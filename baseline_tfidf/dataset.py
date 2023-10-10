import os
import random
import logging
from collections import defaultdict
from itertools import chain

import csv
import pdb
import torch
from tqdm import tqdm

from .utils.data import pad_ids, truncate_sequences
from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords, wordnet

from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words("english")
stop_words.extend(["wa", "would", "also", "Also"])
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")
logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<speaker1>",
        "<speaker2>",
        "<knowledge_sep>",
        "<knowledge_tag>",
    ],
}


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.SPECIAL_TOKENS = SPECIAL_TOKENS

        (
            self.speaker1,
            self.speaker2,
            self.knowledge_sep,
            self.knowledge_tag,
        ) = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]
        self.dataset_walker = DatasetWalker(
            split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file
        )
        self.dialogs = self._prepare_conversations()
        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.snippets = self._prepare_knowledge()
        self._create_examples()

    def _prepare_conversations(self):
        """Tokenize and encode the dialog data"""
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        for i, (log, label) in enumerate(
            tqdm(self.dataset_walker, disable=False, desc="tokenizing...")
        ):
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs

    def _clean_text_noun(self, text):
        text = text.lower()
        text = text.split()
        temp = []
        for txt in text:
            txt = [
                word
                for (word, pos) in nltk.pos_tag(nltk.word_tokenize(txt))
                if pos[0] == "N"
            ]
            temp.append(" ".join(txt))
        text = temp
        # text = [ps.lemmatize(word) for word in text]
        text = [word for word in text if not word in set(stop_words)]
        text = " ".join(text)
        return text

    def _clean_text_lem(self, text):
        text = re.sub("[^a-zA-Z]", " ", text)
        ps = WordNetLemmatizer()

        text = text.lower()
        text = text.split()
        text = [ps.lemmatize(word) for word in text]
        text = [word for word in text if not word in set(stop_words)]
        text = " ".join(text)
        return text

    def _tf_idf_vec(self, corpus, n_gram):
        cv = TfidfVectorizer(ngram_range=(1, n_gram))
        corpus = cv.fit_transform(corpus)
        avg = corpus.mean(axis=0)
        avg = pd.DataFrame(avg, columns=cv.get_feature_names()).T
        avg = avg.rename(columns={0: "score"})
        avg["word"] = avg.index
        avg = avg.sort_values("score", ascending=False)
        key_words = avg[:]["word"].tolist()
        corpus_dense = corpus.todense().tolist()
        # dense_sort = np.sort(corpus_dense, axis=1)[:,-3:]
        indexs = []
        for dense_list in tqdm(corpus_dense):
            dense_list = [(i, v) for (i, v) in enumerate(dense_list)]
            sorted_list = sorted(dense_list, key=lambda item: -item[1])[:3]
            indexs.append([k for k, v in sorted_list])

        return indexs, cv.get_feature_names(), key_words

    def _prepare_knowledge(self):

        """Tokenize and encode the knowledge snippets"""
        self.knowledge_docs = self._get_snippet_list()
        corpus = []

        tokenized_snippets = defaultdict(dict)
        token = "<knowledge_sep> "

        for snippet_id, snippet in enumerate(self.knowledge_docs):
            key = "{}__{}__{}".format(
                snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"]
            )
            knowledge = self._knowledge_to_string(
                snippet["doc"], name=snippet["entity_name"] or ""
            )

            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(knowledge)
            )
            place_name = knowledge[: knowledge.index(token)].strip().lower()
            knowledge = knowledge.replace(place_name, "").lower()
            corpus.append(knowledge[knowledge.index(token) + len(token) :])
            tokenized_snippets[key]["token_ids"] = tokenized_knowledge[
                : self.args.knowledge_max_tokens
            ]
        # corpus1 = [self._clean_text_noun(t) for t in corpus]
        corpus2 = [self._clean_text_lem(t) for t in corpus]
        # index, words = self._tf_idf_vec(corpus1, 1)
        # onegram = [words[i.item()] for i in index]
        index, words, keywords = self._tf_idf_vec(corpus2, 3)
        threegram = []
        for idxs in index:
            temp = []
            for idx in idxs:
                if words[idx] in keywords:
                    temp.append(words[idx])
            if len(temp) == 0:
                temp = ["None"]
            threegram.append(temp)

        import csv

        f = open("test.csv", "w", encoding="utf-8", newline="")
        wr = csv.writer(f)
        for cor, three in zip(corpus, threegram):
            wr.writerow([three, cor])
        f.close()
        for key, gram in zip(list(tokenized_snippets.keys()), threegram):
            adding_text = "keywords : " + " ".join(gram)
            adding_text = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(adding_text)
            )
            tokenized_snippets[key]["token_ids"] += adding_text

        return tokenized_snippets

    def _get_snippet_list(self):
        """Get all knowledge snippets in the dataset"""
        result = []
        for domain in self.knowledge_reader.get_domain_list():
            for entity_id in self.knowledge_reader.knowledge[domain].keys():
                for review_doc_id in self.knowledge_reader.get_review_doc_ids(
                    domain, entity_id
                ):
                    review_doc = self.knowledge_reader.get_review_doc(
                        domain, entity_id, review_doc_id
                    )
                    for review_sent_id, review_sent in review_doc["sentences"].items():
                        result.append(
                            {
                                "domain": domain,
                                "entity_id": entity_id,
                                "entity_name": review_doc["entity_name"],
                                "doc_id": f"{review_doc_id}-{review_sent_id}",
                                "doc": {"body": review_sent},
                            }
                        )
                for faq_doc_id in self.knowledge_reader.get_faq_doc_ids(
                    domain, entity_id
                ):
                    faq_doc = self.knowledge_reader.get_faq_doc(
                        domain, entity_id, faq_doc_id
                    )
                    result.append(
                        {
                            "domain": domain,
                            "entity_id": entity_id,
                            "entity_name": faq_doc["entity_name"],
                            "doc_id": faq_doc_id,
                            "doc": {
                                "body": f"{faq_doc['question']} {faq_doc['answer']}"
                            },
                        }
                    )
        return result

    def _knowledge_to_string(self, doc, name=""):
        """Convert a knowledge snippet to a string"""
        doc_body = f"{name.title()}: {doc['body']}"
        return doc_body

    def _create_examples(self):
        """Creating examples for model training and evaluation"""
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=False, desc="creating examples"):
            if self.args.debug > 0 and len(self.examples) >= self.args.debug:
                break
            dialog_id = dialog["id"]
            label = dialog["label"]

            dialog = dialog["log"]
            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue

            history = [
                self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(turn["text"])
                )
                for turn in dialog
            ]
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(gt_resp)
            )

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances :]

            # perform token-level truncation of history from the left
            truncated_history = truncate_sequences(
                truncated_history, self.args.history_max_tokens
            )

            if target:
                knowledge_keys = []
                knowledge_candidates = defaultdict(lambda: 0)
                used_knowledge = []
                knowledge_prefix_visited = set()

                if "knowledge" not in label:
                    raise ValueError(
                        "Please run entity matching before running knowledge selection"
                    )

                label_knowledge = label["knowledge"]

                for knowledge in label_knowledge:
                    if not (self.args.task == "selection" and self.args.eval_only):
                        if knowledge["doc_type"] == "review":
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
                        else:
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"

                    # find snippets with same entity as candidates
                    prefix = "{}__{}".format(
                        knowledge["domain"], knowledge["entity_id"]
                    )
                    if prefix not in knowledge_prefix_visited:
                        knowledge_prefix_visited.add(prefix)
                        _knowledge_candidates = [
                            cand
                            for cand in self.snippets.keys()
                            if "__".join(cand.split("__")[:-1]) == prefix
                        ]

                        for _knowledge_cand_idx, _knowledge_cand in enumerate(
                            _knowledge_candidates
                        ):
                            knowledge_candidates[_knowledge_cand] = 1
                    if (
                        self.split_type == "train"
                        and self.args.negative_sample_method == "oracle"
                    ):
                        # if there's not enough candidates during training, we just skip this example
                        if len(knowledge_candidates) < self.args.n_candidates or len(
                            knowledge_candidates
                        ) <= len(label["knowledge"]):
                            logger.info("Not enough candidates. Skip this example...")
                            continue

                    if not (self.args.task == "selection" and self.args.eval_only):
                        used_knowledge.append(
                            self.snippets[knowledge_key]["token_ids"][
                                : self.args.knowledge_max_tokens
                            ]
                        )
                        knowledge_keys.append(knowledge_key)
                knowledge_candidates = [k for k, v in knowledge_candidates.items()]

            else:
                knowledge_candidates = None
                used_knowledge = []
                knowledge_keys = []

            self.examples.append(
                {
                    "history": truncated_history,
                    "knowledge": used_knowledge,
                    "knowledge_keys": knowledge_keys,
                    "candidates": knowledge_candidates,
                    "response": tokenized_gt_resp,
                    "response_text": gt_resp,
                    "label": label,
                    "knowledge_seeking": target,
                    "dialog_id": dialog_id,
                }
            )

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(
            args, tokenizer, split_type, labels, labels_file
        )

    def build_input_from_segments(self, history):
        """Build a sequence of input from history"""
        instance = {}

        sequence = [[self.cls]] + history[:-1] + [history[-1]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence0 = [sequence[0]] + sequence_with_speaker[:-1] + [[self.sep]]
        sequence0 = list(chain(*sequence0))
        sequence1 = sequence_with_speaker[-1]

        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]
        data_info = {"dialog_ids": [ins["dialog_id"] for ins in batch]}

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, attention_mask, labels, data_info


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(
            args, tokenizer, split_type, labels, labels_file
        )

        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            # Negative sampling method for knowledge selection
            # all: use all knowledge snippets of all entities as candidates
            # oracle: use all knowledge snippets of oracle entities as candidates
            # mix: use oracle candidates & equally sized candidates sampled from other entities
            raise ValueError(
                "negative_sample_method must be all, mix, or oracle, got %s"
                % self.args.negative_sample_method
            )

    def _knowledge_to_string(self, doc, name=""):
        """convert a knowlege snippet to a string"""
        join_str = " %s " % self.knowledge_sep_token
        doc_body = doc["body"]
        knowledge_string = join_str.join([name.title(), doc_body])
        return knowledge_string

    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": [],
        }

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates with no sampling
            if self.args.eval_all_snippets:
                candidates = list(self.snippets.keys())
            else:
                candidates = example["candidates"]
        else:
            if self.args.negative_sample_method == "all":
                candidates = list(self.snippets.keys())
            elif self.args.negative_sample_method == "mix":
                candidates = example["candidates"] + random.sample(
                    list(self.snippets.keys()), k=len(example["candidates"])
                )
            elif self.args.negative_sample_method == "oracle":
                candidates = example["candidates"]
            else:  # although we have already checked for this, still adding this here to be sure
                raise ValueError(
                    "negative_sample_method must be all, mix, or oracle, got %s"
                    % self.args.negative_sample_method
                )

        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        candidates = [self.snippets[cand_key]["token_ids"] for cand_key in candidates]

        if self.split_type == "train":
            candidates = self._shrink_label_cands(example["knowledge"], candidates)

        label_idx = [candidates.index(knowledge) for knowledge in example["knowledge"]]

        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance, _ = self.build_input_from_segments(cand, example["history"])
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])
        return this_inst

    def build_input_from_segments(self, knowledge, history):
        """Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
        history = history[-2:]
        sequence = [[self.cls]] + history
        if len(history) == 2:
            sequence_with_speaker = (
                [self.speaker2] + history[0] + [self.speaker1] + history[1]
            )

        else:
            sequence_with_speaker = [self.speaker2] + history[0]
        # sequence = [[self.cls]] + history[:-1] + [history[-1]]

        # sequence_with_speaker = [
        #     [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
        #     for i, s in enumerate(sequence[1:])
        # ]
        # sequence_with_speaker = list(chain(*sequence_with_speaker))

        sequence0 = [self.cls] + sequence_with_speaker + [self.sep]
        sequence1 = knowledge + [self.sep]

        if "roberta" in str(type(self.tokenizer)):
            sequence0 += [self.sep]
        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence

    def _shrink_label_cands(self, label, candidates):
        """remove positive knowledge snippets from the candidates"""
        shrunk_label_cands = candidates.copy()
        for l in label:
            if l in shrunk_label_cands:
                shrunk_label_cands.remove(l)
        sample_size = min(len(label), len(shrunk_label_cands))
        shrunk_label_cands = random.sample(shrunk_label_cands, k=sample_size)

        shrunk_label_cands.extend(label)
        random.shuffle(shrunk_label_cands)
        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [
            1 if i in ins["label_idx"] else 0
            for ins in batch
            for i in range(len(ins["input_ids"]))
        ]
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch],
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        token_type_ids = torch.tensor(pad_ids(token_type_ids, 0))
        label_idx = torch.tensor(label_idx)
        return input_ids, token_type_ids, attention_mask, label_idx, data_info


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(
            args, tokenizer, split_type, labels, labels_file
        )

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"], example["history"], example["response"]
        )
        return instance

    def build_input_from_segments(self, knowledge, history, response):
        """Build a sequence of input from 3 segments: knowledge, history and last reply"""
        instance = {}
        knowledge = [[self.knowledge_sep] + k for k in knowledge]
        knowledge = [w for k in knowledge for w in k]

        # 3: special tokens; len(history): special speaker tokens
        entire_input_len = self.tokenizer.model_max_length - 3

        entire_knowledge_len, entire_history_len = len(knowledge), len(
            list(chain(*history))
        )
        max_history_len = int(
            (entire_history_len * entire_input_len)
            / (entire_knowledge_len + entire_history_len)
        )
        max_history_len = min(
            entire_history_len + len(history), max(max_history_len, 256)
        )
        max_knowledge_len = entire_input_len - max_history_len  # - len(history)

        if max_knowledge_len < entire_knowledge_len:
            logger.warning(
                f"Knowledge too long! Have been truncated from {entire_knowledge_len} to {max_knowledge_len}"
            )
            knowledge = knowledge[:max_knowledge_len]
        if max_history_len < entire_history_len:
            logger.warning(
                f"History too long! Have been truncated from {entire_history_len} to {max_history_len}"
            )

        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]  # speaker 2 (user)
        history = list(chain(*sequence_with_speaker[:-1]))[:max_history_len]
        sequence = (
            [[self.bos]]
            + [sequence[0]]
            + [[self.knowledge_tag]]
            + [history]
            + [[self.eos]]
        )
        instance["input_ids"] = list(chain(*sequence))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]
        return instance, sequence

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, attention_mask, lm_labels


class ResponseGenerationEvalDataset(ResponseGenerationDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(
            args, tokenizer, split_type, labels, labels_file
        )

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
