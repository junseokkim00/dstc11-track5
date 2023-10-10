import argparse
import logging
import os
import random
import json
import pdb
from typing import Dict, Tuple
from argparse import Namespace

import numpy as np
import torch
from sklearn.metrics import (
    recall_score,
    precision_score,
    average_precision_score,
    classification_report,
    f1_score,
)
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration,
)

from .dataset import (
    KnowledgeSelectionAugDataset,
    SPECIAL_TOKENS,
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args,
)
from .utils.model import (
    run_batch_selection_aug_train,
    run_batch_selection_aug_eval,
)
from .utils.data import write_selection_preds, write_detection_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

os.environ["TRANSFORMERS_OFFLINE"] = "1"


def get_classes(args):
    """Get classes for dataset, model, training func, and eval func"""
    task, model = args.task, args.model_name_or_path

    if task.lower() == "selection_aug":
        return (
            KnowledgeSelectionAugDataset,
            T5ForConditionalGeneration,
            run_batch_selection_aug_train,
            run_batch_selection_aug_eval,
        )

    else:
        raise ValueError(
            "args.task not in ['generation_review', 'selection_review', 'detection_review'], got %s"
            % task
        )


def evaluate(
    args, eval_dataset, model: PreTrainedModel, run_batch_fn, tokenizer, desc=""
) -> Dict:
    """Model evaluation for knowledge seeking turn detection and knowledge selection
    Report evaluation results if gold labels are available
    """
    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    # eval_batch_size for selection must be 1 to handle different number of candidates
    # args.eval_batch_size = 1

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn,
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    all_augs = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
        if nb_eval_steps > 20:
            pass
        with torch.no_grad():
            loss, aug, labels = run_batch_fn(args, model, batch)
            aug = [(k, tokenizer.decode(a, skip_special_tokens=True)) for (k, a) in aug]
            all_augs.extend(aug)
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps

    result = {"loss": eval_loss, "val_measure": eval_loss, "aug": all_augs}
    return result


def train(
    args,
    train_dataset,
    eval_dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    run_batch_fn_train,
    run_batch_fn_eval,
) -> Tuple[int, float]:
    """Model training and evaluation"""
    log_dir = os.path.join("runs", args.exp_name) if args.exp_name else None
    tb_writer = SummaryWriter(log_dir)
    args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size
    args.eval_batch_size = args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        # shuffle=True,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
    )

    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
    )
    if 0 < args.warmup_steps < 1:
        args.warmup_steps = int(args.warmup_steps * t_total)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # for reproducibility
    val_loss = float("inf")
    best_aug = []
    for _ in train_iterator:
        local_steps = 0  # update step
        tr_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        step = 0  # backward step
        total_log_loss = 0
        for _, batch in enumerate(epoch_iterator):
            model.train()
            if step > 20:
                pass
            for loss, logits, _ in run_batch_fn_train(
                args, model, batch, global_step=global_step
            ):
                if step % 30 == 0:
                    logger.info(loss.mean().item())
                    ans_text = tokenizer.decode(batch[3][0], skip_special_tokens=True)
                    pred_text = tokenizer.decode(
                        torch.argmax(logits[0], dim=1), skip_special_tokens=True
                    )
                    logger.info(f"ans : {ans_text}")
                    logger.info(f"pred : {pred_text}")
                step += 1
                total_log_loss += loss.mean().item()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    local_steps += 1
                    epoch_iterator.set_postfix(Loss=tr_loss / local_steps)
                    total_log_loss = 0
        results = evaluate(
            args,
            eval_dataset,
            model,
            run_batch_fn_eval,
            tokenizer,
            desc=str(global_step),
        )

        for key, value in results.items():
            if key == "aug":
                continue
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)

        if results["val_measure"] < val_loss:
            logger.info(f"Find a smaller val loss measure {results['val_measure']}")
            val_loss = results["val_measure"]
            best_aug = results["aug"]
            # Save model checkpoint
            save_model(args, args.output_dir, model, tokenizer)
        else:
            logger.info(
                f"The val loss measure {results['val_measure']} is larger than "
                f"the smallest val loss {val_loss}, continue to train ... "
            )

    tb_writer.flush()
    tb_writer.close()

    return (global_step, tr_loss / local_steps, best_aug)


def save_model(args, output_dir, model, tokenizer):
    """Save model, tokenizer, and params to the output dir"""
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model

    logger.info("Saving model checkpoint to %s", output_dir)
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
        json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
    logger.info("Saving model checkpoint to %s", output_dir)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, help="JSON configuration file")
    parser.add_argument(
        "--model_name_or_path", type=str, help="model_name_or_path", default="gpt2"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Perform evaluation only"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=("selection_aug"),
        help="to specify the task. Will overwrite the setting in params.json",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Saved checkpoint directory"
    )
    parser.add_argument(
        "--history_max_tokens",
        type=int,
        default=-1,
        help="Maximum length in tokens for history, will override that value in config.",
    )
    parser.add_argument(
        "--knowledge_max_tokens",
        type=int,
        default=-1,
        help="Maximum length in tokens for knowledge, will override that value in config.",
    )
    parser.add_argument("--dataroot", type=str, default="data", help="Path to dataset.")
    parser.add_argument(
        "--knowledge_file",
        type=str,
        default="knowledge.json",
        help="knowledge file name.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="val",
        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="Read a dataset without labels.json. This option is useful when running "
        "knowledge-seeking turn detection on test dataset where labels.json is not available.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default=None,
        help="If set, the labels will be loaded not from the default path, but from this file instead."
        "This option is useful to take the outputs from the previous task in the pipe-lined evaluation.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Predictions will be written to this file.",
    )
    parser.add_argument(
        "--negative_sample_method",
        type=str,
        choices=["all", "mix", "oracle"],
        default="",
        help="Negative sampling method for knowledge selection, will override the value in config.",
    )
    parser.add_argument(
        "--eval_all_snippets",
        action="store_true",
        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="If set, will only use a small number (==debug) of data for training and test.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        default="",
        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}",
    )
    parser.add_argument(
        "--eval_desc",
        type=str,
        default="",
        help="Optional description to be listed in eval_results.txt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    verify_args(args, parser)

    # load args from params file and update the args Namespace
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)

        update_additional_params(params, args)
        args.update(params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    set_default_dataset_params(dataset_args)
    dataset_args.task = args.task
    dataset_args.eval_only = args.eval_only
    dataset_args.debug = args.debug

    # Setup CUDA & GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = device

    # Set seed
    set_seed(args)

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(
        args
    )

    if args.checkpoint is not None:
        model = model_class.from_pretrained(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = nn.DataParallel(model)
        model.to(args.device)
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        tokenizer.model_max_length = min(1024, tokenizer.model_max_length)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))
        model = nn.DataParallel(model).to("cuda")
    logger.info("Training/evaluation parameters %s", args)

    # load datasets and train the model
    train_dataset = dataset_class(dataset_args, tokenizer, split_type="train")
    eval_dataset = dataset_class(
        dataset_args, tokenizer, split_type="val"
    )  # main difference is during evaluation, val need to go through all snippets
    global_step, tr_loss, best_aug = train(
        args,
        train_dataset,
        eval_dataset,
        model,
        tokenizer,
        run_batch_fn_train,
        run_batch_fn_eval,
    )

    results = evaluate(
        args,
        train_dataset,
        model,
        run_batch_fn_eval,
        tokenizer,
        desc=str(global_step),
    )
    aug_file_name = os.path.join("runs", args.exp_name, "augdata.json")
    with open(aug_file_name, "w") as f:
        json.dump(results["aug"], f, ensure_ascii=False, indent=4)

    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


def get_key(knowledge):
    if "sent-id" in knowledge:
        knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
    else:
        knowledge_key = (
            f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"
        )

    return knowledge_key


if __name__ == "__main__":
    main()
