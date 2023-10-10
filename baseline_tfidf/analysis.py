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

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    BartForConditionalGeneration,
    AutoModelForSequenceClassification,
)

from .dataset import (
    KnowledgeTurnDetectionDataset,
    KnowledgeSelectionDataset,
    ResponseGenerationDataset,
    SPECIAL_TOKENS,
)
from .utils.argument import (
    set_default_params,
    set_default_dataset_params,
    update_additional_params,
    verify_args,
)
from .utils.model import (
    run_batch_detection_train,
    run_batch_detection_eval,
    run_batch_selection_train,
    run_batch_selection_eval,
    run_batch_generation_train,
    run_batch_generation_eval,
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
    if task.lower() == "generation":
        return (
            ResponseGenerationDataset,
            BartForConditionalGeneration,
            run_batch_generation_train,
            run_batch_generation_eval,
        )
    elif task.lower() == "selection":
        return (
            KnowledgeSelectionDataset,
            AutoModelForSequenceClassification,
            run_batch_selection_train,
            run_batch_selection_eval,
        )
    elif task.lower() == "detection":
        return (
            KnowledgeTurnDetectionDataset,
            AutoModelForSequenceClassification,
            run_batch_detection_train,
            run_batch_detection_eval,
        )
    else:
        raise ValueError(
            "args.task not in ['generation_review', 'selection_review', 'detection_review'], got %s"
            % task
        )


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
        choices=("detection", "selection", "generation"),
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Set seed
    set_seed(args)

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = get_classes(
        args
    )

    args.output_dir = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    eval_dataset = dataset_class(
        dataset_args,
        tokenizer,
        split_type=args.eval_dataset,
        labels=not args.no_labels,
        labels_file=args.labels_file,
    )
    pdb.set_trace()


if __name__ == "__main__":
    main()
