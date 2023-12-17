import sys
import os
sys.path.insert(0, os.getcwd() + '/')

import argparse
import json
import random
import logging
import numpy as np

import traceback
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, DataCollatorForSeq2Seq, TrainingArguments
from transformers.models.t5 import T5ForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint

from src.pretraining.atomic_code_util import DelayedKeyboardInterrupt
from src.pretraining.semcode_trainer import SemCodeTrainer


def get_logger():
    logger = logging.getLogger()
    return logger


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def find_langs_in_data_dir(data_dir):
    return list(set(
        ["_".join(f[:-6].split("_")[:-1]) for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    ))


def num_parameters(model):
    model_parameters = model.parameters()
    return sum([np.prod(p.size()) for p in model_parameters])


logging.basicConfig(level=logging.NOTSET,
					handlers = [logging.FileHandler('pretrain.log', 'w')],
					format="%(asctime)s;%(levelname)s;%(message)s",
					datefmt='%Y-%m-%d %H:%M:%S')

logger = get_logger()


def main(args):
    set_seeds(args.seed) # set the seed for all randomness sources

    experiment_name = args.output_dir.split("/")[-1] if "/" in args.output_dir else args.output_dir

    # Step 1: Initialize the models.
    if args.initial_model == 'codet5-base':
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    else:
        tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

    logger.info(f"starting Training from {args.initial_model}")
    logger.info(f"total parameters : {num_parameters(model)}")

    # Steps 2: Prepare the datasets.
    data_dir = args.data_path
    languages = args.languages
    cache_dir = args.data_cache_path

    if cache_dir is None:
        cache_dir = os.path.join(data_dir, "cached-" + experiment_name)
        os.makedirs(cache_dir, exist_ok=True)

    if languages is None:
        languages = sorted(find_langs_in_data_dir(data_dir))

    logger.info(languages)
    data_files = {"train": ["train.jsonl"], "validation": ["valid.jsonl"]}
    raw_datasets = load_dataset(path=data_dir, data_dir=data_dir, data_files=data_files, cache_dir=cache_dir, name=experiment_name)
    column_names = raw_datasets["train"].column_names

    training_argument_dict = json.load(open(args.training_config))
    training_argument_dict["output_dir"] = args.output_dir
    training_args = TrainingArguments(**training_argument_dict)
    training_args.dataloader_num_workers = args.workers
    if args.local_rank != -1:
        training_args.local_rank = args.local_rank

    def prepare_features(examples):
        inputs = examples["transformed"]
        targets = examples["original"]
        model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding="max_length", truncation=True)
        model_outputs = tokenizer(targets, max_length=tokenizer.model_max_length, padding="max_length", truncation=True)

        model_inputs["labels_original"] = model_outputs["input_ids"]
        # If we are padding here, replace all tokenizer.pad_token_id (0) in the outputs by -100 when we want to ignore
        # padding in the loss for the decoder.
        model_outputs["input_ids"] = [
            [(_l if _l != tokenizer.pad_token_id else -100) for _l in output] for output in model_outputs["input_ids"]
        ]
        model_inputs["labels"] = model_outputs["input_ids"]
        return model_inputs

    train_dataset = raw_datasets["train"]
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            prepare_features,
            batched=True,
            num_proc=args.workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    eval_dataset = raw_datasets["validation"]
    with training_args.main_process_first(desc="eval dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            prepare_features,
            batched=True,
            num_proc=args.workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on eval dataset",
        )

    # Step 3: Run the training.
    trainer = SemCodeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            model=model,
            tokenizer=tokenizer,
            padding='max_length',
        ),
        tokenizer=tokenizer,
    )

    try:
        if args.do_not_reload_from_checkpoint:
            trainer.train()
        else:
            last_checkpoint = get_last_checkpoint(args.output_dir)
            try:
                trainer.train(resume_from_checkpoint=last_checkpoint)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as ex:
                traceback.print_exc()
                logger.info(f"Found and exception {ex} of type {type(ex)}. Carefully inspect the stacktrace")
                logger.info(f"Invalid checkpoint found in {last_checkpoint}")
                logger.info("Please delete this directory and try again!")
    except KeyboardInterrupt:
        with DelayedKeyboardInterrupt():
            logger.info("*" * 150)
            logger.info("*" * 70, "CAUTION", "*" * 70)
            logger.info("*" * 70, "CAUTION", "*" * 70)
            logger.info("Keyboard Interrupt encountered!!")
            logger.info("Saving the checkpoint in ", trainer.state.global_step)
            trainer.save_checkpoint()
            logger.info("*" * 70, "CAUTION", "*" * 70)
            logger.info("*" * 70, "CAUTION", "*" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, help="Path of the training configuration file", required=True)
    parser.add_argument("--output_dir", type=str, help="Path of the output directory", required=True)
    parser.add_argument("--initial_model", type=str, choices=['codet5-base', 'random', 'codet5-small'], default='codet5-base')
    parser.add_argument("--data_path", type=str, help="Base Directory of processed data", required=True)
    parser.add_argument("--workers", help="Number of worker CPU", type=int, default=20)
    parser.add_argument("--data_cache_path", type=str, help="Caching Directory of processed data", default=None)
    parser.add_argument("--languages", type=str, nargs="+", help="List of languages to train on! Training will select all if none is given", default=['java'])
    parser.add_argument("--do_not_reload_from_checkpoint", action="store_true", help="Flag to forcefully stop reloading from the checkpoint")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--overwrite_cache", help='Overwrite the cache dataset directory, if such exists', action='store_true')
    parser.add_argument("--local_rank", help="The local rank in distributed training", type=int, default=-1)    
    args = parser.parse_args()
    main(args)
