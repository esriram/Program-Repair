import argparse
import json
import os
import random
from datetime import timedelta
from timeit import default_timer as get_now

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import zero_to_fp32
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AdamW, AutoTokenizer, T5ForConditionalGeneration
from util.input_args import ScheduleArgs
from util.util import get_optimizer, get_scheduler, set_seed, to_device

from Dataset import Dataset, DeviceDataLoader, _grab_raw_dataset_num, build_dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Adopted from 24hourBERT
def _prepare_optimizer_parameters(model: T5ForConditionalGeneration, weight_decay):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    no_decay = []
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def prepare_optimizer_and_model(model, args, pretrained_path=None):
    optimizer_grouped_parameters = _prepare_optimizer_parameters(
        model, weight_decay=0.0
    )
    # Current Optimizer and Scheduler based on 24HourBERT implementation
    optimizer = get_optimizer(
        type="adamw",
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
        params=optimizer_grouped_parameters,
    )
    schedule_args = ScheduleArgs(
        lr_schedule="step", curve="linear", warmup_proportion=0.1
    )
    lr_scheduler = get_scheduler(
        schedule_args=schedule_args, optimizer=optimizer, extra_args=args
    )

    # TODO look into ZERO optimization (can offload some memory), only works with Adam
    ds_config = {
        "train_batch_size": args.batch_size,
        # set micro batch size to be the same as the training batch size for now
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "steps_per_print": 10000,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "fp16": {  # enable fp16
            "enabled": args.fp16,
            "loss_scale": 0.0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            # https://github.com/microsoft/DeepSpeed/issues/1022
            "min_loss_scale": 1,
        },
        "local_rank": args.local_rank,
    }

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        # args=args,
        model=model,
        model_parameters=optimizer_grouped_parameters,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config_params=ds_config,
    )

    print(f"optimizer type: {type(optimizer)}")
    print(f"optimizer description: {optimizer}")

    return model, optimizer, lr_scheduler


def load_base_model(pretraining_objective, pretrained_path=None):
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
    if pretrained_path is not None:
        print("Loading Pretrained Model from Path: {}".format(pretrained_path))
        if pretraining_objective == "MLM":
            model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
        model = to_device(model, device)
    else:
        print("Initializing from Pretrained CodeT5")
        if pretraining_objective == "MLM":
            model = T5ForConditionalGeneration.from_pretrained(
                "Salesforce/codet5-large"
            )
        else:
            raise ValueError("Pretraining Objective not supported")
    return tokenizer, model


# Training Loop
def train(
    args,
    model_name,
    lr,
    batch_size,
    pretraining_objective,
    masking_rate=0.15,
    masking_style=1,
    static_repeat=10,
    dataset_name=None,
    bug_id=None,
    extract_type=None,
    repo_name=None,
    pretrained_path=None,
    data_path=None,
    data_type=None,
):
    if dataset_name == "DefextsKotlin":
        writer = SummaryWriter(
            "runs/{}_{}_{}_{}".format(
                repo_name, str(masking_rate), str(masking_style), model_name
            )
        )
    else:
        writer = SummaryWriter(
            "runs/{}_{}_{}_{}_{}".format(
                repo_name, bug_id, str(masking_rate), str(masking_style), model_name
            )
        )

    tokenizer, model = load_base_model(pretraining_objective, pretrained_path)
    args.max_steps = args.max_steps * static_repeat
    model, optimizer, lr_scheduler = prepare_optimizer_and_model(model, args)

    data_path_info = [data_path, dataset_name, extract_type, bug_id, data_type]
    dataset = build_dataset(
        tokenizer,
        masking_rate,
        masking_style,
        pretraining_objective,
        static_repeat,
        repo_name,
        data_path_info,
        device,
        args.seed,
    )
    # return
    epochs = round(args.max_steps / (dataset.__len__() / args.batch_size))
    print("Epochs:", epochs)
    sampler = DistributedSampler(
        dataset, num_replicas=args.num_gpus, rank=args.local_rank
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=sampler,
    )

    if dataset_name == "DefextsKotlin":
        last_output_dir = "trained_models/{}_{}_{}_{}".format(
            repo_name, str(masking_rate), str(masking_style), model_name
        )
        save_output_dir = "saved_models/{}_{}_{}_{}".format(
            repo_name, str(masking_rate), str(masking_style), model_name
        )
    else:
        last_output_dir = "trained_models/{}_{}_{}_{}_{}".format(
            repo_name, bug_id, str(masking_rate), str(masking_style), model_name
        )
        save_output_dir = "saved_models/{}_{}_{}_{}_{}".format(
            repo_name, bug_id, str(masking_rate), str(masking_style), model_name
        )

    lr_scheduler_last = os.path.join(last_output_dir, "scheduler.pt")
    if os.path.exists(lr_scheduler_last):
        lr_scheduler.load_state_dict(torch.load(lr_scheduler_last))

    optimizer_last = os.path.join(last_output_dir, "optimizer.pt")
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    running_loss = 0.0
    t_steps = 0
    r_steps = 0
    for epoch in range(epochs):
        model.train()
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        random_show_idxs = [random.randint(0, len(loop) - 1) for _ in range(20)]
        for i, batch in enumerate(loop):
            if pretraining_objective == "MLM":
                batch = (batch[0].to(model.local_rank), batch[1].to(model.local_rank))
                input_ids, labels = batch
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            model.backward(loss)
            model.step()
            # print relevant info to progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()
            t_steps += 1
            if (
                t_steps
                % ((10 * args.batch_size / args.micro_batch_size) / args.num_gpus)
                == 0
                and model.local_rank == 0
            ):
                r_steps += 1
                writer.add_scalar(
                    "training loss",
                    running_loss
                    / ((args.batch_size / args.micro_batch_size) / args.num_gpus),
                    r_steps,
                )
                running_loss = 0.0
                writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], r_steps)
                print("Example", i)
                print("Input_ids:", tokenizer.decode(input_ids[0]).replace("<pad>", ""))
                print("Labels:", tokenizer.decode(labels[0]).replace("<pad>", ""))
                print(
                    "Outputs:",
                    tokenizer.decode(logits[0].argmax(dim=-1)).replace("<pad>", ""),
                )
                print("*" * 10)

        model.save_pretrained(save_output_dir)

        print("Done")
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masking_rate", type=int, default=15)
    parser.add_argument(
        "--masking_style",
        type=int,
        default=3,
        help="1: Default; 2: Masking variables; 3: Avoid masking syntax tokens; 4: Only masking syntax tokens; 5: Masking spans",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--static_repeat",
        type=int,
        default=10,
        help="number of times the training data is randomly masked to produce new input",
    )
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument(
        "--pretraining-objective", type=str, default="MLM", help="MLM; SBO; NSP; RTD"
    ),
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="defects4j")
    parser.add_argument("--data_type", type=str, default="NL+PL")
    parser.add_argument("--extract_type", type=str, default="function")
    parser.add_argument("--bug_id", type=str, default="oldest")
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="use FP16 training"
    )
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--repo_name", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--max_steps", type=int, default=9200)
    # Note Number of GPUS should be preset using deepspeed
    args = parser.parse_args()
    # get current time
    args.exp_start_marker = get_now()
    print("Run with setting:")
    print(args)
    set_seed(args.seed)  # set seed for deterministic runs
    deepspeed.init_distributed()
    train(
        args=args,
        model_name=args.name,
        lr=args.lr,
        batch_size=args.batch_size,
        pretraining_objective=args.pretraining_objective,
        masking_rate=args.masking_rate / 100,
        masking_style=args.masking_style,
        static_repeat=args.static_repeat,
        dataset_name=args.dataset_name,
        bug_id=args.bug_id,
        extract_type=args.extract_type,
        repo_name=args.repo_name,
        pretrained_path=args.pretrained_path,
        data_path=args.data_path,
        data_type=args.data_type,
    )


if __name__ == "__main__":
    main()
