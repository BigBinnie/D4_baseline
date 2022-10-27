from dataclasses import dataclass
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    HfArgumentParser,
)
from tqdm import tqdm

import time
import os
import sys
from allennlp.nn import util

from dataclasses import dataclass, field
from typing import Optional
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import (
    ContextResponseDataset,
    preprocess_json_transformer,
    add_special_tokens_to_tokenizer,
    TOD_SPECIAL_TOKENS,
    set_seed,
)
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from model import Transformer


@dataclass
class RunAruguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    reload_from: Optional[str] = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    seed: Optional[int] = field(default=42, metadata={"help": "random seed"})
    epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    start_epoch: int = field(default=0, metadata={"help": "Start epoch."})
    reload_from: Optional[str] = field(default="")
    use_cached_dataset: bool = field(default=False)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    do_test: bool = field(default=False)
    output_path: Optional[str] = field(default='')
    train_data_path: Optional[str] = field(default='./dataset/tod_train.json')
    val_data_path: Optional[str] = field(default='dataset/tod_val.json')
    test_data_path: Optional[str] = field(default='dataset/tod_test.json')


def evaluate(model, val_dataloader, device, tokenizer):
    model.eval()

    perplexity = 0
    temp_loss = 0
    batch_count = 0
    print("start calculate the perplexity....")

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            batch = [item.to(device) for item in batch]

            (
                encoder_input,
                decoder_input,
                mask_encoder_input,
                mask_decoder_input,
            ) = batch

            # (batch_size, max_length, vocab_size)
            logits = model(
                encoder_input,
                mask_encoder_input,
                decoder_input,
                mask_decoder_input,
            )

            out = logits[:, :-1].contiguous()
            target = decoder_input[:, 1:].contiguous()
            target_mask = mask_decoder_input[:, 1:].contiguous()

            loss = util.sequence_cross_entropy_with_logits(
                out, target, target_mask, average="token"
            )

            temp_loss += loss.item()
            perplexity += np.exp(loss.item())

            batch_count += 1

    print("validate perplexity:" + str(perplexity / batch_count))
    print("validate loss:" + str(temp_loss / batch_count))

    print(f"perplexity: {str(perplexity / batch_count)}" + "\n")
    print(f"loss: {str(temp_loss / batch_count)}" + "\n\n")


def train_model(
    args,
    epochs=10,
    num_gradients_accumulation=4,
    batch_size=16,
    lr=1e-4,
    start_epoch=0,
    reload_from=None,
):
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    test_data_path = args.test_data_path

    # make sure your model is on GPU
    device = torch.device("cuda")

    # ------------------------LOAD MODEL-----------------
    print("load the model....")

    tokenizer = BertTokenizer.from_pretrained("/mnt/lustre/sjtu/home/lfd98/Dataset/transformers/bart-base-chinese")
    add_special_tokens_to_tokenizer(tokenizer, TOD_SPECIAL_TOKENS)
    vocab_size = len(tokenizer)
    print("vocab_size: ", vocab_size)
    model = Transformer(vocab_size)
    model = model.to(device)
    if reload_from is not None:
        print("reload from " + reload_from)
        model.reload_from(reload_from, resize_vocab_to=vocab_size, device=device)

    print(f"load success")

    if args.use_cached_dataset and os.path.exists("valid_dataset.pkl"):
        val_dataset = pickle.load(open("valid_dataset.pkl", "rb"))
    else:
        val_dataset = [
            preprocess_json_transformer(x, tokenizer, "max_length", 400)
            for x in ContextResponseDataset(val_data_path)
        ]
        pickle.dump(val_dataset, open("valid_dataset.pkl", "wb"))
    if args.use_cached_dataset and os.path.exists("test_dataset.pkl"):
        test_dataset = pickle.load(open("test_dataset.pkl", "rb"))
    else:
        test_dataset = [
            preprocess_json_transformer(x, tokenizer, "max_length", 400)
            for x in ContextResponseDataset(test_data_path)
        ]
        pickle.dump(test_dataset, open("test_dataset.pkl", "wb"))

    val_dataloader = DataLoader(
        dataset=val_dataset, shuffle=True, batch_size=batch_size
    )
    if args.do_train:
        if args.use_cached_dataset and os.path.exists("train_dataset.pkl"):
            train_dataset = pickle.load(open("train_dataset.pkl", "rb"))
        else:
            train_dataset = [
                preprocess_json_transformer(x, tokenizer, "max_length", 400)
                for x in ContextResponseDataset(train_data_path)
                # for x in ContextResponseDataset("./dataset/tod_dev.json")
            ]
            pickle.dump(train_dataset, open("train_dataset.pkl", "wb"))
        train_dataloader = DataLoader(
            dataset=train_dataset, shuffle=True, batch_size=batch_size
        )

        num_train_optimization_steps = (
            len(train_dataset) * epochs // batch_size // num_gradients_accumulation
        )

        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            weight_decay=0.01,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=num_train_optimization_steps,
        )
        # ------------------------END SET OPTIMIZER--------------
        update_count = 0

        f = open("valid_loss.txt", "w")
        start = time.time()
        print("start training....")
        for epoch in range(start_epoch, start_epoch + epochs):
            # ------------------------training------------------------
            model.train()
            losses = 0
            times = 0
            for batch in tqdm(train_dataloader):
                batch = [item.to(device) for item in batch]

                (
                    encoder_input,
                    decoder_input,
                    mask_encoder_input,
                    mask_decoder_input,
                ) = batch
                #            print (knowledge_input.size())
                logits = model(
                    encoder_input, mask_encoder_input, decoder_input, mask_decoder_input
                )

                #            logits = nn.parallel.data_parallel(model, encoder_input, mask_encoder_input, decoder_input, mask_decoder_input, knowledge_input, device_ids=[5,6,7])

                out = logits[:, :-1].contiguous()
                target = decoder_input[:, 1:].contiguous()
                target_mask = mask_decoder_input[:, 1:].contiguous()

                loss = util.sequence_cross_entropy_with_logits(
                    out, target, target_mask, average="token"
                )
                loss.backward()

                losses += loss.item()

                times += 1
                update_count += 1
                max_grad_norm = 1.0

                if (
                    update_count % num_gradients_accumulation
                    == num_gradients_accumulation - 1
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            end = time.time()
            print("-" * 20 + "epoch" + str(epoch) + "-" * 20)
            print("time:" + str(end - start))
            print("loss:" + str(losses / times))
            start = end

            os.makedirs(args.output_path, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.output_path, f"epoch{epoch}.pth"),
            )
        # ------------------------validate------------------------
    if args.do_eval:
        evaluate(model, val_dataloader, device, tokenizer)


parser = HfArgumentParser((RunAruguments))
(args,) = parser.parse_args_into_dataclasses()
if __name__ == "__main__":
    set_seed(args.seed)
    train_model(
        args,
        epochs=args.epochs, start_epoch=args.start_epoch, reload_from=args.reload_from
    )
