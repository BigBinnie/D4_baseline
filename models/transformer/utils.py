from dataclasses import dataclass, field
from typing import Optional
import torch
import json
import random
import numpy as np

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_class: str = field(metadata={"help": "The model class to use."})
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a jsonlines or csv file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "A prefix to add before every source text (useful for T5 models)."
        },
    )
    lr: Optional[str] = field(default=None, metadata={"help": "The learning rate."})


TOK_RESP_BOS = "<resp_bos>"
TOD_SPECIAL_TOKENS = ["<doc_bos>", "<pat_bos>", "<act>"]
TOD_LM_SPECIAL_TOKENS = TOD_SPECIAL_TOKENS + [TOK_RESP_BOS]

class ContextResponseDataset:
    def __init__(self, filename) -> None:
        with open(filename, "r") as f:
            self.dataset = json.load(f)

    def __getitem__(self, index):
        return (self.dataset[index]["src"], self.dataset[index]["tgt"])

    def __len__(self):
        return len(self.dataset)


class GptDataset(ContextResponseDataset):
    def __getitem__(self, index):
        src, tgt = super().__getitem__(index)
        line = src + " " + TOK_RESP_BOS + " " + tgt
        return (line, line)

def preprocess_json_transformer(sample, tokenizer, padding, max_source_length):
    src, tgt = sample
    model_inputs = tokenizer(
        src, padding=padding, truncation=True, return_token_type_ids=False
    )
    if len(model_inputs["input_ids"]) > max_source_length:
        model_inputs["input_ids"] = model_inputs["input_ids"][-max_source_length:]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][
            -max_source_length:
        ]
    else:
        model_inputs = tokenizer.pad(
            model_inputs, padding="max_length", max_length=max_source_length
        )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt,
            padding=padding,
            truncation=True,
            max_length=max_source_length,
        )

    return (
        torch.tensor(model_inputs["input_ids"]),
        torch.tensor(labels["input_ids"]),
        torch.tensor(model_inputs["attention_mask"]),
        torch.tensor(labels["attention_mask"]),
    )

def preprocess_json(sample, tokenizer, padding, max_source_length):
    src, tgt = sample
    model_inputs = tokenizer(
        src, padding=padding, truncation=True, return_token_type_ids=False,
    )
    if len(model_inputs["input_ids"]) > max_source_length:
        model_inputs["input_ids"] = model_inputs["input_ids"][
            -max_source_length :
        ]
        model_inputs["attention_mask"] = model_inputs["attention_mask"][
            -max_source_length :
        ]
    else:
        model_inputs = tokenizer.pad(
            model_inputs, padding="max_length", max_length=max_source_length
        )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt,
            padding=padding,
            truncation=True,
            max_length=max_source_length,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def add_special_tokens_to_tokenizer(tokenizer, special_tokens):
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)