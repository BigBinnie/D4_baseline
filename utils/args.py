import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class _Args:
    def to_dict(self):
        return asdict(self)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ModelConstructArgs(_Args):
    model_path: Optional[str] = field(default=None, metadata={"help": "Pretrained model path"})
    model_type: Optional[str] = field(default="bart",
                                      metadata={"choices": ["bart-base", "bart-large",
                                                            "cpt-base", "cpt-large", "bert-base", "bert-large"],
                                                "help": "Pretrained model path"})
    init_model: Optional[int] = field(default=0, metadata={"choices": [0, 1], "help": "Init models' parameters"})


@dataclass
class DataArgs(_Args):
    n_class: Optional[int] = field(default=2, metadata={"choices": [2, 3, 4, 6,  11]})
    task: str = field(default="dialog", metadata={"choices": ["dialog", "summary", "drisk", "srisk", "prediction"],
                                                  "help": "Task name"})
    max_len: Optional[int] = field(default=512, metadata={"help": "Maximum sequence length"})
    # keep_doc: Optional[bool] = field(default=False, metadata={"help": "Whether to keep doctors' tokens"})
    add_portrait: Optional[bool] = field(default=True, metadata={"help": "Whether to keep patients' portraits"})
    truncation: Optional[str] = field(default="mid", metadata={"choices": ["mid", "left", "right"],
                                                               "help": "Truncate method. 'Left' means keep left"})

    train_text: Optional[str] = field(default=None, metadata={"help": "Training corpus"})
    dev_text: Optional[str] = field(default=None, metadata={"help": "Dev corpus"})
    test_text: Optional[str] = field(default=None, metadata={"help": "Dev corpus"})
