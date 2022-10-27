import time
from logging import Logger

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


class LoggerCallback(TrainerCallback):
    def __init__(self, logger: Logger = None):
        self.logger = logger

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs: dict = None, **kwargs):
        if logs is None:
            return
        _msg = []
        for k in sorted(logs.keys()):
            if k == "epoch":
                _msg.append(f"Ep: {logs[k]:.2f}")
            elif k in ("learning_rate", "loss", "eval_loss", "test_loss"):
                _msg.append(f"{k}: {logs[k]:.2e}")
            elif k in ("train_samples_per_second", "train_runtime"):
                _msg.append(f"{k}: {logs[k]:.2f}")
            else:
                _msg.append(f"{k}: {logs[k]}")
        msg = " | ".join(_msg)
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        setattr(self, "epoch_tic", time.time())

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tic = getattr(self, "epoch_tic")
        self.logger.info(f"Epoch {round(state.epoch)} finished. Cost {(time.time() - tic):.2f} s")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        setattr(self, "train_tic", time.time())

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        tic = getattr(self, "train_tic")
        self.logger.info(f"Training finished. Cost {(time.time() - tic):.2f} s")
