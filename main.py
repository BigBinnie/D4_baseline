import json
from os.path import join
from typing import List

from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed, Seq2SeqTrainer, \
    BartForConditionalGeneration, BartForSequenceClassification, BertTokenizer, PreTrainedTokenizer, Trainer, \
    AutoModelForSequenceClassification

from models.modeling_cpt import CPTForConditionalGeneration, CPTForSequenceClassification
from utils.args import ModelConstructArgs, DataArgs
from utils.callback import LoggerCallback
from utils.data_process import DialogDataset, CollateFnForSummary, CollateFnForDialog, CollateFnForClassify, CollateFnForPrediction
from utils.metrics import MetricComputer, strip_special_tokens
from utils.utils import get_logger


def add_special_tokens(tokenizer: PreTrainedTokenizer):
    special_token = ["<doc_bos>", "<pat_bos>", "<act>", "<por>"]
    for i in range(0, 12):
        special_token.append(f"<s_{i}>")
    tokenizer.add_special_tokens({"additional_special_tokens": special_token})
    return tokenizer


def get_model_and_tokenizer(task: str, model_type: str, model_path: str, n_class):
    print("model",model_type)
    if "bart" in model_type:
        if "srisk" in task:
            model = BartForSequenceClassification.from_pretrained(model_path, num_labels=4)
        elif "drisk" in task:
            model = BartForSequenceClassification.from_pretrained(model_path, num_labels=4)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_path)
    elif "cpt" in model_type:
        if "srisk" in task:
            model = CPTForSequenceClassification.from_pretrained(model_path, num_labels=4, cls_mode=3)
        elif "drisk" in task:
            model = CPTForSequenceClassification.from_pretrained(model_path, num_labels=4, cls_mode=3)
        else:
            model = CPTForConditionalGeneration.from_pretrained(model_path)
    elif "bert" in model_type:
        if "srisk" in task:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=n_class)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=n_class)
    else:
        raise ValueError(f"Unsupported model type: {model_path}")

    tokenizer = add_special_tokens(BertTokenizer.from_pretrained(model_path))
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def main(_args: List[str] = None):
    # ===== Parse arguments =====
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ModelConstructArgs, DataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(__name__, exp_dir=train_args.output_dir, rank=train_args.local_rank)

    for _log_name, _logger in logger.manager.loggerDict.items():
        if _log_name.startswith("transformers.trainer"):
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    print("model_type-1",model_args.model_type)
    model, tokenizer = get_model_and_tokenizer(task=data_args.task, model_type=model_args.model_type, model_path=model_args.model_path, n_class=data_args.n_class)

    # ===== Get datasets =====
    trainset = DialogDataset(data_args.train_text)
    devset = DialogDataset(data_args.dev_text)
    testset = DialogDataset(data_args.test_text)

    if data_args.task == "dialog":
        _CollateFn = CollateFnForDialog(tokenizer, max_len=data_args.max_len, truncation="right")
    elif data_args.task == "prediction":
        _CollateFn = CollateFnForPrediction(tokenizer, max_len=data_args.max_len, truncation="right")
    elif data_args.task == "summary":
        _CollateFn = CollateFnForSummary(tokenizer, max_len=data_args.max_len, truncation="mid",
                                         add_portrait=data_args.add_portrait)
    else:
        _CollateFn = CollateFnForClassify(tokenizer, max_len=data_args.max_len, truncation="mid",
                                          add_portrait=data_args.add_portrait, risk=data_args.task)

    # ===== Metrics =====
    compute_metrics = MetricComputer(tokenizer, task=data_args.task).compute_metrics

    # ===== Callbacks =====
    callbacks = [LoggerCallback(logger)]

    # ===== Trainer =====
    _Trainer = Trainer if "risk" in data_args.task or "prediction" in data_args.task else Seq2SeqTrainer

    trainer = _Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=_CollateFn.collate_fn,
        train_dataset=trainset,
        eval_dataset=devset,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        predictions, labels, metrics = trainer.predict(testset, metric_key_prefix="predict")
        callbacks[0].on_log(train_args, trainer.state, trainer.control, logs=metrics)

        if "risk" in data_args.task:
            if "bert" in model_args.model_type:
                result = {
                    "preds": str(predictions.argmax(axis=1).tolist()),
                    "labels": str(labels.tolist()),
                }
            else:
                result = {
                    "preds": str(predictions[0].argmax(axis=1).tolist()),
                    "labels": str(labels.tolist()),
                }
            with open(join(train_args.output_dir, "result.json"), "w", encoding='utf8') as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
        elif data_args.task == "prediction":
            result = {
                "preds": str(predictions.argmax(axis=1).tolist()),
                "labels": str(labels.tolist()),
            }
            with open(join(train_args.output_dir, "result.json"), "w", encoding='utf8') as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
        else:
            test_preds = strip_special_tokens(tokenizer.batch_decode(predictions, skip_special_tokens=False))
            labels = strip_special_tokens(tokenizer.batch_decode(labels, skip_special_tokens=False))

            if len(test_preds) == len(labels):
                result = [group for group in zip(test_preds, labels)]
                with open(join(train_args.output_dir, "result.json"), "w", encoding='utf8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=1)
            else:
                logger.warning("Length of test_preds != labels")
                with open(join(train_args.output_dir, "preds.json"), "w", encoding='utf8') as f:
                    json.dump(test_preds, f, ensure_ascii=False, indent=1)
                with open(join(train_args.output_dir, "labels.json"), "w", encoding='utf8') as f:
                    json.dump(labels, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
