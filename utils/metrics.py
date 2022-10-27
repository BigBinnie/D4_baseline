import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge import Rouge
from sklearn.metrics import classification_report
from transformers import PreTrainedTokenizer


def strip_special_tokens(preds: List[str]) -> List[str]:
    """ Strip special tokens of BertTokenizer """
    return [re.sub(r"\[\w+]", "", s).strip() for s in preds]


def post_process_text(preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    preds = strip_special_tokens(preds)
    labels = strip_special_tokens(labels)

    # # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    while '' in preds:
        idx = preds.index('')
        preds[idx] = '。'

    return preds, labels


def compute_bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for _ in range(n)))


def compute_nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)


def compute_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g) - n):
                ngram = " ".join(g[idx: idx + n + 1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += -(v + 0.0) / total * (np.log(v + 0.0) - np.log(total))
        div_score[n] = (len(counter[n].values()) + 0.0) / total
    return etp_score, div_score


def calculate_bleu(preds: List[str], labels: List[str], avg=True):
    BLEUs: List[float] = []

    for pred, label in zip(preds, labels):
        pred = pred.split()
        label = label.split()
        # references: List of references, List[List[str]]
        # pred:       Prediction, List[str]
        b = sentence_bleu(references=[label], hypothesis=pred)
        BLEUs.append(b)

    if avg:
        return sum(BLEUs) / len(BLEUs)
    return BLEUs


class MetricComputer:
    def __init__(self, tokenizer: PreTrainedTokenizer, task: str):
        self.tokenizer = tokenizer
        self.task = task
        self.rouge = None if "risk" in task or "prediction" in task else Rouge()

    def compute_metrics(self, eval_preds):
        if "risk" in self.task or "prediction" in self.task:
            return self._metric_for_classify(eval_preds)
        return self._metric_for_seq2seq(eval_preds)

    @staticmethod
    def _metric_for_classify(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]  # Already formatted as np.ndarray
        preds = preds.argmax(axis=1)
        print(type(preds), preds)
        print(type(labels), labels)
        result = classification_report(labels, preds, output_dict=True)  # type: dict
        macro_avg = result.pop("macro avg")
        # Add f1-score, precision, recall
        result.update(macro_avg)
        return result

    def _metric_for_seq2seq(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # preds_tokens = [
        #     self.tokenizer.convert_ids_to_tokens(sample, skip_special_tokens=True)
        #     for sample in preds
        # ]
        # labels_tokens = [
        #     self.tokenizer.convert_ids_to_tokens(sample, skip_special_tokens=True)
        #     for sample in labels
        # ]
        # if data_args.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        # Some simple post-processing
        decoded_preds, decoded_labels = post_process_text(decoded_preds, decoded_labels)

        scores = self.rouge.get_scores(
            decoded_preds, decoded_labels, avg=True, ignore_empty=True
        )
        for key in scores:
            scores[key] = scores[key]["f"] * 100

        result = scores

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["avg_len"] = np.mean(prediction_lens)

        bleu_2 = np.average(
            [compute_bleu(pred, label, 2) for pred, label in zip(decoded_preds, decoded_labels)]
        )
        bleu_4 = np.average(
            [compute_bleu(pred, label, 4) for pred, label in zip(decoded_preds, decoded_labels)]
        )
        nist_2 = np.average(
            [compute_nist(pred, label, 2) for pred, label in zip(decoded_preds, decoded_labels)]
        )
        nist_4 = np.average(
            [compute_nist(pred, label, 4) for pred, label in zip(decoded_preds, decoded_labels)]
        )
        print(len(decoded_labels))
        print(len(decoded_preds))
        meteor = np.average([
            meteor_score([label], pred)
            for pred, label in zip(decoded_preds, decoded_labels)
        ])
        entropy, dist = compute_entropy(decoded_preds)

        result["bleu_2"] = bleu_2
        result["bleu_4"] = bleu_4
        result["nist_2"] = nist_2
        result["nist_4"] = nist_4
        result["meteor"] = meteor
        for i in range(4):
            result[f"entropy-{i}"] = entropy[i]
            result[f"dist-{i}"] = dist[i]

        # result = {k: round(v, 4) for k, v in result.items()}
        return result

    def _compute_metrics(self, eval_preds):
        """ Calculate: BLEU, ROUGE """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)

        # Some simple post-processing
        decoded_preds, decoded_labels = post_process_text(decoded_preds, decoded_labels)

        # Calculate Rouge score
        scores = self.rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        for key in scores:
            scores[key] = scores[key]['f'] * 100

        # Calculate BLEU score
        scores["bleu"] = 100 * calculate_bleu(decoded_preds, decoded_labels, avg=True)

        result = scores

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result
