import json
import sys
import os
from dataclasses import dataclass, field
from typing import Optional
from rouge.rouge import Rouge
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer

from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from nltk.util import ngrams

from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import (
    TOD_SPECIAL_TOKENS,
    add_special_tokens_to_tokenizer,
    preprocess_json,
    ContextResponseDataset,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from model import Transformer


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain"""
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )


@dataclass
class TestArgs:
    model_name: str = field()
    test_file: str = field(default="dataset/tod_test.json")
    no_cuda: bool = field(default=False)
    output_path: Optional[str] = field(default=None)


def bleu(predict, target, n):
    return sentence_bleu([target], predict, weights=tuple(1 / n for i in range(n)))


def nist(predict, target, n):
    if len(predict) < n or len(target) < n:
        return 0
    return sentence_nist([target], predict, n)


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g) - n):
                ngram = " ".join(g[idx : idx + n + 1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += -(v + 0.0) / total * (np.log(v + 0.0) - np.log(total))
        div_score[n] = (len(counter[n].values()) + 0.0) / total
    return etp_score, div_score


def cal_length(sentences):
    sen_length = [len(s) for s in sentences]
    return np.mean(sen_length), np.var(sen_length)


def calculate_metrics(predict, reference, predict_tok, ref_tok):
    # -------------------bleu----------
    bleu_2 = bleu(predict, reference, 2)
    bleu_4 = bleu(predict, reference, 4)
    # -------------------nist----------
    nist_2 = nist(predict, reference, 2)
    nist_4 = nist(predict, reference, 4)
    # -------------------meteor----------
    predict = " ".join(predict)
    reference = " ".join(reference)
    # meteor_scores = meteor_score([ref_tok], predict_tok)
    meteor_scores = 0
    rouge_scores = (
        Rouge().get_scores(predict, reference, avg=True, ignore_empty=True)
    )
    return bleu_2, bleu_4, nist_2, nist_4, meteor_scores, rouge_scores


def sample_generate(args, top_k=50, temperature=1.0):
    decoder_path = args.model_name
    output_path = args.output_path
    # make sure your model is on GPU
    device = torch.device(f"cuda") if not args.no_cuda else torch.device("cpu")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    add_special_tokens_to_tokenizer(tokenizer, TOD_SPECIAL_TOKENS)
    model = Transformer(vocab_size=len(tokenizer))
    model.reload_from(decoder_path, len(tokenizer), device=device)

    model.to(device)
    model.eval()

    test_dataset = [
        preprocess_json(x, tokenizer, "max_length", 400)
        for x in ContextResponseDataset(args.test_file)
    ]
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)
    update_count = 0

    bleu_2scores = 0
    bleu_4scores = 0
    nist_2scores = 0
    nist_4scores = 0

    meteor_scores = 0
    rouge_1f = (
        rouge_1p
    ) = rouge_1r = rouge_2f = rouge_2p = rouge_2r = rouge_lf = rouge_lp = rouge_lr = 0
    sentences = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_result = []
    for batch in tqdm(test_dataloader):
        with torch.no_grad():
            # batch = [item.to(device) for item in batch]

            encoder_input = torch.tensor([batch['input_ids']], device=device)
            decoder_input = torch.tensor([batch['labels']], device=device)
            mask_encoder_input = torch.tensor([batch['attention_mask']], device=device)

            past = model.encoder(encoder_input, mask_encoder_input)
            # doc_bos_idx = torch.where(decoder_input == tokenizer.convert_tokens_to_ids("<doc_bos>"))

            # prev_pred = decoder_input[:, :doc_bos_idx[1] + 1]
            prev_pred = decoder_input[:, :1]
            sentence = prev_pred

            # decoding loop
            for i in range(100):
                outputs = model.decoder(
                    sentence,
                    encoder_hidden_states=past[0],
                )
                logits = model.linear(outputs[0])  # (batch_size, seq_len, vocab_size)
                #                print (logits.size())

                logits = logits[:, -1]  # (batch_size, vocab_size)
                logits = logits.squeeze(1) / temperature

                logits = top_k_logits(logits, k=top_k)
                probs = F.softmax(logits, dim=-1)
                prev_pred = torch.multinomial(probs, num_samples=1)
                sentence = torch.cat([sentence, prev_pred], dim=-1)
                if prev_pred[0][0] == 102:
                    break

            predict = tokenizer.convert_ids_to_tokens(sentence[0].tolist())

            encoder_input = encoder_input.squeeze(dim=0)
            encoder_input_num = (encoder_input != 0).sum()
            inputs = tokenizer.convert_ids_to_tokens(
                encoder_input[:encoder_input_num].tolist()
            )

            decoder_input = decoder_input.squeeze(dim=0)
            decoder_input_num = (decoder_input != 0).sum()

            reference = tokenizer.convert_ids_to_tokens(
                decoder_input[:decoder_input_num].tolist()
            )

            preds_tokens = [
                tokenizer.convert_ids_to_tokens(sample, skip_special_tokens=False)
                for sample in sentence[0].tolist()
            ]
            labels_tokens = [
                tokenizer.convert_ids_to_tokens(sample, skip_special_tokens=True)
                for sample in decoder_input[:decoder_input_num].tolist()
            ]
            predict_text = "".join(predict).replace('[CLS]', '').replace('[SEP]', '')
            reference_text = "".join(reference).replace('[CLS]', '').replace('[SEP]', '')
            output_result.append([predict_text, reference_text])
            json.dump(
                output_result,
                open(os.path.join(output_path, f"result_transformer.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )
            (
                temp_bleu_2,
                temp_bleu_4,
                temp_nist_2,
                temp_nist_4,
                temp_meteor_scores,
                temp_rouge_scores,
            ) = calculate_metrics(
                predict[1:-1], reference[1:-1], preds_tokens, labels_tokens
            )

            bleu_2scores += temp_bleu_2
            bleu_4scores += temp_bleu_4
            nist_2scores += temp_nist_2
            nist_4scores += temp_nist_4

            meteor_scores += temp_meteor_scores
            rouge_1f += temp_rouge_scores["rouge-1"]["f"]
            rouge_1p += temp_rouge_scores["rouge-1"]["p"]
            rouge_1r += temp_rouge_scores["rouge-1"]["r"]
            rouge_2f += temp_rouge_scores["rouge-2"]["f"]
            rouge_2p += temp_rouge_scores["rouge-2"]["p"]
            rouge_2r += temp_rouge_scores["rouge-2"]["r"]
            rouge_lf += temp_rouge_scores["rouge-l"]["f"]
            rouge_lp += temp_rouge_scores["rouge-l"]["p"]
            rouge_lr += temp_rouge_scores["rouge-l"]["r"]

            sentences.append(" ".join(predict[1:-1]))
            update_count += 1

    entro, dist = cal_entropy(sentences)
    mean_len, var_len = cal_length(sentences)

    f = open(os.path.join(output_path, "test_res.txt"), "w")
    print(f"avg: {mean_len}, var: {var_len}")
    print(f"entro: {entro}")
    print(f"dist: {dist}")
    print(f"test bleu_2scores: {bleu_2scores / update_count}")
    print(f"test bleu_4scores: {bleu_4scores / update_count}")
    print(f"test nist_2scores: {nist_2scores / update_count}")
    print(f"test nist_4scores: {nist_4scores / update_count}")
    print(f"test meteor_scores: {meteor_scores / update_count}")
    f.write(f"avg: {mean_len}, var: {var_len}\n")
    f.write(f"entro: {entro}\n")
    f.write(f"dist: {dist}\n")
    f.write(f"test bleu_2scores: {bleu_2scores / update_count}\n")
    f.write(f"test bleu_4scores: {bleu_4scores / update_count}\n")
    f.write(f"test nist_2scores: {nist_2scores / update_count}\n")
    f.write(f"test nist_4scores: {nist_4scores / update_count}\n")
    f.write(f"test meteor_scores: {meteor_scores / update_count}\n")
    f.write(f"test rouge_1f: {rouge_1f / update_count}\n")
    f.write(f"test rouge_1p: {rouge_1p / update_count}\n")
    f.write(f"test rouge_1r: {rouge_1r / update_count}\n")
    f.write(f"test rouge_2f: {rouge_2f / update_count}\n")
    f.write(f"test rouge_2p: {rouge_2p / update_count}\n")
    f.write(f"test rouge_2r: {rouge_2r / update_count}\n")
    f.write(f"test rouge_lf: {rouge_lf / update_count}\n")
    f.write(f"test rouge_lp: {rouge_lp / update_count}\n")
    f.write(f"test rouge_lr: {rouge_lr / update_count}\n")
    f.close()


if __name__ == "__main__":
    parser = HfArgumentParser([TestArgs])
    (args,) = parser.parse_args_into_dataclasses()
    sample_generate(args)
