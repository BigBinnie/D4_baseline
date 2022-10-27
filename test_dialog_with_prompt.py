import argparse
import json

import torch
from torch.utils.data import DataLoader

from main import get_model_and_tokenizer
from utils.data_process import DialogDataset, CollateFnForDialog


def generate(model_type: str, ckpt_path: str, dataset: DialogDataset, save_to: str):
    print(model_type)
    model, tokenizer = get_model_and_tokenizer("dialog", model_type, ckpt_path, n_class=-1)
    _collate = CollateFnForDialog(tokenizer, 512, truncation="right", action_prompt=True)
    model.cuda().eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate.collate_fn)

    result = []

    for i, inputs in enumerate(dataloader):
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.cuda()
        labels = inputs.pop("labels")
        outputs = model.generate(**inputs, num_beams=8, max_length=128)
        pair = [
            tokenizer.batch_decode(outputs)[0].replace("[SEP]", "").replace("[CLS] ", "").strip(),
            tokenizer.batch_decode(labels)[0].replace("[SEP]", "").replace("[CLS] ", "").strip(),
        ]
        result.append(pair)
        if i % 50 == 0:
            print(f"'{i} | {pair[0]}', '{pair[1]}'")

    with open(save_to, "w", encoding="utf8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"Save to {save_to}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    dataset = DialogDataset(args.test_data)

    generate(args.model_type, args.ckpt_path, dataset, args.save_to)


if __name__ == '__main__':
    main()
