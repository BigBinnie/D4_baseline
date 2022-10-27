import imp
import json
import re
from typing import List, Dict, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, BartTokenizer


class DialogDataset(Dataset):
    def __init__(self, filename: str):
        with open(filename, "r", encoding="utf8") as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class _CollateFn:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len=512, truncation="mid"):
        # assert len(tokenizer.additional_special_tokens) == 3, \
            # f"3 special tokens required, got {len(tokenizer.additional_special_tokens)}"

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation

    def batch_encode(self, sentences: List[str]) -> dict:
        input_ids = [self.tokenizer.encode(s) for s in sentences]
        attention_mask = []
        pad_token_id = self.tokenizer.pad_token_id

        max_len = min(self.max_len, max(map(len, input_ids)))

        # Do padding & truncating
        for i, ids in enumerate(input_ids):
            _len = len(ids)
            if _len >= max_len:
                if self.truncation == "mid":
                    # Pick the middle segment
                    ids = ids[(_len - max_len) // 2: (_len + max_len) // 2]
                    ids[0] = self.tokenizer.cls_token_id
                    ids[-1] = self.tokenizer.sep_token_id
                elif self.truncation == "left":
                    ids = ids[:max_len]
                    ids[-1] = self.tokenizer.sep_token_id
                else:
                    ids = ids[-max_len:]
                    ids[0] = self.tokenizer.cls_token_id
                input_ids[i] = ids
                attention_mask.append([1] * max_len)
            else:
                input_ids[i] = ids + [pad_token_id] * (max_len - _len)
                attention_mask.append([1] * _len + [0] * (max_len - _len))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

    def collate_fn(self, batch: List[Dict[str, str]]) -> Union[dict, BatchEncoding]:
        raise NotImplementedError


class CollateFnForDialog(_CollateFn):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len=512, truncation="right", action_prompt=False):
        super().__init__(tokenizer, max_len, truncation)
        self.action_prompt = action_prompt
        self._eos_token_id = tokenizer("test")["input_ids"][-1]

    def collate_fn(self, batch: List[Dict[str, str]]) -> Union[dict, BatchEncoding]:
        srcs = [item["src"] for item in batch]
        tgts = [item["tgt"] for item in batch]

        if self.action_prompt:
            assert len(srcs) == 1, "When add action prompt, the batch size is limit to 1."

        inputs = self.batch_encode(srcs)

        with self.tokenizer.as_target_tokenizer():
            labels = self.batch_encode(tgts)
            if self.action_prompt:
                doc_bos_idx = torch.where(labels["input_ids"] == self.tokenizer.convert_tokens_to_ids("<doc_bos>"))
                tmp = labels["input_ids"][:, :doc_bos_idx[1] + 2].clone()
                tmp[:, 1:] = tmp[:, :-1].clone()
                tmp[:, 0] = self._eos_token_id
                inputs["decoder_input_ids"] = tmp
            inputs["labels"] = labels["input_ids"]

        return inputs


class CollateFnForSummary(_CollateFn):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len=512, truncation="mid", add_portrait=True):
        super().__init__(tokenizer, max_len, truncation)
        self.add_portrait = add_portrait

    @staticmethod
    def process_portrait(item: dict) -> str:
        portrait = item["portrait"]
        return f"{portrait['age']}岁，" \
               f"性别{portrait['gender']}，" \
               f"{portrait['martial_status']}，" \
               f"{portrait['occupation']}"

    def collate_fn(self, batch: List[Dict[str, Union[str, dict]]]) -> Union[dict, BatchEncoding]:
        srcs, tgts = [], []

        for item in batch:
            dialog = item["dialog"]
            summary = item["summary"]
            srcs.append(f"{dialog}|{self.process_portrait(item)}" if self.add_portrait else dialog)
            tgts.append(summary)

        inputs = self.batch_encode(srcs)

        with self.tokenizer.as_target_tokenizer():
            labels = self.batch_encode(tgts)
            inputs["labels"] = labels["input_ids"]

        return inputs


class CollateFnForClassify(_CollateFn):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len=512, truncation="mid", add_portrait=True, risk="drisk"):
        assert risk in ("drisk", "srisk")
        super().__init__(tokenizer, max_len, truncation)
        self.add_portrait = add_portrait
        self.risk = risk

    def collate_fn(self, batch: List[Dict[str, Union[str, dict]]]) -> Union[dict, BatchEncoding]:
        srcs, tgts = [], []

        for item in batch:
            dialog = item["dialog"]
            # dialog = item["summary"]
            srcs.append(f"{dialog}|{CollateFnForSummary.process_portrait(item)}" if self.add_portrait else dialog)
            tgts.append(item[self.risk])

        inputs = self.batch_encode(srcs)
        inputs["labels"] = torch.tensor(tgts, dtype=torch.long)

        return inputs

class CollateFnForPrediction(_CollateFn):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len=512, truncation="right"):
        super().__init__(tokenizer, max_len, truncation)
    
    # def map_label(self, label):
    #     item_map = {"共情安慰":0, "兴趣": 1, "其它": 2, "情绪": 3, "睡眠": 4, "社会功能": 5,
    #             "筛查": 6, "精神状态": 7, "自杀倾向": 8, "躯体症状":9, "食欲": 10} 
    #     if label in item_map:
    #         return item_map[label]
    #     else:
    #         return 2
    def map_label(self, label):
        item_map = {"共情安慰":0, "核心": 1, "其它": 2, "行为": 3, "自杀倾向": 4,
                "筛查": 5} 
        if label in item_map:
            return item_map[label]
        else:
            return 2

    def collate_fn(self, batch: List[Dict[str, Union[str, dict]]]) -> Union[dict, BatchEncoding]:
        srcs, tgts = [], []

        srcs = [item["src"] for item in batch]
        for item in batch:
            try:
                topic = ''.join(re.search("<act>(.*?)<", item["tgt"]).group(1).split())
                print(item["tgt"], topic, "\n")
                tgts.append(self.map_label(topic))
            except AttributeError:
                tgts.append(self.map_label("其它"))
                print(f"No action found in: ",item["tgt"])
        print(len(tgts),tgts)
        inputs = self.batch_encode(srcs)
        inputs["labels"] = torch.tensor(tgts, dtype=torch.long)

        return inputs
