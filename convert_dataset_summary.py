import json
import os
from typing import Dict, List
import argparse
from tqdm import tqdm
skip_doc_utterence_in_context = False

def process(raw_file: str, summary_file: str):
    with open(raw_file, encoding="utf8") as f:
        sessions = json.load(f)

    for_summary = []  # dialog.keys: src, tgt
    # for_dialog = []  # summary.keys: dialog, portrait, summary

    for session in tqdm(sessions):
        logs: List[Dict[str, str]] = session["log"]
        portrait: Dict[str, str] = session["portrait"]
        portrait.pop("drisk")
        portrait.pop("srisk")
        drisk: int = session["record"]["drisk"]
        srisk: int = session["record"]["srisk"]
        summary: str = session["record"]["summary"]

        history = ""

        for i, log in enumerate(logs):  # type: int, dict
            speaker = log["speaker"]
            action = log["action"]
            text = log["text"]

            if speaker == "doctor":
                # for_dialog.append({
                #     "src": history, "tgt": f"<act>{action}<doc_bos>{text}"
                # })
                # if i + 1 < len(logs) and logs[i + 1]["speaker"] == "patient":
                history += f"<act>{action}<doc_bos>{text}"
            else:
                history += f"<pat_bos>{text}"

        for_summary.append({
            "dialog": history,
            "portrait": portrait,
            "drisk": min(3, drisk),
            "srisk": min(3, srisk),
            "summary": summary
        })

    with open(summary_file, "w", encoding="utf8") as f:
        json.dump(for_summary, f, indent=1, ensure_ascii=False)

    # with open(dialog_file, "w", encoding="utf8") as f:
    #     json.dump(for_dialog, f, indent=1, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser('Data preprocess',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src_dir', type=str, required=True,
                        help='file directory to preprocess')
    parser.add_argument('-t', '--tgt_dir', type=str, required=True,
                        help='target directory')
    args = parser.parse_args()

    os.makedirs(f"{args.tgt_dir}/summary/", exist_ok=True)

    for name in ["val", "train", "test"]:
        raw_file = f"{args.src_dir}/raw_data_{name}.json"
        summary_file = f"{args.tgt_dir}/summary/{name}.json"
        process(raw_file, summary_file)

if __name__ == '__main__':
    main()
