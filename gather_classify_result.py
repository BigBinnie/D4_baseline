import glob
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize 
import argparse

def gather_classify(src_dir):
    files = sorted(glob.glob(f"{src_dir}/*/result.json"))
    for file in files:
        with open(file) as f:
            ans = json.load(f)
        preds = eval(ans["preds"])
        labels = eval(ans["labels"])
        print(f"======== {file} ========")
        print(classification_report(labels, preds))
        print()
    
    acc = 0
    for index in range(len(preds)):
        acc += (preds[index] == labels[index])
    print(f"{acc} / {len(preds)} = {(acc / len(preds)):.3%}")

def main():
    parser = argparse.ArgumentParser('Gather Summary Result',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src_dir', type=str, required=True,
                        help='file directory to preprocess')
    args = parser.parse_args()
    
    gather_classify(args.src_dir)


if __name__ == '__main__':
    main()
