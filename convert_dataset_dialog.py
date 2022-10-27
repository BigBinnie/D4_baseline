# generate tod data
import json
from xmlrpc.client import Boolean
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List
import numpy as np
import argparse
import os

skip_doc_utterence_in_context = False
skip_action_in_context = False
simulate_patient = False
add_portrait = False
# custom tokens
TOK_DOC_BOS = "<doc_bos>"
TOK_PAT_BOS = "<pat_bos>"
TOK_ACT = "<act>"
PORTRAIT_ACT = "<por>"
action_stat_dict = defaultdict(int)

def get_topic(topic, topic_num):
    if topic_num == 11:
        return topic
    elif topic_num == 2:
        return get_empathy_topic(topic)
    
    _labels = ['兴趣','情绪','精神状态', '社会功能','躯体症状', '睡眠','食欲',
        '筛查', '自杀倾向', '共情安慰', '其它']
    index = _labels.index(topic)
    if index == -1:
        topic = "其它"
    elif index >= 0 and index <= 3:
        topic = "核心"
    elif index >=4 and index <= 6:
        topic = "行为"
    return topic

def get_empathy_topic(topic):
    _labels = ['兴趣','情绪','精神状态', '社会功能','躯体症状', '睡眠','食欲',
               '筛查', '自杀倾向', '共情安慰', '其它']
    if _labels.index(topic) == 9:
        return topic
    else: 
        return "其它"

def process_portrait(portrait: dict) -> str:
    return f"{portrait['age']}岁，" \
            f"性别{portrait['gender']}，" \
            f"{portrait['martial_status']}，" \
            f"{portrait['occupation']}，" \
            f"{portrait['symptoms']}"
        
def main():
    parser = argparse.ArgumentParser('Data preprocess',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--src_dir', type=str, required=True,
                        help='file directory to preprocess')
    parser.add_argument('-t', '--tgt_dir', type=str, required=True,
                        help='target directory')
    parser.add_argument('-a', '--add_topic', type=Boolean, required=True,
                        help='whether or not add topic')
    parser.add_argument('-n', '--topic_num', type=int, default=11, help='topic number', choices= [2, 6, 11])
    args = parser.parse_args()
    
    skip_action_in_context = not args.add_topic
    
    os.makedirs(f"{args.tgt_dir}/dialog/{args.topic_num}topic", exist_ok=True)

    for filename in ["train", "test", "val"]:
        with open(f"{args.src_dir}/raw_data_{filename}.json", "r", encoding='utf-8') as data_file:
            data = json.load(data_file)
            output = []

            for dialog in tqdm(data):
                grouped_messages = []
                tmp = []
                last_speaker = None
                portrait: Dict[str, str] = dialog["portrait"]
                for message in dialog["log"]:
                    speaker = message["speaker"]
                    if speaker != last_speaker and len(tmp) > 0:
                        grouped_messages.append(tmp)
                        tmp = []
                    last_speaker = speaker
                    tmp.append(message)
                if len(tmp) > 0:
                    grouped_messages.append(tmp)

                context = ""
                for speaker_messages in grouped_messages:
                    speaker = speaker_messages[0]["speaker"]
                    if speaker == "patient":
                        patient_str = (
                            TOK_PAT_BOS
                            +
                            " ".join([message["text"] for message in speaker_messages])
                        )
                        if simulate_patient:
                            if add_portrait:
                                output.append({"src": context + PORTRAIT_ACT + process_portrait(portrait), "tgt": patient_str})
                            else:
                                output.append({"src": context, "tgt": patient_str})
                        context += patient_str
                    else:
                        actions = []
                        utterances = []
                        for msg in speaker_messages:
                            action_str = (
                                TOK_ACT + " " + msg["action"] if msg["action"] else ""
                            )
                            a = msg["action"]
                            if a != None:
                                if a != '其它':
                                    actions.append(a)
                                utterances.append(msg['text'])
                        action_stat_dict[len(set(actions))] += 1
                        action_str = actions[-1] if len(actions) > 0 and actions[-1]!='' else "其它"
                        action_str = get_topic(action_str, args.topic_num)

                        utt_str = " ".join(utterances)
                        if args.add_topic:
                            doctor_str = TOK_ACT + action_str + TOK_DOC_BOS + utt_str
                        else:
                            doctor_str = TOK_DOC_BOS + utt_str
                        if simulate_patient != True:
                            output.append({"src": context, "tgt": doctor_str})
                        if not skip_action_in_context:
                            context += TOK_ACT + action_str
                        if not skip_doc_utterence_in_context:
                            context += TOK_DOC_BOS + utt_str
            
            json.dump(
                output,
                open(f"{args.tgt_dir}/dialog/{args.topic_num}topic/{filename}_test.json", "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            with open(f"{args.tgt_dir}/dialog/{args.topic_num}topic/statistics.txt", "a", encoding='utf-8') as logfile:
                logfile.write(
                    f"{filename} number of pairs: {len(output)}, #utterances {len(output) * 2}\n"
                )
                logfile.write(str(action_stat_dict))


if __name__ == '__main__':
    main()
