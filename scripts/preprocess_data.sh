#!/bin/bash
# pre-process data
raw_data_directory="./D4"
target_directory="./data"
topic_num=6

python convert_dataset_dialog.py \
--src_dir ${raw_data_directory} \
--tgt_dir ${target_directory} \
--add_topic true \
--topic_num ${topic_num}

python convert_dataset_summary.py \
--src_dir ${raw_data_directory} \
--tgt_dir ${target_directory} \

