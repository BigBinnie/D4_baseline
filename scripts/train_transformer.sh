#!/bin/bash
MODEL_PATH=""
OUTPUT_PATH=""
train_data_path=""
val_data_path=""
test_data_path=""

python transformer/train.py \
	--reload_from=$MODEL_PATH \
	--do_train \
	--do_eval \
	--start_epoch 0 \
	--epochs 20 \
	--seed 233 \
	--output_path $OUTPUT_PATH \
	--train_data_path $train_data_path \
	--val_data_path $val_data_path \
	--test_data_path $test_data_path \
