#!/bin/bash

EPOCH=""
MODEL_PATH=""
OUTPUT_PATH=""
TEST_FILE=""
python transformer/generate.py \
	--model_name $MODEL_PATH \
	--output_path $OUTPUT_PATH \
	--test_file $TEST_FILE