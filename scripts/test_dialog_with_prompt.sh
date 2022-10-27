#!/bin/bash
MODEL_TYPE=""
CKPT_PATH=""
SAVE_TO=""
TEST_FILE=""

python test_dialog_with_prompt.py \
  --model_type  ${MODEL_TYPE} \
  --ckpt_path   ${CKPT_PATH} \
  --save_to     ${SAVE_TO} \
  --test_data   ${TEST_FILE}
