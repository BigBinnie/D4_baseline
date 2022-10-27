#!/bin/bash

per_device_train_batch_size=2
per_device_eval_batch_size=8
gradient_accumulation_steps=4

MODEL_TYPE=bart-base
MODEL_PATH=""
train_data=""
valid_data=""
test_data=""
output_dir=""

python \
  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main.py \
  --output_dir  "${output_dir}" \
  --do_train    true \
  --do_eval     true \
  --do_predict  true \
  --evaluation_strategy     epoch \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size  ${per_device_eval_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --learning_rate               1e-5 \
  --weight_decay                1e-6 \
  --num_train_epochs            50 \
  --lr_scheduler_type           cosine \
  --warmup_steps                100 \
  --logging_steps               10 \
  --save_strategy               epoch \
  --save_total_limit            1 \
  --load_best_model_at_end      true \
  --seed                        42 \
  --dataloader_num_workers      3 \
  --disable_tqdm                true \
  --label_smoothing_factor      0 \
  --ddp_find_unused_parameters  true \
  --dataloader_pin_memory       false \
  \
  --metric_for_best_model       rouge-1 \
  --greater_is_better           true \
  \
  --predict_with_generate       true \
  --generation_max_length       200 \
  --generation_num_beams        6 \
  \
  --model_type    "${MODEL_TYPE}" \
  --model_path    "${MODEL_PATH}" \
  \
  --max_len       512 \
  --add_portrait  true \
  --task          summary4 \
  --train_text    ${train_data} \
  --dev_text      ${valid_data} \
  --test_text     ${valid_data}
