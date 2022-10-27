#!/bin/bash
#SBATCH --job-name=train_classify_model
#SBATCH --partition=2080ti,gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=90G
#SBATCH --output=exp_summary_classify/train_classify_model-%A-%a.out
#SBATCH --error=exp_summary_classify/train_classify_model-%A-%a.err
#SBATCH --exclude=gqxx-01-[027,118,175,085]
#SBATCH --array=2,4

hostname
HOME=/mnt/lustre/sjtu/home/lfd98
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=4

if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 0 ]; then
  RISK=drisk
else
  RISK=srisk
fi

# train_text=data/summary/train.json
# dev_text=data/summary/val.json
# test_text=data/summary/test.json

train_text=data/summary_labelled/train.json
dev_text=data/summary_labelled/val.json
test_text=data/summary_labelled/test.json
n_class=4

case ${SLURM_ARRAY_TASK_ID} in
1|2)
  MODEL_TYPE=bart-base
  MODEL_PATH="${HOME}/Dataset/transformers/bart-base-chinese"
  ;;
3|4)
  MODEL_TYPE=cpt-base
  MODEL_PATH="${HOME}/Dataset/transformers/cpt-base"
  ;;
5|6)
  MODEL_TYPE=bart-large
  MODEL_PATH="${HOME}/Dataset/transformers/bart-large-chinese"
  per_device_train_batch_size=1
  per_device_eval_batch_size=2
  gradient_accumulation_steps=8
  ;;
7|8)
  MODEL_TYPE=cpt-large
  MODEL_PATH="${HOME}/Dataset/transformers/cpt-large"
  per_device_train_batch_size=1
  per_device_eval_batch_size=1
  gradient_accumulation_steps=8
  ;;
9|10)
  n_class=2
  MODEL_TYPE=bert-base
  MODEL_PATH="${HOME}/Dataset/transformers/chinese-macbert-base"
  train_text=data/summary2/train.json
  dev_text=data/summary2/val.json
  test_text=data/summary2/test.json
  ;;
11|12)
  n_class=4
  MODEL_TYPE=bert-base
  MODEL_PATH="${HOME}/Dataset/transformers/chinese-macbert-base"
  train_text=data/summary4/train.json
  dev_text=data/summary4/val.json
  test_text=data/summary4/test.json
  ;;
*)
  echo "Unrecognized task id: ${SLURM_ARRAY_TASK_ID}"
  exit 1
  ;;
esac

python \
  -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=$((SLURM_ARRAY_TASK_ID + 8889)) \
main.py \
  --output_dir  "exp_summary_classify/${RISK}${n_class}/${MODEL_TYPE}" \
  --do_train    true \
  --do_eval     true \
  --do_predict  true \
  --evaluation_strategy     epoch \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size  ${per_device_eval_batch_size} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --eval_accumulation_steps     4 \
  --learning_rate               1e-5 \
  --weight_decay                1e-6 \
  --num_train_epochs            30 \
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
  --ddp_find_unused_parameters  false \
  --dataloader_pin_memory       false \
  \
  --metric_for_best_model       f1-score \
  --greater_is_better           true \
  \
  --predict_with_generate       false \
  --generation_max_length       128 \
  --generation_num_beams        4 \
  \
  --model_type    "${MODEL_TYPE}" \
  --model_path    "${MODEL_PATH}" \
  \
  --n_class       ${n_class} \
  --max_len       512 \
  --add_portrait  false \
  --task          "${RISK}" \
  --train_text    ${train_text} \
  --dev_text      ${dev_text}\
  --test_text     ${test_text}
