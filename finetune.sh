#! /bin/bash
dataset_name=${1:-"testForDelete"}
dataset_dir=${2:-"/root/autodl-tmp/Flow-MAE/data/encryptedtestData/"}
test_ratio=${3:-0.1}
THIS_DIR=$(dirname "$(readlink -f "$0")")


torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  "$THIS_DIR"/finetune.py \
  --dataset_name "$dataset_name" \
  --dataset_dir "$dataset_dir" \
  --train_dir "/root/autodl-tmp/Flow-MAE/data/PLUS_VPN_TOPtest/pretrain4096_0.75train.pkl" \
  --validation_dir "/root/autodl-tmp/Flow-MAE/data/PLUS_VPN_TOPtest/pretrain4096_0.75test.pkl" \
  --train_val_split "$test_ratio" \
  --dataloader_num_workers 4 \
  --output_dir "./vit-mae-finetune100PLUS_VPN_TOPtest" \
  --overwrite_output_dir \
  --model_name_or_path "/root/autodl-tmp/Flow-MAE/vit-mae-4097/checkpoint-360500" \
  --return_entity_level_metrics True \
  --remove_unused_columns False \
  --num_channels 1 \
  --num_attention_heads 2 \
  --hidden_dropout_prob 0.1 \
  --attention_probs_dropout_prob 0.1 \
  --mask_ratio 0 \
  --image_column_name "layers_layerData" \
  --norm_pix_loss \
  --do_train \
  --do_eval \
  --base_learning_rate 1e-4 \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.08 \
  --num_train_epochs 100 \
  --warmup_ratio 0.0 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --logging_strategy steps \
  --logging_steps 50 \
  --evaluation_strategy "epoch" \
  --save_strategy "epoch" \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_f1" \
  --greater_is_better True \
  --save_total_limit 3 \
  --seed 1337

#deepspeed --num_gpus=2 \
#  "$THIS_DIR"/pretrain.py \
#  --deepspeed "$THIS_DIR"/pretrain/ds_config.json \
#  --dataset_name "/root/PycharmProjects/DATA/IDS2018Pretrain_single" \
#  --output_dir "./vit-mae-demo" \
#  --overwrite_output_dir \
#  --remove_unused_columns False \
#  --num_channels 1 \
#  --mask_ratio 0.15 \
#  --image_column_name "tcp.payload" \
#  --norm_pix_loss \
#  --do_train \
#  --do_eval \
#  --base_learning_rate 1.5e-4 \
#  --lr_scheduler_type "cosine" \
#  --weight_decay 0.05 \
#  --num_train_epochs 100 \
#  --warmup_ratio 0.01 \
#  --per_device_train_batch_size 128 \
#  --per_device_eval_batch_size 32 \
#  --logging_strategy steps \
#  --logging_steps 10 \
#  --evaluation_strategy "epoch" \
#  --save_strategy "epoch" \
#  --load_best_model_at_end True \
#  --save_total_limit 3 \
#  --seed 1337 \
#  --fp16
