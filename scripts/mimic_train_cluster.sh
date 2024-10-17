#!/bin/bash

cd "$(dirname "$0")"

python ../train_two_stage_mimic.py \
    --dataset mimic_4  \
    --classifier_name transformer \
    --expert_exp Clusters \
    --NB_experts 2 \
    --batch_size 1024 \
    --path_dataset ../data/mimic_4/mimic-iv-demo/2.2/hosp \
    --name_classifier ./exp_transformer_42_lr_0.001_lambda_cla1.0/model_eval_steps_0.pth \
    --lr 0.001 \
    --epochs 75 \
    --save_freq 1 \
    --lambda_1 1 \
    --lambda_2 0.2 \
    --bound 5 \
    --alpha 1 \
    --beta 0 \
    --warmup 0.1 \
    --save 1 \
    --seed 42 \
    --dev 1 \
    --overfit 0 \
    --test 0

