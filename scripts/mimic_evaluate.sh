#!/bin/bash

cd "$(dirname "$0")"

python statistics_model_mimic.py \
    --dataset mimic_4 \
    --classifier transformer \
    --path_dataset data/mimic_4/mimic-iv-demo/hosp \
    --expert_exp Cluster \
    --classifier_name model_eval_steps_80.pth \
    --path_rejector \ temp
    --two_stage 0 \
    --batch_size 8 \
    --lr 0.0001 \
    --epochs 20 \
    --save_freq 10 \
    --log_freq 5 \
    --lambda_1 1 \
    --lambda_2 1 \
    --alpha 1 \
    --beta 0 \
    --warmup 0.1 \
    --n_points 100 
