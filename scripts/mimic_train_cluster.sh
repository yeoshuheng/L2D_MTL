#!/bin/bash

cd "$(dirname "$0")"

python ../train_classifier_mimic.py \
    --dataset mimic_4  \
    --classifier_name transformer \
    --expert_exp Clusters \
    --batch_size 32 \
    --NB_experts 2 \
    --lr 0.0001 \
    --path_dataset ../data/mimic_4/mimic-iv-demo/2.2/hosp \
    --epochs 5 \
    --save_freq 1 \
    --log_freq 1 \
    --lambda_1 1 \
    --lambda_2 0.2 \
    --warmup 0.1 \
    --n_points 100\
    --save 1 

python ../train_two_stage_mimic.py \
    --dataset mimic_4  \
    --classifier_name transformer \
    --expert_exp Clusters \
    --NB_experts 2 \
    --batch_size 32 \
    --path_dataset ../data/mimic_4/mimic-iv-demo/2.2/hosp \
    --lr 0.0001 \
    --epochs 5 \
    --save_freq 1 \
    --log_freq 1 \
    --lambda_1 1 \
    --lambda_2 0.2 \
    --alpha 1 \
    --beta 0 \
    --warmup 0.1 \
    --n_points 100 \
    --save 1 
