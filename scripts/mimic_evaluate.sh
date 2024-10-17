#!/bin/bash

cd "$(dirname "$0")"


python ../statistics_model_mimic.py \
    --dataset mimic_4 \
    --classifier_name transformer \
    --path_dataset ../data/mimic_4/mimic-iv-demo/2.2/hosp \
    --expert_exp Oracle \
    --NB_experts 1 \
    --name_classifier model_eval_final.pth \
    --path_rejector ./logs/train/mimic_4/exp_transformer_42_lr_0.0001_lambda_cla1.0/model_final.pth \
    --batch_size 8 \
    --lambda_1 1 \
    --lambda_2 0.2 \
    --alpha 1 \
    --beta 0 
