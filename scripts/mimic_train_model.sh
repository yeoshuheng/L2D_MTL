#!/bin/bash

cd "$(dirname "$0")"

python ../train_classifier_mimic.py \
    --dataset mimic_4  \
    --classifier_name transformer \
    --batch_size 1024 \
    --path_dataset ../data/mimic_4/mimic-iv-demo/2.2/hosp \
    --lr 0.001 \
    --epochs 75 \
    --save_freq 1 \
    --lambda_1 1 \
    --lambda_2 0.2 \
    --bound 5 \
    --warmup 0.1 \
    --save 1 \
    --seed 42 \
    --dev 1 \
    --overfit 0 \

