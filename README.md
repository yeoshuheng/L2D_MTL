# Learning-To-Defer For Multi-Task Problems

This repository contains the code for Learning-To-Defer for Multi-Task Problems.

## Setup
Firstly, install the requirements with 
```bash
pip install requirements.txt
```
Download the full MIMIC-IV dataset and add it into the `./data/mimic_4`folder.
Afterwards, unzip the downloaded data into the folder within `./data/mimic_4`, and update the `--path_dataset` argument when running the scripts.

Setup a WandB account and set your public API key as a environment variable.
```bash
export WANDB_API_KEY=yourPublicApiKey
```

## Training
We provide an example for training both the classifier & two-stage system, as well as evaluating their performance. Run the script with,
```bash
bash mimic_train.sh
```
The `--expert_exp` argument determines the training setting, `Oracle` represents the Single-Expert oracle experiment whereas `Cluster` represents the multi-expert cluster setting.

## Testing
We also provide a pre-trained two-stage L2D model to replicate the results within the paper. To evaluate this checkpoint, run the script
```bash
bash mimic_evaluate.sh
```