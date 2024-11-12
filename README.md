
# Two-Stage Learning-To-Defer for Multi-Task Learning

This repository contains the code for the **Two-Stage Learning-To-Defer Multi-Task Learning** project.

Our paper can be found [here](https://arxiv.org/abs/2410.15729v2).

## Setup

### Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Dataset

1. Download the full MIMIC-IV dataset from [PhysioNet](https://physionet.org/content/mimiciv/3.0/). You will need to create an account and request access to the dataset. 
We provide a public demo subset of this dataset in the `./data/mimic_4` folder for testing purposes.

2. Once you have downloaded the full dataset, place it into the `./data/mimic_4` folder.
3. Unzip the downloaded data within the `./data/mimic_4` folder.
4. Update the `--path_dataset` argument in the scripts to point to the correct dataset path when running them (hosp folder).

### Weights & Biases Setup

1. Set up a [Weights & Biases](https://wandb.ai/) account for experiment tracking.
2. Set your WandB API key as an environment variable:
```bash
export WANDB_API_KEY=yourPublicApiKey
```
3. Replace `yourPublicApiKey` with your actual WandB API key.

## Training

We provide scripts for training the two-stage system from a pre-trained classifier and evaluating its performance.

The `--expert_exp` argument determines the training setting:
- `Oracle`: Represents the **Oracle Expert** experiment.
- `Cluster`: Represents the **Specialized Experts** setting.

To train and evaluate the **Specialized Experts** setting, run:

```bash
bash mimic_train_cluster.sh
```

For the **Oracle Expert setting**, run:

```bash
bash mimic_train_oracle.sh
```

Hyperparameters can be adjusted in the scripts. 
