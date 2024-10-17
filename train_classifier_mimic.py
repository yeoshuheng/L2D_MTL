import os
import torch
import torch.nn as nn
import argparse as arg
import random
import numpy as np
from transformers import AlbertTokenizerFast, get_scheduler
from tqdm import tqdm
import wandb
from losses.mimic import MimicLoss
from models.mimic.mlp import ClassifierRegressor, MLP
from dataset.mimic_4_dataset import preprocess_data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def setup_wandb(args, name, param):
    wandb.init(project=args.name_project, name=name)
    wandb.define_metric('custom_epoch')
    wandb.define_metric('steps')
    wandb.define_metric('Loss/*', step_metric='custom_epoch')
    wandb.define_metric('Val_Loss/*', step_metric='steps')
    config_wandb = wandb.config
    config_wandb.lambda_1 = param['lambda_1']
    config_wandb.lambda_2 = args.lambda_2
    config_wandb.lr = param['lr']
    config_wandb.name = args.batch_size
    config_wandb.epochs = args.epochs
    config_wandb.expert_exp = args.expert_exp
    config_wandb.NB_experts = args.NB_experts
    config_wandb.seed = args.seed
    config_wandb.warmup = args.warmup
    config_wandb.classifier_name = args.classifier_name
    config_wandb.log_freq = args.log_freq
    config_wandb.save_freq = args.save_freq
    config_wandb.dev = args.dev
    config_wandb.overfit = args.overfit
    config_wandb.name_project = args.name_project
    return wandb

def setup_args():
    parser = arg.ArgumentParser(description='MIMIC4: Classifier Training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, nargs='+' , default=[0.0001, 0.01], help='Learning rate')
    parser.add_argument('--dataset', type=str, default='mimic_4', help='Dataset type') #/media/yannis/T7 Touch/physionet.org/files/mimiciv/3.0/hosp

    parser.add_argument('--n_points', type=int, default=100, help='Number of points to use')
    parser.add_argument("--classifier_name", type=str, default="transformer", help='Classifier name')
    parser.add_argument("--NB_experts", type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--warmup', type=float, default=0.1, help='warmup')
    parser.add_argument('--expert_exp', type=str, default='Oracle', help='expert type')
    parser.add_argument('--lambda_1', type=float, nargs='+' , default=[1], help='lambda_pred')
    parser.add_argument('--lambda_2', type=float, default=1, help='lambda_regression')
    parser.add_argument("--log_freq", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=2)
    parser.add_argument("--dev", type=int, default=0) # if 0 we use dev set (subsample) 1 is for full dataset
    parser.add_argument('--name', type=str, default='exp', help='Name experiment')
    parser.add_argument('--overfit', type=int, default=0, help='overfit: 1 ')
    parser.add_argument('--name_project', type=str, default='training_mimic', help='Name')
    parser.add_argument('--path_dataset', type=str, default='/data/mimic_4/',
                        help='Path to dataset')
    parser.add_argument('--extra_data', type=int, default=0, help='Extra data')
    parser.add_argument('--loss', type=str, default='l1', help='l1/smape')
    args = parser.parse_args()
#'/media/yannis/T7 Touch/physionet.org/files/mimic-iv-demo/2.2/hosp/'
#'/media/yannis/T7 Touch/1.4'
    args.lr = [float(i) for i in args.lr]
    args.lambda_1 = [float(i) for i in args.lambda_1]
    return args

def train_loop(args, device, param, dataloader, dataset, name):

    dir_name = f"./logs/train/{args.dataset}/{name}"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if args.classifier_name == "mlp":
        model_init = MLP(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key='label_mortality_class',  # ignore this for now
            mode="multiclass",
        )
    if args.classifier_name == "transformer":
        import models.mimic.trans_v2 as tr
        model_init = tr.Transformer(
            dataset=dataset,
            # look up what are available for "feature_keys" and "label_keys" in dataset.samples[0]
            feature_keys=["conditions", "procedures"],
            label_key="label_mortality_class",
            mode="binary",
        )

    model = nn.DataParallel(model_init.to(device))


    warmup_proportion = args.warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'])

    num_training_steps = args.epochs * len(dataloader['train'])
    num_warmup_steps = int(warmup_proportion * num_training_steps)
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    criterion = MimicLoss(param['lambda_1'], args.lambda_2, args)
    steps = 0
    N = len(dataloader['train'])
    for epoch in tqdm(range(args.epochs)):
        cls_loss, reg_loss, running_loss = 0, 0, 0
        for i, x in enumerate(dataloader['train']):
            model.train()
            optimizer.zero_grad()
            loss, loss_dict, _ = criterion.get_loss(model, x, device)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if steps % args.save_freq == 0:
                eval_loop(args, device, model, dataloader, criterion, dir_name, steps)

            cls_loss = loss_dict["cls_loss"] + cls_loss
            reg_loss = loss_dict["reg_loss"] + reg_loss
            running_loss = loss.item() + running_loss
            steps = 1 + steps
        print(f"cls_loss: {cls_loss/N} | reg_loss: {reg_loss/N} | total: {running_loss/N}")
        wandb.log({
            "custom_epoch" : epoch,
            "Loss/cls_loss" : cls_loss/N,
            "Loss/reg_loss" : reg_loss/N,
            "Loss/total_loss" : running_loss/N
        })


    checkpoint_name = "model_final.pth"
    save_path = os.path.join(dir_name, checkpoint_name)
    torch.save(model.state_dict(), save_path)

def eval_loop(args, device, model, dataloader, criterion, dir_name, steps):
    model.eval()
    cls_loss, reg_loss, running_loss = 0, 0, 0
    output_list = []
    for i, x in enumerate(dataloader['val']):
        loss, loss_dict, output = criterion.get_loss(model, x, device)
        cls_loss = loss_dict["cls_loss"] + cls_loss
        reg_loss = loss_dict["reg_loss"] + reg_loss
        running_loss = loss.item() + running_loss
        output_list.append(output)
    wandb.log({
        "steps": steps,
        "Val_Loss/cls_loss": cls_loss/len(dataloader['val']),
        "Val_Loss/reg_loss": reg_loss/len(dataloader['val']),
        "Val_Loss/total_loss": running_loss/len(dataloader['val'])
    })
    checkpoint_name = f"model_eval_steps_{steps}.pth"
    save_path = os.path.join(dir_name, checkpoint_name)
    torch.save(model.state_dict(), save_path)

    # Metrics evaluation
    accuracy, balance_accuracy, f1 = metrics(output_list)
    wandb.log({
        "Val_Metrics/accuracy": accuracy,
        "Val_Metrics/balance_accuracy": balance_accuracy,
        "Val_Metrics/f1_0": f1[0],
        "Val_Metrics/f1_1": f1[1],
        'steps': steps,
        'Val_Metrics/L1': reg_loss/len(dataloader['val']),
    })

def metrics(output_list):
    classifier_fc_list = torch.cat([item['classifier_fc'] for item in output_list], dim=0)
    classifier_preds = torch.argmax(classifier_fc_list, dim=1).cpu().numpy()
    labels_class_list = torch.cat([item['label_class'] for item in output_list], dim=0).cpu().numpy()
    # Calculate accuracy
    _, _, f1, _ = precision_recall_fscore_support(labels_class_list, classifier_preds, average=None, zero_division=0)
    balance_accuracy = balanced_accuracy_score(labels_class_list, classifier_preds)
    accuracy = accuracy_score(labels_class_list, classifier_preds)
    return accuracy, balance_accuracy, f1

def main():
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    args = setup_args()
    random_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.device = device
    data_loader, dataset = preprocess_data(args)

    for i in range(len(args.lr)):
        for j in range(len(args.lambda_1)):
            name = f"{args.name}_{args.classifier_name}_{args.seed}_lr_{args.lr[i]}_lambda_cla{args.lambda_1[j]}"
            setup_wandb(args, name, {"lr" : args.lr[i], 'lambda_1':args.lambda_1[j]})
            train_loop(args, device, {"lr" : args.lr[i], 'lambda_1':args.lambda_1[j]}, data_loader, dataset, name)
            wandb.finish()


if __name__ == "__main__":
    main()

