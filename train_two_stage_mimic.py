import os
import torch
import torch.nn as nn
import argparse as arg
import random
import numpy as np
from tqdm import tqdm
from transformers import get_scheduler
import wandb
from losses.two_stage import TwoStage_Mimic, SMAPE_loss
# from models.mimic.mlp import ClassifierRegressor, ClassifierRejector, MLP, Custom
from models.mimic.trans_v2 import Transformer, Transformer_rejector, Custom
from dataset.mimic_4_dataset import preprocess_data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
import pyhealth

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def load_classifier(args, classifier, device, name):
    dir_name = f"./logs/train/{args.dataset}/{args.name_classifier}"
    classifier.load_state_dict(torch.load(dir_name, map_location=device))

    # freeze classifier
    for param in classifier.parameters():
        param.requires_grad = False

    return classifier

def setup_wandb(args, name):
    wandb.init(config=args, project=args.name_project, name=name)
    wandb.define_metric('custom_epoch')
    wandb.define_metric('steps')
    wandb.define_metric('Loss/*', step_metric='custom_epoch')
    wandb.define_metric('Val_Loss/*', step_metric='steps')
    return wandb

def setup_args():
    parser = arg.ArgumentParser(description='MIMIC4: Classifier Training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, nargs='+' , default=[0.0001], help='Learning rate')
    parser.add_argument('--dataset', type=str, default='mimic_4', help='Dataset type')
    parser.add_argument('--n_points', type=int, default=100, help='Number of points to use')
    parser.add_argument("--classifier_name", type=str, default="transformer")
    parser.add_argument("--NB_experts", type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--warmup', type=float, default=0.1, help='warmup')
    parser.add_argument('--expert_exp', type=str, default='Clusters', help='expert type')
    parser.add_argument('--lambda_1', type=float, nargs='+' , default=[1, 10], help='lambda_pred')
    parser.add_argument('--lambda_2', type=float, default=1, help='lambda_regression')
    parser.add_argument("--log_freq", type=int, default=2)
    parser.add_argument("--save_freq", type=int, default=15)
    parser.add_argument("--save", type=int, default=0)
    parser.add_argument("--dev", type=int, default=1) # if 0 we use dev set (subsample) 1 is for full dataset
    parser.add_argument('--name', type=str, default='exp', help='Name experiment')
    parser.add_argument('--name_classifier', type=str, default='model_eval_steps_80.pth', help='classifier name')
    parser.add_argument('--beta', type=float, nargs='+', default=[10], help='cost to ask')
    parser.add_argument('--overfit', type=int, default=0, help='overfit: 1 ')
    parser.add_argument('--device', type=str, default='cuda', help='cuda')
    parser.add_argument('--name_project', type=str, default='two_stage_training_mimic', help='Name')
    parser.add_argument('--path_dataset', type=str, default='/data/mimic_4/', help='Path to dataset')
    parser.add_argument('--extra_data', type=int, default=0, help='Extra data')
    parser.add_argument('--test', type=int, default=1, help='Extra data')


    args = parser.parse_args()

    if args.expert_exp == 'Oracle':
        assert args.NB_experts == 1, "Oracle only works with 1 expert"
    if args.expert_exp == 'Clusters':
        assert args.NB_experts == 2, "Clusters only works with 2 expert2"

    args.beta = [float(i) for i in args.beta]
    args.lr = [float(i) for i in args.lr]
    args.lambda_1 = [float(i) for i in args.lambda_1]

    return args



def train_loop(args, device, param, dataloader, dataset, name):

    dir_name = f"./logs/train/{args.dataset}/two_stage_{name}"

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    classifier_init = Transformer(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key='label_mortality_class',  # ignore this for now
        mode="multiclass",
    )
    classifier = nn.DataParallel(classifier_init.to(device)).eval()
    if not args.test:
        classifier = load_classifier(args, classifier, device, name)

    rejector_init = Transformer_rejector(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key='label_mortality_class',  # ignore this for now
        mode="multiclass",
        number_agents=args.NB_experts + 1,
    )
    rejector = nn.DataParallel(rejector_init.to(device))

    two_stage = TwoStage_Mimic(n_experts=args.NB_experts,
                                 classifier=classifier.eval(),
                                 rejector=rejector,
                                 lambda_1=param['lambda_1'],
                                 lambda_2=args.lambda_2,
                                 alpha_1=param['lambda_1'],
                                 alpha_2=args.lambda_2,
                                 beta=param['beta'],)

    warmup_proportion = args.warmup
    optimizer = torch.optim.AdamW(two_stage.parameters(), lr=param['lr'])

    num_training_steps = args.epochs * len(dataloader['train'])
    num_warmup_steps = int(warmup_proportion * num_training_steps)

    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    steps = 0
    N = len(dataloader['train'])
    for epoch in tqdm(range(args.epochs)):
        running_loss = 0
        for i, x in enumerate(dataloader['train']):
            rejector.train()
            optimizer.zero_grad()
            loss, loss_dict, agent_preds, r_vector,_ = two_stage.get_loss(rejector, x, device, args)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if steps % args.save_freq == 0:
                eval_loop(args, device, rejector, dataloader, two_stage, dir_name, steps)

            running_loss = loss.item() + running_loss
            steps = 1 + steps
        print(f"total: {running_loss / N}")
        wandb.log({
            "custom_epoch": epoch,
            "Loss/total_loss": running_loss / N
        })
    checkpoint_name = "model_final.pth"
    save_path = os.path.join(dir_name, checkpoint_name)
    two_stage.save_model(save_path)


def eval_loop(args, device, model, dataloader, criterion, dir_name, steps):
    model.eval()
    running_loss = 0
    rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla = [], [], [], [], []
    with torch.no_grad():
        for i, x in enumerate(dataloader['val']):
            loss, loss_dict, agent_preds, r_vector, labels = criterion.get_loss(model,x, device, args)
            running_loss = loss.item() + running_loss
            rejector_list.append(r_vector)
            agent_preds_reg.append(agent_preds['agent_regression'])
            agent_preds_cla.append(agent_preds['agent_classification'])
            labels_reg.append(labels['label_regression'])
            labels_cla.append(labels['label_classification'])
    wandb.log({
        "steps": steps,
        "Val_Loss/total_loss": running_loss/len(dataloader['val'])
    })
    if args.save:
        checkpoint_name = f"model_eval_steps_{steps}.pth"
        save_path = os.path.join(dir_name, checkpoint_name)
        torch.save(model.state_dict(), save_path)

    # Metrics evaluation
    sys_metrics, classifier_metrics, deferral_ratio = metrics_L2D(rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla, args)
    wandb.log({
        "Val_Metrics/sys_accuracy": sys_metrics["accuracy"],
        "Val_Metrics/sys_balance_accuracy": sys_metrics["balance_accuracy"],
        "Val_Metrics/sys_f1_0": sys_metrics["f1_0"],
        "Val_Metrics/sys_f1_1": sys_metrics["f1_1"],
        'steps': steps,
        'Val_Metrics/sys_L1': sys_metrics['L1_loss'],
    })
    for i in range(args.NB_experts + 1):
        wandb.log({f"Val_Metrics/deferral_ratio_{i}": deferral_ratio[i].item(),
                   f"cla_Val_Metrics/accuracy_agent_{i}": classifier_metrics["accuracy"][i][0],
                   f"cla_Val_Metrics/balanced_accuracy_agent_{i}": classifier_metrics["balance_accuracy"][i][0],
                   f"cla_Val_Metrics/agent_{i}_f1_0": classifier_metrics["f1_0"][i],
                   f"cla_Val_Metrics/agent_{i}_f1_1": classifier_metrics["f1_1"][i],
                    f"cla_Val_Metrics/agent_{i}_L1": classifier_metrics["L1_loss"][i],
                   'steps': steps,
                   })



def metrics_L2D(rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla, args):
    rejector_arg = torch.argmax(torch.cat(rejector_list, dim=0), dim=1)
    # System labels
    labels_cla = torch.cat(labels_cla, dim=0).cpu().numpy()
    labels_reg = torch.cat(labels_reg, dim=0).cpu().numpy()
    # System allocated
    sys_pred_reg = torch.cat(agent_preds_reg, dim=0)[torch.arange(rejector_arg.size(0)), rejector_arg].cpu().numpy()
    sys_pred_cla = torch.cat(agent_preds_cla, dim=0)[torch.arange(rejector_arg.size(0)),rejector_arg].cpu().numpy()
    # Model and agents
    preds_agent = torch.cat(agent_preds_cla, dim=0).cpu().numpy()
    reg_agent = torch.cat(agent_preds_reg, dim=0).cpu().numpy()
    # Init metrics
    f1_agent = np.zeros((args.NB_experts + 1, 2))
    smape_agent = np.zeros((args.NB_experts + 1, 1))
    accuracy_agent = np.zeros((args.NB_experts + 1, 1))
    balanced_agent = np.zeros((args.NB_experts + 1, 1))
    for i in range(args.NB_experts + 1):
        _, _, f1_agent[i], _ = precision_recall_fscore_support(labels_cla, preds_agent[:,i], average=None, zero_division=0)
        smape_agent[i] = nn.SmoothL1Loss(reduction="none")(torch.tensor(reg_agent[:,i]), torch.tensor(labels_reg)).clamp(0.0,5.0).mean().item()
        accuracy_agent[i] = accuracy_score(labels_cla, preds_agent[:,i])
        balanced_agent[i] = balanced_accuracy_score(labels_cla, preds_agent[:,i])


    # Calculate metrics system
    _, _, f1, _ = precision_recall_fscore_support(labels_cla, sys_pred_cla, average=None, zero_division=0)
    balance_accuracy = balanced_accuracy_score(labels_cla, sys_pred_cla)
    accuracy = accuracy_score(labels_cla, sys_pred_cla)
    L1_loss = nn.SmoothL1Loss(reduction="none")(torch.tensor(sys_pred_reg), torch.tensor(labels_reg)).clamp(0.0,5.0).mean().item()
    sys_metrics = {"accuracy": accuracy, "balance_accuracy": balance_accuracy, "f1_0": f1[0], "f1_1": f1[1], 'L1_loss': L1_loss}

    # Calculate metrics agents
    agent_metrics = {"accuracy": accuracy_agent, "balance_accuracy": balanced_agent,
                     "f1_0": f1_agent[:,0], "f1_1": f1_agent[:,1], 'L1_loss': smape_agent}

    deferral_ratio = torch.bincount(rejector_arg, minlength=args.NB_experts + 1).float() / len(rejector_arg)

    return sys_metrics, agent_metrics, deferral_ratio


def main():
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    args = setup_args()
    random_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader, dataset = preprocess_data(args)
    for i in range(len(args.lr)):
        for j in range(len(args.lambda_1)):
            for k in range(len(args.beta)):
                name = f"{args.name}_{args.classifier_name}_{args.seed}_lr_{args.lr[i]}_lambda_cla{args.lambda_1[j]}_beta_{args.beta[k]}"
                setup_wandb(args, name)
                train_loop(args, device, {"lr" : args.lr[i], 'lambda_1':args.lambda_1[j], 'beta': args.beta[k]},
                           data_loader, dataset, name)
                wandb.finish()

if __name__ == "__main__":
    main()

