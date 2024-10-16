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



def eval_model(args, device, dataloader, name, dataset, path_rejector):

    classifier_init = Transformer(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key='label_mortality_class',  # ignore this for now
        mode="multiclass",
    )
    classifier = nn.DataParallel(classifier_init.to(device)).eval()
    if not args.test:
        dir_name_classifier = f"./logs/train/{args.dataset}/{args.name_classifier}"
        classifier.load_state_dict(torch.load(dir_name_classifier, map_location=device))

    rejector_init = Transformer_rejector(
        dataset=dataset,
        feature_keys=["conditions", "procedures"],
        label_key='label_mortality_class',  # ignore this for now
        mode="multiclass",
        number_agents=args.NB_experts + 1,
    )
    rejector = nn.DataParallel(rejector_init.to(device))
    if not args.test:
        dir_name_rejector = f"./logs/train/{args.dataset}/{args.name_rejector}"
        rejector.load_state_dict(torch.load(dir_name_rejector, map_location=device))


    two_stage = TwoStage_Mimic(n_experts=args.NB_experts,
                               classifier=classifier.eval(),
                               rejector=rejector,
                               lambda_1=args.lambda_1,
                               lambda_2=args.lambda_2,
                               alpha_1=args.lambda_1,
                               alpha_2=args.lambda_2,
                               beta=args.beta, )
    rejector.eval()
    classifier.eval()
    running_loss = 0
    rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla = [], [], [], [], []
    with torch.no_grad():
        for i, x in enumerate(dataloader['val']):
            loss, loss_dict, agent_preds, r_vector, labels = two_stage.get_loss(rejector,x, device, args)
            running_loss = loss.item() + running_loss
            rejector_list.append(r_vector)
            agent_preds_reg.append(agent_preds['agent_regression'])
            agent_preds_cla.append(agent_preds['agent_classification'])
            labels_reg.append(labels['label_regression'])
            labels_cla.append(labels['label_classification'])

    # Metrics evaluation
    sys_metrics, classifier_metrics, deferral_ratio = metrics_L2D(rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla, args)
    eval_stat(rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla, args)

    wandb.log({
        "Val_Metrics/sys_accuracy": sys_metrics["accuracy"],
        "Val_Metrics/sys_balance_accuracy": sys_metrics["balance_accuracy"],
        "Val_Metrics/sys_f1_0": sys_metrics["f1_0"],
        "Val_Metrics/sys_f1_1": sys_metrics["f1_1"],
        'Val_Metrics/sys_L1': sys_metrics['L1_loss'],
    })
    for i in range(args.NB_experts + 1):
        wandb.log({f"Val_Metrics/deferral_ratio_{i}": deferral_ratio[i].item(),
                   f"cla_Val_Metrics/accuracy_agent_{i}": classifier_metrics["accuracy"][i][0],
                   f"cla_Val_Metrics/balanced_accuracy_agent_{i}": classifier_metrics["balance_accuracy"][i][0],
                   f"cla_Val_Metrics/agent_{i}_f1_0": classifier_metrics["f1_0"][i],
                   f"cla_Val_Metrics/agent_{i}_f1_1": classifier_metrics["f1_1"][i],
                    f"cla_Val_Metrics/agent_{i}_L1": classifier_metrics["L1_loss"][i],
                   })


def eval_stat(rejector_list, agent_preds_reg, agent_preds_cla, labels_reg, labels_cla, args):
    print('test')
    size = rejector_list[0].size(1)
    loss_agent = []
    for i in range(size):
        correct_agent = torch.where(agent_preds_cla[0][:,i] == labels_cla[0], 1, 0)
        l1_agent = nn.SmoothL1Loss(reduction="none")(agent_preds_reg[0][:,i], labels_reg[0]).clamp(0.0,args.clamp)
        if i==0:
            loss_agent.append(args.lambda_1 * (1-correct_agent) + args.lambda_2 * l1_agent)
        else:
            loss_agent.append(args.lambda_1 * (1-correct_agent) + args.lambda_2 * l1_agent + args.beta)

    loss_agent = torch.stack(loss_agent, dim=1)
    arg_loss_agent = torch.argmin(loss_agent, dim=1)
    arg_rejector = torch.argmax(rejector_list[0], dim=1)
    # correctly_deferred_id = torch.where(arg_loss_agent == arg_rejector)
    # difference_loss = loss_agent[:,0] - torch.min(loss_agent[:,1:], dim=1)[0]
    # difference_correctly_deferred = difference_loss[correctly_deferred_id]
    # difference_not_correctly_deferred = difference_loss[torch.where(arg_loss_agent != arg_rejector)]
    #
    min_loss_agent = torch.min(loss_agent[:,1:], dim=1)[0]
    difference_loss = loss_agent[:,0] - min_loss_agent
    loss_agent_combined = torch.cat((loss_agent[:,0][:,None], min_loss_agent[:,None]), dim=1)
    rejector_adapted = torch.where(arg_rejector!=0, 1,0)

    # should_be_deferred_theory = torch.where(difference_loss>=0, 1,0)

    #Deferred to model
    count_correct_behavior_model = []
    count_incorrect_behavior_model = []
    #Deferred to expert
    count_correct_behavior_expert = []
    count_incorrect_behavior_expert = []
    cumulative_loss_store = []
    tot = len(rejector_list[0])
    for q in range(tot):
        t = difference_loss[q]
        if rejector_adapted[q]==0:
            if t<=0: # we did not defer and model was better (1)
                count_correct_behavior_model.append(q)
            else: # we did not defer but expert was better (2)
                count_incorrect_behavior_model.append(q)
        else:
            if t<=0: # we deferred but model was better (3)
                count_incorrect_behavior_expert.append(q)
            else: # we deferred and expert was better than model (4)
                count_correct_behavior_expert.append(q)

        cumulative_loss_store.append({'model': loss_agent_combined[q,0].item(), 'best_expert': loss_agent_combined[q,1].item(),
                           'system': loss_agent_combined[q,rejector_adapted[q]].item(), 'theory': torch.min(loss_agent[q]).item(),
                                     'loss_agents': loss_agent[q].cpu().numpy()})

    percentage = {'1': len(count_correct_behavior_model)/tot, '2': len(count_incorrect_behavior_model)/tot,
                  '3': len(count_incorrect_behavior_expert)/tot, '4': len(count_correct_behavior_expert)/tot}

    expected_loss =  {'1': difference_loss[count_correct_behavior_model].sum().item(),
                      '2': difference_loss[count_incorrect_behavior_model].sum().item(),
                  '3': difference_loss[count_incorrect_behavior_expert].sum().item(),
                      '4': difference_loss[count_correct_behavior_expert].sum().item()}

    normalizer = torch.abs(difference_loss).sum().item()
    expected_loss_normalize = {k: v/normalizer for k,v in expected_loss.items()}

    cumulative_loss = {'model': sum([i['model'] for i in cumulative_loss_store]), 'best_expert': sum([i['best_expert'] for i in cumulative_loss_store]),
                       'system': sum([i['system'] for i in cumulative_loss_store]), 'theory': sum([i['theory'] for i in cumulative_loss_store]),
                       'loss_agents': [i['loss_agents'] for i in cumulative_loss_store]}

    my_dict = {'percentage': percentage, 'expected_loss': expected_loss, 'expected_loss_normalize': expected_loss_normalize,
               'cumulative_loss': cumulative_loss}

    wandb.summary.update(my_dict)
    return percentage, expected_loss, expected_loss_normalize

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
        smape_agent[i] = nn.SmoothL1Loss(reduction="none")(torch.tensor(reg_agent[:,i]),
                                                           torch.tensor(labels_reg)).clamp(0.0,args.clamp).mean().item()
        accuracy_agent[i] = accuracy_score(labels_cla, preds_agent[:,i])
        balanced_agent[i] = balanced_accuracy_score(labels_cla, preds_agent[:,i])


    # Calculate metrics system
    _, _, f1, _ = precision_recall_fscore_support(labels_cla, sys_pred_cla, average=None, zero_division=0)
    balance_accuracy = balanced_accuracy_score(labels_cla, sys_pred_cla)
    accuracy = accuracy_score(labels_cla, sys_pred_cla)
    L1_loss = nn.SmoothL1Loss(reduction="none")(torch.tensor(sys_pred_reg), torch.tensor(labels_reg)).clamp(0.0,args.clamp).mean().item()
    sys_metrics = {"accuracy": accuracy, "balance_accuracy": balance_accuracy, "f1_0": f1[0], "f1_1": f1[1], 'L1_loss': L1_loss}

    # Calculate metrics agents
    agent_metrics = {"accuracy": accuracy_agent, "balance_accuracy": balanced_agent,
                     "f1_0": f1_agent[:,0], "f1_1": f1_agent[:,1], 'L1_loss': smape_agent}

    deferral_ratio = torch.bincount(rejector_arg, minlength=args.NB_experts + 1).float() / len(rejector_arg)

    return sys_metrics, agent_metrics, deferral_ratio

def setup_args():
    parser = arg.ArgumentParser(description='MIMIC3: Classifier Training')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--dataset', type=str, default='mimic_4', help='Dataset type')
    parser.add_argument("--classifier_name", type=str, default="transformer")
    parser.add_argument("--NB_experts", type=int, default=2)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--expert_exp', type=str, default='Clusters', help='expert type')
    parser.add_argument('--lambda_1', type=float, nargs='+' , default=[1], help='lambda_pred')
    parser.add_argument('--lambda_2', type=float, default=0.2, help='lambda_ref')
    parser.add_argument("--dev", type=int, default=1) # if 0 we use dev set (subsample) 1 is for full dataset
    parser.add_argument('--name', type=str, default='exp', help='Name experiment')
    parser.add_argument('--name_classifier', type=str, default='model_eval_steps_80.pth', help='classifier name')
    parser.add_argument('--beta', type=float, nargs='+', default=[0], help='cost to ask')
    parser.add_argument('--overfit', type=int, default=0, help='overfit: 1 ')
    parser.add_argument('--device', type=str, default='cuda', help='cuda')
    parser.add_argument('--name_project', type=str, default='two_stage_mimic_metrics', help='Name')
    parser.add_argument('--path_dataset', type=str, default='./data/mimic-'
                                                            , help='Path to dataset')
    parser.add_argument('--name_rejector', type=str, default='', help='path to the rejector')
    parser.add_argument('--clamp', type=float, default=5, help='clamp')
    parser.add_argument('--test', type=int, default=1, help='test')
    args = parser.parse_args()

    assert args.clamp==1/args.lambda_2, "clamp should be 1/lambda_2 for lambda_1=1"
    return args

def main():
    os.environ["WANDB_INIT_TIMEOUT"] = "300"
    args = setup_args()
    # Setting
    args.lambda_1 = args.lambda_1[0]
    args.beta = args.beta[0]
    #
    random_seed(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_loader, dataset = preprocess_data(args)
    name = f"test"
    setup_wandb(args, name)
    eval_model(args, device, data_loader, name, dataset, args.path_rejector)
    wandb.finish()
    print('done')

if __name__ == "__main__":
    main()

