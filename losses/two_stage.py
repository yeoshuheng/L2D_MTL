import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings
from sklearn.exceptions import ConvergenceWarning

def SMAPE_loss(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    # Prevent division by zero
    denominator = torch.where(denominator == 0, 1e-8, denominator)
    scaled_smape = (numerator / denominator) / 2  # Scale to [0, 1]
    return scaled_smape

class TwoStage_Mimic(nn.Module):
    """
    This class represents the Two-Stage Mimic L2D system.
    """
    def __init__(self, n_experts, classifier, rejector, lambda_1, lambda_2, alpha_1, alpha_2, beta):
        super().__init__()

        self.classifier = classifier
        self.n_experts = n_experts
        self.rejector = rejector

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # non-expert deferral properties
        self.num_labels_cls = 1
        self.num_labels_reg = 1

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta = beta

        self.l1 = nn.SmoothL1Loss(reduction="none")
        self.re = nn.LogSoftmax(dim = 1)

    def get_cost(self, expert_cls, expert_reg, target_cls, target_reg, expert=False):
        """
        Returns the cost of defering to a particular agent.
        :param expert_cls: Agent's predicted class.
        :param expert_cls: Agent's predicted regression values.
        :param target_cls: The target class.
        :param target_reg: The target regression value.
        :param expert: Boolean flag to add alpha & beta penalities if we are evaluating a expert.
        """

        target_cls = target_cls.long()
        cls_loss = 1 - (expert_cls == target_cls.unsqueeze(dim=1)).long()
        if expert:
            reg_loss = torch.zeros(expert_reg.shape).to(expert_reg.device)
            for i in range(expert_reg.shape[1]):
                reg_loss[:, i] = self.l1(expert_reg[:, i], target_reg).clamp(0.0, 5.0)
            expert_loss = self.alpha_1 * cls_loss + self.alpha_2 * reg_loss
            cost = expert_loss + self.beta
        else:
            reg_loss = self.l1(expert_reg, target_reg.unsqueeze(dim=1)).clamp(0.0, 5.0)
            cost = self.lambda_1 * cls_loss \
                + self.lambda_2 * reg_loss

        return cost.detach()

    def get_loss(self, rejector, x_, device, args):
        """
        Gets loss for current batch.
        """
        out = rejector(**x_)
        r_vector, labels_class, labels_reg = (out["rejector_fc"], out["label_class"], out["label_reg"])

        with torch.no_grad():
            self.classifier.eval()
            out_class, data = self.classifier(**x_)
            cls_pred, reg_pred = (out_class["classifier_fc"], out_class["regressor_fc"])
            cls_pred = torch.argmax(cls_pred, dim=-1).detach()
            cls_pred = cls_pred.unsqueeze(-1)
            classifier_cost = self.get_cost(cls_pred, reg_pred, labels_class, labels_reg, expert=False).detach()

        # Experts sampling
        expert_cls, expert_reg = torch.tensor(x_['expert_class'], device=device), torch.tensor(x_['expert_reg'], device=device)
        if len(expert_reg.shape) == 1:
            expert_reg = expert_reg.unsqueeze(dim=1)
        if len(expert_cls.shape) == 1:
            expert_cls = expert_cls.unsqueeze(dim=1)

        expert_costs = self.get_cost(expert_cls, expert_reg, labels_class, labels_reg, expert=True).detach()

        sum_experts = torch.sum(expert_costs, dim = -1)
        r_vector_output = - self.re(r_vector)
        lhs = r_vector_output[:, 0] * sum_experts

        agents_cost = torch.cat((classifier_cost, expert_costs), dim=1)
        rhs = 0
        c_0 = classifier_cost.squeeze()
        for j in range(1, self.n_experts+1):
            sum_cost_k_acc = 0
            for k in range(1, self.n_experts+1):
                if k!=j:
                    c_k = agents_cost[:,k]
                    sum_cost_k_acc += c_k
            acc = c_0 + sum_cost_k_acc
            rhs += acc*r_vector_output[:,j]

        loss = (lhs + rhs).mean()
        loss_dict = {"l2d_loss" : loss.item()}
        agent_preds = {"agent_classification": torch.cat((cls_pred, expert_cls), dim=1),
                       "agent_regression": torch.cat((reg_pred, expert_reg), dim=1)}
        labels = {"label_classification": labels_class, "label_regression": labels_reg}
        return loss, loss_dict, agent_preds, r_vector, labels

    def save_model(self, path):
        """Saves model"""
        torch.save(self.rejector.state_dict(), path)

    def load_model(self, path, classifier, device):
        """Loads model"""
        self.classifier = classifier
        self.rejector.load_state_dict(torch.load(path, map_location=device))

    def forward(self, x_, device):
        """
        Get prediction from two-stage L2D system.
        """
        x, labels_class, labels_reg, expert_cls, expert_reg, _ = x_

        x = x.to(device)
        labels_class = labels_class.to(device)
        labels_reg = labels_reg.to(device)
        expert_cls = expert_cls.to(device)
        expert_reg = expert_reg.to(device)

        r_vector = self.rejector(x)
        r_vector = self.sm(r_vector)
        selected = torch.argmax(r_vector, dim=-1)

        defer_ratio = np.array([torch.sum(selected == num).item() for num in np.arange(1 + self.n_experts)]) / len(selected)

        bs = x.shape[0]

        # create new output size
        output_reg = torch.zeros((bs, self.num_labels_reg))
        output_cls = torch.zeros((bs, self.num_labels_cls)).long()

        cls_pred, reg_pred = self.classifier(x)
        cls_pred = self.sm(cls_pred)
        cls_pred = torch.argmax(cls_pred, dim=-1).unsqueeze(-1).long()

        cls_mask = (selected == 0)

        output_reg[cls_mask] = reg_pred[cls_mask]
        output_cls[cls_mask] = cls_pred[cls_mask]
        output_reg[~cls_mask] = expert_reg[~cls_mask]
        output_cls[~cls_mask] = expert_cls[~cls_mask]

        output_cls = output_cls.squeeze(-1)

        return output_cls, output_reg, defer_ratio















