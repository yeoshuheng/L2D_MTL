import torch
import torch.nn as nn

def SMAPE_loss(y_pred, y_true):
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    # Prevent division by zero
    denominator = torch.where(denominator == 0, 1e-8, denominator)
    scaled_smape = (numerator / denominator) / 2  # Scale to [0, 1]
    return scaled_smape

def SmoothL1Loss(y_pred, y_true):
    return nn.SmoothL1Loss(y_pred, y_true)

class MimicLoss:
    """ This class represents the loss function used to train the classifier on MIMIC-IV"""
    def __init__(self, lambda_pred, lambda_reg, args):
        self.lambda_pred = lambda_pred
        self.lambda_reg = lambda_reg
        
        self.ce = nn.CrossEntropyLoss()
        if args.loss == "l1":
            self.l1 = nn.SmoothL1Loss()
        if args.loss == "smape":
            self.l1 = SMAPE_loss

    def get_loss(self, model, x_, device):
        """ Get combined regressive and classification loss components. """
        out, _ = model(**x_)
        cls_pred, reg_pred, labels_class, labels_reg = (out["classifier_fc"], out["regressor_fc"],
                                                        out["label_class"], out["label_reg"])
        labels_class = labels_class.long()

        cls_loss = self.cls_loss(cls_pred, labels_class)
        reg_loss = self.reg_loss(reg_pred.squeeze(), labels_reg).mean()

        loss_dict = { "cls_loss" : self.lambda_pred * cls_loss.item(), "reg_loss" : self.lambda_reg * reg_loss.item() }
        full_loss = self.lambda_pred * cls_loss + self.lambda_reg * reg_loss
        return full_loss, loss_dict, out

    def cls_loss(self, cls_pred, cls_target):
        """ Get classification loss """
        return self.ce(cls_pred, cls_target)

    def reg_loss(self, reg_pred, reg_target):
        """ Get regression loss """
        return self.l1(reg_pred, reg_target)