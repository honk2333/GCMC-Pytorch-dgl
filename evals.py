import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def MAE(predictions, ratings):
    dev = predictions - ratings
    mae = torch.norm(dev, p=1) / dev.size()[0]

    return mae.item()


def RMSE(predictions, ratings):
    dev = predictions - ratings
    rmse = torch.sqrt(torch.sum(torch.mul(dev, dev)) / dev.size()[0])

    return rmse.item()


def MSE_loss(predictions, ratings):
    return F.mse_loss(predictions, ratings, reduction='sum')


def CE_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(logits, labels.view(-1))
