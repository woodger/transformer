import torch


def combined_loss(preds, targets):
    return torch.mean((preds - targets) ** 2)
