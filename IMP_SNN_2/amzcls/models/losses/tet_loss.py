import torch
from torch.nn import functional as F


def tet_loss(outputs, labels, criterion, means, lamb):
    time, batch = outputs.shape[:2]
    loss_ce = criterion(outputs.view(time * batch, -1), torch.cat([labels] * time, dim=0))
    if lamb == 0:
        return loss_ce
    loss_mse = F.mse_loss(outputs, torch.full_like(outputs, means))
    return (1. - lamb) * loss_ce + lamb * loss_mse
