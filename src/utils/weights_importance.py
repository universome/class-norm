import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from firelab.utils.training_utils import get_module_device

from src.utils.lll import prune_logits
from src.utils.constants import NEG_INF


def compute_diagonal_fisher(model: nn.Module, dataloader: DataLoader, output_mask: np.ndarray, normalize_fisher=True) -> Tensor:
    """
    Computes approximate diagonal Fisher matrix

    :param model:
    :param dataloader:
    :return:
    """
    fisher = compute_grad(model, nn.CrossEntropyLoss(), dataloader, output_mask, 'square')

    if normalize_fisher:
        f_min, f_max = fisher.min().item(), fisher.max().item()
        fisher = (fisher - f_min) / (f_max - f_min)

    return fisher


def compute_mse_grad(model: nn.Module, dataloader: DataLoader, output_mask: np.ndarray) -> Tensor:
    """
    Computes absolute value of gradient of mse loss

    :param model:
    :param dataloader:
    :return:
    """
    mse_criterion = lambda logits, _: logits[logits != NEG_INF].pow(2).sum()

    return compute_grad(model, mse_criterion, dataloader, output_mask, 'abs')


def compute_grad(model: nn.Module, criterion: nn.Module, dataloader: DataLoader,
                 output_mask: np.ndarray, elementwise_grad_norm: str) -> Tensor:
    """
    Computes gradient of the given loss across the dataset

    :param model:
    :param dataloader:
    :return:
    """
    dummy_optim = torch.optim.SGD(model.parameters(), lr=1) # Just for zeroing gradient
    num_samples = 0
    num_params = sum(p.numel() for p in model.parameters())
    device = get_module_device(model)
    grad = torch.zeros(num_params).to(device)

    for x, y in dataloader:
        x = torch.tensor(x).to(device)
        y = torch.tensor(y).to(device)
        logits = model(x)
        pruned_logits = prune_logits(logits, output_mask)
        loss = criterion(pruned_logits, y)

        dummy_optim.zero_grad()
        loss.backward()
        curr_grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()])

        if elementwise_grad_norm == 'square':
            curr_grad = curr_grad.pow(2)
        elif elementwise_grad_norm == 'abs':
            curr_grad = curr_grad.abs()
        else:
            raise NotImplementedError(f'Unknown elementwise grad norm: {elementwise_grad_norm}')

        grad += curr_grad
        num_samples += len(x)

    return grad / num_samples
