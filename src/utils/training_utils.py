import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from firelab.config import Config


def validate_clf(clf_model: nn.Module, dataloader, device: str='cpu'):
    losses = []
    accs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = torch.from_numpy(x).to(device)
            y = torch.tensor(y).to(device)

            logits = clf_model(x)
            loss = F.cross_entropy(logits, y, reduction='none').cpu().tolist()
            acc = compute_guessed(logits, y).tolist()

            losses.extend(loss)
            accs.extend(acc)

    return np.mean(losses), np.mean(accs)


def construct_optimizer(parameters, optim_config: Config):
    if optim_config.type == 'sgd':
        return torch.optim.SGD(parameters, **optim_config.kwargs.to_dict())
    elif optim_config.type == 'adam':
        return torch.optim.Adam(parameters, **optim_config.kwargs.to_dict())
    else:
        raise NotImplementedError(f'Unknown optimizer: {optim_config.type}')


def compute_accuracy(logits: Tensor, targets: Tensor, *args, **kwargs) -> Tensor:
    return compute_guessed(logits, targets, *args, **kwargs).mean()


def compute_guessed(logits: Tensor, targets: Tensor, to_device='cpu') -> Tensor:
    assert logits.ndim == 2
    assert targets.ndim == 1
    assert len(logits) == len(targets)

    return (logits.argmax(dim=1) == targets).float().detach().to(to_device)
