import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from firelab.config import Config


def validate_clf(clf_model: nn.Module, dataloader, device: str='cpu'):
    losses = []
    accs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits = clf_model(x)
            loss = F.cross_entropy(logits, y, reduction='none').cpu().tolist()
            acc = (logits.argmax(dim=1) == y).float().cpu().tolist()

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
