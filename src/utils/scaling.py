from typing import Callable

import numpy as np
import torch
from scipy.optimize import fsolve, bisect
from torch import Tensor


def compute_optimal_temperature(logits: Tensor, mode:str, **kwargs) -> Tensor:
    assert mode in ['entropic', 'max_prob']

    logits_np = logits.detach().cpu().numpy()
    optimal_scalers = []

    # Finding sequentially (for now)
    for ls in logits_np:
        if mode == 'entropic':
            func = create_entropic_scaler(ls, kwargs['target_entropy_val'])
        else:
            func = create_max_prob_scaler(ls, kwargs['target_max_prob'])

        try:
            scale = bisect(func, 1e-12, 1e+12)
        except:
            scale = 1.

        optimal_scalers.append(scale)

    return torch.tensor(optimal_scalers).to(logits.device)


def create_entropic_scaler(logits, target_entropy_val: float) -> Callable:
    return (lambda s: entropy_for_logits(logits, s) - target_entropy_val)


def create_max_prob_scaler(logits, target_max_prob: float) -> Callable:
    return (lambda s: softmax(logits, s).max() - target_max_prob)


def entropy_for_logits(logits, *args):
    return entropy(softmax(logits, *args))


def entropy(probs):
    probs = np.array(probs)[np.array(probs) != 0]
    return -(np.log(probs) * probs).sum()


def softmax(values, scale=1):
    raw_log_probs = np.array(values) * scale
    raw_log_probs = raw_log_probs - raw_log_probs.max()
    raw_probs = np.exp(raw_log_probs)
    probs = raw_probs / raw_probs.sum()

    return probs


def linear_softmax(values, scale=1):
    log_probs = values - values.min()
    probs = log_probs / log_probs.sum()

    return probs
