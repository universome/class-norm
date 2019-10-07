import numpy as np
import torch
from torch import Tensor
from src.utils.constants import NEG_INF


def prune_logits(logits: Tensor, output_mask:np.ndarray) -> Tensor:
    """
    Takes logits and sets those classes which do not participate
    in the current task to -infinity so they are not explicitly penalized and forgotten
    """
    mask_idx = np.nonzero(~output_mask)[0]
    pruned = logits.index_fill(1, torch.tensor(mask_idx).to(logits.device), NEG_INF)

    return pruned
