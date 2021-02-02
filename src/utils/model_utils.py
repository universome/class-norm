from typing import List
import torch.nn as nn


def filter_params(model: nn.Module, exclude_prefix: str) -> List[nn.Parameter]:
    return [p for n, p in model.named_parameters() if not n.startswith(exclude_prefix)]


def get_number_of_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
