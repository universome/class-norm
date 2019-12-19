from typing import Tuple
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization by @Kaixhin
    https://github.com/pytorch/pytorch/issues/8985
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        return out


class Reshape(nn.Module):
    def __init__(self, target_shape: Tuple[int]):
        super(Reshape, self).__init__()

        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
