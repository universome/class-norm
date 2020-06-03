import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize


class AttrsHead(nn.Module):
    def __init__(self, config, attrs):
        super().__init__()

        self.config = config
        self.attrs = nn.Parameter(torch.from_numpy(attrs), requires_grad=False)

        self.output_layer = nn.Linear(self.config.hid_dim, self.config.feat_dim)
        self.bn_layer = nn.BatchNorm1d(self.config.hid_dim, affine=False) if self.config.init.type == 'proper' else nn.Identity()

        self.transform = nn.Sequential(
            nn.Linear(self.attrs.shape[1], self.config.hid_dim),
            nn.ReLU(),
            self.bn_layer,
            self.output_layer,
            nn.ReLU()
        )

        if self.config.init.type == 'proper':
            if self.config.init.with_relu:
                var = 2 / (self.config.hid_dim * self.config.feat_dim * (1 - 1/np.pi))
            else:
                var = 1 / (self.config.hid_dim * self.config.feat_dim)

            if self.config.init.dist == 'uniform':
                b = np.sqrt(3 * var)
                self.output_layer.weight.data.uniform_(-b, b)
            else:
                self.output_layer.weight.data.normal_(0, np.sqrt(var))
        elif self.config.init.type == 'xavier':
            if self.config.init.dist == 'uniform':
                init.xavier_uniform_(self.output_layer.weight)
            else:
                init.xavier_normal_(self.output_layer.weight)
        elif self.config.init.type == 'kaiming':
            if self.config.init.dist == 'uniform':
                init.kaiming_uniform_(self.output_layer.weight, mode=self.config.init.mode, nonlinearity='relu')
            else:
                init.kaiming_normal_(self.output_layer.weight, mode=self.config.init.mode, nonlinearity='relu')
        else:
            raise ValueError(f'Unknown init type: {self.config.init.type}')

    def forward(self, x: Tensor) -> Tensor:
        protos = self.transform(self.attrs)

        if self.config.get('normalize_feats', True):
            x = normalize(x, self.config.scale)

        if self.config.get('normalize_protos', True):
            protos = normalize(protos, self.config.scale)

        return x @ protos.t()
