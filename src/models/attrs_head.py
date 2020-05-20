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

        if self.config.get('normalize_attrs'):
            attrs = (attrs - attrs.mean(axis=0)) / attrs.std(axis=0)
        self.register_buffer('attrs', torch.from_numpy(attrs))

        self.transform = nn.Linear(self.attrs.shape[1], self.config.hid_dim)
        self.init_weights()

    def init_weights(self):
        if self.config.init.type == "orthogonal":
            init.orthogonal_(self.transform.weight, gain=self.config.init.get('gain', 1.0))
        elif self.config.init.type == "xavier":
            initializer = {'normal': init.xavier_normal_, 'uniform': init.xavier_uniform_}[self.config.init.dist_type]
            initializer(self.transform.weight, gain=self.config.init.get('gain', 1.0))
        elif self.config.init.type == "kaiming":
            initializer = {'normal': init.kaiming_normal_, 'uniform': init.kaiming_uniform_}[self.config.init.dist_type]
            initializer(self.transform.weight,
                mode=self.config.init.get('mode', 'fan_in'),
                nonlinearity=self.config.init.get('nonlinearity', 'leaky_relu'))
        elif self.config.init.type == 'proper':
            std = self.get_proper_std()

            if self.config.init.dist_type == 'normal':
                init.normal_(self.transform.weight, std=std)
            elif self.config.init.dist_type == 'uniform':
                b = std * np.sqrt(3)
                init.uniform_(self.transform.weight, a=-b, b=b)
            else:
                raise NotImplementedError(f'Unknown dist type: {self.config.init.dist_type}')
        else:
            print('Not using any specific init.')

    def get_proper_std(self):
        attrs_std = self.attrs.std(dim=0).mean()
        fan_in_std = 1 / np.sqrt(self.config.hid_dim * self.attrs.shape[1]) * attrs_std
        fan_out_std = 1 / np.sqrt(self.config.hid_dim * self.config.init.n_classes) * attrs_std
        fan_harmonic_std = 2 * fan_in_std * fan_out_std / (fan_in_std * fan_out_std)

        if self.config.init.proper_std_type == 'fan_in':
            return fan_in_std
        elif self.config.init.proper_std_type == 'fan_out':
            return fan_out_std
        elif self.config.init.proper_std_type == 'fan_harmonic':
            return fan_harmonic_std
        else:
            raise NotImplementedError(f"Unknown init type: {self.config.init.proper_std_type}")

    def forward(self, x: Tensor) -> Tensor:
        protos = self.transform(self.attrs)
        x = normalize(x, self.config.scale)
        protos = normalize(protos, self.config.scale)

        return x @ protos.t()
