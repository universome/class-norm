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

        if self.config.get('renormalize_attrs'):
            attrs = (attrs - attrs.mean(axis=0)) / attrs.std(axis=0)
        self.register_buffer('attrs', torch.from_numpy(attrs))

        self.transform = nn.Linear(self.attrs.shape[1], self.config.hid_dim, bias=self.config.get('has_bias', True))
        self.init_weights()

    def init_weights(self):
        if self.config.init.type == "orthogonal":
            init.orthogonal_(self.transform.weight, gain=self.config.init.get('gain', 1.0))
        elif self.config.init.type == "xavier":
            initializer = {
                'normal': init.xavier_normal_,
                'uniform': init.xavier_uniform_
            }[self.config.init.dist_type]

            initializer(self.transform.weight, gain=self.config.init.get('gain', 1.0))
            # import math
            # fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.transform.weight)
            # std = 0.5 * math.sqrt(2.0 / float(fan_in + fan_out))
            # a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            # print(a, std)

            # init._no_grad_uniform_(self.transform.weight, -a, a)
        elif self.config.init.type == "kaiming":
            initializer = {
                'normal': init.kaiming_normal_,
                'uniform': init.kaiming_uniform_
            }[self.config.init.dist_type]

            initializer(self.transform.weight,
                mode=self.config.init.get('mode', 'fan_in'),
                nonlinearity=self.config.init.get('nonlinearity', 'linear'))
        elif self.config.init.type == 'proper':
            std = self.get_proper_std()

            if self.config.init.dist_type == 'normal':
                init.normal_(self.transform.weight, std=std)
            elif self.config.init.dist_type == 'uniform':
                b = std * np.sqrt(3)
                init.uniform_(self.transform.weight, a=-b, b=b)
            else:
                raise NotImplementedError(f'Unknown dist type: {self.config.init.dist_type}')
        elif self.config.init.has('gain'):
            self.transform.weight.data.mul_(self.config.init.gain)
        else:
            print('Not using any specific init.')

    def get_proper_std(self):
        attrs_std = self.attrs.std(dim=0).mean()
        fan_in_std = 1 / (np.sqrt(self.config.hid_dim * self.attrs.shape[1]) * attrs_std)
        fan_out_std = 1 / (np.sqrt(self.config.hid_dim * self.config.init.n_classes) * attrs_std)
        fan_harmonic_std = 2 * fan_in_std * fan_out_std / (fan_in_std * fan_out_std)

        if self.config.init.mode == 'fan_in':
            return fan_in_std
        elif self.config.init.mode == 'fan_out':
            return fan_out_std
        elif self.config.init.mode == 'fan_harmonic':
            return fan_harmonic_std
        else:
            raise NotImplementedError(f"Unknown init mode: {self.config.init.mode}")

    def forward(self, x: Tensor) -> Tensor:
        if self.config.get('attrs_dropout.p') and self.training:
            mask = (torch.rand(self.attrs.shape[1]) > self.config.attrs_dropout.p)
            mask = mask.unsqueeze(0).float().to(x.device)
            attrs = self.attrs * mask
            attrs = attrs / (1 - self.config.attrs_dropout.p)
        else:
            attrs = self.attrs

        if self.config.get('attrs_noise.std') and self.training:
            noise = torch.randn(self.attrs.shape[1]) * self.config.attrs_noise.std
            noise = noise.unsqueeze(0).float().to(x.device)
            attrs += noise

        if self.config.get('feats_dropout.p'):
            x = F.dropout(x, p=self.config.feats_dropout.p, training=self.training)

        W = self.transform.weight # [hid_dim, attr_dim]
        if self.config.get('normalize_weights'):
            avg_w_norm = W.norm(dim=0).mean() # Compute avg norm for each attr column
            W = normalize(W, avg_w_norm, dim=0) # Renormalize
        protos = attrs @ W.t() # [n_classes, hid_dim]

        if self.config.get('protos_dropout.p'):
            protos = F.dropout(protos, p=self.config.protos_dropout.p, training=self.training)

        if self.config.get('normalize_feats', True):
            x = normalize(x, self.config.scale)

        if self.config.get('normalize_protos', True):
            protos = normalize(protos, self.config.scale)

        return x @ protos.t()
