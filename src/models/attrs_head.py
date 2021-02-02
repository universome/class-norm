import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize


class AttrsHead(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.config = config
        if self.config.standardize_attrs:
            attrs = (attrs - attrs.mean(axis=0, keepdims=True)) / attrs.std(axis=0, keepdims=True)
        else:
            attrs = self.config.attrs_additional_scale * np.sqrt(attrs.shape[1]) * attrs / ((attrs ** 2).sum(axis=1, keepdims=True) ** 0.5)
        self.attrs = nn.Parameter(torch.from_numpy(attrs), requires_grad=False)

        if self.config.type == 'deep':
            penultimate_dim = self.config.hid_dim
            self.early_layers = nn.Sequential(
                nn.Linear(self.attrs.shape[1], penultimate_dim),
                nn.ReLU(),
            )
        elif self.config.type == 'linear':
            penultimate_dim = self.attrs.shape[1]
            self.early_layers = nn.Identity()
        else:
            raise NotImplementedError(f'Unknown model type: {self.config.type}')

        if self.config.final_activation == 'relu':
            final_activation = nn.ReLU()
        elif self.config.final_activation is None:
            final_activation = nn.Identity()
        else:
            raise NotImplementedError(f'Unknown final activation: {self.config.final_activation}')

        if self.config.type == 'deep' and self.config.num_additional_hidden_layers > 0:
            self.additional_hidden_layers = []
            for _ in range(self.config.num_additional_hidden_layers):
                self.additional_hidden_layers.append(nn.Linear(self.config.hid_dim, self.config.hid_dim))
                if self.config.has_bn:
                    self.additional_hidden_layers.append(nn.BatchNorm1d(self.config.hid_dim))
                self.additional_hidden_layers.append(nn.ReLU())
            self.additional_hidden_layers = nn.Sequential(*self.additional_hidden_layers)
        else:
            self.additional_hidden_layers = nn.Identity()

        if self.config.has_bn:
            if self.config.bn_type == 'batch_norm':
                bn_layer = nn.BatchNorm1d(penultimate_dim, affine=self.config.bn_affine)
            elif self.config.bn_type == 'layer_norm':
                bn_layer = nn.LayerNorm(penultimate_dim, elementwise_affine=self.config.bn_affine)
            else:
                raise NotImplementedError(f'Unknwn bn layer type: {self.config.bn_type}')
        else:
            bn_layer = nn.Identity()

        if self.config.has_dn:
            dn_layer = DynamicNormalization()
        else:
            dn_layer = nn.Identity()

        self.output_layer = nn.Linear(penultimate_dim, self.config.feat_dim)

        self.transform = nn.Sequential(
            self.early_layers,
            self.additional_hidden_layers,
            bn_layer,
            dn_layer,
            self.output_layer,
            final_activation
        )

        if self.config.init.type == 'proper':
            if self.config.init.with_relu:
                var = 2 / (penultimate_dim * self.config.feat_dim * (1 - 1/np.pi))
            else:
                var = 1 / (penultimate_dim * self.config.feat_dim)

            if self.config.init.dist == 'uniform':
                b = np.sqrt(3 * var)
                self.output_layer.weight.data.uniform_(-b, b)
            else:
                self.output_layer.weight.data.normal_(0, np.sqrt(var))
        elif self.config.init.type == 'xavier':
            init.xavier_uniform_(self.output_layer.weight)
        elif self.config.init.type == 'kaiming':
            init.kaiming_uniform_(self.output_layer.weight, mode=self.config.init.mode, nonlinearity='relu')
        else:
            raise ValueError(f'Unknown init type: {self.config.init.type}')

    def forward(self, x: Tensor, attrs_mask: bool=None, return_prelogits: bool=False) -> Tensor:
        attrs = self.attrs if attrs_mask is None else self.attrs[attrs_mask]
        protos = self.transform(attrs)

        if self.config.get('normalize_and_scale', True):
            x_ns = normalize(x, self.config.scale)
            protos_ns = normalize(protos, self.config.scale)
        else:
            x_ns = x
            protos_ns = protos

        logits = x_ns @ protos_ns.t()

        if return_prelogits:
            prelogits = x @ protos.t()
            # print('Protos mean value:', protos.mean().cpu().item())
            # print('Feats mean value:', x.mean().cpu().item())
            # print('Prelogits mean value:', prelogits.mean().cpu().item())
            return logits, prelogits
        else:
            return logits


class DynamicNormalization(nn.Module):
    def forward(self, x):
        assert x.ndim == 2, f"Wrong shape: {x.shape}"

        mean_norm = x.norm(dim=1).mean()
        return x / mean_norm.pow(2)
