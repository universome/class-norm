from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.constants import RESNET_FEAT_DIM


class FeatVAE(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAE, self).__init__()

        self.encoder = FeatVAEEncoder(config, attrs)
        self.decoder = FeatVAEDecoder(config, attrs)
        self.prior = FeatVAEPrior(config, attrs)

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_var = self.encoder(x, y)
        z = self.sample(mean, log_var)
        x_rec = self.decoder(z, y)

        return x_rec, mean, log_var

    def sample(self, mean: Tensor, log_var: Tensor, noise_level: float=1.) -> Tensor:
        """Samples z ~ N(mean, sigma)"""
        return mean + noise_level * torch.randn_like(log_var) * (log_var / 2).exp()

    def sample_from_prior(self, y: Tensor) -> Tensor:
        mean, log_var = self.prior(y)
        eps = torch.randn_like(log_var)
        z = mean + eps * (log_var / 2).exp()

        return z

    def generate(self, y: Tensor) -> Tensor:
        z = self.sample_from_prior(y)
        x = self.decoder(z, y)

        return x


class FeatVAEEncoder(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAEEncoder, self).__init__()

        self.config = config
        self.embedder = FeatVAEEmbedder(config, attrs)
        self.model = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type] + self.config.emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.z_dim * 2),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_emb = self.embedder(y)
        inputs = torch.cat([x, y_emb], dim=1)
        encodings = self.model(inputs)
        mean = encodings[:, :self.config.z_dim]
        log_var = encodings[:, self.config.z_dim:]

        return mean, log_var


class FeatVAEDecoder(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAEDecoder, self).__init__()

        self.config = config
        self.embedder = FeatVAEEmbedder(config, attrs)
        self.model = nn.Sequential(
            nn.Linear(self.config.z_dim + self.config.emb_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, RESNET_FEAT_DIM[self.config.resnet_type]),
        )

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        y_emb = self.embedder(y)
        inputs = torch.cat([z, y_emb], dim=1)
        x_rec = self.model(inputs)

        return x_rec


class FeatVAEPrior(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAEPrior, self).__init__()

        self.config = config
        self.embedder = FeatVAEEmbedder(config, attrs)
        self.model = nn.Sequential(
            nn.Linear(self.config.emb_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.z_dim * 2),
        )

    def forward(self, y: Tensor):
        if self.config.get('learn_prior'):
            y_emb = self.embedder(y)
            encodings = self.model(y_emb)
            mean = encodings[:, :self.config.z_dim]
            log_var = encodings[:, self.config.z_dim:]
        else:
            mean = torch.zeros(y.size(0), self.config.z_dim).to(y.device)
            log_var = torch.zeros(y.size(0), self.config.z_dim).to(y.device)

        return mean, log_var


class FeatVAEEmbedder(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAEEmbedder, self).__init__()

        self.config = config
        if self.config.get('use_attrs_in_vae'):
            self.register_buffer('attrs', torch.tensor(attrs))
            self.model = nn.Linear(self.attrs.size[1], self.config.emb_dim)
        else:
            self.model = nn.Embedding(self.config.num_classes, self.config.emb_dim)

    def forward(self, y: Tensor) -> Tensor:
        # TODO: let's use both attrs and class labels!
        if self.config.get('use_attrs_in_vae'):
            return self.model(self.attrs[y])
        else:
            return self.model(y)
