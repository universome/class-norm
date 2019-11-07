from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.classifier import FeatClassifier


class FeatVAEClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super(FeatVAEClassifier, self).__init__()

        self.classifier = FeatClassifier(config, attrs)
        self.vae = FeatCVAE(config)

    def forward(self, *input):
        return self.classifier.forward(*input)

    def compute_pruned_predictions(self, *args):
        return self.classifier.compute_pruned_predictions(*args)


class FeatCVAE(nn.Module):
    def __init__(self, config: Config):
        super(FeatCVAE, self).__init__()

        self.config = config
        self.enc_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim)
        self.encoder = nn.Sequential(
            nn.Linear(config.feat_dim + self.config.class_emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.z_dim * 2),
        )

        self.dec_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim) # TODO: share weights?
        self.decoder = nn.Sequential(
            nn.Linear(config.z_dim + self.config.class_emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.feat_dim),
        )
        self.prior_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim)
        self.prior = nn.Sequential(
            nn.Linear(self.config.class_emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, config.z_dim * 2),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_var = self.encode(x, y)
        z = self.sample(mean, log_var)
        x_rec = self.decode(z, y)

        return x_rec, mean, log_var

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_emb = self.enc_class_emb(y)
        inputs = torch.cat([x, y_emb], dim=1)
        encodings = self.encoder(inputs)
        mean = encodings[:, :self.config.z_dim]
        log_var = encodings[:, self.config.z_dim:]

        return mean, log_var

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        y_emb = self.dec_class_emb(y)
        inputs = torch.cat([z, y_emb], dim=1)

        return self.decoder(inputs)

    def sample(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Samples z ~ N(mean, sigma)"""
        return torch.randn_like(log_var) * (log_var / 2).exp() + mean

    def sample_z_from_prior(self, y: Tensor) -> Tensor:
        mean, log_var = self.get_prior_distribution(y)
        eps = torch.randn_like(log_var)
        z = mean + eps * (log_var / 2).exp()

        return z

    def get_prior_distribution(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_emb = self.prior_class_emb(y)
        encodings = self.prior(y_emb)
        mean = encodings[:, :self.config.z_dim]
        log_var = encodings[:, self.config.z_dim:]

        return mean, log_var

    def generate(self, y: Tensor) -> Tensor:
        z = self.sample_z_from_prior(y)
        x = self.decode(z, y)

        return x
