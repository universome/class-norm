from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.classifier import ZSClassifier


class FeatVAEClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super(FeatVAEClassifier, self).__init__()

        self.config = config
        self.vae = FeatVAE(config, attrs)
        self.classifier = ZSClassifier(pretrained=self.config.hp.model.get('pretrained'))

    def forward(self, x: Tensor):
        f = self.feat_extractor(x)
        logits = self.classifier(f)

        return logits

    def compute_pruned_predictions(self, x, output_mask):
        f = self.feat_extractor(x)
        logits = self.classifier.compute_pruned_predictions(f, output_mask)

        return logits


class FeatVAE(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatVAE, self).__init__()

        self.config = config

        if config.get('use_attrs_in_vae'):
            assert not attrs is None

            self.register_buffer('attrs', torch.tensor(attrs).clone().detach())
            self.enc_attr_emb = nn.Linear(attrs.shape[1], config.emb_dim)
            self.dec_attr_emb = nn.Linear(attrs.shape[1], config.emb_dim)
            self.prior_attr_emb = nn.Linear(attrs.shape[1], config.emb_dim)
        else:
            self.enc_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim)
            self.dec_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim) # TODO: share weights?
            self.prior_class_emb = nn.Embedding(self.config.num_classes, self.config.class_emb_dim)

        self.encoder = nn.Sequential(
            nn.Linear(config.feat_dim + self.config.emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.z_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(config.z_dim + self.config.emb_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, config.feat_dim),
        )

        self.prior = nn.Sequential(
            nn.Linear(self.config.emb_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.hid_dim),
            nn.ReLU(),
            nn.Linear(self.config.hid_dim, self.config.z_dim * 2),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mean, log_var = self.encode(x, y)
        z = self.sample(mean, log_var)
        x_rec = self.decode(z, y)

        return x_rec, mean, log_var

    def compute_enc_class_emb(self, y: Tensor) -> Tensor:
        if self.config.get('use_attrs_in_vae'):
            return self.enc_attr_emb(self.attrs[y])
        else:
            return self.enc_class_emb(y)

    def compute_dec_class_emb(self, y: Tensor) -> Tensor:
        if self.config.get('use_attrs_in_vae'):
            return self.dec_attr_emb(self.attrs[y])
        else:
            return self.dec_class_emb(y)

    def compute_prior_class_emb(self, y: Tensor) -> Tensor:
        if self.config.get('use_attrs_in_vae'):
            return self.prior_attr_emb(self.attrs[y])
        else:
            return self.prior_class_emb(y)

    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        y_emb = self.compute_enc_class_emb(y)
        inputs = torch.cat([x, y_emb], dim=1)
        encodings = self.encoder(inputs)
        mean = encodings[:, :self.config.z_dim]
        log_var = encodings[:, self.config.z_dim:]

        return mean, log_var

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        y_emb = self.compute_dec_class_emb(y)
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
        if self.config.get('learn_prior_dist'):
            y_emb = self.compute_prior_class_emb(y)
            encodings = self.prior(y_emb)
            mean = encodings[:, :self.config.z_dim]
            log_var = encodings[:, self.config.z_dim:]
        else:
            mean = torch.zeros(y.size(0), self.config.z_dim).to(y.device)
            log_var = torch.zeros(y.size(0), self.config.z_dim).to(y.device)

        return mean, log_var

    def generate(self, y: Tensor) -> Tensor:
        z = self.sample_z_from_prior(y)
        x = self.decode(z, y)

        return x
