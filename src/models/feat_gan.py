from typing import Tuple, Iterable
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import INPUT_DIMS
from src.models.classifier import ClassifierHead


class FeatGenerator(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatGenerator, self).__init__()

        if config.hp.generator.use_attrs:
            assert not attrs is None

            self.register_buffer('attrs', torch.tensor(attrs))
            self.attr_emb = nn.Linear(attrs.shape[1], config.hp.generator.emb_dim)
        else:
            self.cls_emb = nn.Embedding(config.data.num_classes, config.hp.generator.emb_dim)

        self.config = config
        self.init_model()

    def init_model(self):
        layers = [
            nn.Linear(self.config.hp.generator.z_dim + self.config.hp.generator.emb_dim, self.config.hp.generator.hid_dim),
            nn.LeakyReLU(),
        ]

        for _ in range(self.config.hp.generator.num_layers - 1):
            layers.append(nn.Linear(self.config.hp.generator.hid_dim, self.config.hp.generator.hid_dim))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(self.config.hp.generator.hid_dim, INPUT_DIMS[self.config.data.input_type]))

        if self.config.get('use_tanh_in_gen'):
            layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        assert z.size(0) == y.size(0), "You should specify necessary y label for each z"

        if self.config.hp.generator.use_attrs:
            embs = self.attr_emb(self.attrs[y])
        else:
            embs = self.cls_emb(y)

        x = torch.cat([z, embs], dim=1)
        x = self.model(x)

        return x

    def forward_with_attr(self, z: Tensor, attr: Tensor) -> Tensor:
        assert z.size(0) == attr.size(0), "You should specify necessary attr for each z yourself"

        attr_feats = self.attr_emb(attr)
        x = torch.cat([z, attr_feats], dim=1)
        x = self.model(x)

        return x

    def sample_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.config.hp.generator.z_dim)

    def sample(self, y: Tensor) -> Tensor:
        return self.forward(self.sample_noise(len(y)).to(y.device), y)


class FeatDiscriminator(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatDiscriminator, self).__init__()

        self.config = config
        if self.config.hp.discriminator.use_attrs:
            assert (not attrs is None)

        self.adv_body = self.init_body()

        if self.config.hp.discriminator.share_body:
            self.cls_body = self.adv_body
        else:
            self.cls_body = self.init_body()

        self.adv_head = nn.Linear(self.config.hp.discriminator.hid_dim, 1)
        self.cls_head = ClassifierHead(self.config.hp.classifier, attrs)

    def init_body(self, conditional:bool=False) -> nn.Module:
        layers = [
            nn.Linear(INPUT_DIMS[self.config.data.input_type], self.config.hp.discriminator.hid_dim),
            nn.ReLU(),
        ]

        for _ in range(self.config.hp.discriminator.num_layers - 1):
            layers.append(nn.Linear(self.config.hp.discriminator.hid_dim, self.config.hp.discriminator.hid_dim))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def get_adv_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.adv_body.parameters(), self.adv_head.parameters())

    def get_cls_parameters(self) -> Iterable[nn.Parameter]:
        return chain(self.cls_body.parameters(), self.cls_head.parameters())

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        adv_feats, cls_feats = self.compute_feats(x)
        discr_logits = self.adv_head(adv_feats)
        cls_logits = self.cls_head(cls_feats)

        return discr_logits, cls_logits

    def compute_feats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        adv_feats = self.adv_body(x)

        if self.config.share_body_in_discr:
            cls_feats = adv_feats
        else:
            cls_feats = self.cls_body(x)

        return adv_feats, cls_feats

    def run_adv_head(self, x: Tensor) -> Tensor:
        return self.adv_head(self.adv_body(x))

    def run_cls_head(self, x: Tensor) -> Tensor:
        return self.cls_head(self.cls_body(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.run_cls_head(x), output_mask)


class ConditionalFeatDiscriminator(FeatDiscriminator):
    def __init__(self, config):
        nn.Module.__init__(self)

        self.embedder = nn.Embedding(config.data.num_classes, config.hp.discriminator.emb_dim)
        self.model = nn.Sequential(
            nn.Linear(config.hp.discriminator.emb_dim + INPUT_DIMS[config.data.input_type], config.hp.discriminator.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hp.discriminator.hid_dim, 1)
        )

    def forward(self, x, y):
        inputs = torch.cat([x, self.embedder(y)], dim=1)

        return self.model(inputs)

class FeatDiscriminatorWithoutClsHead(nn.Module):
    def __init__(self, config: Config):
        super(FeatDiscriminatorWithoutClsHead, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(INPUT_DIMS[config.data.input_type], config.hp.discriminator.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hp.discriminator.hid_dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
