from typing import Tuple, Iterable
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import RESNET_FEAT_DIM
from src.models.classifier import ClassifierHead


class FeatGenerator(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatGenerator, self).__init__()

        if config.get('use_attrs_in_gen'):
            assert not attrs is None

            self.register_buffer('attrs', torch.tensor(attrs))
            self.attr_emb = nn.Linear(attrs.shape[1], config.emb_dim)
        else:
            self.cls_emb = nn.Embedding(config.num_classes, config.emb_dim)

        self.config = config
        self.init_model()

    def init_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.config.z_dim + self.config.emb_dim, self.config.hid_dim),
            nn.LeakyReLU(),
            nn.Linear(self.config.hid_dim, RESNET_FEAT_DIM[self.config.resnet_type]),
        )

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        assert z.size(0) == y.size(0), "You should specify necessary y label for each z"

        if self.config.use_attrs_in_gen:
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
        return torch.randn(batch_size, self.config.z_dim)

    def sample(self, y: Tensor) -> Tensor:
        return self.forward(self.sample_noise(len(y)).to(y.device), y)


class FeatDiscriminator(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatDiscriminator, self).__init__()

        assert (not attrs is None) == config.use_attrs_in_discr

        self.config = config

        if self.config.share_body_in_discr:
            self.adv_body = self.init_body()
            self.cls_body = self.adv_body
        else:
            self.adv_body = self.init_body()
            self.cls_body = self.init_body()

        self.adv_head = nn.Linear(self.config.hid_dim, 1)
        self.cls_head = ClassifierHead(self.config, attrs)

    def init_body(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], self.config.hid_dim),
            nn.ReLU()
        )

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


class FeatDiscriminatorWithoutClsHead(nn.Module):
    def __init__(self, config: Config):
        super(FeatDiscriminatorWithoutClsHead, self).__init__()

        self.config = config
        self.model = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], config.hid_dim),
            nn.ReLU(),
            nn.Linear(config.hid_dim, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
