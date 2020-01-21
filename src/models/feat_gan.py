from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import RESNET_FEAT_DIM


class FeatGenerator(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(FeatGenerator, self).__init__()

        if config.get('use_attrs_in_gen'):
            assert not attrs is None

            self.register_buffer('attrs', torch.tensor(attrs).clone().detach())
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

        self.config = config
        self.init_body()
        self.adv_head = nn.Linear(self.config.hid_dim, 1)

        if config.use_attrs_in_discr:
            assert not attrs is None, "You should provide attrs to use attrs"

            self.register_buffer('attrs', torch.tensor(attrs).clone().detach())
            self.cls_attr_emb = nn.Linear(attrs.shape[1], config.hid_dim)
            self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))
        else:
            self.cls_head = nn.Linear(self.config.hid_dim, self.config.num_classes)

    def init_body(self):
        self.body = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], self.config.hid_dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.body(x)
        discr_logits = self.adv_head(feats)
        cls_logits = self.compute_cls_logits(feats)

        return discr_logits, cls_logits

    def run_adv_head(self, x: Tensor) -> Tensor:
        return self.adv_head(self.body(x))

    def run_cls_head(self, x: Tensor) -> Tensor:
        return self.compute_cls_logits(self.body(x))

    def compute_cls_logits(self, feats: Tensor) -> Tensor:
        if self.config.use_attrs_in_discr:
            attr_embs = self.cls_attr_emb(self.attrs)
            return torch.mm(feats, attr_embs.t()) + self.biases
        else:
            return self.cls_head(feats)

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
