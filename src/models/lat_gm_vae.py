import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.feat_vae import FeatVAE
from src.models.classifier import ResnetEmbedder
from src.utils.lll import prune_logits
from src.utils.constants import RESNET_FEAT_DIM

class LatGMVAE(nn.Module):
    def __init__(self, config, attrs: np.ndarray=None):
        super(LatGMVAE, self).__init__(config, attrs)

        self.config = config
        self.vae = FeatVAE(self.config, attrs)
        self.embedder = ResnetEmbedder(config.pretrained)
        self.clf_body = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], self.config.hid_dim),
            nn.ReLU()
        )

        if config.use_attrs:
            assert not attrs is None, "You should provide attrs to use attrs"

            self.register_buffer('attrs', torch.tensor(attrs))
            self.cls_attr_emb = nn.Linear(attrs.shape[1], config.hid_dim)
            self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))
        else:
            self.cls_head = nn.Linear(self.config.hid_dim, self.config.num_classes)

    def forward(self, x) -> Tensor:
        return self.compute_cls_logits(self.embedder(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)

    def compute_cls_logits(self, embs: Tensor) -> Tensor:
        feats = self.cls_body(embs)

        if self.config.use_attrs:
            attr_embs = self.cls_attr_emb(self.attrs)
            return torch.mm(feats, attr_embs.t()) + self.biases
        else:
            return self.cls_head(feats)
