import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import RESNET_FEAT_DIM


RESNET_CLS = {18: resnet18, 34: resnet34, 50: resnet50}


class ResnetClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(ResnetClassifier, self).__init__()

        self.embedder = ResnetEmbedder(config.pretrained, config.resnet_type)
        self.head = ClassifierHead(config, attrs)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embedder(x))

    def get_head_size(self) -> int:
        return sum(p.numel() for p in self.head.parameters())


class ResnetEmbedder(nn.Module):
    def __init__(self, pretrained: bool=True, resnet_type: int=18):
        super(ResnetEmbedder, self).__init__()

        self.resnet = RESNET_CLS[resnet_type](pretrained=pretrained)

        del self.resnet.fc # So it's not included in parameters

    def forward(self, x):
        return resnet_embedder_forward(self.resnet, x)


class FeatClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray = None):
        super(FeatClassifier, self).__init__()

        self.body = nn.Sequential(
            nn.Linear(RESNET_FEAT_DIM[config.resnet_type], config.hid_dim),
            nn.ReLU(),
        )
        self.head = ClassifierHead(config, attrs)

    def forward(self, x) -> Tensor:
        return self.head(self.body(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)


class ClassifierHead(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray = None):
        super(ClassifierHead, self).__init__()

        self.use_attrs = not attrs is None

        if self.use_attrs:
            self.register_buffer('attrs', torch.tensor(attrs))
            self.cls_attr_emb = nn.Linear(attrs.shape[1], config.hid_dim)
            self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))
        else:
            self.head = nn.Linear(config.hid_dim, config.num_classes)

    def forward(self, feats: Tensor) -> Tensor:
        if self.use_attrs:
            attr_embs = self.cls_attr_emb(self.attrs)
            return torch.mm(feats, attr_embs.t()) + self.biases
        else:
            return self.head(feats)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)


def resnet_embedder_forward(resnet: nn.Module, x: Tensor) -> Tensor:
    """
    Runs embedder part of a resnet model
    """
    x = resnet.conv1(x)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)

    x = resnet.avgpool(x)
    x = torch.flatten(x, 1)

    return x
