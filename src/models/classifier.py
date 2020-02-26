import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import INPUT_DIMS
from src.models.layers import ResNetLastBlock, GaussianDropout


RESNET_CLS = {18: resnet18, 34: resnet34, 50: resnet50}


class ResnetClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(ResnetClassifier, self).__init__()

        self.embedder = ResnetEmbedder(
            config.hp.classifier.resnet_n_layers,
            config.hp.classifier.pretrained)
        self.head = ClassifierHead(
            INPUT_DIMS[f'resnet{config.hp.classifier.resnet_n_layers}_feat'],
            config.data.num_classes, attrs)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.embedder(x))

    def get_head_size(self) -> int:
        return sum(p.numel() for p in self.head.parameters())


class ResnetEmbedder(nn.Module):
    def __init__(self, resnet_n_layers: int=18, pretrained: bool=True):
        super(ResnetEmbedder, self).__init__()

        self.resnet = RESNET_CLS[resnet_n_layers](pretrained=pretrained)

        del self.resnet.fc # So it's not included in parameters

    def forward(self, x):
        return resnet_embedder_forward(self.resnet, x)


class FeatClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray = None):
        super(FeatClassifier, self).__init__()

        self.config = config
        self.body = self.create_body()
        self.head = ClassifierHead(config.hp.classifier.hid_dim, config.data.num_classes, attrs)

    def create_body(self) -> nn.Module:
        return nn.Sequential(
            # GaussianDropout(self.config.get('cls_gaussian_dropout_sigma', 0.)),
            nn.Linear(self.config.hp.classifier.data_dim, self.config.hp.classifier.hid_dim),
            nn.ReLU(),
        )

    def forward(self, x) -> Tensor:
        return self.head(self.body(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)


class ClassifierHead(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, attrs: np.ndarray = None):
        super(ClassifierHead, self).__init__()

        self.use_attrs = not attrs is None

        if self.use_attrs:
            self.register_buffer('attrs', torch.tensor(attrs))
            self.cls_attr_emb = nn.Linear(attrs.shape[1], hid_dim)
            self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))
        else:
            self.head = nn.Linear(hid_dim, num_classes)

    def forward(self, feats: Tensor) -> Tensor:
        if self.use_attrs:
            attr_embs = self.cls_attr_emb(self.attrs)
            return torch.mm(feats, attr_embs.t()) + self.biases
        else:
            return self.head(feats)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)


class ConvFeatClassifier(FeatClassifier):
    def create_body(self) -> nn.Module:
        return nn.Sequential(
            ResNetLastBlock(self.config.input_type, self.config.pretrained),
            nn.ReLU(),
        )


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
