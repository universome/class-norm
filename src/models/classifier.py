import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50
from firelab.config import Config

from src.utils.training_utils import prune_logits
from src.utils.constants import INPUT_DIMS
from src.models.layers import ResNetLastBlock, GaussianDropout
from src.models.attrs_head import AttrsHead


RESNET_CLS = {18: resnet18, 34: resnet34, 50: resnet50}


class Classifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(Classifier, self).__init__()

        self.config = config
        self.attrs = attrs

        self.init_embedder()
        self.init_head()

    @property
    def hid_dim(self) -> int:
        raise NotImplementedError()

    def init_embedder(self):
        raise NotImplementedError('`init_embedder` method is not implemented')

    def init_head(self):
        if self.attrs is None:
            self.head = nn.Linear(self.hid_dim, self.config.data.num_classes)
        else:
            self.head = AttrsHead(self.config.hp.get('head'), self.attrs)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return prune_logits(self.forward(x), output_mask)

    def forward(self, x: Tensor, **head_kwargs) -> Tensor:
        return self.head(self.embedder(x), **head_kwargs)

    def get_head_size(self) -> int:
        return sum(p.numel() for p in self.head.parameters())


class ResnetClassifier(Classifier):
    @property
    def hid_dim(self) -> int:
        return INPUT_DIMS[f'resnet{self.config.hp.classifier.resnet_n_layers}_feat']

    def init_embedder(self):
        self.embedder = ResnetEmbedder(
            self.config.hp.classifier.resnet_n_layers,
            self.config.hp.classifier.pretrained)


class FeatClassifier(Classifier):
    @property
    def hid_dim(self) -> int:
        return self.config.hp.classifier.hid_dim

    def init_embedder(self):
        self.embedder = nn.Sequential(
            # GaussianDropout(self.config.get('cls_gaussian_dropout_sigma', 0.)),
            nn.Linear(self.config.hp.classifier.data_dim, self.config.hp.classifier.hid_dim),
            nn.ReLU(),
        )

class ResnetEmbedder(nn.Module):
    def __init__(self, resnet_n_layers: int=18, pretrained: bool=True):
        super(ResnetEmbedder, self).__init__()

        self.resnet = RESNET_CLS[resnet_n_layers](pretrained=pretrained)

        del self.resnet.fc # So it's not included in the parameters

    def forward(self, x):
        return resnet_embedder_forward(self.resnet, x)


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
