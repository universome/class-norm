import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.models.resnet import resnet18
from firelab.config import Config

from src.utils.lll import prune_logits


class ZSClassifier(nn.Module):
    def __init__(self, attrs:np.ndarray, pretrained:bool=False):
        super(ZSClassifier, self).__init__()

        self.embedder = ResnetEmbedder(pretrained=pretrained)
        self.register_buffer('attrs', torch.tensor(attrs).float())
        self.attr_emb = nn.Linear(attrs.shape[1], 512, bias=False)
        self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, x: Tensor) -> Tensor:
        img_feats = self.embedder(x)
        attrs_feats = self.attr_emb(self.attrs)
        logits = torch.mm(img_feats, attrs_feats.t()) + self.biases

        return logits

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        logits = self.forward(x)
        pruned_logits = prune_logits(logits, output_mask)

        return pruned_logits

    def get_head_size(self) -> int:
        return self.biases.numel() + self.attr_emb.weight.numel()


class ResnetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool=False):
        super(ResnetClassifier, self).__init__()

        self.embedder = ResnetEmbedder(pretrained=pretrained)
        self.head = nn.Linear(512, num_classes)

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        logits = self.forward(x)
        pruned_logits = prune_logits(logits, output_mask)

        return pruned_logits

    def forward(self, x: Tensor) -> Tensor:
        feats = self.embedder(x)
        logits = self.head(feats)

        return logits


class ResnetEmbedder(nn.Module):
    def __init__(self, pretrained: bool=True):
        super(ResnetEmbedder, self).__init__()

        self.resnet = resnet18(pretrained=pretrained)

        del self.resnet.fc # So it's not included in parameters

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class FeatClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super(FeatClassifier, self).__init__()

        self.embedder = nn.Sequential(
            nn.Linear(config.feat_dim, config.hid_dim),
            # nn.ReLU(),
            # nn.Linear(config.hid_dim, config.hid_dim)
        )
        self.register_buffer('attrs', torch.tensor(attrs).float())
        self.attr_emb = nn.Sequential(
            nn.Linear(attrs.shape[1], config.hid_dim, bias=False),
            # nn.ReLU(),
            # nn.Linear(config.hid_dim, config.hid_dim, bias=False),
        )
        self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, x: Tensor) -> Tensor:
        img_feats = self.embedder(x)
        attrs_feats = self.attr_emb(self.attrs)
        logits = torch.mm(img_feats, attrs_feats.t()) + self.biases

        return logits

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        logits = self.forward(x)
        pruned_logits = prune_logits(logits, output_mask)

        return pruned_logits

# class FeatClassifier(nn.Module):
#     def __init__(self, config: Config, attrs: np.ndarray):
#         super(FeatClassifier, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(config.feat_dim, config.hid_dim),
#             nn.ReLU(),
#             nn.Linear(config.hid_dim, 200)
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#     def compute_pruned_predictions(self, x, output_mask):
#         logits = self.model(x)
#         pruned = prune_logits(logits, output_mask)
#
#         return pruned
