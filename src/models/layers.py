from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet18, resnet34, resnet50


RESNET_TYPE_TO_CLS = {18: resnet18, 34: resnet34, 50: resnet50}


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization by @Kaixhin
    https://github.com/pytorch/pytorch/issues/8985
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

        return out


class Reshape(nn.Module):
    def __init__(self, target_shape: Tuple[int]):
        super(Reshape, self).__init__()

        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ResNetLastBlock(nn.Module):
    def __init__(self, resnet_type: int, pretrained: bool):
        super(ResNetLastBlock, self).__init__()

        self.resnet = RESNET_TYPE_TO_CLS[resnet_type](pretrained=pretrained)

        del self.resnet.conv1
        del self.resnet.bn1
        del self.resnet.relu
        del self.resnet.maxpool
        del self.resnet.layer1
        del self.resnet.layer2
        del self.resnet.layer3
        del self.resnet.fc

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNetConvEmbedder(nn.Module):
    def __init__(self, resnet_type: int, pretrained: bool):
        super(ResNetConvEmbedder, self).__init__()

        self.resnet = RESNET_TYPE_TO_CLS[resnet_type](pretrained=pretrained)

        del self.resnet.layer4
        del self.resnet.avgpool
        del self.resnet.fc

    def forward(self, x: Tensor) -> Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: Any) -> Any:
        return x
