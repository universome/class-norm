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
    def __init__(self, resnet_type: int, pretrained: bool, should_pool: bool=True):
        super(ResNetLastBlock, self).__init__()

        self.resnet = RESNET_TYPE_TO_CLS[resnet_type](pretrained=pretrained)
        self.should_pool = should_pool

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

        if self.should_pool:
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


class ConvTransposeBNReLU(nn.Module):
    def __init__(self, num_in_c: int, num_out_c: int, kernel_size: int, *conv_args):
        super(ConvTransposeBNReLU, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(num_in_c, num_out_c, kernel_size, *conv_args),
            nn.BatchNorm2d(num_out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class RepeatToSize(nn.Module):
    def __init__(self, target_size: int):
        super(RepeatToSize, self).__init__()

        self.target_size = target_size

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2

        return x.view(x.size(0), x.size(1), 1, 1).repeat(1, 1, self.target_size, self.target_size)


class GaussianDropout(nn.Module):
    def __init__(self, sigma: float):
        super(GaussianDropout, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training or self.sigma == 0:
            return x
        else:
            return x + self.sigma * torch.randn_like(x)
