import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import ConvBNReLU, ConvTransposeBNReLU, Reshape


import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import ConvBNReLU, ConvTransposeBNReLU, Reshape


class AutoEncoder(nn.Module):
    def __init__(self, config: Config):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            ConvBNReLU(3, 16, 3, False),
            ConvBNReLU(16, 64, 3, False, stride=2),
            ConvBNReLU(64, 256, 3, False, stride=2),
            ConvBNReLU(256, 512, 3, False, stride=2),
            Reshape([-1, 2048])
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            Reshape([-1, 512, 2, 2]),
            ConvTransposeBNReLU(512, 256, 3, stride=2, output_padding=1),
            ConvTransposeBNReLU(256, 256, 3, stride=2, output_padding=1),
            ConvTransposeBNReLU(256, 256, 3, stride=2, output_padding=1),
            ConvTransposeBNReLU(256, 3, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ShapePrinter(nn.Module):
    def __init__(self, title: str='unknown'):
        super(ShapePrinter, self).__init__()
        self.title = title

    def forward(self, x):
        print(self.title, x.shape, x.flatten(1).shape)
        return x