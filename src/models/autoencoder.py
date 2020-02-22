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
        # self.encoder = nn.Conv2d(3, 16, 3, padding=1)
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(16, 3, 3, padding=1),
        #     nn.Sigmoid()
        # )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
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