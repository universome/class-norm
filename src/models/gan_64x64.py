import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.layers import Reshape, Flatten
from src.models.gan import GAN


class GAN64x64(GAN):
    def __init__(self, config: Config):
        nn.Module.__init__(self)

        self.generator = Generator64x64(config)
        self.discriminator = Discriminator64x64(config)


class Generator64x64(nn.Module):
    # TODO: why not using conditional batchnorm in generator 64x64?
    def __init__(self, config: Config):
        super(Generator64x64, self).__init__()

        self.config = config
        self.cls_embedder = nn.Embedding(config.num_classes, config.emb_dim)
        self.model = nn.Sequential(
            nn.Linear(config.z_dim + config.emb_dim, 4 * 4 * 8 * 64),
            Reshape([-1, 8 * 64, 4, 4]),
            GeneratorResidualBlock(8 * 64, 8 * 64),
            GeneratorResidualBlock(8 * 64, 4 * 64),
            GeneratorResidualBlock(4 * 64, 2 * 64),
            GeneratorResidualBlock(2 * 64, 1 * 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, y: Tensor):
        cls_embs = self.cls_embedder(y)
        inputs = torch.cat([z, cls_embs], dim=1)

        return self.model(inputs)

    def sample_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.config.z_dim)

    def sample(self, y: Tensor) -> Tensor:
        z = self.sample_noise(len(y)).to(y.device)

        return self.forward(z, y)


class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(GeneratorResidualBlock, self).__init__()

        self.shortcut = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=True, stride=2, output_padding=1)
        self.main = nn.Sequential(
            # TODO: starting with batchnorm is crazy...
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, bias=False, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)


class Discriminator64x64(nn.Module):
    # TODO: why layernorm and not batchnorm in discriminator?
    def __init__(self, config: Config):
        super(Discriminator64x64, self).__init__()

        self.embedder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), # TODO: no he_init
            DiscriminatorResidualBlock(64, 64, 2 * 64),
            DiscriminatorResidualBlock(32, 2 * 64, 4 * 64),
            DiscriminatorResidualBlock(16, 4 * 64, 8 * 64),
            DiscriminatorResidualBlock(8, 8 * 64, 8 * 64),
            # TODO: add some pooling because output is too large
            Flatten(),
        )

        self.adv_head = nn.Linear(4 * 4 * 8 * 64, 1)
        self.cls_head = nn.Linear(4 * 4 * 8 * 64, config.num_classes)

    def forward(self, x):
        feats = self.embedder(x)

        return self.adv_head(feats), self.cls_head(feats)

    def run_adv_head(self, x: Tensor) -> Tensor:
        return self.adv_head(self.embedder(x))


class DiscriminatorResidualBlock(nn.Module):
    def __init__(self, size: int, in_channels: int, out_channels: int):
        super(DiscriminatorResidualBlock, self).__init__()

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, 2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

        self.main = nn.Sequential(
            nn.LayerNorm([in_channels, size, size]),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, bias=False, padding=1),
            nn.LayerNorm([in_channels, size, size]),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + self.shortcut(x)
