from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.lll import prune_logits
from src.models.layers import ConditionalBatchNorm2d, Reshape, Flatten


class GAN(nn.Module):
    def __init__(self, config: Config):
        super(GAN, self).__init__()

        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, *input):
        _, cls_logits = self.discriminator(*input)

        return cls_logits

    def compute_pruned_predictions(self, x, output_mask) -> Tensor:
        logits = self.forward(x)
        pruned_logits = prune_logits(logits, output_mask)

        return pruned_logits


class Generator(nn.Module):
    def __init__(self, config: Config):
        super(Generator, self).__init__()

        self.config = config

        self.linear1 = nn.Linear(config.hp.generator.z_dim, 4 * 4 * 4 * config.hp.generator.dim)
        self.reshape1 = Reshape([-1, 4 * config.hp.generator.dim, 4, 4])
        self.bn1 = ConditionalBatchNorm2d(4 * config.hp.generator.dim, config.data.num_classes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(config.hp.generator.dim * 4, config.hp.generator.dim * 2, 4, stride=1)
        self.bn2 = ConditionalBatchNorm2d(2 * config.hp.generator.dim, config.data.num_classes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(config.hp.generator.dim * 2, config.hp.generator.dim, 4, stride=2)
        self.bn3 = ConditionalBatchNorm2d(config.hp.generator.dim, config.data.num_classes)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(config.hp.generator.dim, config.data.num_img_channels, 4, stride=2, padding=1)
        self.tanh4 = nn.Tanh()

    def sample_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.config.hp.generator.z_dim)

    def sample(self, y: Tensor) -> Tensor:
        z = self.sample_noise(len(y)).to(y.device)

        return self.forward(z, y)

    def forward(self, z: Tensor, y: Tensor):
        x = self.linear1(z)
        x = self.reshape1(x)
        x = self.bn1(x, y)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, y)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x, y)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.tanh4(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, config: Config):
        super(Discriminator, self).__init__()

        self.feat_extractor = nn.Sequential(
            Reshape([-1, config.data.num_img_channels, config.data.img_size, config.data.img_size]),
            nn.Conv2d(config.data.num_img_channels, config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(config.hp.discriminator.dim, 2 * config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(2 * config.hp.discriminator.dim, 4 * config.hp.discriminator.dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            Flatten(),
        )

        self.adv_head = nn.Linear(4 * 4 * 4 * config.hp.discriminator.dim, 1)
        self.cls_head = nn.Linear(4 * 4 * 4 * config.hp.discriminator.dim, config.data.num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.feat_extractor(x)
        discr_out = self.adv_head(feats)
        cls_out = self.cls_head(feats)

        return discr_out, cls_out

    def run_adv_head(self, x: Tensor) -> Tensor:
        return self.adv_head(self.feat_extractor(x))
