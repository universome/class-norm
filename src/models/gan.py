from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.lll import prune_logits
from .layers import ConditionalBatchNorm2d, Reshape, Flatten


CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?


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

        self.linear1 = nn.Linear(config.z_dim, 4 * 4 * 4 * config.gen_dim)
        self.reshape1 = Reshape([-1, 4 * config.gen_dim, 4, 4])
        self.bn1 = ConditionalBatchNorm2d(4 * config.gen_dim, config.num_classes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(config.gen_dim * 4, config.gen_dim * 2, 4, stride=1)
        self.bn2 = ConditionalBatchNorm2d(2 * config.gen_dim, config.num_classes)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(config.gen_dim * 2, config.gen_dim, 4, stride=2)
        self.bn3 = ConditionalBatchNorm2d(config.gen_dim, config.num_classes)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(config.gen_dim, config.num_img_channels, 4, stride=2, padding=1)
        self.tanh4 = nn.Tanh()

    def sample_noise(self, batch_size: int) -> Tensor:
        return torch.randn(batch_size, self.config.z_dim)

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
            Reshape([-1, config.num_img_channels, config.img_size, config.img_size]),
            nn.Conv2d(config.num_img_channels, config.discr_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(config.discr_dim, 2 * config.discr_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            nn.Conv2d(2 * config.discr_dim, 4 * config.discr_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),

            Flatten(),
        )

        self.discr_head = nn.Linear(4 * 4 * 4 * config.discr_dim, 1)
        self.cls_head = nn.Linear(4 * 4 * 4 * config.discr_dim, config.num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feats = self.feat_extractor(x)
        discr_out = self.discr_head(feats)
        cls_out = self.cls_head(feats)

        return discr_out, cls_out

    def run_discr_head(self, x: Tensor) -> Tensor:
        return self.discr_head(self.feat_extractor(x))
