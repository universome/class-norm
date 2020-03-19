import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet34, conv3x3
from firelab.config import Config

from src.utils.training_utils import prune_logits
from src.utils.constants import INPUT_DIMS
from src.models.layers import ResNetLastBlock, ResNetConvEmbedder, Reshape, ConvTransposeBNReLU
from src.models.feat_gan import FeatGenerator, FeatDiscriminator


class ConvFeatEmbedder(nn.Module):
    def __init__(self, config):
        super(ConvFeatEmbedder, self).__init__()

        self.model = nn.Sequential(
            ResNetConvEmbedder(config.input_type, config.pretrained),
            # conv3x3(INPUT_DIMS[config.input_type], INPUT_DIMS[config.input_type]),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class ConvFeatDiscriminator(FeatDiscriminator):
    def init_body(self) -> nn.Module:
        return nn.Sequential(
            ResNetLastBlock(self.config.input_type, self.config.pretrained),
            nn.ReLU(),
            nn.Linear(INPUT_DIMS[self.config.input_type], self.config.hid_dim),
            nn.ReLU()
        )


class ConvFeatGenerator(FeatGenerator):
    def init_model(self):
        self.model = nn.Sequential(
            Reshape([-1, self.config.z_dim + self.config.emb_dim, 1, 1]),
            ConvTransposeBNReLU(self.config.z_dim + self.config.emb_dim, 512, 6),
            ConvTransposeBNReLU(512, 512, 5),
            nn.ConvTranspose2d(512, INPUT_DIMS[self.config.input_type], 5),
            nn.Tanh(),
        )
