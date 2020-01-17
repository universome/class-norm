import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from .feat_gan import FeatGenerator, FeatDiscriminator
from .conv_feat_gan import ConvFeatGenerator, ConvFeatDiscriminator, ConvFeatEmbedder
from .classifier import ResnetEmbedder


class LatGM(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(LatGM, self).__init__()

        self.config = config
        self.register_buffer('attrs', torch.tensor(attrs).clone().detach())

        if self.config.feat_level == 'fc':
            self.embedder = ResnetEmbedder(config)
            self.generator = FeatGenerator(config, attrs)
            self.discriminator = FeatDiscriminator(config, attrs)
        elif self.config.feat_level == 'conv':
            self.embedder = ConvFeatEmbedder(config)
            self.generator = ConvFeatGenerator(config, attrs)
            self.discriminator = ConvFeatDiscriminator(config, attrs)
        else:
            raise NotImplementedError(f'Unknown feat level: {self.config.feat_level}')

    def forward(self, x) -> Tensor:
        return self.discriminator.run_cls_head(self.embedder(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return self.discriminator.compute_pruned_predictions(self.embedder(x), output_mask)


class ConvDecoder(nn.Module):
    """
    If you build an autoencoder, the this is a Decoder part
    """
    def __init__(self, config: Config):
        pass
