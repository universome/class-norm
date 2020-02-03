import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.feat_gan import FeatGenerator, FeatDiscriminator
from src.models.conv_feat_gan import ConvFeatGenerator, ConvFeatDiscriminator, ConvFeatEmbedder
from src.models.classifier import ResnetEmbedder
from src.models.layers import Identity


class LatGM(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(LatGM, self).__init__()

        self.config = config
        self.register_buffer('attrs', torch.tensor(attrs))

        if self.config.feat_level == 'fc':
            if self.config.get('identity_embedder'):
                self.embedder = Identity()
            else:
                self.embedder = ResnetEmbedder(pretrained=config.pretrained)
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

    def reset_discriminator(self):
        if self.config.feat_level == 'fc':
            new_state_dict = FeatDiscriminator(self.config, self.attrs.cpu().numpy()).state_dict()
        elif self.config.feat_level == 'conv':
            new_state_dict = ConvFeatDiscriminator(self.config, self.attrs.cpu().numpy()).state_dict()
        else:
            raise NotImplementedError(f'Unknown feat level: {self.config.feat_level}')

        self.discriminator.load_state_dict(new_state_dict)

    def sample(self, y: Tensor) -> Tensor:
        return self.generator.sample(y)


class ConvDecoder(nn.Module):
    """
    If you build an autoencoder, the this is a Decoder part
    """
    def __init__(self, config: Config):
        pass
