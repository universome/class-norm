import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.models.feat_gan import FeatGenerator, FeatDiscriminator
from src.models.conv_feat_gan import ConvFeatGenerator, ConvFeatDiscriminator, ConvFeatEmbedder
from src.models.classifier import ResnetEmbedder
from src.models.layers import Identity


class LGM(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        super(LGM, self).__init__()

        self.config = config

        if not attrs is None:
            self.register_buffer('attrs', torch.tensor(attrs))

        if len(self.config.data.feat_dims) == 1:
            if self.config.hp.embedder.get('use_identity_embedder'):
                self.embedder = Identity()
            else:
                self.embedder = ResnetEmbedder(
                    self.config.hp.embedder.resnet_n_layers,
                    self.config.hp.embedder.pretrained)
            self.generator = FeatGenerator(config, attrs)
            self.discriminator = FeatDiscriminator(config, attrs)
        elif len(self.config.data.feat_dims) == 3:
            self.embedder = ConvFeatEmbedder(config)
            self.generator = ConvFeatGenerator(config, attrs)
            self.discriminator = ConvFeatDiscriminator(config, attrs)
        else:
            raise NotImplementedError(f'Unsupported feat dim: {self.config.data.feat_dims}')

    def forward(self, x) -> Tensor:
        return self.discriminator.run_cls_head(self.embedder(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return self.discriminator.compute_pruned_predictions(self.embedder(x), output_mask)

    def reset_discriminator(self):
        if self.config.feat_level == 'fc':
            new_state_dict = FeatDiscriminator(self.config, self._get_attrs()).state_dict()
        elif self.config.feat_level == 'conv':
            new_state_dict = ConvFeatDiscriminator(self.config, self._get_attrs()).state_dict()
        else:
            raise NotImplementedError(f'Unsupported feat dim: {self.config.data.feat_dims}')

        self.discriminator.load_state_dict(new_state_dict)

    def sample(self, y: Tensor) -> Tensor:
        return self.generator.sample(y)

    def _get_attrs(self):
        if hasattr(self, 'attrs'):
            return self.attrs.cpu().numpy()
        else:
            return None


class ConvDecoder(nn.Module):
    """
    If you build an autoencoder, the this is a Decoder part
    """
    def __init__(self, config: Config):
        pass
