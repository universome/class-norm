import numpy as np
import torch
import torch.nn as nn
from firelab.config import Config

from src.utils.constants import INPUT_DIMS
from src.models.layers import ResNetLastBlock, Reshape, ConvTransposeBNReLU, RepeatToSize
from src.models.feat_vae import FeatVAE, FeatVAEEncoder, FeatVAEDecoder, FeatVAEPrior, FeatVAEEmbedder


class ConvFeatVAE(FeatVAE):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        nn.Module.__init__(self)

        self.config = config
        self.encoder = ConvFeatVAEEncoder(self.config, attrs)
        self.decoder = ConvFeatVAEDecoder(self.config, attrs)
        self.prior = ConvFeatVAEPrior(self.config, attrs)


class ConvFeatVAEEncoder(FeatVAEEncoder):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        nn.Module.__init__(self)

        self.config = config
        self.embedder = ConvFeatVAEEmbedder(config, attrs, self.config.conv_feat_spatial_size)
        self.model = ResNetLastBlock(self.config.input_type, self.config.pretrained, should_pool=False)


class ConvFeatVAEDecoder(FeatVAEDecoder):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        nn.Module.__init__(self)

        self.config = config
        self.embedder = ConvFeatVAEEmbedder(config, attrs, self.config.z_spatial_size)
        self.model = nn.Sequential(
            ConvTransposeBNReLU(self.config.z_dim + self.config.emb_dim, self.config.hid_dim, 5),
            ConvTransposeBNReLU(self.config.hid_dim, self.config.hid_dim, 6),
            nn.ConvTranspose2d(self.config.hid_dim, INPUT_DIMS[self.config.input_type], 6),
        )


class ConvFeatVAEPrior(FeatVAEPrior):
    def __init__(self, config: Config, attrs: np.ndarray=None):
        nn.Module.__init__(self)

        self.config = config
        self.embedder = ConvFeatVAEEmbedder(config, attrs, self.config.z_spatial_size)
        self.model = nn.Sequential(
            Reshape([-1, self.config.emb_dim, 1, 1]),
            ConvTransposeBNReLU(self.config.emb_dim, self.config.hid_dim, 6),
            ConvTransposeBNReLU(self.config.hid_dim, self.config.hid_dim, 5),
            nn.ConvTranspose2d(self.config.hid_dim, INPUT_DIMS[self.config.input_type], 5),
        )


class ConvFeatVAEEmbedder(FeatVAEEmbedder):
    def __init__(self, config: Config, attrs: np.ndarray=None, target_spatial_size: int=1):
        nn.Module.__init__(self)

        self.config = config

        if self.config.get('use_attrs_in_vae'):
            self.register_buffer('attrs', torch.tensor(attrs))
            first_layer = nn.Linear(self.attrs.size[1], self.config.emb_dim)
        else:
            first_layer = nn.Embedding(self.config.num_classes, self.config.emb_dim)

        self.model = nn.Sequential(
            first_layer,
            nn.ReLU(),
            RepeatToSize(target_spatial_size)
        )
