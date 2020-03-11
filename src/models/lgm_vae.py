import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.feat_vae import FeatVAE
from src.models.conv_feat_vae import ConvFeatVAE
from src.models.classifier import ResnetEmbedder, FeatClassifier, ConvFeatClassifier
from src.utils.training_utils import prune_logits
from src.models.layers import Identity, ResNetConvEmbedder


class LGMVAE(nn.Module):
    def __init__(self, config, attrs: np.ndarray=None):
        super(LGMVAE, self).__init__()

        self.config = config
        if self.config.get('feat_level', 'fc') == 'conv':
            VAEClass = ConvFeatVAE
            ClassifierClass = ConvFeatClassifier
            EmbedderClass = ResNetConvEmbedder
        elif self.config.get('feat_level', 'fc') == 'fc':
            VAEClass = FeatVAE
            ClassifierClass = FeatClassifier
            EmbedderClass = ResnetEmbedder
        else:
            raise NotImplementedError(f'Unknown feat level: {self.config.feat_level}')

        if not attrs is None: self.register_buffer('attrs', torch.tensor(attrs))
        self.vae = VAEClass(self.config, attrs)

        if self.config.get('use_identity_embedder'):
            self.embedder = Identity()
        else:
            self.embedder = EmbedderClass(config.input_type, config.pretrained)

        self.classifier = ClassifierClass(config, attrs)

    def forward(self, x) -> Tensor:
        return self.classifier(self.embedder(x))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        return self.classifier.compute_pruned_predictions(self.embedder(x), output_mask)

    def sample(self, y: Tensor) -> Tensor:
        return self.vae.generate(y)
