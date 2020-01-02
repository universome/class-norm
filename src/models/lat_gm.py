import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .feat_gan import FeatGenerator, FeatDiscriminator
from .classifier import ZSClassifier, ResnetClassifier


class LatGM(nn.Module):
    def __init__(self, config, attrs: np.ndarray=None):
        super(LatGM, self).__init__()

        self.config = config
        self.register_buffer('attrs', torch.tensor(attrs))

        if attrs is None:
            self.classifier = ResnetClassifier(pretrained=config.get('pretrained'))
        else:
            self.classifier = ZSClassifier(attrs, pretrained=config.get('pretrained'))

        self.generator = FeatGenerator(config, attrs)
        self.discriminator = FeatDiscriminator(config)

    def forward(self, *inputs) -> Tensor:
        return self.classifier(*inputs)

    def compute_pruned_predictions(self, *inputs) -> Tensor:
        return self.classifier.compute_pruned_predictions(*inputs)

    def embed(self, x: Tensor) -> Tensor:
        return self.classifier.embedder(x)
