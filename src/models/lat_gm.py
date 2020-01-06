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
        self.register_buffer('attrs', torch.tensor(attrs).clone().detach())

        if attrs is None:
            self.classifier = ResnetClassifier(pretrained=config.get('pretrained'))
        else:
            self.classifier = ZSClassifier(attrs, pretrained=config.get('pretrained'))

        self.init_gm()

    def init_gm(self):
        if self.config.use_attrs_in_gen:
            self.generator = FeatGenerator(self.config, self.attrs.cpu().numpy())
        else:
            self.generator = FeatGenerator(self.config)

        self.discriminator = FeatDiscriminator(self.config)

    def forward(self, *inputs) -> Tensor:
        return self.classifier(*inputs)

    def compute_pruned_predictions(self, *inputs) -> Tensor:
        return self.classifier.compute_pruned_predictions(*inputs)

    def embed(self, x: Tensor) -> Tensor:
        return self.classifier.embedder(x)
