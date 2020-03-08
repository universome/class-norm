import torch
import torch.nn as nn
from firelab.config import Config

from src.models.layers import Identity


class Upsampler(nn.Module):
    def __init__(self, config: Config):
        super(Upsampler, self).__init__()

        self.config = config

        if self.config.hp.upsampler.mode == 'none':
            self.model = Identity()
        elif self.config.hp.upsampler.mode == 'learnable':
            self.model = nn.Sequential(
                nn.ConvTranspose2d(3, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                nn.Tanh(),
            )
        else:
            self.model = nn.Upsample((256, 256), mode=self.config.hp.upsampler.mode)

    def forward(self, x):
        return self.model(x)
