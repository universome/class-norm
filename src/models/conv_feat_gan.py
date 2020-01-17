import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet34, conv3x3
from firelab.config import Config

from src.utils.lll import prune_logits
from src.utils.constants import RESNET_FEAT_DIM, RESNET_CONV_FEAT_DIM
from src.models.layers import ResNetLastBlock, ResNetConvEmbedder, Reshape
from src.models.feat_gan import FeatGenerator, FeatDiscriminator


class ConvFeatEmbedder(nn.Module):
    def __init__(self, config):
        super(ConvFeatEmbedder, self).__init__()

        self.model = nn.Sequential(
            ResNetConvEmbedder(config.resnet_type, config.pretrained),
            conv3x3(RESNET_CONV_FEAT_DIM[config.resnet_type], RESNET_CONV_FEAT_DIM[config.resnet_type]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class ConvFeatDiscriminator(FeatDiscriminator):
    def init_model(self):
        self.body = ResNetLastBlock(self.config.resnet_type, pretrained=False)
        self.cls_head = nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], self.config.num_classes)
        self.adv_head = nn.Linear(RESNET_FEAT_DIM[self.config.resnet_type], 1)


class ConvFeatGenerator(FeatGenerator):
    def init_model(self):
        self.model = nn.Sequential(
            Reshape([-1, self.config.z_dim + self.config.emb_dim, 1, 1]),
            ConvBatchNormReluBlock(self.config.z_dim + self.config.emb_dim, 512, 6),
            ConvBatchNormReluBlock(512, 512, 5),
            nn.ConvTranspose2d(512, RESNET_CONV_FEAT_DIM[self.config.resnet_type], 5),
            nn.Tanh(),
        )


class ConvBatchNormReluBlock(nn.Module):
    def __init__(self, num_in_c: int, num_out_c: int, kernel_size: int, *conv_args):
        super(ConvBatchNormReluBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(num_in_c, num_out_c, kernel_size, *conv_args),
            nn.BatchNorm2d(num_out_c),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
