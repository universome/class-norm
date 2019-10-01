import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


class ZSClassifier(nn.Module):
    def __init__(self, attrs:np.ndarray, pretrained:bool=False):
        super(ZSClassifier, self).__init__()

        self.resnet_emb = ResnetEmbedder(pretrained=pretrained)
        self.register_buffer('attrs', torch.tensor(attrs).float())
        self.attr_emb = nn.Linear(attrs.shape[1], 512)
        self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, x):
        img_feats = self.resnet_emb(x)
        attrs_feats = self.attr_emb(self.attrs)
        logits = torch.mm(img_feats, attrs_feats.t()) + self.biases

        return logits


class ResnetEmbedder(nn.Module):
    def __init__(self, pretrained:bool=True):
        super(ResnetEmbedder, self).__init__()

        self.resnet = resnet18(pretrained=pretrained)

        del self.resnet.fc # So it's not included in parameters

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x