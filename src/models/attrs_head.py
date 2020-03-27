import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize


class AttrsHead(nn.Module):
    """
    An entry point for all types of attribute-based heads
    """
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()
        self.model = {
            'simple': SimpleAttrsHead,
            'joint_head': JointHead,
            'deterministic_linear': DeterministicLinearMultiProtoHead
        }[config.type](config, attrs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x, **kwargs)


class SimpleAttrsHead(nn.Module):
    def __init__(self, attrs: np.ndarray, config: Config):
        self.register_buffer('attrs', torch.from_numpy(attrs))
        self.projector = nn.Linear(attrs.shape[1], config.hid_dim)
        self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, feats: Tensor) -> Tensor:
        attr_embs = self.projector(self.attrs)
        logits = torch.mm(feats, attr_embs.t()) + self.biases

        return logits


class MultiProtoHead(nn.Module):
    def generate_prototypes(self) -> Tensor:
        raise NotImplementedError('You forgot to implement `.generate_prototypes()` method')

    def forward(self, feats: Tensor, return_protos: bool=False):
        batch_size = feats.size(0)
        n_protos = self.config.num_prototypes
        n_classes = self.attrs.shape[0]
        hid_dim = self.config.hid_dim

        protos = self.generate_prototypes()

        assert protos.shape == (n_protos, n_classes, hid_dim)

        protos = protos.view(n_protos * n_classes, hid_dim)
        if self.config.normalize: protos = normalize(protos, self.config.scale_value)
        if self.config.normalize: feats = normalize(feats, self.config.scale_value)

        logit_groups = torch.mm(feats, protos.t()) # [batch_size x (n_protos * n_classes)]
        logit_groups = logit_groups.view(batch_size, n_protos, n_classes)
        logits = logit_groups.mean(dim=1) # [batch_size x n_classes]

        if return_protos:
            return logits, protos.view(n_protos, n_classes, hid_dim)
        else:
            return logits


class JointHead(nn.Module):
    """
    This head implements the following strategies to produce an label embedding matrix:
        - just a normal attrs head with/without biases
        - attrs head with normalized attribute embeddings
        - attribute is [original attribute] + [one-hot class representation] concatenated
        - learned attributes

        TODO: several layers for attributes embeddings
    """
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        print('Running with a "complex" attrs head.')

        self.config = config
        self.attrs_to_proj_matrix = nn.Linear(attrs.shape[1], self.config.hid_dim)

        if self.config.learnable_attrs:
            self.attrs = nn.Parameter(torch.from_numpy(attrs))
        else:
            self.register_buffer('attrs', torch.from_numpy(attrs))

        if self.config.use_cls_proj:
            self.cls_proj_matrix = nn.Linear(attrs.shape[0], self.config.hid_dim, bias=False)

            if self.config.combine_strategy == 'learnable':
                self.attrs_weight = nn.Parameter(torch.tensor(0.))
            elif self.config.combine_strategy == 'concat':
                self.register_buffer('attrs_weight', torch.tensor(0.))

            if self.config.use_biases:
                self.cls_proj_biases = nn.Parameter(torch.zeros(attrs.shape[0]))

        if self.config.use_biases:
            self.attrs_proj_biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, x_feats: Tensor) -> Tensor:
        if self.config.normalize_feats:
            x_feats = normalize(x_feats, self.config.scale_value)

        # Computing logits from attrs
        attrs_proj_matrix = self.attrs_to_proj_matrix(self.attrs)
        if self.config.normalize: attrs_proj_matrix = normalize(attrs_proj_matrix, self.config.scale_value)
        attrs_logits = torch.mm(x_feats, attrs_proj_matrix.t())
        if self.config.use_biases: attrs_logits += self.attrs_proj_biases

        if self.config.use_cls_proj:
            cls_proj_matrix = self.cls_proj_matrix.weight.t()
            if self.config.normalize: cls_proj_matrix = normalize(cls_proj_matrix, self.config.scale_value)
            cls_logits = torch.mm(x_feats, cls_proj_matrix.t())
            if self.config.use_biases: cls_logits += self.cls_proj_biases

            alpha = self.attrs_weight.sigmoid()
            logits = alpha * attrs_logits + (1 - alpha) * cls_logits

            if self.config.combine_strategy == 'concat':
                logits = 2 * logits # Since our weights were equal to 0.5
        else:
            logits = attrs_logits

        return logits


class DeterministicLinearMultiProtoHead(MultiProtoHead):
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.config = config
        self.register_buffer('attrs', torch.from_numpy(attrs))

        # TODO: we should better create a single large matrix
        # and do this by a single matmul. This will speed the things up.
        self.projectors = nn.ModuleList([nn.Linear(attrs.shape[1], config.hid_dim) for _ in range(config.num_prototypes)])

    def generate_prototypes(self) -> Tensor:
        prototypes = [p(self.attrs) for p in self.projectors] # [num_prototypes x num_attrs x hid_dim]
        prototypes = torch.stack(prototypes) # [num_prototypes x num_attrs x hid_dim]

        return prototypes
