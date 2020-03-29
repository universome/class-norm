import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize
from src.models.layers import MILayer, ConcatLayer


class AttrsHead(nn.Module):
    """
    An entry point for all types of attribute-based heads
    """
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()
        self.model = {
            'simple': SimpleAttrsHead,
            'joint_head': JointHead,
            'deterministic_linear': DeterministicLinearMultiProtoHead,
            'embedding_based': EmbeddingBasedMPHead
        }[config.type](config, attrs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x, **kwargs)


class SimpleAttrsHead(nn.Module):
    def __init__(self, attrs: np.ndarray, config: Config):
        super().__init__()

        self.register_buffer('attrs', torch.from_numpy(attrs))
        self.projector = nn.Linear(attrs.shape[1], config.hid_dim)
        self.biases = nn.Parameter(torch.zeros(attrs.shape[0]))

    def forward(self, feats: Tensor) -> Tensor:
        attr_embs = self.projector(self.attrs)
        logits = torch.mm(feats, attr_embs.t()) + self.biases

        return logits


class MultiProtoHead(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.config = config
        self.register_buffer('attrs', torch.from_numpy(attrs))

        self.init_modules()

    def generate_prototypes(self) -> Tensor:
        raise NotImplementedError('You forgot to implement `.generate_prototypes()` method')

    def aggregate_logits(self, logit_groups: Tensor) -> Tensor:
        batch_size = logit_groups.size(0)
        n_protos = self.config.num_prototypes
        n_classes = self.attrs.shape[0]

        assert logit_groups.shape == (batch_size, n_protos, n_classes), f"Wrong shape: {logit_groups.shape}"

        if self.config.combine_strategy == 'mean':
            logits = logit_groups.mean(dim=1) # [batch_size x n_classes]
        elif self.config.combine_strategy == 'max':
            logits = logit_groups.max(dim=1)[0]
        elif self.config.combine_strategy == 'min':
            logits = logit_groups.min(dim=1)[0]
        elif self.config.combine_strategy == 'sum':
            logits = logit_groups.sum(dim=1)
        elif self.config.combine_strategy == 'softmax_mean':
            prob_groups = logit_groups.view(batch_size, n_protos * n_classes).softmax(dim=1) # [batch_suze, n_protos * n_classes]
            probs = prob_groups.view(batch_size, n_protos, n_classes).sum(dim=1) # [batch_size x n_classes]
            logits = probs.log() # [batch_size x n_classes]
        else:
            raise NotImplementedError(f'Unknown combine strategy: {self.config.combine_strategy}')

        return logits

    def forward(self, feats: Tensor, return_protos: bool=False):
        batch_size = feats.size(0)
        n_protos = self.config.num_prototypes
        n_classes = self.attrs.shape[0]
        hid_dim = self.config.hid_dim

        protos = self.generate_prototypes()

        assert protos.shape == (n_protos, n_classes, hid_dim)

        if self.config.get('senet.enabled'):
            attns = self.generete_senet_attns(feats)
            assert attns.shape == (n_protos, batch_size, hid_dim)

            se_protos = protos.view(n_protos, 1, n_classes, hid_dim) * attns.view(n_protos, batch_size, 1, hid_dim)
            se_protos = se_protos.permute(1, 0, 2, 3)

            assert se_protos.shape == (batch_size, n_protos, n_classes, hid_dim)

            if self.config.normalize: se_protos = normalize(se_protos, self.config.scale_value)
            if self.config.normalize: feats = normalize(feats, self.config.scale_value)

            feats = feats.view(batch_size, 1, hid_dim).repeat(1, n_protos, 1)
            logit_groups = torch.matmul(
                se_protos.view(batch_size * n_protos, n_classes, hid_dim),
                feats.view(batch_size * n_protos, hid_dim, 1)) # [batch_size * n_protos, n_classes]
            logit_groups = logit_groups.view(batch_size, n_protos, n_classes)
        else:
            if self.config.normalize: protos = normalize(protos, self.config.scale_value)
            if self.config.normalize: feats = normalize(feats, self.config.scale_value)

            logit_groups = torch.matmul(protos, feats.t()) # [n_protos, n_classes, batch_size]
            logit_groups = logit_groups.permute(2, 0, 1) # [batch_size, n_protos, n_classes]

        logits = self.aggregate_logits(logit_groups)

        if self.config.get('logits_scaling.enabled', False):
            logits *= (n_protos * self.config.logits_scaling.scale_value)

        if return_protos:
            return logits, protos
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
    def init_modules(self):
        # TODO: we should better create a single large matrix
        # and do this by a single matmul. This will speed the things up.
        self.embedders = nn.ModuleList([nn.Linear(self.attrs.shape[1], self.config.hid_dim) for _ in range(self.config.num_prototypes)])

        if self.config.get('senet.enabled'):
            self.senets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.config.hid_dim, self.config.senet.reduction_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.config.senet.reduction_dim, self.config.hid_dim),
                    nn.Sigmoid()
                ) for _ in range(self.config.num_prototypes)])

    def generete_senet_attns(self, x: Tensor) -> Tensor:
        return torch.stack([s(x) for s in self.senets])

    def generate_prototypes(self) -> Tensor:
        prototypes = [e(self.attrs) for e in self.embedders] # [n_protos x n_classes x hid_dim]
        prototypes = torch.stack(prototypes) # [n_protos x n_classes x hid_dim]

        return prototypes


class EmbeddingBasedMPHead(MultiProtoHead):
    def init_modules(self):
        self.context_embedder = self.create_context_embedder()
        self.fuser = self.create_fuser()

    def create_context_embedder(self) -> nn.Module:
        if self.config.context.type == 'gaussian_noise':
            assert self.config.context.proto_emb_size == self.config.context.z_size
            return nn.Identity()
        elif self.config.context.type == 'embeddings':
            return nn.Embedding(self.config.num_prototypes, self.config.context.proto_emb_size)
        elif self.config.context.type == 'transformed_gaussian_noise':
            return nn.Sequential(
                nn.Linear(self.config.context.z_size, self.config.context.proto_emb_size),
                nn.ReLU() if self.config.context.use_activation else nn.Identity()
            )
        else:
            raise NotImplementedError(f'Unknown context type: {self.config.context.type}')

    def create_fuser(self) -> nn.Module:
        if self.config.fusing_type == 'pure_mult_int':
            return MILayer(self.attrs.shape[1], self.config.context.proto_emb_size, self.config.hid_dim, False)
        if self.config.fusing_type == 'full_mult_int':
            return MILayer(self.attrs.shape[1], self.config.context.proto_emb_size, self.config.hid_dim, True)
        elif self.config.fusing_type == 'concat':
            return ConcatLayer(self.attrs.shape[1], self.config.context.proto_emb_size, self.config.hid_dim)
        else:
            raise NotImplementedError('Unknown fusing type: {self.config.fusing_type}')

    def generate_proto_embeddings(self) -> Tensor:
        n_classes = self.attrs.shape[0]
        n_protos = self.config.num_prototypes

        if self.config.context.type in ['gaussian_noise', 'transformed_gaussian_noise']:
            if self.config.context.same_for_each_class:
                noise = torch.randn(1, n_protos, self.config.context.z_size).repeat(n_classes, 1, 1)
            else:
                noise = torch.randn(n_classes, n_protos, self.config.context.z_size)

            context = noise * self.config.context.std
        elif self.config.context.type == 'embeddings':
            context = torch.arange(n_protos).view(1, n_protos).repeat(n_classes, 1) # [n_classes, n_protos]
        else:
            raise NotImplementedError(f'Unknown context type: {self.config.context.type}')

        context = context.to(self.attrs.device)
        embeddings = self.context_embedder(context) # [n_classes, n_protos, proto_emb_size]

        return embeddings

    def generate_prototypes(self) -> Tensor:
        proto_embeddings = self.generate_proto_embeddings() # [n_classes, n_protos, proto_emb_size]
        n_classes, n_protos, proto_emb_size = proto_embeddings.shape
        proto_embeddings = proto_embeddings.view(n_classes * n_protos, proto_emb_size)
        prototypes = self.fuser(self.attrs.repeat(n_protos, 1), proto_embeddings) # [n_classes * n_protos, hid_dim]

        assert prototypes.shape == (n_classes * n_protos, self.config.hid_dim), f"Wrong shape: {prototypes.shape}"

        prototypes = prototypes.view(n_classes, n_protos, self.config.hid_dim)
        prototypes = prototypes.permute(1, 0, 2)

        return prototypes
