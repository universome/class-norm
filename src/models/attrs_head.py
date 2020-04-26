import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize
from src.models.layers import MILayer, ConcatLayer, create_sequential_model, create_fuser


def create_attrs_head(config: Config, attrs: np.ndarray) -> nn.Module:
    return {
        'simple': SimpleAttrsHead,
        'multi_headed': MultiHeadedMPHead,
        'random_embeddings': RandomEmbeddingMPHead,
        'static_embeddings': StaticEmbeddingMPHead,
        'dropout_attrs': DropoutMPH
    }[config.type](config, attrs)


class SimpleAttrsHead(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.scale = config.scale.value
        self.register_buffer('attrs', torch.from_numpy(attrs))
        self.projector = nn.Linear(attrs.shape[1], config.hid_dim)

    def forward(self, feats: Tensor) -> Tensor:
        prototypes = self.projector(self.attrs)
        feats = normalize(feats, self.scale)
        prototypes = normalize(prototypes, self.scale)
        logits = torch.mm(feats, prototypes.t())

        return logits


class MultiProtoHead(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.config = config
        self.register_buffer('attrs', torch.from_numpy(attrs))

        self.init_modules()
        self.init_scaling()

    def generate_prototypes(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError('You forgot to implement `.generate_prototypes()` method')

    def compute_n_protos(self) -> int:
        if (not self.training and self.config.get('num_test_prototypes')):
            return self.config.num_test_prototypes
        else:
            return self.config.num_prototypes

    def init_scaling(self):
        if self.config.scale.type == 'learnable':
            self.scale = nn.Parameter(torch.tensor(self.config.scale.value))
        elif self.config.scale.type == 'constant':
            self.register_buffer('scale', torch.tensor(self.config.scale.value))
        else:
            raise NotImplementedError(f'Unknown scaling type: {self.config.scale.type}')

    def get_aggregation_type(self):
        train_time_aggr = self.config.get('aggregation_type', 'average_prototypes')

        if self.training:
            return train_time_aggr
        else:
            return self.config.get('test_aggregation_type', train_time_aggr)

    def forward(self, feats: Tensor, return_protos: bool=False, aggregation_type: str=None):
        aggregation_type = aggregation_type or self.get_aggregation_type()
        batch_size = feats.size(0)
        n_protos = self.compute_n_protos()
        n_classes = self.attrs.shape[0]
        hid_dim = self.config.hid_dim

        if aggregation_type == "golden_prototype":
            n_protos = 1
            protos = self.generate_prototypes(n_protos=n_protos, golden=True)
        else:
            protos = self.generate_prototypes()

        assert protos.shape == (n_protos, n_classes, hid_dim)
        assert feats.shape == (batch_size, hid_dim)

        if self.config.use_tanh_on_top:
            protos = F.tanh(protos)

        if self.training:
            feats = feats + torch.randn_like(feats) * self.config.feats_noise_std

        feats = normalize(feats, self.scale)
        protos = normalize(protos, self.scale)

        if aggregation_type == 'average_prototypes':
            if n_protos > 1 and self.config.get('golden_proto.enabled') and self.config.golden_proto.weight != "same":
                # Distribute remaining weights equally between other prototypes
                weight_others = (1 - self.config.golden_proto.weight) / (n_protos - 1)
                protos_weighted = protos
                protos_weighted[0] *= self.config.golden_proto.weight
                protos_weighted[1:] *= weight_others
                main_protos = protos_weighted.mean(dim=0) # [n_classes, hid_dim]
            else:
                main_protos = protos.mean(dim=0) # [n_classes, hid_dim]

            main_protos = normalize(main_protos, self.scale) # Normalizing once again
            logits = torch.matmul(feats, main_protos.t()) # [batch_size, n_classes]
        elif aggregation_type == 'golden_prototype':
            assert n_protos == 1
            logits = torch.matmul(feats, protos.squeeze(0).t()) # [batch_size, n_classes]
        elif aggregation_type == 'mean_distance':
            protos_for_dist = protos.view(n_protos * n_classes, hid_dim).unsqueeze(0) # [1, n_protos * n_classes, hid_dim]
            distances = (feats.unsqueeze(1) - protos_for_dist).norm(dim=2).pow(2) # [batch_size, n_protos * n_classes]
            distances = distances.view(batch_size, n_protos, n_classes) # [batch_size, n_protos, n_classes]
            logits = -distances.mean(dim=1) # [batch_size, n_classes]
        elif aggregation_type == 'shortest_distance':
            protos_for_dist = protos.view(n_protos * n_classes, hid_dim).unsqueeze(0) # [1, n_protos * n_classes, hid_dim]
            distances = (feats.unsqueeze(1) - protos_for_dist).norm(dim=2).pow(2) # [batch_size, n_protos * n_classes]
            distances = distances.view(batch_size, n_protos, n_classes) # [batch_size, n_protos, n_classes]
            logits = -(distances.min(dim=1)[0]) # [batch_size, n_classes]
        elif aggregation_type == 'softmax':
            logit_groups = protos @ feats.t() # [n_protos, n_classes, batch_size]
            logit_groups = logit_groups.permute(2, 1, 0) # [batch_size, n_protos, n_classes]
            prob_groups = logit_groups.view(batch_size, n_protos * n_classes).softmax(dim=1) # [batch_suze, n_protos * n_classes]
            probs = prob_groups.view(batch_size, n_protos, n_classes).sum(dim=1) # [batch_size x n_classes]
            logits = probs.log() # [batch_size x n_classes]
        else:
            raise NotImplementedError(f'Unknown aggregation_type: {aggregation_type}')

        if return_protos:
            return logits, protos
        else:
            return logits


class MultiHeadedMPHead(MultiProtoHead):
    def init_modules(self):
        # TODO: we should better create a single large matrix
        # and do this by a single matmul. This will speed the things up.
        self.embedders = nn.ModuleList([self.create_embedder() for _ in range(self.config.num_prototypes)])

    def create_embedder(self):
        # assert len(self.config.embedder_hidden_layers) > 0, \
        #     "Several linear models are equivalent to a single one"

        return create_sequential_model([
            self.attrs.shape[1],
            *self.config.embedder_hidden_layers,
            self.config.hid_dim
        ], final_activation=self.config.use_final_activation)

    def generate_prototypes(self) -> Tensor:
        prototypes = [e(self.attrs) for e in self.embedders] # [n_protos x n_classes x hid_dim]
        prototypes = torch.stack(prototypes) # [n_protos x n_classes x hid_dim]

        return prototypes


class EmbeddingMPHead(MultiProtoHead):
    def init_modules(self):
        self.context_embedder = self.create_context_embedder()
        self.fuser = self.create_fuser()

    def generate_prototypes(self) -> Tensor:
        proto_embeddings = self.generate_proto_embeddings() # [n_classes, n_protos, proto_emb_size]
        n_classes, n_protos, proto_emb_size = proto_embeddings.shape
        proto_embeddings = proto_embeddings.view(n_classes * n_protos, proto_emb_size)
        prototypes = self.fuser(self.attrs.repeat(n_protos, 1), proto_embeddings) # [n_classes * n_protos, hid_dim]

        assert prototypes.shape == (n_classes * n_protos, self.config.hid_dim), f"Wrong shape: {prototypes.shape}"

        prototypes = prototypes.view(n_protos, n_classes, self.config.hid_dim)

        return prototypes


class StaticEmbeddingMPHead(EmbeddingMPHead):
    def create_context_embedder(self) -> nn.Module:
        return nn.Sequential(
            nn.Embedding(self.config.num_prototypes, 512),
            nn.ReLU(),
            nn.Linear(512, self.config.context.proto_emb_size)
        )
        # return nn.Sequential(
        #     nn.Embedding(self.config.num_prototypes, self.config.context.proto_emb_size),
        # )

    def create_fuser(self):
        return create_fuser(
            self.config.fusing_type, self.attrs.shape[1],
            self.config.context.proto_emb_size, self.config.hid_dim)

    def generate_proto_embeddings(self) -> Tensor:
        n_classes = self.attrs.shape[0]
        n_protos = self.config.num_prototypes

        context = torch.arange(n_protos, device=self.attrs.device)
        context = context.view(1, n_protos).repeat(n_classes, 1) # [n_classes, n_protos]
        embeddings = self.context_embedder(context) # [n_classes, n_protos, proto_emb_size]

        return embeddings


class RandomEmbeddingMPHead(MultiProtoHead):
    def init_modules(self):
        self.noise_transform = create_sequential_model(self.config.noise.transform_layers)
        self.attrs_transform = create_sequential_model(self.config.attrs_transform_layers)
        self.fuser = create_fuser(
            self.config.fusing_type,
            self.config.attrs_transform_layers[-1],
            self.config.noise.transform_layers[-1],
            self.config.after_fuse_transform_layers[0],
            activation=self.config.get('after_fuse_activation', ('relu' if len(self.config.after_fuse_transform_layers) > 1 else 'none'))
        )
        self.after_fuse_transform = create_sequential_model(self.config.after_fuse_transform_layers)

        if self.config.get('dae.enabled'):
            self.encoder = create_sequential_model(self.config.dae.encoder_layers)

        if self.config.get('after_fuse_identity_init'):
            n_in, n_out = self.after_fuse_transform[0].weight.data.shape
            assert n_in == n_out, f"Cannot use identity init for non-square transforms: {n_in, n_out}"
            self.after_fuse_transform[0].weight.data.mul_(0.001)
            self.after_fuse_transform[0].weight.data.add_(torch.eye(n_in))

    def get_transformed_noise(self, n_protos: int, golden: bool=False) -> Tensor:
        n_classes = self.attrs.shape[0]
        z_size = self.config.noise.transform_layers[0]

        if self.config.noise.same_for_each_class:
            noise = torch.randn(1, n_protos, z_size, device=self.attrs.device)
            noise = noise.repeat(n_classes, 1, 1)
        else:
            noise = torch.randn(n_classes, n_protos, z_size, device=self.attrs.device)

        if golden:
            noise[:, 0, :] = 0. # First noise emb is zero.

        noise = noise * self.config.noise.std
        transformed_noise = self.noise_transform(noise) # [n_classes, n_protos, proto_emb_size]

        return transformed_noise

    def compute_dae_reconstructions(self, feats: Tensor, y: Tensor) -> Tensor:
        if self.config.dae.input_noise_type == 'bernoulli':
            feats_noised = F.dropout(feats, self.config.dae.input_dropout_p)
        elif self.config.dae.input_noise_type == 'gaussian':
            feats_noised = feats + torch.randn_like(feats) * self.config.dae.input_std
        else:
            raise NotImplementedError('Unknown input noise type: {self.config.dae.input_noise_type}')

        z = self.encoder(feats_noised) # [batch_size, z_size]
        z += torch.randn_like(z) * self.config.dae.get('codes_std', 0.)
        z_transformed = self.noise_transform(z) # [batch_size, noise.transform_layers[-1]]
        attrs_transformed = self.attrs_transform(self.attrs) # [n_classes, attrs.transform_layers[-1]]
        attrs_transformed = attrs_transformed[y] # [batch_size, attrs.transform_layers[-1]]
        contextualized_attrs = self.fuser(z_transformed, attrs_transformed) # [batch_size, after_fuse_transform_layers[0]]
        feats_rec = self.after_fuse_transform(contextualized_attrs) # [batch_size, hid_dim]

        assert feats.shape == feats_rec.shape, f"Wrong shape: {feats_rec.shape}"

        return feats_rec

    def generate_prototypes(self, n_protos: int=None, golden: bool=False) -> Tensor:
        n_protos = self.compute_n_protos() if n_protos is None else n_protos

        if golden:
            z_transormed = self.get_transformed_noise(n_protos, self.config.golden_proto.enabled) # [n_classes, n_protos, transformed_noise_size]
        else:
            z_transormed = self.get_transformed_noise(n_protos, True)

        attrs_transformed = self.attrs_transform(self.attrs) # [n_classes, transformed_attrs_size]

        assert z_transormed.shape[0] == attrs_transformed.shape[0]

        n_classes, n_protos, transformed_noise_size = z_transormed.shape

        z_transormed = z_transormed.view(n_classes * n_protos, transformed_noise_size)
        attrs_transformed = attrs_transformed.unsqueeze(1).repeat(1, n_protos, 1).view(n_classes * n_protos, -1)
        contextualized_attrs = self.fuser(attrs_transformed, z_transormed) # [n_classes * n_protos, after_fuse_transform_layers[0]]
        prototypes = self.after_fuse_transform(contextualized_attrs) # [n_classes * n_protos, hid_dim]

        assert prototypes.shape == (n_classes * n_protos, self.config.hid_dim), f"Wrong shape: {prototypes.shape}"

        prototypes = prototypes.view(n_classes, n_protos, self.config.hid_dim)
        prototypes = prototypes.permute(1, 0, 2)

        return prototypes


class DropoutMPH(MultiHeadedMPHead):
    def init_modules(self):
        self.attrs_transform = create_sequential_model(self.config.attrs_transform_layers)

    def compute_noise_attrs(self):
        if self.config.dropout.type == 'bernoulli':
            return self.compute_bernoulli_noised_attrs()
        elif self.config.dropout.type == 'gaussian':
            return self.compute_gaussian_noised_attrs()
        else:
            raise NotImplementedError(f'Unknown dropout type: {self.config.dropout.type}')

    def compute_gaussian_noised_attrs(self):
        n_protos = self.compute_n_protos()
        n_classes, attr_dim = self.attrs.shape
        noise = torch.randn(n_protos, n_classes, attr_dim).to(self.attrs.device)
        noise *= self.config.dropout.std

        return self.attrs.unsqueeze(0) + noise

    def compute_bernoulli_noised_attrs(self):
        n_protos = self.compute_n_protos()
        n_classes, attr_dim = self.attrs.shape

        if not self.training and self.compute_n_protos() == 1:
            p = 0.0
        else:
            p = self.config.dropout.p

        if self.config.dropout.level == 'attribute_wise':
            scale = 1 / (1 - p)

            # Creating a mask per prototype
            masks = torch.rand(n_protos, attr_dim) >= p
            masks = masks.view(1, n_protos, attr_dim).float()
            masks = masks.to(self.attrs.device)

            # Replicating attrs per each prototype
            attrs = self.attrs.view(n_classes, 1, attr_dim)
            attrs = attrs.repeat(1, n_protos, 1)

            # Applying the masks
            attrs = attrs * masks

            # Scaling attrs during training
            attrs = attrs * scale
            attrs = attrs.permute(1, 0, 2) # [n_protos, n_classes, attr_dim]
        elif self.config.dropout.level == 'element_wise':
            attrs = self.attrs.view(n_classes, 1, attr_dim)
            attrs = attrs.repeat(1, n_protos, 1)
            attrs = attrs.permute(1, 0, 2)
            attrs = F.dropout(attrs, p=p)
        else:
            raise NotImplementedError(f'Unknown dropout level: {self.config.dropout.level}')

        return attrs

    def generate_prototypes(self):
        attrs = self.compute_noise_attrs() # [n_protos, n_clasess, attr_dim]
        prototypes = self.attrs_transform(attrs)

        assert prototypes.shape == (self.compute_n_protos(), self.attrs.shape[0], self.config.hid_dim), \
            f"Wrong shape: {prototypes.shape}"

        return prototypes
