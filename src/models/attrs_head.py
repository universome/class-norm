import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from firelab.config import Config

from src.utils.training_utils import normalize
from src.models.layers import MILayer, ConcatLayer, create_sequential_model, create_fuser


class AttrsHead(nn.Module):
    """
    An entry point for all types of attribute-based heads
    """
    def __init__(self, config: Config, attrs: np.ndarray):
        super().__init__()

        self.model = {
            'simple': SimpleAttrsHead,
            'multi_headed': MultiHeadedMPHead,
            'random_embeddings': RandomEmbeddingMPHead,
            'static_embeddings': StaticEmbeddingMPHead
        }[config.type](config, attrs)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x, **kwargs)


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

        # TODO: - compute std and change it in such a way that it equals to He/Xavier's std.
        if self.config.scale.type == 'learnable':
            self.scale = nn.Parameter(torch.tensor(self.config.scale.value))
        elif self.config.scale.type == 'constant':
            self.register_buffer('scale', torch.tensor(self.config.scale.value))
        elif self.config.scale.type == 'predict_from_attrs':
            self.predict_scale = create_sequential_model(self.config.scale.layers_sizes)
            self.predict_scale[-2].bias.data = torch.ones_like(self.predict_scale[-2].bias.data) * (self.config.scale.value ** 2)
        elif self.config.scale.type == 'predict_from_logits':
            self.predict_scale = create_sequential_model(self.config.scale.layers_sizes)
            self.predict_scale[-2].bias.data = torch.ones_like(self.predict_scale[-2].bias.data) * (self.config.scale.value ** 2)
        else:
            raise NotImplementedError(f'Unknown scaling type: {self.config.scale.type}')

    def generate_prototypes(self) -> Tensor:
        raise NotImplementedError('You forgot to implement `.generate_prototypes()` method')

    def compute_n_protos(self) -> int:
        if (not self.training and self.config.get('num_test_prototypes')):
            return self.config.num_test_prototypes
        else:
            return self.config.num_prototypes

    def aggregate_logits(self, logit_groups: Tensor) -> Tensor:
        batch_size = logit_groups.size(0)
        n_protos = self.compute_n_protos()
        n_classes = self.attrs.shape[0]

        assert logit_groups.shape == (batch_size, n_protos, n_classes), f"Wrong shape: {logit_groups.shape}"

        if self.config.combine_strategy == 'mean':
            if n_protos > 1 and self.config.golden_proto.enabled and self.config.golden_proto.weight != "same":
                # Distribute remaining weights equally between other prototypes
                weight_others = (1 - self.config.golden_proto.weight) / (n_protos - 1)
                logit_groups[:, 0, :] *= self.config.golden_proto.weight
                logit_groups[:, 1:, :] *= weight_others
                logits = logit_groups.sum(dim=1)
            else:
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

    def forward(self, feats: Tensor):
        batch_size = feats.size(0)
        n_protos = self.compute_n_protos()
        n_classes = self.attrs.shape[0]
        hid_dim = self.config.hid_dim

        protos = self.generate_prototypes()

        assert protos.shape == (n_protos, n_classes, hid_dim)
        assert feats.shape == (batch_size, hid_dim)

        if self.config.get('senet.enabled'):
            attns = self.generete_senet_attns(feats)
            assert attns.shape == (n_protos, batch_size, hid_dim)

            se_protos = protos.view(n_protos, 1, n_classes, hid_dim) * attns.view(n_protos, batch_size, 1, hid_dim)
            se_protos = se_protos.permute(1, 0, 2, 3)

            assert se_protos.shape == (batch_size, n_protos, n_classes, hid_dim)

            if self.config.normalize: se_protos = normalize(se_protos, self.scale)
            if self.config.normalize: feats = normalize(feats, self.scale)

            feats = feats.view(batch_size, 1, hid_dim).repeat(1, n_protos, 1)
            logit_groups = torch.matmul(
                se_protos.view(batch_size * n_protos, n_classes, hid_dim),
                feats.view(batch_size * n_protos, hid_dim, 1)) # [batch_size * n_protos, n_classes]
            logit_groups = logit_groups.view(batch_size, n_protos, n_classes)
        else:
            if self.config.scale.type == 'predict_from_attrs':
                scales = self.predict_scale(self.attrs).view(1, n_classes, 1)
                protos = normalize(protos, scales)
                feats = normalize(feats)
            elif self.config.scale.type == 'predict_from_logits':
                protos = normalize(protos)
                feats = normalize(feats)
            else:
                protos = normalize(protos, self.scale)
                feats = normalize(feats, self.scale)

            if self.config.output_dist == 'von_mises':
                logit_groups = torch.matmul(protos, feats.t()) # [n_protos, n_classes, batch_size]
                logit_groups = logit_groups.permute(2, 0, 1) # [batch_size, n_protos, n_classes]
            else:
                assert False, "Gaussian output distribution is deprecated"
                x = feats.view(batch_size, 1, 1, hid_dim)
                mu = protos.view(1, n_protos, n_classes, hid_dim)
                logit_groups = -(x - mu).pow(2).sum(dim=3) * self.scale

                assert logit_groups.shape == (batch_size, n_protos, n_classes)

        logits = self.aggregate_logits(logit_groups)

        if self.config.scale.type == 'predict_from_logits':
            scales = self.predict_scale(logits)
            logits = logits * scales

        return logits


class MultiHeadedMPHead(MultiProtoHead):
    def init_modules(self):
        # TODO: we should better create a single large matrix
        # and do this by a single matmul. This will speed the things up.
        self.embedders = nn.ModuleList([self.create_embedder() for _ in range(self.config.num_prototypes)])

        if self.config.get('senet.enabled'):
            self.senets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.config.hid_dim, self.config.senet.reduction_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.config.senet.reduction_dim, self.config.hid_dim),
                    nn.Sigmoid()
                ) for _ in range(self.config.num_prototypes)])

    def create_embedder(self):
        # assert len(self.config.embedder_hidden_layers) > 0, \
        #     "Several linear models are equivalent to a single one"

        return create_sequential_model([
            self.attrs.shape[1],
            *self.config.embedder_hidden_layers,
            self.config.hid_dim
        ], final_activation=self.config.use_final_activation)

    def generete_senet_attns(self, x: Tensor) -> Tensor:
        return torch.stack([s(x) for s in self.senets])

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
            self.config.after_fuse_transform_layers[0])
        self.after_fuse_transform = create_sequential_model(
            self.config.after_fuse_transform_layers, final_activation=False)

    def get_transformed_noise(self) -> Tensor:
        n_classes = self.attrs.shape[0]
        n_protos = self.compute_n_protos()
        z_size = self.config.noise.transform_layers[0]

        if self.config.noise.same_for_each_class:
            noise = torch.randn(1, n_protos, z_size, device=self.attrs.device)
            noise = noise.repeat(n_classes, 1, 1)
        else:
            noise = torch.randn(n_classes, n_protos, z_size, device=self.attrs.device)

        if self.config.golden_proto.enabled:
            noise[:, 0, :] = 0. # First noise emb is zero.

        noise = noise * self.config.noise.std
        transformed_noise = self.noise_transform(noise) # [n_classes, n_protos, proto_emb_size]

        assert torch.all(noise == transformed_noise)

        return transformed_noise

    def generate_prototypes(self) -> Tensor:
        transformed_noise = self.get_transformed_noise() # [n_classes, n_protos, transformed_noise_size]
        transformed_attrs = self.attrs_transform(self.attrs) # [n_classes, transformed_attrs_size]

        assert transformed_noise.shape[0] == transformed_attrs.shape[0]

        n_classes, n_protos, transformed_noise_size = transformed_noise.shape

        transformed_noise = transformed_noise.view(n_classes * n_protos, transformed_noise_size)
        transformed_attrs = transformed_attrs.repeat(n_protos, 1)
        contextualized_attrs = self.fuser(transformed_attrs, transformed_noise) # [n_classes * n_protos, after_fuse_transform_layers[0]]
        prototypes = self.after_fuse_transform(contextualized_attrs) # [n_classes * n_protos, hid_dim]

        assert prototypes.shape == (n_classes * n_protos, self.config.hid_dim), f"Wrong shape: {prototypes.shape}"

        prototypes = prototypes.view(n_protos, n_classes, self.config.hid_dim)

        return prototypes
