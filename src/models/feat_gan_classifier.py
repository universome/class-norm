import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from firelab.config import Config

from src.utils.constants import POS_INF
from src.utils.lll import prune_logits
from .feat_gan import FeatGenerator, FeatDiscriminatorWithoutClsHead
from .classifier import FeatClassifier


class FeatGANClassifier(nn.Module):
    def __init__(self, config: Config, attrs: np.ndarray):
        super(FeatGANClassifier, self).__init__()

        self.config = config
        self.register_buffer('attrs', torch.tensor(attrs).float())
        self.generator = FeatGenerator(config, attrs)
        self.discriminator = FeatDiscriminatorWithoutClsHead(config)
        self.classifier = FeatClassifier(config, attrs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Traditional forward pass for this model
        is used only for evaluation purposes
        """
        assert self.training is False

        return self.compute_pruned_predictions(x, np.ones(self.attrs.shape[0]).astype(bool))

    def compute_pruned_predictions(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        """
        :param x:
        :param output_mask: 1D boolean vector
        :return: pruned logits
        """
        if self.config.prediction_strategy == 'nearest_neighbor':
            return self.compute_pruned_predictions_via_centroids(x, output_mask)
        elif self.config.prediction_strategy == 'classifier':
            return self.compute_pruned_predictions_via_classifier(x, output_mask)
        else:
            raise NotImplementedError(f"Unknown prediction strategy: {self.config.prediction_strategy}")

    def compute_pruned_predictions_via_centroids(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        assert self.training is False, "Backprop through this thing is not supported (but can be, actually)"

        attrs = self.attrs[output_mask]
        centroids = self.compute_class_centroids(attrs)
        closest_centroid_idx = self.find_closest_centroid(centroids, x)

        # Now we know what we want to return,
        # but should convert the result into a more convenient form
        closest_class_idx = output_mask.nonzero()[0][closest_centroid_idx.data.cpu().tolist()]
        pseudo_logits = torch.zeros(len(x), len(self.attrs)).to(x.device)
        pseudo_logits[torch.arange(len(x)), closest_class_idx] = POS_INF

        return pseudo_logits

    def compute_pruned_predictions_via_classifier(self, x: Tensor, output_mask: np.ndarray) -> Tensor:
        logits = self.classifier(x)
        pruned = prune_logits(logits, output_mask)

        return pruned

    def compute_class_centroids(self, attrs) -> Tensor:
        num_classes = len(attrs)
        z = torch.randn(self.config.num_hallicinated_samples_per_class * num_classes, self.config.z_dim).to(attrs.device)
        x_fake = self.generator(z, attrs.repeat(self.config.num_hallicinated_samples_per_class, 1)) # [NUM_HALLICINATED_SAMPLES x NUM_CLASSES, X_DIM]
        x_fake = x_fake.view(self.config.num_hallicinated_samples_per_class, num_classes, x_fake.size(1))
        centroids = x_fake.mean(axis=0) # [NUM_CLASSES, X_DIM]

        return centroids

    def find_closest_centroid(self, centroids: Tensor, x: Tensor) -> Tensor:
        """
        Finds and returns idx of the closest centroid"
        :param centroids: matrix of size [NUM_CLASSES x X_DIM]
        :param x: matrix of size [BATCH_SIZE x X_DIM]

        :return: vector of size [BATCH_SIZE], where each idx indicates the closest centroid
        """
        distances = (centroids.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=2) # [NUM_CLASSES x BATCH_SIZE]
        closest = distances.argmin(dim=0)

        return closest
