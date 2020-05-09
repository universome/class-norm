import sys; sys.path.append('.')

import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange

from src.utils.gaussian import compute_ll_decomposed_gaussian_log_density


def test_gaussain():
    for _ in trange(10, desc='Testing Gaussian'):
        batch_size, feat_dim, num_dists = 8, 32, 20
        x = torch.randn(batch_size, feat_dim)
        mean = torch.randn(num_dists, feat_dim)
        cov_l_inv = torch.rand(num_dists, feat_dim, feat_dim).tril() + torch.eye(feat_dim, feat_dim).unsqueeze(0) / 2
        cov_l = [torch.inverse(l_inv) for l_inv in cov_l_inv]

        ds = [MultivariateNormal(mu, scale_tril=l) for mu, l in zip(mean, cov_l)]
        result_torch = torch.stack([d.log_prob(x) for d in ds]).t()
        result = compute_ll_decomposed_gaussian_log_density(x, mean, cov_l_inv)

        assert torch.allclose(result_torch, result)
