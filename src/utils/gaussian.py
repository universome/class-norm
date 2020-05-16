from math import pi

import torch
from torch import Tensor
from firelab.utils.training_utils import compute_pairwise_l2_dists

# def compute_low_rank_gaussian_log_density(x, mean, cov_left, cov_right, cov_diag) -> Tensor:
#     """
#     Computes density of gaussian distribution with low-rank covariance matrix approximation
#     I.e. Sigma = AB + D, where A and B are low-rank matrices and D is diagonal

#     x: points to compute density for [batch_size, feat_dim]
#     mean: mean vectors [n_dists, feat_dim]
#     cov_left: [n_dists, feat_dim, rank]
#     cov_right: [n_dists, rank, feat_dim]
#     cov_diag: [n_dists, feat_dim]

#     @result: log-densities of size [batch_size, n_dists]
#     """

#     cov_invs = compute_low_rank_inverse(cov_left, cov_right, cov_diag)


# def compute_low_rank_inverse(m_left, m_right, m_diag):
#     """
#     Computes the inverse of the matrix M of the form M = AB + D using Woodbury's identity
#     https://en.wikipedia.org/wiki/Woodbury_matrix_identity

#     m_left: [batch_size, feat_dim, rank]
#     m_right: [batch_size, rank, feat_dim]
#     m_diag: [batch_size, feat_dim]
#     """
#     batch_size, feat_dim, rank == m_left.shape
#     assert m_right.shape == (batch_size, rank, feat_dim), f"Wrong shape: {m_right.shape}"
#     assert m_diag.shape == (batch_size, feat_dim), f"Wrong shape: {m_diag.shape}"

#     m_diag_inv_with_m_left_product = m_left / m_diag.view(batch_size, feat_dim, 1)
#     m_right_with_m_diag_inv_product = m_right / m_diag.view(batch_size, 1, feat_dim)
#     m_diag_inv = torch.diag_embed(1 / m_diag)

def compute_ll_decomposed_gaussian_log_density(x: Tensor, mean: Tensor, cov_l_inv: Tensor) -> Tensor:
    """
    Computes density of gaussian distribution with cholesky decomposed covariance matrix
    I.e. Sigma = LL^T, L is a lower-triangluar matrix

    x: [batch_size, feat_dim] — points to compute log-density for
    mean: [n_dists, feat_dim] — mean vectors
    cov_l_inv: [n_dists, feat_dim, feat_dim] — inverse of L matrix

    @result: log-densities of size [batch_size, n_dists]
    """
    batch_size, feat_dim, n_dists = x.shape[0], x.shape[1], mean.shape[0]

    assert mean.shape == (n_dists, feat_dim), f"Wrong shape: {mean.shape}"
    assert cov_l_inv.shape == (n_dists, feat_dim, feat_dim), f"Wrong shape: {cov_l_inv.shape}"

    const_term = -0.5 * feat_dim * torch.tensor(2 * pi, device=x.device).log()
    logdet_term = 0.5 * torch.diagonal(cov_l_inv, dim1=1, dim2=2).pow(2).log().sum(dim=1) # [n_dists]
    # xt_lt_inv_product = (x.view(batch_size, feat_dim, 1) * cov_l_inv.permute(2, 1, 0).view()).sum(dim=1) # [batch_size, feat_dim, n_dists]
    #exp_term_lhs = x.view(batch_size, feat_dim, 1) - mean.permute() # [batch_size]
    x_minus_mu = (x.view(batch_size, feat_dim, 1) - mean.permute(1, 0).view(1, feat_dim, n_dists)) # [batch_size, feat_dim, n_dists]
    x_minus_mu_t_with_l_t_product = torch.einsum('bifn,fjn->bijn', x_minus_mu.unsqueeze(1), cov_l_inv.permute(2,1,0)) # [batch_size, feat_dim, n_dists]
    x_minus_mu_t_with_l_t_product = x_minus_mu_t_with_l_t_product.squeeze(1)

    assert x_minus_mu_t_with_l_t_product.shape == (batch_size, feat_dim, n_dists)

    exp_term = -0.5 * x_minus_mu_t_with_l_t_product.pow(2).sum(dim=1) # [batch_size, n_dists]
    # result = const_term + logdet_term.unsqueeze(0) + exp_term
    result = exp_term

    return result


def compute_diag_gaussian_log_density(x: Tensor, mean: Tensor, cov_diag: Tensor) -> Tensor:
    """
    Computes density of gaussian distribution with diagonal covariance matrix

    x: [batch_size, feat_dim] — points to compute log-density for
    mean: [n_dists, feat_dim] — mean vectors
    cov_diag: [n_dists, feat_dim] — diagonal covariance matrix

    @result: log-densities of size [batch_size, n_dists]
    """
    batch_size, feat_dim, n_dists = x.shape[0], x.shape[1], mean.shape[0]

    assert mean.shape == (n_dists, feat_dim), f"Wrong shape: {mean.shape}"
    assert cov_diag.shape == (n_dists, feat_dim), f"Wrong shape: {cov_diag.shape}"

    const_term = -0.5 * feat_dim * torch.tensor(2 * pi, device=x.device).log() # []
    logdet_term = -0.5 * cov_diag.log().sum(dim=1) # [n_dists]

    mean_reshaped = mean.permute(1, 0).view(1, feat_dim, n_dists) # [1, feat_dim, n_dists]
    cov_diag_reshaped = cov_diag.permute(1, 0).view(1, feat_dim, n_dists) # [1, feat_dim, n_dists]
    exp_term = ((x.view(batch_size, feat_dim, 1) - mean_reshaped).pow(2) / cov_diag_reshaped).sum(dim=1) # [batch_size, n_dists]

    result = const_term + logdet_term.unsqueeze(0) + exp_term

    return result
