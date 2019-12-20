import torch
from torch import Tensor, autograd


def compute_gradient_penalty(discriminator, x_real, x_fake):
    """
    Computes gradient penalty according to WGAN-GP paper
    """
    assert x_real.size() == x_fake.size()

    shape = [x_real.size(0)] + [1] * (x_real.dim() - 1)
    alpha = torch.rand(shape).to(x_real.device)
    interpolations = x_real + alpha * (x_fake - x_real)

    interpolations = interpolations.to(x_real.device)
    interpolations.requires_grad_(True)
    outputs = discriminator(interpolations)
    grads = autograd.grad(
        outputs,
        interpolations,
        grad_outputs=torch.ones(outputs.size()).to(interpolations.device),
        create_graph=True
    )[0].view(interpolations.size(0), -1)

    return ((grads.norm(p=2, dim=1) - 1) ** 2).mean()


def compute_kld_with_standard_gaussian(mean: Tensor, log_var: Tensor) -> Tensor:
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()


def compute_kld_between_diagonal_gaussians(mean_a: Tensor, log_var_a: Tensor, mean_b: Tensor, log_var_b: Tensor) -> Tensor:
    assert mean_a.ndim == log_var_a.ndim == mean_b.ndim == log_var_b.ndim == 2

    var_term = (log_var_a.exp() / log_var_b.exp()).sum(dim=1)
    quadr_term = ((mean_a - mean_b).pow(2) / log_var_b.exp()).sum(dim=1)
    log_var_term = (log_var_b - log_var_a).sum(dim=1)
    k = mean_a.size(1)
    kld = 0.5 * (var_term + quadr_term - k + log_var_term)

    return kld.mean()
