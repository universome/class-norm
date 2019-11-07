import torch
from torch import Tensor, autograd


def compute_gradient_penalty(discriminator, x_real, x_fake) -> Tensor:
    """
    Computes gradient penalty according to WGAN-GP paper
    """
    assert x_real.ndim == 2, "If you want to use other input sizes, implement interpolations for them first."
    assert x_real.size() == x_fake.size()

    eps = torch.rand(x_real.size(0), 1).to(x_real.device)
    interpolations = eps * x_real + (1 - eps) * x_fake
    interpolations.requires_grad_(True)
    outputs = discriminator(interpolations)

    grads = autograd.grad(
        outputs=outputs,
        inputs=interpolations,
        grad_outputs=torch.ones(outputs.size()).to(x_real.device),
        retain_graph=True, create_graph=True, only_inputs=True
    )[0]

    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


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
