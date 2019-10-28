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
