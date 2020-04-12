import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, autograd

from src.utils.training_utils import normalize


def compute_gradient_penalty(discriminator, x_real, x_fake, y: Tensor=None):
    """
    Computes gradient penalty according to WGAN-GP paper
    Args:
    - y — class labels for cGAN
    """
    assert x_real.size() == x_fake.size()

    shape = [x_real.size(0)] + [1] * (x_real.dim() - 1)
    alpha = torch.rand(shape).to(x_real.device)
    interpolations = x_real + alpha * (x_fake - x_real)

    interpolations = interpolations.to(x_real.device)
    interpolations.requires_grad_(True)

    if y is None:
        outputs = discriminator(interpolations)
    else:
        outputs = discriminator(interpolations, y)

    grads = autograd.grad(
        outputs,
        interpolations,
        grad_outputs=torch.ones(outputs.size()).to(interpolations.device),
        create_graph=True
    )[0].view(interpolations.size(0), -1)

    return ((grads.norm(p=2, dim=1) - 1) ** 2).mean()


def compute_kld_with_standard_gaussian(mean: Tensor, log_var: Tensor) -> Tensor:
    return -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1).mean()


def compute_kld_between_diagonal_gaussians(
        mean_a: Tensor,
        log_var_a: Tensor,
        mean_b: Tensor,
        log_var_b: Tensor,
        reduction:str='mean') -> Tensor:

    assert mean_a.ndim == log_var_a.ndim == mean_b.ndim == log_var_b.ndim == 2

    var_term = (log_var_a.exp() / log_var_b.exp()).sum(dim=1)
    quadr_term = ((mean_a - mean_b).pow(2) / log_var_b.exp()).sum(dim=1)
    log_var_term = (log_var_b - log_var_a).sum(dim=1)
    k = mean_a.size(1)
    kld = 0.5 * (var_term + quadr_term - k + log_var_term)

    if reduction == 'mean':
        return kld.mean()
    else:
        return kld


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing by @PistonY
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    """
    def __init__(self, num_classes: int, smoothing_coef=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()

        self.confidence = 1.0 - smoothing_coef
        self.smoothing_coef = smoothing_coef
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, logits, target):
        logits = logits.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing_coef / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * logits, dim=self.dim))


def compute_mean_distance(protos: Tensor, metric: str='cosine') -> Tensor:
    """
    pass
    """
    assert metric == 'cosine', "Other metrics are not implemented"
    assert protos.ndim == 3, "Dimensionality must be [n_protos x n_classes x feat_dim]"

    n_protos, n_classes, feat_dim = protos.shape
    protos = normalize(protos) # [n_protos x n_classes x feat_dim]
    cosines = torch.matmul(protos.permute(1, 0, 2), protos.permute(1, 2, 0))

    assert cosines.shape == (n_classes, n_protos, n_protos)

    # Let's remove the main diag
    # TODO: do we really need it? Cosine distance with itself would be equal to 1 anyway...
    cosines = cosines * (1 - torch.eye(n_protos, device=cosines.device)).view(1, n_protos, n_protos)

    return (2 - cosines).mean()


def compute_gdpp_loss(phiFake, phiReal):
    r"""
    Copypasted from: https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/loss_criterions/GDPP_loss.py

    Implementation of the GDPP loss. Can be used with any kind of GAN
    architecture.
    Args:
        phiFake (tensor) : last feature layer of the discriminator on real data
        phiReal (tensor) : last feature layer of the discriminator on fake data
    Returns:
        Loss's value. The backward operation in performed within this operator
    """
    fakeEigVals, fakeEigVecs = gdpp_compute_diversity(phiFake)
    realEigVals, realEigVecs = gdpp_compute_diversity(phiReal)

    # Scaling factor to make the two losses operating in comparable ranges.
    magnitudeLoss = 0.0001 * F.mse_loss(target=realEigVals, input=fakeEigVals)
    structureLoss = -torch.sum(torch.mul(fakeEigVecs, realEigVecs), 0)
    normalizedRealEigVals = gdpp_normalize_min_max(realEigVals)
    weightedStructureLoss = torch.sum(torch.mul(normalizedRealEigVals, structureLoss))
    gdppLoss = magnitudeLoss + weightedStructureLoss

    return gdppLoss


def gdpp_compute_diversity(phi):
    phi = F.normalize(phi, p=2, dim=1)
    SB = torch.mm(phi, phi.t())
    eig_vals, eig_vecs = torch.symeig(SB, eigenvectors=True)

    return eig_vals, eig_vecs


def gdpp_normalize_min_max(eig_vals):
    minV, maxV = torch.min(eig_vals), torch.max(eig_vals)
    if abs(minV - maxV) < 1e-10:
        return eig_vals
    return (eig_vals - minV) / (maxV - minV)
