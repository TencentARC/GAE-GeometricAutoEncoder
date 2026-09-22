"""Adapted and modified from https://github.com/CompVis/taming-transformers"""

import torch
import torch.nn.functional as F


def hinge_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Hinge discriminator loss used by VQGAN."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    loss_real = reduce(F.relu(1.0 - logits_real))
    loss_fake = reduce(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN discriminator loss."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    return 0.5 * (
        reduce(F.softplus(-logits_real)) + reduce(F.softplus(logits_fake))
    )


def vanilla_g_loss(logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN generator loss."""
    if reduction == "mean":
        return -torch.mean(logits_fake)
    if reduction == "sum":
        return -torch.sum(logits_fake)
    raise ValueError(f"Unsupported reduction '{reduction}'")
