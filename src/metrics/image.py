"""Image-level reconstruction and generation metrics — paper Tables 3 and 5.

Paired fidelity (comparing each generated target view with its real frame):

* ``psnr``  — peak signal-to-noise ratio, higher is better.
* ``ssim``  — structural similarity (uniform 11x11 window), higher is better.
* ``lpips`` — learned perceptual similarity, lower is better.

Distributional quality (FID / rFID for frames, FVD / rFVD for clips) needs
feature statistics over a whole split and is provided by
:mod:`src.metrics.distribution` rather than as a per-pair function.

All functions accept float tensors in ``[0, 1]`` with shape ``[..., C, H, W]``
and reduce over the leading batch dimensions.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = ["psnr", "ssim", "lpips", "LPIPSMetric"]


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x.flatten(0, -4)


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR per sample, computed on the full value range [0, 1].

    Args:
        pred: Generated images, ``[N, C, H, W]`` in ``[0, 1]``.
        target: Reference images, same shape and range.

    Returns:
        ``[N]`` PSNR in dB.
    """
    p, t = _as_bchw(pred.float()), _as_bchw(target.float())
    mse = (p - t).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g).expand(1, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SSIM per sample (uniform-window formulation of Wang et al., 2004).

    Args:
        pred: ``[N, C, H, W]`` in ``[0, 1]``.
        target: Same shape and range.
        window_size: Gaussian window size.
        sigma: Gaussian standard deviation.

    Returns:
        ``[N]`` SSIM in ``[-1, 1]``, higher is better.
    """
    p, t = _as_bchw(pred.float()), _as_bchw(target.float())
    c = p.shape[1]
    win = _gaussian_window(window_size, sigma, p.device, p.dtype).repeat(c, 1, 1, 1)

    def _stats(x: torch.Tensor):
        mu = F.conv2d(x, win, padding=window_size // 2, groups=c)
        mu2 = mu.pow(2)
        sigma2 = F.conv2d(x * x, win, padding=window_size // 2, groups=c) - mu2
        return mu, mu2, sigma2

    mu_p, mu_p2, var_p = _stats(p)
    mu_t, mu_t2, var_t = _stats(t)
    cov = F.conv2d(p * t, win, padding=window_size // 2, groups=c) - mu_p * mu_t

    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_p * mu_t + c1) * (2 * cov + c2)
    den = (mu_p2 + mu_t2 + c1) * (var_p + var_t + c2)
    return (num / (den + eps)).flatten(1).mean(dim=1)


class LPIPSMetric:
    """Thin wrapper around the bundled LPIPS implementation.

    Keeps one frozen network alive across calls, which matters because
    instantiating it per batch is expensive. Inputs are ``[0, 1]``; they are
    mapped to ``[-1, 1]`` internally, as LPIPS expects.
    """

    def __init__(self, net: str = "alex", device: Optional[torch.device] = None) -> None:
        from disc.lpips import LPIPS  # local import: pulls in torchvision weights

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fn = LPIPS(net=net).to(self.device).eval()
        self._fn.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _as_bchw(pred.float()).to(self.device) * 2.0 - 1.0
        t = _as_bchw(target.float()).to(self.device) * 2.0 - 1.0
        return self._fn(p, t).flatten()


def lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """LPIPS per sample. Convenience wrapper; reuse :class:`LPIPSMetric` in loops."""
    return LPIPSMetric(net=net, device=device)(pred, target)
