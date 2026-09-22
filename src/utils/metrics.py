"""Reconstruction / depth evaluation metrics (torch-only).

Image quality metrics that need extra packages (LPIPS, SSIM via skimage) live
with their call sites; this module keeps only the dependency-light metrics used
across the eval scripts.
"""
from __future__ import annotations

import torch
from einops import reduce
from torch import Tensor


@torch.no_grad()
def compute_psnr(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b ... -> b", "mean")
    return -10 * mse.log10()


@torch.no_grad()
def compute_mse(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Per-sample MSE (e.g. for latent-space evaluation)."""
    return reduce((ground_truth - predicted) ** 2, "b ... -> b", "mean")


@torch.no_grad()
def compute_cosine_similarity(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    """Per-sample cosine similarity in [-1, 1] (1 = perfect alignment)."""
    gt_flat = ground_truth.flatten(start_dim=1)
    pred_flat = predicted.flatten(start_dim=1)
    return torch.nn.functional.cosine_similarity(gt_flat, pred_flat, dim=1)


# ==============================================================================
# Depth evaluation metrics
# ==============================================================================

def _as_hw(x: Tensor) -> Tensor:
    if x.dim() == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    return x


@torch.no_grad()
def compute_abs_rel(predicted: Tensor, ground_truth: Tensor,
                    valid_mask: Tensor = None) -> Tensor:
    """Absolute Relative Error: mean(|pred - gt| / gt) over valid pixels."""
    predicted = _as_hw(predicted)
    ground_truth = _as_hw(ground_truth)
    if valid_mask is None:
        valid_mask = ground_truth > 0
    out = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            out.append(torch.tensor(0.0, device=pred.device))
            continue
        out.append((torch.abs(pred[mask] - gt[mask]) / gt[mask]).mean())
    return torch.stack(out)


@torch.no_grad()
def compute_depth_rmse(predicted: Tensor, ground_truth: Tensor,
                       valid_mask: Tensor = None) -> Tensor:
    """RMSE = sqrt(mean((pred - gt)^2)) over valid pixels."""
    predicted = _as_hw(predicted)
    ground_truth = _as_hw(ground_truth)
    if valid_mask is None:
        valid_mask = ground_truth > 0
    out = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            out.append(torch.tensor(0.0, device=pred.device))
            continue
        out.append(torch.sqrt(((pred[mask] - gt[mask]) ** 2).mean()))
    return torch.stack(out)


@torch.no_grad()
def compute_delta(predicted: Tensor, ground_truth: Tensor,
                  threshold: float = 1.25, valid_mask: Tensor = None) -> Tensor:
    """delta accuracy: fraction of pixels with max(pred/gt, gt/pred) < threshold.

    Common thresholds: 1.25 (d1), 1.25**2 (d2), 1.25**3 (d3).
    """
    predicted = _as_hw(predicted)
    ground_truth = _as_hw(ground_truth)
    if valid_mask is None:
        valid_mask = ground_truth > 0
    out = []
    for pred, gt, mask in zip(predicted, ground_truth, valid_mask):
        if mask.sum() == 0:
            out.append(torch.tensor(1.0, device=pred.device))
            continue
        pv, gv = pred[mask], gt[mask]
        ratio = torch.maximum(pv / gv, gv / pv)
        out.append((ratio < threshold).float().mean())
    return torch.stack(out)
