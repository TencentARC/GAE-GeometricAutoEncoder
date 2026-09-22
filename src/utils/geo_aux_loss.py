"""Geometry auxiliary loss for diffusion training (VAE decode → DPT).

Target geometry always comes from **original / GT video** features:
``DPT(frozen_encoder(GT_RGB))``.  Gradients flow into the DiT x-prediction
through ``latent → VAE decode → DPT`` while encoder + DPT stay frozen.

Do **not** use DA3-on-generated-RGB as the training target — that is reserved
for eval / post-hoc distillation (expensive, biased, circular at train time).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from utils.dpt_helpers import (
    _format_feats_for_dpt,
    _format_recon_for_dpt,
    _geo_view_indices,
)


def _flatten_scene_output(output: dict) -> dict:
    """Remove DPT's singleton scene axis while preserving the view axis."""
    flattened = {}
    for key, value in output.items():
        if value is None:
            flattened[key] = None
        elif torch.is_tensor(value):
            if value.shape[0] != 1:
                raise ValueError(
                    f"DPT output {key!r} must have a singleton scene axis, "
                    f"got shape {tuple(value.shape)}"
                )
            flattened[key] = value.squeeze(0)
        else:
            flattened[key] = value
    return flattened


def _concat_scene_outputs(outputs: list[dict]) -> dict:
    """Concatenate per-scene DPT outputs along their flattened view axis."""
    if not outputs:
        raise ValueError("At least one scene output is required")
    result = {}
    for key in outputs[0]:
        values = [output.get(key) for output in outputs]
        tensors = [value for value in values if torch.is_tensor(value)]
        if not tensors:
            result[key] = None
        elif len(tensors) != len(values):
            raise ValueError(f"DPT output {key!r} is missing for some scenes")
        else:
            result[key] = torch.cat(tensors, dim=0)
    return result


@torch.no_grad()
def precompute_gt_dpt_geometry(
    all_feats: dict,
    backbone_norm,
    dpt_decoder,
    H: int,
    W: int,
    *,
    embed_dim: int,
    total_view: int,
) -> dict:
    """Run frozen DPT on each GT scene independently (no grad)."""
    bv = next(iter(all_feats.values())).shape[0]
    if bv % total_view != 0:
        raise ValueError(f"Feature batch {bv} is not divisible by total_view={total_view}")

    scene_outputs = []
    with torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, bv, total_view):
            scene_feats = {
                key: value[start:start + total_view]
                for key, value in all_feats.items()
            }
            dpt_in = _format_feats_for_dpt(
                scene_feats, backbone_norm, embed_dim=embed_dim,
            )
            scene_outputs.append(
                _flatten_scene_output(
                    dpt_decoder(dpt_in, H, W, patch_start_idx=0)
                )
            )
    return _concat_scene_outputs(scene_outputs)


def latent_raw_to_dpt_geometry(
    z_raw: torch.Tensor,
    vae,
    backbone_norm,
    dpt_decoder,
    H: int,
    W: int,
    *,
    embed_dim: int,
) -> dict:
    """Decode whitened/raw latent → VAE features → DPT (grad through z_raw)."""
    with torch.autocast(device_type="cuda", enabled=False):
        raw = vae.decode(z_raw.float())
        recon_feats = vae.denormalize_and_split(raw)
        bv = z_raw.shape[0]
        dpt_in = _format_recon_for_dpt(
            recon_feats, backbone_norm, bv, embed_dim=embed_dim,
        )
        return dpt_decoder(dpt_in, H, W, patch_start_idx=0)


def _ray_l1(pred_ray: torch.Tensor, gt_ray: torch.Tensor,
            gt_ray_conf: Optional[torch.Tensor]) -> torch.Tensor:
    if gt_ray_conf is not None:
        conf = gt_ray_conf.detach().sigmoid()
        while conf.ndim < gt_ray.ndim:
            conf = conf.unsqueeze(-1)
        return (conf * (pred_ray - gt_ray).abs()).sum() / (conf.sum() + 1e-8)
    return F.l1_loss(pred_ray, gt_ray)


def _depth_vhw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (V,H,W), got {tuple(depth.shape)}")
    return depth


def cross_view_depth_reprojection_loss(
    depth: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    *,
    stride: int = 8,
) -> torch.Tensor:
    """Adjacent-view depth consistency under fixed metric cameras."""
    depth = _depth_vhw(depth).float()
    views, height, width = depth.shape
    if views < 2:
        return depth.new_zeros(())

    ys = torch.arange(0, height, stride, device=depth.device, dtype=depth.dtype)
    xs = torch.arange(0, width, stride, device=depth.device, dtype=depth.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    sampled_source_depth = depth[:, ::stride, ::stride].reshape(views, -1)

    losses = []
    for source, target in zip(range(views - 1), range(1, views)):
        rays = pixels @ torch.linalg.inv(K[source].float()).T
        source_points = rays * sampled_source_depth[source, :, None]
        world_points = (
            source_points @ c2w[source, :3, :3].float().T
            + c2w[source, :3, 3].float()
        )
        target_w2c = torch.linalg.inv(c2w[target].float())
        target_points = (
            world_points @ target_w2c[:3, :3].T + target_w2c[:3, 3]
        )
        projected = target_points @ K[target].float().T
        z_projected = target_points[:, 2]
        u = projected[:, 0] / z_projected.clamp_min(1e-6)
        v = projected[:, 1] / z_projected.clamp_min(1e-6)
        grid = torch.stack([
            2.0 * u / max(width - 1, 1) - 1.0,
            2.0 * v / max(height - 1, 1) - 1.0,
        ], dim=-1).reshape(1, 1, -1, 2)
        sampled_target = F.grid_sample(
            depth[target][None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(-1)
        valid = (
            torch.isfinite(z_projected)
            & torch.isfinite(sampled_target)
            & (z_projected > 1e-4)
            & (sampled_target > 1e-4)
            & (grid.reshape(-1, 2).abs() <= 1.0).all(dim=-1)
        )
        if valid.any():
            scale = (
                0.5 * (z_projected[valid].detach() + sampled_target[valid].detach())
            ).clamp_min(1e-4)
            losses.append(
                F.l1_loss(
                    z_projected[valid] / scale,
                    sampled_target[valid] / scale,
                )
            )
    return torch.stack(losses).mean() if losses else depth.new_zeros(())


def compute_geo_aux_loss(
    x_pred: torch.Tensor,
    t: torch.Tensor,
    *,
    gt_ray: torch.Tensor,
    gt_depth: Optional[torch.Tensor],
    gt_ray_conf: Optional[torch.Tensor],
    denormalize_latent_fn: Callable[[torch.Tensor], torch.Tensor],
    geo_decode_fn: Callable[[torch.Tensor], dict],
    total_view: int,
    geo_t_max: float = 0.4,
    geo_max_views: int = 0,
    ray_weight: float = 1.0,
    depth_weight: float = 1.0,
    reproj_weight: float = 0.0,
    reproj_stride: int = 8,
    c2w: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 ray (+ optional depth) loss on low-noise x-pred steps.

    ``gt_*`` tensors must come from ``precompute_gt_dpt_geometry`` on the
    **original** video features for this batch.

    ``geo_max_views``: 0 or >= ``total_view`` → all views; otherwise randomly
    subsample that many views per batch element (cost cap).
    """
    if ray_weight <= 0 and depth_weight <= 0 and reproj_weight <= 0:
        return x_pred.new_zeros(())

    BV = x_pred.shape[0]
    assert BV % total_view == 0, (BV, total_view)
    B = BV // total_view
    device = x_pred.device

    low_t = t < geo_t_max
    if not low_t.any():
        return x_pred.new_zeros(())

    view_mask = torch.ones(B, total_view, dtype=torch.bool, device=device)
    if geo_max_views > 0 and geo_max_views < total_view:
        view_mask.zero_()
        for b in range(B):
            pick = torch.randperm(total_view, device=device)[:geo_max_views]
            view_mask[b, pick] = True

    token_mask = (low_t & view_mask.reshape(B * total_view)).reshape(B, total_view)
    scene_outputs = []
    selected_indices = []
    reproj_losses = []
    for b in range(B):
        scene_mask = token_mask[b]
        if not scene_mask.any():
            continue
        scene_start = b * total_view
        scene_indices = scene_start + scene_mask.nonzero(as_tuple=False).flatten()
        z_raw = denormalize_latent_fn(x_pred[scene_indices])
        scene_output = _flatten_scene_output(geo_decode_fn(z_raw.float()))
        scene_outputs.append(scene_output)
        selected_indices.append(scene_indices)
        if reproj_weight > 0:
            if c2w is None or K is None:
                raise ValueError("c2w and K are required when geo reprojection is enabled")
            local_indices = scene_indices - scene_start
            reproj_losses.append(
                cross_view_depth_reprojection_loss(
                    scene_output["depth"],
                    c2w[b, local_indices],
                    K[b, local_indices],
                    stride=reproj_stride,
                )
            )

    if not scene_outputs:
        return x_pred.new_zeros(())

    pred_out = _concat_scene_outputs(scene_outputs)
    flat_indices = torch.cat(selected_indices)
    pred_ray = pred_out["ray"]
    gt_ray_sel = gt_ray[flat_indices]
    gt_rc_sel = gt_ray_conf[flat_indices] if gt_ray_conf is not None else None

    loss = x_pred.new_zeros(())
    if ray_weight > 0:
        loss = loss + ray_weight * _ray_l1(pred_ray, gt_ray_sel.detach(), gt_rc_sel)

    pred_depth = pred_out.get("depth")
    if depth_weight > 0 and gt_depth is not None and pred_depth is not None:
        loss = loss + depth_weight * F.l1_loss(
            pred_depth, gt_depth[flat_indices].detach(),
        )
    if reproj_weight > 0 and reproj_losses:
        loss = loss + reproj_weight * torch.stack(reproj_losses).mean()

    return loss
