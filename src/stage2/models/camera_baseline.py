"""GLD-original baseline camera conditioning (spatial embed + ProPE).

Mirrors ``DiTwDDTHead`` in ``DDT.py`` / ``DA3_level1.yaml``:
  1. Per-pixel camera map (mask + Plücker/camray channels) → PatchEmbed → add to encoder tokens.
  2. ``viewmats`` (w2c) + ``Ks`` → ProPE inside cross-view attention.

Used when ``stage_2.params.camera_conditioning: baseline_prope``. The default
``plucker_flip_pe`` path keeps per-token ``plucker_6d`` + Plücker Flip PE on Q/K.
"""
from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from utils.camera.camera import get_camera_embedding


def build_baseline_camera_inputs(
    c2w: torch.Tensor,
    Ks: torch.Tensor,
    *,
    H: int,
    W: int,
    cond_num: int,
    total_view: int,
    camera_mode: str = "plucker",
    v4_cond_extend: bool = False,
) -> dict[str, torch.Tensor | int]:
    """Build baseline camera tensors for DiT encoder.

    Args:
        c2w: (B, V, 4, 4) camera-to-world (same normalisation as plucker path).
        Ks: (B, V, 3, 3) intrinsics.
        v4_cond_extend: if True, duplicate cond-view cameras as a prefix
            (V4 token-concat: K cond + V view → K+V encoder views).

    Returns:
        camera_embedding: (B*V_enc, 7, H, W) — mask + 6ch plucker (default).
        viewmats: (B, V_enc, 4, 4) world-to-camera for ProPE.
        Ks: (B, V_enc, 3, 3).
        total_enc_views: V_enc (= V or K+V).
    """
    B, V = c2w.shape[:2]
    device, dtype = c2w.device, c2w.dtype

    extri = c2w[..., :3, :4]
    extri_flat = rearrange(extri, "b v r c -> (b v) r c")
    intri_flat = rearrange(Ks, "b v r c -> (b v) r c")

    cam_emb, scale = get_camera_embedding(
        intri_flat, extri_flat, B, V, H, W,
        mode=camera_mode, return_scale=True,
    )
    masks = torch.ones(B, V, 1, H, W, device=device, dtype=cam_emb.dtype)
    masks[:, :cond_num] = 0.0
    camera_map = torch.cat([masks, cam_emb], dim=2)  # (B, V, 7, H, W)

    c2w_4 = c2w.float()
    w2c = torch.linalg.inv(c2w_4)
    if scale is not None:
        w2c = w2c.clone()
        w2c[..., :3, 3] = w2c[..., :3, 3] * scale.to(w2c.dtype)

    if v4_cond_extend and cond_num > 0:
        cam_cond = camera_map[:, :cond_num]
        camera_map = torch.cat([cam_cond, camera_map], dim=1)
        w2c_cond = w2c[:, :cond_num]
        w2c = torch.cat([w2c_cond, w2c], dim=1)
        Ks = torch.cat([Ks[:, :cond_num], Ks], dim=1)
        total_enc_views = cond_num + V
    else:
        total_enc_views = V

    camera_embedding = rearrange(camera_map, "b v c h w -> (b v) c h w").to(dtype)
    return dict(
        camera_embedding=camera_embedding,
        viewmats=w2c.to(dtype),
        Ks=Ks.to(dtype),
        total_enc_views=total_enc_views,
    )


def apply_baseline_camera_dropout(
    camera_embedding: torch.Tensor,
    viewmats: torch.Tensor,
    drop_mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample camera dropout (training CFG channel).

    Zeros pose channels (1:) of the spatial map and replaces viewmats with
    identity for dropped samples — mirrors ``camera_drop`` on ``plucker_6d``.
    """
    if drop_mask_b is None:
        return camera_embedding, viewmats
    # drop_mask_b: (B, 1, 1), 1=keep, 0=drop
    B = drop_mask_b.shape[0]
    V_enc = camera_embedding.shape[0] // B
    m = drop_mask_b.to(camera_embedding.dtype).view(B, 1, 1, 1, 1)
    m_flat = m.expand(B, V_enc, 1, 1, 1).reshape(B * V_enc, 1, 1, 1)
    cam = camera_embedding.clone()
    cam[:, 1:] = cam[:, 1:] * m_flat

    vm = viewmats.clone()
    eye = torch.eye(4, device=vm.device, dtype=vm.dtype)
    drop_idx = (drop_mask_b.view(B) < 0.5).nonzero(as_tuple=False).view(-1)
    if drop_idx.numel() > 0:
        vm[drop_idx] = eye.unsqueeze(0).expand(drop_idx.numel(), V_enc, 4, 4)
    return cam, vm


def make_uncond_baseline_camera(
    camera_embedding: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unconditional branch for CFG (baseline): zero pose channels, identity extrinsics."""
    uncond_cam = camera_embedding.clone()
    uncond_cam[:, 1:] = 0.0
    eye4 = torch.eye(4, device=viewmats.device, dtype=viewmats.dtype)
    uncond_vm = eye4.view(1, 1, 4, 4).expand_as(viewmats).clone()
    return uncond_cam, uncond_vm, Ks
