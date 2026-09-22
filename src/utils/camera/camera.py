"""
Camera embedding utilities for multi-view diffusion.

Camera Convention
=================
The ``extrinsic`` argument throughout this module is **c2w** (camera-to-world)
with an **OpenCV camera frame** (X-right, Y-down, Z-forward), matching the
pinhole back-projection performed in ``batch_sample_rays``.

Per-dataset conventions (verified by Procrustes alignment vs DPT-recovered
OpenCV poses + multi-view point cloud overlap in world space):

* **RE10K** — ``transforms.json``'s ``transform_matrix`` is **OpenCV** c2w
  (despite the NeRF-style filename). Used as-is in the dataset loader.

* **DL3DV** — ``transforms.json``'s ``transform_matrix`` is **OpenGL** c2w
  (standard NeRF/instant-ngp: camera looks along -Z, Y-up). The dataset
  loader converts to OpenCV by ``c2w @ diag(1, -1, -1, 1)`` (flip camera
  Y/Z). Without this conversion, per-view direction agreement vs
  DPT-recovered poses is only cos≈+0.13~+0.86 (Z reversed); after
  conversion it becomes +0.99.

* **MVS-Synth** — ``c2w = inv(extrinsic)`` directly. Its rotation has
  ``det(R) = -1`` only because GTA-V's *world* frame is left-handed; the
  *camera* frame is still OpenCV, so projection is correct as-is. Do NOT
  flip Z or Y to "normalize" the rotation — doing so breaks the warp.

NOTE: Datasets with `transforms.json` come in *both* OpenCV (RE10K) and
OpenGL (DL3DV) flavours, so per-dataset detection is essential.

PRoPE consumes **w2c** (world-to-camera) matrices. Training code computes
``w2c = inv(c2w)`` before passing to ``model_kwargs['viewmats']``.

Plücker coordinates: ``[ray_direction, origin × direction]`` (6 channels).
Translation is normalized by the max absolute component per batch.

Reference-view convention
-------------------------
We use the **last view** (``index = -1``) as the geometric reference for
ref-centric normalization. This is intentional and decoupled from the
"condition view" assignment in ``cut3r_adapter._get_view_order`` (which
puts condition views in the **prefix**, indices ``0..cond_num-1``):

  * ``cut3r_adapter`` decides which views are conditions (RGB given to
    the model as known) — by default the prefix.
  * PRoPE / ``batch_sample_rays`` decides which view is the **geometric
    origin** (``c2w_norm[:, ref] = I``) — fixed to ``-1``.

These can be (and usually are) different views — that is by design, not a
bug. ``normalize_extrinsic_tgt=-1`` should be kept as the default unless
both call sites are changed in lockstep (training + eval + adapter).
"""

import einops
import torch
import torch.nn.functional as F

from utils.camera.position_encoding import freq_encoding


@torch.cuda.amp.autocast(enabled=False)
def sample_rays(intrinsic, extrinsic, image_h=None, image_w=None,
                normalize_extrinsic=False, normalize_std=False):
    """Sample per-pixel rays from camera parameters.

    Args:
        intrinsic: [B, 3, 3] camera intrinsic matrix (OpenCV convention: fx, fy, cx, cy)
        extrinsic: [B, 4, 4] camera-to-world (OpenCV c2w).
        image_h, image_w: output ray grid resolution
        normalize_extrinsic: if True, normalize first camera to identity

    Returns:
        rays_o, rays_d: [B, N, 3] ray origins and directions in world frame
    """

    device = intrinsic.device
    B = intrinsic.shape[0]
    if normalize_extrinsic:
        extrinsic = extrinsic[0:1].inverse() @ extrinsic

    c2w = extrinsic[:, :3, :4]  # [B,3,4]
    # OpenCV pixel-center sampling: pixel (i, j) center is at (i+0.5, j+0.5).
    # Symmetric +0.5 on both axes → optical axis (cx, cy) maps to ray ≈ +Z.
    x = torch.arange(image_w, device=device).float() + 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [B,N,3]

    rays_o = c2w[..., :3, 3]  # [B, 3]

    if normalize_std:
        rays_o = rays_o / rays_o.std(dim=0, keepdim=True)

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [B, N, 3]

    return rays_o, rays_d


@torch.cuda.amp.autocast(enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None,
                      normalize_extrinsic=False, normalize_t=False, nframe=1,
                      normalize_extrinsic_tgt=-1,
                      ):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4], camera-to-world (OpenCV c2w)
        h, w: int
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    device = intrinsic.device
    B = intrinsic.shape[0]
    if extrinsic.shape[-2] == 3:
        new_extrinsic = torch.zeros((B, 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
        new_extrinsic[:, :3, :4] = extrinsic
        new_extrinsic[:, 3, 3] = 1.0
        extrinsic = new_extrinsic

    if normalize_extrinsic:
        extri_ = einops.rearrange(extrinsic, "(b f) r c -> b f r c", f=nframe)
        # Normalize relative to reference view: make ref view identity
        ref_c2w_inv = extri_[:, normalize_extrinsic_tgt].inverse().to(device)  # [B,4,4]
        ref_c2w_inv = ref_c2w_inv.repeat_interleave(nframe, dim=0)  # [BF,4,4]
        extrinsic = ref_c2w_inv @ extrinsic

    c2w = extrinsic[:, :3, :4].to(device)  # [BF,3,4]
    # OpenCV pixel-center sampling: pixel (i, j) center is at (i+0.5, j+0.5).
    # Symmetric +0.5 on both axes → optical axis (cx, cy) maps to ray ≈ +Z.
    x = torch.arange(image_w, device=device).float() + 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    scale = None
    if normalize_t:
        rays_o = einops.rearrange(rays_o, "(b f) c -> b f c", f=nframe)
        # normalize the farthest to 1.0 (direct3d)
        farthest, _ = rays_o.abs().max(dim=1, keepdim=True)
        farthest, _ = farthest.max(dim=2, keepdim=True)
        scale = 1.0 / (farthest + 1e-8)
        rays_o = rays_o * scale
        rays_o = einops.rearrange(rays_o, "b f c -> (b f) c")

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d, scale

@torch.cuda.amp.autocast(enabled=False)
def embed_rays(
    rays_o,
    rays_d,
    nframe,
    mode: str = "plucker",  # "plucker" | "camray"
    fourier_embedding: bool = False,
    fourier_embed_dim: int = 16,
    camera_longest_side=None,
):
    """
    Embed rays for camera conditioning.
    
    Args:
        rays_o: Ray origins [b,f,n,3] or [(b f),n,3]
        rays_d: Ray directions [b,f,n,3] or [(b f),n,3]
        nframe: Number of frames
        mode: "plucker" for [d, o×d] or "camray" for direction only
        fourier_embedding: Whether to add Fourier positional encoding
        fourier_embed_dim: Dimension of Fourier embedding
        camera_longest_side: For Fourier encoding normalization
    
    Returns:
        cam_emb: Camera embedding [b, f, n, c]
    """
    # Accept [b,f,n,3] or [(b f),n,3]
    if len(rays_d.shape) == 4:  # [b,f,n,3]
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
        if rays_o is not None:
            rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")

    if mode.lower() == "camray":
        # CamRay: use direction only (works well with PRoPE for extrinsics)
        if fourier_embedding:
            fourier_pe = freq_encoding(
                rays_d, embed_dim=fourier_embed_dim, camera_longest_side=camera_longest_side
            )
            cam_emb = torch.cat([rays_d, fourier_pe], dim=-1)
        else:
            cam_emb = rays_d

    elif mode.lower() == "plucker":
        # Plücker: [d, o × d] (+ optional Fourier)
        cross_od = torch.cross(rays_o, rays_d, dim=-1)
        if fourier_embedding:
            fourier_pe = freq_encoding(
                cross_od, embed_dim=fourier_embed_dim, camera_longest_side=camera_longest_side
            )
            cam_emb = torch.cat([rays_d, cross_od, fourier_pe], dim=-1)
        else:
            cam_emb = torch.cat([rays_d, cross_od], dim=-1)

    else:
        raise ValueError(f"Unknown camera embedding mode: {mode}. Use 'plucker' or 'camray'.")

    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


def get_camera_embedding(
    intrinsic,
    extrinsic,
    b,
    f,
    h,
    w,
    mode: str = "camray",  # "plucker" | "camray" (default: camray for PRoPE compatibility)
    normalize_extrinsic: bool = True,
    normalize_t: bool = None,  # Optional override; if None, inferred from mode
    return_scale: bool = False,
):
    """
    Get camera embedding from intrinsics and extrinsics (OpenCV c2w).

    Args:
        intrinsic: [BF, 3, 3] camera intrinsic matrix.
        extrinsic: [BF, 4, 4] or [BF, 3, 4] camera-to-world (OpenCV).
        b, f, h, w: batch size, views, image height, image width.
        mode: "plucker" for [d, o×d] (6ch) or "camray" for direction only (3ch).
        normalize_extrinsic: Normalize poses relative to reference frame.
        normalize_t: Normalize translation scale (default: True for plucker).
        return_scale: Return the normalization scale factor.

    Returns:
        camera_embedding: [B, F, C, H, W]
        scale (optional): [B, 1, 1]
    """
    # CamRay doesn't use translation, so normalize_t is unnecessary (default False)
    # Plücker's moment term is scale-sensitive, so normalize_t is beneficial (default True)
    if normalize_t is None:
        if return_scale:
            normalize_t = True
        else:
            normalize_t = (mode.lower() == "plucker")

    rays_o, rays_d, scale = batch_sample_rays(
        intrinsic, extrinsic,
        image_h=h, image_w=w,
        normalize_extrinsic=normalize_extrinsic,
        normalize_t=normalize_t,
        normalize_extrinsic_tgt=-1,
        nframe=f,
    )

    camera_embedding = embed_rays(
        rays_o if mode.lower() == "plucker" else None,
        rays_d,
        nframe=f,
        mode=mode,
        fourier_embedding=False,
        fourier_embed_dim=None,
        camera_longest_side=None
    )

    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b f c h w", h=h, w=w)
    
    if return_scale:
        return camera_embedding, scale
    
    return camera_embedding
