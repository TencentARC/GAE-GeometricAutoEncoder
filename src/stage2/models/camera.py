"""Metric ray-space camera conditioning — paper Section 3.3, Eq. (13).

For each latent cell and camera we form the world-space Plucker ray
``r = (d, m)`` with ``d`` the direction and ``m = o x d`` the moment, then
decompose it as

    d,    m_hat = m / (||m|| + eps),    s = log(||m|| + eps)

and inject the resulting ray embedding into query/key self-attention,
following the ray-space conditioning philosophy of RayPE. The log-scale
channel is what preserves metric translation magnitude when metric camera
sidecars are available. Reference tokens receive the ray embeddings of their
observed cameras, which spatially registers the clean evidence.

Two modules live here:

* :func:`compute_plucker_6d_per_token` / :func:`compute_plucker_6d` — ray
  construction from camera matrices (``c2w``, ``K``) onto the patch grid.
* :class:`PluckerFlipPE_V13` — normalize-gate-inject positional encoding.
  Query sees ``(d, m_hat, s)`` and key sees ``(m_hat, d, s)``; the "flip"
  makes ``q^T k`` approximate the Klein form ``m_i . d_j + d_i . m_j``, which
  encodes ray-intersection geometry rather than direction similarity alone.

.. note::
   ``normalize_moment`` defaults to ``False``. Training performs per-batch
   translation normalisation (max |t| = 1) at the dataset level, which already
   keeps ``||m||`` in a healthy range, so the explicit median rescale is off by
   default. Enable it if you feed unnormalised metric cameras.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _compute_rays(
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    patches_x: int,
    patches_y: int,
    image_width: int,
    image_height: int,
    is_c2w: bool = True,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute world-frame ray origin & unit direction for each patch centre.

    Args:
        viewmats: (B, V, 4, 4) camera matrices (c2w by default).
        Ks: (B, V, 3, 3) intrinsics in pixel space.
        patches_x, patches_y: spatial token grid (e.g. 36×36).
        image_width, image_height: pixel-space image dims (e.g. 504×504).
        is_c2w: True if viewmats is c2w; otherwise w2c is converted to c2w.
        eps: numerical safety for normalisation.

    Returns:
        rays_o: (B, V, Py, Px, 3) origin (camera centre in world frame).
        rays_d: (B, V, Py, Px, 3) unit direction in world frame.
    """
    B, V = viewmats.shape[:2]
    device = viewmats.device
    dtype = viewmats.dtype

    patch_w = image_width / patches_x
    patch_h = image_height / patches_y
    u = torch.linspace(
        0.5 * patch_w,
        image_width - 0.5 * patch_w,
        patches_x,
        device=device,
        dtype=dtype,
    )
    v = torch.linspace(
        0.5 * patch_h,
        image_height - 0.5 * patch_h,
        patches_y,
        device=device,
        dtype=dtype,
    )
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")  # (Px, Py) → broadcast as (Py, Px)

    grid_u = grid_u.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
    grid_v = grid_v.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)

    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    x_cam = (grid_u - cx[..., None, None]) / fx[..., None, None]
    y_cam = (grid_v - cy[..., None, None]) / fy[..., None, None]
    z_cam = torch.ones_like(x_cam)
    d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    d_cam = F.normalize(d_cam, dim=-1, eps=eps)

    if is_c2w:
        R = viewmats[..., :3, :3]
        t = viewmats[..., :3, 3]
    else:
        R_w2c = viewmats[..., :3, :3]
        t_w2c = viewmats[..., :3, 3]
        R = R_w2c.transpose(-1, -2)
        t = -torch.einsum("bvij,bvj->bvi", R, t_w2c)

    d_world = torch.einsum("bvij,bvhwj->bvhwi", R, d_cam)
    d_world = F.normalize(d_world, dim=-1, eps=eps)

    rays_o = t[..., None, None, :].expand_as(d_world)
    return rays_o, d_world


def compute_plucker_6d_per_token(
    c2w: torch.Tensor,
    Ks: torch.Tensor,
    patches_x: int,
    patches_y: int,
    image_height: int,
    image_width: int,
    normalize_moment: bool = False,
    is_c2w: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-token 6D Plücker coordinates.

    Args:
        c2w: (B, V, 4, 4) camera matrices (or w2c if `is_c2w=False`).
        Ks: (B, V, 3, 3) intrinsics.
        patches_x, patches_y: spatial token grid.
        image_height, image_width: pixel-space image dims.
        normalize_moment: if True, divide moment m by per-batch median ‖m‖.
            Default False; the diffusion training already does per-batch
            translation normalisation, which keeps ‖m‖ in a healthy range.
        is_c2w: True if viewmats argument is c2w.
        eps: numerical safety for moment normalisation.

    Returns:
        plucker_6d: (B, V*Py*Px, 6) where channels 0..2 = d, channels 3..5 = m.
    """
    rays_o, rays_d = _compute_rays(
        viewmats=c2w,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        is_c2w=is_c2w,
    )
    rays_m = torch.cross(rays_o, rays_d, dim=-1)

    if normalize_moment:
        B = rays_m.shape[0]
        m_norm = rays_m.flatten(1, 3).norm(dim=-1)
        scale = m_norm.median(dim=-1).values.clamp(min=eps)
        rays_m = rays_m / scale.view(B, 1, 1, 1, 1)

    B = rays_d.shape[0]
    d_flat = rays_d.reshape(B, -1, 3)
    m_flat = rays_m.reshape(B, -1, 3)
    return torch.cat([d_flat, m_flat], dim=-1)


class _RMSNorm(nn.Module):
    """Per-token RMSNorm with learnable scale (matches WAN/UCPE QKNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


class PluckerFlipPE_V13(nn.Module):
    """
    Plücker Flip PE with Normalize-Gate-Inject.

    Args:
        dim: attention feature dimension (= num_heads * head_dim).
        plucker_init: "zero" or "small" for E_q/E_k output-layer initialization.
            "zero" makes the model start as a no-camera baseline (recommended).
        plucker_init_scale: std for "small" init.
        plucker_mlp_hidden: if > 0, use 7→hidden→dim MLP; if 0, use 7→dim Linear.
        plucker_scale: if > 0, add learnable α_q/α_k initialised to this value.
            With PE RMSNorm, α=1.0 means geometry and content contribute equally.
        gate_init_bias: initial value for cam_residual gate logit (only used
            when enable_cam_residual=True).
        enable_cam_residual: whether to add frame-uniform gated camera residual.
        scale_gate_hidden: hidden dim of the scale gate MLP. If 0, defaults to
            max(dim // 4, 1).
        log_scale_aug_prob: training-time probability for perturbing only the
            scale-gate log‖m‖ input.
        log_scale_aug_range: uniform additive range for the perturbation.
    """

    def __init__(
        self,
        dim: int,
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 0,
        plucker_scale: float = 0.0,
        gate_init_bias: float = -2.0,
        enable_cam_residual: bool = False,
        scale_gate_hidden: int = 0,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple[float, float] = (-1.2, 1.6),
    ):
        super().__init__()
        self.dim = dim
        self.use_mlp = plucker_mlp_hidden > 0
        self.use_scale = plucker_scale > 0
        self.enable_cam_residual = enable_cam_residual
        self.log_scale_aug_prob = float(log_scale_aug_prob)
        self.log_scale_aug_range = tuple(float(x) for x in log_scale_aug_range)
        if self.log_scale_aug_prob < 0.0 or self.log_scale_aug_prob > 1.0:
            raise ValueError("log_scale_aug_prob must be in [0, 1]")
        if len(self.log_scale_aug_range) != 2:
            raise ValueError("log_scale_aug_range must have two values")
        if self.log_scale_aug_range[0] > self.log_scale_aug_range[1]:
            raise ValueError("log_scale_aug_range min must be <= max")

        in_dim = 7  # (d(3), m̂(3), log_s(1))

        # ── Q/K geometric projections ────────────────────────────────────
        if self.use_mlp:
            self.eq = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
            self.ek = nn.Sequential(
                nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                nn.GELU(),
                nn.Linear(plucker_mlp_hidden, dim, bias=False),
            )
        else:
            self.eq = nn.Linear(in_dim, dim, bias=False)
            self.ek = nn.Linear(in_dim, dim, bias=False)

        # ── PE RMSNorm: align PE magnitude with content QKNorm ──────────
        self.norm_pe_q = _RMSNorm(dim)
        self.norm_pe_k = _RMSNorm(dim)

        # ── Scale gate: log_scale → (0, 1) per-dim ──────────────────────
        sg_hidden = scale_gate_hidden if scale_gate_hidden > 0 else max(dim // 4, 1)
        self.scale_gate = nn.Sequential(
            nn.Linear(1, sg_hidden),
            nn.SiLU(),
            nn.Linear(sg_hidden, dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.scale_gate[0].bias)
        nn.init.zeros_(self.scale_gate[2].bias)

        # ── Learnable per-layer scale α ──────────────────────────────────
        if self.use_scale:
            self.alpha_q = nn.Parameter(torch.tensor(plucker_scale))
            self.alpha_k = nn.Parameter(torch.tensor(plucker_scale))

        # ── Optional cam_residual (kept for API parity with V13) ────────
        if self.enable_cam_residual:
            if self.use_mlp:
                self.ev = nn.Sequential(
                    nn.Linear(in_dim, plucker_mlp_hidden, bias=False),
                    nn.GELU(),
                    nn.Linear(plucker_mlp_hidden, dim, bias=False),
                )
                self.gate_proj = nn.Sequential(
                    nn.Linear(in_dim, plucker_mlp_hidden, bias=True),
                    nn.GELU(),
                    nn.Linear(plucker_mlp_hidden, dim, bias=False),
                )
            else:
                self.ev = nn.Linear(in_dim, dim, bias=False)
                self.gate_proj = nn.Linear(in_dim, dim, bias=False)
            self.gate_logit = nn.Parameter(torch.full((dim,), gate_init_bias))

        self._init_weights(plucker_init, plucker_init_scale)

    def _gate_log_scale(self, log_scale: torch.Tensor) -> torch.Tensor:
        """Return log scale for the gate; augmentation does not alter Q/K PE."""
        if (
            not self.training
            or self.log_scale_aug_prob <= 0.0
            or torch.rand((), device=log_scale.device) >= self.log_scale_aug_prob
        ):
            return log_scale
        lo, hi = self.log_scale_aug_range
        delta = torch.empty_like(log_scale).uniform_(lo, hi)
        return log_scale + delta

    def _init_weights(self, mode: str, scale: float):
        if self.use_mlp:
            # For 2-layer MLP with zero init: only zero the OUTPUT layer.
            # Zeroing both creates dead gradients (h=GELU(0)=0 → ∂L/∂W=0).
            qk_output_layers = [self.eq[2], self.ek[2]]
            qk_input_layers = [self.eq[0], self.ek[0]]
        else:
            qk_output_layers = [self.eq, self.ek]
            qk_input_layers = []

        for m in qk_output_layers:
            if mode == "zero":
                nn.init.zeros_(m.weight)
            else:
                nn.init.normal_(m.weight, 0.0, scale)

        for m in qk_input_layers:
            if mode == "zero":
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
            else:
                nn.init.normal_(m.weight, 0.0, scale)

        if self.enable_cam_residual:
            v_modules = [self.ev[0], self.ev[2]] if self.use_mlp else [self.ev]
            for m in v_modules:
                nn.init.normal_(m.weight, 0.0, scale)
            if self.use_mlp:
                nn.init.xavier_uniform_(self.gate_proj[0].weight)
                nn.init.zeros_(self.gate_proj[0].bias)
                nn.init.zeros_(self.gate_proj[2].weight)
            else:
                nn.init.zeros_(self.gate_proj.weight)

    @staticmethod
    def decompose_plucker(plucker_6d: torch.Tensor):
        """Decompose (d, m) → (d, m̂, log‖m‖).

        Args:
            plucker_6d: (..., 6) raw Plücker (d direction, m moment).

        Returns:
            feat_q: (..., 7) = (d, m̂, log_s) for E_q projection.
            feat_k: (..., 7) = (m̂, d, log_s) for E_k projection (flip d ↔ m̂).
            log_scale: (..., 1) for scale gate input.
        """
        d = plucker_6d[..., :3]
        m = plucker_6d[..., 3:]

        m_norm = m.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        m_hat = m / m_norm
        log_scale = torch.log(m_norm)

        feat_q = torch.cat([d, m_hat, log_scale], dim=-1)
        feat_k = torch.cat([m_hat, d, log_scale], dim=-1)
        return feat_q, feat_k, log_scale

    def apply_to_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
    ):
        """Add scale-gated Plücker PE to Q and K.

        Args:
            q: (B, S, D) query (post-RoPE if applicable).
            k: (B, S, D) key (post-RoPE if applicable).
            plucker_6d: (B, S, 6) raw Plücker coordinates (d, m).

        Returns:
            (q, k) with additive Normalize-Gate-Inject PE applied.
        """
        feat_q, feat_k, log_scale = self.decompose_plucker(plucker_6d)

        pe_q = self.norm_pe_q(self.eq(feat_q.to(q.dtype)))
        pe_k = self.norm_pe_k(self.ek(feat_k.to(k.dtype)))

        gate = self.scale_gate(self._gate_log_scale(log_scale).to(q.dtype))  # (B, S, D)
        pe_q = gate * pe_q
        pe_k = gate * pe_k

        if self.use_scale:
            pe_q = self.alpha_q * pe_q
            pe_k = self.alpha_k * pe_k

        return q + pe_q, k + pe_k

    def apply_to_qk_and_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
        num_frames: int = 1,
    ):
        """Apply Plücker PE to Q/K, optionally compute frame-uniform cam_residual.

        Args:
            q, k: (B, S, D) where S = num_frames * spatial.
            plucker_6d: (B, S, 6) raw Plücker coordinates.
            num_frames: latent frame count, for frame-level averaging of cam_residual.

        Returns:
            q, k with additive PE applied.
            cam_residual: (B, S, D) frame-uniform residual, or None if disabled.
        """
        orig_dtype = q.dtype
        feat_q, feat_k, log_scale = self.decompose_plucker(plucker_6d)
        feat_q = feat_q.to(orig_dtype)
        feat_k = feat_k.to(orig_dtype)

        pe_q = self.norm_pe_q(self.eq(feat_q))
        pe_k = self.norm_pe_k(self.ek(feat_k))

        gate = self.scale_gate(self._gate_log_scale(log_scale).to(orig_dtype))
        pe_q = gate * pe_q
        pe_k = gate * pe_k

        if self.use_scale:
            pe_q = self.alpha_q * pe_q
            pe_k = self.alpha_k * pe_k

        q = q + pe_q
        k = k + pe_k

        cam_residual = None
        if self.enable_cam_residual:
            B, S, C = feat_q.shape
            spatial = S // num_frames
            feat_frame = (
                feat_q.reshape(B, num_frames, spatial, C)
                .mean(dim=2, keepdim=True)
                .expand(B, num_frames, spatial, C)
                .reshape(B, S, C)
            )
            cam_gate = torch.sigmoid(self.gate_logit + self.gate_proj(feat_frame))
            cam_residual = cam_gate.to(orig_dtype) * self.ev(feat_frame)

        return q.to(orig_dtype), k.to(orig_dtype), cam_residual
