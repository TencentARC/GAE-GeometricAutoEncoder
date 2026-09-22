"""
Plücker-conditioned attention used by the DDT head.

PluckerAttention — a NormAttention variant with two additions:

  1. **ref_key_scale**: per-head learnable scale applied to K vectors of ref
     views before attention.  Forces the model to attend more strongly to clean
     reference tokens.
  2. The forward signature accepts `cond_num` so the module knows which views
     are reference (first `cond_num` views) vs target (the rest).

Everything else (Plücker Flip PE, RoPE, xformers, PAG, gradient-checkpoint)
follows the base NormAttention.

Shape contract:
    x:           (BV, N, C)
    plucker_6d:  (B, V*N_spatial, 6)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from einops import rearrange

from .model_utils import RMSNorm
from .camera import PluckerFlipPE_V13

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except Exception:
    xops = None  # type: ignore[assignment]
    XFORMERS_AVAILABLE = False

# FlashAttention v2 — preferred on Hopper (H100/H200). For head_dim=48 it is
# ~1.16× faster than xformers' memory_efficient_attention (incl. permute
# overhead). See docs/notes/2026-05-26-raev2-analysis.md.
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class PluckerAttention(nn.Module):
    """Cross-view attention with Plücker PE + learnable ref-K scale."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
        # Plücker Flip PE
        plucker_pe_dim: Optional[int] = None,
        plucker_init: str = "zero",
        plucker_init_scale: float = 0.01,
        plucker_mlp_hidden: int = 64,
        plucker_scale: float = 1.0,
        scale_gate_hidden: int = 0,
        enable_cam_residual: bool = False,
        plucker_pe_checkpoint: bool = True,
        log_scale_aug_prob: float = 0.0,
        log_scale_aug_range: tuple[float, float] = (-1.2, 1.6),
        # baseline GLD: ProPE in attention instead of Plücker Flip PE
        use_baseline_prope: bool = False,
        # ref-K amplification (Helios-style)
        ref_key_scale: bool = True,
        ref_key_max_scale: float = 10.0,
        ref_key_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.plucker_pe_checkpoint = plucker_pe_checkpoint
        self.use_baseline_prope = use_baseline_prope

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Plücker Flip PE (skipped when use_baseline_prope)
        self.plucker_pe = None
        if not use_baseline_prope:
            pe_dim = plucker_pe_dim if plucker_pe_dim is not None else dim
            assert pe_dim == dim
            self.plucker_pe = PluckerFlipPE_V13(
                dim=dim,
                plucker_init=plucker_init,
                plucker_init_scale=plucker_init_scale,
                plucker_mlp_hidden=plucker_mlp_hidden,
                plucker_scale=plucker_scale,
                enable_cam_residual=enable_cam_residual,
                scale_gate_hidden=scale_gate_hidden,
                log_scale_aug_prob=log_scale_aug_prob,
                log_scale_aug_range=log_scale_aug_range,
            )

        # Helios-style per-head ref-K scale: sigmoid maps raw param to [1, max].
        # ref_key_init_bias shifts the raw init:
        #   bias= 0.0 → gate_init = 1 + 0.5*(max-1)     (e.g. 5.5 if max=10, 2.0 if max=3)
        #   bias=-4.0 → gate_init ≈ 1 + 0.018*(max-1)  (essentially no amplification)
        # -4.0 strongly recommended when max > 3 to avoid the checkerboard
        # artefact caused by aggressive ref-K gating early in training.
        self.use_ref_key_scale = ref_key_scale
        self.ref_key_max_scale = ref_key_max_scale
        if ref_key_scale:
            self.ref_key_scale_param = nn.Parameter(
                torch.full((num_heads,), float(ref_key_init_bias))
            )

    def _get_ref_key_scale(self) -> torch.Tensor:
        return 1.0 + torch.sigmoid(self.ref_key_scale_param) * (
            self.ref_key_max_scale - 1.0
        )

    # ── Plücker PE helper (identical to V2) ──────────────────────────────

    def _apply_plucker_pe(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        plucker_6d: torch.Tensor,
        V: int,
        N: int,
        num_prefix_tokens: int,
    ):
        H = self.num_heads
        D = self.head_dim
        BV = q.shape[0]
        B = BV // V
        K = num_prefix_tokens
        N_spatial = N - K

        plucker_6d = plucker_6d.to(q.dtype)
        if plucker_6d.shape[1] != V * N_spatial:
            raise ValueError(
                f"plucker_6d length {plucker_6d.shape[1]} != V*N_spatial = "
                f"{V}*{N_spatial} = {V * N_spatial}"
            )

        q_bvnhd = rearrange(q, "(b v) h n d -> b v n (h d)", v=V)
        k_bvnhd = rearrange(k, "(b v) h n d -> b v n (h d)", v=V)

        if K > 0:
            q_prefix = q_bvnhd[:, :, :K]
            k_prefix = k_bvnhd[:, :, :K]
            q_spatial = q_bvnhd[:, :, K:]
            k_spatial = k_bvnhd[:, :, K:]
        else:
            q_prefix = k_prefix = None
            q_spatial = q_bvnhd
            k_spatial = k_bvnhd

        q_flat = rearrange(q_spatial, "b v n d -> b (v n) d")
        k_flat = rearrange(k_spatial, "b v n d -> b (v n) d")
        q_flat, k_flat = self.plucker_pe.apply_to_qk(q_flat, k_flat, plucker_6d)

        q_spatial = rearrange(q_flat, "b (v n) d -> b v n d", v=V)
        k_spatial = rearrange(k_flat, "b (v n) d -> b v n d", v=V)

        if K > 0:
            q_bvnhd = torch.cat([q_prefix, q_spatial], dim=2)
            k_bvnhd = torch.cat([k_prefix, k_spatial], dim=2)
        else:
            q_bvnhd = q_spatial
            k_bvnhd = k_spatial

        q = rearrange(q_bvnhd, "b v n (h d) -> (b v) h n d", h=H, d=D)
        k = rearrange(k_bvnhd, "b v n (h d) -> (b v) h n d", h=H, d=D)
        return q, k

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        total_view: int,
        rope=None,
        plucker_6d: Optional[torch.Tensor] = None,
        cond_num: int = 0,
        pag_mode: bool = False,
        num_prefix_tokens: int = 0,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        prope_image_size=None,
        patches_layout=None,
        **_,
    ) -> torch.Tensor:
        """
        Args:
            x: (BV, N, C).
            total_view: V.
            rope: optional VisionRotaryEmbeddingFast.
            plucker_6d: (B, V*N_spatial, 6); None → skip PE.
            cond_num: number of ref views (first cond_num in V dim).
            pag_mode: if True, identity attention.
            num_prefix_tokens: K prefix tokens per view that skip RoPE+PE.
        """
        BV, N, C = x.shape
        V = total_view
        assert BV % V == 0
        B = BV // V
        KP = num_prefix_tokens

        qkv = (
            self.qkv(x)
            .reshape(BV, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # 2D RoPE
        if rope is not None:
            if KP > 0:
                q_prefix, q_spatial = q[:, :, :KP], q[:, :, KP:]
                k_prefix, k_spatial = k[:, :, :KP], k[:, :, KP:]
                q_spatial = rope(q_spatial)
                k_spatial = rope(k_spatial)
                q = torch.cat([q_prefix, q_spatial], dim=2)
                k = torch.cat([k_prefix, k_spatial], dim=2)
            else:
                q = rope(q)
                k = rope(k)

        # Plücker Flip PE (per-token; disabled when use_baseline_prope)
        if plucker_6d is not None and self.plucker_pe is not None:
            if self.plucker_pe_checkpoint and self.training and torch.is_grad_enabled():
                q, k = ckpt.checkpoint(
                    self._apply_plucker_pe,
                    q, k, plucker_6d, V, N, KP,
                    use_reentrant=False,
                )
            else:
                q, k = self._apply_plucker_pe(q, k, plucker_6d, V, N, KP)

        # 3D-inflated cross-view: stack V views into seq dim.
        q = rearrange(q, "(b v) h n c -> b h (v n) c", v=V)
        k = rearrange(k, "(b v) h n c -> b h (v n) c", v=V)
        v = rearrange(v, "(b v) h n c -> b h (v n) c", v=V)

        # ── ref-K scale (out-of-place to avoid autograd in-place error) ──
        if self.use_ref_key_scale and cond_num > 0:
            scale = self._get_ref_key_scale()  # (num_heads,)
            ref_token_end = cond_num * N
            k_ref = k[:, :, :ref_token_end] * scale.view(1, -1, 1, 1)
            k = torch.cat([k_ref, k[:, :, ref_token_end:]], dim=2)

        # ── attention ────────────────────────────────────────────────
        if pag_mode:
            x_attn = v
        elif self.use_baseline_prope and viewmats is not None and Ks is not None:
            from .prope import prope_dot_product_attention

            if isinstance(prope_image_size, (tuple, list)):
                image_h, image_w = int(prope_image_size[0]), int(prope_image_size[1])
            else:
                size = int(prope_image_size or 252)
                image_h = image_w = size
            if patches_layout is not None:
                patches_h, patches_w = int(patches_layout[0]), int(patches_layout[1])
            else:
                patches_h = patches_w = int((N - KP) ** 0.5)
            x_attn = prope_dot_product_attention(
                q, k, v,
                viewmats=viewmats,
                Ks=Ks.to(q.dtype) if Ks.dtype != q.dtype else Ks,
                patches_x=patches_w,
                patches_y=patches_h,
                image_width=image_w,
                image_height=image_h,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        elif self.fused_attn:
            q = q.to(v.dtype)
            k = k.to(v.dtype)
            orig_dtype = q.dtype

            # Preferred fast path on Hopper: FlashAttention v2.
            # Layout: PyTorch SDPA = (B, H, N, D); FA expects (B, N, H, D).
            if FLASH_ATTN_AVAILABLE and orig_dtype in (torch.bfloat16, torch.float16):
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()
                x_attn = flash_attn_func(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
                x_attn = x_attn.permute(0, 2, 1, 3)
            elif XFORMERS_AVAILABLE:
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                if orig_dtype == torch.float32:
                    q = q.to(torch.bfloat16)
                    k = k.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)

                x_attn = xops.memory_efficient_attention(
                    q, k, v,
                    p=self.attn_drop.p if self.training else 0.0,
                )
                if x_attn.dtype != orig_dtype:
                    x_attn = x_attn.to(orig_dtype)
                x_attn = x_attn.permute(0, 2, 1, 3)
            else:
                x_attn = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v

        x_attn = x_attn.transpose(1, 2).reshape(B, V * N, C)
        x_attn = rearrange(x_attn, "b (v n) c -> (b v) n c", v=V)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        return x_attn
