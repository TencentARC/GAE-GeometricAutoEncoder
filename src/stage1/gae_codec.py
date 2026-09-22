"""GAE codec (Stage 1) — Geometry-Native Autoencoder.

Compresses the four-level frozen DA3 feature hierarchy into one compact per-view
spatial latent, and reconstructs all four levels so the frozen DA3 DPT head can
still read geometry from them. A separate learned RGB head renders appearance
from the same latent.

Architecture:
    Encoder: Conv stack (6144→bottleneck) + spatial self-attention → (mu, logvar)
    Decoder: latent → spatial self-attention (shared trunk) →
               ├── Conv stack → 6144d features  (frozen DA3 DPT head → depth/ray/pointmap)
               └── RGBHead (MAE-style transformer decoder) → RGB pixels

Corresponds to paper Section 3.1; equations are cited at each method below.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

NUM_DA3_LEVELS = 4
DA3_LEVEL_DIM = 1536
DA3_TOTAL_DIM = NUM_DA3_LEVELS * DA3_LEVEL_DIM  # 6144
DINOV2_PATCH = 14




def _tokens_to_hw(n: int) -> Tuple[int, int]:
    for h_cand in (36, 32, 28, 24, 20, 16):
        if n % h_cand == 0:
            return h_cand, n // h_cand
    h = int(math.isqrt(n))
    while h > 0 and n % h != 0:
        h -= 1
    return h, n // h


def _sincos_pos_embed_2d(h: int, w: int, dim: int, dtype=torch.float32) -> torch.Tensor:
    assert dim % 4 == 0
    half = dim // 2
    gy = torch.arange(h, dtype=dtype).unsqueeze(1).expand(h, w).reshape(-1)
    gx = torch.arange(w, dtype=dtype).unsqueeze(0).expand(h, w).reshape(-1)
    omega = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=dtype) / half))
    pe = torch.zeros(h * w, dim, dtype=dtype)
    pe[:, 0::4] = (gy.unsqueeze(1) * omega.unsqueeze(0)).sin()
    pe[:, 1::4] = (gy.unsqueeze(1) * omega.unsqueeze(0)).cos()
    pe[:, 2::4] = (gx.unsqueeze(1) * omega.unsqueeze(0)).sin()
    pe[:, 3::4] = (gx.unsqueeze(1) * omega.unsqueeze(0)).cos()
    return pe


def _sincos_pos_embed_1d(length: int, dim: int, dtype=torch.float32) -> torch.Tensor:
    """1D sin-cos PE over ordinal slot index (frame position within the window).

    Slot-based on purpose: the temporal decoder only needs visual adjacency
    ("these frames are neighbors, share texture"), not physical Δt. This makes
    it robust to variable frame strides — train with mixed intervals and the
    decoder learns to borrow more from similar neighbors, less from distant ones.
    """
    assert dim % 2 == 0
    pos = torch.arange(length, dtype=dtype).unsqueeze(1)
    omega = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    pe = torch.zeros(length, dim, dtype=dtype)
    pe[:, 0::2] = (pos * omega).sin()
    pe[:, 1::2] = (pos * omega).cos()
    return pe


# ── Building blocks ──────────────────────────────────────────────────────

class ResBlock1x1(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class SelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN."""

    def __init__(self, dim: int, num_heads: int = 8, ffn_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAttentionBlock(nn.Module):
    """Factorized temporal attention over the view axis at each spatial token.

    Input/output keep the per-frame layout (BV, 1+N, dim). Internally the patch
    tokens (CLS excluded) are regrouped to (B*N, V, dim) so attention runs across
    the V neighboring frames at a fixed spatial location, then scattered back.
    This is the standard cheap factorization (spatial N² + temporal V²) used by
    video-VAE decoders to kill high-frequency texture flicker.

    `gated=True` (default, warm-start): zero-init residual gates (`gate_attn`,
    `gate_ffn`) make the block an exact identity at step 0, so a per-frame
    RGBHead checkpoint can be warm-started without disturbing its output; the
    temporal path is learned from zero.

    `gated=False` (from-scratch): the gates are dropped and the block is a plain
    pre-norm residual (like the spatial blocks). There is no per-frame ckpt to
    protect, so the temporal path contributes and receives full gradient from
    step 0 instead of ramping up through a single scalar — the zero-gate design
    is redundant (and slows convergence) when the whole head trains from scratch.
    """

    def __init__(self, dim: int, num_heads: int = 8, ffn_ratio: float = 4.0,
                 max_views: int = 64, gated: bool = True):
        super().__init__()
        self.gated = gated
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate_attn = nn.Parameter(torch.zeros(1)) if gated else None
        # FFN is optional: temporal mixing is mostly an attention job, so
        # ffn_ratio<=0 drops the (dominant) FFN params for a much lighter block.
        self.use_ffn = ffn_ratio is not None and ffn_ratio > 0
        if self.use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            ffn_dim = int(dim * ffn_ratio)
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, dim),
            )
            self.gate_ffn = nn.Parameter(torch.zeros(1)) if gated else None
        self.register_buffer(
            "temporal_pe", _sincos_pos_embed_1d(max_views, dim), persistent=False)

    def forward(self, x: torch.Tensor, num_views: int) -> torch.Tensor:
        bv, t, d = x.shape
        v = num_views
        b = bv // v
        cls, pat = x[:, :1, :], x[:, 1:, :]
        n = pat.shape[1]
        # (BV, N, d) → (B, V, N, d) → (B, N, V, d) → (B*N, V, d)
        pat = pat.reshape(b, v, n, d).permute(0, 2, 1, 3).reshape(b * n, v, d)
        pe = self.temporal_pe[:v].to(pat.dtype).unsqueeze(0)
        h = self.norm1(pat) + pe
        h = self.attn(h, h, h, need_weights=False)[0]
        pat = pat + (self.gate_attn * h if self.gated else h)
        if self.use_ffn:
            f = self.ffn(self.norm2(pat))
            pat = pat + (self.gate_ffn * f if self.gated else f)
        # (B*N, V, d) → (B, N, V, d) → (B, V, N, d) → (BV, N, d)
        pat = pat.reshape(b, n, v, d).permute(0, 2, 1, 3).reshape(bv, n, d)
        return torch.cat([cls, pat], dim=1)


# ── Rotary position embedding (RoPE) for the divided space-time RGB head ──
# Interleaved-pair convention (matches src/stage1/raev2/encoders/models/rope.py):
# adjacent channels (2k, 2k+1) form one rotation pair sharing frequency f_k, so
# attention logits depend only on the *relative* offset between two positions.


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    lead = x.shape[:-1]
    x = x.reshape(*lead, d // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(*lead, d)


def _rope_angles(positions: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Per-position rotation angles, shape (L, dim). `dim` must be even.

    `positions` are ordinal indices (float). Angles are interleave-repeated so
    each adjacent channel pair used by `_rope_rotate_half` shares one frequency.
    """
    device = positions.device
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    ang = positions.float()[:, None] * inv_freq[None, :]      # (L, dim/2)
    return ang.repeat_interleave(2, dim=-1)                    # (L, dim)


def _rope_angles_2d(hp: int, wp: int, head_dim: int, device, n_prefix: int = 1,
                    theta: float = 10000.0) -> torch.Tensor:
    """Axial 2D-RoPE angles for a hp×wp grid, shape (n_prefix+hp*wp, head_dim).

    `head_dim` splits into a y-half and an x-half, each rotated by its own axis
    index (so the encoding is relative in both spatial dims and transfers across
    resolutions). The `n_prefix` leading rows (CLS) get zero angle → no rotation.
    """
    assert head_dim % 4 == 0, "2D RoPE needs head_dim divisible by 4"
    half = head_dim // 2
    gy = torch.arange(hp, device=device).repeat_interleave(wp).float()   # (N,)
    gx = torch.arange(wp, device=device).repeat(hp).float()              # (N,)
    ang = torch.cat([_rope_angles(gy, half, theta),
                     _rope_angles(gx, half, theta)], dim=-1)             # (N, head_dim)
    if n_prefix:
        ang = torch.cat([ang.new_zeros(n_prefix, head_dim), ang], dim=0)
    return ang


def _apply_rope(t: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rotate `t` (..., L, head_dim) by angles `ang` (L, head_dim), done in fp32."""
    dt = t.dtype
    tf = t.float()
    out = tf * ang.cos() + _rope_rotate_half(tf) * ang.sin()
    return out.to(dt)


class _RoPESelfAttention(nn.Module):
    """Multi-head self-attention (fused SDPA) with optional rotary PE on Q/K."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, ang: Optional[torch.Tensor] = None) -> torch.Tensor:
        bg, seq, _ = x.shape
        qkv = self.qkv(x).reshape(bg, seq, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, bg, H, L, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if ang is not None:
            q = _apply_rope(q, ang)
            k = _apply_rope(k, ang)
        out = F.scaled_dot_product_attention(q, k, v)  # (bg, H, L, hd)
        out = out.transpose(1, 2).reshape(bg, seq, self.num_heads * self.head_dim)
        return self.proj(out)


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer-style divided space-time block for the RGB decoder.

    Order (each pre-norm + residual): temporal-attn → spatial-attn → FFN.
      * temporal-attn: at each spatial patch, attends across the V frames (CLS
        excluded), with ordinal temporal RoPE — relative and length-extrapolating
        (transfers across frame counts) and, being ordinal, invariant to the
        sampling stride.
      * spatial-attn: within each frame, attends across CLS + N patch tokens with
        axial 2D-RoPE — relative, so it transfers across resolutions.
      * FFN: shared channel mixing after both attentions — this is what fills the
        non-linearity gap an attention-only temporal block would leave before the
        final `pred`.

    No zero-gates: every sublayer is a plain residual that contributes from step
    0, because this head is meant to be trained from scratch.
    """

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0,
                 temporal_num_heads: Optional[int] = None):
        super().__init__()
        t_heads = temporal_num_heads if temporal_num_heads is not None else num_heads
        self.t_norm = nn.LayerNorm(dim)
        self.t_attn = _RoPESelfAttention(dim, t_heads)
        self.s_norm = nn.LayerNorm(dim)
        self.s_attn = _RoPESelfAttention(dim, num_heads)
        self.m_norm = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor, num_views: Optional[int],
                spatial_ang: torch.Tensor,
                temporal_ang: Optional[torch.Tensor]) -> torch.Tensor:
        bv, _, d = x.shape
        v = num_views if (num_views and num_views > 0 and bv % num_views == 0) else 1
        b = bv // v
        cls, pat = x[:, :1, :], x[:, 1:, :]
        n = pat.shape[1]
        # temporal: (BV, N, d) → (B*N, V, d), attend across frames, scatter back.
        # temporal_ang is None at V=1 (per-frame decoding), where the length-1
        # attention keeps the temporal params in the graph but adds no mixing.
        p = pat.reshape(b, v, n, d).permute(0, 2, 1, 3).reshape(b * n, v, d)
        p = p + self.t_attn(self.t_norm(p), temporal_ang if v > 1 else None)
        pat = p.reshape(b, n, v, d).permute(0, 2, 1, 3).reshape(bv, n, d)
        x = torch.cat([cls, pat], dim=1)
        # spatial: within-frame attention over CLS + N patches
        x = x + self.s_attn(self.s_norm(x), spatial_ang)
        # FFN
        x = x + self.mlp(self.m_norm(x))
        return x


class RGBHead(nn.Module):
    """MAE-style transformer decoder: shared trunk features → RGB pixels.

    Architecture follows GeneralDecoder_Variable:
        trunk features (BV, N, trunk_dim)
        → Linear proj to decoder_dim
        + trainable CLS token + 2D sincos positional encoding
        → depth × [LayerNorm → MHSA → FFN] transformer blocks
        → LayerNorm → Linear → patch²×3
        → strip CLS → unpatchify → (BV, 3, H_px, W_px)

    Args:
        trunk_dim: Input dim from shared decoder trunk (bottleneck_dim).
        hidden_dim: Transformer hidden size.
        depth: Number of transformer layers.
        num_heads: Attention heads.
        ffn_ratio: FFN expansion ratio.
        patch_size: Pixel patch size for unpatchify (must match DINOv2).
    """

    def __init__(
        self,
        trunk_dim: int = 512,
        hidden_dim: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        patch_size: int = DINOV2_PATCH,
        temporal: bool = False,
        temporal_num_heads: Optional[int] = None,
        temporal_max_views: int = 64,
        temporal_every: int = 1,
        temporal_ffn_ratio: Optional[float] = None,
        temporal_gated: bool = True,
        temporal_mode: str = "interleaved",
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.temporal = temporal
        self.temporal_mode = temporal_mode
        self.num_heads = num_heads
        self.temporal_num_heads = (
            temporal_num_heads if temporal_num_heads is not None else num_heads)
        self.rope_theta = rope_theta

        self.proj_in = nn.Linear(trunk_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # `temporal_mode` selects how time is modelled:
        #   * "interleaved": per-frame spatial blocks with sparse temporal-attn
        #     blocks inserted between them (sincos PE; optional zero-gates for
        #     warm-start). This is the default / backward-compatible path.
        #   * "divided": every layer is a TimeSformer-style divided space-time
        #     block (temporal-attn → spatial-attn → FFN) with rotary PE — spatial
        #     2D-RoPE (resolution-extrapolating) and ordinal temporal RoPE
        #     (frame-count-extrapolating, stride-invariant). Un-gated; meant to
        #     train the whole video RGB decoder from scratch.
        self.temporal_blocks = None
        if temporal_mode == "divided":
            if not temporal:
                raise ValueError("temporal_mode='divided' requires temporal=True")
            if (hidden_dim // num_heads) % 4 != 0 or (hidden_dim // self.temporal_num_heads) % 2 != 0:
                raise ValueError(
                    "divided mode needs head_dim%4==0 (spatial 2D-RoPE) and "
                    "temporal head_dim%2==0 (temporal RoPE)")
            self.blocks = nn.ModuleList([
                DividedSpaceTimeBlock(hidden_dim, num_heads, ffn_ratio,
                                      temporal_num_heads=self.temporal_num_heads)
                for _ in range(depth)
            ])
        elif temporal_mode == "interleaved":
            self.blocks = nn.ModuleList([
                SelfAttentionBlock(hidden_dim, num_heads, ffn_ratio)
                for _ in range(depth)
            ])
            # Temporal blocks are interleaved *sparsely*: one after every
            # `temporal_every` spatial blocks (always including the last block so
            # the final output is temporally mixed). Keyed by spatial-block index
            # in a ModuleDict. `temporal_gated=True` (default) zero-gates them at
            # init (identity) → safe on top of a per-frame ckpt;
            # `temporal_gated=False` makes them plain residuals for from-scratch.
            # `temporal_ffn_ratio<=0` drops the FFN for a lean block.
            if temporal:
                t_heads = self.temporal_num_heads
                t_ffn = temporal_ffn_ratio if temporal_ffn_ratio is not None else ffn_ratio
                every = max(1, int(temporal_every))
                positions = [i for i in range(depth) if (i + 1) % every == 0]
                self.temporal_blocks = nn.ModuleDict({
                    str(i): TemporalAttentionBlock(hidden_dim, t_heads, t_ffn,
                                                   max_views=temporal_max_views,
                                                   gated=temporal_gated)
                    for i in positions
                })
        else:
            raise ValueError(
                f"temporal_mode must be 'interleaved' or 'divided', got {temporal_mode!r}")
        self.norm = nn.LayerNorm(hidden_dim)
        self.pred = nn.Linear(hidden_dim, patch_size ** 2 * 3)

        self._pos_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def _get_pos(self, hp: int, wp: int, device, dtype) -> torch.Tensor:
        key = (hp, wp)
        if key not in self._pos_cache or self._pos_cache[key].device != device:
            patch_pe = _sincos_pos_embed_2d(hp, wp, self.hidden_dim, dtype=torch.float32)
            cls_pe = torch.zeros(1, self.hidden_dim)
            full = torch.cat([cls_pe, patch_pe], dim=0).to(device=device, dtype=dtype)
            self._pos_cache[key] = full
        return self._pos_cache[key]

    def forward(self, seq: torch.Tensor, hp: int, wp: int,
                num_views: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            seq: (BV, N, trunk_dim) from shared decoder attention trunk
            hp, wp: patch grid dimensions
            num_views: V (frames per scene). Required to enable temporal
                attention. If None, the head degrades to per-frame decoding
                (e.g. the latent-diffusion stage). When given, temporal blocks
                always run — even at V=1 (a length-1 attention, mathematically a
                no-op but keeping all temporal params in the graph so DDP without
                `find_unused_parameters` stays happy under dynamic views).
        Returns:
            (BV, 3, hp*patch_size, wp*patch_size) RGB prediction
        """
        bv = seq.shape[0]
        x = self.proj_in(seq)
        cls = self.cls_token.expand(bv, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.temporal_mode == "divided":
            return self._forward_divided(x, hp, wp, num_views)

        # interleaved: absolute 2D sincos PE added once at the input
        x = x + self._get_pos(hp, wp, x.device, x.dtype).unsqueeze(0)
        do_temporal = (self.temporal_blocks is not None and num_views is not None)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if do_temporal and str(i) in self.temporal_blocks:
                x = self.temporal_blocks[str(i)](x, num_views)

        x = self.norm(x)
        patches = self.pred(x[:, 1:, :])  # strip CLS
        return self._unpatchify(patches, hp, wp)

    def _forward_divided(self, x: torch.Tensor, hp: int, wp: int,
                         num_views: Optional[int]) -> torch.Tensor:
        # No absolute PE: position enters through RoPE inside each block.
        v = num_views if (num_views and num_views > 0 and x.shape[0] % num_views == 0) else 1
        spatial_ang = _rope_angles_2d(
            hp, wp, self.hidden_dim // self.num_heads, x.device,
            n_prefix=1, theta=self.rope_theta)
        temporal_ang = None
        if v > 1:
            positions = torch.arange(v, device=x.device, dtype=torch.float32)
            temporal_ang = _rope_angles(
                positions, self.hidden_dim // self.temporal_num_heads, theta=self.rope_theta)
        for blk in self.blocks:
            x = blk(x, num_views, spatial_ang, temporal_ang)
        x = self.norm(x)
        patches = self.pred(x[:, 1:, :])  # strip CLS
        return self._unpatchify(patches, hp, wp)

    def _unpatchify(self, patches: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        p = self.patch_size
        bv = patches.shape[0]
        x = patches.reshape(bv, hp, wp, p, p, 3)
        return torch.einsum("bhwpqc->bchpwq", x).reshape(bv, 3, hp * p, wp * p)


class _TemporalGeoAdapterBlock(nn.Module):
    """One residual spatio-temporal block at ``dim`` channels."""

    def __init__(self, dim, hidden_ratio, num_heads, max_views, use_spatial):
        super().__init__()
        hidden = max(dim, int(dim * hidden_ratio))
        if use_spatial:
            self.spatial_norm = nn.GroupNorm(min(32, dim), dim)
            self.spatial_dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.spatial_pw = nn.Conv2d(dim, dim, 1)
        else:
            self.spatial_norm = None
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.register_buffer(
            "temporal_pe", _sincos_pos_embed_1d(max_views, dim), persistent=False)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self._init_weights()

    def _init_weights(self):
        if self.spatial_norm is not None:
            nn.init.kaiming_normal_(self.spatial_dw.weight, nonlinearity="linear")
            nn.init.zeros_(self.spatial_dw.bias)
            nn.init.kaiming_normal_(self.spatial_pw.weight, nonlinearity="linear")
            nn.init.zeros_(self.spatial_pw.bias)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, b, v, n):
        """x: (B*V, C, h, w) at block dim. Returns same shape."""
        bv, c, h, w = x.shape
        if self.spatial_norm is not None:
            s = self.spatial_norm(x)
            s = self.spatial_pw(F.silu(self.spatial_dw(s)))
            x = x + s
        tok = x.reshape(bv, c, n).permute(0, 2, 1)
        tt = tok.reshape(b, v, n, c).permute(0, 2, 1, 3).reshape(b * n, v, c)
        h_t = self.temporal_norm(tt) + self.temporal_pe[:v].to(tt.dtype).unsqueeze(0)
        h_t = self.temporal_attn(h_t, h_t, h_t, need_weights=False)[0]
        tt = tt + h_t
        tt = tt + self.mlp(self.mlp_norm(tt))
        tok = tt.reshape(b, n, v, c).permute(0, 2, 1, 3).reshape(bv, n, c)
        return tok.permute(0, 2, 1).reshape(bv, c, h, w)


class TemporalGeoAdapter(nn.Module):
    """Zero-init residual spatio-temporal adapter on the GEOMETRY latent.

    Operates on the raw latent ``(B*V, C, h, w)`` and returns
    ``z_geo = z + gate * A(z)``. Only the geometry branch (``_decode_trunk`` ->
    ``dec_conv`` -> DPT) consumes ``z_geo``; the RGB head keeps decoding from the
    original ``z`` so refining geometry never perturbs RGB. The scalar gate is
    zero-init, so the module is an exact identity until finetuned
    (``scripts/finetune_geo_decoder.py``). Supports variable ``V``.
    """

    def __init__(self, channels, hidden_ratio=2.0, num_heads=4, max_views=64,
                 use_spatial=True, num_blocks=1, width_mult=1.0):
        super().__init__()
        self.channels = channels
        self.num_blocks = max(1, int(num_blocks))
        self.width_mult = float(width_mult)
        work_dim = max(channels, int(round(channels * self.width_mult)))
        heads = int(num_heads)
        while heads > 1 and work_dim % heads != 0:
            heads -= 1
        self.work_dim = work_dim
        self.in_proj = (nn.Identity() if work_dim == channels
                        else nn.Conv2d(channels, work_dim, 1))
        self.blocks = nn.ModuleList([
            _TemporalGeoAdapterBlock(work_dim, hidden_ratio, heads, max_views, use_spatial)
            for _ in range(self.num_blocks)
        ])
        self.proj_out = nn.Conv2d(work_dim, channels, 1)
        self.gate = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.in_proj, nn.Conv2d):
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, z, num_views):
        """z: (B*V, C, h, w). Returns residual-updated latent, same shape."""
        bv, c, h, w = z.shape
        v = int(num_views) if num_views is not None else 1
        if v <= 0 or bv % v != 0:
            v = 1
        b = bv // v
        n = h * w
        x = self.in_proj(z)
        for blk in self.blocks:
            x = blk(x, b, v, n)
        out = self.proj_out(x)
        return z + self.gate * out


def apply_rgb_latent_jitter(
    z: torch.Tensor,
    num_views: Optional[int],
    mode: str,
    sigma: float,
    *,
    relative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perturb the latent before RGB decoding (RGB-only robustness training).

    Runs INSIDE ``GAECodec.forward`` so the RGB head always stays in the DDP
    autograd graph (the project's DDP has find_unused_parameters=False).

    Args:
        z: latent laid out as contiguous ``(B*V, C, H, W)``.
        num_views: V frames per scene; <=1 (or non-divisible) falls back to iid.
        mode: ``iid`` / ``frame_diff`` / ``mixed``.
        sigma: absolute std, or a relative multiplier of the batch latent std.
        relative: scale ``sigma`` by the current batch latent std.

    Returns:
        ``(z_jittered, sigma_eff)``. ``sigma <= 0`` or ``mode="none"`` returns
        ``z`` unchanged.
    """
    mode = str(mode or "none").lower()
    if sigma <= 0 or mode in ("none", "off", "false"):
        return z, z.new_zeros(())

    v = int(num_views or 1)
    sigma_t = z.new_tensor(float(sigma))
    if relative:
        sigma_t = sigma_t * z.detach().float().std().to(dtype=z.dtype).clamp_min(1e-6)

    eps = torch.randn_like(z)
    if mode == "iid" or v <= 1 or z.shape[0] % v != 0:
        jitter = eps
    else:
        bsz = z.shape[0] // v
        eps_v = eps.view(bsz, v, *z.shape[1:])
        if mode in ("frame_diff", "mixed"):
            jitter_v = eps_v.clone()
            jitter_v[:, 1:] = eps_v[:, 1:] - eps_v[:, :-1]
            jitter_v[:, 0] = eps_v[:, 0] - eps_v[:, 1]
            fd = jitter_v.reshape_as(z)
            jitter = fd if mode == "frame_diff" else 0.5 * eps + 0.5 * fd
        else:
            raise ValueError(
                f"rgb_latent_jitter_mode must be none/iid/frame_diff/mixed, got {mode!r}"
            )

    return z + sigma_t * jitter, sigma_t.detach()


class GAECodec(nn.Module):
    """
    Feature VAE with attention bottleneck and optional RGB decoder.

    The decoder has a shared attention trunk that branches into:
    - Feature branch (Conv stack → 6144d) for depth/pose
    - RGB branch (RGBHead: MAE-style transformer decoder → pixels)

    Args:
        latent_dim: Bottleneck channel count.
        hidden_dims: Conv stack channel sizes.
        num_attn_blocks: Attention blocks in encoder/decoder shared trunk.
        attn_heads: Attention heads for shared trunk.
        kl_weight: KL weight (>0 for diffusion-ready latent).
        free_bits: Per-channel KL floor in nats (Kingma+16). 0 disables it.
        use_spatial: Add 3×3 depthwise conv in conv blocks.
        level_weights: Per-level feature loss weights [w0, w1, w2, w3].
        rgb_decoder: Optional dict → RGBHead config:
            hidden_dim (int): transformer hidden dim (default 768)
            depth (int): transformer layers (default 8)
            num_heads (int): attention heads (default 12)
            ffn_ratio (float): FFN expansion (default 4.0)
            patch_size (int): DINOv2 patch size (default 14)
        stats_dir: Path to per-level normalization stats.
    """

    def __init__(
        self,
        latent_dim: int = 384,
        hidden_dims: Optional[List[int]] = None,
        num_attn_blocks: int = 4,
        attn_heads: int = 8,
        kl_weight: float = 1e-6,
        free_bits: float = 0.0,
        use_spatial: bool = True,
        level_weights: Optional[List[float]] = None,
        rgb_decoder: Optional[dict] = None,
        stats_dir: str = "model_stats/da3",
        feature_level_dim: int = None,  # Per-level feature dim. None → DA3_LEVEL_DIM (1536)
        spatial_downsample: int = 1,    # 2 = compress latent H,W by 2x (e.g. 36->18)
        repa_proj_dim: Optional[int] = None,   # >0 → 建 REPA 投影头，输出维须等于目标特征维
        repa_proj_hidden: int = 1024,          # REPA 投影头隐藏维
        geo_adapter: Optional[dict] = None,     # >None -> zero-init geometry-branch adapter (finetune_geo_decoder)
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]
        if spatial_downsample not in (1, 2):
            raise ValueError(f"spatial_downsample must be 1 or 2, got {spatial_downsample}")
        self.spatial_downsample = int(spatial_downsample)

        # Allow dynamic feature dim (1536 for DA3-Base, 2048 for DA3-Large)
        self._level_dim = feature_level_dim if feature_level_dim is not None else DA3_LEVEL_DIM
        self._total_dim = NUM_DA3_LEVELS * self._level_dim

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.free_bits = float(free_bits)
        self.use_spatial = use_spatial
        self.num_attn_blocks = num_attn_blocks
        bottleneck_dim = hidden_dims[-1]

        if level_weights is not None:
            self.register_buffer("level_weights", torch.tensor(level_weights, dtype=torch.float32))
        else:
            self.level_weights = None

        self._load_normalization_stats(stats_dir)

        # ── Conv encoder: total_dim → bottleneck_dim ──
        enc_conv = []
        in_ch = self._total_dim
        for h_ch in hidden_dims:
            enc_conv.append(nn.Conv2d(in_ch, h_ch, 1))
            enc_conv.append(nn.GroupNorm(32, h_ch))
            enc_conv.append(nn.SiLU(inplace=True))
            if use_spatial:
                enc_conv.append(nn.Conv2d(h_ch, h_ch, 3, padding=1, groups=h_ch))
                enc_conv.append(nn.SiLU(inplace=True))
            enc_conv.append(ResBlock1x1(h_ch))
            in_ch = h_ch
        self.enc_conv = nn.Sequential(*enc_conv)

        # ── Attention encoder ──
        self.enc_attn = nn.ModuleList([
            SelfAttentionBlock(bottleneck_dim, attn_heads) for _ in range(num_attn_blocks)
        ])
        self.enc_attn_norm = nn.LayerNorm(bottleneck_dim)

        # ── Optional 2× spatial downsample (after enc attention, before mu/logvar) ──
        # Placement rationale: keep self-attention on the original (uncompressed)
        # spatial grid so all pretrained attention weights apply unchanged. The
        # compression happens only at the bottleneck, immediately before the
        # latent projections. Initialised as a 2×2 average-pool (Conv2d k=2 s=2
        # with weight = 0.25·I) → on step 0, the new layer is a no-parameter
        # 2× avg pool, so the model behaves exactly like the pretrained VAE
        # squeezed through a 4× spatial info bottleneck. The bias-free, identity-
        # like init gives the cleanest finetune starting point — see
        # `docs/notes/2026-06-01-vae-spatial-downsample-ds2.md`.
        if self.spatial_downsample == 2:
            self.enc_downsample = nn.Conv2d(bottleneck_dim, bottleneck_dim,
                                            kernel_size=2, stride=2, padding=0, bias=False)
            self.dec_upsample = nn.ConvTranspose2d(bottleneck_dim, bottleneck_dim,
                                                   kernel_size=2, stride=2, padding=0,
                                                   bias=False)
        else:
            self.enc_downsample = None
            self.dec_upsample = None

        # ── Latent projections ──
        self.mu_proj = nn.Conv2d(bottleneck_dim, latent_dim, 1)
        self.logvar_proj = nn.Conv2d(bottleneck_dim, latent_dim, 1)

        # ── Decoder: latent → shared attention trunk ──
        self.dec_proj = nn.Conv2d(latent_dim, bottleneck_dim, 1)
        self.dec_attn = nn.ModuleList([
            SelfAttentionBlock(bottleneck_dim, attn_heads) for _ in range(num_attn_blocks)
        ])
        self.dec_attn_norm = nn.LayerNorm(bottleneck_dim)

        # ── Feature branch: bottleneck_dim → 6144 ──
        dec_conv = []
        in_ch = bottleneck_dim
        for h_ch in reversed(hidden_dims[:-1]):
            dec_conv.append(nn.Conv2d(in_ch, h_ch, 1))
            dec_conv.append(nn.GroupNorm(32, h_ch))
            dec_conv.append(nn.SiLU(inplace=True))
            if use_spatial:
                dec_conv.append(nn.Conv2d(h_ch, h_ch, 3, padding=1, groups=h_ch))
                dec_conv.append(nn.SiLU(inplace=True))
            dec_conv.append(ResBlock1x1(h_ch))
            in_ch = h_ch
        dec_conv.append(nn.Conv2d(in_ch, self._total_dim, 1))
        self.dec_conv = nn.Sequential(*dec_conv)

        # ── RGB branch (optional): MAE-style transformer decoder ──
        self.rgb_head = None
        if rgb_decoder is not None:
            self.rgb_head = RGBHead(
                trunk_dim=bottleneck_dim,
                hidden_dim=rgb_decoder.get("hidden_dim", 768),
                depth=rgb_decoder.get("depth", 8),
                num_heads=rgb_decoder.get("num_heads", 12),
                ffn_ratio=rgb_decoder.get("ffn_ratio", 4.0),
                patch_size=rgb_decoder.get("patch_size", DINOV2_PATCH),
                temporal=rgb_decoder.get("temporal", False),
                temporal_num_heads=rgb_decoder.get("temporal_num_heads", None),
                temporal_max_views=rgb_decoder.get("temporal_max_views", 64),
                temporal_every=rgb_decoder.get("temporal_every", 1),
                temporal_ffn_ratio=rgb_decoder.get("temporal_ffn_ratio", None),
                temporal_gated=rgb_decoder.get("temporal_gated", True),
                temporal_mode=rgb_decoder.get("temporal_mode", "interleaved"),
                rope_theta=rgb_decoder.get("rope_theta", 10000.0),
            )

        # ── REPA 投影头（可选）：latent → 目标 encoder 特征维，逐 token MLP ──
        # 推理（precompute / diffusion）时不使用，latent 不受影响；仅训练期参与
        # self-REPA 对齐损失。新参数，load 预训练 ckpt 时 strict=False 随机初始化。
        self.repa_proj_dim = repa_proj_dim
        self.repa_proj = None
        if repa_proj_dim is not None and repa_proj_dim > 0:
            self.repa_proj = nn.Sequential(
                nn.Linear(latent_dim, repa_proj_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(repa_proj_hidden, repa_proj_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(repa_proj_hidden, repa_proj_dim),
            )

        # Optional geometry-branch adapter. Absent from base ckpts -> loaded
        # with strict=False; identity at init so enabling it never changes
        # existing geometry until finetuned.
        self.geo_adapter = None
        if geo_adapter is not None:
            self.geo_adapter = TemporalGeoAdapter(
                channels=latent_dim,
                hidden_ratio=geo_adapter.get("hidden_ratio", 2.0),
                num_heads=geo_adapter.get("num_heads", 4),
                max_views=geo_adapter.get("max_views", 64),
                use_spatial=geo_adapter.get("use_spatial", True),
                num_blocks=geo_adapter.get("num_blocks", 1),
                width_mult=geo_adapter.get("width_mult", 1.0),
            )

        self._pos_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._init_weights()

    def repa_project(self, mu: torch.Tensor) -> torch.Tensor:
        """latent mu (B, C, h, w) → 投影 token (B, h*w, repa_proj_dim)。"""
        assert self.repa_proj is not None, "repa_proj 未配置（repa_proj_dim 须 >0）"
        b, c, h, w = mu.shape
        z = mu.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.repa_proj(z)

    def _init_weights(self):
        for m in (list(self.enc_conv.modules()) + list(self.dec_conv.modules())
                  + [self.mu_proj, self.logvar_proj, self.dec_proj]):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in (list(self.enc_attn.modules()) + list(self.dec_attn.modules())
                  + [self.enc_attn_norm, self.dec_attn_norm]):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.dec_conv[-1].weight)
        nn.init.zeros_(self.dec_conv[-1].bias)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)
        # Identity-pool init for optional 2x spatial down/up (see __init__ docstring):
        # enc_downsample = per-channel 2x2 avg-pool (0.25*I), dec_upsample = per-channel
        # nearest upsample (1.0*I) -> enc->dec round-trip is a pure low-pass at init.
        if self.spatial_downsample == 2:
            C = self.enc_downsample.weight.shape[0]
            with torch.no_grad():
                self.enc_downsample.weight.zero_()
                eye = torch.eye(C).view(C, C, 1, 1)
                self.enc_downsample.weight.copy_(eye.expand(C, C, 2, 2) * 0.25)
                # ConvTranspose2d weight is (C_in, C_out, kH, kW).
                self.dec_upsample.weight.zero_()
                self.dec_upsample.weight.copy_(eye.expand(C, C, 2, 2))

    # ── Normalization stats ──

    def _load_normalization_stats(self, stats_dir: str):
        means, stds = [], []
        for lvl in range(NUM_DA3_LEVELS):
            s = torch.load(f"{stats_dir}/normalization_stats_level{lvl}.pt",
                           map_location="cpu", weights_only=True)
            means.append(s["mean"].squeeze())
            stds.append(s["std"].squeeze())
        self.register_buffer("feat_mean", torch.cat(means).reshape(1, -1, 1, 1))
        self.register_buffer("feat_std", torch.cat(stds).reshape(1, -1, 1, 1))

    def normalize_levels(self, feats: Dict[int, torch.Tensor],
                         image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Fuse the four DA3 levels into one tensor — paper Eq. (5).

        Level-wise per-channel normalization (fixed training-set statistics)
        prevents high-variance channels from dominating the concatenation.
        """
        bv, n, c = feats[0].shape
        if image_size is not None:
            h, w = image_size[0] // DINOV2_PATCH, image_size[1] // DINOV2_PATCH
        else:
            h, w = _tokens_to_hw(n)
        parts = [feats[lvl].transpose(1, 2).reshape(bv, self._level_dim, h, w)
                 for lvl in range(NUM_DA3_LEVELS)]
        x = torch.cat(parts, dim=1)
        return (x - self.feat_mean) / (self.feat_std + 1e-5)

    def denormalize_and_split(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x = x * (self.feat_std + 1e-5) + self.feat_mean
        chunks = x.split(self._level_dim, dim=1)
        return {lvl: ch.reshape(ch.shape[0], ch.shape[1], -1).transpose(1, 2)
                for lvl, ch in enumerate(chunks)}

    # ── Positional encoding ──

    def _get_pos(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w)
        if key not in self._pos_cache or self._pos_cache[key].device != device:
            dim = self.enc_attn[0].norm1.normalized_shape[0]
            self._pos_cache[key] = _sincos_pos_embed_2d(h, w, dim, dtype=dtype).to(device)
        return self._pos_cache[key]

    # ── Encode / Decode ──

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bv, _, h, w = x.shape
        h_conv = self.enc_conv(x)
        c = h_conv.shape[1]
        seq = h_conv.reshape(bv, c, h * w).permute(0, 2, 1)
        seq = seq + self._get_pos(h, w, seq.device, seq.dtype).unsqueeze(0)
        for blk in self.enc_attn:
            seq = blk(seq)
        seq = self.enc_attn_norm(seq)
        feat = seq.permute(0, 2, 1).reshape(bv, c, h, w)
        # Optional 2× spatial compression before latent projection (e.g. 36→18).
        # CONSTRAINT: h and w must be even (the k=2 s=2 round-trip is only invertible
        # for even dims), i.e. image H, W must be multiples of DINOV2_PATCH*2 = 28.
        if self.enc_downsample is not None:
            if h % 2 or w % 2:
                raise ValueError(
                    f"spatial_downsample=2 requires even latent (h, w); got ({h}, {w}). "
                    f"Image dims must be multiples of {DINOV2_PATCH * 2} = 28.")
            feat = self.enc_downsample(feat)
        return self.mu_proj(feat), self.logvar_proj(feat)

    def _decode_trunk(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Shared decoder attention trunk.
        Returns: (sequence (BV, N, C), h_patches, w_patches)
        """
        bv, _, h, w = z.shape
        feat = self.dec_proj(z)
        # Mirror the encode-side 2× compression: upsample back to the full grid so
        # dec_attn runs at the original resolution (e.g. 18→36).
        if self.dec_upsample is not None:
            feat = self.dec_upsample(feat)
            h, w = feat.shape[-2], feat.shape[-1]
        c = feat.shape[1]
        seq = feat.reshape(bv, c, h * w).permute(0, 2, 1)
        seq = seq + self._get_pos(h, w, seq.device, seq.dtype).unsqueeze(0)
        for blk in self.dec_attn:
            seq = blk(seq)
        return self.dec_attn_norm(seq), h, w

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent → 6144d features (feature branch only)."""
        seq, h, w = self._decode_trunk(z)
        c = seq.shape[-1]
        return self.dec_conv(seq.permute(0, 2, 1).reshape(-1, c, h, w))

    def apply_geo_adapter(self, z: torch.Tensor,
                          num_views: Optional[int] = None) -> torch.Tensor:
        """z -> z + gate*A_geo(z) (identity when no adapter is configured)."""
        if self.geo_adapter is None:
            return z
        return self.geo_adapter(z, num_views)

    def decode_geo(self, z: torch.Tensor,
                   num_views: Optional[int] = None) -> torch.Tensor:
        """Geometry-branch decode: (optional adapter) -> trunk -> dec_conv.

        Numerically identical to ``decode`` when no adapter is configured.
        """
        z_geo = self.apply_geo_adapter(z, num_views)
        seq, h, w = self._decode_trunk(z_geo)
        c = seq.shape[-1]
        return self.dec_conv(seq.permute(0, 2, 1).reshape(-1, c, h, w))



    def decode_rgb(self, z: torch.Tensor, num_views: Optional[int] = None) -> torch.Tensor:
        """Decode latent → RGB pixels (BV, 3, H_px, W_px).

        Pass `num_views` (V) to enable temporal decoding; the BV batch must be
        laid out as contiguous (B, V) groups so frames of one scene are adjacent.
        """
        assert self.rgb_head is not None, "rgb_decoder not configured"
        seq, h, w = self._decode_trunk(z)
        return self.rgb_head(seq, h, w, num_views=num_views)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + sigma * eps — paper Eq. (7).

        Sampled only during codec training; the posterior mean ``mu`` is used
        deterministically for flow training and inference.
        """
        if self.training and self.kl_weight > 0:
            return mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        return mu

    def forward(self, x: torch.Tensor, num_views: Optional[int] = None,
                rgb_jitter: Optional[Dict] = None,
                decode_features: bool = True) -> Dict[str, torch.Tensor]:
        """Encode + decode.

        The feature branch always decodes from the clean latent. When
        ``rgb_jitter`` is given (a dict with ``sigma`` / ``mode`` / ``relative``),
        the RGB head instead decodes from a perturbed latent, which makes the
        decoder robust to the noisy latents Stage 2 emits at sample time. The
        RGB head ALWAYS runs inside this forward (never skipped) so all its
        params stay in the DDP autograd graph; the project's DDP uses
        find_unused_parameters=False.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        seq, h, w = self._decode_trunk(z)
        result = {"mu": mu, "logvar": logvar, "z": z}
        if decode_features:
            c = seq.shape[-1]
            result["recon"] = self.dec_conv(
                seq.permute(0, 2, 1).reshape(-1, c, h, w))
        if self.rgb_head is not None:
            do_jitter = (
                rgb_jitter is not None and float(rgb_jitter.get("sigma", 0.0)) > 0
            )
            if do_jitter:
                z_rgb, sigma_eff = apply_rgb_latent_jitter(
                    z, num_views,
                    rgb_jitter.get("mode", "frame_diff"),
                    float(rgb_jitter["sigma"]),
                    relative=bool(rgb_jitter.get("relative", True)),
                )
                seq_r, h_r, w_r = self._decode_trunk(z_rgb)
                result["rgb_pred"] = self.rgb_head(seq_r, h_r, w_r, num_views=num_views)
                result["rgb_jitter_sigma_eff"] = sigma_eff
            else:
                result["rgb_pred"] = self.rgb_head(seq, h, w, num_views=num_views)
        return result

    def compute_loss(self, x, output, per_level=True):
        """L_feat + lambda_kl * L_kl — the first two terms of paper Eq. (12).

        L_rgb / L_geo / L_repr are added by the training loop (see
        ``scripts/train_codec.py``) because they need extra teachers/targets.
        """
        recon = output["recon"]
        mu, logvar = output["mu"], output["logvar"]

        if per_level and self.level_weights is not None:
            recon_chunks = recon.split(self._level_dim, dim=1)
            input_chunks = x.split(self._level_dim, dim=1)
            recon_loss = sum(
                w * F.l1_loss(rc, ic)
                for w, rc, ic in zip(self.level_weights, recon_chunks, input_chunks)
            ) / self.level_weights.sum()
        else:
            recon_loss = F.l1_loss(recon, x)

        # Per-element KL: KL(N(μ,σ²) || N(0,1)) = -0.5*(1 + logvar - μ² - exp(logvar))
        # Shape: (B, C, H, W)
        if self.kl_weight > 0:
            kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            # Per-channel mean over batch + spatial — this is the "info budget" the
            # encoder allocates to each latent channel.
            kl_per_ch = kl_elem.mean(dim=(0, 2, 3))  # (C,)
            kl_loss = kl_per_ch.mean()  # raw value — logged for diagnostics

            if self.free_bits > 0:
                # Free-bits (Kingma+16): each channel keeps `free_bits` nats of
                # KL "for free" — channels below the floor contribute a constant
                # (zero gradient), channels above contribute their excess. Mean
                # over channels keeps the scale comparable to the raw KL.
                kl_for_loss = torch.clamp(kl_per_ch, min=self.free_bits).mean()
            else:
                kl_for_loss = kl_loss

        else:
            kl_loss = torch.zeros(1, device=x.device)
            kl_for_loss = kl_loss

        losses = {
            "loss": recon_loss + self.kl_weight * kl_for_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_loss_used": kl_for_loss,
        }
        if per_level:
            rc = recon.split(self._level_dim, dim=1)
            ic = x.split(self._level_dim, dim=1)
            for lvl, (r, i) in enumerate(zip(rc, ic)):
                losses[f"recon_l{lvl}"] = F.l1_loss(r, i)
        return losses
