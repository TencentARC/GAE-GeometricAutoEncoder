"""
Temporal (frame-axis) RoPE for video DiT, composable with the existing 2D
spatial RoPE (`VisionRotaryEmbeddingFast`).

Motivation
----------
`TokenConcatDDT` distinguishes frames only via Plücker camera-pose PE + full
cross-view attention; there is NO rotary position over the frame axis. For
夸张点说 "static camera + moving object" 两帧 pose 相同 → 位置上不可区分。
Temporal RoPE adds an ordinal frame-index rotation so tokens at the same
spatial location but different frames get distinct phases.

Design (axial 3D RoPE, disjoint bands)
--------------------------------------
head_dim is split into two disjoint sub-bands::

    [ 0 ........ head_dim - dim_t )   ← spatial RoPE (h, w)  (unchanged)
    [ head_dim - dim_t ... head_dim ) ← temporal RoPE (frame index)

Disjoint bands (vs. composing both on the full dim) keep spatial and temporal
relative positions independent — the standard choice in CogVideoX / OpenSora
3D RoPE. Both use the SAME interleaved-pair convention as
`VisionRotaryEmbeddingFast` (freqs repeated r=2, paired by `rotate_half`), so
the two halves concatenate seamlessly.

No learnable parameters; `inv_freq` is a constant buffer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model_utils import rotate_half


class TemporalRoPE(nn.Module):
    """Axial RoPE over the frame (view) axis, applied to a ``dim_t`` sub-band.

    Args:
        dim_t: size of the temporal head-dim band (even). Rotates this many dims.
        theta: RoPE base. Frames are integer indices 0..V-1, so the default
               10000 spreads phases sensibly for the typical V≈8..64 range.
    """

    def __init__(self, dim_t: int, theta: float = 10000.0):
        super().__init__()
        if dim_t % 2 != 0:
            raise ValueError(f"TemporalRoPE dim_t must be even, got {dim_t}")
        self.dim_t = int(dim_t)
        self.theta = float(theta)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_t, 2).float() / dim_t))  # (dim_t/2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        xt: torch.Tensor,
        total_view: int,
        frame_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rotate ``xt`` by its per-frame angle.

        Args:
            xt: (B*V, heads, N, dim_t) — q/k temporal band. Row-major layout is
                ``bv = b*V + v`` (matches the model's ``(b v)`` flatten), so the
                view index is ``bv % V`` and is shared across all N spatial
                tokens of that frame.
            total_view: V (number of views packed in the sequence dim).
            frame_idx: optional (V,) long tensor mapping each view slot to its
                temporal frame index. Lets the caller decouple the ordinal frame
                phase from the physical slot — e.g. so prepended cond views reuse
                the frame index of the target view they replicate. When ``None``
                the slot index itself is used (``bv % V``).

        Returns:
            (B*V, heads, N, dim_t) rotated.
        """
        BV = xt.shape[0]
        device = xt.device
        slot = torch.arange(BV, device=device) % total_view  # (BV,) view slot
        if frame_idx is None:
            frame = slot.float()
        else:
            frame = frame_idx.to(device=device)[slot].float()  # (BV,)
        ang = torch.outer(frame, self.inv_freq.to(device))              # (BV, dim_t/2)
        ang = ang.repeat_interleave(2, dim=-1)                          # (BV, dim_t)
        cos = ang.cos().view(BV, 1, 1, -1)
        sin = ang.sin().view(BV, 1, 1, -1)
        return xt * cos + rotate_half(xt) * sin


class SpatialTemporalRoPE:
    """Per-forward composite: split head_dim, apply spatial RoPE to the first
    ``spatial_dim`` dims and temporal RoPE to the rest.

    Lightweight plain object (no params) so it can be created fresh inside each
    attention forward and passed where a spatial ``rope`` callable is expected.
    """

    def __init__(self, spatial_rope, temporal_rope: TemporalRoPE,
                 total_view: int, spatial_dim: int,
                 frame_idx: torch.Tensor | None = None):
        self.spatial_rope = spatial_rope
        self.temporal_rope = temporal_rope
        self.total_view = total_view
        self.spatial_dim = spatial_dim
        self.frame_idx = frame_idx

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B*V, heads, N, head_dim). spatial_rope expects last dim == its rot_dim.
        xs = self.spatial_rope(t[..., : self.spatial_dim])
        xt = self.temporal_rope(
            t[..., self.spatial_dim:], self.total_view, frame_idx=self.frame_idx,
        )
        return torch.cat([xs, xt], dim=-1)


def build_temporal_frame_idx(
    total_view: int,
    cond_num: int,
    has_cond: bool,
    device,
    view_frame_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-view temporal frame index for an encoder/decoder token layout.

    Token layout is ``[cond_0..cond_{K-1}, view_0..view_{V-1}]`` when cond tokens
    are prepended (``has_cond``), else just ``[view_0..view_{V-1}]``. Cond view
    ``k`` duplicates batch slot ``k`` (see ``cut3r_adapter._get_view_order``),
    so both get ``view_frame_idx[k]``.

    Args:
        view_frame_idx: optional ``(V,)`` long tensor — physical timeline index
            per batch slot (``cut3r_adapter`` reorder ``order``). When ``None``,
            slot index ``0..V-1`` is used (correct for prefix / chronological).

    Returns a 1-D long tensor of length ``K+V`` (has_cond) or ``V``.
    """
    if view_frame_idx is None:
        tgt = torch.arange(total_view, device=device, dtype=torch.long)
    else:
        tgt = view_frame_idx.to(device=device, dtype=torch.long).reshape(-1)
        if tgt.numel() != total_view:
            raise ValueError(
                f"view_frame_idx length {tgt.numel()} != total_view {total_view}"
            )
    if has_cond and cond_num > 0:
        return torch.cat([tgt[:cond_num], tgt])
    return tgt
