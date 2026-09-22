"""Helpers for feeding features to the frozen DA3 DPT head.

Shared by the Stage 1 codec trainer and the Stage 2 geometry auxiliary loss.
The DPT head expects the DA3 token layout (a CLS token plus per-level patch
tokens), so reconstructed features have to be re-shaped before the head can
read depth, rays and point maps out of them.
"""

from __future__ import annotations

import torch

# DA3-GIANT embed dim. Each level's patch tokens are a concatenation of local
# and global halves, i.e. 2 * embed_dim channels wide (3072 for DA3-GIANT).
_DA3_EMBED_DIM = 1536


def _format_feats_for_dpt(feats_with_cls, backbone_norm, embed_dim=None):
    """DA3 features (with CLS token) → DPT decoder input format."""
    edim = embed_dim or _DA3_EMBED_DIM
    result = []
    for lvl in sorted(feats_with_cls.keys()):
        feat = feats_with_cls[lvl]            # (BV, N+1, C)
        cls_raw = feat[:, 0, :]                # (BV, C)
        patches = feat[:, 1:, :]               # (BV, N, C)
        local = patches[:, :, :edim]
        glob_ln = backbone_norm(patches[:, :, edim:])
        result.append((torch.cat([local, glob_ln], dim=-1).unsqueeze(0),
                        cls_raw.unsqueeze(0)))
    return result


def _format_recon_for_dpt(recon_feats, backbone_norm, bv, embed_dim=None):
    """VAE-reconstructed features (no CLS) → DPT decoder input format."""
    edim = embed_dim or _DA3_EMBED_DIM
    hidden_size = edim * 2  # [local, global] concatenation
    result = []
    for lvl in sorted(recon_feats.keys()):
        patches = recon_feats[lvl]             # (BV, N, C)
        cls_zeros = torch.zeros(bv, hidden_size, device=patches.device, dtype=patches.dtype)
        local = patches[:, :, :edim]
        glob_ln = backbone_norm(patches[:, :, edim:])
        result.append((torch.cat([local, glob_ln], dim=-1).unsqueeze(0),
                        cls_zeros.unsqueeze(0)))
    return result


def _geo_view_indices(num_views: int, max_views: int, device: torch.device) -> torch.Tensor:
    """Evenly cover a scene's views for geometry supervision."""
    if max_views <= 0 or num_views <= max_views:
        return torch.arange(num_views, device=device)
    idx = torch.linspace(0, num_views - 1, steps=max_views, device=device)
    return idx.round().long().unique(sorted=True)
