"""Framework-matched raw-DA3 latent codec for the ``da3_direct`` baseline.

This wraps a frozen :class:`DA3Backbone` so raw DA3 features can be plugged into the
V4-RAE latent-diffusion pipeline *without* the compact GAECodec bottleneck,
isolating "latent quality" as the only changed variable.

Design (see chat 2026-07-04):
    * Diffusion runs on the **shallowest** DA3 out-layer (level 0) raw patch
      features — shape ``(B*V, C, h, w)`` with ``C = rae.latent_dim`` (3072 for
      DA3-GIANT). This is what the DiT denoises (``in_channels = C``).
    * **RGB** is decoded by a trainable trunk + MAE-style ``RGBHead`` that mirrors
      GAECodec's RGB branch: ``level-0 feature → dec_proj → dec_attn trunk →
      RGBHead → pixels``. Only the trunk + rgb_head are trainable; they are meant
      to be co-trained with the DiT via the auxiliary RGB loss.
    * **Depth** (optional, eval only) reuses the frozen DA3 stack: the level-0
      feature is propagated through the remaining backbone blocks
      (``forward_from_layer``) to recover the deeper levels, then the DA3 DPT head
      decodes depth. The generation-time CLS token is taken from the GT encode
      (``encode_all``), matching the existing GLD baseline's ``replace_level_features``.

The codec exposes the same ``encode_views`` / ``decode_rgb`` / ``latent_dim``
surface the V4-RAE loop already probes for (``hasattr(vae, "encode_views")``), so
it drops into training/T2I with no changes to ``prepare_data``.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .gae_codec import (
    DINOV2_PATCH,
    RGBHead,
    SelfAttentionBlock,
    _sincos_pos_embed_2d,
)
from .da3 import DA3Backbone


class DA3DirectCodec(nn.Module):
    """Raw-DA3 latent codec with a trainable RGB head + optional frozen DPT depth.

    Args:
        rae: A frozen :class:`DA3Backbone` (DA3 encoder + DPT head). Must be built
            with ``reshape_to_2d=true`` and **no** ``normalization_stat_path``.
        level: DA3 out-layer level to encode/decode with the RGB head. Use
            ``0``/``-4`` for the da3_direct diffusion baseline; other levels are
            supported for RGB-head diagnostics/pretraining only.
        trunk_dim: Shared decoder trunk width (mirrors GAECodec bottleneck).
        num_trunk_blocks: Self-attention blocks in the trunk.
        trunk_heads: Attention heads in the trunk.
        rgb_decoder: RGBHead config dict (hidden_dim/depth/num_heads/...).
    """

    def __init__(
        self,
        rae: DA3Backbone,
        level: int = 0,
        trunk_dim: int = 1024,
        num_trunk_blocks: int = 4,
        trunk_heads: int = 8,
        rgb_decoder: Optional[dict] = None,
        latent_stats_path: Optional[str] = None,
    ):
        super().__init__()
        if level not in (0, 1, 2, 3, -1, -2, -3, -4):
            raise ValueError(
                f"DA3DirectCodec level must be one of 0..3 or -1..-4, got {level!r}"
            )
        self.rae = rae
        self.rae.requires_grad_(False)
        self.level = int(level)
        self.feat_dim = int(rae.latent_dim)   # 3072 for DA3-GIANT
        self.latent_dim = self.feat_dim       # DiT in_channels
        self.trunk_dim = int(trunk_dim)

        rgb_decoder = rgb_decoder or {}

        # ── Self-contained per-channel latent normalization (like sd/wan) ──
        # Diffusion runs in normalized space; encode_views normalizes, decode_*
        # de-normalizes back to raw before the frozen DA3 stack / RGB head. The
        # stats are non-persistent buffers (loaded from file, never from a ckpt),
        # so loading a pretrained RGB-head state_dict never clobbers them.
        self.has_stats = False
        self.register_buffer("latent_mean", None, persistent=False)
        self.register_buffer("latent_std", None, persistent=False)
        if latent_stats_path:
            self.load_latent_stats(latent_stats_path)

        # ── Shared decoder trunk: raw DA3 feature → trunk features ──
        self.dec_proj = nn.Conv2d(self.feat_dim, self.trunk_dim, 1)
        self.dec_attn = nn.ModuleList(
            [SelfAttentionBlock(self.trunk_dim, trunk_heads) for _ in range(num_trunk_blocks)]
        )
        self.dec_attn_norm = nn.LayerNorm(self.trunk_dim)
        self._pos_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        # ── RGB branch (MAE-style), identical architecture to GAECodec ──
        self.rgb_head = RGBHead(
            trunk_dim=self.trunk_dim,
            hidden_dim=rgb_decoder.get("hidden_dim", 1024),
            depth=rgb_decoder.get("depth", 12),
            num_heads=rgb_decoder.get("num_heads", 16),
            ffn_ratio=rgb_decoder.get("ffn_ratio", 4.0),
            patch_size=rgb_decoder.get("patch_size", DINOV2_PATCH),
            temporal=rgb_decoder.get("temporal", False),
            temporal_num_heads=rgb_decoder.get("temporal_num_heads", None),
            temporal_max_views=rgb_decoder.get("temporal_max_views", 64),
            temporal_every=rgb_decoder.get("temporal_every", 1),
            temporal_ffn_ratio=rgb_decoder.get("temporal_ffn_ratio", None),
        )

    # ── Latent normalization ──
    def load_latent_stats(self, path: str) -> None:
        """Load per-channel mean/std (C,) from a stats file → (1, C, 1, 1) buffers."""
        import os

        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"da3_direct latent_stats not found: {path!r}")
        stats = torch.load(path, map_location="cpu")
        mean = stats["mean"].float().reshape(1, -1, 1, 1)
        std = stats["std"].float().reshape(1, -1, 1, 1)
        if mean.numel() != self.feat_dim:
            raise ValueError(
                f"latent_stats channels {mean.numel()} != feat_dim {self.feat_dim}"
            )
        self.latent_mean = mean
        self.latent_std = std
        self.has_stats = True

    def _normalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_stats:
            return z
        return (z - self.latent_mean.to(z)) / (self.latent_std.to(z) + 1e-6)

    def _denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_stats:
            return z
        return z * (self.latent_std.to(z) + 1e-6) + self.latent_mean.to(z)

    def train(self, mode: bool = True):
        """Keep the frozen DA3 encoder in eval mode even when DDP sets us train()."""
        super().train(mode)
        self.rae.eval()
        return self

    # ── Trainable-parameter helper (RAE stays frozen) ──
    def trainable_parameters(self):
        for m in (self.dec_proj, self.dec_attn, self.dec_attn_norm, self.rgb_head):
            for p in m.parameters():
                yield p

    # ── Positional encoding ──
    def _get_pos(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w)
        cached = self._pos_cache.get(key)
        if cached is None or cached.device != device:
            cached = _sincos_pos_embed_2d(h, w, self.trunk_dim, dtype=dtype).to(device)
            self._pos_cache[key] = cached
        return cached

    # ── Encode: [0,1] images → selected DA3 level features (B*V, C, h, w) ──
    @torch.no_grad()
    def encode_views(self, imgs_5d: torch.Tensor) -> torch.Tensor:
        """(B, V, 3, H, W) in [0,1] → selected DA3 level features (B*V, C, h, w).

        Returns raw features when no latent_stats are loaded (e.g. during stats
        computation or RGB-head pretraining); normalized otherwise.
        """
        norm = (imgs_5d - self.rae.encoder_mean[None]) / self.rae.encoder_std[None]
        z_raw = self.rae.encode(norm, mode="single", level=self.level)
        return self._normalize(z_raw)

    @torch.no_grad()
    def encode_all(self, imgs_5d: torch.Tensor) -> Dict[int, torch.Tensor]:
        """(B, V, 3, H, W) in [0,1] → GT all-level raw features (with CLS at idx 0)."""
        norm = (imgs_5d - self.rae.encoder_mean[None]) / self.rae.encoder_std[None]
        return self.rae.encode(norm, mode="all")

    # ── Shared trunk ──
    def _decode_trunk(self, z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        bv, _, h, w = z.shape
        feat = self.dec_proj(z)
        seq = feat.reshape(bv, self.trunk_dim, h * w).permute(0, 2, 1)
        seq = seq + self._get_pos(h, w, seq.device, seq.dtype).unsqueeze(0)
        for blk in self.dec_attn:
            seq = blk(seq)
        return self.dec_attn_norm(seq), h, w

    # ── Forward = RGB decode (so DDP can wrap the trainable branch) ──
    def forward(self, z: torch.Tensor, num_views: Optional[int] = None) -> torch.Tensor:
        return self.decode_rgb(z, num_views=num_views)

    # ── RGB decode (trainable branch) ──
    def decode_rgb(self, z: torch.Tensor, num_views: Optional[int] = None) -> torch.Tensor:
        """Selected DA3-level latent (B*V, C, h, w) → ImageNet-normalized RGB.

        ``z`` is the (self-)normalized diffusion latent; it is de-normalized to
        the raw level-0 feature the trunk/RGBHead were trained on.
        """
        seq, h, w = self._decode_trunk(self._denormalize(z))
        return self.rgb_head(seq, h, w, num_views=num_views)

    # ── Depth decode (frozen DA3 propagation + DPT head, eval only) ──
    @torch.no_grad()
    def decode_depth(
        self,
        z_level0: torch.Tensor,
        cls_token: Optional[torch.Tensor],
        total_view: int,
        H: int,
        W: int,
    ) -> dict:
        """Raw level-0 feature → {rgb, depth, depth_conf, ...} via forward-propagation + DPT.

        ``z_level0``: (B*V, C, h, w) raw level-0 feature.
        ``z_level0``: (self-)normalized level-0 latent; de-normalized here to raw
            before propagation through the frozen DA3 backbone.
        ``cls_token``: (B*V, C) raw level-0 CLS (from ``encode_all``[level][:, 0]);
            required for a faithful propagation. If ``None``, zeros are used.
        """
        if self.level not in (0, -4):
            raise ValueError(
                "decode_depth requires the shallowest DA3 level (0/-4) so deeper "
                f"features can be propagated forward; got level={self.level!r}."
            )
        feats = self.rae.propagate_features(
            self._denormalize(z_level0), from_level=self.level,
            total_view=total_view, cls_token=cls_token,
        )
        dpt_decoder = self.rae.rae_cl_decoder
        if dpt_decoder is None:
            raise RuntimeError("DA3DirectCodec.decode_depth requires the DA3 DPT decoder")

        # The DA3-GIANT DPT head's `rgb` entry is a nested structure. The
        # direct path only consumes geometry (depth/ray/conf), so bypass
        # DA3Backbone.decode(), which assumes `rgb` is an image tensor.
        with torch.autocast(device_type=z_level0.device.type, enabled=False):
            return dpt_decoder(feats, H, W, patch_start_idx=0)
