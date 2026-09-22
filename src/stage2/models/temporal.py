"""
TemporalTokenConcatDDT — `TokenConcatDDT` + axial temporal (frame-axis) RoPE.

What it adds vs the base DDT head
------------------
The base head has only 2D spatial RoPE; frames are distinguished by Plücker pose + global
cross-view attention (no frame-axis position). This variant adds **temporal
RoPE** on a disjoint head-dim band (see `temporal_rope.py`), giving each frame
an ordinal phase — important for dynamic video (e.g. static-camera object
motion) where pose alone can't order frames.

Integration (zero new parameters, weights preserved)
----------------------------------------------------
* The spatial-RoPE band is shrunk from ``head_dim`` to ``head_dim - dim_t``
  (we just lower ``enc_half_head_dim`` / ``dec_half_head_dim`` and clear the
  rope caches; the parent forward rebuilds a smaller spatial rope on demand).
* Each block's ``PluckerAttention`` is upgraded in place to
  ``TemporalPluckerAttention`` — same qkv / Plücker / ref-K / cross-attn weights,
  it merely wraps the spatial ``rope`` into a spatial+temporal composite at
  forward time. So a from-scratch run and the base DDT head share identical learnable
  state (the only extra buffer is a non-persistent ``inv_freq``).

Enable via config::

    stage_2:
      target: stage2.models.temporal.TemporalTokenConcatDDT
      params:
        ...                       # same as TokenConcatDDT
        temporal_band_frac: 0.25  # fraction of head_dim reserved for temporal
        temporal_theta: 10000.0
"""
from __future__ import annotations

from .plucker_attention import PluckerAttention
from .token_concat_ddt import TokenConcatDDT
from .temporal_rope import TemporalRoPE, SpatialTemporalRoPE


def _even_band(head_dim: int, frac: float) -> int:
    """Temporal band size: round(frac*head_dim) clamped to an even value in
    [2, head_dim-2] so the spatial band stays non-empty and pair-aligned."""
    dt = int(round(head_dim * frac))
    dt -= dt % 2
    return max(2, min(dt, head_dim - 2))


def install_temporal_rope(
    model: TokenConcatDDT,
    *,
    temporal_band_frac: float = 0.25,
    temporal_theta: float = 10000.0,
) -> dict:
    """Upgrade an existing DDT backbone with axial temporal RoPE in-place.

    Callable on any ``TokenConcatDDT`` subclass (``GAEFlow``,
    ``TemporalTokenConcatDDT``, etc.) after its ``__init__`` completes.
    """
    if not model.use_rope:
        raise ValueError(
            "install_temporal_rope requires use_rope=true (temporal RoPE "
            "shares the head-dim band split with spatial RoPE)."
        )

    enc_heads, dec_heads = model.num_heads[0], model.num_heads[1]
    enc_head_dim = model.encoder_hidden_size // enc_heads
    dec_head_dim = model.decoder_hidden_size // dec_heads
    dim_t_enc = _even_band(enc_head_dim, temporal_band_frac)
    dim_t_dec = _even_band(dec_head_dim, temporal_band_frac)

    model.enc_half_head_dim = (enc_head_dim - dim_t_enc) // 2
    model.dec_half_head_dim = (dec_head_dim - dim_t_dec) // 2
    model._enc_rope_cache.clear()
    model._dec_rope_cache.clear()

    for i, blk in enumerate(model.blocks):
        dt = dim_t_enc if i < model.num_encoder_blocks else dim_t_dec
        TemporalPluckerAttention.upgrade(blk.attn, dt, temporal_theta)

    return dict(
        band_frac=temporal_band_frac,
        theta=temporal_theta,
        enc_head_dim=enc_head_dim,
        dec_head_dim=dec_head_dim,
        dim_t_enc=dim_t_enc,
        dim_t_dec=dim_t_dec,
    )


class TemporalPluckerAttention(PluckerAttention):
    """`PluckerAttention` whose spatial RoPE is composed with a temporal RoPE.

    Created by :meth:`upgrade` (in-place class swap) so all trained weights of
    an existing attention are kept; only a parameter-free ``TemporalRoPE`` and
    the band size ``_dim_t`` are attached.
    """

    @classmethod
    def upgrade(cls, attn: PluckerAttention, dim_t: int, theta: float) -> "TemporalPluckerAttention":
        attn.__class__ = cls
        attn._dim_t = int(dim_t)
        attn.temporal_rope = TemporalRoPE(dim_t, theta=theta)
        return attn  # type: ignore[return-value]

    def forward(self, x, total_view, rope=None, **kwargs):
        # Wrap the spatial rope into a disjoint-band spatial+temporal composite.
        # When rope is None (use_rope=false) we leave it untouched.
        # ``temporal_frame_idx`` (set by the model forward) remaps cond/target
        # view slots to their physical frame index; absent → slot index is used.
        if rope is not None and getattr(self, "temporal_rope", None) is not None:
            rope = SpatialTemporalRoPE(
                spatial_rope=rope,
                temporal_rope=self.temporal_rope,
                total_view=total_view,
                spatial_dim=self.head_dim - self._dim_t,
                frame_idx=kwargs.get("temporal_frame_idx"),
            )
        return super().forward(x, total_view, rope=rope, **kwargs)


class TemporalTokenConcatDDT(TokenConcatDDT):
    """DDT backbone with axial temporal RoPE added to every attention block."""

    def __init__(self, *, temporal_band_frac: float = 0.25,
                 temporal_theta: float = 10000.0, **kwargs):
        super().__init__(**kwargs)
        self._temporal_cfg = install_temporal_rope(
            self,
            temporal_band_frac=temporal_band_frac,
            temporal_theta=temporal_theta,
        )
