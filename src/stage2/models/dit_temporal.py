"""GAEFlowTemporal — GAEFlow plus axial temporal RoPE.

Combines :class:`dit.GAEFlow` (x-prediction + IG baseline
head, dict return for ``training_losses_rae``) with the frame-axis temporal
RoPE from :class:`temporal.TemporalTokenConcatDDT`.

T2I cotrain (V=1) is safe: temporal RoPE degenerates to a constant phase when
``total_view=1``. Warm-start from a T2I-RAE ckpt remains weight-compatible
(``install_temporal_rope`` adds no learnable parameters).

Enable via config::

    stage_2:
      target: stage2.models.dit_temporal.GAEFlowTemporal
      params:
        temporal_band_frac: 0.25
        temporal_theta: 10000.0
        ...  # same as GAEFlow
"""
from __future__ import annotations

from .dit import GAEFlow
from .temporal import install_temporal_rope


class GAEFlowTemporal(GAEFlow):
    """DDT backbone with temporal RoPE on every attention block."""

    def __init__(
        self,
        *,
        temporal_band_frac: float = 0.25,
        temporal_theta: float = 10000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._temporal_cfg = install_temporal_rope(
            self,
            temporal_band_frac=temporal_band_frac,
            temporal_theta=temporal_theta,
        )
