"""Optimizer + scheduler builders for the RAEv2 T2I recipe.

Why a separate module from ``utils/optim_utils.py``:
  * ``optim_utils.build_optimizer`` is exclusively AdamW (raises on anything
    else) and consumed by every other GLD trainer. Touching it would put the
    whole repo at risk for a feature that only this trainer wants.
  * GMuon adds a runtime dependency (``gram-newton-schulz``) that may not be
    installed on every node yet. We isolate the import + fallback here.

Public API:
  * :func:`build_optimizer_rae`  → ``(optimizer, name_str)``
  * :func:`build_lr_lambda_rae`  → callable suitable for ``LambdaLR``

The builders intentionally take plain dicts (not OmegaConf / dataclass) so
the trainer can pass ``OmegaConf.to_container(...)`` directly.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


def _as_tuple(values: Any, length: int = 2) -> tuple:
    if isinstance(values, (list, tuple)):
        if len(values) != length:
            raise ValueError(f"expected length {length}, got {len(values)}: {values!r}")
        return tuple(float(v) for v in values)
    return tuple(float(values) for _ in range(length))


def _split_2d_vs_other(
    parameters: Iterable[torch.nn.Parameter],
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Muon orthogonalizes 2D weight matrices; 1D / scalar parameters
    (biases, RMSNorm scales, embeddings, conv weights flattened to 4D, etc.)
    are routed to AdamW. Embedding tables are 2D but better off in AdamW
    (orthogonalization on a token table is meaningless), so we explicitly
    keep ``ndim == 2`` AND not part of an Embedding lookup. Detection is by
    parameter name pattern in the trainer; here we just slice by ndim==2.
    """
    p_list = list(parameters)
    muon = [p for p in p_list if p.ndim == 2]
    rest = [p for p in p_list if p.ndim != 2]
    return muon, rest


def build_optimizer_rae(
    parameters: Iterable[torch.nn.Parameter],
    *,
    optim_cfg: Dict[str, Any],
    base_lr_default: float,
) -> Tuple[Optimizer, str]:
    """Build optimizer for the T2I RAE trainer.

    ``optim_cfg`` keys (all optional unless noted):
        type:      'gmuon' | 'adamw' (default 'adamw')
        lr:        float — base learning rate
        adamw_lr:  float — separate LR for the AdamW group when type=='gmuon'.
                   Falls back to lr.
        betas:     [b1, b2] (default (0.9, 0.95))
        weight_decay: float (default 0.0)
        eps:       float  (default 1e-8)
        # gmuon-only:
        momentum:  float (default 0.95)
        nesterov:  bool  (default True)
        ns_coefficients_preset: 'POLAR_EXPRESS_COEFFICIENTS' | 'YOU_COEFFICIENTS'
        ns_use_kernels: bool
        gmuon_num_restarts: int (default 1)

    Returns the optimizer and a human-readable description for logging.
    Raises ``RuntimeError`` only on misconfigured arguments — missing
    ``gram-newton-schulz`` triggers a warning + AdamW fallback so a node
    without the optional package can still train (recipe degrades to b only).
    """
    opt_type = str(optim_cfg.get("type", "adamw")).lower()
    base_lr = float(optim_cfg.get("lr", base_lr_default))
    betas = _as_tuple(optim_cfg.get("betas", optim_cfg.get("beta", (0.9, 0.95))))
    wd = float(optim_cfg.get("weight_decay", optim_cfg.get("wd", 0.0)))
    eps = float(optim_cfg.get("eps", 1e-8))

    if opt_type == "adamw":
        opt = torch.optim.AdamW(
            list(parameters), lr=base_lr, betas=betas, weight_decay=wd, eps=eps,
            fused=torch.cuda.is_available(),
        )
        return opt, f"AdamW(lr={base_lr}, betas={betas}, wd={wd})"

    if opt_type == "gmuon":
        try:
            from gram_newton_schulz import Muon  # type: ignore

            try:
                from gram_newton_schulz import POLAR_EXPRESS_COEFFICIENTS  # type: ignore
            except ImportError:
                POLAR_EXPRESS_COEFFICIENTS = None  # type: ignore
            try:
                from gram_newton_schulz import YOU_COEFFICIENTS  # type: ignore
            except ImportError:
                YOU_COEFFICIENTS = None  # type: ignore
        except ImportError as e:
            logger.warning(
                "gram-newton-schulz not installed (%s); falling back to AdamW. "
                "Run `uv sync` to install the optional dependency.",
                e,
            )
            opt = torch.optim.AdamW(
                list(parameters), lr=base_lr, betas=betas, weight_decay=wd, eps=eps,
                fused=torch.cuda.is_available(),
            )
            return opt, f"AdamW(lr={base_lr}, fallback from gmuon)"

        muon_params, fallback_params = _split_2d_vs_other(parameters)
        if not muon_params:
            raise RuntimeError(
                "gmuon requested but no 2D parameters found — every weight "
                "matrix would route to AdamW, defeating the purpose."
            )

        adamw_lr = float(optim_cfg.get("adamw_lr", base_lr))
        scalar_optimizer = torch.optim.AdamW(
            fallback_params if fallback_params else [torch.nn.Parameter(torch.zeros(1))],
            lr=adamw_lr, betas=betas, weight_decay=wd, eps=eps,
        )

        preset_name = str(optim_cfg.get("ns_coefficients_preset", "POLAR_EXPRESS"))
        coeff_table = {
            "POLAR_EXPRESS_COEFFICIENTS": POLAR_EXPRESS_COEFFICIENTS,
            "POLAR_EXPRESS": POLAR_EXPRESS_COEFFICIENTS,
            "YOU_COEFFICIENTS": YOU_COEFFICIENTS,
            "YOU": YOU_COEFFICIENTS,
        }
        ns_coeffs = coeff_table.get(preset_name, POLAR_EXPRESS_COEFFICIENTS)
        if ns_coeffs is None:
            raise RuntimeError(
                f"ns_coefficients_preset {preset_name!r} not exported by your "
                f"gram_newton_schulz install. Update the package."
            )

        muon_kwargs: Dict[str, Any] = dict(
            params=[{"params": muon_params, "lr": base_lr,
                     "weight_decay": wd, "momentum": float(optim_cfg.get("momentum", 0.95))}],
            scalar_optimizer=scalar_optimizer,
            lr=base_lr,
            momentum=float(optim_cfg.get("momentum", 0.95)),
            nesterov=bool(optim_cfg.get("nesterov", True)),
            weight_decay=wd,
            adjust_lr="rms_norm",
            ns_algorithm="gram_newton_schulz",
            ns_use_kernels=bool(optim_cfg.get("ns_use_kernels", False)),
            ns_coefficients=ns_coeffs,
            gram_newton_schulz_num_restarts=int(
                optim_cfg.get("gmuon_num_restarts", 1)
            ),
        )
        try:
            opt = Muon(**muon_kwargs)
        except TypeError as e:
            # Older releases (<0.1.4) had a different keyword surface
            # (``ns_coefficients_preset`` instead of ``ns_coefficients`` /
            # ``ns_algorithm`` not present). Try the legacy interface once.
            logger.warning(
                "Muon() rejected kwargs (%s); retrying with legacy interface", e
            )
            legacy_kwargs = dict(
                params=muon_params,
                lr=base_lr,
                momentum=float(optim_cfg.get("momentum", 0.95)),
                nesterov=bool(optim_cfg.get("nesterov", True)),
                weight_decay=wd,
                adjust_lr="rms_norm",
            )
            if hasattr(Muon, "__init__") and "ns_coefficients_preset" in Muon.__init__.__code__.co_varnames:
                legacy_kwargs["ns_coefficients_preset"] = preset_name
            opt = Muon(**legacy_kwargs)
            # Wrap so a single .step() drives both halves.
            opt = _MuonAdamW(opt, scalar_optimizer)

        msg = (
            f"GMuon(lr={base_lr}, momentum={float(optim_cfg.get('momentum', 0.95))}, "
            f"preset={preset_name}, kernels={bool(optim_cfg.get('ns_use_kernels', False))}, "
            f"{len(muon_params)} 2D params, {len(fallback_params)} fallback)"
        )
        return opt, msg

    raise ValueError(f"Unsupported optimizer type {opt_type!r}; expected 'adamw' or 'gmuon'.")


class _MuonAdamW(Optimizer):
    """Composite optimizer used only on the legacy gram_newton_schulz path
    where ``Muon`` does not accept a ``scalar_optimizer`` argument. New
    versions wrap this internally."""

    def __init__(self, muon_opt: Optimizer, adamw_opt: Optimizer):
        self._muon = muon_opt
        self._adamw = adamw_opt
        self.param_groups = list(muon_opt.param_groups) + list(adamw_opt.param_groups)
        self.defaults = {}

    @property
    def state(self):
        merged = {}
        merged.update(self._muon.state)
        merged.update(self._adamw.state)
        return merged

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        self._muon.step(closure=closure)
        self._adamw.step(closure=closure)

    def state_dict(self):  # type: ignore[override]
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict):  # type: ignore[override]
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = list(self._muon.param_groups) + list(self._adamw.param_groups)


def build_lr_lambda_rae(
    *,
    schedule_type: str,
    base_lr: float,
    final_lr: float,
    warmup_steps: int,
    decay_end_steps: int,
    warmup_from_zero: bool = False,
):
    """Returns a ``lambda step → multiplier`` for ``torch.optim.lr_scheduler.LambdaLR``.

    Shared between cosine / linear schedules — keeps the trainer file
    smaller. Multiplier is relative to ``base_lr`` (so the optimizer should
    have its lr set to ``base_lr`` before construction)."""
    warmup_steps = max(int(warmup_steps), 0)
    decay_end_steps = max(int(decay_end_steps), warmup_steps + 1)
    final_ratio = (final_lr / base_lr) if base_lr > 0 else 1.0
    total_decay = max(decay_end_steps - warmup_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return ((step + 1) / max(warmup_steps, 1)) if warmup_from_zero else 1.0
        if step >= decay_end_steps:
            return final_ratio
        progress = (step - warmup_steps) / total_decay
        if schedule_type == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return final_ratio + (1.0 - final_ratio) * cosine
        if schedule_type == "linear":
            return 1.0 - (1.0 - final_ratio) * progress
        if schedule_type == "constant":
            return 1.0
        raise ValueError(f"Unsupported schedule_type {schedule_type!r}")

    return lr_lambda
