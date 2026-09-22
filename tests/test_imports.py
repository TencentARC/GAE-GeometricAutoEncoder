"""The public facade and core research modules must import without GPU/weights."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Core modules require torch; skip on the torch-less CI runner.
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_public_facade_imports() -> None:
    import gae

    for name in ("GAE", "load_codec", "load_backbone", "load_flow"):
        assert hasattr(gae, name)


def test_core_modules_import() -> None:
    import importlib

    for module in (
        "stage1.gae_codec",
        "stage2.models.dit_temporal",
        "stage2.transport.flow",
        "utils.train_runtime",
        "metrics.latent",
    ):
        importlib.import_module(module)


def test_sample_signature_is_public() -> None:
    import inspect

    from gae import GAE

    params = inspect.signature(GAE.sample).parameters
    for name in ("z_ref", "total_views", "cond_num", "plucker_6d", "ref_global"):
        assert name in params
