"""Run the stubbed end-to-end pipeline check as a pytest case (no downloads)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

torch = pytest.importorskip("torch")


def test_stub_pipeline_smoke() -> None:
    import smoke_test

    assert smoke_test.main() == 0
