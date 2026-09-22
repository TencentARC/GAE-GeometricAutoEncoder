"""Every shipped config must load and resolve its dataset targets."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("config", CONFIGS, ids=lambda p: p.name)
def test_config_resolves(config: Path) -> None:
    os.environ.setdefault("GAE_DATA_ROOT", "/tmp/gae_data")
    cfg = OmegaConf.load(config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    # Codec configs expose codec; flow configs expose stage_2.
    assert ("codec" in resolved) or ("stage_2" in resolved)


def test_at_least_one_codec_and_flow_config() -> None:
    names = {p.name for p in CONFIGS}
    assert any(n.startswith("gae_") for n in names)
    assert any(n.startswith("flow_") for n in names)
