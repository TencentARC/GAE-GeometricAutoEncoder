"""Download / resolve GAE weights from the Hugging Face Hub."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

DEFAULT_REPO = os.environ.get("GAE_HF_REPO", "TencentARC/GAE-D64-1B")

# filename -> (subset, size|"shared", description)
MANIFEST: Dict[str, tuple] = {
    "gae_64.pt": ("codec", "64", "GAE-64 codec."),
    "gae_128.pt": ("codec", "128", "GAE-128 codec."),
    "latent_stats_gae_64.pt": ("codec", "64", "Per-channel latent mean/std for GAE-64."),
    "latent_stats_gae_128.pt": ("codec", "128", "Per-channel latent mean/std for GAE-128."),
    "da3_stats_giant_5ds.tar": (
        "codec",
        "shared",
        "DA3-GIANT feature-level stats. Extract to model_stats/da3_giant_5ds/.",
    ),
    "flow_gae64.pt": ("flow", "64", "GAE-64 flow model."),
    "flow_gae128.pt": ("flow", "128", "GAE-128 flow model."),
}


def files_for(size: str = "64", subset: str = "all") -> List[str]:
    out: List[str] = []
    for name, (sub, sz, _desc) in MANIFEST.items():
        if subset not in ("all", sub):
            continue
        if size == "all" or sz == "shared" or sz == str(size):
            out.append(name)
    return out


def download_weights(
    repo_id: Optional[str] = None,
    *,
    size: str = "64",
    subset: str = "all",
    out_dir: str | Path = "ckpts",
    filenames: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Fetch Hub files into ``out_dir``. Skips files that already exist."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required (pip install huggingface-hub)") from exc

    repo = repo_id or DEFAULT_REPO
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    names = list(filenames) if filenames is not None else files_for(str(size), subset)
    paths: Dict[str, Path] = {}
    for name in names:
        local = dest / name
        if local.is_file() and local.stat().st_size > 0:
            paths[name] = local
            continue
        paths[name] = Path(
            hf_hub_download(repo_id=repo, filename=name, local_dir=str(dest))
        )
    return paths


def extract_da3_stats(tar_path: str | Path, dest_dir: str | Path) -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "normalization_stats_level0.pt"
    if marker.is_file():
        return dest
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest)
    return dest
