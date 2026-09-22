"""
ScanNet++ pre-computed latent dataset for long-sequence GLD training.

Each sample is one pre-computed chunk file produced by
``scripts/precompute_scannetpp_latents.py``. Two physical layouts are
supported transparently:

    Flat  (long-sequence source):  ROOT/<scene_id>/chunk_*.pt
    Nested (multi-clip source):    ROOT/<scene_id>/clip_*/chunk_*.pt

Each chunk holds::

    {
        "z_all":            (V, C, h, w) bf16  — diffusion target
        "z_ref":            (1, C, h, w) bf16  — conditioning
        "c2w":              (V, 4, 4) fp32
        "intrinsics":       (V, 3, 3) fp32
        "frame_idx_in_mp4": (V,) int32
        "scene_id":         str  ("<scene>" or "<scene>/<clip>")
        ...
    }

``split`` partitioning is done on ``<scene_id>`` (top-level dir name) so that
the flat and nested datasets sharing the same scene names land in the same
split, preventing train/val leakage when both are mixed in one yaml.

This bypasses ALL image / DA3 / VAE work at train time. The training loop
detects ``is_precomputed=True`` in the batch and skips ``prepare_data``.

It does NOT inherit ``BaseMultiViewDataset`` (which assumes raw RGB ↔ DA3
encoding) — it's a plain ``EasyDataset`` so the existing ``MulDataset`` /
``ResizedDataset`` syntax (``50000 @ ds``) still works.
"""
from __future__ import annotations

import hashlib
import os
import os.path as osp
from pathlib import Path
from typing import Any

import torch

from cut3r_data.base.dataset_index_cache import (
    load_or_build,
    mirror_shared_path,
    resolve_shared_cache_dir,
)
from cut3r_data.base.easy_dataset import EasyDataset
from cut3r_data.base.batched_sampler import BatchedRandomSampler, CustomRandomSampler


class ScanNetppLatent_Multi(EasyDataset):
    """Pre-computed latent chunks → directly emitted as a single dict per sample.

    Args:
        ROOT:        Latent root, e.g. ``$GAE_DATA_ROOT/scannetpp_latents``.
        num_views:   Expected V per chunk; chunks not matching are dropped.
        resolution:  Kept for sampler compatibility; not used to alter latents.
                     Must be a single ``(H, W)`` tuple (latents have fixed H/W).
        split:       Optional "train"/"val" using a deterministic scene hash.
        val_frac:    Fraction of scenes reserved for val (default 0.02).
        allow_repeat: Sampler kwarg (unused; kept for parity).
        ref_view_sampling: Always ``"prefix"`` because z_ref is fixed at index 0.
    """

    def __init__(
        self,
        *,
        ROOT: str,
        num_views: int,
        resolution: list[tuple[int, int]] | tuple[int, int],
        split: str | None = None,
        val_frac: float = 0.02,
        allow_repeat: bool = False,
        ref_view_sampling: str = "prefix",
        seed: int | None = None,
        **_ignored_kwargs,  # for forward-compat with cfg passes (skip_bad_poses, etc.)
    ):
        self.ROOT = ROOT
        self.num_views = int(num_views)
        self.split = split
        self.val_frac = float(val_frac)
        self.allow_repeat = bool(allow_repeat)
        self.ref_view_sampling = ref_view_sampling
        self.seed = seed

        # Sampler compatibility: it needs a list of resolutions (we have a fixed one).
        if isinstance(resolution, tuple):
            self._resolutions = [tuple(resolution)]
        else:
            # If multiple resolutions given, take the first (latents are fixed-shape).
            self._resolutions = [tuple(resolution[0])]

        # is_t2v handling for adapter compatibility
        self.is_metric = True
        self.video = True

        self._load_index()

    # ── Indexing ─────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        """Cache key ``_v2``: v1 indexer missed nested ``<scene>/<clip>/chunk_*.pt``."""
        key = f"scannetpp_latent_v2__{self.ROOT}__V{self.num_views}__split{self.split}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        local_root = Path("/local-ssd/scannetpp_latent_cache")
        if Path("/local-ssd").is_dir():
            local_root.mkdir(parents=True, exist_ok=True)
            return str(local_root / f"scannetpp_index_{tag}.pkl")
        return osp.join(self.ROOT, f".scannetpp_index_{tag}.pkl")

    def _shared_cache_path(self) -> str | None:
        local = self._cache_path()
        shared = mirror_shared_path(local, resolve_shared_cache_dir())
        if shared is not None:
            return shared
        shared_root = resolve_shared_cache_dir()
        if shared_root is None:
            return None
        return osp.join(shared_root, "scannetpp_latent_cache", osp.basename(local))

    def _scene_in_split(self, scene_id: str) -> bool:
        if self.split is None:
            return True
        h = int(hashlib.md5(scene_id.encode()).hexdigest()[:8], 16) / 0xffffffff
        is_val = h < self.val_frac
        return (self.split == "val") if is_val else (self.split == "train")

    def _build_index(self) -> list[str]:
        print(f"[ScanNetppLatent] scanning {self.ROOT} …", flush=True)
        chunk_paths: list[str] = []
        scene_dirs = sorted(
            d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))
        )
        n_flat = 0
        n_nested = 0
        for sid in scene_dirs:
            if not self._scene_in_split(sid):
                continue
            scene_dir = osp.join(self.ROOT, sid)
            for entry in sorted(os.listdir(scene_dir)):
                entry_path = osp.join(scene_dir, entry)
                if entry.startswith("chunk_") and entry.endswith(".pt"):
                    chunk_paths.append(entry_path)
                    n_flat += 1
                elif osp.isdir(entry_path) and entry.startswith("clip_"):
                    for fn in sorted(os.listdir(entry_path)):
                        if fn.startswith("chunk_") and fn.endswith(".pt"):
                            chunk_paths.append(osp.join(entry_path, fn))
                            n_nested += 1
        print(
            f"[ScanNetppLatent] {len(chunk_paths)} chunks indexed "
            f"(split={self.split}, flat={n_flat}, nested={n_nested}).",
            flush=True,
        )
        return chunk_paths

    def _load_index(self) -> None:
        self.chunk_paths = load_or_build(
            log_prefix="ScanNetppLatent",
            local_path=self._cache_path(),
            shared_path=self._shared_cache_path(),
            build_fn=self._build_index,
        )
        print(
            f"[ScanNetppLatent] split={self.split}  chunks={len(self.chunk_paths)}",
            flush=True,
        )

    # ── Required-by-EasyDataset bits ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self.chunk_paths)

    def get_stats(self) -> str:
        return f"{len(self)} chunks"

    def __repr__(self) -> str:
        return (f"ScanNetppLatent_Multi(ROOT={self.ROOT}, "
                f"V={self.num_views}, split={self.split}, "
                f"chunks={len(self)})")

    def set_epoch(self, epoch: int) -> None:
        # Latents are static — nothing per-epoch to refresh.
        pass

    # ── Sampler (overridden because we don't use multi-resolution / cut3r tuple idx) ──

    def make_sampler(self, batch_size, shuffle=True, drop_last=True,
                     world_size=1, rank=0, fixed_length=False):
        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios=1,            # one fixed resolution
            num_of_views=self.num_views if fixed_length else 4,
            max_num_of_views=self.num_views,
            world_size=world_size,
            rank=rank,
            warmup=1,
            drop_last=drop_last,
        )
        return BatchedRandomSampler(sampler, batch_size, drop_last)

    # ── Item access ──────────────────────────────────────────────────────

    def __getitem__(self, idx) -> dict[str, Any]:
        if isinstance(idx, (tuple, list)):
            # CustomRandomSampler emits (idx, ar_idx, nview); we only care about idx.
            idx = idx[0]
        if not (0 <= idx < len(self.chunk_paths)):
            raise IndexError(idx)

        chunk_path = self.chunk_paths[idx]
        try:
            obj = torch.load(chunk_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {chunk_path}: {e}") from e

        z_all = obj["z_all"]          # (V, C, h, w) bf16
        z_ref = obj["z_ref"]          # (1, C, h, w) bf16
        if z_all.shape[0] != self.num_views:
            raise RuntimeError(
                f"V mismatch in {chunk_path}: chunk V={z_all.shape[0]} vs config V={self.num_views}"
            )

        return {
            "is_precomputed": True,
            "z_all": z_all,                              # (V, C, h, w) bf16
            "z_ref": z_ref,                              # (1, C, h, w) bf16
            "c2w": obj["c2w"].float(),                   # (V, 4, 4)
            "intrinsics": obj["intrinsics"].float(),     # (V, 3, 3)
            "frame_idx_in_mp4": obj["frame_idx_in_mp4"], # (V,) int32
            "scene_id": obj["scene_id"],
            "caption": "",  # ScanNet++ has no captions; empty for compat with cap-aware loops
        }
