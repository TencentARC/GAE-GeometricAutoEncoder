"""
RE10K_Packed pre-computed latent dataset for 504² V=33 cond1-4 i2v training.

Each sample is one ``win_*.pt`` produced by
``scripts/precompute_re10k_504_v33_latents.py``::

    ROOT/<scene_id>/win_<start_pos:05d>.pt

Each window holds::

    {
        "z_all":         (V, C, h, w) bf16  — all-view DA3 ctx (diffusion target)
        "z_ref_1..4":    (k, C, h, w) bf16  — ref-only DA3 ctx over first k frames
        "c2w":           (V, 4, 4) fp32
        "intrinsics":    (V, 3, 3) fp32
        "frame_indices": (V,) int64
        "caption":       str
        "scene_id":      str
        ...
    }

The trainer samples ``cond_num`` in 1-4 per step and picks the matching
``z_ref_k`` from the batch. This mirrors the online ``prepare_data`` path.
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


class RE10KLatent_Multi(EasyDataset):
    """Pre-computed RE10K_Packed latent windows → dict batch for RAE trainer."""

    def __init__(
        self,
        *,
        ROOT: str,
        num_views: int,
        resolution: list[tuple[int, int]] | tuple[int, int],
        cond_max: int = 4,
        split: str | None = None,
        val_frac: float = 0.02,
        allow_repeat: bool = False,
        ref_view_sampling: str = "prefix",
        seed: int | None = None,
        **_ignored_kwargs,
    ):
        self.ROOT = ROOT
        self.num_views = int(num_views)
        self.cond_max = int(cond_max)
        self.split = split
        self.val_frac = float(val_frac)
        self.allow_repeat = bool(allow_repeat)
        self.ref_view_sampling = ref_view_sampling
        self.seed = seed

        if isinstance(resolution, tuple):
            self._resolutions = [tuple(resolution)]
        else:
            self._resolutions = [tuple(resolution[0])]

        self.is_metric = True
        self.video = True

        self._load_index()

    def _cache_path(self) -> str:
        key = (
            f"re10k_latent_v1__{self.ROOT}__V{self.num_views}"
            f"__cond{self.cond_max}__split{self.split}"
        )
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        local_root = Path("/local-ssd/re10k_latent_cache")
        if Path("/local-ssd").is_dir():
            local_root.mkdir(parents=True, exist_ok=True)
            return str(local_root / f"re10k_index_{tag}.pkl")
        return osp.join(self.ROOT, f".re10k_latent_index_{tag}.pkl")

    def _shared_cache_path(self) -> str | None:
        local = self._cache_path()
        shared = mirror_shared_path(local, resolve_shared_cache_dir())
        if shared is not None:
            return shared
        shared_root = resolve_shared_cache_dir()
        if shared_root is None:
            return None
        return osp.join(shared_root, "re10k_latent_cache", osp.basename(local))

    def _scene_in_split(self, scene_id: str) -> bool:
        if self.split is None:
            return True
        h = int(hashlib.md5(scene_id.encode()).hexdigest()[:8], 16) / 0xffffffff
        is_val = h < self.val_frac
        return (self.split == "val") if is_val else (self.split == "train")

    def _build_index(self) -> list[str]:
        print(f"[RE10KLatent] scanning {self.ROOT} …", flush=True)
        window_paths: list[str] = []
        scene_dirs = sorted(
            d for d in os.listdir(self.ROOT)
            if osp.isdir(osp.join(self.ROOT, d)) and not d.startswith("_")
        )
        for sid in scene_dirs:
            if not self._scene_in_split(sid):
                continue
            scene_dir = osp.join(self.ROOT, sid)
            for fn in sorted(os.listdir(scene_dir)):
                if fn.startswith("win_") and fn.endswith(".pt"):
                    window_paths.append(osp.join(scene_dir, fn))
        print(
            f"[RE10KLatent] {len(window_paths)} windows indexed "
            f"(split={self.split}, scenes={len(scene_dirs)}).",
            flush=True,
        )
        return window_paths

    def _load_index(self) -> None:
        self.window_paths = load_or_build(
            log_prefix="RE10KLatent",
            local_path=self._cache_path(),
            shared_path=self._shared_cache_path(),
            build_fn=self._build_index,
        )
        print(
            f"[RE10KLatent] split={self.split}  windows={len(self.window_paths)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.window_paths)

    def get_stats(self) -> str:
        return f"{len(self)} windows"

    def __repr__(self) -> str:
        return (
            f"RE10KLatent_Multi(ROOT={self.ROOT}, V={self.num_views}, "
            f"cond_max={self.cond_max}, split={self.split}, windows={len(self)})"
        )

    def set_epoch(self, epoch: int) -> None:
        pass

    def make_sampler(self, batch_size, shuffle=True, drop_last=True,
                     world_size=1, rank=0, fixed_length=False):
        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios=1,
            num_of_views=self.num_views if fixed_length else 4,
            max_num_of_views=self.num_views,
            world_size=world_size,
            rank=rank,
            warmup=1,
            drop_last=drop_last,
        )
        return BatchedRandomSampler(sampler, batch_size, drop_last)

    def __getitem__(self, idx) -> dict[str, Any]:
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        if not (0 <= idx < len(self.window_paths)):
            raise IndexError(idx)

        window_path = self.window_paths[idx]
        try:
            obj = torch.load(window_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {window_path}: {e}") from e

        z_all = obj["z_all"]
        if z_all.shape[0] != self.num_views:
            raise RuntimeError(
                f"V mismatch in {window_path}: chunk V={z_all.shape[0]} "
                f"vs config V={self.num_views}"
            )

        out: dict[str, Any] = {
            "is_precomputed": True,
            "z_all": z_all,
            "c2w": obj["c2w"].float(),
            "intrinsics": obj["intrinsics"].float(),
            "scene_id": obj["scene_id"],
            "caption": str(obj.get("caption", "")),
        }

        frame_idx = obj.get("frame_indices")
        if frame_idx is None:
            frame_idx = obj.get("frame_idx")
        if frame_idx is not None:
            frame_idx = torch.as_tensor(frame_idx).long()
            out["frame_indices"] = frame_idx
            out["physical_frame_indices"] = frame_idx

        for k in range(1, self.cond_max + 1):
            key = f"z_ref_{k}"
            if key not in obj:
                raise RuntimeError(f"Missing {key} in {window_path}")
            out[key] = obj[key]

        return out
