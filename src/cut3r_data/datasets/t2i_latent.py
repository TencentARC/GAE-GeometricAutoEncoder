"""Flat per-frame T2I latent dataset (V=1).

One sample per ``.pt`` produced by either:
  - ``scripts/precompute_t2i_latents.py``           (BLIP3o / JourneyDB / ImageNet)
  - ``scripts/precompute_scannetpp_t2i.py``         (ScanNet++ video frames)

Both write the same schema::

    {
        "z_all":       (1, C, h, w) bf16   — diffusion target
        "z_ref":       (1, C, h, w) bf16   — alias for I2V compat
        "c2w":         (1, 4, 4) fp32      — identity (camera_drop=1.0 zeros plücker)
        "intrinsics":  (1, 3, 3) fp32      — placeholder
        "caption":     str                 — raw text (UMT5 cache uses caption_key)
        "caption_key": str                 — UMT5 lookup key
        "source":      str                 — "blip3o-journeydb", "scannetpp-short", ...
        "tier":        int                 — sampling priority (smaller = higher pref)
        "is_t2i":      True
        "is_t2v":      True                — forces trainer's cond_num=0 path
        "V":           1
        ...
    }

Layout under ``ROOT``::

    <ROOT>/<source>/<bucket>/<file>.pt          # BLIP3o: source/tar_stem/<n>.pt
    <ROOT>/<source>/<scene>/<clip>__f<idx>.pt   # ScanNet++ short
    <ROOT>/<source>/<scene>/f<idx>.pt           # ScanNet++ long (no clip)

One instance handles ONE source; mix multiple sources via config-level weighted
concat — keeps index sizes bounded and makes per-source skip/test trivial.

Split: hash on ``caption_key`` so all frames sharing a caption (per-clip ScanNet++
or per-image BLIP3o) land in the same split — no leakage.
"""
from __future__ import annotations

import hashlib
import os
import os.path as osp
import pickle
import random
import time
from pathlib import Path
from typing import Any

import torch

from cut3r_data.base.easy_dataset import EasyDataset
from cut3r_data.base.batched_sampler import BatchedRandomSampler, CustomRandomSampler


class T2ILatentFlat_Multi(EasyDataset):
    """Single-source flat T2I latent dataset (V=1).

    Args:
        ROOT:        Root for ALL t2i source subdirs, e.g.
                     ``$GAE_DATA_ROOT/t2i_latents``.
        source:      Source subdir name under ROOT (one per instance).
                     E.g. "scannetpp-short", "scannetpp-long", "blip3o-journeydb".
        num_views:   Must be 1 (V=1 t2i).
        resolution:  Sampler-compat; latent shape is fixed by precompute.
        split:       Optional "train" / "val" via caption_key hash.
        val_frac:    Fraction reserved for val (default 0.005 — t2i needs tiny val).
        recursive_depth: How many directory levels under ROOT/source/ to walk for
                         .pt files. 2 covers both BLIP3o (tar_stem/file.pt) and
                         ScanNet++ (scene/file.pt). Set higher only if a new
                         source nests deeper.
    """

    def __init__(
        self,
        *,
        ROOT: str,
        source: str,
        num_views: int = 1,
        resolution: list[tuple[int, int]] | tuple[int, int] = (252, 252),
        split: str | None = None,
        val_frac: float = 0.005,
        allow_repeat: bool = False,
        ref_view_sampling: str = "prefix",
        seed: int | None = None,
        recursive_depth: int = 2,
        **_ignored_kwargs,
    ):
        if int(num_views) != 1:
            raise ValueError(
                f"T2ILatentFlat_Multi expects num_views=1 (V=1 t2i); got {num_views}"
            )
        self.ROOT = ROOT
        self.source = source
        self.num_views = 1
        self.split = split
        self.val_frac = float(val_frac)
        self.allow_repeat = bool(allow_repeat)
        self.ref_view_sampling = ref_view_sampling
        self.seed = seed
        self.recursive_depth = int(recursive_depth)

        if isinstance(resolution, tuple):
            self._resolutions = [tuple(resolution)]
        else:
            self._resolutions = [tuple(resolution[0])]

        self.is_metric = True
        self.video = False

        self._load_index()

    # ── Indexing ─────────────────────────────────────────────────────────

    def _source_root(self) -> str:
        return osp.join(self.ROOT, self.source)

    def _cache_path(self) -> str:
        """Local-ssd-backed pickle cache. Key contains source + ROOT + depth so
        bumping any of them invalidates correctly."""
        key = f"t2i_flat_v1__{self.ROOT}__{self.source}__d{self.recursive_depth}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        local_root = Path("/local-ssd/t2i_latent_cache")
        if Path("/local-ssd").is_dir():
            local_root.mkdir(parents=True, exist_ok=True)
            return str(local_root / f"t2i_index_{self.source}_{tag}.pkl")
        return osp.join(self.ROOT, self.source, f".t2i_index_{tag}.pkl")

    def _walk_pt_files(self, root: str, depth: int) -> list[str]:
        """Walk ROOT/source/ for .pt files up to ``depth`` directory levels.
        Returns ROOT-relative paths so the cache is portable across mounts.
        """
        out: list[str] = []
        # iterative DFS bounded by depth to avoid recursing into stray symlinks
        stack: list[tuple[str, int]] = [(root, 0)]
        while stack:
            cur, d = stack.pop()
            try:
                entries = os.listdir(cur)
            except OSError:
                continue
            for name in entries:
                p = osp.join(cur, name)
                if osp.isfile(p):
                    if name.endswith(".pt") and not name.startswith("."):
                        out.append(p)
                elif osp.isdir(p) and d < depth:
                    stack.append((p, d + 1))
        out.sort()
        return out

    def _build_index(self) -> list[str]:
        src_root = self._source_root()
        if not osp.isdir(src_root):
            raise RuntimeError(
                f"T2ILatentFlat source not found: {src_root}"
            )
        t0 = time.time()
        print(f"[T2ILatent/{self.source}] scanning {src_root} "
              f"(depth={self.recursive_depth}) ...", flush=True)
        paths = self._walk_pt_files(src_root, self.recursive_depth)
        dt = time.time() - t0
        print(f"[T2ILatent/{self.source}] {len(paths)} .pt files "
              f"({dt:.1f}s walk)", flush=True)
        return paths

    def _load_or_build_index(self) -> list[str]:
        cache_path = self._cache_path()
        if osp.isfile(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    paths = pickle.load(f)
                print(f"[T2ILatent/{self.source}] loaded {len(paths)} paths "
                      f"from cache {cache_path}", flush=True)
                return paths
            except (pickle.UnpicklingError, EOFError, OSError) as e:
                print(f"[T2ILatent/{self.source}] cache corrupt ({e}); rebuilding",
                      flush=True)
        paths = self._build_index()
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(paths, f)
            os.replace(tmp, cache_path)
            print(f"[T2ILatent/{self.source}] cached index → {cache_path}",
                  flush=True)
        except OSError as e:
            print(f"[T2ILatent/{self.source}] cache write failed ({e}); "
                  f"will rebuild next run", flush=True)
        return paths

    def _sample_in_split(self, caption_key: str) -> bool:
        """Hash on caption_key so all frames of one clip / image stay together."""
        if self.split is None:
            return True
        h = int(hashlib.md5(caption_key.encode()).hexdigest()[:8], 16) / 0xffffffff
        is_val = h < self.val_frac
        return (self.split == "val") if is_val else (self.split == "train")

    def _caption_key_from_path(self, path: str) -> str:
        """Approximate caption_key from path WITHOUT loading the .pt.
        Used for split filtering at index time. Format must mirror
        precompute writers' caption_key construction so the hash matches.

        - ScanNet++ short:  <scene>/<clip>__f<idx>.pt
            file stem is "<clip>__f<idx>"; caption_key was
            "scannetpp__<scene>__<clip>__f<idx>" but the per-CLIP key (shared
            across frames) is "scannetpp__<scene>__<clip>__f000000". To keep
            all frames of one clip together, hash on (source, scene, clip).
        - ScanNet++ long:   <scene>/f<idx>.pt → hash on (source, scene)
        - BLIP3o:           <tar_stem>/<n>.pt → hash on (source, tar_stem, n)
                            (each image is its own caption — no sharing)
        """
        rel = osp.relpath(path, self._source_root())
        parts = rel.split(os.sep)
        if len(parts) < 2:
            return f"{self.source}/{rel}"  # unexpected shape

        scene_or_bucket = parts[0]
        fname = parts[-1]
        stem = fname[:-3] if fname.endswith(".pt") else fname

        if self.source.startswith("scannetpp-short"):
            # stem = "<clip>__f<idx>"; group by clip
            clip = stem.rsplit("__f", 1)[0] if "__f" in stem else stem
            return f"{self.source}/{scene_or_bucket}/{clip}"
        if self.source.startswith("scannetpp-long"):
            return f"{self.source}/{scene_or_bucket}"
        # BLIP3o-style: tar_stem/<n>; one caption per file
        return f"{self.source}/{scene_or_bucket}/{stem}"

    def _load_index(self) -> None:
        full = self._load_or_build_index()
        if self.split is None:
            self.paths = full
        else:
            self.paths = [
                p for p in full if self._sample_in_split(self._caption_key_from_path(p))
            ]
        print(f"[T2ILatent/{self.source}] split={self.split}  "
              f"items={len(self.paths)} (of {len(full)} total)", flush=True)

    # ── EasyDataset bits ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.paths)

    def get_stats(self) -> str:
        return f"{len(self)} t2i frames"

    def __repr__(self) -> str:
        return (f"T2ILatentFlat_Multi(source={self.source}, split={self.split}, "
                f"items={len(self)})")

    def set_epoch(self, epoch: int) -> None:
        pass

    def make_sampler(self, batch_size, shuffle=True, drop_last=True,
                     world_size=1, rank=0, fixed_length=False):
        sampler = CustomRandomSampler(
            self,
            batch_size,
            num_of_aspect_ratios=1,
            num_of_views=1,
            max_num_of_views=1,
            world_size=world_size,
            rank=rank,
            warmup=1,
            drop_last=drop_last,
        )
        return BatchedRandomSampler(sampler, batch_size, drop_last)

    # ── Item access ───────────────────────────────────────────────────────

    _MAX_LOAD_RETRIES = 8
    _CORRUPT_MARKERS = (
        "central directory", "pytorchstreamreader", "failed reading zip",
        "zip archive", "invalid load key", "unpicklingerror", "truncated",
    )

    def _load_sample(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        obj = torch.load(path, map_location="cpu", weights_only=False)

        z_all = obj["z_all"]      # (1, C, h, w) bf16
        z_ref = obj.get("z_ref", z_all)
        if z_all.shape[0] != 1:
            raise RuntimeError(
                f"V mismatch in {path}: t2i expects V=1, got {z_all.shape[0]}"
            )

        # caption_key for UMT5 lookup. Always prefer the stored field.
        caption_key = obj.get("caption_key")
        if not caption_key:
            raise RuntimeError(f"{path}: missing caption_key field")

        return {
            "is_precomputed": True,
            "z_all": z_all,
            "z_ref": z_ref,
            "c2w": obj["c2w"].float() if "c2w" in obj else torch.eye(4).unsqueeze(0),
            "intrinsics": (
                obj["intrinsics"].float() if "intrinsics" in obj
                else torch.eye(3).unsqueeze(0)
            ),
            "caption": caption_key,   # trainer uses this field as UMT5 cache key
            "is_t2v": True,           # cond_num=0 path
            "is_t2i": True,           # informational
            "disable_plucker": True,  # explicit: zero plücker for t2i
        }

    def __getitem__(self, idx) -> dict[str, Any]:
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        n = len(self.paths)
        if not (0 <= idx < n):
            raise IndexError(idx)

        cur = idx
        last_err: Exception | None = None
        for _ in range(self._MAX_LOAD_RETRIES):
            try:
                return self._load_sample(cur)
            except Exception as e:
                last_err = e
                bad = self.paths[cur]
                print(f"[T2ILatent/{self.source}] WARN: skipping bad sample {bad}: {e}",
                      flush=True)
                cur = random.randint(0, n - 1)
        raise RuntimeError(
            f"T2ILatent/{self.source}: {self._MAX_LOAD_RETRIES} consecutive "
            f"load failures (last: {last_err})"
        )
