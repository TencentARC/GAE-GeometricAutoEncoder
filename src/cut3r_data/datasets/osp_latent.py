"""
OSP iStock pre-computed latent dataset for camera-aware T2V training (V=33).

Each sample is one chunk file produced by
``scripts/precompute_osp_t2v_latents.py``::

    ROOT/<caption_key>/chunk_<chunk_idx:03d>.pt

Each chunk holds::

    {
        "z_all":       (V, C, h, w) bf16  — diffusion target (cross-view DA3)
        "z_ref":       (1, C, h, w) bf16  — frame-0 single-view feature (I2V prep)
        "c2w":         (V, 4, 4) fp32     — interpolated chunk-local poses
        "intrinsics":  (V, 3, 3) fp32
        "caption_key": str                — UMT5 cache lookup key
        "chunk_idx":   int
        "is_t2v":      True
        ...
    }

Vs ``ScanNetppLatent_Multi``: this emits the per-video ``caption`` key (for the
pre-encoded UMT5 cache lookup) plus ``is_t2v=True`` / ``disable_plucker=False``,
so the precomputed trainer runs the pure-T2V (cond_num=0) path while keeping
Plücker as a camera-control signal. ``z_ref`` is carried for forward-compat /
future I2V but is unused by the T2V loss.

Like ``ScanNetppLatent_Multi`` it is a plain ``EasyDataset`` (no DA3/VAE).
``split`` is hashed on ``<caption_key>`` (top-level dir) so all chunks of one
video land in the same split (no train/val leakage across a video).
"""
from __future__ import annotations

import hashlib
import os
import os.path as osp
import random
import shutil
import subprocess
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


class OSPLatent_Multi(EasyDataset):
    """Pre-computed OSP T2V latent chunks → one dict per sample.

    Args:
        ROOT:        Latent root, e.g. ``$GAE_DATA_ROOT/osp_istock_t2v_latents``.
        num_views:   Expected V per chunk (33); mismatching chunks raise.
        resolution:  Kept for sampler compatibility; latents are fixed-shape.
        split:       Optional "train"/"val" via deterministic caption_key hash.
        val_frac:    Fraction of videos reserved for val (default 0.02).
        ref_view_sampling: Always "prefix" (z_ref fixed at frame 0).
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
        **_ignored_kwargs,
    ):
        self.ROOT = ROOT
        self.num_views = int(num_views)
        self.split = split
        self.val_frac = float(val_frac)
        self.allow_repeat = bool(allow_repeat)
        self.ref_view_sampling = ref_view_sampling
        self.seed = seed
        # 命中明确损坏的 chunk 时是否顺手删掉源文件（不可逆）。默认关；
        # 设 OSP_DELETE_CORRUPT=1 开启。需挂载允许删除 (mountpoint-s3 --allow-delete)。
        self._delete_corrupt = (
            os.environ.get("OSP_DELETE_CORRUPT", "").strip().lower()
            in ("1", "true", "yes")
        )

        if isinstance(resolution, tuple):
            self._resolutions = [tuple(resolution)]
        else:
            self._resolutions = [tuple(resolution[0])]

        self.is_metric = True
        self.video = True

        self._load_index()

    # ── Indexing ─────────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        # split-independent: cache the FULL chunk list once; train/val share it
        # and filter in-process at load (see _load_index). v2 marks the new
        # S3-LIST builder + full-list semantics (old v1 per-split caches differ).
        key = f"osp_latent_v2_all__{self.ROOT}__V{self.num_views}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        local_root = Path("/local-ssd/osp_latent_cache")
        if Path("/local-ssd").is_dir():
            local_root.mkdir(parents=True, exist_ok=True)
            return str(local_root / f"osp_latent_index_{tag}.pkl")
        return osp.join(self.ROOT, f".osp_latent_index_{tag}.pkl")

    def _shared_cache_path(self) -> str | None:
        local = self._cache_path()
        shared = mirror_shared_path(local, resolve_shared_cache_dir())
        if shared is not None:
            return shared
        shared_root = resolve_shared_cache_dir()
        if shared_root is None:
            return None
        return osp.join(shared_root, "osp_latent_cache", osp.basename(local))

    def _sample_in_split(self, caption_key: str) -> bool:
        if self.split is None:
            return True
        h = int(hashlib.md5(caption_key.encode()).hexdigest()[:8], 16) / 0xffffffff
        is_val = h < self.val_frac
        return (self.split == "val") if is_val else (self.split == "train")

    def _s3_uri(self) -> str | None:
        """Map the FUSE ROOT to its backing ``s3://`` URI, or ``None`` if the
        path is not on a known mountpoint-s3 mount.

        ``OSP_LATENT_NO_S3=1`` forces the os.listdir path; ``OSP_LATENTS_S3``
        overrides with an explicit ``s3://bucket/prefix`` for the latents root.
        """
        if os.environ.get("OSP_LATENT_NO_S3", "").strip() in ("1", "true", "yes"):
            return None
        env = os.environ.get("OSP_LATENTS_S3", "").strip()
        if env:
            return env.rstrip("/")
        # mountpoint-s3 mounts (see `mount | grep mountpoint-s3`).
        # Set OSP_LATENTS_S3="s3://bucket/prefix" to enable the fast s5cmd path.
        mount_to_bucket = {}
        root = osp.abspath(self.ROOT)
        for mnt, bucket in mount_to_bucket.items():
            if root == mnt or root.startswith(mnt + "/"):
                rel = root[len(mnt):].lstrip("/")
                return f"{bucket}/{rel}".rstrip("/")
        return None

    def _list_chunks_via_s3(self, s3_uri: str) -> list[str] | None:
        """List every ``chunk_*.pt`` under ``s3_uri`` via one recursive S3 LIST
        (s5cmd) — ~100x faster than walking 76k dirs over FUSE. Returns
        ROOT-relative ``<cap>/chunk_xxx.pt`` paths, or ``None`` to signal the
        caller to fall back to os.listdir (s5cmd missing / list failed)."""
        s5 = shutil.which(os.environ.get("S5CMD", "s5cmd"))
        if s5 is None:
            print("[OSPLatent] s5cmd not found; fallback to os.listdir", flush=True)
            return None
        try:
            proc = subprocess.run(
                [s5, "ls", f"{s3_uri}/*"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[OSPLatent] s5cmd ls failed ({e}); fallback to os.listdir",
                  flush=True)
            return None
        rels: list[str] = []
        for line in proc.stdout.splitlines():
            # object line: "<date> <time> <size> <key>"; DIR line: "DIR <key>".
            # split(None, 3) keeps any whitespace inside <key> intact.
            parts = line.split(None, 3)
            if len(parts) != 4:
                continue
            key = parts[3]
            cap, _, fn = key.rpartition("/")
            if cap and fn.startswith("chunk_") and fn.endswith(".pt"):
                rels.append(key)
        rels.sort()
        return rels

    def _list_chunks_via_fs(self) -> list[str]:
        """Fallback: walk ROOT with os.listdir. Returns ROOT-relative paths."""
        rels: list[str] = []
        sample_dirs = sorted(
            d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))
        )
        for cap in sample_dirs:
            sample_dir = osp.join(self.ROOT, cap)
            for fn in sorted(os.listdir(sample_dir)):
                if fn.startswith("chunk_") and fn.endswith(".pt"):
                    rels.append(f"{cap}/{fn}")
        return rels

    def _build_index(self) -> list[str]:
        """Index ALL chunk_*.pt under ROOT (split filtering happens at load, so
        train+val share one cache instead of scanning the tree twice)."""
        print(f"[OSPLatent] scanning {self.ROOT} …", flush=True)
        s3_uri = self._s3_uri()
        rels = self._list_chunks_via_s3(s3_uri) if s3_uri else None
        src = f"s3({s3_uri})" if rels is not None else "os.listdir"
        if rels is None:
            rels = self._list_chunks_via_fs()
        chunk_paths = [osp.join(self.ROOT, r) for r in rels]
        n_videos = len({r.split("/", 1)[0] for r in rels})
        print(
            f"[OSPLatent] {len(chunk_paths)} chunks indexed via {src} "
            f"(videos={n_videos}).",
            flush=True,
        )
        return chunk_paths

    def _load_index(self) -> None:
        full = load_or_build(
            log_prefix="OSPLatent",
            local_path=self._cache_path(),
            shared_path=self._shared_cache_path(),
            build_fn=self._build_index,
        )
        if self.split is None:
            self.chunk_paths = full
        else:
            self.chunk_paths = [
                p for p in full
                if self._sample_in_split(osp.basename(osp.dirname(p)))
            ]
        print(
            f"[OSPLatent] split={self.split}  chunks={len(self.chunk_paths)} "
            f"(of {len(full)} total)",
            flush=True,
        )

    # ── EasyDataset bits ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.chunk_paths)

    def get_stats(self) -> str:
        return f"{len(self)} chunks"

    def __repr__(self) -> str:
        return (f"OSPLatent_Multi(ROOT={self.ROOT}, V={self.num_views}, "
                f"split={self.split}, chunks={len(self)})")

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

    # ── Item access ───────────────────────────────────────────────────────

    # 单个坏样本最多换几次 index；连续都失败基本是系统性损坏，应抛出而非空转。
    _MAX_LOAD_RETRIES = 8
    # 只对「明确的文件损坏」特征才考虑删除；瞬时 IO/限流错误只跳过不删，避免误删好文件。
    _CORRUPT_MARKERS = (
        "central directory", "pytorchstreamreader", "failed reading zip",
        "zip archive", "invalid load key", "unpicklingerror", "truncated",
    )

    def _load_sample(self, idx: int) -> dict[str, Any]:
        """Load one chunk → sample dict. Raises on corrupt/invalid chunk."""
        chunk_path = self.chunk_paths[idx]
        obj = torch.load(chunk_path, map_location="cpu", weights_only=False)

        z_all = obj["z_all"]          # (V, C, h, w) bf16
        z_ref = obj["z_ref"]          # (1, C, h, w) bf16
        if z_all.shape[0] != self.num_views:
            raise RuntimeError(
                f"V mismatch in {chunk_path}: chunk V={z_all.shape[0]} "
                f"vs config V={self.num_views}"
            )

        # caption_key drives the UMT5 cache lookup. Prefer the stored field;
        # fall back to the parent dir name (== caption_key by construction).
        caption_key = obj.get("caption_key") or osp.basename(osp.dirname(chunk_path))

        return {
            "is_precomputed": True,
            "z_all": z_all,                              # (V, C, h, w) bf16
            "z_ref": z_ref,                              # (1, C, h, w) bf16
            "c2w": obj["c2w"].float(),                   # (V, 4, 4)
            "intrinsics": obj["intrinsics"].float(),     # (V, 3, 3)
            "caption": caption_key,                      # UMT5 cache key
            "is_t2v": True,                              # force cond_num=0 path
            "disable_plucker": False,                    # keep Plücker (camera-aware)
        }

    def __getitem__(self, idx) -> dict[str, Any]:
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        n = len(self.chunk_paths)
        if not (0 <= idx < n):
            raise IndexError(idx)

        # 个别 chunk 可能在 precompute/上传时被写坏或截断（torch.load 报
        # "failed finding central directory" 等）。单个坏样本不应拖垮整个多机
        # job：随机换一个 index 重试，连续多次都失败再抛（多半是系统性损坏）。
        cur = idx
        last_err: Exception | None = None
        for _ in range(self._MAX_LOAD_RETRIES):
            try:
                return self._load_sample(cur)
            except Exception as e:
                last_err = e
                bad_path = self.chunk_paths[cur]
                action = ""
                # 仅对明确损坏特征删源文件（best-effort）；瞬时错误不删。
                is_corrupt = any(m in str(e).lower() for m in self._CORRUPT_MARKERS)
                if is_corrupt and self._delete_corrupt:
                    try:
                        os.remove(bad_path)
                        action = " (已删除损坏文件)"
                    except OSError as de:
                        action = f" (删除失败: {de})"
                print(f"[OSPLatent] WARN: 跳过坏样本 {bad_path}: {e}{action}",
                      flush=True)
                cur = random.randint(0, n - 1)
        raise RuntimeError(
            f"OSPLatent: 连续 {self._MAX_LOAD_RETRIES} 个样本加载失败，"
            f"疑似系统性损坏；最后错误: {last_err}"
        )
