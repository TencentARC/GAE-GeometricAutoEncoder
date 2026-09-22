"""OSP iStock RGB dataset for camera-aware Text-to-Video training.

每个 sample = 1 chunk = 33 raw video frames，跨 chunk 之间不 overlap。
DA3 估的 9 个 anchor pose（chunk-local raw idx [0,4,8,...,32]）经 slerp + lerp 插值
扩展到 33 帧，提供给 DiT 作为 per-frame plucker 输入。

Layout:
    FEATURE_ROOT (.pt files):
        <sample_id>_<hash>_<start>-<end>_<N>_<H>_<W>.pt
            ├── vae_latent      (chunks, 16, 9, 48, 80)  ← Wan VAE，本类不用
            ├── prompt_embed    (512, 4096)              ← UMT5-XXL，由 extract 脚本另存
            └── ...
    CAMERA_ROOT (.json files): 同名 .json，结构见 _load_meta。

JSON 里的视频路径如与本机布局不同，通过 ``VIDEO_ROOT_REMAP`` 改写。

Config example::

    dataset_osp: OSPIstockRGB_Multi(
        FEATURE_ROOT="${GAE_DATA_ROOT}/osp/osp_istock_camera_features",
        CAMERA_ROOT="${GAE_DATA_ROOT}/osp/osp_istock_camera_features_camera_json",
        VIDEO_ROOT_REMAP=[],
        num_views=33,
        resolution=[(504, 504)],
        pose_frame="rel_chunk",
        min_valid_ratio=1.0,
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R_scipy, Slerp

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.base.dataset_index_cache import (
    load_or_build,
    resolve_shared_cache_dir,
)
from cut3r_data.base.fast_meta_index import _log_index_progress


# 数据集出厂约定：每 chunk 33 raw frames、9 anchor，anchor 在 chunk-local raw idx
# [0, 4, 8, 12, 16, 20, 24, 28, 32]（latent_window_size=9, Wan VAE 4x time stride）。
_FRAMES_PER_CHUNK = 33
_ANCHORS_PER_CHUNK = 9
_ANCHOR_STRIDE = 4
_EXPECTED_ANCHOR_LOCAL = tuple(range(0, _FRAMES_PER_CHUNK, _ANCHOR_STRIDE))  # (0, 4, ..., 32)

# Prebuilt chunk-mode index for the default CAMERA_ROOT + vr=1.0 + pf=rel_chunk
# + req=0. Env ``OSP_INDEX_PATH`` overrides this. Only consulted in chunk mode
# (num_views <= 33); clip mode builds its own index. Tag must match ``_cache_tag()``.
_DEFAULT_OSP_INDEX = os.path.join(
    os.environ.get("GAE_CACHE_DIR", "cache"), "osp_index_7792fe6eb91e.pkl"
)


def _default_index_workers() -> int:
    """Parallel JSON parsers for index build. Override via ``OSP_INDEX_WORKERS``."""
    raw = os.environ.get("OSP_INDEX_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    ncpu = os.cpu_count() or 8
    return max(8, min(64, ncpu * 2))


def _osp_index_backend() -> str:
    return os.environ.get("OSP_INDEX_BACKEND", "thread").strip().lower()


def _json_loads(raw: bytes | str) -> object:
    try:
        import orjson  # type: ignore[import-not-found]

        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        return orjson.loads(raw)
    except ImportError:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)


def _parse_one_camera_json(
    json_path: str,
    min_valid_ratio: float,
    require_feature_pt: bool,
    feature_root: str,
) -> tuple[list[tuple[str, int]], bool, int]:
    """Parse one camera JSON → (chunk index entries, json_bad, n_chunks_dropped)."""
    try:
        with open(json_path, "rb") as f:
            meta = _json_loads(f.read())
    except (json.JSONDecodeError, OSError, ValueError, UnicodeDecodeError):
        return [], True, 0

    num_chunks = int(meta.get("camera_num_chunks", 0))
    num_anchors = int(meta.get("camera_num_anchors_per_chunk", 0))
    if num_chunks <= 0 or num_anchors != _ANCHORS_PER_CHUNK:
        return [], True, 0

    if require_feature_pt:
        pt_path = osp.join(feature_root, osp.basename(json_path).replace(".json", ".pt"))
        if not osp.isfile(pt_path):
            return [], True, 0

    chunks: list[tuple[str, int]] = []
    n_dropped = 0
    valid_masks = meta.get("camera_valid_mask", [])
    for ci in range(num_chunks):
        if ci >= len(valid_masks):
            n_dropped += 1
            continue
        vm = valid_masks[ci]
        if not vm or len(vm) != _ANCHORS_PER_CHUNK:
            n_dropped += 1
            continue
        vr = sum(bool(x) for x in vm) / float(_ANCHORS_PER_CHUNK)
        if vr < min_valid_ratio:
            n_dropped += 1
            continue
        chunks.append((json_path, ci))

    return chunks, False, n_dropped


def _parse_one_camera_json_task(
    args: tuple[str, float, bool, str],
) -> tuple[list[tuple[str, int]], bool, int]:
    return _parse_one_camera_json(*args)


def _parse_one_camera_json_clip(args):
    """Clip 粒度解析（cross-clip 长序列模式，num_views > 33）。

    Returns ``(entry_or_None, json_bad, drop_reason)``，其中
    ``entry = (json_path, max_valid_anchor_idx, num_raw_frames, scale_factor)``。
    drop_reason ∈ {"", "invalid", "short", "static"}（json_bad=False 时才看）。
    """
    (json_path, min_valid_ratio, min_metric_motion, min_span,
     require_feature_pt, feature_root) = args
    try:
        with open(json_path, "rb") as f:
            meta = _json_loads(f.read())
    except (json.JSONDecodeError, OSError, ValueError, UnicodeDecodeError):
        return None, True, ""

    num_chunks = int(meta.get("camera_num_chunks", 0))
    num_anchors = int(meta.get("camera_num_anchors_per_chunk", 0))
    if num_chunks <= 0 or num_anchors != _ANCHORS_PER_CHUNK:
        return None, True, ""

    if require_feature_pt:
        pt_path = osp.join(feature_root, osp.basename(json_path).replace(".json", ".pt"))
        if not osp.isfile(pt_path):
            return None, True, ""

    try:
        anchor_idx = np.asarray(meta["camera_anchor_frame_idx"], dtype=np.int64).reshape(-1)
        valid = np.asarray(meta["camera_valid_mask"]).reshape(-1).astype(bool)
        pose = np.asarray(meta["camera_pose_rel_video_c2w"], dtype=np.float64).reshape(-1, 4, 4)
    except (KeyError, ValueError, TypeError):
        return None, True, ""

    if not (anchor_idx.shape[0] == valid.shape[0] == pose.shape[0]) or anchor_idx.shape[0] < 2:
        return None, True, ""

    if float(valid.sum()) / float(len(valid)) < min_valid_ratio:
        return None, False, "invalid"

    vidx = anchor_idx[valid]
    vpose = pose[valid]
    if vidx.shape[0] < 2:
        return None, False, "invalid"

    max_anchor = int(vidx.max())
    num_raw = int(meta.get("camera_num_video_raw_frames", 0))
    usable = max_anchor + 1
    if num_raw > 0:
        usable = min(usable, num_raw)
    if usable < min_span:
        return None, False, "short"

    scale_factor = float(meta.get("camera_pose_scale_factor", 1.0))
    if min_metric_motion > 0.0:
        # metric 位移（米）= raw_max_disp / scale_factor；丢静止相机 clip。
        t = vpose[:, :3, 3]
        raw_max_disp = float(np.linalg.norm(t - t[:1], axis=-1).max())
        metric_disp = raw_max_disp / max(scale_factor, 1e-6)
        if metric_disp < min_metric_motion:
            return None, False, "static"

    return (json_path, max_anchor, num_raw, scale_factor), False, ""


class OSPIstockRGB_Multi(BaseMultiViewDataset):
    """OSP iStock T2V dataset with anchor-pose interpolation.

    Each sample = 1 chunk × 33 raw frames，每个 view 都带插值后的 c2w / K。

    Args:
        FEATURE_ROOT: ``.pt`` 文件所在目录（本类不直接读，仅用于索引校验可选）。
        CAMERA_ROOT:  ``.json`` 文件所在目录，dataset 主索引来源。
        VIDEO_ROOT_REMAP: ``[(old_prefix, new_prefix), ...]``，把 JSON 内的
            ``source_video_path`` 改写成本机挂载路径。
        pose_frame: ``"rel_chunk"`` 用 chunk-local 坐标系；``"rel_video"`` 用整段
            视频第一帧的坐标系。
        min_valid_ratio: chunk 级阈值，invalid anchor 比例超过 ``1 - ratio`` 的
            chunk 直接 drop。
        max_index_videos: 调试用，仅扫前 N 个 JSON 建索引（None=全量）。
        require_feature_pt: True 时索引阶段会校验 ``.pt`` 是否存在；默认 False
            以避免 listdir 数百万文件的代价（FEATURE_ROOT 与 CAMERA_ROOT 由
            offload pipeline 保证一一对应）。

    Notes:
        * ``is_t2v`` is configurable. Set ``is_t2v=False`` to train OSP in the
          same I2V prefix-ref regime as RE10K/DL3DV; keep ``True`` for pure T2V.
        * ``pose_frame`` (``rel_chunk`` / ``rel_video``) selects which pose field
          is read from OSP JSON. This is independent of training
          ``pose_origin="first"``, which re-normalizes loaded poses to the first
          sampled view inside ``prepare_data``.
        * ``view_choices`` is supported via ``BaseMultiViewDataset``; when set,
          the sampler requests 9/17/33 views and this class subsamples uniformly
          from the 33-frame chunk.
    """

    NUM_FRAMES_PER_CHUNK = _FRAMES_PER_CHUNK
    NUM_ANCHORS_PER_CHUNK = _ANCHORS_PER_CHUNK

    def __init__(
        self,
        *,
        FEATURE_ROOT: str,
        CAMERA_ROOT: str,
        VIDEO_ROOT_REMAP: Optional[list] = None,
        DA3_POSE_ROOT: Optional[str] = None,
        require_da3_sidecar: bool = False,
        pose_frame: str = "rel_chunk",
        min_valid_ratio: float = 1.0,
        max_index_videos: Optional[int] = None,
        require_feature_pt: bool = False,
        emit_caption_key: bool = True,
        is_t2v: bool = True,
        caption_source: str = "key",
        # ── Cross-clip 长序列模式（num_views > 33）参数 ──
        # OSP JSON 把整段 clip 切成多个连续 33 帧 chunk（每 chunk 9 anchor）。
        # num_views ≤ 33 走原 chunk 模式（1 sample = 1 chunk）；num_views > 33 走
        # clip 模式：跨 chunk 用 rel_video anchor 拼一条连续轨迹，按 stride 采
        # num_views 帧（参考已验证的 ucpe src/train_dataset_istock.py）。
        sample_stride: int = 4,
        sampling_strategy: str = "random_stride",
        random_start_offset: bool = True,
        min_metric_motion: float = 0.0,
        SHARED_CACHE_DIR: Optional[str] = os.environ.get("GAE_CACHE_DIR", "cache"),
        **kwargs,
    ):
        self.FEATURE_ROOT = FEATURE_ROOT
        self.CAMERA_ROOT = CAMERA_ROOT
        self.VIDEO_ROOT_REMAP = list(VIDEO_ROOT_REMAP or [])
        # Optional per-frame DA3 metric pose sidecars (see
        # scripts/export_osp_da3_perframe_poses.py). When set and a chunk has a
        # ``<json-stem>__chunk_XXXX/meta_da3_metric.json`` sidecar, its per-frame
        # c2w+K REPLACE the 9-anchor Slerp/Lerp interpolation.
        # ``require_da3_sidecar`` controls the missing/invalid-sidecar policy:
        #   * False (default): fall back to 9→33 Slerp/Lerp interpolation.
        #   * True: drop the chunk (skip to another sample), so only chunks with
        #     a valid per-frame DA3 sidecar are ever trained on.
        self.DA3_POSE_ROOT = (DA3_POSE_ROOT or "").strip() or None
        self.require_da3_sidecar = bool(require_da3_sidecar)
        self.pose_frame = pose_frame
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_index_videos = max_index_videos
        self.emit_caption_key = bool(emit_caption_key)
        self.is_t2v = bool(is_t2v)
        self.caption_source = str(caption_source)
        # Persistent cross-restart / cross-node index cache on shared FS
        # (S3-FUSE / EFS). On every node /local-ssd is wiped between jobs, so
        # without a shared layer each restart pays the ~110k JSON scan again.
        # Override via init kwarg or env ``OSP_SHARED_CACHE_DIR``; set to
        # ``""`` / ``"none"`` / ``"off"`` to disable.
        self.shared_cache_dir = SHARED_CACHE_DIR
        self.require_feature_pt = bool(require_feature_pt)

        # chunk 模式（num_views ≤ 33）：1 sample = 1 chunk，从 33 帧均匀抽 N 帧
        # （pose/K/img 同步），兼容 VAE 训练（V∈[4,8]）与 T2V（V=33）。
        # clip 模式（num_views > 33）：跨 chunk 拼长序列（见 __init__ 上方说明）。
        nv = int(kwargs.get("num_views", _FRAMES_PER_CHUNK))
        if nv < 1:
            raise ValueError(f"OSPIstockRGB_Multi requires num_views >= 1 (got {nv}).")
        self._clip_mode = nv > _FRAMES_PER_CHUNK
        if self._clip_mode and self.pose_frame != "rel_video":
            raise ValueError(
                f"OSPIstockRGB_Multi cross-clip mode (num_views={nv} > "
                f"{_FRAMES_PER_CHUNK}) requires pose_frame='rel_video' (got "
                f"pose_frame={self.pose_frame!r}); rel_chunk poses live in per-chunk "
                "coordinate frames and cannot be concatenated across chunks."
            )
        self.sample_stride = max(1, int(sample_stride))
        if sampling_strategy not in ("fixed_stride", "random_stride"):
            raise ValueError(
                "sampling_strategy must be 'fixed_stride' or 'random_stride', "
                f"got {sampling_strategy!r}"
            )
        self.sampling_strategy = sampling_strategy
        self.random_start_offset = bool(random_start_offset)
        self.min_metric_motion = float(min_metric_motion)
        kwargs["num_views"] = nv

        self.video = True
        self.is_metric = True
        # cond_num=0 时 ref_view_sampling 实际不生效，给 prefix 占位。
        self.ref_view_sampling = "prefix"

        super().__init__(**kwargs)
        self._load_index()

    # ─── Index 建立 / 缓存 ────────────────────────────────────────────────

    def _cache_tag(self) -> str:
        key_parts = [
            "osp_istock_v1",
            self.CAMERA_ROOT,
            f"vr{self.min_valid_ratio:.3f}",
            f"pf{self.pose_frame}",
            f"req{int(self.require_feature_pt)}",
        ]
        # clip 模式的索引与 chunk 模式不同（单元=整 clip + 依赖 num_views/motion 过滤）。
        # clip 模式追加判别项，避免失效已有 chunk-mode cache tag。
        if self._clip_mode:
            key_parts.append("clip")
            key_parts.append(f"nv{int(self.num_views)}")
            key_parts.append(f"mm{self.min_metric_motion:.3f}")
        if self.max_index_videos is not None:
            key_parts.append(f"max{int(self.max_index_videos)}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]

    def _cache_path(self) -> str:
        """Per-node fast cache on /local-ssd (or fallback to CAMERA_ROOT)."""
        tag = self._cache_tag()
        local = Path("/local-ssd/osp_istock_cache")
        if local.parent.is_dir():
            local.mkdir(parents=True, exist_ok=True)
            return str(local / f"osp_index_{tag}.pkl")
        return osp.join(self.CAMERA_ROOT, f".osp_index_{tag}.pkl")

    def _shared_cache_path(self) -> Optional[str]:
        """Persistent cross-restart cache (flat ``osp_index_<tag>.pkl`` in shared root)."""
        base = resolve_shared_cache_dir()
        if base is None:
            base = (self.shared_cache_dir or "").strip()
            if not base or base.lower() in ("none", "off", "disable", "false", "0"):
                return None
            if not Path(base).parent.exists():
                return None
        return str(Path(base) / f"osp_index_{self._cache_tag()}.pkl")

    def _collect_json_paths(self) -> list[str]:
        paths: list[str] = []
        with os.scandir(self.CAMERA_ROOT) as it:
            for entry in it:
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                paths.append(entry.path)
                if self.max_index_videos is not None and len(paths) >= self.max_index_videos:
                    break
        return paths

    def _log_scan_progress(
        self, n_videos: int, n_chunks_kept: int, n_chunks_dropped: int, n_videos_bad: int,
    ) -> None:
        print(
            f"[OSPIstockRGB] scan: {n_videos} json, "
            f"{n_chunks_kept} chunks kept, "
            f"{n_chunks_dropped} chunks dropped, "
            f"{n_videos_bad} json bad",
            flush=True,
        )

    def _scan_camera_root_serial(self, json_paths: list[str]) -> list[tuple[str, int]]:
        chunks: list[tuple[str, int]] = []
        n_videos_bad = 0
        n_chunks_dropped = 0
        n_chunks_kept = 0

        desc = "Indexing OSPIstockRGB"
        n_json = len(json_paths)
        for i, json_path in enumerate(json_paths, start=1):
            part, bad, dropped = _parse_one_camera_json(
                json_path,
                self.min_valid_ratio,
                self.require_feature_pt,
                self.FEATURE_ROOT,
            )
            if bad:
                n_videos_bad += 1
            else:
                chunks.extend(part)
                n_chunks_kept += len(part)
                n_chunks_dropped += dropped
            _log_index_progress(desc, i, n_json)

        self._log_scan_done(len(json_paths), n_chunks_kept, n_chunks_dropped, n_videos_bad)
        chunks.sort()
        return chunks

    def _scan_camera_root_parallel(
        self, json_paths: list[str], workers: int,
    ) -> list[tuple[str, int]]:
        chunks: list[tuple[str, int]] = []
        n_videos_bad = 0
        n_chunks_dropped = 0
        n_chunks_kept = 0
        print(
            f"[OSPIstockRGB] parallel scan: {len(json_paths)} json, "
            f"backend={_osp_index_backend()}, workers={workers}",
            flush=True,
        )

        task_args = [
            (p, self.min_valid_ratio, self.require_feature_pt, self.FEATURE_ROOT)
            for p in json_paths
        ]
        n_json = len(json_paths)
        desc = "Indexing OSPIstockRGB"

        if _osp_index_backend() == "process":
            chunksize = max(1, min(64, n_json // (workers * 16) or 1))
            done = 0
            with Pool(processes=workers) as pool:
                for part, bad, dropped in pool.imap_unordered(
                    _parse_one_camera_json_task, task_args, chunksize=chunksize,
                ):
                    if bad:
                        n_videos_bad += 1
                    else:
                        chunks.extend(part)
                        n_chunks_kept += len(part)
                        n_chunks_dropped += dropped
                    done += 1
                    _log_index_progress(desc, done, n_json)
        else:
            batch = max(workers * 8, 512)
            disable_bar = os.environ.get("DISABLE_INDEX_TQDM", "").strip().lower() in (
                "1", "true", "yes",
            )
            pbar = None
            if not disable_bar:
                try:
                    import sys
                    from tqdm import tqdm

                    force = os.environ.get("FORCE_INDEX_TQDM", "").strip().lower() in (
                        "1", "true", "yes",
                    )
                    pbar = tqdm(
                        total=n_json,
                        desc=desc,
                        unit="json",
                        dynamic_ncols=True,
                        disable=(not force) and (not sys.stderr.isatty()),
                        mininterval=0.5,
                    )
                except ImportError:
                    pass
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for start in range(0, n_json, batch):
                    chunk = task_args[start : start + batch]
                    futures = [ex.submit(_parse_one_camera_json_task, a) for a in chunk]
                    for fut in as_completed(futures):
                        part, bad, dropped = fut.result()
                        if bad:
                            n_videos_bad += 1
                        else:
                            chunks.extend(part)
                            n_chunks_kept += len(part)
                            n_chunks_dropped += dropped
                        done += 1
                        if pbar is not None:
                            pbar.update(1)
                        _log_index_progress(desc, done, n_json)
            if pbar is not None:
                pbar.close()

        self._log_scan_done(len(json_paths), n_chunks_kept, n_chunks_dropped, n_videos_bad)
        chunks.sort()
        return chunks

    def _log_scan_done(
        self, n_videos: int, n_chunks_kept: int, n_chunks_dropped: int, n_videos_bad: int,
    ) -> None:
        print(
            f"[OSPIstockRGB] scan done: {n_videos} json, "
            f"{n_chunks_kept} chunks kept, "
            f"{n_chunks_dropped} chunks dropped (valid<{self.min_valid_ratio}), "
            f"{n_videos_bad} json bad/malformed",
            flush=True,
        )

    def _scan_camera_root(self) -> list[tuple[str, int]]:
        """扫 CAMERA_ROOT 列所有 .json，解析建 (json_path, chunk_idx) 列表。"""
        json_paths = self._collect_json_paths()
        workers = _default_index_workers()
        if workers <= 1 or len(json_paths) <= 1:
            return self._scan_camera_root_serial(json_paths)
        return self._scan_camera_root_parallel(json_paths, workers)

    def _scan_camera_root_clips(self) -> list:
        """Clip 粒度索引（cross-clip 长序列模式，num_views > 33）。

        每条 entry = ``(json_path, max_valid_anchor_idx, num_raw_frames, scale_factor)``。
        过滤：valid_ratio < min_valid_ratio / usable_span < num_views / metric 静止相机。
        """
        json_paths = self._collect_json_paths()
        min_span = int(self.num_views)  # stride=1 时至少要 num_views 个 raw 帧
        workers = _default_index_workers()
        task_args = [
            (p, self.min_valid_ratio, self.min_metric_motion, min_span,
             self.require_feature_pt, self.FEATURE_ROOT)
            for p in json_paths
        ]
        n_json = len(json_paths)
        desc = "Indexing OSPIstockRGB(clip)"
        print(
            f"[OSPIstockRGB] clip scan: {n_json} json, workers={workers}, "
            f"min_span={min_span}, min_valid_ratio={self.min_valid_ratio}, "
            f"min_metric_motion={self.min_metric_motion}",
            flush=True,
        )

        entries: list = []
        n_bad = n_invalid = n_short = n_static = 0
        batch = max(workers * 8, 512)
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start in range(0, n_json, batch):
                chunk = task_args[start : start + batch]
                futures = [ex.submit(_parse_one_camera_json_clip, a) for a in chunk]
                for fut in as_completed(futures):
                    entry, bad, reason = fut.result()
                    if bad:
                        n_bad += 1
                    elif entry is None:
                        if reason == "invalid":
                            n_invalid += 1
                        elif reason == "short":
                            n_short += 1
                        elif reason == "static":
                            n_static += 1
                    else:
                        entries.append(entry)
                    done += 1
                    _log_index_progress(desc, done, n_json)

        entries.sort()
        print(
            f"[OSPIstockRGB] clip scan done: kept {len(entries)}/{n_json} "
            f"(bad={n_bad}, invalid(<{self.min_valid_ratio})={n_invalid}, "
            f"short(<{min_span})={n_short}, static(<{self.min_metric_motion}m)={n_static})",
            flush=True,
        )
        return entries

    def _index_override_path(self) -> Optional[str]:
        """Env / default pickle override (skip lock/copy/build), chunk mode only.

        Default only applies when ``_cache_tag()`` matches the prebuilt
        ``7792fe6eb91e`` index (vr=1.0, pf=rel_chunk, req=0, default CAMERA_ROOT).
        Clip mode (num_views > 33) has a distinct cache tag and never matches.
        """
        p = os.environ.get("OSP_INDEX_PATH", "").strip()
        if p:
            if osp.isfile(p):
                return p
            print(f"[OSPIstockRGB] OSP_INDEX_PATH={p} not a file; ignoring", flush=True)
        if self._cache_tag() == "7792fe6eb91e" and osp.isfile(_DEFAULT_OSP_INDEX):
            return _DEFAULT_OSP_INDEX
        return None

    def _load_index(self) -> None:
        if self._clip_mode:
            index = load_or_build(
                log_prefix="OSPIstockRGB",
                local_path=self._cache_path(),
                shared_path=self._shared_cache_path(),
                build_fn=self._scan_camera_root_clips,
            )
            self.clips = index
            self.chunks = None
            print(
                f"[OSPIstockRGB] index ready: {len(self.clips)} clips "
                f"(cross-clip mode, num_views={self.num_views})",
                flush=True,
            )
            return
        # chunk mode (num_views <= 33): honor a prebuilt-index override if present.
        override = self._index_override_path()
        if override is not None:
            print(f"[OSPIstockRGB] using index override → {override}", flush=True)
        index = load_or_build(
            log_prefix="OSPIstockRGB",
            local_path=override or self._cache_path(),
            shared_path=None if override is not None else self._shared_cache_path(),
            build_fn=self._scan_camera_root,
        )
        self.chunks = index
        self.clips = None
        print(
            f"[OSPIstockRGB] index ready: {len(self.chunks)} chunks",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.clips) if self._clip_mode else len(self.chunks)

    def get_stats(self) -> str:
        return f"{len(self)} clips" if self._clip_mode else f"{len(self)} chunks"

    def __repr__(self) -> str:
        res = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"OSPIstockRGB_Multi(chunks={len(self)}, num_views={self.num_views}, "
            f"resolutions={res}, pose_frame={self.pose_frame}, "
            f"min_valid_ratio={self.min_valid_ratio})"
        )

    # ─── Path remap ───────────────────────────────────────────────────────

    def _remap_video_path(self, p: str) -> str:
        for old, new in self.VIDEO_ROOT_REMAP:
            if p.startswith(old):
                return new + p[len(old):]
        return p

    # ─── Pose 插值（slerp + lerp）─────────────────────────────────────────

    @staticmethod
    def _interp_poses(
        anchor_c2w: np.ndarray,
        anchor_local_idx: np.ndarray,
        num_frames: int,
    ) -> np.ndarray:
        """把 9 个 anchor c2w 在 chunk-local frame 维度上插值到 num_frames 个。

        旋转：scipy.Slerp；平移：lerp。

        Args:
            anchor_c2w:  (9, 4, 4) c2w matrices.
            anchor_local_idx: (9,) chunk-local raw frame idx (e.g. [0, 4, ..., 32]).
            num_frames:  返回多少帧（=33）。

        Returns:
            (num_frames, 4, 4) fp32 c2w，target frame i 的 c2w 落在
            anchor_local_idx 的封闭区间内做插值。
        """
        assert anchor_c2w.shape == (_ANCHORS_PER_CHUNK, 4, 4)
        anchor_t = np.asarray(anchor_local_idx, dtype=np.float64)

        # 检查旋转矩阵是否合法（DA3 偶尔出过 numerical drift）
        Rs = anchor_c2w[:, :3, :3].astype(np.float64)
        # scipy 的 from_matrix 会自动做 SVD 正交化（since 1.4），稳健
        rotations = R_scipy.from_matrix(Rs)
        slerp = Slerp(anchor_t, rotations)

        target_t = np.arange(num_frames, dtype=np.float64)
        target_t = np.clip(target_t, anchor_t[0], anchor_t[-1])  # slerp 不外推

        R_interp = slerp(target_t).as_matrix()  # (num_frames, 3, 3)

        # Lerp translation: 找到 bracketing anchors j, j+1
        j = np.clip(
            np.searchsorted(anchor_t, target_t, side="right") - 1,
            0,
            len(anchor_t) - 2,
        )
        t0 = anchor_t[j]
        t1 = anchor_t[j + 1]
        alpha = (target_t - t0) / np.maximum(t1 - t0, 1e-9)
        trans = anchor_c2w[:, :3, 3].astype(np.float64)
        T_interp = (1.0 - alpha)[:, None] * trans[j] + alpha[:, None] * trans[j + 1]

        out = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
        out[:, :3, :3] = R_interp
        out[:, :3, 3] = T_interp
        return out.astype(np.float32)

    # ─── Intrinsic ────────────────────────────────────────────────────────

    @staticmethod
    def _K_norm_to_pixel(K_norm: np.ndarray, W: int, H: int) -> np.ndarray:
        """``[fx/W, fy/H, cx/W, cy/H]`` → 3x3 pixel-space K."""
        fx_n, fy_n, cx_n, cy_n = K_norm.astype(np.float64)
        return np.array(
            [
                [fx_n * W, 0.0, cx_n * W],
                [0.0, fy_n * H, cy_n * H],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # ─── 视频读取 ─────────────────────────────────────────────────────────

    def _extract_chunk_frames(
        self, video_path: str, chunk_start: int, num_frames: int = _FRAMES_PER_CHUNK,
    ) -> Optional[list[Image.Image]]:
        """连续读 num_frames 个 raw frame 起点 chunk_start。"""
        cap = cv2.VideoCapture(video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0 or chunk_start + num_frames > total:
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk_start)
            frames: list[Image.Image] = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    return None
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            return frames
        finally:
            cap.release()

    # ─── 主入口 ───────────────────────────────────────────────────────────

    @staticmethod
    def _raw_is_useful(raw: str) -> bool:
        text = str(raw or "").strip()
        return len(text.split()) >= 2 and text.lower() not in {"a", "an", "the"}

    @staticmethod
    def _caption_from_filename(stem: str) -> str:
        """Derive a readable prompt from OSP's descriptive filename stem."""
        text = re.sub(r"_[0-9a-f]{8,}_[0-9]+-[0-9]+.*$", "", stem)
        text = re.sub(r"_[0-9]+-[0-9]+.*$", "", text)
        text = re.sub(r"_part[0-9]+$", "", text)
        text = re.sub(r"_gm[0-9]+-[0-9]+$", "", text)
        text = text.replace("_", " ").replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip(" .,_-")
        return text

    @classmethod
    def _filename_caption_is_useful(cls, stem: str) -> bool:
        """Reject filename fallbacks that collapse to gm-ID / UUID noise."""
        text = cls._caption_from_filename(stem)
        if not cls._raw_is_useful(text):
            return False
        # e.g. "gm1311237282 400393531" — only stock IDs, no natural language.
        words = [w for w in text.split() if w]
        alpha_words = [
            w for w in words
            if re.search(r"[a-zA-Z]{3,}", w) and not re.fullmatch(r"gm\d+", w, flags=re.I)
        ]
        return len(alpha_words) >= 2

    def _caption_for_meta(self, meta: dict, json_path: str, caption_key: str) -> str:
        if self.emit_caption_key or self.caption_source == "key":
            return caption_key
        if self.caption_source == "none":
            return ""
        raw = str(meta.get("prompt_raw", "") or "").strip()
        if self.caption_source == "prompt_raw":
            return raw if self._raw_is_useful(raw) else ""
        if self.caption_source == "prompt_or_filename":
            # Good prompt_raw → Qwen text; bad/missing → drop text (null embed).
            return raw if self._raw_is_useful(raw) else ""
        if self.caption_source == "filename":
            stem = str(meta.get("sample_id", "") or osp.splitext(osp.basename(json_path))[0])
            return (
                self._caption_from_filename(stem)
                if self._filename_caption_is_useful(stem) else ""
            )
        raise ValueError(
            "caption_source must be one of key/none/prompt_raw/filename/"
            f"prompt_or_filename, got {self.caption_source!r}"
        )

    def _get_views(self, idx, resolution, rng, num_views):
        if self._clip_mode:
            return self._get_views_clip(idx, resolution, rng, num_views)
        # ── chunk 模式（num_views ≤ 33，原实现，未改动）──
        # 内部统一按 33 帧生成（视频抽帧 + slerp+lerp 插值需要完整 chunk），
        # 在 return 之前再按 num_views 从 33 帧均匀抽样。
        if not (1 <= num_views <= _FRAMES_PER_CHUNK):
            raise ValueError(
                f"OSPIstockRGB_Multi supports 1 <= num_views <= {_FRAMES_PER_CHUNK}, "
                f"got {num_views}"
            )

        max_retries = 50
        current_idx = idx
        for retry in range(max_retries):
            if retry > 0:
                current_idx = int(rng.integers(0, len(self.chunks)))
            json_path, chunk_idx = self.chunks[current_idx]

            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            video_path = self._remap_video_path(meta["source_video_path"])
            if not osp.isfile(video_path):
                continue

            try:
                anchor_local_idx, anchor_c2w, anchor_K_norm, chunk_start_raw = (
                    self._load_chunk_pose(meta, chunk_idx)
                )
            except (KeyError, IndexError, AssertionError, ValueError):
                continue

            frames = self._extract_chunk_frames(
                video_path, chunk_start_raw, _FRAMES_PER_CHUNK
            )
            if frames is None or len(frames) != _FRAMES_PER_CHUNK:
                continue

            W_raw, H_raw = frames[0].size

            # 优先用 DA3 per-frame sidecar 的逐帧 metric c2w+K（da3_K_33 在
            # raw 像素空间 H_raw×W_raw，逐帧生效）。命中时**不做插值**；仅当
            # sidecar 缺失/损坏时才回退 9→33 Slerp/Lerp 插值 + 常量 anchor K。
            c2w_33 = None
            da3_K_33 = None
            if self.DA3_POSE_ROOT is not None:
                da3 = self._load_da3_perframe(
                    json_path, chunk_idx, chunk_start_raw, H_raw, W_raw
                )
                if da3 is not None:
                    c2w_33, da3_K_33 = da3
                elif self.require_da3_sidecar:
                    # 显式要求逐帧 DA3 pose：sidecar 缺失/损坏的 chunk 直接放弃，
                    # 换下一个样本，绝不回退插值。
                    continue

            if c2w_33 is None:
                # 回退：9 → 33 c2w 插值；chunk 内 K 基本不变，取 anchor 0。
                c2w_33 = self._interp_poses(
                    anchor_c2w, anchor_local_idx, _FRAMES_PER_CHUNK
                )
            K_pixel = (
                None if da3_K_33 is not None
                else self._K_norm_to_pixel(anchor_K_norm[0], W_raw, H_raw)
            )

            # caption_key per-video（同一视频 ≥1 chunk 共用同一 prompt_embed）。
            # 用文件名 stem（去 .json 后缀）作为唯一 key，与
            # ``scripts/extract_osp_caption_embeddings.py`` 的存 key 规则保持一致。
            # .pt 内部 *没有* sample_id 字段（offload 输出未带），用 JSON 里的
            # sample_id 会跟 .pt 文件名 stem 不一致 → caption_cache lookup miss。
            caption_key = osp.splitext(osp.basename(json_path))[0]
            caption = self._caption_for_meta(meta, json_path, caption_key)

            # num_views < 33 时（如 VAE 训练 V=8），从 33 帧均匀抽 N 帧。
            # 索引含端点 0 和 32，确保覆盖 anchor 0/4/8/.../32。
            if num_views == _FRAMES_PER_CHUNK:
                frame_sel = list(range(_FRAMES_PER_CHUNK))
            else:
                frame_sel = (
                    np.linspace(0, _FRAMES_PER_CHUNK - 1, num_views)
                    .round()
                    .astype(int)
                    .tolist()
                )

            views = []
            for v in frame_sel:
                # BaseMultiViewDataset._crop_resize_if_necessary 要求 depthmap 入参
                # （即便不用），尺寸与原图一致即可。
                depth_dummy = np.zeros((H_raw, W_raw, 3), dtype=np.uint8)
                K_in = da3_K_33[v] if da3_K_33 is not None else K_pixel
                img_proc, _, K_proc = self._crop_resize_if_necessary(
                    frames[v], depth_dummy, K_in.copy(), resolution,
                    rng=rng, info=(current_idx, v),
                )
                views.append(
                    dict(
                        img=img_proc,
                        camera_pose=c2w_33[v].astype(np.float32),
                        camera_intrinsics=K_proc.astype(np.float32),
                        caption=caption,
                        is_t2v=self.is_t2v,
                        # 阶段 2 训练脚本会读这个 flag，让 plucker 不被 T2V 默认清零
                        disable_plucker=False,
                    )
                )
            return views

        raise RuntimeError(
            f"[OSPIstockRGB] failed to produce a valid chunk after {max_retries} retries "
            f"(start idx={idx})"
        )

    # ─── Cross-clip 长序列（num_views > 33）─────────────────────────────────

    @staticmethod
    def _interp_anchor_to_clip(
        anchor_idx: np.ndarray,
        anchor_c2w: np.ndarray,
        query_idx: np.ndarray,
    ) -> np.ndarray:
        """把（可跨 chunk 的）N 个 anchor c2w 插值到 query 帧。

        旋转 SLERP，平移线性（np.interp）。参考 ucpe
        ``train_dataset_istock._interp_anchor_to_dense``。

        Args:
            anchor_idx: (N,) 全局 raw frame idx（valid 过滤后，可能未排序/有重复）。
            anchor_c2w: (N, 4, 4) rel_video c2w。
            query_idx:  (M,) 目标 raw frame idx。

        Returns:
            (M, 4, 4) fp32 c2w。SLERP 不外推，query 会 clip 到 anchor 区间内。
        """
        anchor_idx = np.asarray(anchor_idx)
        anchor_c2w = np.asarray(anchor_c2w)
        # 去重 + 升序（跨 chunk 边界偶有重复 idx；Slerp 要求严格递增）。
        uniq_idx, keep = np.unique(anchor_idx, return_index=True)
        anchor_idx = uniq_idx
        anchor_c2w = anchor_c2w[keep]
        if anchor_idx.shape[0] < 2:
            raise ValueError("need >= 2 unique anchors to interpolate")

        q = np.asarray(query_idx, dtype=np.float64)
        q = np.clip(q, anchor_idx[0], anchor_idx[-1])

        rots = R_scipy.from_matrix(anchor_c2w[:, :3, :3].astype(np.float64))
        slerp = Slerp(anchor_idx.astype(np.float64), rots)
        interp_R = slerp(q).as_matrix()  # (M, 3, 3)

        a = anchor_idx.astype(np.float64)
        interp_t = np.stack(
            [np.interp(q, a, anchor_c2w[:, ax, 3].astype(np.float64)) for ax in range(3)],
            axis=-1,
        )  # (M, 3)

        out = np.tile(np.eye(4, dtype=np.float32), (len(q), 1, 1))
        out[:, :3, :3] = interp_R.astype(np.float32)
        out[:, :3, 3] = interp_t.astype(np.float32)
        return out

    def _extract_frames_at_indices(
        self, video_path: str, raw_indices: np.ndarray,
    ) -> Optional[list[Image.Image]]:
        """按 raw_indices（可跨 chunk、可 stride>1）读帧。

        cv2 顺序解码到最大 idx、命中即取（避免部分编码器上 cv2 随机 seek 不可靠）。
        缺帧/越界返回 None（调用方换样本）。
        """
        want = {int(i) for i in raw_indices}
        max_idx = int(max(raw_indices))
        cap = cv2.VideoCapture(video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0 and max_idx >= total:
                return None
            grabbed: dict[int, Image.Image] = {}
            pos = 0
            while pos <= max_idx:
                ret, frame = cap.read()
                if not ret:
                    return None
                if pos in want:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    grabbed[pos] = Image.fromarray(rgb)
                pos += 1
            try:
                return [grabbed[int(i)] for i in raw_indices]
            except KeyError:
                return None
        finally:
            cap.release()

    def _get_views_clip(self, idx, resolution, rng, num_views):
        """clip 模式：跨 chunk 用 rel_video anchor 拼连续轨迹，按 stride 采 num_views 帧。

        pose 输出 metric 米制（÷ camera_pose_scale_factor），与其它 metric 几何源同尺度
        （训练 ``pose_translation_norm="none"`` 不抵消 scale）。
        """
        max_retries = 50
        current_idx = idx
        for retry in range(max_retries):
            if retry > 0:
                current_idx = int(rng.integers(0, len(self.clips)))
            json_path, max_anchor, num_raw, scale_factor = self.clips[current_idx]

            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            video_path = self._remap_video_path(meta["source_video_path"])
            if not osp.isfile(video_path):
                continue

            try:
                anchor_idx = np.asarray(
                    meta["camera_anchor_frame_idx"], dtype=np.int64
                ).reshape(-1)
                valid = np.asarray(meta["camera_valid_mask"]).reshape(-1).astype(bool)
                anchor_c2w = np.asarray(
                    meta["camera_pose_rel_video_c2w"], dtype=np.float32
                ).reshape(-1, 4, 4)
                K_norm = np.asarray(meta["camera_K_norm"], dtype=np.float32).reshape(-1, 4)
            except (KeyError, ValueError, TypeError):
                continue
            if not (
                anchor_idx.shape[0] == valid.shape[0]
                == anchor_c2w.shape[0] == K_norm.shape[0]
            ):
                continue
            if int(valid.sum()) < 2:
                continue
            anchor_idx_v = anchor_idx[valid]
            anchor_c2w_v = anchor_c2w[valid]
            K_norm_v = K_norm[valid]

            usable = int(anchor_idx_v.max()) + 1
            if num_raw > 0:
                usable = min(usable, num_raw)
            if usable < num_views:
                continue

            # stride（受 usable 约束）+ random start，参考 ucpe。
            max_possible_stride = max(1, (usable - 1) // max(1, num_views - 1))
            if self.sampling_strategy == "random_stride":
                hi = min(self.sample_stride, max_possible_stride)
                stride = int(rng.integers(1, hi + 1))
            else:
                stride = min(self.sample_stride, max_possible_stride)
            needed_span = (num_views - 1) * stride + 1
            start_max = max(0, usable - needed_span)
            raw_start = (
                int(rng.integers(0, start_max + 1))
                if (self.random_start_offset and start_max > 0)
                else 0
            )
            raw_indices = raw_start + np.arange(num_views) * stride

            # rel_video anchor（跨 chunk）→ 插值 → ÷scale_factor 转 metric（米）。
            try:
                c2w = self._interp_anchor_to_clip(anchor_idx_v, anchor_c2w_v, raw_indices)
            except (ValueError, IndexError):
                continue
            sf = max(float(scale_factor), 1e-6)
            c2w[:, :3, 3] /= sf

            frames = self._extract_frames_at_indices(video_path, raw_indices)
            if frames is None or len(frames) != num_views:
                continue
            W_raw, H_raw = frames[0].size

            # K：valid anchor 的 K_norm 均值（clip 内基本不变）→ 像素 K。
            K_pixel = self._K_norm_to_pixel(K_norm_v.mean(axis=0), W_raw, H_raw)

            caption_key = osp.splitext(osp.basename(json_path))[0]
            caption = self._caption_for_meta(meta, json_path, caption_key)

            views = []
            for v in range(num_views):
                depth_dummy = np.zeros((H_raw, W_raw, 3), dtype=np.uint8)
                img_proc, _, K_proc = self._crop_resize_if_necessary(
                    frames[v], depth_dummy, K_pixel.copy(), resolution,
                    rng=rng, info=(current_idx, v),
                )
                views.append(
                    dict(
                        img=img_proc,
                        camera_pose=c2w[v].astype(np.float32),
                        camera_intrinsics=K_proc.astype(np.float32),
                        caption=caption,
                        is_t2v=self.is_t2v,
                        disable_plucker=False,
                    )
                )
            return views

        raise RuntimeError(
            f"[OSPIstockRGB] (clip mode) failed to produce a valid clip after "
            f"{max_retries} retries (start idx={idx})"
        )

    # ─── meta unpack 帮助 ─────────────────────────────────────────────────

    def _load_da3_perframe(
        self, json_path: str, chunk_idx: int, chunk_start_raw: int,
        H_raw: int, W_raw: int,
    ):
        """读 ``<stem>__chunk_XXXX/meta_da3_metric.json`` 的逐帧 metric c2w+K。

        返回 ``(c2w_33 (33,4,4) fp32, K_33 (33,3,3) fp32)``；缺失/损坏/不对齐
        时返回 ``None``（调用方回退到 anchor 插值）。K 按 sidecar 记录的分辨率
        缩放到当前解码帧的 (H_raw, W_raw) 像素空间。
        """
        stem = osp.splitext(osp.basename(json_path))[0]
        scene_name = f"{stem}__chunk_{chunk_idx:04d}"
        sidecar = osp.join(self.DA3_POSE_ROOT, scene_name, "meta_da3_metric.json")
        if not osp.isfile(sidecar):
            return None
        try:
            with open(sidecar) as f:
                dp = json.load(f)
            frames_meta = dp.get("frames", [])
            if int(dp.get("num_frames", 0)) != _FRAMES_PER_CHUNK:
                return None
            if len(frames_meta) != _FRAMES_PER_CHUNK:
                return None
            # 起点必须与本次解码一致，否则 pose 与帧错位。
            if int(dp.get("chunk_start_raw", chunk_start_raw)) != int(chunk_start_raw):
                return None
            c2w = np.asarray([fr["c2w"] for fr in frames_meta], dtype=np.float32)
            K = np.asarray([fr["K"] for fr in frames_meta], dtype=np.float32)
            if c2w.shape != (_FRAMES_PER_CHUNK, 4, 4):
                return None
            if K.shape != (_FRAMES_PER_CHUNK, 3, 3):
                return None
            if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
                return None
            sc_h = int(dp.get("height", H_raw)) or H_raw
            sc_w = int(dp.get("width", W_raw)) or W_raw
            if (sc_h, sc_w) != (H_raw, W_raw):
                sx = float(W_raw) / float(max(sc_w, 1))
                sy = float(H_raw) / float(max(sc_h, 1))
                K = K.copy()
                K[:, 0, 0] *= sx
                K[:, 0, 2] *= sx
                K[:, 1, 1] *= sy
                K[:, 1, 2] *= sy
            return c2w, K
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def _load_chunk_pose(self, meta: dict, chunk_idx: int):
        """从 JSON meta 解析一个 chunk 的 (anchor_local_idx, anchor_c2w, K_norm, chunk_start_raw)。

        失败时抛 KeyError / IndexError / AssertionError，调用方会换 idx 重试。
        """
        if self.pose_frame == "rel_chunk":
            anchor_c2w = np.asarray(
                meta["camera_pose_rel_chunk_c2w"][chunk_idx], dtype=np.float32
            )
        elif self.pose_frame == "rel_video":
            anchor_c2w = np.asarray(
                meta["camera_pose_rel_video_c2w"][chunk_idx], dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown pose_frame: {self.pose_frame}")

        if anchor_c2w.shape != (_ANCHORS_PER_CHUNK, 4, 4):
            raise ValueError(f"Bad anchor_c2w shape: {anchor_c2w.shape}")

        anchor_idx_raw = np.asarray(
            meta["camera_anchor_frame_idx"][chunk_idx], dtype=np.int64
        )
        if anchor_idx_raw.shape != (_ANCHORS_PER_CHUNK,):
            raise ValueError(f"Bad anchor_idx_raw shape: {anchor_idx_raw.shape}")

        chunk_start_raw = int(anchor_idx_raw[0])
        anchor_local_idx = anchor_idx_raw - chunk_start_raw

        # OSP 的固定 layout：anchor 在 chunk-local [0, 4, 8, ..., 32]
        if tuple(int(x) for x in anchor_local_idx) != _EXPECTED_ANCHOR_LOCAL:
            raise ValueError(
                f"Unexpected anchor layout for chunk {chunk_idx}: {anchor_local_idx.tolist()} "
                f"(expected {list(_EXPECTED_ANCHOR_LOCAL)})"
            )

        anchor_K_norm = np.asarray(
            meta["camera_K_norm"][chunk_idx], dtype=np.float32
        )
        if anchor_K_norm.shape != (_ANCHORS_PER_CHUNK, 4):
            raise ValueError(f"Bad anchor_K_norm shape: {anchor_K_norm.shape}")

        return anchor_local_idx, anchor_c2w, anchor_K_norm, chunk_start_raw
