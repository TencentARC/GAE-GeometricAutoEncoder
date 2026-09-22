"""Generic per-scene packed dataset: video.mp4 + meta.json (+ caption.txt).

This class reads any dataset preprocessed into the same on-disk layout used by
``preprocess_scannetpp.py``:

    ROOT/<scene_id>/
        video.mp4    — H.264 (yuv420p, crf=18), N frames at preprocess-time resolution
        meta.json    — {"num_frames": int,
                        "rgb_resolution"|"height"|"width": ...,
                        "frames": [{"c2w": 4x4, "K": 3x3, ...}, ...]}
        caption.txt  — (optional) single-line scene caption

Compared to ``ScanNetppRGB_Multi`` (which is ScanNet++ specific), this class
exposes:

  * ``is_metric``         — whether camera poses are in metric units
                            (RE10K/DL3DV/MVS-Synth → False)
  * ``caption_map_path``  — optional path to a global ``{scene_id: caption}``
                            JSON for datasets where captions live outside the
                            scene directory (RE10K legacy ``train_caption.json``)
  * ``min_interval`` / ``max_interval`` — same semantics as RE10K/DL3DV
  * ``split`` / ``val_frac`` — same hash-based scene partition

Used by the VAE fine-tune pipeline once RE10K / MVS-Synth / DL3DV have been
repacked into this layout by their respective ``preprocess_*_to_video.py``
scripts. The original PNG-sequence dataset classes
(``RE10K_Multi`` / ``MVSSynth_Multi`` / ``DL3DV_Multi``) remain for backward
compatibility.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image

from cut3r_data.base.base_multiview_dataset import (
    BadPoseSampleError,
    BaseMultiViewDataset,
)
from cut3r_data.base.dataset_index_cache import (
    load_or_build,
    mirror_shared_path,
    resolve_shared_cache_dir,
)
from cut3r_data.base.fast_meta_index import (
    _vms_index_backend,
    build_packed_scene_index,
    default_vms_index_workers,
    fast_load_packed_meta_header,
)


_CAM_CACHE_MAX = 4096
_LOCAL_CACHE_ROOT = "/local-ssd/video_meta_scene_cache"

# Prebuilt index pickles loaded directly (skip lock/copy/build) unless the
# matching ``VMS_INDEX_<TAG>`` / ``VMS_INDEX_PATH`` env overrides it. Keyed by
# ``dataset_tag``.
# Exact nv33 + pose_meta=meta_da3_metric.json hashes for the current packed
# ROOTs. Used as a direct pickle override so train/index jobs skip cold FUSE
# scans when the shared cache is present.
_DEFAULT_VMS_INDEX: dict[str, str] = {}


class VideoMetaScene_Multi(BaseMultiViewDataset):
    """Read per-scene packed mp4 + meta.json across heterogeneous datasets.

    Args:
        ROOT:               Output dir of the dataset preprocessing scripts
                            (e.g. ``scripts/data/prepare_data.py``).
        dataset_tag:        Short string identifier mixed into index cache key
                            so RE10K / DL3DV / MVSSynth indices don't collide
                            even if their ROOT happens to share a prefix.
        is_metric:          Whether c2w in meta.json is in metric units.
        min_interval:       Min frame stride between sampled views.
        max_interval:       Max frame stride between sampled views.
        split:              Optional ``"train"`` / ``"val"`` partition by
                            scene hash. ``None`` = use all scenes.
        val_frac:           Fraction of scenes held out for ``"val"``.
        caption_map_path:   Optional path to a JSON file mapping
                            ``scene_id -> caption_string``. Used when captions
                            are not inlined as ``<scene>/caption.txt``.
        pose_meta_name:     Optional sidecar JSON carrying DA3 metric ``c2w``/``K``.
                            When set, root scenes and one-level ``chunk_*`` dirs
                            are indexed only if this sidecar exists.
        scene_whitelist_path:
                            Optional mandatory/base plain-text or gzip whitelist,
                            typically produced by visual-quality QC.
        motion_whitelist_path:
                            Optional additional motion whitelist. When both
                            paths are set, only their intersection is indexed.
        ref_view_sampling:  Forwarded as-is into each view dict.
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        dataset_tag: str = "vms",
        is_metric: bool = False,
        min_interval: int = 1,
        max_interval: int = 128,
        split: str | None = None,
        val_frac: float = 0.02,
        caption_map_path: str | None = None,
        pose_meta_name: str | None = None,
        scene_whitelist_path: str | None = None,
        motion_whitelist_path: str | None = None,
        ref_view_sampling: str = "prefix",
        **kwargs,
    ):
        self.ROOT = ROOT
        self.dataset_tag = str(dataset_tag)
        self.video = True
        self.is_metric = bool(is_metric)
        self.ref_view_sampling = ref_view_sampling
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.val_frac = float(val_frac)
        self.caption_map_path = caption_map_path
        self.pose_meta_name = pose_meta_name
        self.scene_whitelist_path = scene_whitelist_path
        self.motion_whitelist_path = motion_whitelist_path
        self._scene_whitelist: set[str] | None = None
        self._scene_whitelist_tag = self._hash_scene_whitelist()
        if self._scene_whitelist_paths() and not self.pose_meta_name:
            raise ValueError(
                f"[VMS:{self.dataset_tag}] scene/motion whitelist requires "
                "pose_meta_name (whitelist is keyed by DA3 sidecar units)."
            )
        super().__init__(*args, split=split, **kwargs)

        self._cam_cache: collections.OrderedDict = collections.OrderedDict()
        self._caption_map: dict[str, str] = {}

        if self.caption_map_path and osp.isfile(self.caption_map_path):
            try:
                with open(self.caption_map_path) as f:
                    raw_map = json.load(f)
                # Caption files (e.g. RE10K train_caption.json) often have
                # trailing "\n" baked in — strip uniformly so downstream
                # tokenizers don't see stray whitespace.
                self._caption_map = {
                    k: (v.strip() if isinstance(v, str) else "")
                    for k, v in raw_map.items()
                }
                print(
                    f"[VMS:{self.dataset_tag}] Loaded "
                    f"{len(self._caption_map)} captions from "
                    f"{self.caption_map_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[VMS:{self.dataset_tag}] WARN: could not load captions "
                    f"from {self.caption_map_path}: {e}",
                    flush=True,
                )

        self._load_data()

    # ── Index building ────────────────────────────────────────────────────

    def _scene_whitelist_paths(self) -> list[tuple[str, str]]:
        paths: list[tuple[str, str]] = []
        scene_path = getattr(self, "scene_whitelist_path", None)
        motion_path = getattr(self, "motion_whitelist_path", None)
        if scene_path:
            paths.append(("quality", scene_path))
        if motion_path:
            paths.append(("motion", motion_path))
        return paths

    def _hash_scene_whitelist(self) -> str | None:
        """Short content hash of all whitelist files, for the cache key."""
        paths = self._scene_whitelist_paths()
        if not paths:
            return None
        components: list[tuple[str, str]] = []
        for role, path in paths:
            if not osp.isfile(path):
                raise FileNotFoundError(
                    f"[VMS:{self.dataset_tag}] {role} whitelist not found: {path}"
                )
            file_hash = hashlib.md5()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    file_hash.update(block)
            components.append((role, file_hash.hexdigest()))
        if len(components) == 1:
            return components[0][1][:12]
        combined = hashlib.md5()
        for role, digest in components:
            combined.update(f"{role}:{digest}\n".encode())
        return combined.hexdigest()[:12]

    def _load_scene_whitelist(self) -> set[str]:
        """Load the quality/motion whitelist intersection by first tab field."""
        paths = self._scene_whitelist_paths()
        assert paths
        units: set[str] | None = None
        for role, path in paths:
            selected: set[str] = set()
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    unit_id = line.split("\t", 1)[0]
                    if units is None or unit_id in units:
                        selected.add(unit_id)
            units = selected
            print(
                f"[VMS:{self.dataset_tag}] {role} whitelist -> "
                f"{len(units)} retained from {path}",
                flush=True,
            )
        assert units is not None
        print(
            f"[VMS:{self.dataset_tag}] effective whitelist: "
            f"{len(units)} kept units (tag={self._scene_whitelist_tag})",
            flush=True,
        )
        return units

    def _cache_path(self) -> str:
        key = (
            f"vms_{self.dataset_tag}_{self.ROOT}__v5__nv{self.num_views}"
            f"__split{self.split}__vf{self.val_frac}"
            f"__pose{self.pose_meta_name or 'meta'}"
        )
        if self._scene_whitelist_tag:
            key += f"__wl{self._scene_whitelist_tag}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        os.makedirs(_LOCAL_CACHE_ROOT, exist_ok=True)
        return osp.join(
            _LOCAL_CACHE_ROOT,
            f"vms_index_{self.dataset_tag}_{tag}.pkl",
        )

    def _shared_cache_path(self) -> str | None:
        return mirror_shared_path(self._cache_path(), resolve_shared_cache_dir())

    def _scene_in_split(self, scene_id: str) -> bool:
        if self.split is None:
            return True
        h = int(hashlib.md5(scene_id.encode()).hexdigest()[:8], 16)
        is_val = (h % 10000) < int(10000 * self.val_frac)
        return is_val if self.split == "val" else (not is_val)

    def _build_index(self) -> dict:
        w = default_vms_index_workers()
        print(
            f"[VMS:{self.dataset_tag}] scanning {self.ROOT} "
            f"(fast header, backend={_vms_index_backend()}, workers={w}) …",
            flush=True,
        )
        if self.pose_meta_name:
            return self._build_pose_sidecar_index()
        return build_packed_scene_index(
            self.ROOT,
            cut_off=max(2, self.num_views),
            split=self.split,
            val_frac=self.val_frac,
            with_caption=True,
            workers=w,
            progress_desc=f"Indexing VMS:{self.dataset_tag}",
        )

    def _has_frame_sidecar(self, pose_path: str) -> bool:
        try:
            with open(pose_path, encoding="utf-8") as f:
                return '"frames"' in f.read(8192)
        except OSError:
            return False

    def _load_pose_sidecar_frame_count(self, pose_path: str) -> int | None:
        """Return DA3 sidecar frame count without parsing poses when possible."""
        try:
            fields = fast_load_packed_meta_header(pose_path)
        except Exception:
            fields = None
        if fields is not None:
            n = int(fields.get("num_frames", 0))
            if n > 0:
                return n
        try:
            with open(pose_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        frames = meta.get("frames")
        if not isinstance(frames, list):
            return None
        return len(frames)

    def _load_caption_txt(self, meta_path: str) -> str:
        caption_path = osp.join(osp.dirname(meta_path), "caption.txt")
        try:
            with open(caption_path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _maybe_add_sidecar_scene(
        self,
        *,
        scene_id: str,
        meta_path: str,
        video_path: str,
        pose_meta_path: str,
        scenes: list[str],
        scene_data: dict[str, dict],
        start_img_ids: list[tuple[str, int]],
    ) -> bool:
        if not self._scene_in_split(scene_id):
            return False
        if self._scene_whitelist is not None and scene_id not in self._scene_whitelist:
            return False
        if not (osp.isfile(meta_path) and osp.isfile(video_path) and osp.isfile(pose_meta_path)):
            return False
        fields = fast_load_packed_meta_header(meta_path)
        if fields is None:
            return False
        meta_num_frames = int(fields["num_frames"])
        pose_num_frames = self._load_pose_sidecar_frame_count(pose_meta_path)
        if pose_num_frames is None:
            return False
        num_frames = min(meta_num_frames, int(pose_num_frames))
        cut_off = max(2, self.num_views)
        if num_frames < cut_off:
            return False
        scenes.append(scene_id)
        scene_data[scene_id] = {
            "video_path": video_path,
            "meta_path": meta_path,
            "pose_meta_path": pose_meta_path,
            "num_frames": num_frames,
            "rgb_resolution": int(fields["rgb_resolution"]),
            "inline_caption": fields["inline_caption"],
            "caption_txt": self._load_caption_txt(meta_path),
        }
        num_start = max(1, num_frames - cut_off + 1)
        start_img_ids.extend((scene_id, si) for si in range(num_start))
        return True

    def _add_valid_immediate_chunks(
        self,
        *,
        parent_dir: str,
        parent_scene_id: str,
        scenes: list[str],
        scene_data: dict[str, dict],
        start_img_ids: list[tuple[str, int]],
    ) -> tuple[int, int]:
        """Add valid ``parent/chunk_*`` entries.

        Returns:
            (candidate_chunks, added_chunks). If candidate_chunks > 0, callers
            should not fall back to the parent video even when all chunks are
            filtered out for being too short.
        """
        candidates = 0
        added = 0
        manifest_path = osp.join(parent_dir, "chunks_manifest.json")
        if osp.isfile(manifest_path):
            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                chunks = manifest.get("chunks") or []
            except (OSError, json.JSONDecodeError):
                return 1, 0
            if not isinstance(chunks, list):
                return 1, 0
            for ch in chunks:
                if not isinstance(ch, dict):
                    continue
                chunk_name = str(ch.get("dir", ""))
                if not chunk_name.startswith("chunk_"):
                    continue
                chunk_dir = osp.join(parent_dir, chunk_name)
                meta_name = str(ch.get("meta") or "meta.json")
                video_name = str(ch.get("video") or "video.mp4")
                pose_name = str(ch.get("pose_sidecar") or self.pose_meta_name)
                candidates += 1
                if self._maybe_add_sidecar_scene(
                    scene_id=f"{parent_scene_id}/{chunk_name}",
                    meta_path=osp.join(chunk_dir, meta_name),
                    video_path=osp.join(chunk_dir, video_name),
                    pose_meta_path=osp.join(chunk_dir, pose_name),
                    scenes=scenes,
                    scene_data=scene_data,
                    start_img_ids=start_img_ids,
                ):
                    added += 1
            return candidates, added

        try:
            with os.scandir(parent_dir) as it:
                for child in it:
                    if not (child.is_dir() and child.name.startswith("chunk_")):
                        continue
                    chunk_dir = child.path
                    if not (
                        osp.isfile(osp.join(chunk_dir, "meta.json"))
                        and osp.isfile(osp.join(chunk_dir, "video.mp4"))
                        and osp.isfile(osp.join(chunk_dir, self.pose_meta_name))
                    ):
                        continue
                    candidates += 1
                    if self._maybe_add_sidecar_scene(
                        scene_id=f"{parent_scene_id}/{child.name}",
                        meta_path=osp.join(chunk_dir, "meta.json"),
                        video_path=osp.join(chunk_dir, "video.mp4"),
                        pose_meta_path=osp.join(chunk_dir, self.pose_meta_name),
                        scenes=scenes,
                        scene_data=scene_data,
                        start_img_ids=start_img_ids,
                    ):
                        added += 1
        except OSError:
            return candidates, added
        return candidates, added

    def _build_pose_sidecar_index(self) -> dict:
        if self._scene_whitelist_paths():
            self._scene_whitelist = self._load_scene_whitelist()
        scenes: list[str] = []
        scene_data: dict[str, dict] = {}
        start_img_ids: list[tuple[str, int]] = []
        n_dirs = 0
        with os.scandir(self.ROOT) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                n_dirs += 1
                scene_dir = entry.path
                # DA3 sidecars are generated per processed unit. Long videos are
                # exported as chunk_*/video.mp4; use those chunks instead of
                # also indexing the parent video and duplicating samples.
                root_chunk_candidates, _added_root_chunks = self._add_valid_immediate_chunks(
                    parent_dir=scene_dir,
                    parent_scene_id=entry.name,
                    scenes=scenes,
                    scene_data=scene_data,
                    start_img_ids=start_img_ids,
                )
                if root_chunk_candidates == 0:
                    self._maybe_add_sidecar_scene(
                        scene_id=entry.name,
                        meta_path=osp.join(scene_dir, "meta.json"),
                        video_path=osp.join(scene_dir, "video.mp4"),
                        pose_meta_path=osp.join(scene_dir, self.pose_meta_name),
                        scenes=scenes,
                        scene_data=scene_data,
                        start_img_ids=start_img_ids,
                    )
                try:
                    with os.scandir(scene_dir) as child_it:
                        for child in child_it:
                            if not child.is_dir():
                                continue
                            child_dir = child.path
                            if child.name.startswith("chunk_"):
                                continue

                            child_scene_id = f"{entry.name}/{child.name}"
                            child_chunk_candidates, _added_child_chunks = self._add_valid_immediate_chunks(
                                parent_dir=child_dir,
                                parent_scene_id=child_scene_id,
                                scenes=scenes,
                                scene_data=scene_data,
                                start_img_ids=start_img_ids,
                            )
                            if child_chunk_candidates == 0:
                                self._maybe_add_sidecar_scene(
                                    scene_id=child_scene_id,
                                    meta_path=osp.join(child_dir, "meta.json"),
                                    video_path=osp.join(child_dir, "video.mp4"),
                                    pose_meta_path=osp.join(child_dir, self.pose_meta_name),
                                    scenes=scenes,
                                    scene_data=scene_data,
                                    start_img_ids=start_img_ids,
                                )
                except OSError:
                    continue
        print(
            f"[VMS:{self.dataset_tag}] sidecar={self.pose_meta_name} "
            f"root_dirs={n_dirs} scenes/chunks={len(scenes)} "
            f"start_positions={len(start_img_ids)}",
            flush=True,
        )
        return {
            "scenes": scenes,
            "scene_data": scene_data,
            "start_img_ids": start_img_ids,
        }

    def _index_override_path(self) -> str | None:
        """Env override to load a prebuilt index pickle directly.

        Set ``VMS_INDEX_<TAG>`` (tag upper-cased, non-alnum → ``_``) or the
        generic ``VMS_INDEX_PATH`` to an existing ``.pkl``. When present it is
        passed to ``load_or_build`` as ``local_path`` so the very first
        ``try_load_pickle`` hits and we skip the lock / copy / build entirely
        (loads straight from the shared/network path).
        """
        tag_env = "VMS_INDEX_" + "".join(
            c if c.isalnum() else "_" for c in self.dataset_tag
        ).upper()
        for name in (tag_env, "VMS_INDEX_PATH"):
            p = os.environ.get(name, "").strip()
            if not p:
                continue
            if osp.isfile(p):
                return p
            print(
                f"[VMS:{self.dataset_tag}] {name}={p} not a file; ignoring override",
                flush=True,
            )
        # Built-in default for known tags (env still wins above).
        default = _DEFAULT_VMS_INDEX.get(self.dataset_tag)
        if default and osp.isfile(default):
            return default
        return None

    def _load_data(self):
        log = f"VMS:{self.dataset_tag}"
        override = self._index_override_path()
        if override is not None:
            print(f"[{log}] using index override → {override}", flush=True)
        d = load_or_build(
            log_prefix=log,
            local_path=override or self._cache_path(),
            shared_path=None if override is not None else self._shared_cache_path(),
            build_fn=self._build_index,
        )
        self.scenes = d["scenes"]
        self.scene_data = d["scene_data"]
        self.start_img_ids = d["start_img_ids"]
        self.invalid_scenes = {s: False for s in self.scenes}
        print(
            f"[{log}] split={self.split}  "
            f"scenes={len(self.scenes)}  "
            f"start_positions={len(self.start_img_ids)}",
            flush=True,
        )

    def __len__(self):
        return len(self.start_img_ids)

    # ── Camera loading (LRU-cached) ──────────────────────────────────────

    def _load_cameras(self, scene_id: str):
        if scene_id in self._cam_cache:
            self._cam_cache.move_to_end(scene_id)
            return self._cam_cache[scene_id]

        sd = self.scene_data[scene_id]
        with open(sd.get("pose_meta_path") or sd["meta_path"]) as f:
            meta = json.load(f)

        frames = meta["frames"]
        c2w_arr = np.asarray([fr["c2w"] for fr in frames], dtype=np.float32)
        K_arr = np.asarray([fr["K"] for fr in frames], dtype=np.float32)

        finite = (
            np.isfinite(c2w_arr).all(axis=(1, 2))
            & np.isfinite(K_arr).all(axis=(1, 2))
        )
        if not finite.all():
            for i in np.where(~finite)[0]:
                c2w_arr[i] = np.eye(4, dtype=np.float32)
                K_arr[i] = np.eye(3, dtype=np.float32)

        self._cam_cache[scene_id] = (c2w_arr, K_arr)
        while len(self._cam_cache) > _CAM_CACHE_MAX:
            self._cam_cache.popitem(last=False)
        return c2w_arr, K_arr

    # ── Frame reading ────────────────────────────────────────────────────

    def _read_frames(
        self, video_path: str, frame_indices: list[int]
    ) -> list[Image.Image]:
        """Sequential single-pass read of the requested frame indices."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise IOError(f"Empty/unreadable video: {video_path}")

        order = np.argsort(frame_indices)
        sorted_targets = [int(frame_indices[i]) for i in order]
        out_sorted: list[Image.Image | None] = [None] * len(sorted_targets)

        cur = 0
        pos = 0
        while pos < len(sorted_targets) and cur < total:
            target = sorted_targets[pos]
            if target - cur > 16:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if cur == target:
                while pos < len(sorted_targets) and sorted_targets[pos] == cur:
                    out_sorted[pos] = Image.fromarray(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                    pos += 1
            cur += 1

        cap.release()

        if any(im is None for im in out_sorted):
            got = sum(im is not None for im in out_sorted)
            raise IOError(
                f"Could not read all requested frames from {video_path} "
                f"(got {got}/{len(sorted_targets)})"
            )

        out: list[Image.Image | None] = [None] * len(frame_indices)
        for sorted_i, orig_i in enumerate(order):
            out[int(orig_i)] = out_sorted[sorted_i]
        return out  # type: ignore[return-value]

    # ── Main data method ─────────────────────────────────────────────────

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        scene_id, start_pos = self.start_img_ids[idx]

        for _ in range(max_retries):
            while self.invalid_scenes.get(scene_id, False):
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_id, start_pos = self.start_img_ids[idx]

            sd = self.scene_data[scene_id]
            total = int(sd["num_frames"])
            all_ids = list(range(total))

            pos = self.get_seq_from_start_id_novid(
                num_views, start_pos, all_ids, rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=1,
            )
            if pos is None:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_id, start_pos = self.start_img_ids[idx]
                continue

            try:
                c2w_all, K_all = self._load_cameras(scene_id)
            except Exception:
                self.invalid_scenes[scene_id] = True
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_id, start_pos = self.start_img_ids[idx]
                continue

            # Screen poses before paying for the video decode; __getitem__ owns the
            # retry bookkeeping for the raised rejection.
            if self._pose_filter_enabled:
                sel = [int(p) for p in pos]
                if max(sel) < len(c2w_all):
                    reason = self.pose_violation_reason(np.asarray(c2w_all)[sel])
                    if reason is not None:
                        raise BadPoseSampleError(
                            f"{self.dataset_tag}:{scene_id} pos={sel[0]}..{sel[-1]} {reason}"
                        )

            try:
                pil_imgs = self._read_frames(
                    sd["video_path"], [int(p) for p in pos]
                )
            except Exception:
                self.invalid_scenes[scene_id] = True
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_id, start_pos = self.start_img_ids[idx]
                continue

            caption = (
                sd.get("caption_txt")
                or sd.get("inline_caption")
                or self._caption_map.get(scene_id, "")
                or ""
            )

            views = []
            ok = True
            for pi, img in zip(pos, pil_imgs):
                pi = int(pi)
                if pi >= len(c2w_all):
                    ok = False
                    break

                c2w = c2w_all[pi]
                intr = K_all[pi].copy()

                depth_dummy = np.zeros(np.array(img).shape[:2], dtype=np.float32)
                rgb_pil, _, intr_scaled = self._crop_resize_if_necessary(
                    img, depth_dummy, intr, resolution, rng=rng, info=pi,
                )

                views.append(dict(
                    img=rgb_pil,
                    camera_pose=c2w.astype(np.float32),
                    camera_intrinsics=intr_scaled.astype(np.float32),
                    caption=caption,
                    ref_view_sampling=self.ref_view_sampling,
                    scene_id=str(scene_id),
                    start_pos=int(start_pos),
                ))

            if ok and len(views) == num_views:
                return views

            idx = rng.integers(low=0, high=len(self.start_img_ids))
            scene_id, start_pos = self.start_img_ids[idx]

        raise RuntimeError(
            f"[VMS:{self.dataset_tag}] could not assemble valid view set "
            f"after {max_retries} retries"
        )
