"""
ScanNet++ RGB multi-view dataset (preprocessed mp4 + meta.json).

Reads from the output of ``scripts/preprocess_scannetpp.py``:

    ROOT/
        <scene_id>/
            video.mp4    — 504x504 H.264, N=256 evenly-spaced frames
            depth.npz    — uint16 mm (N, 192, 192)   [not used here]
            meta.json    — per-frame name / original_idx / c2w / K

Each ``__getitem__`` returns ``num_views`` view dicts in the same format as
``RE10K_Multi`` / ``DL3DV_Multi`` / ``OpenVid_Multi``: ``img`` (PIL),
``camera_pose`` (4x4), ``camera_intrinsics`` (3x3), ``caption`` (empty),
``ref_view_sampling="prefix"``.

The video is read on the fly with ``cv2.VideoCapture`` and seeked by frame
index; the per-scene camera arrays are LRU-cached.

Used by the codec fine-tune pipeline (``scripts/train_codec.py``), which
encodes RGB through DA3 → VAE and reconstructs DA3 features. This class does
NOT load precomputed latents — for that, see ``ScanNetppLatent_Multi``.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.base.dataset_index_cache import (
    load_or_build,
    mirror_shared_path,
    resolve_shared_cache_dir,
)
from cut3r_data.base.fast_meta_index import (
    build_packed_scene_index,
    default_vms_index_workers,
)


_CAM_CACHE_MAX = 4096
_LOCAL_CACHE_ROOT = "/local-ssd/scannetpp_rgb_cache"


class ScanNetppRGB_Multi(BaseMultiViewDataset):
    """ScanNet++ multi-view RGB dataset backed by preprocessed mp4 files.

    Args:
        ROOT:          Output dir of ``scripts/preprocess_scannetpp.py``.
        split:         Optional ``"train"`` / ``"val"`` partition by scene hash.
        val_frac:      Fraction of scenes held out for val (default 0.02).
        min_interval:  Min frame stride between sampled views (default 1).
        max_interval:  Max frame stride between sampled views (default 8).
                       Smaller than OpenVid because preprocessed mp4 is already
                       down-sampled to 256 frames per scene.
    """

    def __init__(
        self,
        *args,
        ROOT,
        split: str | None = None,
        val_frac: float = 0.02,
        min_interval: int = 1,
        max_interval: int = 8,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.val_frac = float(val_frac)
        self.video = True
        self.is_metric = True  # ScanNet++ poses are metric (meters)
        self.ref_view_sampling = "prefix"
        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        # Pass `split` through to BaseMultiViewDataset so it's not clobbered.
        super().__init__(*args, split=split, **kwargs)
        self._cam_cache: collections.OrderedDict = collections.OrderedDict()
        self._load_data()

    # ── Index building ────────────────────────────────────────────────────

    def _cache_path(self) -> str:
        key = (
            f"scannetpp_{self.ROOT}__nv{self.num_views}"
            f"__split{self.split}__vf{self.val_frac}"
        )
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        os.makedirs(_LOCAL_CACHE_ROOT, exist_ok=True)
        return osp.join(_LOCAL_CACHE_ROOT, f"scannetpp_rgb_index_{tag}.pkl")

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
            f"[ScanNetppRGB] scanning {self.ROOT} (fast header + {w} workers) …",
            flush=True,
        )
        return build_packed_scene_index(
            self.ROOT,
            cut_off=max(2, self.num_views),
            split=self.split,
            val_frac=self.val_frac,
            with_caption=False,
            workers=w,
            progress_desc="Indexing ScanNetppRGB",
        )

    def _load_data(self):
        d = load_or_build(
            log_prefix="ScanNetppRGB",
            local_path=self._cache_path(),
            shared_path=self._shared_cache_path(),
            build_fn=self._build_index,
        )
        self.scenes = d["scenes"]
        self.scene_data = d["scene_data"]
        self.start_img_ids = d["start_img_ids"]
        self.invalid_scenes = {s: False for s in self.scenes}
        print(
            f"[ScanNetppRGB] split={self.split}  scenes={len(self.scenes)}  "
            f"start_positions={len(self.start_img_ids)}",
            flush=True,
        )

    def __len__(self):
        return len(self.start_img_ids)

    # ── Camera loading (LRU-cached) ──────────────────────────────────────

    def _load_cameras(self, scene_id: str):
        """Return (c2w_arr, K_arr) — both float32, shapes (N, 4, 4) and (N, 3, 3)."""
        if scene_id in self._cam_cache:
            self._cam_cache.move_to_end(scene_id)
            return self._cam_cache[scene_id]

        sd = self.scene_data[scene_id]
        with open(sd["meta_path"]) as f:
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

    def _read_frames(self, video_path: str, frame_indices: list[int]) -> list[Image.Image]:
        """Sequential single-pass read of the requested frame indices (sorted)."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise IOError(f"Empty/unreadable video: {video_path}")

        # Sorted unique targets so we can scan in one pass.
        order = np.argsort(frame_indices)
        sorted_targets = [int(frame_indices[i]) for i in order]
        out_sorted: list[Image.Image | None] = [None] * len(sorted_targets)

        cur = 0
        pos = 0
        while pos < len(sorted_targets) and cur < total:
            target = sorted_targets[pos]
            # Seek when the target is far ahead; otherwise stream forward.
            if target - cur > 16:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if cur == target:
                # Skip duplicate targets that resolve to the same frame.
                while pos < len(sorted_targets) and sorted_targets[pos] == cur:
                    out_sorted[pos] = Image.fromarray(
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    )
                    pos += 1
            cur += 1

        cap.release()

        if any(im is None for im in out_sorted):
            raise IOError(
                f"Could not read all requested frames from {video_path} "
                f"(got {sum(im is not None for im in out_sorted)}/{len(sorted_targets)})"
            )

        # Restore original (caller) order.
        out = [None] * len(frame_indices)
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

            try:
                pil_imgs = self._read_frames(sd["video_path"], [int(p) for p in pos])
            except Exception:
                self.invalid_scenes[scene_id] = True
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_id, start_pos = self.start_img_ids[idx]
                continue

            views = []
            ok = True
            for pi, img in zip(pos, pil_imgs):
                pi = int(pi)
                if pi >= len(c2w_all):
                    ok = False
                    break

                c2w = c2w_all[pi]
                intr = K_all[pi].copy()

                # All preprocessed frames are 504x504 with adjusted intrinsics, so
                # _crop_resize_if_necessary is effectively a no-op when resolution
                # also equals (504, 504). For (504, H<504) variants it will center
                # crop the principal point — safe because cx≈cy≈252.
                depth_dummy = np.zeros(np.array(img).shape[:2], dtype=np.float32)
                rgb_pil, _, intr_scaled = self._crop_resize_if_necessary(
                    img, depth_dummy, intr, resolution, rng=rng, info=pi,
                )

                views.append(dict(
                    img=rgb_pil,
                    camera_pose=c2w.astype(np.float32),
                    camera_intrinsics=intr_scaled.astype(np.float32),
                    caption="",
                    ref_view_sampling="prefix",
                ))

            if ok and len(views) == num_views:
                return views

            idx = rng.integers(low=0, high=len(self.start_img_ids))
            scene_id, start_pos = self.start_img_ids[idx]

        raise RuntimeError(
            f"[ScanNetppRGB] could not assemble valid view set after {max_retries} retries"
        )
