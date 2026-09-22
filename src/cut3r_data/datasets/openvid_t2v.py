"""
OpenVid-1M T2V dataset for GLD training (no camera pose, text-only).

Reads OpenVid-1M CSV + video files directly. No pre-computed poses needed.
All views use identity camera pose → model learns text-conditioned generation.

Layout:
    CSV_PATH:   /path/to/OpenVid-1M/data/train/OpenVid-1M.csv
    VIDEO_DIR:  /path/to/OpenVid-1M/video/

Config example:
    dataset_openvid_t2v: OpenVidT2V_Multi(
        CSV_PATH="/path/to/OpenVid-1M/data/train/OpenVid-1M.csv",
        VIDEO_DIR="/path/to/OpenVid-1M/video",
        resolution=[(504, 504), (504,378)],
        num_views=8, n_corres=0, min_interval=5, max_interval=30)
"""

import csv
import hashlib
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset


class OpenVidT2V_Multi(BaseMultiViewDataset):
    """OpenVid-1M text-to-video dataset. No camera pose — identity c2w."""

    def __init__(self, *args, CSV_PATH, VIDEO_DIR, **kwargs):
        self.CSV_PATH = CSV_PATH
        self.VIDEO_DIR = VIDEO_DIR
        self.video = True
        self.is_metric = False
        self.min_interval = kwargs.pop("min_interval", 5)
        self.max_interval = kwargs.pop("max_interval", 30)
        self.min_aesthetic = kwargs.pop("min_aesthetic", 4.0)
        self.ref_view_sampling = "prefix"  # T2V uses cond_num=0 anyway
        super().__init__(*args, **kwargs)
        self._load_data()

    # ── Index ─────────────────────────────────────────────────────────────

    def _cache_path(self):
        key = f"openvid_t2v_{self.CSV_PATH}_{self.VIDEO_DIR}_nv{self.num_views}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.VIDEO_DIR, f".openvid_t2v_index_{tag}.pkl")

    def _load_data(self):
        cache = self._cache_path()
        if osp.exists(cache):
            print(f"[OpenVidT2V] Loading cached index …")
            with open(cache, "rb") as f:
                d = pickle.load(f)
            self.samples = d["samples"]
            self.start_ids = d["start_ids"]
            self.invalid = set()
            print(f"[OpenVidT2V] {len(self.samples)} videos, "
                  f"{len(self.start_ids)} start positions ready.")
            return

        print(f"[OpenVidT2V] Building index from {self.CSV_PATH} …")
        samples = []  # list of dicts: video_path, caption, num_frames
        # First pass: filter by metadata only (fast, no disk I/O)
        candidates = []
        with open(self.CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row["video"]
                caption = row.get("caption", "")
                num_frames = int(row.get("frame", 0))
                aesthetic = float(row.get("aesthetic score", 0))

                # Filters (metadata only — no os.path.exists)
                if num_frames < self.num_views * 5:
                    continue
                if aesthetic < self.min_aesthetic:
                    continue

                candidates.append({
                    "video_path": osp.join(self.VIDEO_DIR, video_name),
                    "caption": caption,
                    "num_frames": num_frames,
                })
        print(f"[OpenVidT2V] {len(candidates)} candidates after metadata filter.")

        # Second pass: batch-check which videos exist on disk
        # Build set of existing filenames for O(1) lookup (one listdir vs 1M stat calls)
        print(f"[OpenVidT2V] Listing {self.VIDEO_DIR} for existence check ...")
        try:
            existing_files = set(os.listdir(self.VIDEO_DIR))
        except OSError:
            existing_files = None
            print("[OpenVidT2V] WARNING: could not list VIDEO_DIR, skipping existence check")

        for c in tqdm(candidates, desc="Filtering OpenVid"):
            if existing_files is not None:
                basename = osp.basename(c["video_path"])
                if basename not in existing_files:
                    continue
            samples.append(c)

        # Build start_ids: (sample_idx, start_frame)
        start_ids = []
        for si, s in enumerate(samples):
            # How many valid starting positions
            max_span = (self.num_views - 1) * self.max_interval
            num_starts = max(1, s["num_frames"] - max_span)
            # Limit to avoid huge index
            num_starts = min(num_starts, 10)
            for j in range(num_starts):
                start_frame = j * (s["num_frames"] // num_starts)
                start_ids.append((si, start_frame))

        self.samples = samples
        self.start_ids = start_ids
        self.invalid = set()

        print(f"[OpenVidT2V] {len(samples)} videos, {len(start_ids)} start positions.")

        try:
            with open(cache, "wb") as f:
                pickle.dump({"samples": samples, "start_ids": start_ids}, f)
            print(f"[OpenVidT2V] Cached to {cache}")
        except Exception as e:
            print(f"[OpenVidT2V] Warning: cache write failed: {e}")

    def __len__(self):
        return len(self.start_ids)

    # ── Frame extraction ──────────────────────────────────────────────────

    def _extract_frames(self, video_path, start_frame, num_views, rng):
        """Extract num_views frames and their physical frame indices."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None

        # Compute intervals
        available = total - start_frame
        if available < num_views:
            cap.release()
            return None

        max_int = min(self.max_interval, (available - 1) // (num_views - 1))
        max_int = max(max_int, 1)  # At least 1, but don't force min_interval up
        min_int = min(self.min_interval, max_int)

        intervals = [rng.integers(min_int, max_int + 1) for _ in range(num_views - 1)]
        indices = [start_frame]
        for iv in intervals:
            indices.append(indices[-1] + iv)

        # Clamp to valid range
        indices = [min(i, total - 1) for i in indices]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        cap.release()
        return frames, indices

    # ── Main ──────────────────────────────────────────────────────────────

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 50
        sample_idx, start_frame = self.start_ids[idx]

        for _ in range(max_retries):
            while sample_idx in self.invalid:
                idx = rng.integers(0, len(self.start_ids))
                sample_idx, start_frame = self.start_ids[idx]

            sample = self.samples[sample_idx]
            scene_id = osp.splitext(osp.basename(sample["video_path"]))[0]
            extracted = self._extract_frames(
                sample["video_path"], start_frame, num_views, rng
            )

            if extracted is None:
                self.invalid.add(sample_idx)
                idx = rng.integers(0, len(self.start_ids))
                sample_idx, start_frame = self.start_ids[idx]
                continue

            frames, frame_indices = extracted
            if len(frames) != num_views or len(frame_indices) != num_views:
                self.invalid.add(sample_idx)
                idx = rng.integers(0, len(self.start_ids))
                sample_idx, start_frame = self.start_ids[idx]
                continue

            # Build views with identity pose + dummy intrinsics
            views = []
            for frame_img, frame_idx in zip(frames, frame_indices):
                w, h = frame_img.size
                # Dummy intrinsics: focal ~ image size, principal point at center (pixels)
                focal = max(w, h)
                intrinsics = np.array([
                    [focal, 0.0, w / 2.0],
                    [0.0, focal, h / 2.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32)

                depthmap = np.zeros((h, w, 3), dtype=np.uint8)
                frame_img_proc, depthmap, intrinsics = self._crop_resize_if_necessary(
                    frame_img, depthmap, intrinsics, resolution, rng=rng, info=idx,
                )

                views.append(dict(
                    img=frame_img_proc,
                    camera_pose=np.eye(4, dtype=np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    frame_idx=int(frame_idx),
                    caption=sample["caption"],
                    ref_view_sampling=self.ref_view_sampling,
                    is_t2v=True,
                    scene_id=scene_id,
                    start_pos=int(start_frame),
                ))

            return views

        raise RuntimeError(
            f"[OpenVidT2V] Could not find valid video after {max_retries} retries"
        )
