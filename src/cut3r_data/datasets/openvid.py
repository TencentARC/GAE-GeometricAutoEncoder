"""
OpenVid-1M multi-view dataset for GLD training.

Reads pre-computed camera poses from preprocess_openvid_cameras.py output.
Each JSON file stores ALL frame poses for one video; this class samples
subsets at training time with configurable interval, matching the
BaseMultiViewDataset interface used by RE10K and DL3DV.

Layout:
    ROOT/               (camera JSON dir, e.g. /path/to/openvid_cameras)
        {video_stem}.json
    VIDEO_DIR/          (video mp4 dir, e.g. /path/to/OpenVid-1M/video)
        {video_stem}.mp4

Each JSON contains:
    c2w:            list of (4,4) c2w matrices (one per extracted frame)
    intrinsics:     list of (3,3) intrinsic matrices
    frame_indices:  list of raw video frame indices
    caption:        str
    video_path:     str (original path, may differ from VIDEO_DIR)
    image_height, image_width: int
"""

import collections
import hashlib
import json
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2


_CAM_CACHE_MAX = 2048


class OpenVid_Multi(BaseMultiViewDataset):
    """OpenVid-1M multi-view dataset with pre-computed DA3 camera poses."""

    def __init__(self, *args, ROOT, VIDEO_DIR=None, **kwargs):
        """
        Args:
            ROOT:       Directory with camera JSON files from preprocess_openvid_cameras.py
            VIDEO_DIR:  Directory with .mp4 files. If None, uses video_path from JSON.
        """
        self.ROOT = ROOT
        self.VIDEO_DIR = VIDEO_DIR
        self.video = True
        self.is_metric = True  # OpenVid uses DA3 metric poses
        # Video data: use prefix ref_view_sampling (first N frames as ref)
        self.ref_view_sampling = "prefix"
        self.min_interval = kwargs.pop("min_interval", 1)
        self.max_interval = kwargs.pop("max_interval", 50)
        super().__init__(*args, **kwargs)
        self._cam_cache = collections.OrderedDict()
        self._load_data()

    # ── Index building ────────────────────────────────────────────────────

    def _cache_path(self):
        key = f"openvid_{self.ROOT}__nv{self.num_views}__ar{self.allow_repeat}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.ROOT, f".openvid_index_{tag}.pkl")

    def _load_data(self):
        cache = self._cache_path()
        if osp.exists(cache):
            print(f"[OpenVid] Loading cached index from {cache} …")
            with open(cache, "rb") as f:
                d = pickle.load(f)
            self.scenes = d["scenes"]
            self.scene_data = d["scene_data"]
            self.start_img_ids = d["start_img_ids"]
            self.invalid_scenes = {s: False for s in self.scenes}
            print(f"[OpenVid] {len(self.scenes)} videos, "
                  f"{len(self.start_img_ids)} start positions ready.")
            return

        print(f"[OpenVid] Building index for {self.ROOT} (first time only) …")
        json_files = sorted([f for f in os.listdir(self.ROOT) if f.endswith(".json")])
        print(f"[OpenVid] Found {len(json_files)} camera JSON files.")

        scenes = []
        scene_data = {}  # scene_name → metadata
        start_img_ids = []

        cut_off = max(2, self.num_views)
        for jf in tqdm(json_files, desc="Indexing OpenVid"):
            scene_name = jf[:-5]  # strip .json
            json_path = osp.join(self.ROOT, jf)
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception:
                continue

            num_frames = meta.get("num_frames", len(meta.get("frame_indices", [])))
            if num_frames < cut_off:
                continue

            # Only use videos with metric camera poses
            if not meta.get("is_metric", False):
                continue

            # Validate video exists
            video_path = self._resolve_video_path(meta, scene_name)
            if video_path is None:
                continue

            scenes.append(scene_name)
            scene_data[scene_name] = {
                "json_path": json_path,
                "video_path": video_path,
                "num_frames": num_frames,
                "caption": meta.get("caption", ""),
                "frame_indices": meta["frame_indices"],
                "image_height": meta.get("image_height", 384),
                "image_width": meta.get("image_width", 640),
                "scale_factor": meta.get("scale_factor", 1.0),
                "is_metric": meta.get("is_metric", 0),
            }

            # Build start positions: each valid starting frame
            num_start = max(1, num_frames - cut_off + 1)
            for si in range(num_start):
                start_img_ids.append((scene_name, si))

        self.scenes = scenes
        self.scene_data = scene_data
        self.start_img_ids = start_img_ids
        self.invalid_scenes = {s: False for s in scenes}

        # Save cache
        try:
            os.makedirs(self.ROOT, exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump({
                    "scenes": scenes,
                    "scene_data": scene_data,
                    "start_img_ids": start_img_ids,
                }, f)
            print(f"[OpenVid] Cached index to {cache}")
        except Exception as e:
            print(f"[OpenVid] Warning: could not cache index: {e}")

        print(f"[OpenVid] {len(scenes)} videos, {len(start_img_ids)} start positions.")

    def _resolve_video_path(self, meta, scene_name):
        """Find the actual video file path."""
        # Try VIDEO_DIR first
        if self.VIDEO_DIR:
            for ext in [".mp4", ""]:
                p = osp.join(self.VIDEO_DIR, scene_name + ext)
                if osp.exists(p):
                    return p
        # Try original path from JSON
        orig = meta.get("video_path", "")
        if orig and osp.exists(orig):
            return orig
        return None

    def __len__(self):
        return len(self.start_img_ids)

    # ── Camera loading (with cache) ──────────────────────────────────────

    def _load_cameras(self, scene_name):
        """Load c2w and intrinsics for a scene (cached)."""
        if scene_name in self._cam_cache:
            self._cam_cache.move_to_end(scene_name)
            return self._cam_cache[scene_name]  # (c2w, intr, frame_indices, scale_factor)

        sd = self.scene_data[scene_name]
        with open(sd["json_path"]) as f:
            meta = json.load(f)

        c2w_list = np.array(meta["c2w"], dtype=np.float32)       # (N, 4, 4)
        intr_list = np.array(meta["intrinsics"], dtype=np.float32)  # (N, 3, 3)
        frame_indices = meta["frame_indices"]
        scale_factor = float(meta.get("scale_factor", 1.0))

        # Validate metric poses: check for NaN/Inf
        valid = (np.isfinite(c2w_list).all(axis=(1, 2))
                 & np.isfinite(intr_list).all(axis=(1, 2)))
        if not valid.all():
            # Replace invalid poses with identity
            for i in range(len(c2w_list)):
                if not valid[i]:
                    c2w_list[i] = np.eye(4, dtype=np.float32)

        self._cam_cache[scene_name] = (c2w_list, intr_list, frame_indices, scale_factor)
        while len(self._cam_cache) > _CAM_CACHE_MAX:
            self._cam_cache.popitem(last=False)

        return c2w_list, intr_list, frame_indices, scale_factor

    # ── Frame reading ────────────────────────────────────────────────────

    def _read_video_frame(self, video_path, frame_idx):
        """Read a single frame from video by index."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IOError(f"Cannot read frame {frame_idx} from {video_path}")
        # BGR → RGB, then to PIL
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ── Main data method ─────────────────────────────────────────────────

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        scene_name, start_pos = self.start_img_ids[idx]

        for retry in range(max_retries):
            while self.invalid_scenes.get(scene_name, False):
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_name, start_pos = self.start_img_ids[idx]

            sd = self.scene_data[scene_name]
            total_poses = sd["num_frames"]

            # Select frame positions using interval sampling (same as RE10K/DL3DV)
            all_ids = list(range(total_poses))
            pos = self.get_seq_from_start_id_novid(
                num_views, start_pos, all_ids, rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=1,
            )
            if pos is None:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_name, start_pos = self.start_img_ids[idx]
                continue

            # Load cameras
            try:
                c2w_all, intr_all, frame_indices_all, scale_factor = self._load_cameras(scene_name)
            except Exception:
                self.invalid_scenes[scene_name] = True
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene_name, start_pos = self.start_img_ids[idx]
                continue

            # Read selected frames from video
            views = []
            ok = True
            for pi in pos:
                if pi >= len(frame_indices_all):
                    ok = False
                    break

                raw_frame_idx = frame_indices_all[pi]
                c2w = c2w_all[pi]    # (4, 4)
                intr = intr_all[pi].copy()  # (3, 3) — from DA3 processing resolution

                try:
                    rgb = self._read_video_frame(sd["video_path"], raw_frame_idx)
                except Exception:
                    self.invalid_scenes[scene_name] = True
                    ok = False
                    break

                # Scale intrinsics from DA3 processing resolution to actual frame size.
                # DA3 intrinsics (fx, cx, fy, cy) are for the DA3 process_res (~504),
                # NOT the original video resolution. Infer DA3 res from cx/cy ≈ w/2, h/2.
                actual_w, actual_h = rgb.size  # PIL Image
                da3_w = round(intr[0, 2] * 2)  # cx ≈ da3_w / 2
                da3_h = round(intr[1, 2] * 2)  # cy ≈ da3_h / 2
                if da3_w > 0 and da3_h > 0 and (actual_w != da3_w or actual_h != da3_h):
                    sx = actual_w / da3_w
                    sy = actual_h / da3_h
                    intr[0, 0] *= sx  # fx
                    intr[0, 2] *= sx  # cx
                    intr[1, 1] *= sy  # fy
                    intr[1, 2] *= sy  # cy

                depthmap = np.zeros_like(np.array(rgb))
                rgb_arr, depthmap, intr_scaled = self._crop_resize_if_necessary(
                    rgb, depthmap, intr, resolution, rng=rng, info=pi,
                )

                views.append(dict(
                    img=rgb_arr,
                    camera_pose=c2w.astype(np.float32),
                    camera_intrinsics=intr_scaled.astype(np.float32),
                    caption=sd["caption"],
                    ref_view_sampling="prefix",
                ))

            if ok and len(views) == num_views:
                return views

            idx = rng.integers(low=0, high=len(self.start_img_ids))
            scene_name, start_pos = self.start_img_ids[idx]

        raise RuntimeError(
            f"[OpenVid] Could not find valid scene after {max_retries} retries"
        )
