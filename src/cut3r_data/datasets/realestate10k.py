# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - RealEstate10K Multi-View Dataset

import collections
import hashlib
import json
import os
import os.path as osp
import pickle
import cv2
import numpy as np
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2

_CAM_CACHE_MAX = 512


class RE10K_Multi(BaseMultiViewDataset):
    """
    Supports two data layouts:

    1. Flat layout (original):
       ROOT/{scene_id}/{timestamp}.png + {timestamp}_cam.npz

    2. Nested/NeRF layout:
       ROOT/{shard}/{scene_id}/{timestamp}.png + transforms.json

    The layout is auto-detected: if ROOT contains subdirectories that
    themselves contain scene subdirectories with transforms.json, the
    nested layout is used.
    """

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        # Pop interval kwargs before forwarding: BaseMultiViewDataset does not accept them.
        self.min_interval = kwargs.pop('min_interval', 1)
        self.max_interval = kwargs.pop('max_interval', 128)
        super().__init__(*args, **kwargs)
        self._cam_cache = collections.OrderedDict()
        self.loaded_data = self._load_data()

        # Load scene-level captions (if available).
        # train_caption.json lives 2 levels up from ROOT (process/train → rel10k/).
        self._caption_map = {}
        for caption_dir in [osp.dirname(osp.dirname(self.ROOT)), osp.dirname(self.ROOT), self.ROOT]:
            for cname in ["train_caption.json", "test_caption.json"]:
                cpath = osp.join(caption_dir, cname)
                if osp.exists(cpath):
                    try:
                        with open(cpath) as _cf:
                            self._caption_map = json.load(_cf)
                        print(f"[RE10K] Loaded {len(self._caption_map)} captions from {cpath}")
                    except Exception:
                        pass
                    break
            if self._caption_map:
                break

    # ------------------------------------------------------------------
    # Layout detection & scene discovery (with disk cache)
    # ------------------------------------------------------------------

    def _cache_path(self):
        key = f"{self.ROOT}__nv{self.num_views}__ar{self.allow_repeat}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.ROOT, f".re10k_index_{tag}.pkl")

    def _discover_scenes_cached(self):
        """Discover scenes and cache to disk; subsequent calls load from cache."""
        cache = self._cache_path()
        if osp.exists(cache):
            with open(cache, "rb") as f:
                data = pickle.load(f)
            print(f"[RE10K] Loaded cached index ({len(data)} scenes) from {cache}")
            return data

        print(f"[RE10K] Building scene index for {self.ROOT} (first time only) ...")
        result = self._discover_scenes_raw()

        try:
            with open(cache, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[RE10K] Saved index cache to {cache}")
        except OSError:
            pass
        return result

    def _discover_scenes_raw(self):
        entries = sorted(os.listdir(self.ROOT))
        for e in entries:
            epath = osp.join(self.ROOT, e)
            if not osp.isdir(epath):
                continue
            children = os.listdir(epath)
            has_png = any(c.endswith('.png') for c in children)
            if has_png:
                return [
                    (d, osp.join(self.ROOT, d))
                    for d in entries
                    if osp.isdir(osp.join(self.ROOT, d))
                ]
            else:
                break

        scene_list = []
        for shard in sorted(entries):
            shard_dir = osp.join(self.ROOT, shard)
            if not osp.isdir(shard_dir):
                continue
            for scene in sorted(os.listdir(shard_dir)):
                scene_dir = osp.join(shard_dir, scene)
                if osp.isdir(scene_dir):
                    scene_list.append((scene, scene_dir))
        return scene_list

    # ------------------------------------------------------------------
    # Camera loading helpers
    # ------------------------------------------------------------------

    def _load_cam_npz(self, scene_dir, basename):
        cam = np.load(osp.join(scene_dir, basename + "_cam.npz"))
        return cam["intrinsics"], cam["pose"]

    def _load_cam_transforms_json(self, scene_dir, basename):
        """Load camera from transforms.json (custom NeRF-style format).

        RealEstate10K's official format (Google Research) stores per-frame
        3×4 **w2c** matrices in ``.txt`` files under **OpenCV / COLMAP**
        convention (X-right, Y-down, Z-forward). The preprocessing pipeline
        simply inverts each w2c → c2w and
        writes them as 4×4 ``transform_matrix`` entries in
        ``transforms.json``. It does NOT apply the standard
        nerfstudio/Blender OpenGL conversion; ``transforms.json`` here lacks
        the usual ``camera_model`` / ``applied_transform`` fields.

        Verification: per-view direction agreement vs DPT-recovered OpenCV
        poses gives cos≈+0.96~+0.97 (raw), but cos≈-0.78 if we wrongly
        treat it as OpenGL and convert. → Use as-is, no conversion.
        """
        if scene_dir in self._cam_cache:
            self._cam_cache.move_to_end(scene_dir)
        else:
            tf_path = osp.join(scene_dir, "transforms.json")
            with open(tf_path, "r") as f:
                tf = json.load(f)
            intrinsics = np.array([
                [tf["fl_x"], 0.0,       tf["cx"]],
                [0.0,        tf["fl_y"], tf["cy"]],
                [0.0,        0.0,       1.0],
            ], dtype=np.float32)
            frame_map = {}
            for fr in tf["frames"]:
                fp = fr["file_path"]
                name = osp.splitext(osp.basename(fp))[0]
                frame_map[name] = np.array(fr["transform_matrix"], dtype=np.float32)
            self._cam_cache[scene_dir] = (intrinsics, frame_map)
            while len(self._cam_cache) > _CAM_CACHE_MAX:
                self._cam_cache.popitem(last=False)

        intrinsics, frame_map = self._cam_cache[scene_dir]
        pose = frame_map[basename]
        return intrinsics.copy(), pose.copy()

    def _load_camera(self, scene_dir, basename):
        npz_path = osp.join(scene_dir, basename + "_cam.npz")
        if osp.exists(npz_path):
            return self._load_cam_npz(scene_dir, basename)
        return self._load_cam_transforms_json(scene_dir, basename)

    # ------------------------------------------------------------------
    # Data indexing
    # ------------------------------------------------------------------

    def _load_data(self):
        cache = self._cache_path()
        cache_full = cache.replace(".pkl", "_full.pkl")

        if osp.exists(cache_full):
            print(f"[RE10K] Loading full data index from cache ...")
            with open(cache_full, "rb") as f:
                d = pickle.load(f)
            self.scenes = d["scenes"]
            self.scene_dirs = d["scene_dirs"]
            self.sceneids = np.asarray(d["sceneids"], dtype=np.int32)
            self.images = d["images"]
            self.start_img_ids = d["start_img_ids"]
            self.scene_img_list = d["scene_img_list"]
            self.invalid_scenes = {s: False for s in self.scenes}
            print(f"[RE10K] {len(self.scenes)} scenes, {len(self.images)} images ready.")
            return

        raw_scenes = self._discover_scenes_cached()

        offset = 0
        scenes = []
        scene_dirs = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene_name, scene_dir in tqdm(raw_scenes, desc="Loading RE10K"):
            basenames = sorted(
                [f[:-4] for f in os.listdir(scene_dir) if f.endswith(".png")],
                key=lambda x: int(x),
            )

            num_imgs = len(basenames)
            if num_imgs == 0:
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = max(2, self.num_views // 3) if not self.allow_repeat else max(2, self.num_views // 3)

            if num_imgs < cut_off:
                continue

            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            start_img_ids.extend([(scene_name, id_) for id_ in start_img_ids_])
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene_name)
            scene_dirs.append(scene_dir)
            scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.scene_dirs = scene_dirs
        self.sceneids = np.array(sceneids, dtype=np.int32)
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list
        self.invalid_scenes = {s: False for s in self.scenes}

        try:
            with open(cache_full, "wb") as f:
                pickle.dump({
                    "scenes": scenes, "scene_dirs": scene_dirs,
                    "sceneids": sceneids, "images": images,
                    "start_img_ids": start_img_ids,
                    "scene_img_list": scene_img_list,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[RE10K] Saved full data index to {cache_full}")
        except OSError:
            pass

        print(f"[RE10K] {len(self.scenes)} scenes, {len(self.images)} images ready.")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]
        max_retries = 100

        retry_count = 0
        while invalid_seq and retry_count < max_retries:
            retry_count += 1
            
            while self.invalid_scenes[scene]:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]
            pos = self.get_seq_from_start_id_novid(
                num_views, start_id, all_image_ids, rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=1,  # keep time order (each "block" is 1 view → no permute)
            )
            
            if pos is None:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]
                continue
            
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            load_failed = False
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = self.scene_dirs[scene_id]
                basename = self.images[view_idx]
                
                try:
                    rgb_image = imread_cv2(osp.join(scene_dir, basename + ".png"))
                    intrinsics, camera_pose = self._load_camera(scene_dir, basename)
                except Exception:
                    self.invalid_scenes[scene] = True
                    load_failed = True
                    break
                
                depthmap = np.zeros_like(rgb_image)
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                views.append(dict(
                    img=rgb_image,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                ))

            if load_failed:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]
                continue
            
            if len(views) == num_views:
                invalid_seq = False
        
        if retry_count >= max_retries:
            raise RuntimeError(f"Could not find scene with enough frames for {num_views} views after {max_retries} retries")

        # Attach scene-level caption (if available).
        caption = self._caption_map.get(scene, "") or ""  # Never None
        for v in views:
            v["caption"] = caption

        return views
