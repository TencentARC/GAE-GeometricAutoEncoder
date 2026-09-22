import collections
import hashlib
import json
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2

_CAM_CACHE_MAX = 512


class DL3DV_Multi(BaseMultiViewDataset):
    """DL3DV-10K multi-view dataset.

    Layout:  ROOT/{split}/{scene_hash}/images_4/{frame_XXXXX}.png
             ROOT/{split}/{scene_hash}/transforms.json

    transforms.json stores intrinsics for the ORIGINAL resolution (w x h).
    images_4/ contains 4x-downsampled images; intrinsics are scaled accordingly.
    """

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        # Pop interval kwargs before forwarding: BaseMultiViewDataset does not accept them.
        self.min_interval = kwargs.pop("min_interval", 1)
        self.max_interval = kwargs.pop("max_interval", 128)
        super().__init__(*args, **kwargs)
        self._cam_cache = collections.OrderedDict()
        self._load_data()

    def _cache_path(self):
        key = f"dl3dv_{self.ROOT}__nv{self.num_views}__ar{self.allow_repeat}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.ROOT, f".dl3dv_index_{tag}.pkl")

    def _discover_scenes(self):
        cache = self._cache_path()
        if osp.exists(cache):
            with open(cache, "rb") as f:
                data = pickle.load(f)
            print(f"[DL3DV] Loaded cached index ({len(data)} scenes)")
            return data

        print(f"[DL3DV] Building scene index for {self.ROOT} …")
        scene_list = []
        for split in sorted(os.listdir(self.ROOT)):
            split_dir = osp.join(self.ROOT, split)
            if not osp.isdir(split_dir):
                continue
            for scene in sorted(os.listdir(split_dir)):
                scene_dir = osp.join(split_dir, scene)
                if not osp.isdir(scene_dir):
                    continue
                # Support nerfstudio sub-dir variant
                ns = osp.join(scene_dir, "nerfstudio")
                actual = ns if osp.isdir(ns) else scene_dir
                img_dir = osp.join(actual, "images_4")
                tf_path = osp.join(actual, "transforms.json")
                if osp.isdir(img_dir) and osp.isfile(tf_path):
                    scene_list.append((scene, actual))
        try:
            with open(cache, "wb") as f:
                pickle.dump(scene_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
        print(f"[DL3DV] Found {len(scene_list)} scenes")
        return scene_list

    # ── Camera loading (from transforms.json, with intrinsic scaling) ─────

    def _load_camera(self, scene_dir, basename):
        if scene_dir in self._cam_cache:
            self._cam_cache.move_to_end(scene_dir)
        else:
            tf_path = osp.join(scene_dir, "transforms.json")
            with open(tf_path, "r") as f:
                tf = json.load(f)

            orig_w = float(tf.get("w", 3840))
            # Compute actual scale from an image on disk
            img_dir = osp.join(scene_dir, "images_4")
            sample = next((fn for fn in os.listdir(img_dir) if fn.endswith(".png")), None)
            if sample is not None:
                import struct
                with open(osp.join(img_dir, sample), "rb") as fh:
                    fh.read(16)
                    actual_w = struct.unpack(">I", fh.read(4))[0]
                scale = actual_w / orig_w
            else:
                scale = 0.25

            intrinsics = np.array([
                [tf["fl_x"] * scale, 0.0, tf["cx"] * scale],
                [0.0, tf["fl_y"] * scale, tf["cy"] * scale],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)

            # DL3DV-10K's transforms.json follows the **Nerfstudio / Blender
            # OpenGL** convention (X-right, Y-up, Z-back; camera looks along
            # -Z). Confirmed by:
            #   * Official docs: DL3DV-10K Issue #4 explicitly states
            #     "transforms.json uses OpenGL space, consistent with Blender /
            #      Nerfstudio convention"
            #   * The ``applied_transform`` field present in the file is the
            #     standard nerfstudio colmap→nerfstudio world-axis swap
            #   * (``camera_model: OPENCV`` in the file refers to the
            #     INTRINSICS distortion model — pinhole+radial — NOT the
            #     extrinsics frame.)
            #
            # We convert to OpenCV camera frame (X-right, Y-down, Z-forward)
            # to match pinhole back-projection used in ProPE / DPT ray
            # supervision. Minimal conversion = flip camera Y and Z axes:
            #   c2w_opencv = c2w_opengl @ diag(1, -1, -1, 1)
            #
            # Note: the official conversion in DL3DV docs additionally
            # undoes ``applied_transform`` to land in colmap-style world
            # coordinates. We skip that step because it only affects
            # absolute world-frame orientation, not multi-view geometric
            # consistency (which is what ProPE / DPT actually use).
            #
            # Empirical verification: per-view direction agreement vs
            # DPT-recovered OpenCV poses jumps from cos≈+0.13~+0.86 (raw,
            # Z reversed) to cos≈+0.99 after this conversion.
            _GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
            frame_map = {}
            for fr in tf["frames"]:
                name = osp.splitext(osp.basename(fr["file_path"]))[0]
                c2w_gl = np.array(fr["transform_matrix"], dtype=np.float32)
                frame_map[name] = c2w_gl @ _GL2CV

            self._cam_cache[scene_dir] = (intrinsics, frame_map)
            while len(self._cam_cache) > _CAM_CACHE_MAX:
                self._cam_cache.popitem(last=False)

        intrinsics, frame_map = self._cam_cache[scene_dir]
        pose = frame_map[basename]
        return intrinsics.copy(), pose.copy()

    # ── Data indexing ─────────────────────────────────────────────────────

    def _load_data(self):
        cache_full = self._cache_path().replace(".pkl", "_full.pkl")
        if osp.exists(cache_full):
            print("[DL3DV] Loading full data index from cache …")
            with open(cache_full, "rb") as f:
                d = pickle.load(f)
            self.scenes = d["scenes"]
            self.scene_dirs = d["scene_dirs"]
            self.sceneids = np.asarray(d["sceneids"], dtype=np.int32)
            self.images = d["images"]
            self.start_img_ids = d["start_img_ids"]
            self.scene_img_list = d["scene_img_list"]
            self.invalid_scenes = {s: False for s in self.scenes}
            print(f"[DL3DV] {len(self.scenes)} scenes, {len(self.images)} images ready.")
            return

        raw_scenes = self._discover_scenes()

        offset, j = 0, 0
        scenes, scene_dirs, sceneids = [], [], []
        images, start_img_ids, scene_img_list = [], [], []

        for scene_name, scene_dir in tqdm(raw_scenes, desc="Loading DL3DV"):
            img_dir = osp.join(scene_dir, "images_4")
            basenames = sorted(
                [f[:-4] for f in os.listdir(img_dir) if f.endswith(".png")]
            )
            num_imgs = len(basenames)
            cut_off = max(2, self.num_views // 3)
            if num_imgs < cut_off:
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_ids = img_ids[: num_imgs - cut_off + 1]
            start_img_ids.extend([(scene_name, sid) for sid in start_ids])
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
            print(f"[DL3DV] Saved full data index to {cache_full}")
        except OSError:
            pass
        print(f"[DL3DV] {len(self.scenes)} scenes, {len(self.images)} images ready.")

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        scene, start_id = self.start_img_ids[idx]
        for _ in range(max_retries):
            while self.invalid_scenes.get(scene, False):
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
            views, ok = [], True
            for vi in image_idxs:
                sid = self.sceneids[vi]
                scene_dir = self.scene_dirs[sid]
                basename = self.images[vi]
                try:
                    rgb = imread_cv2(osp.join(scene_dir, "images_4", basename + ".png"))
                    intrinsics, camera_pose = self._load_camera(scene_dir, basename)
                except Exception:
                    self.invalid_scenes[scene] = True
                    ok = False
                    break

                depthmap = np.zeros_like(rgb)
                rgb, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb, depthmap, intrinsics, resolution, rng=rng, info=vi)
                views.append(dict(
                    img=rgb,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                ))

            if ok and len(views) == num_views:
                # Load scene-level caption from wan2_caption.json (if available).
                caption = None
                caption_path = osp.join(scene_dir, "wan2_caption.json")
                if osp.exists(caption_path):
                    try:
                        with open(caption_path) as _cf:
                            caption = json.load(_cf).get("caption", None)
                    except Exception:
                        pass
                for v in views:
                    v["caption"] = caption or ""  # same caption for all views in scene
                return views
            idx = rng.integers(low=0, high=len(self.start_img_ids))
            scene, start_id = self.start_img_ids[idx]

        raise RuntimeError(
            f"[DL3DV] Could not find valid scene after {max_retries} retries")
