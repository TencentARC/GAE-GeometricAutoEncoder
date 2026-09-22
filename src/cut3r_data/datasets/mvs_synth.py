import hashlib
import json
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2

_POSE_CACHE_MAX = 256


class MVSSynth_Multi(BaseMultiViewDataset):
    """MVS-Synth (GTA V) multi-view dataset.

    Layout:  ROOT/{scene_id}/images/{XXXX}.png
             ROOT/{scene_id}/poses/{XXXX}.json
             ROOT/{scene_id}/depths/{XXXX}.exr  (optional, unused)

    Per-frame JSON contains intrinsics (f_x, f_y, c_x, c_y) and a 4×4
    ``extrinsic`` matrix.

    Convention notes (per official MVS-Synth docs at
    https://phuang17.github.io/DeepMVS/mvs-synth.html):

      * ``extrinsic`` is **w2c** (world-to-camera).
      * The official docs warn: "rotation matrices have determinants of
        -1 instead of 1; this is an unusual convention and further
        processing may be necessary in some applications" — without
        explaining the cause.

    Empirically, the ``det(R) = -1`` is solely due to GTA V's *world*
    frame being left-handed (X-right, Y-forward, Z-up). The *camera*
    frame is still OpenCV (X-right, Y-down, Z-forward), so
    ``c2w = inv(extrinsic)`` is directly usable for OpenCV pinhole
    back-projection without any chirality "fix".

    DO NOT flip Z or Y when det(R)<0 — that breaks projection because
    it would inadvertently change the *camera* frame instead of the
    world frame. Verified by direct GT-pose + GT-depth point-cloud
    overlap (multi-view points align in world coordinates only with
    raw inv(extrinsic)).
    """

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        # Pop interval kwargs before forwarding: BaseMultiViewDataset does not accept them.
        self.min_interval = kwargs.pop("min_interval", 1)
        self.max_interval = kwargs.pop("max_interval", 64)
        super().__init__(*args, **kwargs)
        self._pose_cache = {}
        self._load_data()

    def _cache_path(self):
        key = f"mvssynth_{self.ROOT}__nv{self.num_views}__ar{self.allow_repeat}"
        tag = hashlib.md5(key.encode()).hexdigest()[:12]
        return osp.join(self.ROOT, f".mvssynth_index_{tag}.pkl")

    # ── Camera loading ────────────────────────────────────────────────────

    def _load_camera(self, scene_dir, basename):
        """Load per-frame intrinsics + c2w from pose JSON."""
        cache_key = (scene_dir, basename)
        if cache_key in self._pose_cache:
            return self._pose_cache[cache_key]

        pose_path = osp.join(scene_dir, "poses", basename + ".json")
        with open(pose_path, "r") as f:
            d = json.load(f)

        intrinsics = np.array([
            [d["f_x"], 0.0,      d["c_x"]],
            [0.0,      d["f_y"], d["c_y"]],
            [0.0,      0.0,      1.0],
        ], dtype=np.float32)

        extrinsic = np.array(d["extrinsic"], dtype=np.float64)
        # extrinsic is w2c with an OpenCV camera frame; raw inv() gives the
        # geometrically-correct c2w. Do NOT chirality-fix here.
        c2w = np.linalg.inv(extrinsic).astype(np.float32)

        if len(self._pose_cache) < _POSE_CACHE_MAX:
            self._pose_cache[cache_key] = (intrinsics, c2w)
        return intrinsics, c2w

    # ── Data indexing ─────────────────────────────────────────────────────

    def _load_data(self):
        cache_full = self._cache_path().replace(".pkl", "_full.pkl")
        if osp.exists(cache_full):
            print("[MVS-Synth] Loading full data index from cache …")
            with open(cache_full, "rb") as f:
                d = pickle.load(f)
            self.scenes = d["scenes"]
            self.scene_dirs = d["scene_dirs"]
            self.sceneids = np.asarray(d["sceneids"], dtype=np.int32)
            self.images = d["images"]
            self.start_img_ids = d["start_img_ids"]
            self.scene_img_list = d["scene_img_list"]
            self.invalid_scenes = {s: False for s in self.scenes}
            print(f"[MVS-Synth] {len(self.scenes)} scenes, "
                  f"{len(self.images)} images ready.")
            return

        print(f"[MVS-Synth] Building scene index for {self.ROOT} …")
        offset, j = 0, 0
        scenes, scene_dirs, sceneids = [], [], []
        images, start_img_ids, scene_img_list = [], [], []

        entries = sorted(os.listdir(self.ROOT))
        for scene_name in tqdm(entries, desc="Loading MVS-Synth"):
            scene_dir = osp.join(self.ROOT, scene_name)
            img_dir = osp.join(scene_dir, "images")
            pose_dir = osp.join(scene_dir, "poses")
            if not osp.isdir(img_dir) or not osp.isdir(pose_dir):
                continue

            basenames = sorted(
                f[:-4] for f in os.listdir(img_dir) if f.endswith(".png")
            )
            # Only keep frames that also have a pose JSON
            basenames = [
                bn for bn in basenames
                if osp.isfile(osp.join(pose_dir, bn + ".json"))
            ]
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
            print(f"[MVS-Synth] Saved full data index to {cache_full}")
        except OSError:
            pass
        print(f"[MVS-Synth] {len(self.scenes)} scenes, "
              f"{len(self.images)} images ready.")

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
                    rgb = imread_cv2(
                        osp.join(scene_dir, "images", basename + ".png"))
                    intrinsics, camera_pose = self._load_camera(
                        scene_dir, basename)
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
                return views
            idx = rng.integers(low=0, high=len(self.start_img_ids))
            scene, start_id = self.start_img_ids[idx]

        raise RuntimeError(
            f"[MVS-Synth] Could not find valid scene after {max_retries} retries")
