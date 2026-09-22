# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only)
# Modified for GLD project - HyperSim Multi-View Dataset

import os
import os.path as osp
import cv2
import numpy as np

from cut3r_data.base.base_multiview_dataset import BaseMultiViewDataset
from cut3r_data.utils.image import imread_cv2


class HyperSim_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.min_interval = kwargs.get('min_interval', 1)
        self.max_interval = kwargs.get('max_interval', 4)
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):
        self.all_scenes = sorted([
            f for f in os.listdir(self.ROOT) 
            if os.path.isdir(osp.join(self.ROOT, f))
        ])
        subscenes = []
        for scene in self.all_scenes:
            subscenes.extend([
                osp.join(scene, f)
                for f in os.listdir(osp.join(self.ROOT, scene))
                if os.path.isdir(osp.join(self.ROOT, scene, f))
                and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
            ])

        offset = 0
        scenes = []
        sceneids = []
        images = []
        start_img_ids = []
        scene_img_list = []
        j = 0
        
        for scene in subscenes:
            scene_dir = osp.join(self.ROOT, scene)
            rgb_paths = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            assert len(rgb_paths) > 0, f"{scene_dir} is empty."
            
            num_imgs = len(rgb_paths)
            # Use minimum cut_off of 2 - adaptive sampling handles the rest
            cut_off = max(2, self.num_views // 3) if not self.allow_repeat else max(2, self.num_views // 3)
            if num_imgs < cut_off:
                print(f"Skipping {scene} hypersim (only {num_imgs} frames)")
                continue
                
            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.scene_img_list = scene_img_list
        self.start_img_ids = start_img_ids

    def __len__(self):
        return len(self.start_img_ids) * 10

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        max_retries = 100
        retry_count = 0
        
        while retry_count < max_retries:
            retry_count += 1
            actual_idx = idx // 10
            start_id = self.start_img_ids[actual_idx]
            scene_id = self.sceneids[start_id]
            all_image_ids = self.scene_img_list[scene_id]
            pos = self.get_seq_from_start_id_novid(
                num_views,
                start_id,
                all_image_ids,
                rng,
                min_interval=self.min_interval,
                max_interval=self.max_interval,
                block_shuffle=16,
            )
            
            # Skip if not enough frames
            if pos is None:
                idx = rng.integers(low=0, high=len(self) // 10) * 10
                continue
                
            image_idxs = np.array(all_image_ids)[pos]
            
            views = []
            for v, view_idx in enumerate(image_idxs):
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                rgb_path = self.images[view_idx]
                cam_path = rgb_path.replace("rgb.png", "cam.npz")

                rgb_image = imread_cv2(osp.join(scene_dir, rgb_path), cv2.IMREAD_COLOR)
                cam_file = np.load(osp.join(scene_dir, cam_path))
                intrinsics = cam_file["intrinsics"].astype(np.float32)
                camera_pose = cam_file["pose"].astype(np.float32)
                
                depthmap = np.zeros_like(rgb_image)
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.75, 0.2, 0.05])

                views.append(dict(
                    img=rgb_image,
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                ))
            
            if len(views) == num_views:
                return views
        
        raise RuntimeError(f"Could not find scene with enough frames for {num_views} views")
