import os
import torch
import random
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
# from torchvision.io import read_video
from video.utils.io import read_video
from depth_anything_3.utils.io.input_processor import InputProcessor
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union, List


################################################################################
#                           Abstract Base Class                                 #
################################################################################

class MultiviewDatasetBase(ABC):
    """
    Abstract base class for multiview datasets.
    All multiview datasets must implement this interface.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Returns:
            dict with keys:
                - "gt_inp": (V, 3, H, W) tensor in [0, 1] range
                - "fxfycxcy": (V, 4) intrinsics [fx, fy, cx, cy]
                - "c2w": (V, 4, 4) camera-to-world extrinsics
                - "video_id": str identifier
                - "frame_indices": (V,) frame indices
        """
        pass

class DA3VideoDataset(Dataset):
    """
    Loads an MP4, picks 2 random frames, preprocesses using DA3 InputProcessor.
    Returns (1, 2, 3, H, W)
    """

    def __init__(
        self,
        video_path,
        process_res=224,
        process_res_method="upper_bound_resize",
        l=100,
    ):
        mp4_files=os.listdir(video_path)
        self.video_paths = [os.path.join(video_path,vid) for vid in mp4_files]
        if 'test' in video_path:
            self.video_paths=self.video_paths[:l]

        self.process_res = process_res
        self.process_res_method = process_res_method

        self.processor = InputProcessor()

    def __len__(self):
        return len(self.video_paths)

    def _load_video_frames(self, path):
        """Reads video -> list of PIL frames."""
        video= read_video(path, pts_unit="sec")  # T,H,W,C

        frames = []
        for frame in video:
            arr = frame.numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
            frames.append(pil)

        return frames

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        # --------------------------------------
        # Load video frames
        # --------------------------------------
        frames = self._load_video_frames(path)
        T = len(frames)

        if T < 2:
            print(path)
            raise ValueError(f"Video {path} has fewer than 2 frames.")

        # --------------------------------------
        # Pick 2 random DIFFERENT frames
        # --------------------------------------
        idx1, idx2 = random.sample(range(T), 2)
        selected_frames = [frames[idx1], frames[idx2]]

        # --------------------------------------
        # Preprocess using DA3 InputProcessor
        # IMPORTANT: disable internal multiprocessing!
        # --------------------------------------
        batch_tensor, _, _ = self.processor(
            image=selected_frames,
            intrinsics=None,
            extrinsics=None,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            num_workers=1,            # ← AVOID nested multiprocessing
            sequential=True,
        )
        batch_tensor_gt, _, _ = self.processor(
            image=selected_frames,
            intrinsics=None,
            extrinsics=None,
            process_res=self.process_res,  # Same as encoder resolution
            process_res_method=self.process_res_method,
            num_workers=1,            # ← AVOID nested multiprocessing
            sequential=True,
        )
        # Output: (1, 2, 3, H, W)
        return {"enc_inp": batch_tensor,
                "gt_inp": batch_tensor_gt}
    

class RealEstate10KDataset(MultiviewDatasetBase, Dataset):
    """
    RealEstate-10K multiview dataset.
    
    Loads an MP4, picks N views (frames), preprocesses using DA3 InputProcessor.
    Additionally loads RealEstate-10K pose .pt and returns intrinsics + c2w
    *aligned with the encoder resolution* (e.g., 252).

    Returns:
        {
            "gt_inp":       (V, 3, H_enc, W_enc),
            "fxfycxcy":     (V, 4),    # intrinsics for H_enc,W_enc
            "c2w":          (V, 4, 4), # extrinsics (resolution-independent)
            "video_id":     str,
            "frame_indices": (V,)
        }
    """

    def __init__(
        self,
        video_path,
        pose_path,                # directory containing *.pt pose files
        num_views: int,           # number of views (frames) per sample
        cond_num: int,            # number of reference views (must be < num_views)
        ref_view_sampling: str,   # "prefix" | "random" | "interpolate"
        process_res=252,          # you use 252 for the model
        mode="train",             # train / test
        overfit=False,
        process_res_method="upper_bound_resize",
        l=100,
        test_baseline=None,
    ):
        mp4_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
        mp4_files = sorted(mp4_files)
        self.video_paths = [os.path.join(video_path, vid) for vid in mp4_files]

        # if "test" in video_path:
        #     self.video_paths = self.video_paths[:l]

        # Pose directory (e.g., .../real-estate-10k/test_poses)
        self.pose_path = pose_path
        self.mode = mode
        self.test_baseline = test_baseline

        if num_views is None:
            raise ValueError("DA3VideoDataset_Pose requires explicit num_views (no implicit default).")
        if not isinstance(num_views, int):
            raise TypeError(f"num_views must be int, got {type(num_views)}")
        if num_views < 2:
            raise ValueError(f"num_views must be >= 2 for multiview training, got {num_views}")
        self.num_views = int(num_views)

        if cond_num is None:
            raise ValueError("cond_num must be provided explicitly.")
        if not isinstance(cond_num, int):
            raise TypeError(f"cond_num must be int, got {type(cond_num)}")
        if not (0 <= int(cond_num) <= self.num_views):
            raise ValueError(f"cond_num must satisfy 0 <= cond_num <= num_views, got cond_num={cond_num}, num_views={self.num_views}")
        self.cond_num = int(cond_num)

        if ref_view_sampling is None:
            raise ValueError("ref_view_sampling must be provided explicitly.")
        if not isinstance(ref_view_sampling, str):
            raise TypeError(f"ref_view_sampling must be str, got {type(ref_view_sampling)}")
        ref_view_sampling = ref_view_sampling.lower()
        if ref_view_sampling not in {"prefix", "random", "interpolate"}:
            raise ValueError(f"Unknown ref_view_sampling={ref_view_sampling}. Use one of: prefix, random, interpolate.")
        self.ref_view_sampling = ref_view_sampling

        self.process_res = process_res
        self.process_res_method = process_res_method

        self.processor = InputProcessor()
        
        self.overfit=overfit

    def __len__(self):
        return len(self.video_paths)

    def _load_video_frames(self, path):
        """Reads video -> list of PIL frames."""
        video = read_video(path, pts_unit="sec")  # [T, H, W, C]

        frames = []
        for frame in video:
            arr = frame.numpy().astype(np.uint8)
            pil = Image.fromarray(arr)
            frames.append(pil)

        return frames

    @staticmethod
    def _pose_row_to_intrinsics_c2w_at_original(pose_row, width, height):
        """
        pose_row: 1D tensor of shape [18]
            [ fx_n, fy_n, cx_n, cy_n,  w2c_flat(12), extra0, extra1 ]

        width, height: original image size (e.g. 256,256).

        Returns:
            fxfycxcy_orig: torch.FloatTensor [4] at (width,height)
            c2w:           torch.FloatTensor [4, 4]
        """
        pose_row = pose_row.cpu()

        # ---- Intrinsics at original resolution ----
        fx_n, fy_n, cx_n, cy_n = pose_row[:4].tolist()  # normalized (0~1)

        fx = fx_n * width
        fy = fy_n * height
        cx = cx_n * width
        cy = cy_n * height

        fxfycxcy_orig = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

        # ---- Extrinsics: 3x4 world-to-camera ----
        w2c_3x4 = pose_row[6:18].view(3, 4).numpy()

        w2c_4x4 = np.eye(4, dtype=np.float32)
        w2c_4x4[:3, :4] = w2c_3x4

        c2w_4x4 = np.linalg.inv(w2c_4x4).astype(np.float32)
        c2w = torch.from_numpy(c2w_4x4)  # [4,4]

        return fxfycxcy_orig, c2w

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        video_id = os.path.splitext(os.path.basename(path))[0]

        # --------------------------------------
        # Load video frames
        # --------------------------------------
        frames = self._load_video_frames(path)
        T = len(frames)

        if T < self.num_views:
            print(path)
            raise ValueError(
                f"Video {path} has fewer than num_views frames: T={T}, requested num_views={self.num_views}."
            )

        # --------------------------------------
        # Load corresponding pose tensor .pt
        # --------------------------------------
        pose_file = os.path.join(self.pose_path, f"{video_id}.pt")
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not found: {pose_file}")

        poses = torch.load(pose_file)  # [num_frames, 18]
        if len(poses) < T:
            T = min(T, len(poses))
            frames = frames[:T]
            poses = poses[:T]

        if T < self.num_views:
            raise ValueError(
                f"After aligning frames/poses, video {path} has insufficient frames: "
                f"T={T}, requested num_views={self.num_views}."
            )

        # --------------------------------------
        # Pick V different frames (views)
        # --------------------------------------
        if self.overfit:
            # Deterministic evenly-spaced indices for overfit/debug.
            # Fail loudly if the clip is too short.
            V = self.num_views
            if V == 2:
                # Keep previous behavior where possible (legacy overfit indices).
                idx1, idx2 = 80, 160
                if idx2 >= T:
                    raise ValueError(
                        f"Overfit indices out of range for video {path}: T={T}, "
                        f"requested indices={[idx1, idx2]}."
                    )
                frame_indices = [idx1, idx2]
            else:
                # Evenly spaced on [0, T-1]
                frame_indices = [int(round(i * (T - 1) / (V - 1))) for i in range(V)]
        elif self.mode == 'test':
            V = self.num_views
            idx1 = 0
            if self.test_baseline is not None:
                baseline = int(self.test_baseline)
                if baseline < 1:
                    raise ValueError(f"test_baseline must be >= 1, got {baseline}")
                max_required = idx1 + (V - 1) * baseline
                if max_required >= T:
                    # Clamp baseline to max possible value
                    max_baseline = (T - 1 - idx1) // (V - 1)
                    if max_baseline < 1:
                        # Fallback to evenly spaced if clamping fails
                        frame_indices = [int(round(i * (T - 1) / (V - 1))) for i in range(V)]
                    else:
                        baseline = max_baseline
                        frame_indices = [idx1 + i * baseline for i in range(V)]
                else:
                    frame_indices = [idx1 + i * baseline for i in range(V)]
            else:
                # Evenly spaced deterministic indices on [0, T-1]
                frame_indices = [int(round(i * (T - 1) / (V - 1))) for i in range(V)]
        else:
            frame_indices = random.sample(range(T), self.num_views)

        # Sort by time for consistent "interpolate" / "prefix" semantics.
        order_by_time = sorted(range(len(frame_indices)), key=lambda k: int(frame_indices[k]))
        frame_indices = [frame_indices[k] for k in order_by_time]
        selected_frames = [frames[i] for i in frame_indices]
        selected_poses = poses[frame_indices]  # [V,18]

        # --------------------------------------
        # Reorder views so that "first cond_num views are reference".
        # This keeps downstream training/validation unchanged (it assumes ref = prefix).
        # --------------------------------------
        V = int(self.num_views)
        cond_num = int(self.cond_num)
        if cond_num >= V:
            raise ValueError(f"cond_num must be < num_views for multiview conditioning, got cond_num={cond_num}, num_views={V}")

        if self.ref_view_sampling == "prefix":
            # After sorting by time, prefix = earliest frames are references.
            order = list(range(V))
        elif self.ref_view_sampling == "interpolate":
            # Interpolation: only outermost views are references.
            if cond_num != 2:
                raise ValueError(f"ref_view_sampling='interpolate' requires cond_num==2, got cond_num={cond_num}")
            order = [0, V - 1] + list(range(1, V - 1))
        elif self.ref_view_sampling == "random":
            # Randomly choose reference views among the sampled V views.
            # In test mode, keep it deterministic per video for reproducibility.
            if cond_num < 1:
                raise ValueError(f"ref_view_sampling='random' requires cond_num>=1, got cond_num={cond_num}")
            if self.mode == "test":
                import hashlib
                h = hashlib.md5(video_id.encode("utf-8")).hexdigest()
                seed = int(h[:8], 16)
                g = torch.Generator(device="cpu").manual_seed(seed)
                perm = torch.randperm(V, generator=g).tolist()
                ref_pos = sorted(perm[:cond_num])
            else:
                ref_pos = sorted(random.sample(range(V), cond_num))
            tgt_pos = [i for i in range(V) if i not in set(ref_pos)]
            order = ref_pos + tgt_pos
        else:
            raise ValueError(f"Unhandled ref_view_sampling={self.ref_view_sampling}")

        frame_indices = [frame_indices[i] for i in order]
        selected_frames = [selected_frames[i] for i in order]
        selected_poses = selected_poses[order]
            

        # --------------------------------------
        # Intrinsics & c2w at ORIGINAL resolution (e.g. 256x256)
        # --------------------------------------
        fxfycxcy_orig_list = []
        c2w_list = []
        orig_sizes = []

        for frame, pose_row in zip(selected_frames, selected_poses):
            w_orig, h_orig = frame.size  # PIL: (width, height)
            orig_sizes.append((w_orig, h_orig))

            fxfycxcy_orig, c2w = self._pose_row_to_intrinsics_c2w_at_original(
                pose_row, width=w_orig, height=h_orig
            )
            fxfycxcy_orig_list.append(fxfycxcy_orig)
            c2w_list.append(c2w)

        fxfycxcy_orig = torch.stack(fxfycxcy_orig_list, dim=0)  # [V, 4]
        c2w = torch.stack(c2w_list, dim=0)                      # [V, 4, 4]

        # --------------------------------------
        # Preprocess using DA3 InputProcessor
        #   enc_inp : resized to self.process_res (e.g., 252x252)
        #   gt_inp  : kept at 256 with process_res=256, method="leave"
        # --------------------------------------
        batch_tensor, _, _ = self.processor(
            image=selected_frames,
            intrinsics=None,
            extrinsics=None,
            process_res=self.process_res,           # 252
            process_res_method=self.process_res_method,
            num_workers=1,
            sequential=True,
        )

        # For GT: use same resize method to get proper size (252x252), then denormalize
        batch_tensor_gt, _, _ = self.processor(
            image=selected_frames,
            intrinsics=None,
            extrinsics=None,
            process_res=self.process_res,  # Same as encoder resolution
            process_res_method=self.process_res_method,  # Same resize method
            num_workers=1,
            sequential=True,
        )
        # Denormalize from ImageNet norm to [0,1]
        # ImageNet mean/std used by InputProcessor
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        batch_tensor_gt = batch_tensor_gt * imagenet_std + imagenet_mean

        # enc_inp shape: (V, 3, H_enc, W_enc)
        # print("batch_tensor.shape:", batch_tensor.shape)
        _, _, H_enc, W_enc = batch_tensor.shape

        # --------------------------------------
        # Scale intrinsics from original -> encoder resolution (process_res)
        # --------------------------------------
        # The image is resized from original (e.g., 256) to process_res (e.g., 252).
        # Intrinsics must be scaled accordingly for ProPE to compute correct
        # camera projection matrices.
        
        fxfycxcy_scaled_list = []
        for (w_orig, h_orig), fxfycxcy in zip(orig_sizes, fxfycxcy_orig):
            sx = W_enc / float(w_orig)  # e.g., 252 / 256
            sy = H_enc / float(h_orig)

            fx, fy, cx, cy = fxfycxcy.tolist()
            fx_scaled = fx * sx
            fy_scaled = fy * sy
            cx_scaled = cx * sx
            cy_scaled = cy * sy

            fxfycxcy_scaled_list.append(
                torch.tensor([fx_scaled, fy_scaled, cx_scaled, cy_scaled], dtype=torch.float32)
            )

        fxfycxcy_scaled = torch.stack(fxfycxcy_scaled_list, dim=0)  # [V,4]

        return {
            # "enc_inp":      batch_tensor,       # (V, 3, H_enc, W_enc)
            "gt_inp":       batch_tensor_gt,   # (V, 3, H_enc, W_enc) - same as process_res
            "fxfycxcy": fxfycxcy_scaled,      # intrinsics scaled to process_res
            "c2w":          c2w,               # extrinsics (resolution-independent)
            "video_id":     video_id,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
        }


# Backward compatibility alias
DA3VideoDataset_Pose = RealEstate10KDataset


################################################################################
#                           Dummy Dataset for Testing                           #
################################################################################

class DummyMultiviewDataset(MultiviewDatasetBase, Dataset):
    """
    Dummy dataset for testing/debugging without real data.
    Generates random images with identity camera poses.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_views: int = 4,
        image_size: Union[int, Tuple[int, int], List[Tuple[int, int]]] = 256,
        cond_num: int = 1,
        **kwargs,  # Accept and ignore extra args for compatibility
    ):
        self.num_samples = num_samples
        self.num_views = num_views
        
        # Handle various input formats for image_size
        # 1. int -> [(size, size)]
        # 2. tuple -> [tuple]
        # 3. list of tuples -> list of tuples
        if isinstance(image_size, int):
            self.image_size_candidates = [(image_size, image_size)]
        elif isinstance(image_size, tuple):
            self.image_size_candidates = [image_size]
        elif isinstance(image_size, list):
            # Check if it's a list of ints [h, w] or list of tuples [(h,w), (h,w)]
            if len(image_size) == 2 and isinstance(image_size[0], int):
                # Assumed to be [H, W] single resolution
                self.image_size_candidates = [tuple(image_size)]
            elif len(image_size) > 0 and isinstance(image_size[0], (list, tuple)):
                # List of candidates
                self.image_size_candidates = [tuple(x) for x in image_size]
            else:
                self.image_size_candidates = [(256, 256)] # Fallback
        else:
            self.image_size_candidates = [(256, 256)]

        self.cond_num = cond_num
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        V = self.num_views
        
        # Randomly select resolution from candidates
        # Note: If batch_size > 1, this requires a custom collate_fn or lucky consistency.
        # User requested this feature for "dummy" testing, likely for dynamic resolution checks.
        H, W = random.choice(self.image_size_candidates)
        
        # Generate random RGB images in [0, 1]
        gt_inp = torch.rand(V, 3, H, W)
        
        # Default intrinsics (focal length = image size, principal point = center)
        fx = fy = float(H) # Just a dummy value
        cx, cy = float(W) / 2, float(H) / 2
        fxfycxcy = torch.tensor([[fx, fy, cx, cy]] * V, dtype=torch.float32)
        
        # Identity pose for all views
        c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(V, 1, 1)
        
        return {
            "gt_inp": gt_inp,
            "fxfycxcy": fxfycxcy,
            "c2w": c2w,
            "video_id": f"dummy_{idx}",
            "frame_indices": torch.arange(V, dtype=torch.long),
        }


################################################################################
#                           Dataset Registry & Factory                          #
################################################################################

# CUT3R datasets are loaded lazily to avoid import errors if CUT3R is not available
CUT3R_DATASETS = {
    "cut3r_dl3dv": "DL3DV_Multi",
    "cut3r_hypersim": "HyperSim_Multi",
    "cut3r_re10k": "RE10K_Multi",
    "cut3r_combined": "COMBINED",  # Special: uses all 3 datasets
}

DATASET_REGISTRY: Dict[str, type] = {
    "re10k": RealEstate10KDataset,
    "realestate10k": RealEstate10KDataset,
    "dummy": DummyMultiviewDataset,
}


def create_multiview_dataloader(
    dataset_name: str,
    video_path: Optional[Path] = None,
    pose_path: Optional[Path] = None,
    image_size: int = 256,
    num_views: int = 4,
    cond_num: int = 1,
    ref_view_sampling: str = "prefix",
    batch_size: int = 1,
    workers: int = 16,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
    mode: str = "train",
    overfit: bool = False,
    desired_steps: int = 1,
    test_baseline: Optional[int] = None,
    **dataset_kwargs,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Factory function to create multiview dataloaders.
    
    Args:
        dataset_name: "re10k", "dummy", or other registered datasets
        video_path: Path to video files (not needed for dummy)
        pose_path: Path to pose files (not needed for dummy)
        image_size: Target image resolution
        num_views: Number of views per sample
        cond_num: Number of conditioning views
        ref_view_sampling: "prefix", "random", or "interpolate"
        batch_size: Batch size per GPU
        workers: Number of dataloader workers
        rank: Current process rank
        world_size: Total number of processes
        shuffle: Whether to shuffle data
        mode: "train" or "test"
        overfit: If True, use single sample for overfitting
        desired_steps: Number of steps for overfit mode
        test_baseline: Frame offset for test mode
        **dataset_kwargs: Additional dataset-specific arguments
    
    Returns:
        (DataLoader, DistributedSampler) tuple
    """
    # Handle None or empty dataset_name -> use dummy
    if dataset_name is None or dataset_name.lower() == "dummy":
        dataset = DummyMultiviewDataset(
            num_samples=dataset_kwargs.get("num_samples", 100),
            num_views=num_views,
            image_size=image_size,
            cond_num=cond_num,
        )
    # Handle CUT3R datasets (lazy import)
    elif dataset_name.lower() in CUT3R_DATASETS:
        try:
            from cut3r_data import get_data_loader, DL3DV_Multi, HyperSim_Multi, RE10K_Multi, TartanAir_Multi
        except ImportError as e:
            raise ImportError(
                f"cut3r_data import failed. Ensure src/cut3r_data package exists. Error: {e}"
            )
        
        # Convert image_size to resolution list
        if isinstance(image_size, int):
            resolution = [(image_size, image_size)]
        elif isinstance(image_size, (list, tuple)) and len(image_size) == 2 and isinstance(image_size[0], int):
            resolution = [tuple(image_size)]
        elif isinstance(image_size, list) and all(isinstance(r, (list, tuple)) for r in image_size):
            resolution = [tuple(r) for r in image_size]
        else:
            resolution = image_size
        
        dataset_type = CUT3R_DATASETS[dataset_name.lower()]
        
        # Determine seed for test mode determinism
        # When seed is set, view selection becomes deterministic (reproducible)
        test_seed = 42 if mode == 'test' else None
        
        # Handle cut3r_combined: use all 3 datasets
        if dataset_type == "COMBINED":
            # Training paths (required for train mode)
            dl3dv_root = dataset_kwargs.get("dl3dv_root")
            hypersim_root = dataset_kwargs.get("hypersim_root")
            re10k_root = dataset_kwargs.get("re10k_root")
            tartan_root = dataset_kwargs.get("tartan_root")
            # Config compatibility: also check 'tartanair_root'
            if tartan_root is None:
                tartan_root = dataset_kwargs.get("tartanair_root")


            # Validation paths (required for test mode to prevent leakage)
            val_dl3dv_root = dataset_kwargs.get("val_dl3dv_root")
            val_hypersim_root = dataset_kwargs.get("val_hypersim_root")
            val_re10k_root = dataset_kwargs.get("val_re10k_root")
            val_tartan_root = dataset_kwargs.get("val_tartan_root", None)
            # Config compatibility: also check 'val_tartanair_root'
            if val_tartan_root is None:
                val_tartan_root = dataset_kwargs.get("val_tartanair_root", None)
            
            # Optional pose filtering (explicit; never silent)
            skip_bad_poses = dataset_kwargs.get("skip_bad_poses", False)
            max_translation_norm = dataset_kwargs.get("max_translation_norm", None)
            max_pose_retries = dataset_kwargs.get("max_pose_retries", None)
            log_bad_pose_every = dataset_kwargs.get("log_bad_pose_every", None)
            
            # LEAKAGE PROTECTION: In test mode, only use datasets with explicit validation paths
            if mode == 'test':
                # Track which datasets have validation paths
                use_dl3dv = val_dl3dv_root is not None
                use_hypersim = val_hypersim_root is not None
                use_re10k = val_re10k_root is not None
                use_tartan = val_tartan_root is not None
                
                if not any([use_dl3dv, use_hypersim, use_re10k]):
                    raise ValueError(
                        "[CUT3R Leakage Protection] mode='test' requires at least one val_*_root path. "
                        "Set val_dl3dv_root, val_hypersim_root, or val_re10k_root in config. "
                        "Datasets without validation paths are excluded to prevent train/val leakage."
                    )
                
                # Log which datasets are used for validation
                excluded = []
                if not use_dl3dv:
                    excluded.append("DL3DV")
                if not use_hypersim:
                    excluded.append("HyperSim")
                if not use_re10k:
                    excluded.append("RE10K")
                if not use_tartan:
                    excluded.append("Tartan")
                if excluded:
                    print(f"[CUT3R Leakage Protection] Excluding from validation (no val_*_root): {', '.join(excluded)}")
                
                # Use validation paths
                dl3dv_root = val_dl3dv_root
                hypersim_root = val_hypersim_root
                re10k_root = val_re10k_root
                tartan_root = val_tartan_root
                
                if use_dl3dv:
                    print(f"[CUT3R] Using validation path for DL3DV: {dl3dv_root}")
                if use_hypersim:
                    print(f"[CUT3R] Using validation path for HyperSim: {hypersim_root}")
                if use_re10k:
                    print(f"[CUT3R] Using validation path for RE10K: {re10k_root}")
                if use_tartan:
                    print(f"[CUT3R] Using validation path for Tartan: {tartan_root}")
            else:
                # Train mode: require all training paths
                use_dl3dv = dl3dv_root is not None
                use_hypersim = hypersim_root is not None
                use_re10k = re10k_root is not None
                use_tartan = tartan_root is not None
                
                # if not all([use_dl3dv, use_hypersim, use_re10k]):
                #     raise ValueError(
                #         "cut3r_combined requires: dl3dv_root, hypersim_root, re10k_root for training"
                #     )

            # Create individual datasets only for those with valid paths
            datasets_to_combine = []
            mix_ratio = dataset_kwargs.get("mix_ratio", {})
            equal_ratio = dataset_kwargs.get("equal_ratio", False)
            
            if use_dl3dv and dl3dv_root:
                dl3dv = DL3DV_Multi(
                    ROOT=dl3dv_root,
                    resolution=resolution,
                    num_views=num_views,
                    split='train' if mode == 'train' else None,
                    allow_repeat=False,
                    seed=test_seed,
                    skip_bad_poses=skip_bad_poses,
                    max_translation_norm=max_translation_norm,
                    max_pose_retries=max_pose_retries,
                    log_bad_pose_every=log_bad_pose_every,
                )
                dl3dv_ratio = mix_ratio.get("dl3dv", 84000) if not equal_ratio else None
                print(f"[CUT3R] Loaded DL3DV: {len(dl3dv)} samples (Ratio: {dl3dv_ratio})")
                datasets_to_combine.append(("dl3dv", dl3dv, dl3dv_ratio))
            
            if use_hypersim and hypersim_root:
                hypersim = HyperSim_Multi(
                    ROOT=hypersim_root,
                    resolution=resolution,
                    num_views=num_views,
                    split='train' if mode == 'train' else None,
                    allow_repeat=False,
                    seed=test_seed,
                    skip_bad_poses=skip_bad_poses,
                    max_translation_norm=max_translation_norm,
                    max_pose_retries=max_pose_retries,
                    log_bad_pose_every=log_bad_pose_every,
                )
                hypersim_ratio = mix_ratio.get("hypersim", 5600) if not equal_ratio else None
                print(f"[CUT3R] Loaded HyperSim: {len(hypersim)} samples (Ratio: {hypersim_ratio})")
                datasets_to_combine.append(("hypersim", hypersim, hypersim_ratio))

            if use_re10k and re10k_root:
                re10k = RE10K_Multi(
                    ROOT=re10k_root,
                    resolution=resolution,
                    num_views=num_views,
                    split=None,  # RE10K doesn't use split
                    allow_repeat=False,
                    seed=test_seed,
                    skip_bad_poses=skip_bad_poses,
                    max_translation_norm=max_translation_norm,
                    max_pose_retries=max_pose_retries,
                    log_bad_pose_every=log_bad_pose_every,
                )
                re10k_ratio = mix_ratio.get("re10k", 9600) if not equal_ratio else None
                print(f"[CUT3R] Loaded RE10K: {len(re10k)} samples (Ratio: {re10k_ratio})")
                datasets_to_combine.append(("re10k", re10k, re10k_ratio))
            if use_tartan and tartan_root:
                tartan = TartanAir_Multi(
                    ROOT=tartan_root,
                    resolution=resolution,
                    num_views=num_views,
                    split=None,  # RE10K doesn't use split
                    allow_repeat=False,
                    seed=test_seed,
                    skip_bad_poses=skip_bad_poses,
                    max_translation_norm=max_translation_norm,
                    max_pose_retries=max_pose_retries,
                    log_bad_pose_every=log_bad_pose_every,
                )
                tartan_ratio = mix_ratio.get("tartan")
                # Config compatibility: also check 'tartanair'
                if tartan_ratio is None:
                     tartan_ratio = mix_ratio.get("tartanair")
                
                # Default fallback
                if tartan_ratio is None:
                     tartan_ratio = 9600
                elif not equal_ratio:
                     # Already got ratio
                     pass
                else: 
                     tartan_ratio = None # equal_ratio handles it
                
                print(f"[CUT3R] Loaded TartanAir: {len(tartan)} samples (Ratio: {tartan_ratio})")
                datasets_to_combine.append(("tartan", tartan, tartan_ratio))
            # Combine datasets
            if len(datasets_to_combine) == 0:
                print(f"[CUT3R] ERROR: No datasets selected! Check paths and active flags.")
                print(f"  use_dl3dv={use_dl3dv}, root={dl3dv_root}")
                print(f"  use_hypersim={use_hypersim}, root={hypersim_root}")
                print(f"  use_re10k={use_re10k}, root={re10k_root}")
                print(f"  use_tartan={use_tartan}, root={tartan_root}")
                raise ValueError("No datasets available to combine!")
            
            if equal_ratio:
                # 1:1:1 ratio - use the largest dataset size for all
                max_len = max(len(ds) for _, ds, _ in datasets_to_combine)
                dataset = None
                for name, ds, _ in datasets_to_combine:
                    if dataset is None:
                        dataset = max_len @ ds
                    else:
                        dataset = dataset + max_len @ ds
            else:
                # Use specified mix ratios
                dataset = None
                for name, ds, ratio in datasets_to_combine:
                    if dataset is None:
                        dataset = ratio @ ds
                    else:
                        dataset = dataset + ratio @ ds
        else:
            # Single CUT3R dataset
            dataset_cls = {"TartanAir_Multi":TartanAir_Multi,"DL3DV_Multi": DL3DV_Multi, "HyperSim_Multi": HyperSim_Multi, "RE10K_Multi": RE10K_Multi}[dataset_type]
            cut3r_root = dataset_kwargs.get("root") or dataset_kwargs.get("ROOT")
            if cut3r_root is None:
                raise ValueError(f"CUT3R dataset '{dataset_name}' requires 'root' in dataset_kwargs.")
            
            dataset = dataset_cls(
                ROOT=cut3r_root,
                resolution=resolution,
                num_views=num_views,
                split='train' if mode == 'train' else None,
                allow_repeat=False,
                seed=test_seed,
                skip_bad_poses=dataset_kwargs.get("skip_bad_poses", False),
                max_translation_norm=dataset_kwargs.get("max_translation_norm", None),
                max_pose_retries=dataset_kwargs.get("max_pose_retries", None),
                log_bad_pose_every=dataset_kwargs.get("log_bad_pose_every", None),
            )
        
        # Use CUT3R's BatchedRandomSampler for multi-resolution support
        #
        # IMPORTANT (Fail Loud / Reproducibility):
        # - In CUT3R train mode, the default sampler varies the number of views per batch (e.g., 4~num_views),
        #   which can conflict with our training code that assumes a fixed total_view.
        # - We therefore force fixed_length=True so every batch has exactly `num_views` views.
        loader = get_data_loader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=shuffle,
            drop_last=True,
            world_size=world_size,
            rank=rank,
            fixed_length=True,
        )

        loader.dataset.set_epoch(0)
        if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
            if hasattr(loader.batch_sampler, 'set_epoch'):
                loader.batch_sampler.set_epoch(0)  # Required for multi-resolution sampling
        
        # Debug: Print multi-resolution info
        print(f"[CUT3R DataLoader] mode={mode}, batch_size={batch_size}, num_workers={workers}")
        print(f"[CUT3R DataLoader] Resolutions: {dataset._resolutions} (count: {len(dataset._resolutions)})")
        print(f"[CUT3R DataLoader] Dataset length: {len(dataset)}, num_views: {dataset.num_views}")
        if hasattr(loader, 'batch_sampler') and loader.batch_sampler is not None:
            bs = loader.batch_sampler
            if hasattr(bs, 'sampler') and hasattr(bs.sampler, 'pool_size'):
                sampler = bs.sampler
                print(f"[CUT3R DataLoader] Sampler pool_size={sampler.pool_size}, epoch={sampler.epoch}")
            else:
                print(f"[CUT3R DataLoader] batch_sampler type: {type(bs).__name__}")
        else:
            print(f"[CUT3R DataLoader] WARNING: No custom batch_sampler - multi-resolution may not work!")
        
        # Return loader with its batch_sampler (not DistributedSampler)
        return loader, loader.batch_sampler
    else:
        dataset_cls = DATASET_REGISTRY.get(dataset_name.lower())
        if dataset_cls is None:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_REGISTRY.keys()) + list(CUT3R_DATASETS.keys())}"
            )
        
        if dataset_cls == DummyMultiviewDataset:
            dataset = DummyMultiviewDataset(
                num_samples=dataset_kwargs.get("num_samples", 100),
                num_views=num_views,
                image_size=image_size,
                cond_num=cond_num,
            )
        else:
            # Real dataset requires paths
            if video_path is None or pose_path is None:
                raise ValueError(
                    f"Dataset '{dataset_name}' requires video_path and pose_path."
                )
            
            # Remove keys that are already passed as explicit arguments or not needed
            for key in ["name", "video_path", "pose_path", "num_views", "cond_num", "ref_view_sampling", "process_res", "mode", "overfit", "test_baseline"]:
                if key in dataset_kwargs:
                    del dataset_kwargs[key]
            
            dataset = dataset_cls(
                video_path=video_path,
                pose_path=pose_path,
                num_views=num_views,
                cond_num=cond_num,
                ref_view_sampling=ref_view_sampling,
                process_res=image_size,
                mode=mode,
                overfit=overfit,
                test_baseline=test_baseline,
                **dataset_kwargs,
            )
    
    
    # Check dataset size (Fail Loud)
    if hasattr(dataset, "__len__"):
        ds_len = len(dataset)
        print(f"[INFO] Successfully loaded dataset '{dataset_name}' with {ds_len} samples.")
        if ds_len == 0:
            raise ValueError(f"Dataset '{dataset_name}' is empty! Check dataset paths, num_views, or image availability.")
    
    # Handle overfit mode
    if overfit:
        steps = desired_steps if desired_steps is not None else 10
        total_len = steps * batch_size * world_size
        overfit_index = 0
        dataset = Subset(dataset, [overfit_index] * total_len)
    
    # Create distributed sampler and dataloader
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=shuffle
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader, sampler
