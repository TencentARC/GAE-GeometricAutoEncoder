#!/usr/bin/env python3
"""
Evaluate **3D geometric consistency** of generated novel views.

Follows the protocol of Geometric Latent Diffusion (GLD) paper, Sec. 5.1.4:
   "To assess the 3D geometric consistency of generated views, we further
    incorporate camera estimation errors, reprojection error, and MEt3R.
    Specifically, for camera errors, we extract camera poses from the
    generated views using an external estimator (VGGT) to compute
    Absolute Trajectory Error (ATE), and Relative Pose Errors for rotation
    (RPEr) and translation (RPEt). ... Reprojection error measures the
    spatial re-alignment accuracy of reconstructed 3D points, while MEt3R
    evaluates multi-view consistency using projected feature similarity."

This script is a *backend-agnostic* evaluator: it reads the **already
generated** novel views produced by the diffusion eval script
(`scripts/eval/eval_generation.py`) and computes only the 3D-consistency metrics. The
generation step is *not* repeated, so a single VGGT pass is amortized over
all baselines.

──────────────────────────────────────────────────────────────────────
Inputs
──────────────────────────────────────────────────────────────────────
Each diffusion eval script writes per-scene predictions as:
    <output_dir>/<dataset>/<sceneIdx>_pred.mp4   (V frames, generated)
    <output_dir>/<dataset>/<sceneIdx>_gt.mp4     (V frames, ground truth)

This script pairs each `_pred.mp4` with the GT camera pose loaded from the
**original dataset** (re-using `load_image_and_camera` from
`eval_data`), then runs VGGT once on the predicted frames to
estimate (pose, depth, world points) and computes:

    1. ATE / RPEr / RPEt  — VGGT-predicted pose vs GT pose (evo), reported both
                            Sim3 (scale-aligned) and SE3 (no scale) aligned
    2. Reproj_100/80/50   — 2D cycle reprojection of VGGT pointmaps (norm. by
                            image diagonal), keeping top 100/80/50% conf pixels
    3. MEt3R              — official MEt3R (MASt3R + FeatUp + PyTorch3D)

──────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────
    cd /path/to/GLD

    # Evaluate generated novel views
    PYTHONPATH=src uv run python scripts/eval/eval_3d_consistency.py \\
        --pred-dir results/eval_generation \\
        --scene-manifest configs/eval/re10k_v3_16scenes_seed42.json \\
        --num-views 9 --cond-num 1

    # Evaluate another run
    PYTHONPATH=src uv run python scripts/eval/eval_3d_consistency.py --pred-dir results/eval_generation --scene-manifest configs/eval/re10k_v3_16scenes_seed42.json --num-views 9 --cond-num 1

    # Evaluate a baseline run
    PYTHONPATH=src uv run python scripts/eval/eval_3d_consistency.py --pred-dir results/eval_generation --scene-manifest configs/eval/re10k_v3_16scenes_seed42.json --num-views 9 --cond-num 1

    # Sanity check: evaluate the GT mp4s themselves (should give near-zero errors)
    PYTHONPATH=src uv run python scripts/eval/eval_3d_consistency.py \\
        --pred-dir results/eval_generation --pred-source gt \\
        --scene-manifest configs/eval/re10k_v3_16scenes_seed42.json \\
        --num-views 9
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import sys

os.environ.setdefault("TMPDIR", "/tmp")

# Reuse local-ssd HF cache pattern from sibling eval scripts.
_LOCAL_HF_CACHE = "/local-ssd/hf_cache"
if os.path.isdir("/local-ssd"):
    os.makedirs(_LOCAL_HF_CACHE, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/local-ssd/hf_home")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _LOCAL_HF_CACHE)
else:
    os.environ.setdefault("HF_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/xdg-cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# FeatUp is loaded via torch.hub inside official MEt3R; keep cache on writable storage.
if os.path.isdir("/local-ssd"):
    os.environ.setdefault("TORCH_HOME", "/local-ssd/torch_home")
else:
    os.environ.setdefault("TORCH_HOME", "/tmp/torch_home")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(ROOT, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC_ROOT)


@contextlib.contextmanager
def _met3r_import_context():
    """Keep GLD ``src/utils`` from shadowing DINO/croco ``utils`` during MEt3R load."""
    saved_path = sys.path.copy()
    saved_utils = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "utils" or name.startswith("utils.")
    }
    sys.path = [
        p for p in sys.path
        if os.path.normpath(p) != os.path.normpath(SRC_ROOT)
    ]
    try:
        yield
    finally:
        sys.path = saved_path
        sys.modules.update(saved_utils)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eval_data import (
    DATASET_ROOTS,
    collect_scenes,
    collect_scenes_re10k,
    load_scenes_from_manifest,
    load_image_and_camera,
)


# ── Pred frame loader ───────────────────────────────────────────────────────

def load_video_frames(mp4_path: str) -> np.ndarray:
    """Load all frames from mp4 as (V, H, W, 3) uint8 RGB."""
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"Predicted video not found: {mp4_path}")
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path}")
    return np.stack(frames, axis=0)


def find_pred_mp4(pred_dir: str, dataset: str, s_idx: int,
                  pred_source: str = "pred") -> str:
    """Locate the predicted-frames mp4 for scene index `s_idx`.

    Tries the canonical layout used by the three diffusion eval scripts:
        <pred_dir>/<dataset>/<sIdx:03d>_<source>.mp4
        <pred_dir>/<sIdx:03d>_<source>.mp4   (when manifest dataset='custom')
    """
    candidates = [
        os.path.join(pred_dir, dataset, f"{s_idx:03d}_{pred_source}.mp4"),
        os.path.join(pred_dir, f"{s_idx:03d}_{pred_source}.mp4"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"No {pred_source}.mp4 for scene {s_idx} under {pred_dir} "
        f"(tried {candidates})"
    )


def collect_video_pairs(pred_dir: str, dataset: str,
                        pred_source: str = "pred",
                        gt_tag: str = "gt") -> list:
    """Discover paired ``<idx>_<pred_source>.mp4`` / ``<idx>_<gt_tag>.mp4`` files.

    This bypasses any dataset re-sampling: both the generated video and its
    ground-truth video already live side-by-side in the eval output directory
    and are frame-aligned by construction. Returns a sorted list of
    ``(name, gt_mp4_path, pred_mp4_path)``.
    """
    import glob
    suffix = f"_{pred_source}.mp4"
    for d in (os.path.join(pred_dir, dataset), pred_dir):
        preds = sorted(glob.glob(os.path.join(d, f"*{suffix}")))
        pairs = []
        for p in preds:
            idx = os.path.basename(p)[:-len(suffix)]
            gt = os.path.join(d, f"{idx}_{gt_tag}.mp4")
            if os.path.isfile(gt):
                pairs.append((idx, gt, p))
        if pairs:
            return pairs
    return []


# ── GT loader (reuses eval_data preprocessing) ──────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _denorm_imagenet_01(t: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(t.device, t.dtype)
    std = _IMAGENET_STD.to(t.device, t.dtype)
    return (t * std + mean).clamp(0, 1)


def load_gt_for_scene(scene_dir: str, img_names: list, resolution: tuple,
                      ds_type: str, device: torch.device):
    """Return (gt_imgs_01[V,3,H,W], gt_intrinsics[V,3,3], gt_c2w[V,4,4])."""
    img_tensors, intri_list, pose_list = [], [], []
    basenames = [os.path.splitext(n)[0] for n in img_names]
    for bn in basenames:
        img_t, intr, pose = load_image_and_camera(
            scene_dir, bn, resolution, ds_type=ds_type)
        img_tensors.append(img_t)
        intri_list.append(intr)
        pose_list.append(pose)
    imgs_inet = torch.stack(img_tensors).to(device)
    imgs_01 = _denorm_imagenet_01(imgs_inet)
    intri = torch.from_numpy(np.stack(intri_list)).float().to(device)  # (V, 3, 3)
    c2w = torch.from_numpy(np.stack(pose_list)).float().to(device)     # (V, 4, 4)
    return imgs_01, intri, c2w


# ── VGGT estimator wrapper ──────────────────────────────────────────────────

def _load_vggt(pretrained_path: str, device: torch.device):
    """Load VGGT-1B for pose+depth+point estimation."""
    from vggt.models.vggt import VGGT
    print(f"  [VGGT] loading {pretrained_path} ...", flush=True)
    model = VGGT.from_pretrained(pretrained_path).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [VGGT] loaded — {n_params:.1f}M params")
    return model


def _resize_for_vggt(images_01: torch.Tensor, target_long: int = 518) -> torch.Tensor:
    """VGGT default img_size is 518; rescale (keep aspect, multiples of 14)."""
    V, C, H, W = images_01.shape
    scale = float(target_long) / max(H, W)
    new_h = int(round(H * scale / 14) * 14)
    new_w = int(round(W * scale / 14) * 14)
    if (new_h, new_w) == (H, W):
        return images_01
    return F.interpolate(images_01, size=(new_h, new_w),
                         mode="bilinear", align_corners=False)


@torch.no_grad()
def vggt_infer(model, images_01: torch.Tensor, device: torch.device,
               amp_dtype=torch.bfloat16) -> dict:
    """Run VGGT on (V, 3, H, W) in [0,1]. Returns dict on CPU as numpy arrays.

    Keys:
        c2w:          (V, 4, 4)  recovered camera-to-world (OpenCV convention)
        K:            (V, 3, 3)  recovered intrinsics (pixels at VGGT input res)
        depth:        (V, H_v, W_v)
        depth_conf:   (V, H_v, W_v)
        world_points: (V, H_v, W_v, 3)
        wp_conf:      (V, H_v, W_v)
        input_hw:     (H_v, W_v)
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    imgs_resized = _resize_for_vggt(images_01.to(device), target_long=518)
    _, _, Hv, Wv = imgs_resized.shape
    imgs_in = imgs_resized.unsqueeze(0)  # (1, V, 3, H, W)

    with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
        preds = model(imgs_in)

    pose_enc = preds["pose_enc"].float()                # (1, V, 9)
    extr_w2c, K = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(Hv, Wv))               # (1, V, 3, 4), (1, V, 3, 3)
    extr_w2c = extr_w2c[0].cpu().numpy()                # (V, 3, 4)
    K_np = K[0].cpu().numpy()                            # (V, 3, 3)

    V = extr_w2c.shape[0]
    c2w_list = []
    for v in range(V):
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :4] = extr_w2c[v]
        c2w_list.append(np.linalg.inv(T_w2c))
    c2w = np.stack(c2w_list, axis=0).astype(np.float32)

    depth = preds["depth"][0, ..., 0].float().cpu().numpy()      # (V, H, W)
    depth_conf = preds["depth_conf"][0].float().cpu().numpy()
    world_points = preds["world_points"][0].float().cpu().numpy()  # (V, H, W, 3)
    wp_conf = preds["world_points_conf"][0].float().cpu().numpy()

    return {
        "c2w": c2w,
        "K": K_np,
        "depth": depth,
        "depth_conf": depth_conf,
        "world_points": world_points,
        "wp_conf": wp_conf,
        "input_hw": (Hv, Wv),
    }


# ── DA3 estimator wrapper ───────────────────────────────────────────────────

def _stage_da3_to_local(src_dir: str, cache_root: str | None = None) -> str:
    """Copy a DA3 checkpoint dir to fast local disk and return the local dir.

    DA3-GIANT is ~5.4 GB; reading it from a shared/network filesystem (e.g.
    ``/threed-code``) is very slow. Staging to node-local NVMe (``/local-ssd``
    by default, override via ``DA3_LOCAL_CACHE``) makes subsequent loads fast.

    Skips copying when the source already lives under the cache root, when a
    complete size-matching copy already exists, or when staging is disabled
    (``DA3_LOCAL_CACHE=""``). Falls back to the source dir on any failure.
    """
    if cache_root is None:
        cache_root = os.environ.get("DA3_LOCAL_CACHE", "/local-ssd")
    src_dir = os.path.abspath(os.path.expanduser(src_dir))
    if not cache_root or not os.path.isdir(src_dir):
        return src_dir
    try:
        cr = os.path.abspath(os.path.expanduser(cache_root))
        if src_dir == cr or src_dir.startswith(cr + os.sep):
            return src_dir  # already on local disk
        os.makedirs(cr, exist_ok=True)
        dst_dir = os.path.join(cr, os.path.basename(src_dir))
        os.makedirs(dst_dir, exist_ok=True)
        for fname in sorted(os.listdir(src_dir)):
            sp = os.path.join(src_dir, fname)
            if not os.path.isfile(sp):
                continue
            dp = os.path.join(dst_dir, fname)
            ssize = os.path.getsize(sp)
            if os.path.isfile(dp) and os.path.getsize(dp) == ssize:
                continue  # already staged
            tmp = f"{dp}.partial.{os.getpid()}"
            print(f"  [DA3] staging {fname} ({ssize / 1e9:.2f} GB) -> {dst_dir} ...",
                  flush=True)
            shutil.copy2(sp, tmp)
            os.replace(tmp, dp)
        print(f"  [DA3] using local copy: {dst_dir}", flush=True)
        return dst_dir
    except Exception as e:  # noqa: BLE001 - staging is best-effort
        print(f"  [DA3] local staging failed ({e}); loading from {src_dir}",
              flush=True)
        return src_dir


def _load_da3(pretrained_path: str, device: torch.device):
    """Load Depth Anything 3 for pose+depth estimation."""
    from depth_anything_3.api import DepthAnything3
    model_path = os.path.abspath(os.path.expanduser(pretrained_path))
    if not os.path.exists(model_path):
        model_path = pretrained_path
    if os.path.isdir(model_path):
        model_path = _stage_da3_to_local(model_path)
    local_only = os.path.isdir(model_path)
    print(f"  [DA3] loading {model_path} (local_only={local_only}) ...", flush=True)
    model = DepthAnything3.from_pretrained(
        model_path,
        local_files_only=local_only,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [DA3] loaded — {n_params:.1f}M params")
    return model


def _as_4x4_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """Convert DA3 extrinsics to homogeneous (V, 4, 4)."""
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3:
        raise ValueError(f"Expected extrinsics (V,3/4,4), got {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] == (3, 4):
        V = extrinsics.shape[0]
        out = np.broadcast_to(np.eye(4, dtype=np.float32), (V, 4, 4)).copy()
        out[:, :3, :4] = extrinsics
        return out
    raise ValueError(f"Expected extrinsics (V,3/4,4), got {extrinsics.shape}")


def _depth_to_world_points(
    depth: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Back-project dense depth maps to world points using OpenCV pinhole cameras."""
    depth_t = torch.from_numpy(np.asarray(depth)).float()
    c2w_t = torch.from_numpy(_as_4x4_extrinsics(c2w)).float()
    K_t = torch.from_numpy(np.asarray(K)).float()
    V, H, W = depth_t.shape
    device = depth_t.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=depth_t.dtype),
        torch.arange(W, device=device, dtype=depth_t.dtype),
        indexing="ij",
    )
    z = depth_t.clamp(min=1e-6)
    fx = K_t[:, 0, 0].view(V, 1, 1).clamp(min=1e-6)
    fy = K_t[:, 1, 1].view(V, 1, 1).clamp(min=1e-6)
    cx = K_t[:, 0, 2].view(V, 1, 1)
    cy = K_t[:, 1, 2].view(V, 1, 1)

    x_cam = (x.view(1, H, W) - cx) / fx * z
    y_cam = (y.view(1, H, W) - cy) / fy * z
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (V,H,W,3)
    R = c2w_t[:, :3, :3]
    t = c2w_t[:, :3, 3]
    pts_world = torch.einsum("vhwc,vdc->vhwd", pts_cam, R) + t[:, None, None, :]
    return pts_world.numpy().astype(np.float32)


@torch.no_grad()
def da3_infer(
    model,
    frames_np: np.ndarray,
    *,
    gt_c2w: np.ndarray | None = None,
    gt_K: np.ndarray | None = None,
    process_res: int = 504,
    use_gt_cameras: bool = False,
    use_ray_pose: bool = False,
) -> dict:
    """Run DA3 on RGB frames and return a VGGT-like geometry dict.

    DA3 ``Prediction.extrinsics`` uses the OpenCV **world-to-camera** (w2c)
    convention, so we invert it here to obtain the camera-to-world (c2w) matrices
    expected by the downstream metrics (verified empirically on GT frames: the
    inverted poses recover the GT trajectory to ATE < 2mm, whereas using the raw
    extrinsics as c2w gives large errors). When ``use_gt_cameras`` is enabled,
    the GT c2w cameras are converted to w2c before being handed to DA3.
    """
    if use_gt_cameras and gt_c2w is not None:
        extrinsics = np.linalg.inv(_as_4x4_extrinsics(gt_c2w))[:, :3, :].astype(np.float32)
    else:
        extrinsics = None
    intrinsics = gt_K if use_gt_cameras else None
    pred = model.inference(
        [im for im in frames_np],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        infer_gs=False,
        use_ray_pose=use_ray_pose,
        process_res=process_res,
        export_format="mini_npz",
    )
    if pred.depth is None or pred.extrinsics is None or pred.intrinsics is None:
        raise RuntimeError("DA3 did not return depth/extrinsics/intrinsics")

    # DA3 returns w2c extrinsics; invert to c2w for the metric pipeline.
    w2c = _as_4x4_extrinsics(pred.extrinsics).astype(np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    K = np.asarray(pred.intrinsics, dtype=np.float32)
    depth = np.asarray(pred.depth, dtype=np.float32)
    conf = pred.conf
    if conf is None:
        conf = np.ones_like(depth, dtype=np.float32)
    else:
        conf = np.asarray(conf, dtype=np.float32)

    world_points = _depth_to_world_points(depth, c2w, K)
    return {
        "c2w": c2w,
        "K": K,
        "depth": depth,
        "depth_conf": conf,
        "world_points": world_points,
        "wp_conf": conf,
        "input_hw": tuple(depth.shape[-2:]),
    }


# ── Metric 1: ATE / RPEr / RPEt ─────────────────────────────────────────────

def _c2w_list_to_tum(c2w_np: np.ndarray):
    """Convert (V, 4, 4) c2w to TUM (V, 8) [t, tx, ty, tz, qx, qy, qz, qw]."""
    from utils.evo_utils import get_tum_poses
    return get_tum_poses([c2w_np[v] for v in range(c2w_np.shape[0])])


def compute_camera_metrics(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict:
    """ATE/RPEt/RPEr via evo, both Sim3 (with scale) and SE3 (no scale) aligned.

    Returns dict with Sim3 keys ``ate``/``rpe_trans``/``rpe_rot`` and SE3 keys
    ``ate_noscale``/``rpe_trans_noscale``/``rpe_rot_noscale``. ATE/RPEt in the
    GT-video estimator scale, RPEr in degrees. All RMSE. Returns NaN on failure
    (e.g. evo not installed, or degenerate trajectory).
    """
    nan_out = {
        "ate": float("nan"), "rpe_trans": float("nan"), "rpe_rot": float("nan"),
        "ate_noscale": float("nan"), "rpe_trans_noscale": float("nan"),
        "rpe_rot_noscale": float("nan"),
    }
    try:
        from utils.evo_utils import eval_metrics
    except ImportError as e:
        print(f"  [WARN] evo not available, skipping ATE/RPE: {e}")
        return dict(nan_out)

    try:
        pred_tum = _c2w_list_to_tum(pred_c2w)
        gt_tum = _c2w_list_to_tum(gt_c2w)
        # Sim3 (with scale) and SE3 (no scale) alignments side by side.
        ate, rpe_t, rpe_r = eval_metrics(
            pred_tum, gt_tum, seq="3d_consistency", correct_scale=True)
        ate_ns, rpe_t_ns, rpe_r_ns = eval_metrics(
            pred_tum, gt_tum, seq="3d_consistency", correct_scale=False)
        return {
            "ate": float(ate), "rpe_trans": float(rpe_t), "rpe_rot": float(rpe_r),
            "ate_noscale": float(ate_ns), "rpe_trans_noscale": float(rpe_t_ns),
            "rpe_rot_noscale": float(rpe_r_ns),
        }
    except Exception as e:
        print(f"  [WARN] ATE/RPE failed: {e}")
        return dict(nan_out)


# ── Metric 2: Reprojection error ────────────────────────────────────────────

# Reproj is reported at several confidence-keep fractions: 1.0 = all pixels,
# 0.8 = top-80% confident pixels, 0.5 = top-50%. Column key = reproj_<pct>.
REPROJ_KEEP_FRACTIONS = (1.0, 0.8, 0.5)


def _reproj_key(frac: float) -> str:
    return f"reproj_{int(round(frac * 100))}"


def _build_pixel_grid(H: int, W: int, device, dtype) -> torch.Tensor:
    """Return (H, W, 2) [u, v] pixel coords."""
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([u, v], dim=-1)


def _scale_intrinsics(K: torch.Tensor, src_hw: tuple, dst_hw: tuple) -> torch.Tensor:
    """Scale (V, 3, 3) intrinsics from src resolution to dst resolution."""
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    K_scaled = K.clone()
    K_scaled[:, 0, 0] *= sx
    K_scaled[:, 1, 1] *= sy
    K_scaled[:, 0, 2] *= sx
    K_scaled[:, 1, 2] *= sy
    return K_scaled


def _sim3_align_world_points_to_gt(
    pred_c2w_np: np.ndarray,
    gt_c2w_np: np.ndarray,
    world_points_np: np.ndarray,
) -> np.ndarray:
    """Sim3 (Umeyama) align VGGT world points into the GT camera frame."""
    from evo.core.trajectory import PoseTrajectory3D
    from scipy.spatial.transform import Rotation

    V = pred_c2w_np.shape[0]
    quats_wxyz = []
    for v in range(V):
        q = Rotation.from_matrix(pred_c2w_np[v, :3, :3]).as_quat()  # xyzw
        quats_wxyz.append([q[3], q[0], q[1], q[2]])
    traj_pred = PoseTrajectory3D(
        positions_xyz=pred_c2w_np[:, :3, 3].astype(np.float64),
        orientations_quat_wxyz=np.asarray(quats_wxyz, dtype=np.float64),
        timestamps=np.arange(V, dtype=np.float64),
    )
    quats_gt = []
    for v in range(V):
        q = Rotation.from_matrix(gt_c2w_np[v, :3, :3]).as_quat()
        quats_gt.append([q[3], q[0], q[1], q[2]])
    traj_gt = PoseTrajectory3D(
        positions_xyz=gt_c2w_np[:, :3, 3].astype(np.float64),
        orientations_quat_wxyz=np.asarray(quats_gt, dtype=np.float64),
        timestamps=np.arange(V, dtype=np.float64),
    )
    r_a, t_a, s = traj_pred.align(traj_gt, correct_scale=True)
    pts = world_points_np.reshape(-1, 3).astype(np.float64)
    aligned = s * (pts @ r_a.T) + t_a
    return aligned.reshape(world_points_np.shape).astype(np.float32)


def compute_reprojection_error(
    world_points: torch.Tensor,   # (V, H, W, 3) VGGT world points (VGGT world frame)
    vggt_c2w: torch.Tensor,       # (V, 4, 4) VGGT predicted camera-to-world
    vggt_K: torch.Tensor,         # (V, 3, 3) VGGT intrinsics at world_points resolution
    conf: torch.Tensor,           # (V, H, W) — VGGT world-point confidence
    conf_thresh_quantile: float = 0.0,
    max_points_per_view: int | None = None,
) -> float:
    """2D cycle reprojection error, normalized by image diagonal.

    ``conf_thresh_quantile=0`` keeps all pixels (no confidence filtering);
    ``max_points_per_view=None/0`` disables subsampling (use every pixel).

    For each directed pair (i -> j), project high-confidence pointmap pixels
    from view i into view j, sample view-j's pointmap at that projected location,
    then project the sampled 3D point back into view i. The metric is the
    original-vs-cycle pixel distance divided by sqrt(H^2 + W^2).

    Lower means different views agree on the same 3D surface under VGGT's
    estimated cameras and pointmaps.
    """
    V, H, W, _ = world_points.shape
    device = world_points.device
    dtype = world_points.dtype
    diag = math.sqrt(float(H * H + W * W))

    # VGGT world-to-camera
    w2c = torch.linalg.inv(vggt_c2w)  # (V, 4, 4)
    R = w2c[:, :3, :3]   # (V, 3, 3)
    t = w2c[:, :3, 3]    # (V, 3)

    # Confidence mask: keep top (1 - quantile) fraction per view
    flat_conf = conf.reshape(V, -1)
    thresh = torch.quantile(flat_conf, conf_thresh_quantile, dim=1, keepdim=True)
    mask = conf >= thresh.reshape(V, 1, 1)

    wp_chw = world_points.permute(0, 3, 1, 2).contiguous()  # (V, 3, H, W)
    pix = _build_pixel_grid(H, W, device, dtype).reshape(-1, 2)

    def _project_to_pixels(X: torch.Tensor, view_idx: int):
        """Project (N, 3) world points to pixel coords in view_idx."""
        cam = X @ R[view_idx].T + t[view_idx]   # (N, 3) in camera frame
        z = cam[:, 2]
        uv = cam[:, :2] / z.unsqueeze(-1).clamp(min=1e-6)
        u = uv[:, 0] * vggt_K[view_idx, 0, 0] + vggt_K[view_idx, 0, 2]
        v = uv[:, 1] * vggt_K[view_idx, 1, 1] + vggt_K[view_idx, 1, 2]
        return u, v, z

    def _pixels_to_grid(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert pixel coords to grid_sample coords (align_corners=True)."""
        return torch.stack([
            u / max(W - 1, 1) * 2.0 - 1.0,
            v / max(H - 1, 1) * 2.0 - 1.0,
        ], dim=-1).view(1, -1, 1, 2)

    errs = []
    for i in range(V):
        m_i = mask[i].reshape(-1)
        if m_i.sum() < 16:
            continue
        sel = m_i.nonzero(as_tuple=True)[0]
        if max_points_per_view and sel.numel() > max_points_per_view:
            sub = torch.linspace(
                0, sel.numel() - 1, max_points_per_view,
                device=sel.device,
            ).long()
            sel = sel[sub]
        wp_i = world_points[i].reshape(-1, 3)[sel]  # (N_sel, 3)
        pix_i = pix[sel]

        for j in range(V):
            if j == i:
                continue
            # Project view-i points into view-j pixel space
            u_j, v_j, z_j = _project_to_pixels(wp_i, j)
            valid = ((z_j > 1e-3) &
                     (u_j >= 0) & (u_j <= W - 1) &
                     (v_j >= 0) & (v_j <= H - 1))
            if valid.sum() < 16:
                continue

            # Sample view-j pointmap at projected locations -> 3D points
            wp_j_sampled = F.grid_sample(
                wp_chw[j:j + 1],
                _pixels_to_grid(u_j[valid], v_j[valid]),
                mode="bilinear", padding_mode="border", align_corners=True,
            ).squeeze(-1).squeeze(0).T  # (N_valid, 3)

            # Cycle back to view i and measure normalized pixel reprojection error
            u_i2, v_i2, z_i2 = _project_to_pixels(wp_j_sampled, i)
            pix_i_valid = pix_i[valid]
            valid2 = ((z_i2 > 1e-3) &
                      (u_i2 >= 0) & (u_i2 <= W - 1) &
                      (v_i2 >= 0) & (v_i2 <= H - 1))
            if valid2.sum() < 16:
                continue
            err = torch.stack([
                u_i2[valid2] - pix_i_valid[valid2, 0],
                v_i2[valid2] - pix_i_valid[valid2, 1],
            ], dim=-1).norm(dim=-1) / max(diag, 1e-6)
            errs.append(err.mean().item())

    if not errs:
        return float("nan")
    return float(np.mean(errs))


# ── Metric 3: MEt3R (official implementation) ───────────────────────────────

def _load_met3r(device: torch.device, img_size: int = 256):
    """Load official MEt3R metric (MASt3R + FeatUp + PyTorch3D)."""
    print(f"  [MEt3R] loading official metric (img_size={img_size}) ...", flush=True)
    with _met3r_import_context():
        from met3r import MEt3R
        metric = MEt3R(
            img_size=img_size,
            use_norm=True,
            backbone="mast3r",
            feature_backbone="dino16",
            feature_backbone_weights="mhamilton723/FeatUp",
            upsampler="featup",
            distance="cosine",
            freeze=True,
        ).to(device).eval()
    print("  [MEt3R] loaded")
    return metric


def compute_met3r(
    images_01: torch.Tensor,      # (V, 3, H, W) in [0, 1]
    metric,
    img_size: int = 256,
    pair_mode: str = "all",
    batch_pairs: int = 4,
) -> float:
    """Official MEt3R over image pairs. Lower = more consistent."""
    V = images_01.shape[0]
    if pair_mode == "consecutive":
        pairs = [(i, i + 1) for i in range(V - 1)]
    else:
        pairs = [(i, j) for i in range(V) for j in range(i + 1, V)]
    if not pairs:
        return float("nan")

    imgs = images_01
    if imgs.shape[-2:] != (img_size, img_size):
        imgs = F.interpolate(
            imgs, size=(img_size, img_size),
            mode="bilinear", align_corners=False)

    scores = []
    for start in range(0, len(pairs), batch_pairs):
        batch = pairs[start:start + batch_pairs]
        pair_tensors = [torch.stack([imgs[i], imgs[j]], dim=0) for i, j in batch]
        inputs = torch.stack(pair_tensors, dim=0) * 2.0 - 1.0  # → [-1, 1]
        out = metric(images=inputs)
        batch_scores = out[0]
        if batch_scores.ndim == 0:
            scores.append(float(batch_scores.item()))
        else:
            scores.extend(batch_scores.detach().float().cpu().tolist())

    return float(np.mean(scores))


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate 3D geometric consistency of generated views "
                    "(ATE / RPEr / RPEt / Reproj / MEt3R) following GLD §5.1.4")
    p.add_argument("--pred-dir", required=True,
                   help="Output directory of a diffusion eval script "
                        "(contains <sceneIdx>_pred.mp4 etc.)")
    p.add_argument("--pred-source", default="pred",
                   choices=["pred", "gt"],
                   help="Which mp4 to evaluate. 'gt' = sanity check on the GT "
                        "video itself (should give ≈0 errors).")
    p.add_argument("--gt-source", default="dataset",
                   choices=["dataset", "video"],
                   help="Reference cameras source. 'dataset' loads GT poses "
                        "from the dataset by re-sampling scenes (index-aligned). "
                        "'video' directly pairs <idx>_gt.mp4 with <idx>_pred.mp4 "
                        "in --pred-dir and runs the recon model on the GT video "
                        "to obtain the reference trajectory (no re-sampling; "
                        "robust to ordering/seed drift).")
    p.add_argument("--data-root", default=None)
    p.add_argument("--dataset", default="re10k",
                   choices=["re10k", "re10k_packed", "dl3dv", "dl3dv_packed",
                            "mvssynth", "scannetpp"])
    p.add_argument("--scene-manifest", default=None,
                   help="JSON manifest with fixed scene list (matches the one "
                        "used by the diffusion eval).")
    p.add_argument("--num-scenes", type=int, default=16,
                   help="Used only when --scene-manifest is not given.")
    p.add_argument("--num-views", type=int, default=9)
    p.add_argument("--cond-num", type=int, default=1,
                   help="Number of source views (for reporting only).")
    p.add_argument("--resolution", type=int, nargs=2, default=[504, 504],
                   help="GT image resolution to load (should match the "
                        "diffusion-script resolution).")
    p.add_argument("--recon-model", default="vggt", choices=["vggt", "da3"],
                   help="External geometry estimator for camera/depth/pointmaps.")
    p.add_argument("--vggt-ckpt", default="facebook/VGGT-1B")
    p.add_argument("--da3-ckpt", default="pretrained_models/da3_giant",
                   help="Depth Anything 3 checkpoint/repo for --recon-model da3.")
    p.add_argument("--da3-process-res", type=int, default=504,
                   help="DA3 processing resolution. Default: 504.")
    p.add_argument("--da3-use-gt-cameras", action="store_true",
                   help="Pass GT cameras to DA3 and align depth to GT scale. "
                        "Camera metrics then compare GT to GT and are not an "
                        "independent pose-estimation score.")
    p.add_argument("--da3-use-ray-pose", action="store_true",
                   help="Use DA3 ray-pose mode instead of camera decoder.")
    p.add_argument("--met3r-img-size", type=int, default=256,
                   help="Input resolution for official MEt3R rasterization.")
    p.add_argument("--met3r-pair-mode", default="all",
                   choices=["all", "consecutive"],
                   help="View-pair strategy for MEt3R averaging.")
    p.add_argument("--met3r-batch-pairs", type=int, default=4,
                   help="Number of image pairs per MEt3R forward pass.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--reproj-sample-points", type=int, default=0,
                   help="Max pointmap pixels per source view for cycle "
                        "reprojection. 0 = no cap (use all pixels). Default: 0.")
    p.add_argument("--sample-interval", type=int, default=None)
    p.add_argument("--output-name", default="3d_consistency.json")
    p.add_argument("--csv-output", default=None,
                   help="Optional CSV path. Defaults next to --output-name.")
    p.add_argument("--skip-met3r", action="store_true",
                   help="Skip MEt3R (saves GPU memory; avoids MASt3R/FeatUp load).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate first N scenes (for quick debugging).")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_recon(recon_model, args, frames_np, device, amp_dtype,
               gt_c2w_np=None, gt_K_np=None):
    """Run the selected reconstruction model on uint8 (V,H,W,3) frames."""
    if args.recon_model == "vggt":
        imgs01 = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
        H, W = args.resolution
        imgs01 = imgs01.to(device)
        if imgs01.shape[-2:] != (H, W):
            imgs01 = F.interpolate(imgs01, size=(H, W),
                                   mode="bilinear", align_corners=False)
        return vggt_infer(recon_model, imgs01, device, amp_dtype)
    return da3_infer(
        recon_model, frames_np,
        gt_c2w=gt_c2w_np, gt_K=gt_K_np,
        process_res=args.da3_process_res,
        use_gt_cameras=args.da3_use_gt_cameras,
        use_ray_pose=args.da3_use_ray_pose,
    )


def eval_one(name, frames_np, gt_c2w_np, gt_K_np,
             recon_model, met3r_model, met3r_skip_reason,
             args, device, amp_dtype):
    """Evaluate one sample. Returns a per-scene dict or None on failure."""
    H, W = args.resolution

    try:
        recon_out = _run_recon(recon_model, args, frames_np, device, amp_dtype,
                               gt_c2w_np=gt_c2w_np, gt_K_np=gt_K_np)
    except Exception as e:
        print(f"  [WARN] {args.recon_model} inference failed: {e}; skipping")
        return None
    pred_c2w = recon_out["c2w"]
    if not np.isfinite(pred_c2w).all():
        print(f"  [WARN] {args.recon_model} returned non-finite poses; skipping")
        return None

    cam_m = compute_camera_metrics(pred_c2w, gt_c2w_np)
    print(f"    [Sim3] ATE={cam_m['ate']:.4f}  RPEt={cam_m['rpe_trans']:.4f}  "
          f"RPEr={cam_m['rpe_rot']:.4f}°")
    print(f"    [SE3 ] ATE={cam_m['ate_noscale']:.4f}  "
          f"RPEt={cam_m['rpe_trans_noscale']:.4f}  "
          f"RPEr={cam_m['rpe_rot_noscale']:.4f}°")

    wp_t = torch.from_numpy(recon_out["world_points"]).to(device).float()
    conf_t = torch.from_numpy(recon_out["wp_conf"]).to(device).float()
    recon_c2w_t = torch.from_numpy(recon_out["c2w"]).to(device).float()
    recon_K_t = torch.from_numpy(recon_out["K"]).to(device).float()
    reproj_pct = {}
    for frac in REPROJ_KEEP_FRACTIONS:
        reproj_pct[frac] = compute_reprojection_error(
            wp_t, recon_c2w_t, recon_K_t, conf_t,
            conf_thresh_quantile=(1.0 - frac),
            max_points_per_view=args.reproj_sample_points)
    reproj_norm = reproj_pct[REPROJ_KEEP_FRACTIONS[0]]
    print("    " + "  ".join(
        f"Reproj@{int(round(f * 100))}%={reproj_pct[f]:.4f}"
        for f in REPROJ_KEEP_FRACTIONS) + " (2D cycle / image diagonal)")

    if met3r_model is not None:
        pred_imgs_01 = torch.from_numpy(frames_np).float().permute(0, 3, 1, 2) / 255.0
        pred_imgs_01 = pred_imgs_01.to(device)
        if pred_imgs_01.shape[-2:] != (H, W):
            pred_imgs_01 = F.interpolate(pred_imgs_01, size=(H, W),
                                         mode="bilinear", align_corners=False)
        met3r = compute_met3r(
            pred_imgs_01, met3r_model,
            img_size=args.met3r_img_size,
            pair_mode=args.met3r_pair_mode,
            batch_pairs=args.met3r_batch_pairs)
        print(f"    MEt3R={met3r:.4f}")
    else:
        met3r = float("nan")
        if met3r_skip_reason:
            print(f"    MEt3R=nan ({met3r_skip_reason})")

    return {
        "scene": name,
        "ate": cam_m["ate"],
        "rpe_trans": cam_m["rpe_trans"],
        "rpe_rot": cam_m["rpe_rot"],
        "ate_noscale": cam_m["ate_noscale"],
        "rpe_trans_noscale": cam_m["rpe_trans_noscale"],
        "rpe_rot_noscale": cam_m["rpe_rot_noscale"],
        "reproj_norm": reproj_norm,
        **{_reproj_key(f): reproj_pct[f] for f in REPROJ_KEEP_FRACTIONS},
        "reproj": reproj_norm,
        "met3r": met3r,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.resolution
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    print("=" * 70)
    print("3D Geometric Consistency Evaluation (GLD §5.1.4)")
    print(f"  pred-dir:       {args.pred_dir}")
    print(f"  pred-source:    {args.pred_source}")
    print(f"  dataset:        {args.dataset}")
    print(f"  num-views:      {args.num_views} (cond={args.cond_num})")
    print(f"  resolution:     {H}x{W}")
    print(f"  recon-model:    {args.recon_model}")
    if args.recon_model == "vggt":
        print(f"  VGGT:           {args.vggt_ckpt}")
    else:
        print(f"  DA3:            {args.da3_ckpt} "
              f"(process_res={args.da3_process_res}, "
              f"gt_cameras={args.da3_use_gt_cameras}, "
              f"ray_pose={args.da3_use_ray_pose})")
    print(f"  MEt3R:          {'(skipped)' if args.skip_met3r else f'official, img={args.met3r_img_size}, pairs={args.met3r_pair_mode}'}")
    print(f"  reproj-keep:    {', '.join(f'{int(round(f*100))}%' for f in REPROJ_KEEP_FRACTIONS)}")
    print("=" * 70)

    print(f"  gt-source:      {args.gt_source}")

    # ── Build work items ──
    #   video  : pair <idx>_gt.mp4 with <idx>_pred.mp4 directly (no re-sampling)
    #   dataset: re-sample scenes and pair by index with the pred mp4s
    video_pairs = None
    scenes = None
    if args.gt_source == "video":
        video_pairs = collect_video_pairs(
            args.pred_dir, args.dataset, pred_source=args.pred_source)
        if args.limit:
            video_pairs = video_pairs[:args.limit]
        print(f"Found {len(video_pairs)} gt/pred video pairs\n")
        if not video_pairs:
            print("No <idx>_gt.mp4 / <idx>_pred.mp4 pairs found; nothing to evaluate.")
            return
    else:
        if args.scene_manifest:
            scenes = load_scenes_from_manifest(
                args.scene_manifest, num_views=args.num_views,
                interval=args.sample_interval)
        elif args.data_root:
            scenes = collect_scenes(
                args.dataset, args.data_root, args.num_scenes, args.num_views,
                args.seed, interval=args.sample_interval)
        else:
            ds_root = DATASET_ROOTS.get(args.dataset, "")
            scenes = collect_scenes(
                args.dataset, ds_root, args.num_scenes, args.num_views,
                args.seed, interval=args.sample_interval)
        if args.limit:
            scenes = scenes[:args.limit]
        print(f"Loaded {len(scenes)} scenes\n")
        if not scenes:
            print("No scenes selected; nothing to evaluate.")
            return

    # ── Load estimators (lazy, only once) ──
    if args.recon_model == "vggt":
        recon_model = _load_vggt(args.vggt_ckpt, device)
    else:
        recon_model = _load_da3(args.da3_ckpt, device)
    met3r_model = None
    met3r_skip_reason = None
    if args.skip_met3r:
        met3r_skip_reason = "--skip-met3r"
    else:
        try:
            met3r_model = _load_met3r(device, img_size=args.met3r_img_size)
        except Exception as e:
            met3r_skip_reason = str(e)
            print(f"  [WARN] Failed to load MEt3R ({e}); disabling MEt3R.")
            print(f"  [WARN] install met3r/featup/pytorch3d (see README).")
            met3r_model = None

    # ── Evaluate ──
    per_scene = []
    if args.gt_source == "video":
        n = len(video_pairs)
        for i, (name, gt_path, pred_path) in enumerate(video_pairs):
            print(f"[{i+1}/{n}] {name}")
            try:
                gt_frames = load_video_frames(gt_path)
                pred_frames = load_video_frames(pred_path)
            except Exception as e:
                print(f"  [WARN] failed to load videos: {e}; skipping")
                continue
            # Reference trajectory: run the same recon model on the GT video.
            try:
                gt_recon = _run_recon(recon_model, args, gt_frames, device, amp_dtype)
            except Exception as e:
                print(f"  [WARN] recon on GT video failed: {e}; skipping")
                continue
            gt_c2w_np = gt_recon["c2w"]
            if not np.isfinite(gt_c2w_np).all():
                print(f"  [WARN] recon returned non-finite GT poses; skipping")
                continue
            row = eval_one(name, pred_frames, gt_c2w_np, None,
                           recon_model, met3r_model, met3r_skip_reason,
                           args, device, amp_dtype)
            if row is not None:
                per_scene.append(row)
    else:
        n = len(scenes)
        for s_idx, (scene_name, scene_dir, img_names, ds_type) in enumerate(scenes):
            actual_v = len(img_names)
            if actual_v != args.num_views:
                print(f"  [WARN] Scene {scene_name} has {actual_v} views, "
                      f"expected {args.num_views}. Skipping.")
                continue

            print(f"[{s_idx+1}/{n}] {scene_name}")
            try:
                mp4_path = find_pred_mp4(args.pred_dir, args.dataset, s_idx,
                                         pred_source=args.pred_source)
            except FileNotFoundError as e:
                print(f"  [WARN] {e}; skipping")
                continue

            try:
                frames_np = load_video_frames(mp4_path)
            except Exception as e:
                print(f"  [WARN] failed to load {mp4_path}: {e}; skipping")
                continue
            if frames_np.shape[0] != actual_v:
                print(f"  [WARN] mp4 has {frames_np.shape[0]} frames, "
                      f"expected {actual_v}; truncating")
                frames_np = frames_np[:actual_v]

            try:
                _, gt_K, gt_c2w = load_gt_for_scene(
                    scene_dir, img_names, (H, W), ds_type, device)
            except Exception as e:
                print(f"  [WARN] GT load failed: {e}; skipping")
                continue
            gt_c2w_np = gt_c2w.cpu().numpy()

            row = eval_one(scene_name, frames_np, gt_c2w_np, gt_K.cpu().numpy(),
                           recon_model, met3r_model, met3r_skip_reason,
                           args, device, amp_dtype)
            if row is not None:
                per_scene.append(row)

    # ── Aggregate ──
    print("\n" + "=" * 70)
    print(f"Summary  ({len(per_scene)} scenes evaluated)")
    print("=" * 70)
    if not per_scene:
        print("No scenes evaluated; nothing to report.")
        return

    def _mean(key):
        vals = [s[key] for s in per_scene
                if s[key] is not None and not math.isnan(float(s[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "ATE":   _mean("ate"),
        "RPEt":  _mean("rpe_trans"),
        "RPEr":  _mean("rpe_rot"),
        "ATE_noscale":  _mean("ate_noscale"),
        "RPEt_noscale": _mean("rpe_trans_noscale"),
        "RPEr_noscale": _mean("rpe_rot_noscale"),
        "Reproj_norm": _mean("reproj_norm"),
        **{f"Reproj_{int(round(f*100))}": _mean(_reproj_key(f))
           for f in REPROJ_KEEP_FRACTIONS},
        "MEt3R": _mean("met3r"),
        # Paper table aliases.
        "Reproj": _mean("reproj_norm"),
        "num_scenes": len(per_scene),
    }
    print("  Reproj = " + "  ".join(
        f"{int(round(f*100))}%:{summary[f'Reproj_{int(round(f*100))}']:.4f}"
        for f in REPROJ_KEEP_FRACTIONS) + "  (2D cycle / image diagonal)")
    print(f"  ATE   = {summary['ATE']:.4f}   (Sim3-aligned, GT scale)")
    print(f"  RPEt  = {summary['RPEt']:.4f}")
    print(f"  RPEr  = {summary['RPEr']:.4f}°")
    print(f"  ATE (no-scale/SE3)  = {summary['ATE_noscale']:.4f}")
    print(f"  RPEt(no-scale/SE3)  = {summary['RPEt_noscale']:.4f}")
    print(f"  RPEr(no-scale/SE3)  = {summary['RPEr_noscale']:.4f}°")
    print(f"  Reproj_norm= {summary['Reproj_norm']:.4f}  (2D cycle / image diagonal)")
    print(f"  MEt3R = {summary['MEt3R']:.4f}  (official, lower=better)")

    # ── Save JSON ──
    out = {
        "config": {
            "pred_dir": args.pred_dir,
            "pred_source": args.pred_source,
            "gt_source": args.gt_source,
            "dataset": args.dataset,
            "scene_manifest": args.scene_manifest,
            "num_views": args.num_views,
            "cond_num": args.cond_num,
            "resolution": [H, W],
            "recon_model": args.recon_model,
            "vggt_ckpt": args.vggt_ckpt,
            "da3_ckpt": args.da3_ckpt,
            "da3_process_res": args.da3_process_res,
            "da3_use_gt_cameras": args.da3_use_gt_cameras,
            "da3_use_ray_pose": args.da3_use_ray_pose,
            "met3r_img_size": (None if args.skip_met3r else args.met3r_img_size),
            "met3r_pair_mode": (None if args.skip_met3r else args.met3r_pair_mode),
            "met3r_skip_reason": met3r_skip_reason,
            "reproj_keep_fractions": list(REPROJ_KEEP_FRACTIONS),
            "reproj_sample_points": args.reproj_sample_points,
        },
        "summary": summary,
        "per_scene": per_scene,
    }
    out_path = os.path.join(args.pred_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    csv_path = args.csv_output
    if csv_path is None:
        root, _ = os.path.splitext(out_path)
        csv_path = root + ".csv"
    with open(csv_path, "w", newline="") as f:
        reproj_cols = [_reproj_key(f) for f in REPROJ_KEEP_FRACTIONS]
        fieldnames = [
            "scene", "ate", "rpe_trans", "rpe_rot",
            "ate_noscale", "rpe_trans_noscale", "rpe_rot_noscale",
            *reproj_cols, "met3r",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_scene:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        mean_row = {
            "scene": "MEAN",
            "ate": summary["ATE"],
            "rpe_trans": summary["RPEt"],
            "rpe_rot": summary["RPEr"],
            "ate_noscale": summary["ATE_noscale"],
            "rpe_trans_noscale": summary["RPEt_noscale"],
            "rpe_rot_noscale": summary["RPEr_noscale"],
            "met3r": summary["MEt3R"],
        }
        for f_ in REPROJ_KEEP_FRACTIONS:
            mean_row[_reproj_key(f_)] = summary[f"Reproj_{int(round(f_*100))}"]
        writer.writerow(mean_row)
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
