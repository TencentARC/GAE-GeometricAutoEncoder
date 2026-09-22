#!/usr/bin/env python3
"""
Evaluate **NGD's native (direct) geometry** against external reconstructions.

Motivation
──────────
`scripts/eval_3d_consistency.py` answers a *different* question: it takes the
**generated RGB video** and runs an external estimator (VGGT / DA3) on it to
recover pose/depth/points, then scores that reconstruction. It never touches
NGD's own geometry head.

But NGD's VAE *directly decodes* depth + rays (→ camera pose) + point maps from
its latent (`z → D → (I_hat, G_hat)`; see `method.tex`). The question this
script answers is therefore:

    How good is NGD's **direct** depth/pose/point-map output, compared to what a
    dedicated estimator (DA3 / VGGT) reconstructs from RGB?

It builds a single table whose rows are geometry *sources* and whose columns are
direct-geometry metrics (Depth AbsRel/RMSE/δ1, Pose ATE/RPEt/RPEr, PointCloud
Chamfer/PointMap-L1). No reprojection / MEt3R here — those are indirect
consistency proxies handled by `eval_3d_consistency.py`.

    ┌───────────────────┬─────────── Depth ──────────┬──── Pose (Sim3) ───┬─ PointCloud ─┐
    │ source            │ AbsRel ↓  RMSE ↓  δ1 ↑      │ ATE ↓ RPEt ↓ RPEr ↓│ Chamfer ↓ L1 ↓│
    ├───────────────────┼────────────────────────────┼────────────────────┼──────────────┤
    │ DA3   (from RGB)  │  ...                        │  ...               │  ...          │
    │ VGGT  (from RGB)  │  ...                        │  ...               │  ...          │
    │ NGD-VAE  (direct) │  ...                        │  ...               │  ...          │
    │ NGD-Gen  (direct) │  ...                        │  ...               │  ...          │
    └───────────────────┴────────────────────────────┴────────────────────┴──────────────┘

Inputs
──────
Per-scene ``<idx>_geom.npz`` files produced by ``eval_generation.py``
run with ``--dump-geometry``. Each holds (keys present depend on eval mode):

    gen_depth (V,H,W)  gen_ray (V,H,W,6)  gen_ray_conf (V,H,W)   # NGD generated-direct
    vae_depth          vae_ray            vae_ray_conf           # NGD VAE-direct (encode→decode)
    gtmodel_depth      gtmodel_ray        gtmodel_ray_conf       # NGD DPT on real GT features (pseudo-GT)
    gen_rgb (V,3,H,W)  gt_rgb (V,3,H,W)                          # RGB in [0,1]
    gt_c2w (V,4,4)     gt_K (V,3,3)                              # real dataset cameras (world frame)
    cond_num (scalar)  image_size (2,)

Reference (ground truth)
────────────────────────
  * Pose  : the **real dataset cameras** ``gt_c2w`` (true GT trajectory).
  * Depth : a pseudo-GT depth chosen by ``--gt-depth-source`` (default ``model``
            = NGD's DPT on real frozen features — the exact target ``L_geo`` is
            trained against, and neutral w.r.t. every RGB-estimator row).
  * Points: ``gt_depth`` back-projected with the real cameras.

Alignment
─────────
NGD's recovered poses are ref-centric & up-to-scale; DA3/VGGT live in their own
frame/scale. All metrics are therefore alignment-invariant:
  * Depth : per-view median-scale alignment (standard monocular protocol).
  * Pose  : evo Sim3(Umeyama) + SE3 alignment (reuses eval_3d_consistency).
  * Points: two Sim3 alignments — (1) camera-trajectory Umeyama applied to
            points (``chamfer``/``pmap_l1``, couples pose); (2) dense point-map
            Umeyama with no cameras (``chamfer_pc``/``pmap_l1_pc``).

Usage
─────
    cd /path/to/GLD

    # 1) dump geometry during the normal v4 recon eval
    PYTHONPATH=src uv run python scripts/eval_generation.py \
        --dit-ckpt <ckpt> --mode recon --dump-geometry \
        --output-dir results/eval_v4_geom --dataset re10k --num-views 8

    # 2) score NGD-direct vs DA3/VGGT
    PYTHONPATH=src uv run python scripts/eval_direct_geometry.py \
        --geom-dir results/eval_v4_geom/re10k \
        --estimators da3,vggt --rgb-sources gen,gt
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import math
import os
import sys

os.environ.setdefault("TMPDIR", "/tmp")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC_ROOT)

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the battle-tested estimator wrappers + pose/point helpers.
from scripts.eval_3d_consistency import (
    _as_4x4_extrinsics,
    _depth_to_world_points,
    _load_da3,
    _load_vggt,
    _sim3_align_world_points_to_gt,
    compute_camera_metrics,
    da3_infer,
    vggt_infer,
)
from utils.camera_from_ray import recover_poses
from utils.metrics import compute_abs_rel, compute_delta, compute_depth_rmse


# ── Row labels ───────────────────────────────────────────────────────────────
# Canonical, stable order for the output table (only present rows are emitted).
ROW_ORDER = [
    # reconstruction block (real frames)
    "DA3(real)", "VGGT(real)", "PI3(real)", "DA3(gt)", "VGGT(gt)", "NGD-VAE",
    # generation block (generated frames)
    "DA3(gen)", "VGGT(gen)", "PI3(gen)", "NGD-Gen",
    # Gen3R baseline (its own generated frames)
    "DA3(gen3r)", "VGGT(gen3r)", "PI3(gen3r)", "Gen3R", "Gen3R*",
]


# ── Pi3 (π³) reference estimator ─────────────────────────────────────────────

_PI3_ROOT_DEFAULT = "/local-ssd/thirdparty/Pi3"
_PI3_CKPT_DEFAULT = "/local-ssd/_eval_cache/pi3/model.safetensors"


def _load_pi3(ckpt: str, device: torch.device, pi3_root: str | None = None):
    """Load Pi3 weights from a local safetensors / pt checkpoint."""
    root = pi3_root or _PI3_ROOT_DEFAULT
    if root not in sys.path:
        sys.path.insert(0, root)
    from pi3.models.pi3 import Pi3  # noqa: WPS410

    model = Pi3().to(device).eval()
    ckpt = ckpt or _PI3_CKPT_DEFAULT
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Pi3 checkpoint not found: {ckpt}")
    if ckpt.endswith(".safetensors"):
        from safetensors.torch import load_file
        weight = load_file(ckpt, device="cpu")
    else:
        weight = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(weight, strict=True)
    print(f"  [pi3] loaded {ckpt}")
    return model


def pi3_infer(model, images_01: torch.Tensor, device: torch.device,
              amp_dtype) -> dict:
    """Run Pi3 on (V,3,H,W) RGB in [0,1] → geometry dict matching da3/vggt_infer.

    Pi3 predicts scale-invariant local point maps + OpenCV c2w. Depth is the
    camera-frame ``z`` of ``local_points``. Intrinsics are a placeholder pinhole
    (unused by depth/pose metrics; points come from the model directly).
    """
    if images_01.ndim != 4 or images_01.shape[1] != 3:
        raise ValueError(f"pi3_infer expects (V,3,H,W), got {tuple(images_01.shape)}")
    imgs = images_01.float().to(device)
    V, _, H, W = imgs.shape
    # DINOv2 patch size 14 — bilinear-resize if needed.
    if H % 14 != 0 or W % 14 != 0:
        Ht, Wt = (H // 14) * 14, (W // 14) * 14
        imgs = F.interpolate(imgs, size=(Ht, Wt), mode="bilinear",
                             align_corners=False)
        H, W = Ht, Wt
    dtype = amp_dtype if device.type == "cuda" else torch.float32
    with torch.no_grad():
        ctx = (torch.amp.autocast("cuda", dtype=dtype)
               if device.type == "cuda" else contextlib.nullcontext())
        with ctx:
            res = model(imgs[None])
    local = res["local_points"][0].float().cpu().numpy()       # (V,H,W,3)
    points = res["points"][0].float().cpu().numpy()            # (V,H,W,3)
    c2w = res["camera_poses"][0].float().cpu().numpy()         # (V,4,4)
    depth = local[..., 2].astype(np.float32)
    # Placeholder K (depth metrics are median-scale; points are used as-is).
    fx = fy = float(max(H, W))
    K = np.zeros((V, 3, 3), dtype=np.float32)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = W * 0.5
    K[:, 1, 2] = H * 0.5
    K[:, 2, 2] = 1.0
    return {
        "c2w": c2w.astype(np.float32),
        "K": K,
        "depth": depth,
        "world_points": points.astype(np.float32),
    }


def _maybe_load_gt_rgb(npz_path: str, npz: dict, args) -> np.ndarray | None:
    """Return gt_rgb from npz, or from a sibling recon dump if configured."""
    rgb = _get(npz, "gt_rgb")
    if rgb is not None:
        return rgb
    src_dir = getattr(args, "gt_rgb_from_geom_dir", None)
    if not src_dir:
        return None
    base = os.path.basename(npz_path)
    alt = os.path.join(src_dir, base)
    if not os.path.isfile(alt):
        return None
    try:
        z = np.load(alt, allow_pickle=True)
        return _get({k: z[k] for k in z.files}, "gt_rgb")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] failed to load gt_rgb from {alt}: {e}")
        return None


# ── Small numeric helpers ────────────────────────────────────────────────────

def _finite_mean(vals) -> float:
    vals = [float(v) for v in vals
            if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _resize_depth(depth: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize (V,H,W) depth to target (H,W)."""
    if depth.shape[-2:] == tuple(hw):
        return depth
    t = torch.from_numpy(np.ascontiguousarray(depth)).float().unsqueeze(1)
    t = F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)
    return t.squeeze(1).numpy()


def _resize_points(points: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize (V,H,W,3) world points to target (H,W)."""
    if points.shape[1:3] == tuple(hw):
        return points
    t = torch.from_numpy(np.ascontiguousarray(points)).float().permute(0, 3, 1, 2)
    t = F.interpolate(t, size=tuple(hw), mode="bilinear", align_corners=False)
    return t.permute(0, 2, 3, 1).numpy()


def _median_scale(pred: torch.Tensor, gt: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """Per-view median-scale align: pred *= median(gt)/median(pred).

    Standard scale-invariant monocular-depth protocol. Operates on (V,H,W).
    """
    out = pred.clone()
    for v in range(pred.shape[0]):
        m = mask[v]
        if m.sum() < 16:
            continue
        pm = torch.median(pred[v][m])
        gm = torch.median(gt[v][m])
        if pm > 1e-6:
            out[v] = pred[v] * (gm / pm)
    return out


# ── Depth metrics (scale-invariant, no reproj) ───────────────────────────────

def depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray,
                  view_slice: slice | None = None,
                  max_depth: float = 100.0) -> dict:
    """AbsRel / RMSE / δ1 with per-view median-scale alignment.

    Both inputs (V,H,W). ``pred_depth`` is resized to the GT grid. Returns NaNs
    when no valid pixels. ``view_slice`` restricts to a subset of views
    (e.g. target views only).
    """
    nan = {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    Hg, Wg = gt_depth.shape[-2:]
    pred = _resize_depth(pred_depth, (Hg, Wg))
    p = torch.from_numpy(np.ascontiguousarray(pred)).float()
    g = torch.from_numpy(np.ascontiguousarray(gt_depth)).float()
    if view_slice is not None:
        p, g = p[view_slice], g[view_slice]
    if p.shape[0] == 0:
        return dict(nan)

    # Valid where BOTH depths are finite and positive (guards the gt/pred
    # ratio in δ from sign flips and division-by-zero).
    mask = (torch.isfinite(g) & torch.isfinite(p)
            & (g > 1e-3) & (g < max_depth)
            & (p > 1e-3) & (p < max_depth))
    p = _median_scale(p, g, mask)  # positive scale → mask stays valid
    g_masked = torch.where(mask, g, torch.zeros_like(g))

    abs_rel = compute_abs_rel(p, g_masked, valid_mask=mask)
    rmse = compute_depth_rmse(p, g_masked, valid_mask=mask)
    delta1 = compute_delta(p, g_masked, threshold=1.25, valid_mask=mask)

    # Average only over views with enough valid pixels — the metric helpers
    # emit a misleading "perfect" value (0 / 0 / 1.0) for empty views.
    ok = mask.reshape(mask.shape[0], -1).sum(dim=1) >= 16
    if ok.sum() == 0:
        return dict(nan)
    return {
        "abs_rel": float(abs_rel[ok].mean().item()),
        "rmse": float(rmse[ok].mean().item()),
        "delta1": float(delta1[ok].mean().item()),
    }


# ── Point-cloud metrics (Sim3-aligned) ───────────────────────────────────────

def _subsample(pts: np.ndarray, max_pts: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] <= max_pts:
        return pts
    idx = rng.choice(pts.shape[0], max_pts, replace=False)
    return pts[idx]


def _chamfer(pred_pts: np.ndarray, gt_pts: np.ndarray,
             device: torch.device, block: int = 2048) -> float:
    """Symmetric Chamfer distance (mean of both directions), chunked NN.

    Falls back to CPU if the GPU is out of memory (the estimators already hold
    several GB), so a large point set never crashes the whole eval.
    """
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("nan")

    def _run(dev):
        a = torch.from_numpy(pred_pts).float().to(dev)
        b = torch.from_numpy(gt_pts).float().to(dev)

        def _nn_mean(src, tgt):
            acc = 0.0
            for i in range(0, src.shape[0], block):
                d = torch.cdist(src[i:i + block], tgt)  # (blk, M)
                acc += d.min(dim=1).values.sum().item()
            return acc / src.shape[0]

        return float(0.5 * (_nn_mean(a, b) + _nn_mean(b, a)))

    try:
        return _run(device)
    except RuntimeError as e:  # typically CUDA OOM
        if device.type == "cuda":
            torch.cuda.empty_cache()
            return _run(torch.device("cpu"))
        raise e


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama Sim(3): find ``s, R, t`` s.t. ``dst ≈ s R src + t``.

    ``src`` / ``dst`` are corresponding ``(N, 3)`` point sets (N>=3).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"umeyama expects (N,3) pairs, got {src.shape}/{dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"umeyama needs >=3 points, got {n}")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float((src_c ** 2).sum() / n)
    cov = (dst_c.T @ src_c) / n  # (3, 3)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (var_s + 1e-12))
    t = mu_d - s * (R @ mu_s)
    return R.astype(np.float64), t.astype(np.float64), s


def _sim3_align_points_direct(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    *,
    stride: int = 4,
    max_corr: int = 50000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sim(3)-align ``src_points`` to ``dst_points`` using *point* correspondences.

    Both are dense ``(V,H,W,3)`` maps on the same view grid (src is resized to
    dst's HxW). Finite pixel pairs are treated as correspondences for Umeyama —
    no camera poses involved. Returns aligned src at dst resolution.
    """
    rng = rng or np.random.default_rng(0)
    dst = np.asarray(dst_points, dtype=np.float32)
    src = _resize_points(np.asarray(src_points, dtype=np.float32),
                         (dst.shape[1], dst.shape[2]))
    mask = (np.isfinite(src).all(axis=-1) & np.isfinite(dst).all(axis=-1))
    # Subsample on the strided grid for a stable, cheap Umeyama fit.
    mask_s = np.zeros_like(mask)
    mask_s[:, ::stride, ::stride] = mask[:, ::stride, ::stride]
    src_c = src[mask_s]
    dst_c = dst[mask_s]
    if src_c.shape[0] > max_corr:
        idx = rng.choice(src_c.shape[0], max_corr, replace=False)
        src_c, dst_c = src_c[idx], dst_c[idx]
    if src_c.shape[0] < 32:
        raise RuntimeError(
            f"too few finite correspondences for point Sim3 ({src_c.shape[0]})")
    R, t, s = _umeyama_sim3(src_c, dst_c)
    flat = src.reshape(-1, 3).astype(np.float64)
    aligned = (s * (flat @ R.T) + t).reshape(src.shape).astype(np.float32)
    return aligned


def _pc_metrics_from_aligned(aligned: np.ndarray, gt_points: np.ndarray,
                             device: torch.device, *, stride: int,
                             max_pts: int, rng: np.random.Generator
                             ) -> tuple[float, float]:
    a = aligned[:, ::stride, ::stride, :].reshape(-1, 3)
    b = gt_points[:, ::stride, ::stride, :].reshape(-1, 3)
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    a = _subsample(a, max_pts, rng)
    b = _subsample(b, max_pts, rng)
    chamfer = _chamfer(a, b, device)
    Hg, Wg = gt_points.shape[1:3]
    aligned_r = _resize_points(aligned, (Hg, Wg))
    diff = np.linalg.norm(aligned_r - gt_points, axis=-1)
    finite = np.isfinite(diff)
    pmap_l1 = float(diff[finite].mean()) if finite.any() else float("nan")
    return chamfer, pmap_l1


def pointcloud_metrics(sample: dict, gt_points: np.ndarray, gt_c2w: np.ndarray,
                       device: torch.device, *, stride: int = 4,
                       max_pts: int = 30000,
                       rng: np.random.Generator | None = None) -> dict:
    """Chamfer + PMap-L1 under two Sim3 alignments.

    * ``chamfer`` / ``pmap_l1``: Sim3 from **camera trajectories** (pose-mediated),
      then applied to points — couples pose error into the point metrics.
    * ``chamfer_pc`` / ``pmap_l1_pc``: Sim3 from **dense point-map correspondences**
      (Umeyama on finite pixel pairs) — pose-free shape agreement.
    """
    rng = rng or np.random.default_rng(0)
    src_pts = np.asarray(sample["points"], dtype=np.float32)
    src_c2w = sample["c2w"]
    gt_points = np.asarray(gt_points, dtype=np.float32)
    nan = float("nan")
    out = {"chamfer": nan, "pmap_l1": nan, "chamfer_pc": nan, "pmap_l1_pc": nan}

    # ── (1) pose-mediated Sim3 ──
    try:
        aligned_pose = _sim3_align_world_points_to_gt(
            np.asarray(src_c2w, dtype=np.float64),
            np.asarray(gt_c2w, dtype=np.float64),
            src_pts,
        )
        out["chamfer"], out["pmap_l1"] = _pc_metrics_from_aligned(
            aligned_pose, gt_points, device, stride=stride,
            max_pts=max_pts, rng=rng)
    except Exception as e:  # noqa: BLE001
        print(f"      [WARN] pose-Sim3 point align failed: {e}")

    # ── (2) point-direct Sim3 (no cameras) ──
    try:
        aligned_pc = _sim3_align_points_direct(
            src_pts, gt_points, stride=stride, rng=rng)
        out["chamfer_pc"], out["pmap_l1_pc"] = _pc_metrics_from_aligned(
            aligned_pc, gt_points, device, stride=stride,
            max_pts=max_pts, rng=rng)
    except Exception as e:  # noqa: BLE001
        print(f"      [WARN] point-Sim3 align failed: {e}")

    return out


# ── Geometry source adapters ─────────────────────────────────────────────────

def _ngd_sample(depth: np.ndarray, ray: np.ndarray,
                ray_conf: np.ndarray | None, H: int, W: int) -> dict | None:
    """Build a geometry sample from NGD direct depth + ray head output.

    Recovers per-view c2w and intrinsics from the ray head (DA3 official fit),
    then back-projects depth to world points in the recovered frame.
    """
    if depth is None or ray is None:
        return None
    ray = np.asarray(ray, dtype=np.float32)
    # ``ray`` should be (V,H,W,6); collapse any leading batch dims.
    while ray.ndim > 4:
        ray = ray[0]
    if ray_conf is not None:
        ray_conf = np.asarray(ray_conf, dtype=np.float32)
        # dump stores conf as (1,V,H,W); recover_poses wants (V,H,W) after a
        # single [None] wrap, so strip any extra leading singleton dims here.
        while ray_conf.ndim > 3:
            ray_conf = ray_conf[0]
    c2w_list, K_v = recover_poses(
        ray[None], None if ray_conf is None else ray_conf[None],
        ref_view=0, subsample=4, input_size=(H, W),
        return_per_view_intrinsics=True,
    )
    V = len(c2w_list)
    c2w = np.stack(c2w_list, axis=0).astype(np.float32)
    K_v = np.asarray(K_v, dtype=np.float32)
    depth = np.ascontiguousarray(depth.astype(np.float32))
    points = _depth_to_world_points(depth, c2w, K_v)
    return {"c2w": c2w, "K": K_v, "depth": depth, "points": points}


def _estimator_sample(estimator: str, model, rgb_01: np.ndarray,
                      gt_c2w: np.ndarray, gt_K: np.ndarray,
                      device: torch.device, amp_dtype, args) -> dict | None:
    """Run DA3/VGGT/Pi3 on (V,3,H,W)[0,1] RGB → geometry sample."""
    if rgb_01 is None:
        return None
    if estimator == "vggt":
        imgs = torch.from_numpy(np.ascontiguousarray(rgb_01)).float().to(device)
        out = vggt_infer(model, imgs, device, amp_dtype)
    elif estimator == "pi3":
        imgs = torch.from_numpy(np.ascontiguousarray(rgb_01)).float().to(device)
        out = pi3_infer(model, imgs, device, amp_dtype)
    else:  # da3 — feed uint8 (V,H,W,3) frames
        frames = (np.transpose(rgb_01, (0, 2, 3, 1)) * 255.0)
        frames = frames.clip(0, 255).astype(np.uint8)
        out = da3_infer(
            model, frames,
            gt_c2w=gt_c2w, gt_K=gt_K,
            process_res=args.da3_process_res,
            use_gt_cameras=args.da3_use_gt_cameras,
            use_ray_pose=args.da3_use_ray_pose,
        )
    return {
        "c2w": np.asarray(out["c2w"], dtype=np.float32),
        "K": np.asarray(out["K"], dtype=np.float32),
        "depth": np.asarray(out["depth"], dtype=np.float32),
        "points": np.asarray(out["world_points"], dtype=np.float32),
    }


# ── Per-scene evaluation ─────────────────────────────────────────────────────

def _score_source(name: str, sample: dict, gt_ref: dict, gt_c2w_real: np.ndarray,
                  cond_num: int, device: torch.device,
                  args, rng) -> dict:
    """All direct-geometry metrics for one source vs GT.

    ``gt_ref`` is the internally-consistent geometry reference
    (``depth``/``points``/``c2w``) used for depth + point-cloud metrics.
    ``gt_c2w_real`` is the true dataset trajectory used for pose metrics.
    """
    V = gt_c2w_real.shape[0]
    tgt = slice(cond_num, V) if cond_num < V else slice(0, V)

    dep = depth_metrics(sample["depth"], gt_ref["depth"], view_slice=tgt)
    dep_all = depth_metrics(sample["depth"], gt_ref["depth"], view_slice=None)

    # Pose: reuse evo Sim3 + SE3 aligned ATE/RPEt/RPEr vs the REAL trajectory.
    pose = compute_camera_metrics(
        np.asarray(sample["c2w"], dtype=np.float64),
        np.asarray(gt_c2w_real, dtype=np.float64))

    # Point cloud: align to the consistent geometry reference frame.
    pc = pointcloud_metrics(sample, gt_ref["points"], gt_ref["c2w"], device,
                            stride=args.pc_stride, rng=rng)

    return {
        "source": name,
        # depth (target views)
        "abs_rel": dep["abs_rel"], "rmse": dep["rmse"], "delta1": dep["delta1"],
        # depth (all views)
        "abs_rel_all": dep_all["abs_rel"], "rmse_all": dep_all["rmse"],
        "delta1_all": dep_all["delta1"],
        # pose (Sim3)
        "ate": pose["ate"], "rpe_trans": pose["rpe_trans"],
        "rpe_rot": pose["rpe_rot"],
        # pose (SE3, no scale)
        "ate_noscale": pose["ate_noscale"],
        "rpe_trans_noscale": pose["rpe_trans_noscale"],
        "rpe_rot_noscale": pose["rpe_rot_noscale"],
        # point cloud (pose-Sim3 + point-Sim3)
        "chamfer": pc["chamfer"], "pmap_l1": pc["pmap_l1"],
        "chamfer_pc": pc["chamfer_pc"], "pmap_l1_pc": pc["pmap_l1_pc"],
    }


def _estimator_ref(est: str, model, rgb: np.ndarray | None,
                   gt_c2w: np.ndarray, gt_K: np.ndarray, H: int, W: int,
                   device, amp_dtype, args) -> dict | None:
    """Reference geometry from an estimator on a given frame set (depth→(H,W))."""
    s = _estimator_sample(est, model, rgb, gt_c2w, gt_K, device, amp_dtype, args)
    if s is None:
        return None
    return {"depth": _resize_depth(s["depth"], (H, W)),
            "points": s["points"], "c2w": s["c2w"], "_sample": s}


def _pose_only_row(name: str, sample: dict, gt_c2w_real: np.ndarray) -> dict:
    """Reference row: pose vs the REAL trajectory; depth/points = NaN.

    Used for the reference estimator itself in --degradation mode so its pose
    accuracy is shown as the Δ baseline without a (trivially-zero) self depth.
    """
    pose = compute_camera_metrics(
        np.asarray(sample["c2w"], dtype=np.float64),
        np.asarray(gt_c2w_real, dtype=np.float64))
    nan = float("nan")
    return {
        "source": name,
        "abs_rel": nan, "rmse": nan, "delta1": nan,
        "abs_rel_all": nan, "rmse_all": nan, "delta1_all": nan,
        "ate": pose["ate"], "rpe_trans": pose["rpe_trans"],
        "rpe_rot": pose["rpe_rot"],
        "ate_noscale": pose["ate_noscale"],
        "rpe_trans_noscale": pose["rpe_trans_noscale"],
        "rpe_rot_noscale": pose["rpe_rot_noscale"],
        "chamfer": nan, "pmap_l1": nan,
        "chamfer_pc": nan, "pmap_l1_pc": nan,
    }


def _scene_dims(npz: dict, npz_path: str):
    """(H, W, cond_num) from an <idx>_geom.npz, or (None, None, None)."""
    if "image_size" in npz:
        H, W = int(npz["image_size"][0]), int(npz["image_size"][1])
    else:
        H = W = None
        for k in ("gtmodel_depth", "gen_depth", "vae_depth"):
            if k in npz:
                H, W = (int(x) for x in npz[k].shape[-2:])
                break
    cond = int(npz["cond_num"]) if "cond_num" in npz else None
    return H, W, cond


def evaluate_scene_degradation(npz_path: str, ref_name: str, ref_model,
                               cross_models: dict, args, device, amp_dtype,
                               rng) -> list[dict]:
    """No-degradation eval: NGD-direct vs the reference estimator on the SAME
    frames (VAE↔real, Gen↔gen). Pose is always scored vs the real cameras."""
    z = np.load(npz_path, allow_pickle=True)
    npz = {k: z[k] for k in z.files}
    if "gt_c2w" not in npz:
        print(f"  [WARN] {os.path.basename(npz_path)} missing gt_c2w; skipping")
        return []
    gt_c2w = np.asarray(npz["gt_c2w"], dtype=np.float64)
    gt_K = np.asarray(npz["gt_K"], dtype=np.float64)
    H, W, cond = _scene_dims(npz, npz_path)
    if H is None:
        print(f"  [WARN] cannot infer image size for "
              f"{os.path.basename(npz_path)}; skipping")
        return []
    cond_num = cond if cond is not None else args.cond_num

    rgb_gt = _maybe_load_gt_rgb(npz_path, npz, args)
    rgb_gen = _get(npz, "gen_rgb")

    # Paired reference geometry (reference estimator on each frame set).
    refs: dict[str, dict] = {}
    if rgb_gt is not None:
        r = _estimator_ref(ref_name, ref_model, rgb_gt, gt_c2w, gt_K, H, W,
                           device, amp_dtype, args)
        if r is not None:
            refs["real"] = r
    if rgb_gen is not None:
        r = _estimator_ref(ref_name, ref_model, rgb_gen, gt_c2w, gt_K, H, W,
                           device, amp_dtype, args)
        if r is not None:
            refs["gen"] = r
    if not refs:
        print(f"  [WARN] no reference frames for "
              f"{os.path.basename(npz_path)}; skipping")
        return []

    rows: list[dict] = []
    up = ref_name.upper()
    # Reference-estimator baseline rows (pose vs real GT; depth/points = ref).
    if "real" in refs:
        rows.append(_pose_only_row(f"{up}(real)", refs["real"]["_sample"], gt_c2w))
    if "gen" in refs:
        rows.append(_pose_only_row(f"{up}(gen)", refs["gen"]["_sample"], gt_c2w))

    # Frame set the reference estimator uses to score GENERATED geometry.
    # 'real' (default) => non-circular: generated geometry is scored against
    # DA3 on the REAL target frames (same viewpoints), directly comparable to
    # the reconstruction block. 'gen' => legacy self-consistency reference
    # (DA3 on the model's own generated frames).
    gen_ref = getattr(args, "gen_ref_frame", "real")
    if gen_ref not in refs:  # fall back if the chosen ref frames are absent
        gen_ref = "gen" if "gen" in refs else "real"

    # NGD-direct sources paired to the matched-frame reference.
    ngd_specs = []
    if "vae_depth" in npz:
        s = _ngd_sample(npz["vae_depth"], npz.get("vae_ray"),
                        npz.get("vae_ray_conf"), H, W)
        if s is not None:
            ngd_specs.append(("NGD-VAE", s, "real"))
    if "gen_depth" in npz:
        s = _ngd_sample(npz["gen_depth"], npz.get("gen_ray"),
                        npz.get("gen_ray_conf"), H, W)
        if s is not None:
            ngd_specs.append(("NGD-Gen", s, gen_ref))

    # Optional cross-check estimators (natural estimator-disagreement floor).
    # Generated-frame estimators are scored against ``gen_ref`` too so the whole
    # generation block shares one non-circular reference.
    for cname, cmodel in cross_models.items():
        rgb_map = {"real": rgb_gt, "gen": rgb_gen}
        for frame, rgb in rgb_map.items():
            if rgb is None:
                continue
            score_frame = gen_ref if frame == "gen" else frame
            if score_frame not in refs:
                continue
            cs = _estimator_sample(cname, cmodel, rgb, gt_c2w, gt_K,
                                   device, amp_dtype, args)
            if cs is not None:
                ngd_specs.append((f"{cname.upper()}({frame})", cs, score_frame))

    for name, sample, frame in ngd_specs:
        ref = refs.get(frame)
        if ref is None:
            continue
        try:
            rows.append(_score_source(name, sample,
                                      {"depth": ref["depth"],
                                       "points": ref["points"],
                                       "c2w": ref["c2w"]},
                                      gt_c2w, cond_num, device, args, rng))
        except Exception as e:  # noqa: BLE001
            print(f"      [WARN] scoring {name} failed: {e}")
    return rows


def evaluate_gen3r_scene(gen3r_path: str, ref_name: str, ref_model,
                         args, device, amp_dtype, rng) -> list[dict]:
    """Score the Gen3R baseline exactly like NGD-Gen.

    Gen3R generates RGB + native geometry (depth/pose/points) from frame-0 + the
    real cameras. Its directly-decoded geometry is scored against DA3 run on its
    OWN generated frames (same-frame agreement, mirroring NGD-Gen↔DA3(gen)); pose
    is scored vs the real dataset cameras. With --gen3r-neutral we additionally
    score depth/points against DA3 on the REAL frames (a method-neutral
    reference that does not favour a DA3-native latent).
    """
    z = np.load(gen3r_path, allow_pickle=True)
    npz = {k: z[k] for k in z.files}
    for req in ("gen3r_depth", "gen3r_c2w", "gen3r_points", "gt_c2w"):
        if req not in npz:
            print(f"  [WARN] {os.path.basename(gen3r_path)} missing {req}; skip")
            return []
    gt_c2w = np.asarray(npz["gt_c2w"], dtype=np.float64)
    gt_K = np.asarray(npz["gt_K"], dtype=np.float64)
    H, W, cond = _scene_dims(npz, gen3r_path)
    cond_num = cond if cond is not None else args.cond_num

    sample = {
        "depth": np.asarray(npz["gen3r_depth"], dtype=np.float32),
        "c2w": np.asarray(npz["gen3r_c2w"], dtype=np.float32),
        "points": np.asarray(npz["gen3r_points"], dtype=np.float32),
    }
    rgb_gen3r = _get(npz, "gen3r_rgb")
    rgb_gt = _get(npz, "gt_rgb")
    up = ref_name.upper()
    rows: list[dict] = []

    # Same-frame reference: DA3 on Gen3R's own generated frames.
    ref_g = _estimator_ref(ref_name, ref_model, rgb_gen3r, gt_c2w, gt_K, H, W,
                           device, amp_dtype, args)
    if ref_g is not None:
        rows.append(_pose_only_row(f"{up}(gen3r)", ref_g["_sample"], gt_c2w))
        try:
            rows.append(_score_source(
                "Gen3R", sample,
                {"depth": ref_g["depth"], "points": ref_g["points"],
                 "c2w": ref_g["c2w"]},
                gt_c2w, cond_num, device, args, rng))
        except Exception as e:  # noqa: BLE001
            print(f"      [WARN] scoring Gen3R failed: {e}")

    # Method-neutral reference: DA3 on the REAL frames.
    if args.gen3r_neutral and rgb_gt is not None:
        ref_r = _estimator_ref(ref_name, ref_model, rgb_gt, gt_c2w, gt_K, H, W,
                               device, amp_dtype, args)
        if ref_r is not None:
            try:
                rows.append(_score_source(
                    "Gen3R*", sample,
                    {"depth": ref_r["depth"], "points": ref_r["points"],
                     "c2w": ref_r["c2w"]},
                    gt_c2w, cond_num, device, args, rng))
            except Exception as e:  # noqa: BLE001
                print(f"      [WARN] scoring Gen3R* failed: {e}")
    return rows


def _run_gen3r(args, device, amp_dtype, rng):
    """Gen3R-baseline driver: score <idx>_gen3r.npz vs the DA3 reference."""
    ref_name = args.ref_estimator
    ref_model = _load_ref_estimator(ref_name, args, device)
    npz_files = sorted(glob.glob(os.path.join(args.gen3r_dir, "*_gen3r.npz")))
    if args.limit:
        npz_files = npz_files[:args.limit]
    print(f"  gen3r-dir:       {args.gen3r_dir}")
    print(f"  gen3r scenes:    {len(npz_files)}")
    print(f"  ref-estimator:   {ref_name}  (neutral row: {args.gen3r_neutral})")
    print("=" * 70)
    if not npz_files:
        print("No <idx>_gen3r.npz found.")
        return

    per_scene_rows: list[dict] = []
    for i, npz_path in enumerate(npz_files):
        print(f"[{i + 1}/{len(npz_files)}] {os.path.basename(npz_path)}")
        try:
            rows = evaluate_gen3r_scene(npz_path, ref_name, ref_model,
                                        args, device, amp_dtype, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] scene failed: {e}")
            continue
        for r in rows:
            r["scene"] = os.path.basename(npz_path)[:-len("_gen3r.npz")]
            print(f"    {r['source']:<12} "
                  f"AbsRel={r['abs_rel']:.4f} δ1={r['delta1']:.4f} | "
                  f"ATE={r['ate']:.4f} RPEt={r['rpe_trans']:.4f} "
                  f"RPEr={r['rpe_rot']:.3f}° | "
                  f"Cham↓pose={r['chamfer']:.4f} Cham↓pc={r.get('chamfer_pc', float('nan')):.4f}")
        per_scene_rows.extend(rows)

    if not per_scene_rows:
        print("\nNo rows scored; nothing to report.")
        return
    summary = _aggregate(per_scene_rows)
    _print_table(summary)
    out = {
        "config": {
            "gen3r_dir": args.gen3r_dir,
            "mode": "gen3r-baseline",
            "ref_estimator": ref_name,
            "gen3r_neutral": args.gen3r_neutral,
            "da3_process_res": args.da3_process_res,
            "da3_ckpt": args.da3_ckpt,
            "vggt_ckpt": args.vggt_ckpt,
            "pi3_ckpt": args.pi3_ckpt,
            "pc_stride": args.pc_stride,
        },
        "summary": summary,
        "per_scene": per_scene_rows,
    }
    # Save alongside the gen3r npzs.
    args.geom_dir = args.gen3r_dir
    _save_outputs(out, summary, per_scene_rows, args)


def _build_gt_reference(npz: dict, args, estimators: dict,
                        rgb_gt: np.ndarray | None,
                        gt_c2w_real: np.ndarray, gt_K: np.ndarray,
                        H: int, W: int, device, amp_dtype) -> dict | None:
    """Build an internally-consistent GT geometry reference (depth+points+c2w).

    The (depth, cameras) pair is kept consistent so back-projected world points
    are geometrically valid:
      * ``model``    : NGD DPT depth + poses from its own ray head (real frames).
      * ``da3``/``vggt``: estimator depth + its recovered cameras (GT frames).
      * ``npz``      : dataset depth ``gt_depth_real`` + real cameras.

    Depth is returned on the (H,W) grid. ``points``/``c2w`` live in the
    reference frame (used only after per-source Sim3 alignment).
    """
    src = args.gt_depth_source

    if src == "model":
        d = npz.get("gtmodel_depth")
        ray = npz.get("gtmodel_ray")
        if d is None or ray is None:
            return None
        s = _ngd_sample(np.asarray(d, np.float32), np.asarray(ray, np.float32),
                        _get(npz, "gtmodel_ray_conf"), H, W)
        if s is None:
            return None
        return {"depth": _resize_depth(s["depth"], (H, W)),
                "points": s["points"], "c2w": s["c2w"]}

    if src == "npz":
        d = npz.get("gt_depth_real")
        if d is None:
            return None
        depth = _resize_depth(np.asarray(d, np.float32), (H, W))
        pts = _depth_to_world_points(depth, gt_c2w_real.astype(np.float32),
                                     np.asarray(gt_K, np.float32))
        return {"depth": depth, "points": pts,
                "c2w": gt_c2w_real.astype(np.float32)}

    # da3 / vggt on real frames
    if rgb_gt is None:
        return None
    model = estimators.get(src)
    if model is None:
        return None
    s = _estimator_sample(src, model, rgb_gt, gt_c2w_real, gt_K,
                          device, amp_dtype, args)
    if s is None:
        return None
    return {"depth": _resize_depth(s["depth"], (H, W)),
            "points": s["points"], "c2w": s["c2w"]}


def _get(npz: dict, key: str):
    v = npz.get(key)
    return None if v is None else np.asarray(v)


def evaluate_scene(npz_path: str, row_estimators: dict, all_estimators: dict,
                   args, device, amp_dtype, rng) -> list[dict]:
    z = np.load(npz_path, allow_pickle=True)
    npz = {k: z[k] for k in z.files}

    if "gt_c2w" not in npz:
        print(f"  [WARN] {os.path.basename(npz_path)} missing gt_c2w; skipping")
        return []
    gt_c2w = np.asarray(npz["gt_c2w"], dtype=np.float64)
    gt_K = np.asarray(npz["gt_K"], dtype=np.float64)
    if "image_size" in npz:
        H, W = int(npz["image_size"][0]), int(npz["image_size"][1])
    else:
        # Fall back to any available depth map resolution.
        H = W = None
        for k in ("gtmodel_depth", "gen_depth", "vae_depth"):
            if k in npz:
                H, W = (int(x) for x in npz[k].shape[-2:])
                break
        if H is None:
            print(f"  [WARN] cannot infer image size for "
                  f"{os.path.basename(npz_path)}; skipping")
            return []
    cond_num = int(npz["cond_num"]) if "cond_num" in npz else args.cond_num

    rgb_gen = _get(npz, "gen_rgb")
    rgb_gt = _get(npz, "gt_rgb")

    # ── Consistent GT geometry reference (depth+points+cameras) ──
    gt_ref = _build_gt_reference(npz, args, all_estimators, rgb_gt,
                                 gt_c2w, gt_K, H, W, device, amp_dtype)
    if gt_ref is None:
        print(f"  [WARN] no GT reference (source={args.gt_depth_source}) for "
              f"{os.path.basename(npz_path)}; skipping scene "
              f"(need recon-mode dump or --gt-depth-source with GT RGB)")
        return []

    # ── Build every requested source ──
    sources: dict[str, dict] = {}

    # NGD direct sources (from npz, no model load).
    if "vae_depth" in npz:
        s = _ngd_sample(npz["vae_depth"], npz.get("vae_ray"),
                        npz.get("vae_ray_conf"), H, W)
        if s is not None:
            sources["NGD-VAE"] = s
    if "gen_depth" in npz:
        s = _ngd_sample(npz["gen_depth"], npz.get("gen_ray"),
                        npz.get("gen_ray_conf"), H, W)
        if s is not None:
            sources["NGD-Gen"] = s

    # External estimator sources (from RGB).
    rgb_map = {"gt": rgb_gt, "gen": rgb_gen}
    for est_name, model in row_estimators.items():
        for rgb_key in args.rgb_sources:
            rgb = rgb_map.get(rgb_key)
            if rgb is None:
                continue
            s = _estimator_sample(est_name, model, rgb, gt_c2w, gt_K,
                                  device, amp_dtype, args)
            if s is not None:
                sources[f"{est_name.upper()}({rgb_key})"] = s

    rows = []
    for name, sample in sources.items():
        try:
            rows.append(_score_source(name, sample, gt_ref, gt_c2w,
                                      cond_num, device, args, rng))
        except Exception as e:  # noqa: BLE001
            print(f"      [WARN] scoring {name} failed: {e}")
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare NGD's direct depth/pose/point-map against "
                    "DA3/VGGT reconstructions (no reprojection/MEt3R).")
    p.add_argument("--geom-dir", default=None,
                   help="Directory with <idx>_geom.npz (e.g. "
                        "<eval-output>/<dataset>). Not required with --gen3r-dir.")
    p.add_argument("--estimators", default="da3,vggt",
                   help="Comma list of external estimators to compare against "
                        "(da3, vggt). Empty = NGD-direct only.")
    p.add_argument("--rgb-sources", default="gen,gt",
                   help="Which RGB the estimators reconstruct from: 'gen' "
                        "(generated frames), 'gt' (real frames), or both.")
    p.add_argument("--gt-depth-source", default="model",
                   choices=["model", "da3", "vggt", "npz"],
                   help="Pseudo-GT depth reference. 'model' = NGD DPT on real "
                        "features (default, neutral); 'da3'/'vggt' = estimator "
                        "on GT RGB (needs gt cameras); 'npz' = gt_depth_real.")
    p.add_argument("--degradation", action="store_true",
                   help="No-degradation mode: pair each NGD source with the "
                        "reference estimator run on the SAME frames "
                        "(NGD-VAE↔REF(real), NGD-Gen↔REF(gen)). Pose is scored "
                        "vs the real dataset cameras for every row and a "
                        "Δ(NGD−REF) readout is printed. Overrides "
                        "--gt-depth-source / --estimators / --rgb-sources.")
    p.add_argument("--gen3r-dir", default=None,
                   help="Directory with <idx>_gen3r.npz (from run_gen3r_re10k.py). "
                        "When set, score the Gen3R baseline against the SAME "
                        "same-frame DA3 reference + real cameras as NGD-Gen "
                        "(and, with --gen3r-neutral, also vs DA3 on real frames).")
    p.add_argument("--gen3r-neutral", action="store_true",
                   help="Also emit a Gen3R row scored against DA3 on the REAL "
                        "frames (a method-neutral depth/point reference).")
    p.add_argument("--ref-estimator", default="da3", choices=["da3", "vggt", "pi3"],
                   help="Estimator building the paired pseudo-GT in "
                        "--degradation mode (default da3 = NGD's own backbone, "
                        "so parity means 'no geometry lost').")
    p.add_argument("--cross-check", default="",
                   help="Comma list of extra estimators (e.g. 'vggt') to also "
                        "score against the same paired reference in "
                        "--degradation mode, as an estimator-disagreement floor.")
    p.add_argument("--gen-ref-frame", default="real", choices=["real", "gen"],
                   help="In --degradation mode, which frames the reference "
                        "estimator scores GENERATED geometry against. 'real' "
                        "(default) = DA3 on the REAL target frames "
                        "(non-circular, comparable to the reconstruction "
                        "block); 'gen' = DA3 on the model's own generated "
                        "frames (self-consistency only, the legacy behavior).")
    p.add_argument("--gt-rgb-from-geom-dir", default=None,
                   help="If the scored geom dump lacks gt_rgb (common for "
                        "MODE=generate dumps), load gt_rgb from the matching "
                        "<idx>_geom.npz under this directory (typically the "
                        "paired recon dump).")
    p.add_argument("--cond-num", type=int, default=1,
                   help="Fallback #source views if npz lacks cond_num "
                        "(target-view depth metrics use views >= cond_num).")
    p.add_argument("--pc-stride", type=int, default=4,
                   help="Pixel stride for Chamfer point subsampling.")
    p.add_argument("--vggt-ckpt", default="facebook/VGGT-1B")
    p.add_argument("--da3-ckpt", default="pretrained_models/da3_giant")
    p.add_argument("--pi3-ckpt", default=_PI3_CKPT_DEFAULT,
                   help="Local Pi3 checkpoint (.safetensors / .pt).")
    p.add_argument("--pi3-root", default=_PI3_ROOT_DEFAULT,
                   help="Pi3 repo root (added to sys.path for `import pi3`).")
    p.add_argument("--da3-process-res", type=int, default=504)
    p.add_argument("--da3-use-gt-cameras", action="store_true",
                   help="Feed GT cameras to DA3 (depth anchored to GT scale; "
                        "recommended when DA3 is a depth reference, not a pose "
                        "row).")
    p.add_argument("--da3-use-ray-pose", action="store_true")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate first N scenes.")
    p.add_argument("--output-name", default="direct_geometry.json")
    p.add_argument("--csv-output", default=None)
    return p.parse_args()


# ── Aggregation / reporting ──────────────────────────────────────────────────

_COLS = [
    ("abs_rel", "AbsRel↓", "{:.4f}"), ("rmse", "RMSE↓", "{:.4f}"),
    ("delta1", "δ1↑", "{:.4f}"),
    ("ate", "ATE↓", "{:.4f}"), ("rpe_trans", "RPEt↓", "{:.4f}"),
    ("rpe_rot", "RPEr↓", "{:.3f}"),
    ("chamfer", "Cham↓pose", "{:.4f}"), ("pmap_l1", "PMap↓pose", "{:.4f}"),
    ("chamfer_pc", "Cham↓pc", "{:.4f}"), ("pmap_l1_pc", "PMap↓pc", "{:.4f}"),
]


def _aggregate(per_scene_rows: list[dict]) -> dict[str, dict]:
    """Group per-scene rows by source and average each metric."""
    by_source: dict[str, list[dict]] = {}
    for r in per_scene_rows:
        by_source.setdefault(r["source"], []).append(r)
    metric_keys = [k for k in per_scene_rows[0] if k not in ("source", "scene")] \
        if per_scene_rows else []
    summary = {}
    for src, rows in by_source.items():
        summary[src] = {k: _finite_mean([r.get(k) for r in rows])
                        for k in metric_keys}
        summary[src]["num_scenes"] = len(rows)
    return summary


def _ordered_sources(summary: dict) -> list[str]:
    present = list(summary.keys())
    ordered = [s for s in ROW_ORDER if s in summary]
    ordered += [s for s in present if s not in ordered]
    return ordered


def _print_table(summary: dict):
    header = "  " + f"{'source':<12}" + "  ".join(f"{lbl:>9}" for _, lbl, _ in _COLS)
    print("\n" + "=" * len(header))
    print("Direct-geometry comparison (Sim3-aligned; no reproj/MEt3R)")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src in _ordered_sources(summary):
        row = summary[src]
        cells = []
        for key, _, fmt in _COLS:
            v = row.get(key)
            cells.append(fmt.format(v) if v is not None and np.isfinite(v) else "n/a")
        print("  " + f"{src:<12}" + "  ".join(f"{c:>9}" for c in cells))


_POSE_KEYS = [("ate", "ATE"), ("rpe_trans", "RPEt"), ("rpe_rot", "RPEr")]


def _print_degradation_delta(summary: dict, ref_name: str,
                             gen_ref_frame: str = "real"):
    """Print Δ(NGD − REF) on the real-GT-anchored pose metrics.

    Depth/points have no REF counterpart (REF *is* the reference), so the
    'no-degradation' evidence there is simply the small NGD residual in the
    table above; pose is the one axis with a true side-by-side baseline.
    """
    up = ref_name.upper()
    delta_pairs = [("NGD-VAE", "real"), ("NGD-Gen", gen_ref_frame)]
    lines = []
    for ngd_src, frame in delta_pairs:
        ref_src = f"{up}({frame})"
        if ngd_src not in summary or ref_src not in summary:
            continue
        cells = []
        for key, lbl in _POSE_KEYS:
            n, r = summary[ngd_src].get(key), summary[ref_src].get(key)
            if n is None or r is None or not (np.isfinite(n) and np.isfinite(r)):
                cells.append(f"{lbl}=n/a")
            else:
                d = n - r
                cells.append(f"{lbl} Δ={d:+.4f} ({n:.4f} vs {r:.4f})")
        lines.append(f"  {ngd_src:<9} − {ref_src:<9}  " + "  ".join(cells))
    if lines:
        print("\nΔ pose (NGD − reference; ≤0 ⇒ no degradation; vs real cameras)")
        print("  " + "-" * 90)
        for ln in lines:
            print(ln)


def _write_markdown(summary: dict, path: str):
    headers = ["source"] + [lbl for _, lbl, _ in _COLS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for src in _ordered_sources(summary):
        row = summary[src]
        cells = [src]
        for key, _, fmt in _COLS:
            v = row.get(key)
            cells.append(fmt.format(v) if v is not None and np.isfinite(v) else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _save_outputs(out: dict, summary: dict, per_scene_rows: list[dict], args):
    out_path = os.path.join(args.geom_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    md_path = os.path.splitext(out_path)[0] + ".md"
    _write_markdown(summary, md_path)
    print(f"Saved table: {md_path}")

    csv_path = args.csv_output or (os.path.splitext(out_path)[0] + ".csv")
    fieldnames = ["scene", "source"] + [k for k in per_scene_rows[0]
                                        if k not in ("scene", "source")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_scene_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved CSV: {csv_path}")


def _load_ref_estimator(name: str, args, device: torch.device):
    if name == "vggt":
        return _load_vggt(args.vggt_ckpt, device)
    if name == "pi3":
        return _load_pi3(args.pi3_ckpt, device, pi3_root=args.pi3_root)
    return _load_da3(args.da3_ckpt, device)


def _run_degradation(args, npz_files, cross_names, device, amp_dtype, rng):
    """No-degradation driver: NGD-direct vs reference estimator on same frames."""
    ref_name = args.ref_estimator
    ref_model = _load_ref_estimator(ref_name, args, device)
    cross_models = {n: (ref_model if n == ref_name else _load_ref_estimator(n, args, device))
                    for n in cross_names if n != ref_name}

    per_scene_rows: list[dict] = []
    for i, npz_path in enumerate(npz_files):
        print(f"[{i + 1}/{len(npz_files)}] {os.path.basename(npz_path)}")
        try:
            rows = evaluate_scene_degradation(
                npz_path, ref_name, ref_model, cross_models,
                args, device, amp_dtype, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] scene failed: {e}")
            continue
        for r in rows:
            r["scene"] = os.path.basename(npz_path)[:-len("_geom.npz")]
            print(f"    {r['source']:<11} "
                  f"AbsRel={r['abs_rel']:.4f} δ1={r['delta1']:.4f} | "
                  f"ATE={r['ate']:.4f} RPEt={r['rpe_trans']:.4f} "
                  f"RPEr={r['rpe_rot']:.3f}° | "
                  f"Cham↓pose={r['chamfer']:.4f} Cham↓pc={r.get('chamfer_pc', float('nan')):.4f}")
        per_scene_rows.extend(rows)

    if not per_scene_rows:
        print("\nNo rows scored; nothing to report.")
        return

    summary = _aggregate(per_scene_rows)
    _print_table(summary)
    _print_degradation_delta(summary, ref_name, args.gen_ref_frame)

    out = {
        "config": {
            "geom_dir": args.geom_dir,
            "mode": "no-degradation",
            "ref_estimator": ref_name,
            "gen_ref_frame": args.gen_ref_frame,
            "gt_rgb_from_geom_dir": args.gt_rgb_from_geom_dir,
            "cross_check": list(cross_models.keys()),
            "da3_process_res": args.da3_process_res,
            "da3_ckpt": args.da3_ckpt,
            "vggt_ckpt": args.vggt_ckpt,
            "pi3_ckpt": args.pi3_ckpt,
            "pc_stride": args.pc_stride,
        },
        "summary": summary,
        "per_scene": per_scene_rows,
    }
    _save_outputs(out, summary, per_scene_rows, args)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    rng = np.random.default_rng(args.seed)

    args.rgb_sources = [s.strip() for s in args.rgb_sources.split(",") if s.strip()]
    est_names = [s.strip().lower() for s in args.estimators.split(",") if s.strip()]

    # Gen3R-baseline mode: score <idx>_gen3r.npz and exit.
    if args.gen3r_dir:
        if args.output_name == "direct_geometry.json":
            args.output_name = "gen3r_baseline.json"
        print("=" * 70)
        print("Gen3R Baseline Evaluation (vs same-frame DA3 + real cameras)")
        _run_gen3r(args, device, amp_dtype, rng)
        return

    if not args.geom_dir:
        raise SystemExit("--geom-dir is required unless --gen3r-dir is set.")

    npz_files = sorted(glob.glob(os.path.join(args.geom_dir, "*_geom.npz")))
    if args.limit:
        npz_files = npz_files[:args.limit]

    cross_names = [s.strip().lower() for s in args.cross_check.split(",")
                   if s.strip()]

    print("=" * 70)
    print("NGD Direct-Geometry Evaluation")
    print(f"  geom-dir:        {args.geom_dir}")
    print(f"  scenes:          {len(npz_files)}")
    if args.degradation:
        print(f"  mode:            no-degradation (paired same-frame reference)")
        print(f"  ref-estimator:   {args.ref_estimator}")
        print(f"  gen-ref-frame:   {args.gen_ref_frame}"
              f"{'  (non-circular)' if args.gen_ref_frame == 'real' else '  (self-consistency)'}")
        print(f"  cross-check:     {cross_names or '(none)'}")
    else:
        print(f"  estimators:      {est_names or '(none)'}")
        print(f"  rgb-sources:     {args.rgb_sources}")
        print(f"  gt-depth-source: {args.gt_depth_source}")
    print("=" * 70)
    if not npz_files:
        print("No <idx>_geom.npz found. Run eval_generation.py "
              "with --dump-geometry first.")
        return

    if args.degradation:
        _run_degradation(args, npz_files, cross_names, device, amp_dtype, rng)
        return

    # Load estimators once (heavy). Also load whichever the GT depth needs.
    estimators: dict[str, object] = {}
    needed = set(est_names)
    if args.gt_depth_source in ("da3", "vggt"):
        needed.add(args.gt_depth_source)
    for name in needed:
        if name == "vggt":
            estimators["vggt"] = _load_vggt(args.vggt_ckpt, device)
        elif name == "da3":
            estimators["da3"] = _load_da3(args.da3_ckpt, device)
        else:
            print(f"  [WARN] unknown estimator '{name}', ignoring")
    # Estimators actually used as comparison rows (GT-depth estimator may be
    # loaded but not emitted as a row unless also in --estimators).
    row_estimators = {n: estimators[n] for n in est_names if n in estimators}

    per_scene_rows: list[dict] = []
    for i, npz_path in enumerate(npz_files):
        print(f"[{i + 1}/{len(npz_files)}] {os.path.basename(npz_path)}")
        try:
            rows = evaluate_scene(npz_path, row_estimators, estimators, args,
                                  device, amp_dtype, rng)
        except Exception as e:  # noqa: BLE001
            print(f"  [WARN] scene failed: {e}")
            continue
        for r in rows:
            r["scene"] = os.path.basename(npz_path)[:-len("_geom.npz")]
            print(f"    {r['source']:<11} "
                  f"AbsRel={r['abs_rel']:.4f} RMSE={r['rmse']:.4f} "
                  f"δ1={r['delta1']:.4f} | ATE={r['ate']:.4f} "
                  f"RPEt={r['rpe_trans']:.4f} RPEr={r['rpe_rot']:.3f}° | "
                  f"Cham↓pose={r['chamfer']:.4f} Cham↓pc={r.get('chamfer_pc', float('nan')):.4f}")
        per_scene_rows.extend(rows)

    if not per_scene_rows:
        print("\nNo rows scored; nothing to report.")
        return

    summary = _aggregate(per_scene_rows)
    _print_table(summary)

    # ── Save JSON / CSV / Markdown ──
    out = {
        "config": {
            "geom_dir": args.geom_dir,
            "estimators": est_names,
            "rgb_sources": args.rgb_sources,
            "gt_depth_source": args.gt_depth_source,
            "cond_num_fallback": args.cond_num,
            "pc_stride": args.pc_stride,
            "da3_ckpt": args.da3_ckpt,
            "da3_process_res": args.da3_process_res,
            "da3_use_gt_cameras": args.da3_use_gt_cameras,
            "vggt_ckpt": args.vggt_ckpt,
        },
        "summary": summary,
        "per_scene": per_scene_rows,
    }
    out_path = os.path.join(args.geom_dir, args.output_name)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    md_path = os.path.splitext(out_path)[0] + ".md"
    _write_markdown(summary, md_path)
    print(f"Saved table: {md_path}")

    csv_path = args.csv_output or (os.path.splitext(out_path)[0] + ".csv")
    fieldnames = ["scene", "source"] + [k for k in per_scene_rows[0]
                                        if k not in ("scene", "source")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_scene_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
