"""Evaluate GAE flow generation and reconstruction.

Use ``scripts/demo/generate.py`` for one-image + prompt generation. This lower-level
entry point supports dataset manifests, long rollouts, video metrics, direct
geometry dumps, and PLY export. The default ``euler_v3`` sampler is the sampler
used by the released GAE checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts" / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("TMPDIR", "/tmp")

# Use local-ssd for HuggingFace cache (fast NVMe) if available;
# copies from /tmp or downloads on first run, then instant on subsequent runs.
_LOCAL_HF_CACHE = "/local-ssd/hf_cache"
if os.path.isdir("/local-ssd"):
    os.makedirs(_LOCAL_HF_CACHE, exist_ok=True)
    # If HF models already cached in /tmp, migrate to local-ssd
    _tmp_hf = "/tmp/xdg-cache/huggingface/hub"
    if os.path.isdir(_tmp_hf) and not os.listdir(_LOCAL_HF_CACHE):
        import shutil
        for d in os.listdir(_tmp_hf):
            src = os.path.join(_tmp_hf, d)
            dst = os.path.join(_LOCAL_HF_CACHE, d)
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst, symlinks=True)
    os.environ["HF_HOME"] = "/local-ssd/hf_home"
    os.environ["HUGGINGFACE_HUB_CACHE"] = _LOCAL_HF_CACHE
else:
    os.environ.setdefault("HF_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/xdg-cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Insert scripts/ before ROOT so sibling imports (eval_data, …) resolve
# even when PYTHONPATH contains another top-level ``scripts`` package that would
# shadow ``ROOT/scripts`` for ``from scripts.*`` imports.
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

_HF_TO_LOCAL_TEXT = {
    "Qwen/Qwen3-0.6B": "pretrained_models/qwen3_0.6b",
}


def _resolve_text_model_path(model_name: str) -> str:
    """Resolve known HF text encoders to checked-in local model dirs when present."""
    candidates = [os.environ.get("QWEN3_LOCAL_PATH", ""), model_name]
    mapped = _HF_TO_LOCAL_TEXT.get(model_name)
    if mapped:
        candidates.append(mapped)
    for raw in candidates:
        if not raw:
            continue
        probes = [raw] if os.path.isabs(raw) else [
            os.path.join(ROOT, raw),
            os.path.join(os.getcwd(), raw),
        ]
        for path in probes:
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
                return path
    return model_name


def _cache_pretrained_dir_to_local_ssd(path: str, tag: str, timeout_s: float = 1800.0) -> str:
    """Copy a local pretrained directory to node-local SSD once and reuse it.

    Loading large safetensors from a network filesystem makes every concurrent
    eval worker re-read roughly 1.5 GB. A completion marker and exclusive lock
    let one worker populate a local SSD copy while the others wait for it.
    """
    if not (path and os.path.isdir(path)):
        return path

    cache_root = os.path.join(_eval_cache_base(), "_eval_cache")
    if os.path.abspath(path).startswith(os.path.abspath(cache_root) + os.sep):
        return path

    source_files: list[tuple[str, str]] = []
    for root, _, files in os.walk(path):
        for name in files:
            src = os.path.join(root, name)
            source_files.append((src, os.path.relpath(src, path)))
    if not source_files:
        return path

    digest = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
    local_dir = os.path.join(cache_root, f"{tag}_{digest}")
    done_path = os.path.join(local_dir, ".complete")
    lock_path = f"{local_dir}.lock"

    def _is_complete() -> bool:
        if not os.path.isfile(done_path):
            return False
        try:
            return all(
                os.path.isfile(os.path.join(local_dir, rel))
                and os.path.getsize(os.path.join(local_dir, rel)) == os.path.getsize(src)
                for src, rel in source_files
            )
        except OSError:
            return False

    if _is_complete():
        print(f"  [cache] Using {local_dir}")
        return local_dir

    owns_lock = False
    try:
        os.makedirs(cache_root, exist_ok=True)
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_fd)
            owns_lock = True
        except FileExistsError:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if _is_complete():
                    print(f"  [cache] Using {local_dir}")
                    return local_dir
                time.sleep(1.0)
            print(f"  [cache] timed out waiting for {tag}; using {path}")
            return path

        os.makedirs(local_dir, exist_ok=True)
        print(f"  [cache] {path} -> {local_dir} ...", flush=True)
        for src, rel in source_files:
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                valid = os.path.isfile(dst) and os.path.getsize(dst) == os.path.getsize(src)
            except OSError:
                valid = False
            if valid:
                continue
            tmp = f"{dst}.tmp.{os.getpid()}"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        with open(done_path, "w", encoding="utf-8") as handle:
            handle.write(os.path.abspath(path))
        print("  [cache] Qwen3 local copy ready", flush=True)
        return local_dir
    except OSError as exc:
        print(f"  [cache] {tag} local-ssd copy failed ({exc}); using {path}", flush=True)
        return path
    finally:
        if owns_lock:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass


def _to_container_or_empty(node) -> dict:
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True) or {}
    if isinstance(node, dict):
        return dict(node)
    return {}

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, AutoencoderKLWan
from einops import rearrange
from PIL import Image
from omegaconf import OmegaConf

from stage1.da3 import DA3Backbone
from stage1.gae_codec import GAECodec, DA3_LEVEL_DIM
from stage1.da3_direct_codec import DA3DirectCodec
from stage2.transport.flow import (
    convert_x_to_v,
    sample_logit_normal_t,
    shift_from_latent_dim,
)
from stage2.models.camera import compute_plucker_6d_per_token
from utils.model_utils import instantiate_from_config

from eval_data import (
    DATASET_ROOTS,
    DATASET_DEFAULT_INTERVAL,
    collect_scenes,
    collect_scenes_re10k,
    load_scenes_from_manifest,
    load_image_and_camera,
    clear_scannetpp_frame_cache,
    prefetch_scannetpp_frames,
    tensor_to_numpy_img,
    depth_to_numpy_img,
    compute_metrics,
    visualize_trajectory,
    raw_no_cls_to_dpt_input,
    raw_to_dpt_input,
    decode_to_depth,
)

from utils.train_runtime import normalise_c2w_for_batch, intrinsic_to_K

from eval_reconstruction import (
    decode_dpt,
    _ray_to_numpy,
    _rayconf_to_numpy,
)
from utils.camera_from_ray import recover_poses


# ── Latent normalization (per-channel scaling OR full-covariance whitening) ──
# When stats file contains "whiten" / "unwhiten" matrices (Σ^(-1/2) / Σ^(+1/2)),
# diffusion operates in the whitened ≈ N(0, I) space — see docs/notes/
# 2026-05-26-raev2-analysis.md ("Final verdict"). Falls back to per-channel.

def _norm_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                 whiten: torch.Tensor | None) -> torch.Tensor:
    if whiten is None:
        return (z - mean) / std
    # einsum requires uniform dtype. Whitening (Σ^(-1/2)) is precision-sensitive
    # for anisotropic latents (cond≈100), so always run the matmul in fp32 and
    # cast the result back to z's dtype. This is also what makes the helper
    # safe to call outside an autocast context.
    out_dtype = z.dtype
    out = torch.einsum("ij,bjhw->bihw", whiten.float(),
                       (z.float() - mean.float()))
    return out.to(out_dtype)


def _denorm_latent(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                   unwhiten: torch.Tensor | None) -> torch.Tensor:
    if unwhiten is None:
        return y * std + mean
    # See _norm_latent: fp32 matmul + cast back to caller dtype.
    out_dtype = y.dtype
    out = (torch.einsum("ij,bjhw->bihw", unwhiten.float(), y.float())
           + mean.float())
    return out.to(out_dtype)


def compute_ssim(img1, img2):
    """Compute SSIM between two (B, C, H, W) tensors in [0,1]."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ks, sigma = 11, 1.5
    coords = torch.arange(ks, dtype=img1.dtype, device=img1.device) - ks // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    pad = ks // 2
    vals = []
    for c in range(img1.shape[1]):
        g1, p1 = img1[:, c:c+1], img2[:, c:c+1]
        mu_g = F.conv2d(g1, window, padding=pad)
        mu_p = F.conv2d(p1, window, padding=pad)
        sig_gg = F.conv2d(g1 * g1, window, padding=pad) - mu_g * mu_g
        sig_pp = F.conv2d(p1 * p1, window, padding=pad) - mu_p * mu_p
        sig_gp = F.conv2d(g1 * p1, window, padding=pad) - mu_g * mu_p
        ssim_map = ((2 * mu_g * mu_p + C1) * (2 * sig_gp + C2)) / \
                   ((mu_g**2 + mu_p**2 + C1) * (sig_gg + sig_pp + C2))
        vals.append(ssim_map.mean().item())
    return float(np.mean(vals))


def _fast_load(path, **kwargs):
    """torch.load with local caching for S3-FUSE/EFS paths.

    The cache key is hashed from the FULL source path (not just the basename):
    different runs all name their checkpoints ``0008000.pt`` etc., so a
    basename-only key silently served a stale checkpoint from another run
    (e.g. a d64 ckpt loaded against a d128 config → state_dict size mismatch).
    A size check additionally re-copies if the cached file is incomplete/stale.
    """
    import hashlib
    import subprocess
    path = os.path.abspath(os.fspath(path))
    if path.startswith("/threed-code/") or path.startswith("/efs/"):
        cache_dir = os.path.join(_eval_cache_base(), "_eval_cache")
        os.makedirs(cache_dir, exist_ok=True)
        src = path
        h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
        local = os.path.join(cache_dir, f"{h}_{os.path.basename(path)}")
        try:
            src_stat = os.stat(src)
            local_stat = os.stat(local)
            # Size alone is insufficient: checkpoints are occasionally replaced
            # in-place, and two torch archives can have the same byte length.
            # A normal ``cp`` gives the destination a current mtime, so an older
            # or equal source mtime means the cached copy still covers the source.
            fresh = (
                local_stat.st_size == src_stat.st_size
                and local_stat.st_mtime_ns >= src_stat.st_mtime_ns
            )
        except OSError:
            fresh = False
        if not fresh:
            print(f"  [cache] {src} -> {local} ...", end=" ", flush=True)
            # atomic: cp to a pid-unique tmp then rename, so concurrent shards
            # on the same node never read a half-written file.
            _tmp = f"{local}.tmp.{os.getpid()}"
            subprocess.run(["cp", src, _tmp], check=True)
            os.replace(_tmp, local)
            print("done (%.1f GB)" % (os.path.getsize(local) / 1e9))
        else:
            print(f"  [cache] Using {local}")
        return torch.load(local, **kwargs)
    return torch.load(path, **kwargs)


def _unwrap_codec_state(ckpt):
    """Release ckpts use ``codec``; research dumps use ``ema_vae`` / ``vae``."""
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("ema_codec", "codec", "ema_vae", "vae", "model"):
        inner = ckpt.get(key)
        if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
            return inner
    return ckpt


def _eval_cache_base() -> str:
    """Local scratch root for weight caches (callers append '/_eval_cache').

    Falls back off ``/local-ssd`` so the distributed cache job also works in
    environments that lack ``/local-ssd``: reading a ~1.6 GB safetensors
    directly over S3-FUSE via mmap otherwise fails with
    'incomplete metadata, file not fully covered'. Override with EVAL_CACHE_ROOT.
    """
    root = os.environ.get("EVAL_CACHE_ROOT")
    if not root:
        root = "/local-ssd" if os.path.isdir("/local-ssd") else "/tmp"
    return root


_HF_TO_LOCAL_DA3 = {
    "depth-anything/DA3-LARGE-1.1": "pretrained_models/da3_large",
    "depth-anything/DA3-Large": "pretrained_models/da3_large",
    "depth-anything/DA3-Base": "pretrained_models/da3",
}


def _resolve_da3_encoder_path(pretrained_path: str, da3_weights_path: str | None = None) -> str:
    """Map HF repo ids to local pretrained_models/ when available (offline eval)."""
    candidates: list[str] = []
    if pretrained_path:
        candidates.append(pretrained_path)
    mapped = _HF_TO_LOCAL_DA3.get(pretrained_path)
    if mapped:
        candidates.append(mapped)
    if da3_weights_path:
        parent = os.path.dirname(da3_weights_path)
        if parent:
            candidates.append(parent)
    for raw in candidates:
        p = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
            if raw != pretrained_path:
                print(f"  [da3] encoder_pretrained_path: {pretrained_path!r} -> {p}")
            return p
    return pretrained_path


def _cache_da3_dir(path: str) -> str:
    """Mirror a local DA3 model directory to /local-ssd for fast from_pretrained.

    ``DepthAnything3.from_pretrained`` reads ``model.safetensors`` (~1.6 GB)
    from the directory; on a network filesystem that read is the slow
    "Loading weights from local directory" step. Copy once to local NVMe and reuse.

    A big weight file already pre-cached by ``_cache_file`` (as
    ``<dirname>_<filename>``) is symlinked rather than re-copied.
    """
    if not (path and os.path.isdir(path)):
        return path  # HF id / missing → leave as-is for offline fallback
    import shutil

    cache_root = os.path.join(_eval_cache_base(), "_eval_cache")
    if path.startswith(cache_root):
        return path
    name = os.path.basename(path.rstrip("/"))
    local = os.path.join(cache_root, name)
    done = local + ".done"
    if os.path.isdir(local) and os.path.isfile(done):
        print(f"  [cache] Using {local}")
        return local

    os.makedirs(local, exist_ok=True)
    for fn in sorted(os.listdir(path)):
        src = os.path.join(path, fn)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(local, fn)
        if os.path.islink(dst) or os.path.isfile(dst):
            continue
        prior = os.path.join(cache_root, f"{name}_{fn}")  # _cache_file naming
        try:
            reuse = os.path.isfile(prior) and os.path.getsize(prior) == os.path.getsize(src)
        except OSError:
            reuse = False
        if reuse:
            os.symlink(prior, dst)
        else:
            print(f"  [cache] {src} -> {dst} ...", end=" ", flush=True)
            _tmp = f"{dst}.tmp.{os.getpid()}"
            shutil.copy2(src, _tmp)
            os.replace(_tmp, dst)
            print("done (%.0f MB)" % (os.path.getsize(dst) / 1e6))
    open(done, "w").close()
    print(f"  [cache] da3 encoder dir -> {local}")
    return local



def depth_to_point_map(depth, K):
    """Convert depth map to 3D point map in camera coordinates.

    Args:
        depth: (BV, H, W) depth values
        K:     (BV, 3, 3) camera intrinsic matrices
    Returns:
        pmap: (BV, H, W, 3) xyz in camera coordinates
    """
    import torch as _torch
    BV, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    v_coords, u_coords = _torch.meshgrid(
        _torch.arange(H, device=device, dtype=dtype),
        _torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = K[:, 0, 0].view(BV, 1, 1)
    fy = K[:, 1, 1].view(BV, 1, 1)
    cx = K[:, 0, 2].view(BV, 1, 1)
    cy = K[:, 1, 2].view(BV, 1, 1)
    x = (u_coords.unsqueeze(0) - cx) * depth / (fx + 1e-8)
    y = (v_coords.unsqueeze(0) - cy) * depth / (fy + 1e-8)
    return _torch.stack([x, y, depth], dim=-1)


def save_pointcloud_ply(path, xyz, rgb):
    """Save a colored point cloud to PLY.

    Args:
        path: output .ply file path
        xyz:  (N, 3) world-space coordinates, float
        rgb:  (N, 3) colors in [0, 255], uint8
    """
    import os as _os
    _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
    N = xyz.shape[0]
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(N):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")


def _save_geom_npz(path: str, arrays: dict) -> None:
    """Write geometry NPZ robustly (fuse/NFS-safe via local temp)."""
    import shutil
    import tempfile

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = "/local-ssd" if os.path.isdir("/local-ssd") else None
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=tmp_dir)
    os.close(fd)
    try:
        try:
            np.savez_compressed(tmp, **arrays)
        except OSError:
            np.savez(tmp, **arrays)
        try:
            os.replace(tmp, path)
        except OSError:
            # Cross-filesystem or fuse mounts may reject replace; copy then unlink.
            shutil.copyfile(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _umeyama_sim3(src: np.ndarray, tgt: np.ndarray):
    """Estimate similarity transform: tgt ≈ scale * (src @ R.T) + t.

    Args:
        src, tgt: (N, 3) corresponding points in row-vector convention.
    Returns:
        scale, R (3, 3), t (3,)
    """
    assert src.shape == tgt.shape and src.shape[1] == 3
    n = src.shape[0]
    if n < 3:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    src_c = src - mu_s
    tgt_c = tgt - mu_t
    cov = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (src_c * src_c).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_t - scale * (R @ mu_s)
    return scale, R, t


def _apply_sim3(xyz: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (xyz @ R.T)) + t


def _view_pointcloud_from_depth(
    depth_hw: np.ndarray,
    rgb_chw: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    *,
    stride: int = 4,
    max_depth: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject one view to world-space points (subsampled grid)."""
    H, W = depth_hw.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d = depth_hw[yy, xx].astype(np.float64)
    valid = (d > 1e-3) & (d < max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)

    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    z = d[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / (fx + 1e-8)
    y = (v - cy) * z / (fy + 1e-8)
    pts_cam = np.stack([x, y, z], axis=-1)
    R_w = c2w[:3, :3]
    t_w = c2w[:3, 3]
    pts_w = (pts_cam @ R_w.T) + t_w

    if rgb_chw.ndim == 3 and rgb_chw.shape[0] == 3:
        rgb_hwc = np.transpose(rgb_chw, (1, 2, 0))
    else:
        rgb_hwc = rgb_chw
    cols = (rgb_hwc[yy, xx][valid] * 255.0).clip(0, 255).astype(np.uint8)
    return pts_w, cols


def _subsample_xyz(pts: np.ndarray, max_pts: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] <= max_pts:
        return pts
    idx = rng.choice(pts.shape[0], max_pts, replace=False)
    return pts[idx]


def _bidirectional_chamfer(
    a: np.ndarray,
    b: np.ndarray,
    device: torch.device,
    *,
    block: int = 2048,
) -> float:
    """Symmetric Chamfer (mean of both NN directions), chunked for VRAM."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")

    def _run(dev: torch.device) -> float:
        ta = torch.from_numpy(a).float().to(dev)
        tb = torch.from_numpy(b).float().to(dev)

        def _nn_mean(src, tgt):
            acc = 0.0
            for i in range(0, src.shape[0], block):
                d = torch.cdist(src[i : i + block], tgt)
                acc += d.min(dim=1).values.sum().item()
            return acc / src.shape[0]

        return float(0.5 * (_nn_mean(ta, tb) + _nn_mean(tb, ta)))

    try:
        return _run(device)
    except RuntimeError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            return _run(torch.device("cpu"))
        raise


def _median_scale_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale both clouds so median ||p|| of the union is 1 (scene-invariant)."""
    if a.shape[0] == 0 and b.shape[0] == 0:
        return a, b
    all_pts = np.concatenate([p for p in (a, b) if p.shape[0] > 0], axis=0)
    norms = np.linalg.norm(all_pts, axis=-1)
    valid = norms > 1e-6
    if not np.any(valid):
        return a, b
    scale = float(np.median(norms[valid]))
    if scale < 1e-8:
        return a, b
    return a / scale, b / scale


def compute_ref_tgt_pc_gap(
    dpt_out,
    depth_tensor,
    rgb_imgs,
    H: int,
    W: int,
    device,
    *,
    cond_num: int,
    stride: int = 4,
    max_pts: int = 30000,
    rng: np.random.Generator | None = None,
) -> float:
    """Point-cloud discrepancy between ref views [:cond] and tgt views [cond:].

    Used by tab:ref_ablation. Decodes poses from DPT rays, unprojects depth,
    median-scales the scene, then reports bidirectional Chamfer.
    """
    if cond_num <= 0:
        return float("nan")
    n_views = int(depth_tensor.shape[0])
    if cond_num >= n_views:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    view_pcs = _recover_view_pointclouds(
        dpt_out, depth_tensor, rgb_imgs, H, W, device, stride=stride,
    )
    ref = np.concatenate([view_pcs[i][0] for i in range(cond_num)], axis=0)
    tgt = np.concatenate(
        [view_pcs[i][0] for i in range(cond_num, n_views)], axis=0,
    )
    ref = ref[np.isfinite(ref).all(axis=1)]
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    ref, tgt = _median_scale_pair(ref, tgt)
    ref = _subsample_xyz(ref, max_pts, rng)
    tgt = _subsample_xyz(tgt, max_pts, rng)
    return _bidirectional_chamfer(ref, tgt, device)


def _recover_view_pointclouds(
    dpt_out, depth_tensor, rgb_imgs, H, W, device, *,
    stride: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-view world point clouds from the decoded DA3 ray/depth pair.

    ``recover_poses`` is still run on the decoded ray to keep the official DA3
    camera-pose convention and validate the ray output. Point coordinates,
    however, are built directly from that same ray: its first three channels
    are world-frame directions and its last three channels are the per-view
    camera center. Mixing the ray translation scale with a separate pinhole
    back-projection from recovered ``K/c2w`` causes geoft point clouds to drift.
    """
    ray = dpt_out.get("ray")
    rc = dpt_out.get("ray_conf")
    if ray is None:
        raise RuntimeError("DPT output missing ray — cannot build point cloud")
    ray_np = _ray_to_numpy(ray)
    rc_np = _rayconf_to_numpy(rc)
    # Keep the official ray-pose fit in the evaluation path. The point map
    # below intentionally uses the ray's native origin/direction representation
    # so all geometry terms share one metric gauge.
    recover_poses(
        ray_np, rc_np, input_size=(H, W),
        return_per_view_intrinsics=True,
    )
    if ray_np.ndim != 4 or ray_np.shape[-1] != 6:
        raise ValueError(f"expected ray shape (V,H,W,6), got {ray_np.shape}")
    n_views, ray_h, ray_w = ray_np.shape[:3]

    d_t = depth_tensor.float()
    if d_t.ndim == 4:
        d_t = d_t.squeeze(1)
    if d_t.ndim != 3 or d_t.shape[0] != n_views:
        raise ValueError(
            f"depth/ray view mismatch: depth={tuple(d_t.shape)} ray={ray_np.shape}"
        )
    if d_t.shape[-2:] != (ray_h, ray_w):
        d_t = F.interpolate(
            d_t.unsqueeze(1), size=(ray_h, ray_w), mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    d_hw = d_t.clamp(min=1e-3).cpu().numpy()
    points = ray_np[..., 3:] + d_hw[..., None] * ray_np[..., :3]

    if rgb_imgs is None:
        rgb_np = np.zeros((n_views, 3, ray_h, ray_w), dtype=np.float32)
    else:
        rgb_t = rgb_imgs.float()
        if rgb_t.ndim != 4 or rgb_t.shape[0] != n_views:
            raise ValueError(
                f"rgb/ray view mismatch: rgb={tuple(rgb_t.shape)} ray={ray_np.shape}"
            )
        if rgb_t.shape[-2:] != (ray_h, ray_w):
            rgb_t = F.interpolate(
                rgb_t, size=(ray_h, ray_w), mode="bilinear", align_corners=False,
            )
        rgb_np = rgb_t.cpu().numpy()

    view_pcs: list[tuple[np.ndarray, np.ndarray]] = []
    for vi in range(n_views):
        xyz_v = points[vi][::stride, ::stride]
        depth_v = d_hw[vi][::stride, ::stride]
        valid = np.isfinite(xyz_v).all(axis=-1) & (depth_v > 1e-3) & (depth_v < 100.0)
        xyz = xyz_v[valid].astype(np.float64, copy=False)
        rgb_hwc = np.transpose(rgb_np[vi], (1, 2, 0))
        cols = (rgb_hwc[::stride, ::stride][valid] * 255.0).clip(0, 255).astype(np.uint8)
        view_pcs.append((xyz, cols))
    return view_pcs


def _merge_overlap_aligned_pointclouds(
    per_chunk_view_pcs: list[list[tuple[np.ndarray, np.ndarray]]],
    roll_cond_num: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Align chunk-local PCs via overlap views (Sim3) and merge.

    Chunk 0 defines the global frame. For chunk i>0, estimate Sim3 mapping
    overlap views [0:roll) onto the last ``roll`` global views, apply to
    the non-overlap views only, then append.
    """
    if not per_chunk_view_pcs:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), []

    global_views: list[tuple[np.ndarray, np.ndarray]] = []
    acc_xyz, acc_rgb = [], []
    residuals: list[float] = []

    for xyz, rgb in per_chunk_view_pcs[0]:
        global_views.append((xyz, rgb))
        acc_xyz.append(xyz)
        acc_rgb.append(rgb)

    for ci in range(1, len(per_chunk_view_pcs)):
        chunk_pcs = per_chunk_view_pcs[ci]
        R = roll_cond_num
        if len(chunk_pcs) < R:
            raise ValueError(f"chunk {ci} has {len(chunk_pcs)} views < roll {R}")
        if len(global_views) < R:
            raise ValueError("not enough global views for overlap alignment")

        src_pts, tgt_pts = [], []
        for r in range(R):
            s_xyz, _ = chunk_pcs[r]
            t_xyz, _ = global_views[-R + r]
            n = min(len(s_xyz), len(t_xyz))
            if n == 0:
                continue
            src_pts.append(s_xyz[:n])
            tgt_pts.append(t_xyz[:n])
        if not src_pts:
            scale, Rm, t = 1.0, np.eye(3), np.zeros(3)
        else:
            src_all = np.concatenate(src_pts, axis=0)
            tgt_all = np.concatenate(tgt_pts, axis=0)
            scale, Rm, t = _umeyama_sim3(src_all, tgt_all)
            pred = _apply_sim3(src_all, scale, Rm, t)
            residuals.append(float(np.linalg.norm(pred - tgt_all, axis=1).mean()))

        for vi in range(R, len(chunk_pcs)):
            xyz, rgb = chunk_pcs[vi]
            xyz_g = _apply_sim3(xyz, scale, Rm, t)
            global_views.append((xyz_g, rgb))
            acc_xyz.append(xyz_g)
            acc_rgb.append(rgb)

    xyz = np.concatenate(acc_xyz, axis=0)
    rgb = np.concatenate(acc_rgb, axis=0)
    max_pts = 5_000_000
    if xyz.shape[0] > max_pts:
        idx = np.random.choice(xyz.shape[0], max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb, residuals


def _smooth_latents_temporal(z: torch.Tensor, strength: float) -> torch.Tensor:
    """Symmetric 3-frame smoothing over latent time/view axis."""
    if strength <= 0 or z.shape[0] < 2:
        return z
    s = float(max(0.0, min(strength, 0.95)))
    out = z.clone()
    if z.shape[0] == 2:
        out[0] = (1.0 - s) * z[0] + s * z[1]
        out[1] = s * z[0] + (1.0 - s) * z[1]
        return out
    out[0] = (1.0 - s) * z[0] + s * z[1]
    out[-1] = s * z[-2] + (1.0 - s) * z[-1]
    out[1:-1] = (
        (0.5 * s) * z[:-2]
        + (1.0 - s) * z[1:-1]
        + (0.5 * s) * z[2:]
    )
    return out


def _inject_rgb_decoder_from_ckpt(
    vae_cfg: dict,
    vae_sd: dict,
    temporal_mode: str = "config",
) -> None:
    """Build/reconcile RGB decoder architecture with checkpoint keys."""
    block_idxs = [
        int(k.split(".")[2])
        for k in vae_sd
        if k.startswith("rgb_head.blocks.") and k.split(".")[2].isdigit()
    ]
    if not block_idxs:
        print("  [vae] ckpt has no rgb_head weights → RGB output disabled")
        return
    detected_depth = max(block_idxs) + 1
    injected = "rgb_decoder" not in vae_cfg
    if injected:
        vae_cfg["rgb_decoder"] = dict(
            hidden_dim=int(vae_cfg.get("rgb_decoder_hidden", 1024)),
            depth=detected_depth,
            num_heads=int(vae_cfg.get("rgb_decoder_heads", 16)),
            ffn_ratio=4.0,
            patch_size=14,
        )
    rgb_dec = vae_cfg["rgb_decoder"]

    # A TimeSformer-style "divided" head keeps time inside every block
    # (t_attn -> s_attn -> mlp) instead of interleaving separate
    # rgb_head.temporal_blocks.*, so the interleaved probe below sees nothing and
    # would rebuild it as a per-frame head. Detect it up front and pin the mode.
    _divided_blocks = any(
        k.startswith("rgb_head.blocks.") and ".t_attn." in k for k in vae_sd
    )
    if _divided_blocks:
        if temporal_mode == "off":
            raise RuntimeError(
                "ckpt has a divided space-time rgb_head but RGB_HEAD_TEMPORAL=off "
                "was requested; a per-frame decode of these weights is impossible.")
        _proj_in = vae_sd.get("rgb_head.proj_in.weight")
        if _proj_in is not None:
            rgb_dec["hidden_dim"] = int(_proj_in.shape[0])
        rgb_dec["depth"] = detected_depth
        rgb_dec["temporal"] = True
        rgb_dec["temporal_mode"] = "divided"
        rgb_dec.setdefault("temporal_num_heads", int(rgb_dec.get("num_heads", 16)))
        # Interleaved-only knobs; RGBHead ignores them in divided mode, but leaving
        # them in the dict makes the printed config misleading.
        for key in ("temporal_every", "temporal_max_views",
                    "temporal_ffn_ratio", "temporal_gated"):
            rgb_dec.pop(key, None)
        print(f"  [vae] divided space-time RGBHead detected "
              f"(depth={detected_depth}, hidden={rgb_dec.get('hidden_dim')}, "
              f"heads={rgb_dec.get('num_heads')})")
        if injected:
            print(f"  [vae] injecting rgb_decoder (auto-detected from ckpt)")
        return

    tblock_idxs = sorted({
        int(k.split(".")[2])
        for k in vae_sd
        if k.startswith("rgb_head.temporal_blocks.")
        and len(k.split(".")) > 2
        and k.split(".")[2].isdigit()
    })
    if tblock_idxs:
        temporal_every = (
            tblock_idxs[1] - tblock_idxs[0]
            if len(tblock_idxs) >= 2
            else max(1, detected_depth)
        )
        has_ffn = any(
            ".ffn." in k
            for k in vae_sd
            if k.startswith("rgb_head.temporal_blocks.")
        )
        detected_temporal = dict(
            temporal=True,
            temporal_num_heads=int(rgb_dec.get("num_heads", 16)),
            temporal_max_views=64,
            temporal_every=temporal_every,
            temporal_ffn_ratio=2.0 if has_ffn else 0.0,
        )
    else:
        temporal_every = None
        has_ffn = False
        detected_temporal = {"temporal": False}

    if temporal_mode == "auto" or (temporal_mode == "config" and injected):
        for key in (
            "temporal_num_heads",
            "temporal_max_views",
            "temporal_every",
            "temporal_ffn_ratio",
        ):
            rgb_dec.pop(key, None)
        rgb_dec.update(detected_temporal)
    elif temporal_mode == "off":
        rgb_dec["temporal"] = False
    elif temporal_mode == "on":
        rgb_dec["temporal"] = True
        if tblock_idxs:
            rgb_dec.update(detected_temporal)
        else:
            rgb_dec.setdefault("temporal_num_heads", int(rgb_dec.get("num_heads", 16)))
            rgb_dec.setdefault("temporal_max_views", 64)
            rgb_dec.setdefault("temporal_every", 2)
            rgb_dec.setdefault("temporal_ffn_ratio", 0.0)
    elif temporal_mode != "config":
        raise ValueError(f"Unknown RGB head temporal mode: {temporal_mode!r}")

    temporal_enabled = bool(rgb_dec.get("temporal", False))
    if temporal_enabled and tblock_idxs:
        print(
            f"  [vae] temporal RGBHead enabled "
            f"(blocks={tblock_idxs}, every={temporal_every}, ffn={has_ffn})"
        )
    else:
        print(
            f"  [vae] temporal RGBHead {'enabled' if temporal_enabled else 'disabled'} "
            f"(mode={temporal_mode}, ckpt_temporal={bool(tblock_idxs)})"
        )
    if injected:
        print(
            f"  [vae] injecting rgb_decoder (depth={detected_depth} "
            f"auto-detected from ckpt) so RGB head is built"
        )


def _latent_chunk_to_view_pointclouds(
    z_chunk, vae, rae, backbone_norm, rgb_imgs,
    H, W, device, use_amp, amp_dtype, *,
    stride: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        seq, h_l, w_l = vae._decode_trunk(z_chunk)
        c_trunk = seq.shape[-1]
        recon_raw = vae.dec_conv(
            seq.permute(0, 2, 1).reshape(-1, c_trunk, h_l, w_l))
        recon_feats = vae.denormalize_and_split(recon_raw)
    n_views = z_chunk.shape[0]
    gen_dpt_in = raw_no_cls_to_dpt_input(recon_feats, backbone_norm, n_views)
    gen_dpt_out = decode_dpt(gen_dpt_in, rae.rae_cl_decoder, H, W)
    depth = gen_dpt_out["depth"]
    return _recover_view_pointclouds(
        gen_dpt_out, depth, rgb_imgs, H, W, device, stride=stride,
    )


def _scene_pointcloud_from_dpt(
    dpt_out, depth_tensor, rgb_imgs, n_views, H, W, device, *,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge all views of one scene/chunk into a single point cloud."""
    view_pcs = _recover_view_pointclouds(
        dpt_out, depth_tensor, rgb_imgs, H, W, device, stride=stride,
    )
    xyz = np.concatenate([v[0] for v in view_pcs], axis=0)
    rgb = np.concatenate([v[1] for v in view_pcs], axis=0)
    max_pts = 5_000_000
    if xyz.shape[0] > max_pts:
        idx = np.random.choice(xyz.shape[0], max_pts, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz, rgb


def _save_pointcloud_from_latents(
    z_sampled: torch.Tensor,
    rgb_imgs: torch.Tensor,
    *,
    s_idx: int,
    ds_out_dir: str,
    tag: str,
    vae,
    rae,
    backbone_norm,
    H: int,
    W: int,
    device,
    use_amp: bool,
    amp_dtype,
    pc_stride: int,
) -> None:
    """Build + save a world-centric point cloud from denormalized latents."""
    z = z_sampled.to(device, non_blocking=True)
    rgb = rgb_imgs.to(device, non_blocking=True)
    actual_v = z.shape[0]
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        seq, h_l, w_l = vae._decode_trunk(z)
        c_trunk = seq.shape[-1]
        raw = vae.dec_conv(
            seq.permute(0, 2, 1).reshape(-1, c_trunk, h_l, w_l))
        recon_feats = vae.denormalize_and_split(raw)
    gen_dpt_in = raw_no_cls_to_dpt_input(recon_feats, backbone_norm, actual_v)
    gen_dpt_out = decode_dpt(gen_dpt_in, rae.rae_cl_decoder, H, W)
    gen_depth = gen_dpt_out["depth"]
    xyz, pc_rgb = _scene_pointcloud_from_dpt(
        gen_dpt_out, gen_depth, rgb, actual_v, H, W, device, stride=pc_stride,
    )
    ply_path = os.path.join(ds_out_dir, f"{s_idx:03d}_{tag}_pointcloud.ply")
    save_pointcloud_ply(ply_path, xyz, pc_rgb)
    print(f"    {tag.upper()} point cloud: {ply_path} ({xyz.shape[0]} pts)")


def _save_autoregress_pointclouds(
    z_sampled: torch.Tensor,
    rgb_imgs: torch.Tensor,
    chunk_z_list: list[torch.Tensor],
    *,
    s_idx: int,
    ds_out_dir: str,
    vae,
    rae,
    backbone_norm,
    H: int,
    W: int,
    device,
    use_amp: bool,
    amp_dtype,
    pc_stride: int,
    roll_cond_num: int,
    has_rgb: bool,
    denorm_fn,
) -> None:
    """Deferred autoregress point clouds (aligned + unaligned)."""
    per_chunk_pcs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        for zc_cpu in chunk_z_list:
            zc = zc_cpu.to(device, non_blocking=True)
            if has_rgb:
                seq_c, h_c, w_c = vae._decode_trunk(zc)
                rgb_c = denorm_fn(vae.rgb_head(seq_c, h_c, w_c, num_views=zc.shape[0]))
            else:
                rgb_c = rgb_imgs.to(device, non_blocking=True)[: zc.shape[0]]
            view_pcs = _latent_chunk_to_view_pointclouds(
                zc, vae, rae, backbone_norm, rgb_c,
                H, W, device, use_amp, amp_dtype, stride=pc_stride,
            )
            per_chunk_pcs.append(view_pcs)

    xyz_aligned, rgb_aligned, align_res = _merge_overlap_aligned_pointclouds(
        per_chunk_pcs, roll_cond_num,
    )
    ply_aligned = os.path.join(ds_out_dir, f"{s_idx:03d}_pred_pointcloud.ply")
    save_pointcloud_ply(ply_aligned, xyz_aligned, rgb_aligned)
    res_msg = (
        f"mean overlap residual={align_res[-1]:.4f}m"
        if align_res else "single chunk"
    )
    print(f"    PRED point cloud (overlap-aligned): "
          f"{ply_aligned} ({xyz_aligned.shape[0]} pts, {res_msg})")

    z_all = z_sampled.to(device, non_blocking=True)
    rgb_all = rgb_imgs.to(device, non_blocking=True)
    out_v = z_all.shape[0]
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        seq_all, h_a, w_a = vae._decode_trunk(z_all)
        c_a = seq_all.shape[-1]
        raw_a = vae.dec_conv(
            seq_all.permute(0, 2, 1).reshape(-1, c_a, h_a, w_a))
        feats_a = vae.denormalize_and_split(raw_a)
    dpt_in_a = raw_no_cls_to_dpt_input(feats_a, backbone_norm, out_v)
    dpt_out_a = decode_dpt(dpt_in_a, rae.rae_cl_decoder, H, W)
    xyz_u, rgb_u = _scene_pointcloud_from_dpt(
        dpt_out_a, dpt_out_a["depth"], rgb_all, out_v, H, W, device,
        stride=pc_stride,
    )
    ply_unaligned = os.path.join(
        ds_out_dir, f"{s_idx:03d}_pred_pointcloud_unaligned.ply")
    save_pointcloud_ply(ply_unaligned, xyz_u, rgb_u)
    print(f"    PRED point cloud (single-pass, no chunk align): "
          f"{ply_unaligned} ({xyz_u.shape[0]} pts)")


def _synthesize_free_trajectory(
    anchor_c2w: np.ndarray,
    n: int,
    *,
    motion: str = "wander",
    speed: float = 0.06,
    yaw_deg: float = 24.0,
    pitch_deg: float = 6.0,
    fwd_sign: float = 1.0,
    seed: int = 0,
) -> list:
    """Build an ``n``-length synthetic camera-to-world trajectory anchored at
    ``anchor_c2w`` for unbounded ("free") rollout demos.

    Design (per demo spec): *varied* motion (not a monotonic dolly) and no
    head-on "wall crash". The path is turn-dominant — heading (yaw) sweeps as a
    sum of low harmonics while forward advance is small and speed-modulated (it
    eases / nearly pauses), with gentle strafe + vertical bob + slight pitch for
    cinematic variety. Each rollout chunk is re-normalised pose-origin-relative,
    so only the RELATIVE inter-frame motion matters; ``speed``/``yaw_deg`` are
    the main knobs (env: FREE_ROLLOUT_SPEED / _YAW_DEG / _PITCH_DEG / _FWD_SIGN).
    'orbit' and 'spiral' are provided as alternates.
    """
    rng = np.random.default_rng(seed)
    A = np.asarray(anchor_c2w, dtype=np.float64).copy()
    R0 = A[:3, :3]
    p0 = A[:3, 3].copy()
    ph = rng.uniform(0.0, 2.0 * np.pi, size=6)
    yaw_a = np.deg2rad(yaw_deg)
    pitch_a = np.deg2rad(pitch_deg)
    w = (2.0 * np.pi) / max(n - 1, 1)          # 1 cycle over the whole clip
    f = np.array([1.0, 2.3, 0.5]) * w          # a few low harmonics = smooth+varied
    poses = []
    pos = p0.copy()
    for t in range(n):
        if motion == "orbit":
            yaw = yaw_a * 3.0 * (t / max(n - 1, 1))     # steady sweep
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd = 0.0                                    # pure rotation (radius via strafe)
            strafe = speed
            bob = 0.0
        elif motion == "spiral":
            yaw = yaw_a * 2.0 * (t / max(n - 1, 1))
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd = speed * 0.5
            strafe = speed * 0.5
            bob = speed * 0.2 * np.sin(f[2] * t + ph[5])
        elif motion == "drive":
            # forward/backward + turning ONLY: no vertical bob, no pitch tilt,
            # no lateral strafe. yaw sweeps (varied, sum of low harmonics) and
            # the signed advance eases through 0 so the camera goes forward then
            # backward without ever craning up/down.
            yaw = yaw_a * (0.6 * np.sin(f[0] * t + ph[0])
                           + 0.4 * np.sin(f[1] * t + ph[1]))
            pitch = 0.0
            spd = speed * np.sin(f[1] * t + ph[3])   # signed: forward & back
            strafe = 0.0
            bob = 0.0
        else:  # "wander" (default): varied, turn-dominant, no charging
            yaw = yaw_a * (0.6 * np.sin(f[0] * t + ph[0])
                           + 0.4 * np.sin(f[1] * t + ph[1]))
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd = speed * (0.5 + 0.5 * np.sin(f[1] * t + ph[3]))   # ease / pause
            strafe = speed * 0.6 * np.sin(f[0] * t + ph[4])
            bob = speed * 0.25 * np.sin(2.0 * f[2] * t + ph[5])
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        R = R0 @ (Ry @ Rx)
        fwd = fwd_sign * (-R[:, 2])            # camera looks down local -Z (OpenGL)
        right = R[:, 0]
        up = R[:, 1]
        pos = pos + spd * fwd + strafe * right + bob * up
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        poses.append(c2w)
    return poses


def _save_mp4(frames_rgb: list, path: str, fps: int = 4):
    """Save uint8 RGB frames as H.264 mp4 via ffmpeg pipe (S3-FUSE safe)."""
    import subprocess
    h, w = frames_rgb[0].shape[:2]
    tmp_root = "/local-ssd" if os.path.isdir("/local-ssd") else "/tmp"
    tmp_path = os.path.join(tmp_root, f"_mp4_{os.getpid()}_{os.path.basename(path)}")
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
            "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast", "-movflags", "+faststart",
            tmp_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in frames_rgb:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            print(f"[WARN] ffmpeg failed (rc={proc.returncode}) for {path}")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        subprocess.call(["cp", tmp_path, path])
    except Exception as e:
        print(f"[WARN] _save_mp4 failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _maybe_stage_output_dir(args) -> tuple[str, str | None]:
    """Redirect output_dir to local NVMe, returning (final_dir, staged_dir)."""
    final_dir = os.path.abspath(args.output_dir)
    use_stage = bool(getattr(args, "stage_output_local", False)) or _env_flag(
        "EVAL_STAGE_OUTPUT_LOCAL", "0"
    )
    if not use_stage:
        os.makedirs(final_dir, exist_ok=True)
        args.output_dir = final_dir
        return final_dir, None
    if not os.path.isdir("/local-ssd"):
        print("[stage-output] /local-ssd not found; writing directly to final output dir.")
        os.makedirs(final_dir, exist_ok=True)
        args.output_dir = final_dir
        return final_dir, None
    if final_dir.startswith("/local-ssd/"):
        os.makedirs(final_dir, exist_ok=True)
        args.output_dir = final_dir
        return final_dir, None

    stage_root = os.environ.get(
        "EVAL_LOCAL_OUTPUT_ROOT", "/local-ssd/gld_eval_outputs"
    )
    digest = hashlib.sha1(final_dir.encode("utf-8")).hexdigest()[:10]
    stage_dir = os.path.join(
        stage_root, f"{os.path.basename(final_dir)}_{digest}_{os.getpid()}"
    )
    os.makedirs(stage_dir, exist_ok=True)
    args.output_dir = stage_dir
    print(f"[stage-output] writing artifacts to local NVMe: {stage_dir}")
    print(f"[stage-output] final sync target: {final_dir}")
    return final_dir, stage_dir


def _sync_staged_output(stage_dir: str | None, final_dir: str) -> None:
    if not stage_dir:
        return
    print(f"[stage-output] syncing {stage_dir} -> {final_dir} ...")
    os.makedirs(final_dir, exist_ok=True)
    for root, dirs, files in os.walk(stage_dir):
        rel = os.path.relpath(root, stage_dir)
        dst_root = final_dir if rel == "." else os.path.join(final_dir, rel)
        os.makedirs(dst_root, exist_ok=True)
        for dirname in dirs:
            os.makedirs(os.path.join(dst_root, dirname), exist_ok=True)
        for filename in files:
            src = os.path.join(root, filename)
            dst = os.path.join(dst_root, filename)
            shutil.copyfile(src, dst)
    print("[stage-output] sync done")
    if not _env_flag("EVAL_KEEP_LOCAL_OUTPUT", "0"):
        shutil.rmtree(stage_dir, ignore_errors=True)
        print(f"[stage-output] removed local staging dir: {stage_dir}")


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", type=str, required=True)
    p.add_argument("--vae-ckpt", type=str, default=None,
                   help="Override yaml vae_checkpoint (e.g. temporal RGBHead ckpt).")
    p.add_argument("--gld-config", type=str, default=None,
                   help="Load codec from a GLD/VAE training yaml "
                        "(e.g. ..._temporal.yaml with rgb_decoder.temporal=true). "
                        "Default: yaml gld_config if set, else diffusion codec.")
    p.add_argument(
        "--rgb-head-temporal",
        choices=("config", "auto", "on", "off"),
        default="config",
        help="Control FeatureVAE RGBHead temporal blocks. 'config' preserves the "
        "YAML, while 'auto' follows temporal keys in the VAE checkpoint.",
    )
    p.add_argument("--config", type=str,
                   default="configs/flow_gae64.yaml")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--dataset", type=str, default="all",
                   choices=["re10k", "re10k_packed", "dl3dv", "dl3dv_packed",
                            "mvssynth", "scannetpp", "all"],
                   help="dataset to evaluate (default: all). scannetpp + "
                        "re10k_packed both use <scene>/video.mp4 + meta.json "
                        "layout (pass --data-root to override the default root). "
                        "Neither is included in 'all'.")
    p.add_argument("--dpt-decoder", type=str,
                   default="pretrained_models/da3/dpt_decoder.pt")
    p.add_argument("--da3-weights", type=str,
                   default="pretrained_models/da3/model.safetensors")
    p.add_argument("--num-scenes", type=int, default=4)
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--total-views", type=int, default=None,
                   help="Autoregressive rollout target length. When > num-views, "
                        "generate in overlapping chunks of num-views, reusing the "
                        "last roll-cond-num frames as refs for the next chunk.")
    p.add_argument("--roll-cond-num", type=int, default=None,
                   help="Trailing frames reused as ref for the next chunk "
                        "(default: same as --cond-num). Must satisfy "
                        "0 < roll-cond-num < num-views.")
    p.add_argument("--cond-num", type=int, default=1)
    p.add_argument("--cond-num-range", type=str, default=None)
    p.add_argument("--ref-view-sampling", type=str, default=None,
                   help="Match training dataset.ref_view_sampling (prefix, "
                        "interpolate, prefix_fl, random). Default: read from "
                        "config dataset.ref_view_sampling, else prefix.")
    p.add_argument("--output-dir", type=str,
                   default="results/eval_generation")
    p.add_argument("--stage-output-local", action="store_true",
                   help="Write artifacts under /local-ssd first, then sync "
                        "the completed output tree back to --output-dir. "
                        "Can also be enabled with EVAL_STAGE_OUTPUT_LOCAL=1.")
    p.add_argument("--resolution", type=int, nargs=2, default=None,
                   help="(H W) eval resolution. Default None → derive from "
                        "config stage_1.encoder_input_size (square). The old "
                        "hardcoded 504×504 default silently ran the whole "
                        "pipeline at 2× the trained 252×252, corrupting the "
                        "latent grid size, time_dist_shift, and feature stats.")
    p.add_argument("--video-fps", type=int, default=12,
                   help="FPS for saved pred/gt/compare mp4s (was hardcoded 4).")
    p.add_argument("--mode", choices=["recon", "generate", "t2v"], default="recon")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    # 引导方式 (对齐 scripts/eval_t2i.py):
    #   none   — 裸条件输出, 无引导。
    #   cfg    — classifier-free, 2 次前向/步 (uncond + cfg*(cond-uncond))。
    #   ig     — internal guidance (autoguidance), 1 次前向/步: cond + w*(cond-base),
    #            base 是 base_final_layer 浅层头, 与 main 同一前向产出 (需 base_model_depth!=null)。
    #   cfg_ig — 先对 cond 做 IG, 再与 uncond 做 CFG (2 次前向/步)。
    p.add_argument("--guidance", choices=["none", "cfg", "ig", "cfg_ig"], default="cfg",
                   help="none/cfg/ig/cfg_ig. cfg(默认)保持旧行为; ig/cfg_ig 需 base 头。")
    p.add_argument("--ig-scale", type=float, default=2.0,
                   help="Internal-guidance 权重 w (cond + w*(cond-base)), 仅 --guidance 含 ig 时生效。")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-corr", type=float, default=0.0,
                   help="跨帧初始噪声相关度 ∈[0,1]，降低帧间抖动。0=逐视图独立(默认)，"
                        "1=所有视图共享同一噪声。每视图边缘分布仍为 N(0,I)。"
                        "注意:训练用独立噪声，>0 属推理期 off-distribution 技巧。")
    p.add_argument("--latent-smooth", type=float, default=0.0,
                   help="Decode-time symmetric 3-frame smoothing on generated "
                        "latents before VAE decode. 0 disables; typical 0.1-0.25. "
                        "This is an inference-only deflicker probe.")
    p.add_argument("--model-interp-factor", type=int, default=0,
                   help="Model-based interpolation between generated keyframes. "
                        "factor=10 turns 17 stride-10 keyframes into 161 dense "
                        "frames (16 intervals * 10 + final endpoint).")
    p.add_argument("--model-interp-views", type=int, default=None,
                   help="Views per model interpolation pass. Default: --num-views "
                        "(V17 model uses 17).")
    p.add_argument("--time-dist-shift", type=float, default=None)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--loss-t", type=str, default="0.5")
    p.add_argument("--loss-samples", type=int, default=1)
    p.add_argument("--loss-only", action="store_true")
    p.add_argument("--metrics-only", action="store_true",
                   help="Run full eval and save only metrics.json; skip videos, "
                        "images, point clouds, prompt text, and trajectory plots.")
    p.add_argument("--timing-json", type=str, default=None,
                   help="Write compute-only timing totals for diffusion, latent/VAE, and RGB/geometry decode.")
    p.add_argument("--sample-interval", type=int, default=None)
    p.add_argument("--scene-manifest", type=str, default=None,
                   help="Fixed scene list JSON (e.g. configs/eval/re10k_v3_16scenes_seed42.json)")
    p.add_argument("--sampler", choices=["euler_v3", "transport"],
                   default="euler_v3",
                   help="euler_v3: custom Euler with ref-clamp every step "
                        "(now supports time_dist_shift warp); "
                        "transport: reuse stage2.transport.Sampler.sample_ode")
    p.add_argument("--sample-eps", type=float, default=1.0 / 1000,
                   help="endpoint epsilon for time grid "
                        "(matches transport.check_interval default)")
    p.add_argument("--save-pointcloud", action="store_true", default=True,
                   help="Save world-coordinate point cloud (.ply) + depth "
                        "(DPT + recovered poses). Default on.")
    p.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false",
                   help="Skip depth / .ply dumps.")
    p.add_argument("--dump-latents", action="store_true",
                   help="Dump z_pred/z_vae/pred_rgb NPZ without writing PLY. "
                        "Used by temporal decoder-amplification diagnostics.")
    p.add_argument("--ref-tgt-pc-gap", action="store_true",
                   help="Compute Ref–tgt point-cloud Chamfer gap after generate "
                        "(tab:ref_ablation): median-scale per-view PCs from DPT "
                        "rays, then bidirectional Chamfer between views[:cond] "
                        "and views[cond:]. Written to metrics.json.")
    p.add_argument("--pc-stride", type=int, default=1,
                   help="Point cloud pixel stride (1 = full per-pixel, no "
                        "downsampling; 4 = every 4th pixel). Default: 1.")
    p.add_argument("--dump-geometry", action="store_true",
                   help="Dump per-scene <idx>_geom.npz with NGD direct geometry "
                        "(generated + VAE-encode/decode depth & ray), model "
                        "pseudo-GT depth/ray, RGB (gen + GT) and real dataset "
                        "cameras. Consumed by scripts/eval_direct_geometry.py to "
                        "compare NGD's native depth/pose/point-map against "
                        "DA3/VGGT reconstructions. Requires a DPT-capable feature "
                        "backend (has_dpt); no-op for RGB-only latent codecs.")
    p.add_argument("--denoise-all", action="store_true", default=None,
                   help="Denoise all views including ref (adds ref_noise_frac noise to ref). "
                        "Auto-enabled if config has denoise_all_views=true.")
    p.add_argument("--ref-noise-frac", type=float, default=None,
                   help="Noise fraction for ref views in denoise-all mode. "
                        "Default: read from config training.ref_noise_frac.")
    # T2V mode
    p.add_argument("--prompt", type=str, default=None,
                   help="Text prompt for t2v mode (camera poses still read from dataset)")
    p.add_argument("--no-text", action="store_true",
                   help="Skip Qwen3 text conditioning; for ablation/debug only.")
    p.add_argument("--no-camera", action="store_true",
                   help="t2v/T2I: force plucker_6d=None (skip the camera branch "
                        "entirely), reproducing the exact T2I-stage forward "
                        "(z_ref_clean=None, plucker_6d=None, cond_num=0). Use this "
                        "to test the warm-started model's pure T2I ability without "
                        "the (possibly under-trained) camera branch perturbing it.")
    p.add_argument("--text-encoder", type=str,
                   default="google/umt5-base",
                   help="UMT5 text encoder path (used only when --prompt is given)")
    p.add_argument("--tokenizer", type=str,
                   default="google/umt5-base",
                   help="UMT5 tokenizer path")
    p.add_argument("--text-max-length", type=int, default=226)
    return p.parse_args()


# ── V3 inference inputs (simplified — no z_cond) ─────────────────────────

def reorder_scene_for_ref_sampling(
    imgs: torch.Tensor,
    intri_list: list,
    pose_list: list,
    cond_num: int,
    ref_view_sampling: str,
):
    """Reorder views like training ``convert_cut3r_batch`` + RoPE frame indices."""
    from video.cut3r_adapter import compute_view_order, resolve_ref_view_sampling

    V = imgs.shape[0]
    effective = resolve_ref_view_sampling(cond_num, ref_view_sampling)
    if effective == "interpolate" and cond_num != 2:
        raise ValueError(
            f"ref_view_sampling={ref_view_sampling!r} → {effective!r} "
            f"requires cond_num=2, got {cond_num}"
        )
    order = compute_view_order(V, cond_num, ref_view_sampling)
    order_t = torch.tensor(order, dtype=torch.long, device=imgs.device)
    if order != list(range(V)):
        imgs = imgs[order_t]
        intri_list = [intri_list[i] for i in order]
        pose_list = [pose_list[i] for i in order]
    return imgs, intri_list, pose_list, order_t


def _apply_pose_interval_scale(c2w_norm, pose_scale):
    """Scale only the translation of normalized poses by ``pose_scale``.

    Mirrors the training-side ``training.pose_interval_scale`` so inference feeds
    the model pose magnitudes under the same interval→scale convention. ``None``
    leaves poses untouched (feature disabled or unknown frame interval).
    """
    if pose_scale is None:
        return c2w_norm
    c2w_norm[..., :3, 3] = c2w_norm[..., :3, 3] * float(pose_scale)
    return c2w_norm


def build_inference_inputs(
    rae, vae, img_tensors, intrinsics_list, pose_list,
    H_img, W_img, device, latent_mean, latent_std,
    cond_num: int = 1,
    origin_idx: int = -1,
    translation_norm: str = "batch_max",
    latent_whiten: torch.Tensor | None = None,
    use_baseline_camera: bool = False,
    baseline_camera_mode: str = "plucker",
    pose_scale: float | None = None,
    s_patch_size: int = 1,
):
    """Build V3 inputs for a single (B=1) eval scene.

    DA3 is video-level (cross-view attention), so encoding ref images alone
    yields different features than encoding them together with tgt images.
    To match training (which now also encodes refs separately), we run DA3+VAE
    twice: once on refs only (z_ref) and once on all V views (z_all).  The
    returned z_input has refs from z_ref and tgt slots from z_all.

    Returns:
        z_input:    (V, C, h, w) — refs from refs-only ctx, tgt from all-view ctx.
        plucker_6d: (1, V*Hs*Ws, 6) per-token rays.
        x_norm:     normalised features (all-view ctx) for metric computation.
        feats_all:  raw DA3 features (all-view ctx) for DPT depth.
    """
    V = img_tensors.shape[0]
    imgs_5d = img_tensors.unsqueeze(0).to(device)

    def _encode(imgs):
        feats = rae.encode(imgs, mode="all")
        feats_no_cls = {k: v[:, 1:, :] for k, v in feats.items()}
        x_n = vae.normalize_levels(feats_no_cls, image_size=(H_img, W_img))
        z = vae.encode(x_n)[0]
        return z, x_n, feats

    z_all, x_norm, feats_all = _encode(imgs_5d)

    if cond_num > 0:
        z_ref, _, _ = _encode(imgs_5d[:, :cond_num])
    else:
        z_ref = None

    if latent_mean is not None:
        z_all = _norm_latent(z_all, latent_mean, latent_std, latent_whiten)
        if z_ref is not None:
            z_ref = _norm_latent(z_ref, latent_mean, latent_std, latent_whiten)

    if z_ref is not None:
        z_input = z_all.clone()
        z_input[:cond_num] = z_ref
    else:
        z_input = z_all

    pose_5d = torch.from_numpy(np.stack(pose_list)).float().unsqueeze(0).to(device)
    intri_5d = torch.from_numpy(np.stack(intrinsics_list)).float().unsqueeze(0).to(device)
    c2w_norm = normalise_c2w_for_batch(
        pose_5d, device, origin_idx=origin_idx, translation_norm=translation_norm,
    )
    c2w_norm = _apply_pose_interval_scale(c2w_norm, pose_scale)
    Ks = intrinsic_to_K(intri_5d, device)

    with torch.amp.autocast("cuda", enabled=False):
        if use_baseline_camera:
            from stage2.models.camera_baseline import build_baseline_camera_inputs
            cam_pack = build_baseline_camera_inputs(
                c2w_norm.float(), Ks.float(),
                H=H_img, W=W_img,
                cond_num=cond_num,
                total_view=V,
                camera_mode=baseline_camera_mode,
                v4_cond_extend=True,
            )
            plucker_6d = None
            baseline_cam_kwargs = dict(
                camera_embedding=cam_pack["camera_embedding"],
                viewmats=cam_pack["viewmats"],
                Ks=cam_pack["Ks"],
            )
        else:
            baseline_cam_kwargs = {}
            plucker_6d = compute_plucker_6d_per_token(
                c2w=c2w_norm.float(),
                Ks=Ks.float(),
                patches_x=z_input.shape[-1] // s_patch_size,
                patches_y=z_input.shape[-2] // s_patch_size,
                image_height=H_img,
                image_width=W_img,
                is_c2w=True,
            )

    return z_input, plucker_6d, x_norm, feats_all, z_ref, z_all, baseline_cam_kwargs


def build_rgb_codec_inference_inputs(
    codec, latent_backend: str, img_tensors, intrinsics_list, pose_list,
    H_img, W_img, device,
    cond_num: int = 1,
    origin_idx: int = -1,
    translation_norm: str = "batch_max",
    use_baseline_camera: bool = False,
    baseline_camera_mode: str = "plucker",
    pose_scale: float | None = None,
    s_patch_size: int = 1,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    latent_whiten: torch.Tensor | None = None,
):
    """Build eval inputs for RGB latent codecs: RAEv2, SD VAE, Wan2.1, da3_direct."""
    imgs_01 = (
        img_tensors.to(device)
        * torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        + torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    ).clamp(0, 1)
    if latent_backend == "da3_direct":
        # Codec self-normalizes raw DA3 level-0 with da3_direct.latent_stats_path.
        # Do not apply the framework latent_stats a second time.
        z_all = codec.encode_views(imgs_01.unsqueeze(0)).float()
    elif latent_backend == "sd_vae":
        scaling = float(getattr(codec.config, "scaling_factor", 0.18215))
        imgs_sd = imgs_01 * 2.0 - 1.0
        z_all = codec.encode(imgs_sd).latent_dist.sample().float() * scaling
    elif latent_backend == "wan2_1":
        imgs_wan = (imgs_01 * 2.0 - 1.0).unsqueeze(2)
        z_all = codec.encode(imgs_wan).latent_dist.sample().float().squeeze(2)
        if codec.config.latents_mean is not None and codec.config.latents_std is not None:
            mean = torch.tensor(
                codec.config.latents_mean, dtype=torch.float32, device=device,
            ).view(1, -1, 1, 1)
            std = torch.tensor(
                codec.config.latents_std, dtype=torch.float32, device=device,
            ).view(1, -1, 1, 1)
            z_all = (z_all - mean) / std
    else:
        raise ValueError(f"unsupported rgb codec backend: {latent_backend}")
    z_ref = z_all[:cond_num] if cond_num > 0 else None
    z_input = z_all.clone()

    pose_5d = torch.from_numpy(np.stack(pose_list)).float().unsqueeze(0).to(device)
    intri_5d = torch.from_numpy(np.stack(intrinsics_list)).float().unsqueeze(0).to(device)
    c2w_norm = normalise_c2w_for_batch(
        pose_5d, device, origin_idx=origin_idx, translation_norm=translation_norm,
    )
    c2w_norm = _apply_pose_interval_scale(c2w_norm, pose_scale)
    Ks = intrinsic_to_K(intri_5d, device)

    with torch.amp.autocast("cuda", enabled=False):
        if use_baseline_camera:
            from stage2.models.camera_baseline import build_baseline_camera_inputs
            cam_pack = build_baseline_camera_inputs(
                c2w_norm.float(), Ks.float(),
                H=H_img, W=W_img,
                cond_num=cond_num,
                total_view=img_tensors.shape[0],
                camera_mode=baseline_camera_mode,
                v4_cond_extend=True,
            )
            plucker_6d = None
            baseline_cam_kwargs = dict(
                camera_embedding=cam_pack["camera_embedding"],
                viewmats=cam_pack["viewmats"],
                Ks=cam_pack["Ks"],
            )
        else:
            baseline_cam_kwargs = {}
            plucker_6d = compute_plucker_6d_per_token(
                c2w=c2w_norm.float(),
                Ks=Ks.float(),
                patches_x=z_input.shape[-1] // s_patch_size,
                patches_y=z_input.shape[-2] // s_patch_size,
                image_height=H_img,
                image_width=W_img,
                is_c2w=True,
            )

    return z_input, plucker_6d, None, None, z_ref, z_all, baseline_cam_kwargs, imgs_01


def decode_rgb_codec(
    codec, latent_backend: str, z: torch.Tensor,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    latent_unwhiten: torch.Tensor | None = None,
    num_views: int | None = None,
) -> torch.Tensor:
    if latent_backend == "da3_direct":
        # codec.decode_rgb returns ImageNet-normalized pixels, matching
        # GAECodec's RGBHead. Convert to [0,1] for metrics/artifacts.
        rgb_norm = codec.decode_rgb(z.float(), num_views=num_views).float()
        mean = codec.rae.encoder_mean.to(rgb_norm.device, rgb_norm.dtype)
        std = codec.rae.encoder_std.to(rgb_norm.device, rgb_norm.dtype)
        return (rgb_norm * std + mean).clamp(0, 1)
    if latent_backend == "sd_vae":
        scaling = float(getattr(codec.config, "scaling_factor", 0.18215))
        dec = codec.decode((z / scaling).float()).sample
        return ((dec + 1.0) * 0.5).float().clamp(0, 1)
    if latent_backend == "wan2_1":
        z_dec = z.float()
        if codec.config.latents_mean is not None and codec.config.latents_std is not None:
            mean = torch.tensor(
                codec.config.latents_mean, dtype=torch.float32, device=z.device,
            ).view(1, -1, 1, 1)
            std = torch.tensor(
                codec.config.latents_std, dtype=torch.float32, device=z.device,
            ).view(1, -1, 1, 1)
            z_dec = z_dec * std + mean
        dec = codec.decode(z_dec.unsqueeze(2)).sample.squeeze(2)
        return ((dec + 1.0) * 0.5).float().clamp(0, 1)
    raise ValueError(f"unsupported rgb codec backend: {latent_backend}")


def build_plucker_from_cameras(
    intrinsics_list, pose_list, H_img, W_img, patches_y, patches_x,
    device, origin_idx: int = -1, translation_norm: str = "batch_max",
    pose_scale: float | None = None,
):
    """Build Plucker rays from camera data only (no image encoding)."""
    pose_5d = torch.from_numpy(np.stack(pose_list)).float().unsqueeze(0).to(device)
    intri_5d = torch.from_numpy(np.stack(intrinsics_list)).float().unsqueeze(0).to(device)
    c2w_norm = normalise_c2w_for_batch(
        pose_5d, device, origin_idx=origin_idx, translation_norm=translation_norm,
    )
    c2w_norm = _apply_pose_interval_scale(c2w_norm, pose_scale)
    Ks = intrinsic_to_K(intri_5d, device)

    with torch.amp.autocast("cuda", enabled=False):
        plucker_6d = compute_plucker_6d_per_token(
            c2w=c2w_norm.float(),
            Ks=Ks.float(),
            patches_x=patches_x,
            patches_y=patches_y,
            image_height=H_img,
            image_width=W_img,
            is_c2w=True,
        )
    return plucker_6d


def build_baseline_cam_from_cameras(
    intrinsics_list, pose_list, H_img, W_img,
    device, cond_num: int = 0, total_view: int | None = None,
    origin_idx: int = -1, baseline_camera_mode: str = "plucker",
    v4_cond_extend: bool = False, translation_norm: str = "batch_max",
    pose_scale: float | None = None,
) -> dict:
    """Build baseline camera kwargs from pose/intrinsics only (t2v / no-encode path)."""
    if total_view is None:
        total_view = len(pose_list)
    pose_5d = torch.from_numpy(np.stack(pose_list)).float().unsqueeze(0).to(device)
    intri_5d = torch.from_numpy(np.stack(intrinsics_list)).float().unsqueeze(0).to(device)
    c2w_norm = normalise_c2w_for_batch(
        pose_5d, device, origin_idx=origin_idx, translation_norm=translation_norm,
    )
    c2w_norm = _apply_pose_interval_scale(c2w_norm, pose_scale)
    Ks = intrinsic_to_K(intri_5d, device)

    with torch.amp.autocast("cuda", enabled=False):
        from stage2.models.camera_baseline import build_baseline_camera_inputs
        cam_pack = build_baseline_camera_inputs(
            c2w_norm.float(), Ks.float(),
            H=H_img, W=W_img,
            cond_num=cond_num,
            total_view=total_view,
            camera_mode=baseline_camera_mode,
            v4_cond_extend=v4_cond_extend,
        )
    return dict(
        camera_embedding=cam_pack["camera_embedding"],
        viewmats=cam_pack["viewmats"],
        Ks=cam_pack["Ks"],
    )


def _required_camera_frames(total_views: int, chunk_size: int, roll_cond_num: int) -> int:
    """Camera poses needed to autoregressively generate ``total_views`` frames."""
    if total_views <= chunk_size:
        return total_views
    out_len = chunk_size
    n_chunks = 1
    while out_len < total_views:
        out_len += chunk_size - roll_cond_num
        n_chunks += 1
    return (n_chunks - 1) * (chunk_size - roll_cond_num) + chunk_size


def _plan_autoregress_chunks(total_views: int, chunk_size: int, roll_cond_num: int):
    """Yield per-chunk (cam_start, cam_end, skip_out_prefix) for sliding-window rollout."""
    if total_views <= 0:
        return
    if total_views <= chunk_size:
        yield (0, total_views, 0)
        return
    out_len = 0
    cam_start = 0
    while out_len < total_views:
        skip = 0 if out_len == 0 else roll_cond_num
        cam_end = cam_start + chunk_size
        yield (cam_start, cam_end, skip)
        out_len += chunk_size - skip
        cam_start = cam_end - roll_cond_num


def _validate_rollout_cond(
    *,
    chunk_size: int,
    roll_cond_num: int,
    first_cond_max: int,
    mode: str,
    autoregress: bool,
) -> None:
    """Fail fast on invalid cond / roll combinations."""
    if roll_cond_num <= 0 or roll_cond_num >= chunk_size:
        raise ValueError(
            f"roll-cond-num must satisfy 0 < roll < num-views "
            f"(got roll={roll_cond_num}, num-views={chunk_size})"
        )
    if mode in ("recon", "generate"):
        if first_cond_max < 0:
            raise ValueError(f"cond-num must be >= 0 (got {first_cond_max})")
        if first_cond_max >= chunk_size:
            raise ValueError(
                f"cond-num must be < num-views so each chunk can denoise at least "
                f"one view (got cond={first_cond_max}, num-views={chunk_size})"
            )
    if autoregress and mode in ("recon", "generate") and first_cond_max == 0:
        print("  [WARN] autoregressive recon/generate with cond-num=0: chunk 0 starts "
              "from pure noise (same as generate); only later chunks use rolled refs.")
    if autoregress and mode in ("recon", "generate") and roll_cond_num > first_cond_max:
        print(f"  [WARN] roll-cond-num ({roll_cond_num}) > first-chunk cond ({first_cond_max}): "
              f"later chunks use fewer rolled latents than initial GT refs — intentional "
              f"sliding-window, not a bug.")


def _assert_chunk_plan(total_views: int, chunk_size: int, roll_cond_num: int) -> None:
    """Sanity-check planner output length matches total_views."""
    chunks = list(_plan_autoregress_chunks(total_views, chunk_size, roll_cond_num))
    out = 0
    for i, (a, b, skip) in enumerate(chunks):
        out += (b - a) if i == 0 else (b - a - skip)
    if out < total_views:
        raise RuntimeError(
            f"chunk planner produced {out} frames < total_views={total_views}"
        )


def _encode_ref_latent(
    rae, vae, ref_imgs, H_img, W_img, device,
    latent_mean, latent_std, latent_whiten=None,
):
    """Encode (K,3,H,W) ref images → (K,C,h,w) refs-only normalized latent."""
    if ref_imgs is None or ref_imgs.shape[0] == 0:
        return None
    imgs_5d = ref_imgs.unsqueeze(0).to(device)
    feats = rae.encode(imgs_5d, mode="all")
    feats_no_cls = {k: v[:, 1:, :] for k, v in feats.items()}
    x_n = vae.normalize_levels(feats_no_cls, image_size=(H_img, W_img))
    z_ref = vae.encode(x_n)[0]
    if latent_mean is not None:
        z_ref = _norm_latent(z_ref, latent_mean, latent_std, latent_whiten)
    return z_ref


def _build_chunk_camera_kwargs(
    intri_list, pose_list, cam_start, cam_end,
    H_img, W_img, device, *,
    cond_num: int,
    no_camera: bool,
    use_baseline_camera: bool,
    baseline_camera_mode: str,
    origin_idx: int,
    translation_norm: str,
    plucker_h: int, plucker_w: int,
    pose_scale: float | None = None,
):
    """Build plucker_6d / baseline kwargs for one autoregressive chunk.

    Normalises poses **within the chunk only** (same as training: each V-view
    clip is pose_origin-relative, then Plücker / baseline cameras are built).
    """
    chunk_intr = intri_list[cam_start:cam_end]
    chunk_pose = pose_list[cam_start:cam_end]
    chunk_v = len(chunk_pose)
    if no_camera:
        return None, {}
    if use_baseline_camera:
        baseline_cam_kwargs = build_baseline_cam_from_cameras(
            chunk_intr, chunk_pose, H_img, W_img, device,
            cond_num=cond_num, total_view=chunk_v,
            origin_idx=origin_idx,
            baseline_camera_mode=baseline_camera_mode,
            v4_cond_extend=True,
            translation_norm=translation_norm,
            pose_scale=pose_scale,
        )
        return None, baseline_cam_kwargs
    plucker_6d = build_plucker_from_cameras(
        chunk_intr, chunk_pose, H_img, W_img,
        patches_y=plucker_h, patches_x=plucker_w,
        device=device, origin_idx=origin_idx,
        translation_norm=translation_norm,
        pose_scale=pose_scale,
    )
    return plucker_6d, {}


def sample_long_sequence_v4(
    dit, rae, vae,
    *,
    img_tensors, intri_list, pose_list,
    chunk_size: int, total_views: int,
    first_cond_num: int, roll_cond_num: int,
    mode: str,
    H, W, device, latent_h, latent_w, plucker_h, plucker_w,
    latent_mean, latent_std, latent_whiten, latent_unwhiten,
    in_ch, use_amp, amp_dtype,
    plucker_6d=None, ref_global=None,
    cfg_uncond_ref_global=None,
    baseline_cam_kwargs=None,
    no_camera=False,
    reference_conditioning="clean_token_v4",
    use_baseline_camera=False,
    baseline_camera_mode="plucker",
    pose_origin_idx=-1,
    translation_norm: str = "batch_max",
    pose_scale: float | None = None,
    num_steps=50, cfg_scale=1.0, cfg_interval=(0.0, 1.0),
    guidance_mode="cfg", ig_scale=2.0,
    time_dist_shift=1.0, eps=1e-3,
    prediction="velocity", noise_corr=0.0,
):
    """Autoregressive rollout: chunk_size views per step, roll_cond_num-frame overlap.

    First chunk:
      - recon/generate with first_cond_num>0: GT first frames → z_ref_clean.
      - t2v: z_ref_clean=None (cond_num=0).
    Later chunks: previous chunk's last roll_cond_num latents → z_ref_clean.
    """
    if roll_cond_num <= 0 or roll_cond_num >= chunk_size:
        raise ValueError(
            f"roll_cond_num must be in [1, chunk_size-1], got {roll_cond_num}"
        )

    z_parts: list[torch.Tensor] = []
    z_ref_clean = None
    chunk_infos: list[dict] = []

    for chunk_idx, (cam_start, cam_end, skip_out) in enumerate(
        _plan_autoregress_chunks(total_views, chunk_size, roll_cond_num)
    ):
        chunk_v = cam_end - cam_start
        chunk_cond = first_cond_num if chunk_idx == 0 else roll_cond_num

        if chunk_idx == 0 and mode in ("recon", "generate") and first_cond_num > 0:
            z_ref_clean = _encode_ref_latent(
                rae, vae, img_tensors[:first_cond_num], H, W, device,
                latent_mean, latent_std, latent_whiten,
            )
        elif chunk_idx > 0:
            z_ref_clean = z_parts[-1][-roll_cond_num:]

        if mode == "t2v" and chunk_idx == 0 and first_cond_num == 0:
            z_ref_clean = None
            chunk_cond = 0

        # Per-chunk camera: normalise only this chunk's poses (matches training
        # where each V-view clip is pose_origin-relative independently).
        # plucker_6d / baseline_cam_kwargs passed from main() are NOT sliced here.
        chunk_plucker, chunk_baseline = _build_chunk_camera_kwargs(
            intri_list, pose_list, cam_start, cam_end,
            H, W, device,
            cond_num=chunk_cond,
            no_camera=no_camera,
            use_baseline_camera=use_baseline_camera,
            baseline_camera_mode=baseline_camera_mode,
            origin_idx=pose_origin_idx,
            translation_norm=translation_norm,
            plucker_h=plucker_h, plucker_w=plucker_w,
            pose_scale=pose_scale,
        )

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if z_ref_clean is not None:
                z_chunk = sample_v4_euler(
                    dit, z_ref_clean, chunk_v, chunk_cond,
                    plucker_6d=chunk_plucker, ref_global=ref_global,
                    cfg_uncond_ref_global=cfg_uncond_ref_global,
                    num_steps=num_steps, cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    guidance_mode=guidance_mode, ig_scale=ig_scale,
                    time_dist_shift=time_dist_shift, eps=eps,
                    prediction=prediction,
                    reference_conditioning=reference_conditioning,
                    baseline_cam_kwargs=chunk_baseline or None,
                    noise_corr=noise_corr,
                )
            else:
                z_chunk = sample_v4_euler(
                    dit, z_ref_clean=None, total_view=chunk_v, cond_num=0,
                    plucker_6d=chunk_plucker, ref_global=ref_global,
                    cfg_uncond_ref_global=cfg_uncond_ref_global,
                    num_steps=num_steps, cfg_scale=cfg_scale,
                    cfg_interval=cfg_interval,
                    guidance_mode=guidance_mode, ig_scale=ig_scale,
                    time_dist_shift=time_dist_shift, eps=eps,
                    noise_shape=(in_ch, latent_h, latent_w),
                    device=device, dtype=amp_dtype,
                    prediction=prediction,
                    reference_conditioning=reference_conditioning,
                    baseline_cam_kwargs=chunk_baseline or None,
                    noise_corr=noise_corr,
                )

        chunk_infos.append({
            "z_full": z_chunk.detach(),
            "skip": skip_out,
            "cam_start": cam_start,
            "cam_end": cam_end,
        })
        z_parts.append(z_chunk[skip_out:])

    z_out = torch.cat(z_parts, dim=0)
    return z_out[:total_views], chunk_infos


# ── V3 Euler ODE sampler ─────────────────────────────────────────────────

@torch.no_grad()
def sample_v3_euler(
    dit, z_ref, noise_tgt,
    plucker_6d, total_view, cond_num,
    num_steps=50, cfg_scale=1.0, cfg_interval=(0.0, 1.0),
    guidance_mode: str = "cfg", ig_scale: float = 2.0,
    time_dist_shift: float = 1.0,
    eps: float = 1.0 / 1000,
    ref_global=None,
    denoise_all: bool = False,
    ref_noise_frac: float = 0.1,
):
    """Custom Euler ODE sampler for V3: clamp ref, evolve tgt only.

    Flow direction: t=1 (noise) → t=0 (clean data).
    Velocity ut = noise - clean (pointing toward noise).
    ODE step: z_{t_next} = z_t + dt * v(z_t, t)  where dt = t_next - t_cur < 0.

    Time grid mirrors integrators.ode:
        t_lin = 1 - linspace(0, 1-eps, N+1)           # descending [≈1, eps]
        t     = shift * t_lin / (1 + (shift-1)*t_lin) # same warp as transport

    Setting time_dist_shift=1.0 recovers the original uniform grid.

    When denoise_all=True, ref views start from lightly noised latent
    (ref_noise_frac blend) and are also denoised by the model (not clamped).

    Args:
        z_ref:     (cond_num, C, h, w) clean ref latent.
        noise_tgt: (V-cond_num, C, h, w) noise for tgt views.
        denoise_all: if True, also denoise ref views (start from noised ref).
        ref_noise_frac: noise fraction for ref init when denoise_all=True.
    Returns:
        z_tgt:     (V-cond_num, C, h, w) denoised tgt latent.
                   If denoise_all, returns (V, C, h, w) — all views denoised.
    """
    device = z_ref.device

    if denoise_all:
        # Training: ref xt is fixed at (1-frac)*z_ref + frac*noise, doesn't evolve.
        # Model learns vel_ref = z_all_gt[ref] - z_ref (constant correction direction).
        # Correct inference: z_ref_final = z_ref + vel_ref_predicted.
        noise_ref = torch.randn_like(z_ref)
        z_ref_init = (1.0 - ref_noise_frac) * z_ref + ref_noise_frac * noise_ref
        z_tgt = noise_tgt.clone()
    else:
        z_tgt = noise_tgt.clone()

    # Build warped time grid identical to integrators.ode.
    t_lin = 1.0 - torch.linspace(0.0, 1.0 - eps, num_steps + 1, device=device)
    t_grid = time_dist_shift * t_lin / (1.0 + (time_dist_shift - 1.0) * t_lin)

    # Accumulate ref velocity across steps for a stable estimate
    vel_ref_accum = None

    for step in range(num_steps):
        t_cur = float(t_grid[step].item())
        t_next = float(t_grid[step + 1].item())
        dt = t_next - t_cur  # negative, non-uniform when shift > 1

        if denoise_all:
            # Ref stays fixed throughout ODE (matches training)
            z_in = torch.cat([z_ref_init, z_tgt], dim=0)
        else:
            z_in = torch.cat([z_ref, z_tgt], dim=0)

        t_tensor = torch.full((total_view,), t_cur, device=device)

        in_interval = cfg_interval[0] <= t_cur <= cfg_interval[1]
        use_cfg = guidance_mode in ("cfg", "cfg_ig")
        use_ig = guidance_mode in ("ig", "cfg_ig")

        if use_ig and in_interval:
            out_cond = dit(
                z_in, t_tensor, total_view,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                ref_global=ref_global,
                return_dict=True,
            )
            p_cond = out_cond["main"]
            if out_cond.get("base") is None:
                raise RuntimeError(
                    "guidance=ig/cfg_ig 需要 base 头 (base_model_depth!=null), 但模型返回 "
                    "base=None。改用 --guidance=cfg/none, 或换一个带 base_model_depth 的 ckpt。"
                )
            p_cond = p_cond + ig_scale * (p_cond - out_cond["base"])
            if use_cfg and cfg_scale > 1.0:
                p_uncond = dit(
                    z_in, t_tensor, total_view,
                    plucker_6d=plucker_6d,
                    cond_num=cond_num,
                    ref_global=None,
                )
                vel = p_uncond + cfg_scale * (p_cond - p_uncond)
            else:
                vel = p_cond
        elif use_cfg and cfg_scale > 1.0 and in_interval:
            vel = dit.forward_with_cfg(
                z_in, t_tensor, total_view,
                plucker_6d=plucker_6d,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cond_num=cond_num,
                ref_global=ref_global,
            )
        else:
            vel = dit(
                z_in, t_tensor, total_view,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                ref_global=ref_global,
            )

        # Only evolve tgt views
        vel_tgt = vel[cond_num:]
        z_tgt = z_tgt + dt * vel_tgt

        # Accumulate ref velocity (should be ~constant across steps)
        if denoise_all:
            if vel_ref_accum is None:
                vel_ref_accum = vel[:cond_num].clone()
            else:
                vel_ref_accum = vel_ref_accum + vel[:cond_num]

    if denoise_all:
        # Average ref velocity across all steps (stable estimate of the constant)
        vel_ref_avg = vel_ref_accum / num_steps
        # Correct formula: z_ref_final = z_ref_clean + vel_ref
        # (vel_ref ≈ z_all_gt[ref] - z_ref, so z_ref + vel_ref ≈ z_all_gt[ref])
        z_ref_final = z_ref + vel_ref_avg
        return torch.cat([z_ref_final, z_tgt], dim=0)
    return z_tgt


# ── V3 transport-based sampler (reuses stage2.transport.Sampler) ────────

@torch.no_grad()
def sample_v3_transport(
    dit, transport, z_ref, noise_tgt,
    plucker_6d, total_view, cond_num,
    num_steps=50, cfg_scale=1.0, cfg_interval=(0.0, 1.0),
    sampling_method: str = "euler",
    ref_global=None,
):
    """Alternative V3 sampler that reuses stage2.transport.Sampler.sample_ode.

    The transport's velocity_ode non-concat branch already zeros velocity for
    ref views (so they stay clean throughout the ODE), which is numerically
    equivalent to our manual cat([z_ref, z_tgt]) clamp at every step.

    Benefits:
      * Automatically uses the warped time grid (time_dist_shift).
      * Can swap sampling_method to 'heun' / 'dopri5' for free.

    Args:
        transport: configured transport object (same used for loss eval).
        z_ref:     (cond_num, C, h, w) clean ref latent.
        noise_tgt: (V-cond_num, C, h, w) noise for tgt views.
    Returns:
        z_tgt:     (V-cond_num, C, h, w) denoised tgt latent.
    """
    try:
        from stage2.transport.transport import Sampler
    except ImportError as exc:
        raise RuntimeError(
            "--sampler transport is not shipped in the public release; "
            "use the default --sampler euler_v3"
        ) from exc
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(
        sampling_method=sampling_method,
        num_steps=num_steps,
    )

    # Build init: refs stay clean, tgt slots are noise.
    init = torch.cat([z_ref, noise_tgt], dim=0)

    model_kwargs = dict(
        total_view=total_view,
        cond_num=cond_num,
        plucker_6d=plucker_6d,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        ref_global=ref_global,
    )

    xs = sample_fn(init, dit, **model_kwargs)
    # odeint returns a tensor stack along dim 0: (T, BV, C, H, W)
    z_final = xs[-1]
    return z_final[cond_num:]



# ── V4 Standard Euler ODE sampler (all views from noise) ─────────────────

@torch.no_grad()
def _rae_eval_loss(
    dit,
    z_all_gt: torch.Tensor,         # (BV, C, h, w) all-view-ctx clean latent (whitened)
    z_ref_clean: torch.Tensor | None,  # (B*cond_num, C, h, w) refs-only-ctx clean latent
    total_view: int,
    cond_num: int,
    model_kwargs: dict,
    *,
    time_dist_shift: float,
    time_dist_type: str = "logit-normal_0_1",
    reference_conditioning: str = "clean_token_v4",
    t_override: float | None = None,
    t_eps_loss: float = 0.05,
):
    """Eval-side mirror of ``stage2.transport.flow.training_losses_rae``.

    Reason this is needed:
      * The trainer uses ``training_losses_rae`` which assumes the model output
        is **x-prediction** and converts it to velocity via
        ``convert_x_to_v(x_pred, xt, t)`` before computing MSE against the
        velocity target ``v_target = noise - x_clean``.
      * The default ``stage2.transport.create_transport`` does NOT recognize
        ``prediction='x'`` — it silently maps the model to ``ModelType.VELOCITY``
        and then does ``MSE(x_pred, ut)`` which is meaningless (x_pred lives in
        clean-latent space, ut lives in noise-clean space → loss values are
        wildly inflated, ~8 at t=0.5 with full-cov whitening instead of ~0.5).
      * For eval-time loss reporting we need the trainer's exact formula so
        loss numbers are comparable across training/eval and across ckpts.

    Loss formula (matches training_losses_rae main loss, REPA / base heads
    omitted because eval only needs scalar "tgt loss"):
        xt        = (1-t) * x_clean + t * noise           # all views noised
        x_pred    = model(xt, t, ..., return_dict=False)  # main head only
        v_pred    = (xt - x_pred) / max(t, t_eps_loss)
        v_target  = noise - x_clean
        loss_main = MSE(v_pred[tgt views], v_target[tgt views])

    Returns: ``{"loss": scalar tensor on tgt views}`` to mimic the legacy
    transport API the caller already uses.
    """
    BV = z_all_gt.shape[0]
    assert BV % total_view == 0, (BV, total_view)
    B = BV // total_view
    device, dtype = z_all_gt.device, z_all_gt.dtype

    # 1) Sample t. Trainer uses logit-normal_0_1 + the resolution-aware shift
    #    (already pre-resolved by caller into ``time_dist_shift``). For fixed-t
    #    eval (loss_t=0.5) we ignore the time_dist_shift warp since the user
    #    is asking "loss exactly at t=0.5", not "logit-normal sample".
    if t_override is not None:
        t_b = torch.full((B,), float(t_override), device=device, dtype=dtype)
    elif time_dist_type.startswith("logit-normal"):
        parts = time_dist_type.split("_")
        mu, sigma = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.0, 1.0)
        t_b = sample_logit_normal_t(
            B, mu=mu, sigma=sigma, shift=time_dist_shift,
            device=device, dtype=dtype,
        )
    else:
        # Uniform with optional shift warp (rare; covered for completeness).
        u = torch.rand(B, device=device, dtype=dtype)
        if time_dist_shift != 1.0:
            u = time_dist_shift * u / (1.0 + (time_dist_shift - 1.0) * u)
        t_b = u.clamp(1e-3, 1.0 - 1e-3)

    # Broadcast t to per-view (all V views share the same t per sample — this
    # is the trainer's convention; the model expects a flat (BV,) tensor).
    t = t_b.unsqueeze(1).expand(B, total_view).reshape(BV)

    noise = torch.randn_like(z_all_gt)
    t_view = t.view(BV, *([1] * (z_all_gt.dim() - 1)))
    xt = (1.0 - t_view) * z_all_gt + t_view * noise
    v_target = noise - z_all_gt
    state_v3 = reference_conditioning == "state_v3"
    if state_v3 and cond_num > 0:
        if z_ref_clean is None:
            raise ValueError("state_v3 eval loss requires clean refs")
        xt_5d = xt.reshape(B, total_view, *xt.shape[1:])
        xt_5d[:, :cond_num] = z_ref_clean.reshape(
            B, cond_num, *z_ref_clean.shape[1:]
        )
        xt = xt_5d.reshape(BV, *xt.shape[1:])
        t_2d = t.reshape(B, total_view)
        t_2d[:, :cond_num] = 0.0
        t = t_2d.reshape(BV)

    # 2) Forward in main-head mode. The RAE training path uses return_dict=True
    #    only to also fetch base/repa for the dual losses; for eval main loss
    #    the default scalar-tensor return is exactly what we want.
    fwd_kwargs = dict(model_kwargs)
    fwd_kwargs.pop("total_view", None)
    fwd_kwargs.pop("cond_num", None)
    if state_v3:
        fwd_kwargs["reference_in_state"] = True
    x_pred = dit(
        xt, t, total_view,
        z_ref_clean=None if state_v3 else z_ref_clean,
        cond_num=cond_num,
        **fwd_kwargs,
    )

    # 3) x → v at the model boundary, then MSE on tgt views only (eval reports
    #    "Tgt Loss" specifically — ref views are conditioning, not a target).
    v_pred = convert_x_to_v(x_pred, xt, t, t_eps=t_eps_loss)

    v_pred_5d = rearrange(v_pred, "(b v) c h w -> b v c h w", v=total_view)
    v_target_5d = rearrange(v_target, "(b v) c h w -> b v c h w", v=total_view)

    # all-view MSE == training_losses_rae `main` (it does NOT slice off refs).
    # Report this so the eval number is directly comparable to the trainer's
    # logged `main=...`. tgt-only is the stricter novel-view metric. For
    # state_v3 the ref slots are clamped clean (t=0) and are NOT trained, so an
    # all-view average would be polluted by the ref slots — report the tgt-only
    # loss for both to stay meaningful.
    if state_v3 and cond_num > 0:
        loss_all = (v_pred_5d[:, cond_num:] - v_target_5d[:, cond_num:]).pow(2).mean()
    else:
        loss_all = (v_pred - v_target).pow(2).mean()

    if cond_num > 0:
        v_pred_5d = v_pred_5d[:, cond_num:]
        v_target_5d = v_target_5d[:, cond_num:]
    loss_tgt = (v_pred_5d - v_target_5d).pow(2).mean()

    return {"loss": loss_tgt, "loss_all": loss_all, "t_mean": t_b.mean().item()}


def _make_view_correlated_noise(
    total_view: int, C: int, h: int, w: int,
    device: torch.device, dtype: torch.dtype = torch.float32,
    corr: float = 0.0,
) -> torch.Tensor:
    """初始噪声 (total_view, C, h, w)，跨视图相关度 corr ∈ [0,1]。

    corr 把"全视图共享基底噪声"与"逐视图独立噪声"按方差加权混合：
    ``z = sqrt(corr)*base + sqrt(1-corr)*per_view``。每个视图的边缘分布仍是
    N(0, I)（corr*1 + (1-corr)*1 = 1），但相邻帧噪声相关 → 降低解码后帧间抖动。
    corr=0 复现逐视图独立（旧行为），corr=1 所有视图同一噪声。
    """
    per_view = torch.randn(total_view, C, h, w, device=device, dtype=dtype)
    corr = float(min(max(corr, 0.0), 1.0))
    if corr <= 0.0:
        return per_view
    base = torch.randn(1, C, h, w, device=device, dtype=dtype)
    return (corr ** 0.5) * base + ((1.0 - corr) ** 0.5) * per_view


def sample_v4_euler(
    dit, z_ref_clean, total_view, cond_num,
    plucker_6d=None, ref_global=None, cfg_uncond_ref_global=None,
    num_steps=50, cfg_scale=1.0, cfg_interval=(0.0, 1.0),
    guidance_mode: str = "cfg", ig_scale: float = 2.0,
    time_dist_shift: float = 1.0, eps: float = 1.0 / 1000,
    noise_shape: tuple[int, ...] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    prediction: str = "velocity",
    reference_conditioning: str = "clean_token_v4",
    t_eps_pred: float = 0.05,
    baseline_cam_kwargs: dict | None = None,
    noise_corr: float = 0.0,
    view_frame_idx: torch.Tensor | None = None,
    cfg_drop_text: bool = True,
    cfg_drop_camera: bool = True,
    cfg_drop_ref: bool = True,
):
    """Euler ODE for V4 clean-token or V3 reference-in-state conditioning.

    Args:
        z_ref_clean: (cond_num, C, h, w) clean ref-only latent, or None for
                     unconditional (t2v) mode.
        cfg_uncond_ref_global: unconditional text context used by CFG. For
                    Qwen-trained models this must be the encoding of the empty
                    string. Passing ``None`` disables cross-attention entirely
                    and does not match the trainer's text-dropout branch.
        noise_shape: (C, h, w) required when z_ref_clean is None.
        device/dtype: required when z_ref_clean is None.
        prediction: ``"velocity"`` (model output is v_t directly, Euler step
                    is ``z += dt * vel``) or ``"x"`` (RAE recipe: model output
                    is x_pred, convert to velocity via ``(z - x_pred)/t`` before
                    the Euler step). The yaml's ``transport.params.prediction``
                    is what drives this; caller is responsible for passing it.
                    Misalignment between the model's actual head and this flag
                    silently produces wrong samples (no shape error).
        t_eps_pred: clamp on t when converting x→v (mirrors trainer's
                    ``t_eps_loss=0.05`` for stable bf16 division near t=0).
    """
    if z_ref_clean is not None:
        device = z_ref_clean.device
        C, h, w = z_ref_clean.shape[1:]
        dtype = z_ref_clean.dtype
    else:
        assert noise_shape is not None, "noise_shape required when z_ref_clean is None"
        assert device is not None, "device required when z_ref_clean is None"
        C, h, w = noise_shape
    if reference_conditioning not in ("clean_token_v4", "state_v3"):
        raise ValueError(
            "reference_conditioning must be 'clean_token_v4' or 'state_v3'; "
            f"got {reference_conditioning!r}"
        )
    state_v3 = reference_conditioning == "state_v3"
    if state_v3 and cond_num > 0 and z_ref_clean is None:
        raise ValueError("state_v3 sampling requires clean refs when cond_num > 0")
    z = _make_view_correlated_noise(total_view, C, h, w, device, dtype, corr=noise_corr)
    if state_v3 and cond_num > 0:
        z[:cond_num] = z_ref_clean

    t_lin = 1.0 - torch.linspace(0.0, 1.0 - eps, num_steps + 1, device=device)
    t_grid = time_dist_shift * t_lin / (1.0 + (time_dist_shift - 1.0) * t_lin)

    cam_kw = dict(baseline_cam_kwargs or {})
    model_kw = {}
    if state_v3:
        model_kw["reference_in_state"] = True
    if view_frame_idx is not None:
        model_kw["view_frame_idx"] = view_frame_idx
    cond_kw = {**cam_kw, **model_kw}
    # CFG uncond camera: drop to the identity baseline (and plucker=None) when
    # cfg_drop_camera, else reuse the conditional camera unchanged.
    if cam_kw and cfg_drop_camera:
        from stage2.models.camera_baseline import make_uncond_baseline_camera
        u_cam, u_vm, u_Ks = make_uncond_baseline_camera(
            cam_kw["camera_embedding"], cam_kw["viewmats"], cam_kw["Ks"],
        )
        uncond_cam_kw = dict(
            camera_embedding=u_cam, viewmats=u_vm, Ks=u_Ks,
        )
    else:
        uncond_cam_kw = dict(cam_kw)
    uncond_plucker = None if cfg_drop_camera else plucker_6d
    # Match training ref-drop: the encoder then sees no cond tokens. Baseline
    # cameras may still be cond-extended to K+V; slice the prefix to V views.
    if cfg_drop_ref and uncond_cam_kw and cond_num > 0:
        vm = uncond_cam_kw.get("viewmats")
        if vm is not None and vm.dim() >= 2 and vm.shape[1] == total_view + cond_num:
            from einops import rearrange as _rearrange
            uncond_cam_kw = dict(uncond_cam_kw)
            uncond_cam_kw["viewmats"] = vm[:, cond_num:]
            if uncond_cam_kw.get("Ks") is not None:
                uncond_cam_kw["Ks"] = uncond_cam_kw["Ks"][:, cond_num:]
            emb = uncond_cam_kw.get("camera_embedding")
            if emb is not None and emb.shape[0] % (total_view + cond_num) == 0:
                cam5 = _rearrange(
                    emb, "(b v) c h w -> b v c h w", v=total_view + cond_num,
                )
                uncond_cam_kw["camera_embedding"] = _rearrange(
                    cam5[:, cond_num:], "b v c h w -> (b v) c h w",
                )
    uncond_kw = {**uncond_cam_kw, **model_kw}
    # uncond reference/text: cond_num=0 (no clean refs) + null-caption text.
    uncond_cnum = 0 if cfg_drop_ref else cond_num
    uncond_text = cfg_uncond_ref_global if cfg_drop_text else ref_global

    for step in range(num_steps):
        t_cur = float(t_grid[step].item())
        t_next = float(t_grid[step + 1].item())
        dt = t_next - t_cur

        t_tensor = torch.full((total_view,), t_cur, device=device)
        model_z_ref_clean = z_ref_clean
        if state_v3 and cond_num > 0:
            z[:cond_num] = z_ref_clean
            t_tensor[:cond_num] = 0.0
            model_z_ref_clean = None

        # 引导在模型 NATIVE 输出空间组合 (x-pred 即 x 空间)。IG/CFG 都是 x 的线性
        # 组合, 与下方 x→v 转换交换, 故等价于在 v 空间组合 (对齐 eval_t2i._sample_batch)。
        in_interval = cfg_interval[0] <= t_cur <= cfg_interval[1]
        use_cfg = guidance_mode in ("cfg", "cfg_ig")
        use_ig = guidance_mode in ("ig", "cfg_ig")

        if use_ig and in_interval:
            # Internal guidance: 单次前向同时拿 main + base (浅层头快照), 放大 (main-base)。
            out_cond = dit(
                z, t_tensor, total_view,
                z_ref_clean=model_z_ref_clean,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                ref_global=ref_global,
                return_dict=True,
                **cond_kw,
            )
            p_cond = out_cond["main"]
            if out_cond.get("base") is None:
                raise RuntimeError(
                    "guidance=ig/cfg_ig 需要 base 头 (base_model_depth!=null), 但模型返回 "
                    "base=None。改用 --guidance=cfg/none, 或换一个带 base_model_depth 的 ckpt。"
                )
            p_cond = p_cond + ig_scale * (p_cond - out_cond["base"])
            if use_cfg and cfg_scale > 1.0:
                # CFG: 把 IG 后的 cond 与 uncond (丢 ref) 混合 (第 2 次前向)。
                p_uncond = dit(
                    z, t_tensor, total_view,
                    z_ref_clean=(None if cfg_drop_ref else model_z_ref_clean),
                    plucker_6d=uncond_plucker,
                    cond_num=uncond_cnum,
                    ref_global=uncond_text,
                    **uncond_kw,
                )
                model_out = p_uncond + cfg_scale * (p_cond - p_uncond)
            else:
                model_out = p_cond
        elif use_cfg and cfg_scale > 1.0 and in_interval:
            # Explicit forwards are required here: ``forward_with_cfg`` uses
            # ``ref_global=None`` for uncond, which skips cross-attention instead
            # of reproducing the trained empty-caption branch.
            p_cond = dit(
                z, t_tensor, total_view,
                z_ref_clean=model_z_ref_clean,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                ref_global=ref_global,
                **cond_kw,
            )
            p_uncond = dit(
                z, t_tensor, total_view,
                z_ref_clean=(None if cfg_drop_ref else model_z_ref_clean),
                plucker_6d=uncond_plucker,
                cond_num=uncond_cnum,
                ref_global=uncond_text,
                **uncond_kw,
            )
            model_out = p_uncond + cfg_scale * (p_cond - p_uncond)
        else:
            model_out = dit(
                z, t_tensor, total_view,
                z_ref_clean=model_z_ref_clean,
                plucker_6d=plucker_6d,
                cond_num=cond_num,
                ref_global=ref_global,
                **cond_kw,
            )

        # Convert model output to velocity for the Euler step.
        # prediction='x': model emits x_pred (RAE recipe), v = (z - x_pred)/t.
        # prediction='velocity': model emits v directly (legacy V3/V4 flow).
        # Misalignment between the trained head and this flag is silent — the
        # samples will be visually wrong but no shape error.
        if prediction == "x":
            t_view = t_tensor.view(total_view, 1, 1, 1).clamp_min(t_eps_pred)
            vel = (z - model_out) / t_view
        else:
            vel = model_out

        if state_v3 and cond_num > 0:
            z[cond_num:] = z[cond_num:] + dt * vel[cond_num:]
            z[:cond_num] = z_ref_clean
        else:
            z = z + dt * vel

    return z


def _interpolate_cameras_between(
    K0: np.ndarray,
    K1: np.ndarray,
    c2w0: np.ndarray,
    c2w1: np.ndarray,
    n_views: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Interpolate n camera slots from endpoint 0 to endpoint 1 inclusive."""
    alphas = np.linspace(0.0, 1.0, n_views, dtype=np.float32)
    Ks = [(1.0 - a) * K0 + a * K1 for a in alphas]
    try:
        from scipy.spatial.transform import Rotation, Slerp

        key_rots = Rotation.from_matrix(
            np.stack([c2w0[:3, :3], c2w1[:3, :3]], axis=0)
        )
        rots = Slerp([0.0, 1.0], key_rots)(alphas).as_matrix().astype(np.float32)
    except Exception:
        rots = []
        for a in alphas:
            R = (1.0 - a) * c2w0[:3, :3] + a * c2w1[:3, :3]
            U, _, Vt = np.linalg.svd(R)
            R = (U @ Vt).astype(np.float32)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = (U @ Vt).astype(np.float32)
            rots.append(R)
        rots = np.stack(rots, axis=0)

    poses = []
    for a, R in zip(alphas, rots):
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = (1.0 - a) * c2w0[:3, 3] + a * c2w1[:3, 3]
        poses.append(c2w)
    return Ks, poses


@torch.no_grad()
def model_interpolate_generated_keyframes(
    dit,
    z_keyframes: torch.Tensor,
    intri_list: list,
    pose_list: list,
    *,
    factor: int,
    interp_views: int,
    H: int,
    W: int,
    device,
    plucker_h: int,
    plucker_w: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    use_baseline_camera: bool,
    no_camera: bool,
    reference_conditioning: str,
    baseline_camera_mode: str,
    pose_origin_idx: int,
    translation_norm: str,
    ref_global,
    cfg_uncond_ref_global=None,
    num_steps: int,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    guidance_mode: str,
    ig_scale: float,
    time_dist_shift: float,
    eps: float,
    prediction: str,
    noise_corr: float,
    pose_scale: float | None = None,
) -> torch.Tensor:
    """Use the V4 model as a first/last-frame interpolator for generated keys."""
    if factor < 2:
        raise ValueError(f"model-interp-factor must be >= 2, got {factor}")
    if interp_views < 3:
        raise ValueError(f"model-interp-views must be >= 3, got {interp_views}")
    if z_keyframes.shape[0] != len(intri_list) or z_keyframes.shape[0] != len(pose_list):
        raise ValueError(
            f"keyframe/camera length mismatch: z={z_keyframes.shape[0]}, "
            f"K={len(intri_list)}, pose={len(pose_list)}"
        )

    model_order = [0, interp_views - 1] + list(range(1, interp_views - 1))
    view_frame_idx = torch.tensor(model_order, device=device, dtype=torch.long)
    select = np.rint(np.linspace(0, interp_views - 1, factor + 1)).astype(np.int64)
    if len(set(select.tolist())) != len(select):
        raise ValueError(
            f"interp_views={interp_views} is too small for factor={factor}; "
            f"selection has duplicates: {select.tolist()}"
        )

    dense_parts: list[torch.Tensor] = []
    for i in range(z_keyframes.shape[0] - 1):
        K_chron, pose_chron = _interpolate_cameras_between(
            np.asarray(intri_list[i], dtype=np.float32),
            np.asarray(intri_list[i + 1], dtype=np.float32),
            np.asarray(pose_list[i], dtype=np.float32),
            np.asarray(pose_list[i + 1], dtype=np.float32),
            interp_views,
        )
        K_model = [K_chron[j] for j in model_order]
        pose_model = [pose_chron[j] for j in model_order]

        if no_camera:
            chunk_plucker = None
            chunk_baseline = {}
        elif use_baseline_camera:
            chunk_plucker = None
            chunk_baseline = build_baseline_cam_from_cameras(
                K_model, pose_model, H, W, device,
                cond_num=2, total_view=interp_views,
                origin_idx=pose_origin_idx,
                baseline_camera_mode=baseline_camera_mode,
                v4_cond_extend=True,
                translation_norm=translation_norm,
                pose_scale=pose_scale,
            )
        else:
            chunk_baseline = {}
            chunk_plucker = build_plucker_from_cameras(
                K_model, pose_model, H, W,
                patches_y=plucker_h, patches_x=plucker_w,
                device=device, origin_idx=pose_origin_idx,
                translation_norm=translation_norm,
                pose_scale=pose_scale,
            )

        z_ref_clean = torch.stack([z_keyframes[i], z_keyframes[i + 1]], dim=0)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            z_seg = sample_v4_euler(
                dit, z_ref_clean, interp_views, 2,
                plucker_6d=chunk_plucker,
                ref_global=ref_global,
                cfg_uncond_ref_global=cfg_uncond_ref_global,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                cfg_interval=cfg_interval,
                guidance_mode=guidance_mode,
                ig_scale=ig_scale,
                time_dist_shift=time_dist_shift,
                eps=eps,
                prediction=prediction,
                reference_conditioning=reference_conditioning,
                baseline_cam_kwargs=chunk_baseline or None,
                noise_corr=noise_corr,
                view_frame_idx=view_frame_idx,
            )

        z_chron = torch.empty_like(z_seg)
        for slot, phys in enumerate(model_order):
            z_chron[phys] = z_seg[slot]
        z_chron[0] = z_keyframes[i]
        z_chron[-1] = z_keyframes[i + 1]

        chosen = torch.as_tensor(select, device=z_chron.device, dtype=torch.long)
        if i < z_keyframes.shape[0] - 2:
            chosen = chosen[:-1]
        dense_parts.append(z_chron.index_select(0, chosen))

    return torch.cat(dense_parts, dim=0)

# ── Main ─────────────────────────────────────────────────────────────────


def _count_module_params(*modules):
    """Count module or standalone parameter tensors once across output branches."""
    seen = set()
    total = 0
    for module in modules:
        if module is None:
            continue
        if isinstance(module, torch.Tensor):
            params = (module,)
        elif hasattr(module, "parameters"):
            params = module.parameters()
        else:
            continue
        for param in params:
            if id(param) not in seen:
                seen.add(id(param))
                total += param.numel()
    return total


def _da3_direct_geometry_propagation_modules(codec):
    """Return learned DA3 modules used to propagate a level-0 latent to geometry."""
    if getattr(codec, "level", None) not in (0, -4):
        return []

    encoder = getattr(codec.rae, "encoder", None)
    backbone = getattr(encoder, "backbone", None)
    transformer = getattr(backbone, "pretrained", None)
    out_layers = getattr(encoder, "OUT_LAYERS", None)
    blocks = getattr(transformer, "blocks", None)
    if transformer is None or not out_layers or blocks is None:
        return []

    start_layer_idx = out_layers[getattr(codec, "level")]
    # `_forward_from_layer` resumes at start_layer_idx + 1, then applies final
    # normalization; camera_token is read when propagation reaches alt_start.
    return [
        *list(blocks[start_layer_idx + 1:]),
        getattr(transformer, "norm", None),
        getattr(transformer, "camera_token", None),
    ]

@torch.no_grad()
def main():
    args = parse_args()
    primary_output_timing = os.environ.get(
        "TIMING_PRIMARY_OUTPUT_ONLY", "1").lower() not in ("0", "false", "no")
    timing = {
        "diffusion": 0.0,
        "latent_vae_decode": 0.0,
        "rgb_geometry_decode": 0.0,
        "scenes": 0,
        "primary_output_active": False,
    }

    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed(fn, key):
        def wrapped(*fn_args, **fn_kwargs):
            if (key != "diffusion" and primary_output_timing
                    and not timing["primary_output_active"]):
                return fn(*fn_args, **fn_kwargs)
            _sync_cuda()
            started = time.perf_counter()
            result = fn(*fn_args, **fn_kwargs)
            _sync_cuda()
            timing[key] += time.perf_counter() - started
            return result
        return wrapped

    @contextmanager
    def _primary_output_timing():
        previous = timing["primary_output_active"]
        timing["primary_output_active"] = True
        try:
            yield
        finally:
            timing["primary_output_active"] = previous

    if args.timing_json:
        global sample_v4_euler, sample_v3_euler, decode_rgb_codec
        global decode_dpt, decode_to_depth
        sample_v4_euler = _timed(sample_v4_euler, "diffusion")
        sample_v3_euler = _timed(sample_v3_euler, "diffusion")
        decode_rgb_codec = _timed(decode_rgb_codec, "rgb_geometry_decode")
        decode_dpt = _timed(decode_dpt, "rgb_geometry_decode")
        decode_to_depth = _timed(decode_to_depth, "rgb_geometry_decode")

    save_artifacts = not args.metrics_only
    if args.metrics_only:
        args.save_pointcloud = False
    if args.save_pointcloud and args.pc_stride < 1:
        raise ValueError(f"--pc-stride must be >= 1 (got {args.pc_stride})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_output_dir, staged_output_dir = _maybe_stage_output_dir(args)
    V = args.num_views
    T = args.total_views if args.total_views is not None else V
    cond_num = args.cond_num
    if args.mode == "t2v":
        cond_num = 0
    # roll overlap is independent of t2v's cond_num=0; default to --cond-num CLI value
    roll_cond_num = (
        args.roll_cond_num if args.roll_cond_num is not None else args.cond_num
    )
    autoregress = T > V

    cond_num_range = None
    if args.cond_num_range:
        a, b = args.cond_num_range.split("-")
        cond_num_range = (int(a), int(b))
        if int(a) < 0 or int(b) < int(a):
            raise ValueError(f"Invalid --cond-num-range {args.cond_num_range!r}")

    first_cond_max = (
        cond_num_range[1] if cond_num_range else args.cond_num
    )
    if args.mode == "t2v":
        first_cond_max = 0

    _validate_rollout_cond(
        chunk_size=V,
        roll_cond_num=roll_cond_num,
        first_cond_max=first_cond_max,
        mode=args.mode,
        autoregress=autoregress,
    )
    if not autoregress and args.mode in ("recon", "generate"):
        if cond_num >= V:
            raise ValueError(
                f"--cond-num ({cond_num}) must be < --num-views ({V})"
            )
    if autoregress:
        if T <= V:
            raise ValueError(
                f"--total-views ({T}) must exceed --num-views ({V}) for rollout"
            )
        _assert_chunk_plan(T, V, roll_cond_num)
    collect_views = (
        _required_camera_frames(T, V, roll_cond_num) if autoregress else V
    )
    # FREE_ROLLOUT overrides the GT camera path with a synthetic trajectory and
    # only ever uses frame-0 (the cond image + its K), so requiring the full
    # rollout length of real frames per scene is pointless — and it makes the
    # collector reject/scan most short clips over slow S3-FUSE (hangs on re10k
    # where the target length rarely fits). Cap the collect requirement to one
    # chunk so scene selection stays fast; the synthetic path handles the rest.
    if os.environ.get("FREE_ROLLOUT") and args.mode in ("recon", "generate"):
        collect_views = min(collect_views, V)

    if args.loss_t.lower() == "shifted":
        loss_t_mode = "shifted"
        loss_t_fixed = None
    else:
        loss_t_mode = "fixed"
        loss_t_fixed = float(args.loss_t)

    if cond_num_range is not None and autoregress:
        if cond_num_range[1] >= V:
            raise ValueError(
                f"--cond-num-range upper bound ({cond_num_range[1]}) must be < "
                f"num-views ({V}) for autoregressive rollout"
            )
    torch.manual_seed(args.seed)

    # has_dpt will be re-evaluated after reading config stage_1 params
    has_dpt = (
        os.path.isfile(args.dpt_decoder) and os.path.isfile(args.da3_weights)
    )

    cfg = OmegaConf.load(args.config)
    latent_backend = str(cfg.get("latent_backend", "feature_vae")).lower()
    if latent_backend == "wan21":
        latent_backend = "wan2_1"
    gld_config_path = args.gld_config or cfg.get("gld_config")
    if gld_config_path:
        gld_cfg = OmegaConf.load(str(gld_config_path))
        vae_cfg = _to_container_or_empty(gld_cfg.get("codec"))
        if not vae_cfg:
            raise ValueError(f"No codec in gld config: {gld_config_path}")
        print(f"  [vae] codec from gld config: {gld_config_path}")
    else:
        vae_cfg = _to_container_or_empty(cfg.get("codec"))
    model_cfg = cfg.get("stage_2")
    model_params = _to_container_or_empty(model_cfg.get("params"))
    transport_cfg = _to_container_or_empty(cfg.get("transport", {}).get("params", {}))
    misc_cfg = _to_container_or_empty(cfg.get("misc"))
    training_cfg = _to_container_or_empty(cfg.get("training"))
    stage1_cfg_node = cfg.get("stage_1")
    stage1_cfg = _to_container_or_empty(stage1_cfg_node)
    stage1_params = _to_container_or_empty(stage1_cfg_node.get("params")) if stage1_cfg_node else {}

    # Resolution: derive from the trained encoder input size unless explicitly
    # overridden. The DA3 encoder, VAE whitening stats, and latent grid are all
    # pinned to stage_1.encoder_input_size at train time (252 here). Running eval
    # at a different resolution silently corrupts every downstream quantity
    # (latent h/w, time_dist_shift, feature distribution vs whitening stats).
    if args.resolution is not None:
        H, W = args.resolution
    else:
        if latent_backend in ("sd_vae", "wan2_1"):
            enc_size = int(training_cfg.get("image_size", 256))
        else:
            enc_size = int(
                stage1_params.get(
                    "encoder_input_size",
                    stage1_params.get("resolution", training_cfg.get("image_size", 252)),
                )
            )
        H = W = enc_size
        print(f"  [resolution] derived {H}×{W} from config")
    if latent_backend == "feature_vae" and (H % 14 != 0 or W % 14 != 0):
        raise ValueError(
            f"resolution {H}×{W} not divisible by 14 (DA3 patch). "
            "Latent grid would be wrong. Pass a 14-aligned --resolution.")

    # Model output convention (drives loss + sampler — must match the trainer).
    # yaml.transport.params.prediction is the source of truth. The default
    # `create_transport` only recognizes velocity/noise/score → "x" silently
    # falls through to VELOCITY, breaking both loss and sampler (samples look
    # like noise, loss numbers are 5–10× inflated). We dispatch on this flag
    # explicitly below: loss uses _rae_eval_loss, sampler converts x→v inline.
    prediction = str(transport_cfg.get("prediction", "velocity")).lower()
    if prediction not in ("velocity", "x"):
        raise ValueError(
            f"transport.params.prediction={prediction!r} not supported by eval "
            f"(only 'velocity' and 'x'). Add a new branch in _rae_eval_loss + "
            f"sample_v4_euler if you need 'noise'/'score'.")
    if prediction == "x" and args.sampler == "transport":
        raise ValueError(
            "yaml says prediction='x' but --sampler=transport uses the legacy "
            "Transport.sample_ode path which has no x→v conversion → samples "
            "will be garbage. Use --sampler euler_v3 (the default) for x-pred "
            "models, or fix Transport to know about ModelType.X.")
    time_dist_type = str(transport_cfg.get("time_dist_type", "logit-normal_0_1"))

    pose_origin = str(training_cfg.get("pose_origin", "last"))
    pose_origin_idx = 0 if pose_origin == "first" else -1
    pose_translation_norm = str(training_cfg.get("pose_translation_norm", "batch_max"))

    # Interval-aware pose translation scaling — must match training so the model
    # sees the same pose magnitude convention at inference. ``mean_gap`` is the
    # average real frame gap of the clip: keyframe clips sampled uniformly at
    # ``--sample-interval`` have mean_gap == sample_interval; the model-interp
    # stage spans one keyframe gap (== sample_interval) across interp_views slots.
    pose_interval_scale_enabled = bool(training_cfg.get("pose_interval_scale", False))
    pose_interval_canonical = float(training_cfg.get("pose_interval_canonical", 10.0))
    pose_interval_min_scale = float(training_cfg.get("pose_interval_min_scale", 0.1))
    pose_interval_max_scale = float(training_cfg.get("pose_interval_max_scale", 1.0))

    def _eval_interval_scale(mean_gap: float | None):
        if not pose_interval_scale_enabled:
            return None
        if mean_gap is None:
            raise ValueError(
                "--sample-interval is required when training.pose_interval_scale=true"
            )
        s = mean_gap / pose_interval_canonical
        return float(min(max(s, pose_interval_min_scale), pose_interval_max_scale))

    eval_pose_scale = _eval_interval_scale(args.sample_interval)
    if pose_interval_scale_enabled:
        print(f"  pose_interval:   scale enabled "
              f"(canonical={pose_interval_canonical:g}, "
              f"keyframe_scale={eval_pose_scale:g} @ interval={args.sample_interval})")

    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True) or {}
    ref_view_sampling = args.ref_view_sampling
    if ref_view_sampling is None:
        ref_view_sampling = str(dataset_cfg.get("ref_view_sampling", "prefix"))

    camera_conditioning = str(model_params.get("camera_conditioning", "plucker_flip_pe"))
    baseline_camera_mode = str(model_params.get("baseline_camera_mode", "plucker"))
    use_baseline_camera = camera_conditioning == "baseline_prope"
    config_no_camera = camera_conditioning == "none"
    if camera_conditioning not in ("plucker_flip_pe", "baseline_prope", "none"):
        raise ValueError(
            f"Unsupported stage_2.params.camera_conditioning={camera_conditioning!r}"
        )
    no_camera = args.no_camera or config_no_camera
    reference_conditioning = str(
        training_cfg.get("reference_conditioning", "clean_token_v4")
    )
    if reference_conditioning not in ("clean_token_v4", "state_v3"):
        raise ValueError(
            "training.reference_conditioning must be 'clean_token_v4' or "
            f"'state_v3', got {reference_conditioning!r}"
        )

    # denoise_all mode: auto-detect from config or CLI override
    if args.denoise_all is not None:
        denoise_all = args.denoise_all
    else:
        denoise_all = bool(training_cfg.get("denoise_all_views", False))
    ref_noise_frac = args.ref_noise_frac if args.ref_noise_frac is not None else \
        float(training_cfg.get("ref_noise_frac", 0.1))

    print("=" * 60)
    print("Latent Diffusion V3 Evaluation")
    print(f"  DiT ckpt:        {args.dit_ckpt}")
    print(f"  Weights:         {'EMA' if args.use_ema else 'raw model'}")
    print(f"  Mode:            {args.mode}")
    print(f"  pose_origin:     {pose_origin} (idx={pose_origin_idx})")
    print(f"  pose_trans_norm: {pose_translation_norm}")
    print(f"  ref_view_sampling: {ref_view_sampling}")
    cond_desc = (f"{cond_num_range[0]}-{cond_num_range[1]} (random)"
                 if cond_num_range else str(cond_num))
    print(f"  Scenes: {args.num_scenes}, Chunk views: {V}, Cond views: {cond_desc}")
    if autoregress:
        print(f"  Autoregressive:  total_views={T}, roll_cond={roll_cond_num}, "
              f"camera_frames={collect_views}")
    print(f"  Sample steps:    {args.sample_steps}, CFG: {args.cfg_scale}")
    _guid_extra = ""
    if args.guidance in ("ig", "cfg_ig"):
        _guid_extra = f"  (ig_scale={args.ig_scale})"
    print(f"  Guidance:        {args.guidance}{_guid_extra}")
    if args.guidance in ("ig", "cfg_ig") and args.sampler == "transport":
        print("  [WARN] guidance=ig/cfg_ig 仅 euler_v3 采样器支持; transport 采样器会忽略 IG。")
    print(f"  Sampler:         {args.sampler}  (eps={args.sample_eps})")
    print(f"  Prediction:      {prediction!r}  (loss + sampler dispatch)")
    print(f"  Camera cond:     {camera_conditioning}"
          + (f" (mode={baseline_camera_mode})" if use_baseline_camera else ""))
    print(f"  Reference cond:  {reference_conditioning}")
    print(f"  Time dist:       {time_dist_type!r}")
    if denoise_all:
        print(f"  Denoise all:     YES (ref_noise_frac={ref_noise_frac})")
    if args.save_pointcloud:
        print(f"  Point cloud:     enabled (deferred until all videos saved; "
              f"stride={args.pc_stride})")
    if args.ref_tgt_pc_gap:
        print(f"  Ref–tgt PC gap:  enabled (stride={args.pc_stride})")
    if args.metrics_only:
        print("  Metrics only:    YES (skip videos/images/pointclouds)")
    print(f"  Output:          {args.output_dir}")
    print("=" * 60)

    use_amp = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    rae = None

    # ── 1. Stage-1 codec ──
    print("\n[1/4] Loading stage-1 codec ...")

    # Pre-cache da3_weights to local-ssd (safetensors loaded inside DA3Backbone init)
    def _cache_file(path):
        _cbase = _eval_cache_base()
        if path and os.path.isfile(path) and not path.startswith(_cbase):
            cache_dir = os.path.join(_cbase, "_eval_cache")
            os.makedirs(cache_dir, exist_ok=True)
            parent = os.path.basename(os.path.dirname(path))
            local = os.path.join(cache_dir, "%s_%s" % (parent, os.path.basename(path)))
            if not os.path.isfile(local):
                import shutil
                print("  [cache] %s -> %s ..." % (path, local), end=" ", flush=True)
                _tmp = "%s.tmp.%d" % (local, os.getpid())
                shutil.copy2(path, _tmp)
                os.replace(_tmp, local)
                print("done (%.0f MB)" % (os.path.getsize(local) / 1e6))
            else:
                print("  [cache] Using %s" % local)
            return local
        return path

    if latent_backend == "sd_vae":
        sd_vae_model = str(cfg.get("sd_vae_model", "stabilityai/sd-vae-ft-mse"))
        if not os.path.isdir(sd_vae_model):
            local_sd_candidates = [
                os.path.join(ROOT, "pretrained_models", "sd-vae-ft-mse"),
                os.path.join(ROOT, "pretrained_models", "sd_vae_ft_mse"),
                "/local-ssd/hf_cache/models--stabilityai--sd-vae-ft-mse",
            ]
            for cand in local_sd_candidates:
                if os.path.isfile(os.path.join(cand, "config.json")):
                    sd_vae_model = cand
                    break
        print(f"  SD VAE: {sd_vae_model}")
        if not os.path.isdir(sd_vae_model) and os.environ.get("HF_HUB_OFFLINE") == "1":
            print("  [sd_vae] local cache not found; temporarily enabling HF download for SD VAE")
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                import huggingface_hub.constants as hf_constants
                hf_constants.HF_HUB_OFFLINE = False
            except Exception:
                pass
        vae = AutoencoderKL.from_pretrained(sd_vae_model, torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        vae.requires_grad_(False)
        has_dpt = False
        has_rgb = True
        backbone_norm = None
        encoder_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        encoder_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print(f"  SD VAE ready: scaling_factor={getattr(vae.config, 'scaling_factor', 0.18215):.5f}")
    elif latent_backend == "wan2_1":
        wan_vae_path = str(
            cfg.get("wan_vae_path", os.environ.get("WAN_VAE_PATH", ""))
        )
        if not os.path.isdir(wan_vae_path):
            raise FileNotFoundError(f"wan_vae_path not found: {wan_vae_path!r}")
        if os.path.isdir("/local-ssd") and not wan_vae_path.startswith("/local-ssd"):
            wan_key = os.path.abspath(wan_vae_path.rstrip("/"))
            wan_digest = hashlib.sha1(wan_key.encode("utf-8")).hexdigest()[:12]
            local_wan = f"/local-ssd/_eval_cache/wan_vae_{wan_digest}"
            if not os.path.isdir(local_wan):
                print(f"  [cache] {wan_vae_path} -> {local_wan} ...", flush=True)
                tmp_wan = f"{local_wan}.staging_{os.getpid()}"
                if os.path.isdir(tmp_wan):
                    shutil.rmtree(tmp_wan)
                shutil.copytree(wan_vae_path, tmp_wan)
                os.replace(tmp_wan, local_wan)
            wan_vae_path = local_wan
        print(f"  Wan2.1 VAE: {wan_vae_path}")
        vae = AutoencoderKLWan.from_pretrained(wan_vae_path, torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        vae.requires_grad_(False)
        has_dpt = False
        has_rgb = True
        backbone_norm = None
        encoder_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        encoder_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        print("  Wan2.1 VAE ready")
    elif latent_backend == "da3_direct":
        # Raw-DA3 (level-0) baseline: DA3Backbone (frozen, reshape_to_2d) + a
        # pretrained trunk+RGBHead. Diffusion runs on the raw level-0 feature;
        # RGB via decode_rgb, depth via forward-propagation + DPT.
        enc_pretrained = _resolve_da3_encoder_path(
            stage1_params.get("encoder_pretrained_path", "depth-anything/DA3-GIANT-1.1"),
            stage1_params.get("da3_weights_path") or args.da3_weights,
        )
        enc_pretrained = _cache_da3_dir(enc_pretrained)
        rae_kwargs = dict(
            encoder_pretrained_path=enc_pretrained,
            encoder_input_size=stage1_params.get("encoder_input_size", H),
            encoder_type=stage1_params.get("encoder_type", "DA3EncoderDirect"),
            reshape_to_2d=True,
            dpt_model_type=stage1_params.get("dpt_model_type", "da3-giant"),
        )
        da3_wt = _cache_file(stage1_params.get("da3_weights_path") or args.da3_weights)
        dpt_dec = _cache_file(stage1_params.get("dpt_decoder_path") or args.dpt_decoder)
        if da3_wt and os.path.isfile(da3_wt):
            rae_kwargs["da3_weights_path"] = da3_wt
            rae_kwargs["dpt_decoder_path"] = dpt_dec
        print(f"  DA3 encoder (da3_direct): {rae_kwargs['encoder_pretrained_path']} "
              f"(input={rae_kwargs['encoder_input_size']}, dpt={rae_kwargs['dpt_model_type']})")
        rae = DA3Backbone(**rae_kwargs).to(device).eval()
        rae.requires_grad_(False)
        da3_direct_cfg = _to_container_or_empty(cfg.get("da3_direct"))
        vae = DA3DirectCodec(
            rae=rae,
            level=int(da3_direct_cfg.get("level", 0)),
            trunk_dim=int(da3_direct_cfg.get("trunk_dim", 1024)),
            num_trunk_blocks=int(da3_direct_cfg.get("num_trunk_blocks", 4)),
            trunk_heads=int(da3_direct_cfg.get("trunk_heads", 8)),
            rgb_decoder=da3_direct_cfg.get("rgb_decoder"),
            latent_stats_path=da3_direct_cfg.get("latent_stats_path"),
        ).to(device).eval()
        vae.requires_grad_(False)
        if args.timing_json:
            # Level-0 geometry is generated by decode_depth(): latent
            # propagation through DA3 attention blocks followed by DPT decode.
            # Time the complete path rather than only the final DPT head.
            vae.decode_depth = _timed(vae.decode_depth, "rgb_geometry_decode")
        rgb_ckpt = str(args.vae_ckpt or cfg.get("da3_direct_rgb_checkpoint", "") or "")
        rgb_ckpt_local = _cache_file(rgb_ckpt) if rgb_ckpt else ""
        if rgb_ckpt_local and os.path.isfile(rgb_ckpt_local):
            state = _fast_load(rgb_ckpt_local, map_location="cpu")
            rgb_sd = state.get("ema_codec", state.get("codec", state))
            m, u = vae.load_state_dict(rgb_sd, strict=False)
            rgb_bad = [k for k in u if k.startswith(("rgb_head", "dec_"))]
            if rgb_bad:
                raise RuntimeError(
                    f"da3_direct RGB head load mismatch ({len(rgb_bad)} keys), "
                    f"e.g. {rgb_bad[:3]}")
            print(f"  da3_direct RGB head loaded from {rgb_ckpt_local}")
        else:
            print(f"  [WARN] da3_direct_rgb_checkpoint missing ({rgb_ckpt!r}); "
                  "RGB decode uses random-init head")
        has_dpt = rae.rae_cl_decoder is not None
        has_rgb = True
        encoder_mean = rae.encoder_mean
        encoder_std = rae.encoder_std
        backbone_norm = rae.encoder.backbone.pretrained.norm
        print(f"  DA3DirectCodec ready (latent_dim={vae.latent_dim}, level={vae.level})")
    else:
        # Read DA3 config from stage_1 (supports DA3-Base and DA3-Large)
        # Same approach as eval_wan_vs_feature_vaev2.py
        enc_pretrained = _resolve_da3_encoder_path(
            stage1_params.get("encoder_pretrained_path", "depth-anything/DA3-Base"),
            stage1_params.get("da3_weights_path") or args.da3_weights,
        )
        # Mirror the encoder dir to /local-ssd so from_pretrained reads weights off
        # NVMe instead of the slow NFS "Loading weights from local directory" path.
        enc_pretrained = _cache_da3_dir(enc_pretrained)
        rae_kwargs = dict(
            encoder_pretrained_path=enc_pretrained,
            encoder_input_size=stage1_params.get("encoder_input_size", H),
            encoder_type=stage1_params.get("encoder_type", "DA3EncoderDirect"),
            reshape_to_2d=False,
            dpt_model_type=stage1_params.get("dpt_model_type", "da3-base"),
        )
        da3_wt = _cache_file(stage1_params.get("da3_weights_path") or args.da3_weights)
        dpt_dec = _cache_file(stage1_params.get("dpt_decoder_path") or args.dpt_decoder)
        if da3_wt and os.path.isfile(da3_wt):
            rae_kwargs["da3_weights_path"] = da3_wt
            rae_kwargs["dpt_decoder_path"] = dpt_dec
        elif has_dpt:
            rae_kwargs["dpt_decoder_path"] = _cache_file(args.dpt_decoder)
            rae_kwargs["da3_weights_path"] = _cache_file(args.da3_weights)
        print(f"  DA3 encoder: {rae_kwargs['encoder_pretrained_path']} "
              f"(input={rae_kwargs['encoder_input_size']}, dpt_model={rae_kwargs['dpt_model_type']})")
        rae = DA3Backbone(**rae_kwargs).to(device).eval()
        has_dpt = rae.rae_cl_decoder is not None
        encoder_mean = rae.encoder_mean
        encoder_std = rae.encoder_std
        backbone_norm = rae.encoder.backbone.pretrained.norm

        # ── 2. VAE ──
        print("[2/4] Loading Feature VAE v2 ...")
        vae_ckpt_path = str(args.vae_ckpt or cfg.get("vae_checkpoint"))
        print(f"  vae ckpt: {vae_ckpt_path}")
        vae_ckpt = _fast_load(vae_ckpt_path, map_location="cpu")
        vae_sd = _unwrap_codec_state(vae_ckpt)

        # rgb_head: training yaml strips rgb_decoder; auto-detect spatial depth and
        # optional temporal blocks from ckpt keys before building the head.
        _inject_rgb_decoder_from_ckpt(
            vae_cfg,
            vae_sd,
            temporal_mode=args.rgb_head_temporal,
        )
        vae_cfg.pop("rgb_decoder_hidden", None)
        vae_cfg.pop("rgb_decoder_heads", None)

        vae = GAECodec(**vae_cfg).to(device).eval()
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        if args.timing_json:
            vae._decode_trunk = _timed(vae._decode_trunk, "latent_vae_decode")
            # dec_conv is a registered nn.Module; replacing the module itself
            # with a function is invalid. Wrap its bound forward method instead.
            vae.dec_conv.forward = _timed(
                vae.dec_conv.forward, "latent_vae_decode")
            vae.denormalize_and_split = _timed(
                vae.denormalize_and_split, "latent_vae_decode")
            if vae.rgb_head is not None:
                vae.rgb_head.forward = _timed(
                    vae.rgb_head.forward, "rgb_geometry_decode")
        has_rgb = vae.rgb_head is not None
        # Fail loud if rgb_head shape disagrees with ckpt — a silent strict=False
        # truncation produces grid / oil-painting artifacts (see eval_t2i notes).
        rgb_bad = [k for k in (*missing, *unexpected) if k.startswith("rgb_head")]
        if has_rgb and rgb_bad:
            raise RuntimeError(
                f"rgb_head load mismatch ({len(rgb_bad)} keys), e.g. {rgb_bad[:3]}. "
                "RGBHead config disagrees with ckpt — decode would be corrupted.")
        print(f"  latent_dim={vae.latent_dim}  RGB head: {'OK' if has_rgb else 'no'}"
              f"  (missing={len(missing)} unexpected={len(unexpected)})")

    # ── 3. DDTHead ──
    print("[3/4] Loading DDTHead ...")
    # PatchEmbed / init-time RoPE need latent grid size. Trainer injects
    # input_size=encoder_input_size//14; without it patch_size>1 → num_patches=0.
    if model_cfg.get("params", {}).get("input_size") is None:
        latent_h = H // 14
        if "params" not in model_cfg:
            model_cfg["params"] = {}
        model_cfg["params"]["input_size"] = latent_h
        print(f"  [dit] auto input_size={latent_h} (from {H}×{W} RGB)")
    dit = instantiate_from_config(model_cfg).to(device).eval()
    dit_ckpt = _fast_load(args.dit_ckpt, map_location="cpu")
    # Released checkpoints are bare state_dicts; training checkpoints wrap the
    # weights under 'ema' / 'model' / 'state_dict'.
    _wrappers = ("ema", "model", "state_dict") if args.use_ema else ("model", "ema", "state_dict")
    dit_key, dit_sd = "state_dict", None
    for _k in _wrappers:
        _inner = dit_ckpt.get(_k)
        if isinstance(_inner, dict) and any(torch.is_tensor(v) for v in _inner.values()):
            dit_key, dit_sd = _k, _inner
            break
    if dit_sd is None:
        if not any(torch.is_tensor(v) for v in dit_ckpt.values()):
            raise KeyError(
                f"{args.dit_ckpt} holds no tensors under {_wrappers} nor at the top "
                "level; is this a flow checkpoint?")
        dit_key, dit_sd = "<raw state_dict>", dit_ckpt
    elif args.use_ema and dit_key != "ema":
        print(f"  [WARN] no 'ema' key → using '{dit_key}'")
    model_keys = set(dit.state_dict().keys())
    stale = [k for k in dit_sd if k not in model_keys]
    if stale:
        print(f"  [WARN] dropping {len(stale)} stale checkpoint key(s): {stale}")
        for k in stale:
            del dit_sd[k]
    missing, unexpected = dit.load_state_dict(dit_sd, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} model key(s) not in ckpt (random init): "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected ckpt key(s) after load")
    n_params = sum(p.numel() for p in dit.parameters()) / 1e6
    print(f"  Loaded '{dit_key}' — {n_params:.1f}M params")

    # ── Transport ──
    in_ch = int(model_params.get("in_channels", 384))
    if latent_backend in ("sd_vae", "wan2_1"):
        latent_h = H // 8
        latent_w = W // 8
    else:
        latent_h = H // 14
        latent_w = W // 14
    s_patch_size = int(getattr(dit, "s_patch_size", model_params.get("patch_size", 1)))
    plucker_h = latent_h // s_patch_size
    plucker_w = latent_w // s_patch_size
    if s_patch_size > 1:
        print(
            f"  [dit] patch_size={s_patch_size} → plucker token grid "
            f"{plucker_h}×{plucker_w} (latent {latent_h}×{latent_w})"
        )
    # time_dist_shift MUST be computed exactly as the trainer does
    # (as train_flow.py does): explicit misc.time_dist_shift wins,
    # else shift_from_latent_dim(misc.time_dist_shift_dim or prod(latent),
    # base=misc.time_dist_shift_base). The old eval path used a hardcoded
    # /4096 base and ignored time_dist_shift_dim/base → at the wrong resolution
    # it produced 6.36 instead of the trained 3.18.
    if args.time_dist_shift is not None:
        time_dist_shift = float(args.time_dist_shift)
        shift_src = "cli"
    elif "time_dist_shift" in misc_cfg:
        time_dist_shift = float(misc_cfg["time_dist_shift"])
        shift_src = "yaml.misc.time_dist_shift"
    else:
        shift_dim = int(misc_cfg.get("time_dist_shift_dim",
                                     math.prod((in_ch, latent_h, latent_w))))
        shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
        time_dist_shift = shift_from_latent_dim(shift_dim, base=shift_base)
        shift_src = f"shift_from_latent_dim(dim={shift_dim}, base={shift_base})"
    print(f"  time_dist_shift: {time_dist_shift:.4f}  [{shift_src}]"
          f"  (used by {'euler_v3 sampler + loss' if args.sampler == 'euler_v3' else 'transport ODE + loss'})")
    transport = None
    if args.sampler == "transport":
        try:
            from stage2.transport import create_transport
        except ImportError as exc:
            raise RuntimeError(
                "--sampler transport is not shipped in the public release; "
                "use the default --sampler euler_v3"
            ) from exc
        transport = create_transport(**transport_cfg, time_dist_shift=time_dist_shift)

    latent_mean, latent_std, latent_whiten, latent_unwhiten = None, None, None, None
    stats_path = str(cfg.get("latent_stats", ""))
    if latent_backend == "da3_direct" and stats_path:
        print("  [da3_direct] ignoring framework latent_stats; codec self-normalizes")
    elif stats_path and os.path.isfile(stats_path):
        stats = _fast_load(stats_path, map_location="cpu")
        latent_mean = stats["mean"].float().to(device).reshape(1, -1, 1, 1)
        latent_std = stats["std"].float().to(device).reshape(1, -1, 1, 1).clamp(min=1e-5)
        if "whiten" in stats:
            latent_whiten = stats["whiten"].float().to(device)         # (C, C)
            latent_unwhiten = stats["unwhiten"].float().to(device)     # (C, C)
            print(f"  Latent stats: FULL-COV whitening enabled "
                  f"(W shape={tuple(latent_whiten.shape)})")
        else:
            print(f"  Latent stats: per-channel only (legacy)")

    # ── Text embeddings (from config, same as training) ──
    # t2v + --prompt 时用在线编码器，跳过预计算 embedding 的加载
    use_text = cfg.get("use_text", True)
    text_cfg = OmegaConf.to_container(cfg.get("text_encoder", {}), resolve=True) if use_text else {}
    text_emb_dict = None
    if args.mode == "t2v" and args.prompt:
        print(f"  Text embeddings: skipped (t2v with --prompt, will encode on-the-fly)")
    else:
        text_embeddings_path = str(text_cfg.get("embeddings_path", "")) if text_cfg else ""
        if text_embeddings_path and os.path.isfile(text_embeddings_path):
            raw = _fast_load(text_embeddings_path, map_location="cpu")
            text_emb_dict = raw["caption_to_emb"] if isinstance(raw, dict) and "caption_to_emb" in raw else raw
            print(f"  Text embeddings: {len(text_emb_dict)} captions from {text_embeddings_path}")
        else:
            print(f"  Text embeddings: none (use_text={use_text})")

    # Scene→caption mapping (same json files the dataset uses at training time)
    import json as _json
    _scene_caption_map = {}
    _caption_root = os.environ.get("GAE_DATA_ROOT", "/data/gae")
    for _cj in [os.path.join(_caption_root, "re10k_packed/train_caption.json"),
                os.path.join(_caption_root, "re10k_packed/test_caption.json")]:
        if os.path.isfile(_cj):
            with open(_cj) as _f:
                _scene_caption_map.update(_json.load(_f))
    if _scene_caption_map:
        print(f"  Scene→caption map: {len(_scene_caption_map)} scenes")

    def _lookup_text_emb(scene_name):
        """scene_name → caption (via json) → text embedding (via precomputed dict)."""
        if text_emb_dict is None:
            return None
        scene_id = scene_name.split("/")[-1]
        caption = _scene_caption_map.get(scene_id) or _scene_caption_map.get(scene_name)
        if caption is None:
            return None
        # Try exact match, then stripped (caption json may have trailing \n)
        for key in [caption, caption.strip(), caption.rstrip("\n")]:
            emb = text_emb_dict.get(key)
            if emb is not None:
                return emb.unsqueeze(0).to(device)
        return None

    # ── Qwen3 online text encoder (RAE recipe) ──────────────────────────────
    # CRITICAL train/eval parity: DiT cross-attn only fires when ref_global is
    # not None (ddt_head.py: `if self.use_cross_attn and ref_global is not None`).
    # The trainer ALWAYS passes a real tensor — either the Qwen3 encoding of the
    # scene caption, or the Qwen3 encoding of "" (null_text_tokens) for
    # CFG-dropout / blank-caption scenes. If eval passes None, cross-attn is
    # skipped entirely → the model runs in a regime it never saw → inflated loss
    # + garbage RGB. So when the config's text encoder is Qwen*, we load it here
    # and feed the real caption (empty → null token), mirroring training.
    _qwen3_encoder = None
    _qwen3_null_tokens = None
    _text_model_name = str(text_cfg.get("model_name", "")) if text_cfg else ""
    if use_text and args.no_text:
        print("[text] disabled by --no-text (cross-attn off)")
    elif use_text and "qwen" in _text_model_name.lower():
        _text_model_path = _resolve_text_model_path(_text_model_name)
        _text_model_path = _cache_pretrained_dir_to_local_ssd(_text_model_path, "qwen3")
        _text_is_local = (
            os.path.isdir(_text_model_path)
            and os.path.isfile(os.path.join(_text_model_path, "config.json"))
        )
        if _text_model_path != _text_model_name:
            print(
                "[text] Loading Qwen3 encoder for RAE cross-attn parity: "
                f"{_text_model_name} -> {_text_model_path}"
            )
        else:
            print(f"[text] Loading Qwen3 encoder for RAE cross-attn parity: {_text_model_name}")
        from stage2.models.text_encoder import Qwen3TextEncoder, prefill_hf_cache
        # Mirror the trainer (train_flow.py): single-process prefill, then
        # offline load. Without this, from_pretrained races HTTP on a partial cache
        # (tokenizer-only snapshot) and sporadically SSL-handshake-times out.
        if _text_is_local:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        elif (
            os.environ.get("HF_HUB_OFFLINE", "0") != "1"
            and os.environ.get("SKIP_HF_PREFILL", "0") != "1"
        ):
            print(f"  [hf-cache] pre-fetching {_text_model_path} to "
                  f"{os.environ.get('HUGGINGFACE_HUB_CACHE', '~/.cache/huggingface')} ...")
            prefill_hf_cache(_text_model_path, torch_dtype=torch.bfloat16)
            print("  [hf-cache] pre-fetch done")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _qwen3_encoder = Qwen3TextEncoder(
            model_name=_text_model_path,
            max_length=int(text_cfg.get("max_length", 256)),
            torch_dtype=torch.bfloat16,
        ).to(device).eval()
        for p in _qwen3_encoder.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            _qwen3_null_tokens = _qwen3_encoder([""])["tokens"].to(device)  # (1, T, 1024)
        print(f"  Qwen3 loaded (null-token shape={tuple(_qwen3_null_tokens.shape)})")

    def _read_scene_caption(scene_dir, scene_name):
        """RE10K_Packed / scannetpp caption: meta.json['caption'] or caption.txt."""
        cap = None
        meta_p = os.path.join(scene_dir, "meta.json")
        if os.path.isfile(meta_p):
            try:
                with open(meta_p) as _f:
                    cap = _json.load(_f).get("caption")
            except Exception:
                cap = None
        if not cap:
            cap_p = os.path.join(scene_dir, "caption.txt")
            if os.path.isfile(cap_p):
                try:
                    with open(cap_p) as _f:
                        cap = _f.read().strip()
                except Exception:
                    cap = None
        if not cap:
            # Fall back to the legacy json map (re10k sharded layout).
            sid = scene_name.split("/")[-1]
            cap = _scene_caption_map.get(sid) or _scene_caption_map.get(scene_name)
        return cap

    def _resolve_ref_global(scene_name, scene_dir):
        """Unified ref_global for the loss + sampler, matching training policy.

        Priority:
          1. Qwen3 (RAE recipe): encode the scene caption; empty → null token.
             NEVER None — cross-attn must stay active as in training.
          2. Legacy UMT5 precomputed dict (older configs).
          3. None (no text conditioning configured).
        """
        if _qwen3_encoder is not None:
            cap = _read_scene_caption(scene_dir, scene_name)
            if cap and cap.strip():
                with torch.no_grad():
                    return _qwen3_encoder([cap.strip()])["tokens"].to(device)
            return _qwen3_null_tokens
        return _lookup_text_emb(scene_name)

    # ── T2V: on-the-fly text encoder for --prompt ──
    # RAE configs use Qwen3 (already loaded above) → reuse it. Only fall back to
    # UMT5 for legacy configs whose DiT was trained with UMT5 (cross_attn_kdim
    # 4096). Loading UMT5 against a Qwen3-trained 1024-dim model would crash.
    _t2v_text_encoder = None
    _t2v_tokenizer = None
    if args.mode == "t2v" and args.prompt and _qwen3_encoder is None:
        print("[T2V] Loading UMT5 text encoder ...")
        from transformers import AutoTokenizer, UMT5EncoderModel
        _t2v_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        _t2v_text_encoder = UMT5EncoderModel.from_pretrained(args.text_encoder).to(device).eval()
        for p in _t2v_text_encoder.parameters():
            p.requires_grad_(False)
        print(f"  UMT5 loaded (d_model={_t2v_text_encoder.config.d_model})")

    def _encode_prompt(prompt_str):
        """Encode a raw text prompt → (1, T, D) embedding tensor."""
        if _qwen3_encoder is not None:
            with torch.no_grad():
                return _qwen3_encoder([prompt_str])["tokens"].to(device)
        tokens = _t2v_tokenizer(
            [prompt_str], padding="max_length", max_length=args.text_max_length,
            truncation=True, return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            return _t2v_text_encoder(**tokens).last_hidden_state  # (1, T, D)

    # ── 4. Evaluate ──
    if args.scene_manifest:
        import json as _json_manifest
        with open(args.scene_manifest) as _mf:
            _manifest_meta = _json_manifest.load(_mf)
        manifest_ds = _manifest_meta.get("dataset", args.dataset if args.dataset != "all" else "re10k")
        datasets_to_eval = [(manifest_ds, DATASET_ROOTS.get(manifest_ds, args.data_root or ""))]
    elif args.data_root:
        # Preserve the dataset type when overriding the root. This is important
        # for re10k_packed: routing it as "custom" would fall back to the legacy
        # flat-PNG RE10K collector and hang/scan the packed train tree.
        ds_name = args.dataset if args.dataset != "all" else "custom"
        datasets_to_eval = [(ds_name, args.data_root)]
    elif args.dataset == "all":
        datasets_to_eval = [
            ("re10k", DATASET_ROOTS["re10k"]),
            ("dl3dv", DATASET_ROOTS["dl3dv"]),
            ("mvssynth", DATASET_ROOTS["mvssynth"]),
        ]
    else:
        datasets_to_eval = [(args.dataset, DATASET_ROOTS[args.dataset])]

    denorm = (lambda t: (t * encoder_std.squeeze(0)
                         + encoder_mean.squeeze(0)).clamp(0, 1))

    cfg_interval = (0.0, 1.0)
    val_guidance = OmegaConf.to_container(
        cfg.get("validation", {}).get("guidance", {}), resolve=True)
    if val_guidance:
        cfg_interval = (
            float(val_guidance.get("t_min", 0.0)),
            float(val_guidance.get("t_max", 1.0)),
        )

    global_results = {}

    for ds_name, ds_root in datasets_to_eval:
        if args.scene_manifest:
            scenes = load_scenes_from_manifest(
                args.scene_manifest, num_views=collect_views,
                interval=args.sample_interval)
            ds_out_dir = (args.output_dir if ds_name == "custom"
                          else os.path.join(args.output_dir, ds_name))
            os.makedirs(ds_out_dir, exist_ok=True)
            print(f"\n{'─' * 50}")
            _view_desc = (f"{T} out / {collect_views} cams (chunk={V})"
                          if autoregress else f"{V} views")
            print(f"Dataset: {ds_name} (manifest: {len(scenes)} scenes × {_view_desc})")
            print(f"  manifest: {args.scene_manifest}")
            print(f"{'─' * 50}")
            if not scenes:
                raise SystemExit(
                    f"[fatal] scene manifest loaded 0 scenes "
                    f"(path={args.scene_manifest}, num_views={collect_views}, "
                    f"interval={args.sample_interval}). Usually the manifest "
                    f"was built with too-large SAMPLE_INTERVAL for this view "
                    f"count, or all scene_dir paths are missing on this node. "
                    f"Fix: use V>=17 intervals (scannetpp/re10k iv=1) and "
                    f"SCENE_MANIFEST_REGEN=1, then rerun."
                )
        elif ds_name == "custom":
            scenes = collect_scenes_re10k(
                ds_root, args.num_scenes, collect_views, args.seed,
                interval=args.sample_interval)
            ds_out_dir = args.output_dir
        else:
            scenes = collect_scenes(
                ds_name, ds_root, args.num_scenes, collect_views, args.seed,
                interval=args.sample_interval)
            ds_out_dir = os.path.join(args.output_dir, ds_name)
            os.makedirs(ds_out_dir, exist_ok=True)
            print(f"\n{'─' * 50}")
            _view_desc = (f"{T} out / {collect_views} cams (chunk={V})"
                          if autoregress else f"{V} views")
            print(f"Dataset: {ds_name} ({len(scenes)} scenes × {_view_desc})")
            print(f"{'─' * 50}")

        all_rgb, all_depth, all_feat = [], [], []
        all_rgb_vae_recon = []
        all_ref_tgt_pc_gap: list[float] = []
        all_loss = []
        rng = random.Random(args.seed)
        deferred_pc_jobs: list[dict] = []
        _pc_gap_rng = np.random.default_rng(args.seed)

        _shard_n = int(os.environ.get("GEO_CACHE_NUM_SHARDS", "0") or "0")
        _shard_i = int(os.environ.get("GEO_CACHE_SHARD_INDEX", "0") or "0")
        for s_idx, (scene_name, scene_dir, img_names, ds_type) in enumerate(scenes):
            # Multi-GPU/multi-node sharding: each worker owns the scenes with
            # s_idx % NUM_SHARDS == SHARD_INDEX (round-robin, embarrassingly
            # parallel). Silent skip so the shard's log stays readable.
            if _shard_n > 1 and (s_idx % _shard_n) != _shard_i:
                continue
            # Resume: skip scenes whose immutable geo-cache npz already exists
            # (lets a crashed/OOM'd cache run be re-launched to fill the rest).
            _gcd_resume = os.environ.get("GEO_CACHE_DIR")
            if _gcd_resume:
                _dsn = os.environ.get("GEO_CACHE_DS_NAME") or ds_name
                if os.path.exists(
                        os.path.join(_gcd_resume, _dsn, f"{s_idx:03d}.npz")):
                    print(f"  [{s_idx+1}/{len(scenes)}] {scene_name}: "
                          f"geo-cache {_dsn}/{s_idx:03d}.npz exists — skip")
                    continue
            actual_v = len(img_names)
            expected_frames = collect_views if autoregress else V
            if actual_v != expected_frames:
                print(f"  [WARN] Scene {scene_name} has {actual_v} views, "
                      f"expected {expected_frames}. Skipping.")
                continue
            scene_cond = cond_num
            if cond_num_range is not None:
                scene_cond = rng.randint(cond_num_range[0], cond_num_range[1])

            _view_msg = (f"{T} out / {actual_v} cams, chunk={V}, roll={roll_cond_num}"
                         if autoregress else f"{actual_v} views")
            print(f"  [{s_idx+1}/{len(scenes)}] {scene_name} "
                  f"({_view_msg}, cond={scene_cond})")

            basenames = [os.path.splitext(n)[0] for n in img_names]

            # ScanNet++ and RE10K_Packed both read from a single mp4; pre-decode
            # all requested frames in one cv2 sorted-pass so the per-frame loop
            # is cache hits. Re-opening the mp4 per frame over S3-FUSE is
            # unusably slow (~seconds each after the initial keyframe seek).
            if ds_type in ("scannetpp", "re10k_packed"):
                prefetch_scannetpp_frames(scene_dir, [int(bn) for bn in basenames])

            img_tensors, intri_list, pose_list = [], [], []
            try:
                for bn in basenames:
                    img_t, intr, pose = load_image_and_camera(
                        scene_dir, bn, (H, W), ds_type=ds_type)
                    img_tensors.append(img_t)
                    intri_list.append(intr)
                    pose_list.append(pose)
            except (KeyError, OSError, ValueError, RuntimeError) as exc:
                # Pose-less parent metas (empty frames[]) used to hard-crash the
                # whole job mid-run. Skip the scene and keep evaluating the rest.
                print(f"  [WARN] Scene {scene_name}: failed to load views "
                      f"({type(exc).__name__}: {exc}); skipping")
                continue

            imgs = torch.stack(img_tensors).to(device)
            imgs_01 = denorm(imgs)
            view_frame_idx = torch.arange(actual_v, dtype=torch.long, device=device)

            # ── FREE (unbounded) rollout: replace the GT camera path with a
            # synthetic, arbitrarily-long trajectory so the demo can "keep
            # extending" from a single conditioning image, independent of the
            # source clip length. Only frame-0 (the cond image + its K) is kept;
            # the rest of pose_list is synthesised. Downstream GT/metrics/PC are
            # skipped (we `continue` after the long _pred.mp4). Gated on env so
            # normal eval/cache runs are completely unaffected.
            _free = (bool(os.environ.get("FREE_ROLLOUT"))
                     and autoregress and args.mode in ("recon", "generate"))
            if _free:
                _tv = int(os.environ.get("FREE_ROLLOUT_VIEWS", str(T)))
                T = _tv
                _ncam = _required_camera_frames(T, V, roll_cond_num)
                _anchor = np.asarray(pose_list[0], dtype=np.float64)
                _K0 = intri_list[0]
                pose_list = _synthesize_free_trajectory(
                    _anchor, _ncam,
                    motion=os.environ.get("FREE_ROLLOUT_MOTION", "wander"),
                    speed=float(os.environ.get("FREE_ROLLOUT_SPEED", "0.06")),
                    yaw_deg=float(os.environ.get("FREE_ROLLOUT_YAW_DEG", "24")),
                    pitch_deg=float(os.environ.get("FREE_ROLLOUT_PITCH_DEG", "6")),
                    fwd_sign=float(os.environ.get("FREE_ROLLOUT_FWD_SIGN", "1")),
                    seed=int(args.seed) + s_idx,
                )
                intri_list = [_K0 for _ in range(_ncam)]
                img_tensors = [img_tensors[0]]
                imgs = imgs[:1]
                imgs_01 = imgs_01[:1]
                actual_v = _ncam
                print(f"    [free-rollout] synthetic '{os.environ.get('FREE_ROLLOUT_MOTION','wander')}' "
                      f"trajectory: {T} frames from {_ncam} cam poses "
                      f"(chunk={V}, roll={roll_cond_num}, cond={scene_cond})")

            # ── Autoregressive long-sequence rollout (chunk=num_views, overlap=roll_cond) ──
            if autoregress and not args.loss_only:
                first_cond = scene_cond if args.mode in ("recon", "generate") else 0
                plucker_full = None
                baseline_full: dict = {}
                x_norm = feats_all = None

                if args.mode == "t2v":
                    if save_artifacts and not no_camera:
                        visualize_trajectory(
                            pose_list, 0,
                            os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))
                    if no_camera:
                        plucker_full = None
                        baseline_full = {}
                    elif use_baseline_camera:
                        plucker_full = None
                        baseline_full = build_baseline_cam_from_cameras(
                            intri_list, pose_list, H, W, device,
                            cond_num=0, total_view=actual_v,
                            origin_idx=pose_origin_idx,
                            baseline_camera_mode=baseline_camera_mode,
                            translation_norm=pose_translation_norm,
                            pose_scale=eval_pose_scale,
                        )
                    else:
                        plucker_full = build_plucker_from_cameras(
                            intri_list, pose_list, H, W,
                            patches_y=plucker_h, patches_x=plucker_w,
                            device=device, origin_idx=pose_origin_idx,
                            translation_norm=pose_translation_norm,
                            pose_scale=eval_pose_scale,
                        )
                    if args.prompt:
                        text_emb = _encode_prompt(args.prompt)
                        print(f"    Prompt: {args.prompt[:80]}")
                    else:
                        text_emb = _resolve_ref_global(scene_name, scene_dir)
                        if text_emb is not None:
                            cap = _read_scene_caption(scene_dir, scene_name)
                            print(f"    Caption: {str(cap)[:80]}")
                        else:
                            print("    [WARN] no text embedding found, running unconditioned")
                elif _free:
                    # Free rollout: imgs holds only the cond frame, so we cannot
                    # (and need not) run build_inference_inputs over all views.
                    # The cond image is encoded inside sample_long_sequence_v4;
                    # per-chunk cameras come from the synthetic pose_list.
                    if save_artifacts:
                        visualize_trajectory(
                            pose_list, scene_cond,
                            os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))
                    text_emb = _resolve_ref_global(scene_name, scene_dir)
                else:
                    if save_artifacts and not args.loss_only:
                        visualize_trajectory(
                            pose_list, scene_cond,
                            os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        (
                            _z_input, plucker_full, x_norm, feats_all,
                            _z_ref_clean_enc, _z_all_gt_enc, baseline_full,
                        ) = build_inference_inputs(
                            rae, vae, imgs, intri_list, pose_list,
                            H, W, device, latent_mean, latent_std,
                            latent_whiten=latent_whiten,
                            cond_num=scene_cond,
                            origin_idx=pose_origin_idx,
                            translation_norm=pose_translation_norm,
                            use_baseline_camera=use_baseline_camera,
                            baseline_camera_mode=baseline_camera_mode,
                            pose_scale=eval_pose_scale,
                            s_patch_size=s_patch_size,
                        )
                    text_emb = _resolve_ref_global(scene_name, scene_dir)

                n_chunks = sum(1 for _ in _plan_autoregress_chunks(T, V, roll_cond_num))
                print(f"    Autoregressive: {T} frames, {n_chunks} chunks "
                      f"(chunk={V}, roll={roll_cond_num}, first_cond={first_cond})")

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z_sampled, chunk_infos = sample_long_sequence_v4(
                        dit, rae, vae,
                        img_tensors=imgs, intri_list=intri_list, pose_list=pose_list,
                        chunk_size=V, total_views=T,
                        first_cond_num=first_cond, roll_cond_num=roll_cond_num,
                        mode=args.mode,
                        H=H, W=W, device=device,
                        latent_h=latent_h, latent_w=latent_w,
                        plucker_h=plucker_h, plucker_w=plucker_w,
                        latent_mean=latent_mean, latent_std=latent_std,
                        latent_whiten=latent_whiten, latent_unwhiten=latent_unwhiten,
                        in_ch=in_ch, use_amp=use_amp, amp_dtype=amp_dtype,
                        plucker_6d=plucker_full, ref_global=text_emb,
                        cfg_uncond_ref_global=_qwen3_null_tokens,
                        baseline_cam_kwargs=baseline_full,
                        no_camera=no_camera,
                        reference_conditioning=reference_conditioning,
                        use_baseline_camera=use_baseline_camera,
                        baseline_camera_mode=baseline_camera_mode,
                        pose_origin_idx=pose_origin_idx,
                        translation_norm=pose_translation_norm,
                        pose_scale=eval_pose_scale,
                        num_steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                        cfg_interval=cfg_interval,
                        guidance_mode=args.guidance,
                        ig_scale=args.ig_scale,
                        time_dist_shift=time_dist_shift,
                        eps=args.sample_eps,
                        prediction=prediction,
                        noise_corr=args.noise_corr,
                    )

                if latent_mean is not None:
                    z_sampled = _denorm_latent(
                        z_sampled, latent_mean, latent_std, latent_unwhiten)
                if args.latent_smooth > 0:
                    z_sampled = _smooth_latents_temporal(
                        z_sampled, args.latent_smooth,
                    )

                out_v = z_sampled.shape[0]
                imgs = imgs[:out_v]
                imgs_01 = imgs_01[:out_v]

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    seq, h_l, w_l = vae._decode_trunk(z_sampled)

                if has_rgb:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        rgb_pred = vae.rgb_head(seq, h_l, w_l, num_views=out_v)
                    rgb_pred_01 = denorm(rgb_pred)

                    if save_artifacts:
                        rows = [tensor_to_numpy_img(rgb_pred_01[vi]) for vi in range(out_v)]
                        Image.fromarray(np.concatenate(rows, axis=0)).save(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_rgb.png"))
                        _save_mp4(rows, os.path.join(ds_out_dir, f"{s_idx:03d}_pred.mp4"), fps=args.video_fps)
                        print(f"    Saved: {s_idx:03d}_pred.mp4 ({out_v} frames)")

                        if _free:
                            # No GT exists for a synthetic trajectory, but the
                            # generated latents still support geometry decoding.
                            if (args.save_pointcloud and has_dpt and
                                    rae.rae_cl_decoder is not None):
                                chunk_z_cpus = []
                                for info in chunk_infos:
                                    zc = info["z_full"]
                                    if latent_mean is not None:
                                        zc = _denorm_latent(
                                            zc, latent_mean, latent_std,
                                            latent_unwhiten)
                                    chunk_z_cpus.append(zc.detach().cpu())
                                deferred_pc_jobs.append(dict(
                                    kind="autoregress", s_idx=s_idx,
                                    ds_out_dir=ds_out_dir,
                                    z_sampled=z_sampled.detach().cpu(),
                                    rgb_imgs=rgb_pred_01.detach().cpu(),
                                    chunk_z_list=chunk_z_cpus,
                                    has_rgb=has_rgb,
                                ))
                            if save_artifacts and args.prompt:
                                with open(os.path.join(
                                        ds_out_dir, f"{s_idx:03d}_prompt.txt"), "w") as handle:
                                    handle.write(args.prompt)
                            print(f"    [free-rollout] done: {s_idx:03d}_pred.mp4 "
                                  f"({out_v} frames)")
                            continue

                        # GT target frames are identical regardless of mode; dump
                        # them for every mode so paired FVD/FID/LPIPS can pair
                        # generated (generate) videos against real videos.
                        _save_mp4(
                            [tensor_to_numpy_img(imgs_01[vi]) for vi in range(out_v)],
                            os.path.join(ds_out_dir, f"{s_idx:03d}_gt.mp4"), fps=args.video_fps)
                        if args.mode in ("recon", "t2v", "generate"):
                            _save_mp4(
                                [np.concatenate([tensor_to_numpy_img(imgs_01[vi]),
                                                 tensor_to_numpy_img(rgb_pred_01[vi])], axis=1)
                                 for vi in range(out_v)],
                                os.path.join(ds_out_dir, f"{s_idx:03d}_compare.mp4"), fps=args.video_fps)
                else:
                    print("    [WARN] No RGB head — skipping visualization")

                if (
                    args.save_pointcloud
                    and has_dpt
                    and rae.rae_cl_decoder is not None
                ):
                    chunk_z_cpus = []
                    for info in chunk_infos:
                        zc = info["z_full"]
                        if latent_mean is not None:
                            zc = _denorm_latent(
                                zc, latent_mean, latent_std, latent_unwhiten)
                        chunk_z_cpus.append(zc.detach().cpu())
                    deferred_pc_jobs.append(dict(
                        kind="autoregress",
                        s_idx=s_idx,
                        ds_out_dir=ds_out_dir,
                        z_sampled=z_sampled.detach().cpu(),
                        rgb_imgs=(rgb_pred_01 if has_rgb else imgs_01).detach().cpu(),
                        chunk_z_list=chunk_z_cpus,
                        has_rgb=has_rgb,
                    ))

                if save_artifacts and args.prompt:
                    with open(os.path.join(ds_out_dir, f"{s_idx:03d}_prompt.txt"), "w") as _pf:
                        _pf.write(args.prompt)
                continue

            # ── T2V branch: camera-conditioned text-to-3D generation ──
            if args.mode == "t2v":
                if save_artifacts and not no_camera:
                    visualize_trajectory(
                        pose_list, 0,
                        os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))

                if no_camera:
                    # Pure T2I forward: skip the camera branch entirely so the
                    # model runs exactly as it did during the T2I stage
                    # (plucker_6d=None → `if plucker_6d is not None` short-circuits
                    # the under-trained camera path). z_ref_clean / cond_num are
                    # already None / 0 in t2v mode.
                    plucker_6d = None
                    baseline_cam_kwargs = {}
                elif use_baseline_camera:
                    plucker_6d = None
                    baseline_cam_kwargs = build_baseline_cam_from_cameras(
                        intri_list, pose_list, H, W, device,
                        cond_num=0, total_view=actual_v,
                        origin_idx=pose_origin_idx,
                        baseline_camera_mode=baseline_camera_mode,
                        translation_norm=pose_translation_norm,
                        pose_scale=eval_pose_scale,
                    )
                else:
                    plucker_6d = build_plucker_from_cameras(
                        intri_list, pose_list, H, W,
                        patches_y=plucker_h, patches_x=plucker_w,
                        device=device, origin_idx=pose_origin_idx,
                        translation_norm=pose_translation_norm,
                        pose_scale=eval_pose_scale,
                    )
                    baseline_cam_kwargs = {}

                if args.prompt:
                    text_emb = _encode_prompt(args.prompt)
                    print(f"    Prompt: {args.prompt[:80]}")
                else:
                    text_emb = _resolve_ref_global(scene_name, scene_dir)
                    if text_emb is not None:
                        cap = _read_scene_caption(scene_dir, scene_name)
                        print(f"    Caption: {str(cap)[:80]}")
                    else:
                        print("    [WARN] no text embedding found, running unconditioned")

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z_sampled = sample_v4_euler(
                        dit, z_ref_clean=None, total_view=actual_v, cond_num=0,
                        plucker_6d=plucker_6d, ref_global=text_emb,
                        cfg_uncond_ref_global=_qwen3_null_tokens,
                        num_steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                        cfg_interval=cfg_interval,
                        guidance_mode=args.guidance,
                        ig_scale=args.ig_scale,
                        time_dist_shift=time_dist_shift,
                        eps=args.sample_eps,
                        noise_shape=(in_ch, latent_h, latent_w),
                        device=device, dtype=amp_dtype,
                        prediction=prediction,
                        reference_conditioning=reference_conditioning,
                        baseline_cam_kwargs=baseline_cam_kwargs or None,
                        noise_corr=args.noise_corr,
                    )

                if latent_mean is not None:
                    z_sampled = _denorm_latent(
                        z_sampled, latent_mean, latent_std, latent_unwhiten)
                if args.latent_smooth > 0:
                    z_sampled = _smooth_latents_temporal(
                        z_sampled, args.latent_smooth,
                    )

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    seq, h_l, w_l = vae._decode_trunk(z_sampled)

                if has_rgb:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        rgb_pred = vae.rgb_head(seq, h_l, w_l, num_views=actual_v)
                    rgb_pred_01 = denorm(rgb_pred)

                    if save_artifacts:
                        rows = [tensor_to_numpy_img(rgb_pred_01[vi]) for vi in range(actual_v)]
                        Image.fromarray(np.concatenate(rows, axis=0)).save(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_rgb.png"))
                        _save_mp4(rows, os.path.join(ds_out_dir, f"{s_idx:03d}_pred.mp4"), fps=args.video_fps)
                        print(f"    Saved: {s_idx:03d}_pred.mp4")

                        # GT target frames for paired FVD/FID/LPIPS (all modes).
                        _save_mp4(
                            [tensor_to_numpy_img(imgs_01[vi]) for vi in range(actual_v)],
                            os.path.join(ds_out_dir, f"{s_idx:03d}_gt.mp4"), fps=args.video_fps)
                        # Compare with GT if available
                        _save_mp4(
                            [np.concatenate([tensor_to_numpy_img(imgs_01[vi]),
                                             tensor_to_numpy_img(rgb_pred_01[vi])], axis=1)
                             for vi in range(actual_v)],
                            os.path.join(ds_out_dir, f"{s_idx:03d}_compare.mp4"), fps=args.video_fps)
                else:
                    print("    [WARN] No RGB head — skipping visualization")

                if (
                    args.save_pointcloud
                    and has_dpt
                    and rae.rae_cl_decoder is not None
                ):
                    deferred_pc_jobs.append(dict(
                        kind="t2v",
                        s_idx=s_idx,
                        ds_out_dir=ds_out_dir,
                        z_sampled=z_sampled.detach().cpu(),
                        rgb_imgs=(rgb_pred_01 if has_rgb else imgs_01).detach().cpu(),
                    ))

                if save_artifacts and args.prompt:
                    with open(os.path.join(ds_out_dir, f"{s_idx:03d}_prompt.txt"), "w") as _pf:
                        _pf.write(args.prompt)
                continue  # skip recon/generate logic below

            if scene_cond > 0:
                imgs, intri_list, pose_list, view_frame_idx = reorder_scene_for_ref_sampling(
                    imgs, intri_list, pose_list, scene_cond, ref_view_sampling,
                )
                imgs_01 = denorm(imgs)

            if save_artifacts and not args.loss_only:
                visualize_trajectory(
                    pose_list, scene_cond,
                    os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if latent_backend in ("sd_vae", "wan2_1", "da3_direct"):
                    (
                        z_input, plucker_6d, x_norm, feats_all,
                        z_ref_clean_enc, z_all_gt_enc, baseline_cam_kwargs,
                        imgs_01,
                    ) = build_rgb_codec_inference_inputs(
                        vae, latent_backend, imgs, intri_list, pose_list,
                        H, W, device,
                        cond_num=scene_cond,
                        origin_idx=pose_origin_idx,
                        translation_norm=pose_translation_norm,
                        use_baseline_camera=use_baseline_camera,
                        baseline_camera_mode=baseline_camera_mode,
                        pose_scale=eval_pose_scale,
                        s_patch_size=s_patch_size,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                        latent_whiten=latent_whiten,
                    )
                else:
                    (
                        z_input, plucker_6d, x_norm, feats_all,
                        z_ref_clean_enc, z_all_gt_enc, baseline_cam_kwargs,
                    ) = build_inference_inputs(
                        rae, vae, imgs, intri_list, pose_list,
                        H, W, device, latent_mean, latent_std,
                        latent_whiten=latent_whiten,
                        cond_num=scene_cond,
                        origin_idx=pose_origin_idx,
                        translation_norm=pose_translation_norm,
                        use_baseline_camera=use_baseline_camera,
                        baseline_camera_mode=baseline_camera_mode,
                        pose_scale=eval_pose_scale,
                        s_patch_size=s_patch_size,
                    )

            if no_camera:
                plucker_6d = None
                baseline_cam_kwargs = {}

            text_emb = _resolve_ref_global(scene_name, scene_dir)
            model_kwargs = dict(
                plucker_6d=plucker_6d,
                total_view=actual_v,
                cond_num=scene_cond,
                ref_global=text_emb,
                view_frame_idx=view_frame_idx,
                **(baseline_cam_kwargs or {}),
            )

            # ── Loss ──
            if args.mode == "recon":
                use_shifted_t = (loss_t_mode == "shifted")
                n_loss_samples = args.loss_samples if use_shifted_t else 1
                t_override_val = None if use_shifted_t else loss_t_fixed

                scene_losses = []
                scene_losses_all = []
                for _ in range(n_loss_samples):
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if prediction == "x":
                            # RAE recipe: model emits x_pred, loss is in v-space
                            # after x→v conversion. Mirrors training_losses_rae.
                            # x_target = z_all_gt_enc (all-view-ctx clean latent);
                            # ref conditioning flows via z_ref_clean kwarg, NOT
                            # via xt clamping (trainer's convention).
                            loss_dict = _rae_eval_loss(
                                dit,
                                z_all_gt=z_all_gt_enc,
                                z_ref_clean=z_ref_clean_enc,
                                total_view=actual_v,
                                cond_num=scene_cond,
                                model_kwargs=model_kwargs,
                                time_dist_shift=time_dist_shift,
                                time_dist_type=time_dist_type,
                                reference_conditioning=reference_conditioning,
                                t_override=t_override_val,
                            )
                        else:
                            # Legacy V3/V4 velocity path (transport API). Uses
                            # the mixed z_input (ref slots = z_ref, tgt slots
                            # = z_all) and clamps refs to clean in xt.
                            loss_dict = transport.training_multiview_losses(
                                dit, z_input, actual_v, scene_cond,
                                model_kwargs=model_kwargs.copy(),
                                t_override=t_override_val,
                            )
                    scene_losses.append(loss_dict["loss"].mean().item())
                    if "loss_all" in loss_dict:
                        scene_losses_all.append(loss_dict["loss_all"].mean().item())

                loss_val = float(np.mean(scene_losses))
                all_loss.append(loss_val)
                t_desc = "shifted" if use_shifted_t else f"t={t_override_val}"
                if scene_losses_all:
                    loss_all_val = float(np.mean(scene_losses_all))
                    # all-view == trainer's `main=...`; tgt-only is novel-view only
                    print(f"    Tgt Loss [{t_desc}]: {loss_val:.4f}  "
                          f"(all-view/main-equiv: {loss_all_val:.4f})")
                else:
                    print(f"    Tgt Loss [{t_desc}]: {loss_val:.4f}")

            if args.loss_only:
                continue

            # ── ODE sampling ──
            C, h_lat, w_lat = z_input.shape[1:]
            z_ref = z_input[:scene_cond]  # refs-only DA3 ctx

            # V4: use z_ref_clean as conditioning, all views from noise
            if z_ref_clean_enc is not None:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z_sampled = sample_v4_euler(
                        dit, z_ref_clean_enc, actual_v, scene_cond,
                        plucker_6d=plucker_6d,
                        ref_global=text_emb,
                        cfg_uncond_ref_global=_qwen3_null_tokens,
                        num_steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                        cfg_interval=cfg_interval,
                        guidance_mode=args.guidance,
                        ig_scale=args.ig_scale,
                        time_dist_shift=time_dist_shift,
                        eps=args.sample_eps,
                        prediction=prediction,
                        reference_conditioning=reference_conditioning,
                        baseline_cam_kwargs=baseline_cam_kwargs or None,
                        noise_corr=args.noise_corr,
                        view_frame_idx=view_frame_idx,
                    )
            else:
                # Fallback to V3 sampling
                noise_tgt = _make_view_correlated_noise(
                    actual_v - scene_cond, C, h_lat, w_lat, device, corr=args.noise_corr,
                )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z_tgt_sampled = sample_v3_euler(
                        dit, z_ref, noise_tgt,
                        plucker_6d, actual_v, scene_cond,
                        num_steps=args.sample_steps,
                        cfg_scale=args.cfg_scale,
                        cfg_interval=cfg_interval,
                        guidance_mode=args.guidance,
                        ig_scale=args.ig_scale,
                        time_dist_shift=time_dist_shift,
                        eps=args.sample_eps,
                        ref_global=text_emb,
                        denoise_all=denoise_all,
                        ref_noise_frac=ref_noise_frac,
                    )
                if denoise_all:
                    z_sampled = z_tgt_sampled
                else:
                    z_sampled = torch.cat([z_ref, z_tgt_sampled], dim=0)

            z_model_interp = None
            if args.model_interp_factor > 1:
                interp_views = args.model_interp_views or actual_v
                # Each interp segment spans one keyframe gap (== sample_interval
                # real frames) across interp_views slots → its mean real frame
                # gap is sample_interval/(interp_views-1).
                interp_pose_scale = _eval_interval_scale(
                    None if args.sample_interval is None else
                    float(args.sample_interval) / max(interp_views - 1, 1)
                )
                print(
                    f"    Model interpolation: {actual_v} keyframes, "
                    f"factor={args.model_interp_factor}, interp_views={interp_views}"
                    + (f", interp_pose_scale={interp_pose_scale:g}"
                       if interp_pose_scale is not None else "")
                )
                z_model_interp = model_interpolate_generated_keyframes(
                    dit,
                    z_sampled,
                    intri_list,
                    pose_list,
                    factor=args.model_interp_factor,
                    interp_views=interp_views,
                    H=H,
                    W=W,
                    device=device,
                    plucker_h=plucker_h,
                    plucker_w=plucker_w,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    use_baseline_camera=use_baseline_camera,
                    no_camera=no_camera,
                    reference_conditioning=reference_conditioning,
                    baseline_camera_mode=baseline_camera_mode,
                    pose_origin_idx=pose_origin_idx,
                    translation_norm=pose_translation_norm,
                    ref_global=text_emb,
                    cfg_uncond_ref_global=_qwen3_null_tokens,
                    num_steps=args.sample_steps,
                    cfg_scale=args.cfg_scale,
                    cfg_interval=cfg_interval,
                    guidance_mode=args.guidance,
                    ig_scale=args.ig_scale,
                    time_dist_shift=time_dist_shift,
                    eps=args.sample_eps,
                    prediction=prediction,
                    noise_corr=args.noise_corr,
                    pose_scale=interp_pose_scale,
                )
                print(f"    Model interpolation frames: {z_model_interp.shape[0]}")

            if latent_mean is not None:
                z_sampled = _denorm_latent(z_sampled, latent_mean, latent_std, latent_unwhiten)
                if z_model_interp is not None:
                    z_model_interp = _denorm_latent(
                        z_model_interp, latent_mean, latent_std, latent_unwhiten,
                    )
            if args.latent_smooth > 0:
                z_sampled = _smooth_latents_temporal(
                    z_sampled, args.latent_smooth,
                )
                if z_model_interp is not None:
                    z_model_interp = _smooth_latents_temporal(
                        z_model_interp, args.latent_smooth,
                    )

            recon_feats = None
            seq = h_l = w_l = None
            if latent_backend not in ("sd_vae", "wan2_1", "da3_direct"):
                # ── VAE decode ──
                with _primary_output_timing():
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        seq, h_l, w_l = vae._decode_trunk(z_sampled)
                        c_trunk = seq.shape[-1]
                        recon_raw = vae.dec_conv(
                            seq.permute(0, 2, 1).reshape(-1, c_trunk, h_l, w_l))
                        recon_feats = vae.denormalize_and_split(recon_raw)

            if args.mode == "recon" and latent_backend not in ("sd_vae", "wan2_1", "da3_direct"):
                recon_norm = vae.normalize_levels(recon_feats, image_size=(H, W))
                recon_ch = recon_norm.split(DA3_LEVEL_DIM, dim=1)
                input_ch = x_norm.split(DA3_LEVEL_DIM, dim=1)
                feat_l1 = {f"l{i}": F.l1_loss(rc, ic).item()
                           for i, (rc, ic) in enumerate(zip(recon_ch, input_ch))}
                feat_l1["avg"] = float(np.mean(list(feat_l1.values())))
                all_feat.append(feat_l1)
                print("    Feature L1: " +
                      ", ".join(f"{k}={v:.4f}" for k, v in feat_l1.items()))

                # Separate ref vs tgt feature L1
                if scene_cond > 0:
                    ref_l1 = F.l1_loss(recon_norm[:scene_cond], x_norm[:scene_cond]).item()
                    tgt_l1 = F.l1_loss(recon_norm[scene_cond:], x_norm[scene_cond:]).item()
                    print(f"    Ref L1={ref_l1:.4f}  Tgt L1={tgt_l1:.4f}")

                # Source view latent loss: denoised ref latent vs GT all-view ref latent
                if denoise_all and scene_cond > 0:
                    # z_sampled is already denormalized; z_input[:cond_num] is ref-only ctx (normalized)
                    # Compare in normalized latent space
                    z_sampled_norm = (_norm_latent(z_sampled, latent_mean, latent_std, latent_whiten)
                                      if latent_mean is not None else z_sampled)
                    # GT all-view ref latent (normalized)
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        imgs_5d_all = imgs.unsqueeze(0).to(device)
                        feats_gt = rae.encode(imgs_5d_all, mode="all")
                        feats_gt_nc = {k: v[:, 1:, :] for k, v in feats_gt.items()}
                        x_gt = vae.normalize_levels(feats_gt_nc, image_size=(H, W))
                        z_gt_all = vae.encode(x_gt)[0]
                        if latent_mean is not None:
                            z_gt_all_norm = _norm_latent(z_gt_all, latent_mean, latent_std, latent_whiten)
                        else:
                            z_gt_all_norm = z_gt_all
                    ref_latent_l1 = F.l1_loss(
                        z_sampled_norm[:scene_cond], z_gt_all_norm[:scene_cond]).item()
                    tgt_latent_l1 = F.l1_loss(
                        z_sampled_norm[scene_cond:], z_gt_all_norm[scene_cond:]).item()
                    print(f"    Latent L1: ref={ref_latent_l1:.4f}  tgt={tgt_latent_l1:.4f}")

            if has_rgb:
                with _primary_output_timing():
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if latent_backend in ("sd_vae", "wan2_1", "da3_direct"):
                            rgb_pred_01 = decode_rgb_codec(vae, latent_backend, z_sampled)
                        else:
                            rgb_pred = vae.rgb_head(seq, h_l, w_l, num_views=actual_v)
                            rgb_pred_01 = denorm(rgb_pred)
                if args.mode == "recon":
                    rgb_m = compute_metrics(imgs_01.float(), rgb_pred_01.float())
                    rgb_m["ssim"] = compute_ssim(imgs_01.float(), rgb_pred_01.float())
                    all_rgb.append(rgb_m)
                    print(f"    RGB  PSNR={rgb_m['psnr']:.2f}dB  "
                          f"L1={rgb_m['l1']:.4f}  SSIM={rgb_m['ssim']:.4f}")

                if save_artifacts:
                    rows = []
                    for vi in range(actual_v):
                        row = [tensor_to_numpy_img(rgb_pred_01[vi])]
                        if args.mode == "recon":
                            row.insert(0, tensor_to_numpy_img(imgs_01[vi]))
                        rows.append(np.concatenate(row, axis=1))
                    Image.fromarray(np.concatenate(rows, axis=0)).save(
                        os.path.join(ds_out_dir, f"{s_idx:03d}_rgb.png"))

                    _save_mp4(
                        [tensor_to_numpy_img(rgb_pred_01[vi]) for vi in range(actual_v)],
                        os.path.join(ds_out_dir, f"{s_idx:03d}_pred.mp4"), fps=args.video_fps)
                    # GT target frames for paired FVD/FID/LPIPS (all modes).
                    _save_mp4(
                        [tensor_to_numpy_img(imgs_01[vi]) for vi in range(actual_v)],
                        os.path.join(ds_out_dir, f"{s_idx:03d}_gt.mp4"), fps=args.video_fps)
                    if args.mode in ("recon", "generate"):
                        _save_mp4(
                            [np.concatenate([tensor_to_numpy_img(imgs_01[vi]),
                                             tensor_to_numpy_img(rgb_pred_01[vi])], axis=1)
                             for vi in range(actual_v)],
                            os.path.join(ds_out_dir, f"{s_idx:03d}_compare.mp4"), fps=args.video_fps)

                if save_artifacts and z_model_interp is not None:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if latent_backend in ("sd_vae", "wan2_1", "da3_direct"):
                            rgb_interp_01 = decode_rgb_codec(vae, latent_backend, z_model_interp)
                        else:
                            seq_i, h_i, w_i = vae._decode_trunk(z_model_interp)
                            rgb_interp = vae.rgb_head(
                                seq_i, h_i, w_i, num_views=z_model_interp.shape[0],
                            )
                            rgb_interp_01 = denorm(rgb_interp)
                    interp_frames = [
                        tensor_to_numpy_img(rgb_interp_01[vi])
                        for vi in range(rgb_interp_01.shape[0])
                    ]
                    interp_fps = max(1, int(args.video_fps) * int(args.model_interp_factor))
                    _save_mp4(
                        interp_frames,
                        os.path.join(
                            ds_out_dir,
                            f"{s_idx:03d}_pred_model_interp_x{args.model_interp_factor}.mp4",
                        ),
                        fps=interp_fps,
                    )
                    print(
                        f"    Saved model interpolation: "
                        f"{s_idx:03d}_pred_model_interp_x{args.model_interp_factor}.mp4 "
                        f"({len(interp_frames)} frames @ {interp_fps} fps)"
                    )

            # ── Ref view RGB metrics ──
            if has_rgb and args.mode == "recon" and scene_cond > 0:
                ref_rgb_m = compute_metrics(imgs_01[:scene_cond].float(), rgb_pred_01[:scene_cond].float())
                ref_rgb_m["ssim"] = compute_ssim(imgs_01[:scene_cond].float(), rgb_pred_01[:scene_cond].float())
                print(f"    Ref view:   PSNR={ref_rgb_m['psnr']:.2f}dB  "
                      f"L1={ref_rgb_m['l1']:.4f}")

                if save_artifacts:
                    ref_gt_np = tensor_to_numpy_img(imgs_01[0])
                    ref_pred_np = tensor_to_numpy_img(rgb_pred_01[0])
                    ref_grid = np.concatenate([ref_gt_np, ref_pred_np], axis=1)
                    Image.fromarray(ref_grid).save(
                        os.path.join(ds_out_dir, f"{s_idx:03d}_ref.png"))

            # ── VAE Recon RGB baseline (encode → decode, no diffusion) ──
            if has_rgb and args.mode == "recon":
                try:
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        if latent_backend in ("sd_vae", "wan2_1", "da3_direct"):
                            z_vae_recon = z_all_gt_enc
                            rgb_vae_01 = decode_rgb_codec(vae, latent_backend, z_vae_recon)
                        else:
                            _imgs_5d = imgs.unsqueeze(0).to(device)
                            _feats = rae.encode(_imgs_5d, mode="all")
                            _feats_nc = {k: v[:, 1:, :] for k, v in _feats.items()}
                            _x_n = vae.normalize_levels(_feats_nc, image_size=(H, W))
                            z_vae_recon = vae.encode(_x_n)[0]
                            seq_vr, h_vr, w_vr = vae._decode_trunk(z_vae_recon)
                            rgb_vae = vae.rgb_head(seq_vr, h_vr, w_vr, num_views=actual_v)
                            rgb_vae_01 = denorm(rgb_vae)

                    vae_rgb_m = compute_metrics(imgs_01.float(), rgb_vae_01.float())
                    vae_rgb_m["ssim"] = compute_ssim(imgs_01.float(), rgb_vae_01.float())
                    all_rgb_vae_recon.append(vae_rgb_m)
                    print(f"    VAE Recon:  PSNR={vae_rgb_m['psnr']:.2f}dB  "
                          f"L1={vae_rgb_m['l1']:.4f}  SSIM={vae_rgb_m['ssim']:.4f}")

                    if save_artifacts:
                        _save_mp4(
                            [tensor_to_numpy_img(rgb_vae_01[vi]) for vi in range(actual_v)],
                            os.path.join(ds_out_dir, f"{s_idx:03d}_vae_recon.mp4"), fps=args.video_fps)
                except Exception as e:
                    print(f"    [WARN] VAE recon RGB failed: {e}")

            if (has_dpt and rae is not None and rae.rae_cl_decoder is not None
                    and latent_backend == "da3_direct"):
                # da3_direct DPT depth = propagate sampled level-0 through the
                # frozen DA3 backbone (GT CLS token) → DPT head. GT-depth
                # alignment / pointclouds are handled separately. When timing,
                # run this primary geometry path unconditionally and fail loud:
                # silently skipping it would produce an RGB-only timing result.
                primary_direct_output = None
                if args.timing_json or save_artifacts:
                    with torch.amp.autocast("cuda", enabled=False):
                        _gt_all = vae.encode_all(imgs.unsqueeze(0).to(device))
                        _cls = _gt_all[vae.level][:, 0].float()
                        with _primary_output_timing():
                            primary_direct_output = vae.decode_depth(
                                z_sampled.float(), _cls, total_view=actual_v, H=H, W=W,
                            )
                if save_artifacts and primary_direct_output is not None:
                    _dep = primary_direct_output.get("depth")
                    if _dep is not None:
                        # DPT returns [B, V, 1, H, W] for DA3-direct; PNG rows
                        # are indexed by view, not the singleton batch dimension.
                        _dep_views = _dep[0] if _dep.ndim >= 4 and _dep.shape[0] == 1 else _dep
                        if _dep_views.shape[0] != actual_v:
                            raise ValueError(
                                "DA3-direct depth view count mismatch: "
                                f"got {tuple(_dep.shape)}, expected {actual_v} views")
                        rows = [depth_to_numpy_img(_dep_views[vi]) for vi in range(actual_v)]
                        Image.fromarray(np.concatenate(rows, axis=0)).save(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_depth.png"))

                # ── Dump direct geometry for da3_direct (level-0 propagate) ──
                # Same NPZ schema as the VAE backend so scripts/
                # eval_direct_geometry.py reuses it unchanged. da3_direct has no
                # VAE bottleneck, so vae/gtmodel == propagate(GT level-0) = the
                # DA3-native ceiling reachable from the diffused level.
                if (args.dump_geometry
                        and len(pose_list) == actual_v
                        and len(intri_list) == actual_v):
                    try:
                        _dump = {}

                        def _ray_pack_d(dpt_out, prefix):
                            d = dpt_out.get("depth")
                            r = dpt_out.get("ray")
                            rc = dpt_out.get("ray_conf")
                            if d is not None:
                                _dump[f"{prefix}_depth"] = (
                                    d.squeeze(1).float().cpu().numpy())
                            if r is not None:
                                _dump[f"{prefix}_ray"] = _ray_to_numpy(r)
                            if rc is not None:
                                _dump[f"{prefix}_ray_conf"] = _rayconf_to_numpy(rc)

                        with torch.amp.autocast("cuda", enabled=False):
                            _gt_all = vae.encode_all(imgs.unsqueeze(0).to(device))
                            # CLS source for propagation. Default 'gt' uses the
                            # real-frame CLS token (faithful propagation). 'zero'
                            # withholds it (propagate uses zeros) for a fair,
                            # self-contained comparison to the VAE backend whose
                            # DPT decode is CLS-free (raw_no_cls_to_dpt_input).
                            _dump_cls_mode = os.environ.get(
                                "DA3_DIRECT_DUMP_CLS", "gt").lower()
                            if _dump_cls_mode == "zero":
                                _cls = None
                            else:
                                _cls = _gt_all[vae.level][:, 0].float()

                            # Propagate diffused level-0 through the frozen DA3
                            # backbone → all levels → DPT head (depth+ray+conf).
                            # We call decode_dpt directly (the tested path) rather
                            # than rae.decode, which also runs the RGB/MAE denorm
                            # branch we do not need here.
                            def _da3_direct_dpt(z_norm):
                                _raw = vae._denormalize(z_norm)
                                _feats = vae.rae.propagate_features(
                                    _raw, from_level=vae.level,
                                    total_view=actual_v, cls_token=_cls)
                                return decode_dpt(_feats, rae.rae_cl_decoder, H, W)

                            # NGD generated: flow-sampled level-0 → propagate → DPT
                            _ray_pack_d(_da3_direct_dpt(z_sampled.float()), "gen")
                            if args.mode == "recon":
                                # Encode→decode (no bottleneck) = DA3-native ceiling
                                # encode_views applies its own ImageNet norm,
                                # so it takes imgs_01, not the ImgNorm imgs.
                                _zgt = vae.encode_views(
                                    imgs_01.unsqueeze(0).to(device)).float()
                                _rec = _da3_direct_dpt(_zgt)
                                _ray_pack_d(_rec, "vae")
                                _ray_pack_d(_rec, "gtmodel")

                        if has_rgb:
                            _dump["gen_rgb"] = rgb_pred_01.float().cpu().numpy()
                        if args.mode == "recon":
                            _dump["gt_rgb"] = imgs_01.float().cpu().numpy()
                        _dump["gt_c2w"] = np.stack(
                            [np.asarray(p, dtype=np.float64) for p in pose_list])
                        _intr_t = torch.from_numpy(
                            np.stack([np.asarray(k, dtype=np.float32)
                                      for k in intri_list]))
                        _dump["gt_K"] = (
                            intrinsic_to_K(_intr_t, device).cpu().numpy()
                            .astype(np.float64))
                        _dump["cond_num"] = np.int64(scene_cond)
                        _dump["image_size"] = np.array([H, W], dtype=np.int64)

                        _save_geom_npz(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_geom.npz"),
                            _dump)
                        print(f"    [dump-geometry] {s_idx:03d}_geom.npz "
                              f"({', '.join(sorted(_dump.keys()))})")
                    except Exception as e:
                        print(f"    [WARN] da3_direct dump-geometry failed: {e}")
            elif has_dpt and rae is not None and rae.rae_cl_decoder is not None:
                gen_dpt_in = raw_no_cls_to_dpt_input(
                    recon_feats, backbone_norm, actual_v)

                gen_dpt_out = None
                need_full_dpt = bool(args.save_pointcloud or args.ref_tgt_pc_gap)
                if need_full_dpt:
                    # Use decode_dpt to get depth + ray + ray_conf (all consistent)
                    with _primary_output_timing():
                        gen_dpt_out = decode_dpt(gen_dpt_in, rae.rae_cl_decoder, H, W)
                    gen_depth = gen_dpt_out['depth']
                else:
                    with _primary_output_timing():
                        gen_depth = decode_to_depth(gen_dpt_in, rae.rae_cl_decoder, H, W)

                gt_depth = None
                gt_dpt_out = None
                if args.mode == "recon":
                    gt_dpt_in = raw_to_dpt_input(feats_all, backbone_norm, actual_v)
                    if need_full_dpt:
                        gt_dpt_out = decode_dpt(gt_dpt_in, rae.rae_cl_decoder, H, W)
                        gt_depth = gt_dpt_out['depth']
                    else:
                        gt_depth = decode_to_depth(gt_dpt_in, rae.rae_cl_decoder, H, W)

                if gen_depth is not None:
                    if gt_depth is not None:
                        depth_m = compute_metrics(
                            gt_depth.float(), gen_depth.float())
                        all_depth.append(depth_m)
                        print(f"    Depth PSNR={depth_m['psnr']:.2f}dB  "
                              f"L1={depth_m['l1']:.4f}")

                    if (
                        args.ref_tgt_pc_gap
                        and gen_dpt_out is not None
                        and scene_cond > 0
                        and scene_cond < actual_v
                    ):
                        try:
                            _rgb_for_gap = (
                                rgb_pred_01 if has_rgb else None
                            )
                            gap = compute_ref_tgt_pc_gap(
                                gen_dpt_out,
                                gen_depth,
                                _rgb_for_gap,
                                H,
                                W,
                                device,
                                cond_num=scene_cond,
                                stride=max(int(args.pc_stride), 1),
                                rng=_pc_gap_rng,
                            )
                            if math.isfinite(gap):
                                all_ref_tgt_pc_gap.append(gap)
                                print(f"    Ref–tgt PC gap: {gap:.6f}")
                            else:
                                print("    Ref–tgt PC gap: nan (empty cloud)")
                        except Exception as e:  # noqa: BLE001
                            print(f"    [WARN] Ref–tgt PC gap failed: {e}")

                    if save_artifacts:
                        rows = []
                        for vi in range(actual_v):
                            row = [depth_to_numpy_img(gen_depth[vi])]
                            if gt_depth is not None:
                                row.insert(0, depth_to_numpy_img(gt_depth[vi]))
                            rows.append(np.concatenate(row, axis=1))
                        Image.fromarray(np.concatenate(rows, axis=0)).save(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_depth.png"))

                # ── Save input (GT) + predicted camera poses to compact NPZ ──
                # input_c2w : dataset camera-to-world poses fed to the model
                #             (post ref-sampling reorder, so view order matches
                #             the predictions), plus input_K.
                # pred_c2w  : per-view poses recovered from the generated ray
                #             head (the same recover_poses used for the point
                #             cloud), plus pred_K.
                # Frames differ (recover_poses is canonicalised), align with
                # Umeyama/Sim3 downstream if you need them in one frame.
                if (save_artifacts and gen_dpt_out is not None
                        and gen_dpt_out.get("ray") is not None
                        and len(pose_list) == actual_v
                        and len(intri_list) == actual_v):
                    try:
                        _pray = _ray_to_numpy(gen_dpt_out["ray"])
                        _prc = _rayconf_to_numpy(gen_dpt_out.get("ray_conf"))
                        _pred_c2w, _pred_K = recover_poses(
                            _pray, _prc, input_size=(H, W),
                            return_per_view_intrinsics=True)
                        _intr_pt = torch.from_numpy(
                            np.stack([np.asarray(k, dtype=np.float32)
                                      for k in intri_list]))
                        _pose_arrays = dict(
                            input_c2w=np.stack(
                                [np.asarray(p, dtype=np.float64)
                                 for p in pose_list]),
                            input_K=(intrinsic_to_K(_intr_pt, device)
                                     .cpu().numpy().astype(np.float64)),
                            pred_c2w=np.asarray(_pred_c2w, dtype=np.float64),
                            pred_K=np.asarray(_pred_K, dtype=np.float64),
                            cond_num=np.int64(scene_cond),
                            image_size=np.array([H, W], dtype=np.int64),
                        )
                        _save_geom_npz(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_poses.npz"),
                            _pose_arrays)
                        print(f"    [poses] {s_idx:03d}_poses.npz "
                              f"(input_c2w {tuple(_pose_arrays['input_c2w'].shape)}"
                              f", pred_c2w {tuple(_pose_arrays['pred_c2w'].shape)})")
                    except Exception as e:  # noqa: BLE001
                        print(f"    [WARN] pose save failed: {e}")

                # ── Dump direct geometry (NGD gen + VAE + pseudo-GT) to NPZ ──
                # Consumed by scripts/eval_direct_geometry.py for the
                # depth/pose/point-map comparison against DA3/VGGT (no reproj).
                if (args.dump_geometry and recon_feats is not None
                        and len(pose_list) == actual_v
                        and len(intri_list) == actual_v):
                    try:
                        _dump = {}

                        def _ray_pack(dpt_out, prefix):
                            d = dpt_out.get("depth")
                            r = dpt_out.get("ray")
                            rc = dpt_out.get("ray_conf")
                            if d is not None:
                                _dump[f"{prefix}_depth"] = (
                                    d.squeeze(1).float().cpu().numpy())
                            if r is not None:
                                _dump[f"{prefix}_ray"] = _ray_to_numpy(r)
                            if rc is not None:
                                _dump[f"{prefix}_ray_conf"] = _rayconf_to_numpy(rc)

                        # NGD generated (flow-sampled latent → decode)
                        _gen_in = raw_no_cls_to_dpt_input(
                            recon_feats, backbone_norm, actual_v)
                        _ray_pack(
                            decode_dpt(_gen_in, rae.rae_cl_decoder, H, W), "gen")

                        # Model pseudo-GT (DA3 features of GT frames → DPT)
                        if args.mode == "recon":
                            _gt_in = raw_to_dpt_input(
                                feats_all, backbone_norm, actual_v)
                            _ray_pack(
                                decode_dpt(_gt_in, rae.rae_cl_decoder, H, W),
                                "gtmodel")

                            # NGD VAE-direct (encode GT → z → decode, no diffusion)
                            with torch.amp.autocast(
                                    "cuda", enabled=use_amp, dtype=amp_dtype):
                                _im5 = imgs.unsqueeze(0).to(device)
                                _ft = rae.encode(_im5, mode="all")
                                _ftnc = {k: v[:, 1:, :] for k, v in _ft.items()}
                                _xn = vae.normalize_levels(_ftnc, image_size=(H, W))
                                _zv = vae.encode(_xn)[0]
                                _sq, _hv, _wv = vae._decode_trunk(_zv)
                                _cv = _sq.shape[-1]
                                _raw = vae.dec_conv(
                                    _sq.permute(0, 2, 1).reshape(-1, _cv, _hv, _wv))
                                _vfeats = vae.denormalize_and_split(_raw)
                            _vae_in = raw_no_cls_to_dpt_input(
                                _vfeats, backbone_norm, actual_v)
                            _ray_pack(
                                decode_dpt(_vae_in, rae.rae_cl_decoder, H, W),
                                "vae")

                        # RGB (generated + GT) for DA3/VGGT reconstruction
                        if has_rgb:
                            _dump["gen_rgb"] = rgb_pred_01.float().cpu().numpy()
                        if args.mode == "recon":
                            _dump["gt_rgb"] = imgs_01.float().cpu().numpy()

                        # Real dataset cameras (world c2w + pixel intrinsics).
                        # intrinsic_to_K handles both (3,3) and [fx,fy,cx,cy]
                        # layouts → canonical (V,3,3), matching the K the model
                        # is conditioned on in build_inference_inputs.
                        _dump["gt_c2w"] = np.stack(
                            [np.asarray(p, dtype=np.float64) for p in pose_list])
                        _intr_t = torch.from_numpy(
                            np.stack([np.asarray(k, dtype=np.float32)
                                      for k in intri_list]))
                        _dump["gt_K"] = (
                            intrinsic_to_K(_intr_t, device).cpu().numpy()
                            .astype(np.float64))
                        _dump["cond_num"] = np.int64(scene_cond)
                        _dump["image_size"] = np.array([H, W], dtype=np.int64)

                        _save_geom_npz(
                            os.path.join(ds_out_dir, f"{s_idx:03d}_geom.npz"),
                            _dump)
                        print(f"    [dump-geometry] {s_idx:03d}_geom.npz "
                              f"({', '.join(sorted(_dump.keys()))})")
                    except Exception as e:
                        print(f"    [WARN] dump-geometry failed: {e}")
                elif args.dump_geometry and recon_feats is not None:
                    print(f"    [dump-geometry] skip {s_idx:03d}: cameras "
                          f"({len(pose_list)}) != decoded views ({actual_v}) "
                          f"— not supported for autoregressive rollout")

                # ── Point clouds from DPT depth + recovered poses ──
                if args.save_pointcloud or args.dump_latents:
                    # When only the geo-cache is wanted, skip the (expensive,
                    # high-VRAM) point-cloud backprojection + PLY write but keep
                    # the decode_dpt calls above that feed the cache.
                    _skip_ply = (
                        args.dump_latents and not args.save_pointcloud
                    ) or bool(os.environ.get("GEO_CACHE_NO_PLY"))
                    def _build_pointcloud(dpt_out, depth_tensor, rgb_imgs,
                                          n_views, tag):
                        if _skip_ply or dpt_out is None or depth_tensor is None:
                            return
                        try:
                            xyz, rgb = _scene_pointcloud_from_dpt(
                                dpt_out, depth_tensor, rgb_imgs,
                                n_views, H, W, device,
                                stride=args.pc_stride,
                            )
                            ply_path = os.path.join(
                                ds_out_dir,
                                f"{s_idx:03d}_{tag}_pointcloud.ply")
                            save_pointcloud_ply(ply_path, xyz, rgb)
                            print(f"    {tag.upper()} point cloud: "
                                  f"{ply_path} ({xyz.shape[0]} pts)")
                        except Exception as e:
                            print(f"    [WARN] {tag} point cloud failed: {e}")

                    # Predicted point cloud (pred RGB + pred depth/ray)
                    pred_rgb = rgb_pred_01 if has_rgb else imgs_01
                    _build_pointcloud(gen_dpt_out, gen_depth, pred_rgb,
                                      actual_v, "pred")

                    # GT point cloud (GT RGB + GT depth/ray)
                    if args.mode == "recon":
                        _build_pointcloud(gt_dpt_out, gt_depth, imgs_01,
                                          actual_v, "gt")

                    # VAE-recon point cloud (encode→decode, no diffusion)
                    # z_input is the normalized latent from VAE encoder;
                    # denormalize → VAE decode → DPT → point cloud
                    if args.mode == "recon":
                        try:
                            # Re-encode ALL views together with DA3
                            # cross-view attention to preserve multi-view
                            # consistency. (z_input mixes ref-only and
                            # all-view contexts which breaks alignment.)
                            with torch.amp.autocast("cuda",
                                                    enabled=use_amp,
                                                    dtype=amp_dtype):
                                _imgs_5d = imgs.unsqueeze(0).to(device)
                                _feats = rae.encode(_imgs_5d, mode="all")
                                _feats_nc = {k: v[:, 1:, :]
                                             for k, v in _feats.items()}
                                _x_n = vae.normalize_levels(
                                    _feats_nc, image_size=(H, W))
                                z_vae = vae.encode(_x_n)[0]
                                # z_vae is NOT normalized by latent_mean/std
                                # (vae.encode output is raw latent)
                                seq_v, h_v, w_v = vae._decode_trunk(z_vae)
                                c_v = seq_v.shape[-1]
                                raw_v = vae.dec_conv(
                                    seq_v.permute(0, 2, 1).reshape(
                                        -1, c_v, h_v, w_v))
                                vae_recon_feats = vae.denormalize_and_split(
                                    raw_v)
                            vae_dpt_in = raw_no_cls_to_dpt_input(
                                vae_recon_feats, backbone_norm, actual_v)
                            vae_dpt_out = decode_dpt(
                                vae_dpt_in, rae.rae_cl_decoder, H, W)
                            vae_depth = vae_dpt_out['depth']
                            # Use GT RGB for coloring
                            _build_pointcloud(vae_dpt_out, vae_depth,
                                              imgs_01, actual_v, "vae_recon")
                        except Exception as e:
                            print(f"    [WARN] vae_recon point cloud "
                                  f"failed: {e}")

                    # ── Re-encode the GENERATED video through DA3+VAE ──
                    # Feed the diffusion-generated RGB frames back into the DA3
                    # multi-view encoder (cross-view attention) + feature VAE,
                    # then decode → DPT. Tests whether re-encoding the generated
                    # pixels yields more multi-view-consistent geometry than the
                    # diffused latent decoded directly ("pred").
                    if has_rgb:
                        try:
                            with torch.amp.autocast("cuda",
                                                    enabled=use_amp,
                                                    dtype=amp_dtype):
                                # rgb_pred_01 is (V,3,H,W) in [0,1]; invert the
                                # encoder normalization to match DA3 input.
                                _regen = (
                                    (rgb_pred_01.to(device)
                                     - encoder_mean.squeeze(0))
                                    / encoder_std.squeeze(0)
                                )
                                _regen_5d = _regen.unsqueeze(0)
                                _rf = rae.encode(_regen_5d, mode="all")
                                _rf_nc = {k: v[:, 1:, :]
                                          for k, v in _rf.items()}
                                _rx_n = vae.normalize_levels(
                                    _rf_nc, image_size=(H, W))
                                _rz = vae.encode(_rx_n)[0]
                                _rseq, _rh, _rw = vae._decode_trunk(_rz)
                                _rc = _rseq.shape[-1]
                                _rraw = vae.dec_conv(
                                    _rseq.permute(0, 2, 1).reshape(
                                        -1, _rc, _rh, _rw))
                                _regen_feats = vae.denormalize_and_split(_rraw)
                            regen_dpt_in = raw_no_cls_to_dpt_input(
                                _regen_feats, backbone_norm, actual_v)
                            regen_dpt_out = decode_dpt(
                                regen_dpt_in, rae.rae_cl_decoder, H, W)
                            regen_depth = regen_dpt_out['depth']
                            _build_pointcloud(regen_dpt_out, regen_depth,
                                              rgb_pred_01, actual_v,
                                              "regen_from_video")
                        except Exception as e:
                            print(f"    [WARN] regen_from_video point cloud "
                                  f"failed: {e}")

                        # ── Dump latents for pred vs regen vs vae_recon ──
                        # All three are RAW VAE-encoder-space latents (V,C,h,w):
                        #   z_sampled : diffusion output      -> pred geometry
                        #   _rz       : DA3-encode(pred RGB)   -> regen geometry
                        #   z_vae     : DA3-encode(GT frames)  -> vae_recon geom
                        # Lets us quantify why identical-ish RGB yields totally
                        # different geometry (pred misaligned, regen aligned).
                        try:
                            _lat = {
                                "z_pred": z_sampled.float().cpu().numpy(),
                                "z_regen": _rz.float().cpu().numpy(),
                            }
                            if "z_vae" in dir():
                                _lat["z_vae"] = z_vae.float().cpu().numpy()

                            # Decoded geometry (depth + world-frame ray dirs) so
                            # we can prove: pred vs regen share ~same RGB but
                            # differ in per-view ray geometry (alignment).
                            def _packg(dpt_out, pfx):
                                if dpt_out is None:
                                    return
                                _d = dpt_out.get("depth")
                                _r = dpt_out.get("ray")
                                if _d is not None:
                                    _lat[f"{pfx}_depth"] = (
                                        _d.squeeze(1).float().cpu().numpy())
                                if _r is not None:
                                    _lat[f"{pfx}_ray"] = _ray_to_numpy(_r)
                            if "gen_dpt_out" in dir():
                                _packg(gen_dpt_out, "pred")
                            _packg(regen_dpt_out, "regen")
                            if "vae_dpt_out" in dir():
                                _packg(vae_dpt_out, "vae")

                            # Decoded RGB for pred and regen (test "same RGB").
                            _lat["pred_rgb"] = rgb_pred_01.float().cpu().numpy()
                            with torch.amp.autocast("cuda", enabled=use_amp,
                                                    dtype=amp_dtype):
                                _rrgb = vae.rgb_head(_rseq, _rh, _rw,
                                                     num_views=actual_v)
                            _lat["regen_rgb"] = denorm(_rrgb).float().cpu().numpy()

                            _save_geom_npz(
                                os.path.join(
                                    ds_out_dir, f"{s_idx:03d}_latents.npz"),
                                _lat)
                            print(f"    [dump-latents] {s_idx:03d}_latents.npz "
                                  f"({', '.join(sorted(_lat.keys()))})")
                        except Exception as e:
                            print(f"    [WARN] latent dump failed: {e}")

                        # ── Immutable geometry-alignment cache (Step 2) ──
                        # One self-contained npz per scene with everything the
                        # geometry-decoder finetune needs, all from the FROZEN
                        # source DiT + OLD VAE. Requires --mode recon so z_clean
                        # and the identity targets exist. Teacher geometry ==
                        # OLD dec_conv+DPT on z_regen (== G_old(z_regen)); it is
                        # NEVER recomputed with the student decoder.
                        _geo_cache_dir = os.environ.get("GEO_CACHE_DIR")
                        if (_geo_cache_dir and "z_vae" in dir()
                                and regen_dpt_out is not None
                                and vae_dpt_out is not None
                                and len(pose_list) == actual_v
                                and len(intri_list) == actual_v):
                            try:
                                import hashlib as _hl
                                _cache = {}

                                def _packf(dpt_out, pfx):
                                    _d = dpt_out.get("depth")
                                    _r = dpt_out.get("ray")
                                    _rc = dpt_out.get("ray_conf")
                                    if _d is not None:
                                        _cache[f"{pfx}_depth"] = (
                                            _d.squeeze(1).float().cpu()
                                            .numpy().astype(np.float16))
                                    if _r is not None:
                                        _cache[f"{pfx}_ray"] = _ray_to_numpy(
                                            _r).astype(np.float16)
                                    if _rc is not None:
                                        _cache[f"{pfx}_ray_conf"] = (
                                            _rayconf_to_numpy(_rc)
                                            .astype(np.float16))

                                # Raw VAE latents (student inputs) — keep fp32.
                                _cache["z_pred"] = z_sampled.float().cpu().numpy()
                                _cache["z_regen"] = _rz.float().cpu().numpy()
                                _cache["z_clean"] = z_vae.float().cpu().numpy()
                                # Teacher = G_old(z_regen); regen-identity target
                                # is identical so we alias on load.
                                _packf(regen_dpt_out, "teacher")
                                _packf(vae_dpt_out, "cleanid")
                                # Validation-only GT geometry (not a train loss).
                                if ("gt_dpt_out" in dir()
                                        and gt_dpt_out is not None):
                                    _packf(gt_dpt_out, "gt")
                                if "gen_dpt_out" in dir() and gen_dpt_out is not None:
                                    _packf(gen_dpt_out, "pred")

                                _cache["gt_c2w"] = np.stack(
                                    [np.asarray(p, dtype=np.float64)
                                     for p in pose_list])
                                _intr_c = torch.from_numpy(
                                    np.stack([np.asarray(k, dtype=np.float32)
                                              for k in intri_list]))
                                _cache["gt_K"] = (
                                    intrinsic_to_K(_intr_c, device).cpu()
                                    .numpy().astype(np.float64))
                                _cache["cond_num"] = np.int64(scene_cond)
                                _cache["view_count"] = np.int64(actual_v)
                                _cache["image_size"] = np.array(
                                    [H, W], dtype=np.int64)
                                _cache["raw_latent"] = np.bool_(True)
                                _rgb_bytes = (
                                    rgb_pred_01.float().cpu().numpy().tobytes())
                                _cache["rgb_pred_sha"] = np.bytes_(
                                    _hl.sha1(_rgb_bytes).hexdigest().encode())
                                _cache["dit_ckpt"] = np.bytes_(
                                    str(args.dit_ckpt).encode())
                                _cache["pose_conv_version"] = np.bytes_(
                                    b"camera_from_ray_fixed_v1")

                                # The multi-DS launcher hardcodes --dataset
                                # scannetpp for every dataset, so honour an
                                # explicit override for the cache sub-dir to
                                # keep RE10K/DL3DV/mvssynth/ScanNet++ separate.
                                _cds_name = (os.environ.get("GEO_CACHE_DS_NAME")
                                             or ds_name)
                                _cds = os.path.join(
                                    _geo_cache_dir, _cds_name)
                                os.makedirs(_cds, exist_ok=True)
                                _save_geom_npz(
                                    os.path.join(_cds, f"{s_idx:03d}.npz"),
                                    _cache)
                                print(f"    [geo-cache] {_cds_name}/"
                                      f"{s_idx:03d}.npz "
                                      f"({', '.join(sorted(_cache.keys()))})")
                            except Exception as e:
                                print(f"    [WARN] geo-cache dump failed: {e}")

                        # ── Per-channel RGB-vs-geometry sensitivity probe ──
                        # Property of the (fixed) feature-VAE decoder + RGB/DPT
                        # heads, independent of the diffusion ckpt. For each of
                        # the 128 latent channels, perturb it by +1 global std on
                        # a clean-frame latent (z_vae) and measure how much the
                        # decoded RGB vs decoded geometry (depth+ray) changes.
                        if (os.environ.get("PROBE_CHANNELS") == "1"
                                and "z_vae" in dir()):
                            try:
                                _probe_out = os.environ.get(
                                    "PROBE_OUT",
                                    "/local-ssd/gld_eval_logs/"
                                    "channel_probe.npz")
                                _Vp = min(4, actual_v)
                                _z0 = z_vae.detach()[:_Vp].clone()
                                _gstd = float(z_vae.std().item())
                                _C = _z0.shape[1]

                                def _dec_both(z):
                                    s, hh, ww = vae._decode_trunk(z)
                                    _rgb = denorm(vae.rgb_head(
                                        s, hh, ww, num_views=z.shape[0]))
                                    cc = s.shape[-1]
                                    raw = vae.dec_conv(
                                        s.permute(0, 2, 1).reshape(
                                            -1, cc, hh, ww))
                                    feats = vae.denormalize_and_split(raw)
                                    din = raw_no_cls_to_dpt_input(
                                        feats, backbone_norm, z.shape[0])
                                    dpt = decode_dpt(
                                        din, rae.rae_cl_decoder, H, W)
                                    return _rgb.float(), dpt

                                drgb = np.zeros(_C)
                                ddep = np.zeros(_C)
                                dray = np.zeros(_C)
                                with torch.amp.autocast(
                                        "cuda", enabled=use_amp,
                                        dtype=amp_dtype):
                                    rgb0, g0 = _dec_both(_z0)
                                    d0 = g0["depth"].float()
                                    r0 = g0["ray"].float()
                                    for c in range(_C):
                                        zp = _z0.clone()
                                        zp[:, c] = zp[:, c] + _gstd
                                        rgb1, g1 = _dec_both(zp)
                                        drgb[c] = (rgb1 - rgb0).abs().mean().item()
                                        ddep[c] = (g1["depth"].float()
                                                   - d0).abs().mean().item()
                                        dray[c] = (g1["ray"].float()
                                                   - r0).abs().mean().item()
                                        if c % 32 == 0:
                                            print(f"    [probe] ch {c}/{_C}")
                                os.makedirs(os.path.dirname(_probe_out),
                                            exist_ok=True)
                                np.savez(_probe_out, drgb=drgb, ddep=ddep,
                                         dray=dray, gstd=_gstd,
                                         d0_scale=float(d0.abs().mean().item()),
                                         r0_scale=float(r0.abs().mean().item()))
                                print(f"    [probe] saved {_probe_out}")
                            except Exception as e:
                                print(f"    [WARN] channel probe failed: {e}")

            # Free per-scene scannetpp / re10k_packed frame cache (~50 MB/scene
            # at V=64) so multi-scene eval doesn't accumulate hundreds of MB of
            # decoded RGB. No-op for sharded-PNG datasets.
            if ds_type in ("scannetpp", "re10k_packed"):
                clear_scannetpp_frame_cache(scene_dir)

        if deferred_pc_jobs and args.save_pointcloud:
            print(f"\n  [pc] generating {len(deferred_pc_jobs)} deferred point cloud(s) "
                  f"(after all scene videos saved) ...")
            for job in deferred_pc_jobs:
                try:
                    if job["kind"] == "t2v":
                        _save_pointcloud_from_latents(
                            job["z_sampled"], job["rgb_imgs"],
                            s_idx=job["s_idx"], ds_out_dir=job["ds_out_dir"],
                            tag="pred", vae=vae, rae=rae, backbone_norm=backbone_norm,
                            H=H, W=W, device=device, use_amp=use_amp,
                            amp_dtype=amp_dtype, pc_stride=args.pc_stride,
                        )
                    elif job["kind"] == "autoregress":
                        _save_autoregress_pointclouds(
                            job["z_sampled"], job["rgb_imgs"], job["chunk_z_list"],
                            s_idx=job["s_idx"], ds_out_dir=job["ds_out_dir"],
                            vae=vae, rae=rae, backbone_norm=backbone_norm,
                            H=H, W=W, device=device, use_amp=use_amp,
                            amp_dtype=amp_dtype, pc_stride=args.pc_stride,
                            roll_cond_num=roll_cond_num,
                            has_rgb=job["has_rgb"], denorm_fn=denorm,
                        )
                except Exception as e:
                    print(f"    [WARN] deferred point cloud failed for "
                          f"{job['s_idx']:03d}: {e}")
                finally:
                    torch.cuda.empty_cache()

        # ── Summary ──
        ds_result = {}
        print(f"\n  [{ds_name}] Summary:")
        if all_loss:
            ds_result["tgt_loss"] = float(np.mean(all_loss))
            print(f"    Tgt Loss: {ds_result['tgt_loss']:.4f}")
        if all_feat:
            avg = {k: float(np.mean([m[k] for m in all_feat])) for k in all_feat[0]}
            ds_result["feat_l1"] = avg
            print("    Feature L1: " +
                  ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        if all_rgb:
            ds_result["rgb_psnr"] = float(np.mean([m["psnr"] for m in all_rgb]))
            ds_result["rgb_l1"] = float(np.mean([m["l1"] for m in all_rgb]))
            ds_result["rgb_ssim"] = float(np.mean([m.get("ssim", 0) for m in all_rgb]))
            print(f"    RGB   PSNR: {ds_result['rgb_psnr']:.2f} dB  "
                  f"L1: {ds_result['rgb_l1']:.4f}  SSIM: {ds_result['rgb_ssim']:.4f}")
        if all_rgb_vae_recon:
            ds_result["vae_recon_psnr"] = float(np.mean([m["psnr"] for m in all_rgb_vae_recon]))
            ds_result["vae_recon_l1"] = float(np.mean([m["l1"] for m in all_rgb_vae_recon]))
            ds_result["vae_recon_ssim"] = float(np.mean([m.get("ssim", 0) for m in all_rgb_vae_recon]))
            print(f"    VAE Recon: PSNR: {ds_result['vae_recon_psnr']:.2f} dB  "
                  f"L1: {ds_result['vae_recon_l1']:.4f}  SSIM: {ds_result['vae_recon_ssim']:.4f}")
        if all_depth:
            ds_result["depth_psnr"] = float(np.mean([m["psnr"] for m in all_depth]))
            ds_result["depth_l1"] = float(np.mean([m["l1"] for m in all_depth]))
            print(f"    Depth PSNR: {ds_result['depth_psnr']:.2f} dB  "
                  f"L1: {ds_result['depth_l1']:.4f}")
        if all_ref_tgt_pc_gap:
            ds_result["ref_tgt_pc_gap"] = float(np.mean(all_ref_tgt_pc_gap))
            ds_result["ref_tgt_pc_gap_n"] = int(len(all_ref_tgt_pc_gap))
            print(f"    Ref–tgt PC gap: {ds_result['ref_tgt_pc_gap']:.6f}  "
                  f"(n={ds_result['ref_tgt_pc_gap_n']})")
        global_results[ds_name] = ds_result

    print("\n" + "=" * 60)
    print(f"Global Summary  (V4, mode={args.mode})")
    print("=" * 60)
    for ds_name, r in global_results.items():
        parts = [f"[{ds_name}]"]
        if "tgt_loss" in r:
            parts.append(f"TgtLoss={r['tgt_loss']:.4f}")
        if "rgb_psnr" in r:
            parts.append(f"RGB={r['rgb_psnr']:.2f}dB")
        if "rgb_ssim" in r:
            parts.append(f"SSIM={r['rgb_ssim']:.4f}")
        if "vae_recon_psnr" in r:
            parts.append(f"VAE={r['vae_recon_psnr']:.2f}dB")
        if "depth_psnr" in r:
            parts.append(f"Depth={r['depth_psnr']:.2f}dB")
        if "feat_l1" in r:
            parts.append(f"FeatL1={r['feat_l1']['avg']:.4f}")
        if "ref_tgt_pc_gap" in r:
            parts.append(f"RefTgtPC={r['ref_tgt_pc_gap']:.6f}")
        print("  " + "  ".join(parts))
    if args.timing_json:
        _sync_cuda()
        decoder_modules = [
            # `post_quant_conv` is traversed immediately before `decoder` by
            # diffusers SD/Wan VAEs.  FeatureVAE/codec backends simply lack it.
            getattr(vae, "post_quant_conv", None),
            getattr(vae, "dec_proj", None),
            getattr(vae, "dec_upsample", None),
            getattr(vae, "dec_attn", None),
            getattr(vae, "dec_attn_norm", None),
            getattr(vae, "dec_conv", None),
            getattr(vae, "rgb_head", None),
            getattr(vae, "decoder", None),
        ]
        geometry_decoder = getattr(rae, "rae_cl_decoder", None) if rae is not None else None
        geometry_propagation_modules = []
        if latent_backend == "da3_direct":
            if not args.metrics_only:
                geometry_propagation_modules = _da3_direct_geometry_propagation_modules(vae)
            # Level-3 DA3-direct cannot propagate a level-0 latent through the
            # remaining backbone to create geometry, so it correctly stays RGB-only.
            if not geometry_propagation_modules:
                geometry_decoder = None
        timing_payload = {
            "diffusion": timing["diffusion"],
            "latent_vae_decode": timing["latent_vae_decode"],
            "rgb_geometry_decode": timing["rgb_geometry_decode"],
            "compute_total": (
                timing["diffusion"]
                + timing["latent_vae_decode"]
                + timing["rgb_geometry_decode"]
            ),
            "dit_params": _count_module_params(dit),
            "rgb_decoder_params": _count_module_params(*decoder_modules),
            "geometry_decoder_params": _count_module_params(geometry_decoder),
            "geometry_propagation_params": _count_module_params(
                *geometry_propagation_modules),
            "vae_decoder_params": _count_module_params(*decoder_modules),
            "rgb_geometry_output_params": _count_module_params(
                *decoder_modules, *geometry_propagation_modules, geometry_decoder),
            "timing_scope": (
                "primary_output" if primary_output_timing else "all_decoder_calls"),
            "note": "rgb_geometry_output_params includes the RGB latent decoder and every learned geometry module exercised by this evaluation. DA3-direct level-0 geometry includes propagation attention blocks, final norm, camera token, and the DA3 DPT head. Excludes model/data loading, metrics, and artifact saving; CUDA-synchronized function timings.",
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.timing_json)), exist_ok=True)
        with open(args.timing_json, "w") as f:
            json.dump(timing_payload, f, indent=2, sort_keys=True)
        print(f"Timing saved to {args.timing_json}: {timing_payload}")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "dataset": args.dataset,
            "num_scenes": args.num_scenes,
            "num_views": args.num_views,
            "cond_num": args.cond_num,
            "sample_steps": args.sample_steps,
            "cfg_scale": args.cfg_scale,
            "guidance": args.guidance,
            "sample_interval": args.sample_interval,
            "reference_conditioning": reference_conditioning,
            "ref_tgt_pc_gap": args.ref_tgt_pc_gap,
            "ckpt": args.dit_ckpt,
            "config": args.config,
            "results": global_results,
        }, f, indent=2, sort_keys=True)
    print(f"\nMetrics saved to {metrics_path}")
    _sync_staged_output(staged_output_dir, final_output_dir)
    if not args.loss_only:
        print(f"\nResults saved to {final_output_dir}/")


if __name__ == "__main__":
    main()
