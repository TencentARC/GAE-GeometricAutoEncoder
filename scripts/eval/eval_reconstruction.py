"""
Evaluate Feature VAE v2 reconstruction quality across datasets.

v2 has an integrated MAE-style RGB decoder (RGBHead), so no external MAE
decoder is needed.  Depth evaluation still uses the DPT decoder.

Usage:
    # RE10K (default)
    PYTHONPATH=src python scripts/eval/eval_reconstruction.py \
        --vae-ckpt ckpts/gae_128.pt \
        --dataset re10k --num-scenes 10 --num-views 1

    # DL3DV
    PYTHONPATH=src python scripts/eval/eval_reconstruction.py \
        --vae-ckpt ckpts/gae_64.pt \
        --dataset dl3dv --num-scenes 10 --num-views 4

    # MVS-Synth
    PYTHONPATH=src python scripts/eval/eval_reconstruction.py \
        --vae-ckpt ckpts/gae_64.pt \
        --dataset mvssynth --num-scenes 10 --num-views 4

    # Sintel (dynamic scenes — 50-frame mp4 clips with GT depth/pose)
    PYTHONPATH=src python scripts/eval/eval_reconstruction.py \
        --vae-ckpt ckpts/gae_64.pt \
        --dataset sintel --num-scenes 10 --num-views 4

    # All datasets at once (now includes sintel)
    PYTHONPATH=src python scripts/eval/eval_reconstruction.py \
        --vae-ckpt ckpts/gae_64.pt \
        --dataset all --num-scenes 8 --num-views 4 --fast
"""

import argparse
import json
import os
import sys
import random

os.environ.setdefault("TMPDIR", "/tmp")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf

from stage1.da3 import DA3Backbone
from stage1.gae_codec import GAECodec, DA3_LEVEL_DIM


_DATA_ROOT = os.environ.get("GAE_DATA_ROOT", "/data/gae")
DATASET_ROOTS = {
    "re10k": os.path.join(_DATA_ROOT, "raw/re10k/test"),
    "re10k_packed": os.path.join(_DATA_ROOT, "re10k_packed/test"),
    "dl3dv": os.path.join(_DATA_ROOT, "raw/dl3dv"),
    "dl3dv_packed": os.path.join(_DATA_ROOT, "dl3dv_packed"),
    "mvssynth": os.path.join(_DATA_ROOT, "raw/mvssynth"),
    "sintel": os.path.join(_DATA_ROOT, "raw/sintel"),
}

# Where to drop the per-scene PNG cache decoded from Sintel mp4 clips.
# /local-ssd is preferred (fast NVMe, survives across runs); /tmp is fallback.
SINTEL_CACHE_ROOT = "/local-ssd/_eval_cache/sintel" if os.path.isdir("/local-ssd") \
    else "/tmp/_eval_cache/sintel"

# Pre-defined scene directories for fast startup (avoids slow EFS scanning)
FAST_SCENES = {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/gae_64.yaml")
    p.add_argument("--dataset", type=str, default="re10k",
                   choices=["re10k", "re10k_packed", "dl3dv", "dl3dv_packed",
                            "mvssynth", "sintel", "all"],
                   help="dataset to evaluate on (default: re10k). "
                        "'sintel' is a dynamic-scene video benchmark "
                        "(GeometryCrafter Sintel_video).")
    p.add_argument("--data-root", type=str, default=None,
                   help="override dataset root (auto-detected from --dataset)")
    p.add_argument("--dpt-decoder", type=str, default=None,
                   help="override stage_1.params.dpt_decoder_path")
    p.add_argument("--da3-weights", type=str, default=None,
                   help="override stage_1.params.da3_weights_path")
    p.add_argument("--num-scenes", type=int, default=4)
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="results/eval_reconstruction")
    p.add_argument("--resolution", type=int, nargs=2, default=[504, 504])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fast", action="store_true",
                   help="use pre-defined scene paths (skip slow EFS scanning)")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────

def collect_packed_scenes(data_root, num_scenes, num_views, seed, ds_type):
    """Collect flat <scene>/{video.mp4,meta.json} packed scenes."""
    rng = random.Random(seed)
    candidates = []
    for entry in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, entry)
        if (os.path.isfile(os.path.join(scene_dir, "video.mp4")) and
                os.path.isfile(os.path.join(scene_dir, "meta.json"))):
            candidates.append((entry, scene_dir))
    rng.shuffle(candidates)
    scenes = []
    for scene_name, scene_dir in candidates:
        try:
            with open(os.path.join(scene_dir, "meta.json")) as handle:
                meta = json.load(handle)
            count = int(meta.get("num_frames", len(meta.get("frames", []))))
        except Exception:
            continue
        if count < num_views:
            continue
        idx = np.linspace(0, count - 1, num_views).round().astype(int)
        scenes.append((scene_name, scene_dir, idx.tolist(), ds_type))
        if len(scenes) >= num_scenes:
            break
    return scenes


def load_packed_image(scene_dir, frame_index, resolution):
    import cv2
    """Decode one packed MP4 frame and apply the evaluator's ImageNet transform."""
    cap = cv2.VideoCapture(os.path.join(scene_dir, "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"cannot decode frame {frame_index} from {scene_dir}/video.mp4")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame).resize((resolution[1], resolution[0]), Image.Resampling.LANCZOS)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])(image)



def collect_scenes_re10k(data_root, num_scenes, num_views, seed):
    """RE10K: ROOT/{shard}/{scene_id}/{timestamp}.png"""
    rng = random.Random(seed)
    candidates = []
    for shard in sorted(os.listdir(data_root)):
        shard_dir = os.path.join(data_root, shard)
        if not os.path.isdir(shard_dir):
            continue
        for scene in sorted(os.listdir(shard_dir)):
            scene_dir = os.path.join(shard_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            pngs = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            if len(pngs) >= num_views:
                candidates.append((scene, scene_dir, pngs))
    rng.shuffle(candidates)
    results = []
    for scene, scene_dir, pngs in candidates[:num_scenes]:
        chosen = rng.sample(pngs, num_views) if num_views < len(pngs) else pngs[:num_views]
        paths = [os.path.join(scene_dir, p) for p in sorted(chosen)]
        results.append((f"re10k/{scene}", paths))
    return results


def collect_scenes_dl3dv(data_root, num_scenes, num_views, seed):
    """DL3DV: ROOT/{split}/{scene_hash}/images_4/{frame}.png
    or ROOT/{split}/{scene_hash}/nerfstudio/images_4/{frame}.png

    Only scans the first split that yields enough scenes to avoid
    slow EFS enumeration of 10K+ directories.
    """
    rng = random.Random(seed)
    candidates = []
    need = num_scenes * 3
    for split in sorted(os.listdir(data_root)):
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for scene in sorted(os.listdir(split_dir)):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            ns = os.path.join(scene_dir, "nerfstudio")
            actual = ns if os.path.isdir(ns) else scene_dir
            img_dir = os.path.join(actual, "images_4")
            if not os.path.isdir(img_dir):
                continue
            pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if len(pngs) >= num_views:
                candidates.append((f"{split}/{scene}", img_dir, pngs))
            if len(candidates) >= need:
                break
        if len(candidates) >= need:
            break
    rng.shuffle(candidates)
    results = []
    for scene, img_dir, pngs in candidates[:num_scenes]:
        chosen = rng.sample(pngs, num_views) if num_views < len(pngs) else pngs[:num_views]
        paths = [os.path.join(img_dir, p) for p in sorted(chosen)]
        results.append((f"dl3dv/{scene}", paths))
    return results


def collect_scenes_mvssynth(data_root, num_scenes, num_views, seed):
    """MVS-Synth: ROOT/{scene_id}/images/{XXXX}.png"""
    rng = random.Random(seed)
    candidates = []
    for scene in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, scene)
        img_dir = os.path.join(scene_dir, "images")
        if not os.path.isdir(img_dir):
            continue
        pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if len(pngs) >= num_views:
            candidates.append((scene, img_dir, pngs))
    rng.shuffle(candidates)
    results = []
    for scene, img_dir, pngs in candidates[:num_scenes]:
        chosen = rng.sample(pngs, num_views) if num_views < len(pngs) else pngs[:num_views]
        paths = [os.path.join(img_dir, p) for p in sorted(chosen)]
        results.append((f"mvssynth/{scene}", paths))
    return results


def _decode_sintel_mp4_to_pngs(mp4_path, png_dir):
    """Decode every frame of a Sintel mp4 into <png_dir>/{NNNNN}.png (idempotent).

    Sintel clips are short (50 frames at 872×436) so we just dump all frames on
    the first access. Subsequent calls hit the existing cache and return fast.
    Returns the sorted list of png basenames.
    """
    import cv2
    existing = sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))
    if existing:
        return existing
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {mp4_path}")
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(png_dir, f"{i:05d}.png"), frame)
        i += 1
    cap.release()
    return sorted(f for f in os.listdir(png_dir) if f.endswith(".png"))


def collect_scenes_sintel(data_root, num_scenes, num_views, seed):
    """Sintel video: ROOT/<scene>/00000_NNNNN_rgb.mp4 (+ .hdf5 with GT).

    Each scene is a short dynamic-content clip (~50 frames at 872×436, 24 fps).
    We decode every clip on first access into SINTEL_CACHE_ROOT/<scene>/ so the
    rest of the eval pipeline (PIL load_image) works unchanged.

    Frames are picked EVENLY SPACED across the clip, not random — for a short
    dynamic sequence this gives a representative temporal slice rather than
    accidentally drawing several near-duplicate frames.
    """
    rng = random.Random(seed)
    if not os.path.isdir(data_root):
        print(f"  [sintel] data_root not found: {data_root}")
        return []
    os.makedirs(SINTEL_CACHE_ROOT, exist_ok=True)

    candidates = []
    for scene in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        mp4s = sorted(f for f in os.listdir(scene_dir) if f.endswith("_rgb.mp4"))
        if not mp4s:
            continue
        candidates.append((scene, os.path.join(scene_dir, mp4s[0])))

    if not candidates:
        print(f"  [sintel] no *_rgb.mp4 found under {data_root}")
        return []

    rng.shuffle(candidates)
    results = []
    skipped_short = 0
    for scene, mp4_path in candidates[:num_scenes]:
        png_dir = os.path.join(SINTEL_CACHE_ROOT, scene)
        os.makedirs(png_dir, exist_ok=True)
        try:
            pngs = _decode_sintel_mp4_to_pngs(mp4_path, png_dir)
        except Exception as e:
            print(f"  [sintel/{scene}] decode failed: {e}")
            continue
        if len(pngs) < num_views:
            skipped_short += 1
            continue

        # Evenly spaced selection — deterministic and better for a 50-frame clip.
        if num_views == 1:
            idxs = [len(pngs) // 2]
        else:
            step = (len(pngs) - 1) / (num_views - 1)
            idxs = [int(round(i * step)) for i in range(num_views)]
        paths = [os.path.join(png_dir, pngs[i]) for i in idxs]
        results.append((f"sintel/{scene}", paths))

    if not results:
        # Common cause: --num-views (or --v-long) exceeds every sintel clip length.
        print(f"  [sintel] returning 0 scenes — found {len(candidates)} mp4 clips, "
              f"but {skipped_short} were shorter than num_views={num_views}. "
              f"Sintel clips are typically 20–50 frames; lower --num-views/--v-long.")
    return results


def collect_scenes_fast(dataset, num_scenes, num_views, seed):
    """Use pre-defined scene paths — no EFS directory scanning."""
    rng = random.Random(seed)
    dirs = FAST_SCENES.get(dataset, [])
    rng.shuffle(dirs)
    results = []
    for scene_dir in dirs[:num_scenes]:
        if not os.path.isdir(scene_dir):
            continue
        pngs = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
        if len(pngs) < num_views:
            continue
        chosen = rng.sample(pngs, num_views) if num_views < len(pngs) else pngs[:num_views]
        paths = [os.path.join(scene_dir, p) for p in sorted(chosen)]
        tag = os.path.basename(os.path.dirname(scene_dir))
        results.append((f"{dataset}/{tag}", paths))
    return results


def collect_scenes(dataset, data_root, num_scenes, num_views, seed, fast=False):
    # Sintel only has ~20 clips; the directory scan is essentially free, and its
    # mp4 layout doesn't fit the FAST collector's "PNG dir per scene" assumption.
    # Always route to the dedicated collector (which also handles mp4 decode + cache).
    if dataset == "sintel":
        return collect_scenes_sintel(data_root, num_scenes, num_views, seed)
    if fast:
        return collect_scenes_fast(dataset, num_scenes, num_views, seed)
    collectors = {
        "re10k": collect_scenes_re10k,
        "dl3dv": collect_scenes_dl3dv,
        "mvssynth": collect_scenes_mvssynth,
    }
    return collectors[dataset](data_root, num_scenes, num_views, seed)


def load_image(path, resolution):
    img = Image.open(path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img)


def tensor_to_numpy_img(t):
    if t.ndim == 4:
        t = t[0]
    return (t.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def depth_to_numpy_img(d):
    import matplotlib.cm as cm
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    d = d.float().cpu()
    dmin, dmax = d.min(), d.max()
    if dmax - dmin > 1e-6:
        d = (d - dmin) / (dmax - dmin)
    else:
        d = torch.zeros_like(d)
    colored = cm.viridis(d.numpy())[:, :, :3]
    return (colored * 255).astype(np.uint8)


def compute_metrics(gt, pred):
    mse = F.mse_loss(pred, gt).item()
    psnr = -10 * np.log10(mse + 1e-10)
    l1 = F.l1_loss(pred, gt).item()
    return {"psnr": psnr, "l1": l1}


# ── DPT helpers ───────────────────────────────────────────────────────────

def raw_to_dpt_input(feats_with_cls, backbone_norm, V):
    # Infer embed_dim from backbone_norm's normalized_shape
    embed_dim = backbone_norm.normalized_shape[0]
    result = []
    for lvl in sorted(feats_with_cls.keys()):
        feat = feats_with_cls[lvl]
        cls_raw = feat[:, 0, :]
        patches = feat[:, 1:, :]
        local_part = patches[:, :, :embed_dim]
        curr_norm = backbone_norm(patches[:, :, embed_dim:])
        patches_ln = torch.cat([local_part, curr_norm], dim=-1)
        result.append((patches_ln.unsqueeze(0), cls_raw.unsqueeze(0)))
    return result


def raw_no_cls_to_dpt_input(feats_no_cls, backbone_norm, V):
    # Infer embed_dim from backbone_norm's normalized_shape
    embed_dim = backbone_norm.normalized_shape[0]
    hidden_size = embed_dim * 2
    result = []
    for lvl in sorted(feats_no_cls.keys()):
        patches = feats_no_cls[lvl]
        cls_raw = torch.zeros(V, hidden_size, device=patches.device, dtype=patches.dtype)
        local_part = patches[:, :, :embed_dim]
        curr_norm = backbone_norm(patches[:, :, embed_dim:])
        patches_ln = torch.cat([local_part, curr_norm], dim=-1)
        result.append((patches_ln.unsqueeze(0), cls_raw.unsqueeze(0)))
    return result


@torch.no_grad()
def decode_dpt(dpt_feats, dpt_decoder, H, W):
    """Run DPT decoder → dict with 'depth', 'ray', 'ray_conf'."""
    with torch.autocast(device_type=dpt_feats[0][0].device.type, enabled=False):
        output = dpt_decoder(dpt_feats, H, W, patch_start_idx=0)
    depth = output.get("depth", None)
    if depth is not None:
        if depth.ndim == 5:
            depth = depth.reshape(-1, *depth.shape[2:])
        elif depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0).unsqueeze(1)
    return {
        'depth': depth,
        'ray': output.get('ray', None),
        'ray_conf': output.get('ray_conf', None),
    }


def _c2w_list_to_evo_traj(c2w_list):
    """Convert list of (4,4) c2w matrices to evo PoseTrajectory3D."""
    from evo.core.trajectory import PoseTrajectory3D
    from scipy.spatial.transform import Rotation as R
    poses = [np.asarray(p, dtype=np.float64) for p in c2w_list]
    pos = np.array([p[:3, 3] for p in poses])
    quats = np.array([R.from_matrix(p[:3, :3]).as_quat() for p in poses])
    quats_wxyz = quats[:, [3, 0, 1, 2]]
    return PoseTrajectory3D(
        positions_xyz=pos, orientations_quat_wxyz=quats_wxyz,
        timestamps=np.arange(len(poses), dtype=np.float64))


def _compute_evo_metrics(gt_c2w_list, pred_c2w_list):
    """Compute APE/RPE using evo (Sim3 aligned). Returns dict."""
    import copy
    from evo.core import metrics
    traj_gt = _c2w_list_to_evo_traj(gt_c2w_list)
    traj_pred = _c2w_list_to_evo_traj(pred_c2w_list)

    traj_pred_aligned = copy.deepcopy(traj_pred)
    try:
        traj_pred_aligned.align(traj_gt, correct_scale=True)
    except Exception:
        return None, traj_gt, traj_pred

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_gt, traj_pred_aligned))
    ate_rmse = ape.get_statistic(metrics.StatisticsType.rmse)

    rpe_t = metrics.RPE(metrics.PoseRelation.translation_part,
                        delta=1, delta_unit=metrics.Unit.frames, all_pairs=True)
    rpe_t.process_data((traj_gt, traj_pred_aligned))
    rpe_t_rmse = rpe_t.get_statistic(metrics.StatisticsType.rmse)

    rpe_r = metrics.RPE(metrics.PoseRelation.rotation_angle_deg,
                        delta=1, delta_unit=metrics.Unit.frames, all_pairs=True)
    rpe_r.process_data((traj_gt, traj_pred_aligned))
    rpe_r_rmse = rpe_r.get_statistic(metrics.StatisticsType.rmse)

    result = {
        'ate_rmse': float(ate_rmse),
        'rpe_t_rmse': float(rpe_t_rmse),
        'rpe_r_rmse': float(rpe_r_rmse),
        'ape_error': ape.error,
    }
    return result, traj_gt, traj_pred_aligned


def _save_pose_vis(gt_c2w_list, pred_c2w_list, save_path):
    """Visualize GT vs VAE camera poses using evo (trajectory + APE error)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evo.tools import plot

    evo_result, traj_gt, traj_pred = _compute_evo_metrics(
        gt_c2w_list, pred_c2w_list)

    fig = plt.figure(figsize=(14, 5))

    # Left: 3D trajectory comparison
    ax_traj = fig.add_subplot(121, projection="3d")
    plot.traj(ax_traj, plot.PlotMode.xyz, traj_gt,
             style='-', color='royalblue', label='GT features')
    plot.traj(ax_traj, plot.PlotMode.xyz, traj_pred,
             style='--', color='crimson', label='VAE features')
    positions = traj_gt.positions_xyz
    traj_extent = max(positions.max(0) - positions.min(0))
    axis_scale = max(traj_extent * 0.15, 1e-6)
    plot.draw_coordinate_axes(ax_traj, traj_gt, plot.PlotMode.xyz,
                              marker_scale=axis_scale)
    plot.draw_coordinate_axes(ax_traj, traj_pred, plot.PlotMode.xyz,
                              marker_scale=axis_scale)
    ax_traj.legend(fontsize=8)
    ax_traj.set_title("Trajectory (Sim3 aligned)", fontsize=10)

    # Right: APE error bar
    ax_err = fig.add_subplot(122)
    if evo_result is not None:
        plot.error_array(
            ax_err, evo_result['ape_error'], name="APE",
            statistics={
                'rmse': evo_result['ate_rmse'],
                'rpe_t': evo_result['rpe_t_rmse'],
                'rpe_r(°)': evo_result['rpe_r_rmse'],
            })
        ax_err.set_title(
            f"ATE={evo_result['ate_rmse']:.4f}  "
            f"RPE_t={evo_result['rpe_t_rmse']:.4f}  "
            f"RPE_r={evo_result['rpe_r_rmse']:.2f}°", fontsize=9)
    else:
        ax_err.text(0.5, 0.5, "Alignment failed", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _ray_to_numpy(ray):
    """Convert ray tensor to numpy (V, H, W, C) format for camera_from_ray."""
    if ray is None:
        return None
    if isinstance(ray, torch.Tensor):
        arr = ray.detach().cpu().float().numpy()
    else:
        arr = np.asarray(ray, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] in (3, 6) and arr.shape[-1] != 6:
        arr = arr.transpose(0, 2, 3, 1)
    return arr


def _rayconf_to_numpy(rc):
    """Convert ray_conf tensor to numpy (V, H, W) format."""
    if rc is None:
        return None
    if isinstance(rc, torch.Tensor):
        arr = rc.detach().cpu().float().numpy()
    else:
        arr = np.asarray(rc, dtype=np.float32)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr.squeeze(1)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


# ── Main ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    H, W = args.resolution
    V = args.num_views
    img_size = (H, W)

    cfg = OmegaConf.load(args.config)
    stage1_params = OmegaConf.to_container(
        cfg.get("stage_1", {}).get("params", {}), resolve=True
    )
    if args.dpt_decoder is not None:
        stage1_params["dpt_decoder_path"] = args.dpt_decoder
    if args.da3_weights is not None:
        stage1_params["da3_weights_path"] = args.da3_weights
    da3_weights = str(stage1_params.get("da3_weights_path", ""))
    if da3_weights and not os.path.isabs(da3_weights):
        da3_weights = os.path.join(ROOT, da3_weights)
        stage1_params["da3_weights_path"] = da3_weights
    has_dpt = bool(da3_weights and os.path.isfile(da3_weights))

    print("=" * 60)
    print("Feature VAE v2 Evaluation")
    print(f"  VAE ckpt:    {args.vae_ckpt}")
    print(f"  Config:      {args.config}")
    print(f"  Dataset(s):  {args.dataset}")
    print(f"  DA3 weights: {da3_weights or '<none>'} "
          f"{'✓' if has_dpt else '✗ (skip geometry)'}")
    print(f"  Resolution:  {H}×{W}")
    print(f"  Scenes/ds:   {args.num_scenes}, Views/scene: {V}")
    print(f"  Output:      {args.output_dir}")
    print("=" * 60)

    # ── 1. DA3 encoder ──
    print("\n[1/3] Loading DA3 encoder ...")
    rae_kwargs = dict(stage1_params)
    rae_kwargs["encoder_input_size"] = [H, W]
    rae_kwargs["reshape_to_2d"] = False
    if not has_dpt:
        rae_kwargs["da3_weights_path"] = None
    rae = DA3Backbone(**rae_kwargs).to(device).eval()
    encoder_mean = rae.encoder_mean   # (1, 3, 1, 1)
    encoder_std = rae.encoder_std     # (1, 3, 1, 1)
    backbone_norm = rae.encoder.backbone.pretrained.norm

    # ── 2. Feature VAE v2 ──
    print("[2/3] Loading Feature VAE v2 ...")
    vae_cfg = OmegaConf.to_container(cfg.get("codec", {}), resolve=True)

    vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
    vae_state = vae_ckpt.get(
        "ema_codec",
        vae_ckpt.get("ema_vae", vae_ckpt.get("codec", vae_ckpt.get("vae", vae_ckpt))),
    )

    vae = GAECodec(**vae_cfg).to(device).eval()
    load_result = vae.load_state_dict(vae_state, strict=False)
    if load_result.missing_keys:
        print(f"  ⚠ Missing keys: {load_result.missing_keys[:5]}"
              + (" ..." if len(load_result.missing_keys) > 5 else ""))
    if load_result.unexpected_keys:
        print(f"  ⚠ Unexpected keys: {load_result.unexpected_keys[:5]}"
              + (" ..." if len(load_result.unexpected_keys) > 5 else ""))

    has_rgb = vae.rgb_head is not None
    n_params = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"  Params: {n_params:.1f}M  latent={vae.latent_dim}  "
          f"RGB head: {'✓' if has_rgb else '✗'}")

    # ── 3. Evaluate ──
    print("[3/3] Collecting test scenes ...")

    if args.dataset == "all":
        datasets_to_eval = ["re10k", "dl3dv", "mvssynth", "sintel"]
    else:
        datasets_to_eval = [args.dataset]

    per_dataset_results = {}
    denorm = lambda t: (t * encoder_std.squeeze(0) + encoder_mean.squeeze(0)).clamp(0, 1)

    for ds_name in datasets_to_eval:
        ds_root = args.data_root if args.data_root else DATASET_ROOTS[ds_name]
        if ds_name in ("re10k_packed", "dl3dv_packed"):
            scenes = collect_packed_scenes(
                ds_root, args.num_scenes, V, args.seed, ds_name)
        else:
            scenes = collect_scenes(ds_name, ds_root, args.num_scenes, V, args.seed,
                                    fast=args.fast)
        print(f"\n{'─'*50}")
        print(f"  Dataset: {ds_name}  ({len(scenes)} scenes × {V} views)")
        print(f"  Root: {ds_root}")
        print(f"{'─'*50}")

        all_feat, all_rgb, all_depth, all_cam = [], [], [], []
        ds_outdir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_outdir, exist_ok=True)

        for s_idx, scene in enumerate(scenes):
            if len(scene) == 4:
                scene_name, scene_dir, frame_indices, _packed_type = scene
                imgs = torch.stack([
                    load_packed_image(scene_dir, index, img_size)
                    for index in frame_indices
                ]).to(device)
                actual_v = len(frame_indices)
            else:
                scene_name, img_paths = scene
                imgs = torch.stack([load_image(path, img_size) for path in img_paths]).to(device)
                actual_v = len(img_paths)
            print(f"  [{s_idx+1}/{len(scenes)}] {scene_name} ({actual_v} views)")
            imgs_5d = imgs.unsqueeze(0)
            imgs_01 = denorm(imgs)

            # ── Encode ──
            gt_raw = rae.encode(imgs_5d, mode="all")
            feats_no_cls = {k: v[:, 1:, :] for k, v in gt_raw.items()}

            # ── VAE forward ──
            x_norm = vae.normalize_levels(feats_no_cls, image_size=img_size)
            # num_views is required so the divided temporal RGB head attends across
            # frames (temporal: true) instead of running per-frame (V=1).
            vae_out = vae(x_norm, num_views=actual_v)
            recon_feats = vae.denormalize_and_split(vae_out["recon"])

            # ── Feature metrics ──
            recon_norm = vae.normalize_levels(recon_feats, image_size=img_size)
            level_dim = vae._level_dim
            recon_ch = recon_norm.split(level_dim, dim=1)
            input_ch = x_norm.split(level_dim, dim=1)
            feat_l1 = {f"l{i}": F.l1_loss(rc, ic).item()
                       for i, (rc, ic) in enumerate(zip(recon_ch, input_ch))}
            feat_l1["avg"] = np.mean(list(feat_l1.values()))
            all_feat.append(feat_l1)
            print("    Feature L1: " + ", ".join(f"{k}={v:.4f}" for k, v in feat_l1.items()))

            # ── RGB (from integrated RGBHead) ──
            if has_rgb and "rgb_pred" in vae_out:
                rgb_pred_01 = denorm(vae_out["rgb_pred"])
                rgb_m = compute_metrics(imgs_01, rgb_pred_01)
                all_rgb.append(rgb_m)
                print(f"    RGB  PSNR={rgb_m['psnr']:.2f}dB  L1={rgb_m['l1']:.4f}")

                rows = []
                for vi in range(actual_v):
                    rows.append(np.concatenate([
                        tensor_to_numpy_img(imgs_01[vi]),
                        tensor_to_numpy_img(rgb_pred_01[vi]),
                    ], axis=1))
                Image.fromarray(np.concatenate(rows, axis=0)).save(
                    os.path.join(ds_outdir, f"{s_idx:03d}_rgb.png"))

            # ── Depth + Camera Pose ──
            gt_depth, vae_depth = None, None
            if has_dpt and rae.rae_cl_decoder is not None:
                gt_dpt_in = raw_to_dpt_input(gt_raw, backbone_norm, actual_v)
                gt_dpt_out = decode_dpt(gt_dpt_in, rae.rae_cl_decoder, H, W)
                gt_depth = gt_dpt_out['depth']

                vae_dpt_in = raw_no_cls_to_dpt_input(recon_feats, backbone_norm, actual_v)
                vae_dpt_out = decode_dpt(vae_dpt_in, rae.rae_cl_decoder, H, W)
                vae_depth = vae_dpt_out['depth']

                if gt_depth is not None and vae_depth is not None:
                    depth_m = compute_metrics(gt_depth.float(), vae_depth.float())
                    all_depth.append(depth_m)
                    print(f"    Depth PSNR={depth_m['psnr']:.2f}dB  L1={depth_m['l1']:.4f}")

                    rows = []
                    for vi in range(actual_v):
                        rows.append(np.concatenate([
                            depth_to_numpy_img(gt_depth[vi]),
                            depth_to_numpy_img(vae_depth[vi]),
                        ], axis=1))
                    Image.fromarray(np.concatenate(rows, axis=0)).save(
                        os.path.join(ds_outdir, f"{s_idx:03d}_depth.png"))

                # ── Camera Pose (from ray head) ──
                gt_ray = gt_dpt_out.get('ray')
                vae_ray = vae_dpt_out.get('ray')
                if gt_ray is not None and vae_ray is not None and actual_v >= 2:
                    try:
                        from utils.camera_from_ray import recover_poses, compute_camera_metrics

                        gt_ray_np = _ray_to_numpy(gt_ray)
                        vae_ray_np = _ray_to_numpy(vae_ray)
                        gt_rc_np = _rayconf_to_numpy(gt_dpt_out.get('ray_conf'))
                        vae_rc_np = _rayconf_to_numpy(vae_dpt_out.get('ray_conf'))

                        gt_c2w_list, gt_K_rec = recover_poses(
                            gt_ray_np, gt_rc_np, input_size=(H, W),
                            return_per_view_intrinsics=True,
                        )

                        cam_m = compute_camera_metrics(
                            ray=vae_ray_np,
                            ray_conf=vae_rc_np,
                            gt_K=gt_K_rec,
                            gt_c2w=np.stack(gt_c2w_list),
                            cond_num=0,
                            input_size=(H, W),
                        )
                        vae_c2w_list = cam_m.get('pred_c2w', [])
                        gt_ref = cam_m.get('gt_c2w_ref', gt_c2w_list)

                        # evo metrics (ATE/RPE with Sim3 alignment)
                        evo_m = None
                        if len(vae_c2w_list) >= 2:
                            try:
                                evo_m, _, _ = _compute_evo_metrics(
                                    gt_ref, vae_c2w_list)
                            except Exception:
                                pass
                        if evo_m is not None:
                            cam_m['ate_rmse'] = evo_m['ate_rmse']
                            cam_m['rpe_t_rmse'] = evo_m['rpe_t_rmse']
                            cam_m['rpe_r_rmse'] = evo_m['rpe_r_rmse']

                        all_cam.append(cam_m)

                        # Per-scene log
                        rot_e = cam_m.get('mean_rot_err')
                        parts = [f"AUC@30={cam_m['auc30']:.4f}"]
                        if rot_e is not None:
                            parts.append(f"RotErr={rot_e:.2f}°")
                            parts.append(
                                f"TransErr={cam_m['mean_trans_err']:.2f}°")
                        if evo_m is not None:
                            parts.append(f"ATE={evo_m['ate_rmse']:.4f}")
                            parts.append(
                                f"RPE_r={evo_m['rpe_r_rmse']:.2f}°")
                        print(f"    Pose   {'  '.join(parts)}")

                        # evo visualization
                        if len(vae_c2w_list) >= 2:
                            _save_pose_vis(
                                gt_ref, vae_c2w_list,
                                os.path.join(ds_outdir,
                                             f"{s_idx:03d}_pose.png"))
                    except Exception as e:
                        print(f"    Pose   [error: {e}]")

        per_dataset_results[ds_name] = {
            "feat": all_feat, "rgb": all_rgb, "depth": all_depth, "cam": all_cam
        }

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for ds_name, res in per_dataset_results.items():
        print(f"\n  [{ds_name}]")
        if res["feat"]:
            avg = {k: np.mean([m[k] for m in res["feat"]]) for k in res["feat"][0]}
            print("    Feature L1: " + ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        if res["rgb"]:
            print(f"    RGB   PSNR: {np.mean([m['psnr'] for m in res['rgb']]):.2f} dB  "
                  f"L1: {np.mean([m['l1'] for m in res['rgb']]):.4f}")
        if res["depth"]:
            print(f"    Depth PSNR: {np.mean([m['psnr'] for m in res['depth']]):.2f} dB  "
                  f"L1: {np.mean([m['l1'] for m in res['depth']]):.4f}")
        if res["cam"]:
            cam_list = res["cam"]
            avg_auc30 = np.mean([m['auc30'] for m in cam_list])
            avg_maa = np.mean([m['maa30'] for m in cam_list])
            rot_vals = [m['mean_rot_err'] for m in cam_list if m.get('mean_rot_err') is not None]
            trans_vals = [m['mean_trans_err'] for m in cam_list if m.get('mean_trans_err') is not None]
            focal_vals = [m['focal_rel_err'] for m in cam_list]
            ate_vals = [m['ate_rmse'] for m in cam_list if 'ate_rmse' in m]
            rpe_t_vals = [m['rpe_t_rmse'] for m in cam_list if 'rpe_t_rmse' in m]
            rpe_r_vals = [m['rpe_r_rmse'] for m in cam_list if 'rpe_r_rmse' in m]

            print(f"    Pose (pairwise):")
            parts = [f"AUC@30: {avg_auc30:.4f}", f"mAA@30: {avg_maa:.2f}"]
            if rot_vals:
                parts.append(f"RotErr: {np.mean(rot_vals):.2f}°")
            if trans_vals:
                parts.append(f"TransErr: {np.mean(trans_vals):.2f}°")
            parts.append(f"FocalRelErr: {np.mean(focal_vals):.4f}")
            print(f"      " + "  ".join(parts))
            if ate_vals:
                print(f"    Pose (evo Sim3):")
                print(f"      ATE: {np.mean(ate_vals):.4f}  "
                      f"RPE_t: {np.mean(rpe_t_vals):.4f}  "
                      f"RPE_r: {np.mean(rpe_r_vals):.2f}°")

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
