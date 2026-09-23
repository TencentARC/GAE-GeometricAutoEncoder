"""
Scene-collection helpers shared by the evaluation entry points.

Provides ``DATASET_ROOTS``, ``DATASET_DEFAULT_INTERVAL``, the ``collect_scenes*``
family and ``load_scenes_from_manifest``, which turn a dataset root into a list
of scenes with frames and camera sidecars.

In the release this module is used as a data helper only; the evaluation entry
points are ``scripts/eval/eval_generation.py`` and friends. It retains the original
standalone evaluation CLI it was extracted from.

Run from the repo root so that ``src/`` is importable:

    PYTHONPATH=src python scripts/eval/eval_data.py --help
"""


import argparse
import json
import os
import sys
import random

os.environ.setdefault("TMPDIR", "/tmp")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from einops import rearrange
from omegaconf import OmegaConf

from stage1.da3 import DA3Backbone
from stage1.gae_codec import GAECodec, DA3_LEVEL_DIM
import cut3r_data.utils.cropping as cropping


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/flow_gae64.yaml")
    p.add_argument("--data-root", type=str, default=None,
                       help="explicit data root (overrides --dataset)")
    p.add_argument("--dataset", type=str, default="all",
                       choices=["re10k", "dl3dv", "mvssynth", "scannetpp", "all"],
                       help="dataset to evaluate (default: all). "
                            "scannetpp reads <scene>/video.mp4 + meta.json "
                            "and supports both flat (long-source) and "
                            "nested <scene>/<clip>/ layouts.")
    p.add_argument("--dpt-decoder", type=str, default="pretrained_models/da3/dpt_decoder.pt")
    p.add_argument("--da3-weights", type=str, default="pretrained_models/da3/model.safetensors")
    p.add_argument("--num-scenes", type=int, default=4)
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--cond-num", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="results/eval_generation")
    p.add_argument("--resolution", type=int, nargs=2, default=[504, 504])
    p.add_argument("--mode", choices=["recon", "generate"], default="recon")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--use-ema", action="store_true", default=True,
                       help="use EMA weights (default)")
    p.add_argument("--no-ema", dest="use_ema", action="store_false",
                       help="use raw training weights instead of EMA")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-dist-shift", type=float, default=None,
                       help="override time_dist_shift (default: auto-compute from latent dim)")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                       help="compute precision (training uses bf16)")
    p.add_argument("--loss-t", type=str, default="0.5",
                       help="'shifted' = sample from shifted dist (match training), "
                            "or a float value for fixed t (default: 0.5)")
    p.add_argument("--loss-samples", type=int, default=1,
                       help="number of random t samples to average per scene "
                            "(only used when --loss-t shifted, default: 1)")
    p.add_argument("--loss-only", action="store_true",
                       help="only compute loss, skip ODE sampling and visualization")
    p.add_argument("--cond-num-range", type=str, default=None,
                       help="random cond_num range like training, e.g. '1-4'. "
                            "Overrides --cond-num with random sampling.")
    # Free-form generation from a single image
    p.add_argument("--ref-image", type=str, default=None,
                       help="path to a single reference image (bypasses --data-root)")
    p.add_argument("--ref-intrinsics", type=float, nargs=4, default=None,
                       metavar=("FX", "FY", "CX", "CY"),
                       help="intrinsics for --ref-image (default: estimate from resolution)")
    p.add_argument("--pose-source", choices=["auto", "scene", "random"],
                       default="auto",
                       help="pose source for --ref-image: "
                            "auto=scene if transforms.json exists else random, "
                            "scene=force load from transforms.json, "
                            "random=generate synthetic trajectory (default: auto)")
    p.add_argument("--traj-radius", type=float, default=0.3,
                       help="camera movement radius for random trajectory (default: 0.3)")
    p.add_argument("--traj-type", choices=["circle", "spiral", "forward"],
                       default="circle", help="trajectory shape (default: circle)")
    p.add_argument("--sample-interval", type=int, default=None,
                       help="frame interval between neighbouring views. "
                            "None = dataset-specific default (re10k=4, dl3dv=3, mvssynth=2); "
                            "pass e.g. 1 for consecutive frames, 8 for wider baseline.")
    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────

_DATA_ROOT = os.environ.get("GAE_DATA_ROOT", "/data/gae")
DATASET_ROOTS = {
    # Packed roots produced by scripts/data/prepare_data.py (see docs/DATA.md).
    "re10k_packed": os.path.join(_DATA_ROOT, "re10k_packed/test"),
    "dl3dv_packed": os.path.join(_DATA_ROOT, "dl3dv_packed"),
    "scannetpp": os.path.join(_DATA_ROOT, "scannetpp_preprocessed"),
    "mvssynth": os.path.join(_DATA_ROOT, "mvssynth_packed"),
    # Raw (pre-pack) roots; only used when you point --data-root at originals.
    "re10k": os.path.join(_DATA_ROOT, "raw/re10k/test"),
    "dl3dv": os.path.join(_DATA_ROOT, "raw/dl3dv"),
}

# Default frame-interval per dataset for sequential sampling at eval time.
# Picked as roughly mid-range of training (min=1, max=8/6/4) so the baseline
# distribution matches what the model saw during training.
# scannetpp=2 matches the lower end of precompute_scannetpp_latents.py's
# STRIDE_CHOICES="2 3 4" — densest temporal sampling within a 256-frame mp4.
DATASET_DEFAULT_INTERVAL = {
    "re10k": 6,
    "dl3dv": 4,
    "mvssynth": 2,
    "scannetpp": 2,
    # RE10K_Packed at train uses min/max=5/15 per scene; eval mid-range pick
    # matches training distribution. For test split dataset_test uses 3/50, but
    # 6 here keeps cross-ckpt comparison stable.
    "re10k_packed": 6,
    "dl3dv_packed": 4,
}


def _pick_sequential(pngs_sorted, num_views, interval, rng):
    """Pick num_views frames from pngs_sorted as a monotonically increasing
    sequence with fixed ``interval`` between neighbouring picks.

    - If the scene is too short for the requested span, fall back to evenly
      spaced frames across the whole scene (keeps time-ordered output).
    - ``rng`` is python's ``random.Random``; start index is uniformly chosen.
    """
    total = len(pngs_sorted)
    span = (num_views - 1) * max(1, int(interval))
    if total <= span:
        if num_views == 1:
            return [pngs_sorted[0]]
        step = (total - 1) / (num_views - 1)
        return [pngs_sorted[int(round(i * step))] for i in range(num_views)]
    max_start = total - span - 1
    start = rng.randint(0, max_start)
    return [pngs_sorted[start + i * interval] for i in range(num_views)]


def collect_scenes_re10k(data_root, num_scenes, num_views, seed, interval=None):
    """RE10K: shard_dir/scene_dir/*.png + transforms.json

    Sampling: pick a random start within each scene and take ``num_views``
    consecutive frames with fixed ``interval`` — matches training (which
    samples time-ordered, bounded-interval sequences; see ``block_shuffle=1``
    and ``max_interval`` in the flow config).
    """
    interval = interval if interval is not None else DATASET_DEFAULT_INTERVAL["re10k"]
    print('interval: ', interval)
    rng = random.Random(seed)
    reservoir, seen = [], 0
    shards = sorted(os.listdir(data_root))
    rng.shuffle(shards)
    for shard in shards:
        shard_dir = os.path.join(data_root, shard)
        if not os.path.isdir(shard_dir):
            continue
        try:
            scene_names = os.listdir(shard_dir)
        except OSError:
            continue
        rng.shuffle(scene_names)
        for scene in scene_names:
            scene_dir = os.path.join(shard_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            pngs = sorted([f for f in os.listdir(scene_dir) if f.endswith(".png")])
            if len(pngs) < num_views:
                continue
            seen += 1
            chosen = _pick_sequential(pngs, num_views, interval, rng)
            entry = (scene, scene_dir, chosen, "re10k")
            if len(reservoir) < num_scenes:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen - 1)
                if j < num_scenes:
                    reservoir[j] = entry
            if seen >= num_scenes * 10:
                break
        if seen >= num_scenes * 10:
            break
    return reservoir


def collect_scenes_dl3dv(data_root, num_scenes, num_views, seed, interval=None):
    """DL3DV: {K}K/hash_dir/{images_4/*.png, transforms.json}

    See ``collect_scenes_re10k`` for sampling rationale.
    """
    interval = interval if interval is not None else DATASET_DEFAULT_INTERVAL["dl3dv"]
    print('interval: ', interval)
    rng = random.Random(seed)
    reservoir, seen = [], 0
    buckets = sorted(os.listdir(data_root))
    rng.shuffle(buckets)
    for bucket in buckets:
        bucket_dir = os.path.join(data_root, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        try:
            scene_names = os.listdir(bucket_dir)
        except OSError:
            continue
        rng.shuffle(scene_names)
        for scene in scene_names:
            scene_dir = os.path.join(bucket_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            img_dir = os.path.join(scene_dir, "images_4")
            ns_dir = os.path.join(scene_dir, "nerfstudio")
            if os.path.isdir(os.path.join(ns_dir, "images_4")):
                scene_dir = ns_dir
                img_dir = os.path.join(ns_dir, "images_4")
            if not os.path.isdir(img_dir):
                continue
            tf_path = os.path.join(scene_dir, "transforms.json")
            if not os.path.isfile(tf_path):
                continue
            pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
            if len(pngs) < num_views:
                continue
            seen += 1
            chosen = _pick_sequential(pngs, num_views, interval, rng)
            entry = (scene[:16], scene_dir, chosen, "dl3dv")
            if len(reservoir) < num_scenes:
                reservoir.append(entry)
            else:
                j = rng.randint(0, seen - 1)
                if j < num_scenes:
                    reservoir[j] = entry
            if seen >= num_scenes * 10:
                break
        if seen >= num_scenes * 10:
            break
    return reservoir


def collect_scenes_mvssynth(data_root, num_scenes, num_views, seed, interval=None):
    """MVS-Synth: scene_id/{images/*.png, poses/*.json}"""
    interval = interval if interval is not None else DATASET_DEFAULT_INTERVAL["mvssynth"]
    print('interval: ', interval)
    rng = random.Random(seed)
    scenes_all = sorted([d for d in os.listdir(data_root)
                         if os.path.isdir(os.path.join(data_root, d, "images"))])
    rng.shuffle(scenes_all)
    result = []
    for scene in scenes_all:
        if len(result) >= num_scenes:
            break
        scene_dir = os.path.join(data_root, scene)
        img_dir = os.path.join(scene_dir, "images")
        pngs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        if len(pngs) < num_views:
            continue
        chosen = _pick_sequential(pngs, num_views, interval, rng)
        result.append((scene, scene_dir, chosen, "mvssynth"))
    return result


def collect_scenes_scannetpp(data_root, num_scenes, num_views, seed, interval=None):
    """ScanNet++ scene collector — auto-detects flat vs nested layout.

    Flat layout (long-source, e.g. scannetpp_preprocessed_long/):
        <root>/<scene>/{video.mp4, meta.json}
    Nested layout (short-source, e.g. scannetpp_preprocessed/):
        <root>/<scene>/<clip>/{video.mp4, meta.json}

    For the nested case each clip becomes its own eval scene with name
    "<scene>/<clip>" so the two layouts can share downstream code paths.

    Frame selection mirrors the other collectors: random start within the
    256-frame mp4, then num_views frames at fixed ``interval``. The returned
    img_names are zero-padded frame indices (decoded later by
    load_image_and_camera).
    """
    interval = interval if interval is not None else DATASET_DEFAULT_INTERVAL["scannetpp"]
    print('interval: ', interval)
    rng = random.Random(seed)

    # 1) Enumerate every (scene_name, scene_dir) with a complete mp4+meta pair.
    candidates: list[tuple[str, str]] = []
    for top in sorted(os.listdir(data_root)):
        top_dir = os.path.join(data_root, top)
        if not os.path.isdir(top_dir):
            continue
        if (os.path.isfile(os.path.join(top_dir, "video.mp4")) and
                os.path.isfile(os.path.join(top_dir, "meta.json"))):
            candidates.append((top, top_dir))
            continue
        # Nested
        try:
            sub_entries = sorted(os.listdir(top_dir))
        except OSError:
            continue
        for sub in sub_entries:
            sub_dir = os.path.join(top_dir, sub)
            if (os.path.isdir(sub_dir) and
                    os.path.isfile(os.path.join(sub_dir, "video.mp4")) and
                    os.path.isfile(os.path.join(sub_dir, "meta.json"))):
                candidates.append((f"{top}/{sub}", sub_dir))

    if not candidates:
        raise RuntimeError(
            f"No scannetpp scenes found under {data_root} — expected either "
            f"<scene>/video.mp4 or <scene>/<clip>/video.mp4 layout.")

    rng.shuffle(candidates)
    result: list[tuple[str, str, list[str], str]] = []
    for scene_name, scene_dir in candidates:
        if len(result) >= num_scenes:
            break
        try:
            with open(os.path.join(scene_dir, "meta.json")) as f:
                meta = json.load(f)
            num_frames = int(meta.get("num_frames", 0))
        except Exception:
            continue
        if num_frames < num_views:
            continue
        chosen_indices = _pick_sequential(
            list(range(num_frames)), num_views, interval, rng)
        # Store as zero-padded strings so the (scene_dir, basename) contract
        # used by load_image_and_camera works unchanged.
        img_names = [f"{int(i):06d}" for i in chosen_indices]
        result.append((scene_name, scene_dir, img_names, "scannetpp"))
    return result


def collect_scenes_re10k_packed(data_root, num_scenes, num_views, seed, interval=None):
    """RE10K_Packed scene collector — fast path (shuffle-then-validate).

    Layout: <root>/<scene>/{video.mp4, meta.json, caption.txt}. Schema is
    identical to scannetpp long-source (meta.json["frames"][i] holds c2w + K
    aligned with the i-th mp4 frame) so downstream decode reuses the scannetpp
    path via ds_type="re10k_packed" routing.

    Why not just call collect_scenes_scannetpp:
      That collector pre-stats every top-level entry to also detect the nested
      <scene>/<clip>/ layout. RE10K_Packed has ~7k+ scenes on S3-FUSE and never
      uses nested layout → the full pre-scan is several minutes of pure stat
      latency. Here we shuffle the listdir first and validate at most
      ~num_scenes*1.5 candidates lazily (skip on missing files / bad meta).
    """
    interval = (interval if interval is not None
                else DATASET_DEFAULT_INTERVAL["re10k_packed"])
    print('interval: ', interval)
    rng = random.Random(seed)

    all_entries = sorted(os.listdir(data_root))
    rng.shuffle(all_entries)

    result: list[tuple[str, str, list[str], str]] = []
    examined = 0
    # 1.5× over-sample lets us drop scenes with corrupt meta / short videos
    # without re-scanning the whole listdir.
    max_examine = max(num_scenes * 4, 64)
    for scene in all_entries:
        if len(result) >= num_scenes:
            break
        if examined >= max_examine and len(result) > 0:
            # already found at least one good scene; stop expanding the search
            # rather than walking the full 7k+ tree.
            break
        examined += 1

        scene_dir = os.path.join(data_root, scene)
        meta_path = _scene_meta_path(scene_dir)
        mp4_path = os.path.join(scene_dir, "video.mp4")
        if not (os.path.isfile(meta_path) and os.path.isfile(mp4_path)):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            frames = meta.get("frames", [])
            num_frames = int(meta.get("num_frames") or len(frames))
            if frames:
                num_frames = min(num_frames, len(frames))
        except Exception:
            continue
        if num_frames < num_views:
            continue
        chosen_indices = _pick_sequential(
            list(range(num_frames)), num_views, interval, rng)
        img_names = [f"{int(i):06d}" for i in chosen_indices]
        result.append((scene, scene_dir, img_names, "re10k_packed"))

    if not result:
        raise RuntimeError(
            f"No re10k_packed scenes found under {data_root} after examining "
            f"{examined}/{len(all_entries)} entries — expected "
            f"<scene>/video.mp4 + meta.json layout.")
    return result


def collect_scenes(dataset, data_root, num_scenes, num_views, seed, interval=None):
    """Dispatch to dataset-specific scene collector.

    ``interval`` (int or None): frame spacing between neighbouring views.
    None → use ``DATASET_DEFAULT_INTERVAL``. Overridden by CLI ``--sample-interval``.
    """
    if interval is None:
        interval = DATASET_DEFAULT_INTERVAL.get(dataset)
    if dataset == "re10k":
        return collect_scenes_re10k(data_root, num_scenes, num_views, seed, interval=interval)
    elif dataset == "dl3dv":
        return collect_scenes_dl3dv(data_root, num_scenes, num_views, seed, interval=interval)
    elif dataset == "mvssynth":
        return collect_scenes_mvssynth(data_root, num_scenes, num_views, seed, interval=interval)
    elif dataset == "scannetpp":
        return collect_scenes_scannetpp(data_root, num_scenes, num_views, seed, interval=interval)
    elif dataset in ("re10k_packed", "dl3dv_packed"):
        return collect_scenes_re10k_packed(data_root, num_scenes, num_views, seed, interval=interval)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _list_scene_pngs(scene_dir, ds_type):
    if ds_type == "mvssynth":
        img_dir = os.path.join(scene_dir, "images")
    elif ds_type == "dl3dv":
        img_dir = os.path.join(scene_dir, "images_4")
        if not os.path.isdir(img_dir):
            img_dir = scene_dir
    else:
        img_dir = scene_dir
    return sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])


def _extend_scene_frames(scene_dir, img_names, num_views, interval, ds_type):
    """Extend a fixed frame list to ``num_views`` using the same interval."""
    if len(img_names) >= num_views:
        return img_names[:num_views]
    pngs = _list_scene_pngs(scene_dir, ds_type)
    if not pngs:
        return img_names
    interval = max(1, int(interval))
    first = img_names[0]
    if first in pngs:
        start = pngs.index(first)
        for iv in range(interval, 0, -1):
            span = (num_views - 1) * iv
            if start + span < len(pngs):
                return [pngs[start + i * iv] for i in range(num_views)]
    extended = list(img_names)
    last = extended[-1]
    if last not in pngs:
        return extended
    idx = pngs.index(last)
    while len(extended) < num_views and idx + interval < len(pngs):
        idx += interval
        extended.append(pngs[idx])
    return extended


def save_scenes_manifest(
    scenes,
    path,
    *,
    dataset=None,
    seed=None,
    num_views=None,
    interval=None,
):
    """Save scene list to JSON for reproducible eval across scripts/versions."""
    payload = {
        "dataset": dataset,
        "seed": seed,
        "num_views": num_views,
        "interval": interval,
        "scenes": [
            {
                "scene_name": name,
                "scene_dir": scene_dir,
                "img_names": img_names,
                "ds_type": ds_type,
            }
            for name, scene_dir, img_names, ds_type in scenes
        ],
    }
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_scenes_from_manifest(path, num_views=None, interval=None):
    """Load fixed scene list from JSON manifest.

    When ``num_views`` exceeds the stored frame count, extra frames are appended
    with the same interval (e.g. extend v3's 8-frame clips to 9 for Wan temporal).
    """
    with open(path) as f:
        payload = json.load(f)

    manifest_interval = payload.get("interval")
    interval = interval if interval is not None else manifest_interval
    target_views = num_views if num_views is not None else payload.get("num_views")

    scenes = []
    for entry in payload["scenes"]:
        scene_name = entry["scene_name"]
        scene_dir = entry["scene_dir"]
        img_names = list(entry["img_names"])
        ds_type = entry.get("ds_type", "re10k")

        if target_views is not None and len(img_names) != target_views:
            if len(img_names) < target_views:
                iv = interval
                if iv is None:
                    iv = DATASET_DEFAULT_INTERVAL.get(ds_type, 6)
                img_names = _extend_scene_frames(
                    scene_dir, img_names, target_views, iv, ds_type)
                if len(img_names) < target_views:
                    print(
                        f"  [WARN] Scene {scene_name}: only {len(img_names)} frames, "
                        f"need {target_views}")
            else:
                img_names = img_names[:target_views]

        if not os.path.isdir(scene_dir):
            print(f"  [WARN] Scene dir missing, skipping: {scene_dir}")
            continue
        scenes.append((scene_name, scene_dir, img_names, ds_type))
    return scenes


_IMG_NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_tf_cache = {}  # scene_dir -> (global_intrinsics_3x3, frame_map)


def _load_transforms_json(scene_dir):
    """Load and cache transforms.json for a scene (RE10K / DL3DV)."""
    if scene_dir not in _tf_cache:
        tf_path = os.path.join(scene_dir, "transforms.json")
        with open(tf_path, "r") as f:
            tf = json.load(f)

        orig_w = float(tf.get("w", 0))
        fl_x, fl_y = float(tf["fl_x"]), float(tf["fl_y"])
        cx, cy = float(tf["cx"]), float(tf["cy"])

        # DL3DV may need intrinsic scaling (images_4 is 1/4 of original)
        if orig_w > 0:
            img_dir = os.path.join(scene_dir, "images_4")
            if os.path.isdir(img_dir):
                sample = next((f for f in os.listdir(img_dir)
                               if f.endswith(".png")), None)
                if sample is not None:
                    import struct
                    with open(os.path.join(img_dir, sample), "rb") as fh:
                        fh.read(16)
                        actual_w = struct.unpack(">I", fh.read(4))[0]
                    scale = actual_w / orig_w
                    fl_x *= scale; fl_y *= scale
                    cx *= scale; cy *= scale

        intrinsics = np.array([
            [fl_x, 0.0,  cx],
            [0.0,  fl_y, cy],
            [0.0,  0.0,  1.0],
        ], dtype=np.float32)
        frame_map = {
            os.path.splitext(os.path.basename(fr["file_path"]))[0]: fr["transform_matrix"]
            for fr in tf["frames"]
        }
        _tf_cache[scene_dir] = (intrinsics, frame_map)
    return _tf_cache[scene_dir]


# ── ScanNet++ helpers (mp4 frame + meta.json) ────────────────────────────
# Meta is read once per scene_dir (≈256 frames of c2w/K); frames are read on
# demand via cv2 seek and kept in a per-scene LRU dict so consecutive calls
# from load_image_and_camera don't re-open the mp4. Cleared by
# clear_scannetpp_cache() between scenes to bound memory (~50 MB per scene).

_scannetpp_meta_cache: dict[str, tuple] = {}
_scannetpp_frames_cache: dict[str, dict] = {}


def _scene_video_path(scene_dir):
    """Return scene video path; allow metadata-only scenes to point elsewhere."""
    local_video = os.path.join(scene_dir, "video.mp4")
    if os.path.isfile(local_video):
        return local_video
    meta_path = os.path.join(scene_dir, "meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        source_video = meta.get("source_video")
    except Exception:
        source_video = None
    if source_video and os.path.isfile(source_video):
        return source_video
    return local_video


def _scene_meta_path(scene_dir):
    """Prefer DA3 metric sidecar when present, matching RE10K_Packed training."""
    metric_meta = os.path.join(scene_dir, "meta_da3_metric.json")
    if os.path.isfile(metric_meta):
        return metric_meta
    return os.path.join(scene_dir, "meta.json")


def _load_scannetpp_meta(scene_dir):
    """Return ((c2w_arr (N,4,4), K_arr (N,3,3)) fp32) for an mp4+meta scene_dir."""
    if scene_dir not in _scannetpp_meta_cache:
        meta_path = _scene_meta_path(scene_dir)
        with open(meta_path) as f:
            meta = json.load(f)
        frames = meta.get("frames")
        if not isinstance(frames, list) or not frames:
            raise KeyError(
                f"no frames in {meta_path} (scene_dir={scene_dir}); "
                f"long packed clips store poses under chunk_*/meta_da3_metric.json")
        bad = [i for i, fr in enumerate(frames)
               if not (isinstance(fr, dict) and "c2w" in fr and "K" in fr)]
        if bad:
            sample = frames[bad[0]]
            keys = sorted(sample.keys()) if isinstance(sample, dict) else type(sample)
            raise KeyError(
                f"{len(bad)}/{len(frames)} frames missing c2w/K in {meta_path}; "
                f"first bad frame keys={keys}. Parent metas without DA3 poses "
                f"must not be used — prefer chunk_*/meta_da3_metric.json.")
        c2w_arr = np.asarray([fr["c2w"] for fr in frames], dtype=np.float32)
        K_arr = np.asarray([fr["K"] for fr in frames], dtype=np.float32)
        # Defensive: replace non-finite poses/intrinsics with identity so a
        # corrupt frame doesn't NaN downstream losses. Matches ScanNetppRGB_Multi.
        finite = np.isfinite(c2w_arr).all(axis=(1, 2)) & \
                 np.isfinite(K_arr).all(axis=(1, 2))
        if not finite.all():
            for i in np.where(~finite)[0]:
                c2w_arr[i] = np.eye(4, dtype=np.float32)
                K_arr[i] = np.eye(3, dtype=np.float32)
        _scannetpp_meta_cache[scene_dir] = (c2w_arr, K_arr)
    return _scannetpp_meta_cache[scene_dir]


def prefetch_scannetpp_frames(scene_dir, frame_indices):
    """Decode the requested mp4 frames into the per-scene cache in a single pass.

    Re-opening the mp4 per frame is unusably slow over S3-FUSE (~seconds each
    after the initial keyframe seek). This function does one cv2 capture per
    scene, sorts targets, and uses forward streaming with occasional seek for
    far-ahead jumps — mirrors ScanNetppRGB_Multi._read_frames.
    """
    cache = _scannetpp_frames_cache.setdefault(scene_dir, {})
    missing = sorted(set(int(i) for i in frame_indices) - cache.keys())
    if not missing:
        return
    video_path = _scene_video_path(scene_dir)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise IOError(f"Empty/unreadable mp4: {video_path}")
    try:
        cur = 0
        for target in missing:
            if target >= total:
                raise IOError(
                    f"frame {target} out of bounds for {video_path} (total={total})")
            if target - cur > 16:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                cur = target
            while cur <= target:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    raise IOError(f"Read failed at frame {cur} of {video_path}")
                if cur == target:
                    cache[cur] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                cur += 1
    finally:
        cap.release()


def _load_scannetpp_frame(scene_dir, frame_idx):
    """Return cached RGB uint8 ndarray for mp4 frame frame_idx.

    Falls back to a single-frame prefetch (one cap.open + seek+read) if the
    caller didn't pre-batch. For best performance call
    ``prefetch_scannetpp_frames(scene_dir, all_frame_indices)`` once per scene.
    """
    cache = _scannetpp_frames_cache.setdefault(scene_dir, {})
    if frame_idx in cache:
        return cache[frame_idx]
    prefetch_scannetpp_frames(scene_dir, [frame_idx])
    return cache[frame_idx]


def clear_scannetpp_frame_cache(scene_dir=None):
    """Drop the cached RGB frames for one scene (or all if scene_dir=None).

    Call between scenes during eval to keep memory bounded; meta cache is tiny
    and kept around for the whole run.
    """
    if scene_dir is None:
        _scannetpp_frames_cache.clear()
    else:
        _scannetpp_frames_cache.pop(scene_dir, None)


def _load_mvssynth_camera(scene_dir, basename):
    """Load per-frame intrinsics + OpenCV c2w from MVS-Synth pose JSON.

    ``extrinsic`` is w2c with an OpenCV camera frame; ``inv(extrinsic)`` is
    the geometrically-correct c2w (its ``det(R) = -1`` only reflects that
    GTA-V's world frame is left-handed, which does not affect projection).
    """
    pose_path = os.path.join(scene_dir, "poses", basename + ".json")
    with open(pose_path, "r") as f:
        d = json.load(f)
    intrinsics = np.array([
        [d["f_x"], 0.0,    d["c_x"]],
        [0.0,      d["f_y"], d["c_y"]],
        [0.0,      0.0,    1.0],
    ], dtype=np.float32)
    extrinsic = np.array(d["extrinsic"], dtype=np.float64)
    c2w = np.linalg.inv(extrinsic).astype(np.float32)
    return intrinsics, c2w


def _crop_resize_if_necessary(image_np, intrinsics, resolution):
    """Replicate training's _crop_resize_if_necessary (without aug_crop).

    Args:
        image_np: (H, W, 3) uint8 RGB numpy array
        intrinsics: (3, 3) float32
        resolution: (W_out, H_out) target size — NOTE: PIL convention (W, H)

    Returns:
        pil_image: cropped & resized PIL image at exact resolution
        intrinsics: adjusted (3, 3) float32
    """
    image = Image.fromarray(image_np)
    depthmap = np.zeros(image_np.shape[:2], dtype=np.float32)

    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    min_margin_x = max(min_margin_x, W // 5 + 1)
    min_margin_y = max(min_margin_y, H // 5 + 1)

    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    image, depthmap, intrinsics = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, (l, t, r, b))

    image, depthmap, intrinsics = cropping.rescale_image_depthmap(
        image, depthmap, intrinsics, resolution)

    intrinsics2 = cropping.camera_matrix_of_crop(
        intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(
        intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
        image, depthmap, intrinsics, crop_bbox)

    return image, intrinsics2


def load_image_and_camera(scene_dir, basename, resolution, ds_type="re10k"):
    """Load image + camera with the SAME preprocessing as training.

    Args:
        scene_dir: path to scene directory
        basename: image filename without extension (for scannetpp: zero-padded frame_idx)
        resolution: (H, W) target — converted to PIL's (W, H) internally
        ds_type: "re10k", "dl3dv", "mvssynth", or "scannetpp"

    Returns:
        img_tensor: (3, H, W) ImageNet-normalized tensor
        intrinsics: (3, 3) float32 numpy array (adjusted for crop+resize)
        pose: (4, 4) float32 numpy array (c2w)
    """
    H_out, W_out = resolution

    if ds_type in ("scannetpp", "re10k_packed"):
        # Frame comes from mp4 (no per-frame png); c2w/K from meta.json.
        # basename is the zero-padded frame index emitted by collect_scenes_*.
        # RE10K_Packed shares the exact meta.json schema + OpenCV c2w convention,
        # so the scannetpp decode path is reused as-is.
        frame_idx = int(basename)
        img_rgb = _load_scannetpp_frame(scene_dir, frame_idx)
        c2w_arr, K_arr = _load_scannetpp_meta(scene_dir)
        if frame_idx >= len(K_arr) or frame_idx >= len(c2w_arr):
            raise IndexError(
                f"frame {frame_idx} is out of bounds for camera meta "
                f"{_scene_meta_path(scene_dir)} "
                f"(poses={len(c2w_arr)}, intrinsics={len(K_arr)})"
            )
        intrinsics = K_arr[frame_idx].copy()
        pose = c2w_arr[frame_idx].copy()
    else:
        if ds_type == "mvssynth":
            img_path = os.path.join(scene_dir, "images", basename + ".png")
        elif ds_type == "dl3dv":
            img_path = os.path.join(scene_dir, "images_4", basename + ".png")
        else:
            img_path = os.path.join(scene_dir, basename + ".png")

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Could not load {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if ds_type == "mvssynth":
            intrinsics, pose = _load_mvssynth_camera(scene_dir, basename)
        else:
            global_K, frame_map = _load_transforms_json(scene_dir)
            intrinsics = global_K.copy()
            pose = np.array(frame_map[basename], dtype=np.float32)
            # DL3DV's transforms.json is OpenGL c2w (NeRF/instant-ngp style):
            # camera looks along -Z, Y-up. Convert to OpenCV (Z-forward, Y-down)
            # by flipping camera Y/Z axes so it matches DPT/ProPE pinhole conv.
            # RE10K's transforms.json is empirically already OpenCV.
            if ds_type == "dl3dv":
                pose = pose @ np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    pil_img, intrinsics = _crop_resize_if_necessary(
        img_rgb, intrinsics, (W_out, H_out))
    img_tensor = _IMG_NORM(pil_img)
    return img_tensor, intrinsics, pose


def build_camera_emb(intrinsics_list, poses_list, H, W, cond_num, device,
                     camera_mode="plucker"):
    from utils.camera.camera import get_camera_embedding
    """Build camera embedding with ref/tgt mask.
    Returns: (V, 7, H, W)
    """
    V = len(intrinsics_list)
    intri = torch.from_numpy(np.stack(intrinsics_list)).to(device)
    poses = torch.from_numpy(np.stack(poses_list)).to(device)
    extri = poses[:, :3, :4]

    B = 1
    cam_emb = get_camera_embedding(intri, extri, B, V, H, W, mode=camera_mode)
    cam_emb = rearrange(cam_emb, "b v c h w -> (b v) c h w")

    mask = torch.ones(V, 1, H, W, device=device, dtype=cam_emb.dtype)
    mask[:cond_num] = 0  # ref=0, tgt=1
    return torch.cat([mask, cam_emb], dim=1)


def build_prope_data(intrinsics_list, poses_list, device):
    """Build viewmats (w2c) and Ks for ProPE. Returns float32.

    Geometric reference is the **last view** (`-1`), matching
    the flow trainer's PRoPE branch and
    `utils/camera/camera.py:batch_sample_rays` default
    (`normalize_extrinsic_tgt=-1`). This is intentionally decoupled from the
    condition-view assignment (prefix `0..cond_num-1`).
    """
    V = len(intrinsics_list)
    Ks = torch.from_numpy(np.stack(intrinsics_list)).float().to(device)   # (V, 3, 3)
    c2w = torch.from_numpy(np.stack(poses_list)).float().to(device)       # (V, 4, 4)

    ref_inv = torch.linalg.inv(c2w[-1:])
    c2w = ref_inv @ c2w

    # Translation scale normalization
    t_vec = c2w[:, :3, 3]
    farthest = t_vec.abs().amax()
    scale = 1.0 / (farthest + 1e-8)
    c2w[:, :3, 3] = c2w[:, :3, 3] * scale

    w2c = torch.linalg.inv(c2w)

    return w2c.unsqueeze(0), Ks.unsqueeze(0)  # (1, V, 4, 4), (1, V, 3, 3)


def tensor_to_numpy_img(t):
    if t.ndim == 4:
        t = t[0]
    return (t.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def depth_to_numpy_img(d, dmin=None, dmax=None):
    """Colorize one depth frame with an optional shared normalization range.

    ``dmin``/``dmax`` are used by video writers so every frame in a clip is
    rendered with the same global range.  When omitted, the historical
    single-frame min/max behavior is retained for standalone previews.
    """
    import matplotlib.cm as cm
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[0] == 1:
        d = d[0]
    d = d.float().cpu()
    finite = torch.isfinite(d)
    if dmin is None or dmax is None:
        if finite.any():
            dmin = float(d[finite].min())
            dmax = float(d[finite].max())
        else:
            dmin = dmax = 0.0
    if float(dmax) - float(dmin) > 1e-6:
        d = (d - float(dmin)) / (float(dmax) - float(dmin))
    else:
        d = torch.zeros_like(d)
    d = torch.nan_to_num(d, nan=0.0, posinf=1.0, neginf=0.0)
    colored = cm.viridis(d.numpy())[:, :, :3]
    return (colored * 255).astype(np.uint8)


def depth_to_numpy_video(depths):
    """Colorize a depth clip using one global finite min/max across all frames."""
    d = torch.as_tensor(depths).float().cpu()
    while d.ndim > 3 and d.shape[0] == 1:
        d = d[0]
    if d.ndim == 4 and d.shape[1] == 1:
        d = d[:, 0]
    if d.ndim != 3:
        raise ValueError(f"expected depth clip [V,H,W] or [V,1,H,W], got {tuple(d.shape)}")
    finite = torch.isfinite(d)
    if finite.any():
        dmin = float(d[finite].min())
        dmax = float(d[finite].max())
    else:
        dmin = dmax = 0.0
    return [depth_to_numpy_img(frame, dmin, dmax) for frame in d]


def compute_metrics(gt, pred):
    mse = F.mse_loss(pred, gt).item()
    psnr = -10 * np.log10(mse + 1e-10)
    l1 = F.l1_loss(pred, gt).item()
    return {"psnr": psnr, "l1": l1}


# ── Trajectory & single-image helpers ─────────────────────────────────────

def generate_trajectory(num_views, radius=0.3, traj_type="circle", seed=42):
    """Generate a smooth camera trajectory as a list of (4,4) c2w matrices.

    Convention note
    ---------------
    All training data is fed to the model in **OpenCV camera frame**
    (X-right, Y-down, Z-forward) after the dataset loaders normalize
    things (RE10K = OpenCV native, DL3DV = converted from OpenGL,
    MVS-Synth = OpenCV camera in left-handed world).

    However, the *empirical motion distribution* in the training data
    (especially RE10K) has the camera centre drifting toward **-Z** in
    the reference frame — RE10K clips are often "walk-back" sequences
    where view-0 is closest to the scene and later frames retreat. The
    trajectories below intentionally move along **-Z** to match this
    distribution, since out-of-distribution +Z motion produces visibly
    worse generations.

    Args:
        num_views: total number of poses returned (1 ref + tgt).
        radius: scale of motion (relative to the DA3 unit-depth scale
            the model was trained at, NOT meters).
        traj_type: ``"circle" | "spiral" | "forward"``.
        seed: jitter RNG seed.

    Returns:
        list[(4,4) np.float32 c2w], length ``num_views``. The first
        entry is identity (reference frame).
    """
    rng = np.random.default_rng(seed)
    poses = [np.eye(4, dtype=np.float32)]
    num_tgt = num_views - 1
    if num_tgt <= 0:
        return poses

    def _rot_y(a):
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

    def _rot_x(a):
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

    if traj_type == "circle":
        for i in range(num_tgt):
            t = 2.0 * np.pi * (i + 1) / (num_tgt + 1)
            x = radius * np.sin(t)
            z = radius * (np.cos(t) - 1.0)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = _rot_y(t)
            c2w[:3, 3] = [x, 0.0, z]
            poses.append(c2w)

    elif traj_type == "spiral":
        for i in range(num_tgt):
            frac = (i + 1) / num_tgt
            angle = 2.0 * np.pi * frac
            x = radius * frac * np.sin(angle) * 0.5
            y = radius * 0.1 * np.sin(angle * 0.5)
            z = -radius * frac
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = (_rot_y(angle * 0.15)
                           @ _rot_x(np.sin(angle * 0.5) * 0.05))
            c2w[:3, 3] = [x, y, z]
            poses.append(c2w)

    elif traj_type == "forward":
        for i in range(num_tgt):
            frac = (i + 1) / num_tgt
            z = -radius * frac * 2.0
            x = float(rng.normal(0, radius * 0.03))
            y = float(rng.normal(0, radius * 0.02))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = _rot_y(float(rng.normal(0, 0.03)))
            c2w[:3, 3] = [x, y, z]
            poses.append(c2w)

    return poses


def load_ref_image(image_path, resolution, ref_intrinsics=None):
    """Load a single reference image, applying the same crop+resize as training.

    Intrinsics priority: --ref-intrinsics > transforms.json > estimate.

    Returns:
        img_tensor: (3, H, W) ImageNet-normalized
        K: (3, 3) float32 intrinsics (adjusted for crop+resize)
    """
    H_out, W_out = resolution
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Cannot load {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_src, w_src = img_rgb.shape[:2]

    if ref_intrinsics is not None:
        fx, fy, cx, cy = ref_intrinsics
        src = "manual"
    else:
        scene_dir = os.path.dirname(image_path)
        tf_path = os.path.join(scene_dir, "transforms.json")
        if os.path.isfile(tf_path):
            with open(tf_path, "r") as f:
                tf = json.load(f)
            fx, fy = float(tf["fl_x"]), float(tf["fl_y"])
            cx, cy = float(tf["cx"]), float(tf["cy"])
            src = "transforms.json"
        else:
            fx = fy = float(max(w_src, h_src))
            cx, cy = w_src / 2.0, h_src / 2.0
            src = "estimated"
    print(f"    Intrinsics ({src}): fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}  src_size={w_src}x{h_src}")

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    pil_img, K = _crop_resize_if_necessary(img_rgb, K, (W_out, H_out))
    img_tensor = _IMG_NORM(pil_img)
    return img_tensor, K


def load_scene_poses(scene_dir, ref_basename, num_views, seed=42):
    """Load real poses from transforms.json, selecting evenly-spaced target frames.

    Returns:
        ref_pose:  (4,4) c2w for the reference frame
        tgt_poses: list of (4,4) c2w for target frames
        tgt_bns:   basenames of selected target frames
    """
    tf_path = os.path.join(scene_dir, "transforms.json")
    with open(tf_path, "r") as f:
        tf = json.load(f)

    # DL3DV scenes have an `images_4` directory; detect to apply OpenGL→OpenCV
    is_dl3dv = os.path.isdir(os.path.join(scene_dir, "images_4"))
    _GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)

    frames = []
    for fr in tf["frames"]:
        bn = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        pose = np.array(fr["transform_matrix"], dtype=np.float32)
        if is_dl3dv:
            pose = pose @ _GL2CV
        frames.append((bn, pose))
    frames.sort(key=lambda x: x[0])

    ref_idx = next((i for i, (bn, _) in enumerate(frames) if bn == ref_basename), 0)
    ref_pose = frames[ref_idx][1]

    other = [(bn, pose) for i, (bn, pose) in enumerate(frames) if i != ref_idx]
    num_target = num_views - 1

    if len(other) >= num_target:
        step = len(other) / num_target
        selected = [other[int(i * step)] for i in range(num_target)]
    else:
        selected = other

    return ref_pose, [s[1] for s in selected], [s[0] for s in selected]


def visualize_trajectory(pose_list, cond_num, output_path):
    """Draw 3D camera frustums (blue=source, red=target) with ground grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch

    V = len(pose_list)
    pos = np.array([p[:3, 3] for p in pose_list])

    # world (x,y,z) → plot (x,z,y) so matplotlib vertical = world Y (up)
    def _p(v):
        return np.array([v[0], v[2], v[1]])

    pos_plt = np.column_stack([pos[:, 0], pos[:, 2], pos[:, 1]])
    extent = max(pos_plt.ptp(axis=0).max(), 1e-6)
    fd = extent * 0.15          # frustum depth
    fs = extent * 0.08          # frustum half-width
    fs_h = fs * 0.6             # frustum half-height (non-square)

    src_c, tgt_c = "#4285f4", "#ea4335"

    fig = plt.figure(figsize=(10, 7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    for i in range(V):
        c2w = pose_list[i].astype(np.float64)
        o = c2w[:3, 3]
        r = c2w[:3, 0]
        u = c2w[:3, 1]
        fwd = -c2w[:3, 2]
        color = src_c if i < cond_num else tgt_c

        corners = [
            o + fwd * fd + (-r * fs + u * fs_h),   # top-left
            o + fwd * fd + ( r * fs + u * fs_h),   # top-right
            o + fwd * fd + ( r * fs - u * fs_h),   # bottom-right
            o + fwd * fd + (-r * fs - u * fs_h),   # bottom-left
        ]
        cp = [_p(c) for c in corners]
        op = _p(o)

        for c in cp:
            ax.plot3D([op[0], c[0]], [op[1], c[1]], [op[2], c[2]],
                      color=color, lw=1.0, alpha=0.85)
        base = cp + [cp[0]]
        bx = [c[0] for c in base]
        by = [c[1] for c in base]
        bz = [c[2] for c in base]
        ax.plot3D(bx, by, bz, color=color, lw=1.0, alpha=0.85)

        poly = Poly3DCollection([cp], alpha=0.25,
                                facecolor=color, edgecolor="none")
        ax.add_collection3d(poly)

        # Up indicator on top edge
        mid_top = (_p(corners[0]) + _p(corners[1])) / 2
        up_tip = mid_top + _p(u) * fs * 0.35
        ax.plot3D([mid_top[0], up_tip[0]], [mid_top[1], up_tip[1]],
                  [mid_top[2], up_tip[2]], color=color, lw=2.0, alpha=0.9)

        ax.text(op[0], op[1], op[2], str(i), fontsize=7,
                ha="center", fontweight="bold", zorder=10)

    # Ground grid at lowest camera height
    mid = pos_plt.mean(axis=0)
    gr = extent * 0.8
    z_floor = pos_plt[:, 2].min() - extent * 0.25
    n = 9
    for t in np.linspace(-gr, gr, n):
        ax.plot3D([mid[0] - gr, mid[0] + gr], [mid[1] + t, mid[1] + t],
                  [z_floor, z_floor], color="gray", lw=0.4, alpha=0.35)
        ax.plot3D([mid[0] + t, mid[0] + t], [mid[1] - gr, mid[1] + gr],
                  [z_floor, z_floor], color="gray", lw=0.4, alpha=0.35)

    legend = [Patch(facecolor=src_c, label="Source views"),
              Patch(facecolor=tgt_c, label="Target views")]
    ax.legend(handles=legend, loc="upper right", fontsize=9,
              framealpha=0.9)

    max_r = extent * 0.7
    for setter, c in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        setter(mid[c] - max_r, mid[c] + max_r)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=25, azim=-55)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"    Trajectory plot → {output_path}")


# ── DPT helpers ───────────────────────────────────────────────────────────

def raw_no_cls_to_dpt_input(feats_no_cls, backbone_norm, V):
    embed_dim = backbone_norm.normalized_shape[0]  # 768 for DA3-Base, 1024 for DA3-Large
    hidden_size = embed_dim * 2  # 1536 / 2048
    result = []
    for lvl in sorted(feats_no_cls.keys()):
        patches = feats_no_cls[lvl]
        cls_raw = torch.zeros(V, hidden_size, device=patches.device, dtype=patches.dtype)
        local_part = patches[:, :, :embed_dim]
        curr_norm = backbone_norm(patches[:, :, embed_dim:])
        patches_ln = torch.cat([local_part, curr_norm], dim=-1)
        result.append((patches_ln.unsqueeze(0), cls_raw.unsqueeze(0)))
    return result


def raw_to_dpt_input(feats_with_cls, backbone_norm, V):
    embed_dim = backbone_norm.normalized_shape[0]  # 768 for DA3-Base, 1024 for DA3-Large
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


@torch.no_grad()
def decode_to_depth(dpt_feats, dpt_decoder, H, W):
    with torch.autocast(device_type=dpt_feats[0][0].device.type, enabled=False):
        output = dpt_decoder(dpt_feats, H, W, patch_start_idx=0)
    depth = output.get("depth", None)
    if depth is None:
        return None
    if depth.ndim == 5:
        depth = depth.reshape(-1, *depth.shape[2:])
    elif depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth.squeeze(0).unsqueeze(1)
    return depth


# ── Main ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    try:
        from stage2.transport import create_transport
        from stage2.transport.transport import Sampler
        from utils.model_utils import instantiate_from_config
    except ImportError as exc:
        raise SystemExit(
            "eval_data.py is a helper module in the public release; "
            "use scripts/eval/eval_generation.py instead"
        ) from exc
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    H, W = args.resolution
    V = args.num_views
    cond_num = args.cond_num
    camera_mode = "plucker"

    # Parse --loss-t
    if args.loss_t.lower() == "shifted":
        loss_t_mode = "shifted"
        loss_t_fixed = None
    else:
        loss_t_mode = "fixed"
        loss_t_fixed = float(args.loss_t)

    # Parse --cond-num-range (e.g. "1-4")
    cond_num_range = None
    if args.cond_num_range:
        parts = args.cond_num_range.split("-")
        cond_num_range = (int(parts[0]), int(parts[1]))
    torch.manual_seed(args.seed)

    has_dpt = os.path.isfile(args.dpt_decoder) and os.path.isfile(args.da3_weights)

    print("=" * 60)
    print("Latent Diffusion Evaluation")
    print(f"  DiT ckpt:     {args.dit_ckpt}")
    print(f"  Weights:      {'EMA' if args.use_ema else 'raw model'}")
    print(f"  Mode:         {args.mode}")
    print(f"  Config:       {args.config}")
    print(f"  DPT decoder:  {args.dpt_decoder} {'OK' if has_dpt else '(skip depth)'}")
    print(f"  Resolution:   {H}x{W}")
    cond_desc = f"{cond_num_range[0]}-{cond_num_range[1]} (random)" if cond_num_range else str(cond_num)
    print(f"  Scenes: {args.num_scenes}, Views/scene: {V}, Cond views: {cond_desc}")
    loss_desc = "shifted (match training)" if loss_t_mode == "shifted" else f"fixed t={loss_t_fixed}"
    print(f"  Loss mode:    {loss_desc}, samples={args.loss_samples}")
    print(f"  Loss only:    {args.loss_only}")
    print(f"  Precision:    {args.precision}")
    print(f"  Sample steps: {args.sample_steps}, CFG: {args.cfg_scale}")
    print(f"  Output:       {args.output_dir}")
    print("=" * 60)

    use_amp = args.precision == "bf16"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    cfg = OmegaConf.load(args.config)
    vae_cfg = OmegaConf.to_container(cfg.get("codec", {}), resolve=True)
    model_cfg = cfg.get("stage_2")
    model_params = OmegaConf.to_container(model_cfg.get("params", {}), resolve=True)
    transport_cfg = OmegaConf.to_container(
        cfg.get("transport", {}).get("params", {}), resolve=True)
    sampler_cfg = OmegaConf.to_container(
        cfg.get("sampler", {}), resolve=True)
    use_prope = bool(model_params.get("use_prope", False))

    # ── 1. DA3 encoder ──
    print("\n[1/4] Loading DA3 encoder ...")
    rae_kwargs = dict(
        encoder_pretrained_path="depth-anything/DA3-Base",
        encoder_input_size=H,
        encoder_type="DA3EncoderDirect",
        reshape_to_2d=False,
    )
    if has_dpt:
        rae_kwargs["dpt_decoder_path"] = args.dpt_decoder
        rae_kwargs["da3_weights_path"] = args.da3_weights
    rae = DA3Backbone(**rae_kwargs).to(device).eval()
    encoder_mean = rae.encoder_mean
    encoder_std = rae.encoder_std
    backbone_norm = rae.encoder.backbone.pretrained.norm

    # ── 2. VAE ──
    print("[2/4] Loading Feature VAE v2 ...")
    vae_ckpt_path = str(cfg.get("vae_checkpoint"))
    vae = GAECodec(**vae_cfg).to(device).eval()
    vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    vae_sd = vae_ckpt.get("ema_vae", vae_ckpt.get("codec", vae_ckpt.get("vae", vae_ckpt)))
    vae.load_state_dict(vae_sd, strict=False)
    has_rgb = vae.rgb_head is not None
    print(f"  latent_dim={vae.latent_dim}  RGB head: {'OK' if has_rgb else 'no'}")

    # ── 3. GAEFlow ──
    print("[3/4] Loading GAEFlow ...")
    cam_ch = 4 if camera_mode == "camray" else 7
    if "params" in model_cfg:
        model_cfg["params"]["cam_in_channels"] = cam_ch
    dit = instantiate_from_config(model_cfg).to(device).eval()

    dit_ckpt = torch.load(args.dit_ckpt, map_location="cpu")
    dit_key = "ema" if args.use_ema and "ema" in dit_ckpt else "model"
    dit.load_state_dict(dit_ckpt[dit_key])
    n_params = sum(p.numel() for p in dit.parameters()) / 1e6
    print(f"  Loaded '{dit_key}' — {n_params:.1f}M params  "
          f"enc={dit.encoder_hidden_size} dec={dit.decoder_hidden_size}  "
          f"depth={dit.num_encoder_blocks}+{dit.num_decoder_blocks}")

    # Transport + Sampler
    in_ch = int(model_params.get("in_channels", 384))
    latent_h = H // 14
    latent_w = W // 14
    if args.time_dist_shift is not None:
        time_dist_shift = args.time_dist_shift
    else:
        shift_dim = math.prod((in_ch, latent_h, latent_w))
        shift_base = 4096
        time_dist_shift = math.sqrt(shift_dim / shift_base)
    print(f"  time_dist_shift: {time_dist_shift:.4f}")
    transport = create_transport(**transport_cfg, time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))
    if sampler_mode == "ODE":
        sample_fn = transport_sampler.sample_ode(**sampler_params)
    else:
        sample_fn = transport_sampler.sample_sde(**sampler_params)

    # Latent stats
    latent_mean, latent_std = None, None
    stats_path = str(cfg.get("latent_stats", ""))
    if stats_path and os.path.isfile(stats_path):
        stats = torch.load(stats_path, map_location="cpu")
        latent_mean = stats["mean"].to(device).reshape(1, -1, 1, 1)
        latent_std = stats["std"].to(device).reshape(1, -1, 1, 1).clamp(min=1e-5)
        print(f"  Latent stats: mean={latent_mean.mean():.4f}  std={latent_std.mean():.4f}")

    # ── 4. Collect scenes and evaluate ──
    use_ref_image = args.ref_image is not None

    denorm = lambda t: (t * encoder_std.squeeze(0) + encoder_mean.squeeze(0)).clamp(0, 1)

    if use_ref_image:
        if args.mode != "generate":
            print("  [Note] --ref-image forces mode=generate")
            args.mode = "generate"
        print(f"[4/4] Reference image: {args.ref_image}")
        print(f"  Pose source: {args.pose_source}, views={V}")
        datasets_to_eval = [("ref_image", None)]
    elif args.data_root:
        datasets_to_eval = [("custom", args.data_root)]
    elif args.dataset == "all":
        datasets_to_eval = [
            ("re10k", DATASET_ROOTS["re10k"]),
            ("dl3dv", DATASET_ROOTS["dl3dv"]),
            ("mvssynth", DATASET_ROOTS["mvssynth"]),
        ]
    else:
        datasets_to_eval = [(args.dataset, DATASET_ROOTS[args.dataset])]

    global_results = {}

    for ds_name, ds_root in datasets_to_eval:
        if ds_name == "ref_image":
            scenes = [("ref_image", None, None, "re10k")]
            ds_out_dir = args.output_dir
        else:
            if ds_name == "custom":
                scenes = collect_scenes_re10k(
                    ds_root, args.num_scenes, V, args.seed,
                    interval=args.sample_interval)
            else:
                scenes = collect_scenes(
                    ds_name, ds_root, args.num_scenes, V, args.seed,
                    interval=args.sample_interval)
            ds_out_dir = os.path.join(args.output_dir, ds_name)
            os.makedirs(ds_out_dir, exist_ok=True)
            print(f"\n{'─' * 50}")
            print(f"Dataset: {ds_name} ({len(scenes)} scenes × {V} views)")
            print(f"Root: {ds_root}")
            print(f"{'─' * 50}")

        all_rgb, all_depth, all_feat = [], [], []
        all_loss, all_ref_loss, all_tgt_loss = [], [], []
        rng = random.Random(args.seed)

        for s_idx, (scene_name, scene_dir, img_names, ds_type) in enumerate(scenes):
            if ds_name == "ref_image":
                actual_v = V
                scene_name = os.path.splitext(os.path.basename(args.ref_image))[0]

                ref_tensor, ref_K = load_ref_image(
                    args.ref_image, (H, W), args.ref_intrinsics)
                intri_list = [ref_K] * actual_v

                scene_dir_ref = os.path.dirname(args.ref_image)
                tf_exists = os.path.isfile(
                    os.path.join(scene_dir_ref, "transforms.json"))

                use_scene_poses = (
                    (args.pose_source == "scene") or
                    (args.pose_source == "auto" and tf_exists)
                )

                if use_scene_poses:
                    if not tf_exists:
                        raise FileNotFoundError(
                            f"--pose-source scene requires transforms.json in "
                            f"{scene_dir_ref}")
                    ref_pose, tgt_poses, tgt_bns = load_scene_poses(
                        scene_dir_ref, scene_name, actual_v, args.seed)
                    pose_list = [ref_pose] + tgt_poses
                    print(f"  [{s_idx+1}/{len(scenes)}] {scene_name} "
                          f"({actual_v} views, cond={cond_num}, poses=scene)")

                    gt_imgs = [ref_tensor.to(device)]
                    for tbn in tgt_bns:
                        tgt_t, _, _ = load_image_and_camera(
                            scene_dir_ref, tbn, (H, W))
                        gt_imgs.append(tgt_t.to(device))
                    imgs_01_ref = denorm(torch.stack(gt_imgs))
                else:
                    traj_poses = generate_trajectory(
                        actual_v, args.traj_radius, args.traj_type, args.seed)
                    pose_list = traj_poses
                    print(f"  [{s_idx+1}/{len(scenes)}] {scene_name} "
                          f"({actual_v} views, cond={cond_num}, "
                          f"poses=random/{args.traj_type})")
                    imgs_01_ref = denorm(
                        ref_tensor.unsqueeze(0).to(device))

                ref_imgs = ref_tensor.unsqueeze(0).to(device)
            else:
                actual_v = len(img_names)
                if cond_num_range is not None:
                    cond_num = rng.randint(cond_num_range[0], cond_num_range[1])

                print(f"  [{s_idx+1}/{len(scenes)}] {scene_name} "
                      f"({actual_v} views, cond={cond_num})")

                basenames = [os.path.splitext(n)[0] for n in img_names]
                img_tensors, intri_list, pose_list = [], [], []
                for bn in basenames:
                    img_t, intr, pose = load_image_and_camera(
                        scene_dir, bn, (H, W), ds_type=ds_type)
                    img_tensors.append(img_t)
                    intri_list.append(intr)
                    pose_list.append(pose)
                imgs = torch.stack(img_tensors).to(device)
                imgs_5d = imgs.unsqueeze(0)
                imgs_01 = denorm(imgs)

            if not args.loss_only:
                visualize_trajectory(
                    pose_list, cond_num,
                    os.path.join(ds_out_dir, f"{s_idx:03d}_trajectory.png"))

            cam_emb = build_camera_emb(
                intri_list, pose_list, H, W, cond_num, device, camera_mode)

            prope_kwargs = {}
            if use_prope:
                w2c, Ks = build_prope_data(intri_list, pose_list, device)
                prope_kwargs = dict(
                    viewmats=w2c, Ks=Ks,
                    prope_image_size=(H, W),
                )

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if use_ref_image:
                    ref_5d = ref_imgs.unsqueeze(0)
                    ref_raw = rae.encode(ref_5d, mode="all")
                    ref_feats = {k: v[:, 1:, :] for k, v in ref_raw.items()}
                    ref_norm = vae.normalize_levels(ref_feats, image_size=(H, W))
                    z_ref = vae.encode(ref_norm)[0]
                    if latent_mean is not None:
                        z_ref_n = (z_ref - latent_mean) / latent_std
                    else:
                        z_ref_n = z_ref
                    z_all_n = None
                    gt_raw = None
                    x_norm = None
                else:
                    gt_raw = rae.encode(imgs_5d, mode="all")
                    feats_no_cls = {k: v[:, 1:, :] for k, v in gt_raw.items()}
                    x_norm = vae.normalize_levels(feats_no_cls, image_size=(H, W))
                    z_all = vae.encode(x_norm)[0]

                    ref_raw = rae.encode(imgs_5d[:, :cond_num], mode="all")
                    ref_feats = {k: v[:, 1:, :] for k, v in ref_raw.items()}
                    ref_norm = vae.normalize_levels(ref_feats, image_size=(H, W))
                    z_ref = vae.encode(ref_norm)[0]

                    if latent_mean is not None:
                        z_all_n = (z_all - latent_mean) / latent_std
                        z_ref_n = (z_ref - latent_mean) / latent_std
                    else:
                        z_all_n = z_all
                        z_ref_n = z_ref

            C, h_lat, w_lat = z_ref_n.shape[1:]
            z_cond = torch.zeros(actual_v, C, h_lat, w_lat, device=device, dtype=z_ref_n.dtype)
            z_cond[:cond_num] = z_ref_n
            z_shape = (actual_v, C, h_lat, w_lat)

            model_kwargs = dict(
                camera_embedding=cam_emb,
                total_view=actual_v,
                cond_num=cond_num,
                is_concat_mode=True,
                ref_cond=z_cond,
                x1_global=z_all_n,
                freeze_cond=False,
                **prope_kwargs,
            )
            if args.cfg_scale > 1.0:
                model_kwargs["cfg_scale"] = args.cfg_scale

            if args.mode == "recon":
                use_shifted_t = (loss_t_mode == "shifted")
                n_loss_samples = args.loss_samples if use_shifted_t else 1
                t_override_val = None if use_shifted_t else loss_t_fixed

                scene_losses, scene_ref, scene_tgt = [], [], []
                for li in range(n_loss_samples):
                    loss_kwargs = model_kwargs.copy()
                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                        loss_dict = transport.training_multiview_losses(
                            dit, z_cond, actual_v, cond_num,
                            model_kwargs=loss_kwargs,
                            t_override=t_override_val,
                        )
                    scene_losses.append(loss_dict["loss"].mean().item())
                    scene_ref.append(loss_dict.get("ref_loss", loss_dict["loss"]).mean().item())
                    scene_tgt.append(loss_dict.get("tgt_loss", loss_dict["loss"]).mean().item())

                loss_val = float(np.mean(scene_losses))
                ref_val = float(np.mean(scene_ref))
                tgt_val = float(np.mean(scene_tgt))
                all_loss.append(loss_val)
                all_ref_loss.append(ref_val)
                all_tgt_loss.append(tgt_val)

                t_desc = "shifted" if use_shifted_t else f"t={t_override_val}"
                samples_desc = f" (avg {n_loss_samples} samples)" if n_loss_samples > 1 else ""
                print(f"    Loss [{t_desc}]{samples_desc}: {loss_val:.4f}, "
                      f"Ref: {ref_val:.4f}, Tgt: {tgt_val:.4f}")

            if args.loss_only:
                continue

            noise = torch.randn(z_shape, device=device)
            sample_input = torch.cat([z_cond, noise], dim=1)
            sample_kwargs = model_kwargs.copy()
            samples = sample_fn(sample_input, dit, **sample_kwargs)[-1]
            z_sampled = samples[:, in_ch:]

            if latent_mean is not None:
                z_sampled = z_sampled * latent_std + latent_mean

            seq, h_l, w_l = vae._decode_trunk(z_sampled)
            c_trunk = seq.shape[-1]
            recon_raw = vae.dec_conv(seq.permute(0, 2, 1).reshape(-1, c_trunk, h_l, w_l))
            recon_feats = vae.denormalize_and_split(recon_raw)

            if args.mode == "recon":
                recon_norm = vae.normalize_levels(recon_feats, image_size=(H, W))
                recon_ch = recon_norm.split(DA3_LEVEL_DIM, dim=1)
                input_ch = x_norm.split(DA3_LEVEL_DIM, dim=1)
                feat_l1 = {f"l{i}": F.l1_loss(rc, ic).item()
                           for i, (rc, ic) in enumerate(zip(recon_ch, input_ch))}
                feat_l1["avg"] = float(np.mean(list(feat_l1.values())))
                all_feat.append(feat_l1)
                print("    Feature L1: " +
                      ", ".join(f"{k}={v:.4f}" for k, v in feat_l1.items()))

            if has_rgb:
                rgb_pred = vae.rgb_head(seq, h_l, w_l)
                rgb_pred_01 = denorm(rgb_pred)
                if args.mode == "recon":
                    rgb_m = compute_metrics(imgs_01, rgb_pred_01)
                    all_rgb.append(rgb_m)
                    print(f"    RGB  PSNR={rgb_m['psnr']:.2f}dB  L1={rgb_m['l1']:.4f}")

                rows = []
                for vi in range(actual_v):
                    row = [tensor_to_numpy_img(rgb_pred_01[vi])]
                    if args.mode == "recon":
                        row.insert(0, tensor_to_numpy_img(imgs_01[vi]))
                    elif use_ref_image and vi < len(imgs_01_ref):
                        row.insert(0, tensor_to_numpy_img(imgs_01_ref[vi]))
                    elif use_ref_image:
                        row.insert(0, np.zeros_like(row[0]))
                    rows.append(np.concatenate(row, axis=1))
                Image.fromarray(np.concatenate(rows, axis=0)).save(
                    os.path.join(ds_out_dir, f"{s_idx:03d}_rgb.png"))

            if has_dpt and rae.rae_cl_decoder is not None:
                gen_dpt_in = raw_no_cls_to_dpt_input(recon_feats, backbone_norm, actual_v)
                gen_depth = decode_to_depth(gen_dpt_in, rae.rae_cl_decoder, H, W)

                gt_depth = None
                if args.mode == "recon":
                    gt_dpt_in = raw_to_dpt_input(gt_raw, backbone_norm, actual_v)
                    gt_depth = decode_to_depth(gt_dpt_in, rae.rae_cl_decoder, H, W)

                if gen_depth is not None:
                    if gt_depth is not None:
                        depth_m = compute_metrics(gt_depth.float(), gen_depth.float())
                        all_depth.append(depth_m)
                        print(f"    Depth PSNR={depth_m['psnr']:.2f}dB  L1={depth_m['l1']:.4f}")

                    rows = []
                    for vi in range(actual_v):
                        row = [depth_to_numpy_img(gen_depth[vi])]
                        if gt_depth is not None:
                            row.insert(0, depth_to_numpy_img(gt_depth[vi]))
                        rows.append(np.concatenate(row, axis=1))
                    Image.fromarray(np.concatenate(rows, axis=0)).save(
                        os.path.join(ds_out_dir, f"{s_idx:03d}_depth.png"))

        # ── Per-dataset summary ──
        ds_result = {}
        print(f"\n  [{ds_name}] Summary:")
        if all_loss:
            ds_result["loss"] = float(np.mean(all_loss))
            ds_result["ref_loss"] = float(np.mean(all_ref_loss))
            ds_result["tgt_loss"] = float(np.mean(all_tgt_loss))
            print(f"    Loss: {ds_result['loss']:.4f}, "
                  f"Ref: {ds_result['ref_loss']:.4f}, "
                  f"Tgt: {ds_result['tgt_loss']:.4f}")
        if all_feat:
            avg = {k: np.mean([m[k] for m in all_feat]) for k in all_feat[0]}
            ds_result["feat_l1"] = avg
            print("    Feature L1: " +
                  ", ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        if all_rgb:
            ds_result["rgb_psnr"] = float(np.mean([m["psnr"] for m in all_rgb]))
            ds_result["rgb_l1"] = float(np.mean([m["l1"] for m in all_rgb]))
            print(f"    RGB   PSNR: {ds_result['rgb_psnr']:.2f} dB  "
                  f"L1: {ds_result['rgb_l1']:.4f}")
        if all_depth:
            ds_result["depth_psnr"] = float(np.mean([m["psnr"] for m in all_depth]))
            ds_result["depth_l1"] = float(np.mean([m["l1"] for m in all_depth]))
            print(f"    Depth PSNR: {ds_result['depth_psnr']:.2f} dB  "
                  f"L1: {ds_result['depth_l1']:.4f}")
        global_results[ds_name] = ds_result

    # ── Global summary ──
    print("\n" + "=" * 60)
    loss_desc = "shifted (match training)" if loss_t_mode == "shifted" else f"t={loss_t_fixed}"
    cond_desc = f"{cond_num_range[0]}-{cond_num_range[1]}" if cond_num_range else str(args.cond_num)
    weights_desc = "EMA" if args.use_ema else "raw model"
    print(f"Global Summary  (mode={args.mode}, loss_t={loss_desc}, "
          f"weights={weights_desc}, cond={cond_desc})")
    print("=" * 60)
    for ds_name, r in global_results.items():
        parts = [f"[{ds_name}]"]
        if "loss" in r:
            parts.append(f"Loss={r['loss']:.4f}")
        if "rgb_psnr" in r:
            parts.append(f"RGB={r['rgb_psnr']:.2f}dB")
        if "depth_psnr" in r:
            parts.append(f"Depth={r['depth_psnr']:.2f}dB")
        if "feat_l1" in r:
            parts.append(f"FeatL1={r['feat_l1']['avg']:.4f}")
        print("  " + "  ".join(parts))
    if not args.loss_only:
        print(f"\nResults saved to {args.output_dir}/")
    else:
        print(f"\n(loss-only mode, no visualizations generated)")


if __name__ == "__main__":
    main()
