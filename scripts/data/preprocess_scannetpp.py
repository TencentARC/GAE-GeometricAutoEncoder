#!/usr/bin/env python3
"""Preprocess ScanNet++ scenes into mp4 + meta.json + depth.npz.

Produces the layout consumed by ``cut3r_data.datasets.scannetpp_rgb`` (the
``ScanNetppRGB_Multi`` loader used by the codec configs).

Input layout, per scene under SOURCE/<scene_id>/::

    rgb/frame_*.jpg                    RGB frames
    depth/frame_*.png                  16-bit mm depth
    colmap/pose_intrinsic_imu.json     per-frame c2w (4x4) + intrinsic (3x3)

Output, per scene under OUTPUT/<scene_id>/::

    video.mp4    — H.264 crf=18 yuv420p, RES x RES (center-cropped)
    meta.json    — {scene_id, num_frames, fps, rgb_resolution, frames=[{name,
                    original_idx, c2w 4x4, K 3x3}]}
    depth.npz    — uint16 mm, (num_frames, depth_res, depth_res)

Scenes lacking colmap/, depth/ or rgb/ are skipped.

Usage (8-shard parallel)::

    for i in $(seq 0 7); do
      python scripts/data/preprocess_scannetpp.py \
        --source /datasets/scannetpp \
        --output "$GAE_DATA_ROOT/scannetpp_preprocessed" \
        --num-frames 256 --resolution 504 --depth-resolution 192 \
        --num-shards 8 --shard-index $i &
    done
    wait
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
import pathlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess ScanNet++ into mp4 + meta + depth.")
    p.add_argument("--source", type=Path, required=True, help="ScanNet++ root")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=256,
                   help="Frames to keep per scene.")
    p.add_argument("--resolution", type=int, default=504,
                   help="Square RGB resolution after center-crop+resize.")
    p.add_argument("--depth-resolution", type=int, default=192,
                   help="Square depth resolution after center-crop.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--pose-key", type=str, default="pose",
                   choices=["pose", "aligned_pose"])
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--sampling-mode", type=str, default="progressive",
                   choices=["uniform", "progressive"])
    p.add_argument("--max-interval", type=int, default=8,
                   help="[progressive] Max frames to skip before force-accept.")
    p.add_argument("--min-translation", type=float, default=0.005,
                   help="[progressive] Min normalized displacement to accept.")
    p.add_argument("--max-translation", type=float, default=0.20,
                   help="[progressive] Max normalized displacement (visual overlap).")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def list_valid_scenes(source: Path) -> list[str]:
    scenes = []
    for sd in sorted(source.iterdir()):
        if not sd.is_dir():
            continue
        if not (sd / "colmap" / "pose_intrinsic_imu.json").is_file():
            continue
        if not (sd / "depth").is_dir() or not (sd / "rgb").is_dir():
            continue
        scenes.append(sd.name)
    return scenes


def load_pose_metadata(scene_dir: Path, pose_key: str) -> dict:
    with open(scene_dir / "colmap" / "pose_intrinsic_imu.json") as f:
        raw = json.load(f)
    out = {}
    for frame_name, entry in raw.items():
        c2w = np.asarray(entry[pose_key], dtype=np.float32)
        K = np.asarray(entry["intrinsic"], dtype=np.float32)
        if c2w.shape != (4, 4) or K.shape != (3, 3):
            continue
        if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
            continue
        out[frame_name] = (c2w, K)
    return out


def pick_frame_indices(num_total: int, num_keep: int) -> list[int]:
    if num_total <= num_keep:
        return list(range(num_total))
    if num_keep == 1:
        return [num_total // 2]
    step = (num_total - 1) / (num_keep - 1)
    return [int(round(i * step)) for i in range(num_keep)]


def pick_frames_progressive(poses, valid_names, num_keep, max_interval=8,
                            min_translation=0.005, max_translation=0.20) -> list[int]:
    num_total = len(valid_names)
    if num_total <= num_keep:
        return list(range(num_total))
    translations = np.zeros((num_total, 3), dtype=np.float32)
    for i, name in enumerate(valid_names):
        c2w, _ = poses[name]
        translations[i] = c2w[:3, 3]
    t_range = translations.max(axis=0) - translations.min(axis=0)
    scene_scale = np.linalg.norm(t_range) + 1e-8
    tn = translations / scene_scale

    def _greedy(min_t_val, max_t_val, max_skip):
        accepted = [0]
        last = tn[0]
        skipped = 0
        for i in range(1, num_total):
            if len(accepted) >= num_keep:
                break
            dist = float(np.linalg.norm(tn[i] - last))
            if dist >= min_t_val:
                if dist <= max_t_val:
                    accepted.append(i); last = tn[i]; skipped = 0
                elif skipped >= max_skip:
                    accepted.append(i); last = tn[i]; skipped = 0
                else:
                    skipped += 1
            else:
                skipped += 1
                if skipped >= max_skip:
                    accepted.append(i); last = tn[i]; skipped = 0
        return accepted

    idx = _greedy(min_translation, max_translation, max_interval)
    if len(idx) < num_keep * 0.8:
        idx = _greedy(min_translation * 0.3, max_translation * 1.5, max_interval * 2)
    if len(idx) > num_keep:
        step = (len(idx) - 1) / (num_keep - 1)
        idx = [idx[int(round(j * step))] for j in range(num_keep)]
    while len(idx) < num_keep and len(idx) < num_total:
        new = list(idx)
        for k in range(len(idx) - 1):
            if len(new) >= num_keep:
                break
            mid = (idx[k] + idx[k + 1]) // 2
            if mid not in new and mid not in idx:
                new.append(mid)
        new = sorted(set(new))
        if len(new) == len(idx):
            break
        idx = new[:num_keep]
    return sorted(idx)[:num_keep]


def center_crop_resize_rgb(img_bgr, K, out_res):
    H, W = img_bgr.shape[:2]
    side = min(H, W)
    y0, x0 = (H - side) // 2, (W - side) // 2
    img_sq = img_bgr[y0:y0 + side, x0:x0 + side]
    K_new = K.copy()
    K_new[0, 2] -= x0
    K_new[1, 2] -= y0
    scale = out_res / side
    K_new[0, 0] *= scale
    K_new[1, 1] *= scale
    K_new[0, 2] *= scale
    K_new[1, 2] *= scale
    img_out = cv2.resize(img_sq, (out_res, out_res), interpolation=cv2.INTER_AREA)
    return img_out, K_new


def center_crop_depth(depth_u16, out_res):
    H, W = depth_u16.shape[:2]
    side = min(H, W)
    y0, x0 = (H - side) // 2, (W - side) // 2
    depth_sq = depth_u16[y0:y0 + side, x0:x0 + side]
    if depth_sq.shape[0] == out_res:
        return depth_sq
    return cv2.resize(depth_sq, (out_res, out_res), interpolation=cv2.INTER_NEAREST)


def encode_mp4_ffmpeg(frames_bgr, out_path: Path, fps: int) -> None:
    if not frames_bgr:
        raise RuntimeError("No frames to encode.")
    H, W = frames_bgr[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False,
        dir="/local-ssd" if Path("/local-ssd").is_dir() else None,
    ) as tmp:
        tmp_path = Path(tmp.name)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", str(tmp_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames_bgr:
            assert f.shape == (H, W, 3) and f.dtype == np.uint8
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
        subprocess.run(["cp", str(tmp_path), str(out_path)], check=True)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_copy_write(write_fn, dst: Path, suffix: str) -> None:
    tmp = Path(tempfile.gettempdir()) / f"_{os.getpid()}_{dst.parent.name}_{dst.name}{suffix}"
    write_fn(tmp)
    subprocess.run(["cp", str(tmp), str(dst)], check=True)
    tmp.unlink(missing_ok=True)


def process_scene(scene_id, source, output, num_frames, rgb_res, depth_res, fps,
                  pose_key, skip_existing, sampling_mode, max_interval,
                  min_translation, max_translation) -> str:
    scene_src = source / scene_id
    scene_dst = output / scene_id
    meta_path = scene_dst / "meta.json"
    video_path = scene_dst / "video.mp4"
    depth_path = scene_dst / "depth.npz"

    needs_video = not (skip_existing and video_path.is_file())
    needs_depth = not (skip_existing and depth_path.is_file())
    needs_meta = not (skip_existing and meta_path.is_file())
    if not (needs_video or needs_depth or needs_meta):
        return "skip"

    try:
        pose_meta = load_pose_metadata(scene_src, pose_key)
    except Exception as e:  # noqa: BLE001
        return f"err:pose_load:{e}"
    if not pose_meta:
        return "err:no_valid_poses"

    rgb_dir = scene_src / "rgb"
    depth_dir = scene_src / "depth"
    all_rgb = sorted(p.stem for p in rgb_dir.glob("frame_*.jpg"))
    valid_names = sorted(set(all_rgb) & set(pose_meta.keys())
                         & {p.stem for p in depth_dir.glob("frame_*.png")})
    if not valid_names:
        return "err:no_intersection"

    if sampling_mode == "progressive":
        pick = pick_frames_progressive(pose_meta, valid_names, num_frames,
                                       max_interval, min_translation, max_translation)
    else:
        pick = pick_frame_indices(len(valid_names), num_frames)
    picked = [valid_names[i] for i in pick]

    rgb_frames, depth_frames, c2w_list, K_list, frame_meta = [], [], [], [], []
    for name in picked:
        img = cv2.imread(str(rgb_dir / f"{name}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            return f"err:read_rgb:{name}"
        depth = cv2.imread(str(depth_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            return f"err:read_depth:{name}"
        c2w, K = pose_meta[name]
        rgb_resized, K_new = center_crop_resize_rgb(img, K, rgb_res)
        rgb_frames.append(rgb_resized)
        depth_frames.append(center_crop_depth(depth, depth_res))
        c2w_list.append(c2w.tolist())
        K_list.append(K_new.tolist())
        try:
            original_idx = int(name.replace("frame_", ""))
        except ValueError:
            original_idx = len(frame_meta)
        frame_meta.append({"name": name, "original_idx": original_idx})

    if len(rgb_frames) < 2:
        return "err:too_few_frames_kept"

    scene_dst.mkdir(parents=True, exist_ok=True)
    if needs_video:
        encode_mp4_ffmpeg(rgb_frames, video_path, fps=fps)
    if needs_depth:
        depth_arr = np.stack(depth_frames, axis=0).astype(np.uint16)
        _atomic_copy_write(lambda t: np.savez_compressed(t, depth_mm=depth_arr),
                           depth_path, ".npz")
    if needs_meta:
        meta = {
            "scene_id": scene_id,
            "num_frames": len(rgb_frames),
            "sampling_mode": sampling_mode,
            "fps": int(fps),
            "rgb_resolution": int(rgb_res),
            "depth_resolution": int(depth_res),
            "pose_key": pose_key,
            "source": "scannetpp",
            "frames": [
                {**frame_meta[i], "c2w": c2w_list[i], "K": K_list[i]}
                for i in range(len(rgb_frames))
            ],
        }
        _atomic_copy_write(
            lambda t: pathlib.Path(t).write_text(
                json.dumps(meta, separators=(",", ":"))),
            meta_path, ".json")
    return "ok"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[scan] listing valid scenes in {args.source} ...", flush=True)
    all_scenes = list_valid_scenes(args.source)
    print(f"[scan] {len(all_scenes)} scenes have colmap+depth+rgb.", flush=True)
    my_scenes = [s for i, s in enumerate(all_scenes)
                 if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        my_scenes = my_scenes[: args.limit]
    print(f"[shard {args.shard_index}/{args.num_shards}] processing "
          f"{len(my_scenes)} scenes.", flush=True)

    stats = {"ok": 0, "skip": 0, "err": 0}
    err_log: list[tuple[str, str]] = []
    for sid in tqdm(my_scenes, desc=f"shard{args.shard_index}"):
        try:
            res = process_scene(
                sid, args.source, args.output, args.num_frames, args.resolution,
                args.depth_resolution, args.fps, args.pose_key, args.skip_existing,
                args.sampling_mode, args.max_interval, args.min_translation,
                args.max_translation,
            )
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            res = f"err:exc:{type(e).__name__}:{e}"
            traceback.print_exc()
        if res == "ok":
            stats["ok"] += 1
        elif res == "skip":
            stats["skip"] += 1
        else:
            stats["err"] += 1
            err_log.append((sid, res))
    print(f"\n[shard {args.shard_index}] done: ok={stats['ok']} "
          f"skip={stats['skip']} err={stats['err']}", flush=True)
    if err_log:
        print("[shard] errors:")
        for sid, msg in err_log[:50]:
            print(f"  {sid}: {msg}")
    return 1 if stats["err"] else 0


if __name__ == "__main__":
    sys.exit(main())
