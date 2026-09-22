#!/usr/bin/env python3
"""Pack MVS-Synth scenes into per-scene mp4 + meta.json (VideoMetaScene schema).

Output matches the RE10K/DL3DV packed layout, so MVS-Synth loads through the
same ``VideoMetaScene_Multi`` (dataset_tag="mvssynth") used in the configs.

Input layout::

    SOURCE/<scene_id>/
        images/<XXXX>.png       — RGB frames (e.g. 960x540 GTA-V renders)
        poses/<XXXX>.json       — per-frame {f_x, f_y, c_x, c_y, extrinsic 4x4 (w2c)}
        depths/<XXXX>.exr       — unused here

Output::

    OUTPUT/<scene_id>/
        video.mp4
        meta.json    — {scene_id, num_frames, height, width, rgb_resolution,
                        source, frames=[{name, c2w 4x4, K 3x3}]}

Notes
-----
* The MVS-Synth ``extrinsic`` field is **w2c**; ``c2w = inv(extrinsic)`` is the
  OpenCV camera-to-world used by the loader (no chirality fix needed).
* Per-frame intrinsics are written even though within a scene they are usually
  identical.

Usage::

    python scripts/data/preprocess_mvssynth.py \
        --source /datasets/MVS-Synth/GTAV_540 \
        --output "$GAE_DATA_ROOT/mvssynth_packed"
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _video_pack_helpers import (  # noqa: E402
    encode_mp4_ffmpeg,
    write_meta_json,
    read_png_bgr,
    round_robin_shard,
    ProgressLogger,
    fmt_stats,
)


def parse_args():
    p = argparse.ArgumentParser(description="Pack MVS-Synth into per-scene mp4+meta.")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def list_scenes(source: Path) -> list[tuple[str, Path]]:
    out = []
    for sd in sorted(source.iterdir()):
        if not sd.is_dir():
            continue
        if not (sd / "images").is_dir() or not (sd / "poses").is_dir():
            continue
        out.append((sd.name, sd))
    return out


def process_scene(scene_id, scene_dir, out_dir, fps, crf, skip_existing) -> str:
    dst = out_dir / scene_id
    video_path = dst / "video.mp4"
    meta_path = dst / "meta.json"
    if skip_existing and video_path.is_file() and meta_path.is_file():
        return "skip"

    img_dir = scene_dir / "images"
    pose_dir = scene_dir / "poses"
    basenames = sorted(p.stem for p in img_dir.glob("*.png"))
    basenames = [bn for bn in basenames if (pose_dir / f"{bn}.json").is_file()]
    if len(basenames) < 2:
        return "err:too_few_frames"

    rgb_bgr: list[np.ndarray] = []
    out_frames: list[dict] = []
    H = W = None
    for bn in basenames:
        img = read_png_bgr(img_dir / f"{bn}.png")
        if img is None:
            return f"err:read_png:{bn}"
        if H is None:
            H, W = img.shape[:2]
        elif img.shape[:2] != (H, W):
            return f"err:res_mismatch:{bn}"
        try:
            with open(pose_dir / f"{bn}.json") as f:
                d = json.load(f)
        except Exception as e:  # noqa: BLE001
            return f"err:pose_load:{bn}:{e}"
        K = np.array([[d["f_x"], 0.0, d["c_x"]],
                      [0.0, d["f_y"], d["c_y"]],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        try:
            extr = np.array(d["extrinsic"], dtype=np.float64)
            if extr.shape != (4, 4):
                return f"err:bad_extr_shape:{bn}"
            c2w = np.linalg.inv(extr).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            return f"err:invert:{bn}:{e}"
        if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
            return f"err:nonfinite:{bn}"
        rgb_bgr.append(img)
        out_frames.append({"name": bn, "c2w": c2w.tolist(), "K": K.tolist()})

    dst.mkdir(parents=True, exist_ok=True)
    if not (skip_existing and video_path.is_file()):
        encode_mp4_ffmpeg(rgb_bgr, video_path, fps=fps, crf=crf)
    if not (skip_existing and meta_path.is_file()):
        meta = {
            "scene_id": scene_id,
            "num_frames": len(out_frames),
            "fps": int(fps),
            "rgb_resolution": int(min(H, W)),
            "height": int(H),
            "width": int(W),
            "source": "mvssynth",
            "caption": "",
            "frames": out_frames,
        }
        write_meta_json(meta, meta_path)
    return "ok"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_scenes = list_scenes(args.source)
    print(f"[mvssynth] {len(all_scenes)} scenes total.", flush=True)
    my_scenes = round_robin_shard(all_scenes, args.num_shards, args.shard_index)
    if args.limit is not None:
        my_scenes = my_scenes[: args.limit]
    print(f"[mvssynth shard {args.shard_index}/{args.num_shards}] "
          f"processing {len(my_scenes)} scenes.", flush=True)

    stats = {"ok": 0, "skip": 0, "err": 0}
    err_log: list[tuple[str, str]] = []
    prog = ProgressLogger(len(my_scenes),
                          tag=f"mvssynth sh{args.shard_index}/{args.num_shards}",
                          interval=30.0)
    for scene_id, scene_dir in my_scenes:
        try:
            res = process_scene(scene_id, scene_dir, args.output,
                                args.fps, args.crf, args.skip_existing)
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
            err_log.append((scene_id, res))
        prog.update(1, extra=fmt_stats(stats))

    print(f"\n[mvssynth shard {args.shard_index}] done: ok={stats['ok']} "
          f"skip={stats['skip']} err={stats['err']}", flush=True)
    if err_log:
        print("[mvssynth] sample errors:")
        for sid, msg in err_log[:30]:
            print(f"  {sid}: {msg}")
    return 1 if stats["err"] else 0


if __name__ == "__main__":
    sys.exit(main())
