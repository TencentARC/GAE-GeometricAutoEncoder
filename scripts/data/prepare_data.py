#!/usr/bin/env python3
"""Pack RealEstate10K or DL3DV into GAE's scene format.

Each output scene contains:

    video.mp4
    meta.json
    caption.txt  (when a caption is available)

Examples:
    python scripts/data/prepare_data.py re10k \
        --source /datasets/RealEstate10K --output /data/gae/re10k_packed \
        --split train --workers 8

    python scripts/data/prepare_data.py dl3dv \
        --source /datasets/DL3DV-10K --output /data/gae/dl3dv_packed \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("re10k", "dl3dv"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train",
                        help="RE10K split; ignored for DL3DV")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _encode_video(frame_paths: list[Path], output: Path, fps: int, crf: int) -> tuple[int, int]:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise IOError(f"cannot read {frame_paths[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    os.unlink(tmp_name)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", str(crf), "-pix_fmt", "yuv420p", tmp_name,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for path in frame_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise IOError(f"cannot read {path}")
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"inconsistent image size in {path}: {image.shape[:2]} "
                    f"!= {(height, width)}"
                )
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {output}")
        try:
            os.replace(tmp_name, output)
        except OSError:
            subprocess.run(["cp", tmp_name, str(output)], check=True)
            os.unlink(tmp_name)
    except Exception:
        if proc.poll() is None:
            proc.kill()
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return height, width


def _caption(path: Path) -> str:
    if not path.is_file():
        return ""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return str(data.get("caption", "")).strip()
    return path.read_text().strip()


def _re10k_scenes(args: argparse.Namespace) -> list[dict]:
    root = args.source / "process" / args.split
    if not root.is_dir():
        root = args.source / args.split
    caption_file = args.source / f"{args.split}_caption.json"
    captions = json.loads(caption_file.read_text()) if caption_file.is_file() else {}
    scenes = []
    for shard in sorted(root.iterdir()):
        if not shard.is_dir():
            continue
        for scene in sorted(shard.iterdir()):
            transforms = scene / "transforms.json"
            if scene.is_dir() and transforms.is_file():
                scenes.append({
                    "dataset": "re10k", "id": scene.name, "root": scene,
                    "transforms": transforms, "caption": captions.get(scene.name, ""),
                    "output": args.output / args.split / scene.name,
                    "fps": args.fps, "crf": args.crf,
                    "skip": args.skip_existing,
                })
    return scenes


def _dl3dv_scenes(args: argparse.Namespace) -> list[dict]:
    scenes = []
    for split in sorted(args.source.iterdir()):
        if not split.is_dir():
            continue
        for scene in sorted(split.iterdir()):
            root = scene / "nerfstudio"
            if not root.is_dir():
                root = scene
            transforms = root / "transforms.json"
            if transforms.is_file() and (root / "images_4").is_dir():
                scenes.append({
                    "dataset": "dl3dv", "id": scene.name, "root": root,
                    "transforms": transforms,
                    "caption": _caption(root / "wan2_caption.json"),
                    "output": args.output / scene.name,
                    "fps": args.fps, "crf": args.crf,
                    "skip": args.skip_existing,
                })
    return scenes


def _process(spec: dict) -> tuple[str, str]:
    output = Path(spec["output"])
    video = output / "video.mp4"
    metadata = output / "meta.json"
    if spec["skip"] and video.is_file() and metadata.is_file():
        return spec["id"], "skip"

    tf = json.loads(Path(spec["transforms"]).read_text())
    frames = list(tf.get("frames", []))
    if spec["dataset"] == "re10k":
        try:
            frames.sort(key=lambda x: int(Path(x["file_path"]).stem))
        except ValueError:
            frames.sort(key=lambda x: Path(x["file_path"]).stem)
    else:
        frames.sort(key=lambda x: Path(x["file_path"]).stem)
    if len(frames) < 2:
        return spec["id"], "too-few-frames"

    if spec["dataset"] == "dl3dv":
        image_dir = Path(spec["root"]) / "images_4"
        frame_paths = [image_dir / f"{Path(f['file_path']).stem}.png" for f in frames]
    else:
        root = Path(spec["root"])
        frame_paths = [root / f"{Path(f['file_path']).stem}.png" for f in frames]

    if any(not p.is_file() for p in frame_paths):
        missing = next(p for p in frame_paths if not p.is_file())
        return spec["id"], f"missing:{missing.name}"

    height, width = _encode_video(frame_paths, video, spec["fps"], spec["crf"])
    source_width = float(tf.get("w", width))
    source_height = float(tf.get("h", height))
    sx, sy = width / source_width, height / source_height
    K = np.array([
        [float(tf["fl_x"]) * sx, 0.0, float(tf["cx"]) * sx],
        [0.0, float(tf["fl_y"]) * sy, float(tf["cy"]) * sy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    packed_frames = []
    for index, frame in enumerate(frames):
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, np.array([0.0, 0.0, 0.0, 1.0], np.float32)])
        if c2w.shape != (4, 4):
            raise ValueError(f"{spec['id']} frame {index} has invalid c2w {c2w.shape}")
        if spec["dataset"] == "dl3dv":
            c2w = c2w @ GL_TO_CV
        packed_frames.append({
            "index": index,
            "name": Path(frame["file_path"]).stem,
            "c2w": c2w.tolist(),
            "K": K.tolist(),
        })
    payload = {
        "scene_id": spec["id"],
        "num_frames": len(packed_frames),
        "fps": spec["fps"],
        "height": height,
        "width": width,
        "rgb_resolution": min(height, width),
        "source": spec["dataset"],
        "frames": packed_frames,
    }
    caption = str(spec.get("caption", "")).strip()
    if caption:
        payload["caption"] = caption
    if spec["dataset"] == "dl3dv":
        payload["applied_gl2cv"] = True
    _atomic_text(metadata, json.dumps(payload, indent=2))
    if caption:
        _atomic_text(output / "caption.txt", caption + "\n")
    return spec["id"], "ok"


def main() -> int:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"source does not exist: {args.source}")
    if not shutil_which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    scenes = _re10k_scenes(args) if args.dataset == "re10k" else _dl3dv_scenes(args)
    if args.limit is not None:
        scenes = scenes[:args.limit]
    if not scenes:
        raise SystemExit("no scenes found; check --source and the expected layout")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] {args.dataset}: {len(scenes)} scenes -> {args.output}")
    counts: dict[str, int] = {}
    workers = max(1, args.workers)
    with mp.Pool(workers) if workers > 1 else _SerialPool() as pool:
        for done, (scene_id, status) in enumerate(pool.imap_unordered(_process, scenes), 1):
            counts[status] = counts.get(status, 0) + 1
            if done == 1 or done % 25 == 0 or done == len(scenes):
                print(f"[{done}/{len(scenes)}] {scene_id}: {status}  {counts}", flush=True)
    return 1 if any(k not in ("ok", "skip") for k in counts) else 0


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


class _SerialPool:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @staticmethod
    def imap_unordered(fn, items):
        return map(fn, items)


if __name__ == "__main__":
    raise SystemExit(main())
