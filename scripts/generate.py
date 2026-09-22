#!/usr/bin/env python3
"""Generate a camera-controlled video and point cloud from one image + prompt.

This is the public single-scene wrapper around ``eval_generation.py``. It
creates the small scene manifest expected by the research evaluator, then runs
the released GAE flow sampler.

Example:
    python scripts/generate.py \
        --image examples/scenes/forest_lake_trail.jpg \
        --prompt-file examples/scenes/forest_lake_trail.txt \
        --hf-repo TencentARC/GAE-D64-1B \
        --output results/forest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default=None, help="Prompt text; or use --prompt-file.")
    parser.add_argument("--prompt-file", type=Path, default=None,
                        help="Read the prompt from a text file, e.g. examples/scenes/<scene>.txt.")
    parser.add_argument(
        "--hf-repo", default=None,
        help="Hugging Face repo id (default env GAE_HF_REPO or TencentARC/GAE-D64-1B). "
             "Downloads codec + flow into --ckpt-dir when local files are missing.")
    parser.add_argument("--ckpt-dir", type=Path, default=ROOT / "ckpts")
    parser.add_argument("--flow-ckpt", type=Path, default=None)
    parser.add_argument("--codec-ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/flow_gae64.yaml")
    parser.add_argument("--codec-config", type=Path, default=ROOT / "configs/gae_64.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-views", type=int, default=81,
                        help="views per denoising chunk; 81 matches the shipped "
                             "flow recipe (V=81) and needs no rollout")
    parser.add_argument("--total-views", type=int, default=81,
                        help="total output frames; exceeding --num-views enables "
                             "chunked rollout")
    parser.add_argument("--roll-cond-num", type=int, default=1,
                        help="frames carried over between rollout chunks")
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672),
                        metavar=("HEIGHT", "WIDTH"),
                        help="H W; default 378x672 matches the shipped 672x378 flow recipe")
    parser.add_argument(
        "--poses", type=Path, default=None,
        help="GT camera npz (c2w Nx4x4, K Nx3x3). Default: <image_stem>_poses.npz "
             "next to --image when that file exists.")
    parser.add_argument(
        "--free-rollout", action="store_true",
        help="Ignore GT poses and synthesize a camera path (--trajectory / --speed).")
    parser.add_argument("--trajectory", choices=("wander", "orbit", "spiral", "drive"),
                        default="wander")
    parser.add_argument("--speed", type=float, default=0.06)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--save-pointcloud", action="store_true", default=True,
                        help="Write DPT depth + .ply (default on).")
    parser.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    parser.add_argument("--pc-stride", type=int, default=4)
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            raise SystemExit(f"--prompt-file does not exist: {args.prompt_file}")
        args.prompt = args.prompt_file.read_text().strip()
    if not args.prompt:
        raise SystemExit("Provide a prompt via --prompt or --prompt-file.")
    if args.hf_repo or args.flow_ckpt is None or args.codec_ckpt is None:
        from gae.hub import DEFAULT_REPO, download_weights, extract_da3_stats
        repo = args.hf_repo or os.environ.get("GAE_HF_REPO") or DEFAULT_REPO
        paths = download_weights(repo, size="64", out_dir=args.ckpt_dir)
        tar = paths.get("da3_stats_giant_5ds.tar")
        if tar is not None:
            extract_da3_stats(tar, ROOT / "model_stats" / "da3_giant_5ds")
        if args.codec_ckpt is None:
            args.codec_ckpt = paths["gae_64.pt"]
        if args.flow_ckpt is None:
            args.flow_ckpt = paths["flow_gae64.pt"]
    return args, extra


def _check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")


def _resolve_poses_path(args: argparse.Namespace) -> Path | None:
    if args.free_rollout or "--trajectory" in sys.argv:
        return None
    if args.poses is not None:
        return args.poses
    sibling = args.image.with_name(f"{args.image.stem}_poses.npz")
    return sibling if sibling.is_file() else None


def _load_gt_cameras(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, int | None]:
    data = np.load(path)
    if "c2w" not in data or "K" not in data:
        raise SystemExit(f"{path} must contain 'c2w' (Nx4x4) and 'K' (Nx3x3)")
    c2w = np.asarray(data["c2w"], dtype=np.float32)
    K = np.asarray(data["K"], dtype=np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise SystemExit(f"{path}: c2w expected (N,4,4), got {c2w.shape}")
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise SystemExit(f"{path}: K expected (N,3,3), got {K.shape}")
    n_cam = min(int(n), int(c2w.shape[0]), int(K.shape[0]))
    fps = int(data["fps"]) if "fps" in data else None
    return c2w[:n_cam], K[:n_cam], fps


def _prepare_scene(args: argparse.Namespace, poses_path: Path | None) -> Path:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    height, width = args.resolution
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    n = args.num_views
    if poses_path is not None:
        _check_file(poses_path, "poses")
        c2w, Ks, fps = _load_gt_cameras(poses_path, n)
        n = int(c2w.shape[0])
        args.num_views = n
        args.total_views = min(int(args.total_views), n)
        if fps:
            args.fps = fps
        frames = [
            {
                "index": i,
                "name": f"{i:06d}",
                "c2w": c2w[i].tolist(),
                "K": Ks[i].tolist(),
            }
            for i in range(n)
        ]
        print(f"[generate] using GT cameras from {poses_path} ({n} frames)", flush=True)
    else:
        focal = 0.8 * max(height, width)
        K = [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
        frames = [
            {
                "index": i,
                "name": f"{i:06d}",
                "c2w": np.eye(4, dtype=np.float32).tolist(),
                "K": K,
            }
            for i in range(n)
        ]
        print(f"[generate] no GT poses; synthetic '{args.trajectory}' trajectory", flush=True)

    scene = args.output / "_input_scene"
    scene.mkdir(parents=True, exist_ok=True)
    video = scene / "video.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height)
    )
    if not writer.isOpened():
        raise SystemExit("OpenCV cannot create the temporary input video")
    for _ in range(n):
        writer.write(image)
    writer.release()
    (scene / "meta.json").write_text(json.dumps({
        "scene_id": args.image.stem,
        "num_frames": n,
        "height": height,
        "width": width,
        "rgb_resolution": [height, width],
        "caption": args.prompt,
        "frames": frames,
    }, indent=2))
    (scene / "caption.txt").write_text(args.prompt.strip() + "\n")

    manifest = args.output / "_input_manifest.json"
    manifest.write_text(json.dumps({
        "dataset": "scannetpp",
        "num_views": n,
        "scenes": [{
            "scene_name": args.image.stem,
            "scene_dir": str(scene.resolve()),
            "img_names": [f"{i:06d}" for i in range(n)],
            "ds_type": "scannetpp",
        }],
    }, indent=2))
    return manifest


def main() -> int:
    args, extra = parse_args()
    for path, label in (
        (args.image, "image"),
        (args.flow_ckpt, "flow checkpoint"),
        (args.codec_ckpt, "codec checkpoint"),
        (args.config, "flow config"),
        (args.codec_config, "codec config"),
    ):
        _check_file(path, label)
    if args.total_views > args.num_views and not 0 < args.roll_cond_num < args.num_views:
        raise SystemExit("--roll-cond-num must be in (0, num-views) for long rollout")

    args.output.mkdir(parents=True, exist_ok=True)
    poses_path = _resolve_poses_path(args)
    manifest = _prepare_scene(args, poses_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}:{env.get('PYTHONPATH', '')}"
    env.pop("FREE_ROLLOUT", None)
    if poses_path is None:
        env.update({
            "FREE_ROLLOUT": "1",
            "FREE_ROLLOUT_VIEWS": str(args.total_views),
            "FREE_ROLLOUT_MOTION": args.trajectory,
            "FREE_ROLLOUT_SPEED": str(args.speed),
        })
    command = [
        sys.executable, str(ROOT / "scripts/eval_generation.py"),
        "--dit-ckpt", str(args.flow_ckpt),
        "--vae-ckpt", str(args.codec_ckpt),
        "--gld-config", str(args.codec_config),
        "--config", str(args.config),
        "--scene-manifest", str(manifest),
        "--dataset", "scannetpp",
        "--mode", "generate",
        "--prompt", args.prompt,
        "--cond-num", "1",
        "--num-scenes", "1",
        "--num-views", str(args.num_views),
        "--total-views", str(args.total_views),
        "--roll-cond-num", str(args.roll_cond_num),
        "--resolution", str(args.resolution[0]), str(args.resolution[1]),
        "--sample-steps", str(args.sample_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--seed", str(args.seed),
        "--video-fps", str(args.fps),
        "--pc-stride", str(args.pc_stride),
        "--output-dir", str(args.output),
        *extra,
    ]
    command.append("--save-pointcloud" if args.save_pointcloud else "--no-pointcloud")
    print("[generate]", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
