#!/usr/bin/env python3
"""Encode images or packed videos into posterior-mean GAE latents.

The output can be passed directly to ``scripts/eval_latent.py``.

Examples:
    python scripts/encode_latents.py \
        --input /data/gae/re10k_packed/test \
        --config configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt \
        --output results/re10k_gae64_latents.pt --max-scenes 100

    python scripts/eval_latent.py \
        --latents gae64=results/re10k_gae64_latents.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from gae import GAE, load_backbone, load_codec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="image, video, image directory, or packed dataset root")
    parser.add_argument("--config", default="configs/gae_64.yaml")
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=(504, 504),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--views-per-scene", type=int, default=33)
    parser.add_argument("--max-scenes", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _sample_indices(length: int, count: int) -> list[int]:
    count = min(length, count)
    return np.linspace(0, length - 1, count).round().astype(int).tolist()


def _read_video(path: Path, views: int, height: int, width: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = _sample_indices(length, views)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return torch.stack(frames)


def _read_images(paths: list[Path], views: int, height: int, width: int) -> torch.Tensor:
    selected = [paths[i] for i in _sample_indices(len(paths), views)]
    frames = []
    for path in selected:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
    if not frames:
        raise RuntimeError("no images decoded")
    return torch.stack(frames)


def _collect_inputs(path: Path, max_scenes: int) -> list[tuple[str, Path | list[Path]]]:
    if path.is_file():
        return [(path.stem, path)]
    videos = sorted(path.glob("*/video.mp4"))[:max_scenes]
    if videos:
        return [(p.parent.name, p) for p in videos]
    images = sorted(
        p for p in path.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if images:
        return [(path.name, images)]
    raise SystemExit(f"no videos or images found under {path}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    config = str((ROOT / args.config).resolve()) if not Path(args.config).is_absolute() else args.config

    # Latent extraction only needs the encoder; avoid constructing the DPT head.
    cfg = OmegaConf.load(config)
    cfg.stage_1.params.da3_weights_path = None
    temp_config = args.output.parent / ".encode_latents_config.yaml"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, temp_config)
    try:
        backbone = load_backbone(str(temp_config), device)
        codec = load_codec(config, args.codec_ckpt, device)
        model = GAE(backbone, codec).to(device).eval()

        height, width = args.resolution
        scene_latents = []
        scene_ids = []
        for scene_id, source in _collect_inputs(args.input, args.max_scenes):
            if isinstance(source, list):
                frames = _read_images(source, args.views_per_scene, height, width)
            elif source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
                frames = _read_video(source, args.views_per_scene, height, width)
            else:
                frames = _read_images([source], 1, height, width)
            with torch.inference_mode():
                z = model.encode(frames.unsqueeze(0).to(device))[0].cpu()
            scene_latents.append(z)
            scene_ids.extend([scene_id] * z.shape[0])
            print(f"[encode] {scene_id}: {tuple(frames.shape)} -> {tuple(z.shape)}")

        latents = torch.cat(scene_latents, dim=0)
        torch.save(latents, args.output)
        torch.save({"scene_ids": scene_ids}, args.output.with_suffix(".meta.pt"))
        print(f"[done] {tuple(latents.shape)} -> {args.output}")
    finally:
        temp_config.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
