#!/usr/bin/env python3
"""Evaluate official MEt3R directly from generated multi-view videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--pair-mode", choices=["all", "consecutive"], default="all")
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_video_frames(path: Path) -> torch.Tensor:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Cannot read frames from {path}")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0


def load_metric(device: torch.device, img_size: int) -> Any:
    from met3r import MEt3R

    print(f"[MEt3R] loading official metric (img_size={img_size})", flush=True)
    return MEt3R(
        img_size=img_size,
        use_norm=True,
        backbone="mast3r",
        feature_backbone="dino16",
        feature_backbone_weights="mhamilton723/FeatUp",
        upsampler="featup",
        distance="cosine",
        freeze=True,
    ).to(device).eval()


@torch.inference_mode()
def score_video(
    frames: torch.Tensor,
    metric: Any,
    device: torch.device,
    img_size: int,
    pair_mode: str,
    batch_pairs: int,
) -> float:
    if frames.shape[-2:] != (img_size, img_size):
        frames = F.interpolate(frames, size=(img_size, img_size), mode="bilinear", align_corners=False)
    frame_count = frames.shape[0]
    pairs = (
        [(index, index + 1) for index in range(frame_count - 1)]
        if pair_mode == "consecutive"
        else [(left, right) for left in range(frame_count) for right in range(left + 1, frame_count)]
    )
    scores: list[float] = []
    for start in range(0, len(pairs), batch_pairs):
        batch = pairs[start : start + batch_pairs]
        inputs = torch.stack([torch.stack([frames[left], frames[right]]) for left, right in batch])
        values = metric(images=inputs.to(device) * 2.0 - 1.0)[0]
        scores.extend(np.atleast_1d(values.detach().float().cpu().numpy()).tolist())
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("MEt3R evaluation requires CUDA")
    videos = sorted(args.pred_dir.glob("*_pred.mp4"))
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"No *_pred.mp4 files in {args.pred_dir}")

    metric = load_metric(device, args.img_size)
    rows: list[dict[str, float | str]] = []
    for index, video in enumerate(videos, start=1):
        score = score_video(
            load_video_frames(video), metric, device, args.img_size, args.pair_mode, args.batch_pairs
        )
        rows.append({"scene": video.stem.removesuffix("_pred"), "met3r": score})
        print(f"[{index}/{len(videos)}] {video.stem}: MEt3R={score:.4f}", flush=True)

    summary = {
        "MEt3R": float(np.mean([float(row["met3r"]) for row in rows])),
        "num_scenes": len(rows),
        "config": {"img_size": args.img_size, "pair_mode": args.pair_mode, "batch_pairs": args.batch_pairs},
        "per_scene": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.csv_output or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "met3r"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"MEt3R={summary['MEt3R']:.4f} over {len(rows)} scenes", flush=True)


if __name__ == "__main__":
    main()
