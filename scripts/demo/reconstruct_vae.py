"""Run the released GAE codec reconstruction on one or all scene examples."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import imageio.v3 as iio

from gae import GAE


def _save_depth(depth: torch.Tensor, path: Path) -> None:
    arr = depth.detach().float().cpu().squeeze().numpy()
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    vis = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    Image.fromarray((vis * 255.0 + 0.5).astype(np.uint8)).save(path)


def _read_input(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    h, w = size
    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".avi", ".mkv"}:
        frames = iio.imread(path, index=None)
        fps = float(iio.immeta(path).get("fps", 8.0))
    else:
        frames = np.asarray(Image.open(path).convert("RGB"))[None]
        fps = 1.0
    if frames.ndim != 4:
        raise ValueError(f"expected RGB video frames [V,H,W,3], got {frames.shape}")
    resized = []
    for frame in frames:
        resized.append(np.asarray(Image.fromarray(frame).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)))
    return np.stack(resized), fps


def _save_video(frames: np.ndarray, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(frames, dtype=np.uint8), fps=fps, codec="libx264")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/vae_recon"))
    parser.add_argument("--hf-repo", default="TencentARC/GAE-D64-1B")
    parser.add_argument("--cache-dir", type=Path, default=Path("ckpts"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672), metavar=("H", "W"))
    parser.add_argument("--all-examples", action="store_true")
    args = parser.parse_args()
    if args.image is None and args.video is None and not args.all_examples:
        parser.error("provide --image, --video, or --all-examples")
    inputs = ([args.image] if args.image is not None else
              [args.video] if args.video is not None else
              sorted(Path("examples/scenes").glob("*.jpg")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAE.from_pretrained(args.hf_repo, cache_dir=args.cache_dir, device=device)
    model.eval()
    h, w = args.resolution
    for input_path in inputs:
        frames, fps = _read_input(input_path, (h, w))
        tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.reconstruct(tensor)
        out_dir = args.output / input_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        # ``GAE.reconstruct`` returns [V, 3, H, W] for a single batch. Keep
        # the view dimension for videos while accepting a batched [1,V,...]
        # output from compatible backbones.
        rgb = out["rgb"].float().cpu().clamp(0, 1)
        if rgb.ndim == 5 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4:
            raise RuntimeError(f"unexpected RGB reconstruction shape: {tuple(rgb.shape)}")
        rgb_frames = (rgb.permute(0, 2, 3, 1).numpy() * 255.0 + 0.5).astype(np.uint8)
        if len(rgb_frames) == 1:
            Image.fromarray(rgb_frames[0]).save(out_dir / "rgb_recon.png")
        else:
            _save_video(rgb_frames, out_dir / "rgb_recon.mp4", fps)
        depth = out.get("depth")
        if depth is not None:
            if depth.ndim == 5 and depth.shape[0] == 1:
                depth = depth[0]
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            if depth.ndim != 4:
                raise RuntimeError(f"unexpected depth reconstruction shape: {tuple(depth.shape)}")
            depth_frames = []
            for frame in depth:
                arr = frame.detach().float().cpu().squeeze().numpy()
                lo, hi = np.nanpercentile(arr, [1.0, 99.0])
                vis = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                depth_frames.append((vis * 255.0 + 0.5).astype(np.uint8))
            if len(depth_frames) == 1:
                Image.fromarray(depth_frames[0]).save(out_dir / "depth_recon.png")
            else:
                _save_video(np.stack(depth_frames)[..., None].repeat(3, axis=-1), out_dir / "depth_recon.mp4", fps)
        print(f"{input_path} -> {out_dir}")


if __name__ == "__main__":
    main()
