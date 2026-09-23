"""Run the released GAE codec reconstruction on one or all scene examples."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gae import GAE


def _save_depth(depth: torch.Tensor, path: Path) -> None:
    arr = depth.detach().float().cpu().squeeze().numpy()
    lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    vis = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    Image.fromarray((vis * 255.0 + 0.5).astype(np.uint8)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/vae_recon"))
    parser.add_argument("--hf-repo", default="TencentARC/GAE-D64-1B")
    parser.add_argument("--cache-dir", type=Path, default=Path("ckpts"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672), metavar=("H", "W"))
    parser.add_argument("--all-examples", action="store_true")
    args = parser.parse_args()
    if args.image is None and not args.all_examples:
        parser.error("provide --image or --all-examples")
    images = ([args.image] if args.image is not None else
              sorted(Path("examples/scenes").glob("*.jpg")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAE.from_pretrained(args.hf_repo, cache_dir=args.cache_dir, device=device)
    model.eval()
    h, w = args.resolution
    for image_path in images:
        image = Image.open(image_path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.reconstruct(tensor)
        out_dir = args.output / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb = out["rgb"][0, 0].float().cpu().clamp(0, 1)
        rgb_image = (rgb.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(rgb_image).save(out_dir / "rgb_recon.png")
        depth = out.get("depth")
        if depth is not None:
            _save_depth(depth[0, 0], out_dir / "depth_recon.png")
        print(f"{image_path} -> {out_dir}")


if __name__ == "__main__":
    main()
