#!/usr/bin/env python3
"""Minimal example of the public ``gae.GAE`` API.

Reconstruction needs only the codec; sampling additionally needs a flow model
and camera/text conditioning. Both paths decode RGB **and** DPT geometry
(depth PNG + ``.ply``) via :meth:`GAE.save_outputs`.

    python examples/generate_min.py --image examples/scenes/forest_lake_trail.jpg \
        --codec-cfg configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt \
        --output results/generate_min

Add a flow checkpoint to exercise sampling:

    python examples/generate_min.py --image examples/scenes/forest_lake_trail.jpg \
        --codec-cfg configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt \
        --flow-cfg configs/flow_gae64.yaml --flow-ckpt ckpts/flow_gae64.pt \
        --output results/generate_min
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Works both after `pip install -e .` and from a plain checkout.
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import torch  # noqa: E402

from gae import GAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--codec-cfg", default="configs/gae_64.yaml")
    parser.add_argument("--codec-ckpt", required=True)
    parser.add_argument("--flow-cfg", default=None)
    parser.add_argument("--flow-ckpt", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/generate_min"),
                        help="Directory for RGB / depth / ply dumps.")
    parser.add_argument("--resolution", type=int, nargs=2, default=(378, 672),
                        metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--save-pointcloud", action="store_true", default=True)
    parser.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    parser.add_argument("--pc-stride", type=int, default=4)
    parser.add_argument("--total-views", type=int, default=4,
                        help="Sampled views when a flow checkpoint is given.")
    return parser.parse_args()


def load_image(path: Path, height: int, width: int) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    rgb = cv2.cvtColor(cv2.resize(bgr, (width, height)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]


def _dump(model: GAE, bundle: dict, out_dir: Path, stem: str, args) -> None:
    if not args.save_pointcloud:
        # Still write RGB so the script always produces an image.
        rgb_only = {"rgb": bundle["rgb"]}
        written = model.save_outputs(rgb_only, out_dir, stem, pc_stride=args.pc_stride)
    else:
        written = model.save_outputs(bundle, out_dir, stem, pc_stride=args.pc_stride)
    for k, v in written.items():
        print(f"  wrote {k}: {v}")


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    height, width = args.resolution
    args.output.mkdir(parents=True, exist_ok=True)

    model = GAE.from_configs(
        codec_cfg=args.codec_cfg,
        codec_ckpt=args.codec_ckpt,
        flow_cfg=args.flow_cfg,
        flow_ckpt=args.flow_ckpt,
        device=device,
    )

    images = load_image(args.image, height, width).to(device)

    z = model.encode(images)
    print(f"encode:      {tuple(images.shape)} -> {tuple(z.shape)}")

    recon = model.reconstruct(images)
    print(f"reconstruct: rgb {tuple(recon['rgb'].shape)}, "
          f"keys={sorted(recon)}")
    _dump(model, recon, args.output, "recon", args)

    if model.flow is not None:
        z_ref = z[0]  # [1, C, h, w]
        z_sampled = model.sample(z_ref, total_views=args.total_views, cond_num=1)
        print(f"sample:      -> {tuple(z_sampled.shape)}")
        rgb = model.decode_rgb(z_sampled, num_views=z_sampled.shape[0])
        bundle = {"rgb": rgb}
        if args.save_pointcloud:
            bundle.update(model.decode_geometry(z_sampled, height, width))
        _dump(model, bundle, args.output, "sample", args)
    else:
        print("No flow checkpoint provided; skipped sampling. "
              "Pass --flow-cfg/--flow-ckpt to enable it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
