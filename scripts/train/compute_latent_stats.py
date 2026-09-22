#!/usr/bin/env python3
"""Compute latent standardization statistics for a trained GAE codec (Eq. 15).

Stage 2 flow training and inference standardize the codec posterior mean before
the flow model sees it. The flow configs point at this file:

    latent_stats: "ckpts/latent_stats_gae_64.pt"

If you train your own codec you must produce that file, or Stage 2 will run on an
unnormalized latent space. This script iterates the training distribution,
encodes each view through the frozen DA3 backbone + codec (``GAE.encode``), and
accumulates per-channel mean/std — optionally the full-covariance whitening
operator used by the text-to-image recipe.

Example
-------
    python scripts/train/compute_latent_stats.py \
        --config configs/gae_64.yaml \
        --codec-ckpt ckpts/gae_64.pt \
        --num-batches 500 \
        --output ckpts/latent_stats_gae_64.pt

    # Only have RE10K? Restrict the sampling distribution:
    python scripts/train/compute_latent_stats.py --config configs/gae_64.yaml \
        --codec-ckpt ckpts/gae_64.pt --dataset dataset_re10k \
        --output latent_stats/gae_64_re10k.pt

Output keys: ``mean`` (C,), ``std`` (C,), ``count``; with ``--full-cov`` also
``cov`` (C,C), ``whiten`` (C,C), ``unwhiten`` (C,C) matching the loader
(``utils.train_runtime.load_latent_stats``).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from gae import GAE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Codec config, e.g. configs/gae_64.yaml")
    p.add_argument("--codec-ckpt", required=True, help="Trained codec checkpoint")
    p.add_argument("--output", required=True, help="Destination .pt file")
    p.add_argument("--num-batches", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dataset", default="train_dataset",
                   help="Config key (train_dataset/test_dataset/dataset_re10k...) "
                        "or a raw dataset spec string.")
    p.add_argument("--full-cov", action="store_true",
                   help="Also compute the full-covariance whitening operator.")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    return p.parse_args()


def _save_portable(stats: dict, output: str) -> None:
    stage_dir = "/local-ssd/_stats_tmp" if os.path.isdir("/local-ssd") else "/tmp"
    os.makedirs(stage_dir, exist_ok=True)
    tmp = os.path.join(stage_dir, os.path.basename(output))
    torch.save(stats, tmp)
    out_parent = os.path.dirname(output) or "."
    os.makedirs(out_parent, exist_ok=True)
    try:
        os.remove(output)
    except FileNotFoundError:
        pass
    # cp handles cross-device + S3-FUSE (no in-place / rename support).
    if subprocess.call(["cp", tmp, output]) != 0:
        raise RuntimeError(f"failed to copy stats to {output}")


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    cfg = OmegaConf.load(args.config)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    dataset_spec = resolved.get(args.dataset, args.dataset)
    if not isinstance(dataset_spec, str):
        raise ValueError(f"--dataset '{args.dataset}' did not resolve to a spec string")

    print("[1/3] Building GAE codec ...")
    gae = GAE.from_configs(codec_cfg=args.config, codec_ckpt=args.codec_ckpt, device=device)
    gae.eval()
    C = int(gae.codec.latent_dim)
    print(f"  latent_dim={C}")

    print("[2/3] Building dataloader ...")
    from cut3r_data import get_data_loader, robust_collate
    from cut3r_data.utils.image import imgnorm_to_unit
    loader = get_data_loader(
        dataset_spec, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_mem=True, shuffle=True, drop_last=True, fixed_length=True,
        world_size=1, rank=0, collate_fn=robust_collate,
    )
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)

    print("[3/3] Accumulating latent statistics ...")
    n = 0
    s1 = torch.zeros(C, dtype=torch.float64, device=device)   # sum
    s2 = torch.zeros(C, dtype=torch.float64, device=device)   # sum of squares
    sco = (torch.zeros(C, C, dtype=torch.float64, device=device)
           if args.full_cov else None)                        # sum of outer products

    good = skipped = 0
    it = iter(loader)
    pbar = tqdm(total=args.num_batches, desc="latent stats")
    while good < args.num_batches:
        try:
            batch = next(it)
        except StopIteration:
            print(f"  loader exhausted after {good} batches")
            break
        except RuntimeError as e:
            if "max_pose_retries" in str(e) or "bad poses" in str(e):
                skipped += 1
                continue
            raise
        try:
            # robust_collate yields a list of per-view dicts with ImageNet-normalized
            # "img"; stack to [B,V,3,H,W] and undo the norm so GAE.encode (which
            # re-applies ImageNet norm internally) receives [0,1] pixels.
            if isinstance(batch, list):
                image = torch.stack([d["img"] for d in batch], dim=1)
                image = imgnorm_to_unit(image).to(device, non_blocking=True)
            else:
                image = batch["gt_inp"].to(device, non_blocking=True)  # [B,V,3,H,W] in [0,1]
            with torch.amp.autocast("cuda", enabled=(dtype == torch.bfloat16), dtype=dtype):
                mu = gae.encode(image)                             # [B,V,C,h,w]
            flat = mu.float().permute(2, 0, 1, 3, 4).reshape(C, -1).double()  # (C, N)
            n += flat.shape[1]
            s1 += flat.sum(dim=1)
            s2 += (flat * flat).sum(dim=1)
            if sco is not None:
                sco += flat @ flat.t()
            good += 1
            pbar.update(1)
            if good % 50 == 0:
                mean = s1 / n
                std = (s2 / n - mean * mean).clamp_min(0).sqrt()
                print(f"  [{good}] N={n:,} mean=[{mean.min():.4f},{mean.max():.4f}] "
                      f"std=[{std.min():.4f},{std.max():.4f}] skipped={skipped}")
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped % 20 == 1:
                print(f"  [skip #{skipped}] {e}")
            continue
    pbar.close()

    if n == 0:
        raise RuntimeError("no latents accumulated; check --dataset and data paths")

    mean = s1 / n
    var = (s2 / n - mean * mean).clamp_min(1e-12)
    std = var.sqrt()
    stats = {"mean": mean.cpu().float(), "std": std.cpu().float(), "count": int(n)}
    print(f"\n  samples: {n:,}")
    print(f"  mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  std  range: [{std.min():.4f}, {std.max():.4f}]")

    if sco is not None:
        cov = sco / n - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.t())
        evals, evecs = torch.linalg.eigh(cov)
        evals = evals.clamp_min(1e-8)
        whiten = (evecs * evals.rsqrt()) @ evecs.t()
        unwhiten = (evecs * evals.sqrt()) @ evecs.t()
        stats.update({
            "cov": cov.cpu().float(),
            "whiten": whiten.cpu().float(),
            "unwhiten": unwhiten.cpu().float(),
        })
        print(f"  full-cov: cond={float(evals.max() / evals.min()):.2f} "
              f"eig=[{evals.min():.3e}, {evals.max():.3e}]")

    _save_portable(stats, args.output)
    print(f"  saved -> {args.output}")


if __name__ == "__main__":
    main()
