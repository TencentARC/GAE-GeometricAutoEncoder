#!/usr/bin/env python
"""Pre-build VideoMetaScene / ScanNetppRGB index pickles (local + shared cache).

Run once before codec training (``scripts/train.py codec``) so rank0 does
not block torchrun for hours on a cold FUSE scan. Safe to re-run (idempotent).

Usage::

    VMS_INDEX_WORKERS=32 OSP_INDEX_WORKERS=64 \\
    PYTHONPATH=src python scripts/build_dataset_index.py \\
        --config configs/gae_128.yaml

Speed: VMS reads only meta.json headers (skips ``frames``); OSP uses parallel
JSON parse (``pip install orjson`` optional, ~1.5× faster).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset index caches from a released config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-views", type=int, default=None,
                    help="Override yaml num_views (default: use resolved yaml value)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    nv = int(args.num_views if args.num_views is not None else cfg.num_views)

    import cut3r_data as cd

    resolved = OmegaConf.to_container(cfg, resolve=True)
    specs = {
        k: v for k, v in resolved.items()
        if str(k).startswith("dataset_") and "test" not in str(k)
    }

    print(f"[build_index] num_views={nv}  datasets={list(specs)}", flush=True)
    for name, spec in specs.items():
        spec_s = spec.replace(f"num_views={resolved['num_views']}", f"num_views={nv}")
        print(f"\n[build_index] === {name} ===", flush=True)
        ds = eval(spec_s, vars(cd))
        n = len(ds)
        print(f"[build_index] {name}: {n} samples", flush=True)


if __name__ == "__main__":
    main()
