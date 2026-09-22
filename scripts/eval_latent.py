#!/usr/bin/env python3
"""Latent-space diagnostics CLI — reproduces paper Tables 1 and 2.

Loads posterior-mean latents (``.pt`` / ``.npy``) and reports the intrinsic
diagnostics: rho (transport complexity), kappa (conditioning), effective rank,
LNC@k (semantic organisation), LDS / CDS / SRSS (spatial structure).

Examples
--------
    # a [N, C, H, W] tensor plus an optional [N] label tensor
    python scripts/eval_latent.py --latents z.pt --labels y.pt

    # several latents side by side, each with its own tag
    python scripts/eval_latent.py \\
        --latents gae64=z.pt gae128=z128.pt sdvae=z_sd.pt \\
        --image-res 252

    # skip the O(N^2) spatial metrics on large grids
    python scripts/eval_latent.py --latents z.pt --no-structure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from metrics.latent import latent_diagnostics  # noqa: E402


def _load(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".npy":
        return torch.from_numpy(np.load(p))
    obj = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("z", "latents", "mu", "posterior_mean"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(
                f"{p} is a dict but has none of the latent keys "
                f"(z / latents / mu / posterior_mean); keys={sorted(obj)[:8]}"
            )
    return torch.as_tensor(obj)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Intrinsic latent-space diagnostics (paper Tables 1 and 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--latents",
        nargs="+",
        required=True,
        metavar="[TAG=]PATH",
        help="One or more latent files. Prefix with 'TAG=' to label the output row.",
    )
    ap.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional label file (class / scene id per sample). Enables rho and LNC.",
    )
    ap.add_argument("--image-res", type=int, default=252,
                    help="Input resolution in pixels, used to scale spatial thresholds.")
    ap.add_argument("--lnc-k", type=int, default=5, help="Neighbourhood size for LNC@k.")
    ap.add_argument("--no-structure", action="store_true",
                    help="Skip the O(N^2) LDS / CDS computation.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json", type=str, default=None,
                    help="Also write the results to this JSON file.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    labels: Optional[List[int]] = None
    if args.labels:
        labels = [int(v) for v in _load(args.labels).flatten().tolist()]

    rows: Dict[str, Dict[str, float]] = {}
    for spec in args.latents:
        tag, _, path = spec.rpartition("=")
        if not path:                      # no '=' in the spec
            path, tag = tag, Path(spec).stem
        z = _load(path)
        if z.dim() == 3:                  # [N, C, H*W] or [C, H, W]
            z = z.unsqueeze(0) if z.shape[0] not in (1, 3) else z
        if z.dim() != 4:
            raise ValueError(f"{path}: expected a [N, C, H, W] tensor, got {tuple(z.shape)}")

        y = labels
        if y is not None and len(y) != z.shape[0]:
            print(f"[warn] {path}: {len(y)} labels vs {z.shape[0]} latents; ignoring labels")
            y = None

        rows[tag] = latent_diagnostics(
            z,
            y,
            device=args.device,
            lnc_k=args.lnc_k,
            compute_structure=not args.no_structure,
            image_res=args.image_res,
        )

    # ── report ──
    cols = ["rho", "kappa", "effective_rank", f"lnc@{args.lnc_k}", "lds", "cds", "srss"]
    cols = [c for c in cols if any(c in r for r in rows.values())]
    widths = [max(len(c), 12) for c in cols]
    name_w = max([len("latent")] + [len(t) for t in rows])

    print(" ".join(["latent".ljust(name_w)] + [c.rjust(w) for c, w in zip(cols, widths)]))
    print(" ".join(["-" * name_w] + ["-" * w for w in widths]))
    for tag, r in rows.items():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("n/a".rjust(12) if v is None else f"{v:12.4f}")
        print(" ".join([tag.ljust(name_w)] + cells))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\n[done] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
