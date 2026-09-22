#!/usr/bin/env python3
"""Download GAE checkpoints from the Hugging Face Hub.

Default repo is ``TencentARC/GAE-D64-1B``. ``scripts/run_demo.sh`` calls this
when ``ckpts/`` is empty.

Examples
--------
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --size 64 --out-dir ckpts
    python scripts/download_checkpoints.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gae.hub import DEFAULT_REPO, MANIFEST, download_weights, files_for  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Download GAE checkpoints from the Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--repo",
        default=os.environ.get("GAE_HF_REPO", DEFAULT_REPO),
        help=f"Hugging Face repo id (default: $GAE_HF_REPO or {DEFAULT_REPO}).",
    )
    ap.add_argument(
        "--size",
        choices=("64", "128", "all"),
        default="64",
        help="Which GAE size to fetch (default: 64).",
    )
    ap.add_argument(
        "--subset",
        choices=("all", "codec", "flow"),
        default="all",
        help="Which group of files to fetch (default: all).",
    )
    ap.add_argument("--out-dir", default="ckpts", help="Destination directory.")
    ap.add_argument("--list", action="store_true", help="Print the manifest and exit.")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list:
        print(f"{'file':28s} {'subset':8s} {'size':6s} description")
        print("-" * 100)
        for name, (subset, size, desc) in MANIFEST.items():
            print(f"{name:28s} {subset:8s} {size:6s} {desc}")
        return 0

    names = files_for(args.size, args.subset)
    print(f"repo   : {args.repo}")
    print(f"size   : {args.size}")
    print(f"subset : {args.subset} ({len(names)} file(s))")
    print(f"dest   : {Path(args.out_dir).resolve()}\n")

    try:
        paths = download_weights(
            args.repo, size=args.size, subset=args.subset, out_dir=args.out_dir
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for name in names:
        print(f"  ok   {name:28s} -> {paths[name]}")
    print(f"\n[done] {len(names)} file(s) ready in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
