#!/usr/bin/env python3
"""Public launcher for GAE codec and flow training.

Examples:
    python scripts/train/train.py codec --size 64 --gpus 8
    python scripts/train/train.py codec --size 128 --gpus 8
    python scripts/train/train.py flow --size 64 --gpus 8
    python scripts/train/train.py flow --size 128 --gpus 8 --cotrain-t2i   # i2v + T2I

Arguments after ``--`` are forwarded to the underlying trainer.

Text-to-image is not a separate task: it is co-trained *inside* the Stage 2
flow (i2v/t2v) model. Pass ``--cotrain-t2i`` to interleave single-image
BLIP3o / ImageNet steps into the multi-view loop (prepare the data with
``scripts/data/prepare_t2i_data.py``; see docs/DATA.md). The Stage 1 codec has its
own optional ``cotrain_t2i`` block for RGB-decoder text alignment, enabled via
the config or ``COTRAIN_T2I=1`` rather than a dedicated launcher task.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def _add_distributed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpus", type=_positive_int, default=8,
        help="GPU processes per node (default: 8)",
    )
    parser.add_argument(
        "--nnodes", type=_positive_int, default=1,
        help="Number of training nodes (default: 1)",
    )
    parser.add_argument(
        "--node-rank", "--node_rank", dest="node_rank",
        type=_non_negative_int, default=0,
        help="Rank of this node in [0, nnodes) (default: 0)",
    )
    parser.add_argument(
        "--master-addr", "--master_addr", dest="master_addr",
        default=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        help="Address of rank-0 node (default: $MASTER_ADDR or 127.0.0.1)",
    )
    parser.add_argument(
        "--master-port", "--master_port", dest="master_port",
        type=_port, default=int(os.environ.get("MASTER_PORT", "29500")),
        help="Rendezvous port on rank-0 node (default: $MASTER_PORT or 29500)",
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    codec = sub.add_parser("codec", help="train the Stage 1 DA3-GIANT codec")
    codec.add_argument("--size", type=int, choices=(64, 128), required=True)
    _add_distributed_args(codec)
    codec.add_argument("--results-dir")

    flow = sub.add_parser(
        "flow", help="train the Stage 2 flow model (i2v/t2v; optional T2I co-train)")
    flow.add_argument("--size", type=int, choices=(64, 128), required=True,
                      help="latent dim: 64 -> flow_gae64.yaml, 128 -> flow_gae128.yaml")
    _add_distributed_args(flow)
    flow.add_argument("--vae-ckpt", default=None,
                      help="Override the Flow config codec checkpoint.")
    flow.add_argument("--results-dir")
    flow.add_argument("--cotrain-t2i", action="store_true",
                      help="Interleave single-image T2I steps into the i2v loop.")
    flow.add_argument("--t2i-every-k", type=int, default=3,
                      help="With --cotrain-t2i, run one T2I step every K steps (>=2).")

    args, extra = parser.parse_known_args()
    if args.node_rank >= args.nnodes:
        parser.error(f"--node-rank must be smaller than --nnodes ({args.nnodes})")
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def main() -> int:
    args, extra = parse_args()
    env = os.environ.copy()

    if args.task == "codec":
        config = ROOT / "configs" / f"gae_{args.size}.yaml"
        trainer = ROOT / "scripts" / "train" / "train_codec.py"
        results = args.results_dir or f"results/gae-{args.size}-codec"
    else:
        config = ROOT / "configs" / f"flow_gae{args.size}.yaml"
        trainer = ROOT / "scripts" / "train" / "train_flow.py"
        results = args.results_dir or f"results/gae-{args.size}-flow"
        if args.cotrain_t2i:
            if args.t2i_every_k < 2:
                print("[train] --t2i-every-k must be >=2", file=sys.stderr)
                return 2
            env["COTRAIN_T2I"] = "1"
            env["T2I_EVERY_K"] = str(args.t2i_every_k)

    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    command = [
        "torchrun",
        f"--nnodes={args.nnodes}",
        f"--nproc_per_node={args.gpus}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(trainer),
        "--config", str(config),
        "--results-dir", results,
        *extra,
    ]
    if args.task == "flow" and args.vae_ckpt:
        command.extend(["--vae-ckpt", args.vae_ckpt])

    print("[train]", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
