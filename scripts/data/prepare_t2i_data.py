#!/usr/bin/env python3
"""Fetch / build the T2I co-training data used by GAE Stage 1.

Stage 1 codec training can co-train a single-image text-to-image branch
(``cotrain_t2i`` in configs/gae_64.yaml). It reads two raw sources through the
loaders in ``src/data`` — no bespoke format:

  * **BLIP3o-Pretrain** WebDataset tar shards (``src/data/blip3o_wds.py``)
  * **ImageNet-1k** HF Arrow, captioned by class name (``src/data/imagenet_arrow.py``)

This script prepares both under ``$GAE_DATA_ROOT`` so the config paths resolve.

Subcommands
-----------
    # 1) BLIP3o-Pretrain tar shards from the Hugging Face Hub
    python scripts/data/prepare_t2i_data.py blip3o \
        --output "$GAE_DATA_ROOT/BLIP3o" --splits long short journeydb

    # 2a) ImageNet-1k as HF Arrow (the config default; gated — `huggingface-cli login`)
    python scripts/data/prepare_t2i_data.py imagenet --mode arrow \
        --output "$GAE_DATA_ROOT/imagenet-1k"

    # 2b) ImageNet-1k packed into BLIP3o-style tar shards from a local train dir
    #     (<source>/<wnid>/*.JPEG); captions are the class names.
    python scripts/data/prepare_t2i_data.py imagenet --mode wds \
        --source /datasets/imagenet/train \
        --output "$GAE_DATA_ROOT/Blip3o_style/ImageNet-1K-T2I"

    # 3) Verify a config's cotrain_t2i paths resolve to real data
    python scripts/data/prepare_t2i_data.py verify --config configs/gae_64.yaml
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# On-disk sub-directory names, kept in sync with the loader.
from data.blip3o_wds import GLD_BLIP3O_SUBDIRS  # noqa: E402

# Default Hugging Face dataset repos for each BLIP3o split.
_BLIP3O_HF_REPOS = {
    "long": "BLIP3o/BLIP3o-Pretrain-Long-Caption",
    "short": "BLIP3o/BLIP3o-Pretrain-Short-Caption",
    "journeydb": "BLIP3o/BLIP3o-Pretrain-JourneyDB",
}


# --------------------------------------------------------------------------- #
# blip3o
# --------------------------------------------------------------------------- #
def cmd_blip3o(args) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub required (pip install huggingface-hub)", file=sys.stderr)
        return 2

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        if split not in _BLIP3O_HF_REPOS:
            print(f"[skip] unknown split {split!r} (known: {list(_BLIP3O_HF_REPOS)})")
            continue
        repo = args.repo_override.get(split, _BLIP3O_HF_REPOS[split])
        subdir = GLD_BLIP3O_SUBDIRS[split]
        dest = out_root / subdir
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[blip3o] {split}: {repo} -> {dest}")
        snapshot_download(
            repo_id=repo, repo_type="dataset", local_dir=str(dest),
            allow_patterns=["*.tar"], max_workers=args.workers,
        )
        n = len(list(dest.glob("*.tar")))
        print(f"[blip3o] {split}: {n} tar shard(s) in {dest}")
    print(f"\n[done] set cotrain_t2i.dataset.data_dir to include: {out_root}")
    return 0


# --------------------------------------------------------------------------- #
# imagenet
# --------------------------------------------------------------------------- #
def cmd_imagenet(args) -> int:
    if args.mode == "arrow":
        return _imagenet_arrow(args)
    return _imagenet_wds(args)


def _imagenet_arrow(args) -> int:
    try:
        from datasets import load_dataset
    except ImportError:
        print("error: `datasets` required for --mode arrow (pip install datasets)", file=sys.stderr)
        return 2
    cache_dir = Path(args.output) / "hf_datasets_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[imagenet] materializing ILSVRC/imagenet-1k arrow into {cache_dir}")
    print("           (gated dataset — run `huggingface-cli login` first)")
    load_dataset("ILSVRC/imagenet-1k", split=args.split, cache_dir=str(cache_dir))
    # Report the arrow directory to set as imagenet_arrow_root.
    arrow_dirs = sorted(cache_dir.glob("ILSVRC___imagenet-1k/**/"))
    hint = next((str(d) for d in arrow_dirs if list(d.glob("*.arrow"))), str(cache_dir))
    print(f"\n[done] set cotrain_t2i.dataset.imagenet_arrow_root to:\n  {hint}")
    return 0


def _tar_add(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _imagenet_wds(args) -> int:
    from data.imagenet_classes import IMAGENET_CLASSES

    source = Path(args.source) if args.source else None
    if source is None or not source.is_dir():
        print("error: --mode wds needs --source <imagenet train dir> "
              "(<source>/<wnid>/*.JPEG)", file=sys.stderr)
        return 2
    wnids = sorted(d.name for d in source.iterdir() if d.is_dir())
    if len(wnids) != 1000:
        print(f"[warn] found {len(wnids)} class folders (expected 1000); "
              "captions assume standard sorted-wnid ordering")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpeg", ".jpg", ".png"}
    samples: list[tuple[str, Path, str]] = []
    for idx, wnid in enumerate(wnids):
        name = IMAGENET_CLASSES[idx] if idx < len(IMAGENET_CLASSES) else wnid
        caption = f"a photo of a {name}" if args.caption_template else name
        for img in sorted((source / wnid).iterdir()):
            if img.suffix.lower() in exts:
                samples.append((f"imagenet_{wnid}_{img.stem}", img, caption))
    print(f"[imagenet] packing {len(samples)} images -> {out_dir} "
          f"({args.samples_per_shard}/shard)")

    total = shard = 0
    for start in range(0, len(samples), args.samples_per_shard):
        chunk = samples[start:start + args.samples_per_shard]
        final = out_dir / f"imagenet_{shard:05d}.tar"
        tmp = out_dir / f".imagenet_{shard:05d}.tar.tmp"
        with tarfile.open(tmp, "w") as tf:
            for key, img_path, caption in chunk:
                try:
                    data = img_path.read_bytes()
                except OSError:
                    continue
                _tar_add(tf, f"{key}.jpg", data)
                _tar_add(tf, f"{key}.txt", caption.encode("utf-8"))
                total += 1
        os.replace(tmp, final)
        shard += 1
        print(f"  shard {shard:05d}: {final.name} (packed={total})")
    print(f"\n[done] {total} images in {shard} shard(s) -> {out_dir}\n"
          f"       add split 'imagenet' with data_dir including {out_dir.parent}")
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def cmd_verify(args) -> int:
    from omegaconf import OmegaConf

    os.environ.setdefault("GAE_DATA_ROOT", args.data_root or "/data/gae")
    cfg = OmegaConf.load(args.config)
    node = cfg.get("cotrain_t2i")
    if node is None:
        print(f"[verify] {args.config} has no cotrain_t2i block")
        return 1
    ds = OmegaConf.to_container(node.get("dataset", {}), resolve=True) or {}
    data_dirs = ds.get("data_dir") or []
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    splits = ds.get("splits", [])
    print(f"[verify] data_dir roots: {data_dirs}")
    print(f"[verify] splits: {splits}")
    ok = True
    for split in splits:
        subdir = GLD_BLIP3O_SUBDIRS.get(split, split)
        found = 0
        where = None
        if split == "imagenet" and ds.get("imagenet_arrow_root"):
            root = Path(ds["imagenet_arrow_root"])
            arrow = len(list(root.glob("*.arrow"))) if root.is_dir() else 0
            print(f"  - {split:12s} arrow_root={'OK' if arrow else 'MISSING'} "
                  f"({arrow} .arrow) {root}")
            if not arrow:
                ok = False
            continue
        for root in data_dirs:
            d = Path(root) / subdir
            if d.is_dir():
                n = len(list(d.glob("*.tar")))
                if n:
                    found += n
                    where = d
        status = f"OK ({found} tar)" if found else "MISSING"
        print(f"  - {split:12s} {status} {where or ''}")
        if not found:
            ok = False
    print(f"\n[verify] {'all sources present' if ok else 'some sources missing'}")
    return 0 if ok else 1


def _kv(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("blip3o", help="download BLIP3o-Pretrain tar shards from HF")
    b.add_argument("--output", required=True, help="e.g. $GAE_DATA_ROOT/BLIP3o")
    b.add_argument("--splits", nargs="+", default=["long", "short", "journeydb"],
                   choices=list(_BLIP3O_HF_REPOS))
    b.add_argument("--repo", nargs="*", default=[], metavar="split=repo_id",
                   help="Override HF repo per split, e.g. long=Org/Repo")
    b.add_argument("--workers", type=int, default=8)

    im = sub.add_parser("imagenet", help="prepare ImageNet-1k (arrow or wds)")
    im.add_argument("--mode", choices=("arrow", "wds"), default="arrow")
    im.add_argument("--output", required=True)
    im.add_argument("--source", default=None,
                    help="[wds] ImageNet train dir <source>/<wnid>/*.JPEG")
    im.add_argument("--split", default="train", help="[arrow] HF split")
    im.add_argument("--samples-per-shard", type=int, default=10000)
    im.add_argument("--caption-template", action="store_true",
                    help="[wds] caption 'a photo of a <class>' instead of '<class>'")

    v = sub.add_parser("verify", help="check a config's cotrain_t2i paths")
    v.add_argument("--config", required=True)
    v.add_argument("--data-root", default=None, help="Override GAE_DATA_ROOT")

    args = ap.parse_args()
    if args.cmd == "blip3o":
        args.repo_override = _kv(args.repo)
        return cmd_blip3o(args)
    if args.cmd == "imagenet":
        return cmd_imagenet(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
