#!/usr/bin/env python3
"""Export DA3NESTED-GIANT-LARGE metric c2w+K sidecars for packed datasets.

Writes ``meta_da3_metric.json`` next to each scene's ``meta.json`` without
modifying the original poses.

Usage::

    python scripts/data/export_da3_metric_poses.py \\
        --dataset re10k_packed --split train --max-scenes 1

    python scripts/data/export_da3_metric_poses.py \\
        --datasets re10k_packed,dl3dv_packed --num-shards 32 --shard-index 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if Path("/local-ssd").is_dir():
    os.environ.setdefault("HF_HOME", "/local-ssd/hf_home")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/local-ssd/hf_cache")

from lib.da3_metric_pose import (  # noqa: E402
    CHUNK_MANIFEST_NAME,
    POSE_MODEL_ID,
    SIDEcar_NAME,
    SOURCE_META_NAME,
    PoseExportConfig,
    export_scene_poses,
    scene_export_complete,
    validate_pose_payload,
)
from lib.scene_enumerate import enumerate_scene_dirs  # noqa: E402
from _video_pack_helpers import round_robin_shard  # noqa: E402

_DATA_ROOT = Path(os.environ.get("GAE_DATA_ROOT", "/data/gae"))
DEFAULT_ROOTS = {
    "re10k_packed": _DATA_ROOT / "re10k_packed",
    "dl3dv_packed": _DATA_ROOT / "dl3dv_packed",
    "mvssynth_packed": _DATA_ROOT / "mvssynth_packed",
    "scannetpp": _DATA_ROOT / "scannetpp_preprocessed",
}

ALL_DATASETS = ("re10k_packed", "dl3dv_packed", "mvssynth_packed", "scannetpp")

DATASET_EXCLUDE_TAIL_FRAMES = {
    "re10k_packed": 32,
}


def resolve_exclude_tail_frames(dataset: str, cli_value: int | None) -> int:
    if cli_value is not None:
        return max(0, int(cli_value)) if dataset == "re10k_packed" else 0
    return DATASET_EXCLUDE_TAIL_FRAMES.get(dataset, 0)


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        out = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif args.dataset:
        out = [args.dataset]
    else:
        raise SystemExit("[fatal] specify --dataset or --datasets")
    bad = [d for d in out if d not in ALL_DATASETS]
    if bad:
        raise SystemExit(f"[fatal] unknown dataset(s): {bad}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DA3 nested metric poses (sidecar JSON).")
    p.add_argument("--dataset", type=str, default=None,
                   choices=list(ALL_DATASETS),
                   help="Single dataset (use --datasets for multiple in one model load).")
    p.add_argument("--datasets", type=str, default=None,
                   help="Comma-separated datasets; model is loaded once for all.")
    p.add_argument("--root", type=Path, default=None,
                   help="Override dataset root (single --dataset only).")
    p.add_argument("--split", type=str, default=None, choices=["train", "test"],
                   help="RE10K only; default both train+test.")
    p.add_argument("--chunk-size", type=int, default=400,
                   help="Target chunk length when num_frames > chunk_size.")
    p.add_argument("--min-chunk-frames", type=int, default=200,
                   help="Minimum frames per chunk when splitting; short tails merge.")
    p.add_argument("--exclude-tail-frames", type=int, default=None,
                   help="Drop last N source frames before export (RE10K default: 32).")
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-scenes", type=int, default=None,
                   help="Cap scenes per shard after round-robin partitioning.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--validate-only", action="store_true",
                   help="Only validate existing sidecars (no model load).")
    p.add_argument("--index-only", action="store_true",
                   help="Only build/load scene-list caches (no validation or model load).")
    p.add_argument("--model-load-stagger-sec", type=float, default=0.0,
                   help="Sleep local_gpu_index * this before loading model (avoid RAM spike).")
    p.add_argument("--local-gpu-index", type=int, default=0,
                   help="GPU index on this node (0..NUM_GPUS-1); used for load stagger.")
    p.add_argument("--rebuild-scene-list", action="store_true",
                   help="Ignore cached scene list and rescan dataset root.")
    return p.parse_args()


def load_model(device: str, stagger_sec: float = 0.0):
    import torch
    from depth_anything_3.api import DepthAnything3

    if stagger_sec > 0:
        print(f"[model] stagger sleep {stagger_sec:.1f}s before load ...", flush=True)
        time.sleep(stagger_sec)
    print(f"[model] loading {POSE_MODEL_ID} on {device} ...", flush=True)
    model = DepthAnything3.from_pretrained(POSE_MODEL_ID)
    model = model.to(device).eval()
    print("[model] ready", flush=True)
    return model


def _dataset_root(args: argparse.Namespace, dataset: str) -> Path:
    if args.root is not None:
        if len(resolve_datasets(args)) > 1:
            raise SystemExit("[fatal] --root only supported with a single --dataset")
        return args.root
    return DEFAULT_ROOTS[dataset]


def _validate_scenes(scenes: list[Path], cfg: PoseExportConfig) -> tuple[int, int]:
    ok, bad = 0, 0
    for sd in scenes:
        if not scene_export_complete(sd, cfg):
            bad += 1
            continue
        sidecar = sd / SIDEcar_NAME
        manifest = sd / CHUNK_MANIFEST_NAME
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
        elif sidecar.is_file():
            payload = json.loads(sidecar.read_text())
        else:
            bad += 1
            continue
        errs = validate_pose_payload(payload)
        if errs:
            print(f"[validate] FAIL {sd}: {errs}", flush=True)
            bad += 1
        else:
            ok += 1
    return ok, bad


def _export_scenes(
    model,
    scenes: list[Path],
    root: Path,
    cfg: PoseExportConfig,
    *,
    skip_existing: bool,
) -> tuple[int, int, int]:
    done, skipped, failed = 0, 0, 0
    for i, scene_dir in enumerate(scenes):
        if skip_existing and scene_export_complete(scene_dir, cfg):
            skipped += 1
            continue
        tag = scene_dir.relative_to(root)
        print(f"[{i+1}/{len(scenes)}] {tag}", flush=True)
        try:
            payload = export_scene_poses(model, scene_dir, cfg)
            if payload is None:
                skipped += 1
                print("  skip (missing meta/video or too short)", flush=True)
                continue
            errs = validate_pose_payload(payload)
            if errs:
                raise RuntimeError(f"validation failed: {errs}")
            if payload.get("chunks"):
                for ch in payload["chunks"]:
                    sidecar = scene_dir / ch["dir"] / ch["pose_sidecar"]
                    chunk_payload = json.loads(sidecar.read_text())
                    chunk_errs = validate_pose_payload(chunk_payload)
                    if chunk_errs:
                        raise RuntimeError(
                            f"validation failed for {sidecar}: {chunk_errs}"
                        )
                    if not chunk_payload.get("is_metric"):
                        raise RuntimeError(f"is_metric=False in {sidecar}")
            done += 1
            if payload.get("chunks"):
                print(
                    f"  wrote {CHUNK_MANIFEST_NAME}  frames={payload['num_frames']}  "
                    f"chunks={payload.get('num_chunks')}",
                    flush=True,
                )
            else:
                print(
                    f"  wrote {SIDEcar_NAME}  frames={payload['num_frames']}  "
                    f"metric={payload.get('is_metric')}",
                    flush=True,
                )
        except Exception as e:
            failed += 1
            print(f"  ERROR {e}", flush=True)
            traceback.print_exc()
    return done, skipped, failed


def _prepare_dataset_scenes(
    args: argparse.Namespace,
    dataset: str,
) -> tuple[Path, list[Path], PoseExportConfig]:
    root = _dataset_root(args, dataset)
    if not root.is_dir():
        raise SystemExit(f"[fatal] dataset root not found: {root}")

    scenes = enumerate_scene_dirs(
        dataset, root, args.split, rebuild_cache=args.rebuild_scene_list,
    )
    scenes = round_robin_shard(scenes, args.num_shards, args.shard_index)
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]

    exclude_tail = resolve_exclude_tail_frames(dataset, args.exclude_tail_frames)
    cfg = PoseExportConfig(
        chunk_size=args.chunk_size,
        min_chunk_frames=args.min_chunk_frames,
        exclude_tail_frames=exclude_tail,
        process_res=args.process_res,
    )
    print(
        f"[export] dataset={dataset} root={root} scenes={len(scenes)} "
        f"shard={args.shard_index}/{args.num_shards} exclude_tail_frames={exclude_tail}",
        flush=True,
    )
    return root, scenes, cfg


def run_dataset(
    model,
    args: argparse.Namespace,
    dataset: str,
    *,
    root: Path | None = None,
    scenes: list[Path] | None = None,
    cfg: PoseExportConfig | None = None,
) -> tuple[int, int, int]:
    if root is None or scenes is None or cfg is None:
        root, scenes, cfg = _prepare_dataset_scenes(args, dataset)

    if args.validate_only:
        ok, bad = _validate_scenes(scenes, cfg)
        print(f"[validate] {dataset} ok={ok} bad={bad}", flush=True)
        return ok, 0, bad

    return _export_scenes(model, scenes, root, cfg, skip_existing=args.skip_existing)


def main() -> None:
    args = parse_args()
    datasets = resolve_datasets(args)

    prepared: list[tuple[str, Path, list[Path], PoseExportConfig]] = []
    for dataset in datasets:
        root, scenes, cfg = _prepare_dataset_scenes(args, dataset)
        prepared.append((dataset, root, scenes, cfg))

    if args.index_only:
        total = sum(len(scenes) for _, _, scenes, _ in prepared)
        print(f"[index] total prepared scenes={total} datasets={len(prepared)}", flush=True)
        return

    if args.validate_only:
        total_ok, total_failed = 0, 0
        for dataset, root, scenes, cfg in prepared:
            ok, bad = _validate_scenes(scenes, cfg)
            print(f"[validate] {dataset} ok={ok} bad={bad}", flush=True)
            total_ok += ok
            total_failed += bad
        print(f"[validate] total ok={total_ok} bad={total_failed}", flush=True)
        return

    stagger = max(0.0, float(args.model_load_stagger_sec)) * max(0, int(args.local_gpu_index))
    model = load_model(args.device, stagger)

    total_done, total_skipped, total_failed = 0, 0, 0
    for dataset, root, scenes, cfg in prepared:
        done, skipped, failed = _export_scenes(
            model, scenes, root, cfg, skip_existing=args.skip_existing,
        )
        total_done += done
        total_skipped += skipped
        total_failed += failed

    print(
        f"[done] datasets={len(datasets)} exported={total_done} "
        f"skipped={total_skipped} failed={total_failed}",
        flush=True,
    )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
