"""DA3NESTED-GIANT-LARGE metric pose export helpers.

Videos with ``num_frames > chunk_size`` (default 400) are split into physical
chunk subdirs (``chunk_000/video.mp4``, ...). Each chunk is fed to DA3 in a
**single** forward pass; poses are written to ``chunk_XXX/meta_da3_metric.json``.

Short videos (``num_frames <= chunk_size``) keep the original ``video.mp4`` and
write ``meta_da3_metric.json`` next to ``meta.json``.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from _video_pack_helpers import (  # noqa: E402
    _move_atomic,
    encode_mp4_ffmpeg,
    write_meta_json,
)

POSE_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
SIDEcar_NAME = "meta_da3_metric.json"
SOURCE_META_NAME = "meta.json"
CHUNK_MANIFEST_NAME = "chunks_manifest.json"
CHUNK_VIDEO_NAME = "video.mp4"
CHUNK_META_NAME = "meta.json"


@dataclass
class PoseExportConfig:
    chunk_size: int = 400
    min_chunk_frames: int = 200
    exclude_tail_frames: int = 0
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    video_fps: int = 15
    video_crf: int = 18


def prepare_export_frames(
    meta: dict[str, Any], cfg: PoseExportConfig,
) -> tuple[list[dict[str, Any]], int, int] | None:
    """Truncate tail frames; return (frames_meta, export_num_frames, source_num_frames)."""
    frames_meta = meta.get("frames") or []
    source_num_frames = int(meta.get("num_frames", len(frames_meta)))
    if source_num_frames < 2 or len(frames_meta) < 2:
        return None
    source_num_frames = min(source_num_frames, len(frames_meta))
    drop = max(0, int(cfg.exclude_tail_frames))
    export_num_frames = source_num_frames - drop
    if export_num_frames < 2:
        return None
    return frames_meta[:export_num_frames], export_num_frames, source_num_frames


def _export_meta_fields(
    cfg: PoseExportConfig, source_num_frames: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {"source_num_frames": int(source_num_frames)}
    if cfg.exclude_tail_frames > 0:
        out["exclude_tail_frames"] = int(cfg.exclude_tail_frames)
    return out


def _payload_matches_cfg(payload: dict[str, Any], cfg: PoseExportConfig) -> bool:
    if int(payload.get("exclude_tail_frames", 0)) != int(cfg.exclude_tail_frames):
        return False
    if int(payload.get("chunk_size", cfg.chunk_size)) != int(cfg.chunk_size):
        return False
    if int(payload.get("min_chunk_frames", cfg.min_chunk_frames)) != int(cfg.min_chunk_frames):
        return False
    return True


def _export_cfg_fields(cfg: PoseExportConfig) -> dict[str, int]:
    return {
        "chunk_size": int(cfg.chunk_size),
        "min_chunk_frames": int(cfg.min_chunk_frames),
    }


def _cleanup_stale_export_artifacts(
    scene_dir: Path,
    ranges: list[tuple[int, int]],
    *,
    out_name: str = SIDEcar_NAME,
) -> None:
    """Remove chunk dirs / sidecars left from prior exports with different layout."""
    scene_dir = Path(scene_dir)
    num_chunks = len(ranges)

    for path in sorted(scene_dir.glob("chunk_*")):
        if not path.is_dir():
            continue
        try:
            chunk_index = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if chunk_index >= num_chunks:
            shutil.rmtree(path)

    if num_chunks == 1:
        manifest = scene_dir / CHUNK_MANIFEST_NAME
        if manifest.is_file():
            manifest.unlink()
        for path in sorted(scene_dir.glob("chunk_*")):
            if path.is_dir():
                shutil.rmtree(path)
    else:
        root_sidecar = scene_dir / out_name
        if root_sidecar.is_file():
            root_sidecar.unlink()


def chunk_dir_name(chunk_index: int) -> str:
    return f"chunk_{chunk_index:03d}"


def load_source_meta(meta_path: Path) -> dict[str, Any]:
    with open(meta_path) as f:
        return json.load(f)


def meta_resolution(meta: dict[str, Any]) -> tuple[int, int]:
    h = int(meta.get("height") or 0)
    w = int(meta.get("width") or 0)
    if h <= 0 or w <= 0:
        res = int(meta.get("rgb_resolution") or 0)
        if res > 0:
            if h <= 0:
                h = res
            if w <= 0:
                w = res
    if h <= 0 or w <= 0:
        raise ValueError(f"meta missing height/width: {sorted(meta.keys())}")
    return h, w


def frame_resolution(images_rgb: list[np.ndarray]) -> tuple[int, int]:
    """Return (H, W) shared by all decoded frames."""
    if not images_rgb:
        raise ValueError("frame_resolution: no images")
    h, w = (int(images_rgb[0].shape[0]), int(images_rgb[0].shape[1]))
    for i, im in enumerate(images_rgb[1:], start=1):
        ih, iw = int(im.shape[0]), int(im.shape[1])
        if (ih, iw) != (h, w):
            raise RuntimeError(
                f"frame {i} size {(ih, iw)} != frame 0 size {(h, w)}"
            )
    return h, w


def resolve_export_resolution(
    meta: dict[str, Any], images_rgb: list[np.ndarray],
) -> tuple[int, int]:
    """Use decoded frame size for K/payload; warn if meta disagrees."""
    actual_h, actual_w = frame_resolution(images_rgb)
    try:
        meta_h, meta_w = meta_resolution(meta)
    except ValueError:
        return actual_h, actual_w
    if (actual_h, actual_w) != (meta_h, meta_w):
        print(
            f"[warn] meta resolution {(meta_h, meta_w)} != "
            f"video frames {(actual_h, actual_w)}; using video size for K",
            flush=True,
        )
    return actual_h, actual_w


def chunk_ranges(
    num_frames: int,
    chunk_size: int,
    *,
    min_chunk_frames: int = 200,
) -> list[tuple[int, int]]:
    """Non-overlapping [start, end) ranges. One DA3 forward per range.

    When splitting into multiple chunks, each chunk has at least
    ``min_chunk_frames`` (default 200). A short tail is merged into the
    previous chunk instead of becoming its own chunk.
    """
    if num_frames <= 0:
        return []
    if num_frames <= chunk_size:
        return [(0, num_frames)]
    out: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        remainder = num_frames - end
        if 0 < remainder < min_chunk_frames:
            end = num_frames
        out.append((start, end))
        start = end
    if len(out) > 1:
        for start, end in out:
            if end - start < min_chunk_frames:
                raise ValueError(
                    f"chunk [{start}, {end}) has {end - start} frames "
                    f"< min_chunk_frames={min_chunk_frames}"
                )
    return out


def scene_needs_chunk_dirs(num_frames: int, chunk_size: int) -> bool:
    return num_frames > chunk_size


def decode_video_frames_rgb(
    video_path: Path, frame_indices: list[int],
) -> list[np.ndarray]:
    wanted = sorted(set(int(i) for i in frame_indices))
    if not wanted:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2.VideoCapture failed: {video_path}")
    out_map: dict[int, np.ndarray] = {}
    try:
        cur, wi = 0, 0
        target = wanted[wi]
        while wi < len(wanted):
            if cur < target:
                if not cap.grab():
                    raise IOError(
                        f"video ended before frame {target} (cur={cur}) in {video_path}"
                    )
                cur += 1
                continue
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"cv2 read failed at frame {target} in {video_path}")
            out_map[target] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cur += 1
            wi += 1
            if wi < len(wanted):
                target = wanted[wi]
    finally:
        cap.release()
    return [out_map[i] for i in frame_indices]


def decode_video_frame_range_rgb(
    video_path: Path, start: int, end: int,
) -> list[np.ndarray]:
    """Decode a contiguous [start, end) frame range without rescanning prefixes."""
    start = int(start)
    end = int(end)
    if end <= start:
        return []
    if start <= 0:
        return decode_video_frames_rgb(video_path, list(range(start, end)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cv2.VideoCapture failed: {video_path}")
    try:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
            return decode_video_frames_rgb(video_path, list(range(start, end)))

        # Some backends seek to the nearest earlier keyframe. If that happens,
        # skip forward; if they overshoot, fall back to the exact sequential path.
        pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if pos > start:
            return decode_video_frames_rgb(video_path, list(range(start, end)))
        while pos < start:
            if not cap.grab():
                raise IOError(
                    f"video ended before frame {start} (cur={pos}) in {video_path}"
                )
            pos += 1

        images: list[np.ndarray] = []
        for frame_idx in range(start, end):
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"cv2 read failed at frame {frame_idx} in {video_path}")
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return images
    finally:
        cap.release()


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    m = np.asarray(w2c, dtype=np.float64)
    if m.shape == (3, 4):
        m4 = np.eye(4, dtype=np.float64)
        m4[:3, :4] = m
        m = m4
    return np.linalg.inv(m).astype(np.float32)


def scale_intrinsics_to_original(
    K: np.ndarray, proc_h: int, proc_w: int, orig_h: int, orig_w: int,
) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32).copy()
    sx = float(orig_w) / float(max(proc_w, 1))
    sy = float(orig_h) / float(max(proc_h, 1))
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def _prediction_to_c2w_K(
    prediction, orig_h: int, orig_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 prediction missing extrinsics/intrinsics")
    w2c = np.asarray(prediction.extrinsics, dtype=np.float32)
    K = np.asarray(prediction.intrinsics, dtype=np.float32)
    c2w = np.stack([w2c_to_c2w(e) for e in w2c])
    if prediction.processed_images is not None:
        proc_h = int(prediction.processed_images.shape[1])
        proc_w = int(prediction.processed_images.shape[2])
    else:
        proc_h, proc_w = orig_h, orig_w
    K_out = np.stack([
        scale_intrinsics_to_original(K[i], proc_h, proc_w, orig_h, orig_w)
        for i in range(len(K))
    ])
    return c2w, K_out


def run_da3_inference(
    model, images_rgb: list[np.ndarray], cfg: PoseExportConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Single DA3 forward over all frames in one chunk."""
    if len(images_rgb) < 2:
        raise ValueError(f"DA3 needs >=2 views, got {len(images_rgb)}")
    pil_images = [Image.fromarray(im) for im in images_rgb]
    pred = model.inference(
        pil_images,
        extrinsics=None,
        intrinsics=None,
        align_to_input_ext_scale=True,
        process_res=cfg.process_res,
        process_res_method=cfg.process_res_method,
    )
    oh, ow = images_rgb[0].shape[:2]
    c2w, K = _prediction_to_c2w_K(pred, oh, ow)
    is_metric = int(getattr(pred, "is_metric", 0) or 0)
    return c2w, K, is_metric


def build_chunk_meta(
    source_meta: dict[str, Any],
    frames_meta: list[dict[str, Any]],
    *,
    chunk_index: int,
    total_chunks: int,
    start_frame: int,
    end_frame: int,
    source_meta_name: str = SOURCE_META_NAME,
    cfg: PoseExportConfig | None = None,
    source_num_frames: int | None = None,
    export_height: int | None = None,
    export_width: int | None = None,
) -> dict[str, Any]:
    chunk_frames = frames_meta[start_frame:end_frame]
    if export_height is not None and export_width is not None:
        orig_h, orig_w = int(export_height), int(export_width)
    else:
        orig_h, orig_w = meta_resolution(source_meta)
    out: dict[str, Any] = {
        "source_meta": source_meta_name,
        "chunk_index": int(chunk_index),
        "total_chunks": int(total_chunks),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "num_frames": int(end_frame - start_frame),
        "height": orig_h,
        "width": orig_w,
        "fps": int(source_meta.get("fps", 15)),
        "frames": [],
    }
    for local_i, fr in enumerate(chunk_frames):
        entry = dict(fr)
        entry["frame_idx"] = int(local_i)
        entry["global_frame_idx"] = int(start_frame + local_i)
        out["frames"].append(entry)
    for key in ("scene_id", "clip_id", "rgb_resolution", "caption"):
        if key in source_meta:
            out[key] = source_meta[key]
    if cfg is not None and source_num_frames is not None:
        out.update(_export_meta_fields(cfg, source_num_frames))
    return out


def build_pose_payload(
    *,
    source_meta: dict[str, Any],
    frames_meta: list[dict[str, Any]],
    c2w: np.ndarray,
    K: np.ndarray,
    is_metric_flag: int,
    cfg: PoseExportConfig,
    num_frames: int,
    orig_h: int,
    orig_w: int,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    start_frame: int = 0,
    source_meta_name: str = SOURCE_META_NAME,
    chunk_video_name: str | None = None,
    source_num_frames: int | None = None,
) -> dict[str, Any]:
    out_frames: list[dict[str, Any]] = []
    for i in range(num_frames):
        fr = frames_meta[start_frame + i]
        entry: dict[str, Any] = {
            "name": str(fr.get("name", f"{i:06d}")),
            "frame_idx": int(i),
            "global_frame_idx": int(start_frame + i),
            "c2w": c2w[i].tolist(),
            "K": K[i].tolist(),
        }
        if chunk_index is not None:
            entry["chunk_index"] = int(chunk_index)
        if "original_idx" in fr:
            entry["original_idx"] = fr["original_idx"]
        out_frames.append(entry)

    payload: dict[str, Any] = {
        "source_meta": source_meta_name,
        "pose_model": POSE_MODEL_ID,
        "pose_convention": "opencv_c2w",
        "is_metric": bool(is_metric_flag),
        "process_res": cfg.process_res,
        "num_frames": num_frames,
        "height": orig_h,
        "width": orig_w,
        "frames": out_frames,
    }
    payload.update(_export_cfg_fields(cfg))
    if chunk_index is not None:
        payload["chunk_index"] = int(chunk_index)
        payload["total_chunks"] = int(total_chunks or 1)
        payload["start_frame"] = int(start_frame)
        payload["end_frame"] = int(start_frame + num_frames)
        if chunk_video_name:
            payload["chunk_video"] = chunk_video_name
    if source_meta.get("scene_id"):
        payload["scene_id"] = source_meta["scene_id"]
    if source_meta.get("clip_id"):
        payload["clip_id"] = source_meta["clip_id"]
    if source_num_frames is not None:
        payload.update(_export_meta_fields(cfg, source_num_frames))
    return payload


def _stage_write_json(payload: dict[str, Any], out_path: Path) -> None:
    stage_root = Path(os.environ.get(
        "DA3_POSE_STAGE_DIR",
        "/local-ssd/_da3_pose_stage" if Path("/local-ssd").is_dir() else "/tmp/_da3_pose_stage",
    ))
    stage_root.mkdir(parents=True, exist_ok=True)
    # Unique per process AND per thread/call: out_path.name is the same constant
    # for every chunk, so keying only on PID collides across concurrent consumer
    # threads in one process (one thread's cp races another's move of the shared
    # stage file). urandom guarantees uniqueness regardless of thread count.
    stage_path = stage_root / (
        f"{out_path.name}.{os.getpid()}.{threading.get_ident()}."
        f"{os.urandom(4).hex()}.json"
    )
    write_meta_json(payload, stage_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _move_atomic(stage_path, out_path)


def write_chunk_video(
    images_rgb: list[np.ndarray],
    out_path: Path,
    cfg: PoseExportConfig,
) -> None:
    frames_bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images_rgb]
    encode_mp4_ffmpeg(
        frames_bgr,
        out_path,
        fps=cfg.video_fps,
        crf=cfg.video_crf,
    )


def scene_export_complete(scene_dir: Path, cfg: PoseExportConfig) -> bool:
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / SOURCE_META_NAME
    if not meta_path.is_file():
        return False
    meta = load_source_meta(meta_path)
    prepared = prepare_export_frames(meta, cfg)
    if prepared is None:
        return False
    _, num_frames, _ = prepared
    ranges = chunk_ranges(
        num_frames, cfg.chunk_size, min_chunk_frames=cfg.min_chunk_frames,
    )

    if len(ranges) == 1:
        sidecar = scene_dir / SIDEcar_NAME
        if not sidecar.is_file():
            return False
        with open(sidecar) as f:
            payload = json.load(f)
        if int(payload.get("num_frames", 0)) != num_frames:
            return False
        return _payload_matches_cfg(payload, cfg)

    for ci, (start, end) in enumerate(ranges):
        chunk_dir = scene_dir / chunk_dir_name(ci)
        if not (chunk_dir / CHUNK_VIDEO_NAME).is_file():
            return False
        if not (chunk_dir / CHUNK_META_NAME).is_file():
            return False
        if not (chunk_dir / SIDEcar_NAME).is_file():
            return False
    manifest_path = scene_dir / CHUNK_MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    with open(manifest_path) as f:
        payload = json.load(f)
    if int(payload.get("num_frames", 0)) != num_frames:
        return False
    if int(payload.get("num_chunks", 0)) != len(ranges):
        return False
    return _payload_matches_cfg(payload, cfg)


def export_scene_poses(
    model,
    scene_dir: Path,
    cfg: PoseExportConfig,
    *,
    video_name: str = "video.mp4",
    meta_name: str = SOURCE_META_NAME,
    out_name: str = SIDEcar_NAME,
) -> dict[str, Any] | None:
    scene_dir = Path(scene_dir)
    meta_path = scene_dir / meta_name
    video_path = scene_dir / video_name
    if not meta_path.is_file() or not video_path.is_file():
        return None

    meta = load_source_meta(meta_path)
    prepared = prepare_export_frames(meta, cfg)
    if prepared is None:
        return None
    frames_meta, num_frames, source_num_frames = prepared

    video_fps = int(meta.get("fps", cfg.video_fps))
    ranges = chunk_ranges(
        num_frames, cfg.chunk_size, min_chunk_frames=cfg.min_chunk_frames,
    )
    _cleanup_stale_export_artifacts(scene_dir, ranges, out_name=out_name)
    encode_cfg = PoseExportConfig(
        chunk_size=cfg.chunk_size,
        min_chunk_frames=cfg.min_chunk_frames,
        exclude_tail_frames=cfg.exclude_tail_frames,
        process_res=cfg.process_res,
        process_res_method=cfg.process_res_method,
        video_fps=video_fps,
        video_crf=cfg.video_crf,
    )

    if len(ranges) == 1:
        images = decode_video_frame_range_rgb(video_path, 0, num_frames)
        orig_h, orig_w = resolve_export_resolution(meta, images)
        c2w, K, is_metric_flag = run_da3_inference(model, images, encode_cfg)
        payload = build_pose_payload(
            source_meta=meta,
            frames_meta=frames_meta,
            c2w=c2w,
            K=K,
            is_metric_flag=is_metric_flag,
            cfg=cfg,
            num_frames=num_frames,
            orig_h=orig_h,
            orig_w=orig_w,
            source_meta_name=meta_name,
            source_num_frames=source_num_frames,
        )
        _stage_write_json(payload, scene_dir / out_name)
        return payload

    total_chunks = len(ranges)
    chunk_summaries: list[dict[str, Any]] = []
    for ci, (start, end) in enumerate(ranges):
        chunk_dir = scene_dir / chunk_dir_name(ci)
        chunk_video_path = chunk_dir / CHUNK_VIDEO_NAME
        chunk_meta_path = chunk_dir / CHUNK_META_NAME
        chunk_sidecar_path = chunk_dir / out_name

        images = decode_video_frame_range_rgb(video_path, start, end)
        if len(images) != end - start:
            raise RuntimeError(
                f"decoded {len(images)} frames, expected {end - start} "
                f"for chunk {ci} in {scene_dir}"
            )

        if ci == 0:
            orig_h, orig_w = resolve_export_resolution(meta, images)
        else:
            ch, cw = frame_resolution(images)
            if (ch, cw) != (orig_h, orig_w):
                raise RuntimeError(
                    f"chunk {ci} resolution {(ch, cw)} != {(orig_h, orig_w)} in {scene_dir}"
                )

        chunk_dir.mkdir(parents=True, exist_ok=True)
        write_chunk_video(images, chunk_video_path, encode_cfg)

        chunk_meta = build_chunk_meta(
            meta,
            frames_meta,
            chunk_index=ci,
            total_chunks=total_chunks,
            start_frame=start,
            end_frame=end,
            source_meta_name=meta_name,
            cfg=cfg,
            source_num_frames=source_num_frames,
            export_height=orig_h,
            export_width=orig_w,
        )
        chunk_meta.update(_export_cfg_fields(cfg))
        _stage_write_json(chunk_meta, chunk_meta_path)

        c2w, K, is_metric_flag = run_da3_inference(model, images, encode_cfg)
        chunk_payload = build_pose_payload(
            source_meta=meta,
            frames_meta=frames_meta,
            c2w=c2w,
            K=K,
            is_metric_flag=is_metric_flag,
            cfg=cfg,
            num_frames=end - start,
            orig_h=orig_h,
            orig_w=orig_w,
            chunk_index=ci,
            total_chunks=total_chunks,
            start_frame=start,
            source_meta_name=meta_name,
            chunk_video_name=CHUNK_VIDEO_NAME,
            source_num_frames=source_num_frames,
        )
        _stage_write_json(chunk_payload, chunk_sidecar_path)

        chunk_summaries.append({
            "chunk_index": ci,
            "dir": chunk_dir_name(ci),
            "video": CHUNK_VIDEO_NAME,
            "meta": CHUNK_META_NAME,
            "pose_sidecar": out_name,
            "start_frame": start,
            "end_frame": end,
            "num_frames": end - start,
            "is_metric": bool(is_metric_flag),
        })

    manifest: dict[str, Any] = {
        "source_meta": meta_name,
        "source_video": video_name,
        "pose_model": POSE_MODEL_ID,
        "chunk_size": cfg.chunk_size,
        "min_chunk_frames": cfg.min_chunk_frames,
        "num_frames": num_frames,
        "num_chunks": total_chunks,
        "chunks": chunk_summaries,
    }
    manifest.update(_export_meta_fields(cfg, source_num_frames))
    if meta.get("scene_id"):
        manifest["scene_id"] = meta["scene_id"]
    if meta.get("clip_id"):
        manifest["clip_id"] = meta["clip_id"]
    _stage_write_json(manifest, scene_dir / CHUNK_MANIFEST_NAME)
    return manifest


def validate_pose_payload(payload: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    frames = payload.get("frames") or []
    nf = int(payload.get("num_frames", 0))
    if "frames" in payload and len(frames) != nf:
        errs.append(f"len(frames)={len(frames)} != num_frames={nf}")
    for i, fr in enumerate(frames[: min(5, len(frames))]):
        c2w = np.asarray(fr["c2w"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            errs.append(f"frame {i} bad c2w")
            continue
        if abs(np.linalg.det(c2w[:3, :3]) - 1.0) > 0.05:
            errs.append(f"frame {i} det(R) off")
        K = np.asarray(fr["K"], dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            errs.append(f"frame {i} bad K")
    chunks = payload.get("chunks") or []
    if chunks and not payload.get("num_chunks"):
        errs.append("chunks present but num_chunks missing")
    if errs:
        return errs
    if not payload.get("frames") and chunks:
        return errs
    h = int(payload.get("height", 0))
    w = int(payload.get("width", 0))
    if frames and h > 0 and w > 0:
        K0 = np.asarray(frames[0]["K"], dtype=np.float64)
        if K0[0, 2] > w * 1.25 or K0[1, 2] > h * 1.25:
            errs.append("K principal point far outside image bounds")
        if K0[0, 0] <= 0 or K0[1, 1] <= 0:
            errs.append("K has non-positive focal length")
    if "is_metric" in payload and not payload.get("is_metric"):
        errs.append("is_metric=False (unexpected for DA3NESTED)")
    return errs
