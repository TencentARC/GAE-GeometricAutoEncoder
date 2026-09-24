#!/usr/bin/env python3
"""Render a generated GAE point cloud as a cumulative progressive video.

``eval_generation.py`` writes the prediction cloud as an ASCII PLY and keeps
the geometry required to reproduce its frame ordering in ``*_geom.npz`` and
``*_poses.npz``.  The geo-ft renderer used a binary PLY and ``*_depth.npz``;
this small adapter accepts both layouts, so it can be called immediately after
``scripts/demo/generate.py``/``run_demo.sh``.

Example::

    python scripts/demo/render_progressive_ply.py \
      results/demo/i2v/forest_lake_trail/000_pred_pointcloud.ply
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
BACKGROUND = np.array([238, 242, 239], dtype=np.uint8)
NEW_POINT = np.array([49, 91, 255], dtype=np.uint8)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("point_cloud", type=Path)
    p.add_argument("--depth", type=Path, default=None,
                   help="Depth NPZ/NPY; otherwise read gen_depth from *_geom.npz.")
    p.add_argument("--poses", type=Path, default=None,
                   help="Pose NPZ; otherwise resolve the sibling *_poses.npz.")
    p.add_argument("--geometry", type=Path, default=None,
                   help="Geometry NPZ containing gen_depth and image_size.")
    p.add_argument("--output-prefix", type=Path, default=None)
    p.add_argument("--point-stride", type=int, default=1)
    p.add_argument("--depth-edge-filter", action="store_true", default=True,
                   help="Remove points near depth discontinuities (default on).")
    p.add_argument("--depth-edge-threshold", type=float, default=0.05)
    p.add_argument("--depth-edge-dilation", type=int, default=1)
    p.add_argument("--far-percentile", type=float, default=100.0)
    p.add_argument("--render-voxel-size", type=float, default=0.001,
                   help="Visualization voxel size; default matches GLD renderer.")
    p.add_argument("--point-budget", type=int, default=0)
    p.add_argument("--require-full-resolution", action="store_true",
                   help="Fail if the source PLY uses a stride other than 1.")
    p.add_argument("--save-filtered-ply", action="store_true",
                   help="Write the filtered PLY; the source PLY is never modified.")
    p.add_argument("--no-sky-filter", dest="sky_filter", action="store_false",
                   help="Keep sky-colored points (sky filtering is on by default).")
    p.set_defaults(sky_filter=True)
    p.add_argument("--sky-blue-floor", type=float, default=95.0)
    p.add_argument("--dark-sky-floor", type=float, default=20.0)
    p.add_argument("--point-size", type=int, choices=(1, 2, 4), default=1)
    p.add_argument("--camera-back-offset", type=float, default=0.2,
                   help="Move the render camera backward in normalized scene units.")
    p.add_argument("--camera-smoothing-window", type=int, default=7)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--hold-frames", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def _header(path: Path) -> tuple[bytes, str, int]:
    raw = path.read_bytes()
    marker = b"end_header\n"
    if marker not in raw:
        marker = b"end_header\r\n"
    end = raw.index(marker) + len(marker)
    text = raw[:end].decode("ascii", errors="replace")
    line = next(x for x in text.splitlines() if x.startswith("element vertex "))
    return raw[end:], text, int(line.rsplit(" ", 1)[1])


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload, header, count = _header(path)
    if "format binary_little_endian 1.0" in header:
        vertices = np.frombuffer(payload, dtype=PLY_DTYPE, count=count)
        points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
        colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
        return points.astype(np.float32), colors.astype(np.uint8)
    if "format ascii 1.0" not in header:
        raise ValueError(f"Unsupported PLY format in {path}")
    values = np.loadtxt(__import__("io").BytesIO(payload), dtype=np.float32, max_rows=count)
    values = np.atleast_2d(values)
    if values.shape[1] < 6:
        raise ValueError(f"PLY needs x/y/z/r/g/b properties: {path}")
    return values[:, :3].astype(np.float32), np.clip(values[:, 3:6], 0, 255).astype(np.uint8)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(points, colors):
            f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                    f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")


def _stem_candidates(ply: Path, suffix: str) -> list[Path]:
    stem = ply.with_suffix("")
    out = [stem.with_name(stem.name + suffix)]
    for token in ("_pred_pointcloud_unaligned", "_pred_pointcloud", "_pointcloud"):
        if stem.name.endswith(token):
            out.append(stem.with_name(stem.name[:-len(token)] + suffix))
    return out


def resolve_sidecar(ply: Path, suffix: str, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for candidate in _stem_candidates(ply, suffix):
        if candidate.is_file():
            return candidate
    return None


def load_arrays(ply: Path, depth_path: Path | None, poses_path: Path | None,
                geometry_path: Path | None):
    poses_path = resolve_sidecar(ply, "_poses.npz", poses_path)
    geometry_path = resolve_sidecar(ply, "_geom.npz", geometry_path)
    pose_data = np.load(poses_path) if poses_path else None
    geom_data = np.load(geometry_path) if geometry_path else None
    depth_path = resolve_sidecar(ply, "_depth.npz", depth_path)
    depth = None
    if depth_path:
        loaded = np.load(depth_path)
        depth = np.asarray(loaded["depth"] if hasattr(loaded, "files") and "depth" in loaded else loaded)
    elif geom_data is not None:
        for key in ("gen_depth", "pred_depth", "depth"):
            if key in geom_data:
                depth = np.asarray(geom_data[key])
                break
    if pose_data is None:
        raise FileNotFoundError(f"Could not resolve poses for {ply}; pass --poses")
    cameras = np.asarray(pose_data["pred_c2w"] if "pred_c2w" in pose_data else pose_data["input_c2w"], dtype=np.float32)
    intrinsics = np.asarray(pose_data["pred_K"] if "pred_K" in pose_data else pose_data["input_K"], dtype=np.float32)
    if "image_size" in pose_data:
        image_size = np.asarray(pose_data["image_size"], dtype=np.int64)
    elif geom_data is not None and "image_size" in geom_data:
        image_size = np.asarray(geom_data["image_size"], dtype=np.int64)
    elif depth is not None:
        image_size = np.asarray(depth.shape[-2:], dtype=np.int64)
    else:
        raise ValueError("image_size is missing; pass a geometry/depth sidecar")
    point_size = image_size.copy()
    if geom_data is not None and "gen_ray" in geom_data:
        ray_shape = np.asarray(geom_data["gen_ray"]).shape
        if len(ray_shape) >= 4 and ray_shape[-1] == 6:
            # The released evaluator back-projects from the DA3 ray map
            # (typically 216x384), while depth/K are stored at 378x672.
            point_size = np.asarray(ray_shape[-3:-1], dtype=np.int64)
    if depth is not None:
        depth = np.squeeze(depth)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth [V,H,W], got {depth.shape}")
    return depth, cameras, intrinsics, image_size, point_size, poses_path, geometry_path


def infer_stride(n: int, views: int, size: np.ndarray, requested: int) -> int:
    h, w = map(int, size)
    for stride in (requested, 1, 2, 4, 8, 16):
        if stride > 0 and views * ((h + stride - 1) // stride) * ((w + stride - 1) // stride) == n:
            return stride
    raise ValueError(f"Cannot infer point stride: {n} points, {views} views, {tuple(size)}")


def edge_mask(depth: np.ndarray, point_size: np.ndarray, stride: int,
              threshold: float, dilation: int) -> np.ndarray:
    masks = []
    kernel = np.ones((3, 3), np.uint8)
    target_h, target_w = map(int, point_size)
    for d in depth:
        d = np.asarray(d, np.float32)
        if d.shape != (target_h, target_w):
            d = cv2.resize(d, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(d) & (d > 1e-6)
        logd = np.log(np.maximum(d, 1e-6))
        gx = np.abs(cv2.Sobel(logd, cv2.CV_32F, 1, 0, ksize=3)) / 8.0
        gy = np.abs(cv2.Sobel(logd, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        edge = (np.maximum(gx, gy) > threshold).astype(np.uint8)
        if dilation:
            edge = cv2.dilate(edge, kernel, iterations=dilation)
        masks.append((valid & ~edge.astype(bool))[::stride, ::stride].reshape(-1))
    return np.concatenate(masks)


def sky_keep_mask(colors: np.ndarray, views: int, point_size: np.ndarray,
                  stride: int, blue_floor: float, dark_floor: float) -> np.ndarray:
    """Remove sky-colored connected components touching the sampled top edge."""
    h, w = map(int, point_size)
    sh, sw = (h + stride - 1) // stride, (w + stride - 1) // stride
    frames = colors.reshape(views, sh, sw, 3).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    keeps = []
    for rgb in frames:
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        blue_sky = (blue > blue_floor) & (blue > red + 8.0) & (blue > green - 12.0)
        blue_sky |= ((blue > dark_floor) & (blue > 1.18 * np.maximum(red, 5.0))
                     & (blue > green + 4.0))
        maximum = rgb.max(axis=2)
        saturation = (maximum - rgb.min(axis=2)) / np.maximum(maximum, 1.0)
        bright_cloud = (rgb.mean(axis=2) > 155.0) & (saturation < 0.22)
        candidate = cv2.morphologyEx(
            (blue_sky | bright_cloud).astype(np.uint8),
            cv2.MORPH_CLOSE, kernel, iterations=2)
        _, labels = cv2.connectedComponents(candidate, connectivity=8)
        top_labels = np.unique(labels[0]); top_labels = top_labels[top_labels != 0]
        sky = cv2.dilate(np.isin(labels, top_labels).astype(np.uint8), kernel, iterations=1)
        keeps.append((~sky.astype(bool)).reshape(-1))
    return np.concatenate(keeps)


def normalize_cameras(cameras: np.ndarray, center: np.ndarray, scale: float, back: float):
    flip = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    c = cameras.copy()
    c[:, :3, :3] = flip @ c[:, :3, :3]
    c[:, :3, 3] = ((c[:, :3, 3] - center) / scale) @ flip
    c[:, :3, 3] -= back * c[:, :3, 2]
    return c


def smooth_cameras(cameras: np.ndarray, window: int) -> np.ndarray:
    """Gaussian-smooth render-only camera centers and rotations."""
    if window <= 1:
        return cameras.copy()
    if window % 2 == 0:
        raise ValueError("--camera-smoothing-window must be odd")
    radius = window // 2
    sigma = max(window / 3.0, 1.0)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    out = cameras.copy()
    for i in range(len(cameras)):
        indices = np.clip(i + offsets, 0, len(cameras) - 1)
        out[i, :3, 3] = np.sum(
            cameras[indices, :3, 3] * weights[:, None], axis=0)
        mean_r = np.sum(
            cameras[indices, :3, :3] * weights[:, None, None], axis=0)
        u, _, vh = np.linalg.svd(mean_r)
        r = u @ vh
        if np.sign(np.linalg.det(r)) != np.sign(np.linalg.det(cameras[i, :3, :3])):
            u[:, -1] *= -1
            r = u @ vh
        out[i, :3, :3] = r
    return out


def smooth_intrinsics(intrinsics: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return intrinsics.copy()
    radius = window // 2
    sigma = max(window / 3.0, 1.0)
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    out = intrinsics.copy()
    for i in range(len(intrinsics)):
        indices = np.clip(i + offsets, 0, len(intrinsics) - 1)
        out[i] = np.sum(intrinsics[indices] * weights[:, None, None], axis=0)
    return out


def voxel_first(points: np.ndarray, colors: np.ndarray, frames: np.ndarray,
                voxel: float, budget: int, seed: int):
    center = np.median(points, axis=0)
    scale = max(float(np.percentile(np.linalg.norm(points - center, axis=1), 97.0)), 1e-6)
    q = (points - center) / scale
    q[:, 1] *= -1
    if voxel <= 0:
        return q.astype(np.float32), colors, frames, center, scale
    keys = np.floor((q - q.min(0)) / max(voxel, 1e-6)).astype(np.int32)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    if budget > 0 and len(first) > budget:
        first = np.sort(np.random.default_rng(seed).choice(first, budget, replace=False))
    return q[first].astype(np.float32), colors[first], frames[first], center, scale


def project(frame, points, colors, camera, K, source_size, point_size):
    h, w = frame.shape[:2]
    cam = (points - camera[:3, 3]) @ camera[:3, :3]
    depth = cam[:, 2]
    valid = depth > 1e-5
    cam, depth, src = cam[valid], depth[valid], np.flatnonzero(valid)
    sh, sw = map(int, source_size)
    px = np.rint(float(K[0, 0]) * w / sw * cam[:, 0] / depth + float(K[0, 2]) * w / sw).astype(np.int32)
    py = np.rint(float(K[1, 1]) * h / sh * cam[:, 1] / depth + float(K[1, 2]) * h / sh).astype(np.int32)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py, depth, src = px[inside], py[inside], depth[inside], src[inside]
    order = np.argsort(depth, kind="stable")
    offsets = ((0, 0),) if point_size == 1 else ((0, 0), (1, 0), (0, 1), (1, 1))
    flat = frame.reshape(-1, 3)
    for dx, dy in offsets:
        xx, yy = px + dx, py + dy
        ok = (xx < w) & (yy < h)
        indices = np.flatnonzero(ok)
        if not len(indices):
            continue
        ordered = indices[np.argsort(depth[indices], kind="stable")]
        pix = yy[ordered] * w + xx[ordered]
        _, first = np.unique(pix, return_index=True)
        flat[pix[first]] = colors[src[ordered[first]]]


def encode(frames, path: Path, fps: int, hold: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir="/tmp") as temp:
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp.name], stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for frame in frames:
            for _ in range(max(1, hold)):
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding progressive point-cloud video")
        path.write_bytes(Path(temp.name).read_bytes())


def main() -> None:
    a = args()
    points, colors = read_ply(a.point_cloud)
    source_point_count = int(len(points))
    depth, cameras, K, image_size, point_size, poses_path, geometry_path = load_arrays(
        a.point_cloud, a.depth, a.poses, a.geometry)
    depth_sidecar = a.depth or resolve_sidecar(a.point_cloud, "_depth.npz", None)
    views = min(len(cameras), len(depth) if depth is not None else len(cameras))
    cameras, K = cameras[:views], K[:views]
    stride = infer_stride(len(points), views, point_size, a.point_stride)
    if a.require_full_resolution and stride != 1:
        raise ValueError(
            f"Full-resolution render requested, but source PLY uses point stride {stride}"
        )
    points_per_view = len(points) // views
    if points_per_view * views != len(points):
        raise ValueError(f"PLY point count {len(points)} is not divisible by {views} views")
    frame_index = np.repeat(np.arange(views, dtype=np.int32), points_per_view)
    keep = np.isfinite(points).all(1)
    sky_removed = 0
    if a.sky_filter:
        sky_keep = sky_keep_mask(colors, views, point_size, stride,
                                 a.sky_blue_floor, a.dark_sky_floor)
        sky_removed = int((~sky_keep).sum())
        keep &= sky_keep
    if a.depth_edge_filter:
        if depth is None:
            raise ValueError("--depth-edge-filter requires depth in --depth or *_geom.npz")
        keep &= edge_mask(depth[:views], point_size, stride,
                          a.depth_edge_threshold, a.depth_edge_dilation)
    points, colors, frame_index = points[keep], colors[keep], frame_index[keep]
    if len(points) == 0:
        raise ValueError("all points were filtered")
    if a.far_percentile < 100.0:
        far = np.percentile(
            np.linalg.norm(points - np.median(points, 0), axis=1),
            a.far_percentile,
        )
        keep = np.linalg.norm(points - np.median(points, 0), axis=1) <= far
        points, colors, frame_index = points[keep], colors[keep], frame_index[keep]
    prefix = a.output_prefix or a.point_cloud.with_suffix("").with_name(a.point_cloud.stem + "_progressive")
    filtered = prefix.with_suffix(".ply")
    if a.save_filtered_ply:
        write_ply(filtered, points, colors)
    render_points, render_colors, render_frames, center, scale = voxel_first(
        points, colors, frame_index, a.render_voxel_size, a.point_budget, a.seed)
    render_cameras = normalize_cameras(cameras, center, scale, a.camera_back_offset)
    # The recovered DA3 poses contain small frame-to-frame estimation noise.
    # Smooth only the render path; source frame ordering and point ownership
    # remain unchanged for the progressive accumulation.
    render_cameras = smooth_cameras(render_cameras, a.camera_smoothing_window)
    K = smooth_intrinsics(K, a.camera_smoothing_window)
    frames, cumulative = [], []
    for i in range(views):
        observed = render_frames <= i
        fresh = render_frames == i
        frame = np.empty((a.height, a.width, 3), np.uint8); frame[:] = BACKGROUND
        cc = render_colors[observed].copy(); cc[fresh[observed]] = NEW_POINT
        project(frame, render_points[observed], cc, render_cameras[i], K[i], image_size, a.point_size)
        frames.append(frame); cumulative.append(int(observed.sum()))
        print(f"frame {i + 1:02d}/{views}: {int(observed.sum()):,} observed, +{int(fresh.sum()):,} new", flush=True)
    video = prefix.with_suffix(".mp4"); poster = prefix.with_suffix(".jpg"); report = prefix.with_suffix(".json")
    encode(frames, video, a.fps, a.hold_frames)
    cv2.imwrite(str(poster), frames[-1][:, :, ::-1])
    report.write_text(json.dumps({"source": str(a.point_cloud), "depth": str(depth_sidecar or geometry_path or ""), "poses": str(poses_path or ""), "views": views, "point_stride": stride, "input_points": source_point_count, "filtered_points": int(len(points)), "sky_filter": bool(a.sky_filter), "removed_sky_points": sky_removed, "rendered_points": int(len(render_points)), "depth_edge_filter": bool(a.depth_edge_filter), "depth_edge_threshold": a.depth_edge_threshold, "render_voxel_size": a.render_voxel_size, "point_budget": a.point_budget, "point_size": a.point_size, "camera_back_offset": a.camera_back_offset, "camera_smoothing_window": a.camera_smoothing_window, "fps": a.fps, "hold_frames": a.hold_frames, "cumulative_rendered_points": cumulative}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {video}\nwrote {poster}\nwrote {report}")
    if a.save_filtered_ply:
        print(f"wrote {filtered}")


if __name__ == "__main__":
    main()
