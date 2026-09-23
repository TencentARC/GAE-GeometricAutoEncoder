"""Shared lightweight camera-trajectory utilities for the demo and evaluator."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def path_length(poses: np.ndarray) -> float:
    pos = np.asarray(poses, dtype=np.float64)[:, :3, 3]
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def trajectory_extent(poses: np.ndarray) -> float:
    """Spatial diameter of camera centers, used as the metric pose scale."""
    pos = np.asarray(poses, dtype=np.float64)[:, :3, 3]
    if len(pos) < 2:
        return 0.0
    # N is at most 81 for the public demo; the exact diameter is more robust
    # than path length for comparing straight and turning paths.
    distances = pos[:, None, :] - pos[None, :, :]
    return float(np.linalg.norm(distances, axis=-1).max())


def reference_forward_sign(poses: np.ndarray) -> float:
    """Return the sign matching the reference path's viewing direction."""
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return 1.0
    delta = poses[-1, :3, 3] - poses[0, :3, 3]
    view_forward = -poses[0, :3, 2]
    score = float(np.dot(delta, view_forward))
    return -1.0 if score < 0.0 else 1.0


def load_reference_poses(path: str | Path, n: int) -> np.ndarray:
    data = np.load(path)
    if "c2w" not in data:
        raise ValueError(f"{path} does not contain c2w")
    poses = np.asarray(data["c2w"], dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{path}: expected c2w [N,4,4], got {poses.shape}")
    if len(poses) == 0:
        raise ValueError(f"{path}: empty c2w")
    return poses[:max(1, min(int(n), len(poses)))]


def synthesize_free_trajectory(
    anchor_c2w: np.ndarray,
    n: int,
    *,
    motion: str = "forward",
    speed: float = 0.06,
    yaw_deg: float = 24.0,
    pitch_deg: float = 6.0,
    fwd_sign: float = 1.0,
    seed: int = 0,
    target_path_length: float | None = None,
    target_extent: float | None = None,
    target_extents_xyz: np.ndarray | None = None,
    target_direction: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Generate a smooth path and optionally match a reference path length.

    ``target_extent`` is the spatial diameter of the selected example pose
    prefix, so 17/33/81-view synthetic paths have the same metric scene scale
    as the exact repository example at the same view count.
    ``target_path_length`` is retained for backwards compatibility.  Forward
    and backward travel follow the camera's local viewing direction; turn
    trajectories keep the camera center fixed and apply a small eased yaw.
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    A = np.asarray(anchor_c2w, dtype=np.float64).copy()
    R0 = A[:3, :3]
    p0 = A[:3, 3].copy()
    ph = rng.uniform(0.0, 2.0 * np.pi, size=6)
    yaw_a = np.deg2rad(yaw_deg)
    pitch_a = np.deg2rad(pitch_deg)
    w = (2.0 * np.pi) / max(n - 1, 1)
    f = np.array([1.0, 2.3, 0.5]) * w
    poses: list[np.ndarray] = []
    pos = p0.copy()
    # Keep old names as aliases for callers outside the public app.
    aliases = {"drive": "forward", "wander": "turn_left", "orbit": "turn_right", "spiral": "forward"}
    motion = aliases.get(motion, motion)
    if motion not in {"forward", "backward", "turn_left", "turn_right"}:
        raise ValueError(f"unknown trajectory motion: {motion!r}")
    for t in range(n):
        u = t / max(n - 1, 1)
        # Smoothstep removes the abrupt velocity change at the endpoints.
        ease = u * u * (3.0 - 2.0 * u)
        if motion in ("turn_left", "turn_right"):
            # Positive camera-yaw is a left turn in the OpenCV convention.
            yaw = (1.0 if motion == "turn_left" else -1.0) * np.deg2rad(12.0) * ease
            pitch = np.deg2rad(2.0) * np.sin(np.pi * u)
            spd, strafe, bob = 0.0, 0.0, 0.0
        else:
            yaw = np.deg2rad(1.5) * np.sin(np.pi * u)
            pitch = np.deg2rad(1.0) * np.sin(np.pi * u)
            travel_sign = 1.0 if motion == "forward" else -1.0
            # The sign is applied once below to the local forward vector.
            spd = speed * (0.75 + 0.25 * np.sin(np.pi * u))
            strafe = bob = 0.0
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        R = R0 @ (Ry @ Rx)
        if motion in ("forward", "backward"):
            fwd = (1.0 if motion == "forward" else -1.0) * (-R[:, 2])
        else:
            fwd = np.zeros(3, dtype=np.float64)
        right, up = R[:, 0], R[:, 1]
        pos = pos + spd * fwd + strafe * right + bob * up
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3], c2w[:3, 3] = R, pos
        poses.append(c2w)

    if target_extents_xyz is not None:
        target_vec = np.asarray(target_extents_xyz, dtype=np.float64).reshape(3)
        actual_vec = np.ptp(np.asarray(poses)[:, :3, 3], axis=0)
        scale_vec = np.divide(target_vec, actual_vec, out=np.ones(3), where=actual_vec > 1e-9)
        origin = poses[0][:3, 3].copy()
        for pose in poses:
                pose[:3, 3] = origin + (pose[:3, 3] - origin) * scale_vec
    # Forward/backward are intentional signed motions; do not mirror them to
    # match the example path's direction.  Direction matching is only useful
    # for the turn previews and legacy callers.
    if target_direction is not None and motion not in ("forward", "backward") and len(poses) > 1:
        direction = np.asarray(target_direction, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(direction)
        if norm > 1e-9:
            direction /= norm
            origin = poses[0][:3, 3].copy()
            delta = poses[-1][:3, 3] - origin
            if float(np.dot(delta, direction)) < 0.0:
                for pose in poses:
                    rel = pose[:3, 3] - origin
                    pose[:3, 3] = origin + rel - 2.0 * np.dot(rel, direction) * direction
                if target_extents_xyz is not None:
                    target_vec = np.asarray(target_extents_xyz, dtype=np.float64).reshape(3)
                    actual_vec = np.ptp(np.asarray(poses)[:, :3, 3], axis=0)
                    scale_vec = np.divide(target_vec, actual_vec, out=np.ones(3), where=actual_vec > 1e-9)
                    for pose in poses:
                        pose[:3, 3] = origin + (pose[:3, 3] - origin) * scale_vec
    else:
        target = target_extent if target_extent is not None else target_path_length
        if target is not None and target > 0.0:
            actual = trajectory_extent(np.asarray(poses))
            if actual > 1e-9:
                scale = float(target) / actual
                origin = poses[0][:3, 3].copy()
                for pose in poses:
                    pose[:3, 3] = origin + (pose[:3, 3] - origin) * scale
    return poses


def trajectory_for_preview(
    reference_poses: np.ndarray, motion: str, n: int, *, speed: float = 0.06, seed: int = 42
) -> np.ndarray:
    ref = np.asarray(reference_poses, dtype=np.float64)
    ref = ref[:max(1, min(int(n), len(ref)))]
    if motion == "example":
        return ref
    return np.asarray(synthesize_free_trajectory(
        ref[0], len(ref), motion=motion, speed=speed, seed=seed,
        fwd_sign=reference_forward_sign(ref),
        target_extents_xyz=np.ptp(ref[:, :3, 3], axis=0),
        target_direction=ref[-1, :3, 3] - ref[0, :3, 3]))


def render_trajectory_preview(poses: np.ndarray, output: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    poses = np.asarray(poses, dtype=np.float64)
    pos = poses[:, :3, 3]
    fig = plt.figure(figsize=(8, 7))
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.82)
    ax = fig.add_subplot(111, projection="3d")
    # Draw short segments with a time gradient so the direction is readable
    # even when the trajectory folds back in depth.
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, max(len(pos) - 1, 1)))
    for i in range(len(pos) - 1):
        ax.plot(pos[i:i + 2, 0], pos[i:i + 2, 2], pos[i:i + 2, 1],
                color=colors[i], lw=2.8, solid_capstyle="round")
    ax.scatter(*[pos[0, i] for i in (0, 2, 1)], c="#34a853", s=65, label="start")
    ax.scatter(*[pos[-1, i] for i in (0, 2, 1)], c="#ea4335", s=65, label="end")
    ax.set_xlabel("world x", labelpad=8)
    ax.set_ylabel("world z (depth)", labelpad=8)
    ax.set_zlabel("world y (up)", labelpad=8)
    ax.set_title(f"3D camera pose trajectory\n{title}", pad=14, fontsize=14)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ax.view_init(elev=23, azim=-58)
    span_xyz = np.ptp(pos, axis=0)
    # Keep narrow lateral motion legible while retaining the true numeric axes.
    # A literal aspect ratio would collapse x/y for the long forward examples.
    visual_floor = max(float(span_xyz.max()) * 0.35, 1e-3)
    ax.set_box_aspect(np.maximum(span_xyz[[0, 2, 1]], visual_floor))
    ax.tick_params(axis="both", which="major", pad=1, labelsize=8)
    ax.tick_params(axis="z", which="major", pad=1, labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.94, 0.96, 0.98, 0.7))
    fig.savefig(output, dpi=160)
    plt.close(fig)
