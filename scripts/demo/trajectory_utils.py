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
    # than path length for comparing straight, orbiting, and wandering paths.
    distances = pos[:, None, :] - pos[None, :, :]
    return float(np.linalg.norm(distances, axis=-1).max())


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
    motion: str = "wander",
    speed: float = 0.06,
    yaw_deg: float = 24.0,
    pitch_deg: float = 6.0,
    fwd_sign: float = 1.0,
    seed: int = 0,
    target_path_length: float | None = None,
    target_extent: float | None = None,
) -> list[np.ndarray]:
    """Generate a smooth path and optionally match a reference path length.

    ``target_extent`` is the spatial diameter of the selected example pose
    prefix, so 17/33/81-view synthetic paths have the same metric scene scale
    as the exact repository example at the same view count.
    ``target_path_length`` is retained for backwards compatibility. Drive uses a positive speed
    envelope and therefore never reverses its forward motion.
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
    for t in range(n):
        if motion == "orbit":
            yaw = yaw_a * 3.0 * (t / max(n - 1, 1))
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd, strafe, bob = 0.0, speed, 0.0
        elif motion == "spiral":
            yaw = yaw_a * 2.0 * (t / max(n - 1, 1))
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd, strafe = speed * 0.5, speed * 0.5
            bob = speed * 0.2 * np.sin(f[2] * t + ph[5])
        elif motion == "drive":
            # Forward dolly: speed stays positive while yaw gently varies.
            yaw = yaw_a * (0.6 * np.sin(f[0] * t + ph[0])
                           + 0.4 * np.sin(f[1] * t + ph[1]))
            pitch = 0.0
            spd = speed * (0.65 + 0.35 * np.sin(f[1] * t + ph[3]))
            strafe, bob = 0.0, 0.0
        else:  # wander
            yaw = yaw_a * (0.6 * np.sin(f[0] * t + ph[0])
                           + 0.4 * np.sin(f[1] * t + ph[1]))
            pitch = pitch_a * np.sin(f[2] * t + ph[2])
            spd = speed * (0.5 + 0.5 * np.sin(f[1] * t + ph[3]))
            strafe = speed * 0.6 * np.sin(f[0] * t + ph[4])
            bob = speed * 0.25 * np.sin(2.0 * f[2] * t + ph[5])
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        R = R0 @ (Ry @ Rx)
        fwd = fwd_sign * (-R[:, 2])
        right, up = R[:, 0], R[:, 1]
        pos = pos + spd * fwd + strafe * right + bob * up
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3], c2w[:3, 3] = R, pos
        poses.append(c2w)

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
        target_extent=trajectory_extent(ref)))


def render_trajectory_preview(poses: np.ndarray, output: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(poses)[:, :3, 3]
    fwd = -np.asarray(poses)[:, :3, 2]
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_top.plot(pos[:, 0], pos[:, 2], color="#1a73e8", lw=2.2)
    ax_top.scatter(pos[0, 0], pos[0, 2], c="#34a853", s=55, label="start")
    ax_top.scatter(pos[-1, 0], pos[-1, 2], c="#ea4335", s=55, label="end")
    stride = max(1, len(pos) // 10)
    scale = max(float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 2])), 1e-3) * 0.12
    ax_top.quiver(pos[::stride, 0], pos[::stride, 2], fwd[::stride, 0], fwd[::stride, 2],
                  color="#1a73e8", scale_units="xy", scale=1.0 / scale, width=0.003)
    ax_top.set_title("Top view (world x / z)")
    ax_top.set_xlabel("world x"); ax_top.set_ylabel("world z (depth)")
    ax_top.set_aspect("equal", adjustable="box"); ax_top.grid(alpha=0.25); ax_top.legend()
    ax_side.plot(pos[:, 0], pos[:, 1], color="#a142f4", lw=2.2)
    ax_side.scatter(pos[0, 0], pos[0, 1], c="#34a853", s=55, label="start")
    ax_side.scatter(pos[-1, 0], pos[-1, 1], c="#ea4335", s=55, label="end")
    ax_side.set_title("Side view (world x / y)")
    ax_side.set_xlabel("world x"); ax_side.set_ylabel("world y (up)")
    ax_side.grid(alpha=0.25); ax_side.legend()
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)
