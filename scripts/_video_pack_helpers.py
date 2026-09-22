"""Shared helpers for the dataset preprocessing scripts.

All three scripts (RE10K / MVS-Synth / DL3DV) need the same primitives:

  * Encode a list of BGR uint8 frames to H.264 mp4 via ffmpeg pipe
    (matches ``preprocess_scannetpp.py`` encoding: yuv420p / crf=18 / slow).
  * Atomic write: same-FS tmp + ``os.replace``; cross-FS ``cp`` fallback.
    Works for both ``/local-ssd → /local-ssd`` staging (the common case) and for the legacy direct-to-FUSE
    write path.
  * Read PNG safely with cv2.

The output layout for every dataset is identical:

    OUT/<scene_id>/
        video.mp4    — H.264, per-scene frames at the input resolution
        meta.json    — {"num_frames", "rgb_resolution"|("height","width"),
                        "frames": [{"c2w": 4x4, "K": 3x3, "name", ...}, ...]}
        caption.txt  — optional, single line scene caption
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


class ProgressLogger:
    """Line-buffered progress reporter for use inside container / kubernetes logs.

    tqdm uses ``\\r``-based in-place updates that get swallowed by log files
    and never flushed. This logger emits one plain ``print(..., flush=True)``
    line every ``interval`` seconds (and on the final update), so progress is
    visible in real time both in interactive terminals and in tail-able log
    files of long-running jobs.
    """

    def __init__(self, total: int, tag: str, interval: float = 30.0):
        self.total = max(int(total), 1)
        self.tag = tag
        self.interval = float(interval)
        self.start = time.time()
        self.last_emit = self.start - interval  # force-emit on first update
        self.n = 0

    def update(self, n: int = 1, extra: str = "") -> None:
        self.n += n
        now = time.time()
        last_one = self.n >= self.total
        if now - self.last_emit < self.interval and not last_one:
            return
        elapsed = now - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.n)
        eta = remaining / rate if rate > 0 else float("inf")
        eta_s = f"{eta:6.0f}s" if eta != float("inf") else "    -- "
        pct = 100.0 * self.n / self.total
        print(
            f"[{self.tag}] {self.n:>6d}/{self.total} ({pct:5.1f}%)  "
            f"{extra}  elapsed={elapsed:6.0f}s  rate={rate:5.2f}/s  "
            f"eta={eta_s}",
            flush=True,
        )
        self.last_emit = now


def fmt_stats(stats: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in stats.items())


def _tmp_beside(out_path: Path) -> Path:
    """Return a tmp path in the *same directory* as out_path.

    Why beside? Because ``os.replace(src, dst)`` is only atomic within one
    filesystem. By living next to the target, the tmp is guaranteed to share
    the FS — so the rename is a true atomic in-place move (~free) instead of
    a cross-FS copy.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.parent / (
        f".{out_path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}"
    )


def _local_tmp(suffix: str) -> Path:
    """Tmp on local-ssd (or /tmp). Used only for the ffmpeg side-channel where
    we can't ask ffmpeg to write into a hidden file next to the final mp4
    (because we want to keep stale .tmp out of the final dir on crash)."""
    base = "/local-ssd" if Path("/local-ssd").is_dir() else tempfile.gettempdir()
    return Path(base) / f"_vmpack_{os.getpid()}_{os.urandom(4).hex()}{suffix}"


def _move_atomic(src: Path, dst: Path) -> None:
    """Atomic in-place move src → dst, with cross-FS cp fallback."""
    try:
        os.replace(src, dst)
    except OSError:
        # Different filesystems (e.g. /local-ssd → S3-FUSE). Fall back to cp.
        subprocess.run(["cp", str(src), str(dst)], check=True)
        try:
            src.unlink()
        except FileNotFoundError:
            pass


def encode_mp4_ffmpeg(
    frames_bgr: list[np.ndarray],
    out_path: Path,
    fps: int = 15,
    crf: int = 18,
    preset: str = "medium",
    threads: int = 0,
) -> None:
    """Pipe BGR frames to ffmpeg for H.264 encoding.

    Defaults: libx264 / yuv420p / preset=medium / crf=18 / threads=0(auto).

    Why these defaults
    ------------------
    * preset=medium: benchmarked on 540p×100 frames, ``slow`` takes 2.3s vs
      ``medium`` 1.1s and produces a byte-for-byte indistinguishable 4.8 MB
      mp4. At crf=18 (>~45 dB PSNR, well above the 33–44 dB VAE upper bound)
      the preset choice does not change visual quality, only speed.
    * threads=0 (auto): with ``NUM_SHARDS`` (=8) parallel preprocess workers
      on a 192-core box, letting libx264 auto-thread costs nothing — ``slow``
      preset on 8 ffmpeg in parallel finished in 20s wall (=2.5s/instance),
      so per-scene encode is *not* the bottleneck. (We benchmarked
      ``threads=1`` and it was 4× SLOWER because it leaves 184 cores idle.)

    Writes to a local-ssd temp file first, then ``os.replace``-moves it to
    ``out_path`` (atomic rename if same FS, ``cp`` fallback if cross-FS like
    /local-ssd → S3-FUSE).
    """
    if not frames_bgr:
        raise RuntimeError("No frames to encode.")
    H, W = frames_bgr[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg writes into /local-ssd first (so partial files on crash don't
    # litter the output dir), then we atomically rename it next to out_path.
    ff_tmp = _local_tmp(".mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-threads", str(threads),
        "-pix_fmt", "yuv420p",
        str(ff_tmp),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    success = False
    try:
        for f in frames_bgr:
            if f.shape != (H, W, 3) or f.dtype != np.uint8:
                raise RuntimeError(
                    f"bad frame shape/dtype {f.shape} {f.dtype}"
                )
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed rc={rc}")
        # Atomic move (same-FS rename if possible; copy if the destination
        # is on another filesystem).
        _move_atomic(ff_tmp, out_path)
        success = True
    finally:
        # Kill the encoder if we never closed stdin (exception in the loop).
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if not success:
            try:
                ff_tmp.unlink()
            except FileNotFoundError:
                pass


def atomic_write_text(text: str, out_path: Path) -> None:
    """Write text via same-dir tmp + atomic rename. Falls back to cp cross-FS."""
    tmp = _tmp_beside(out_path)
    try:
        tmp.write_text(text)
        _move_atomic(tmp, out_path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def write_meta_json(meta: dict, out_path: Path) -> None:
    atomic_write_text(json.dumps(meta, separators=(",", ":")), out_path)


def read_png_bgr(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def dataset_root_or_die(p: str) -> Path:
    root = Path(p)
    if not root.is_dir():
        raise SystemExit(f"[fatal] dataset root not found: {root}")
    return root


def round_robin_shard(items: list, num_shards: int, shard_index: int) -> list:
    return [s for i, s in enumerate(items) if i % num_shards == shard_index]
