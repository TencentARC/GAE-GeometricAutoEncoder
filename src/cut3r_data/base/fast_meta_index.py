"""Fast index helpers for packed scene datasets (video.mp4 + meta.json).

Index build only needs header fields (``num_frames``, ``caption``, …). Full
``json.load`` of 300+ frame poses is the main FUSE bottleneck (~1 scene/s).
We parse a truncated prefix ending before ``"frames"``.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")

_HEADER_READ_BYTES = 131072


def scene_in_split(scene_id: str, split: str | None, val_frac: float) -> bool:
    if split is None:
        return True
    h = int(hashlib.md5(scene_id.encode()).hexdigest()[:8], 16)
    is_val = (h % 10000) < int(10000 * val_frac)
    return is_val if split == "val" else (not is_val)


def fast_load_packed_meta_header(meta_path: str) -> dict[str, Any] | None:
    """Load index-time fields without parsing the ``frames`` array."""
    try:
        with open(meta_path, encoding="utf-8") as f:
            buf = f.read(_HEADER_READ_BYTES)
    except OSError:
        return None

    key = '"frames"'
    i = buf.find(key)
    if i < 0:
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    else:
        head = buf[:i].rstrip()
        if head.endswith(","):
            head = head[:-1]
        try:
            meta = json.loads(head + "\n}")
        except json.JSONDecodeError:
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                return None

    cap_val = meta.get("caption", "")
    return {
        "num_frames": int(meta.get("num_frames", 0)),
        "rgb_resolution": int(meta.get("rgb_resolution", meta.get("height", 0))),
        "inline_caption": cap_val.strip() if isinstance(cap_val, str) else "",
    }


def _index_log_interval() -> int:
    raw = os.environ.get("INDEX_LOG_INTERVAL", "100").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _log_index_progress(desc: str, done: int, total: int) -> None:
    """Print ``done/total`` every ``INDEX_LOG_INTERVAL`` items (default 100)."""
    if total <= 0:
        return
    interval = _index_log_interval()
    if done % interval != 0 and done != total:
        return
    pct = 100.0 * done / total
    print(f"{desc}: {done}/{total} ({pct:.1f}%)", flush=True)


def _index_tqdm(
    iterable: Iterable[T],
    *,
    total: int,
    desc: str,
    unit: str = "scene",
) -> Iterator[T]:
    """Progress bar for index builds.

    * ``DISABLE_INDEX_TQDM=1`` — no bar
    * ``FORCE_INDEX_TQDM=1`` — show bar even when stderr is not a TTY (non-TTY logs)
    """
    if os.environ.get("DISABLE_INDEX_TQDM", "").strip().lower() in (
        "1", "true", "yes",
    ):
        return iter(iterable)
    try:
        from tqdm import tqdm
    except ImportError:
        return iter(iterable)
    force = os.environ.get("FORCE_INDEX_TQDM", "").strip().lower() in (
        "1", "true", "yes",
    )
    disable = (not force) and (not sys.stderr.isatty())
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        disable=disable,
        mininterval=1.0,
    )


def default_vms_index_workers() -> int:
    raw = os.environ.get("VMS_INDEX_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    ncpu = os.cpu_count() or 8
    # FUSE I/O-bound: threads avoid fork + mountpoint-s3 stalls seen with Pool.
    return max(8, min(64, ncpu * 4))


def _vms_index_backend() -> str:
    """``thread`` (default on FUSE) or ``process`` (fast local disk only)."""
    return os.environ.get("VMS_INDEX_BACKEND", "thread").strip().lower()


def _index_one_packed_scene(
    args: tuple[str, str, str, int, str | None, float, bool],
) -> tuple[str, dict, list[tuple[str, int]]] | None:
    scene_id, meta_path, video_path, cut_off, split, val_frac, with_caption = args
    if not scene_in_split(scene_id, split, val_frac):
        return None

    fields = fast_load_packed_meta_header(meta_path)
    if fields is None:
        return None
    if not osp.isfile(video_path):
        return None

    num_frames = int(fields["num_frames"])
    if num_frames < cut_off:
        return None

    sd = {
        "video_path": video_path,
        "meta_path": meta_path,
        "num_frames": num_frames,
        "rgb_resolution": int(fields["rgb_resolution"]),
    }
    if with_caption:
        sd["inline_caption"] = fields["inline_caption"]

    num_start = max(1, num_frames - cut_off + 1)
    start_ids = [(scene_id, si) for si in range(num_start)]
    return scene_id, sd, start_ids


def build_packed_scene_index(
    root: str,
    *,
    cut_off: int,
    split: str | None,
    val_frac: float,
    with_caption: bool = True,
    workers: int | None = None,
    progress_desc: str = "VMS index",
) -> dict[str, Any]:
    """Parallel index over ``ROOT/<scene>/{video.mp4,meta.json}``."""
    if workers is None:
        workers = default_vms_index_workers()

    print(f"{progress_desc}: listing scene dirs under {root} …", flush=True)
    tasks: list[tuple[str, str, str, int, str | None, float, bool]] = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir():
                continue
            meta_path = osp.join(entry.path, "meta.json")
            video_path = osp.join(entry.path, "video.mp4")
            tasks.append(
                (entry.name, meta_path, video_path, cut_off, split, val_frac, with_caption)
            )

    scenes: list[str] = []
    scene_data: dict[str, dict] = {}
    start_img_ids: list[tuple[str, int]] = []
    n_total = len(tasks)
    backend = _vms_index_backend()
    print(
        f"{progress_desc}: {n_total} dirs, backend={backend}, workers={workers}",
        flush=True,
    )

    def _merge(item: tuple[str, dict, list[tuple[str, int]]] | None) -> None:
        if item is None:
            return
        sid, sd, starts = item
        scenes.append(sid)
        scene_data[sid] = sd
        start_img_ids.extend(starts)

    done = 0
    if workers <= 1 or n_total <= 1:
        for t in tasks:
            _merge(_index_one_packed_scene(t))
            done += 1
            _log_index_progress(progress_desc, done, n_total)
    elif backend == "process":
        chunksize = max(1, min(64, n_total // (workers * 16) or 1))
        with Pool(processes=workers) as pool:
            for item in pool.imap_unordered(
                _index_one_packed_scene, tasks, chunksize=chunksize,
            ):
                _merge(item)
                done += 1
                _log_index_progress(progress_desc, done, n_total)
    else:
        # Thread pool: shares FUSE mount; avoids Pool fork hangs on mountpoint-s3.
        batch = max(workers * 8, 256)
        disable_bar = os.environ.get("DISABLE_INDEX_TQDM", "").strip().lower() in (
            "1", "true", "yes",
        )
        pbar = None
        if not disable_bar:
            try:
                from tqdm import tqdm

                force = os.environ.get("FORCE_INDEX_TQDM", "").strip().lower() in (
                    "1", "true", "yes",
                )
                pbar = tqdm(
                    total=n_total,
                    desc=progress_desc,
                    unit="scene",
                    dynamic_ncols=True,
                    disable=(not force) and (not sys.stderr.isatty()),
                    mininterval=0.5,
                )
            except ImportError:
                pass

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start in range(0, n_total, batch):
                chunk = tasks[start : start + batch]
                futures = [ex.submit(_index_one_packed_scene, t) for t in chunk]
                for fut in as_completed(futures):
                    _merge(fut.result())
                    done += 1
                    if pbar is not None:
                        pbar.update(1)
                    _log_index_progress(progress_desc, done, n_total)
        if pbar is not None:
            pbar.close()

    scenes.sort()
    return {
        "scenes": scenes,
        "scene_data": scene_data,
        "start_img_ids": start_img_ids,
    }
