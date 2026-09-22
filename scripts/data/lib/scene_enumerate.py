"""Fast scene enumeration for DA3 pose export on network-backed datasets."""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

from lib.da3_metric_pose import SOURCE_META_NAME

_INDEX_WORKERS = int(os.environ.get("DA3_POSE_INDEX_WORKERS", "64"))
_SCENE_LIST_VERSION = 1

# Same VMS caches RE10K_Packed adopts — skip the 60k+ meta.json cold scan.
_RE10K_VMS_HINTS: dict[str, str] = {}


def _shared_cache_dir() -> Path | None:
    value = os.environ.get("DA3_POSE_SHARED_CACHE", "").strip()
    return Path(value) if value else None


def _writable_cache_dir() -> Path:
    """Pick a directory we can actually write to (locks + JSON cache)."""
    env = os.environ.get("DA3_POSE_SCENE_LIST_CACHE", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        Path("/local-ssd/da3_pose_scene_lists"),
        Path.home() / ".cache" / "da3_pose_scene_lists",
        Path(tempfile.gettempdir()) / "da3_pose_scene_lists",
    ])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".__write_probe"
            probe.write_text("ok")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise RuntimeError(
        "[index] no writable scene-list cache directory; "
        "set DA3_POSE_SCENE_LIST_CACHE to a writable path"
    )


def _cache_filename(dataset: str, root: Path, split: str | None) -> str:
    root_s = str(root.resolve())
    tag = hashlib.md5(f"{dataset}|{root_s}|{split or 'all'}".encode()).hexdigest()[:16]
    return f"{dataset}_{split or 'all'}_{tag}.json"


def _cache_paths(dataset: str, root: Path, split: str | None) -> tuple[Path, list[Path]]:
    name = _cache_filename(dataset, root, split)
    writable = _writable_cache_dir() / name
    readers = [writable]
    shared = _shared_cache_dir()
    if shared is not None:
        readers.append(shared / name)
    return writable, readers


def _local_mirror_dir() -> Path | None:
    try:
        return _writable_cache_dir()
    except RuntimeError:
        return None


def _is_exportable_scene_dir(d: Path) -> bool:
    return (d / SOURCE_META_NAME).is_file() and (d / "video.mp4").is_file()


def _scan_one_exportable(path_str: str) -> str | None:
    p = Path(path_str)
    if _is_exportable_scene_dir(p):
        return str(p.resolve())
    return None


def _parallel_filter_exportable(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    items = [str(p) for p in paths]
    out: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=_INDEX_WORKERS) as ex:
        for hit in ex.map(_scan_one_exportable, items, chunksize=64):
            if hit is not None:
                out.append(hit)
    out.sort()
    return [Path(p) for p in out]


def _list_child_dirs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    try:
        with os.scandir(base) as it:
            return sorted(Path(e.path) for e in it if e.is_dir(follow_symlinks=False))
    except OSError:
        return []


def _scan_re10k_split(root: Path, split: str) -> list[Path]:
    base = root / split
    entries = _list_child_dirs(base)
    if not entries:
        return []
    print(
        f"[index] scanning {split} ({len(entries)} entries, "
        f"{_INDEX_WORKERS} workers) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    out = _parallel_filter_exportable(entries)
    print(
        f"[index] scanned {split}: {len(out)} exportable in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _scan_flat_dataset(root: Path) -> list[Path]:
    entries = _list_child_dirs(root)
    print(
        f"[index] scanning root ({len(entries)} entries, {_INDEX_WORKERS} workers) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    out = _parallel_filter_exportable(entries)
    print(
        f"[index] scanned: {len(out)} exportable in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _scan_scannetpp(root: Path) -> list[Path]:
    scene_dirs = _list_child_dirs(root)
    candidates: list[Path] = []
    for scene_dir in scene_dirs:
        # The release preprocess writes a flat layout (scene/{video.mp4,meta.json});
        # older nested data stores multiple clip_* subdirs per scene. Support both:
        # take the scene dir itself when exportable, else fall back to its clips.
        if _is_exportable_scene_dir(scene_dir):
            candidates.append(scene_dir)
        else:
            candidates.extend(
                sorted(p for p in scene_dir.glob("clip_*") if p.is_dir())
            )
    print(
        f"[index] scanning scannetpp ({len(candidates)} scene/clip dirs, "
        f"{_INDEX_WORKERS} workers) ...",
        flush=True,
    )
    t0 = time.perf_counter()
    out = _parallel_filter_exportable(candidates)
    print(
        f"[index] scanned scannetpp: {len(out)} exportable in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _stage_to_local_ssd(src: Path) -> Path:
    """Copy large network-backed pickles to local-ssd once per node."""
    local_root = _local_mirror_dir()
    if local_root is None:
        return src
    dst = local_root / src.name
    try:
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            return dst
        local_root.mkdir(parents=True, exist_ok=True)
        print(f"[index] staging {src.name} -> {dst} ...", flush=True)
        t0 = time.perf_counter()
        dst.write_bytes(src.read_bytes())
        print(f"[index] staged {src.name} in {time.perf_counter() - t0:.1f}s", flush=True)
        return dst
    except OSError as e:
        print(f"[index] local stage failed ({e}); reading {src} directly.", flush=True)
        return src


def _load_re10k_vms_split(split: str) -> list[Path] | None:
    vms_path = os.environ.get("RE10K_PACKED_VMS_CACHE", "").strip()
    if not vms_path:
        vms_path = _RE10K_VMS_HINTS.get(split, "")
    src = Path(vms_path)
    if not vms_path or not src.is_file():
        return None

    load_path = _stage_to_local_ssd(src)
    print(f"[index] loading RE10K VMS cache for {split}: {load_path}", flush=True)
    t0 = time.perf_counter()
    try:
        with open(load_path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        print(f"[index] VMS load failed ({e}); falling back to scan.", flush=True)
        return None

    if not isinstance(d, dict) or not d.get("scenes") or not d.get("scene_data"):
        print("[index] VMS cache shape unexpected; falling back to scan.", flush=True)
        return None

    out: list[Path] = []
    for scene_name in d["scenes"]:
        vsd = d["scene_data"].get(scene_name)
        if not vsd:
            continue
        meta_path = vsd.get("meta_path")
        if not meta_path:
            continue
        # Keep this path operation purely lexical. Calling resolve() for every
        # RE10K scene touches the network-backed dataset mount tens of thousands
        # of times and can make VMS cache loading look hung.
        out.append(Path(meta_path).parent)

    out.sort(key=lambda p: str(p))
    print(
        f"[index] VMS {split}: {len(out)} scenes in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out


def _scan_dataset(dataset: str, root: Path, split: str | None) -> list[Path]:
    dataset = dataset.lower()
    if dataset == "re10k_packed":
        if split:
            vms = _load_re10k_vms_split(split)
            if vms is not None:
                return vms
            return _scan_re10k_split(root, split)
        out: list[Path] = []
        for sp in ("train", "test"):
            vms = _load_re10k_vms_split(sp)
            if vms is not None:
                out.extend(vms)
            else:
                out.extend(_scan_re10k_split(root, sp))
        out.sort(key=lambda p: str(p))
        return out

    if dataset in ("dl3dv_packed", "mvssynth_packed"):
        return _scan_flat_dataset(root)

    if dataset == "scannetpp":
        return _scan_scannetpp(root)

    raise ValueError(f"unknown dataset: {dataset}")


def _write_cache_atomic(writable_path: Path, payload: dict) -> None:
    text = json.dumps(payload, separators=(",", ":"))
    writable_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".__scene_list_", suffix=".json", dir=writable_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, writable_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    shared = _shared_cache_dir()
    if shared is None:
        return
    shared_target = shared / writable_path.name
    try:
        shared.mkdir(parents=True, exist_ok=True)
        shared_target.write_text(text)
    except OSError as e:
        print(
            f"[index] note: shared cache not writable ({shared_target}): {e}",
            flush=True,
        )


def _load_cache(
    read_paths: list[Path], dataset: str, root: Path, split: str | None,
) -> list[Path] | None:
    for candidate in read_paths:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("version") != _SCENE_LIST_VERSION:
            continue
        if payload.get("dataset") != dataset:
            continue
        if payload.get("root") != str(root.resolve()):
            continue
        if payload.get("split") != (split or "all"):
            continue
        scenes = payload.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            continue
        return [Path(p) for p in scenes]
    return None


def enumerate_scene_dirs(
    dataset: str,
    root: Path,
    split: str | None,
    *,
    rebuild_cache: bool = False,
) -> list[Path]:
    """Return sorted exportable scene dirs; uses JSON cache when possible."""
    root = Path(root)
    cache_path, read_paths = _cache_paths(dataset, root, split)

    if not rebuild_cache:
        t0 = time.perf_counter()
        cached = _load_cache(read_paths, dataset, root, split)
        if cached is not None:
            print(
                f"[index] cache hit: {len(cached)} scenes from {cache_path.name} "
                f"({time.perf_counter() - t0:.2f}s)",
                flush=True,
            )
            return cached

    lock_path = cache_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if not rebuild_cache:
            cached = _load_cache(read_paths, dataset, root, split)
            if cached is not None:
                print(
                    f"[index] cache hit after lock: {len(cached)} scenes from {cache_path.name}",
                    flush=True,
                )
                return cached

        t0 = time.perf_counter()
        scenes = _scan_dataset(dataset, root, split)
        payload = {
            "version": _SCENE_LIST_VERSION,
            "dataset": dataset,
            "root": str(root.resolve()),
            "split": split or "all",
            "count": len(scenes),
            "scenes": [str(p) for p in scenes],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            _write_cache_atomic(cache_path, payload)
            print(
                f"[index] wrote cache ({len(scenes)} scenes) -> {cache_path} "
                f"in {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
        except OSError as e:
            print(f"[index] warning: could not write cache {cache_path}: {e}", flush=True)
        return scenes
