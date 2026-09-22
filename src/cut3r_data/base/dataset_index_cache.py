"""Two-tier dataset index cache (local-ssd + shared FUSE) with DDP-safe locking.

Cold index builds on S3-FUSE (e.g. 67k RE10K scenes) take hours. Per-node
``/local-ssd`` is wiped between jobs; a shared pickle on durable storage lets
the next restart skip the scan. An exclusive lock on the local cache path
serializes builders on one node so torchrun ranks do not hammer FUSE in parallel.
"""

from __future__ import annotations

import fcntl
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_DEFAULT_SHARED_ROOT = os.environ.get("GLD_SHARED_CACHE_DIR", "")
_DISABLED = frozenset({"", "none", "off", "disable", "false", "0"})


def resolve_shared_cache_dir() -> str | None:
    """Return a writable shared cache root, or ``None`` if disabled/unavailable."""
    for env_name in ("GLD_SHARED_CACHE_DIR", "OSP_SHARED_CACHE_DIR"):
        base = os.environ.get(env_name, "").strip()
        if base.lower() in _DISABLED:
            continue
        if base:
            parent = Path(base).parent
            if parent.exists():
                return base
            return None

    base = _DEFAULT_SHARED_ROOT
    if not base:
        return None
    parent = Path(base).parent
    if parent.exists():
        return base
    return None


def mirror_shared_path(local_path: str, shared_root: str | None) -> str | None:
    """Map ``/local-ssd/<subdir>/file.pkl`` → ``<shared_root>/<subdir>/file.pkl``."""
    if shared_root is None:
        return None
    parts = Path(local_path).parts
    if "local-ssd" in parts:
        idx = parts.index("local-ssd")
        rel = Path(*parts[idx + 1 :])
        return str(Path(shared_root) / rel)
    return str(Path(shared_root) / Path(local_path).name)


def try_load_pickle(path: str, *, log_prefix: str) -> Any | None:
    if not osp.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
        # EOFError / partial reads happen when another rank is mid-copy and we
        # peek at the local cache without holding the lock; treat as "not ready".
        print(
            f"[{log_prefix}] cache read failed ({path}): {e}; will rebuild.",
            flush=True,
        )
        return None


def write_pickle_atomic(path: str, obj: Any) -> None:
    """Write pickle atomically; per-process temp name avoids DDP races on ``.tmp``."""
    cache_dir = osp.dirname(path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=cache_dir,
        prefix=f".{osp.basename(path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_copy(src: str, dst: str) -> None:
    """Copy ``src`` → ``dst`` via a temp file + ``os.replace`` so readers never
    observe a partially written ``dst`` (avoids EOFError on unlocked peeks)."""
    dst_dir = osp.dirname(dst) or "."
    os.makedirs(dst_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=dst_dir,
        prefix=f".{osp.basename(dst)}.",
        suffix=".tmp",
    )
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_shared_pickle_best_effort(
    path: str, obj: Any, *, log_prefix: str,
) -> None:
    """Best-effort shared write; skip if the object already exists (FUSE EPERM)."""
    if osp.exists(path):
        print(
            f"[{log_prefix}] shared cache already present at {path}, skip write",
            flush=True,
        )
        return
    try:
        os.makedirs(osp.dirname(path), exist_ok=True)
    except OSError as e:
        print(f"[{log_prefix}] shared cache mkdir failed: {e}", flush=True)
        return
    try:
        if osp.exists(path):
            return
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[{log_prefix}] copied index to shared cache {path}", flush=True)
    except OSError as e:
        print(f"[{log_prefix}] shared cache write failed: {e}", flush=True)


def _hydrate_local(
    local_path: str,
    obj: Any,
    *,
    log_prefix: str,
    shared_path: str | None = None,
) -> None:
    """Copy shared pickle to local-ssd once (under lock); skip if already present."""
    if osp.isfile(local_path):
        return

    if shared_path and osp.isfile(shared_path):
        try:
            atomic_copy(shared_path, local_path)
            print(
                f"[{log_prefix}] hydrated local cache (copy) → {local_path}",
                flush=True,
            )
            return
        except OSError as e:
            print(
                f"[{log_prefix}] local cache copy failed ({e}); trying pickle write",
                flush=True,
            )

    try:
        write_pickle_atomic(local_path, obj)
        print(f"[{log_prefix}] hydrated local cache to {local_path}", flush=True)
    except OSError as e:
        if osp.isfile(local_path):
            # Another rank won the race after our write failed.
            print(
                f"[{log_prefix}] hydrate raced; local cache present at {local_path}",
                flush=True,
            )
            return
        print(f"[{log_prefix}] local cache hydrate failed: {e}", flush=True)


def load_or_build(
    *,
    log_prefix: str,
    local_path: str,
    shared_path: str | None,
    build_fn: Callable[[], T],
    wait_interval: float = 2.0,
) -> T:
    """Load index from local → shared → build once under ``local_path.lock``.

    Shared load + local hydrate run **inside** the lock so DDP ranks do not
    all pickle-dump the same multi-GB index to one ``.tmp`` path.
    """
    data = try_load_pickle(local_path, log_prefix=log_prefix)
    if data is not None:
        print(
            f"[{log_prefix}] loaded cached index from {local_path}",
            flush=True,
        )
        return data

    lock_path = local_path + ".lock"
    lock_dir = osp.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    with open(lock_path, "w") as lock_f:
        waited = False
        while True:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waited:
                    print(
                        f"[{log_prefix}] another process is building the index; "
                        "waiting ...",
                        flush=True,
                    )
                    waited = True
                time.sleep(wait_interval)
                data = try_load_pickle(local_path, log_prefix=log_prefix)
                if data is not None:
                    print(
                        f"[{log_prefix}] loaded cached index from {local_path}",
                        flush=True,
                    )
                    return data

        try:
            data = try_load_pickle(local_path, log_prefix=log_prefix)
            if data is not None:
                print(
                    f"[{log_prefix}] loaded cached index from {local_path}",
                    flush=True,
                )
                return data

            if shared_path is not None and osp.isfile(shared_path):
                if not osp.isfile(local_path):
                    try:
                        atomic_copy(shared_path, local_path)
                        print(
                            f"[{log_prefix}] hydrated local cache (copy) "
                            f"→ {local_path}",
                            flush=True,
                        )
                    except OSError as e:
                        print(
                            f"[{log_prefix}] local copy failed ({e}); "
                            f"will load shared pickle directly",
                            flush=True,
                        )
                data = try_load_pickle(local_path, log_prefix=log_prefix)
                if data is None:
                    data = try_load_pickle(shared_path, log_prefix=log_prefix)
                if data is not None:
                    print(
                        f"[{log_prefix}] loaded shared index "
                        f"({shared_path})",
                        flush=True,
                    )
                    return data

            print(f"[{log_prefix}] building index (first time) ...", flush=True)
            data = build_fn()

            try:
                write_pickle_atomic(local_path, data)
                print(f"[{log_prefix}] cached index to {local_path}", flush=True)
            except OSError as e:
                if not osp.isfile(local_path):
                    print(
                        f"[{log_prefix}] local cache write failed: {e}",
                        flush=True,
                    )

            if shared_path is not None:
                write_shared_pickle_best_effort(
                    shared_path, data, log_prefix=log_prefix,
                )
            return data
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
