"""Shared training runtime helpers used by Stage 2.

Public scripts (`scripts/train_flow.py`, `scripts/eval_generation.py`) import
from here. The historical `src/train_flow_from_cache.py` trainer is not an
entry point for this release.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
from collections import OrderedDict
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP


def _to_dict(section):
    if section is None:
        return {}
    return (
        OmegaConf.to_container(section, resolve=True)
        if not isinstance(section, dict) else section
    )


def _strip_rope_buffers(sd):
    """RoPE freqs are deterministic buffers; allow loose checkpoint shapes."""
    if sd is None:
        return sd
    return {
        k: v for k, v in sd.items()
        if not (k.endswith("freqs_cos") or k.endswith("freqs_sin")
                or k.endswith("inv_freq"))
    }


def create_logger(log_dir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger("gae")
    logger.setLevel(logging.INFO)
    logger._log_local_path = None
    logger._log_remote_path = None
    if rank == 0:
        handlers = [logging.StreamHandler()]
        if log_dir:
            remote_log = f"{log_dir}/log.txt"
            local_ssd = "/local-ssd"
            if os.path.isdir(local_ssd):
                local_log = os.path.join(local_ssd, f"log_{os.getpid()}.txt")
                logger._log_local_path = local_log
                logger._log_remote_path = remote_log
            else:
                local_log = remote_log
            handlers.append(logging.FileHandler(local_log))
        formatter = logging.Formatter(
            "[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        for h in handlers:
            h.setFormatter(formatter)
            logger.addHandler(h)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def sync_log(logger):
    if getattr(logger, "_log_local_path", None) and getattr(logger, "_log_remote_path", None):
        for h in logger.handlers:
            if hasattr(h, "flush"):
                h.flush()
        try:
            remote = logger._log_remote_path
            try:
                os.remove(remote)
            except FileNotFoundError:
                pass
            subprocess.call(["cp", logger._log_local_path, remote])
        except OSError:
            pass


@torch.no_grad()
def update_ema(ema, model, decay):
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp.data, alpha=1 - decay)


def _pick_staging_dir(skip_prefix=None):
    candidates = ["/local-ssd", "/dev/shm", "/tmp"]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        if skip_prefix and d == skip_prefix:
            continue
        probe = os.path.join(d, f"._ckpt_probe_{os.getpid()}")
        try:
            with open(probe, "wb") as f:
                f.write(b"\0")
            os.remove(probe)
            return d
        except OSError:
            continue
    return None


def _safe_torch_save(obj, path):
    if not path.endswith(".pt"):
        raise ValueError(f"_safe_torch_save refuses non-.pt path: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    target_prefix = "/local-ssd" if path.startswith("/local-ssd") else None
    stage_dir = _pick_staging_dir(skip_prefix=target_prefix)
    if stage_dir is None:
        torch.save(obj, path)
        return

    tmp = os.path.join(stage_dir, f"_ckpt_{os.getpid()}_{os.path.basename(path)}")
    try:
        torch.save(obj, tmp)
    except (OSError, RuntimeError) as e:
        print(f"[WARN] staging save to {tmp} failed ({e}); writing {path} directly")
        torch.save(obj, path)
        return

    if not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError(f"Local save failed: {tmp}")
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    rc = subprocess.call(["cp", tmp, path])
    if rc != 0:
        print(f"[WARN] cp {tmp} -> {path} failed (rc={rc}); backup kept at {tmp}")
    else:
        os.remove(tmp)


def load_latent_stats(stats_path: str, device):
    """Load mean/std + optional full-covariance whitening matrices."""
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float().to(device).reshape(1, -1, 1, 1)
    std = stats["std"].float().to(device).reshape(1, -1, 1, 1).clamp(min=1e-5)
    whiten = stats["whiten"].float().to(device) if "whiten" in stats else None
    unwhiten = stats["unwhiten"].float().to(device) if "unwhiten" in stats else None
    return mean, std, whiten, unwhiten


def normalize_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                     whiten: torch.Tensor | None) -> torch.Tensor:
    z_centered = z - mean
    if whiten is None:
        return z_centered / std
    return torch.einsum("ij,bjhw->bihw", whiten, z_centered)


def denormalize_latent(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                       unwhiten: torch.Tensor | None) -> torch.Tensor:
    if unwhiten is None:
        return y * std + mean
    return torch.einsum("ij,bjhw->bihw", unwhiten, y) + mean


def normalise_c2w_for_batch(
    extrinsic: torch.Tensor,
    device,
    origin_idx: int = -1,
    translation_norm: str = "batch_max",
):
    c2w = extrinsic.float()
    if c2w.shape[-2:] == (3, 4):
        pad = torch.zeros(c2w.shape[:-2] + (1, 4), device=device, dtype=c2w.dtype)
        pad[..., 3] = 1.0
        c2w = torch.cat([c2w, pad], dim=-2)
    ref_inv = torch.linalg.inv(c2w[:, origin_idx])
    c2w = ref_inv.unsqueeze(1) @ c2w
    if translation_norm in ("none", "identity", ""):
        return c2w
    if translation_norm != "batch_max":
        raise ValueError(
            "translation_norm must be 'batch_max' or 'none', "
            f"got {translation_norm!r}"
        )
    t_vec = c2w[:, :, :3, 3]
    farthest = t_vec.abs().amax(dim=1).amax(dim=1, keepdim=True)
    scale = 1.0 / (farthest + 1e-8)
    c2w[:, :, :3, 3] = c2w[:, :, :3, 3] * scale.unsqueeze(1)
    return c2w


def intrinsic_to_K(intrinsic: torch.Tensor, device):
    intr = intrinsic.float()
    if intr.shape[-1] == 4 and intr.dim() >= 2:
        fx, fy, cx, cy = intr.unbind(dim=-1)
        z_ = torch.zeros_like(fx)
        o_ = torch.ones_like(fx)
        return torch.stack([
            torch.stack([fx, z_, cx], dim=-1),
            torch.stack([z_, fy, cy], dim=-1),
            torch.stack([z_, z_, o_], dim=-1),
        ], dim=-2)
    return intr


def _cache_to_local(path: str) -> str:
    if not path or not os.path.isfile(path):
        return path
    if path.startswith("/local-ssd"):
        return path
    if not os.path.isdir("/local-ssd"):
        return path
    cache_dir = "/local-ssd/_train_cache"
    os.makedirs(cache_dir, exist_ok=True)
    parent = os.path.basename(os.path.dirname(path))
    src_key = os.path.abspath(path)
    digest = hashlib.sha1(src_key.encode("utf-8")).hexdigest()[:12]
    base = os.path.basename(path)
    stem, suffix = os.path.splitext(base)
    local = os.path.join(cache_dir, f"{parent}_{stem}_{digest}{suffix}")
    src_size = os.path.getsize(path)
    try:
        if os.path.isfile(local) and os.path.getsize(local) == src_size:
            print(f"  [cache] Using {local}")
            return local
    except OSError:
        pass
    import shutil
    tmp = os.path.join(cache_dir, f".staging_{os.getpid()}_{parent}_{stem}_{digest}{suffix}")
    print(f"  [cache] {path} -> {local} ...", end=" ", flush=True)
    shutil.copy2(path, tmp)
    if os.path.getsize(tmp) != src_size:
        raise RuntimeError(
            f"cached file size mismatch: got {os.path.getsize(tmp)}, expected {src_size}"
        )
    os.replace(tmp, local)
    sz = os.path.getsize(local)
    print(f"done ({sz/1e6:.0f} MB)" if sz < 1e9 else f"done ({sz/1e9:.1f} GB)")
    return local


class AsyncCheckpointSaver:
    """Non-blocking checkpoint save so DDP ranks are not blocked on FUSE I/O."""

    def __init__(self, logger):
        self.logger = logger
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def _wait_previous(self, timeout: float = 600.0) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            self.logger.warning(
                f"[ckpt] previous save thread ({thread.name}) still running "
                f"after {timeout:.0f}s; forcing new save (prior may be incomplete)"
            )
        self._thread = None

    @staticmethod
    def _snapshot_state_dict(sd) -> dict:
        return {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
            for k, v in sd.items()
        }

    @staticmethod
    def _snapshot_optimizer(opt) -> dict:
        sd = opt.state_dict()
        out_state = {}
        for pid, st in sd.get("state", {}).items():
            out_state[pid] = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in st.items()
            }
        return {"state": out_state, "param_groups": sd.get("param_groups", [])}

    def save(self, path, step, epoch, model, ema, optimizer, scheduler=None):
        with self._lock:
            self._wait_previous()
            t0 = time()
            model_mod = model.module if hasattr(model, "module") else model
            snapshot = {
                "model": self._snapshot_state_dict(model_mod.state_dict()),
                "ema": self._snapshot_state_dict(ema.state_dict()),
                "opt": self._snapshot_optimizer(optimizer),
                "train_steps": step,
                "epoch": epoch,
            }
            if scheduler is not None:
                snapshot["scheduler"] = scheduler.state_dict()
            snap_ms = (time() - t0) * 1000.0

            def _worker():
                t1 = time()
                try:
                    _safe_torch_save(snapshot, path)
                    dt = time() - t1
                    self.logger.info(
                        f"[ckpt] step={step:07d} background save done in "
                        f"{dt:.1f}s → {path}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"[ckpt] step={step:07d} background save FAILED: {e}"
                    )

            self._thread = threading.Thread(
                target=_worker, daemon=True, name=f"ckpt-{step:07d}",
            )
            self._thread.start()
            self.logger.info(
                f"[ckpt] step={step:07d} snapshot done in {snap_ms:.0f}ms; "
                f"background save started"
            )

    def join(self, timeout: float = 600.0) -> None:
        self._wait_previous(timeout=timeout)


def _setup_distributed_with_timeout(watchdog_minutes: int = 15):
    import datetime
    minutes = int(os.environ.get("NCCL_PG_TIMEOUT_MIN", watchdog_minutes))
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if rank == 0:
        print(f"  [dist] NCCL collective timeout = {minutes} min "
              f"(override via NCCL_PG_TIMEOUT_MIN)")
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(minutes=minutes),
    )
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device("cuda", local_rank)
