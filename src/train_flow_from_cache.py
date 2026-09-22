"""NOT A PUBLIC ENTRY POINT. Use scripts/train/train_flow.py.
Kept so historical helpers remain importable; prefer src/utils/train_runtime.py.


Flow trainer — Token-Concat conditioning, **precomputed latent** path.

Differences vs the standard flow trainer (``scripts/train/train_flow.py``):

  * No DA3 (RAE) loaded.
  * No FeatureVAE loaded (training does NOT need encoder/decoder).
  * No text encoder (ScanNet++ chunks carry empty captions).
  * Dataset (`ScanNetppLatent_Multi`) emits batches already containing
    `z_all`, `z_ref`, `c2w`, `intrinsics`. We skip `prepare_data` entirely.
  * Long sequence: trained with V=64 (configurable). All other DiT machinery
    is unchanged.

Why this saves time:
  Training step previously paid two DA3 forwards (all-V and refs-only) plus
  two VAE encode passes per batch. With precomputed latents we only run
  DiT forward + backward → ~3-5× faster per step and frees ~10-30 GB of
  activation memory (depending on DA3 size + V).

Usage:
    torchrun --nproc_per_node=N src/train_flow_from_cache.py \
        --config configs/flow_gae128.yaml \
        --results-dir results/flow-precomputed
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import random
import sys
import tempfile
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

# HF cache → /local-ssd (S3-FUSE write-once is OK but slow; small models cached locally)
if os.path.isdir("/local-ssd"):
    os.makedirs("/local-ssd/hf_cache", exist_ok=True)
    os.environ["HF_HOME"] = "/local-ssd/hf_home"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/local-ssd/hf_cache"
else:
    os.environ.setdefault("HF_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/xdg-cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.backends.cuda
if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
    torch.backends.cuda.is_flash_attention_available = lambda: False

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch.distributed as dist
from einops import rearrange
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import OmegaConf

import torch.nn.functional as F

from stage2.transport import create_transport
from stage2.transport.transport_v4 import TransportV4Wrapper
from stage2.models.camera import compute_plucker_6d_per_token
from utils.model_utils import instantiate_from_config

try:
    import wandb  # noqa: F401
    from utils import wandb_utils
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--results-dir", type=str,
                   default="results/gae-flow-precomputed")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Resume checkpoint (model+optimizer+scheduler+step).")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Init checkpoint (model weights only, training starts from step 0).")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


# ── Distributed ───────────────────────────────────────────────────────────


def setup_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_logger(log_dir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(__name__)
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
        import subprocess
        try:
            remote = logger._log_remote_path
            try:
                os.remove(remote)
            except FileNotFoundError:
                pass
            subprocess.call(["cp", logger._log_local_path, remote])
        except OSError:
            pass


# ── EMA / Checkpoint helpers (S3-FUSE safe) ───────────────────────────────


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
    import subprocess
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


def save_checkpoint(path, step, epoch, model, ema, optimizer, scheduler=None):
    state = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "ema": ema.state_dict(),
        "opt": optimizer.state_dict(),
        "train_steps": step,
        "epoch": epoch,
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    _safe_torch_save(state, path)


def load_latent_stats(stats_path: str, device):
    """Load mean/std + optional full-covariance whitening matrices.

    Returns:
        mean:     (1, C, 1, 1)
        std:      (1, C, 1, 1)      — used in per-channel fallback only
        whiten:   (C, C) or None    — Σ^(-1/2); when present, used instead of std
        unwhiten: (C, C) or None    — Σ^(+1/2); needed at sampling time
    """
    stats = torch.load(stats_path, map_location="cpu")
    mean = stats["mean"].float().to(device).reshape(1, -1, 1, 1)
    std = stats["std"].float().to(device).reshape(1, -1, 1, 1).clamp(min=1e-5)
    whiten = stats["whiten"].float().to(device) if "whiten" in stats else None
    unwhiten = stats["unwhiten"].float().to(device) if "unwhiten" in stats else None
    return mean, std, whiten, unwhiten


def load_frozen_vae(vae_ckpt_path: str, vae_cfg: dict, device):
    """Load a frozen GAECodec (feature-decode branch only) for REPA target.

    Loads the feature-decode branch only. rgb_decoder is dropped
    from the config so no RGBHead is built (lighter); strict=False tolerates the
    missing rgb_head keys in the checkpoint.
    """
    from stage1.gae_codec import GAECodec

    vae_cfg = dict(vae_cfg)
    vae_cfg.pop("rgb_decoder", None)  # feature branch only; skip RGBHead
    vae = GAECodec(**vae_cfg).to(device)
    ckpt = torch.load(vae_ckpt_path, map_location="cpu")
    sd = ckpt.get("ema_vae", ckpt.get("vae", ckpt.get("model", ckpt)))
    vae.load_state_dict(sd, strict=False)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def normalize_latent(z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                     whiten: torch.Tensor | None) -> torch.Tensor:
    """Map raw latent z → whitened y (unit-cov if `whiten` provided)."""
    z_centered = z - mean
    if whiten is None:
        return z_centered / std
    return torch.einsum("ij,bjhw->bihw", whiten, z_centered)


def denormalize_latent(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                       unwhiten: torch.Tensor | None) -> torch.Tensor:
    """Inverse of `normalize_latent`. Used in sampling."""
    if unwhiten is None:
        return y * std + mean
    return torch.einsum("ij,bjhw->bihw", unwhiten, y) + mean


# ── Pose helpers (kept identical to v4) ──────────────────────────────────


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


# ── Cache files to local-ssd ─────────────────────────────────────────────


class LazyTextEmbDir:
    """Lazy per-key UMT5 embedding store: ``<root>/<caption_key>.pt`` loaded on
    demand with a bounded LRU cache.

    Why: the full OSP cache is ~463 GB (110k × (512,4096) bf16) — far above the
    container memory limit, so it cannot be held in RAM as one dict (see
    ``scripts/extract_osp_caption_embeddings.py``). Each training step only needs
    the captions in its batch, so we read those files lazily and cache the most
    recent ``cache_size`` tensors (chunks of one video repeat their caption, so
    the LRU hit-rate is high). RAM stays at ``cache_size × 4 MB``.
    """

    def __init__(self, root: str, cache_size: int = 4096):
        self.root = root
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, key: str):
        if not key:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        path = os.path.join(self.root, f"{key}.pt")
        if not os.path.isfile(path):
            return None
        try:
            emb = torch.load(path, map_location="cpu")
        except Exception:
            return None
        self._cache[key] = emb
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return emb


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
    # Reuse only when the cached copy is COMPLETE (size matches source). A copy
    # killed mid-flight (e.g. an NCCL-watchdog abort during startup) leaves a
    # truncated file; the old `if not isfile` check would silently "Use" it and
    # torch.load then fails / loads garbage.
    try:
        if os.path.isfile(local) and os.path.getsize(local) == src_size:
            print(f"  [cache] Using {local}")
            return local
    except OSError:
        pass
    import shutil
    import subprocess
    # Atomic: stage to a pid-tagged temp then rename, so concurrent ranks and
    # readers never observe a partial file at `local`.
    tmp = os.path.join(cache_dir, f".staging_{os.getpid()}_{parent}_{stem}_{digest}{suffix}")
    print(f"  [cache] {path} -> {local} ...", end=" ", flush=True)
    # Optional fast path: if the cache lives on an s3-backed FUSE mount, set
    # GAE_CACHE_S3_MOUNT=/your/mount and GAE_CACHE_S3_BUCKET=s3://bucket/prefix
    # to stage via s5cmd instead of a slow FUSE copy. Defaults to a plain copy.
    _s3_mount = (os.environ.get("GAE_CACHE_S3_MOUNT") or "").rstrip("/")
    _s3_bucket = (os.environ.get("GAE_CACHE_S3_BUCKET") or "").rstrip("/")
    if _s3_mount and _s3_bucket and path.startswith(_s3_mount + "/") and shutil.which("s5cmd"):
        rel = path[len(_s3_mount) + 1:]
        s3_path = f"{_s3_bucket}/{rel}"
        subprocess.run(["s5cmd", "cp", s3_path, tmp], check=True)
    else:
        shutil.copy2(path, tmp)
    if os.path.getsize(tmp) != src_size:
        raise RuntimeError(
            f"cached file size mismatch: got {os.path.getsize(tmp)}, expected {src_size}"
        )
    os.replace(tmp, local)
    sz = os.path.getsize(local)
    print(f"done ({sz/1e6:.0f} MB)" if sz < 1e9 else f"done ({sz/1e9:.1f} GB)")
    return local


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    cfg = OmegaConf.load(args.config)

    def to_dict(section):
        if section is None:
            return {}
        return OmegaConf.to_container(section, resolve=True) if not isinstance(section, dict) else section

    model_cfg = cfg.get("stage_2")
    transport_cfg = to_dict(cfg.get("transport", {}).get("params", {}))
    training_cfg = to_dict(cfg.get("training", {}))
    validation_cfg = to_dict(cfg.get("validation", {}))
    dataset_cfg = to_dict(cfg.get("dataset", {}))

    # ── Training hyper-params ──
    num_epochs = int(training_cfg.get("epochs", 100))
    batch_size = int(training_cfg.get("batch_size", 1))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 1))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    seed = int(training_cfg.get("global_seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    camera_drop = float(training_cfg.get("camera_drop", 0.1))
    ref_drop_prob = float(training_cfg.get("ref_drop_prob", 0.0))
    pose_origin = str(training_cfg.get("pose_origin", "first"))
    pose_origin_idx = 0 if pose_origin == "first" else -1

    # ── V / cond_num (forced single for precomputed) ──
    top_num_views = cfg.get("num_views", None)
    ds_num_views = dataset_cfg.get("num_views", None)
    if top_num_views is not None:
        total_view = int(top_num_views)
        if ds_num_views is not None and int(ds_num_views) != total_view:
            raise ValueError(
                f"num_views mismatch: top-level={total_view}, dataset={ds_num_views}."
            )
    elif ds_num_views is not None:
        total_view = int(ds_num_views)
    else:
        total_view = 64

    # Precomputed chunks store z_ref only for frame 0, so cond_num is fixed:
    #   cond_num=1 → single-ref conditioning (ScanNet++ pose-conditioned path)
    #   cond_num=0 → pure T2V (OSP camera-aware T2V: text + Plücker, no ref latent)
    cond_num_raw = dataset_cfg.get("cond_num", 1)
    cond_num_cfg = int(cond_num_raw)
    if cond_num_cfg not in (0, 1):
        raise ValueError(
            f"Precomputed mode supports cond_num=0 (T2V) or cond_num=1 (single-ref); "
            f"got cond_num={cond_num_raw!r}."
        )
    is_t2v_mode = (cond_num_cfg == 0)

    ckpt_every = int(validation_cfg.get("ckpt_every", 5000))

    base_lr = float(training_cfg.get("base_lr", 1e-4))
    final_lr = float(training_cfg.get("final_lr", 1e-6))
    warmup_steps = int(training_cfg.get("warmup_steps", 5000))
    schedule_type = str(training_cfg.get("schedule_type", "cosine"))
    betas = tuple(training_cfg.get("beta", [0.9, 0.95]))
    wd = float(training_cfg.get("wd", 0.01))
    ref_key_lr_mult = float(training_cfg.get("ref_key_lr_mult", 1.0))

    misc_cfg = to_dict(cfg.get("misc", {}))
    model_params = to_dict(model_cfg.get("params", {}))
    in_ch = int(model_params.get("in_channels", 128))
    # Latent spatial inferred from chunk shape at first batch, but DiT cfg also has it implicit.
    # For the time_dist shift we just use a representative shape (504/14)².
    latent_h = 504 // 14
    latent_w = latent_h
    latent_size = (in_ch, latent_h, latent_w)
    # DiT 在构造期就要知道 latent 空间尺寸（RoPE / pos-embed 用 num_patches）。
    # precomputed latent 固定 504 → DA3 patch 14 → 36×36。配置没显式 pin 时按此注入，
    # 否则保持配置值（单一真相源）。漏注入会让 input_size 落到默认 1，
    # 与 patch_size>1 相乘得 num_patches=0，RoPE 构造空张量直接崩。
    if "input_size" not in model_cfg["params"]:
        model_cfg["params"]["input_size"] = latent_h
    if "time_dist_shift" in misc_cfg:
        time_dist_shift = float(misc_cfg["time_dist_shift"])
    else:
        shift_dim = misc_cfg.get("time_dist_shift_dim", math.prod(latent_size))
        shift_base = misc_cfg.get("time_dist_shift_base", 4096)
        time_dist_shift = math.sqrt(shift_dim / shift_base)

    # ── Experiment dir ──
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        idx = len(glob(f"{args.results_dir}/*"))
        h_enc = model_params.get("encoder_hidden_size", 768)
        h_dec = model_params.get("hidden_size", 2048)
        if isinstance(h_dec, list):
            h_enc, h_dec = h_dec[0], h_dec[1]
        d_enc = model_params.get("depth", 28)
        d_dec = model_params.get("decoder_depth", 6)
        if isinstance(d_enc, list):
            d_enc, d_dec = d_enc[0], d_enc[1]
        exp_name = (f"{idx:03d}-LatentDDTv4P-enc{h_enc}-dec{h_dec}"
                    f"-d{d_enc}+{d_dec}-v{total_view}-{args.precision}")
        exp_dir = os.path.join(args.results_dir, exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        exp_dir = ckpt_dir = exp_name = None

    logger = create_logger(exp_dir)
    logger.info(f"Experiment: {exp_dir}")
    logger.info(f"Precomputed-latent mode: NO DA3, NO VAE, NO text encoder loaded.")

    # ── Latent stats (mandatory; the chunks store un-normalised mu) ──
    latent_mean, latent_std, latent_whiten, latent_unwhiten = None, None, None, None
    stats_path = cfg.get("latent_stats")
    if stats_path and os.path.isfile(str(stats_path)):
        if device.type == "cuda" and (device.index or 0) == 0:
            _cache_to_local(str(stats_path))
        if dist.is_initialized():
            dist.barrier()
        local_stats = _cache_to_local(str(stats_path))
        latent_mean, latent_std, latent_whiten, latent_unwhiten = load_latent_stats(
            local_stats, device)
        if latent_whiten is not None:
            logger.info(
                f"Latent stats: FULL-COV whitening enabled "
                f"(W shape={tuple(latent_whiten.shape)}, "
                f"mean avg={latent_mean.mean():.4f}, std avg={latent_std.mean():.4f})")
        else:
            logger.info(
                f"Latent stats: per-channel only (legacy) "
                f"mean={latent_mean.mean():.4f}  std={latent_std.mean():.4f}")
    else:
        logger.warning("No latent_stats — training without latent normalization! "
                       "Run scripts/compute_latent_stats_full_cov.py first.")

    # ── Text conditioning (pre-encoded UMT5 cache; T2V path only) ──
    # Gated on cfg.use_text → ScanNet++ (no use_text) path is unchanged.
    use_text = bool(cfg.get("use_text", False))
    text_cfg = to_dict(cfg.get("text_encoder", {})) if use_text else {}
    text_drop_prob = float(text_cfg.get("text_drop_prob", 0.1)) if text_cfg else 0.0
    # text_store exposes ``.get(key) -> Tensor | None``; it is either a per-key
    # lazy dir loader (preferred, bounded RAM) or a legacy in-RAM dict.
    text_store = None
    has_text = False
    text_embeddings_path = str(text_cfg.get("embeddings_path", "")) if text_cfg else ""
    if use_text and text_embeddings_path and os.path.isdir(text_embeddings_path):
        # Per-key layout (scripts/extract_osp_caption_embeddings.py): lazy read,
        # no 463 GB whole-file load / local-ssd copy.
        text_store = LazyTextEmbDir(text_embeddings_path)
        has_text = True
        if rank == 0:
            logger.info(
                f"Text embeddings (per-key lazy dir): {text_embeddings_path}, "
                f"drop_prob={text_drop_prob}, lru={text_store.cache_size}")
    elif use_text and text_embeddings_path and os.path.isfile(text_embeddings_path):
        # Legacy single-dict cache (small datasets only — held fully in RAM).
        if device.type == "cuda" and (device.index or 0) == 0:
            _cache_to_local(text_embeddings_path)
        if dist.is_initialized():
            dist.barrier()
        local_text = _cache_to_local(text_embeddings_path)
        raw = torch.load(local_text, map_location="cpu")
        text_store = (raw["caption_to_emb"]
                      if isinstance(raw, dict) and "caption_to_emb" in raw else raw)
        has_text = True
        if rank == 0:
            meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
            logger.info(
                f"Text embeddings loaded from {local_text}: "
                f"{len(text_store)} captions, drop_prob={text_drop_prob}, meta={meta}")
    elif use_text:
        raise FileNotFoundError(
            f"use_text=true but text embeddings cache not found: {text_embeddings_path!r} "
            "(expected a per-key dir or legacy .pt). "
            "Run scripts/extract_osp_caption_embeddings.py first.")

    def get_text_embeddings(captions) -> torch.Tensor | None:
        """Look up pre-encoded UMT5 embeddings by caption_key. Returns None on
        full miss/empty so the sample trains unconditionally (CFG)."""
        if not has_text or captions is None or text_store is None:
            return None
        if isinstance(captions, str):
            captions = [captions]
        cleaned = [c.strip() if isinstance(c, str) else "" for c in captions]
        if all(c == "" for c in cleaned):
            return None
        embs = []
        for c in cleaned:
            emb = text_store.get(c)
            if emb is None:
                return None
            embs.append(emb.to(device=device, non_blocking=True))
        return torch.stack(embs, dim=0)

    # ── REPA (opt-in): frozen VAE decodes the clean latent → DA3 semantic
    # target; an intermediate encoder layer is aligned to it (cosine). The
    # `repa:` block is the single source of truth — structural params are
    # injected into stage_2.params so the model builds its projector. ──
    repa_cfg = to_dict(cfg.get("repa", {}))
    repa_enable = bool(repa_cfg.get("enable", False))
    repa_weight = float(repa_cfg.get("weight", 0.5))
    repa_target_level = int(repa_cfg.get("target_level", -1))
    repa_vae = None
    repa_level_dim = 0
    if repa_enable:
        vae_cfg = to_dict(cfg.get("codec", {}))
        repa_vae_ckpt = str(repa_cfg.get("vae_checkpoint", ""))
        if not vae_cfg:
            raise ValueError("repa.enable=true but `codec` config is missing.")
        if not (repa_vae_ckpt and os.path.isfile(repa_vae_ckpt)):
            raise FileNotFoundError(
                f"repa.enable=true but repa.vae_checkpoint not found: {repa_vae_ckpt!r}")
        repa_vae = load_frozen_vae(repa_vae_ckpt, vae_cfg, device)
        repa_level_dim = int(repa_vae._level_dim)
        # Inject structural params so TokenConcatDDT builds the matching projector.
        model_cfg["params"]["repa_enable"] = True
        model_cfg["params"]["repa_align_depth"] = int(repa_cfg.get("align_depth", 8))
        model_cfg["params"]["repa_target_dim"] = repa_level_dim
        model_cfg["params"]["repa_proj_hidden"] = int(repa_cfg.get("proj_hidden", 2048))
        logger.info(
            f"REPA enabled: target=DA3 level {repa_target_level} (dim={repa_level_dim}), "
            f"align_depth={model_cfg['params']['repa_align_depth']}, weight={repa_weight}, "
            f"VAE={repa_vae_ckpt}")

    # ── Trainable: TokenConcatDDT ──
    dit = instantiate_from_config(model_cfg).to(device)
    ema_dit = deepcopy(dit).eval().requires_grad_(False)

    train_steps = 0
    start_epoch = 0

    ddp_dit = DDP(dit, device_ids=[device.index], broadcast_buffers=False,
                  gradient_as_bucket_view=False,
                  find_unused_parameters=True) if dist.is_initialized() else dit

    n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    logger.info(f"TokenConcatDDT params: {n_params / 1e6:.2f}M  "
                f"in_ch={dit.in_channels}  hidden={dit.encoder_hidden_size}/{dit.decoder_hidden_size}  "
                f"use_cross_attn={dit.use_cross_attn}")
    logger.info(f"Multiview: V={total_view}  cond_num={cond_num_cfg} (fixed)  "
                f"mode={'T2V' if is_t2v_mode else 'single-ref'}  "
                f"camera_drop={camera_drop}  ref_drop_prob={ref_drop_prob}  "
                f"pose_origin={pose_origin}  has_text={has_text}")

    transport = create_transport(**transport_cfg, time_dist_shift=time_dist_shift)
    transport_v4 = TransportV4Wrapper(time_dist_shift=time_dist_shift)
    logger.info(f"Transport: {transport_cfg} (V4 wrapper enabled)")

    # ── Optimizer: ref_key_scale_param gets its own group ──
    ref_key_params, main_params = [], []
    for name, p in dit.named_parameters():
        if not p.requires_grad:
            continue
        if "ref_key_scale_param" in name:
            ref_key_params.append(p)
        else:
            main_params.append(p)
    param_groups = [
        {"params": main_params, "lr": base_lr, "weight_decay": wd,
         "lr_mult": 1.0, "name": "main"},
    ]
    if ref_key_params:
        param_groups.append({
            "params": ref_key_params, "lr": base_lr * ref_key_lr_mult,
            "weight_decay": 0.0, "lr_mult": ref_key_lr_mult, "name": "ref_key",
        })
    logger.info(
        f"Optimizer groups: main={sum(p.numel() for p in main_params):,} @ lr={base_lr:.2e}, "
        f"ref_key={sum(p.numel() for p in ref_key_params):,} @ lr={base_lr*ref_key_lr_mult:.2e}"
    )
    optimizer = torch.optim.AdamW(param_groups, betas=betas)

    ac_kwargs = (dict(device_type="cuda", enabled=True, dtype=torch.bfloat16)
                 if args.precision == "bf16" else dict(device_type="cuda", enabled=False))

    # ── Dataset / loader ──
    from cut3r_data import get_data_loader
    train_loader = get_data_loader(
        cfg.train_dataset, batch_size=batch_size, num_workers=num_workers,
        pin_mem=True, shuffle=True, drop_last=True,
        fixed_length=True, world_size=world_size, rank=rank,
    )
    steps_per_epoch = len(train_loader) // grad_accum_steps
    loader_batches = len(train_loader)
    nominal_total_steps = num_epochs * steps_per_epoch
    decay_end_steps = int(training_cfg.get("decay_end_steps", nominal_total_steps))
    decay_end_steps = max(decay_end_steps, warmup_steps + 1)
    logger.info(f"Train: {steps_per_epoch} steps/epoch ({loader_batches} micro-batches), "
                f"bs={batch_size}/GPU, world={world_size}, grad_accum={grad_accum_steps}")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if schedule_type == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(decay_end_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return max(final_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"LR: base={base_lr}, final={final_lr}, warmup={warmup_steps}, "
                f"schedule={schedule_type}, decay_end_steps={decay_end_steps}")

    # ── Resume ──
    # RoPE 的 freqs_cos / freqs_sin 是 deterministic buffer（__init__ 时按
    # 当前 config 重算），把它们从 ckpt state_dict 里剔除可以兼容下列情况：
    #   - 旧 ckpt 用更小 pt_seq_len 训练（buffer shape 不同 → 否则 size mismatch
    #     会让 strict=False 也无法加载）
    #   - 当前 model 已用 persistent=False，旧 ckpt 的这些 key 会变成 unexpected
    # 也覆盖 temporal RoPE 的 inv_freq（同性质，本来就 persistent=False，写在这
    # 里只是兜底以防其它分支保存了 persistent 版本）。
    def _strip_rope_buffers(sd):
        if sd is None:
            return sd
        return {
            k: v for k, v in sd.items()
            if not (k.endswith("freqs_cos") or k.endswith("freqs_sin")
                    or k.endswith("inv_freq"))
        }

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        dit_mod = ddp_dit.module if isinstance(ddp_dit, DDP) else ddp_dit
        model_sd = _strip_rope_buffers(ckpt["model"])
        missing, unexpected = dit_mod.load_state_dict(model_sd, strict=False)
        if missing and rank == 0:
            logger.info(f"Checkpoint missing keys ({len(missing)}): "
                        f"{missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected and rank == 0:
            logger.info(f"Checkpoint unexpected keys ({len(unexpected)}): "
                        f"{unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        ema_dit.load_state_dict(_strip_rope_buffers(ckpt["ema"]), strict=False)
        try:
            optimizer.load_state_dict(ckpt["opt"])
        except Exception as e:
            if rank == 0:
                logger.warning(f"Could not load optimizer state: {e}")
                logger.info("Optimizer starts fresh for new parameters.")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        train_steps = int(ckpt.get("train_steps", 0))
        start_epoch = train_steps // steps_per_epoch
        if "scheduler" not in ckpt and train_steps > 0:
            for _ in range(train_steps):
                scheduler.step()
        logger.info(f"Resumed from {args.ckpt} (step={train_steps}, epoch={start_epoch})")
    elif args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        dit_mod = ddp_dit.module if isinstance(ddp_dit, DDP) else ddp_dit
        model_sd = _strip_rope_buffers(ckpt["model"])
        missing, unexpected = dit_mod.load_state_dict(model_sd, strict=False)
        if missing and rank == 0:
            logger.info(f"Init missing keys ({len(missing)}): "
                        f"{missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected and rank == 0:
            logger.info(f"Init unexpected keys ({len(unexpected)}): "
                        f"{unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        if "ema" in ckpt:
            ema_dit.load_state_dict(_strip_rope_buffers(ckpt["ema"]), strict=False)
        logger.info(f"Initialized weights from {args.init_ckpt} (training from step 0)")

    if rank == 0 and args.wandb and HAS_WANDB:
        entity = os.environ.get("ENTITY", "gae")
        project = os.environ.get("PROJECT", "GAEFlowPrecomputed")
        wandb_utils.initialize(args, entity, exp_name, project)

    # ── Training loop ──
    running_loss = 0.0
    running_repa = 0.0
    log_steps = 0
    start_time = time()

    cond_num = cond_num_cfg  # fixed in precomputed mode (0=T2V, 1=single-ref)

    for epoch in range(start_epoch, num_epochs):
        ddp_dit.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, total=loader_batches,
                    desc=f"Epoch {epoch}", disable=(rank != 0))

        accum_counter = 0
        accum_loss = 0.0
        accum_repa = 0.0
        optimizer.zero_grad()

        for batch in pbar:
            if not isinstance(batch, dict) or "z_all" not in batch or "z_ref" not in batch:
                got = list(batch) if isinstance(batch, dict) else type(batch).__name__
                raise RuntimeError(
                    f"Precomputed trainer requires a dict batch with z_all/z_ref keys (got {got}). "
                    "Use ScanNetppLatent_Multi as the dataset."
                )

            # Move chunks to GPU
            z_all = batch["z_all"].to(device, non_blocking=True)            # (B, V, C, h, w) bf16
            z_ref = batch["z_ref"].to(device, non_blocking=True)            # (B, 1, C, h, w) bf16
            c2w = batch["c2w"].to(device, non_blocking=True)                # (B, V, 4, 4)
            intrinsics = batch["intrinsics"].to(device, non_blocking=True)  # (B, V, 3, 3)

            B_batch, V_batch = z_all.shape[0], z_all.shape[1]
            if V_batch != total_view:
                raise ValueError(
                    f"Config/data mismatch: total_view={total_view} but chunk V={V_batch}."
                )

            # Flatten BxV for VAE-latent convention used by the rest of the pipeline.
            z_all_flat = rearrange(z_all, "b v c h w -> (b v) c h w").float()
            z_ref_flat = rearrange(z_ref, "b cn c h w -> (b cn) c h w").float()

            # REPA target is decoded from the *un-whitened* latent (VAE's native
            # space), so snapshot before normalization.
            z_all_repa = z_all_flat.clone() if repa_enable else None

            # Normalize: full-cov whitening (y = Σ^(-1/2)(z - μ)) when available,
            # otherwise per-channel (legacy). Chunks store un-normalized μ.
            if latent_mean is not None:
                z_all_flat = normalize_latent(z_all_flat, latent_mean, latent_std, latent_whiten)
                z_ref_flat = normalize_latent(z_ref_flat, latent_mean, latent_std, latent_whiten)

            H_pixels = z_all_flat.shape[-2] * 14  # latent h * DINOv2 patch (14px/token)
            W_pixels = z_all_flat.shape[-1] * 14

            # Plücker is computed at the DiT token grid (= latent / patch_size),
            # NOT the raw latent grid — required once patch_size > 1.
            s_patch = int(dit.s_patch_size)
            tok_y = z_all_flat.shape[-2] // s_patch
            tok_x = z_all_flat.shape[-1] // s_patch

            with autocast(**ac_kwargs):
                c2w_norm = normalise_c2w_for_batch(c2w, device, origin_idx=pose_origin_idx)
                Ks = intrinsic_to_K(intrinsics, device)

                with torch.amp.autocast("cuda", enabled=False):
                    plucker_6d = compute_plucker_6d_per_token(
                        c2w=c2w_norm.float(),
                        Ks=Ks.float(),
                        patches_x=tok_x,
                        patches_y=tok_y,
                        image_height=H_pixels,
                        image_width=W_pixels,
                        is_c2w=True,
                    )

                if camera_drop > 0:
                    drop_mask_b = (torch.rand(B_batch, 1, 1, device=device) > camera_drop).float()
                    plucker_6d = plucker_6d * drop_mask_b.to(plucker_6d.dtype)

                # ── Conditioning: T2V (text + Plücker) vs single-ref (z_ref) ──
                if is_t2v_mode:
                    # Pure T2V: no ref latent. Text is the global condition, with
                    # text_drop_prob for CFG; Plücker (already camera-dropped above)
                    # carries the camera-control signal.
                    batch_cond_num = 0
                    z_ref_for_loss = None
                    ref_global = None
                    if has_text and random.random() > text_drop_prob:
                        ref_global = get_text_embeddings(batch.get("caption", None))
                else:
                    # Single-ref (ScanNet++): existing behavior, no text.
                    ref_global = None
                    if ref_drop_prob > 0 and random.random() < ref_drop_prob:
                        batch_cond_num = 0
                        z_ref_for_loss = None
                    else:
                        batch_cond_num = cond_num
                        if camera_drop > 0:
                            cond_drop = (torch.rand(B_batch, device=device) < camera_drop)
                            if cond_drop.any():
                                z_ref_for_loss = z_ref_flat.clone()
                                for bi in range(B_batch):
                                    if cond_drop[bi]:
                                        z_ref_for_loss[bi:bi + 1] = 0.0
                            else:
                                z_ref_for_loss = z_ref_flat
                        else:
                            z_ref_for_loss = z_ref_flat

                model_kwargs = dict(
                    plucker_6d=plucker_6d,
                    total_view=total_view,
                    cond_num=batch_cond_num,
                    ref_global=ref_global,
                    denoise_all_views=False,
                    z_ref_clean=z_ref_for_loss,
                )

                loss_dict = transport_v4.training_multiview_losses(
                    ddp_dit, x1_target=z_all_flat, model_kwargs=model_kwargs,
                )
                diff_loss = loss_dict["loss"].mean()

                # ── REPA: align encoder layer (self._repa_pred) to the frozen
                # DA3 semantic target decoded from the clean latent ──
                repa_loss = z_all_flat.new_zeros(())
                if repa_enable and dit._repa_pred is not None:
                    with torch.no_grad():
                        feats = repa_vae.decode(z_all_repa)          # (BV, 4*Ld, h, w)
                        Ld = repa_level_dim
                        lvl = repa_target_level % 4
                        tgt = feats[:, lvl * Ld:(lvl + 1) * Ld]      # (BV, Ld, h, w)
                        tgt = F.avg_pool2d(tgt, kernel_size=s_patch, stride=s_patch)
                        tgt = rearrange(tgt, "bv c h w -> bv (h w) c").float()
                    pred = dit._repa_pred.float()                    # (BV, tok, Ld)
                    pred = F.normalize(pred, dim=-1)
                    tgt = F.normalize(tgt, dim=-1)
                    repa_loss = (1.0 - (pred * tgt).sum(-1)).mean()

                loss = (diff_loss + repa_weight * repa_loss) / grad_accum_steps

            if not torch.isfinite(loss):
                logger.warning(f"(step={train_steps:07d}) NaN/Inf loss — skipping")
                optimizer.zero_grad()
                accum_counter = 0
                accum_loss = 0.0
                accum_repa = 0.0
                continue

            loss.backward()
            accum_loss += loss_dict["loss"].mean().item()
            accum_repa += float(repa_loss.detach())
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            if clip_grad:
                grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), clip_grad)
                if not torch.isfinite(grad_norm):
                    logger.warning(f"(step={train_steps:07d}) NaN/Inf gradient — skipping")
                    optimizer.zero_grad()
                    accum_counter = 0
                    accum_loss = 0.0
                    accum_repa = 0.0
                    continue
            else:
                grad_norm = None

            optimizer.step()
            scheduler.step()
            update_ema(ema_dit, dit, ema_decay)
            optimizer.zero_grad()

            running_loss += accum_loss / grad_accum_steps
            running_repa += accum_repa / grad_accum_steps
            accum_counter = 0
            accum_loss = 0.0
            accum_repa = 0.0
            log_steps += 1
            train_steps += 1

            if log_every > 0 and train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if dist.is_initialized():
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss /= world_size

                avg_repa = running_repa / log_steps if repa_enable else 0.0
                cur_lr = optimizer.param_groups[0]["lr"]
                gn_str = f", GradNorm: {grad_norm:.2f}" if grad_norm is not None else ""
                repa_str = f", REPA: {avg_repa:.4f}" if repa_enable else ""
                logger.info(
                    f"(step={train_steps:07d}) Loss: {avg_loss.item():.4f}{repa_str}, "
                    f"LR: {cur_lr:.2e}{gn_str}, Steps/Sec: {steps_per_sec:.2f}"
                )
                if args.wandb and HAS_WANDB:
                    wb = {
                        "tgt_loss": avg_loss.item(),
                        "lr": cur_lr,
                        "train steps/sec": steps_per_sec,
                    }
                    if repa_enable:
                        wb["repa_loss"] = avg_repa
                    if grad_norm is not None:
                        wb["grad_norm"] = (
                            grad_norm.item() if hasattr(grad_norm, "item")
                            else float(grad_norm)
                        )
                    wandb_utils.log(wb, step=train_steps)

                running_loss = 0.0
                running_repa = 0.0
                log_steps = 0
                start_time = time()

            if ckpt_every > 0 and train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    save_checkpoint(
                        f"{ckpt_dir}/{train_steps:07d}.pt",
                        train_steps, epoch, ddp_dit, ema_dit, optimizer, scheduler,
                    )
                    sync_log(logger)
                    logger.info(f"Saved checkpoint at step {train_steps}")
                if dist.is_initialized():
                    dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
