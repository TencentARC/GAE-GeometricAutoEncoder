"""Train the GAE flow model (Stage 2) — paper Sections 3.2-3.3.

The Stage 1 codec is frozen; its posterior mean is standardized per channel and
a single conditional flow model is trained over that state with RAEv2-style
x-prediction. Conditioning is deliberately limited to three controls that
specify the state without evolving with it:

* clean reference latents   ("evidence rather than state", Eq. 14)
* metric Plucker ray maps   (Eq. 13)
* text, via cross-attention

Reference views are encoded in their inference-time context and prepended as
clean tokens at t=0; only generated-view tokens are predicted and integrated.
Text / reference / camera dropout exposes the same weights to T2I, T2V and
reference-conditioned NVS regimes.

``latent_backend`` selects the generated state. ``feature_vae`` is GAE itself;
``da3_direct`` (raw DA3 L0/L3), ``sd_vae`` and ``wan2_1`` are the controlled
comparison latents from the paper's tables.

Usage:
    torchrun --nproc_per_node=8 scripts/train/train_flow.py \
        --config configs/flow_gae64.yaml \
        --results-dir results/gae-64-re10k

Internals: the trainer combines the RAE recipe (x-prediction, Qwen3-0.6B online
text encoding, async checkpointing, NaN-safe steps, resume / init-ckpt
plumbing) with the multi-view video pipeline (multi-view loaders, DA3
multi-view encoding with refs-only context for reference slots, Plucker
computation, camera- and reference-dropout).
"""
from __future__ import annotations

import argparse
import hashlib
import logging  # noqa: F401  (kept so create_logger works exactly as in v4)
import math
import os
import random
import shutil
import sys
import tempfile
from copy import deepcopy
from glob import glob
from time import sleep, time

os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"
if os.path.isdir("/local-ssd"):
    os.makedirs("/local-ssd/hf_cache", exist_ok=True)
    os.environ.setdefault("HF_HOME", "/local-ssd/hf_home")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/local-ssd/hf_cache")
else:
    os.environ.setdefault("HF_HOME", "/tmp/xdg-cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/xdg-cache/huggingface/hub")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import torch.distributed as dist
from diffusers import AutoencoderKL
from diffusers import AutoencoderKLWan
from einops import rearrange
from omegaconf import OmegaConf
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
SRC_DIR = os.path.join(REPO_ROOT, "src")
for _p in (SRC_DIR, ROOT, os.path.join(REPO_ROOT, "scripts", "eval"), os.path.join(REPO_ROOT, "scripts", "train")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from cut3r_data.utils.image import imgnorm_to_unit  # noqa: E402
from stage1.gae_codec import GAECodec  # noqa: E402
from stage1.da3_direct_codec import DA3DirectCodec  # noqa: E402
from stage2.models.camera import compute_plucker_6d_per_token  # noqa: E402
from stage2.models.text_encoder import Qwen3TextEncoder, prefill_hf_cache  # noqa: E402
from stage2.transport.flow import (  # noqa: E402
    shift_from_latent_dim,
    training_losses_rae,
)
from utils.train_runtime import (  # noqa: E402
    AsyncCheckpointSaver,
    _cache_to_local,
    _setup_distributed_with_timeout,
    _strip_rope_buffers,
    _to_dict,
    create_logger,
    denormalize_latent,
    intrinsic_to_K,
    load_latent_stats,
    normalise_c2w_for_batch,
    normalize_latent,
    sync_log,
    update_ema,
)
from utils.build_optim_rae import build_lr_lambda_rae, build_optimizer_rae  # noqa: E402
from utils.geo_aux_loss import (  # noqa: E402
    latent_raw_to_dpt_geometry,
    precompute_gt_dpt_geometry,
)
from utils.model_utils import instantiate_from_config  # noqa: E402
from video.cut3r_adapter import convert_cut3r_batch, resolve_ref_view_sampling  # noqa: E402

try:
    import wandb  # noqa: F401
    from utils import wandb_utils
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--results-dir", type=str,
        default="results/gae-flow",
    )
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Resume checkpoint (model + optimizer + scheduler + step).")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Init checkpoint (model weights only, training starts from step 0).")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


_HF_TO_LOCAL_DA3 = {
    "depth-anything/DA3-LARGE-1.1": "pretrained_models/da3_large",
    "depth-anything/DA3-Large": "pretrained_models/da3_large",
    "depth-anything/DA3-Base": "pretrained_models/da3",
}


def _resolve_da3_encoder_path(pretrained_path, da3_weights_path=None):
    """Map a DA3 HF repo id to a local ``pretrained_models/`` dir when present.

    ``DepthAnything3.from_pretrained`` otherwise hits the HF Hub for every rank;
    on a multi-node job a node with a cold HF cache / flaky egress fails the TLS
    handshake (``httpx.ConnectTimeout``) and aborts the whole run at startup.
    Loading from a local dir (config.json + model.safetensors) is fully offline.
    """
    proj_root = REPO_ROOT
    candidates = []
    if pretrained_path:
        candidates.append(pretrained_path)
    mapped = _HF_TO_LOCAL_DA3.get(pretrained_path)
    if mapped:
        candidates.append(mapped)
    if da3_weights_path:
        par = os.path.dirname(da3_weights_path)
        if par:
            candidates.append(par)
    for raw in candidates:
        probes = [raw] if os.path.isabs(raw) else [
            os.path.join(proj_root, raw), os.path.join(os.getcwd(), raw),
        ]
        for p in probes:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                return p
    return pretrained_path


_HF_TO_LOCAL_TEXT = {
    "Qwen/Qwen3-0.6B": "pretrained_models/qwen3_0.6b",
}


def _resolve_text_model_path(model_name):
    """Map a text-encoder HF repo id to a local dir (with config.json) when
    present, so the tokenizer/model load fully offline.

    Same motivation as the DA3 resolver: ``AutoTokenizer/AutoModel.from_pretrained``
    otherwise hits the HF Hub for every node's local-rank-0, and a cold cache +
    flaky egress fails the TLS handshake (``httpx.ConnectTimeout``) at startup.
    """
    proj_root = REPO_ROOT
    candidates = [model_name]
    mapped = _HF_TO_LOCAL_TEXT.get(model_name)
    if mapped:
        candidates.append(mapped)
    for raw in candidates:
        if not raw:
            continue
        probes = [raw] if os.path.isabs(raw) else [
            os.path.join(proj_root, raw), os.path.join(os.getcwd(), raw),
        ]
        for p in probes:
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                return p
    return model_name


def _cache_dir_to_local_ssd(src_dir, tag, timeout_s: float = 1800.0):
    """Mirror a (flat) pretrained-model dir to node-local /local-ssd once per
    node, returning the local path. Reading a 1.5 GB model from NFS on all 8
    ranks at once is slow and adds contention; copying once per node and reading
    from NVMe is ~8× lighter on NFS.

    Coordination is via a ``.done`` marker + polling — deliberately NOT
    ``dist.barrier()``: a slow NFS copy behind a barrier trips the NCCL watchdog
    and aborts the whole job (observed). Only LOCAL_RANK==0 copies; other ranks
    poll. On any error / timeout we fall back to ``src_dir`` so training still
    proceeds (just reading from NFS).
    """
    if not (src_dir and os.path.isdir(src_dir)) or not os.path.isdir("/local-ssd"):
        return src_dir
    h = hashlib.sha1(os.path.abspath(src_dir).encode()).hexdigest()[:8]
    dst = os.path.join("/local-ssd/_train_cache", f"{tag}_{h}")
    done = os.path.join(dst, ".done")
    files = sorted(
        n for n in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, n))
    )

    def _valid():
        if not os.path.isfile(done):
            return False
        for n in files:
            dp = os.path.join(dst, n)
            if (not os.path.isfile(dp)
                    or os.path.getsize(dp) != os.path.getsize(os.path.join(src_dir, n))):
                return False
        return True

    if _valid():
        return dst

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        try:
            os.makedirs(dst, exist_ok=True)
            if os.path.isfile(done):
                os.remove(done)
            for n in files:
                sp, dp = os.path.join(src_dir, n), os.path.join(dst, n)
                if os.path.isfile(dp) and os.path.getsize(dp) == os.path.getsize(sp):
                    continue
                tmp = f"{dp}.staging_{os.getpid()}"
                shutil.copyfile(sp, tmp)
                os.replace(tmp, dp)
            with open(done, "w") as f:
                f.write("ok")
            print(f"  [cache] {src_dir} -> {dst}", flush=True)
            return dst
        except Exception as e:  # noqa: BLE001
            print(f"  [cache] {tag}: local-ssd copy failed ({e}); using {src_dir}",
                  flush=True)
            return src_dir

    t0 = time()
    while time() - t0 < timeout_s:
        if _valid():
            return dst
        sleep(2.0)
    print(f"  [cache] {tag}: timed out waiting for local-ssd; using {src_dir}",
          flush=True)
    return src_dir


# ── prepare_data: multi-view DA3 + VAE encode (refs-only ctx for ref slots) ──
@torch.no_grad()
def prepare_data(
    rae, vae, images, intrinsic, extrinsic, device,
    cond_num: int = 1,
    latent_mean=None, latent_std=None, latent_whiten=None,
    origin_idx: int = 0,
    translation_norm: str = "batch_max",
    latent_backend: str = "feature_vae",
    sd_vae_scaling: float = 1.0,
    wan_latent_mean=None,
    wan_latent_std=None,
    dataset_cfg=None,
):
    """T2I latent-prep variant using the full-covariance whitening from
    `latent_stats` (T2I path) rather than per-channel mean/std division.

    Returns:
        z_input:  (BV, C, h, w) — ref slots from refs-only DA3 ctx, tgt slots
                  from all-view DA3 ctx. Used to construct z_ref_clean for
                  conditioning.
        z_all_gt: (BV, C, h, w) — all-view DA3 ctx for every view. This is the
                  diffusion target (every view is denoised).
        c2w_norm: (B, V, 4, 4) pose_origin-relative c2w.
        Ks:       (B, V, 3, 3) intrinsics matrix.
    """
    B, V, _, H, W = images.shape

    dataset_cfg = dataset_cfg or {}

    def _encode_vae(imgs_5d):
        flat = rearrange(imgs_5d, "b v c h w -> (b v) c h w")
        if latent_backend == "wan2_1":
            # V4 DiT is view-slot based, so keep one latent per view. Full
            # temporal Wan encoding would compress T and break total_view=V.
            imgs_wan = (flat * 2.0 - 1.0).unsqueeze(2)
            z_chunks = []
            wan_chunk = int(dataset_cfg.get("wan_encode_chunk", 8))
            for ci in range(0, imgs_wan.shape[0], wan_chunk):
                z_c = vae.encode(imgs_wan[ci : ci + wan_chunk]).latent_dist.sample()
                z_chunks.append(z_c.squeeze(2))
            mu = torch.cat(z_chunks, dim=0)
            if wan_latent_mean is not None and wan_latent_std is not None:
                mu = (mu - wan_latent_mean) / wan_latent_std
            return mu
        if latent_backend == "sd_vae":
            imgs_sd = flat * 2.0 - 1.0
            z_chunks = []
            sd_chunk = int(dataset_cfg.get("sd_encode_chunk", 16))
            for ci in range(0, imgs_sd.shape[0], sd_chunk):
                z_c = vae.encode(imgs_sd[ci : ci + sd_chunk]).latent_dist.sample()
                z_chunks.append(z_c * sd_vae_scaling)
            return torch.cat(z_chunks, dim=0)
        if hasattr(vae, "encode_views"):
            return vae.encode_views(imgs_5d)

        # GAECodec path mirrors the original implementation:
        # [0,1] -> DA3-normalized images -> multiview DA3 features -> VAE latent.
        images_norm = (imgs_5d - rae.encoder_mean[None]) / rae.encoder_std[None]
        all_feats = rae.encode(images_norm, mode="all")
        feats = {k: v[:, 1:, :] for k, v in all_feats.items()}
        x_norm = vae.normalize_levels(feats, image_size=(H, W))
        mu, _logvar = vae.encode(x_norm)
        return mu  # (B*V, C, h, w)

    z_all = _encode_vae(images)  # all-view context

    if cond_num > 0:
        z_ref = _encode_vae(images[:, :cond_num])  # ref-only context
    else:
        z_ref = None

    if latent_mean is not None:
        z_all = normalize_latent(z_all, latent_mean, latent_std, latent_whiten)
        if z_ref is not None:
            z_ref = normalize_latent(z_ref, latent_mean, latent_std, latent_whiten)

    if z_ref is not None:
        z_input_5d = rearrange(z_all, "(b v) c h w -> b v c h w", v=V).clone()
        z_ref_5d = rearrange(z_ref, "(b cn) c h w -> b cn c h w", cn=cond_num)
        z_input_5d[:, :cond_num] = z_ref_5d
        z_input = rearrange(z_input_5d, "b v c h w -> (b v) c h w")
    else:
        z_input = z_all

    c2w_norm = normalise_c2w_for_batch(
        extrinsic,
        device,
        origin_idx=origin_idx,
        translation_norm=translation_norm,
    )
    Ks = intrinsic_to_K(intrinsic, device)

    return z_input, z_all, c2w_norm, Ks


def prepare_data_precomputed(
    z_all, z_ref_cn, c2w, intrinsics, device,
    cond_num: int, total_view: int,
    latent_mean=None, latent_std=None, latent_whiten=None,
    origin_idx: int = 0,
    translation_norm: str = "batch_max",
):
    """Mirror ``prepare_data`` using baked ``z_all`` / ``z_ref_k`` tensors."""
    z_all_flat = rearrange(z_all.float(), "b v c h w -> (b v) c h w")

    if cond_num > 0:
        z_ref_flat = rearrange(z_ref_cn.float(), "b cn c h w -> (b cn) c h w")
    else:
        z_ref_flat = None

    if latent_mean is not None:
        z_all_flat = normalize_latent(z_all_flat, latent_mean, latent_std, latent_whiten)
        if z_ref_flat is not None:
            z_ref_flat = normalize_latent(z_ref_flat, latent_mean, latent_std, latent_whiten)

    if z_ref_flat is not None:
        z_input_5d = rearrange(z_all_flat, "(b v) c h w -> b v c h w", v=total_view).clone()
        z_ref_5d = rearrange(z_ref_flat, "(b cn) c h w -> b cn c h w", cn=cond_num)
        z_input_5d[:, :cond_num] = z_ref_5d
        z_input = rearrange(z_input_5d, "b v c h w -> (b v) c h w")
    else:
        z_input = z_all_flat

    c2w_norm = normalise_c2w_for_batch(
        c2w,
        device,
        origin_idx=origin_idx,
        translation_norm=translation_norm,
    )
    Ks = intrinsics.to(device)
    return z_input, z_all_flat, c2w_norm, Ks


def main():
    args = parse_args()
    rank, world_size, device = _setup_distributed_with_timeout(watchdog_minutes=15)

    cfg = OmegaConf.load(args.config)
    # 启动期覆盖 view_choices（顶层键，被各 dataset 的 ${view_choices} 插值引用）。
    # dataset 在下方 build loader 时才解析 train_dataset 字符串，故此处赋值可传播；
    # 所有 dataset 共用同一顶层键，CatDataset 的一致性断言也不受影响。
    # 用法：VIEW_CHOICES="9,17,33,33"（或 "[9,17,33,33]"）作为环境变量前缀。
    _vc_env = os.environ.get("VIEW_CHOICES", "").strip()
    if _vc_env:
        _vc = [int(x) for x in _vc_env.strip("[]() ").replace(" ", "").split(",") if x != ""]
        cfg.view_choices = _vc
        if rank == 0:
            print(f"[cfg] VIEW_CHOICES override -> {_vc}", flush=True)
    model_cfg = cfg.get("stage_2")
    training_cfg = _to_dict(cfg.get("training", {}))
    validation_cfg = _to_dict(cfg.get("validation", {}))
    misc_cfg = _to_dict(cfg.get("misc", {}))
    repa_cfg = _to_dict(cfg.get("repa", {}))
    self_flow_cfg = _to_dict(cfg.get("self_flow", {}))
    motion_loss_cfg = _to_dict(cfg.get("motion_loss", {}))
    text_cfg = _to_dict(cfg.get("text_encoder", {}))
    stage1_cfg = _to_dict(cfg.get("stage_1", {}))
    vae_cfg = _to_dict(cfg.get("codec", {}))
    dataset_cfg = _to_dict(cfg.get("dataset", {}))
    transport_cfg = _to_dict(cfg.get("transport", {}).get("params", {}))
    latent_backend = str(cfg.get("latent_backend", "feature_vae")).lower()
    if latent_backend == "wan21":
        latent_backend = "wan2_1"

    # ── Hyper-params ────────────────────────────────────────────────────
    seed = int(training_cfg.get("global_seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epochs = int(training_cfg.get("epochs", 1000000))
    micro_batch_size = int(training_cfg.get("batch_size", 4))
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    global_batch_size = micro_batch_size * world_size * grad_accum_steps

    ema_decay = float(os.environ.get("EMA_DECAY", training_cfg.get("ema_decay", 0.9995)))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    clip_grad = float(training_cfg.get("clip_grad", 1.0))
    cfg_dropout_prob = float(training_cfg.get("text_drop_prob", 0.1))
    # Optional launcher override. Keep this opt-in so unrelated V4 recipes
    # continue to use their YAML values. The override pins both endpoints;
    # this is the expected meaning for constant-LR fine-tuning recipes.
    _lr_override_raw = os.environ.get("LR_OVERRIDE", "").strip()
    lr_override = float(_lr_override_raw) if _lr_override_raw else None
    if lr_override is not None and (
        not math.isfinite(lr_override) or lr_override <= 0.0
    ):
        raise ValueError(
            f"LR_OVERRIDE must be finite and > 0, got {_lr_override_raw!r}"
        )
    base_lr = (
        lr_override
        if lr_override is not None
        else float(training_cfg.get("base_lr", 1e-4))
    )
    final_lr = (
        lr_override
        if lr_override is not None
        else float(training_cfg.get("final_lr", 1e-6))
    )
    warmup_steps = int(training_cfg.get("warmup_steps", 2000))
    schedule_type = str(training_cfg.get("schedule_type", "cosine"))

    camera_drop = float(training_cfg.get("camera_drop", 0.05))
    ref_drop_prob = float(training_cfg.get("ref_drop_prob", 0.0))
    reference_conditioning = str(
        training_cfg.get("reference_conditioning", "clean_token_v4")
    )
    if reference_conditioning not in ("clean_token_v4", "state_v3"):
        raise ValueError(
            "training.reference_conditioning must be 'clean_token_v4' or "
            f"'state_v3', got {reference_conditioning!r}"
        )
    reference_in_state = reference_conditioning == "state_v3"
    pose_origin = str(training_cfg.get("pose_origin", "first"))
    pose_origin_idx = 0 if pose_origin == "first" else -1
    pose_translation_norm = str(training_cfg.get("pose_translation_norm", "batch_max"))
    if pose_translation_norm not in ("batch_max", "none"):
        raise ValueError(
            "training.pose_translation_norm must be 'batch_max' or 'none', "
            f"got {pose_translation_norm!r}"
        )
    pose_interval_scale = bool(training_cfg.get("pose_interval_scale", False))
    pose_interval_canonical = float(training_cfg.get("pose_interval_canonical", 10.0))
    pose_interval_min_scale = float(training_cfg.get("pose_interval_min_scale", 0.1))
    pose_interval_max_scale = float(training_cfg.get("pose_interval_max_scale", 1.0))
    if pose_interval_canonical <= 0:
        raise ValueError("training.pose_interval_canonical must be > 0")
    if pose_interval_min_scale <= 0 or pose_interval_max_scale <= 0:
        raise ValueError("pose interval scale bounds must be > 0")
    if pose_interval_min_scale > pose_interval_max_scale:
        raise ValueError("pose_interval_min_scale must be <= pose_interval_max_scale")

    rgb_loss_weight = float(os.environ.get(
        "RGB_LOSS_WEIGHT", training_cfg.get("rgb_loss_weight", 0.0),
    ))
    rgb_t_max = float(os.environ.get(
        "RGB_LOSS_T_MAX", training_cfg.get("rgb_loss_t_max", 0.4),
    ))
    rgb_max_views = int(os.environ.get(
        "RGB_LOSS_MAX_VIEWS", training_cfg.get("rgb_loss_max_views", 8),
    ))
    rgb_loss_start_step = int(os.environ.get(
        "RGB_LOSS_START_STEP", training_cfg.get("rgb_loss_start_step", 0),
    ))
    if rgb_t_max <= 0.0 or rgb_t_max > 1.0:
        raise ValueError(f"training.rgb_loss_t_max must be in (0, 1], got {rgb_t_max}")
    if rgb_max_views < 0:
        raise ValueError(f"training.rgb_loss_max_views must be >= 0, got {rgb_max_views}")
    use_rgb_loss = rgb_loss_weight > 0.0
    if use_rgb_loss and latent_backend not in ("feature_vae", "da3_direct"):
        raise ValueError(
            "RGB aux loss is only supported for feature_vae / da3_direct backends"
        )

    geo_loss_weight = float(os.environ.get(
        "GEO_LOSS_WEIGHT", training_cfg.get("geo_loss_weight", 0.0),
    ))
    geo_t_max = float(os.environ.get(
        "GEO_LOSS_T_MAX", training_cfg.get("geo_loss_t_max", 0.4),
    ))
    geo_max_views = int(training_cfg.get("geo_loss_max_views", 0))
    geo_ray_weight = float(training_cfg.get("geo_ray_weight", 1.0))
    geo_depth_weight = float(training_cfg.get("geo_depth_weight", 1.0))
    geo_reproj_weight = float(os.environ.get(
        "GEO_REPROJ_WEIGHT", training_cfg.get("geo_reproj_weight", 0.0),
    ))
    geo_reproj_stride = int(training_cfg.get("geo_reproj_stride", 8))
    geo_loss_start_step = int(training_cfg.get("geo_loss_start_step", 0))
    if geo_t_max <= 0.0 or geo_t_max > 1.0:
        raise ValueError(f"training.geo_loss_t_max must be in (0, 1], got {geo_t_max}")
    if geo_max_views < 0:
        raise ValueError(f"training.geo_loss_max_views must be >= 0, got {geo_max_views}")
    if geo_reproj_stride < 1:
        raise ValueError("training.geo_reproj_stride must be >= 1")
    use_geo_loss = geo_loss_weight > 0.0
    if use_geo_loss and latent_backend != "feature_vae":
        raise ValueError("Geo aux loss is only supported for feature_vae backend")

    model_params = _to_dict(model_cfg.get("params", {}))
    camera_conditioning = str(model_params.get("camera_conditioning", "plucker_flip_pe"))
    baseline_camera_mode = str(model_params.get("baseline_camera_mode", "plucker"))
    use_baseline_camera = camera_conditioning == "baseline_prope"
    use_no_camera = camera_conditioning == "none"
    if camera_conditioning not in ("plucker_flip_pe", "baseline_prope", "none"):
        raise ValueError(f"Unsupported camera_conditioning: {camera_conditioning!r}")

    # Recipe knobs (see stage2.transport.flow)
    # Optional launcher override.  Keeping the base head instantiated preserves
    # checkpoint compatibility, while BASE_MODEL_COEFF=0 disables its loss and
    # gradient contribution in the transport.
    base_model_coeff = float(
        os.environ.get(
            "BASE_MODEL_COEFF",
            training_cfg.get("base_model_coeff", 1.0),
        )
    )
    if base_model_coeff < 0.0:
        raise ValueError("BASE_MODEL_COEFF/training.base_model_coeff must be >= 0")
    repa_weight = float(repa_cfg.get("weight", 0.5))
    repa_enable = bool(repa_cfg.get("enable", False))
    self_flow_enable = bool(int(os.environ.get(
        "SELF_FLOW_ENABLE", "1" if self_flow_cfg.get("enable", False) else "0"
    )))
    self_flow_mask_ratio = float(os.environ.get(
        "SELF_FLOW_MASK_RATIO", self_flow_cfg.get("mask_ratio", 0.1)
    ))
    self_flow_weight = float(os.environ.get(
        "SELF_FLOW_WEIGHT", self_flow_cfg.get("weight", 0.8)
    ))
    self_flow_warmup_steps = int(os.environ.get(
        "SELF_FLOW_WARMUP_STEPS", self_flow_cfg.get("warmup_steps", 1000)
    ))
    self_flow_student_depth = int(os.environ.get(
        "SELF_FLOW_STUDENT_DEPTH", self_flow_cfg.get("student_depth", 8)
    ))
    self_flow_teacher_depth = int(os.environ.get(
        "SELF_FLOW_TEACHER_DEPTH", self_flow_cfg.get("teacher_depth", 20)
    ))
    self_flow_proj_hidden = int(os.environ.get(
        "SELF_FLOW_PROJ_HIDDEN", self_flow_cfg.get("proj_hidden", 2048)
    ))
    self_flow_exclude_cond_views = bool(int(os.environ.get(
        "SELF_FLOW_EXCLUDE_COND", "1" if self_flow_cfg.get("exclude_cond_views", True) else "0"
    )))
    if self_flow_enable:
        # The released transport does not ship the self-flow (teacher / DDT)
        # objective: src/stage2/transport/flow.py:training_losses_rae has no
        # teacher path, so enabling it here cannot compute a self-flow term.
        # Fail fast with guidance instead of silently ignoring the setting.
        raise NotImplementedError(
            "self_flow is not available in the released transport; set "
            "self_flow.enable=false (or SELF_FLOW_ENABLE=0). The released GAE "
            "flow checkpoints were trained without the self-flow objective."
        )
    motion_loss_enable = bool(motion_loss_cfg.get("enable", False))
    motion_loss_strength = float(motion_loss_cfg.get("strength", 1.0))
    motion_loss_max_multiplier = float(
        motion_loss_cfg.get("max_multiplier", 4.0)
    )
    if motion_loss_strength < 0.0:
        raise ValueError("motion_loss.strength must be >= 0")
    if motion_loss_max_multiplier < 1.0:
        raise ValueError("motion_loss.max_multiplier must be >= 1")
    if not motion_loss_enable:
        motion_loss_strength = 0.0

    # Multi-view shape comes from top-level `num_views` (matches v4 yaml convention).
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
        total_view = 8

    cond_num_raw = dataset_cfg.get("cond_num", 1)
    if isinstance(cond_num_raw, str) and "-" in cond_num_raw:
        cond_num_min, cond_num_max = map(int, cond_num_raw.split("-"))
    else:
        cond_num_min = cond_num_max = int(cond_num_raw)
    ref_view_sampling = dataset_cfg.get("ref_view_sampling", "random")
    # prefix_fl regime split: probability of the first/last interpolation regime
    # (cond=2, ref_view_sampling="interpolate"); the rest is prefix generation
    # with cond uniform in [min,max] (which now also includes cond=2). Decoupling
    # the regime from cond lets generation see 2-frame prefixes too.
    interp_ratio = float(dataset_cfg.get("interp_ratio", 0.5))
    if not 0.0 <= interp_ratio <= 1.0:
        raise ValueError(f"dataset.interp_ratio must be in [0,1], got {interp_ratio}")
    if cond_num_min < 0 or cond_num_max >= total_view:
        raise ValueError(
            f"cond_num range must satisfy 0 <= min and max < num_views "
            f"(got {cond_num_min}-{cond_num_max}, num_views={total_view})"
        )
    if ref_view_sampling == "interpolate" and not (
        cond_num_min <= 2 <= cond_num_max
    ):
        raise ValueError(
            "ref_view_sampling='interpolate' requires cond_num range to include 2"
        )
    if ref_view_sampling == "prefix_fl" and not (
        cond_num_min <= 2 <= cond_num_max
    ):
        raise ValueError(
            "ref_view_sampling='prefix_fl' requires cond_num range to include 2 "
            "(首尾帧 mode activates when batch cond_num==2)"
        )

    # ── Experiment dir ──────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        idx = len(glob(f"{args.results_dir}/*"))
        h_dec = model_cfg.get("params", {}).get("hidden_size", [768, 2048])
        h_enc = h_dec[0] if isinstance(h_dec, (list, tuple)) else h_dec
        h_dec = h_dec[1] if isinstance(h_dec, (list, tuple)) else h_dec
        d = model_cfg.get("params", {}).get("depth", [28, 6])
        d_enc = d[0] if isinstance(d, (list, tuple)) else d
        d_dec = d[1] if isinstance(d, (list, tuple)) else d
        exp_name = (
            f"{idx:03d}-gaeflow-enc{h_enc}-dec{h_dec}"
            f"-d{d_enc}+{d_dec}-v{total_view}-{args.precision}"
        )
        exp_dir = os.path.join(args.results_dir, exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
    else:
        exp_dir = ckpt_dir = exp_name = None
    logger = create_logger(exp_dir)
    logger.info(f"Experiment: {exp_dir}")
    logger.info(
        f"global_bs={global_batch_size} micro_bs={micro_batch_size} "
        f"world_size={world_size} grad_accum={grad_accum_steps}"
    )
    logger.info(
        f"Multiview: V={total_view} cond_num={cond_num_raw} "
        f"ref_view_sampling={ref_view_sampling} "
        f"camera_drop={camera_drop} ref_drop_prob={ref_drop_prob} "
        f"reference_conditioning={reference_conditioning} "
        f"camera_cond={camera_conditioning} "
        f"pose_origin={pose_origin} "
        f"pose_translation_norm={pose_translation_norm}"
    )
    if pose_interval_scale:
        logger.info(
            f"Pose interval translation scale: enabled "
            f"(scale=clamp(mean_frame_gap/{pose_interval_canonical:g}, "
            f"{pose_interval_min_scale:g}, {pose_interval_max_scale:g}))"
        )
    if use_rgb_loss:
        logger.info(
            f"RGB aux loss: weight={rgb_loss_weight} t_max={rgb_t_max} "
            f"max_views={rgb_max_views or 'all'} start_step={rgb_loss_start_step}"
        )
    _train_ds_str = str(cfg.get("train_dataset", ""))
    mv_precomputed = "RE10KLatent_Multi" in _train_ds_str
    if mv_precomputed:
        logger.info("Multiview data: precomputed latent (RE10KLatent_Multi)")

    # ── Latent stats (whitening) ────────────────────────────────────────
    latent_mean = latent_std = latent_whiten = latent_unwhiten = None
    stats_path = cfg.get("latent_stats")
    if latent_backend == "da3_direct" and stats_path:
        logger.info("da3_direct: ignoring framework latent_stats; codec self-normalizes")
    elif stats_path and os.path.isfile(str(stats_path)):
        if device.type == "cuda" and (device.index or 0) == 0:
            _cache_to_local(str(stats_path))
        if dist.is_initialized():
            dist.barrier()
        local_stats = _cache_to_local(str(stats_path))
        latent_mean, latent_std, latent_whiten, latent_unwhiten = load_latent_stats(
            local_stats, device
        )
        logger.info(
            f"latent_stats: {'full-cov' if latent_whiten is not None else 'per-channel'} "
            f"mean.avg={latent_mean.mean():.4f} std.avg={latent_std.mean():.4f}"
        )
    else:
        logger.warning(
            "latent_stats not provided / not found — training without whitening "
            "is supported but loses RAEv2's CFG / IG headroom."
        )

    # ── Stage 1 / latent backend (frozen, online encode each step) ───────
    logger.info(f"latent_backend={latent_backend}")
    if latent_backend not in ("feature_vae", "sd_vae", "wan2_1", "da3_direct"):
        raise ValueError(f"unsupported latent_backend: {latent_backend!r}")
    if latent_backend in ("feature_vae", "da3_direct") and not stage1_cfg:
        raise ValueError(f"config must specify stage_1 for {latent_backend}")

    rae = None
    sd_vae_scaling = 1.0
    wan_latent_mean = wan_latent_std = None

    if latent_backend == "wan2_1":
        wan_vae_path = str(
            cfg.get("wan_vae_path", os.environ.get("WAN_VAE_PATH", ""))
        )
        if not os.path.isdir(wan_vae_path):
            raise FileNotFoundError(f"wan_vae_path not found: {wan_vae_path!r}")
        if os.path.isdir("/local-ssd") and not wan_vae_path.startswith("/local-ssd"):
            wan_key = os.path.abspath(wan_vae_path.rstrip("/"))
            wan_digest = hashlib.sha1(wan_key.encode("utf-8")).hexdigest()[:12]
            local_wan = f"/local-ssd/_train_cache/wan_vae_{wan_digest}"
            if (device.index or 0) == 0 and not os.path.isdir(local_wan):
                print(f"  [cache] {wan_vae_path} -> {local_wan} ...", flush=True)
                tmp_wan = f"{local_wan}.staging_{os.getpid()}"
                if os.path.isdir(tmp_wan):
                    shutil.rmtree(tmp_wan)
                shutil.copytree(wan_vae_path, tmp_wan)
                os.replace(tmp_wan, local_wan)
            if dist.is_initialized():
                dist.barrier()
            if os.path.isdir(local_wan):
                wan_vae_path = local_wan
        vae = AutoencoderKLWan.from_pretrained(wan_vae_path, torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        vae.requires_grad_(False)
        _wan_mean = vae.config.latents_mean
        _wan_std = vae.config.latents_std
        if _wan_mean is not None and _wan_std is not None:
            wan_latent_mean = torch.tensor(
                _wan_mean, dtype=torch.float32, device=device
            ).view(1, -1, 1, 1)
            wan_latent_std = torch.tensor(
                _wan_std, dtype=torch.float32, device=device
            ).view(1, -1, 1, 1)
        logger.info(f"Wan2.1 VAE ready from {wan_vae_path}")
    elif latent_backend == "sd_vae":
        sd_vae_model = str(cfg.get("sd_vae_model", "stabilityai/sd-vae-ft-mse"))
        logger.info(f"Loading Stable Diffusion VAE: {sd_vae_model}")
        vae = AutoencoderKL.from_pretrained(sd_vae_model, torch_dtype=torch.float32)
        vae = vae.to(device).eval()
        vae.requires_grad_(False)
        sd_vae_scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
        logger.info(f"SD VAE ready (scaling_factor={sd_vae_scaling:.5f})")
    elif latent_backend == "da3_direct":
        # Raw-DA3 (level-0) latent baseline: DA3Backbone (frozen) provides the DA3
        # encoder + DPT head; a small trunk+RGBHead (pretrained, frozen) decodes
        # RGB. Diffusion runs on the raw level-0 feature (in_channels = C).
        _s1_params = stage1_cfg.get("params") or {}
        _enc_path = _s1_params.get("encoder_pretrained_path")
        _enc_local = _resolve_da3_encoder_path(
            _enc_path, _s1_params.get("da3_weights_path"),
        )
        if os.path.isdir(_enc_local):
            _enc_local = _cache_dir_to_local_ssd(_enc_local, "da3")
        if _enc_local != _enc_path:
            _s1_params["encoder_pretrained_path"] = _enc_local
            stage1_cfg["params"] = _s1_params
        logger.info("Loading DA3 encoder (da3_direct) ...")
        rae = instantiate_from_config(stage1_cfg).to(device).eval()
        rae.requires_grad_(False)
        da3_direct_cfg = _to_dict(cfg.get("da3_direct", {}))
        vae = DA3DirectCodec(
            rae=rae,
            level=int(da3_direct_cfg.get("level", 0)),
            trunk_dim=int(da3_direct_cfg.get("trunk_dim", 1024)),
            num_trunk_blocks=int(da3_direct_cfg.get("num_trunk_blocks", 4)),
            trunk_heads=int(da3_direct_cfg.get("trunk_heads", 8)),
            rgb_decoder=da3_direct_cfg.get("rgb_decoder"),
            latent_stats_path=da3_direct_cfg.get("latent_stats_path"),
        ).to(device).eval()
        vae.requires_grad_(False)
        if use_rgb_loss:
            rgb_ckpt = str(cfg.get("da3_direct_rgb_checkpoint", "") or "")
            if not rgb_ckpt or not os.path.isfile(rgb_ckpt):
                raise FileNotFoundError(
                    "da3_direct with rgb_loss_weight > 0 requires "
                    f"da3_direct_rgb_checkpoint (pretrained RGB head); got {rgb_ckpt!r}"
                )
            if device.type == "cuda" and (device.index or 0) == 0:
                _cache_to_local(rgb_ckpt)
            if dist.is_initialized():
                dist.barrier()
            local_rgb = _cache_to_local(rgb_ckpt)
            state = torch.load(local_rgb, map_location="cpu")
            rgb_sd = state.get("ema_codec", state.get("codec", state))
            missing, unexpected = vae.load_state_dict(rgb_sd, strict=False)
            # `rae.*` keys are intentionally absent from the RGB ckpt (loaded from
            # the frozen encoder above); only report non-rae misses.
            non_rae_missing = [m for m in missing if not m.startswith("rae.")]
            logger.info(
                f"da3_direct RGB head loaded ({len(non_rae_missing)} missing, "
                f"{len(unexpected)} unexpected)"
            )
        logger.info(f"DA3DirectCodec ready (latent_dim={vae.latent_dim}, level={vae.level})")
    else:
        # Resolve the DA3 encoder repo id → local dir so from_pretrained loads
        # offline (no HF Hub download → no per-node ConnectTimeout at startup).
        _s1_params = stage1_cfg.get("params") or {}
        _enc_path = _s1_params.get("encoder_pretrained_path")
        _enc_local = _resolve_da3_encoder_path(
            _enc_path, _s1_params.get("da3_weights_path"),
        )
        if _enc_local != _enc_path:
            logger.info(f"[da3] encoder_pretrained_path: {_enc_path!r} -> {_enc_local}")
        if os.path.isdir(_enc_local):
            _enc_local = _cache_dir_to_local_ssd(_enc_local, "da3")
        if _enc_local != _enc_path:
            _s1_params["encoder_pretrained_path"] = _enc_local
            stage1_cfg["params"] = _s1_params
        logger.info("Loading DA3 encoder ...")
        rae = instantiate_from_config(stage1_cfg).to(device).eval()
        rae.requires_grad_(False)

        if not vae_cfg:
            raise ValueError("config must specify codec: ...")
        vae_ckpt_path = str(cfg.get("vae_checkpoint", "") or "")
        if not vae_ckpt_path or not os.path.isfile(vae_ckpt_path):
            raise FileNotFoundError(
                f"vae_checkpoint not found: {vae_ckpt_path!r}. Set it in the config."
            )
        if device.type == "cuda" and (device.index or 0) == 0:
            _cache_to_local(vae_ckpt_path)
        if dist.is_initialized():
            dist.barrier()
        local_vae_ckpt = _cache_to_local(vae_ckpt_path)
        logger.info(f"Loading VAE ckpt: {local_vae_ckpt}")
        vae_cfg_for_load = dict(vae_cfg)
        if use_rgb_loss:
            if not vae_cfg_for_load.get("rgb_decoder"):
                raise ValueError(
                    "training.rgb_loss_weight > 0 requires codec.rgb_decoder"
                )
        else:
            vae_cfg_for_load.pop("rgb_decoder", None)
        vae = GAECodec(**vae_cfg_for_load).to(device).eval()
        vae_state = torch.load(local_vae_ckpt, map_location="cpu")
        vae_sd = vae_state.get("ema_vae", vae_state.get("vae", vae_state))
        missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
        if missing or unexpected:
            logger.info(
                f"VAE load partial: {len(missing)} missing, {len(unexpected)} unexpected"
            )
        vae.requires_grad_(False)
        if use_rgb_loss:
            if vae.rgb_head is None:
                raise RuntimeError(
                    "rgb_loss_weight > 0 but VAE has no rgb_head after loading ckpt"
                )
            logger.info(
                f"VAE ready (latent_dim={vae.latent_dim}, rgb_head enabled)"
            )
        else:
            logger.info(f"VAE ready (latent_dim={vae.latent_dim})")

        if use_geo_loss:
            if rae.rae_cl_decoder is None:
                raise RuntimeError(
                    "geo_loss_weight > 0 requires RAE DPT decoder (rae_cl_decoder)"
                )
            logger.info(
                f"Geo aux loss: weight={geo_loss_weight} t_max={geo_t_max} "
                f"max_views={geo_max_views if geo_max_views > 0 else 'all'} "
                f"ray_w={geo_ray_weight} depth_w={geo_depth_weight} "
                f"reproj_w={geo_reproj_weight} reproj_stride={geo_reproj_stride} "
                f"target=GT_encoder_features (original video)"
            )

    # ── Text encoder (Qwen3-0.6B, frozen, online) ───────────────────────
    if not text_cfg:
        raise ValueError("config must specify text_encoder: ...")
    text_model_name = str(text_cfg.get("model_name", "Qwen/Qwen3-0.6B"))
    text_max_length = int(text_cfg.get("max_length", 256))

    # Resolve the text-encoder repo id → local dir so the tokenizer/model load
    # offline (no HF Hub download → no per-node ConnectTimeout at startup).
    _text_local = _resolve_text_model_path(text_model_name)
    if _text_local != text_model_name:
        logger.info(f"[text] model_name: {text_model_name!r} -> {_text_local}")
        text_model_name = _text_local
    if os.path.isdir(text_model_name):
        text_model_name = _cache_dir_to_local_ssd(text_model_name, "qwen3")
    _text_is_local = os.path.isdir(text_model_name)

    # Cache race avoidance: same idea as t2i_rae, but multi-node aware.
    # Each node has its own /local-ssd/hf_cache (node-local NVMe), so we
    # need *every node's local rank 0* to prefill — not just the global
    # rank 0. Otherwise non-master nodes hit HF_HUB_OFFLINE=1 with an
    # empty cache and crash on Qwen3 load. Within a node, only local
    # rank 0 pulls, so there's still no fcntl contention.
    # When model_name resolves to a local dir there's nothing to fetch.
    local_rank_env = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank_env == 0 and not _text_is_local:
        logger.info(
            f"  [hf-cache] local-rank-0 pre-fetching {text_model_name} to "
            f"{os.environ.get('HUGGINGFACE_HUB_CACHE', '~/.cache/huggingface')}"
        )
        prefill_hf_cache(text_model_name, torch_dtype=torch.bfloat16)
        logger.info("  [hf-cache] local-rank-0 pre-fetch done")
    if dist.is_initialized():
        dist.barrier()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info(f"Loading text encoder: {text_model_name} (max_length={text_max_length})")
    text_encoder = Qwen3TextEncoder(
        model_name=text_model_name, max_length=text_max_length,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    if dist.is_initialized():
        dist.barrier()

    with torch.no_grad():
        null_text_out = text_encoder([""])
    null_text_tokens = null_text_out["tokens"].to(device)        # (1, T, D)
    text_embed_dim = int(null_text_tokens.shape[-1])
    logger.info(f"text_embed_dim={text_embed_dim} (Qwen3 hidden)")
    expected_text_dim = int(model_cfg.get("params", {}).get("text_embed_dim", text_embed_dim))
    if expected_text_dim != text_embed_dim:
        raise ValueError(
            f"config text_embed_dim={expected_text_dim} but encoder hidden={text_embed_dim}; "
            "set stage_2.params.text_embed_dim and cross_attn_kdim to match."
        )

    # ── DiT (trainable, RAE recipe) ─────────────────────────────────────
    in_ch = int(model_cfg.get("params", {}).get("in_channels", 128))
    if latent_backend in ("sd_vae", "wan2_1"):
        image_size = int(training_cfg.get("image_size", 256))
        latent_h = latent_w = image_size // 8
    elif latent_backend == "da3_direct":
        image_size = int(stage1_cfg.get("params", {}).get("encoder_input_size", 252))
        latent_h = latent_w = image_size // 14
        if in_ch != int(vae.latent_dim):
            raise ValueError(
                f"da3_direct: stage_2.in_channels={in_ch} must equal DA3 feature "
                f"dim {vae.latent_dim} (level-0 raw features are diffused directly)"
            )
    else:
        image_size = int(stage1_cfg.get("params", {}).get("encoder_input_size", 252))
        latent_h = image_size // 14
        latent_w = latent_h
    if "input_size" not in model_cfg["params"]:
        model_cfg["params"]["input_size"] = latent_h
    latent_size = (in_ch, latent_h, latent_w)
    if "time_dist_shift" in misc_cfg:
        time_dist_shift = float(misc_cfg["time_dist_shift"])
    else:
        shift_dim = int(misc_cfg.get(
            "time_dist_shift_dim", math.prod(latent_size)
        ))
        shift_base = int(misc_cfg.get("time_dist_shift_base", 4096))
        time_dist_shift = shift_from_latent_dim(shift_dim, base=shift_base)
    _shift_env = os.environ.get("TIME_DIST_SHIFT", "").strip()
    if _shift_env:
        time_dist_shift = float(_shift_env)
        if local_rank_env == 0:
            logger.info(f"[t] TIME_DIST_SHIFT override -> {time_dist_shift}")
    time_dist_type = str(transport_cfg.get("time_dist_type", "logit-normal_0_1"))
    prediction = str(transport_cfg.get("prediction", "x"))
    if prediction not in ("x", "velocity"):
        raise ValueError(
            f"transport.prediction must be 'x' or 'velocity'; got {prediction!r}"
        )
    model_pred_type = str(model_cfg.get("params", {}).get("prediction_type", "x"))
    if model_pred_type != prediction:
        logger.warning(
            f"transport.prediction={prediction!r} vs stage_2.prediction_type={model_pred_type!r}; "
            "these should match for the loss/sampler to interpret the head correctly."
        )

    if repa_enable:
        model_cfg["params"]["repa_enable"] = True
        model_cfg["params"]["repa_align_depth"] = int(repa_cfg.get("align_depth", 8))
        model_cfg["params"]["repa_target_dim"] = int(
            repa_cfg.get("target_dim", vae_cfg.get("feature_level_dim", 2048))
        )
        model_cfg["params"]["repa_proj_hidden"] = int(repa_cfg.get("proj_hidden", 2048))
    if self_flow_enable:
        model_cfg["params"]["self_flow_enable"] = True
        model_cfg["params"]["self_flow_student_depth"] = self_flow_student_depth
        model_cfg["params"]["self_flow_teacher_depth"] = self_flow_teacher_depth
        model_cfg["params"]["self_flow_proj_hidden"] = self_flow_proj_hidden

    logger.info(f"Building DiT: {model_cfg.get('target')}")
    dit = instantiate_from_config(model_cfg).to(device)
    ema_dit = deepcopy(dit).eval().requires_grad_(False)

    if dist.is_initialized():
        ddp_dit = DDP(
            dit, device_ids=[device.index], broadcast_buffers=False,
            gradient_as_bucket_view=False, find_unused_parameters=True,
            bucket_cap_mb=200,
        )
    else:
        ddp_dit = dit

    n_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    logger.info(
        f"DiT params: {n_params/1e6:.2f}M  in_ch={in_ch}  latent {latent_h}x{latent_w}  "
        f"time_dist_shift={time_dist_shift:.3f}"
    )
    if self_flow_enable:
        logger.info(
            "Self-Flow enabled: "
            f"mask_ratio={self_flow_mask_ratio:.3f} weight={self_flow_weight:.3f} "
            f"warmup_steps={self_flow_warmup_steps} "
            f"student_depth={self_flow_student_depth} teacher_depth={self_flow_teacher_depth} "
            f"exclude_cond_views={self_flow_exclude_cond_views} ema_decay={ema_decay:.5f}"
        )

    # ── Optimizer + scheduler ───────────────────────────────────────────
    optim_cfg = _to_dict(training_cfg.get("optimizer", {}))
    if "type" not in optim_cfg and "type" in training_cfg:
        optim_cfg["type"] = training_cfg["type"]
    if lr_override is not None:
        # optimizer.lr is explicit in most YAMLs and otherwise wins over
        # base_lr_default, so override it as well.
        optim_cfg["lr"] = lr_override
        if "adamw_lr" in optim_cfg:
            optim_cfg["adamw_lr"] = lr_override
    optimizer, optim_msg = build_optimizer_rae(
        [p for p in dit.parameters() if p.requires_grad],
        optim_cfg=optim_cfg,
        base_lr_default=base_lr,
    )
    logger.info(f"Optimizer: {optim_msg}")

    # Steps-per-epoch is set after dataloader is built, but the LR schedule
    # needs ``decay_end_steps`` upfront — we honour the explicit yaml value
    # (`training.decay_end_steps`) and only fall back to a large default
    # because the cut3r video loader's epoch length is huge.
    decay_end_steps = int(training_cfg.get("decay_end_steps", 100_000))
    # Env override so finetune launchers can match the cosine horizon to a short
    # run (e.g. DECAY_END_STEPS=FINETUNE_STEPS) without editing the shared yaml.
    _decay_env = os.environ.get("DECAY_END_STEPS", "").strip()
    if _decay_env:
        decay_end_steps = int(_decay_env)
        if rank == 0:
            logger.info(f"[lr] DECAY_END_STEPS override -> {decay_end_steps}")
    _warmup_env = os.environ.get("WARMUP_STEPS", "").strip()
    if _warmup_env:
        warmup_steps = int(_warmup_env)
        if rank == 0:
            logger.info(f"[lr] WARMUP_STEPS override -> {warmup_steps}")
    decay_end_steps = max(decay_end_steps, warmup_steps + 1)
    lr_lambda = build_lr_lambda_rae(
        schedule_type=schedule_type,
        base_lr=base_lr,
        final_lr=final_lr,
        warmup_steps=warmup_steps,
        decay_end_steps=decay_end_steps,
        warmup_from_zero=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(
        f"LR: base={base_lr}, final={final_lr}, warmup={warmup_steps}, "
        f"schedule={schedule_type}, decay_end_steps={decay_end_steps}"
    )

    # ── Resume / init ───────────────────────────────────────────────────
    train_steps = 0
    start_epoch = 0
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        dit_mod = ddp_dit.module if isinstance(ddp_dit, DDP) else ddp_dit
        miss, unexp = dit_mod.load_state_dict(
            _strip_rope_buffers(ckpt["model"]), strict=False,
        )
        if rank == 0:
            logger.info(f"Resumed model: missing {len(miss)}, unexpected {len(unexp)}")
        ema_dit.load_state_dict(_strip_rope_buffers(ckpt["ema"]), strict=False)
        try:
            optimizer.load_state_dict(ckpt["opt"])
        except Exception as e:
            logger.warning(f"optimizer state load failed: {e}; starting fresh state")
        train_steps = int(ckpt.get("train_steps", 0))
        if lr_override is not None:
            # optimizer.load_state_dict restores the old checkpoint LR, and
            # scheduler.load_state_dict restores its old base_lrs. Reapply the
            # requested schedule at the resumed global step while retaining
            # Adam moments and every other optimizer state.
            lr_factor = float(lr_lambda(train_steps))
            resumed_lrs = []
            scheduler.base_lrs = [base_lr] * len(optimizer.param_groups)
            scheduler.last_epoch = train_steps
            for group in optimizer.param_groups:
                group["initial_lr"] = base_lr
                group["lr"] = base_lr * lr_factor
                resumed_lrs.append(group["lr"])
            scheduler._last_lr = resumed_lrs
            logger.info(
                f"[lr] LR_OVERRIDE={lr_override:.8g} reapplied after resume "
                f"at step={train_steps}; effective_lr={resumed_lrs}"
            )
        else:
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            elif train_steps > 0:
                for _ in range(train_steps):
                    scheduler.step()
        logger.info(f"Resumed from {args.ckpt} (step={train_steps})")
    elif args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu")
        dit_mod = ddp_dit.module if isinstance(ddp_dit, DDP) else ddp_dit

        # Shape-aware warm start. load_state_dict(strict=False) only tolerates
        # MISSING/UNEXPECTED keys — a shape MISMATCH still raises. Cross-latent
        # warm-start (e.g. d128 → d64) keeps the transformer trunk (hidden_size
        # unchanged) but changes every in_channels-dependent projection.
        #
        # Special case: patch_size 1 → 2 keeps the latent channel count but
        # changes PatchEmbed kernels and final heads. Migrate those weights
        # explicitly so the patch2 run starts closer to the patch1 function.
        def _migrate_patch1_to_patch2(key: str, src: torch.Tensor, dst: torch.Tensor):
            if key.endswith(".proj.weight") and src.ndim == 4 and dst.ndim == 4:
                same_io = src.shape[:2] == dst.shape[:2]
                if same_io and src.shape[2:] == (1, 1) and dst.shape[2:] == (2, 2):
                    # PatchEmbed Conv2d sums over the 2x2 patch. Filling each
                    # sub-kernel with w/4 makes it approximate average-pool
                    # followed by the old patch1 projection.
                    return src.expand(-1, -1, 2, 2).contiguous() / 4.0

            if (
                (
                    key.endswith("final_layer.linear.weight")
                    or key.endswith("final_layer.linear.bias")
                )
                and src.ndim in (1, 2)
                and dst.ndim == src.ndim
                and dst.shape[0] == src.shape[0] * 4
                and (src.ndim == 1 or dst.shape[1:] == src.shape[1:])
            ):
                # DDTFinalLayer flattens patch outputs as (p, q, C). For patch2,
                # initialize all four sub-pixels from the old patch1 prediction.
                return (
                    src.view(1, 1, *src.shape)
                    .expand(2, 2, *src.shape)
                    .reshape_as(dst)
                    .contiguous()
                )

            return None

        def _filter_loadable(src_sd):
            src_sd = _strip_rope_buffers(src_sd)
            if src_sd is None:
                return {}, [], []
            cur = dit_mod.state_dict()
            keep = {}
            shape_skip = []
            migrated = []
            for k, v in src_sd.items():
                if k not in cur:
                    continue
                if cur[k].shape == v.shape:
                    keep[k] = v
                    continue
                v_migrated = _migrate_patch1_to_patch2(k, v, cur[k])
                if v_migrated is not None and v_migrated.shape == cur[k].shape:
                    keep[k] = v_migrated.to(dtype=cur[k].dtype)
                    migrated.append(k)
                else:
                    shape_skip.append(k)
            return keep, shape_skip, migrated

        model_sd, shape_skip, migrated = _filter_loadable(ckpt["model"])
        miss, unexp = dit_mod.load_state_dict(model_sd, strict=False)
        if rank == 0:
            logger.info(
                f"Init weights: loaded {len(model_sd)}, "
                f"patch-migrated {len(migrated)}, "
                f"shape-skipped {len(shape_skip)}, "
                f"missing {len(miss)}, unexpected {len(unexp)}"
            )
            if migrated:
                logger.info(
                    f"  patch1→patch2 migrated: {migrated[:12]}"
                    f"{'...' if len(migrated) > 12 else ''}"
                )
            if shape_skip:
                logger.info(
                    f"  shape-mismatch (kept random init): {shape_skip[:10]}"
                    f"{'...' if len(shape_skip) > 10 else ''}"
                )
        if "ema" in ckpt:
            ema_sd, _, ema_migrated = _filter_loadable(ckpt["ema"])
            ema_dit.load_state_dict(ema_sd, strict=False)
            if rank == 0 and ema_migrated:
                logger.info(
                    f"  EMA patch1→patch2 migrated: {ema_migrated[:12]}"
                    f"{'...' if len(ema_migrated) > 12 else ''}"
                )
        logger.info(f"Initialized weights from {args.init_ckpt} (step starts at 0)")

    # ── Dataloader (cut3r video, e.g. RE10K_Multi) ──────────────────────
    from cut3r_data import get_data_loader, robust_collate
    train_loader = get_data_loader(
        cfg.train_dataset, batch_size=micro_batch_size, num_workers=num_workers,
        pin_mem=True, shuffle=True, drop_last=True,
        fixed_length=True, world_size=world_size, rank=rank,
        collate_fn=robust_collate,
    )
    steps_per_epoch = max(len(train_loader) // grad_accum_steps, 1)
    loader_batches = len(train_loader)
    logger.info(
        f"Loader: {loader_batches} micro-batches/epoch, "
        f"{steps_per_epoch} optim-steps/epoch"
    )

    # ── T2I co-training loader (anti-forgetting) ────────────────────────
    # Interleaves pure-T2I (BLIP3o single-image) optim steps into the
    # multi-view loop to keep the warm-started T2I trunk from being
    # catastrophically forgotten. The T2I step reproduces the exact T2I-stage
    # forward (V=1, z_ref_clean=None, plucker_6d=None, cond_num=0). Domain is
    # chosen deterministically by global ``train_steps`` so every rank runs the
    # same forward graph each step (else DDP/NCCL desync hang).
    cotrain_cfg = _to_dict(cfg.get("cotrain_t2i", {}))
    cotrain_enable = bool(cotrain_cfg.get("enable", False))
    # Env overrides (launcher-friendly): COTRAIN_T2I / T2I_EVERY_K / T2I_BATCH_SIZE.
    _env_cotrain = os.environ.get("COTRAIN_T2I", "")
    if _env_cotrain != "":
        cotrain_enable = _env_cotrain not in ("0", "false", "False", "no")
    # T2I_ONLY: pure-T2I warmup — every optim step is a BLIP3o T2I step and the
    # RE10K multi-view loop is skipped entirely (no DA3/VAE multi-view encode).
    # Useful to let in_channels-reinitialised projections (e.g. d128→d64
    # warm-start) recover T2I capability before tackling the 3D task.
    t2i_only = os.environ.get("T2I_ONLY", "") not in ("", "0", "false", "False", "no")
    if t2i_only and not cotrain_enable:
        raise ValueError(
            "T2I_ONLY=1 requires cotrain_t2i enabled (set COTRAIN_T2I=1); the "
            "pure-T2I loop is driven by the BLIP3o co-training stream."
        )
    t2i_every_k = int(os.environ.get("T2I_EVERY_K", cotrain_cfg.get("every_k", 3)))
    # Mirror the T2I config's shift resolution: explicit `time_dist_shift` wins,
    # else dim-aware sqrt(dim/base) — matching the T2I co-train recipe
    # (no explicit key → sqrt(41472/4096) ≈ 3.18).
    if "time_dist_shift" in cotrain_cfg:
        t2i_shift = float(cotrain_cfg["time_dist_shift"])
    else:
        t2i_shift = shift_from_latent_dim(
            int(cotrain_cfg.get("time_dist_shift_dim", math.prod(latent_size))),
            base=int(cotrain_cfg.get("time_dist_shift_base", 4096)),
        )
    _t2i_shift_env = os.environ.get("T2I_TIME_DIST_SHIFT", "").strip()
    if _t2i_shift_env:
        t2i_shift = float(_t2i_shift_env)
        if rank == 0:
            logger.info(f"[t] T2I_TIME_DIST_SHIFT override -> {t2i_shift}")
    t2i_loss_weight = float(cotrain_cfg.get("loss_weight", 1.0))
    t2i_iter = None
    if cotrain_enable:
        try:
            from data.blip3o_wds import build_blip3o_wds_loader
            from data.imagenet_arrow import build_imagenet_arrow_loader
        except ImportError as exc:
            raise RuntimeError(
                "cotrain_t2i requires the data extras: pip install -e '.[data]'"
            ) from exc
        if grad_accum_steps != 1:
            raise ValueError(
                "cotrain_t2i requires grad_accum_steps==1 (domain is chosen per "
                f"optim step); got grad_accum_steps={grad_accum_steps}. Disable "
                "co-training or set grad_accum_steps=1."
            )
        if t2i_every_k < 2:
            raise ValueError(f"cotrain_t2i.every_k must be >=2; got {t2i_every_k}.")
        t2i_ds = _to_dict(cotrain_cfg.get("dataset", {}))
        t2i_bs = int(os.environ.get(
            "T2I_BATCH_SIZE", cotrain_cfg.get("batch_size", micro_batch_size)
        ))
        # int (square) or (W, H) for widescreen T2I matching video frames.
        _t2i_image_size_raw = t2i_ds.get("image_size", image_size)
        if (
            hasattr(_t2i_image_size_raw, "__len__")
            and not isinstance(_t2i_image_size_raw, (str, bytes, int))
            and len(_t2i_image_size_raw) == 2
        ):
            t2i_image_size = (int(_t2i_image_size_raw[0]), int(_t2i_image_size_raw[1]))
        else:
            t2i_image_size = int(_t2i_image_size_raw)
        imagenet_arrow_root = t2i_ds.get("imagenet_arrow_root")
        if imagenet_arrow_root:
            # Ablation path: same ImageNet class-name Arrow source as the
            # dedicated single-image T2I co-train path.
            prompt_template = t2i_ds.get(
                "prompt_template", "a photo of a {class_name}"
            )
            t2i_loader = build_imagenet_arrow_loader(
                root=str(imagenet_arrow_root),
                split=str(t2i_ds.get("imagenet_arrow_split", "train")),
                image_size=t2i_image_size,
                batch_size=t2i_bs,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                virtual_epoch_steps=int(t2i_ds.get("virtual_epoch_steps", 10000)),
                shuffle_buffer=int(t2i_ds.get("shuffle_buffer", 2000)),
                seed=seed,
                prompt_template=str(prompt_template) if prompt_template else None,
                pixel_norm=str(t2i_ds.get("pixel_norm", "imagenet")),
            )
            t2i_ds = {
                **t2i_ds,
                "splits": ["imagenet_class_arrow"],
                "prompt_template": prompt_template,
            }
        else:
            t2i_loader = build_blip3o_wds_loader(
                data_dir=t2i_ds.get("data_dir"),
                splits=t2i_ds.get("splits"),
                image_size=t2i_image_size,
                batch_size=t2i_bs,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                virtual_epoch_steps=int(t2i_ds.get("virtual_epoch_steps", 10000)),
                shuffle_buffer=int(t2i_ds.get("shuffle_buffer", 2000)),
                seed=seed,
                dual_caption_prob=float(t2i_ds.get("dual_caption_prob", 0.5)),
                split_probs=t2i_ds.get("split_probs"),
            )

        def _t2i_batches():
            """Infinite stream over the WDS loader; rebuilds the pipeline with a
            fresh seed each virtual epoch (WDS is an IterableDataset)."""
            ep = 0
            while True:
                t2i_loader.set_epoch(ep)
                for b in t2i_loader:
                    yield b
                ep += 1

        t2i_iter = _t2i_batches()
        _t2v_every_k = int(os.environ.get("T2V_EVERY_K", "0") or "0")
        if t2i_only and _t2v_every_k >= 2:
            _sched_str = (
                f"T2I_ONLY + t2v-inject every_k={_t2v_every_k} "
                f"(~{100.0/_t2v_every_k:.0f}% t2v, rest T2I)"
            )
        elif t2i_only:
            _sched_str = "T2I_ONLY (100% T2I, MV loop skipped)"
        else:
            _sched_str = f"every_k={t2i_every_k} (~{100.0/t2i_every_k:.0f}% T2I)"
        logger.info(
            f"[cotrain_t2i] enabled: {_sched_str} "
            f"bs={t2i_bs} shift={t2i_shift} loss_weight={t2i_loss_weight} "
            f"image_size={t2i_image_size} splits={t2i_ds.get('splits')}"
        )
    else:
        logger.info("[cotrain_t2i] disabled")

    # ── Training loop setup ─────────────────────────────────────────────
    ac_kwargs = (
        dict(device_type="cuda", enabled=True, dtype=torch.bfloat16)
        if args.precision == "bf16" else dict(device_type="cuda", enabled=False)
    )
    ckpt_every = int(os.environ.get(
        "CKPT_EVERY", validation_cfg.get("ckpt_every", 1000),
    ))
    _max_train_steps_raw = os.environ.get("MAX_TRAIN_STEPS", "").strip()
    max_train_steps = int(_max_train_steps_raw) if _max_train_steps_raw else None
    if max_train_steps is not None and rank == 0:
        logger.info(f"[debug] MAX_TRAIN_STEPS={max_train_steps} (early exit enabled)")

    if rank == 0 and args.wandb and HAS_WANDB:
        entity = os.environ.get("ENTITY", "gae")
        project = os.environ.get("PROJECT", "GAEFlow")
        wandb_utils.initialize(args, entity, exp_name, project)

    if rank == 0:
        logger.info("Starting training loop")

    async_ckpt = AsyncCheckpointSaver(logger) if rank == 0 else None

    # Separate running stats for the two interleaved domains so each can be
    # compared to its own reference (MV vs T2I). `ctr` holds plain counters in a
    # dict so the nested helpers can mutate them without `nonlocal`.
    running = dict(
        loss=0.0, main=0.0, base=0.0, repa=0.0, self_flow=0.0, rgb=0.0, geo=0.0,
        motion_score=0.0, motion_wmax=0.0,
    )  # multi-view
    t2i_run = dict(loss=0.0, main=0.0, base=0.0)                # T2I
    ctr = dict(log_steps=0, mv=0, t2i=0, t_start=time(), last_cn=0)
    # Per-view-count MV loss for mixed-length training: view_count -> raw loss
    # sum + micro-batch count over the current log window.
    running_by_view = {}
    accum_counter = 0
    optimizer.zero_grad()

    def _accum_view_loss(view_count, loss_val):
        s = running_by_view.get(view_count)
        if s is None:
            s = running_by_view[view_count] = dict(loss=0.0, n=0)
        s["loss"] += loss_val
        s["n"] += 1

    def _denorm_latent(y: torch.Tensor) -> torch.Tensor:
        # No framework stats (e.g. da3_direct self-normalizes in the codec) → the
        # DiT already operates in the decoder's input space, so this is identity.
        if latent_mean is None:
            return y
        return denormalize_latent(y, latent_mean, latent_std, latent_unwhiten)

    def _geo_loss_kwargs(
        gt_geo: dict | None,
        H_px: int,
        W_px: int,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> dict:
        if not use_geo_loss or gt_geo is None or train_steps < geo_loss_start_step:
            return {}
        embed_dim = rae.encoder.backbone.pretrained.embed_dim
        backbone_norm = rae.encoder.backbone.pretrained.norm
        dpt_decoder = rae.rae_cl_decoder
        vae_mod = vae.module if isinstance(vae, DDP) else vae

        def geo_decode_fn(z_raw: torch.Tensor) -> dict:
            return latent_raw_to_dpt_geometry(
                z_raw, vae_mod, backbone_norm, dpt_decoder,
                H_px, W_px, embed_dim=embed_dim,
            )

        return dict(
            gt_ray=gt_geo["ray"],
            gt_depth=gt_geo.get("depth"),
            gt_ray_conf=gt_geo.get("ray_conf"),
            geo_decode_fn=geo_decode_fn,
            # denormalize_latent_fn is shared with RGB aux — pass once at the
            # call site (both helpers used to include it → TypeError).
            geo_loss_weight=geo_loss_weight,
            geo_t_max=geo_t_max,
            geo_max_views=geo_max_views,
            geo_ray_weight=geo_ray_weight,
            geo_depth_weight=geo_depth_weight,
            geo_reproj_weight=geo_reproj_weight,
            geo_reproj_stride=geo_reproj_stride,
            geo_c2w=c2w,
            geo_K=intrinsics,
        )

    def _rgb_loss_kwargs(rgb_target: torch.Tensor | None) -> dict:
        if not use_rgb_loss or rgb_target is None or train_steps < rgb_loss_start_step:
            return {}
        return dict(
            rgb_decoder=vae.decode_rgb,
            rgb_target=rgb_target,
            # denormalize_latent_fn shared with geo aux — pass once at call site.
            rgb_loss_weight=rgb_loss_weight,
            rgb_t_max=rgb_t_max,
            rgb_max_views=rgb_max_views,
        )

    def _denorm_loss_kwargs(rgb_kw: dict, geo_kw: dict) -> dict:
        """Pass denormalize_latent_fn at most once when either aux needs it."""
        if not rgb_kw and not geo_kw:
            return {}
        return dict(denormalize_latent_fn=_denorm_latent)

    def _t2i_forward_loss(images, captions):
        """One T2I (V=1) forward → loss_dict for the co-train step:
        z_ref_clean=None, plucker_6d=None,
        cond_num=0, time_dist_shift=t2i_shift.

        Pixel convention differs from the MV path: T2I loaders hand over
        ``ImgNorm`` tensors, which is the space GAECodec and latent_stats
        were built in, so that branch encodes them as-is. The other backends
        want ``[0, 1]`` and convert via ``imgnorm_to_unit``. MV ``prepare_data``
        instead receives ``[0, 1]`` and normalizes on the way in — do not copy
        its ``(x - encoder_mean) / encoder_std`` here.
        """
        B = images.shape[0]
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            imgs_5d = images.unsqueeze(1)  # (B, V=1, 3, H, W)
            with autocast(**ac_kwargs):
                if latent_backend == "wan2_1":
                    imgs_wan = (imgnorm_to_unit(images) * 2.0 - 1.0).unsqueeze(2)
                    z_chunks = []
                    wan_chunk = int(dataset_cfg.get("wan_encode_chunk", 8))
                    for ci in range(0, imgs_wan.shape[0], wan_chunk):
                        z_c = vae.encode(
                            imgs_wan[ci : ci + wan_chunk]
                        ).latent_dist.sample()
                        z_chunks.append(z_c.squeeze(2))
                    mu = torch.cat(z_chunks, dim=0)
                    if wan_latent_mean is not None and wan_latent_std is not None:
                        mu = (mu - wan_latent_mean) / wan_latent_std
                elif latent_backend == "sd_vae":
                    imgs_sd = imgnorm_to_unit(images) * 2.0 - 1.0
                    z_chunks = []
                    sd_chunk = int(dataset_cfg.get("sd_encode_chunk", 16))
                    for ci in range(0, imgs_sd.shape[0], sd_chunk):
                        z_c = vae.encode(
                            imgs_sd[ci : ci + sd_chunk]
                        ).latent_dist.sample()
                        z_chunks.append(z_c * sd_vae_scaling)
                    mu = torch.cat(z_chunks, dim=0)
                elif hasattr(vae, "encode_views"):
                    mu = vae.encode_views(imgnorm_to_unit(images).unsqueeze(1))
                else:
                    feats_all = rae.encode(imgs_5d, mode="all")
                    feats = {k: v[:, 1:, :] for k, v in feats_all.items()}
                    _t2i_h, _t2i_w = images.shape[-2], images.shape[-1]
                    x_norm = vae.normalize_levels(
                        feats, image_size=(_t2i_h, _t2i_w)
                    )
                    mu, _ = vae.encode(x_norm)
            z_all = mu.float()
        if latent_mean is not None:
            z_all = normalize_latent(z_all, latent_mean, latent_std, latent_whiten)
        with torch.no_grad():
            if cfg_dropout_prob > 0 and random.random() < cfg_dropout_prob:
                ref_global = null_text_tokens.to(
                    dtype=torch.bfloat16
                ).expand(B, -1, -1)
            else:
                text_out = text_encoder(list(captions))
                ref_global = text_out["tokens"].to(device, dtype=torch.bfloat16)
        with autocast(**ac_kwargs):
            return training_losses_rae(
                ddp_dit if dist.is_initialized() else dit,
                x_target=z_all,
                total_view=1,
                time_dist_shift=t2i_shift,
                time_dist_type=time_dist_type,
                prediction=prediction,
                base_model_coeff=base_model_coeff,
                repa_coeff=repa_weight if repa_enable else None,
                repa_target=None,
                z_ref_clean=None,
                plucker_6d=None,
                ref_global=ref_global,
                cond_num=0,
            )

    def _pose_interval_translation_scale(batch, batch_size: int):
        """Return per-sample post-normalize translation scale from real frame gaps."""
        if not pose_interval_scale:
            return None
        frame_idx = batch.get("physical_frame_indices")
        if frame_idx is None:
            return None
        if not torch.is_tensor(frame_idx):
            frame_idx = torch.as_tensor(frame_idx)
        frame_idx = frame_idx.to(device=device, dtype=torch.float32, non_blocking=True)
        if frame_idx.ndim == 1:
            frame_idx = frame_idx.unsqueeze(0).expand(batch_size, -1)
        if frame_idx.shape[1] < 2:
            return None

        span = frame_idx.amax(dim=1) - frame_idx.amin(dim=1)
        mean_gap = span / max(frame_idx.shape[1] - 1, 1)
        scale = mean_gap / pose_interval_canonical
        scale = scale.clamp(pose_interval_min_scale, pose_interval_max_scale)
        return scale.view(batch_size, 1, 1)

    def _mv_forward(batch):
        """One multi-view (t2v) forward → ``(loss_dict, batch_cond_num, batch_total_view)``.

        Online DA3+VAE encode, camera conditioning, online text encode and the
        RAE loss — identical to the main-loop MV step. Returns the *unscaled*
        loss_dict; the caller owns grad-accum scaling / backward / logging so
        this can be shared by the main loop and the T2I-only warmup's optional
        t2v injection.
        """
        # ``T2V_NO_CAMERA`` (set by the t2i+t2v warmup launcher) trains every MV
        # batch as pure text-to-video: no reference views (cond_num=0) and the
        # camera signal (Plücker) zeroed — re10k/OpenVid frames become plain
        # text-conditioned video sequences. Datasets that self-report ``is_t2v``
        # (e.g. OpenVidT2V) get the same treatment even without the env flag.
        force_no_cam = os.environ.get("T2V_NO_CAMERA", "0") == "1"
        batch_rvs = ref_view_sampling
        if force_no_cam:
            batch_cond_num = 0
        elif ref_drop_prob > 0 and random.random() < ref_drop_prob:
            batch_cond_num = 0
        elif ref_view_sampling == "prefix_fl":
            # Decouple regime from cond_num: interp_ratio of conditioned steps
            # are first/last interpolation (cond=2), the rest are prefix
            # generation with cond uniform in [min,max] (may include 2).
            if random.random() < interp_ratio:
                batch_cond_num = 2
                batch_rvs = "interpolate"
            else:
                batch_cond_num = random.randint(cond_num_min, cond_num_max)
                batch_rvs = "prefix"
        else:
            batch_cond_num = random.randint(cond_num_min, cond_num_max)
            batch_rvs = resolve_ref_view_sampling(batch_cond_num, ref_view_sampling)

        if isinstance(batch, list) and isinstance(batch[0], dict):
            batch = convert_cut3r_batch(
                batch, batch_cond_num, batch_rvs,
            )

        is_t2v_raw = batch.get("is_t2v", False)
        if isinstance(is_t2v_raw, torch.Tensor):
            is_t2v_batch = bool(is_t2v_raw.all())
            is_t2v_any = bool(is_t2v_raw.any())
        else:
            is_t2v_batch = bool(is_t2v_raw)
            is_t2v_any = is_t2v_batch
        disable_plucker_raw = batch.get("disable_plucker", True)
        if isinstance(disable_plucker_raw, torch.Tensor):
            disable_plucker_tensor = disable_plucker_raw.to(device).float()
            disable_plucker_batch = bool(disable_plucker_raw.all())
            disable_plucker_any = bool(disable_plucker_raw.any())
        else:
            disable_plucker_batch = bool(disable_plucker_raw)
            disable_plucker_any = disable_plucker_batch
            disable_plucker_tensor = None
        if force_no_cam or is_t2v_batch:
            batch_cond_num = 0

        is_precomputed_raw = batch.get("is_precomputed", False)
        if isinstance(is_precomputed_raw, torch.Tensor):
            is_precomputed = bool(is_precomputed_raw.all())
        else:
            is_precomputed = bool(is_precomputed_raw)

        captions = batch.get("caption", None)
        view_frame_idx = None

        if is_precomputed:
            z_all = batch["z_all"].to(device, non_blocking=True)
            extrinsic = batch["c2w"].to(device, non_blocking=True)
            intrinsic = batch["intrinsics"].to(device, non_blocking=True)
            B_batch, V_batch = z_all.shape[0], z_all.shape[1]
            batch_total_view = int(V_batch)
            if batch_total_view > total_view:
                raise ValueError(
                    f"Config/Dataloader mismatch: max total_view={total_view} but V={V_batch}."
                )
            if batch_cond_num > 0:
                ref_key = f"z_ref_{batch_cond_num}"
                if ref_key not in batch:
                    raise KeyError(
                        f"Precomputed batch missing {ref_key!r} "
                        f"(batch_cond_num={batch_cond_num})"
                    )
                z_ref_cn = batch[ref_key].to(device, non_blocking=True)
            else:
                z_ref_cn = None

            fi = batch.get("frame_indices")
            if fi is not None and torch.is_tensor(fi):
                view_frame_idx = fi[0].to(device, non_blocking=True).long()

            H_img = W_img = int(stage1_cfg.get("params", {}).get("encoder_input_size", 504))
        else:
            image = batch["gt_inp"]
            intrinsic = batch["fxfycxcy"]
            extrinsic = batch["c2w"]

            fi = batch.get("frame_indices")
            if fi is not None and torch.is_tensor(fi):
                view_frame_idx = fi[0].to(device, non_blocking=True).long()

            if image.ndim != 5:
                raise ValueError(
                    f"Expected gt_inp shape (B, V, C, H, W), got {tuple(image.shape)}"
                )
            B_batch, V_batch = image.shape[0], image.shape[1]
            batch_total_view = int(V_batch)
            if batch_total_view > total_view:
                raise ValueError(
                    f"Config/Dataloader mismatch: max total_view={total_view} but V={V_batch}."
                )
            image = image.to(device, non_blocking=True)
            intrinsic = intrinsic.to(device, non_blocking=True)
            extrinsic = extrinsic.to(device, non_blocking=True)
            H_img, W_img = image.shape[-2], image.shape[-1]
        if batch_cond_num >= batch_total_view:
            raise ValueError(
                f"batch_cond_num must be < actual batch views "
                f"(got cond={batch_cond_num}, V={batch_total_view})"
            )

        # --- (1) Latent encode (online) or load precomputed ----------------
        with autocast(**ac_kwargs):
            if is_precomputed:
                z_input, z_all_gt, c2w_norm, Ks = prepare_data_precomputed(
                    z_all, z_ref_cn, extrinsic, intrinsic, device,
                    cond_num=batch_cond_num,
                    total_view=batch_total_view,
                    latent_mean=latent_mean, latent_std=latent_std,
                    latent_whiten=latent_whiten,
                    origin_idx=pose_origin_idx,
                    translation_norm=pose_translation_norm,
                )
            else:
                z_input, z_all_gt, c2w_norm, Ks = prepare_data(
                    rae, vae, image, intrinsic, extrinsic, device,
                    cond_num=batch_cond_num,
                    latent_mean=latent_mean, latent_std=latent_std,
                    latent_whiten=latent_whiten,
                    origin_idx=pose_origin_idx,
                    translation_norm=pose_translation_norm,
                    latent_backend=latent_backend,
                    sd_vae_scaling=sd_vae_scaling,
                    wan_latent_mean=wan_latent_mean,
                    wan_latent_std=wan_latent_std,
                    dataset_cfg=dataset_cfg,
                )
            pose_scale = _pose_interval_translation_scale(batch, B_batch)
            if pose_scale is not None:
                c2w_norm[:, :, :3, 3] = (
                    c2w_norm[:, :, :3, 3] * pose_scale.to(c2w_norm.dtype)
                )

            with torch.amp.autocast("cuda", enabled=False):
                if use_no_camera:
                    extra_cam_kwargs = {}
                    plucker_6d = None
                elif use_baseline_camera:
                    from stage2.models.camera_baseline import (
                        apply_baseline_camera_dropout,
                        build_baseline_camera_inputs,
                    )
                    cam_pack = build_baseline_camera_inputs(
                        c2w_norm.float(), Ks.float(),
                        H=H_img, W=W_img,
                        cond_num=batch_cond_num,
                        total_view=batch_total_view,
                        camera_mode=baseline_camera_mode,
                        v4_cond_extend=not reference_in_state,
                    )
                    camera_embedding = cam_pack["camera_embedding"]
                    viewmats_enc = cam_pack["viewmats"]
                    Ks_enc = cam_pack["Ks"]
                    if camera_drop > 0:
                        drop_mask_b = (
                            torch.rand(B_batch, 1, 1, device=device) > camera_drop
                        ).float()
                        camera_embedding, viewmats_enc = apply_baseline_camera_dropout(
                            camera_embedding, viewmats_enc, drop_mask_b,
                        )
                    if force_no_cam or (is_t2v_batch and disable_plucker_batch):
                        camera_embedding = torch.zeros_like(camera_embedding)
                    extra_cam_kwargs = dict(
                        camera_embedding=camera_embedding,
                        viewmats=viewmats_enc,
                        Ks=Ks_enc,
                    )
                    plucker_6d = None
                else:
                    extra_cam_kwargs = {}
                    # Plucker rays must live on the DiT token grid. With
                    # patch_size=2, a 36x36 latent becomes an 18x18 token grid.
                    s_patch = int(getattr(dit, "s_patch_size", 1))
                    if (
                        z_input.shape[-1] % s_patch != 0
                        or z_input.shape[-2] % s_patch != 0
                    ):
                        raise ValueError(
                            "latent size must be divisible by stage_2 patch_size; "
                            f"got latent={tuple(z_input.shape[-2:])}, patch={s_patch}"
                        )
                    plucker_6d = compute_plucker_6d_per_token(
                        c2w=c2w_norm.float(),
                        Ks=Ks.float(),
                        patches_x=z_input.shape[-1] // s_patch,
                        patches_y=z_input.shape[-2] // s_patch,
                        image_height=H_img,
                        image_width=W_img,
                        is_c2w=True,
                    )
                    # Camera dropout for CFG: drop the WHOLE batch's camera
                    # (deterministic per training step, so all ranks agree) by
                    # setting plucker_6d=None, which short-circuits PluckerFlipPE
                    # in forward so attention keeps only the content path. Do NOT
                    # zero the tensor — a zero Plücker still routes through the PE
                    # module and produces a learned constant Q/K bias; the model
                    # then reads "zero plucker" as a strong "camera not moving"
                    # signal and collapses to near-static outputs. None also
                    # matches the eval --no-camera path (which passes None).
                    drop_cam = (
                        camera_drop > 0
                        and random.Random(train_steps).random() < camera_drop
                    )
                    # Full-batch nocam: camera dropout, force_no_cam, or every row
                    # is t2v with disable_plucker. Mixed batches (some rows t2v,
                    # others i2v) fall through to the per-sample mask branch below,
                    # where plucker_6d must stay a tensor for the rows needing PE.
                    if drop_cam or force_no_cam or (is_t2v_batch and disable_plucker_batch):
                        plucker_6d = None
                    elif is_t2v_any and disable_plucker_any:
                        if isinstance(is_t2v_raw, torch.Tensor):
                            t2v_mask = is_t2v_raw.to(device).float()
                        else:
                            t2v_mask = torch.full(
                                (B_batch,), float(is_t2v_batch), device=device
                            )
                        if disable_plucker_tensor is not None:
                            zero_mask = t2v_mask * disable_plucker_tensor
                        else:
                            zero_mask = t2v_mask * float(disable_plucker_batch)
                        plucker_6d = plucker_6d * (
                            1.0 - zero_mask.view(-1, 1, 1)
                        ).to(plucker_6d.dtype)

            # --- (2) Online text encode (frozen) -------------------------
            captions_blank = (
                [not (c or "").strip() for c in captions]
                if captions is not None else [True] * B_batch
            )
            captions_all_blank = all(captions_blank)
            text_dropout_roll = (
                random.random() if cfg_dropout_prob > 0 else 1.0
            )
            use_null_text_batch = (
                captions is None
                or captions_all_blank
                or text_dropout_roll < cfg_dropout_prob
            )
            with torch.no_grad():
                if use_null_text_batch:
                    ref_global = null_text_tokens.to(
                        dtype=torch.bfloat16,
                    ).expand(B_batch, -1, -1)
                else:
                    text_out = text_encoder(list(captions))
                    ref_global = text_out["tokens"].to(
                        device, dtype=torch.bfloat16,
                    )
                    if any(captions_blank):
                        blank_idx = torch.as_tensor(
                            [i for i, b in enumerate(captions_blank) if b],
                            device=device, dtype=torch.long,
                        )
                        ref_global[blank_idx] = null_text_tokens[0].to(
                            dtype=ref_global.dtype,
                        )

            # --- (3) Build conditioning slice + RAE loss -----------------
            if batch_cond_num > 0:
                z_5d_for_cond = rearrange(
                    z_input, "(b v) c h w -> b v c h w", v=batch_total_view,
                )
                z_ref_clean = rearrange(
                    z_5d_for_cond[:, :batch_cond_num],
                    "b v c h w -> (b v) c h w",
                )
            else:
                z_ref_clean = None

            mv_model_kwargs = dict(extra_cam_kwargs or {})
            if view_frame_idx is not None:
                mv_model_kwargs["view_frame_idx"] = view_frame_idx

            rgb_target = None
            if (
                use_rgb_loss
                and not is_precomputed
                and latent_backend in ("feature_vae", "da3_direct")
                and rae is not None
            ):
                rgb_target = rearrange(
                    (image - rae.encoder_mean[None]) / rae.encoder_std[None],
                    "b v c h w -> (b v) c h w",
                ).float()

            gt_geo = None
            if use_geo_loss and not is_precomputed and rae is not None:
                with torch.no_grad():
                    images_norm = (
                        (image - rae.encoder_mean[None]) / rae.encoder_std[None]
                    )
                    all_feats = rae.encode(images_norm, mode="all")
                    embed_dim = rae.encoder.backbone.pretrained.embed_dim
                    gt_geo = precompute_gt_dpt_geometry(
                        all_feats,
                        rae.encoder.backbone.pretrained.norm,
                        rae.rae_cl_decoder,
                        H_img,
                        W_img,
                        embed_dim=embed_dim,
                        total_view=batch_total_view,
                    )

            _rgb_kw = _rgb_loss_kwargs(rgb_target)
            _geo_kw = _geo_loss_kwargs(gt_geo, H_img, W_img, c2w_norm, Ks)
            loss_dict = training_losses_rae(
                ddp_dit if dist.is_initialized() else dit,
                x_target=z_all_gt,
                total_view=batch_total_view,
                time_dist_shift=time_dist_shift,
                time_dist_type=time_dist_type,
                prediction=prediction,
                base_model_coeff=base_model_coeff,
                repa_coeff=repa_weight if repa_enable else None,
                repa_target=None,
                z_ref_clean=z_ref_clean,
                reference_conditioning=reference_conditioning,
                plucker_6d=plucker_6d,
                ref_global=ref_global,
                cond_num=batch_cond_num,
                motion_loss_strength=motion_loss_strength,
                motion_loss_max_multiplier=motion_loss_max_multiplier,
                extra_model_kwargs=mv_model_kwargs or None,
                **_rgb_kw,
                **_geo_kw,
                **_denorm_loss_kwargs(_rgb_kw, _geo_kw),
            )
        return loss_dict, batch_cond_num, batch_total_view

    def _stop_early() -> bool:
        return max_train_steps is not None and train_steps >= max_train_steps

    def _apply_optimizer_step():
        """Shared NaN-scrub + clip + optimizer/scheduler/EMA step. Returns the
        grad norm (or None). Used by both the MV and T2I branches so the two
        domains advance the schedule and EMA identically."""
        num_bad = 0
        for p in dit.parameters():
            if p.grad is not None:
                bad = ~torch.isfinite(p.grad)
                if bad.any():
                    num_bad += int(bad.sum().item())
                    p.grad.masked_fill_(bad, 0.0)
        if num_bad > 0 and rank == 0:
            logger.warning(
                f"(step={train_steps:07d}) zeroed {num_bad} NaN/Inf grad elements"
            )
        gn = None
        if clip_grad:
            gn = torch.nn.utils.clip_grad_norm_(dit.parameters(), clip_grad)
        optimizer.step()
        scheduler.step()
        update_ema(ema_dit, dit, ema_decay)
        optimizer.zero_grad()
        return gn

    def _maybe_log_and_ckpt(grad_norm):
        """Shared logging + checkpoint, called at the end of every optim step
        (MV or T2I) so a log/ckpt boundary is never missed regardless of which
        domain landed on it."""
        if log_every > 0 and train_steps % log_every == 0 and rank == 0:
            torch.cuda.synchronize()
            dt = time() - ctr["t_start"]
            steps_per_sec = ctr["log_steps"] / max(dt, 1e-6)
            cur_lr = optimizer.param_groups[0]["lr"]
            gn_str = (
                f", grad={grad_norm.item():.2f}" if grad_norm is not None else ""
            )
            mv_n = max(ctr["mv"], 1)
            t2i_n = max(ctr["t2i"], 1)
            # MV branch label: "t2v" when camera is forced off (cond_num=0,
            # Plücker zeroed), else the legacy "i2v" (image-conditioned).
            mv_tag = "t2v" if os.environ.get("T2V_NO_CAMERA", "0") == "1" else "i2v"
            mv_str = (
                f"{mv_tag}[n={ctr['mv']} loss={running['loss']/mv_n:.4f} "
                f"main={running['main']/mv_n:.4f} base={running['base']/mv_n:.4f} "
                f"sf={running['self_flow']/mv_n:.4f} "
                f"rgb={running['rgb']/mv_n:.4f} geo={running['geo']/mv_n:.4f} "
                f"motion={running['motion_score']/mv_n:.4f} "
                f"wmax={running['motion_wmax']/mv_n:.2f} cn={ctr['last_cn']}]"
            )
            t2i_str = (
                f" t2i[n={ctr['t2i']} loss={t2i_run['loss']/t2i_n:.4f} "
                f"main={t2i_run['main']/t2i_n:.4f}]"
                if cotrain_enable else ""
            )
            by_view_str = ""
            if running_by_view:
                by_view_str = " by_view[" + " ".join(
                    f"v{v}={s['loss']/max(s['n'],1):.4f}(n={s['n']})"
                    for v, s in sorted(running_by_view.items())
                ) + "]"
            logger.info(
                f"(step={train_steps:07d}) {mv_str}{t2i_str}{by_view_str} "
                f"lr={cur_lr:.2e}{gn_str} step/s={steps_per_sec:.2f}"
            )
            if args.wandb and HAS_WANDB:
                wandb_utils.log({
                    "loss/total": running["loss"] / mv_n,
                    "loss/main": running["main"] / mv_n,
                    "loss/base": running["base"] / mv_n,
                    "loss/repa": running["repa"] / mv_n,
                    "loss/self_flow": running["self_flow"] / mv_n,
                    "loss/rgb": running["rgb"] / mv_n,
                    "loss/geo": running["geo"] / mv_n,
                    "motion/score_mean": running["motion_score"] / mv_n,
                    "motion/weight_max": running["motion_wmax"] / mv_n,
                    "lr": cur_lr,
                    "steps_per_sec": steps_per_sec,
                    "batch_cond_num": ctr["last_cn"],
                    **{
                        f"loss/view_{v}": s["loss"] / max(s["n"], 1)
                        for v, s in running_by_view.items()
                    },
                    **(
                        {"t2i/loss": t2i_run["loss"] / t2i_n,
                         "t2i/main": t2i_run["main"] / t2i_n}
                        if cotrain_enable and ctr["t2i"] > 0 else {}
                    ),
                    **(
                        {"grad_norm": grad_norm.item()}
                        if grad_norm is not None else {}
                    ),
                }, step=train_steps)
            for k in running:
                running[k] = 0.0
            for k in t2i_run:
                t2i_run[k] = 0.0
            running_by_view.clear()
            ctr["log_steps"] = 0
            ctr["mv"] = 0
            ctr["t2i"] = 0
            ctr["t_start"] = time()

        if ckpt_every > 0 and train_steps % ckpt_every == 0 and train_steps > 0:
            if rank == 0 and async_ckpt is not None:
                async_ckpt.save(
                    f"{ckpt_dir}/{train_steps:07d}.pt",
                    train_steps, epoch, ddp_dit, ema_dit, optimizer, scheduler,
                )
                sync_log(logger)
            if dist.is_initialized():
                dist.barrier()

    if t2i_only:
        # ── T2I-driven warmup loop ──────────────────────────────────────
        # Driven by the BLIP3o stream. With T2V_EVERY_K unset it is a pure-T2I
        # warmup (RE10K multi-view loop never iterated → no MV DA3/VAE encode).
        # With T2V_EVERY_K=k (>=2) the loop stays T2I-dominant but injects a
        # real t2v multi-view step every k optim steps so the camera / 3D path
        # is also exercised. Domain is chosen by the global ``train_steps``
        # (identical on every rank) so DDP never desyncs.
        t2v_every_k = int(os.environ.get("T2V_EVERY_K", "0") or "0")
        mv_iter = None
        if t2v_every_k:
            if t2v_every_k < 2:
                raise ValueError(
                    f"T2V_EVERY_K must be >=2 (T2I-dominant); got {t2v_every_k}."
                )

            def _mv_batches():
                """Infinite stream over the RE10K multi-view loader (re-iterated
                each epoch with a fresh shuffle), mirroring the main loop's
                per-epoch ``set_epoch`` wiring."""
                ep = start_epoch
                while True:
                    if hasattr(train_loader.dataset, "set_epoch"):
                        train_loader.dataset.set_epoch(ep)
                    if hasattr(train_loader, "batch_sampler") and hasattr(
                        train_loader.batch_sampler, "set_epoch"
                    ):
                        train_loader.batch_sampler.set_epoch(ep)
                    for b in train_loader:
                        yield b
                    ep += 1

            mv_iter = _mv_batches()

        epoch = start_epoch
        ddp_dit.train()
        if rank == 0:
            if mv_iter is not None:
                logger.info(
                    f"[T2I_ONLY+T2V] T2I-dominant warmup with t2v injection "
                    f"every {t2v_every_k} steps (~{100.0/t2v_every_k:.0f}% t2v); "
                    "RE10K multi-view loader iterated only on injected steps."
                )
            else:
                logger.info(
                    "[T2I_ONLY] pure-T2I training — RE10K multi-view loop skipped; "
                    "every optim step is a BLIP3o T2I step."
                )
        pbar = tqdm(t2i_iter, desc="T2I-only", disable=(rank != 0))
        for t2i_images, t2i_caps in pbar:
            # ── injected t2v multi-view step (deterministic schedule) ──────
            # The t2i batch just pulled is dropped on a t2v step (harmless for
            # an infinite stream), mirroring the main loop's MV-drop on a t2i
            # step.
            if mv_iter is not None and (
                train_steps % t2v_every_k == t2v_every_k - 1
            ):
                mv_loss_dict, batch_cond_num, batch_total_view = _mv_forward(next(mv_iter))
                mv_loss_value = float(mv_loss_dict["loss"].detach())
                mv_loss_dict["loss"].backward()
                running["loss"] += mv_loss_value
                running["main"] += float(mv_loss_dict["loss_main"])
                running["base"] += float(mv_loss_dict["loss_base"])
                running["repa"] += float(mv_loss_dict["loss_repa"])
                running["self_flow"] += float(mv_loss_dict.get("loss_self_flow", 0.0))
                running["rgb"] += float(mv_loss_dict.get("loss_rgb", 0.0))
                running["geo"] += float(mv_loss_dict.get("loss_geo", 0.0))
                running["motion_score"] += float(
                    mv_loss_dict.get("motion_score_mean", 0.0)
                )
                running["motion_wmax"] += float(
                    mv_loss_dict.get("motion_weight_max", 1.0)
                )
                _accum_view_loss(batch_total_view, mv_loss_value)
                grad_norm = _apply_optimizer_step()
                ctr["log_steps"] += 1
                ctr["mv"] += 1
                ctr["last_cn"] = batch_cond_num
                train_steps += 1
                _maybe_log_and_ckpt(grad_norm)
                if _stop_early():
                    break
                continue

            t2i_loss_dict = _t2i_forward_loss(t2i_images, t2i_caps)
            (t2i_loss_dict["loss"] * t2i_loss_weight).backward()
            t2i_run["loss"] += float(t2i_loss_dict["loss"].detach())
            t2i_run["main"] += float(t2i_loss_dict["loss_main"])
            t2i_run["base"] += float(t2i_loss_dict["loss_base"])
            grad_norm = _apply_optimizer_step()
            ctr["log_steps"] += 1
            ctr["t2i"] += 1
            train_steps += 1
            _maybe_log_and_ckpt(grad_norm)
            if _stop_early():
                break
        # The BLIP3o stream is infinite; reaching here means it ended.
        if rank == 0 and async_ckpt is not None:
            async_ckpt.join(timeout=600.0)
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    stop_training = False
    for epoch in range(start_epoch, epochs):
        if stop_training:
            break
        ddp_dit.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        if hasattr(train_loader, "batch_sampler") and hasattr(
            train_loader.batch_sampler, "set_epoch"
        ):
            train_loader.batch_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader, total=loader_batches,
            desc=f"Epoch {epoch}", disable=(rank != 0),
        )

        for batch in pbar:
            # ── T2I co-training step (deterministic schedule) ──────────────
            # Domain is chosen by global ``train_steps`` (identical on every
            # rank) so DDP never desyncs. The MV batch fetched by this iteration
            # is dropped on a T2I step (harmless for an infinite stream).
            if cotrain_enable and (
                train_steps % t2i_every_k == t2i_every_k - 1
            ):
                t2i_images, t2i_caps = next(t2i_iter)
                t2i_loss_dict = _t2i_forward_loss(t2i_images, t2i_caps)
                (t2i_loss_dict["loss"] * t2i_loss_weight).backward()
                t2i_run["loss"] += float(t2i_loss_dict["loss"].detach())
                t2i_run["main"] += float(t2i_loss_dict["loss_main"])
                t2i_run["base"] += float(t2i_loss_dict["loss_base"])
                grad_norm = _apply_optimizer_step()
                ctr["log_steps"] += 1
                ctr["t2i"] += 1
                train_steps += 1
                _maybe_log_and_ckpt(grad_norm)
                if _stop_early():
                    stop_training = True
                    break
                continue

            loss_dict, batch_cond_num, batch_total_view = _mv_forward(batch)
            mv_loss_value = float(loss_dict["loss"].detach())
            loss = loss_dict["loss"] / grad_accum_steps

            loss.backward()
            running["loss"] += mv_loss_value / grad_accum_steps
            running["main"] += float(loss_dict["loss_main"]) / grad_accum_steps
            running["base"] += float(loss_dict["loss_base"]) / grad_accum_steps
            running["repa"] += float(loss_dict["loss_repa"]) / grad_accum_steps
            running["self_flow"] += float(loss_dict.get("loss_self_flow", 0.0)) / grad_accum_steps
            running["rgb"] += float(loss_dict.get("loss_rgb", 0.0)) / grad_accum_steps
            running["geo"] += float(loss_dict.get("loss_geo", 0.0)) / grad_accum_steps
            running["motion_score"] += float(
                loss_dict.get("motion_score_mean", 0.0)
            ) / grad_accum_steps
            running["motion_wmax"] += float(
                loss_dict.get("motion_weight_max", 1.0)
            ) / grad_accum_steps
            _accum_view_loss(batch_total_view, mv_loss_value)
            accum_counter += 1
            if accum_counter < grad_accum_steps:
                continue

            grad_norm = _apply_optimizer_step()
            accum_counter = 0
            ctr["log_steps"] += 1
            ctr["mv"] += 1
            ctr["last_cn"] = batch_cond_num
            train_steps += 1
            _maybe_log_and_ckpt(grad_norm)
            if _stop_early():
                stop_training = True
                break

    if rank == 0 and max_train_steps is not None and train_steps >= max_train_steps:
        logger.info(f"[debug] stopped early at train_steps={train_steps}")
    if rank == 0 and async_ckpt is not None:
        async_ckpt.join(timeout=600.0)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
