"""
Train the GAE codec (Stage 1) — paper Section 3.1.

Compresses the four-level frozen DA3 hierarchy into a compact per-view latent
and trains it to rebuild every level, so the frozen DA3 DPT head can still read
depth / rays / point maps out of the reconstruction. A learned RGB head renders
appearance from the same latent.

Loss (paper Eq. 12):

    L_codec = L_feat + lambda_kl * L_kl + L_rgb + L_geo + L_repr
    L_repr  = lambda_repa * (L_tok + lambda_mu * L_struct)

L_feat / L_kl come from GAECodec.compute_loss; L_rgb / L_geo / L_repr are
computed here because they need the image targets and the frozen teachers.
The two L_struct variants are both available:

* repa_struct_mu_weight — matches DINOv2 similarities in the **raw posterior
  mean** space. This is the paper's term (Eq. 9-10).
* repa_struct_weight    — the same loss applied to the *projected* tokens.
  Kept for the ablation in Table 2; leave at 0 for the paper configuration.

Usage:
    torchrun --nproc_per_node=N scripts/train/train_codec.py \
        --config configs/gae_64.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import tempfile
from pathlib import Path as _Path

# Allow running as a script from the repo root: make src/ importable.
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))

os.environ.setdefault("TMPDIR", "/tmp")
tempfile.tempdir = "/tmp"

import torch
import torch.nn.functional as F
import torch.backends.cuda
if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
    torch.backends.cuda.is_flash_attention_available = lambda: False

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from collections import defaultdict
from copy import deepcopy
from glob import glob
from typing import Dict

import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import default_collate
from tqdm import tqdm
from omegaconf import OmegaConf

from stage1.da3 import DA3Backbone
from stage1.gae_codec import GAECodec
from stage1.repa_target import repa_cosine_loss, repa_similarity_loss
from disc import LPIPS, build_discriminator
from disc.gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from utils.model_utils import instantiate_from_config
from utils.optim_utils import build_optimizer, build_scheduler
from cut3r_data import get_data_loader
from cut3r_data.utils.resolution import parse_resolution_list

try:
    import wandb
    from utils import wandb_utils
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ImageNet normalization constants
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def imagenet_to_pm1(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized → [-1, 1]."""
    mean = _IMG_MEAN.to(x.device, x.dtype)
    std = _IMG_STD.to(x.device, x.dtype)
    return ((x * std + mean) * 2.0 - 1.0).clamp(-1, 1)


def rgb_frames_to_video(frames: torch.Tensor, num_views: int) -> torch.Tensor:
    """Convert contiguous ``[B*V,C,H,W]`` frames to ``[B,C,V,H,W]``."""
    if frames.ndim != 4 or num_views <= 0 or frames.shape[0] % num_views != 0:
        raise ValueError(
            f"Cannot reshape frames={tuple(frames.shape)} with num_views={num_views}"
        )
    batch = frames.shape[0] // num_views
    return frames.reshape(batch, num_views, *frames.shape[1:]).permute(0, 2, 1, 3, 4)


def spatial_crop_last2(
    tensor: torch.Tensor,
    crop_size: int,
    params: tuple[int, int, int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Apply one shared spatial crop to either BCHW or BCTHW tensors."""
    height, width = tensor.shape[-2:]
    crop_h, crop_w = min(crop_size, height), min(crop_size, width)
    if params is None:
        top = int(torch.randint(0, height - crop_h + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_w + 1, (1,)).item())
        params = (top, left, crop_h, crop_w)
    top, left, crop_h, crop_w = params
    return tensor[..., top : top + crop_h, left : left + crop_w], params


def latent_jitter_decision(global_step: int, prob: float, device) -> bool:
    """Decide whether to apply jitter this step, identically across all ranks.

    Seeded by ``global_step`` so every DDP rank makes the same choice, keeping
    the per-step autograd graph identical (required since DDP forward toggles
    the RGB head on/off based on this decision).
    """
    if prob <= 0:
        return False
    if prob >= 1.0:
        return True
    g = torch.Generator(device=device).manual_seed(int(global_step))
    return bool((torch.rand((), generator=g, device=device) < prob).item())


def rgb_frame_diff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_views: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier loss on RGB frame differences, preserving real motion."""
    if num_views <= 1 or pred.shape[0] % num_views != 0:
        return pred.new_zeros(())
    bsz = pred.shape[0] // num_views
    pred_v = pred.view(bsz, num_views, *pred.shape[1:])
    target_v = target.view(bsz, num_views, *target.shape[1:])
    diff = (pred_v[:, 1:] - pred_v[:, :-1]) - (target_v[:, 1:] - target_v[:, :-1])
    return torch.sqrt(diff.float().pow(2) + float(eps) ** 2).mean().to(dtype=pred.dtype)


def rgb_high_freq_frame_diff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_views: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Charbonnier loss on adjacent-frame changes in a 3x3 high-pass band."""
    if num_views <= 1 or pred.shape[0] % num_views != 0:
        return pred.new_zeros(())

    def high_pass(x: torch.Tensor) -> torch.Tensor:
        # Spatial low-pass independently per frame/channel; no temporal blur.
        return x.float() - F.avg_pool2d(x.float(), kernel_size=3, stride=1, padding=1)

    bsz = pred.shape[0] // num_views
    pred_h = high_pass(pred).view(bsz, num_views, *pred.shape[1:])
    target_h = high_pass(target).view(bsz, num_views, *target.shape[1:])
    diff = ((pred_h[:, 1:] - pred_h[:, :-1])
            - (target_h[:, 1:] - target_h[:, :-1]))
    return torch.sqrt(diff.pow(2) + float(eps) ** 2).mean().to(dtype=pred.dtype)


def imagenet_to_rgb01(x: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized RGB frames to display-space ``[0, 1]``."""
    mean = _IMG_MEAN.to(x.device, x.dtype)
    std = _IMG_STD.to(x.device, x.dtype)
    return (x * std + mean).clamp(0, 1)


def rgb_temporal_haar3d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_views: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """L1 on causal Haar-3D temporal-high-pass RGB coefficients.

    Args:
        pred: ImageNet-normalized reconstructed frames, ``[B*V, 3, H, W]``.
        target: Aligned ImageNet-normalized GT frames, ``[B*V, 3, H, W]``.
        num_views: Number of contiguous frames per video.
    """
    if num_views <= 1 or pred.ndim != 4 or pred.shape != target.shape:
        zero = pred.new_zeros(())
        return zero, {name: zero for name in ("hll", "hlh", "hhl", "hhh")}
    if pred.shape[0] % num_views != 0:
        raise ValueError(
            f"Cannot compute Haar-3D loss for frames={tuple(pred.shape)}, "
            f"num_views={num_views}"
        )

    # Exact temporal-high-pass half of WF-VAE's causal Haar-3D filter bank.
    kernels = pred.new_tensor(
        [
            [[[1, 1], [1, 1]], [[-1, -1], [-1, -1]]],  # HLL
            [[[1, -1], [1, -1]], [[-1, 1], [-1, 1]]],  # HLH
            [[[1, 1], [-1, -1]], [[-1, -1], [1, 1]]],  # HHL
            [[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]],  # HHH
        ],
        dtype=torch.float32,
    ).unsqueeze(1) * 0.3536  # [4, 1, 2, 2, 2]

    batch = pred.shape[0] // num_views
    channels = pred.shape[1]

    def transform(frames: torch.Tensor) -> torch.Tensor:
        video = imagenet_to_rgb01(frames.float()).reshape(
            batch, num_views, channels, *frames.shape[-2:]
        )
        video = video.permute(0, 2, 1, 3, 4).float()  # [B, C, V, H, W]
        video = torch.cat((video[:, :, :1], video), dim=2)
        grouped_kernels = kernels.repeat(channels, 1, 1, 1, 1)
        with autocast(device_type=video.device.type, enabled=False):
            coefficients = F.conv3d(
                video,
                grouped_kernels,
                stride=2,
                padding=0,
                groups=channels,
            )
        # Grouped convolution is channel-major: [B, C*4, T', H', W'].
        return coefficients.view(batch, channels, 4, *coefficients.shape[-3:])

    pred_coefficients = transform(pred)
    with torch.no_grad():
        target_coefficients = transform(target)

    names = ("hll", "hlh", "hhl", "hhh")
    band_losses = {
        name: F.l1_loss(
            pred_coefficients[:, :, index],
            target_coefficients[:, :, index],
        )
        for index, name in enumerate(names)
    }
    total = torch.stack(tuple(band_losses.values())).mean()
    return total, band_losses


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """VQGAN/WF-VAE adaptive GAN weight = ‖∇recon‖ / ‖∇gan‖ at `layer`.

    Norms are reduced in fp32: under bf16 autocast the grad magnitudes at the
    RGB head output layer fall near the bf16 resolution limit, which makes the
    ratio jump between steps.
    """
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads.float()) / (torch.norm(gan_grads.float()) + 1e-6)
    return torch.clamp(d_weight, 0.0, max_d_weight).detach()


def select_gan_losses(disc_kind: str, gen_kind: str):
    d_fn = {"hinge": hinge_d_loss, "vanilla": vanilla_d_loss}[disc_kind]
    g_fn = {"vanilla": vanilla_g_loss}[gen_kind]
    return d_fn, g_fn


# ── Pose (ray) supervision helpers ──

_DA3_EMBED_DIM = 768  # Default for DA3-Base; overridden at runtime from encoder

from utils.dpt_helpers import (  # noqa: E402
    _format_feats_for_dpt,
    _format_recon_for_dpt,
    _geo_view_indices,
)
def _pick_scene_indices(
    batch_size: int,
    max_scenes: int,
    global_step: int,
    device: torch.device,
) -> list[int]:
    """DDP-safe random scene subsampling for heavy frozen-head supervision."""
    max_scenes = max(1, max_scenes)
    if batch_size <= max_scenes:
        return list(range(batch_size))
    g = torch.Generator(device=device).manual_seed(int(global_step))
    perm = torch.randperm(batch_size, generator=g, device=device)
    return perm[:max_scenes].tolist()


def _resolve_scene_indices(
    batch_size: int,
    max_scenes: int,
    global_step: int,
    device: torch.device,
) -> list[int]:
    """``max_scenes=0`` → all scenes; ``>0`` → random subset (DDP-synced)."""
    if max_scenes <= 0 or batch_size <= max_scenes:
        return list(range(batch_size))
    return _pick_scene_indices(batch_size, max_scenes, global_step, device)


def _single_scene_geo_loss(
    gt_feats,
    recon_feats,
    vae_mod,
    backbone_norm,
    dpt_decoder,
    H,
    W,
    device,
    embed_dim=None,
):
    n = next(iter(recon_feats.values())).shape[0]

    with torch.no_grad():
        gt_dpt_in = _format_feats_for_dpt(gt_feats, backbone_norm, embed_dim=embed_dim)
        with torch.autocast(device_type=device.type, enabled=False):
            gt_out = dpt_decoder(gt_dpt_in, H, W, patch_start_idx=0)
        gt_ray = gt_out["ray"].detach()
        gt_depth = gt_out.get("depth")
        if gt_depth is not None:
            gt_depth = gt_depth.detach()
        gt_rc = gt_out.get("ray_conf")

    vae_dpt_in = _format_recon_for_dpt(recon_feats, backbone_norm, n, embed_dim=embed_dim)
    with torch.autocast(device_type=device.type, enabled=False):
        vae_out = dpt_decoder(vae_dpt_in, H, W, patch_start_idx=0)
    vae_ray = vae_out["ray"]
    vae_depth = vae_out.get("depth")

    if gt_rc is not None:
        conf = gt_rc.detach().sigmoid()
        while conf.ndim < gt_ray.ndim:
            conf = conf.unsqueeze(-1)
        ray_l1 = (conf * (vae_ray - gt_ray).abs()).sum() / (conf.sum() + 1e-8)
    else:
        ray_l1 = F.l1_loss(vae_ray, gt_ray)

    if gt_depth is not None and vae_depth is not None:
        depth_l1 = F.l1_loss(vae_depth, gt_depth)
    else:
        depth_l1 = torch.zeros(1, device=device)

    return ray_l1, depth_l1


def _compute_geo_loss(recon, all_feats, vae_mod, backbone_norm, dpt_decoder,
                      H, W, device, max_views=8, embed_dim=None, num_views=None):
    """Compare DPT-predicted rays & depth from GT vs VAE-reconstructed features.

    Gradients flow: loss → vae outputs → DPT (frozen) → recon_feats → VAE decoder.
    Returns: (ray_l1, depth_l1) — both scalar tensors with grad.
    """
    recon_feats = vae_mod.denormalize_and_split(recon.float())
    bv = recon.shape[0]
    if num_views is None or num_views <= 0 or bv % num_views != 0:
        num_views = bv
    batch_size = max(1, bv // num_views)
    view_idx = _geo_view_indices(num_views, max_views, device)

    ray_losses = []
    depth_losses = []
    for b in range(batch_size):
        scene_idx = b * num_views + view_idx
        gt_scene = {k: v[scene_idx] for k, v in all_feats.items()}
        recon_scene = {k: v[scene_idx] for k, v in recon_feats.items()}
        ray_l1, depth_l1 = _single_scene_geo_loss(
            gt_scene, recon_scene, vae_mod, backbone_norm, dpt_decoder,
            H, W, device, embed_dim=embed_dim,
        )
        ray_losses.append(ray_l1)
        depth_losses.append(depth_l1)

    return torch.stack(ray_losses).mean(), torch.stack(depth_losses).mean()


def _single_scene_gs_distill_loss(
    gt_feats,
    recon_feats,
    imgs_scene,
    backbone_norm,
    gs_head,
    H,
    W,
    device,
    embed_dim=None,
    conf_weight=1.0,
    eps=1e-6,
):
    """Distill frozen gs_head outputs (camera-space raw gaussians) GT→recon.

    Both branches use the SAME GT images, so only the feature path differs.
    Gradients flow: loss → recon raw_gs → gs_head (frozen) → recon_feats → VAE.
    raw_gs ([1, V, h, w, D]) channels are heterogeneous (scales/quat/sh/...),
    so normalize per-channel by the GT mean-abs; raw_gs_conf (density) is
    normalized by its own mean-abs.
    """
    n = next(iter(recon_feats.values())).shape[0]

    with torch.no_grad():
        gt_in = _format_feats_for_dpt(gt_feats, backbone_norm, embed_dim=embed_dim)
        with torch.autocast(device_type=device.type, enabled=False):
            gt_out = gs_head(gt_in, H, W, patch_start_idx=0, images=imgs_scene)
        gt_raw = gt_out.raw_gs.detach()              # [1, V, h, w, D]
        gt_conf = gt_out.raw_gs_conf.detach()        # [1, V, h, w]

    recon_in = _format_recon_for_dpt(recon_feats, backbone_norm, n, embed_dim=embed_dim)
    with torch.autocast(device_type=device.type, enabled=False):
        recon_out = gs_head(recon_in, H, W, patch_start_idx=0, images=imgs_scene)

    scale = gt_raw.abs().mean(dim=(0, 1, 2, 3), keepdim=True).clamp_min(eps)  # per-channel
    gs_l1 = ((recon_out.raw_gs - gt_raw).abs() / scale).mean()

    conf_scale = gt_conf.abs().mean().clamp_min(eps)
    conf_l1 = ((recon_out.raw_gs_conf - gt_conf).abs() / conf_scale).mean()

    return gs_l1 + conf_weight * conf_l1


def _gs_distill_scene_plan(
    batch_size: int,
    num_views: int,
    max_views: int,
    max_scenes: int,
    global_step: int,
    device: torch.device,
) -> tuple[list[int], torch.Tensor]:
    """Pick scenes and view indices for GS distill."""
    scene_indices = _resolve_scene_indices(
        batch_size, max_scenes, global_step, device)
    view_idx = _geo_view_indices(num_views, max_views, device)
    return scene_indices, view_idx


def _compute_gs_distill_loss(recon, all_feats, vae_mod, backbone_norm, gs_head,
                             images, H, W, device, max_views=8, embed_dim=None,
                             num_views=None, conf_weight=1.0, max_scenes=1,
                             global_step=0):
    """GS distillation scalar loss (train + val).

    ``max_scenes=1`` (default for training) picks one DDP-synced random scene
    per step so a single ``backward()`` suffices without multi-scene gs_head OOM.
    Set ``max_scenes=0`` to average over every scene in the microbatch (val).
    """
    recon_feats = vae_mod.denormalize_and_split(recon.float())
    bv = recon.shape[0]
    if num_views is None or num_views <= 0 or bv % num_views != 0:
        num_views = bv
    batch_size = max(1, bv // num_views)
    scene_indices, view_idx = _gs_distill_scene_plan(
        batch_size, num_views, max_views, max_scenes, global_step, device)

    losses = []
    for b in scene_indices:
        scene_idx = b * num_views + view_idx
        gt_scene = {k: v[scene_idx] for k, v in all_feats.items()}
        recon_scene = {k: v[scene_idx] for k, v in recon_feats.items()}
        imgs_scene = images[b, view_idx].unsqueeze(0)  # [1, V, 3, H, W]
        losses.append(_single_scene_gs_distill_loss(
            gt_scene, recon_scene, imgs_scene, backbone_norm, gs_head,
            H, W, device, embed_dim=embed_dim, conf_weight=conf_weight,
        ))

    return torch.stack(losses).mean()


def parse_args():
    p = argparse.ArgumentParser(description="Train Feature VAE v2")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--results-dir", type=str, default="results/gae-codec")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Resume from checkpoint (weights + optimizer + step)")
    p.add_argument("--init-ckpt", type=str, default=None,
                   help="Init weights only (no optimizer/step)")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


_HF_TO_LOCAL_DA3 = {
    "depth-anything/DA3-LARGE-1.1": "pretrained_models/da3_large",
    "depth-anything/DA3-Large": "pretrained_models/da3_large",
    "depth-anything/DA3-Base": "pretrained_models/da3",
    "depth-anything/DA3-GIANT-1.1": "pretrained_models/da3_giant",
    "depth-anything/DA3-Giant": "pretrained_models/da3_giant",
}


def _resolve_da3_encoder_path(pretrained_path, da3_weights_path=None):
    """Prefer a local ``pretrained_models/`` dir (config.json + weights) over HF Hub."""
    proj_root = str(_Path(__file__).resolve().parents[2])
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


def _ensure_da3_hub_cached(pretrained_path: str, logger) -> None:
    """Download DA3 once per node (LOCAL_RANK 0) before all ranks load it."""
    if not pretrained_path or os.path.isdir(pretrained_path):
        return
    if "/" not in pretrained_path:
        return
    from depth_anything_3.api import DepthAnything3

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logger.info(
            f"[da3] caching {pretrained_path} from HF Hub on local_rank 0 "
            f"(set HF_TOKEN for higher rate limits) ..."
        )
        try:
            DepthAnything3.from_pretrained(pretrained_path)
        except Exception as exc:
            raise RuntimeError(
                f"HF Hub download failed for {pretrained_path!r}. "
                f"If the cache is corrupted, remove incomplete files under "
                f"$HUGGINGFACE_HUB_CACHE/models--{pretrained_path.replace('/', '--')} "
                f"and retry, or place config.json + model.safetensors under "
                f"pretrained_models/da3_giant/. Original error: {exc}"
            ) from exc
    if dist.is_initialized():
        dist.barrier()


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
    """Flush log file and copy from /local-ssd to remote storage."""
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


@torch.no_grad()
def update_ema(ema, model, decay):
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.mul_(decay).add_(mp.data, alpha=1 - decay)


def _robust_collate(batch):
    """default_collate 的容错版：dict 只 collate 所有样本共有的 key。

    bs>1 时一个 batch 可能混入不同子数据集的样本（CatDataset 全局洗牌），
    各库 view dict 的 key 不一致（如 OSP 带 disable_plucker/is_t2v/caption，
    几何库不带）。default_collate 对缺失 key 直接 KeyError。这里在 dict 层只
    保留交集 key——几何 key（img/camera_*）所有库都有故照常保留，数据集专属
    的元数据 key 在混 batch 时被丢弃（feature VAE 训练只消费 img，不受影响）。
    """
    elem = batch[0]
    if isinstance(elem, dict):
        common = set(elem.keys())
        for d in batch[1:]:
            common &= set(d.keys())
        return {k: _robust_collate([d[k] for d in batch])
                for k in elem.keys() if k in common}
    if isinstance(elem, (list, tuple)):
        return [_robust_collate(list(s)) for s in zip(*batch)]
    return default_collate(batch)


def prepare_dataloader(dataset_cfg, batch_size, workers, rank, world_size, test=False):
    return get_data_loader(
        dataset_cfg, batch_size=batch_size, num_workers=workers,
        pin_mem=True, shuffle=not test, drop_last=not test,
        fixed_length=True, world_size=world_size, rank=rank,
        collate_fn=_robust_collate,
    )


def _pick_staging_dir(skip_prefix=None):
    """Return first writable staging dir or None.

    Tested in order: /local-ssd, /dev/shm, /tmp. We probe writability with an
    actual touch because some hosts have /local-ssd mount points whose stat()
    succeeds but writes fail (Links=0 / re-mounted while in use).
    """
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
    """Save to remote `path` via local staging when possible.

    Some network filesystems do not support atomic rename or in-place updates,
    so we stage to a fast local dir (/local-ssd, /dev/shm, or /tmp) and `cp`
    the file over.  When no local staging dir is writable, we fall back to
    writing the target path directly.
    """
    import subprocess
    if not path.endswith(".pt"):
        raise ValueError(f"_safe_torch_save refuses non-.pt path: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    target_prefix = "/local-ssd" if path.startswith("/local-ssd") else None
    stage_dir = _pick_staging_dir(skip_prefix=target_prefix)

    if stage_dir is None:
        # No usable staging area; write target directly. May be slow on S3-FUSE
        # but at least won't lose the checkpoint.
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


def save_checkpoint(path, step, epoch, vae, ema_vae, optimizer, scheduler,
                    disc=None, disc_optimizer=None, disc_scheduler=None):
    state = {
        "step": step, "epoch": epoch,
        "vae": vae.module.state_dict() if isinstance(vae, DDP) else vae.state_dict(),
        "ema_vae": ema_vae.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
    }
    if disc is not None:
        state["disc"] = disc.module.state_dict() if isinstance(disc, DDP) else disc.state_dict()
    if disc_optimizer is not None:
        state["disc_optimizer"] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        state["disc_scheduler"] = disc_scheduler.state_dict()
    _safe_torch_save(state, path)


def load_checkpoint(path, vae_ddp, ema_vae, optimizer, scheduler,
                    disc_ddp=None, disc_optimizer=None, disc_scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    mod = vae_ddp.module if isinstance(vae_ddp, DDP) else vae_ddp
    mod.load_state_dict(ckpt["vae"])
    ema_vae.load_state_dict(ckpt["ema_vae"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if disc_ddp is not None and "disc" in ckpt:
        disc_mod = disc_ddp.module if isinstance(disc_ddp, DDP) else disc_ddp
        disc_mod.load_state_dict(ckpt["disc"])
    if disc_optimizer is not None and "disc_optimizer" in ckpt:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
    if disc_scheduler is not None and "disc_scheduler" in ckpt:
        disc_scheduler.load_state_dict(ckpt["disc_scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0)


def load_weights_only(path, vae_ddp, ema_vae):
    """部分暖启动：strict=False 仅忽略 missing/unexpected key，shape 不一致也要丢弃，
    否则 PyTorch 仍会抛 RuntimeError（典型场景：d128 ckpt → d64 模型，瓶颈投影需重 init）。"""
    ckpt = torch.load(path, map_location="cpu")
    mod = vae_ddp.module if isinstance(vae_ddp, DDP) else vae_ddp
    vae_sd = ckpt.get("vae", ckpt.get("ema_vae", ckpt))

    cur_sd = mod.state_dict()
    filtered_sd = {}
    shape_mismatch = []
    for k, v in vae_sd.items():
        if k in cur_sd and hasattr(v, "shape") and tuple(v.shape) != tuple(cur_sd[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(cur_sd[k].shape)))
            continue
        filtered_sd[k] = v

    missing, unexpected = mod.load_state_dict(filtered_sd, strict=False)
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        log = logging.getLogger(__name__)
        if shape_mismatch:
            log.info(f"[init_ckpt] dropped {len(shape_mismatch)} shape-mismatched key(s); "
                     f"these will keep current-model init:")
            for k, ck, mk in shape_mismatch:
                log.info(f"    {k}: ckpt {ck} vs model {mk}")
        if missing:
            log.info(f"[init_ckpt] missing {len(missing)} key(s) (kept as fresh init): "
                     f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
        if unexpected:
            log.info(f"[init_ckpt] unexpected {len(unexpected)} key(s) (ignored): "
                     f"{unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    ema_vae.load_state_dict(mod.state_dict())


@torch.no_grad()
def run_validation(rae, vae, val_loader, device, ac_kwargs,
                   rgb_weight=0.0, lpips_fn=None, lpips_weight=0.0,
                   rgb_temporal_haar3d_weight=0.0,
                   pose_weight=0.0, geo_depth_weight=0.0, pose_max_views=8,
                   max_batches=20, prefix="val", repa_target=None,
                   repa_struct_weight=0.0, repa_struct_max_tokens=0,
                   repa_struct_mu_weight=0.0, repa_struct_target=None,
                   gs_distill_weight=0.0, gs_distill_max_views=8,
                   gs_distill_conf_weight=1.0, gs_distill_max_scenes=0):
    rae.eval(); vae.eval()
    metrics = defaultdict(float)
    n = 0
    val_loader.dataset.set_epoch(0)
    val_loader.batch_sampler.set_epoch(0)

    need_geo = (pose_weight > 0 or geo_depth_weight > 0) and rae.rae_cl_decoder is not None
    need_gs = gs_distill_weight > 0 and getattr(rae, "gs_head", None) is not None
    backbone_norm = (rae.encoder.backbone.pretrained.norm
                     if (need_geo or need_gs) else None)

    for i, image_dict in enumerate(val_loader):
        if i >= max_batches:
            break
        images = torch.stack([d["img"] for d in image_dict], dim=1).to(device, non_blocking=True)
        img_size = (images.shape[-2], images.shape[-1])
        with autocast(**ac_kwargs):
            all_feats = rae.encode(images, mode="all")
            feats = {k: v[:, 1:, :] for k, v in all_feats.items()}
            x = vae.normalize_levels(feats, image_size=img_size)
            out = vae(x, num_views=images.shape[1])
            losses = vae.compute_loss(x, out, per_level=True)

            if "rgb_pred" in out and rgb_weight > 0:
                gt_flat = images.reshape(-1, 3, *img_size)
                rgb_l1 = F.l1_loss(out["rgb_pred"], gt_flat)
                losses["rgb_l1"] = rgb_l1
                rgb_total = rgb_weight * rgb_l1

                if lpips_fn is not None and lpips_weight > 0:
                    pred_pm1 = imagenet_to_pm1(out["rgb_pred"])
                    gt_pm1 = imagenet_to_pm1(gt_flat)
                    lp = lpips_fn(pred_pm1.float(), gt_pm1.float())
                    losses["rgb_lpips"] = lp
                    rgb_total = rgb_total + lpips_weight * lp

                if rgb_temporal_haar3d_weight > 0:
                    haar3d, haar3d_bands = rgb_temporal_haar3d_loss(
                        out["rgb_pred"], gt_flat, images.shape[1]
                    )
                    losses["rgb_haar3d"] = haar3d
                    for band_name, band_loss in haar3d_bands.items():
                        losses[f"rgb_haar3d_{band_name}"] = band_loss
                    rgb_total = rgb_total + rgb_temporal_haar3d_weight * haar3d

                losses["loss"] = losses["loss"] + rgb_total

            if need_geo:
                H, W = img_size
                vae_mod = vae.module if isinstance(vae, DDP) else vae
                ray_l1, depth_l1 = _compute_geo_loss(
                    out["recon"], all_feats, vae_mod, backbone_norm,
                    rae.rae_cl_decoder, H, W, device, max_views=pose_max_views,
                    num_views=images.shape[1],
                    embed_dim=rae.encoder.backbone.pretrained.embed_dim)
                if pose_weight > 0:
                    losses["ray_l1"] = ray_l1
                    losses["loss"] = losses["loss"] + pose_weight * ray_l1
                if geo_depth_weight > 0:
                    losses["geo_depth_l1"] = depth_l1
                    losses["loss"] = losses["loss"] + geo_depth_weight * depth_l1

            if need_gs:
                H, W = img_size
                vae_mod = vae.module if isinstance(vae, DDP) else vae
                gs_l = _compute_gs_distill_loss(
                    out["recon"], all_feats, vae_mod, backbone_norm, rae.gs_head,
                    images, H, W, device, max_views=gs_distill_max_views,
                    embed_dim=rae.encoder.backbone.pretrained.embed_dim,
                    num_views=images.shape[1], conf_weight=gs_distill_conf_weight,
                    max_scenes=0, global_step=0)
                losses["gs_distill"] = gs_l
                losses["loss"] = losses["loss"] + gs_distill_weight * gs_l

            # Self-REPA 对齐（仅记录原始余弦距离，不计入 val loss）
            if repa_target is not None and vae.repa_proj is not None:
                latent_hw = (out["mu"].shape[-2], out["mu"].shape[-1])
                gt_flat = images.reshape(-1, 3, *img_size)
                gt_flat01 = (gt_flat * _IMG_STD.to(gt_flat.device, gt_flat.dtype)
                             + _IMG_MEAN.to(gt_flat.device, gt_flat.dtype)).clamp(0, 1)
                struct_src = repa_struct_target if repa_struct_target is not None else repa_target
                with autocast(**{**ac_kwargs, "enabled": False}):
                    repa_tgt = repa_target.features(gt_flat01, all_feats, latent_hw)
                    if struct_src is repa_target:
                        struct_tgt = repa_tgt
                    else:
                        struct_tgt = struct_src.features(gt_flat01, all_feats, latent_hw)
                z_proj = vae.repa_project(out["mu"])
                losses["repa"] = repa_cosine_loss(z_proj, repa_tgt.to(z_proj.dtype))
                if repa_struct_weight > 0:
                    losses["repa_struct"] = repa_similarity_loss(
                        z_proj, struct_tgt.to(z_proj.dtype),
                        max_tokens=repa_struct_max_tokens,
                    )
                if repa_struct_mu_weight > 0:
                    b_, c_, h_, w_ = out["mu"].shape
                    mu_tok = out["mu"].permute(0, 2, 3, 1).reshape(b_, h_ * w_, c_)
                    losses["repa_struct_mu"] = repa_similarity_loss(
                        mu_tok, struct_tgt.to(mu_tok.dtype),
                        max_tokens=repa_struct_max_tokens,
                    )

        for k, v in losses.items():
            metrics[k] += v.item()
        n += 1
    return {f"{prefix}/{k}": v / max(n, 1) for k, v in metrics.items()}


def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()

    cfg = OmegaConf.load(args.config)
    train_cfg = OmegaConf.to_container(cfg.get("training", {}), resolve=True)
    vae_cfg = OmegaConf.to_container(cfg.get("codec", {}), resolve=True)
    rgb_decoder_config = os.environ.get("GLD_RGB_DECODER_CONFIG", "")
    if rgb_decoder_config:
        decoder_source = OmegaConf.load(rgb_decoder_config)
        source_decoder = decoder_source.get("codec", {}).get("rgb_decoder")
        if source_decoder is None:
            raise ValueError(
                "GLD_RGB_DECODER_CONFIG must contain codec.rgb_decoder: "
                f"{rgb_decoder_config}"
            )
        vae_cfg["rgb_decoder"] = OmegaConf.to_container(source_decoder, resolve=True)
    rae_cfg = cfg.get("stage_1")
    gan_cfg = OmegaConf.to_container(cfg.get("gan", {}), resolve=True) or {}
    gan_override_config = os.environ.get("GLD_GAN_CONFIG", "")
    if gan_override_config:
        gan_source = OmegaConf.load(gan_override_config)
        source_gan = gan_source.get("gan")
        if source_gan is None:
            raise ValueError(
                f"GLD_GAN_CONFIG must contain gan: {gan_override_config}"
            )
        gan_cfg = OmegaConf.to_container(source_gan, resolve=True)

    # ── Optional: align loss objectives with a reference config (e.g. full-ft) ──
    # GLD_LOSS_ALIGN_CONFIG points to a YAML whose `training` loss weights/flags,
    # `repa` section, and `codec` level_weights/kl_weight are merged over the current
    # config. Lets the head-only spacetime recipe run as a full fine-tune with the
    # exact full-ft objectives. GAN stays independently controlled by GLD_GAN_CONFIG.
    # Unset (default) => no-op.
    loss_align_config = os.environ.get("GLD_LOSS_ALIGN_CONFIG", "")
    if loss_align_config:
        _ref = OmegaConf.load(loss_align_config)
        _ref_train = OmegaConf.to_container(_ref.get("training", {}), resolve=True) or {}
        _loss_keys = (
            "train_rgb_decoder_only", "rgb_only_loss",
            "rgb_weight", "lpips_weight", "lpips_start_step",
            "pose_weight", "geo_depth_weight", "pose_start_step", "pose_max_views",
            "gs_distill_weight", "gs_distill_max_views", "gs_distill_max_scenes",
            "gs_distill_max_num_views", "gs_distill_start_step", "gs_distill_conf_weight",
        )
        _applied = {}
        for _k in _loss_keys:
            if _k in _ref_train:
                train_cfg[_k] = _ref_train[_k]
                _applied[_k] = _ref_train[_k]
        # The full-ft reference omits these flags (defaults false = whole codec
        # trainable); force them so the align path is a real full fine-tune instead
        # of silently inheriting the head-only freeze.
        for _flag in ("train_rgb_decoder_only", "rgb_only_loss"):
            train_cfg[_flag] = bool(_ref_train.get(_flag, False))
            _applied[_flag] = train_cfg[_flag]
        _ref_vae = OmegaConf.to_container(_ref.get("codec", {}), resolve=True) or {}
        for _k in ("level_weights", "kl_weight"):
            if _k in _ref_vae:
                vae_cfg[_k] = _ref_vae[_k]
                _applied[_k] = _ref_vae[_k]
        _ref_repa = _ref.get("repa", None)
        if _ref_repa is not None:
            cfg["repa"] = OmegaConf.to_container(_ref_repa, resolve=True)
            _applied["repa"] = cfg["repa"]
        if rank == 0:
            print(f"[loss-align] merged loss config from {loss_align_config}: {_applied}",
                  flush=True)

    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    gan_batch_size = int(train_cfg.get("gan_batch_size", batch_size))
    gan_grad_accum = int(train_cfg.get("gan_grad_accum", grad_accum))
    num_workers = int(train_cfg.get("num_workers", 8))
    num_epochs = int(train_cfg.get("epochs", 100))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    log_interval = int(train_cfg.get("log_interval", 50))
    ckpt_interval = int(train_cfg.get("checkpoint_interval", 500))
    val_interval = int(train_cfg.get("val_interval", 500))
    val_max_batches = int(train_cfg.get("val_max_batches", 20))
    clip_grad = float(train_cfg.get("clip_grad", 0)) or None
    seed = int(train_cfg.get("global_seed", 0)) * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cotrain_node = cfg.get("cotrain_t2i")
    cotrain_cfg = (
        OmegaConf.to_container(cotrain_node, resolve=True)
        if cotrain_node is not None else {}
    ) or {}
    cotrain_enable = bool(cotrain_cfg.get("enable", False))
    env_cotrain = os.environ.get("COTRAIN_T2I", "")
    if env_cotrain:
        cotrain_enable = env_cotrain not in ("0", "false", "False", "no")
    t2i_every_k = int(os.environ.get("T2I_EVERY_K", cotrain_cfg.get("every_k", 3)))
    if cotrain_enable:
        if grad_accum != 1:
            raise ValueError(
                "cotrain_t2i requires training.grad_accum=1 because the batch "
                f"domain is selected per optimizer step; got grad_accum={grad_accum}."
            )
        if gan_grad_accum != 1:
            raise ValueError(
                "cotrain_t2i requires training.gan_grad_accum=1; otherwise the GAN "
                "phase would change gradient accumulation and mix t2i/geometry "
                f"batches within an accumulation window; got gan_grad_accum={gan_grad_accum}."
            )
        if t2i_every_k < 2:
            raise ValueError(f"cotrain_t2i.every_k must be >=2; got {t2i_every_k}.")
    preserve_geometry_steps = bool(cotrain_cfg.get("preserve_geometry_steps", False))

    rgb_weight = float(train_cfg.get("rgb_weight", 0))
    lpips_weight = float(train_cfg.get("lpips_weight", 0))
    lpips_start_step = int(train_cfg.get("lpips_start_step", 0))
    rgb_jitter_mode = str(train_cfg.get("rgb_latent_jitter_mode", "none")).lower()
    rgb_jitter_sigma = float(train_cfg.get("rgb_latent_jitter_sigma", 0.0))
    rgb_jitter_prob = float(train_cfg.get("rgb_latent_jitter_prob", 0.0))
    rgb_jitter_relative = bool(train_cfg.get("rgb_latent_jitter_relative", True))
    rgb_jitter_start_step = int(train_cfg.get("rgb_latent_jitter_start_step", 0))
    rgb_jitter_warmup_steps = int(train_cfg.get("rgb_latent_jitter_warmup_steps", 0))
    rgb_fdiff_weight = float(train_cfg.get("rgb_fdiff_weight", 0.0))
    rgb_fdiff_start_step = int(train_cfg.get("rgb_fdiff_start_step", 0))
    rgb_fdiff_eps = float(train_cfg.get("rgb_fdiff_eps", 1e-3))
    rgb_hfdiff_weight = float(train_cfg.get("rgb_hfdiff_weight", 0.0))
    rgb_hfdiff_start_step = int(train_cfg.get("rgb_hfdiff_start_step", 0))
    rgb_hfdiff_eps = float(train_cfg.get("rgb_hfdiff_eps", rgb_fdiff_eps))
    rgb_temporal_haar3d_weight = float(
        os.environ.get(
            "TEMPORAL_HAAR3D_WEIGHT",
            train_cfg.get("rgb_temporal_haar3d_weight", 0.0),
        )
    )
    rgb_temporal_haar3d_start_step = int(
        os.environ.get(
            "TEMPORAL_HAAR3D_START_STEP",
            train_cfg.get("rgb_temporal_haar3d_start_step", 0),
        )
    )

    # ── Geometry (ray + depth) supervision config ──
    pose_weight = float(train_cfg.get("pose_weight", 0))
    geo_depth_weight = float(train_cfg.get("geo_depth_weight", 0))
    pose_start_step = int(train_cfg.get("pose_start_step", 0))
    pose_max_views = int(train_cfg.get("pose_max_views", 8))
    # GS distillation (frozen gs_head raw gaussians, GT feats → recon feats)
    gs_distill_weight = float(train_cfg.get("gs_distill_weight", 0))
    gs_distill_max_views = int(train_cfg.get("gs_distill_max_views", 8))
    gs_distill_max_scenes = int(train_cfg.get("gs_distill_max_scenes", 1))
    gs_distill_max_num_views = int(train_cfg.get("gs_distill_max_num_views", 28))
    gs_distill_start_step = int(train_cfg.get("gs_distill_start_step", 0))
    gs_distill_conf_weight = float(train_cfg.get("gs_distill_conf_weight", 1.0))
    need_dpt = pose_weight > 0 or geo_depth_weight > 0 or gs_distill_weight > 0

    if need_dpt:
        if not rae_cfg.params.get("dpt_decoder_path"):
            rae_cfg.params.dpt_decoder_path = train_cfg.get(
                "dpt_decoder_path", "pretrained_models/da3/dpt_decoder.pt")
        if not rae_cfg.params.get("da3_weights_path"):
            rae_cfg.params.da3_weights_path = train_cfg.get(
                "da3_weights_path", "pretrained_models/da3/model.safetensors")

    # ── GAN config ──
    disc_cfg = gan_cfg.get("disc", {})
    loss_cfg = gan_cfg.get("loss", {})
    use_gan_training = bool(gan_cfg) and bool(disc_cfg)
    disc_weight = float(loss_cfg.get("disc_weight", 0)) if use_gan_training else 0
    disc_start_step = int(loss_cfg.get("disc_start_step", 0))
    disc_upd_start_step = int(loss_cfg.get("disc_upd_start_step", disc_start_step))
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    # WF-VAE alternates G/D so the discriminator only steps on half the
    # iterations. disc_update_every=2 reproduces that cadence; 1 keeps the
    # legacy behaviour of stepping D on every generator step.
    disc_update_every = max(1, int(loss_cfg.get("disc_update_every", 1)))
    # R1 gradient penalty on real samples (StyleGAN2). Applied lazily every
    # r1_every discriminator updates with the coefficient scaled by the same
    # factor, which keeps the regularisation strength while paying for the
    # second-order graph only once per interval.
    r1_gamma = float(loss_cfg.get("r1_gamma", 0.0))
    r1_every = max(1, int(loss_cfg.get("r1_every", 16)))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")

    # ── Self-REPA config ──
    repa_cfg = OmegaConf.to_container(cfg.get("repa", {}), resolve=True) or {}
    repa_enabled = bool(repa_cfg.get("enabled", False))
    repa_weight = float(repa_cfg.get("weight", 0.5))
    repa_start_step = int(repa_cfg.get("start_step", 0))
    repa_struct_weight = float(repa_cfg.get("struct_weight", 0.0)) if repa_enabled else 0.0
    repa_struct_mu_weight = float(repa_cfg.get("struct_mu_weight", 0.0)) if repa_enabled else 0.0
    repa_struct_max_tokens = int(repa_cfg.get("struct_max_tokens", 0) or 0)
    # 双老师（方案 1b）：token cosine 用 repa.target（低 VIV，如 cradio/da3），
    # relational struct KD 用 repa.struct_target（spatial 强，如 dinov2）。未设则同源。
    repa_struct_target_name = repa_cfg.get("struct_target", None)
    repa_proj_dim = int(repa_cfg.get("proj_dim", 0)) if repa_enabled else 0
    if repa_enabled:
        if repa_proj_dim <= 0:
            raise ValueError(
                "repa.enabled=true 时必须设 repa.proj_dim "
                "（da3→2048, cradio-b→768, dinov2-large→1024）"
            )
        # 投影头建在 VAE 内，dim 由 repa 块驱动（单一真相源）。
        vae_cfg["repa_proj_dim"] = repa_proj_dim
        vae_cfg["repa_proj_hidden"] = int(repa_cfg.get("proj_hidden", 1024))

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        idx = len(glob(f"{args.results_dir}/*"))
        exp_name = f"{idx:03d}-gae-d{vae_cfg.get('latent_dim', 64)}-attn{vae_cfg.get('num_attn_blocks', 4)}-{args.precision}"
        exp_dir = os.path.join(args.results_dir, exp_name)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment: {exp_dir}")
        if args.wandb and HAS_WANDB:
            wandb_utils.initialize(args, os.environ.get("ENTITY", "gae"), exp_name,
                                   os.environ.get("PROJECT", "GAECodec"))
    else:
        exp_dir = ckpt_dir = None
        logger = create_logger(None)

    # Resolve DA3 encoder to local pretrained_models/ when available so multi-GPU
    # startup does not hammer HF Hub (5+ GB concurrent downloads often fail).
    _s1_params = dict(rae_cfg.get("params") or {})
    _enc_path = _s1_params.get("encoder_pretrained_path")
    _da3_weights = _s1_params.get("da3_weights_path") or train_cfg.get("da3_weights_path")
    _enc_local = _resolve_da3_encoder_path(_enc_path, _da3_weights)
    if _enc_local != _enc_path:
        logger.info(f"[da3] encoder_pretrained_path: {_enc_path!r} -> {_enc_local}")
        _s1_params["encoder_pretrained_path"] = _enc_local
        rae_cfg["params"] = _s1_params
    else:
        _ensure_da3_hub_cached(_enc_path, logger)

    # ── DA3 encoder (frozen) ──
    rae = instantiate_from_config(rae_cfg).to(device).eval()
    rae.requires_grad_(False)
    logger.info("DA3 encoder loaded and frozen.")
    if need_dpt:
        if rae.rae_cl_decoder is not None:
            logger.info(f"Geometry supervision: ray_w={pose_weight}, depth_w={geo_depth_weight}, "
                        f"start={pose_start_step}, max_views={pose_max_views}")
        else:
            logger.warning("Geometry weights > 0 but DPT decoder not loaded; disabling.")
            pose_weight = geo_depth_weight = 0

    if gs_distill_weight > 0:
        if getattr(rae, "gs_head", None) is not None:
            logger.info(f"GS distillation: w={gs_distill_weight}, conf_w={gs_distill_conf_weight}, "
                        f"start={gs_distill_start_step}, max_views={gs_distill_max_views}, "
                        f"max_scenes={gs_distill_max_scenes} "
                        f"(1=one random scene/step, 0=all scenes), "
                        f"train_only_if_num_views<={gs_distill_max_num_views} "
                        f"(0=always)")
        else:
            logger.warning("gs_distill_weight > 0 but gs_head not loaded "
                           "(need da3-giant weights); disabling GS distillation.")
            gs_distill_weight = 0

    # ── Feature VAE v2 ──
    vae = GAECodec(**vae_cfg).to(device)
    has_rgb = vae.rgb_head is not None
    train_rgb_decoder_only = bool(train_cfg.get("train_rgb_decoder_only", False))
    rgb_only_loss = bool(train_cfg.get("rgb_only_loss", False))
    if rgb_only_loss and not has_rgb:
        raise ValueError("training.rgb_only_loss=true requires codec.rgb_decoder")
    if train_rgb_decoder_only:
        if not has_rgb:
            raise ValueError(
                "training.train_rgb_decoder_only=true requires codec.rgb_decoder")
        incompatible = []
        if pose_weight > 0:
            incompatible.append("pose_weight")
        if geo_depth_weight > 0:
            incompatible.append("geo_depth_weight")
        if gs_distill_weight > 0:
            incompatible.append("gs_distill_weight")
        if repa_enabled:
            incompatible.append("repa.enabled")
        if incompatible:
            raise ValueError(
                "training.train_rgb_decoder_only=true freezes/skips the feature "
                "and latent objectives; disable: " + ", ".join(incompatible))
        vae.requires_grad_(False)
        vae.rgb_head.requires_grad_(True)
        logger.info(
            "Decoder-only training: FeatureVAE encoder/feature decoder/latent "
            "projections are frozen; only rgb_head is trainable.")
    elif rgb_only_loss:
        # Train only the path that can affect rgb_pred: feature encoder,
        # latent projections, shared RGB decode trunk, and RGB head.  The
        # feature reconstruction decoder and geometry-only adapter are frozen.
        vae.requires_grad_(False)
        rgb_path_modules = (
            "enc_conv", "enc_attn", "enc_attn_norm", "enc_downsample",
            "mu_proj", "logvar_proj", "dec_proj", "dec_attn",
            "dec_attn_norm", "rgb_head",
        )
        for module_name in rgb_path_modules:
            module = getattr(vae, module_name, None)
            if module is not None:
                module.requires_grad_(True)
        logger.info(
            "RGB-only path training: encoder/latent/shared RGB trunk/RGB head "
            "are trainable; feature decoder and geometry adapter are frozen.")
    if has_rgb:
        logger.info(f"RGB head enabled, rgb_weight={rgb_weight}")
        if rgb_jitter_mode not in ("none", "off", "false") and rgb_jitter_sigma > 0:
            logger.info(
                f"RGB latent jitter: mode={rgb_jitter_mode}, sigma={rgb_jitter_sigma}, "
                f"prob={rgb_jitter_prob}, relative={rgb_jitter_relative}, "
                f"start={rgb_jitter_start_step}, warmup={rgb_jitter_warmup_steps}"
            )
        if rgb_fdiff_weight > 0:
            logger.info(
                f"RGB frame-diff loss: weight={rgb_fdiff_weight}, "
                f"start={rgb_fdiff_start_step}, eps={rgb_fdiff_eps}"
            )
        if rgb_hfdiff_weight > 0:
            logger.info(
                f"RGB high-frequency frame-diff loss: weight={rgb_hfdiff_weight}, "
                f"start={rgb_hfdiff_start_step}, eps={rgb_hfdiff_eps}"
            )
    else:
        logger.info("No RGB head — feature-only training.")
    ema_vae = deepcopy(vae).eval().requires_grad_(False)
    if dist.is_initialized():
        ddp_vae = DDP(
            vae,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=rgb_only_loss,
        )
    else:
        ddp_vae = vae

    n_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"GAE codec params: {n_params / 1e6:.2f}M  latent={vae.latent_dim}  "
                f"attn_blocks={vae.num_attn_blocks}  kl_weight={vae.kl_weight}  "
                f"free_bits={vae.free_bits}")

    # ── Self-REPA target（冻结 encoder / DA3 自身）──
    repa_target = None
    repa_struct_target = None
    if repa_enabled:
        from stage1.repa_target import build_repa_target
        repa_target = build_repa_target(repa_cfg, device)
        # 双老师：仅当 struct_target 与 token target 不同、且确有 struct loss 时才另建
        _use_struct = (repa_struct_weight > 0 or repa_struct_mu_weight > 0)
        _token_name = str(repa_cfg.get("target", "da3")).lower()
        if _use_struct and repa_struct_target_name and \
                str(repa_struct_target_name).lower() != _token_name:
            repa_struct_target = build_repa_target(
                repa_cfg, device, target_name=str(repa_struct_target_name))
        else:
            repa_struct_target = repa_target  # 同源（向后兼容单老师）
        logger.info(f"Self-REPA enabled: target={repa_cfg.get('target', 'da3')}, "
                    f"struct_target={str(repa_struct_target_name).lower() if repa_struct_target is not repa_target else 'same'}, "
                    f"weight={repa_weight}, proj_dim={repa_proj_dim}, "
                    f"proj=MLP, struct_weight={repa_struct_weight}, "
                    f"struct_mu_weight={repa_struct_mu_weight}, "
                    f"start_step={repa_start_step}")

    optimizer, _ = build_optimizer([p for p in vae.parameters() if p.requires_grad], train_cfg)
    ac_kwargs = (dict(device_type="cuda", enabled=True, dtype=torch.bfloat16)
                 if args.precision == "bf16" else dict(device_type="cuda", enabled=False))

    # ── LPIPS (frozen) ──
    lpips_fn = None
    if has_rgb and lpips_weight > 0:
        lpips_fn = LPIPS().to(device).eval()
        lpips_fn.requires_grad_(False)
        logger.info(f"LPIPS enabled, weight={lpips_weight}, start_step={lpips_start_step}")
    if has_rgb and rgb_temporal_haar3d_weight > 0:
        logger.info(
            "Temporal Haar-3D RGB loss enabled: "
            f"weight={rgb_temporal_haar3d_weight}, "
            f"start_step={rgb_temporal_haar3d_start_step}, "
            "bands=HLL/HLH/HHL/HHH"
        )

    # ── Discriminator ──
    discriminator = ddp_disc = disc_optimizer = disc_scheduler = disc_aug = None
    disc_is_3d = False
    disc_crop_size = 224
    disc_loss_fn = gen_loss_fn = None
    if use_gan_training and has_rgb and disc_weight > 0:
        if rgb_weight <= 0:
            raise ValueError(
                "GAN training needs rgb.weight > 0: the adaptive GAN weight "
                "differentiates the reconstruction loss w.r.t. the RGB head "
                "output layer, and rgb_l1 is the only term guaranteed to reach "
                "it on every step."
            )
        discriminator, disc_aug = build_discriminator(disc_cfg, device)
        disc_is_3d = bool(getattr(discriminator, "expects_video", False))
        disc_crop_size = int(disc_cfg.get("arch", {}).get("crop_size", 224))
        ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False) if dist.is_initialized() else discriminator
        disc_params = [p for p in discriminator.parameters() if p.requires_grad]
        disc_optimizer, _ = build_optimizer(disc_params, disc_cfg)
        ddp_disc.train()
        disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)
        if disc_cfg.get("scheduler"):
            pass  # disc_scheduler can be added if needed
        n_disc = sum(p.numel() for p in disc_params)
        disc_domain = "video-3d" if disc_is_3d else "image-2d"
        disc_lr = disc_optimizer.param_groups[0]["lr"]
        logger.info(f"Discriminator enabled: {n_disc / 1e6:.2f}M params, domain={disc_domain}, "
                    f"crop={disc_crop_size}, weight={disc_weight}, start_step={disc_start_step}, "
                    f"loss={disc_loss_type}/{gen_loss_type}, lr={disc_lr:g}, "
                    f"update_every={disc_update_every}, adaptive_weight=on "
                    f"(max={max_d_weight:g}), "
                    + (f"r1_gamma={r1_gamma:g} every={r1_every}"
                       if r1_gamma > 0 else "r1=off"))
    else:
        logger.info("GAN training disabled.")

    # ── Dynamic num_views ──
    min_views = int(train_cfg.get("min_views", 0))
    max_views_cfg = int(train_cfg.get("max_views", 0))
    dynamic_views = min_views > 0 and max_views_cfg > min_views
    # base_bv = tokens budget = batch_size(config) * max_views.
    # When V < max_views, batch_size increases proportionally (B*V ≈ base_bv).
    base_bv = batch_size * max_views_cfg if dynamic_views else 0
    # 自适应 bs 上限：默认 = base_bv（保证 B*V 永不超预算 → 显存不超 V=max 峰值）。
    max_batch_size = int(train_cfg.get("max_batch_size", base_bv if base_bv else batch_size))
    _raw_view_batch_caps = train_cfg.get("view_batch_caps") or {}
    view_batch_caps = {int(k): int(v) for k, v in _raw_view_batch_caps.items()}
    # V 切换频率（optimizer steps）。0 = 旧行为（每 epoch/loader 切一次）。
    view_switch_interval = int(train_cfg.get("view_switch_interval", 0))

    cur_batch_size = batch_size
    cur_grad_accum = grad_accum
    _gan_phase_active = False
    _cur_num_views = max_views_cfg if dynamic_views else 0
    _cur_resolution = None

    # Instantiate train dataset once (keep reference for num_views mutation)
    train_dataset_cfg = cfg.train_dataset
    if isinstance(train_dataset_cfg, str):
        import cut3r_data as _cd
        _train_dataset_obj = eval(train_dataset_cfg, vars(_cd))
    else:
        _train_dataset_obj = train_dataset_cfg

    # ── Optional token-budget batch sizing across resolutions ──
    # base_bv keeps B*V ≈ const; with mixed resolutions we also want B*V*N ≈ const
    # (N = patch tokens), so a small-resolution segment gets a proportionally
    # larger batch. Resolution is pinned per segment (needs dynamic views so the
    # loader is rebuilt each segment). Budget is referenced to the LARGEST
    # resolution so no segment exceeds the baseline per-sample memory.
    try:
        _res_list = [tuple(r) for r in _train_dataset_obj._resolutions]
    except Exception:
        _res_list = []
    _PATCH_TOK = 14
    def _res_tokens(wh):
        w, h = wh
        return max(1, (w // _PATCH_TOK) * (h // _PATCH_TOK))
    adaptive_tokens = (bool(train_cfg.get("adaptive_batch_by_tokens", False))
                       and dynamic_views and len(_res_list) > 1)
    _token_ref = max((_res_tokens(r) for r in _res_list), default=0)
    _token_budget = batch_size * max_views_cfg * _token_ref
    if bool(train_cfg.get("adaptive_batch_by_tokens", False)) and not adaptive_tokens and rank == 0:
        logger.info("[adaptive-batch] adaptive_batch_by_tokens ignored "
                    "(requires dynamic views + >1 resolution).")
    elif adaptive_tokens and rank == 0:
        logger.info(f"[adaptive-batch] token-budget batching on: "
                    f"budget={_token_budget} tokens (ref={_token_ref}), "
                    f"resolutions={_res_list}")

    train_loader = prepare_dataloader(_train_dataset_obj, cur_batch_size, num_workers, rank, world_size)
    val_loader = prepare_dataloader(cfg.test_dataset, cur_batch_size, min(num_workers, 2), rank, world_size, test=True)
    steps_per_epoch = len(train_loader)
    logger.info(f"Train: {steps_per_epoch} steps/epoch, bs={cur_batch_size}/GPU × accum={cur_grad_accum}, "
                f"effective_bs={cur_batch_size * cur_grad_accum}, world={world_size}")
    if dynamic_views:
        logger.info(f"Dynamic views: V ~ [{min_views}, {max_views_cfg}], B*V budget={base_bv}, "
                    f"auto-adjust batch_size per epoch")
    if gan_batch_size != batch_size or gan_grad_accum != grad_accum:
        logger.info(f"GAN phase will switch to bs={gan_batch_size}/GPU × accum={gan_grad_accum}, "
                    f"effective_bs={gan_batch_size * gan_grad_accum} at step {disc_start_step}")

    t2i_iter = None
    if cotrain_enable:
        try:
            from data.blip3o_wds import BLIP3O_NUM_SAMPLES, build_blip3o_wds_loader
            from data.imagenet_arrow import MixedT2ILoader, build_imagenet_arrow_loader
        except ImportError as exc:
            raise RuntimeError(
                "cotrain_t2i requires the data extras: pip install -e '.[data]'"
            ) from exc
        t2i_ds_cfg = cotrain_cfg.get("dataset", {}) or {}
        t2i_max_bs = int(cotrain_cfg.get("max_batch_size", max_batch_size))
        t2i_default_bs = max(1, min(base_bv if dynamic_views else batch_size, t2i_max_bs))
        t2i_bs = int(os.environ.get("T2I_BATCH_SIZE", cotrain_cfg.get("batch_size", t2i_default_bs)))
        t2i_splits_cfg = t2i_ds_cfg.get("splits")
        t2i_splits = [t2i_splits_cfg] if isinstance(t2i_splits_cfg, str) else list(t2i_splits_cfg)
        split_weights = t2i_ds_cfg.get("split_weights") or {}
        imagenet_arrow_root = t2i_ds_cfg.get("imagenet_arrow_root")
        t2i_resolutions = parse_resolution_list(
            t2i_ds_cfg.get("resolution"),
            image_size=t2i_ds_cfg.get("image_size"),
        )
        virtual_epoch_steps = int(t2i_ds_cfg.get("virtual_epoch_steps", 10000))
        # Each enabled t2i source spins up its own DataLoader workers and they
        # stay alive for the whole run (t2i_iter is persistent). With both
        # BLIP3o-WDS and ImageNet-Arrow enabled that is 2x workers on top of the
        # geometry loader; expose a knob to cap it. Default = num_workers.
        t2i_num_workers = int(t2i_ds_cfg.get("num_workers", num_workers))
        t2i_shuffle_buffer = int(t2i_ds_cfg.get("shuffle_buffer", 2000))

        t2i_loaders = []
        t2i_loader_weights = []
        wds_splits = [
            sp for sp in t2i_splits
            if not (sp == "imagenet" and imagenet_arrow_root)
        ]
        if wds_splits:
            wds_split_weights = {sp: split_weights.get(sp, 1) for sp in wds_splits}
            t2i_loaders.append(build_blip3o_wds_loader(
                data_dir=t2i_ds_cfg.get("data_dir"),
                splits=wds_splits,
                resolutions=t2i_resolutions,
                batch_size=t2i_bs,
                num_workers=t2i_num_workers,
                world_size=world_size,
                rank=rank,
                virtual_epoch_steps=virtual_epoch_steps,
                shuffle_buffer=t2i_shuffle_buffer,
                seed=seed,
                dual_caption_prob=float(t2i_ds_cfg.get("dual_caption_prob", 0.5)),
                split_weights=wds_split_weights,
            ))
            wds_effective_samples = sum(
                BLIP3O_NUM_SAMPLES.get(sp, 1_000_000) * int(wds_split_weights.get(sp, 1))
                for sp in wds_splits
            )
            t2i_loader_weights.append(max(1, round(wds_effective_samples / 1_000_000)))
        if imagenet_arrow_root and "imagenet" in t2i_splits:
            imagenet_weight = int(split_weights.get("imagenet", 1))
            t2i_loaders.append(build_imagenet_arrow_loader(
                root=imagenet_arrow_root,
                split=str(t2i_ds_cfg.get("imagenet_arrow_split", "train")),
                resolutions=t2i_resolutions,
                batch_size=t2i_bs,
                num_workers=t2i_num_workers,
                world_size=world_size,
                rank=rank,
                virtual_epoch_steps=virtual_epoch_steps,
                seed=seed,
                shuffle_buffer=t2i_shuffle_buffer,
            ))
            imagenet_effective_samples = BLIP3O_NUM_SAMPLES["imagenet"] * imagenet_weight
            t2i_loader_weights.append(max(1, round(imagenet_effective_samples / 1_000_000)))
        if not t2i_loaders:
            raise RuntimeError("[cotrain_t2i] no enabled single-image datasets")
        t2i_loader = (
            t2i_loaders[0] if len(t2i_loaders) == 1
            else MixedT2ILoader(t2i_loaders, t2i_loader_weights, seed=seed)
        )

        def _t2i_batches():
            ep = 0
            while True:
                t2i_loader.set_epoch(ep)
                for batch in t2i_loader:
                    yield batch
                ep += 1

        t2i_iter = _t2i_batches()
        logger.info(
            "[cotrain_t2i] enabled: every_k=%d (~%.0f%% single-image), "
            "bs=%d, resolutions=%s, splits=%s, split_weights=%s",
            t2i_every_k,
            100.0 / t2i_every_k,
            t2i_bs,
            t2i_resolutions,
            t2i_ds_cfg.get("splits"),
            t2i_ds_cfg.get("split_weights"),
        )
    else:
        logger.info("[cotrain_t2i] disabled")

    scheduler = None
    if train_cfg.get("scheduler"):
        scheduler, _ = build_scheduler(optimizer, steps_per_epoch, train_cfg)

    start_epoch, global_step = 0, 0
    if args.ckpt:
        start_epoch, global_step = load_checkpoint(
            args.ckpt, ddp_vae, ema_vae, optimizer, scheduler,
            ddp_disc, disc_optimizer, disc_scheduler)
        logger.info(f"Resumed from {args.ckpt} (epoch={start_epoch}, step={global_step})")
    elif args.init_ckpt:
        load_weights_only(args.init_ckpt, ddp_vae, ema_vae)
        logger.info(f"Loaded weights from {args.init_ckpt}")

    last_layer = vae.rgb_head.pred.weight if has_rgb else None

    def _pick_views_for_epoch(epoch_idx):
        """Pick num_views for this epoch; broadcast across DDP ranks."""
        if not dynamic_views:
            return 0, False
        v_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if rank == 0:
            v_tensor[0] = random.randint(min_views, max_views_cfg)
        if dist.is_initialized():
            dist.broadcast(v_tensor, src=0)
        return int(v_tensor.item()), True

    def _pick_resolution_for_segment():
        """Pick a resolution index for this segment; broadcast across ranks so
        every rank pins the same (W,H) and derives the same batch size."""
        if not adaptive_tokens:
            return -1
        r_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if rank == 0:
            r_tensor[0] = random.randrange(len(_res_list))
        if dist.is_initialized():
            dist.broadcast(r_tensor, src=0)
        return int(r_tensor.item())

    def _set_num_views_recursive(ds, v):
        """Recursively set num_views on leaf BaseMultiViewDataset objects,
        traversing through CatDataset/MulDataset/ResizedDataset wrappers."""
        if hasattr(ds, 'datasets'):
            for sub in ds.datasets:
                _set_num_views_recursive(sub, v)
        elif hasattr(ds, 'dataset'):
            _set_num_views_recursive(ds.dataset, v)
        else:
            ds.num_views = v

    def _set_resolution_recursive(ds, res):
        """Pin a single (W,H) resolution on all leaf datasets (mirrors the
        num_views setter) so the whole segment renders at one resolution and the
        aspect-ratio sampler pool collapses to 1."""
        if hasattr(ds, 'datasets'):
            for sub in ds.datasets:
                _set_resolution_recursive(sub, res)
        elif hasattr(ds, 'dataset'):
            _set_resolution_recursive(ds.dataset, res)
        else:
            ds._resolutions = [tuple(res)]

    def _rebuild_loader_for_views(v_new, is_gan, res_idx=-1):
        """Set dataset.num_views (+ optional pinned resolution), compute
        batch_size, rebuild dataloader."""
        nonlocal _cur_num_views, cur_batch_size, train_loader, steps_per_epoch
        nonlocal _cur_resolution

        _set_num_views_recursive(_train_dataset_obj, v_new)

        # 自适应 batch。默认：bs = clamp(base_bv // V, 1, cap)，B*V ≈ 恒定。
        # 开 adaptive_batch_by_tokens 时改为 token 预算：bs = clamp(budget //
        # (V*N), 1, cap)，把该段固定到一个分辨率并让 B*V*N ≈ 恒定。
        # view_batch_caps 可对特定 V 再收紧上限；GAN 阶段用 gan_batch_size 作上限。
        cap = gan_batch_size if is_gan else max_batch_size
        if res_idx >= 0:
            res = _res_list[res_idx]
            _set_resolution_recursive(_train_dataset_obj, res)
            _cur_resolution = res
            new_bs = max(1, _token_budget // (v_new * _res_tokens(res)))
        else:
            _cur_resolution = None
            new_bs = max(1, base_bv // v_new)
        view_cap = view_batch_caps.get(v_new)
        if view_cap is not None:
            new_bs = min(new_bs, view_cap)
        new_bs = min(new_bs, cap)
        cur_batch_size = new_bs
        _cur_num_views = v_new

        train_loader = prepare_dataloader(_train_dataset_obj, cur_batch_size, num_workers, rank, world_size)
        steps_per_epoch = len(train_loader)

    # ── Training (step-driven; periodic intra-epoch view switching) ──
    # 旧版按 epoch 切 V（一个 epoch ≈ 5916 步才换一次）。现改为 step-driven：每个
    # "view 段" 重新随机抽 V + 自适应 bs 重建 loader，段长 = min(view_switch_interval,
    # loader 长度)。view_switch_interval=0 时退化为按 loader（≈每 epoch）切一次。
    steps_per_epoch_nominal = max(1, steps_per_epoch)  # 初始 bs 下的名义 epoch 长度
    total_steps = num_epochs * steps_per_epoch_nominal
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    if max_steps > 0:
        max_optimizer_steps = max_steps
        if cotrain_enable and preserve_geometry_steps:
            max_optimizer_steps = math.ceil(max_steps * t2i_every_k / (t2i_every_k - 1))
        total_steps = min(total_steps, max_optimizer_steps)
    if rank == 0:
        cap_note = ""
        if max_steps > 0:
            if cotrain_enable and preserve_geometry_steps:
                cap_note = (
                    f" (geometry max_steps={max_steps}, "
                    f"optimizer_steps≈{max_optimizer_steps})"
                )
            else:
                cap_note = f" (capped by max_steps={max_steps})"
        if dynamic_views and view_switch_interval > 0:
            caps_note = (f", view_batch_caps={view_batch_caps}" if view_batch_caps else "")
            logger.info(f"View switching every {view_switch_interval} optimizer steps; "
                        f"adaptive bs = clamp(base_bv({base_bv})//V, 1, {max_batch_size})"
                        f"{caps_note}; total_steps≈{total_steps}{cap_note}")
        else:
            logger.info(f"View switching per loader (legacy); total_steps≈{total_steps}{cap_note}")

    seg_idx = 0
    done = global_step >= total_steps
    while not done:
        epoch = global_step // steps_per_epoch_nominal  # 名义 epoch（仅用于日志/ckpt）
        if hasattr(ddp_vae, "train"):
            ddp_vae.train()

        # ── GAN phase transition ──
        if (not _gan_phase_active
                and global_step >= disc_start_step
                and (gan_batch_size != batch_size or gan_grad_accum != grad_accum)):
            cur_grad_accum = gan_grad_accum
            _gan_phase_active = True
            torch.cuda.empty_cache()
            logger.info(f"[GAN phase] activated: grad_accum={cur_grad_accum}")

        # ── 本段：抽 V(+分辨率)，自适应 bs，重建 loader ──
        if dynamic_views:
            v_new, _ = _pick_views_for_epoch(seg_idx)
            r_idx = _pick_resolution_for_segment()
            _rebuild_loader_for_views(v_new, _gan_phase_active, r_idx)
            if rank == 0:
                _res_note = (f", res={_cur_resolution[0]}x{_cur_resolution[1]}"
                             f"(N={_res_tokens(_cur_resolution)})"
                             if _cur_resolution is not None else "")
                logger.info(f"[Dynamic views] seg={seg_idx} step={global_step} "
                            f"(epoch~{epoch}): V={v_new}, bs={cur_batch_size}, "
                            f"B*V={cur_batch_size * v_new}{_res_note}, accum={cur_grad_accum}, "
                            f"loader_len={steps_per_epoch}")
        else:
            if _gan_phase_active and cur_batch_size != gan_batch_size:
                cur_batch_size = gan_batch_size
            train_loader = prepare_dataloader(
                _train_dataset_obj, cur_batch_size, num_workers, rank, world_size)
            steps_per_epoch = len(train_loader)

        train_loader.dataset.set_epoch(seg_idx)
        train_loader.batch_sampler.set_epoch(seg_idx)
        seg_idx += 1

        seg_len = steps_per_epoch
        geom_iter = iter(train_loader)
        pbar = tqdm(range(seg_len), total=seg_len,
                     desc=f"seg{seg_idx} V={_cur_num_views} S{global_step}",
                     disable=(rank != 0))
        accum_losses = defaultdict(float)
        # Carries the most recent discriminator measurement, so a log step that
        # lands on an iteration without a discriminator update (disc_update_every
        # > 1) or without an R1 measurement still reports both.
        disc_metrics = {}

        for step in pbar:
            use_t2i_step = (
                cotrain_enable
                and t2i_iter is not None
                and ((global_step + 1) % t2i_every_k == 0)
            )
            if use_t2i_step:
                t2i_images, _captions = next(t2i_iter)
                images = t2i_images.unsqueeze(1).to(device, non_blocking=True)
            else:
                try:
                    image_dict = next(geom_iter)
                except StopIteration:
                    break
                images = torch.cat([d["img"].unsqueeze(1) for d in image_dict], dim=1).to(device, non_blocking=True)
            img_size = (images.shape[-2], images.shape[-1])
            micro_step = step % cur_grad_accum
            is_accum_boundary = (micro_step == cur_grad_accum - 1) or (step == seg_len - 1)

            use_lpips = (lpips_fn is not None and lpips_weight > 0
                         and global_step >= lpips_start_step)
            use_gan = (ddp_disc is not None and disc_weight > 0
                       and global_step >= disc_start_step)
            train_disc = (ddp_disc is not None and disc_weight > 0
                          and global_step >= disc_upd_start_step
                          and global_step % disc_update_every == 0)
            apply_r1 = (r1_gamma > 0
                        and (global_step // disc_update_every) % r1_every == 0)
            use_geo = (not use_t2i_step
                       and (pose_weight > 0 or geo_depth_weight > 0)
                       and global_step >= pose_start_step
                       and rae.rae_cl_decoder is not None)

            # ── Generator forward ──
            if micro_step == 0:
                optimizer.zero_grad(set_to_none=True)
            if ddp_disc is not None:
                ddp_disc.eval()
                ddp_disc.requires_grad_(False)

            gt_flat = images.reshape(-1, 3, *img_size)

            # ── RGB jitter decision (deterministic across ranks) ──
            # Decide BEFORE the forward and pass it INTO the model, so the RGB
            # head always runs inside ddp_vae.forward(). Seeded by global_step
            # → every rank toggles identically.
            jitter_ready = (
                has_rgb
                and global_step >= rgb_jitter_start_step
                and rgb_jitter_sigma > 0
                and rgb_jitter_prob > 0
                and rgb_jitter_mode not in ("none", "off", "false")
            )
            do_jitter = jitter_ready and latent_jitter_decision(
                global_step, rgb_jitter_prob, device)
            rgb_jitter_cfg = None
            if do_jitter:
                if rgb_jitter_warmup_steps > 0:
                    warm = min(
                        1.0,
                        (global_step - rgb_jitter_start_step + 1)
                        / float(rgb_jitter_warmup_steps),
                    )
                else:
                    warm = 1.0
                rgb_jitter_cfg = {
                    "sigma": rgb_jitter_sigma * warm,
                    "mode": rgb_jitter_mode,
                    "relative": rgb_jitter_relative,
                }

            with autocast(**ac_kwargs):
                with torch.no_grad():
                    all_feats = rae.encode(images, mode="all")
                feats = {k: v[:, 1:, :] for k, v in all_feats.items()}
                vae_mod = ddp_vae.module if isinstance(ddp_vae, DDP) else ddp_vae
                x = vae_mod.normalize_levels(feats, image_size=img_size)
                out = ddp_vae(
                    x,
                    num_views=images.shape[1],
                    rgb_jitter=rgb_jitter_cfg,
                    decode_features=not train_rgb_decoder_only,
                )
                if train_rgb_decoder_only or rgb_only_loss:
                    # The frozen feature reconstruction term is constant with
                    # For decoder-only or RGB-only objectives, do not include
                    # disabled feature reconstruction terms in the loss.
                    losses = {}
                    recon_total = out["z"].new_zeros(())
                    if rgb_only_loss and not train_rgb_decoder_only and vae_mod.kl_weight > 0:
                        # Keep the latent information-budget regularizer while
                        # excluding the feature reconstruction objective.
                        kl_elem = -0.5 * (
                            1 + out["logvar"] - out["mu"].pow(2)
                            - out["logvar"].exp()
                        )
                        kl_per_ch = kl_elem.mean(dim=(0, 2, 3))
                        kl_loss = kl_per_ch.mean()
                        if vae_mod.free_bits > 0:
                            kl_used = torch.clamp(
                                kl_per_ch, min=vae_mod.free_bits).mean()
                        else:
                            kl_used = kl_loss
                        losses["kl_loss"] = kl_loss
                        losses["kl_loss_used"] = kl_used
                        recon_total = recon_total + vae_mod.kl_weight * kl_used
                else:
                    losses = vae_mod.compute_loss(x, out, per_level=True)
                    recon_total = losses["loss"]

                # RGB branch. Feature recon/REPA/geo always use clean z; jitter
                # (applied inside forward) only perturbs the RGB decode path to
                # make the decoder robust to generated-latent perturbations.
                rgb_pred_pm1 = None
                rgb_pred = out.get("rgb_pred")
                if rgb_pred is not None:
                    sigma_eff = out.get("rgb_jitter_sigma_eff", out["z"].new_zeros(()))
                    losses["rgb_jitter_applied"] = rgb_pred.new_tensor(
                        1.0 if do_jitter else 0.0)
                    losses["rgb_jitter_sigma_eff"] = sigma_eff.to(dtype=rgb_pred.dtype)

                    if rgb_weight > 0:
                        rgb_l1 = F.l1_loss(rgb_pred, gt_flat)
                        losses["rgb_l1"] = rgb_l1
                        recon_total = recon_total + rgb_weight * rgb_l1

                    if rgb_fdiff_weight > 0 and global_step >= rgb_fdiff_start_step:
                        fdiff_l = rgb_frame_diff_loss(
                            rgb_pred, gt_flat, images.shape[1], eps=rgb_fdiff_eps)
                        losses["rgb_fdiff"] = fdiff_l
                        recon_total = recon_total + rgb_fdiff_weight * fdiff_l

                    if rgb_hfdiff_weight > 0 and global_step >= rgb_hfdiff_start_step:
                        hfdiff_l = rgb_high_freq_frame_diff_loss(
                            rgb_pred, gt_flat, images.shape[1], eps=rgb_hfdiff_eps)
                        losses["rgb_hfdiff"] = hfdiff_l
                        recon_total = recon_total + rgb_hfdiff_weight * hfdiff_l

                    if (
                        rgb_temporal_haar3d_weight > 0
                        and global_step >= rgb_temporal_haar3d_start_step
                    ):
                        haar3d_l, haar3d_bands = rgb_temporal_haar3d_loss(
                            rgb_pred, gt_flat, images.shape[1]
                        )
                        losses["rgb_haar3d"] = haar3d_l
                        for band_name, band_loss in haar3d_bands.items():
                            losses[f"rgb_haar3d_{band_name}"] = band_loss
                        losses["rgb_haar3d_weighted"] = (
                            rgb_temporal_haar3d_weight * haar3d_l
                        )
                        recon_total = recon_total + losses["rgb_haar3d_weighted"]

                    # LPIPS
                    if use_lpips:
                        rgb_pred_pm1 = imagenet_to_pm1(rgb_pred)
                        gt_pm1 = imagenet_to_pm1(gt_flat)
                        lp = lpips_fn(rgb_pred_pm1.float(), gt_pm1.float())
                        losses["rgb_lpips"] = lp
                        recon_total = recon_total + lpips_weight * lp

                # GAN generator loss
                gan_loss = torch.zeros(1, device=device)
                gan_active = False
                if use_gan and "rgb_pred" in out:
                    if rgb_pred_pm1 is None:
                        rgb_pred_pm1 = imagenet_to_pm1(out["rgb_pred"])
                    fake_input = (
                        rgb_frames_to_video(rgb_pred_pm1, images.shape[1])
                        if disc_is_3d else rgb_pred_pm1
                    )
                    fake_crop, _ = spatial_crop_last2(fake_input, disc_crop_size)
                    fake_aug = disc_aug.aug(fake_crop)
                    # Discriminator weights are frozen in the generator phase.
                    # Bypass its DDP wrapper so the reducer does not expect
                    # discriminator gradients from this backward pass.
                    logits_fake, _ = discriminator(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                    gan_active = True
                    losses["gan_g"] = gan_loss

            # ── Geometry (ray + depth) supervision ──
            if use_geo:
                H, W = img_size
                backbone_norm = rae.encoder.backbone.pretrained.norm
                ray_l1, depth_l1 = _compute_geo_loss(
                    out["recon"], all_feats, vae_mod, backbone_norm,
                    rae.rae_cl_decoder, H, W, device, max_views=pose_max_views,
                    num_views=images.shape[1],
                    embed_dim=rae.encoder.backbone.pretrained.embed_dim)
                if pose_weight > 0:
                    losses["ray_l1"] = ray_l1
                    recon_total = recon_total + pose_weight * ray_l1
                if geo_depth_weight > 0:
                    losses["geo_depth_l1"] = depth_l1
                    recon_total = recon_total + geo_depth_weight * depth_l1

            # ── GS distillation (frozen gs_head raw gaussians) ──
            num_views = images.shape[1]
            gs_views_ok = (gs_distill_max_num_views <= 0
                           or num_views <= gs_distill_max_num_views)
            use_gs = (not use_t2i_step and gs_distill_weight > 0
                      and global_step >= gs_distill_start_step
                      and gs_views_ok
                      and getattr(rae, "gs_head", None) is not None)
            gs_l = None
            if use_gs:
                H, W = img_size
                backbone_norm = rae.encoder.backbone.pretrained.norm
                gs_l = _compute_gs_distill_loss(
                    out["recon"], all_feats, vae_mod, backbone_norm, rae.gs_head,
                    images, H, W, device,
                    max_views=gs_distill_max_views,
                    embed_dim=rae.encoder.backbone.pretrained.embed_dim,
                    num_views=images.shape[1], conf_weight=gs_distill_conf_weight,
                    max_scenes=gs_distill_max_scenes, global_step=global_step,
                )
                losses["gs_distill"] = gs_l.detach()
                recon_total = recon_total + gs_distill_weight * gs_l

            # ── Self-REPA 对齐 ──
            # 注意：DDP 未开 find_unused_parameters，所以 repa_proj 每步都必须被用到。
            # 因此即使 global_step < start_step 也照常 forward，仅把权重设 0（梯度为 0
            # 但 param 仍在计算图里 → DDP 不会报 unused param）。
            if repa_target is not None and vae_mod.repa_proj is not None:
                latent_hw = (out["mu"].shape[-2], out["mu"].shape[-1])
                gt_flat01 = (gt_flat * _IMG_STD.to(gt_flat.device, gt_flat.dtype)
                             + _IMG_MEAN.to(gt_flat.device, gt_flat.dtype)).clamp(0, 1)
                # teacher 用 float32、禁用 autocast，避免 bf16 污染 frozen target
                with autocast(**{**ac_kwargs, "enabled": False}):
                    repa_tgt = repa_target.features(gt_flat01, all_feats, latent_hw)
                    # struct 用独立老师（方案 1b）；同源时直接复用 token 特征，零额外 forward
                    if repa_struct_target is repa_target:
                        struct_tgt = repa_tgt
                    else:
                        struct_tgt = repa_struct_target.features(gt_flat01, all_feats, latent_hw)
                with autocast(**ac_kwargs):
                    z_proj = vae_mod.repa_project(out["mu"])
                    if z_proj.shape[-1] != repa_tgt.shape[-1]:
                        raise ValueError(
                            f"REPA dim 不匹配: repa.proj_dim={z_proj.shape[-1]} 但目标特征维="
                            f"{repa_tgt.shape[-1]}。da3→2048, cradio-b→768, dinov2-large→1024，"
                            f"请同步 repa.proj_dim。"
                        )
                    if z_proj.shape[1] != repa_tgt.shape[1]:
                        raise ValueError(
                            f"REPA token 数不匹配: z_proj N={z_proj.shape[1]} vs target N="
                            f"{repa_tgt.shape[1]}（latent {latent_hw}）。检查 teacher 网格插值。"
                        )
                    repa_l = repa_cosine_loss(z_proj, repa_tgt.to(z_proj.dtype))
                    repa_struct_l = None
                    if repa_struct_weight > 0:
                        repa_struct_l = repa_similarity_loss(
                            z_proj, struct_tgt.to(z_proj.dtype),
                            max_tokens=repa_struct_max_tokens,
                        )
                    # raw mu 空间 struct：直接约束 128d latent 的 patch 几何（与 eval/推理同空间）
                    repa_struct_mu_l = None
                    if repa_struct_mu_weight > 0:
                        b_, c_, h_, w_ = out["mu"].shape
                        mu_tok = out["mu"].permute(0, 2, 3, 1).reshape(b_, h_ * w_, c_)
                        repa_struct_mu_l = repa_similarity_loss(
                            mu_tok, struct_tgt.to(mu_tok.dtype),
                            max_tokens=repa_struct_max_tokens,
                        )
                w = repa_weight if global_step >= repa_start_step else 0.0
                losses["repa"] = repa_l
                recon_total = recon_total + w * repa_l
                if repa_struct_l is not None:
                    losses["repa_struct"] = repa_struct_l
                    recon_total = recon_total + w * repa_struct_weight * repa_struct_l
                if repa_struct_mu_l is not None:
                    losses["repa_struct_mu"] = repa_struct_mu_l
                    recon_total = recon_total + w * repa_struct_mu_weight * repa_struct_mu_l

            # Combine losses. The GAN term carries the VQGAN/WF-VAE adaptive
            # weight so its gradient at the RGB head output layer stays at
            # disc_weight × the reconstruction gradient. With a fixed weight a
            # saturated discriminator drives the unbounded generator loss
            # (-mean(logits_fake)) arbitrarily high and training diverges.
            if gan_active:
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, last_layer, max_d_weight)
                losses["gan_d_weight"] = adaptive_weight
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                total_loss = recon_total

            (total_loss / cur_grad_accum).backward()

            # Accumulate metrics for logging
            losses["total_loss"] = total_loss
            for k, v in losses.items():
                accum_losses[k] += (v.item() if isinstance(v, torch.Tensor) else v) / cur_grad_accum

            if not is_accum_boundary:
                continue

            # ── Optimizer step (at accumulation boundary) ──
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)
            optimizer.step()
            if scheduler:
                scheduler.step()
            update_ema(ema_vae, vae, ema_decay)

            # ── Discriminator step ──
            # disc_metrics is not cleared here: with disc_update_every > 1 the
            # log steps can fall on iterations that skip the discriminator, and
            # disc_acc/disc_loss are what reveal a discriminator running away.
            # Keeping the last measurement means the log always carries them.
            if train_disc and "rgb_pred" in out:
                ddp_vae.eval()
                ddp_disc.requires_grad_(True)
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**ac_kwargs):
                        with torch.no_grad():
                            det_out = vae_mod(
                                x,
                                num_views=images.shape[1],
                                decode_features=False,
                            )
                            fake_pm1 = imagenet_to_pm1(det_out["rgb_pred"])
                            fake_pm1 = fake_pm1.clamp(-1, 1)
                            fake_pm1 = torch.round((fake_pm1 + 1) * 127.5) / 127.5 - 1.0

                        gt_pm1_d = imagenet_to_pm1(gt_flat)
                        if disc_is_3d:
                            fake_input = rgb_frames_to_video(fake_pm1, images.shape[1])
                            real_input = rgb_frames_to_video(gt_pm1_d, images.shape[1])
                        else:
                            fake_input, real_input = fake_pm1, gt_pm1_d
                        fake_crop, crop_params = spatial_crop_last2(
                            fake_input, disc_crop_size
                        )
                        real_crop, _ = spatial_crop_last2(
                            real_input, disc_crop_size, params=crop_params
                        )
                        fake_input = disc_aug.aug(fake_crop)
                        real_input = disc_aug.aug(real_crop)
                        # detach() makes real_input a leaf so requires_grad_ is
                        # legal; the real branch never backpropagates upstream.
                        if apply_r1:
                            real_input = real_input.detach().requires_grad_(True)
                        logits_fake_d, logits_real_d = ddp_disc(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real_d, logits_fake_d)
                        accuracy = (logits_real_d > logits_fake_d).float().mean()
                        if apply_r1:
                            r1_grad = torch.autograd.grad(
                                logits_real_d.sum(), real_input, create_graph=True)[0]
                            # Mean over elements, not StyleGAN2's per-sample sum:
                            # this discriminator sees [B, 3, V, H, W] clips whose
                            # element count changes with V, so a sum would make
                            # r1_gamma depend on the view count and resolution.
                            r1 = r1_grad.float().pow(2).mean()
                            d_loss = d_loss + 0.5 * r1_gamma * r1_every * r1

                    d_loss.backward()
                    disc_optimizer.step()
                    # Updated in place rather than rebuilt: r1 is only measured
                    # every r1_every updates and a fresh dict would drop it on
                    # every other step, so it never reached the log.
                    disc_metrics.update({
                        "disc_loss": d_loss.detach().item(),
                        "logits_real": logits_real_d.detach().mean().item(),
                        "logits_fake": logits_fake_d.detach().mean().item(),
                        "disc_acc": accuracy.detach().item(),
                    })
                    if apply_r1:
                        disc_metrics["disc_r1"] = r1.detach().item()

                ddp_disc.eval()
                ddp_disc.requires_grad_(False)
                ddp_vae.train()

            # ── Logging ──
            completed_step = global_step + 1
            if log_interval > 0 and completed_step % log_interval == 0 and rank == 0:
                stats = {f"train/{k}": v for k, v in accum_losses.items()}
                stats["lr"] = optimizer.param_groups[0]["lr"]
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                stats["gpu_mem_gb"] = mem_gb
                for k, v in disc_metrics.items():
                    stats[f"train/{k}"] = v
                temporal_blocks = getattr(vae.rgb_head, "temporal_blocks", None)
                if temporal_blocks is not None:
                    gate_values = []
                    for block_index, block in temporal_blocks.items():
                        # from-scratch heads (gated=False) have no gate to log.
                        gate_attn_param = getattr(block, "gate_attn", None)
                        if gate_attn_param is None:
                            continue
                        gate_attn = gate_attn_param.detach().float().item()
                        stats[f"train/rgb_temporal_gate_attn_b{block_index}"] = gate_attn
                        gate_values.append(abs(gate_attn))
                        gate_ffn_param = getattr(block, "gate_ffn", None)
                        if getattr(block, "use_ffn", False) and gate_ffn_param is not None:
                            stats[f"train/rgb_temporal_gate_ffn_b{block_index}"] = (
                                gate_ffn_param.detach().float().item()
                            )
                    if gate_values:
                        stats["train/rgb_temporal_gate_attn_abs_mean"] = (
                            sum(gate_values) / len(gate_values)
                        )
                logger.info(
                    f"[E{epoch} S{completed_step}] "
                    + ", ".join(f"{k}: {v:.5f}" for k, v in stats.items())
                )
                if args.wandb and HAS_WANDB:
                    wandb_utils.log(stats, step=completed_step)

            accum_losses.clear()

            if ckpt_interval > 0 and completed_step % ckpt_interval == 0 and rank == 0:
                save_checkpoint(f"{ckpt_dir}/{completed_step:07d}.pt",
                                completed_step, epoch, ddp_vae, ema_vae, optimizer, scheduler,
                                ddp_disc, disc_optimizer, disc_scheduler)
                sync_log(logger)

            if val_interval > 0 and completed_step % val_interval == 0 and rank == 0:
                logger.info(f"Validation at step {completed_step}...")
                torch.cuda.empty_cache()
                online = ddp_vae.module if isinstance(ddp_vae, DDP) else ddp_vae
                validation_haar3d_weight = (
                    rgb_temporal_haar3d_weight
                    if completed_step >= rgb_temporal_haar3d_start_step
                    else 0.0
                )
                v_online = run_validation(rae, online, val_loader, device, ac_kwargs,
                                          rgb_weight=rgb_weight, lpips_fn=lpips_fn,
                                          lpips_weight=lpips_weight,
                                          rgb_temporal_haar3d_weight=(
                                              validation_haar3d_weight
                                          ),
                                          pose_weight=pose_weight,
                                          geo_depth_weight=geo_depth_weight,
                                          pose_max_views=pose_max_views,
                                          max_batches=val_max_batches,
                                          prefix="val", repa_target=repa_target,
                                          repa_struct_weight=repa_struct_weight,
                                          repa_struct_max_tokens=repa_struct_max_tokens,
                                          repa_struct_mu_weight=repa_struct_mu_weight,
                                          repa_struct_target=repa_struct_target,
                                          gs_distill_weight=gs_distill_weight,
                                          gs_distill_max_views=gs_distill_max_views,
                                          gs_distill_conf_weight=gs_distill_conf_weight,
                                          gs_distill_max_scenes=gs_distill_max_scenes)
                v_ema = run_validation(rae, ema_vae, val_loader, device, ac_kwargs,
                                       rgb_weight=rgb_weight, lpips_fn=lpips_fn,
                                       lpips_weight=lpips_weight,
                                       rgb_temporal_haar3d_weight=(
                                           validation_haar3d_weight
                                       ),
                                       pose_weight=pose_weight,
                                       geo_depth_weight=geo_depth_weight,
                                       pose_max_views=pose_max_views,
                                       max_batches=val_max_batches,
                                       prefix="val_ema", repa_target=repa_target,
                                       repa_struct_weight=repa_struct_weight,
                                       repa_struct_max_tokens=repa_struct_max_tokens,
                                       repa_struct_mu_weight=repa_struct_mu_weight,
                                       repa_struct_target=repa_struct_target,
                                       gs_distill_weight=gs_distill_weight,
                                       gs_distill_max_views=gs_distill_max_views,
                                       gs_distill_conf_weight=gs_distill_conf_weight,
                                       gs_distill_max_scenes=gs_distill_max_scenes)
                val_stats = {**v_online, **v_ema}
                logger.info("[Val] " + ", ".join(f"{k}: {v:.5f}" for k, v in val_stats.items()))
                if args.wandb and HAS_WANDB:
                    wandb_utils.log(val_stats, step=completed_step)
                ddp_vae.train()

            global_step = completed_step

            # ── 段结束条件：达到总步数 / 到达 view 切换周期 ──
            if global_step >= total_steps:
                done = True
                break
            if (dynamic_views and view_switch_interval > 0
                    and global_step % view_switch_interval == 0):
                break  # 结束本段 → 外层 while 重新抽 V

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
