#!/usr/bin/env python3
"""Geometry-decoder finetune — reproduces the "merged VAE" geometry alignment.

Trains ONLY the codec ``geo_adapter`` (+ optionally ``dec_conv``) so the direct
geometry decode of a frozen flow latent (``z_pred``) matches the FROZEN teacher
geometry ``G_old(z_regen)`` cached before finetuning. The DPT head, shared
decode trunk, RGB head, DA3 encoder and flow model all stay frozen; RGB is never
touched. The zero-init adapter means enabling it is an exact identity until this
script trains it.

Inputs come from the immutable geometry cache produced by
``scripts/eval/eval_generation.py`` run with the ``GEO_CACHE_DIR`` environment
variable set (per-scene ``z_pred`` / ``z_regen`` / ``z_clean`` + teacher
geometry). No flow model / no re-sampling happens here.

After training, fold the trained groups back into a full codec checkpoint with
``scripts/train/merge_geoft_vae_ckpt.py`` and point the flow eval at the merged ckpt.

Usage (single-GPU, adapter only)::

    python scripts/train/finetune_geo_decoder.py \
        --config configs/gae_128.yaml \
        --vae-ckpt ckpts/gae_128.pt \
        --cache-dir results/geo_cache \
        --out-dir results/geoft/run0 \
        --steps 2000 --val-every 100

Multi-GPU (manual grad all-reduce; adapter bypasses DDP hooks)::

    torchrun --nproc_per_node=8 scripts/train/finetune_geo_decoder.py --config ... \
        --vae-ckpt ... --cache-dir ... --out-dir ... --steps 2000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1.da3 import DA3Backbone  # noqa: E402
from stage1.gae_codec import GAECodec  # noqa: E402

from eval_generation import (  # noqa: E402
    _to_container_or_empty,
    _fast_load,
    _cache_da3_dir,
    _eval_cache_base,
    _inject_rgb_decoder_from_ckpt,
)
from eval_reconstruction import raw_no_cls_to_dpt_input  # noqa: E402

from utils import geo_align_loss as GAL  # noqa: E402


def _setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and os.environ.get("RANK") is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, local_rank


_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD = (0.229, 0.224, 0.225)


def _imagenet_to_pm1(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor(_IMNET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMNET_STD).view(1, 3, 1, 1)
    return ((x * std + mean) * 2.0 - 1.0).clamp(-1.0, 1.0)


def decode_dpt_grad(dpt_feats, dpt_decoder, H, W):
    """Grad-enabled DPT decode (eval ``decode_dpt`` is no_grad). DPT stays frozen."""
    with torch.autocast(device_type=dpt_feats[0][0].device.type, enabled=False):
        output = dpt_decoder(dpt_feats, H, W, patch_start_idx=0)
    depth = output.get("depth", None)
    if depth is not None:
        if depth.ndim == 5:
            depth = depth.reshape(-1, *depth.shape[2:])
        elif depth.ndim == 4 and depth.shape[0] == 1:
            depth = depth.squeeze(0).unsqueeze(1)
    return {"depth": depth, "ray": output.get("ray", None),
            "ray_conf": output.get("ray_conf", None)}


def _cache_file(path):
    _base = _eval_cache_base()
    if path and os.path.isfile(path) and not path.startswith(_base):
        cache_dir = os.path.join(_base, "_eval_cache")
        os.makedirs(cache_dir, exist_ok=True)
        parent = os.path.basename(os.path.dirname(path))
        local = os.path.join(cache_dir, f"{parent}_{os.path.basename(path)}")
        if not os.path.isfile(local):
            tmp = "%s.tmp.%d" % (local, os.getpid())
            shutil.copy2(path, tmp)
            os.replace(tmp, local)
        return local
    return path


def build_models(cfg, args, device):
    """Build frozen DA3Backbone (encoder + DPT) + GAECodec with a zero-init adapter."""
    vae_cfg = _to_container_or_empty(cfg.get("codec"))
    if not vae_cfg:
        raise ValueError("codec config not found")

    stage1_params = _to_container_or_empty(cfg.get("stage_1", {}).get("params"))
    # DPT decode resolution. 16:9 recipes store (W, H) on cotrain_t2i.dataset.image_size;
    # a square H=W=encoder_input_size would reshape DPT features (e.g. 48x48) against
    # non-square latents (e.g. 27x48 for 378x672) and corrupt the decoded geometry.
    img = None
    t2i = cfg.get("cotrain_t2i")
    if t2i is not None:
        ds = t2i.get("dataset") if hasattr(t2i, "get") else None
        img = ds.get("image_size") if ds is not None else None
    if img is not None:
        if OmegaConf.is_config(img):
            img = OmegaConf.to_container(img, resolve=True)
        if isinstance(img, (list, tuple)) and len(img) >= 2:
            W, H = int(img[0]), int(img[1])
        else:
            H = W = int(img)
    else:
        enc = stage1_params.get("encoder_input_size", 504)
        if isinstance(enc, (list, tuple)):
            enc = enc[0]
        H = W = int(enc)
    print(f"[rae] decode HxW={H}x{W}")

    # DA3 weights: CLI override > config; resolve ROOT-relative; local-cache.
    da3_wt = stage1_params.get("da3_weights_path") or args.da3_weights
    if da3_wt and not os.path.isabs(da3_wt):
        da3_wt = os.path.join(ROOT, da3_wt)
    da3_wt = _cache_file(da3_wt) if (da3_wt and os.path.isfile(da3_wt)) else None

    rae_kwargs = dict(stage1_params)
    rae_kwargs["encoder_input_size"] = [H, W]
    rae_kwargs["reshape_to_2d"] = False
    rae_kwargs["da3_weights_path"] = da3_wt
    dpt_dec = args.dpt_decoder if args.dpt_decoder not in (None, "none") else \
        stage1_params.get("dpt_decoder_path")
    if dpt_dec:
        rae_kwargs["dpt_decoder_path"] = _cache_file(dpt_dec)
    if stage1_params.get("encoder_pretrained_path"):
        rae_kwargs["encoder_pretrained_path"] = _cache_da3_dir(
            stage1_params["encoder_pretrained_path"])

    print(f"[rae] da3_weights={da3_wt or '<none>'}")
    rae = DA3Backbone(**rae_kwargs).to(device).eval()
    rae.requires_grad_(False)
    if rae.rae_cl_decoder is None:
        raise RuntimeError("DPT decoder not loaded (rae_cl_decoder is None); "
                           "geo-ft needs da3_weights_path/dpt_decoder_path.")
    backbone_norm = rae.encoder.backbone.pretrained.norm
    embed_dim = backbone_norm.normalized_shape[0]

    vae_ckpt_path = str(args.vae_ckpt or cfg.get("vae_checkpoint"))
    print(f"[vae] ckpt {vae_ckpt_path}")
    vae_ckpt = _fast_load(vae_ckpt_path, map_location="cpu")
    vae_sd = vae_ckpt.get("ema_codec", vae_ckpt.get(
        "ema_vae", vae_ckpt.get("codec", vae_ckpt.get("vae", vae_ckpt))))
    _inject_rgb_decoder_from_ckpt(vae_cfg, vae_sd)
    vae_cfg.pop("rgb_decoder_hidden", None)
    vae_cfg.pop("rgb_decoder_heads", None)
    vae_cfg["geo_adapter"] = dict(
        hidden_ratio=args.adapter_hidden_ratio,
        num_heads=args.adapter_heads,
        num_blocks=args.adapter_num_blocks,
        width_mult=args.adapter_width_mult,
        max_views=64,
        use_spatial=True,
    )
    vae = GAECodec(**vae_cfg).to(device)
    missing, unexpected = vae.load_state_dict(vae_sd, strict=False)
    non_adapter_missing = [k for k in missing if not k.startswith("geo_adapter")]
    if non_adapter_missing:
        print(f"[vae][WARN] {len(non_adapter_missing)} non-adapter missing keys: "
              f"{non_adapter_missing[:4]}")
    print(f"[vae] loaded (adapter fresh: "
          f"{sum(k.startswith('geo_adapter') for k in missing)} keys, "
          f"unexpected: {len(unexpected)})")
    return rae, vae, backbone_norm, embed_dim, H, W, vae_ckpt_path


_TRUNK_PREFIXES = ("dec_proj", "dec_attn")
_RGB_PREFIXES = ("rgb_head",)


def set_trainable(vae, freeze_dec_conv, train_trunk=False, train_rgb_head=False):
    """Freeze everything except geo_adapter (+ dec_conv/+trunk/+rgb_head)."""
    vae.requires_grad_(False)
    trainable = []
    for name, p in vae.named_parameters():
        train = name.startswith("geo_adapter")
        if not freeze_dec_conv and name.startswith("dec_conv"):
            train = True
        if train_trunk and name.startswith(_TRUNK_PREFIXES):
            train = True
        if train_rgb_head and name.startswith(_RGB_PREFIXES):
            train = True
        p.requires_grad_(train)
        if train:
            trainable.append(name)
    allowed = ("geo_adapter", "dec_conv") + _TRUNK_PREFIXES + _RGB_PREFIXES
    bad = [n for n in trainable if not n.startswith(allowed)]
    if bad:
        raise RuntimeError(f"unexpected trainable params: {bad[:5]}")
    if not any(n.startswith("geo_adapter") for n in trainable):
        raise RuntimeError("geo_adapter has no trainable params")
    print(f"[trainable] {len(trainable)} tensors; "
          f"adapter={sum(n.startswith('geo_adapter') for n in trainable)} "
          f"dec_conv={sum(n.startswith('dec_conv') for n in trainable)} "
          f"trunk={sum(n.startswith(_TRUNK_PREFIXES) for n in trainable)} "
          f"rgb_head={sum(n.startswith(_RGB_PREFIXES) for n in trainable)}")
    return trainable


def student_geo(vae, rae, backbone_norm, z, V, H, W, idx=None):
    """z (V,C,h,w) raw latent -> student geometry via adapter+dec_conv+DPT."""
    # cross-view temporal adapter runs at latent-res (cheap) -> run once on all views.
    z_geo = vae.apply_geo_adapter(z, V)
    if idx is not None:
        z_geo = z_geo[idx]

    def _decode_views(z_sub):
        seq, h, w = vae._decode_trunk(z_sub)
        c = seq.shape[-1]
        raw = vae.dec_conv(seq.permute(0, 2, 1).reshape(-1, c, h, w))
        feats = vae.denormalize_and_split(raw)
        dpt_in = raw_no_cls_to_dpt_input(feats, backbone_norm, z_sub.shape[0])
        return decode_dpt_grad(dpt_in, rae.rae_cl_decoder, H, W)

    # Optional view-chunked activation checkpointing (GEOFT_GRAD_CKPT=1): backward
    # recomputes ONE chunk of views at a time, bounding DPT peak memory for large V.
    # use_reentrant=False lets grad flow through trainable params even though the
    # cached latent has requires_grad=False.
    _use_ck = (os.environ.get("GEOFT_GRAD_CKPT", "0") == "1" and torch.is_grad_enabled())
    if not _use_ck:
        return _decode_views(z_geo)
    import torch.utils.checkpoint as _uc
    _n = z_geo.shape[0]
    _chunk = int(os.environ.get("GEOFT_DPT_VIEW_CHUNK", "0") or 0)
    if _chunk <= 0 or _chunk >= _n:
        return _uc.checkpoint(_decode_views, z_geo, use_reentrant=False)
    _d, _r, _cf = [], [], []
    for _st in range(0, _n, _chunk):
        _o = _uc.checkpoint(_decode_views, z_geo[_st:_st + _chunk], use_reentrant=False)
        _d.append(_o["depth"]); _r.append(_o["ray"]); _cf.append(_o["ray_conf"])
    def _cat(_lst, _dim):
        _lst = [x for x in _lst if x is not None]
        return torch.cat(_lst, _dim) if _lst else None
    # depth is (V,1,H,W) -> view dim 0; ray/ray_conf keep leading batch -> dim 1.
    return {"depth": _cat(_d, 0), "ray": _cat(_r, 1), "ray_conf": _cat(_cf, 1)}


def slice_geo(g, idx):
    out = {}
    for k, v in g.items():
        if v is None:
            out[k] = None
        elif k == "ray_conf" and v.ndim == 4 and v.shape[0] == 1:
            out[k] = v[:, idx]
        else:
            out[k] = v[idx]
    return out


def ray_to_vhwc(ray):
    if ray is None:
        return None
    if ray.ndim == 5:
        ray = ray[0]
    if ray.ndim == 4 and ray.shape[1] in (3, 6) and ray.shape[-1] != 6:
        ray = ray.permute(0, 2, 3, 1).contiguous()
    return ray


def _split_cache_files(files, *, split_holdout=None, val_scenes=0, val_seed=0):
    files = list(files)
    if split_holdout is not None:
        train, val = [], []
        for f in files:
            (val if os.path.basename(f).startswith(f"{split_holdout:03d}")
             else train).append(f)
        return train, val
    if val_scenes and val_scenes > 0 and len(files) > 1:
        n_val = min(int(val_scenes), len(files) - 1)
        order = list(files)
        rng = np.random.RandomState(int(val_seed))
        rng.shuffle(order)
        val = sorted(order[:n_val])
        val_set = set(val)
        train = [f for f in files if f not in val_set]
        return train, val
    return files, []


def _pick_subset(files, n, seed):
    if n is None or int(n) <= 0 or not files:
        return []
    n = min(int(n), len(files))
    order = list(files)
    rng = np.random.RandomState(int(seed))
    rng.shuffle(order)
    return sorted(order[:n])


class GeoCacheDataset:
    """Lazily loads per-scene npz caches; keeps tensors on CPU until sampled."""

    REQUIRED = ("z_pred", "z_regen", "z_clean", "teacher_depth", "teacher_ray")

    def __init__(self, cache_dir, domains=None, split_holdout=None,
                 want_val=False, val_scenes=0, val_seed=0, files=None, label=None):
        if files is not None:
            self.files = list(files)
            tag = label or "custom"
        else:
            all_files = []
            subdirs = domains or [d for d in os.listdir(cache_dir)
                                  if os.path.isdir(os.path.join(cache_dir, d))]
            for d in subdirs:
                domain_dir = os.path.join(cache_dir, d)
                found = sorted(
                    os.path.join(domain_dir, name)
                    for name in os.listdir(domain_dir)
                    if name.endswith(".npz"))
                eval_latents = [p for p in found if p.endswith("_latents.npz")]
                all_files += eval_latents or found
            if not all_files:
                raise FileNotFoundError(
                    f"no cache npz under {cache_dir} (domains={subdirs})")
            train, val = _split_cache_files(
                all_files, split_holdout=split_holdout,
                val_scenes=val_scenes, val_seed=val_seed)
            self.files = val if want_val else train
            if want_val and not self.files:
                self.files = train
            tag = label or ("val" if want_val else "train")
        print(f"[cache] {len(self.files)} scenes ({tag})")

    def __len__(self):
        return len(self.files)

    def load(self, idx, device):
        path = self.files[idx]
        d = np.load(path, allow_pickle=True)
        direct_eval = path.endswith("_latents.npz")
        if direct_eval:
            required = ("z_pred", "z_regen", "z_vae", "regen_depth", "regen_ray")
        else:
            required = self.REQUIRED
        for k in required:
            if k not in d:
                raise KeyError(f"{path} missing {k}")

        out = {"path": path}
        latent_keys = {
            "z_pred": "z_pred",
            "z_regen": "z_regen",
            "z_clean": "z_vae" if direct_eval else "z_clean",
        }
        for dst, src in latent_keys.items():
            out[dst] = torch.from_numpy(d[src]).float().to(device)

        if direct_eval:
            geometry_aliases = {
                "regen_depth": "teacher_depth",
                "regen_ray": "teacher_ray",
                "regen_ray_conf": "teacher_ray_conf",
                "vae_depth": "cleanid_depth",
                "vae_ray": "cleanid_ray",
                "vae_ray_conf": "cleanid_ray_conf",
                "pred_depth": "pred_depth",
                "pred_ray": "pred_ray",
                "pred_ray_conf": "pred_ray_conf",
            }
            for src, dst in geometry_aliases.items():
                if src in d:
                    out[dst] = torch.from_numpy(d[src].astype(np.float32)).to(device)
            if "teacher_ray_conf" not in out:
                conf_path = path[:-len("_latents.npz")] + "_teacher_ray_conf.npz"
                if os.path.isfile(conf_path):
                    with np.load(conf_path, allow_pickle=False) as conf_data:
                        if "teacher_ray_conf" not in conf_data:
                            raise KeyError(f"{conf_path} missing teacher_ray_conf")
                        conf = conf_data["teacher_ray_conf"]
                    conf_views = (conf.shape[1] if conf.ndim == 4 and conf.shape[0] == 1
                                  else conf.shape[0] if conf.ndim == 3 else -1)
                    if conf_views != out["z_pred"].shape[0]:
                        raise ValueError(
                            f"{conf_path} invalid teacher_ray_conf shape {conf.shape}")
                    out["teacher_ray_conf"] = torch.from_numpy(
                        conf.astype(np.float32)).to(device)
            pose_path = path[:-len("_latents.npz")] + "_poses.npz"
            if not os.path.isfile(pose_path):
                raise FileNotFoundError(f"pose sidecar missing for {path}: {pose_path}")
            poses = np.load(pose_path, allow_pickle=True)
            for key in ("input_c2w", "input_K", "cond_num"):
                if key not in poses:
                    raise KeyError(f"{pose_path} missing {key}")
            out["c2w"] = torch.from_numpy(poses["input_c2w"].astype(np.float32)).to(device)
            out["K"] = torch.from_numpy(poses["input_K"].astype(np.float32)).to(device)
            out["cond_num"] = int(poses["cond_num"])
            out["V"] = int(out["z_pred"].shape[0])
        else:
            for k in d.files:
                if k.endswith("_depth") or k.endswith("_ray") or k.endswith("_ray_conf"):
                    out[k] = torch.from_numpy(d[k].astype(np.float32)).to(device)
            out["c2w"] = torch.from_numpy(d["gt_c2w"].astype(np.float32)).to(device)
            out["K"] = torch.from_numpy(d["gt_K"].astype(np.float32)).to(device)
            out["cond_num"] = int(d["cond_num"])
            out["V"] = int(d["view_count"]) if "view_count" in d else out["z_pred"].shape[0]
        return out


def teacher_dict(sample, prefix):
    return {
        "depth": sample[f"{prefix}_depth"],
        "ray": sample.get(f"{prefix}_ray"),
        "ray_conf": sample.get(f"{prefix}_ray_conf"),
    }


def student_dict(out):
    return {
        "depth": out["depth"],
        "ray": ray_to_vhwc(out.get("ray")),
        "ray_conf": out.get("ray_conf"),
    }


@torch.no_grad()
def eval_gap(vae, rae, backbone_norm, sample, H, W, exclude_cond, max_views=16):
    V = sample["V"]
    z = sample["z_pred"]
    pool = np.arange(exclude_cond, V)
    if max_views > 0 and len(pool) > max_views:
        pool = pool[np.linspace(0, len(pool) - 1, max_views).round().astype(int)]
    idx = torch.from_numpy(pool).long().to(z.device)
    out = student_geo(vae, rae, backbone_norm, z, V, H, W, idx)
    sd = GAL._as_vhw(out["depth"]).float()
    td = GAL._as_vhw(sample["teacher_depth"][idx]).float()
    lp, lt = sd.clamp_min(1e-3).log(), td.clamp_min(1e-3).log()
    shift = (lt - lp).mean()
    rel = (sd * shift.exp() - td).abs() / td.clamp_min(1e-3)
    m = {"depth_absrel": float(rel.mean()), "depth_logstd": float((lt - lp).std())}
    if out.get("ray") is not None and sample.get("teacher_ray") is not None:
        pr = ray_to_vhwc(out["ray"]).float()
        tr = ray_to_vhwc(sample["teacher_ray"][idx]).float()
        if pr.shape[-1] == 6 and tr.shape[-1] == 6:
            pd = F.normalize(pr[..., :3], dim=-1)
            tdir = F.normalize(tr[..., :3], dim=-1)
            cos = (pd * tdir).sum(-1).clamp(-1, 1)
            ang = cos.arccos().rad2deg()
            m["ray_ang_mean"] = float(ang.mean())
            m["ray_ang_p95"] = float(ang.flatten().quantile(0.95))
            scale = GAL._solve_scene_scale(sd, td, weight=None)
            sd_a = sd * scale
            pr_a = torch.cat([pr[..., :3], pr[..., 3:] * scale], dim=-1)
            rc = sample.get("teacher_ray_conf")
            if rc is not None:
                rc = rc[:, idx] if (rc.ndim == 4 and rc.shape[0] == 1) else rc[idx]
            stu_pc = {"depth": sd_a, "ray": pr_a}
            tea_pc = {"depth": td, "ray": tr, "ray_conf": rc}
            pl_all = GAL.self_pose_point_loss(stu_pc, tea_pc, conf=rc, conf_percentile=0.0)
            if pl_all is not None:
                m["point_l1"] = float(pl_all)
            pl_hi = GAL.self_pose_point_loss(stu_pc, tea_pc, conf=rc, conf_percentile=50.0)
            if pl_hi is not None:
                m["point_l1_hiconf"] = float(pl_hi)
    return m


def eval_gap_mean(vae, rae, backbone_norm, val_ds, H, W, exclude_cond, device, max_views=16):
    keys = ("depth_absrel", "depth_logstd", "ray_ang_mean", "ray_ang_p95",
            "point_l1", "point_l1_hiconf")
    acc = {k: 0.0 for k in keys}
    n_ok = {k: 0 for k in keys}
    scenes = []
    for i in range(len(val_ds)):
        vs = val_ds.load(i, device)
        m = eval_gap(vae, rae, backbone_norm, vs, H, W, exclude_cond, max_views=max_views)
        scenes.append(os.path.basename(vs["path"]))
        for k in keys:
            if k in m:
                acc[k] += float(m[k])
                n_ok[k] += 1
    out = {k: acc[k] / n_ok[k] for k in keys if n_ok[k] > 0}
    out["n_val"] = len(scenes)
    out["val_scenes"] = scenes
    return out


def _group_loss(stu, tea, *, c2w, K, w_depth, w_ray, w_point, w_scale, args, scale_align=None):
    sa = args.scale_align if scale_align is None else scale_align
    if args.loss_mode == "simple":
        return GAL.simple_geo_loss(stu, tea, w_depth=w_depth, w_ray=w_ray,
                                   w_normal=args.w_normal, w_point=w_point,
                                   scale_align=sa, conf_percentile=args.conf_percentile,
                                   point_conf_percentile=args.point_conf_percentile)
    return GAL.geometry_group_loss(stu, tea, c2w=c2w, K=K, w_depth=w_depth, w_ray=w_ray,
                                   w_point=w_point, w_scale=w_scale)


@torch.no_grad()
def eval_loss_mean(vae, rae, backbone_norm, val_ds, H, W, exclude_cond, device, args):
    acc: dict[str, float] = {}
    n = 0
    for i in range(len(val_ds)):
        sample = val_ds.load(i, device)
        V = sample["V"]
        vsel = torch.arange(exclude_cond, V, device=device).long()
        c2w_s, K_s = sample["c2w"][vsel], sample["K"][vsel]
        teacher_s = slice_geo(teacher_dict(sample, "teacher"), vsel)
        cleanid_s = slice_geo(teacher_dict(sample, "cleanid"), vsel)
        logd: dict[str, float] = {}
        total = 0.0
        out = student_geo(vae, rae, backbone_norm, sample["z_pred"], V, H, W, vsel)
        cyc = _group_loss(student_dict(out), teacher_s, c2w=c2w_s, K=K_s,
                          w_depth=args.w_cycle_depth, w_ray=args.w_cycle_ray,
                          w_point=args.w_cycle_point, w_scale=args.w_scale, args=args)
        total += float(cyc["total"])
        logd.update({f"cyc_{k}": float(v) for k, v in cyc.items() if k != "total"})
        if args.w_regen_id > 0:
            out_r = student_geo(vae, rae, backbone_norm, sample["z_regen"], V, H, W, vsel)
            rid = _group_loss(student_dict(out_r), teacher_s, c2w=c2w_s, K=K_s,
                              w_depth=1.0, w_ray=0.5, w_point=0.0, w_scale=0.0,
                              args=args, scale_align=False)
            total += args.w_regen_id * float(rid["total"])
            logd.update({f"regen_{k}": float(v) for k, v in rid.items() if k != "total"})
        if args.w_clean_id > 0:
            out_c = student_geo(vae, rae, backbone_norm, sample["z_clean"], V, H, W, vsel)
            cid = _group_loss(student_dict(out_c), cleanid_s, c2w=c2w_s, K=K_s,
                              w_depth=1.0, w_ray=0.5, w_point=0.0, w_scale=0.0,
                              args=args, scale_align=False)
            total += args.w_clean_id * float(cid["total"])
            logd.update({f"clean_{k}": float(v) for k, v in cid.items() if k != "total"})
        logd["loss"] = total
        for k, v in logd.items():
            acc[k] = acc.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in acc.items()}


def build_optimizer(param_groups, args):
    common = dict(betas=(float(args.beta1), float(args.beta2)),
                  weight_decay=float(args.weight_decay), eps=float(args.adam_eps))
    name = str(args.optimizer).lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, **common)
    if name == "adam":
        return torch.optim.Adam(param_groups, **common)
    raise ValueError(f"unknown --optimizer {args.optimizer!r} (use adamw|adam)")


def _s3_cp(local_path, s3_uri):
    if not local_path or not s3_uri:
        return False
    aws = shutil.which("aws")
    if aws is None:
        print(f"    [s3][WARN] aws CLI not found; skip upload → {s3_uri}")
        return False
    try:
        subprocess.run([aws, "s3", "cp", local_path, s3_uri, "--only-show-errors"], check=True)
        print(f"    [s3] → {s3_uri}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    [s3][WARN] upload failed ({e}): {local_path} → {s3_uri}")
        return False


def _s3_join(prefix, name):
    return f"{prefix.rstrip('/')}/{name.lstrip('/')}"


def build_scheduler(optimizer, args):
    name = str(args.sched).lower()
    if name in ("none", "off", ""):
        return None
    warmup = max(int(args.warmup_steps), 0)
    total = max(int(args.decay_end_steps or args.steps), 1)
    total = max(total, warmup + 1)
    min_ratio = min(max(float(args.lr_min_ratio), 0.0), 1.0)
    if name not in ("cosine", "constant"):
        raise ValueError(f"unknown --sched {args.sched!r} (use none|cosine|constant)")

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if name == "constant":
            return 1.0
        progress = min(1.0, max(0.0, float(step - warmup) / float(max(1, total - warmup))))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _load_init_geoft(vae, path, device, *, is_main=True):
    if not path:
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = vae.state_dict()
    counts = {}
    for grp in ("geo_adapter", "dec_conv", "trunk", "rgb_head"):
        n = 0
        for k, v in (ck.get(grp) or {}).items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
                n += 1
        counts[grp] = n
    vae.load_state_dict(sd, strict=False)
    vae.to(device)
    if is_main:
        print(f"[init-geoft] step={ck.get('step')} loaded "
              + " ".join(f"{g}={counts[g]}" for g in counts) + f" from {path}")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Codec config (stage_1 + codec + vae_checkpoint)")
    ap.add_argument("--cache-dir", required=True, help="Geo cache from eval_generation.py GEO_CACHE_DIR")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--vae-ckpt", default=None)
    ap.add_argument("--init-geoft", default="", help="warm-start weights from a prior geoft_*.pt")
    ap.add_argument("--da3-weights", default="pretrained_models/da3_large/model.safetensors")
    ap.add_argument("--dpt-decoder", default="none")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr-adapter", type=float, default=1e-4)
    ap.add_argument("--lr-dec-conv", type=float, default=1e-5)
    ap.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--sched", choices=["none", "cosine", "constant"], default="cosine")
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--decay-end-steps", type=int, default=0)
    ap.add_argument("--freeze-dec-conv", action="store_true")
    ap.add_argument("--train-trunk", action="store_true")
    ap.add_argument("--train-rgb-head", action="store_true")
    ap.add_argument("--w-rgb", type=float, default=1.0)
    ap.add_argument("--w-rgb-lpips", type=float, default=0.0)
    ap.add_argument("--lr-rgb", type=float, default=5e-5)
    ap.add_argument("--adapter-hidden-ratio", type=float, default=4.0)
    ap.add_argument("--adapter-heads", type=int, default=8)
    ap.add_argument("--adapter-num-blocks", type=int, default=4)
    ap.add_argument("--adapter-width-mult", type=float, default=8.0)
    ap.add_argument("--exclude-cond", type=int, default=1)
    ap.add_argument("--views-per-step", type=int, default=8)
    ap.add_argument("--scenes-per-step", type=int, default=1)
    ap.add_argument("--alpha-min", type=float, default=0.25)
    ap.add_argument("--alpha-max", type=float, default=0.0)
    ap.add_argument("--loss-mode", choices=["simple", "rich"], default="simple")
    ap.add_argument("--scale-align", dest="scale_align", action="store_true", default=True)
    ap.add_argument("--no-scale-align", dest="scale_align", action="store_false")
    ap.add_argument("--conf-percentile", type=float, default=0.0)
    ap.add_argument("--point-conf-percentile", type=float, default=50.0)
    ap.add_argument("--w-normal", type=float, default=0.0)
    ap.add_argument("--w-cycle-depth", type=float, default=1.0)
    ap.add_argument("--w-cycle-ray", type=float, default=0.5)
    ap.add_argument("--w-cycle-point", type=float, default=0.25)
    ap.add_argument("--w-scale", type=float, default=0.1)
    ap.add_argument("--w-regen-id", type=float, default=0.5)
    ap.add_argument("--w-clean-id", type=float, default=0.25)
    ap.add_argument("--w-adapter-res", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--s3-out-dir", default="")
    ap.add_argument("--holdout-idx", type=int, default=None)
    ap.add_argument("--val-scenes", type=int, default=8)
    ap.add_argument("--val-seed", type=int, default=0)
    ap.add_argument("--train-val-scenes", type=int, default=8)
    ap.add_argument("--train-val-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dist_on, rank, world, local_rank = _setup_dist()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    if dist_on:
        dist.barrier()
    cfg = OmegaConf.load(args.config)

    rae, vae, backbone_norm, embed_dim, H, W, vae_ckpt_path = build_models(cfg, args, device)
    rgb_ref = None
    lpips_net = None
    if args.train_rgb_head:
        rgb_ref = copy.deepcopy(vae).eval()
        rgb_ref.requires_grad_(False)
        if args.w_rgb_lpips > 0:
            try:
                from disc import LPIPS
            except ImportError as exc:
                raise RuntimeError("--w-rgb-lpips>0 needs the LPIPS module (src/disc)") from exc
            lpips_net = LPIPS().to(device).eval()
            lpips_net.requires_grad_(False)
    _load_init_geoft(vae, args.init_geoft, device, is_main=is_main)
    if dist_on and world > 1:
        for p in vae.parameters():
            dist.broadcast(p.data, src=0)
        for b in vae.buffers():
            dist.broadcast(b.data, src=0)
    set_trainable(vae, args.freeze_dec_conv, train_trunk=args.train_trunk,
                  train_rgb_head=args.train_rgb_head)
    # Per-block activation checkpointing on the RGB head (GEOFT_GRAD_CKPT=1). Its
    # divided temporal attention mixes all V views, so it can't be view-chunked;
    # wrapping EACH block => backward recomputes one block at a time. Student only;
    # rgb_ref is a frozen deepcopy and is untouched.
    if (os.environ.get("GEOFT_GRAD_CKPT", "0") == "1"
            and getattr(vae, "rgb_head", None) is not None
            and hasattr(vae.rgb_head, "blocks")):
        import torch.utils.checkpoint as _uc
        def _wrap_ckpt_block(_m):
            _orig_fwd = _m.forward
            def _ckpt_fwd(*a, **k):
                if torch.is_grad_enabled():
                    return _uc.checkpoint(_orig_fwd, *a, use_reentrant=False, **k)
                return _orig_fwd(*a, **k)
            _m.forward = _ckpt_fwd
        for _blk in vae.rgb_head.blocks:
            _wrap_ckpt_block(_blk)
        if is_main:
            print(f"[grad_ckpt] rgb_head: wrapped {len(vae.rgb_head.blocks)} blocks")
    n_adapter = sum(p.numel() for n, p in vae.named_parameters()
                    if p.requires_grad and n.startswith("geo_adapter"))
    n_dec = sum(p.numel() for n, p in vae.named_parameters()
                if p.requires_grad and n.startswith("dec_conv"))
    if is_main:
        print(f"[params] adapter={n_adapter/1e6:.2f}M dec_conv={n_dec/1e6:.2f}M "
              f"[blocks={args.adapter_num_blocks} width_mult={args.adapter_width_mult} "
              f"hidden_ratio={args.adapter_hidden_ratio} heads={args.adapter_heads}]")

    train_ds = GeoCacheDataset(args.cache_dir, args.domains, split_holdout=args.holdout_idx,
                               want_val=False, val_scenes=args.val_scenes, val_seed=args.val_seed)
    # Build a held-out val set only when one is actually requested (val_scenes>0 or an
    # explicit holdout split). Otherwise leave it None: GeoCacheDataset would fall back to
    # the full train set for want_val=True, leaking train_sub into "val" and tripping the
    # overlap assert / double-counting eval.
    if (args.val_scenes and args.val_scenes > 0) or args.holdout_idx is not None:
        val_ds = GeoCacheDataset(args.cache_dir, args.domains, split_holdout=args.holdout_idx,
                                 want_val=True, val_scenes=args.val_scenes, val_seed=args.val_seed)
    else:
        val_ds = None
    train_sub_files = _pick_subset(train_ds.files, args.train_val_scenes, args.train_val_seed)
    train_sub_ds = (GeoCacheDataset(args.cache_dir, files=train_sub_files, label="train_sub")
                    if train_sub_files else None)
    if is_main:
        print(f"[split] train={len(train_ds)} "
              f"val(holdout)={len(val_ds) if val_ds is not None else 0} "
              f"train_sub={len(train_sub_ds) if train_sub_ds else 0}")
        if train_sub_ds is not None and val_ds is not None:
            overlap = set(train_sub_ds.files) & set(val_ds.files)
            assert not overlap, f"train_sub leaked into holdout: {overlap}"

    param_groups = [{"params": [p for n, p in vae.named_parameters()
                                if p.requires_grad and n.startswith("geo_adapter")],
                     "lr": args.lr_adapter}]
    dec_params = [p for n, p in vae.named_parameters()
                  if p.requires_grad and n.startswith("dec_conv")]
    if dec_params:
        param_groups.append({"params": dec_params, "lr": args.lr_dec_conv})
    rgb_params = [p for n, p in vae.named_parameters()
                  if p.requires_grad and n.startswith(_TRUNK_PREFIXES + _RGB_PREFIXES)]
    rgb_group_idx = -1
    if rgb_params:
        rgb_group_idx = len(param_groups)
        param_groups.append({"params": rgb_params, "lr": args.lr_rgb})
    if not any(g["params"] for g in param_groups):
        raise RuntimeError("no trainable params — check freeze flags / adapter init")
    opt = build_optimizer(param_groups, args)
    if args.decay_end_steps <= 0:
        args.decay_end_steps = args.steps
    sched = build_scheduler(opt, args)
    all_trainable = [p for g in param_groups for p in g["params"]]

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    logf = open(log_path, "a") if is_main else None
    if is_main:
        print(f"[start] steps={args.steps} scenes={len(train_ds)} world={world} "
              f"freeze_dec_conv={args.freeze_dec_conv} out={args.out_dir}")

    t0 = time.time()

    def group(stu, tea, *, c2w=None, K=None, w_depth, w_ray, w_point=0.0, w_scale=0.0,
              scale_align=None):
        sa = args.scale_align if scale_align is None else scale_align
        if args.loss_mode == "simple":
            return GAL.simple_geo_loss(stu, tea, w_depth=w_depth, w_ray=w_ray,
                                       w_normal=args.w_normal, w_point=w_point,
                                       scale_align=sa, conf_percentile=args.conf_percentile,
                                       point_conf_percentile=args.point_conf_percentile)
        return GAL.geometry_group_loss(stu, tea, c2w=c2w, K=K, w_depth=w_depth, w_ray=w_ray,
                                       w_point=w_point, w_scale=w_scale)

    for step in range(1, args.steps + 1):
        vae.train()
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        acc_log: dict[str, float] = {}
        last_scene = ""
        for _micro in range(args.scenes_per_step):
            idx = np.random.randint(len(train_ds))
            sample = train_ds.load(idx, device)
            V = sample["V"]
            teacher = teacher_dict(sample, "teacher")
            cleanid = teacher_dict(sample, "cleanid")
            pool = np.arange(args.exclude_cond, V)
            k = min(args.views_per_step, len(pool)) if args.views_per_step > 0 else len(pool)
            vsel = torch.from_numpy(
                np.sort(np.random.choice(pool, size=k, replace=False))).long().to(device)
            c2w_s, K_s = sample["c2w"][vsel], sample["K"][vsel]
            teacher_s = slice_geo(teacher, vsel)
            cleanid_s = slice_geo(cleanid, vsel)
            scale = 1.0 / max(1, args.scenes_per_step)
            micro_loss = 0.0

            pred_like = [("pred", sample["z_pred"], None)]
            if args.alpha_max > 0:
                a0 = min(args.alpha_min, args.alpha_max)
                a1 = max(args.alpha_min, args.alpha_max)
                alpha = float(np.random.uniform(a0, a1))
                z_synth = sample["z_regen"] + alpha * (sample["z_pred"] - sample["z_regen"])
                pred_like.append(("synth", z_synth, alpha))

            mlogd = {}
            for tag, z_in, alpha in pred_like:
                out = student_geo(vae, rae, backbone_norm, z_in, V, H, W, vsel)
                cyc = group(student_dict(out), teacher_s, c2w=c2w_s, K=K_s,
                            w_depth=args.w_cycle_depth, w_ray=args.w_cycle_ray,
                            w_point=args.w_cycle_point, w_scale=args.w_scale)
                (cyc["total"] * scale).backward()
                micro_loss += float(cyc["total"])
                if tag == "pred":
                    mlogd.update({f"cyc_{kk}": float(v) for kk, v in cyc.items() if kk != "total"})
                else:
                    mlogd.update({f"mix_{kk}": float(v) for kk, v in cyc.items() if kk != "total"})
                    mlogd["mix_alpha"] = float(alpha)
                del out, cyc

            if args.w_regen_id > 0:
                out_regen = student_geo(vae, rae, backbone_norm, sample["z_regen"], V, H, W, vsel)
                rid = group(student_dict(out_regen), teacher_s, c2w=c2w_s, K=K_s,
                            w_depth=1.0, w_ray=0.5, scale_align=False)
                (args.w_regen_id * rid["total"] * scale).backward()
                micro_loss += args.w_regen_id * float(rid["total"])
                mlogd.update({f"regen_{kk}": float(v) for kk, v in rid.items() if kk != "total"})
                del out_regen, rid

            if args.w_clean_id > 0:
                out_clean = student_geo(vae, rae, backbone_norm, sample["z_clean"], V, H, W, vsel)
                cid = group(student_dict(out_clean), cleanid_s, c2w=c2w_s, K=K_s,
                            w_depth=1.0, w_ray=0.5, scale_align=False)
                (args.w_clean_id * cid["total"] * scale).backward()
                micro_loss += args.w_clean_id * float(cid["total"])
                mlogd.update({f"clean_{kk}": float(v) for kk, v in cid.items() if kk != "total"})
                del out_clean, cid

            if rgb_ref is not None and (args.w_rgb > 0 or args.w_rgb_lpips > 0):
                # GEOFT_RGB_VIEWS optionally caps #views for the RGB/LPIPS objective
                # (decode_rgb mixes views temporally, so it can't be view-chunked).
                _rgbv = int(os.environ.get("GEOFT_RGB_VIEWS", "0") or 0)
                if 0 < _rgbv < int(vsel.numel()):
                    _perm = torch.randperm(int(vsel.numel()), device=vsel.device)[:_rgbv]
                    _rsel = vsel[torch.sort(_perm).values]
                else:
                    _rsel = vsel
                kv = int(_rsel.numel())
                rgb_s = vae.decode_rgb(sample["z_pred"][_rsel], num_views=kv)
                with torch.no_grad():
                    rgb_t = rgb_ref.decode_rgb(sample["z_regen"][_rsel], num_views=kv)
                rgb_loss = rgb_s.new_zeros(())
                if args.w_rgb > 0:
                    rgb_l1 = F.l1_loss(rgb_s, rgb_t)
                    rgb_loss = rgb_loss + args.w_rgb * rgb_l1
                    mlogd["rgb_l1"] = float(rgb_l1)
                if lpips_net is not None and args.w_rgb_lpips > 0:
                    with torch.autocast(device_type=device.type, enabled=False):
                        lp = lpips_net(_imagenet_to_pm1(rgb_s.float()),
                                       _imagenet_to_pm1(rgb_t.float()))
                    rgb_loss = rgb_loss + args.w_rgb_lpips * lp
                    mlogd["rgb_lpips"] = float(lp)
                (rgb_loss * scale).backward()
                micro_loss += float(rgb_loss)
                del rgb_s, rgb_t, rgb_loss

            if args.w_adapter_res > 0 and vae.geo_adapter is not None:
                z_out = vae.apply_geo_adapter(sample["z_pred"], V)
                res = GAL.adapter_residual_loss(sample["z_pred"], z_out)
                (args.w_adapter_res * res * scale).backward()
                micro_loss += args.w_adapter_res * float(res)
                mlogd["adapter_res"] = float(res)
                del z_out, res

            step_loss += micro_loss
            for kk, vv in mlogd.items():
                if isinstance(vv, (int, float)):
                    acc_log[kk] = acc_log.get(kk, 0.0) + float(vv)
            last_scene = os.path.basename(sample["path"])
        n_micro = max(1, args.scenes_per_step)
        step_loss /= n_micro
        logd = {kk: vv / n_micro for kk, vv in acc_log.items()}

        if dist_on and world > 1:
            for p in all_trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world)
        gnorm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
        opt.step()
        if sched is not None:
            sched.step()

        if is_main:
            lr_adapter = float(opt.param_groups[0]["lr"])
            lr_dec = float(opt.param_groups[1]["lr"]) if len(opt.param_groups) > 1 else 0.0
            lr_rgb = float(opt.param_groups[rgb_group_idx]["lr"]) if rgb_group_idx >= 0 else 0.0
            gate = float(vae.geo_adapter.gate.detach()) if vae.geo_adapter else 0.0
            logd.update({"step": step, "loss": step_loss, "gnorm": float(gnorm),
                         "scene": last_scene, "lr_adapter": lr_adapter,
                         "lr_dec_conv": lr_dec, "lr_rgb": lr_rgb, "gate": gate})
            logf.write(json.dumps(logd) + "\n")
            logf.flush()
            if step % 10 == 0 or step == 1:
                print(f"[{step:>4}/{args.steps}] loss={step_loss:.4f} gate={gate:.4f} "
                      f"lrA={lr_adapter:.2e} gnorm={float(gnorm):.3f} "
                      f"({(time.time()-t0)/step:.1f}s/it)")

        if step % args.val_every == 0 or step == args.steps:
            if is_main:
                vae.eval()
                if val_ds is not None:
                    m = eval_gap_mean(vae, rae, backbone_norm, val_ds, H, W, args.exclude_cond, device)
                    m["step"] = step
                    ml = eval_loss_mean(vae, rae, backbone_norm, val_ds, H, W, args.exclude_cond, device, args)
                    m.update({f"loss_{k}" if k != "loss" else "loss": v for k, v in ml.items()})
                    short_keys = ("depth_absrel", "depth_logstd", "ray_ang_mean", "ray_ang_p95",
                                  "point_l1", "point_l1_hiconf", "n_val", "step")
                    short = {k: m[k] for k in short_keys if k in m}
                    if "loss" in m:
                        short["loss"] = round(m["loss"], 4)
                    print(f"    [val/holdout] {json.dumps(short)}")
                    logf.write(json.dumps({"val": m}) + "\n")
                if train_sub_ds is not None:
                    mt = eval_gap_mean(vae, rae, backbone_norm, train_sub_ds, H, W, args.exclude_cond, device)
                    mt["step"] = step
                    logf.write(json.dumps({"train_val": mt}) + "\n")
                logf.flush()
                vae.train()
            if dist_on:
                dist.barrier()

        if step % args.save_every == 0 or step == args.steps:
            if is_main:
                sd_cpu = {k: v.cpu() for k, v in vae.state_dict().items()}
                ckpt = {
                    "geo_adapter": {k: v for k, v in sd_cpu.items() if k.startswith("geo_adapter")},
                    "dec_conv": {k: v for k, v in sd_cpu.items() if k.startswith("dec_conv")},
                    "optimizer": opt.state_dict(),
                    "scheduler": None if sched is None else sched.state_dict(),
                    "step": step, "old_vae": vae_ckpt_path, "args": vars(args),
                }
                if args.train_trunk:
                    ckpt["trunk"] = {k: v for k, v in sd_cpu.items() if k.startswith(_TRUNK_PREFIXES)}
                if args.train_rgb_head:
                    ckpt["rgb_head"] = {k: v for k, v in sd_cpu.items() if k.startswith(_RGB_PREFIXES)}
                path = os.path.join(args.out_dir, f"geoft_{step:06d}.pt")
                torch.save(ckpt, path)
                print(f"    [ckpt] {path}")
                if args.s3_out_dir:
                    _s3_cp(path, _s3_join(args.s3_out_dir, os.path.basename(path)))
                    if os.path.isfile(log_path):
                        _s3_cp(log_path, _s3_join(args.s3_out_dir, "train_log.jsonl"))
            if dist_on:
                dist.barrier()

    if is_main and logf is not None:
        logf.close()
    if dist_on:
        dist.barrier()
        dist.destroy_process_group()
    if is_main:
        print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
