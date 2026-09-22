#!/usr/bin/env python3
"""Pure text-to-image generation with a GAE flow model.

This drives the *same* validated Euler + CFG sampler as the video/i2v path
(``scripts/eval/eval_generation.sample_v4_euler``) with ``V=1``, ``cond_num=0`` and
no camera rays, so a single frame is generated from text alone. The generated
latent is decoded to RGB through the codec's RGB head — the same head trained by
the ``cotrain_t2i`` branch of Stage 1 (see ``configs/gae_64.yaml``).

Example
-------
    python scripts/demo/generate_t2i.py \
        --hf-repo TencentARC/GAE-D64-1B \
        --prompts-file examples/t2i_prompts.txt \
        --output results/t2i

    # multiple prompts (';;'-separated) or from a file, N images each:
    python scripts/demo/generate_t2i.py --hf-repo TencentARC/GAE-D64-1B \
        --prompts "a snowy cabin at dusk;;a bowl of ramen" --num-images 2 \
        --output results/t2i

Notes
-----
* ``--codec-ckpt`` must contain the RGB head (any GAE codec trained with
  ``cotrain_t2i`` does). The RGB head is auto-detected from the checkpoint keys.
* Depth and a DPT point cloud are written next to each PNG by default
  (``--no-pointcloud`` to skip). Camera-controlled video still uses
  ``scripts/demo/generate.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src", _ROOT / "scripts" / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

# Reuse the validated building blocks from the i2v/video evaluator.
from eval_generation import (  # noqa: E402
    sample_v4_euler,
    _denorm_latent,
    _inject_rgb_decoder_from_ckpt,
    _fast_load,
    _resolve_text_model_path,
    _resolve_da3_encoder_path,
    _cache_da3_dir,
    _scene_pointcloud_from_dpt,
    save_pointcloud_ply,
)
from eval_data import raw_no_cls_to_dpt_input, depth_to_numpy_img  # noqa: E402
from eval_reconstruction import decode_dpt  # noqa: E402
from stage1.da3 import DA3Backbone  # noqa: E402
from stage1.gae_codec import GAECodec  # noqa: E402
from stage2.transport.flow import shift_from_latent_dim  # noqa: E402
from utils.model_utils import instantiate_from_config  # noqa: E402

# ImageNet mean/std — the space the RGB head decodes into (matches the DA3
# encoder normalization used at Stage 1).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Codec checkpoints may store weights under any of these top-level keys.
_CODEC_KEYS = ("ema_codec", "codec", "ema_vae", "vae", "model")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=str, default=str(_ROOT / "configs/flow_gae64.yaml"),
                   help="Flow config with stage_2 / codec / text_encoder blocks.")
    p.add_argument(
        "--hf-repo", default=None,
        help="Hugging Face repo id (default env GAE_HF_REPO or TencentARC/GAE-D64-1B). "
             "Downloads codec + flow into --ckpt-dir when local files are missing.",
    )
    p.add_argument("--ckpt-dir", type=str, default=str(_ROOT / "ckpts"))
    p.add_argument("--flow-ckpt", type=str, default=None,
                   help="Stage 2 flow checkpoint. Default: download from --hf-repo.")
    p.add_argument("--codec-ckpt", type=str, default=None,
                   help="GAE codec checkpoint with the RGB head. Default: download from --hf-repo.")
    p.add_argument("--output", type=str, required=True, help="Output directory.")
    p.add_argument("--prompt", type=str, default=None, help="A single prompt.")
    p.add_argument("--prompts", type=str, default=None,
                   help="';;'-separated prompts (also accepts newlines).")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="One prompt per line ('#' comments allowed).")
    p.add_argument("--num-images", type=int, default=1,
                   help="Images per prompt (each with its own seed).")
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--guidance", choices=["none", "cfg", "ig", "cfg_ig"], default="ig",
                   help="T2I default is internal guidance (autoguidance).")
    p.add_argument("--ig-scale", type=float, default=2.0)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--sample-eps", type=float, default=1.0 / 1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    p.add_argument("--resolution", type=str, default=None,
                   help="RGB size: '504' (square) or 'WxH' like '672x378'. "
                        "Default = stage_1.encoder_input_size (square).")
    p.add_argument("--time-shift", type=float, default=None,
                   help="Override the Euler time-grid shift (default from config).")
    p.add_argument("--t2i-shift", action="store_true",
                   help="Use cotrain_t2i.time_dist_shift_{dim,base} instead of misc.")
    p.add_argument("--rgb-head-temporal", choices=["auto", "on", "off", "config"],
                   default="on",
                   help="Space-time RGB heads need 'on' even for single frames; "
                        "per-frame heads use 'off'.")
    p.add_argument("--allow-missing-latent-stats", action="store_true")
    p.add_argument("--allow-partial-dit", action="store_true")
    p.add_argument("--save-pointcloud", action="store_true", default=True,
                   help="Decode DPT depth + write a .ply (default on).")
    p.add_argument("--no-pointcloud", dest="save_pointcloud", action="store_false")
    p.add_argument("--pc-stride", type=int, default=4,
                   help="Subsample stride for the T2I point cloud.")
    return p.parse_args()


def _resolve_hub_weights(args) -> None:
    if args.flow_ckpt and args.codec_ckpt and not args.hf_repo:
        return
    from gae.hub import DEFAULT_REPO, download_weights, extract_da3_stats
    repo = args.hf_repo or os.environ.get("GAE_HF_REPO") or DEFAULT_REPO
    paths = download_weights(repo, size="64", out_dir=args.ckpt_dir)
    tar = paths.get("da3_stats_giant_5ds.tar")
    if tar is not None:
        extract_da3_stats(tar, _ROOT / "model_stats" / "da3_giant_5ds")
    if not args.codec_ckpt:
        args.codec_ckpt = str(paths["gae_64.pt"])
    if not args.flow_ckpt:
        args.flow_ckpt = str(paths["flow_gae64.pt"])


def _validate_args(args) -> None:
    if args.num_images < 1:
        raise ValueError(f"--num-images must be >= 1, got {args.num_images}")
    if args.sample_steps < 1:
        raise ValueError(f"--sample-steps must be >= 1, got {args.sample_steps}")
    if not 0.0 < args.sample_eps < 1.0:
        raise ValueError(f"--sample-eps must be in (0, 1), got {args.sample_eps}")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0.0:
        raise ValueError(f"--cfg-scale must be finite and >= 0, got {args.cfg_scale}")


def _load_prompts(args) -> list[str]:
    prompts: list[str] = []
    if args.prompts_file is not None:
        for line in args.prompts_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    for raw in (args.prompt, args.prompts):
        if raw:
            for line in raw.replace(";;", "\n").splitlines():
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        prompts = ["a red apple on a wooden table",
                   "a snow-covered cabin in a pine forest at twilight"]
        print("  [prompts] none given -> using 2 default smoke prompts")
    return prompts


def _ignorable_dit_key(key: str) -> bool:
    return (key.startswith("repa_projector.")
            or key.endswith(("freqs_cos", "freqs_sin", "inv_freq")))


def _load_dit(cfg, ckpt_path, use_ema, device, allow_partial=False):
    model_cfg = OmegaConf.to_container(cfg.stage_2, resolve=True)
    dit = instantiate_from_config(model_cfg).to(device).eval()
    ckpt = _fast_load(ckpt_path, map_location="cpu")
    wrappers = ("ema", "model", "state_dict") if use_ema else ("model", "ema", "state_dict")
    sd = None
    key = None
    if isinstance(ckpt, dict):
        for w in wrappers:
            inner = ckpt.get(w)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                sd, key = dict(inner), w
                break
        if sd is None and any(torch.is_tensor(v) for v in ckpt.values()):
            sd, key = dict(ckpt), "<raw state_dict>"
    if sd is None:
        raise KeyError(
            f"checkpoint has no tensors under {wrappers} nor a raw state_dict")
    model_keys = set(dit.state_dict().keys())
    stale = [k for k in sd if k not in model_keys]
    bad_stale = [k for k in stale if not _ignorable_dit_key(k)]
    if bad_stale and not allow_partial:
        raise RuntimeError(f"DiT ckpt has unexpected keys, e.g. {bad_stale[:5]}; "
                           "use --allow-partial-dit for diagnostics")
    for k in stale:
        del sd[k]
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not _ignorable_dit_key(k)]
    if bad_missing and not allow_partial:
        raise RuntimeError(f"DiT missing keys, e.g. {bad_missing[:5]}; "
                           "use --allow-partial-dit for diagnostics")
    n = sum(p.numel() for p in dit.parameters()) / 1e6
    step = ckpt.get("train_steps", ckpt.get("train_step", "?"))
    print(f"  [dit] loaded '{key}' ({n:.1f}M, missing={len(missing)} "
          f"unexpected={len(unexpected)}) train_step={step}")
    for p in dit.parameters():
        p.requires_grad_(False)
    return dit, model_cfg


def _load_codec(cfg, codec_ckpt, temporal_mode, device):
    if not os.path.isfile(codec_ckpt):
        raise FileNotFoundError(f"codec checkpoint not found: {codec_ckpt}")
    print(f"  [codec] loading {codec_ckpt}")
    ckpt = _fast_load(codec_ckpt, map_location="cpu")
    sd = ckpt
    if isinstance(ckpt, dict):
        for k in _CODEC_KEYS:
            if k in ckpt:
                sd = ckpt[k]
                break
    vae_cfg = OmegaConf.to_container(cfg.codec, resolve=True)
    _inject_rgb_decoder_from_ckpt(vae_cfg, sd, temporal_mode=temporal_mode)
    vae_cfg.pop("rgb_decoder_hidden", None)
    vae_cfg.pop("rgb_decoder_heads", None)
    codec = GAECodec(**vae_cfg).to(device).eval()
    missing, unexpected = codec.load_state_dict(sd, strict=False)
    if codec.rgb_head is None:
        raise RuntimeError("codec built without rgb_head — ckpt has no RGB head weights")
    rgb_bad = [k for k in (*missing, *unexpected) if k.startswith("rgb_head")]
    if rgb_bad:
        raise RuntimeError(f"rgb_head load mismatch, e.g. {rgb_bad[:3]}")
    for p in codec.parameters():
        p.requires_grad_(False)
    print(f"  [codec] OK latent_dim={codec.latent_dim} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    return codec


def _load_backbone(cfg, device):
    """Frozen DA3-GIANT encoder + DPT head (needed for T2I depth / ply)."""
    params = OmegaConf.to_container(cfg.stage_1.params, resolve=True)
    params["reshape_to_2d"] = False
    enc = _resolve_da3_encoder_path(
        params.get("encoder_pretrained_path"),
        params.get("da3_weights_path"),
    )
    params["encoder_pretrained_path"] = _cache_da3_dir(enc)
    rae = DA3Backbone(**params).to(device).eval()
    for p in rae.parameters():
        p.requires_grad_(False)
    ok = rae.rae_cl_decoder is not None
    print(f"  [da3] DPT head: {'OK' if ok else 'MISSING (no depth/ply)'}")
    return rae


def _save_t2i_geometry(z, rgb01, stem, out_dir, codec, rae, H, W, device,
                       use_amp, amp_dtype, pc_stride):
    """One DPT pass -> ``<stem>_depth.png`` + ``<stem>_pointcloud.ply``."""
    if rae is None or rae.rae_cl_decoder is None:
        return
    backbone_norm = rae.encoder.backbone.pretrained.norm
    z = z.to(device)
    rgb = rgb01.unsqueeze(0) if rgb01.ndim == 3 else rgb01
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        seq, h_l, w_l = codec._decode_trunk(z)
        c_trunk = seq.shape[-1]
        raw = codec.dec_conv(
            seq.permute(0, 2, 1).reshape(-1, c_trunk, h_l, w_l))
        recon_feats = codec.denormalize_and_split(raw)
    dpt_in = raw_no_cls_to_dpt_input(recon_feats, backbone_norm, z.shape[0])
    dpt_out = decode_dpt(dpt_in, rae.rae_cl_decoder, H, W)
    depth = dpt_out["depth"]
    Image.fromarray(depth_to_numpy_img(depth[0])).save(out_dir / f"{stem}_depth.png")
    xyz, pc_rgb = _scene_pointcloud_from_dpt(
        dpt_out, depth, rgb, z.shape[0], H, W, device, stride=pc_stride,
    )
    ply = out_dir / f"{stem}_pointcloud.ply"
    save_pointcloud_ply(str(ply), xyz, pc_rgb)
    print(f"    depth+ply {ply.name} ({xyz.shape[0]} pts)")


def _encode_tokens(enc, texts, device, dtype):
    return enc(texts)["tokens"].to(device=device, dtype=dtype)


def _load_text_encoder(cfg, device, output_dtype):
    if not cfg.get("use_text", True):
        return None, None
    text_cfg = OmegaConf.to_container(cfg.get("text_encoder", {}), resolve=True) or {}
    name = str(text_cfg.get("model_name", "Qwen/Qwen3-0.6B"))
    path = _resolve_text_model_path(name)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from stage2.models.text_encoder import Qwen3TextEncoder
    enc = Qwen3TextEncoder(
        model_name=path,
        max_length=int(text_cfg.get("max_length", 256)),
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        null_tokens = _encode_tokens(enc, [""], device, output_dtype)
    print(f"  [text] {name} loaded (null shape={tuple(null_tokens.shape)})")
    return enc, null_tokens


def _resolve_time_shift(args, cfg, in_ch, latent_h, latent_w):
    if args.time_shift is not None:
        return float(args.time_shift), "cli"
    node = "cotrain_t2i" if args.t2i_shift else "misc"
    sub = OmegaConf.to_container(cfg.get(node, {}), resolve=True) or {}
    if "time_dist_shift" in sub:
        return float(sub["time_dist_shift"]), f"{node}.time_dist_shift"
    dim = int(sub.get("time_dist_shift_dim", in_ch * latent_h * latent_w))
    base = int(sub.get("time_dist_shift_base", 4096))
    return shift_from_latent_dim(dim, base=base), f"{node}(dim={dim})"


def _slug(text: str, n: int = 40) -> str:
    return "".join(c if c.isalnum() else "-" for c in text[:n]).strip("-")


def _resolve_rgb_size(spec, cfg):
    if spec:
        s = str(spec).strip().lower().replace("*", "x")
        if "x" in s:
            w_str, h_str = s.split("x", 1)
            W, H = int(w_str), int(h_str)
        else:
            W = H = int(s)
    else:
        # Prefer the 16:9 cotrain image_size (W, H) the shipped flow was trained at;
        # fall back to the square encoder_input_size only when it is absent.
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
                W = H = int(img)
        else:
            W = H = int(cfg.stage_1.params.get("encoder_input_size", 504))
    for name, v in (("width", W), ("height", H)):
        if v <= 0 or v % 14 != 0:
            raise ValueError(f"{name}={v} must be a positive multiple of 14 (DA3 patch)")
    return W, H


def main():
    args = parse_args()
    _resolve_hub_weights(args)
    _validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    ac = dict(device_type=device.type, enabled=(dtype == torch.bfloat16), dtype=dtype)

    cfg = OmegaConf.load(args.config)
    prediction = str(cfg.get("transport", {}).get("params", {})
                     .get("prediction", "x")).lower()
    if prediction not in {"x", "velocity"}:
        raise ValueError(f"unsupported transport prediction {prediction!r}")

    W, H = _resolve_rgb_size(args.resolution, cfg)
    latent_h, latent_w = H // 14, W // 14

    print("=" * 68)
    print(f"  GAE text-to-image (sample_v4_euler, V=1)  prediction={prediction}")
    print(f"  config : {os.path.basename(args.config)}")
    print(f"  rgb    : {W}x{H} (WxH)  latent: {latent_h}x{latent_w}")
    print("=" * 68)

    print("[1/4] flow (DiT) ...")
    dit, model_cfg = _load_dit(cfg, args.flow_ckpt, args.use_ema, device,
                               allow_partial=args.allow_partial_dit)
    in_ch = int(model_cfg["params"].get("in_channels", 128))

    print("[2/5] codec (RGB head) ...")
    codec = _load_codec(cfg, args.codec_ckpt, args.rgb_head_temporal, device)
    if int(codec.latent_dim) != in_ch:
        raise RuntimeError(f"flow in_channels={in_ch} but codec latent_dim="
                           f"{codec.latent_dim}; config/checkpoint mismatch")

    print("[3/5] DA3 DPT (depth / point cloud) ...")
    rae = _load_backbone(cfg, device) if args.save_pointcloud else None
    if args.save_pointcloud and (rae is None or rae.rae_cl_decoder is None):
        print("  [WARN] DPT unavailable; RGB only")
        rae = None

    print("[4/5] text encoder ...")
    text_enc, null_tokens = _load_text_encoder(cfg, device, dtype)
    if text_enc is None and args.guidance in {"cfg", "cfg_ig"}:
        raise ValueError("CFG requested but use_text=false")

    print("[5/5] latent stats + sampler ...")
    latent_mean = latent_std = latent_unwhiten = None
    stats_path = str(cfg.get("latent_stats", "")).strip()
    if stats_path and not os.path.isabs(stats_path):
        stats_path = str(_ROOT / stats_path)
    if stats_path and os.path.isfile(stats_path):
        stats = _fast_load(stats_path, map_location="cpu")
        latent_mean = stats["mean"].float().to(device).reshape(1, -1, 1, 1)
        latent_std = stats["std"].float().to(device).reshape(1, -1, 1, 1).clamp(min=1e-5)
        if latent_mean.shape[1] != in_ch:
            raise RuntimeError(f"latent stats channels {latent_mean.shape[1]} != "
                               f"flow in_channels {in_ch}")
        if "unwhiten" in stats:
            latent_unwhiten = stats["unwhiten"].float().to(device)
            print(f"  latent stats: full-cov whitening ({stats_path})")
        else:
            print(f"  latent stats: per-channel ({stats_path})")
    elif args.allow_missing_latent_stats:
        print("  [WARN] latent stats missing; proceeding (unnormalised)")
    else:
        raise FileNotFoundError(
            f"latent_stats not found: {stats_path or '<unset>'}; pass "
            "--allow-missing-latent-stats only for an unnormalised model")

    time_shift, shift_src = _resolve_time_shift(args, cfg, in_ch, latent_h, latent_w)
    print(f"  time_dist_shift={time_shift:.4f} [{shift_src}]")

    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    prompts = _load_prompts(args)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.txt").write_text("\n".join(prompts))

    print(f"\n[sample] {len(prompts)} prompt(s) x {args.num_images} img -> "
          f"{len(prompts) * args.num_images} total")
    saved = 0
    global_idx = 0
    for p_idx, prompt in enumerate(prompts):
        with torch.no_grad(), torch.amp.autocast(**ac):
            text_emb = (_encode_tokens(text_enc, [prompt], device, dtype)
                        if text_enc is not None else None)
        for img_i in range(args.num_images):
            torch.manual_seed(args.seed + global_idx)
            with torch.no_grad(), torch.amp.autocast(**ac):
                z = sample_v4_euler(
                    dit, z_ref_clean=None, total_view=1, cond_num=0,
                    plucker_6d=None, ref_global=text_emb,
                    cfg_uncond_ref_global=null_tokens,
                    num_steps=args.sample_steps, cfg_scale=args.cfg_scale,
                    guidance_mode=args.guidance, ig_scale=args.ig_scale,
                    time_dist_shift=time_shift, eps=args.sample_eps,
                    noise_shape=(in_ch, latent_h, latent_w),
                    device=device, dtype=torch.float32, prediction=prediction,
                )
                if latent_mean is not None:
                    z = _denorm_latent(z, latent_mean, latent_std, latent_unwhiten)
                rgb = codec.decode_rgb(z.to(dtype), num_views=z.shape[0]).float()
            if tuple(rgb.shape[-2:]) != (H, W):
                raise RuntimeError(f"decoder produced {tuple(rgb.shape[-2:])}, "
                                   f"expected {(H, W)}")
            rgb01 = (rgb * std + mean).clamp(0, 1)[0]
            arr = (rgb01.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype("uint8")
            fname = f"{p_idx:04d}_i{img_i}_s{args.seed + global_idx}_{_slug(prompt)}.png"
            Image.fromarray(arr).save(out_dir / fname)
            if rae is not None:
                stem = Path(fname).stem
                try:
                    _save_t2i_geometry(
                        z, rgb01, stem, out_dir, codec, rae, H, W, device,
                        use_amp=(dtype == torch.bfloat16), amp_dtype=dtype,
                        pc_stride=args.pc_stride,
                    )
                except Exception as exc:
                    print(f"    [WARN] geometry failed for {fname}: {exc}")
            saved += 1
            global_idx += 1
        print(f"  [{p_idx + 1}/{len(prompts)}] {prompt[:60]}  saved={saved}")

    meta = dict(
        flow_ckpt=args.flow_ckpt, codec_ckpt=args.codec_ckpt, config=args.config,
        prediction=prediction, resolution_wh=[W, H], latent=[in_ch, latent_h, latent_w],
        cfg_scale=args.cfg_scale, guidance=args.guidance, sample_steps=args.sample_steps,
        time_dist_shift=time_shift, seed=args.seed, num_images=args.num_images,
        n_prompts=len(prompts), sampler="sample_v4_euler (shared with video eval)",
        save_pointcloud=bool(args.save_pointcloud and rae is not None),
        pc_stride=args.pc_stride,
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
