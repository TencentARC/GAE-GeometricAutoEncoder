#!/usr/bin/env python3
"""Merge a geo-decoder finetune checkpoint into a full GAE codec checkpoint.

``scripts/train/finetune_geo_decoder.py`` saves only the trained module groups
(``geo_adapter``, ``dec_conv``, and optionally the decode trunk / RGB head).
Flow eval / training expect a complete codec checkpoint plus a codec config
whose ``codec`` block instantiates the ``geo_adapter``. This utility
produces both artifacts (the "merged VAE") without modifying the source
checkpoints.

Example
-------
    python scripts/train/merge_geoft_vae_ckpt.py \
        --geoft results/geoft/run0/geoft_002000.pt \
        --base-vae ckpts/gae_128.pt \
        --base-gld-config configs/gae_128.yaml \
        --out-ckpt ckpts/gae_128_geoft.pt \
        --out-gld-config configs/gae_128_geoft.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _state_dict(payload: object) -> dict:
    if isinstance(payload, dict):
        for key in ("ema_codec", "ema_vae", "codec", "vae", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in payload.keys()):
            return payload  # type: ignore[return-value]
    raise TypeError("checkpoint does not contain a codec state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geoft", required=True, help="Path to geoft_*.pt")
    parser.add_argument("--base-vae", default=None,
                        help="Base full codec ckpt. Defaults to geoft['old_vae'].")
    parser.add_argument("--base-gld-config", required=True,
                        help="Codec yaml with a codec block.")
    parser.add_argument("--out-ckpt", required=True, help="Output merged codec checkpoint.")
    parser.add_argument("--out-gld-config", required=True,
                        help="Output codec yaml with geo_adapter enabled.")
    args = parser.parse_args()

    geoft_path = Path(args.geoft)
    geoft = torch.load(geoft_path, map_location="cpu", weights_only=False)
    if not isinstance(geoft, dict):
        raise TypeError(f"unexpected geoft checkpoint type: {type(geoft)!r}")

    base_vae_path = Path(args.base_vae or geoft.get("old_vae", ""))
    if not base_vae_path.is_file():
        raise FileNotFoundError(f"base codec checkpoint not found: {base_vae_path}")

    base_ckpt = torch.load(base_vae_path, map_location="cpu", weights_only=False)
    base_sd = _state_dict(base_ckpt)
    merged = copy.deepcopy(base_sd)

    replaced: dict[str, int] = {}
    added: dict[str, int] = {}
    for group in ("geo_adapter", "dec_conv", "trunk", "rgb_head"):
        group_sd = geoft.get(group, {})
        if not isinstance(group_sd, dict):
            continue
        replaced[group] = 0
        added[group] = 0
        for key, tensor in group_sd.items():
            if key in merged:
                replaced[group] += 1
            else:
                added[group] += 1
            merged[key] = tensor

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema_vae": merged,
            "source_geoft": str(geoft_path),
            "base_vae": str(base_vae_path),
            "geoft_step": geoft.get("step"),
            "merge_replaced": replaced,
            "merge_added": added,
        },
        out_ckpt,
    )

    cfg = OmegaConf.load(args.base_gld_config)
    if "codec" not in cfg:
        raise ValueError(f"no codec in {args.base_gld_config}")
    ft_args = geoft.get("args", {})
    cfg.codec.geo_adapter = {
        "hidden_ratio": float(ft_args.get("adapter_hidden_ratio", 4.0)),
        "num_heads": int(ft_args.get("adapter_heads", 8)),
        "num_blocks": int(ft_args.get("adapter_num_blocks", 4)),
        "width_mult": float(ft_args.get("adapter_width_mult", 8.0)),
        "max_views": 64,
        "use_spatial": True,
    }
    out_cfg = Path(args.out_gld_config)
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_cfg)

    print(f"wrote ckpt: {out_ckpt}")
    print(f"wrote codec config: {out_cfg}")
    print(f"replaced: {replaced}")
    print(f"added: {added}")


if __name__ == "__main__":
    main()
