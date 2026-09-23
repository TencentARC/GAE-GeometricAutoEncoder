"""End-to-end GAE inference pipeline.

Wires the four frozen / trained pieces together so a caller does not have to
know the two-stage layout:

    images --DA3Backbone--> 4-level features --normalize_levels--> fused X
          --GAECodec.encode--> posterior mean z  (the generated state)
          --GAEFlow (Stage 2)--> sampled z_hat
          --GAECodec.decode_rgb--> RGB
          --GAECodec.decode_geo--> rebuilt hierarchy --DA3 DPT head--> depth / rays / pointmap

Reading order for the paper: Sections 3.1 (codec) and 3.2-3.3 (flow and
conditioning). Equation numbers are cited on each method.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_REPO_ROOT / "src")
_EVAL = str(_REPO_ROOT / "scripts" / "eval")
for _p in (_SRC, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf  # noqa: E402

from stage1.da3 import DA3Backbone  # noqa: E402
from stage1.gae_codec import GAECodec  # noqa: E402
from utils.dpt_helpers import _format_recon_for_dpt  # noqa: E402

__all__ = ["GAE", "load_codec", "load_backbone", "load_flow"]

_CODEC_STATE_KEYS = ("ema_codec", "ema_vae", "codec", "vae")
_FLOW_STATE_KEYS = ("ema", "model", "state_dict")


def _filter_ctor_kwargs(cls, params: dict) -> dict:
    """Drop yaml keys the constructor does not accept (train-time toggles)."""
    sig = inspect.signature(cls.__init__)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(params)
    allowed = set(sig.parameters) - {"self"}
    dropped = [k for k in list(params) if k not in allowed]
    out = {k: v for k, v in params.items() if k in allowed}
    if dropped:
        print(f"[{cls.__name__}] ignoring non-constructor keys: {dropped}")
    return out


def _unwrap_state(state, keys: Tuple[str, ...]):
    if not isinstance(state, dict):
        return state
    for key in keys:
        inner = state.get(key)
        if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
            return inner
    inner = state.get("state_dict")
    if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
        return inner
    return state


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _read_latent_stats(path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(stats, dict):
        raise TypeError(f"latent stats at {path} is not a dict")
    mean = stats.get("mean", stats.get("latent_mean"))
    std = stats.get("std", stats.get("latent_std"))
    if mean is None or std is None:
        raise KeyError(f"{path} needs 'mean'/'std' (or latent_mean/latent_std)")
    return torch.as_tensor(mean).float(), torch.as_tensor(std).float()


def _resolve_backbone_norm(backbone) -> Optional[nn.Module]:
    norm = getattr(backbone, "backbone_norm", None)
    if norm is not None and not isinstance(norm, property):
        return norm
    try:
        return backbone.encoder.backbone.pretrained.norm
    except AttributeError:
        return None


def load_backbone(cfg_path: str, device: torch.device) -> DA3Backbone:
    """Build the frozen DA3 backbone from a config's ``stage_1`` block."""
    cfg = OmegaConf.load(cfg_path)
    node = cfg.get("stage_1")
    if node is None:
        raise KeyError(f"{cfg_path} has no 'stage_1' block")
    params = OmegaConf.to_container(node.get("params", {}), resolve=True)
    backbone = DA3Backbone(**params).to(device).eval()
    backbone.requires_grad_(False)
    return backbone


def load_codec(
    cfg_path: str,
    ckpt_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> GAECodec:
    """Build the GAE codec and optionally load trained weights.

    Checkpoints may store weights under ``ema_codec`` / ``codec`` (release names)
    or ``ema_vae`` / ``vae`` (research-tree names).
    """
    cfg = OmegaConf.load(cfg_path)
    vae_cfg = cfg.get("codec")
    if vae_cfg is None:
        raise KeyError(f"{cfg_path} has no 'codec' block")
    params = _filter_ctor_kwargs(
        GAECodec, OmegaConf.to_container(vae_cfg, resolve=True) or {}
    )

    codec = GAECodec(**params).to(device).eval()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _unwrap_state(state, _CODEC_STATE_KEYS)
        missing, unexpected = codec.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.startswith("repa_proj")]
        if missing:
            raise KeyError(f"codec checkpoint is missing keys: {missing[:8]}")
        if unexpected:
            print(f"[load_codec] ignored unexpected keys: {unexpected[:8]}")
    codec.requires_grad_(False)
    return codec


def load_flow(
    cfg_path: str,
    ckpt_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Build the Stage 2 flow model from a ``stage_2`` config block."""
    from stage2.models.dit_temporal import GAEFlowTemporal

    cfg = OmegaConf.load(cfg_path)
    node = cfg.get("stage_2")
    if node is None:
        raise KeyError(f"{cfg_path} has no 'stage_2' block")
    params = OmegaConf.to_container(node.get("params", {}), resolve=True) or {}
    flow = GAEFlowTemporal(**params).to(device).eval()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _unwrap_state(state, _FLOW_STATE_KEYS)
        missing, unexpected = flow.load_state_dict(state, strict=False)
        if missing:
            raise KeyError(f"flow checkpoint is missing keys: {missing[:8]}")
        if unexpected:
            print(f"[load_flow] ignored unexpected keys: {unexpected[:8]}")
    flow.requires_grad_(False)
    return flow


class GAE(nn.Module):
    """Geometry-Native Autoencoder: encode, sample, and read out RGB + geometry.

    Args:
        backbone: Frozen DA3 encoder and DPT geometry head.
        codec: Frozen GAE codec.
        flow: Optional Stage 2 flow model. Required for :meth:`sample`, not for
            reconstruction.
        latent_mean: Per-channel mean for standardizing the latent (Eq. 15).
            ``None`` disables standardization.
        latent_std: Per-channel std for standardizing the latent.
    """

    def __init__(
        self,
        backbone: DA3Backbone,
        codec: GAECodec,
        flow: Optional[nn.Module] = None,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.codec = codec
        self.flow = flow
        if latent_mean is not None:
            self.register_buffer("latent_mean", latent_mean.view(1, -1, 1, 1))
        else:
            self.latent_mean = None
        if latent_std is not None:
            self.register_buffer("latent_std", latent_std.view(1, -1, 1, 1))
        else:
            self.latent_std = None

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Optional[str] = None,
        *,
        size: int = 64,
        cache_dir: Optional[Union[str, Path]] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> "GAE":
        """Build a :class:`GAE` from a Hugging Face Hub repo.

        Downloads codec / flow / latent-stats / DA3 stats into ``cache_dir``
        (default ``ckpts/``) and loads them with the in-repo yaml configs.
        """
        from .hub import DEFAULT_REPO, download_weights, extract_da3_stats

        repo = repo_id or DEFAULT_REPO
        out = Path(cache_dir) if cache_dir is not None else _REPO_ROOT / "ckpts"
        paths = download_weights(repo, size=str(size), out_dir=out)
        tar = paths.get("da3_stats_giant_5ds.tar")
        if tar is not None:
            extract_da3_stats(tar, _REPO_ROOT / "model_stats" / "da3_giant_5ds")
        codec_cfg = str(_REPO_ROOT / f"configs/gae_{size}.yaml")
        flow_cfg = str(_REPO_ROOT / f"configs/flow_gae{size}.yaml")
        codec_ckpt = str(paths[f"gae_{size}.pt"])
        flow_name = f"flow_gae{size}.pt"
        flow_ckpt = str(paths[flow_name]) if flow_name in paths else None
        stats = paths.get(f"latent_stats_gae_{size}.pt")
        return cls.from_configs(
            codec_cfg,
            codec_ckpt,
            flow_cfg=flow_cfg,
            flow_ckpt=flow_ckpt,
            latent_stats=str(stats) if stats is not None else None,
            device=device,
        )

    @classmethod
    def from_configs(
        cls,
        codec_cfg: str,
        codec_ckpt: Optional[str] = None,
        flow_cfg: Optional[str] = None,
        flow_ckpt: Optional[str] = None,
        latent_stats: Optional[str] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> "GAE":
        """Build a :class:`GAE` from config paths, loading weights if given."""
        backbone = load_backbone(flow_cfg or codec_cfg, device)
        codec = load_codec(codec_cfg, codec_ckpt, device)

        flow = None
        if flow_cfg:
            flow = load_flow(flow_cfg, flow_ckpt, device)

        latent_mean = latent_std = None
        stats_src = OmegaConf.load(flow_cfg or codec_cfg)
        stats_path = latent_stats or stats_src.get("latent_stats")
        if stats_path:
            resolved = _resolve_path(str(stats_path))
            if resolved.is_file():
                latent_mean, latent_std = _read_latent_stats(str(resolved))
            else:
                print(f"[from_configs] latent_stats not found: {resolved}")

        model = cls(backbone, codec, flow, latent_mean=latent_mean, latent_std=latent_std)
        if flow_cfg:
            model._configure_sampler(flow_cfg)
        return model.to(device)

    def standardize(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None:
            return z
        return (z - self.latent_mean) / (self.latent_std + 1e-6)

    def destandardize(self, z_bar: torch.Tensor) -> torch.Tensor:
        if self.latent_mean is None:
            return z_bar
        return z_bar * (self.latent_std + 1e-6) + self.latent_mean

    def _configure_sampler(self, flow_cfg: str) -> None:
        """Read the transport / time-shift settings the Euler sampler needs.

        The logit-normal sampling schedule is warped by ``time_dist_shift``
        (paper Eq. 11). Resolution order mirrors ``scripts/train/train_flow.py`` so
        training and inference agree: an explicit ``misc.time_dist_shift`` wins;
        otherwise it is derived from ``misc.time_dist_shift_dim`` (the flattened
        per-sample latent size) with ``misc.time_dist_shift_base``. The codec
        channel count alone is NOT a valid proxy for the sequence length, so we
        never fall back to it silently.
        """
        cfg = OmegaConf.load(flow_cfg)
        transport = OmegaConf.to_container(cfg.get("transport", {}), resolve=True) or {}
        params = transport.get("params", {}) if isinstance(transport, dict) else {}
        self._prediction = str(params.get("prediction", "x"))

        misc = OmegaConf.to_container(cfg.get("misc", {}), resolve=True) or {}
        self._time_dist_shift = None
        if misc.get("time_dist_shift") is not None:
            self._time_dist_shift = float(misc["time_dist_shift"])
        elif misc.get("time_dist_shift_dim") is not None:
            try:
                from stage2.transport.flow import shift_from_latent_dim
                self._time_dist_shift = float(shift_from_latent_dim(
                    int(misc["time_dist_shift_dim"]),
                    base=int(misc.get("time_dist_shift_base", 4096)),
                ))
            except Exception:
                self._time_dist_shift = None
        if self._time_dist_shift is None:
            print(
                "[from_configs] flow config has no misc.time_dist_shift; the "
                "sampling schedule falls back to 1.0 (no warp), which will not "
                "match a checkpoint trained with a shifted schedule."
            )

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Images ``[B, V, 3, H, W]`` in ``[0, 1]`` to posterior mean ``[B, V, C, h, w]``.

        Applies ImageNet normalization when the backbone exposes ``encoder_mean``
        / ``encoder_std`` (the DA3 path). Stub backbones used in smoke tests skip
        that step.
        """
        b, v = images.shape[:2]
        image_size = (images.shape[-2], images.shape[-1])
        mean = getattr(self.backbone, "encoder_mean", None)
        std = getattr(self.backbone, "encoder_std", None)
        if mean is not None and std is not None:
            flat = images.reshape(b * v, *images.shape[2:])
            mean = mean.to(device=flat.device, dtype=flat.dtype)
            std = std.to(device=flat.device, dtype=flat.dtype)
            while mean.ndim < flat.ndim:
                mean = mean.unsqueeze(0)
                std = std.unsqueeze(0)
            images = ((flat - mean) / std).reshape(b, v, *flat.shape[1:])
        all_feats = self.backbone.encode(images, mode="all")
        feats = {k: fv[:, 1:, :] for k, fv in all_feats.items()}
        x = self.codec.normalize_levels(feats, image_size=image_size)
        mu, _ = self.codec.encode(x)
        return mu.reshape(b, v, *mu.shape[1:])

    def _denormalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        """Undo the ImageNet normalization the RGB head regresses in.

        The head is trained against ``(x - mean) / std`` targets (the same
        stats the DA3 encoder consumes), so its raw output lives in normalized
        space. Invert that and clamp to ``[0, 1]`` for displayable pixels.
        Stub backbones without the stats (smoke tests) are returned unchanged.
        """
        mean = getattr(self.backbone, "encoder_mean", None)
        std = getattr(self.backbone, "encoder_std", None)
        if mean is None or std is None:
            return rgb
        mean = mean.to(device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
        std = std.to(device=rgb.device, dtype=rgb.dtype).view(1, -1, 1, 1)
        return (rgb * std + mean).clamp(0.0, 1.0)

    @torch.inference_mode()
    def decode_rgb(self, z: torch.Tensor, num_views: Optional[int] = None) -> torch.Tensor:
        """Latent ``[B*V, C, h, w]`` to display-space RGB via the RGB head (Eq. 4)."""
        rgb = self.codec.decode_rgb(z, num_views=num_views)
        return self._denormalize_rgb(rgb)

    @torch.inference_mode()
    def decode_geometry(
        self,
        z: torch.Tensor,
        height: int,
        width: int,
        backbone_norm: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Latent to geometry through the *frozen* DA3 DPT head (Eq. 4)."""
        if backbone_norm is None:
            backbone_norm = _resolve_backbone_norm(self.backbone)
        if backbone_norm is None:
            raise RuntimeError(
                "decode_geometry needs the DA3 backbone LayerNorm "
                "(encoder.backbone.pretrained.norm); pass backbone_norm=... "
                "explicitly if your backbone does not expose it"
            )
        decoder = getattr(self.backbone, "rae_cl_decoder", None)
        if decoder is None:
            raise RuntimeError("backbone has no DPT decoder (rae_cl_decoder)")

        recon = self.codec.decode(z)
        feats = self.codec.denormalize_and_split(recon)
        dpt_in = _format_recon_for_dpt(feats, backbone_norm, z.shape[0])
        device_type = z.device.type if z.device.type in ("cuda", "cpu", "mps") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return decoder(dpt_in, height, width, patch_start_idx=0)

    @torch.inference_mode()
    def sample(
        self,
        z_ref: torch.Tensor,
        total_views: int,
        cond_num: int = 1,
        *,
        plucker_6d: Optional[torch.Tensor] = None,
        ref_global: Optional[torch.Tensor] = None,
        cfg_uncond_ref_global: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        guidance_mode: str = "cfg",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample ``total_views`` latents from ``cond_num`` clean references.

        This runs the same Euler + CFG sampler as ``scripts/eval/eval_generation.py``.
        For the full single-image + prompt + camera demo (image loading, pose
        synthesis, RGB/point-cloud export) use ``scripts/demo/generate.py``; this
        method is the tensor-level entry point.

        Args:
            z_ref: Reference posterior means ``[cond_num, C, h, w]`` in *raw*
                latent space (as returned by :meth:`encode`). Standardized
                internally.
            total_views: Number of views to produce (``>= cond_num``).
            cond_num: Number of leading views treated as clean evidence.
            plucker_6d: Per-token metric ray maps ``[1, V*h*w, 6]`` (Eq. 13),
                e.g. from ``scripts/eval/eval_generation.build_plucker_from_cameras``.
            ref_global: Text cross-attention context ``[1, T, D]``.
            cfg_uncond_ref_global: Unconditional text context for CFG (encode the
                empty string with the same encoder). Required when ``cfg_scale>1``.
            num_steps: Euler steps.
            cfg_scale: Classifier-free guidance scale.
            guidance_mode: ``"cfg"``, ``"none"``, ``"ig"`` or ``"cfg_ig"``.
            seed: Optional manual seed for the initial noise.

        Returns:
            Raw (destandardized) latents ``[total_views, C, h, w]`` ready for
            :meth:`decode_rgb` / :meth:`decode_geometry`.
        """
        if self.flow is None:
            raise RuntimeError("sample() requires a flow model; pass flow_cfg to from_configs()")
        if z_ref.ndim != 4:
            raise ValueError(f"z_ref must be [cond_num, C, h, w]; got {tuple(z_ref.shape)}")
        if total_views < cond_num:
            raise ValueError(f"total_views ({total_views}) < cond_num ({cond_num})")

        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from eval_generation import sample_v4_euler  # heavy import, done lazily

        device = next(self.flow.parameters()).device
        z_ref_std = self.standardize(z_ref.to(device))
        if seed is not None:
            torch.manual_seed(int(seed))

        z_std = sample_v4_euler(
            self.flow,
            z_ref_clean=z_ref_std,
            total_view=total_views,
            cond_num=cond_num,
            plucker_6d=plucker_6d,
            ref_global=ref_global,
            cfg_uncond_ref_global=cfg_uncond_ref_global,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            guidance_mode=guidance_mode,
            time_dist_shift=getattr(self, "_time_dist_shift", None) or 1.0,
            prediction=getattr(self, "_prediction", "x"),
        )
        return self.destandardize(z_std)

    @torch.inference_mode()
    def save_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        out_dir,
        stem: str = "gae",
        *,
        pc_stride: int = 4,
    ) -> Dict[str, str]:
        """Write RGB PNGs, a DPT depth visualization, and a ``.ply`` point cloud.

        ``outputs`` is the dict from :meth:`reconstruct` or ``decode_rgb`` +
        :meth:`decode_geometry`` (needs ``rgb``, ``depth``, and ``ray``).
        """
        from PIL import Image
        import numpy as np

        scripts_dir = str(_REPO_ROOT / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from eval_data import depth_to_numpy_img, depth_to_numpy_video
        from eval_generation import _scene_pointcloud_from_dpt, save_pointcloud_ply

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rgb = outputs["rgb"]
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        v = rgb.shape[0]
        h, w = int(rgb.shape[-2]), int(rgb.shape[-1])
        written: Dict[str, str] = {}
        for i in range(v):
            arr = (rgb[i].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                   * 255.0 + 0.5).astype("uint8")
            p = out_dir / f"{stem}_{i:02d}.png"
            Image.fromarray(arr).save(p)
            written[f"rgb_{i}"] = str(p)
        depth = outputs.get("depth")
        if depth is not None:
            dep_path = out_dir / f"{stem}_depth.png"
            depth_frames = depth_to_numpy_video(depth[:v])
            Image.fromarray(depth_frames[0]).save(dep_path)
            written["depth"] = str(dep_path)
            if v > 1:
                rows = depth_frames
                strip = np.concatenate(rows, axis=1)
                strip_path = out_dir / f"{stem}_depth_strip.png"
                Image.fromarray(strip).save(strip_path)
                written["depth_strip"] = str(strip_path)
        if depth is not None and outputs.get("ray") is not None:
            xyz, pc_rgb = _scene_pointcloud_from_dpt(
                outputs, depth, rgb, v, h, w, rgb.device, stride=pc_stride,
            )
            ply = out_dir / f"{stem}_pointcloud.ply"
            save_pointcloud_ply(str(ply), xyz, pc_rgb)
            written["ply"] = str(ply)
            written["n_points"] = str(xyz.shape[0])
        return written

    @torch.inference_mode()
    def reconstruct(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode real frames and decode both RGB and geometry back out."""
        b, v = images.shape[:2]
        h, w = images.shape[-2:]
        z = self.encode(images).flatten(0, 1)
        out = {"rgb": self.decode_rgb(z, num_views=v)}
        out.update(self.decode_geometry(z, h, w))
        return out
