"""Flow matching in the GAE latent — paper Section 3.2 (Eqs. 10-11).

The network predicts the *clean* latent from the noisy state (RAEv2-style
x-prediction); :func:`convert_x_to_v` turns that prediction into the velocity
the loss and sampler need (Eq. 11). Time is sampled from a shifted
logit-normal whose shift is derived from the latent dimensionality,
``shift = sqrt(latent_numel / 4096)``, and is shared across all views in a
clip.

Also holds :func:`repa_guidance_step`, the sampling-time composition of
Internal Guidance and classifier-free guidance.

Adapted from RAEv2 ``src/stage2/transport/transport.py`` to GLD's interface:

  * Time sampler: shifted logit-normal. Shift derived from latent dim (RAEv2
    Eq. uses ``shift = sqrt(latent_numel / 4096)``); see
    ``[/tmp/RAEv2/src/train.py](/tmp/RAEv2/src/train.py)`` lines 179–181.
  * Loss: model outputs x-prediction. Target velocity is recovered by
    ``v_pred = (xt - x_pred) / t.clamp_min(t_eps)``; loss is MSE in v-space
    (this is the standard "v-pred space loss with x-pred output head"
    formulation — keeps loss landscape identical to velocity training while
    letting the head output a quantity that lives in the latent's own
    metric space, which is what makes Internal Guidance work).
  * Dual-loss: when the model returns a dict with ``base`` (IG baseline x-pred),
    add ``base_model_coeff * MSE(v_base, vt)``. With ``base_final_layer``
    zero-initialized this term starts at 0 and is learned smoothly.
  * REPA loss: cosine alignment between the DiT's intermediate token
    projection (``out["repa"]``) and a frozen target ``z_repa_target``
    supplied by the trainer (decoded DA3 features, normalized).
  * CFG dropout: trainer is responsible (it sets ``ref_global=None`` with
    probability ``text_drop_prob``). This module does not touch conditioning.

The only call point is :func:`training_losses_rae`. No wrapper class — keeps
the surface minimal.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def shift_from_latent_dim(
    latent_numel: int, base: int = 4096
) -> float:
    """Resolution-aware time-shift used by RAEv2 (DiT_DDT scale-aware schedule).

    Larger sequences need to spend more time on small-t steps where the
    signal-to-noise ratio is harder to learn. The shift constant is
    ``sqrt(N / base)`` where ``N`` is the total per-sample latent element
    count; ``base=4096`` matches RAEv2's reference (16x16 ImageNet at
    1024 ch ≈ 262k for their setup, but they pin ``time_dist_shift_dim``
    independently in config — see RAEv2 train.py).
    """
    return math.sqrt(max(latent_numel, base) / float(base))


def sample_logit_normal_t(
    batch_size: int,
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
    shift: float = 1.0,
    eps: float = 1e-3,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample t in [0,1] from logit-normal(mu, sigma) then apply scale-aware shift.

    Shift warp (RAEv2 ``Transport.sample``): ``t' = shift * t / (1 + (shift-1)*t)``.
    Identity at shift=1; pushes mass toward t→0 for shift>1.
    """
    z = torch.randn(batch_size, device=device, dtype=dtype) * sigma + mu
    t = torch.sigmoid(z)
    if shift != 1.0:
        t = shift * t / (1.0 + (shift - 1.0) * t)
    return t.clamp(eps, 1.0 - eps)


def _expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar/view or spatial-token timesteps over latent channels."""
    if t.ndim == 1:
        return t.view(t.size(0), *([1] * (x.dim() - 1)))
    # Self-Flow timestep maps are (BV,H,W), one noise level per latent token.
    if t.ndim == x.ndim - 1:
        if t.shape[0] != x.shape[0] or t.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"Per-token timestep shape {tuple(t.shape)} incompatible with "
                f"latent shape {tuple(x.shape)}"
            )
        return t.unsqueeze(1)
    if t.ndim == x.ndim:
        return t
    raise ValueError(f"Unsupported timestep shape {tuple(t.shape)} for latent {tuple(x.shape)}")


def compute_rgb_aux_loss(
    x_pred: torch.Tensor,
    t: torch.Tensor,
    *,
    rgb_decoder,
    rgb_target: torch.Tensor,
    denormalize_latent_fn,
    total_view: int,
    rgb_t_max: float = 0.4,
    rgb_max_views: int = 0,
) -> torch.Tensor:
    """L1 RGB reconstruction loss through a frozen VAE RGBHead.

    Gradients flow into ``x_pred`` (the model's x-prediction) while the decoder
    stays frozen. Only tokens with ``t < rgb_t_max`` contribute — low-noise
    steps where pixel fidelity matters most.

    Args:
        x_pred: (BV, C, H, W) whitened latent prediction from the DiT head.
        t: (BV,) noise levels in [0, 1].
        rgb_decoder: callable ``z_raw -> rgb`` (typically ``vae.decode_rgb``).
        rgb_target: (BV, 3, H_px, W_px) in [0, 1].
        denormalize_latent_fn: ``y_whitened -> z_raw``.
        total_view: V views per batch element.
        rgb_t_max: apply loss only where ``t < rgb_t_max``.
        rgb_max_views: if > 0 and < V, randomly subsample this many views per
            batch element to cap decoder cost.
    """
    if rgb_decoder is None or rgb_target is None:
        return x_pred.new_zeros(())

    BV = x_pred.shape[0]
    assert BV % total_view == 0, (BV, total_view)
    B = BV // total_view
    device = x_pred.device

    # RGB decoding is selected per view.  With Self-Flow, reduce the
    # per-token schedule to its view mean for this optional auxiliary loss.
    view_t = t if t.ndim == 1 else t.reshape(t.shape[0], -1).mean(dim=1)
    low_t = view_t < rgb_t_max
    if not low_t.any():
        return x_pred.new_zeros(())

    view_mask = torch.ones(B, total_view, dtype=torch.bool, device=device)
    if rgb_max_views > 0 and rgb_max_views < total_view:
        view_mask.zero_()
        for b in range(B):
            pick = torch.randperm(total_view, device=device)[:rgb_max_views]
            view_mask[b, pick] = True

    token_mask = low_t & view_mask.reshape(B * total_view)

    x_sel = x_pred[token_mask]
    tgt_sel = rgb_target[token_mask]
    if x_sel.shape[0] == 0:
        return x_pred.new_zeros(())

    z_raw = denormalize_latent_fn(x_sel)
    rgb_pred = rgb_decoder(z_raw.float())
    return F.l1_loss(rgb_pred, tgt_sel)


def convert_x_to_v(
    x_pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor, t_eps: float = 0.05
) -> torch.Tensor:
    """x-prediction -> velocity — paper Eq. (11): v = (z_t - z_pred) / max(t, t_eps).

    Matches RAEv2's ``Transport.convert_model_pred`` for ``prediction='x'``.
    """
    t_safe = _expand_t(t, xt).clamp_min(t_eps)
    return (xt - x_pred) / t_safe


def training_losses_rae(
    model,
    *,
    x_target: torch.Tensor,
    total_view: int,
    time_dist_shift: float = 1.0,
    time_dist_type: str = "logit-normal_0_1",
    t_eps_loss: float = 0.05,
    prediction: str = "x",
    base_model_coeff: float = 1.0,
    repa_coeff: Optional[float] = 0.5,
    repa_target: Optional[torch.Tensor] = None,
    z_ref_clean: Optional[torch.Tensor] = None,
    reference_conditioning: str = "clean_token_v4",
    plucker_6d: Optional[torch.Tensor] = None,
    ref_global: Optional[torch.Tensor] = None,
    cond_num: int = 0,
    motion_loss_strength: float = 0.0,
    motion_loss_max_multiplier: float = 4.0,
    extra_model_kwargs: Optional[dict] = None,
    rgb_decoder=None,
    rgb_target: Optional[torch.Tensor] = None,
    denormalize_latent_fn=None,
    rgb_loss_weight: float = 0.0,
    rgb_t_max: float = 0.4,
    rgb_max_views: int = 0,
    geo_loss_weight: float = 0.0,
    geo_t_max: float = 0.4,
    geo_max_views: int = 0,
    geo_ray_weight: float = 1.0,
    geo_depth_weight: float = 1.0,
    geo_reproj_weight: float = 0.0,
    geo_reproj_stride: int = 8,
    gt_ray: Optional[torch.Tensor] = None,
    gt_depth: Optional[torch.Tensor] = None,
    gt_ray_conf: Optional[torch.Tensor] = None,
    geo_c2w: Optional[torch.Tensor] = None,
    geo_K: Optional[torch.Tensor] = None,
    geo_decode_fn=None,
) -> dict:
    """Single training step loss for the RAE recipe.

    Args:
        model: ``GAEFlow`` (or any module accepting V4-style kwargs +
            ``return_dict=True``). Must produce dict {"main", "base", "repa"}.
        x_target: (BV, C, H, W) clean latent (post-whitening).
        total_view: V (the trainer flattens BxV into the leading axis).
        time_dist_shift: see :func:`shift_from_latent_dim`. Pass 1.0 for plain
            uniform/logit-normal sampling.
        time_dist_type: only ``logit-normal_MU_SIGMA`` and ``uniform`` are
            recognized; anything else raises (failing loud is better than a
            silent fallback because misconfigured noise schedule shifts FID
            invisibly).
        t_eps_loss: clamp on ``t`` when converting x→v to avoid division blowup.
            RAEv2 uses 0.05; values <0.05 destabilize bf16 training. Only used
            when ``prediction='x'``; ignored for ``'velocity'``.
        prediction: ``'x'`` (RAEv2 default — model emits x_pred, convert to v
            for loss) or ``'velocity'`` (plain flow-matching — model emits v
            directly, MSE in v-space without the 1/t reweighting). Must match
            the model's ``prediction_type`` semantics and the sampler's
            ``convert_model_pred``.
        base_model_coeff: weight on IG baseline loss. Set to 0 to disable
            (e.g. when warm-starting from a velocity-only ckpt and you want
            the base head to ramp up gradually).
        repa_coeff: weight on REPA cosine loss. ``None`` disables. Has no
            effect if model.repa_pred is None or ``repa_target`` is None.
        repa_target: (BV, T, D) frozen target for REPA cosine align. Trainer
            is responsible for constructing this (decode → pool → flatten).
        z_ref_clean / plucker_6d / ref_global / cond_num: reference, camera,
            and text conditioning. ``reference_conditioning='clean_token_v4'``
            prepends clean encoder tokens; ``'state_v3'`` clamps clean refs into
            the first state slots and supervises target slots only.
        motion_loss_strength: strength of GT adjacent-latent motion weighting.
            Zero preserves the original uniform MSE. Positive values increase
            the contribution of changing latent locations while normalizing the
            supervised weight map to mean one.
        motion_loss_max_multiplier: maximum raw token multiplier before the
            mean-one normalization. Must be at least one.
        extra_model_kwargs: pass-through for new V4 args we don't enumerate.

    Returns:
        dict with scalar tensors ``loss`` (total backward target),
        ``loss_main``, ``loss_base``, ``loss_repa``, plus ``t_mean`` for logging.
    """
    BV = x_target.shape[0]
    assert BV % total_view == 0, (BV, total_view)
    B = BV // total_view
    device, dtype = x_target.device, x_target.dtype
    if reference_conditioning not in ("clean_token_v4", "state_v3"):
        raise ValueError(
            "reference_conditioning must be 'clean_token_v4' or 'state_v3'; "
            f"got {reference_conditioning!r}"
        )
    state_v3 = reference_conditioning == "state_v3"
    if state_v3 and cond_num > 0 and z_ref_clean is None:
        raise ValueError("state_v3 requires z_ref_clean when cond_num > 0")

    def _sample_batch_t() -> torch.Tensor:
        if time_dist_type.startswith("logit-normal"):
            parts = time_dist_type.split("_")
            if len(parts) == 3:
                mu, sigma = float(parts[1]), float(parts[2])
            else:
                mu, sigma = 0.0, 1.0
            return sample_logit_normal_t(
                B, mu=mu, sigma=sigma, shift=time_dist_shift,
                device=device, dtype=dtype,
            )
        if time_dist_type == "uniform":
            u = torch.rand(B, device=device, dtype=dtype)
            if time_dist_shift != 1.0:
                u = time_dist_shift * u / (1.0 + (time_dist_shift - 1.0) * u)
            return u.clamp(1e-3, 1.0 - 1e-3)
        raise ValueError(f"Unsupported time_dist_type: {time_dist_type!r}")

    t_b = _sample_batch_t()
    noise = torch.randn_like(x_target)
    # Shared t across all views in a clip (paper Section 3.2).
    t = t_b.unsqueeze(1).expand(B, total_view).reshape(BV)

    t_view = _expand_t(t, x_target)
    # RAEv2 convention: x_t = (1-t) * x_clean + t * noise; v_target = noise - x_clean
    xt = (1.0 - t_view) * x_target + t_view * noise
    v_target = noise - x_target

    model_kwargs = dict(extra_model_kwargs or {})
    model_z_ref_clean = z_ref_clean
    if state_v3 and cond_num > 0:
        assert z_ref_clean is not None
        xt_5d = xt.reshape(B, total_view, *xt.shape[1:])
        z_ref_5d = z_ref_clean.reshape(B, cond_num, *z_ref_clean.shape[1:])
        xt_5d[:, :cond_num] = z_ref_5d
        xt = xt_5d.reshape(BV, *xt.shape[1:])
        if t.ndim == 1:
            t_state = t.reshape(B, total_view)
            t_state[:, :cond_num] = 0.0
            t = t_state.reshape(BV)
        else:
            # Self-Flow supplies one timestep per spatial token.
            t_state = t.reshape(B, total_view, *t.shape[1:])
            t_state[:, :cond_num] = 0.0
            t = t_state.reshape(BV, *t.shape[1:])
        model_z_ref_clean = None
        model_kwargs["reference_in_state"] = True
    out = model(
        xt,
        t,
        total_view,
        z_ref_clean=model_z_ref_clean,
        plucker_6d=plucker_6d,
        ref_global=ref_global,
        cond_num=cond_num,
        return_dict=True,
        **model_kwargs,
    )

    if prediction == "x":
        v_pred_main = convert_x_to_v(out["main"], xt, t, t_eps=t_eps_loss)
    elif prediction == "velocity":
        v_pred_main = out["main"]
    else:
        raise ValueError(f"Unsupported prediction: {prediction!r} (expected 'x' or 'velocity')")

    motion_weight = None
    motion_score_mean = x_target.new_zeros(())
    motion_weight_mean = x_target.new_ones(())
    motion_weight_max = x_target.new_ones(())
    if motion_loss_strength > 0.0 and total_view > 1:
        if motion_loss_max_multiplier < 1.0:
            raise ValueError(
                "motion_loss_max_multiplier must be >= 1, got "
                f"{motion_loss_max_multiplier}"
            )
        target_5d = x_target.reshape(B, total_view, *x_target.shape[1:])
        # [B, V, 1, H, W]. Weight is derived only from GT clean latents.
        motion_score = target_5d.new_zeros(
            B, total_view, 1, target_5d.shape[-2], target_5d.shape[-1]
        )
        motion_score[:, 1:] = (
            target_5d[:, 1:] - target_5d[:, :-1]
        ).abs().mean(dim=2, keepdim=True)
        score_for_norm = motion_score[:, cond_num:] if cond_num < total_view else motion_score
        score_scale = score_for_norm.mean(dim=(1, 2, 3, 4), keepdim=True)
        normalized_score = motion_score / score_scale.clamp_min(1e-6)
        motion_weight = 1.0 + motion_loss_strength * normalized_score
        motion_weight = motion_weight.clamp(max=motion_loss_max_multiplier)
        supervised_weight = (
            motion_weight[:, cond_num:]
            if state_v3 and cond_num > 0
            else motion_weight
        )
        weight_scale = supervised_weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        motion_weight = motion_weight / weight_scale.clamp_min(1e-6)
        supervised_weight = (
            motion_weight[:, cond_num:]
            if state_v3 and cond_num > 0
            else motion_weight
        )
        motion_score_mean = score_for_norm.detach().mean()
        motion_weight_mean = supervised_weight.detach().mean()
        motion_weight_max = supervised_weight.detach().amax()

    def _diffusion_mse(pred: torch.Tensor) -> torch.Tensor:
        error = (pred - v_target).square().reshape(
            B, total_view, *pred.shape[1:]
        )
        weight = motion_weight
        if state_v3 and cond_num > 0:
            error = error[:, cond_num:]
            if weight is not None:
                weight = weight[:, cond_num:]
        if weight is not None:
            error = error * weight
        return error.mean()

    loss_main = _diffusion_mse(v_pred_main)

    if out.get("base") is not None and base_model_coeff > 0.0:
        if prediction == "x":
            v_pred_base = convert_x_to_v(out["base"], xt, t, t_eps=t_eps_loss)
        else:
            v_pred_base = out["base"]
        loss_base = _diffusion_mse(v_pred_base)
    else:
        loss_base = x_target.new_zeros(())

    if (
        repa_coeff is not None
        and repa_coeff > 0.0
        and out.get("repa") is not None
        and repa_target is not None
    ):
        # Cosine similarity loss in normalized feature space.
        pred = F.normalize(out["repa"].float(), dim=-1)
        tgt = F.normalize(repa_target.float(), dim=-1)
        loss_repa = (1.0 - (pred * tgt).sum(dim=-1)).mean()
    else:
        loss_repa = x_target.new_zeros(())

    loss_rgb = x_target.new_zeros(())
    if rgb_loss_weight > 0.0 and prediction == "x":
        if rgb_target is None:
            raise ValueError("rgb_target is required when RGB auxiliary loss is enabled")
        loss_rgb = compute_rgb_aux_loss(
            out["main"],
            t,
            rgb_decoder=rgb_decoder,
            rgb_target=rgb_target,
            denormalize_latent_fn=denormalize_latent_fn,
            total_view=total_view,
            rgb_t_max=rgb_t_max,
            rgb_max_views=rgb_max_views,
        )

    total = loss_main + base_model_coeff * loss_base
    if repa_coeff is not None:
        total = total + repa_coeff * loss_repa
    if rgb_loss_weight > 0.0:
        total = total + rgb_loss_weight * loss_rgb

    loss_geo = x_target.new_zeros(())
    if geo_loss_weight > 0.0 and prediction == "x":
        if gt_ray is None or geo_decode_fn is None or denormalize_latent_fn is None:
            raise ValueError(
                "gt_ray and geo_decode_fn are required when geo auxiliary loss is enabled"
            )
        from utils.geo_aux_loss import compute_geo_aux_loss

        loss_geo = compute_geo_aux_loss(
            out["main"],
            t,
            gt_ray=gt_ray,
            gt_depth=gt_depth,
            gt_ray_conf=gt_ray_conf,
            denormalize_latent_fn=denormalize_latent_fn,
            geo_decode_fn=geo_decode_fn,
            total_view=total_view,
            geo_t_max=geo_t_max,
            geo_max_views=geo_max_views,
            ray_weight=geo_ray_weight,
            depth_weight=geo_depth_weight,
            reproj_weight=geo_reproj_weight,
            reproj_stride=geo_reproj_stride,
            c2w=geo_c2w,
            K=geo_K,
        )

    if geo_loss_weight > 0.0:
        total = total + geo_loss_weight * loss_geo

    return {
        "loss": total,
        "loss_main": loss_main.detach(),
        "loss_base": loss_base.detach(),
        "loss_repa": loss_repa.detach(),
        "loss_rgb": loss_rgb.detach(),
        "loss_geo": loss_geo.detach(),
        "motion_score_mean": motion_score_mean.detach(),
        "motion_weight_mean": motion_weight_mean.detach(),
        "motion_weight_max": motion_weight_max.detach(),
        "t_mean": t.detach().mean(),
    }


def repa_guidance_step(
    model,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    total_view: int,
    z_ref_clean: Optional[torch.Tensor],
    plucker_6d: Optional[torch.Tensor],
    ref_global: Optional[torch.Tensor],
    cond_num: int,
    ig_scale: float,
    cfg_scale: float = 1.0,
    cfg_uncond_ref_global: Optional[torch.Tensor] = None,
    t_eps: float = 0.05,
    extra_model_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Sampling-time forward that returns a v-pred suitable for an Euler step.

    Implements RAEv2's "internal guidance" composition (Eq. ``x_guided =
    x_full + w*(x_full - x_base)``) operating in x-space, then converts back
    to v. Optionally combines with classical CFG by scaling between the
    text-conditional and the null-conditional output.

    Args:
        ig_scale: weight on (x_full - x_base). 0.0 disables IG.
        cfg_scale: weight on (x_cond - x_uncond). 1.0 disables CFG.
        cfg_uncond_ref_global: pass the null/empty caption embedding when
            CFG is enabled; otherwise ignored.

    Returns velocity prediction (BV, C, H, W).
    """
    model_kwargs = dict(extra_model_kwargs or {})
    out_cond = model(
        x_t, t, total_view,
        z_ref_clean=z_ref_clean,
        plucker_6d=plucker_6d,
        ref_global=ref_global,
        cond_num=cond_num,
        return_dict=True,
        **model_kwargs,
    )
    x_full = out_cond["main"]

    if ig_scale != 0.0 and out_cond.get("base") is not None:
        x_base = out_cond["base"]
        x_full = x_full + ig_scale * (x_full - x_base)

    if cfg_scale != 1.0:
        out_uncond = model(
            x_t, t, total_view,
            z_ref_clean=z_ref_clean,
            plucker_6d=plucker_6d,
            ref_global=cfg_uncond_ref_global,
            cond_num=cond_num,
            return_dict=True,
            **model_kwargs,
        )
        x_uncond = out_uncond["main"]
        x_full = x_uncond + cfg_scale * (x_full - x_uncond)

    return convert_x_to_v(x_full, x_t, t, t_eps=t_eps)
