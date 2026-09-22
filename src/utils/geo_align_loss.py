"""Geometry-alignment losses for the geometry-decoder finetune (Step 4).

Unlike ``geo_aux_loss`` (which supervises DiT x-pred against GT-video DPT), this
module aligns the STUDENT geometry decode of a fixed latent to a FROZEN teacher
geometry that was cached BEFORE finetuning. The student is only
``TemporalGeoAdapter + dec_conv``; DPT, shared trunk, RGB head, encoder and DiT
stay frozen. Targets never flow gradients (all detached / precomputed).

Semantics of the three latent inputs (all raw VAE-encoder-space latents):
  z_pred   : DiT sample            -> aligned to teacher = G_old(z_regen)
  z_regen  : VAE(DA3(pred RGB))    -> identity to G_old(z_regen)
  z_clean  : VAE(DA3(GT RGB))      -> identity to G_old(z_clean)

Depth is compared in log space with a single scene-wide scale alignment (never
per-view median) so genuine cross-frame scale drift is still penalised. All
depth/ray terms are confidence-weighted by the (detached) teacher ray_conf.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _as_vhw(depth: torch.Tensor) -> torch.Tensor:
    """Coerce depth to (V, H, W)."""
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    if depth.ndim != 3:
        raise ValueError(f"expected depth (V,H,W), got {tuple(depth.shape)}")
    return depth


def _conf_weight(conf: Optional[torch.Tensor], ref: torch.Tensor,
                 already_prob: bool = False) -> torch.Tensor:
    """Return a broadcastable non-negative weight for a ``(V,H,W,C?)`` ref.

    ``conf`` is the DA3 ray confidence, typically ``(1,V,H,W)`` or ``(V,H,W)``
    at the ray resolution. It gates only the ray term (depth/point use uniform
    weights) so ``ref`` here is always the ray tensor. Confidence is a logit
    unless ``already_prob``; a leading singleton batch dim is squeezed and a
    trailing channel dim is added to broadcast over the ray's 6 channels.
    """
    if conf is None:
        return torch.ones_like(ref)
    w = conf.detach().float()
    # squeeze a leading singleton batch dim: (1,V,H,W) -> (V,H,W)
    while w.ndim > 3 and w.shape[0] == 1:
        w = w[0]
    if not already_prob:
        w = w.sigmoid()
    w = w.to(ref.dtype)
    while w.ndim < ref.ndim:
        w = w.unsqueeze(-1)
    return w.clamp_min(0.0)


def _weighted_huber(pred: torch.Tensor, target: torch.Tensor,
                    weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    diff = pred - target
    absd = diff.abs()
    quad = torch.minimum(absd, absd.new_tensor(delta))
    lin = absd - quad
    per = 0.5 * quad * quad + delta * lin
    w = weight.expand_as(per)
    return (w * per).sum() / (w.sum() + 1e-8)


def robust_log_depth_loss(
    pred_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    *,
    scene_scale_align: bool = True,
    delta: float = 1.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Confidence-weighted Huber on log-depth with ONE scene-wide log-scale.

    A single scalar log-shift is solved (confidence-weighted mean of the
    log-ratio) so an overall metric-scale ambiguity between student and teacher
    is removed, while per-view / cross-frame scale drift is preserved and
    therefore penalised.
    """
    pd = _as_vhw(pred_depth).float().clamp_min(eps)
    td = _as_vhw(teacher_depth).float().clamp_min(eps)
    lp, lt = pd.log(), td.log()
    w = _conf_weight(conf, lt)
    if scene_scale_align:
        shift = (w * (lt - lp)).sum() / (w.sum() + 1e-8)
        lp = lp + shift.detach()
    return _weighted_huber(lp, lt, w, delta=delta)


def ray_loss(
    pred_ray: torch.Tensor,
    teacher_ray: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    *,
    delta: float = 1.0,
    direction_split: bool = False,
) -> torch.Tensor:
    """Confidence-weighted Huber on the 6D camera ray.

    ``direction_split`` additionally adds a direction cosine term on the last 3
    channels (treated as ray direction). Kept OFF by default until the DA3 ray
    channel convention is confirmed (avoids penalising a mislabelled axis).
    """
    pr = pred_ray.float()
    tr = teacher_ray.float()
    w = _conf_weight(conf, tr)
    loss = _weighted_huber(pr, tr, w, delta=delta)
    if direction_split and pr.shape[-1] == 6:
        pd = F.normalize(pr[..., 3:], dim=-1)
        tdir = F.normalize(tr[..., 3:], dim=-1)
        cos = (pd * tdir).sum(dim=-1, keepdim=True)
        wc = _conf_weight(conf, cos)
        loss = loss + (wc * (1.0 - cos)).sum() / (wc.sum() + 1e-8)
    return loss


def _backproject(depth_vhw: torch.Tensor, c2w: torch.Tensor,
                 K: torch.Tensor) -> torch.Tensor:
    """(V,H,W) depth + metric cameras -> world points (V,H,W,3)."""
    v, h, w = depth_vhw.shape
    ys = torch.arange(h, device=depth_vhw.device, dtype=depth_vhw.dtype)
    xs = torch.arange(w, device=depth_vhw.device, dtype=depth_vhw.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pix = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(-1, 3)
    pts = []
    for i in range(v):
        rays = pix @ torch.linalg.inv(K[i].float()).T
        cam = rays * depth_vhw[i].reshape(-1, 1)
        world = cam @ c2w[i, :3, :3].float().T + c2w[i, :3, 3].float()
        pts.append(world.reshape(h, w, 3))
    return torch.stack(pts, dim=0)


def point_map_loss(
    pred_depth: torch.Tensor,
    teacher_depth: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    *,
    scene_scale_align: bool = True,
    delta: float = 0.5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Robust world-point L1 between student/teacher under the SAME cameras.

    Uses one shared scene-wide depth scale (same convention as log-depth) so the
    point map is compared up to global metric scale, not per-view.
    """
    pd = _as_vhw(pred_depth).float().clamp_min(eps)
    td = _as_vhw(teacher_depth).float().clamp_min(eps)
    if scene_scale_align:
        w2 = _conf_weight(conf, td)
        ratio = (w2 * (td / pd)).sum() / (w2 * torch.ones_like(td)).sum().clamp_min(1e-8)
        pd = pd * ratio.detach()
    pp = _backproject(pd, c2w, K)
    tp = _backproject(td, c2w, K)
    w = _conf_weight(conf, pp[..., 0]).unsqueeze(-1)
    return _weighted_huber(pp, tp, w, delta=delta)


def depth_scale_drift_loss(
    pred_depth: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Penalise cross-view spread of per-view mean log-depth (scale drift)."""
    pd = _as_vhw(pred_depth).float().clamp_min(eps).log()
    if conf is not None:
        w = _conf_weight(conf, pd)
        per_view = (w * pd).sum(dim=(1, 2)) / (w.sum(dim=(1, 2)) + 1e-8)
    else:
        per_view = pd.mean(dim=(1, 2))
    if per_view.numel() < 2:
        return pd.new_zeros(())
    return per_view.var(unbiased=False)


def adapter_residual_loss(z_in: torch.Tensor, z_out: torch.Tensor) -> torch.Tensor:
    """Keep the adapter residual small (regulariser)."""
    return (z_out - z_in).pow(2).mean()


def _solve_scene_scale(
    pred: torch.Tensor, target: torch.Tensor,
    weight: Optional[torch.Tensor] = None, *, eps: float = 1e-8,
) -> torch.Tensor:
    """Detached scalar ``s`` minimising ``||w·(s·pred - target)||`` (LS ratio)."""
    p = pred.float()
    t = target.float()
    if weight is not None:
        w = weight.float().expand_as(p)
        num = (w * p * t).sum()
        den = (w * p * p).sum().clamp_min(eps)
    else:
        num = (p * t).sum()
        den = (p * p).sum().clamp_min(eps)
    return (num / den).detach()


def _hi_conf_mask(conf: Optional[torch.Tensor],
                  percentile: float) -> Optional[torch.Tensor]:
    """Per-scene hard mask keeping the top ``(100-percentile)%`` RAW confidence.

    Returns a float {0,1} mask of shape ``(V,Hc,Wc)`` (leading singleton batch
    squeezed), or ``None`` when disabled / unavailable. Uses the RAW confidence
    (NOT sigmoid): DA3 ray_conf logits sit around 8–80, so sigmoid saturates to
    ~1 and destroys the ranking — thresholding the raw value is what actually
    isolates the high-confidence pixels.
    """
    if conf is None or percentile is None or percentile <= 0:
        return None
    c = conf.detach().float()
    while c.ndim > 3 and c.shape[0] == 1:
        c = c[0]
    finite = torch.isfinite(c)
    if not bool(finite.any()):
        return None
    thr = torch.quantile(c[finite], float(percentile) / 100.0)
    return (c >= thr).to(c.dtype)


def _depth_to_normal(depth: torch.Tensor) -> torch.Tensor:
    """(V,H,W) depth → (V,H,W,3) unit surface normal from central differences.

    Treats depth as a height field: n ∝ (-dz/dx, -dz/dy, 1). Borders use 0
    gradient (normal = +z), so both student/teacher agree there (no spurious
    loss). Direction depends on absolute depth scale, so callers should pass the
    already scale-aligned student depth to compare like-for-like with teacher.
    """
    d = depth
    dx = torch.zeros_like(d)
    dy = torch.zeros_like(d)
    dx[:, :, 1:-1] = (d[:, :, 2:] - d[:, :, :-2]) * 0.5
    dy[:, 1:-1, :] = (d[:, 2:, :] - d[:, :-2, :]) * 0.5
    n = torch.stack([-dx, -dy, torch.ones_like(d)], dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-6)


def _ray_vhwc(ray: torch.Tensor) -> torch.Tensor:
    """Coerce a camera-ray tensor to ``(V,H,W,6)``."""
    if ray.ndim == 5:  # (B,V,H,W,6) or (V,6?,H,W) edge cases
        ray = ray[0]
    if ray.ndim == 4 and ray.shape[1] == 6 and ray.shape[-1] != 6:
        ray = ray.permute(0, 2, 3, 1).contiguous()
    if ray.ndim != 4 or ray.shape[-1] != 6:
        raise ValueError(f"expected ray (V,H,W,6), got {tuple(ray.shape)}")
    return ray


def _points_from_ray_depth(depth: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    """Differentiable self-posed world points from a side's own depth + ray.

    DualDPT ray is ``(V,H,W,6)``: ``ray[...,:3] = R·K⁻¹[u,v,1]`` (world-frame
    pixel ray, camera-frame Z=1) and ``ray[...,3:]`` = per-view camera center in
    that same world frame. Z-depth backprojection is therefore exactly

        P = ray[...,3:] + depth · ray[...,:3]

    (validated against ``recover_poses``+backproject to ~0.4% of scene extent).
    Grad flows into BOTH depth and ray. Depth is bilinear-resized to the ray
    resolution when they differ (ray is typically lower-res, e.g. 288 vs 504).
    NOTE: never routes through ``recover_poses`` (non-differentiable).
    """
    ray = _ray_vhwc(ray).float()
    d = _as_vhw(depth).float()
    V, Hr, Wr, _ = ray.shape
    if d.shape[-2:] != (Hr, Wr):
        d = F.interpolate(d.unsqueeze(1), size=(Hr, Wr), mode="bilinear",
                          align_corners=False).squeeze(1)
    return ray[..., 3:] + d.unsqueeze(-1) * ray[..., :3]


def _normal_from_points(P: torch.Tensor):
    """(V,H,W,3) world points → (unit normal, valid mask) via cross product.

    Uses central differences of the 3D points (not pixel-space depth slopes,
    which collapse to +Z because they ignore the focal-length metric scale).
    Normal is normalized by the clamped cross-product magnitude (so identical
    inputs give an exact unit vector, ``cos=1``). Border pixels (one-sided diff
    missing) and near-degenerate cross products are reported invalid so callers
    can exclude them; the threshold scales with the per-tensor median magnitude
    to stay robust across scene scales.
    """
    dx = torch.zeros_like(P)
    dy = torch.zeros_like(P)
    dx[:, :, 1:-1] = P[:, :, 2:] - P[:, :, :-2]
    dy[:, 1:-1, :] = P[:, 2:, :] - P[:, :-2, :]
    n = torch.cross(dx, dy, dim=-1)
    mag = n.norm(dim=-1, keepdim=True)
    unit = n / mag.clamp_min(1e-12)
    pos = mag[mag > 0]
    thr = 1e-3 * pos.median() if pos.numel() > 0 else mag.new_tensor(0.0)
    valid = (mag.squeeze(-1) > thr)
    return unit, valid


def _umeyama_sim3(Ps: torch.Tensor, Pt: torch.Tensor, *, eps: float = 1e-8):
    """Detached closed-form Sim3 (s,R,t) minimising ``||s·R·Ps + t − Pt||``.

    ``Ps``/``Pt`` are ``(N,3)`` corresponding points. Solved under ``no_grad``
    (gauge only). Returns ``(s, R, t)`` or ``None`` when degenerate/non-finite.
    """
    with torch.no_grad():
        X = Ps.detach().double()
        Y = Pt.detach().double()
        if X.shape[0] < 3 or not (torch.isfinite(X).all() and torch.isfinite(Y).all()):
            return None
        mx = X.mean(0)
        my = Y.mean(0)
        Xc = X - mx
        Yc = Y - my
        var_x = (Xc * Xc).sum() / X.shape[0]
        if not torch.isfinite(var_x) or var_x < eps:
            return None
        cov = (Xc.T @ Yc) / X.shape[0]
        try:
            U, S, Vt = torch.linalg.svd(cov)
        except Exception:  # noqa: BLE001
            return None
        D = torch.eye(3, dtype=X.dtype, device=X.device)
        if torch.det(U @ Vt) < 0:
            D[2, 2] = -1.0
        R = (U @ D @ Vt).T
        s = (torch.diag(S) @ D).trace() / var_x
        if not torch.isfinite(s) or s <= 0:
            return None
        t = my - s * (R @ mx)
        return s.float(), R.float(), t.float()


def self_pose_point_loss(
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    *,
    stride: int = 2,
    conf: Optional[torch.Tensor] = None,
    conf_percentile: float = 50.0,
) -> Optional[torch.Tensor]:
    """Point-cloud alignment loss with each side self-posed (no GT cameras).

    Builds student/teacher clouds from their OWN ray+depth, solves ONE global
    detached Sim3 (scale+rotation+translation) student→teacher, then L1 on
    the aligned residual. The single global gauge is removed, so per-view
    relative pose/scale drift (the multi-view layering signal) stays in the
    residual. Returns ``None`` when rays are missing or the fit degenerates.

    CONFIDENCE GATING (unlike the per-pixel depth/ray terms): the Sim3 is a
    single least-squares gauge SHARED by every point, so a few unreliable
    low-confidence points (sky / textureless / boundaries, where teacher depth
    itself is garbage) drag the alignment off and inject spurious residuals into
    otherwise-perfect regions. Empirically, fitting on all points vs. the
    high-confidence half moves the well-predicted residual from ~0.37 to ~0.
    We therefore fit AND penalise on the top-``(100-conf_percentile)%`` raw
    ``teacher_ray_conf`` only (``conf_percentile<=0`` disables the gate).
    """
    sr, tr = student.get("ray"), teacher.get("ray")
    if sr is None or tr is None:
        return None
    Ps = _points_from_ray_depth(student["depth"], sr)   # (V,Hr,Wr,3)
    Pt = _points_from_ray_depth(teacher["depth"], tr).detach()
    _, Hr, Wr, _ = Ps.shape
    c = None
    if conf is not None:
        c = conf.detach().float()
        while c.ndim > 3 and c.shape[0] == 1:  # (1,V,H,W) -> (V,H,W)
            c = c[0]
        # ray_conf must match ray resolution (point cloud grid); resize if not.
        if c.ndim == 3 and c.shape[-2:] != (Hr, Wr):
            c = F.interpolate(
                c.unsqueeze(1), size=(Hr, Wr), mode="bilinear",
                align_corners=False).squeeze(1)
    if stride > 1:
        Ps = Ps[:, ::stride, ::stride]
        Pt = Pt[:, ::stride, ::stride]
        if c is not None:
            c = c[:, ::stride, ::stride]
    Ps = Ps.reshape(-1, 3)
    Pt = Pt.reshape(-1, 3)

    hi = _hi_conf_mask(c, conf_percentile) if c is not None else None
    if hi is not None:
        m = hi.reshape(-1) > 0.5
        if m.shape[0] != Ps.shape[0]:
            # Defensive: spatial mismatch → disable gating rather than crash.
            m = torch.ones(Ps.shape[0], dtype=torch.bool, device=Ps.device)
        elif int(m.sum().item()) < 8:  # too few high-conf points -> fall back
            m = torch.ones(Ps.shape[0], dtype=torch.bool, device=Ps.device)
    else:
        m = torch.ones(Ps.shape[0], dtype=torch.bool, device=Ps.device)

    # Sim3 estimated on the high-confidence subset (gauge must not be poisoned).
    sim = _umeyama_sim3(Ps[m], Pt[m])
    if sim is None:
        return None
    s, R, t = sim
    Ps_al = s * (Ps @ R.T) + t
    # Penalise the same high-confidence subset (low-conf teacher point = noisy
    # target); {0,1} weight keeps the aligned-residual gradient dense there.
    w = m.float().unsqueeze(-1)
    per = (Ps_al - Pt).abs()
    w = w.expand_as(per)
    return (w * per).sum() / (w.sum() + 1e-8)


def simple_geo_loss(
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    *,
    w_depth: float = 1.0,
    w_ray: float = 1.0,
    w_normal: float = 0.0,
    w_point: float = 0.0,
    scale_align: bool = True,
    conf_percentile: float = 0.0,
    point_stride: int = 2,
    point_conf_percentile: float = 50.0,
) -> Dict[str, torch.Tensor]:
    """All-pixel geo loss: depth + ray + normal + self-posed point alignment.

    Depth/ray/normal supervise ALL pixels (no confidence masking or weighting);
    ``conf_percentile`` is accepted for CLI backward-compatibility but IGNORED
    by those terms. The point term is the ONE exception (see below).
    Teacher tensors are detached (no grad into targets).

    - depth : plain L1. With ``scale_align`` (default) one detached scene-wide
      metric scale (LS ratio) is applied to student depth and to the student
      ray *translation* channels ``ray[...,3:]`` before the L1, removing the
      global metric-scale gauge. Direction channels ``ray[...,:3]`` are left
      untouched (already scale-invariant).
    - ray   : plain L1 (all channels, all pixels).
    - normal: ``mean(1-cos(n_s,n_t))`` from 3D-point cross-product normals (see
      ``_normal_from_points`` — pixel height-field normals collapse to +Z).
    - point : ``self_pose_point_loss`` — each side builds its own cloud from
      ray+depth, aligned by ONE detached global Sim3, L1 on the residual.
      Because that Sim3 is a SHARED global gauge, low-confidence outliers poison
      it; this term therefore fits+penalises only the top
      ``(100-point_conf_percentile)%`` of ``teacher_ray_conf`` (unlike the
      per-pixel terms above). ``point_conf_percentile<=0`` uses all points.
    """
    # Per-term values are *unweighted* (for logging / comparison). ``total`` is
    # the only place weights enter (what train.backward uses).
    terms: Dict[str, torch.Tensor] = {}
    sd = _as_vhw(student["depth"]).float()
    td = _as_vhw(teacher["depth"]).float().detach()

    scale = None
    if scale_align:
        scale = _solve_scene_scale(sd, td, weight=None)
        sd = sd * scale
        terms["scale"] = scale.reshape(())
    terms["depth"] = F.l1_loss(sd, td)

    pr, tr = student.get("ray"), teacher.get("ray")
    tr_scaled = None
    if pr is not None and tr is not None:
        pr = pr.float()
        tr = tr.float().detach()
        # Scale-align ONLY the translation channels (last 3) using the shared
        # scene scale; directions (first 3) are metric-scale invariant.
        if scale is not None and pr.shape[-1] == 6:
            pr = torch.cat([pr[..., :3], pr[..., 3:] * scale], dim=-1)
        terms["ray"] = F.l1_loss(pr, tr)

    # Surface-normal consistency from 3D-point cross-product normals (uses the
    # scale-aligned student depth + its ray so it is comparable to teacher).
    if w_normal > 0 and pr is not None and tr is not None:
        Ps = _points_from_ray_depth(sd, pr)
        Pt = _points_from_ray_depth(td, tr).detach()
        ns, vs = _normal_from_points(Ps)
        nt, vt = _normal_from_points(Pt)
        nt = nt.detach()
        vmask = (vs & vt).float()  # exclude borders / degenerate normals
        ncos = 1.0 - (ns * nt).sum(-1).clamp(-1.0, 1.0)
        terms["normal"] = (vmask * ncos).sum() / (vmask.sum() + 1e-8)

    # Self-posed point-cloud Sim3 alignment loss (uses scale-aligned student
    # depth so it shares the depth gauge; teacher is detached).
    if w_point > 0 and pr is not None and tr is not None:
        pt_loss = self_pose_point_loss(
            {"depth": sd, "ray": pr}, {"depth": td, "ray": tr},
            stride=point_stride,
            conf=teacher.get("ray_conf"),
            conf_percentile=point_conf_percentile)
        if pt_loss is not None:
            terms["point"] = pt_loss

    total = w_depth * terms["depth"]
    if "ray" in terms and w_ray > 0:
        total = total + w_ray * terms["ray"]
    if "normal" in terms and w_normal > 0:
        total = total + w_normal * terms["normal"]
    if "point" in terms and w_point > 0:
        total = total + w_point * terms["point"]
    terms["total"] = total
    return terms


def geometry_group_loss(
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    *,
    c2w: Optional[torch.Tensor] = None,
    K: Optional[torch.Tensor] = None,
    view_mask: Optional[torch.Tensor] = None,
    w_depth: float = 1.0,
    w_ray: float = 0.5,
    w_point: float = 0.25,
    w_scale: float = 0.1,
    delta_depth: float = 1.0,
    delta_ray: float = 1.0,
    scene_scale_align: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compose depth/ray/point/scale terms for one student->teacher group.

    ``student``/``teacher`` are dicts with keys ``depth``/``ray``/``ray_conf``.
    ``view_mask`` (bool, len V) selects which views the differentiable terms use
    (the caller only runs DPT on those views to cap memory); when given, the
    teacher tensors are assumed already sliced to the same views.
    """
    # ray_conf gates ONLY the ray term (DA3 convention + ray is at a different
    # resolution than depth). depth/point/scale use uniform weights.
    # Per-term entries are unweighted; ``total`` applies w_*.
    conf = teacher.get("ray_conf")
    terms: Dict[str, torch.Tensor] = {}
    sd, td = student["depth"], teacher["depth"]
    terms["depth"] = robust_log_depth_loss(
        sd, td, None, scene_scale_align=scene_scale_align, delta=delta_depth)
    if student.get("ray") is not None and teacher.get("ray") is not None:
        terms["ray"] = ray_loss(
            student["ray"], teacher["ray"], conf, delta=delta_ray)
    if w_point > 0 and c2w is not None and K is not None:
        terms["point"] = point_map_loss(
            sd, td, c2w, K, None, scene_scale_align=scene_scale_align)
    if w_scale > 0:
        terms["scale"] = depth_scale_drift_loss(sd, None)
    total = w_depth * terms["depth"]
    if "ray" in terms and w_ray > 0:
        total = total + w_ray * terms["ray"]
    if "point" in terms:
        total = total + w_point * terms["point"]
    if "scale" in terms:
        total = total + w_scale * terms["scale"]
    terms["total"] = total
    return terms
