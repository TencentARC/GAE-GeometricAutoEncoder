"""Recover full camera pose from DualDPT ray head and evaluate with standard metrics.

DualDPT ray head output (6 channels, BVHWC):
ray[:3] ≈ H_v @ [u, v, 1]  where H_v = R_rel(v) @ K⁻¹
ray[3:] ≈ relative displacement vector (spatially near-constant per view)

Metrics implemented:
1) DA3 style:  AUC@3°, AUC@30° — max(rot_err, trans_err) per pair
2) DUSt3R/VGGT style: RRA@τ, RTA@τ, mAA@30
3) Trajectory style: ATE, RPE_t, RPE_r (evo library, Sim3 aligned)
"""

import numpy as np
import torch


# ═══════════════════════════ Low-level helpers ═══════════════════════════

def _fit_homography(ray_view, subsample=4, conf=None):
    """Fit 3x3 H such that ray[:3](u,v) ≈ H @ [u,v,1]."""
    H_ray, W_ray = ray_view.shape[:2]
    direction = ray_view[..., :3].astype(np.float64)

    ys = np.arange(0, H_ray, subsample)
    xs = np.arange(0, W_ray, subsample)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    yy, xx = yy.ravel(), xx.ravel()

    P = np.stack([xx.astype(np.float64), yy.astype(np.float64),
                np.ones_like(xx, dtype=np.float64)], axis=1)
    D = direction[yy, xx]

    if conf is not None:
        w = conf[yy, xx].astype(np.float64)[:, None]
        Pw, Dw = P * w, D * w
    else:
        Pw, Dw = P, D

    Ht = np.linalg.lstsq(Pw, Dw, rcond=None)[0]
    return Ht.T


def _rotation_error_deg(R1, R2):
    """Geodesic rotation error in degrees."""
    R_err = R1 @ R2.T
    cos_a = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _translation_direction_error_deg(t1, t2):
    """Angular error between two translation vectors (scale-free)."""
    n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 180.0
    cos_a = np.clip(np.dot(t1, t2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


# ═══════════════════════════ Pose recovery ═══════════════════════════

def recover_intrinsics(ray, ray_conf=None, ref_view=0, subsample=4, input_size=None):
    """Recover K from ref view's ray[:3] ≈ K⁻¹ @ [u,v,1]."""
    if ray.ndim == 5:
        ray = ray[0]
    if ray_conf is not None and ray_conf.ndim == 4:
        ray_conf = ray_conf[0]

    V, H_ray, W_ray, _ = ray.shape
    conf = ray_conf[ref_view] if ray_conf is not None else None
    H_ref = _fit_homography(ray[ref_view], subsample=subsample, conf=conf)

    try:
        K = np.linalg.inv(H_ref)
    except np.linalg.LinAlgError:
        return np.eye(3)

    if abs(K[2, 2]) > 1e-12:
        K = K / K[2, 2]

    if input_size is not None:
        H_in, W_in = input_size
        K = np.diag([W_in / W_ray, H_in / H_ray, 1.0]) @ K

    return K


def recover_poses(
    ray,
    ray_conf=None,
    ref_view=0,
    subsample=4,
    input_size=None,
    *,
    return_per_view_intrinsics=False,
):
    """Recover cameras using DA3's official ray-pose implementation.

    DA3 fits every view independently with confidence-weighted RANSAC and QL
    decomposition, yielding per-view rotation, translation, focal length, and
    principal point.  This replaces the former GLD approximation that inverted
    only the reference-view homography and forced one shared K on all views.

    Args:
        ray: ``(V,H,W,6)`` or ``(B,V,H,W,6)`` array/tensor.
        ray_conf: matching ``(...,H,W)`` or ``(...,H,W,1)`` confidence.
        ref_view: retained for API compatibility.  It selects the legacy
            single-K return when ``return_per_view_intrinsics=False``; DA3's
            official pose fit itself is per-view and does not use this value.
        subsample: retained for API compatibility; official RANSAC controls its
            own sampling.
        input_size: output image ``(H,W)`` used to convert DA3's normalized
            focal lengths/principal points to pixel intrinsics.  Defaults to the
            ray-map resolution.
        return_per_view_intrinsics: return ``(V,3,3)`` K matrices.  False keeps
            old callers working by returning the selected ``(3,3)`` K.

    Returns:
        c2w_list: list of V ``(4,4)`` numpy arrays.
        K: ``(V,3,3)`` when requested, otherwise one ``(3,3)`` matrix.
    """
    # Import lazily so camera-only utilities remain importable in environments
    # without the DA3 package.
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray

    if torch.is_tensor(ray):
        ray_t = ray.detach().float()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ray_t = torch.as_tensor(np.asarray(ray), dtype=torch.float32, device=device)
    if ray_t.ndim == 4:
        ray_t = ray_t.unsqueeze(0)
    if ray_t.ndim != 5 or ray_t.shape[-1] != 6:
        raise ValueError(
            f"Expected ray shape (V,H,W,6) or (B,V,H,W,6), got {tuple(ray_t.shape)}"
        )
    if ray_t.shape[0] != 1:
        raise ValueError(
            f"recover_poses currently expects one scene (B=1), got B={ray_t.shape[0]}"
        )

    if ray_conf is None:
        conf_t = torch.ones_like(ray_t[..., :1])
    else:
        if torch.is_tensor(ray_conf):
            conf_t = ray_conf.detach().to(device=ray_t.device, dtype=torch.float32)
        else:
            conf_t = torch.as_tensor(
                np.asarray(ray_conf), dtype=torch.float32, device=ray_t.device
            )
        # Canonical confidence shape: (B,V,H,W,1).
        while conf_t.ndim > 5 and conf_t.shape[0] == 1:
            conf_t = conf_t.squeeze(0)
        if conf_t.ndim == 3:
            conf_t = conf_t.unsqueeze(0).unsqueeze(-1)
        elif conf_t.ndim == 4:
            if conf_t.shape[-1] == 1:
                conf_t = conf_t.unsqueeze(0)
            else:
                conf_t = conf_t.unsqueeze(-1)
        if conf_t.ndim != 5 or conf_t.shape[-1] != 1:
            raise ValueError(
                f"Expected ray_conf shape compatible with (B,V,H,W,1), "
                f"got {tuple(conf_t.shape)}"
            )
        if conf_t.shape[:4] != ray_t.shape[:4]:
            raise ValueError(
                f"ray/ray_conf shape mismatch: {tuple(ray_t.shape)} vs "
                f"{tuple(conf_t.shape)}"
            )

    H_ray, W_ray = ray_t.shape[-3:-1]
    H_out, W_out = input_size if input_size is not None else (H_ray, W_ray)
    with torch.inference_mode():
        pred_extrinsic, focal_norm, principal_norm = get_extrinsic_from_camray(
            ray_t, conf_t, H_ray, W_ray, training=False,
        )
        # Convention note: our DPT ray head emits ray directions in the WORLD
        # frame, so the raw [R|T] returned by get_extrinsic_from_camray is
        # already the camera-to-world (c2w) matrix here. DA3's official pipeline
        # applies affine_inverse (w2c -> c2w) because its own ray head uses the
        # opposite (camera-frame) direction convention; replicating that here
        # transposes the rotation and produces a catastrophic per-view rotation
        # drift (~111° vs GT). Verified against GT poses on scannetpp/clip_012:
        # raw [R|T] as c2w gives ~14.7° drift + 0.014 center error, whereas
        # affine_inverse gives ~111° drift. Do NOT re-add affine_inverse.
        pred_c2w = pred_extrinsic

        intrinsics = torch.eye(
            3, dtype=pred_c2w.dtype, device=pred_c2w.device,
        )[None, None].repeat(pred_c2w.shape[0], pred_c2w.shape[1], 1, 1)
        intrinsics[:, :, 0, 0] = focal_norm[:, :, 0] * (float(W_out) / 2.0)
        intrinsics[:, :, 1, 1] = focal_norm[:, :, 1] * (float(H_out) / 2.0)
        intrinsics[:, :, 0, 2] = principal_norm[:, :, 0] * (float(W_out) * 0.5)
        intrinsics[:, :, 1, 2] = principal_norm[:, :, 1] * (float(H_out) * 0.5)

    c2w_np = pred_c2w[0].detach().cpu().double().numpy()
    K_all = intrinsics[0].detach().cpu().double().numpy()
    c2w_list = [c2w_np[v] for v in range(c2w_np.shape[0])]
    if return_per_view_intrinsics:
        return c2w_list, K_all
    return c2w_list, K_all[int(ref_view)]


# ═══════════════════════════ Pairwise errors ═══════════════════════════

def _get_pairwise_errors(pred_c2w_list, gt_c2w_list):
    """Compute per-pair rotation and translation direction errors.

    Returns:
        rot_errors: list of floats (degrees)
        trans_errors: list of floats (degrees)
    """
    V = len(pred_c2w_list)
    rot_errors, trans_errors = [], []

    for i in range(V):
        for j in range(i + 1, V):
            R_rel_pred = pred_c2w_list[i][:3, :3] @ pred_c2w_list[j][:3, :3].T
            R_rel_gt = gt_c2w_list[i][:3, :3] @ gt_c2w_list[j][:3, :3].T
            rot_errors.append(_rotation_error_deg(R_rel_pred, R_rel_gt))

            t_rel_pred = pred_c2w_list[j][:3, 3] - pred_c2w_list[i][:3, 3]
            t_rel_gt = gt_c2w_list[j][:3, 3] - gt_c2w_list[i][:3, 3]
            trans_errors.append(_translation_direction_error_deg(t_rel_pred, t_rel_gt))

    return rot_errors, trans_errors


# ═══════════════════════ DA3 style: AUC@θ ═══════════════════════════

def compute_auc(pred_c2w_list, gt_c2w_list, max_threshold=30.0, num_bins=1000):
    """Compute AUC@θ (DA3 paper style).

    For each pair: error = max(rot_err, trans_dir_err).
    AUC@θ = (1/θ) * ∫₀ᶿ accuracy(τ) dτ  where accuracy(τ) = fraction with error < τ.

    Returns:
        auc: float in [0, 1] (1 = perfect).
    """
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0

    # Per-pair max error
    max_errors = np.maximum(np.array(rot_errors), np.array(trans_errors))

    # Integrate accuracy curve
    thresholds = np.linspace(0, max_threshold, num_bins + 1)
    accuracies = np.array([np.mean(max_errors < t) for t in thresholds])
    auc = float(np.trapz(accuracies, thresholds) / max_threshold)
    return auc


# ═══════════════════ DUSt3R/VGGT style: RRA/RTA/mAA ═══════════════════

def compute_rra_rta(pred_c2w_list, gt_c2w_list, threshold=15.0):
    """Compute RRA@τ and RTA@τ."""
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0, 0.0, [], []

    rot_arr = np.array(rot_errors)
    trans_arr = np.array(trans_errors)
    rra = float(np.mean(rot_arr < threshold) * 100)
    rta = float(np.mean(trans_arr < threshold) * 100)
    return rra, rta, rot_errors, trans_errors


def compute_maa(pred_c2w_list, gt_c2w_list, max_threshold=30):
    """Compute mAA@T (PoseDiffusion/VGGT standard).

    Per pair: error = max(rot_err, trans_dir_err).
    mAA = mean of CDF evaluated at integer-degree bins [1..T].
    Equivalent to mean(cumsum(histogram / num_pairs)).
    """
    rot_errors, trans_errors = _get_pairwise_errors(pred_c2w_list, gt_c2w_list)
    if not rot_errors:
        return 0.0

    max_errors = np.maximum(np.array(rot_errors), np.array(trans_errors))
    # CDF at integer bins [1, 2, ..., max_threshold]
    bins = np.arange(1, max_threshold + 1, dtype=np.float64)
    cdf = np.array([np.mean(max_errors < t) for t in bins])
    return float(np.mean(cdf)) * 100  # percentage


# ═══════════════════ Trajectory style: ATE/RPE (evo) ═══════════════════

def compute_ate_rpe(pred_c2w_list, gt_c2w_list):
    """Compute ATE and RPE using evo library (Sim3 alignment).

    Returns:
        ate: float (RMSE, meters after scale alignment)
        rpe_trans: float (RMSE, meters)
        rpe_rot: float (RMSE, degrees)
    """
    if len(pred_c2w_list) < 2:
        return None, None, None

    try:
        from utils.evo_utils import eval_metrics, get_tum_poses
        import tempfile, os

        pred_traj = get_tum_poses(pred_c2w_list)
        gt_traj = get_tum_poses(gt_c2w_list)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            tmpfile = f.name
        try:
            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj, gt_traj, seq="ray_head", filename=tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
        return float(ate), float(rpe_trans), float(rpe_rot)
    except ImportError:
        return None, None, None
    except Exception as e:
        print(f"[ATE/RPE] Error: {e}")
        return None, None, None


# ═══════════════════════════ Main entry point ═══════════════════════════

def _to_ref_centric(c2w_list, ref_idx=0):
    """Convert c2w poses to ref-centric frame (ref camera = identity)."""
    T_ref_inv = np.linalg.inv(c2w_list[ref_idx])
    return [T_ref_inv @ c2w_list[v] for v in range(len(c2w_list))]


def _scale_align(pred_pts, gt_pts):
    """Least-squares scale: s = (pred . gt) / (pred . pred)."""
    num = np.sum(pred_pts * gt_pts)
    den = np.sum(pred_pts * pred_pts)
    return num / den if den > 1e-12 else 1.0


def compute_camera_metrics(ray, ray_conf, gt_K, gt_c2w, cond_num,
                        subsample=4, input_size=None):
    """Compute ALL camera metrics from ray head output.

    Key: pred poses are in ref-centric frame, so GT is converted to
    ref-centric frame before comparison. ATE/RPE use evo library
    (Sim3 Umeyama, all_pairs, RMSE).

    Returns dict with:
        - focal_rel_err: relative focal length error
        - auc3, auc30: DA3-style AUC (max(rot,trans) per pair, scale-free)
        - rra15, rta15, maa30: DUSt3R/VGGT pairwise metrics (scale-free)
        - mean_rot_err, mean_trans_err: average pairwise errors (scale-free)
        - ate, rpe_trans, rpe_rot: trajectory metrics (evo Sim3 Umeyama, no pre-scaling)
        - scale: LS scale factor (for visualization only)
        - recovered_K, pred_c2w: recovered camera parameters
        - gt_c2w_ref: GT in ref-centric frame
        - pred_c2w_scaled: pred with LS scale-aligned translations (visualization only)
    """
    if ray.ndim == 5:
        ray = ray[0]
    if ray_conf is not None and ray_conf.ndim == 4:
        ray_conf = ray_conf[0]

    V = ray.shape[0]

    # Recover poses via DA3 ray fit (per-view R, t, and intrinsics).
    pred_c2w_list, K_pred = recover_poses(
        ray[None], ray_conf[None] if ray_conf is not None else None,
        ref_view=0, subsample=subsample, input_size=input_size,
        return_per_view_intrinsics=True,
    )

    # Convert GT to ref-centric frame
    gt_c2w_world = [gt_c2w[v].astype(np.float64) for v in range(V)]
    gt_c2w_ref = _to_ref_centric(gt_c2w_world, ref_idx=0)

    # Scale alignment (LS)
    pred_pts = np.array([p[:3, 3] for p in pred_c2w_list])
    gt_pts_ref = np.array([g[:3, 3] for g in gt_c2w_ref])
    scale = _scale_align(pred_pts, gt_pts_ref)

    # Scale-aligned pred (for visualization ONLY — metrics use un-scaled pred)
    pred_scaled = []
    for p in pred_c2w_list:
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = p[:3, :3]
        c2w[:3, 3] = p[:3, 3] * scale
        pred_scaled.append(c2w)

    # Focal length error (mean over views; supports shared or per-view GT K).
    gt_K_arr = np.asarray(gt_K, dtype=np.float64)
    if gt_K_arr.ndim == 2:
        gt_K_arr = np.broadcast_to(gt_K_arr, (V, 3, 3)).copy()
    focal_errs = []
    for v in range(V):
        gt_fx, gt_fy = gt_K_arr[v, 0, 0], gt_K_arr[v, 1, 1]
        pred_fx, pred_fy = abs(K_pred[v, 0, 0]), abs(K_pred[v, 1, 1])
        focal_errs.append(
            (abs(pred_fx - gt_fx) / (abs(gt_fx) + 1e-8)
             + abs(pred_fy - gt_fy) / (abs(gt_fy) + 1e-8)) / 2.0
        )
    focal_rel_err = float(np.mean(focal_errs))

    # Target views only (ref-centric frame, no pre-scaling)
    pred_tgt = [pred_c2w_list[v] for v in range(cond_num, V)]
    gt_tgt = [gt_c2w_ref[v] for v in range(cond_num, V)]

    # Defaults
    auc3, auc30 = 0.0, 0.0
    rra15, rta15, maa30 = 0.0, 0.0, 0.0
    mean_rot_err, mean_trans_err = None, None
    ate, rpe_trans, rpe_rot = None, None, None

    if len(pred_tgt) >= 2:
        # Pairwise metrics (scale-free: rotation=geodesic, translation=angular)
        auc3 = compute_auc(pred_tgt, gt_tgt, max_threshold=3.0)
        auc30 = compute_auc(pred_tgt, gt_tgt, max_threshold=30.0)

        rra15, rta15, rot_errs, trans_errs = compute_rra_rta(pred_tgt, gt_tgt, 15.0)
        maa30 = compute_maa(pred_tgt, gt_tgt, 30)
        mean_rot_err = float(np.mean(rot_errs))
        mean_trans_err = float(np.mean(trans_errs))

        # Trajectory metrics via evo (Sim3 Umeyama alignment inside evo)
        # Pass un-scaled pred — evo handles align=True, correct_scale=True
        ate, rpe_trans, rpe_rot = compute_ate_rpe(pred_tgt, gt_tgt)

    return {
        'focal_rel_err': float(focal_rel_err),
        'auc3': auc3, 'auc30': auc30,
        'rra15': rra15, 'rta15': rta15, 'maa30': maa30,
        'mean_rot_err': mean_rot_err, 'mean_trans_err': mean_trans_err,
        'ate': ate, 'rpe_trans': rpe_trans, 'rpe_rot': rpe_rot,
        'scale': float(scale),
        'recovered_K': K_pred,
        'pred_c2w': pred_c2w_list,
        'gt_c2w_ref': gt_c2w_ref,
        'pred_c2w_scaled': pred_scaled,
    }
