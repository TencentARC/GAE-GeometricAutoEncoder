"""Latent-space diagnostics — paper Section 4.1, Tables 1 and 2.

Every function here answers one question about a candidate generated state:

===========================  ==================================================
``velocity_irreducible_variance``  rho, the within-scene velocity fraction.
                             Lower means the flow transport is easier to model.
``spectral_stats``           kappa (covariance condition number) and effective
                             rank. Lower kappa / higher rank = better
                             conditioned.
``latent_neighbor_consistency``    LNC@k, semantic neighbourhood agreement.
``spatial_structure_metrics``      LDS / CDS / SRSS, spatial relations among
                             tokens (iREPA-style).
===========================  ==================================================

Cross-view retrieval (xLNC*) is measured by
:func:`src.metrics.geometry.cross_view_retrieval`.

All functions take latents as ``[N, C, H, W]`` and are label-free unless a
``labels`` argument is documented.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import math

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

logger = logging.getLogger(__name__)

# Spatial thresholds in IMAGE PIXELS. Shared across encoders so that 18^2 and
# 32^2 grids are compared at the same physical scale; tuned for ~252px inputs,
# where near is roughly the immediate neighbourhood and far is clearly distant.
SPATIAL_NEAR_PX = 24.0
SPATIAL_FAR_PX = 96.0
SPATIAL_CDS_BIN_PX = 16.0
SPATIAL_CDS_MIN_PX = 24.0


def velocity_irreducible_variance(
    latents: torch.Tensor,          # [N, C, H, W]
    labels: List[int],
    num_iters: int = 5,
    device: str = "cpu",
    return_diagnostics: bool = False,
):
    """VIV: Velocity Irreducible Variance.

    Lower is better: the diffusion velocity is easier to predict.
    Paper baseline conv VAE: 1.14; good latent (REPA+CRadio): 0.37.

    Cross-encoder caveat: raw VIV scales ~1/sqrt(C*S) (the mean-of-sqrt-eigenvalues
    shrinks as the latent grows), so it is NOT comparable across latents of very
    different dimensionality. With ``return_diagnostics=True`` this also returns
    ``(viv, diag)`` where ``diag`` carries the pieces needed to de-confound it:
      * ``dim``   = C*S (nominal latent dimensionality),
      * ``nviv``  = viv*sqrt(dim), the empirically dimension-invariant VIV,
      * ``rho``   = mean within-scene variance fraction v_total/(C*S) in [0,1],
                    a dimension-free "how concentrated is the conditional" signal.
    """
    _, C, H, W = latents.shape
    S_dim = H * W

    z = latents.float()
    mean = z.mean(dim=[0, 2, 3], keepdim=True)
    std  = z.std(dim=[0, 2, 3], keepdim=True).clamp(min=1e-8)
    z = (z - mean) / std
    z = z.to(device)

    groups: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for zi, li in zip(z, labels):
        groups[li].append(zi)
    for li in groups:
        groups[li] = torch.stack(groups[li])   # [Ni, C, H, W]

    logger.info(f"VIV: {len(groups)} scenes (classes), C={C}, H={H}, W={W}")

    viv_vals: List[float] = []
    rho_vals: List[float] = []
    for li in sorted(groups):
        zg = groups[li].reshape(len(groups[li]), -1)   # [Ni, C*S]
        Ni = zg.shape[0]
        if Ni < 4:
            continue

        v_total = zg.var(dim=0, unbiased=False).sum()
        # within-scene variance fraction: v_total is summed over C*S dims whose
        # global per-channel variance is ~1, so v_total/(C*S) in [0,1].
        rho_vals.append(float(v_total.item()) / float(C * S_dim))
        z_mat = zg.reshape(Ni, C, S_dim)
        z_mat = z_mat - z_mat.mean(dim=0, keepdim=True)

        W_S: Optional[torch.Tensor] = None
        eig_C = eig_S = None
        for _ in range(num_iters):
            if W_S is None:
                z_C = z_mat.permute(0, 2, 1).reshape(-1, C)
            else:
                z_C = torch.matmul(z_mat, W_S).permute(0, 2, 1).reshape(-1, C)
            _, S_C, Vh_C = torch.linalg.svd(z_C, full_matrices=False)
            eig_C = (S_C ** 2) / max(z_C.shape[0] - 1, 1)
            W_C = Vh_C.T * (1.0 / torch.sqrt(eig_C + 1e-12)).unsqueeze(0)

            z_wC = torch.matmul(z_mat.permute(0, 2, 1), W_C).permute(0, 2, 1)
            z_S = z_wC.reshape(-1, S_dim)
            _, S_S, Vh_S = torch.linalg.svd(z_S, full_matrices=False)
            eig_S = (S_S ** 2) / max(z_S.shape[0] - 1, 1)
            W_S = Vh_S.T * (1.0 / torch.sqrt(eig_S + 1e-12)).unsqueeze(0)

        eig_vals = (eig_C.unsqueeze(1) * eig_S.unsqueeze(0)).flatten()
        eig_vals = torch.sort(eig_vals, descending=True).values
        c_scale = v_total / (eig_vals.sum() + 1e-12)
        lambda_d = c_scale * eig_vals
        viv = ((math.pi / 2) * lambda_d.sqrt()).mean().item()
        viv_vals.append(viv)

    if not viv_vals:
        return (float("nan"), {"dim": C * S_dim, "nviv": float("nan"),
                               "rho": float("nan"), "n_scenes": 0}) \
            if return_diagnostics else float("nan")
    result = float(np.mean(viv_vals))
    logger.info(f"VIV = {result:.4f}  (paper baseline: 1.14, good latent: ~0.37)")
    if return_diagnostics:
        dim = int(C * S_dim)
        diag = {
            "dim": dim,
            "nviv": result * math.sqrt(dim),
            "rho": float(np.mean(rho_vals)) if rho_vals else float("nan"),
            "n_scenes": len(viv_vals),
        }
        return result, diag
    return result


def _dct_1d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """DCT-II along last dim, ortho-normalized.

    Pure-torch implementation numerically identical to torch_dct.dct(norm='ortho').
    Matches the paper's results even when torch_dct is not installed.
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v, dim=1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * math.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == "ortho":
        V[:, 0] = V[:, 0] / (math.sqrt(N) * 2)
        V[:, 1:] = V[:, 1:] / (math.sqrt(N / 2) * 2)
    V = 2 * V.view(*x_shape)
    return V


def _dct_2d(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """2D DCT-II over the last two dims (matches torch_dct.dct_2d)."""
    X1 = _dct_1d(x, norm=norm)
    X2 = _dct_1d(X1.transpose(-1, -2), norm=norm).transpose(-1, -2)
    return X2


def pca_reduce_channels(z: torch.Tensor, k: int) -> torch.Tensor:
    """Project ``[B,C,h,w]`` onto its top-k principal channel directions -> ``[B,k,h,w]``.

    PCA is fit globally over all (sample, position) rows. Returns ``z`` unchanged
    if ``C<=k``.
    """
    b, c, h, w = z.shape
    if k <= 0 or c <= k:
        return z
    x = z.permute(0, 2, 3, 1).reshape(-1, c).float()
    x = x - x.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=k, center=False)  # v: [C, k]
    proj = x @ v[:, :k]
    return proj.reshape(b, h, w, k).permute(0, 3, 1, 2).contiguous()


def gold_standard_latent(z: torch.Tensor, k: int, spatial: int) -> torch.Tensor:
    """Common ``[*, k, spatial, spatial]`` latent (top-k channel PCA + avg-pool) for a
    dimension-controlled ("gold") VIV that removes the 1/sqrt(C*S) confound."""
    zc = pca_reduce_channels(z, k)
    if spatial and spatial > 0 and (zc.shape[-1] != spatial or zc.shape[-2] != spatial):
        zc = torch.nn.functional.adaptive_avg_pool2d(zc, (spatial, spatial))
    return zc


def marginal_channel_spectrum(
    z: torch.Tensor, q_max: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, Tuple[int, int, int]]:
    """Top-``q`` eigenpairs of the *marginal* channel covariance of ``[B,C,h,w]``.

    Returns ``(x_centered [B*h*w, C], V [C, q], eig [q], total_var, (B,h,w))``. The
    near-zero tail of this spectrum is the "dead"/redundant channel subspace (the
    correct basis for dropping dead channels), unlike VIV's internal within-class
    factors whose near-zero directions are the easy-to-diffuse ones.
    """
    b, c, h, w = z.shape
    x = z.permute(0, 2, 3, 1).reshape(-1, c).float()
    x = x - x.mean(0, keepdim=True)
    n = x.shape[0]
    total_var = float((x * x).sum().item()) / max(n - 1, 1)
    q = min(int(q_max), c, n)
    _, s, v = torch.pca_lowrank(x, q=q, center=False)  # v: [C, q]
    eig = (s ** 2) / max(n - 1, 1)
    return x, v, eig, total_var, (b, h, w)


def r_from_energy(eig: torch.Tensor, total_var: float, energy: float) -> int:
    """Smallest ``r`` whose top-r channel eigenvalues cover ``energy`` of ``total_var``.

    Clamped to ``[1, len(eig)]``; returns the cap ``len(eig)`` if the leading
    eigenvalues cannot reach ``energy`` (very flat spectrum).
    """
    frac = torch.cumsum(eig, 0) / max(total_var, 1e-12)
    over = (frac >= float(energy)).nonzero()
    r = int(over[0].item()) + 1 if over.numel() else int(eig.numel())
    return max(1, min(r, int(eig.numel())))


def effective_subspace_viv(
    z_views: torch.Tensor,
    viv_labels: List[int],
    energies: List[float],
    device: str,
    q_max: int = 256,
) -> Dict[float, Dict[str, float]]:
    """eVIV: VIV/rho on each encoder's own effective channel subspace.

    For every energy threshold, project ``z_views`` onto the top-``r`` marginal
    channel eigen-directions (per-encoder ``r``, NO spatial pooling), then compute
    VIV and rho at dimension ``D=r*h*w``. Also returns ``phi = eVIV/((pi/2)sqrt(rho_e))``
    (dimension-free spectrum-shape proxy). Because ``r`` differs across encoders,
    eVIV absolutes compare only at equal ``r``; ``rho_e``/``phi`` carry the ranking.
    """
    x, v, eig, total_var, (b, h, w) = marginal_channel_spectrum(z_views, q_max=q_max)
    out: Dict[float, Dict[str, float]] = {}
    for e in energies:
        r = r_from_energy(eig, total_var, e)
        proj = x @ v[:, :r]  # [B*h*w, r]
        z_eff = proj.reshape(b, h, w, r).permute(0, 3, 1, 2).contiguous()
        viv, diag = velocity_irreducible_variance(
            z_eff, viv_labels, device=device, return_diagnostics=True
        )
        rho_e = float(diag.get("rho", float("nan")))
        phi = (
            viv / ((math.pi / 2) * math.sqrt(rho_e))
            if rho_e and rho_e > 0 and viv == viv
            else float("nan")
        )
        out[e] = {"eviv": float(viv), "rho_e": rho_e, "phi": phi, "r": int(r)}
        del z_eff, proj
    del x, v
    return out


def spectral_energy_concentration(
    latents: torch.Tensor,          # [B, C, H, W]
    thresholds: Tuple[float, ...] = (0.25, 0.5),
    dist_type: str = "Manhattan",
) -> Dict[float, float]:
    """SEC: Spectral Energy Concentration (2D-DCT based).

    Ratio of high-frequency energy to total energy at threshold t.
    Lower means stronger low-frequency concentration -> generation-friendly.
    Paper baseline: 0.40; good latent (REPA+CRadio): 0.04.
    """
    try:
        import torch_dct as dct_mod
        latent_dct = dct_mod.dct_2d(latents.float(), norm="ortho")
    except ImportError:
        latent_dct = _dct_2d(latents.float(), norm="ortho")

    energy = latent_dct ** 2
    B, C, H, W = energy.shape
    avg_energy = energy.mean(dim=(0, 1))   # [H, W]

    u = torch.arange(H, device=latents.device).view(-1, 1).expand(H, W).float()
    v = torch.arange(W, device=latents.device).view(1, -1).expand(H, W).float()
    if dist_type == "Manhattan":
        dist_map = u + v
        max_dist = float((H - 1) + (W - 1))
    else:
        dist_map = (u ** 2 + v ** 2).sqrt()
        max_dist = math.sqrt((H - 1) ** 2 + (W - 1) ** 2)

    total_energy = avg_energy.sum()
    results: Dict[float, float] = {}
    for th in thresholds:
        actual_th = th * max_dist
        hi_mask = dist_map > actual_th
        hi_energy = (avg_energy * hi_mask).sum()
        results[th] = (hi_energy / total_energy).item() if total_energy > 0 else 0.0

    logger.info(
        f"SEC (Manhattan) | "
        + " | ".join(f"t={t:.2f}: {r:.4f}" for t, r in results.items())
        + "  (paper baseline: 0.40, good latent: ~0.04)"
    )
    return results


def spectral_stats(latents: torch.Tensor) -> Dict[str, float]:
    """Covariance eigenvalue spectrum summary: effective rank, condition number."""
    # [N, C, H, W] → flatten spatial → [N*H*W, C]
    z = latents.float().permute(0, 2, 3, 1).reshape(-1, latents.shape[1])
    z = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    _, S, _ = torch.linalg.svd(z, full_matrices=False)
    eigs = (S ** 2) / max(N - 1, 1)

    total = eigs.sum().item()
    eff_rank = (total ** 2) / (eigs ** 2).sum().item() if (eigs ** 2).sum() > 0 else 0.0
    cond = (eigs.max() / eigs.clamp(min=1e-12).min()).item()

    cumvar = torch.cumsum(eigs, dim=0) / total
    top10_var = cumvar[min(9, len(cumvar) - 1)].item()

    stats = {
        "effective_rank": eff_rank,
        "condition_number": cond,
        "top10_cum_var": top10_var,
        "lambda_max": eigs.max().item(),
        "lambda_min": eigs.min().item(),
    }
    logger.info(
        f"Spectrum | eff_rank={eff_rank:.1f}/{latents.shape[1]}  "
        f"cond={cond:.1f}  top10_cumvar={top10_var:.3f}"
    )
    return stats


def _decorrelate(x_mc: torch.Tensor, *, center: bool = True, rm_pc: int = 0) -> torch.Tensor:
    """De-confound ``[M, C]`` samples so cosine reflects structure, not a shared offset.

    An encoder whose features share a dominant common direction (e.g. da3_giant:
    effrank 9 / cond 2e11) makes *every* pair look similar under cosine, inflating
    neighbour similarity and collapsing local-vs-distant margins. Subtracting the
    per-sample-axis mean removes that DC/common component; ``rm_pc>0`` additionally
    projects out the top principal directions. Used for both LNC (M=images) and the
    spatial metrics (M=spatial tokens of one image). L2 distance is translation
    invariant, so centering only changes cosine-based results.
    """
    x = x_mc.float()
    if center:
        x = x - x.mean(0, keepdim=True)
    if rm_pc and rm_pc > 0:
        xc = x - x.mean(0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(xc, full_matrices=False)  # vh: [k, C]
            k = min(int(rm_pc), vh.shape[0])
            basis = vh[:k]  # [k, C]
            x = x - (x @ basis.T) @ basis
        except Exception:  # pragma: no cover - numerical fallback
            pass
    return x


def _pixel_dist_matrix(H: int, W: int, image_res: int, device: str = "cpu") -> torch.Tensor:
    """Euclidean pixel distance between latent-cell centers, ``[T, T]`` (T=H*W).

    Grid-cell Manhattan distance is NOT comparable across encoders (da3 18² grid →
    ~14px/cell vs sd_vae 32² → ~8px/cell, so "dist=1" means different physical
    scales). Mapping cell centers to image pixels lets every encoder share the same
    near/far thresholds in pixels.
    """
    cell_h = float(image_res) / H
    cell_w = float(image_res) / W
    rows = (torch.arange(H, device=device).float() + 0.5) * cell_h
    cols = (torch.arange(W, device=device).float() + 0.5) * cell_w
    rr = rows.repeat_interleave(W)  # [T]
    cc = cols.repeat(H)             # [T]
    coords = torch.stack([rr, cc], dim=1)  # [T, 2]
    return torch.cdist(coords, coords)     # [T, T]


def latent_neighbor_consistency(
    latents: torch.Tensor,                       # [N, C, H, W]
    labels: List[int],
    k_list: Tuple[int, ...] = (5, 10),
    mask: Optional[torch.Tensor] = None,         # [N, h, w] or None (= global mean)
    metric: str = "cosine",
    device: str = "cpu",
    center: bool = True,
    rm_pc: int = 0,
) -> Dict[int, float]:
    """LNC: foreground latent spatial mean -> kNN same-label neighbour ratio (higher is better).

    Reference recipe:
      average each image's latent over its foreground mask to a (C,) vector ->
      pairwise distance -> take each row's k nearest neighbours -> mean fraction
      that share the same label.

    This implementation:
      - mask=None falls back to an all-ones mask (whole-image spatial mean, incl. background);
      - metric='cosine' matches the reference default ('l2' optional);
      - center=True removes the dataset mean from the (N,C) features, dropping the
        shared common component (anisotropy de-confounding); affects cosine only
        (l2 is translation-invariant);
      - self-distance set to inf to exclude it; pure torch, no sklearn dependency.
    """
    N, C, H, W = latents.shape
    z = latents.float().to(device)
    if mask is None:
        feat = z.mean(dim=[2, 3])                # [N, C]
        src = "global-mean"
    else:
        m = mask.float().to(device)
        if m.ndim == 3:
            m = m.unsqueeze(1)                   # [N, 1, h, w]
        if m.shape[-2:] != (H, W):
            m = torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
        m = (m > 0).float()
        fg = m.sum(dim=[2, 3]).clamp(min=1.0)    # [N, 1]
        feat = (z * m).sum(dim=[2, 3]) / fg      # [N, C]
        src = "masked-mean"

    if metric == "cosine" and (center or rm_pc):
        feat = _decorrelate(feat, center=center, rm_pc=rm_pc)
    results = _lnc_knn(feat, labels, k_list, metric=metric, device=device)
    logger.info(
        f"LNC ({metric}, {src}, center={center}) | "
        + " ".join(f"k={k}:{v:.3f}" for k, v in results.items())
    )
    return results


def _lnc_knn(
    feat: torch.Tensor,          # [N, C] per-image feature (already centered if desired)
    labels: List[int],
    k_list: Tuple[int, ...],
    metric: str = "cosine",
    device: str = "cpu",
) -> Dict[int, float]:
    """kNN same-label fraction core, shared by streaming/non-streaming LNC."""
    feat = feat.to(device)
    N = feat.shape[0]
    lbl = torch.tensor(labels, device=device, dtype=torch.long)
    if metric == "cosine":
        feat_n = torch.nn.functional.normalize(feat, dim=1)
        dist = 1.0 - feat_n @ feat_n.T
    elif metric == "l2":
        dist = torch.cdist(feat, feat, p=2)
    else:
        raise ValueError(f"metric must be 'cosine' or 'l2', got {metric!r}")
    dist.fill_diagonal_(float("inf"))            # exclude self
    results: Dict[int, float] = {}
    for k in k_list:
        k_eff = min(k, N - 1)
        nn_idx = torch.topk(dist, k=k_eff, dim=1, largest=False).indices   # [N, k]
        nn_lbl = lbl[nn_idx]                                               # [N, k]
        same = (nn_lbl == lbl.unsqueeze(1)).float().mean(dim=1)            # [N]
        results[k] = float(same.mean().item())
    return results


# ---------------------------------------------------------------------------
# Spatial Structure: LDS / CDS / SRSS  (iREPA, Singh et al. 2025)
# ---------------------------------------------------------------------------


def _manhattan_dist_matrix(H: int, W: int, device: str = "cpu") -> torch.Tensor:
    """Manhattan-distance matrix [T, T] over all patch pairs on the HxW grid, T=H*W."""
    rows = torch.arange(H, device=device).repeat_interleave(W)  # [T]
    cols = torch.arange(W, device=device).repeat(H)             # [T]
    return (rows.unsqueeze(1) - rows.unsqueeze(0)).abs() + \
           (cols.unsqueeze(1) - cols.unsqueeze(0)).abs()         # [T, T]


# Default spatial thresholds in IMAGE PIXELS (shared across encoders so that
# 18² and 32² grids are compared at the same physical scale). Tuned for ~252px
# inputs: near≈"immediate neighbourhood" (>1 coarse cell), far≈"clearly distant".


def _spatial_structure_values(
    latents: torch.Tensor,                       # [N, C, H, W]
    masks: Optional[torch.Tensor],               # [N, H_img, W_img] region-id or None
    device: str,
    *,
    image_res: int,
    center: bool = True,
    rm_pc: int = 0,
    near_px: float = SPATIAL_NEAR_PX,
    far_px: float = SPATIAL_FAR_PX,
    cds_bin_px: float = SPATIAL_CDS_BIN_PX,
    cds_min_px: float = SPATIAL_CDS_MIN_PX,
) -> Dict[str, List[float]]:
    """Per-image LDS/CDS/SRSS lists (single source of truth for streaming + batch).

    Two de-confounding fixes vs the raw iREPA metric:
      * distances are in IMAGE PIXELS (cell centers mapped via image_res), so a
        18² and a 32² grid share the same near/far thresholds instead of "1 cell";
      * ``center=True`` removes each image's shared/anisotropic token component
        before cosine (otherwise a dominant common direction inflates every sim).
    Returns raw per-image values; callers nanmean and may run twice (raw/centered).
    """
    N, C, H, W = latents.shape
    z = latents.float().to(device)
    T = H * W
    pdist = _pixel_dist_matrix(H, W, image_res, device=device)   # [T, T]

    masks_lat: Optional[torch.Tensor] = None
    if masks is not None:
        m = masks.float().to(device).unsqueeze(1)                # [N,1,Hi,Wi]
        m = F.interpolate(m, size=(H, W), mode="nearest").squeeze(1)
        masks_lat = m.round().long()

    lds_near_m = (pdist > 0) & (pdist <= near_px)
    lds_far_m = pdist >= far_px
    lds_ok = bool(lds_near_m.any() and lds_far_m.any())

    edges = torch.arange(0.0, far_px + cds_bin_px, cds_bin_px, device=device)
    ring_lo, ring_hi = edges[:-1], edges[1:]
    ring_ctr = (ring_lo + ring_hi) / 2
    ring_masks_all = [(pdist >= lo) & (pdist < hi) for lo, hi in zip(ring_lo.tolist(), ring_hi.tolist())]
    ring_idx = [
        i for i, mm in enumerate(ring_masks_all)
        if bool(mm.any()) and cds_min_px <= float(ring_ctr[i]) <= far_px
    ]
    cds_ok = len(ring_idx) >= 2
    if cds_ok:
        cds_masks = [ring_masks_all[i] for i in ring_idx]
        cds_x = ring_ctr[ring_idx]
        cds_xm = cds_x - cds_x.mean()
        cds_xm_sq = (cds_xm ** 2).sum().clamp(min=1e-12)

    srss_pos_dist = lds_near_m
    srss_neg_dist = lds_far_m

    vals: Dict[str, List[float]] = {"lds": [], "cds": [], "srss": []}
    for i in range(N):
        x = z[i].reshape(C, T).T                                 # [T, C]
        if center or rm_pc:
            x = _decorrelate(x, center=center, rm_pc=rm_pc)
        xn = F.normalize(x, dim=1)
        sim = xn @ xn.T                                          # [T, T]

        if lds_ok:
            vals["lds"].append((sim[lds_near_m].mean() - sim[lds_far_m].mean()).item())
        else:
            vals["lds"].append(float("nan"))

        if cds_ok:
            g = torch.stack([sim[mm].mean() for mm in cds_masks])
            beta = (cds_xm * (g - g.mean())).sum() / cds_xm_sq
            vals["cds"].append((-beta).item())
        else:
            vals["cds"].append(float("nan"))

        if masks_lat is None:
            continue
        region = masks_lat[i].reshape(T)
        fg = region > 0
        bg = region == 0
        if fg.sum() < 2 or bg.sum() < 1:
            vals["srss"].append(float("nan"))
            continue
        pos_m = (fg.unsqueeze(1) & fg.unsqueeze(0)) & srss_pos_dist
        neg_m = (fg.unsqueeze(1) & bg.unsqueeze(0)) & srss_neg_dist
        pos_cnt = pos_m.sum(1)
        neg_cnt = neg_m.sum(1)
        pos_mean = (sim * pos_m).sum(1) / pos_cnt.clamp(min=1)
        neg_mean = (sim * neg_m).sum(1) / neg_cnt.clamp(min=1)
        anc_ok = fg & (pos_cnt > 0) & (neg_cnt > 0)
        vals["srss"].append(
            (pos_mean[anc_ok] - neg_mean[anc_ok]).mean().item()
            if anc_ok.any() else float("nan")
        )
    return vals


def _nanmean_list(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def spatial_structure_metrics(
    latents: torch.Tensor,                       # [N, C, H, W]
    masks: Optional[torch.Tensor] = None,
    device: str = "cpu",
    *,
    image_res: int,
    center: bool = True,
    rm_pc: int = 0,
    near_px: float = SPATIAL_NEAR_PX,
    far_px: float = SPATIAL_FAR_PX,
    cds_bin_px: float = SPATIAL_CDS_BIN_PX,
    cds_min_px: float = SPATIAL_CDS_MIN_PX,
) -> Dict[str, float]:
    """Aggregated iREPA-style spatial structure (LDS/CDS/SRSS), image-pixel scaled.

    LDS  Local-vs-Distant Similarity (higher better): mean cos(near <= near_px) - mean cos(far >= far_px)
    CDS  Correlation-Decay Slope     (higher better): negative slope of ring-mean cos vs pixel radius
    SRSS Semantic-Region Self-Similarity (higher better, needs mask): mean cos(fg-near) - mean cos(bg-far)
    ``center=True`` de-confounds anisotropy (see ``_decorrelate``). Distances in
    image pixels (``image_res``) make encoders with different grids comparable.
    """
    vals = _spatial_structure_values(
        latents, masks, device,
        image_res=image_res, center=center, rm_pc=rm_pc,
        near_px=near_px, far_px=far_px, cds_bin_px=cds_bin_px, cds_min_px=cds_min_px,
    )
    result: Dict[str, float] = {
        "lds": _nanmean_list(vals["lds"]),
        "cds": _nanmean_list(vals["cds"]),
    }
    if any(not (v != v) for v in vals["srss"]):
        result["srss"] = _nanmean_list(vals["srss"])

    parts = [f"LDS={result['lds']:.4f}", f"CDS={result['cds']:.4f}"]
    if "srss" in result:
        parts.append(f"SRSS={result['srss']:.4f}")
    logger.info(
        "Spatial | " + "  ".join(parts)
        + f"  (center={center}, near≤{near_px:g}px, far≥{far_px:g}px, "
        f"cds≥{cds_min_px:g}px, res={image_res}, grid={latents.shape[2]}x{latents.shape[3]})"
    )
    return result


# ---------------------------------------------------------------------------
# Data loader: ScanNet++ preprocessed latents (precomputed .pt chunks)
# ---------------------------------------------------------------------------


def latent_diagnostics(
    latents: torch.Tensor,
    labels: Optional[Sequence[int]] = None,
    *,
    device: str = "cpu",
    lnc_k: int = 5,
    compute_structure: bool = True,
    image_res: int = 252,
    masks: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute every intrinsic latent diagnostic reported in Tables 1 and 2.

    Args:
        latents: ``[N, C, H, W]`` posterior-mean latents.
        labels: Per-sample class / scene labels for rho and LNC. When ``None``
            every sample is treated as its own class, which disables the
            semantic metrics (rho and LNC) but keeps the spectral and spatial
            ones meaningful.
        device: Device for the rho eigen-decomposition.
        lnc_k: Neighbourhood size for LNC@k (the paper uses 5).
        compute_structure: Run the O(N^2) LDS / CDS computation.
        image_res: Input resolution in pixels, used to convert the spatial
            thresholds so that different latent grids stay comparable.
        masks: Optional [N, H, W] region ids; required for SRSS.

    Returns:
        Dict with ``rho``, ``kappa``, ``effective_rank`` and, when the required
        inputs are available, ``lnc@{lnc_k}``, ``lds``, ``srss``.
    """
    out: Dict[str, float] = {}

    # Spectral: conditioning and rank-carrying channels.
    spec = spectral_stats(latents)
    out["kappa"] = float(spec["condition_number"])
    out["effective_rank"] = float(spec["effective_rank"])

    if labels is not None:
        labels_list = [int(x) for x in labels]
        distinct = len(set(labels_list))
        # rho needs at least two samples per class to form a within-class mean.
        if distinct < len(labels_list):
            _, diag = velocity_irreducible_variance(
                latents, labels_list, device=device, return_diagnostics=True
            )
            out["rho"] = float(diag["rho"])
            lnc = latent_neighbor_consistency(
                latents, labels_list, k_list=(lnc_k,), center=True
            )
            out["lnc@%d" % lnc_k] = float(lnc[lnc_k])

    if compute_structure:
        struct = spatial_structure_metrics(
            latents, masks, device=device, image_res=image_res, center=True
        )
        out["lds"] = float(struct["lds"])
        out["cds"] = float(struct["cds"])
        if "srss" in struct:
            out["srss"] = float(struct["srss"])

    return out
