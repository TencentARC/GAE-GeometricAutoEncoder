"""Evaluation metrics used by the GAE paper.

===================  ==========================================================
Module               Covers
===================  ==========================================================
:mod:`latent`        Intrinsic latent diagnostics: rho, kappa, effective rank,
                     LNC@k, LDS / CDS / SRSS  (Tables 1 and 2).
:mod:`image`         Paired image fidelity: PSNR, SSIM, LPIPS  (Tables 3 and 5).
===================  ==========================================================

Distributional metrics (FID / FVD / rFID / rFVD) and geometry metrics
(VGGT ATE / RPEt / RPEr / reprojection, MEt3R, depth AbsRel / delta1,
Chamfer / point-map error) need an external evaluator or feature statistics
over a whole split. They are driven by the ``scripts/eval/eval_*.py`` entry points
rather than living here, because their cost is dominated by the evaluator, not
by the metric formula.
"""

from . import image, latent

__all__ = ["image", "latent"]
