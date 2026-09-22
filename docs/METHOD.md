# Paper equations to code

Every equation in the paper, and where it lives in this repository. Equation
numbers match Sections 3 and 4 of
`GAE: Learning a Geometry-Native Latent Space for 3D-Consistent World Generation`.

## Stage 1 — the codec (Section 3.1)

| Eq. | What it says | Where |
|---|---|---|
| (4) | RGB and geometry are two readouts of the same latent: `I = D_rgb(z)`, `G = H_DPT(Dec(z))` | `src/stage1/gae_codec.py` — `GAECodec.decode_rgb`, `GAECodec.decode`; the DPT call is wired in `gae/pipeline.py` → `GAE.decode_geometry` |
| (5) | Level-wise normalization and channel-wise fusion of the four DA3 levels into `X` | `src/stage1/gae_codec.py` → `GAECodec.normalize_levels` (and `denormalize_and_split` for the inverse) |
| (6) | Encoder / decoder: `mu, log var = Enc(X)`, `F̂ = Dec(z)` | `GAECodec.encode`, `GAECodec.decode` |
| (7) | Reparameterization `z = mu + sigma ⊙ eps` | `GAECodec.reparameterize` — sampling happens **only** during codec training; the posterior mean is used deterministically for flow training and inference |
| (12) | `L_codec = L_feat + λ_kl L_kl + L_rgb + L_geo + L_repr` | `L_feat` and `L_kl` in `GAECodec.compute_loss`; `L_rgb` / `L_geo` / `L_repr` in `scripts/train_codec.py`, because they need image targets and the frozen teachers |
| (8) | `L_tok`: position-wise cosine alignment to C-RADIO through a learned projector | `src/stage1/repa_target.py` → `repa_cosine_loss`; teacher `CRadioTarget`; projector `GAECodec.repa_proj` / `repa_project` |
| (9)–(10) | `L_struct`: match DINOv2 patch–patch similarity in the raw posterior mean | `src/stage1/repa_target.py` → `repa_similarity_loss`; teacher `DINOv2Target` |
| (15) | Standardize the frozen posterior mean per channel before flow training | `scripts/train_codec.py` emits the stats; `gae/pipeline.py` → `GAE.standardize` / `destandardize` |

### Two `L_struct` variants

The research tree applied the similarity loss in two spaces. Both are kept,
selected by weight:

* `repa.struct_mu_weight` — similarities in the **raw posterior mean**. This is
  the paper's term (Eqs. 9–10) and the one used in every reported model.
* `repa.struct_weight` — the same loss on the **projected** tokens. Used only
  for the Table 2 ablation; leave it at `0`.

In `configs/gae_64.yaml` and `configs/gae_128.yaml`: `weight: 0.25` (λ_repa) and
`struct_mu_weight: 8.0` (λ_μ), matching Eq. (12).

## Stage 2 — flow matching (Sections 3.2–3.3)

| Eq. | What it says | Where |
|---|---|---|
| (10) | Linear path `z_t = (1-t) z_gt + t z_1`, target velocity `u_t = z_1 - z_gt`, MSE loss | `src/stage2/transport/flow.py` → `training_losses_rae` |
| (11) | x-prediction to velocity: `v = (z_t - z_pred) / max(t, t_eps)`, `t_eps = 0.05` | `src/stage2/transport/flow.py` → `convert_x_to_v` |
| (13) | Metric Plücker ray: decompose into `d`, `m̂ = m / ‖m‖`, `s = log ‖m‖`; inject into q/k self-attention | `src/stage2/models/camera.py` — `compute_plucker_6d_per_token` builds the rays, `PluckerFlipPE_V13` builds the embedding and injects it |
| (14) | Clean reference evidence: encode observed references in their inference-time context and prepend as conditioning tokens at `t=0` | `src/stage2/models/dit.py` → `GAEFlow.forward`, argument `z_ref_clean`; the discard-before-the-head step is in the same forward pass |

**A note on Eq. (13).** `compute_plucker_6d_per_token` takes a
`normalize_moment` flag that defaults to `False`. Training performs per-batch
translation normalization (max |t| = 1) at the dataset level, which already
keeps ‖m‖ in a healthy range, so the explicit median rescale is off by default.
If you feed unnormalized metric cameras, enable it.

**Temporal models.** `src/stage2/models/dit_temporal.py` → `GAEFlowTemporal`
adds frame-axis rotary embeddings on top of `GAEFlow`; `src/stage2/models/temporal.py`
installs the spatio-temporal RoPE.

## Evaluation (Section 4)

| Tables | Metric | Where |
|---|---|---|
| 1, 2 | ρ (within-scene velocity fraction) | `src/metrics/latent.py` → `velocity_irreducible_variance(..., return_diagnostics=True)` → `diag["rho"]` |
| 1, 2 | κ (covariance condition number), effective rank | `spectral_stats` |
| 1, 2 | LNC@k | `latent_neighbor_consistency` |
| 1, 2 | LDS / CDS / SRSS | `spatial_structure_metrics` |
| 1, 2 | xLNC* (cross-view retrieval) | driven by `scripts/eval_generation.py` / `scripts/eval_3d_consistency.py` |
| 3, 5 | PSNR, SSIM, LPIPS | `src/metrics/image.py` |
| 3, 5 | rFID / rFVD, FID / FVD | feature statistics over a split — `scripts/eval_reconstruction.py`, `scripts/eval_generation.py` |
| 4, 6, 7 | VGGT ATE / RPEt / RPEr / reprojection | `scripts/eval_3d_consistency.py` |
| 6 | MEt3R | `scripts/eval_met3r.py` |
| 4, 7 | depth AbsRel / δ₁, Chamfer, point-map error | `scripts/eval_geometry.py` |

`latent_diagnostics` in `src/metrics/latent.py` computes ρ, κ, effective rank,
LNC@k and LDS/CDS in one call, which is what `scripts/eval_latent.py` uses.

`src/metrics/latent.py` needs a grid large enough for the spatial thresholds:
`near_px = 24` and `far_px = 96` are in *image pixels*, so on an 18² grid at
252 px a cell is 14 px and both near and far pairs exist. On a much coarser grid
LDS comes back as NaN.

## Renaming map

Earlier research filenames used iteration-specific names. The release renames
the classes and modules that appear in a user-facing API; config **keys** are
untouched so checkpoints still load.

| released name | research name |
|---|---|
| `GAECodec` (`src/stage1/gae_codec.py`) | `FeatureVAEv2` (`src/stage1/feature_vae_v2.py`) |
| `DA3Backbone` (`src/stage1/da3.py`) | `RAE_DA3` (`src/stage1/rae_da3.py`) |
| `GAEFlow` (`src/stage2/models/dit.py`) | `DiTwDDTHeadV4_RAE` (`src/stage2/models/DDT_v4_rae.py`) |
| `GAEFlowTemporal` (`.../dit_temporal.py`) | `DiTwDDTHeadV4Temporal_RAE` (`.../DDT_v4_temporal_rae.py`) |
| `src/stage2/models/camera.py` | `plucker_coords_v2.py` + `plucker_flip_pe_v2.py` (merged) |
| `src/stage2/transport/flow.py` | `transport_rae.py` |
| `src/stage2/models/text_encoder.py` | `text_encoder_qwen3.py` |
| `configs/gae_64.yaml` / `gae_128.yaml` | earlier Stage-1 codec training configs (not shipped) |
| `configs/flow_gae64.yaml` / `flow_gae128.yaml` | earlier Stage-2 flow training configs (not shipped) |
| `scripts/train_codec.py` | `src/train_feature_vae_v2.py` |
| `scripts/train_flow.py` | `src/train_latent_diffusion_v4_rae.py` |

Implementation helpers (`RGBHead`, `SelfAttentionBlock`, `PluckerFlipPE_V13`,
`PluckerAttention`, …) keep their original names.
