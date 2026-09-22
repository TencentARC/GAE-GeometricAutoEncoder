# Codebase map (public release)

This tree is the public GAE release. **Public entry points
are `gae/`, `scripts/`, and `configs/`.** Everything else exists so those
entry points can load the paper checkpoints.

## Use these

| Path | Role |
|---|---|
| `gae/` | Installable facade: `GAE.from_configs`, `load_codec`, `load_flow` |
| `scripts/train_codec.py` | Stage 1 |
| `scripts/train_flow.py` | Stage 2 |
| `scripts/eval_*.py` | Paper tables 1–7 |
| `configs/gae_{64,128}.yaml` | Codec |
| `configs/flow_gae{64,128}.yaml` | Flow |
| `src/stage1/gae_codec.py` | `GAECodec` |
| `src/stage2/models/dit.py` | `GAEFlow` |
| `src/utils/train_runtime.py` | DDP / EMA / latent-stats helpers |

## Do not start training from these

| Path | Why it is still here |
|---|---|
| `src/train_flow_from_cache.py` | Historical trainer; helpers moved to `train_runtime.py` |
| `src/disc/` | GAN path; paper codec has no adversarial term (start step is infinite) |
| `src/stage2/models/{ddt_head,token_concat_ddt,plucker_attention,temporal,lightningDiT}.py` | Internal DDT backbone under `GAEFlow`; not a public API |

Config **keys** such as `codec` are unchanged so released `.pt` files
load. Class names follow the paper (`GAECodec`, `GAEFlow`).

## Dataset roots

Set `GAE_DATA_ROOT` to the directory that contains the packed sources
(`re10k_packed`, `dl3dv_packed`, …). Configs interpolate `${data_root}` from
that env var (default `/data/gae`).
