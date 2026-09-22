# Public scripts

For the project overview and a step-by-step tutorial, see the root
[`README.md`](../README.md). Run commands from the repository root and set
`GAE_DATA_ROOT` to the packed-data root described in [`docs/DATA.md`](../docs/DATA.md).

CLIs are grouped by role:

- [`demo/`](demo/) — Hugging Face download, I2V, T2I, `run_demo.sh`
- [`train/`](train/) — codec / flow trainers, `run_train.sh`
- [`eval/`](eval/) — paper tables, `run_eval.sh`
- [`data/`](data/) — packing, preprocessing, DA3 metric poses

## One-click

Three convenience wrappers drive the Python entry points end-to-end:

```bash
# Demo: venv + pip install -e . + Hugging Face weights + I2V/T2I
bash scripts/demo/run_demo.sh
bash scripts/demo/run_demo.sh --smoke --output results/demo_smoke

# Train: Stage 1 codec, then latent stats, then Stage 2 flow (or use `both`)
scripts/train/run_train.sh --stage codec --size 64 --gpus 8
scripts/train/run_train.sh --stage flow  --size 128 --gpus 8 --cotrain-t2i
scripts/train/run_train.sh --stage both  --size 64

# Eval: default tasks recon,latent,gen (GAE ckpts + data only)
scripts/eval/run_eval.sh --size 64
# Add 3D-consistency / MEt3R / geometry (external recon models via env vars)
VGGT_CKPT=ckpts/vggt.pt PI3_CKPT=ckpts/pi3.pt \
  scripts/eval/run_eval.sh --size 64 --tasks gen,consistency,met3r,geometry
```

`bash scripts/demo/run_demo.sh --help` (and the train/eval wrappers) lists flags.
Extra arguments after `--` are forwarded to `generate.py`.

## Weights

GAE-64 weights live on Hugging Face at
[`TencentARC/GAE-D64-1B`](https://huggingface.co/TencentARC/GAE-D64-1B).
You can either let the demo pull them, download them into `ckpts/`, or load
them through the Python package.

**1. One-click demo (downloads if `ckpts/` is empty)**

```bash
bash scripts/demo/run_demo.sh
```

Override the repo with `GAE_HF_REPO` if needed. Default is `TencentARC/GAE-D64-1B`.

**2. Download into `ckpts/`**

```bash
python scripts/demo/download_checkpoints.py                  # GAE-64 codec + flow + stats
python scripts/demo/download_checkpoints.py --list
python scripts/demo/download_checkpoints.py --subset codec
```

**3. Hugging Face / Python package**

```python
from gae import GAE

gae_model = GAE.from_pretrained("TencentARC/GAE-D64-1B")
```

```bash
python scripts/demo/generate.py \
  --image examples/scenes/forest_lake_trail.jpg \
  --prompt-file examples/scenes/forest_lake_trail.txt \
  --hf-repo TencentARC/GAE-D64-1B \
  --output results/demo
```

`from_pretrained` and `--hf-repo` both call `huggingface_hub` and cache files
under `ckpts/` (or `--ckpt-dir`).

## Data

```bash
python scripts/data/prepare_data.py re10k --source /datasets/RealEstate10K --output "$GAE_DATA_ROOT/re10k_packed" --split train --workers 8
python scripts/data/prepare_data.py dl3dv --source /datasets/DL3DV-10K --output "$GAE_DATA_ROOT/dl3dv_packed" --workers 8
python scripts/data/preprocess_scannetpp.py --source /datasets/scannetpp --output "$GAE_DATA_ROOT/scannetpp_preprocessed" --num-shards 8 --shard-index 0
python scripts/data/preprocess_mvssynth.py  --source /datasets/MVS-Synth/GTAV_540 --output "$GAE_DATA_ROOT/mvssynth_packed"


# Required before Flow/DiT training
python scripts/data/export_da3_metric_poses.py \
  --dataset re10k_packed --root "$GAE_DATA_ROOT/re10k_packed" \
  --device cuda:0 --skip-existing

# Optional: pre-build loader index caches so rank0 does not block torchrun
# on a cold scan at the start of codec training (idempotent).
python scripts/data/build_dataset_index.py --config configs/gae_64.yaml
```

### Text-to-image co-training data

```bash
python scripts/data/prepare_t2i_data.py blip3o   --output "$GAE_DATA_ROOT/BLIP3o" --splits long short journeydb
python scripts/data/prepare_t2i_data.py imagenet --mode arrow --output "$GAE_DATA_ROOT/imagenet-1k"   # gated: huggingface-cli login
python scripts/data/prepare_t2i_data.py verify   --config configs/gae_64.yaml
```

See [`docs/DATA.md`](../docs/DATA.md) for the ImageNet WebDataset packing alternative and layout.

## Training

```bash
python scripts/train/train.py codec --size 64 --gpus 8
python scripts/train/train.py codec --size 128 --gpus 8
python scripts/train/train.py flow --size 64 --gpus 8
python scripts/train/train.py flow --size 128 --gpus 8 --cotrain-t2i   # i2v + T2I
```

`--gpus` is the number of GPU processes per node. For two 8-GPU nodes, run
the same command on both nodes with a unique `--node-rank`:

```bash
# node 0
python scripts/train/train.py codec --size 64 --gpus 8 --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --master-port 29500

# node 1
python scripts/train/train.py codec --size 64 --gpus 8 --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 --master-port 29500
```

All nodes must use the same rendezvous settings and equivalent data/config paths. Use a shared results directory when checkpoints must survive node-local storage.
`train_codec.py` and `train_flow.py` are the underlying trainers. Text-to-image
is co-trained *inside* the flow (i2v/t2v) model: `--cotrain-t2i` interleaves
single-image T2I steps into the multi-view loop (tune with `--t2i-every-k`). The
codec has its own optional `cotrain_t2i` block (RGB-decoder text alignment),
enabled via the config or `COTRAIN_T2I=1`.

## Latent statistics (required after codec, before Flow/DiT)

After Stage 1 codec training, compute the per-channel mean/std used by Stage 2
to standardize codec latents (Eq. 15):

```bash
python scripts/train/compute_latent_stats.py --config configs/gae_64.yaml \
  --codec-ckpt <trained_codec_checkpoint.pt> \
  --num-batches 500 --output ckpts/latent_stats_gae_64.pt
```

Then start Stage 2 Flow/DiT; its config reads `ckpts/latent_stats_gae_64.pt`.
Use `--full-cov` only when the text-to-image recipe needs the whitening operator.


## Codec evaluation

```bash
python scripts/eval/eval_reconstruction.py --config configs/gae_64.yaml --vae-ckpt ckpts/gae_64.pt --dataset re10k_packed --data-root "$GAE_DATA_ROOT/re10k_packed/test"
python scripts/eval/encode_latents.py --input "$GAE_DATA_ROOT/re10k_packed/test" --config configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt --output results/gae64_latents.pt
python scripts/eval/eval_latent.py --latents gae64=results/gae64_latents.pt --json results/gae64_latent_metrics.json
```

`eval_reconstruction.py` reports feature, RGB, depth, and recovered-camera
quality. `eval_geometry.py` reports direct depth/pose/point-cloud metrics from
geometry dumps.

## Generation

One image + prompt -> multi-frame video + point cloud:

```bash
python scripts/demo/generate.py --image examples/scenes/forest_lake_trail.jpg --prompt-file examples/scenes/forest_lake_trail.txt --hf-repo TencentARC/GAE-D64-1B --output results/demo
```

Pure prompt -> image + depth + point cloud (single frame, via the codec RGB + DPT heads):

```bash
python scripts/demo/generate_t2i.py --hf-repo TencentARC/GAE-D64-1B --prompts-file examples/t2i_prompts.txt --output results/t2i
```

Tensor-level API demo (reconstruct / sample, also writes RGB + depth + ply):

```bash
python examples/generate_min.py --image examples/scenes/forest_lake_trail.jpg --codec-cfg configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt --flow-cfg configs/flow_gae64.yaml --flow-ckpt ckpts/flow_gae64.pt --output results/generate_min
```

The generated MP4/PLY (or PNG + `_depth.png` + `_pointcloud.ply`) are written below `--output`. For dataset-scale
metrics use `eval_generation.py`, `eval_3d_consistency.py`, and `eval_met3r.py`. All of these generation
entry points dump depth and a point cloud by default (`--no-pointcloud` to skip).

## Generation evaluation

```bash
# Table 5 (FVD/FID/LPIPS/PSNR/SSIM) + dump point clouds & geometry
python scripts/eval/eval_generation.py --config configs/flow_gae64.yaml \
  --dit-ckpt ckpts/flow_gae64.pt --vae-ckpt ckpts/gae_64.pt \
  --dataset re10k_packed --data-root "$GAE_DATA_ROOT/re10k_packed" \
  --mode generate --cond-num 1 --num-scenes 64 --num-views 9 \
  --sample-steps 50 --cfg-scale 2.0 \
  --output-dir results/gen_gae64 --save-pointcloud --dump-geometry

# Table 6 (VGGT ATE/RPE + MEt3R); external recon models required
python scripts/eval/eval_3d_consistency.py --pred-dir results/gen_gae64 \
  --dataset re10k_packed --data-root "$GAE_DATA_ROOT/re10k_packed" \
  --vggt-ckpt ckpts/vggt.pt --da3-ckpt ckpts/da3_giant.pt \
  --output-name results/gen_gae64/consistency.json
python scripts/eval/eval_met3r.py --pred-dir results/gen_gae64 --output results/gen_gae64/met3r.json

# Table 7 (depth/point-map/pose from sampled latents); needs Pi3
python scripts/eval/eval_geometry.py --geom-dir results/gen_gae64 \
  --pi3-ckpt ckpts/pi3.pt --output-name results/gen_gae64/geometry.json
```

The 3D-consistency and geometry steps consume the `--output-dir` from the first
command. Released weights are a continued-training model, so numbers differ from
the paper — see the Results note in the root [`README.md`](../README.md).
One-click: `scripts/eval/run_eval.sh --size 64 --tasks gen,consistency,met3r,geometry`.

## Checks

```bash
python scripts/eval/smoke_test.py
python scripts/eval/validate_configs.py
python scripts/demo/download_checkpoints.py --list
```
