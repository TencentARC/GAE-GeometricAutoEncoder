# Data preparation

Set one portable root before training:

```bash
export GAE_DATA_ROOT=/data/gae
```

## RealEstate10K

Expected raw layout:

```text
RealEstate10K/
  train_caption.json                 # optional: scene_id -> caption
  test_caption.json                  # optional
  train/<shard>/<scene>/
    123456.png
    ...
    transforms.json
  test/<shard>/<scene>/...
```

`transforms.json` must contain `fl_x`, `fl_y`, `cx`, `cy`, and
`frames[*].{file_path,transform_matrix}`. The transforms are interpreted as
OpenCV camera-to-world matrices.

```bash
python scripts/data/prepare_data.py re10k \
  --source /datasets/RealEstate10K \
  --output "$GAE_DATA_ROOT/re10k_packed" \
  --split train --workers 8

python scripts/data/prepare_data.py re10k \
  --source /datasets/RealEstate10K \
  --output "$GAE_DATA_ROOT/re10k_packed" \
  --split test --workers 8
```

## DL3DV

Expected raw layout:

```text
DL3DV-10K/<split>/<scene>/[nerfstudio/]/
  images_4/frame_00001.png
  ...
  transforms.json
  wan2_caption.json                  # optional
```

DL3DV transforms are interpreted as OpenGL camera-to-world and converted to
OpenCV. Intrinsics are rescaled from the dimensions in `transforms.json` to
the actual `images_4` dimensions.

```bash
python scripts/data/prepare_data.py dl3dv \
  --source /datasets/DL3DV-10K \
  --output "$GAE_DATA_ROOT/dl3dv_packed" \
  --workers 8
```


## ScanNet++

ScanNet++ uses its own loader (`ScanNetppRGB_Multi`) and keeps a depth sidecar,
so it has a dedicated preprocessor. Expected raw layout per scene:

```text
scannetpp/<scene>/
  rgb/frame_*.jpg
  depth/frame_*.png                  # 16-bit mm
  colmap/pose_intrinsic_imu.json     # per-frame c2w (4x4) + intrinsic (3x3)
```

```bash
for i in $(seq 0 7); do
  python scripts/data/preprocess_scannetpp.py \
    --source /datasets/scannetpp --output "$GAE_DATA_ROOT/scannetpp_preprocessed" \
    --num-frames 256 --resolution 504 --depth-resolution 192 \
    --num-shards 8 --shard-index $i &
done; wait
```

Output per scene: `video.mp4`, `meta.json` (frame `name`/`original_idx`/`c2w`/`K`),
and `depth.npz` (uint16 mm). Default sampling is pose-aware (`--sampling-mode
progressive`); pass `--sampling-mode uniform` for evenly-spaced frames.

## MVS-Synth

MVS-Synth packs into the same `VideoMetaScene` schema as RE10K/DL3DV. Expected
raw layout per scene:

```text
MVS-Synth/GTAV_540/<scene>/
  images/<XXXX>.png
  poses/<XXXX>.json                  # {f_x, f_y, c_x, c_y, extrinsic 4x4 (w2c)}
```

The `extrinsic` is world-to-camera; the packer stores `c2w = inv(extrinsic)`.

```bash
python scripts/data/preprocess_mvssynth.py \
  --source /datasets/MVS-Synth/GTAV_540 --output "$GAE_DATA_ROOT/mvssynth_packed"
```

## Stage 2 metric-pose sidecars

Codec training can use the packed scenes directly. Flow/DiT configs additionally
require `meta_da3_metric.json`, generated with DA3NESTED:

```bash
# RE10K train and test
python scripts/data/export_da3_metric_poses.py \
  --dataset re10k_packed --root "$GAE_DATA_ROOT/re10k_packed" \
  --device cuda:0 --skip-existing

# DL3DV
python scripts/data/export_da3_metric_poses.py \
  --dataset dl3dv_packed --root "$GAE_DATA_ROOT/dl3dv_packed" \
  --device cuda:0 --skip-existing
```

Long scenes are split into checkpoint-compatible chunks. RE10K drops the last
32 source frames by default, matching the released flow training recipe. Use
`--num-shards N --shard-index I` to distribute export across GPUs or machines.
For network-backed datasets, pre-build loader indexes before launching many ranks:

```bash
python scripts/data/build_dataset_index.py --config configs/gae_64.yaml
python scripts/data/build_dataset_index.py --config configs/flow_gae64.yaml
```


## Latent statistics (required after codec, before Flow/DiT)

Stage 2 standardizes the codec posterior mean (Eq. 15) using per-channel
mean/std saved to `ckpts/latent_stats_gae_64.pt` (the path the flow configs
read). Download it with `scripts/demo/download_checkpoints.py`, or compute it for a
codec you trained:

```bash
python scripts/train/compute_latent_stats.py \
  --config configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt \
  --num-batches 500 --output ckpts/latent_stats_gae_64.pt
```

`--dataset` restricts the sampling distribution (e.g. `dataset_re10k`) if you do
not have all domains. Add `--full-cov` to also store the full-covariance
whitening operator used by the text-to-image recipe.

## Packed scene schema

Both commands produce:

```text
<output>/<scene>/
  video.mp4
  meta.json
  caption.txt                        # optional
```

`meta.json` stores the frame count, source resolution, and one OpenCV `c2w`
plus `K` matrix per encoded frame. Frame ordering in the MP4 and metadata is
identical. ScanNet++ additionally writes a `depth.npz` sidecar.

Use `--limit N` for a smoke test and `--skip-existing` when resuming.


## Text-to-image (T2I) co-training data

Both the Stage 2 flow (i2v/t2v) model and the Stage 1 codec can co-train a
single-image **text-to-image** branch on this data. In the flow it keeps text
conditioning intact during i2v training (`--cotrain-t2i`); in the codec
(`cotrain_t2i` in `configs/gae_64.yaml`) it makes the RGB head usable for pure
prompt -> image generation (`scripts/demo/generate_t2i.py`).
The branch reads two raw sources through the loaders in `src/data`, so no
bespoke format is needed:

* **BLIP3o-Pretrain** WebDataset tar shards (`src/data/blip3o_wds.py`)
* **ImageNet-1k** as Hugging Face Arrow, captioned by class name
  (`src/data/imagenet_arrow.py`, using `src/data/imagenet_classes.py`)

Prepare both under `$GAE_DATA_ROOT` with `scripts/data/prepare_t2i_data.py`.

### 1. BLIP3o-Pretrain (WebDataset)

```bash
python scripts/data/prepare_t2i_data.py blip3o \
  --output "$GAE_DATA_ROOT/BLIP3o" \
  --splits long short journeydb
```

This downloads the three caption splits from the Hugging Face Hub into the
sub-directories the loader expects:

```text
$GAE_DATA_ROOT/BLIP3o/
  BLIP3o-Pretrain-Long-Caption/*.tar
  BLIP3o-Pretrain-Short-Caption/*.tar
  BLIP3o-Pretrain-JourneyDB/*.tar
```

Override a split's source repo with `--repo long=Org/Repo` if you mirror the
data elsewhere.

### 2. ImageNet-1k

**Option A — Hugging Face Arrow (config default).** ImageNet-1k is gated, so
run `huggingface-cli login` first, then materialize the Arrow cache:

```bash
python scripts/data/prepare_t2i_data.py imagenet --mode arrow \
  --output "$GAE_DATA_ROOT/imagenet-1k"
```

The command prints the exact `imagenet_arrow_root` to set in
`cotrain_t2i.dataset.imagenet_arrow_root`. Captions are the 1000 ImageNet class
names bundled in `src/data/imagenet_classes.py`.

**Option B — packed WebDataset (fast on network filesystems).** Pack a
local ImageNet `train` directory (`<source>/<wnid>/*.JPEG`) into BLIP3o-style
tar shards with class-name captions:

```bash
python scripts/data/prepare_t2i_data.py imagenet --mode wds \
  --source /datasets/imagenet/train \
  --output "$GAE_DATA_ROOT/Blip3o_style/ImageNet-1K-T2I"
```

The `imagenet` split then resolves from any `data_dir` root that contains an
`ImageNet-1K-T2I/` sub-directory. Add `--caption-template` to emit
`"a photo of a <class>"` instead of the bare class name.

### 3. Verify

Confirm a config's `cotrain_t2i` sources resolve to real data:

```bash
python scripts/data/prepare_t2i_data.py verify --config configs/gae_64.yaml
```

It reports each split's tar count (or the ImageNet Arrow status) so you can fix
paths before launching a multi-GPU run.

### 4. Train

```bash
# T2I co-trained inside the flow (i2v) model:
python scripts/train/train.py flow --size 64 --gpus 8 --cotrain-t2i

# Optional: the same data can text-align the Stage 1 codec RGB head:
COTRAIN_T2I=1 python scripts/train/train.py codec --size 64 --gpus 8
```

`--cotrain-t2i` interleaves one single-image T2I step into the i2v loop (tune
with `--t2i-every-k`, `>= 2`). The codec's own `cotrain_t2i` block is enabled via
the config or `COTRAIN_T2I=1`; after codec co-training you can sample images
directly from the RGB head with `scripts/demo/generate_t2i.py` (see `scripts/README.md`).
