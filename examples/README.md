# Examples

## Test scenes (`scenes/`)

`scenes/` holds one reference frame + camera trajectory + prompt per source
dataset, extracted from held-out eval clips (81 views, 672x378, 1 reference
view). For each scene, `<name>.jpg` is frame 0 (the conditioning view), `<name>.txt`
is the text prompt, and `<name>_poses.npz` stores the ground-truth
camera-to-world poses and per-frame intrinsics. `manifest.json` collects all of
these plus the source dataset and license for every scene. The GT clips (`<name>_gt.mp4`) are git-ignored - regenerable
from the eval dumps and not distributed.

This repository ships two Pexels scenes (Pexels License):

| scene | content |
|---|---|
| `forest_lake_trail` | pine-forest lakeside trail |
| `autumn_waterfall` | autumn forest stream + waterfall |

`<name>_poses.npz` contains `c2w` (Nx4x4, OpenCV camera-to-world), `K`
(Nx3x3, pixels), `image_size` (H, W), `cond_num`, and `fps`.

Run a scene through the generator:

```bash
python scripts/generate.py \
  --image examples/scenes/forest_lake_trail.jpg \
  --prompt-file examples/scenes/forest_lake_trail.txt \
  --hf-repo TencentARC/GAE-D64-1B \
  --output results/forest --total-views 81
```

`scripts/generate.py` loads `<name>_poses.npz` next to the image when it exists
(the recorded eval cameras). Pass `--free-rollout` or `--trajectory orbit|spiral|drive|wander`
to ignore GT poses and synthesize a path instead.

## Text-to-image prompts (`t2i_prompts.txt`)

Pure prompt -> image generation has no reference frame or camera, so the T2I
prompts live in a flat list (`t2i_prompts.txt`, one per line) rather than under
`scenes/`. Drive the codec RGB head (plus DPT depth / ply by default) with them:

```bash
python scripts/generate_t2i.py \
  --hf-repo TencentARC/GAE-D64-1B \
  --prompts-file examples/t2i_prompts.txt --output results/t2i
```

Twelve demo captions covering still life, characters, landscape, interior, food, sculpture, and night city scenes.

## `generate_min.py`

A minimal, dependency-light illustration of the `gae.GAE` API (encode /
reconstruct / sample). It writes RGB, DPT depth, and a `.ply` under `--output`
without shelling out to the full evaluator:

```bash
python examples/generate_min.py --image examples/scenes/forest_lake_trail.jpg \
  --codec-cfg configs/gae_64.yaml --codec-ckpt ckpts/gae_64.pt \
  --flow-cfg configs/flow_gae64.yaml --flow-ckpt ckpts/flow_gae64.pt \
  --output results/generate_min
```
