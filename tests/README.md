# Tests

Lightweight checks that keep the public release runnable. They need no GPU, no
model weights, and no dataset downloads; the heavier ones skip automatically
when an optional dependency (`torch`, `cv2`, `ffmpeg`) is missing.

| test | what it guards |
|---|---|
| `test_configs.py` | every `configs/*.yaml` loads and resolves its `${GAE_DATA_ROOT}` targets, and both a codec (`gae_*`) and a flow (`flow_*`) config exist |
| `test_imports.py` | the `gae` public facade and core research modules import, and `GAE.sample` keeps its public signature (needs `torch`; skipped otherwise) |
| `test_pipeline_smoke.py` | `scripts/eval/smoke_test.py` runs the stubbed end-to-end pipeline with no downloads (needs `torch`; skipped otherwise) |
| `test_prepare_data.py` | `scripts/data/prepare_data.py` emits the packed `video.mp4` + `meta.json` schema the loaders expect, with integer-aware frame ordering (needs `ffmpeg` + `cv2`; skipped otherwise) |

## Run

```bash
pip install pytest omegaconf numpy opencv-python-headless   # add torch for the import/smoke tests
pytest -q tests/
```

## CI

`.github/workflows/ci.yml` runs these on a CPU-only, torch-less runner: the
config and data-prep tests execute there, while the import and smoke tests skip
cleanly (they run wherever `torch` is installed, e.g. a local dev checkout).
