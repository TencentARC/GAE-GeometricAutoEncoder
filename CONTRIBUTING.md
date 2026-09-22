# Contributing

Thanks for your interest in GAE. This is a research code release; the notes
below keep changes easy to review.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the `gae` facade and the src/ modules
pip install pytest ruff     # dev tools
```

`pip install -e .` makes both `gae` and the research packages (`stage1`,
`stage2`, `utils`, `metrics`, `video`, `data`, `cut3r_data`, `disc`) importable,
so you no longer need to set `PYTHONPATH` by hand.

## Layout

- `gae/` — small public API. Prefer adding user-facing entry points here.
- `scripts/{demo,train,eval,data}/` — CLIs for generation, training, evaluation, and data prep.
- `src/` — research modules. See [`docs/CODEBASE.md`](docs/CODEBASE.md) for which
  files are load-bearing versus kept only for checkpoint compatibility.
- `configs/` — the released recipes. Keep config *keys* stable so published
  checkpoints keep loading.

## Before you open a PR

```bash
ruff check .
python scripts/eval/validate_configs.py
python scripts/eval/smoke_test.py
pytest -q tests/test_configs.py tests/test_imports.py tests/test_prepare_data.py
```

The full training/eval path needs GPUs, DA3-GIANT, and packed datasets, so CI
only runs the CPU-only config, import, and data tests above.

## Conventions

- Do not hard-code absolute machine paths. Read data roots from `GAE_DATA_ROOT`
  and model/cache locations from documented env vars.
- Keep the public class names (`GAECodec`, `GAEFlow`) and paper equation
  references intact; map any renames in [`docs/METHOD.md`](docs/METHOD.md).
- New optional dependencies go in a `pyproject.toml` extra, imported lazily.
