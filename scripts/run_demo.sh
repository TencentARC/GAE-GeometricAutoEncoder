#!/usr/bin/env bash
# One-click demo: venv + weights from Hugging Face + bundled example scenes.
#
#   bash scripts/run_demo.sh
#   bash scripts/run_demo.sh --smoke
#   bash scripts/run_demo.sh --task i2v -- --trajectory orbit
#
# Options:
#   --size {64|128}       checkpoint pair (default: 64)
#   --task {i2v|t2i|all}  which demos to run (default: all)
#   --ckpt-dir DIR        weight directory (default: ckpts)
#   --output DIR          results root (default: results/demo)
#   --smoke               17 views, 25 steps
#   --skip-install        assume the current Python env already has GAE
#   --skip-download       fail instead of fetching missing weights
#   --guidance MODE       T2I: none|cfg|ig|cfg_ig (default: ig)
#   --ig-scale N          T2I internal-guidance scale (default: 2.0)
#   -h, --help
#
# Env: GAE_HF_REPO (default TencentARC/GAE-D64-1B), HF_TOKEN, PYTHON.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SIZE=64
TASK=all
CKPT_DIR=ckpts
OUT=results/demo
SMOKE=0
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
T2I_GUIDANCE=ig
T2I_IG_SCALE=2.0
EXTRA=()

die() { echo "[run_demo] error: $*" >&2; exit 1; }
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)           SIZE="$2"; shift 2 ;;
    --task)           TASK="$2"; shift 2 ;;
    --ckpt-dir)       CKPT_DIR="$2"; shift 2 ;;
    --output)         OUT="$2"; shift 2 ;;
    --smoke)          SMOKE=1; shift ;;
    --skip-install)   SKIP_INSTALL=1; shift ;;
    --skip-download)  SKIP_DOWNLOAD=1; shift ;;
    --guidance)       T2I_GUIDANCE="$2"; shift 2 ;;
    --ig-scale)       T2I_IG_SCALE="$2"; shift 2 ;;
    -h|--help)        usage 0 ;;
    --)               shift; EXTRA=("$@"); break ;;
    *)                die "unknown option '$1' (use --help)" ;;
  esac
done

[[ "$SIZE" == "64" || "$SIZE" == "128" ]] || die "--size must be 64 or 128"
[[ "$TASK" == "i2v" || "$TASK" == "t2i" || "$TASK" == "all" ]] \
  || die "--task must be i2v, t2i, or all"

HF_REPO="${GAE_HF_REPO:-TencentARC/GAE-D64-1B}"

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "$PYTHON"; return
  fi
  local cand ver
  for cand in python3.10 python3.11 python3.12 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    case "$ver" in
      3.10|3.11|3.12) echo "$cand"; return ;;
    esac
  done
  die "GAE needs CPython 3.10–3.12. Install one, or set PYTHON=/path/to/python3.10"
}

run() { echo "[run_demo] + $*"; "$@"; }

if [[ "$SKIP_INSTALL" != "1" ]]; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ ! -x .venv/bin/python ]]; then
      PY="$(pick_python)"
      echo "[run_demo] creating .venv with $PY"
      run "$PY" -m venv .venv
      # shellcheck disable=SC1091
      source .venv/bin/activate
      run pip install -U pip
      run pip install -e .
    else
      # shellcheck disable=SC1091
      source .venv/bin/activate
    fi
  fi
  if ! python -c "import torch, omegaconf, cv2, gae" >/dev/null 2>&1; then
    run pip install -e .
  fi
fi

python - <<'PY'
import sys, torch
print(f"[run_demo] python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[run_demo] warning: no CUDA; generation will be very slow or fail", file=sys.stderr)
PY

need=0
for f in \
    "$CKPT_DIR/flow_gae${SIZE}.pt" \
    "$CKPT_DIR/gae_${SIZE}.pt" \
    "$CKPT_DIR/latent_stats_gae_${SIZE}.pt" \
    "$CKPT_DIR/da3_stats_giant_5ds.tar"; do
  [[ -f "$f" ]] || need=1
done
if [[ "$need" == "1" ]]; then
  [[ "$SKIP_DOWNLOAD" == "1" ]] && die "missing weights under $CKPT_DIR"
  echo "[run_demo] downloading $HF_REPO -> $CKPT_DIR"
  run python scripts/download_checkpoints.py --repo "$HF_REPO" --size "$SIZE" --out-dir "$CKPT_DIR"
fi

mkdir -p model_stats/da3_giant_5ds
if [[ ! -f model_stats/da3_giant_5ds/normalization_stats_level0.pt ]]; then
  [[ -f "$CKPT_DIR/da3_stats_giant_5ds.tar" ]] || die "missing $CKPT_DIR/da3_stats_giant_5ds.tar"
  echo "[run_demo] extracting DA3 feature stats"
  tar -xf "$CKPT_DIR/da3_stats_giant_5ds.tar" -C model_stats/da3_giant_5ds
fi

FLOW_CKPT="$CKPT_DIR/flow_gae${SIZE}.pt"
CODEC_CKPT="$CKPT_DIR/gae_${SIZE}.pt"
FLOW_CFG="configs/flow_gae${SIZE}.yaml"
CODEC_CFG="configs/gae_${SIZE}.yaml"

I2V_FLAGS=(--total-views 81)
[[ "$SMOKE" == "1" ]] && I2V_FLAGS=(--num-views 17 --total-views 17 --sample-steps 25)

run_i2v() {
  shopt -s nullglob
  local scenes=(examples/scenes/*.jpg)
  [[ ${#scenes[@]} -gt 0 ]] || die "no examples/scenes/*.jpg"
  echo "[run_demo] I2V: ${#scenes[@]} scene(s), GAE-${SIZE}"
  local img name prompt
  for img in "${scenes[@]}"; do
    name="$(basename "$img" .jpg)"
    prompt="examples/scenes/${name}.txt"
    [[ -f "$prompt" ]] || die "missing $prompt"
    echo "[run_demo] --- $name ---"
    run python scripts/generate.py \
      --image "$img" --prompt-file "$prompt" \
      --flow-ckpt "$FLOW_CKPT" --codec-ckpt "$CODEC_CKPT" \
      --config "$FLOW_CFG" --codec-config "$CODEC_CFG" \
      --output "$OUT/i2v/${name}" \
      "${I2V_FLAGS[@]}" "${EXTRA[@]}"
  done
}

run_t2i() {
  local prompts=examples/t2i_prompts.txt
  [[ -f "$prompts" ]] || die "missing $prompts"
  echo "[run_demo] T2I: $prompts, GAE-${SIZE}"
  run python scripts/generate_t2i.py \
    --config "$FLOW_CFG" --flow-ckpt "$FLOW_CKPT" --codec-ckpt "$CODEC_CKPT" \
    --prompts-file "$prompts" --output "$OUT/t2i" --save-pointcloud \
    --guidance "$T2I_GUIDANCE" --ig-scale "$T2I_IG_SCALE"
}

case "$TASK" in
  i2v) run_i2v ;;
  t2i) run_t2i ;;
  all) run_i2v; run_t2i ;;
esac

echo "[run_demo] done -> $OUT"
