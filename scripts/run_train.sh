#!/usr/bin/env bash
# One-click GAE training.
#
#   Stage 1  codec  ->  configs/gae_<size>.yaml       (scripts/train_codec.py)
#   Stage 2  flow   ->  configs/flow_gae<size>.yaml    (scripts/train_flow.py)
#
# Flow training standardizes the codec posterior mean (paper Eq. 15) and reads
# ckpts/latent_stats_gae_<size>.pt. If that file is missing this script computes
# it from a codec checkpoint before launching the flow trainer.
#
# Usage:
#   scripts/run_train.sh [options] [-- extra trainer args]
#
#   --size {64|128}            latent dim                        (default: 64)
#   --stage {codec|flow|both}  what to train                     (default: codec)
#   --gpus N                   GPUs for torchrun                  (default: 8)
#   --cotrain-t2i              (flow) interleave T2I into the i2v loop
#   --codec-ckpt PATH          codec ckpt used to build latent stats for the
#                              flow stage when they are missing
#                              (default: ckpts/gae_<size>.pt)
#   --stats-batches N          batches for compute_latent_stats  (default: 500)
#   -h, --help
#
# Anything after '--' is forwarded verbatim to the underlying trainer, e.g.
#   scripts/run_train.sh --stage flow --size 64 -- --results-dir results/my-run
#
# Examples:
#   scripts/run_train.sh --stage codec --size 64  --gpus 8
#   scripts/run_train.sh --stage flow  --size 128 --gpus 8 --cotrain-t2i
#   scripts/run_train.sh --stage both  --size 64  --codec-ckpt ckpts/gae_64.pt
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZE=64
STAGE=codec
GPUS=8
COTRAIN_T2I=0
CODEC_CKPT=""
STATS_BATCHES=500
EXTRA=()

die() { echo "[run_train] error: $*" >&2; exit 1; }
usage() { sed -n '2,32p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)         SIZE="$2"; shift 2 ;;
    --stage)        STAGE="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --cotrain-t2i)  COTRAIN_T2I=1; shift ;;
    --codec-ckpt)   CODEC_CKPT="$2"; shift 2 ;;
    --stats-batches) STATS_BATCHES="$2"; shift 2 ;;
    -h|--help)      usage 0 ;;
    --)             shift; EXTRA=("$@"); break ;;
    *)              die "unknown option '$1' (use --help)" ;;
  esac
done

[[ "$SIZE" == "64" || "$SIZE" == "128" ]] || die "--size must be 64 or 128"
[[ "$STAGE" == "codec" || "$STAGE" == "flow" || "$STAGE" == "both" ]] \
  || die "--stage must be codec, flow, or both"

CODEC_CFG="configs/gae_${SIZE}.yaml"
FLOW_CFG="configs/flow_gae${SIZE}.yaml"
STATS_PT="ckpts/latent_stats_gae_${SIZE}.pt"
[[ -n "$CODEC_CKPT" ]] || CODEC_CKPT="ckpts/gae_${SIZE}.pt"

run() { echo "[run_train] + $*"; "$@"; }

train_codec() {
  echo "[run_train] === Stage 1 codec (d${SIZE}) ==="
  run python scripts/train.py codec --size "$SIZE" --gpus "$GPUS" "${EXTRA[@]}"
}

ensure_latent_stats() {
  if [[ -f "$STATS_PT" ]]; then
    echo "[run_train] latent stats present: $STATS_PT"
    return
  fi
  echo "[run_train] latent stats missing: $STATS_PT"
  [[ -f "$CODEC_CKPT" ]] || die "cannot build latent stats: codec ckpt not found at '$CODEC_CKPT' (pass --codec-ckpt, or fetch stats with scripts/download_checkpoints.py)"
  echo "[run_train] computing latent stats from $CODEC_CKPT ..."
  mkdir -p "$(dirname "$STATS_PT")"
  run python scripts/compute_latent_stats.py \
    --config "$CODEC_CFG" --codec-ckpt "$CODEC_CKPT" \
    --num-batches "$STATS_BATCHES" --output "$STATS_PT"
}

train_flow() {
  echo "[run_train] === Stage 2 flow (d${SIZE}) ==="
  ensure_latent_stats
  local args=(flow --size "$SIZE" --gpus "$GPUS")
  [[ "$COTRAIN_T2I" == "1" ]] && args+=(--cotrain-t2i)
  run python scripts/train.py "${args[@]}" "${EXTRA[@]}"
}

case "$STAGE" in
  codec) train_codec ;;
  flow)  train_flow ;;
  both)
    train_codec
    echo "[run_train] NOTE: 'both' will build latent stats from '$CODEC_CKPT'."
    echo "[run_train]       Point --codec-ckpt at the codec checkpoint you just trained if it is not there yet."
    train_flow
    ;;
esac

echo "[run_train] done."
