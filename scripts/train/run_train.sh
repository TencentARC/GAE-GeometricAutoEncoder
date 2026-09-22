#!/usr/bin/env bash
# One-click GAE training.
#
#   Stage 1  codec  ->  configs/gae_<size>.yaml       (scripts/train/train_codec.py)
#   Stage 2  flow   ->  configs/flow_gae<size>.yaml    (scripts/train/train_flow.py)
#
# Flow training standardizes the codec posterior mean (paper Eq. 15) and reads
# ckpts/latent_stats_gae_<size>.pt. With --stage both, this script finds the
# checkpoint produced by Stage 1, computes the stats, then starts Stage 2.
#
# Usage:
#   scripts/train/run_train.sh [options] [-- extra trainer args]
#
#   --size {64|128}            latent dim                        (default: 64)
#   --stage {codec|flow|both}  what to train                     (default: codec)
#   --gpus N                   GPUs for torchrun                  (default: 8)
#   --nnodes N                 training nodes                     (default: 1)
#   --node-rank N              rank of this node                  (default: 0)
#   --master-addr HOST         rank-0 rendezvous address          (default: 127.0.0.1)
#   --master-port PORT         rendezvous port                   (default: 29500)
#   --cotrain-t2i              (flow) interleave T2I into the i2v loop
#   --codec-ckpt PATH          codec ckpt used to build latent stats for the
#                              flow stage when they are missing; stage both
#                              auto-detects the new Stage 1 checkpoint
#   --stats-batches N          batches for compute_latent_stats  (default: 500)
#   -h, --help
#
# Anything after '--' is forwarded verbatim to the underlying trainer, e.g.
#   scripts/train/run_train.sh --stage flow --size 64 -- --results-dir results/my-run
#
# Examples:
#   scripts/train/run_train.sh --stage codec --size 64  --gpus 8
#   scripts/train/run_train.sh --stage flow  --size 128 --gpus 8 --cotrain-t2i
#   scripts/train/run_train.sh --stage both  --size 64
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SIZE=64
STAGE=codec
GPUS=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
COTRAIN_T2I=0
CODEC_CKPT=""
FORCE_STATS=0
STATS_BATCHES=500
EXTRA=()

die() { echo "[run_train] error: $*" >&2; exit 1; }
usage() { sed -n '2,32p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)         SIZE="$2"; shift 2 ;;
    --stage)        STAGE="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --nnodes)       NNODES="$2"; shift 2 ;;
    --node-rank)    NODE_RANK="$2"; shift 2 ;;
    --master-addr)  MASTER_ADDR="$2"; shift 2 ;;
    --master-port)  MASTER_PORT="$2"; shift 2 ;;
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
DEFAULT_CODEC_CKPT="ckpts/gae_${SIZE}.pt"

run() { echo "[run_train] + $*"; "$@"; }

latest_codec_checkpoint() {
  local root="results/gae-${SIZE}-codec"
  [[ -d "$root" ]] || return 0
  find "$root" -type f -path '*/checkpoints/*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 1 | cut -d' ' -f2-
}

train_codec() {
  echo "[run_train] === Stage 1 codec (d${SIZE}) ==="
  run python scripts/train/train.py codec --size "$SIZE" --gpus "$GPUS" --nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" "${EXTRA[@]}"
  if [[ "$STAGE" == "both" ]]; then
    FORCE_STATS=1
    local latest
    latest="$(latest_codec_checkpoint || true)"
    if [[ -n "$latest" ]]; then
      CODEC_CKPT="$latest"
      echo "[run_train] using newly trained codec checkpoint: $CODEC_CKPT"
    elif [[ -z "$CODEC_CKPT" ]]; then
      CODEC_CKPT="$DEFAULT_CODEC_CKPT"
    fi
  fi
}

ensure_latent_stats() {
  if [[ "$FORCE_STATS" != "1" && -f "$STATS_PT" ]]; then
    echo "[run_train] latent stats present: $STATS_PT"
    return
  fi
  echo "[run_train] latent stats missing: $STATS_PT"
  [[ -n "$CODEC_CKPT" ]] || CODEC_CKPT="$DEFAULT_CODEC_CKPT"
  [[ -f "$CODEC_CKPT" ]] || die "cannot build latent stats: codec ckpt not found at '$CODEC_CKPT' (pass --codec-ckpt, or fetch stats with scripts/demo/download_checkpoints.py)"
  echo "[run_train] computing latent stats from $CODEC_CKPT ..."
  mkdir -p "$(dirname "$STATS_PT")"
  run python scripts/train/compute_latent_stats.py \
    --config "$CODEC_CFG" --codec-ckpt "$CODEC_CKPT" \
    --num-batches "$STATS_BATCHES" --output "$STATS_PT"
}

train_flow() {
  echo "[run_train] === Stage 2 flow (d${SIZE}) ==="
  ensure_latent_stats
  local args=(flow --size "$SIZE" --gpus "$GPUS" --nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT")
  [[ "$COTRAIN_T2I" == "1" ]] && args+=(--cotrain-t2i)
  run python scripts/train/train.py "${args[@]}" "${EXTRA[@]}"
}

case "$STAGE" in
  codec) train_codec ;;
  flow)  train_flow ;;
  both)
    train_codec
    echo "[run_train] generating latent stats from '$CODEC_CKPT' before Flow/DiT."
    train_flow
    ;;
esac

echo "[run_train] done."
