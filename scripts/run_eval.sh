#!/usr/bin/env bash
# One-click GAE evaluation.
#
# Runs the paper eval pipeline for a given latent size against a packed dataset.
# Tasks (comma-separated, via --tasks):
#
#   recon        RGB / feature / depth reconstruction   (eval_reconstruction.py)  [Table 3]
#   latent       intrinsic latent diagnostics           (encode_latents + eval_latent.py) [Tables 1,2]
#   gen          camera-conditioned generation + dumps  (eval_generation.py)      [Table 5]
#   consistency  generated-view 3D consistency          (eval_3d_consistency.py)  [Table 6] *needs VGGT/DA3
#   met3r        MEt3R inconsistency                     (eval_met3r.py)           [Table 6]
#   geometry     depth / point-map / pose from geom dump (eval_geometry.py)        [Tables 4,7] *needs Pi3
#
# Default tasks (recon,latent,gen) only need the shipped GAE checkpoints + data.
# consistency/geometry additionally need external recon models; point at them
# with VGGT_CKPT / DA3_CKPT / PI3_CKPT env vars.
#
# Usage:
#   scripts/run_eval.sh [options]
#
#   --size {64|128}     latent dim                                (default: 64)
#   --tasks LIST        comma-separated task list                 (default: recon,latent,gen)
#   --dataset NAME      packed dataset name                       (default: re10k_packed)
#   --data-root PATH    packed dataset root (has train/ test/)    (default: $GAE_DATA_ROOT/<dataset>)
#   --out DIR           output dir                                (default: results/eval_gae<size>)
#   --num-scenes N      cap scenes for a quick pass               (forwarded where supported)
#   --num-views N       views per scene                           (forwarded where supported)
#   -h, --help
#
# Checkpoints are expected at:
#   ckpts/gae_<size>.pt         codec
#   ckpts/flow_gae<size>.pt     flow
# (fetch with scripts/download_checkpoints.py)
#
# Examples:
#   scripts/run_eval.sh --size 64
#   scripts/run_eval.sh --size 128 --tasks recon,latent --dataset dl3dv_packed
#   VGGT_CKPT=ckpts/vggt.pt PI3_CKPT=ckpts/pi3.pt scripts/run_eval.sh --tasks gen,consistency,met3r,geometry
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIZE=64
TASKS="recon,latent,gen"
DATASET="re10k_packed"
DATA_ROOT=""
OUT=""
NUM_SCENES=""
NUM_VIEWS=""

die() { echo "[run_eval] error: $*" >&2; exit 1; }
usage() { sed -n '2,44p' "$SELF" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)       SIZE="$2"; shift 2 ;;
    --tasks)      TASKS="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --num-scenes) NUM_SCENES="$2"; shift 2 ;;
    --num-views)  NUM_VIEWS="$2"; shift 2 ;;
    -h|--help)    usage 0 ;;
    *)            die "unknown option '$1' (use --help)" ;;
  esac
done

[[ "$SIZE" == "64" || "$SIZE" == "128" ]] || die "--size must be 64 or 128"
[[ -n "$DATA_ROOT" ]] || DATA_ROOT="${GAE_DATA_ROOT:?set GAE_DATA_ROOT or pass --data-root}/${DATASET}"
[[ -n "$OUT" ]] || OUT="results/eval_gae${SIZE}"

CODEC_CFG="configs/gae_${SIZE}.yaml"
FLOW_CFG="configs/flow_gae${SIZE}.yaml"
VAE="ckpts/gae_${SIZE}.pt"
DIT="ckpts/flow_gae${SIZE}.pt"
TEST_DIR="${DATA_ROOT%/}/test"
[[ -d "$TEST_DIR" ]] || TEST_DIR="$DATA_ROOT"   # some packs have no train/test split

# Optional common flags forwarded only when set.
SCENE_ARGS=(); [[ -n "$NUM_SCENES" ]] && SCENE_ARGS+=(--num-scenes "$NUM_SCENES")
VIEW_ARGS=();  [[ -n "$NUM_VIEWS" ]]  && VIEW_ARGS+=(--num-views "$NUM_VIEWS")

mkdir -p "$OUT"
run() { echo "[run_eval] + $*"; "$@"; }
have() { [[ ",$TASKS," == *",$1,"* ]]; }
need_ckpt() { [[ -f "$1" ]] || die "missing checkpoint: $1 (fetch with scripts/download_checkpoints.py)"; }

echo "[run_eval] size=d${SIZE} dataset=${DATASET} data_root=${DATA_ROOT} out=${OUT} tasks=${TASKS}"

if have recon; then
  echo "[run_eval] === recon (Table 3) ==="
  need_ckpt "$VAE"
  run python scripts/eval_reconstruction.py \
    --config "$CODEC_CFG" --vae-ckpt "$VAE" \
    --dataset "$DATASET" --data-root "$TEST_DIR" \
    --output-dir "$OUT/recon" "${SCENE_ARGS[@]}" "${VIEW_ARGS[@]}"
fi

if have latent; then
  echo "[run_eval] === latent diagnostics (Tables 1,2) ==="
  need_ckpt "$VAE"
  run python scripts/encode_latents.py \
    --input "$TEST_DIR" --config "$CODEC_CFG" --codec-ckpt "$VAE" \
    --output "$OUT/latents.pt"
  run python scripts/eval_latent.py \
    --latents "gae${SIZE}=$OUT/latents.pt" --json "$OUT/latent_metrics.json"
fi

if have gen || have consistency || have met3r || have geometry; then
  echo "[run_eval] === generation (Table 5) + dumps ==="
  need_ckpt "$VAE"; need_ckpt "$DIT"
  run python scripts/eval_generation.py \
    --config "$FLOW_CFG" --dit-ckpt "$DIT" --vae-ckpt "$VAE" \
    --dataset "$DATASET" --data-root "$DATA_ROOT" \
    --output-dir "$OUT/gen" --save-pointcloud --dump-geometry \
    "${SCENE_ARGS[@]}" "${VIEW_ARGS[@]}"
fi

if have consistency; then
  echo "[run_eval] === 3D consistency (Table 6) ==="
  args=(--pred-dir "$OUT/gen" --dataset "$DATASET" --data-root "$DATA_ROOT"
        --output-name "$OUT/consistency.json" --csv-output "$OUT/consistency.csv")
  [[ -n "${VGGT_CKPT:-}" ]] && args+=(--vggt-ckpt "$VGGT_CKPT")
  [[ -n "${DA3_CKPT:-}" ]]  && args+=(--da3-ckpt "$DA3_CKPT")
  run python scripts/eval_3d_consistency.py "${args[@]}"
fi

if have met3r; then
  echo "[run_eval] === MEt3R (Table 6) ==="
  run python scripts/eval_met3r.py \
    --pred-dir "$OUT/gen" --output "$OUT/met3r.json" --csv-output "$OUT/met3r.csv"
fi

if have geometry; then
  echo "[run_eval] === geometry (Tables 4,7) ==="
  args=(--geom-dir "$OUT/gen" --output-name "$OUT/geometry.json" --csv-output "$OUT/geometry.csv")
  [[ -n "${PI3_CKPT:-}" ]]  && args+=(--pi3-ckpt "$PI3_CKPT")
  [[ -n "${VGGT_CKPT:-}" ]] && args+=(--vggt-ckpt "$VGGT_CKPT")
  [[ -n "${DA3_CKPT:-}" ]]  && args+=(--da3-ckpt "$DA3_CKPT")
  run python scripts/eval_geometry.py "${args[@]}"
fi

echo "[run_eval] done. results under $OUT/"
