#!/usr/bin/env bash
# Example 2 of 3 -- SINGLE-TISSUE training with a TILE-LEVEL split.
#
# Trains one tissue-specific head. --by-tile matters most here: a single
# tissue usually has only a handful of slides, so a whole-slide holdout would
# hand one sample's entire cell-type profile to validation.
#
# Usage: bash scripts/train_one_tissue_by_tile.sh <tissue> [input_dir] [output_dir]
#   e.g. bash scripts/train_one_tissue_by_tile.sh breast
#
# Tissue names are the directory names under the input tree (bone, brain,
# breast, cervix, colorectal, heart, kidney, liver, lung, lymph_node, ovary,
# pancreas, prostate, skin, tonsil).
#
# Env:
#   TASK       label space (default pantissue). The
#              celltype_assignment_<TASK>_label.csv files already exist under
#              each sample's outs/, so the annotate stage finds nothing to do
#              and returns immediately -- kurtorank is not invoked.
#   SEGMENTER  stardist (default) | cellpose
#   STARDIST_MODEL_DIR  parent of the csbdeep model folder, e.g.
#              ~/.keras/models/StarDist2D. Auto-detected there when unset.
#   VAL_FRAC   validation fraction (default 0.20)
#   SEED       split seed (default 42)
#   TUNE       auto-tune iterations after training (default 0 = off)
#   STAGE_ONLY run only these stages, e.g. STAGE_ONLY="segment transfer"
#   STAGE_SKIP run everything except these, e.g. STAGE_SKIP="annotate"
#   GPUS       device index, or 'cpu' (default auto)
#   ENVBIN     conda env bin holding wsitrain + torch
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train_one_tissue_by_tile.sh <tissue> [input_dir] [output_dir]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                    # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
ENVBIN="${ENVBIN:-/opt/anaconda3/envs/wsinsight/bin}"
export PATH="$ENVBIN:$PATH"
export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

TISSUE="$1"
case "$TISSUE" in
  *,*|*+*|pantissue)
    echo "ERROR: '$TISSUE' is not a single tissue." >&2
    echo "       Use train_tissues_by_tile.sh for a subset, or" >&2
    echo "       train_pantissue_by_tile.sh for all tissues." >&2
    exit 2 ;;
esac

INPUT="${2:-$ROOT/data/xenium}"
OUT="${3:-$ROOT/models/${TISSUE}_by_tile}"           # own scope: manifest is per output+tissue
TASK="${TASK:-pantissue}"
SEGMENTER="${SEGMENTER:-stardist}"
VAL_FRAC="${VAL_FRAC:-0.20}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
GPUS="${GPUS:-auto}"

STAGE_FLAGS=()
[ -n "${STAGE_ONLY:-}" ] && STAGE_FLAGS=(--stage-only $STAGE_ONLY)
[ -n "${STAGE_SKIP:-}" ] && STAGE_FLAGS=(--stage-skip $STAGE_SKIP)

SD_FLAGS=()
[ -n "${STARDIST_MODEL_DIR:-}" ] && SD_FLAGS=(--stardist-model-dir "$STARDIST_MODEL_DIR")

[ -d "$INPUT" ] || { echo "ERROR: input dir not found: $INPUT" >&2; exit 1; }
[ -d "$INPUT/$TISSUE" ] || { echo "ERROR: no such tissue dir: $INPUT/$TISSUE" >&2; exit 1; }
[ -d "$CELLVIT_ROOT" ] || { echo "ERROR: CELLVIT_ROOT not found: $CELLVIT_ROOT" >&2; exit 1; }

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== $TISSUE, tile-level split (task=$TASK, segmenter=$SEGMENTER, val_frac=$VAL_FRAC) =="
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  --segmenter "$SEGMENTER" \
  "${SD_FLAGS[@]}" \
  --transform affine \
  --by-tile \
  --val-frac "$VAL_FRAC" \
  --seed "$SEED" \
  --output "$OUT" \
  "${STAGE_FLAGS[@]}" \
  --tune "$TUNE" \
  --gpus "$GPUS"

echo
echo "Done."
echo "  model  : $OUT/models/$TISSUE/main/"
echo "  report : $OUT/report/$TISSUE/"
