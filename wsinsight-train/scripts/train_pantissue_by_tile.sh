#!/usr/bin/env bash
# Example 1 of 3 -- PAN-TISSUE training with a TILE-LEVEL split.
#
# --tissue pantissue pools every sample discovered under the input tree into a
# single head. --by-tile then splits at the tile level, so each slide feeds
# both train and val. Whole-slide holdout (--by-slide) is the stricter test of
# generalisation, but with an uneven per-tissue slide count it tends to put
# entire rare tissues on one side of the split.
#
# Usage: bash scripts/train_pantissue_by_tile.sh [input_dir] [output_dir]
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
#   RUN_SKIP   run everything except these, e.g. RUN_SKIP="annotate". To run a
#              single stage instead, call it directly: `wsitrain segment ...`
#   GPUS       device index, or 'cpu' (default auto)
#   ENVBIN     conda env bin holding wsitrain + torch
set -euo pipefail

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

INPUT="${1:-$ROOT/data/xenium}"
OUT="${2:-$ROOT/models/pantissue_by_tile}"
TISSUE=pantissue
TASK="${TASK:-pantissue}"
SEGMENTER="${SEGMENTER:-stardist}"
VAL_FRAC="${VAL_FRAC:-0.20}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
GPUS="${GPUS:-auto}"

STAGE_FLAGS=()
[ -n "${RUN_SKIP:-}" ] && STAGE_FLAGS=(--run-skip $RUN_SKIP)

SD_FLAGS=()
[ -n "${STARDIST_MODEL_DIR:-}" ] && SD_FLAGS=(--stardist-model-dir "$STARDIST_MODEL_DIR")

[ -d "$INPUT" ] || { echo "ERROR: input dir not found: $INPUT" >&2; exit 1; }
[ -d "$CELLVIT_ROOT" ] || { echo "ERROR: CELLVIT_ROOT not found: $CELLVIT_ROOT" >&2; exit 1; }

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== pantissue, tile-level split (task=$TASK, segmenter=$SEGMENTER, val_frac=$VAL_FRAC) =="
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
