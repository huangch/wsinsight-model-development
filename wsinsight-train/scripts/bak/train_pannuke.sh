#!/usr/bin/env bash
# Pan-tissue training against the PanNuke label vocabulary.
# Usage: scripts/train_pannuke.sh [input_dir] [output_dir]
# Env:   TUNE=N auto-tune iters; BY_SLIDE=false for a tile-level split.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                    # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
INPUT="${1:-$ROOT/data/xenium}"
OUT="${2:-$ROOT/models/pannuke}"                     # manifest.json is per --output
TUNE="${TUNE:-0}"
BY_SLIDE="${BY_SLIDE:-true}"

export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

if [ "$BY_SLIDE" = "true" ]; then SPLIT_FLAG=--by-slide; else SPLIT_FLAG=--by-tile; fi

echo "== pannuke (by_slide=$BY_SLIDE, tune=$TUNE) -> $OUT =="
wsitrain run \
  --input "$INPUT" \
  --tissue pantissue \
  --task pannuke \
  "$SPLIT_FLAG" \
  --transform affine \
  --output "$OUT" \
  --tune "$TUNE" \
  --gpus auto

echo "Done. Report: $OUT/report/pantissue/"
