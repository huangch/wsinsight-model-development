#!/usr/bin/env bash
# Full-cycle pan-tissue training: annotate -> segment -> transfer -> tile ->
# split -> train -> validate -> export -> report, pooling EVERY discovered
# tissue into one head. Includes auto-tune and cellpose OOM guards.
#
# Usage: bash scripts/train_pantissue.sh [input_dir] [output_dir]
# Env:   TASK (label space, default pantissue), TUNE, BY_SLIDE, SEGMENTER,
#        CP_BATCH, CP_FLOW; ENVBIN = conda env bin with
#        wsitrain+cellpose+kurtorank+torch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                     # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
ENVBIN="${ENVBIN:-/opt/anaconda3/envs/wsinsight/bin}"
export PATH="$ENVBIN:$PATH"
export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

INPUT="${1:-$ROOT/data/xenium}"
OUT="${2:-$ROOT/models}"                             # manifest.json is per --output, so keep scopes apart
TASK="${TASK:-pantissue}"                            # transfer appends _label.csv
TUNE="${TUNE:-6}"
BY_SLIDE="${BY_SLIDE:-true}"                         # slide-level holdout; false = tile-level split
SEGMENTER="${SEGMENTER:-stardist}"

if [ "$BY_SLIDE" = "true" ]; then SPLIT_FLAG=--by-slide; else SPLIT_FLAG=--by-tile; fi

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue pantissue || true

echo "== full cycle (pantissue, task=$TASK, by_slide=$BY_SLIDE, tune=$TUNE) =="
wsitrain run \
  --input "$INPUT" \
  --tissue pantissue \
  --task "$TASK" \
  "$SPLIT_FLAG" \
  --segmenter "$SEGMENTER" \
  --cellpose-batch-size "${CP_BATCH:-4}" \
  --cellpose-flow-threshold "${CP_FLOW:-0}" \
  --transform affine \
  --output "$OUT" \
  --tune "$TUNE" \
  --gpus auto

echo "Done. Head + report under: $OUT/models/pantissue/  +  $OUT/report/pantissue/"
