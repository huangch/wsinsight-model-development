#!/usr/bin/env bash
# Full-cycle pooled breast+lung training: annotate (kurtorank) -> segment ->
# transfer -> tile -> split -> train -> validate -> export.
# Trained head + report land under models/<tissue>/.
#
# Usage: bash scripts/train_breast_lung.sh [input_dir] [tissue_scope]
# Env:   TASK, TUNE, SEGMENTER, CP_BATCH, CP_FLOW; ENVBIN = conda env bin with
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
TISSUE="${2:-breast,lung}"
TASK="${TASK:-pannuke}"                                 # PanNuke label space (transfer appends _label.csv)
TUNE="${TUNE:-0}"
SEGMENTER="${SEGMENTER:-stardist}"
OUT="$ROOT/models"                                   # all run outputs under models/

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== full cycle ($TISSUE, segmenter=$SEGMENTER, tune=$TUNE) =="
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  --segmenter "$SEGMENTER" \
  --cellpose-batch-size "${CP_BATCH:-4}" \
  --cellpose-flow-threshold "${CP_FLOW:-0}" \
  --transform affine \
  --output "$OUT" \
  --stage-skip report \
  --tune "$TUNE"

echo "Done. Head + report under: $OUT/models/$TISSUE/  +  $OUT/report/$TISSUE/"
