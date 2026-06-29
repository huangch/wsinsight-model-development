#!/usr/bin/env bash
# Full-cycle: annotate (kurtorank markers-v5, STHELAR label) -> segment ->
# transfer -> tile -> split -> train -> validate -> export, for breast+lung.
# Trained head + report land under cellvit-training/models/<tissue>/.
#
# Usage: bash pipeline/train_breast_lung.sh
# Env:   set ENVBIN to a conda env bin with wsitrain+cellpose+kurtorank+torch.
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
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

INPUT="${1:-$ROOT/data/xenium}"
TISSUE="${2:-breast,lung}"
TASK="${TASK:-sthelar_full}"                          # markers-v5 STHELAR label space
OUT="$ROOT/models"                                   # all run outputs under models/

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== full cycle ($TISSUE) =="
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  --segmenter cellpose \
  --transform affine+bspline \
  --output "$OUT" \
  --from annotate --to export

echo "Done. Head + report under: $OUT/models/$TISSUE/  +  $OUT/report/$TISSUE/"
