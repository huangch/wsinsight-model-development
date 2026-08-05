#!/usr/bin/env bash
# Full-cycle pan-tissue training: annotate (kurtorank markers-v5, STHELAR label)
# -> segment -> transfer -> tile -> split -> train -> validate -> export,
# pooling EVERY discovered tissue into one head.
# Outputs under wsinsight-model-development/models/.
#
# Usage: bash scripts/train_pantissue.sh [input_dir]
# Env:   ENVBIN = conda env bin with wsitrain+cellpose+kurtorank+torch.
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
TASK="${TASK:-sthelar_full}"
OUT="$ROOT/models"

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue pantissue || true

echo "== pan-tissue full cycle =="
wsitrain train \
  --input "$INPUT" \
  --tissue pantissue \
  --task "$TASK" \
  --segmenter cellpose \
  --transform affine+bspline \
  --output "$OUT" \
  --from annotate --to export

echo "Done. Head + report under: $OUT/models/pantissue/  +  $OUT/report/pantissue/"
