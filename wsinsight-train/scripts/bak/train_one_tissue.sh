#!/usr/bin/env bash
# Single-tissue full training with optional Cellpose-vs-StarDist parity check.
# Usage: scripts/train_one_tissue.sh <input_dir> <tissue> [output_dir]
# Env:   PARITY=1 to also run StarDist tiling parity; TUNE=N for auto-tune iters.
# Requires the `wsitrain` env (torch + cellpose + kurtorank) and $CELLVIT_ROOT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                     # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
INPUT="${1:?input dir}"
TISSUE="${2:-breast}"
OUT="${3:-$ROOT/models/$TISSUE}"                     # manifest.json is per --output, so keep scopes apart
TASK="${TASK:-sthelar_full}"
PARITY="${PARITY:-0}"   # set to 1 to also tile with StarDist for comparison
TUNE="${TUNE:-0}"       # auto-tune iterations after training (0 = off)

export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== full training: $TISSUE =="
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  --segmenter cellpose \
  --cellpose-batch-size "${CP_BATCH:-4}" \
  --cellpose-flow-threshold "${CP_FLOW:-0}" \
  --transform affine \
  --output "$OUT" \
  --tune "$TUNE" \
  --gpus auto

echo "Done. Model + report under: $OUT/models/$TISSUE/  +  $OUT/report/$TISSUE/"

if [ "${PARITY}" = "1" ]; then
  echo "== StarDist parity tile check =="
  wsitrain run --input "$INPUT" --tissue "$TISSUE" --task "$TASK" --segmenter stardist \
    --output "$OUT/stardist_parity" --to tile
  for s in "$OUT" "$OUT/stardist_parity"; do
    n=$(find "$s/trainingset/$TISSUE/train/labels" -name '*.csv' 2>/dev/null | wc -l || true)
    seg=$([ "$s" = "$OUT" ] && echo cellpose || echo stardist)
    echo "  $seg: $n tiles"
  done
fi
