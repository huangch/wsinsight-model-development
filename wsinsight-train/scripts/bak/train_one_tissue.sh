#!/usr/bin/env bash
# Single-tissue full training, with an optional second-segmenter parity check.
# Usage: scripts/train_one_tissue.sh <input_dir> <tissue> [output_dir]
# Env:   PARITY=1 to also tile with the other segmenter; TUNE=N for auto-tune
#        iters; SEGMENTER=cellpose|stardist.
# Requires the `wsitrain` env and $CELLVIT_ROOT.
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
SEGMENTER="${SEGMENTER:-stardist}"

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
  --segmenter "$SEGMENTER" \
  --cellpose-batch-size "${CP_BATCH:-4}" \
  --cellpose-flow-threshold "${CP_FLOW:-0}" \
  --transform affine \
  --output "$OUT" \
  --tune "$TUNE" \
  --gpus auto

echo "Done. Model + report under: $OUT/models/$TISSUE/  +  $OUT/report/$TISSUE/"

if [ "${PARITY}" = "1" ]; then
  # Compare against the segmenter the main run did NOT use.
  if [ "$SEGMENTER" = "cellpose" ]; then OTHER=stardist; else OTHER=cellpose; fi
  echo "== segmentation parity: $SEGMENTER vs $OTHER =="
  wsitrain run --input "$INPUT" --tissue "$TISSUE" --task "$TASK" --segmenter "$OTHER" \
    --transform affine --output "$OUT/${OTHER}_parity" \
    --stage-only annotate segment transfer tile
  for entry in "$OUT|$SEGMENTER" "$OUT/${OTHER}_parity|$OTHER"; do
    d="${entry%%|*}"; seg="${entry##*|}"
    n=$(find "$d/trainingset/$TISSUE/train/labels" -name '*.csv' 2>/dev/null | wc -l || true)
    echo "  $seg: $n tiles"
  done
fi
