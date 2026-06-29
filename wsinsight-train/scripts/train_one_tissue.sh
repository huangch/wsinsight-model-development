#!/usr/bin/env bash
# One-tissue smoke run + Cellpose-vs-StarDist parity check.
# Usage: scripts/run_one_tissue.sh <input_dir> <tissue> [output_dir]
# Requires the `wsitrain` env (torch + cellpose + kurtorank) and $CELLVIT_ROOT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                     # wsinsight-model-development/
INPUT="${1:?input dir}"
TISSUE="${2:-breast}"
OUT="${3:-$ROOT/models/$TISSUE}"
TASK="${TASK:-sthelar_full}"

export CELLVIT_ROOT="${CELLVIT_ROOT:?set CELLVIT_ROOT to the CellViT-plus-plus checkout}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
wsitrain check --input "$INPUT" --tissue "$TISSUE" || true

echo "== cellpose run =="
wsitrain run --input "$INPUT" --tissue "$TISSUE" --task "$TASK" --segmenter cellpose \
  --output "$OUT/cellpose" --to tile

echo "== stardist run (parity) =="
wsitrain run --input "$INPUT" --tissue "$TISSUE" --task "$TASK" --segmenter stardist \
  --output "$OUT/stardist" --to tile

echo "== compare tile/label counts =="
for s in cellpose stardist; do
  n=$(find "$OUT/$s/trainingset/$TISSUE/train/labels" -name '*.csv' 2>/dev/null | wc -l)
  echo "  $s: $n tiles"
done
echo "Done. Full train: wsitrain run --input $INPUT --tissue $TISSUE --output $OUT/cellpose"
