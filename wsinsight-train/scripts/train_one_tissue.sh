#!/usr/bin/env bash
# One-tissue smoke run + Cellpose-vs-StarDist parity check.
# Usage: scripts/run_one_tissue.sh <input_dir> <tissue> [output_dir]
# Requires the `wsitrain` env (torch + cellpose + kurtorank) and $CELLVIT_ROOT.
set -euo pipefail

INPUT="${1:?input dir}"
TISSUE="${2:-breast}"
OUT="${3:-$INPUT/wsinsight_train_out}"

: "${CELLVIT_ROOT:?set CELLVIT_ROOT to the CellViT-plus-plus checkout}"

echo "== preflight =="
wsitrain check --input "$INPUT" --tissue "$TISSUE"

echo "== cellpose run =="
wsitrain run --input "$INPUT" --tissue "$TISSUE" --segmenter cellpose \
  --output "$OUT/cellpose" --to tile

echo "== stardist run (parity) =="
wsitrain run --input "$INPUT" --tissue "$TISSUE" --segmenter stardist \
  --output "$OUT/stardist" --to tile

echo "== compare tile/label counts =="
for s in cellpose stardist; do
  n=$(find "$OUT/$s/trainingset/$TISSUE/train/labels" -name '*.csv' 2>/dev/null | wc -l)
  echo "  $s: $n tiles"
done
echo "Done. Full train: wsitrain run --input $INPUT --tissue $TISSUE --output $OUT/cellpose"
