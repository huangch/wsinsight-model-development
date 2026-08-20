#!/usr/bin/env bash
# Annotate every Xenium sample under data/xenium EXCEPT heart tissue with
# KurtoRank (markers-v6, STHELAR label space), then train one pooled
# "pantissue" CellViT head on the sthelar_full label vocabulary.
#
# Why the phases are still split:
#   * The annotate stage now derives each sample's --tissue-type from its own
#     folder, so `--tissue pantissue` annotates fine in one pass; the loop below
#     is kept only to annotate tissue-by-tissue for readable progress/logs.
#   * kurtorank emits celltype_assignment_sthelar_full_label.csv itself now, so
#     Phase 2 is a backfill for samples annotated before that change.
#   * A symlink farm (heart excluded) lets the pooled run use --tissue pantissue
#     so the model/report land cleanly under models/pantissue/.
#
# Usage:  bash scripts/annotate_and_train_sthelar_pantissue.sh [input_dir]
# Env:    ENVBIN = conda env bin with wsitrain + cellpose + kurtorank + torch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                     # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
KURTO="$ROOT/kurtorank"
ENVBIN="${ENVBIN:-/opt/anaconda3/envs/wsinsight/bin}"
export PATH="$ENVBIN:$PATH"
export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

INPUT="${1:-$ROOT/data/xenium}"
OUT="$ROOT/models"
TASK="sthelar_full"
EXCLUDE="heart"
MARKERS="${MARKERS:-$KURTO/src/kurtorank/markers/data/markers-v6.csv}"
TOPK="${TOPK:-25}"

[ -d "$INPUT" ]     || { echo "ERROR: input dir not found: $INPUT" >&2; exit 1; }
[ -f "$MARKERS" ]   || { echo "ERROR: markers csv not found: $MARKERS" >&2; exit 1; }

WORK="$(mktemp -d)"
FARM="$WORK/xenium_no_${EXCLUDE}"
trap 'rm -rf "$WORK"' EXIT

# STHELAR labels and lcp_* tags exist in markers-v6.
MARKER_FLAGS=(--markers-csv "$MARKERS" --top-k-markers "$TOPK")

# Enumerate top-level tissue folders that actually contain a sample, minus heart.
mapfile -t TISSUES < <(
  find "$INPUT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | while read -r t; do
    [ "$t" = "$EXCLUDE" ] && continue
    if find "$INPUT/$t" -name cells.parquet -print -quit 2>/dev/null | grep -q .; then
      echo "$t"
    fi
  done
)
[ "${#TISSUES[@]}" -gt 0 ] || { echo "ERROR: no non-$EXCLUDE tissues with samples under $INPUT" >&2; exit 1; }
echo "== tissues to annotate (excluding $EXCLUDE): ${TISSUES[*]} =="

# Symlink farm so the pooled run can use --tissue pantissue with heart absent.
rm -rf "$FARM"; mkdir -p "$FARM"
for t in "${TISSUES[@]}"; do ln -s "$INPUT/$t" "$FARM/$t"; done

# ---------------------------------------------------------------------------
# Phase 1 — annotate each tissue with STHELAR markers (in-place under outs/).
# ---------------------------------------------------------------------------
for t in "${TISSUES[@]}"; do
  echo "== annotate: $t =="
  wsitrain run \
    --input "$FARM" \
    --tissue "$t" \
    --task "$TASK" \
    "${MARKER_FLAGS[@]}" \
    --stage-only annotate
done

# ---------------------------------------------------------------------------
# Phase 2 — derive the sthelar_full_label CSV the transfer stage consumes.
# ---------------------------------------------------------------------------
echo "== derive celltype_assignment_sthelar_full_label.csv =="
"$ENVBIN/python" "$KURTO/scripts/derive_sthelar_full_label_csvs.py" \
  --data-dir "$FARM" \
  --markers-csv "$MARKERS" \
  --overwrite

# ---------------------------------------------------------------------------
# Phase 3 — pooled pan-tissue training on the sthelar_full label space.
# ---------------------------------------------------------------------------
echo "== preflight (pooled, heart excluded) =="
wsitrain check --input "$FARM" --tissue pantissue || true

echo "== pan-tissue full cycle (segment -> export), task=$TASK =="
wsitrain run \
  --input "$FARM" \
  --tissue pantissue \
  --task "$TASK" \
  "${MARKER_FLAGS[@]}" \
  --segmenter "${SEGMENTER:-stardist}" \
  --transform affine \
  --output "$OUT" \
  --stage-skip annotate report

echo "Done. Head + report under: $OUT/models/pantissue/  +  $OUT/report/pantissue/"
