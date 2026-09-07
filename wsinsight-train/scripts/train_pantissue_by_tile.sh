#!/usr/bin/env bash
# Example 1 of 3 -- PAN-TISSUE training with a TILE-LEVEL split.
#
# --tissue pantissue pools every sample discovered under the input tree into a
# single head. --by-tile then splits at the tile level, so each slide feeds
# both train and val. Whole-slide holdout (--by-slide) is the stricter test of
# generalisation, but with an uneven per-tissue slide count it tends to put
# entire rare tissues on one side of the split.
#
# Usage: bash scripts/train_pantissue_by_tile.sh [input_dir] [output_dir]
#
# Env:
#   TASK       label space (default pantissue). The
#              celltype_assignment_<TASK>_label.csv files already exist under
#              each sample's outs/, so the annotate stage finds nothing to do
#              and returns immediately -- kurtorank is not invoked.
#   SEGMENTER  stardist (default) | cellpose
#   STARDIST_MODEL_DIR  parent of the csbdeep model folder, e.g.
#              ~/.keras/models/StarDist2D. Auto-detected there when unset.
#   VAL_FRAC   validation fraction (default 0.20)
#   SEED       split seed (default 42)
#   TUNE       auto-tune iterations after training (default 0 = off)
#   RUN_SKIP   run everything except these, e.g. RUN_SKIP="annotate". To run a
#              single stage instead, call it directly: `wsitrain segment ...`
#   FORCE      set to 1 to re-run stages the manifest marks done, discarding
#              the masks the segment stage would otherwise reuse
#   ARCHITECTURE  'cellvit' | 'hovernet' (default 'cellvit' for end2end). Any
#              torchvision classifier name (resnet50, efficientnet_b0, ...)
#              routes through the cellcls pipeline (--object-detection
#              stardist --architecture <name>) and supports
#              STAIN_NORMALIZATION; end2end runs currently ignore it because
#              CellViT-plus-plus is the only end2end backbone vendored here.
#   STAIN_NORMALIZATION  default 0 (off); set to 1 to Macenko-normalise the
#              training tiles (works for both cellcls and end2end).
#   NORM_SAMPLE_SIZE  cells per slide used to fit the slide-level Macenko
#              source matrix (default 256; consumed on both paths when
#              STAIN_NORMALIZATION=1).
#   GPUS       device index (default auto). 'cpu' segments without a GPU but
#              cannot train, so it needs RUN_SKIP to stop before split.
#   ENVBIN     conda env bin holding wsitrain + torch
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                    # wsinsight-model-development/
CVT="$ROOT/cellvit-training"
# `wsi` is the only env on this host carrying wsitrain + torch + stardist.
ENVBIN="${ENVBIN:-/opt/anaconda3/envs/wsi/bin}"
export PATH="$ENVBIN:$PATH"
export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

INPUT="${1:-$ROOT/data/xenium}"
OUT="${2:-$ROOT/models/pantissue_by_tile}"
TISSUE=pantissue
TASK="${TASK:-pantissue}"
SEGMENTER="${SEGMENTER:-stardist}"
VAL_FRAC="${VAL_FRAC:-0.20}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
GPUS="${GPUS:-auto}"

STAGE_FLAGS=()
[ -n "${RUN_SKIP:-}" ] && STAGE_FLAGS=(--run-skip $RUN_SKIP)

FORCE_FLAGS=()
[ -n "${FORCE:-}" ] && FORCE_FLAGS=(--force)

STAIN_FLAG=()
case "${STAIN_NORMALIZATION:-off}" in
  on|1|true|yes)
    STAIN_FLAG=(--stain-normalization)
    ;;
  off|0|false|no|"")
    STAIN_FLAG=(--no-stain-normalization)
    ;;
  *)
    echo "ERROR: STAIN_NORMALIZATION must be one of on/off, got '${STAIN_NORMALIZATION}'" >&2
    exit 2 ;;
esac

ARCHITECTURE="${ARCHITECTURE:-cellvit}"
case "$ARCHITECTURE" in
  cellvit|hovernet)
    OD_FLAGS=()
    ARCH_FLAGS=()
    ;;
  *)
    : "${PATCH_SIZE_PIXELS:=64}"
    : "${PATCH_SPACING_UM_PX:=0.5}"
    export PATCH_SIZE_PIXELS PATCH_SPACING_UM_PX
    OD_FLAGS=(--object-detection stardist)
    ARCH_FLAGS=(--architecture "$ARCHITECTURE"
                --patch-size-pixels "$PATCH_SIZE_PIXELS"
                --patch-spacing-um-px "$PATCH_SPACING_UM_PX")
    ;;
esac

# split renders the CellViT config, which needs a real device index.
case "$GPUS" in
  cpu|none|false|no)
    case " ${RUN_SKIP:-} " in
      *" split "*) : ;;
      *) echo "ERROR: GPUS=$GPUS turns the GPU off, but CellViT training needs a" >&2
         echo "       device index. Set GPUS=<index>, or stop before training with" >&2
         echo "       RUN_SKIP='split train validate export'." >&2
         exit 2 ;;
    esac ;;
esac

SD_FLAGS=()
[ -n "${STARDIST_MODEL_DIR:-}" ] && SD_FLAGS=(--stardist-model-dir "$STARDIST_MODEL_DIR")

command -v wsitrain >/dev/null || {
  echo "ERROR: wsitrain is not on PATH (ENVBIN=$ENVBIN)" >&2; exit 1; }
[ -d "$INPUT" ] || { echo "ERROR: input dir not found: $INPUT" >&2; exit 1; }
[ -d "$CELLVIT_ROOT" ] || { echo "ERROR: CELLVIT_ROOT not found: $CELLVIT_ROOT" >&2; exit 1; }

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
# --output keeps the sample list out of the (often read-only) data tree.
wsitrain check --input "$INPUT" --tissue "$TISSUE" --output "$OUT" || true

echo "== pantissue, tile-level split (task=$TASK, segmenter=$SEGMENTER, val_frac=$VAL_FRAC, architecture=$ARCHITECTURE) =="
# --reset-config: the flags below are the whole story, never a config an
# earlier command happened to leave in $OUT.
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  --segmenter "$SEGMENTER" \
  "${OD_FLAGS[@]}" \
  "${ARCH_FLAGS[@]}" \
  "${SD_FLAGS[@]}" \
  "${STAIN_FLAG[@]}" \
  --norm-sample-size "${NORM_SAMPLE_SIZE:-256}" \
  --transform affine \
  --by-tile \
  --val-frac "$VAL_FRAC" \
  --seed "$SEED" \
  --output "$OUT" \
  --reset-config \
  "${FORCE_FLAGS[@]}" \
  "${STAGE_FLAGS[@]}" \
  --tune "$TUNE" \
  --gpus "$GPUS"

echo
echo "Done."
echo "  model  : $OUT/models/$TISSUE/main/"
echo "  report : $OUT/report/$TISSUE/"
