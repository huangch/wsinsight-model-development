#!/usr/bin/env bash
# Example 3 of 3 -- MULTI-TISSUE (subset) training with a SLIDE-LEVEL split.
#
# Slide-level counterpart of train_tissues_by_tile.sh. Pools a chosen subset of
# tissues into one head, e.g. the epithelial-rich group, and holds out whole
# slides so no tile from a validation slide is ever seen in training.
#
# The holdout is stratified per tissue (wsitrain/splits.py::split_tiles): each
# tissue in the subset contributes
#   max(1, min(round(n_slides * VAL_FRAC), n_slides - 1))
# validation slides, and a tissue with a single slide is skipped rather than
# turned into an unscoreable leave-tissue-out case. Pooling a slide-starved
# tissue with a richer one is the main reason to use this script over
# train_one_tissue_by_slide.sh: a 3-slide cohort cannot support a credible
# whole-slide holdout on its own.
#
# Any slide that is the sole carrier of a class is tile-split instead so the
# class can appear on both sides; the split then reports mode
# "slide-level+hybrid". Read the [split] lines in the run log before quoting
# the mode anywhere.
#
# Usage: bash scripts/train_tissues_by_slide.sh <tissue,tissue,...> [input_dir] [output_dir]
#   e.g. bash scripts/train_tissues_by_slide.sh breast,lung
#        bash scripts/train_tissues_by_slide.sh breast,lung,colorectal
#
# Tissue names are the directory names under the input tree (bone, brain,
# breast, cervix, colorectal, heart, kidney, liver, lung, lymph_node, ovary,
# pancreas, prostate, skin, tonsil).
#
# Env:
#   TASK       label space (default pantissue). Pooling tissues only makes
#              sense with a shared vocabulary, which is what pantissue is. The
#              celltype_assignment_<TASK>_label.csv files already exist under
#              each sample's outs/, so the annotate stage finds nothing to do
#              and returns immediately -- kurtorank is not invoked.
#   SEGMENTER  stardist (default) | cellpose
#   STARDIST_MODEL_DIR  parent of the csbdeep model folder, e.g.
#              ~/.keras/models/StarDist2D. Auto-detected there when unset.
#   VAL_FRAC   validation fraction (default 0.20); see the rounding rule above
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
#   GPUS       device selection (Docker-style: 'all' = every visible CUDA
#              device; '0' or '0,1' for an explicit subset; 'cpu' = skip GPU).
#              'all' is the default and triggers multi-GPU fan-out for the
#              segment stage when --nuclei-source=he-mask is also set. With
#              the default --nuclei-source=xenium-coords the segment stage
#              is skipped, so --gpus only matters for train/validate.
#              'cpu' segments without a GPU but cannot train, so it needs
#              RUN_SKIP to stop before the split stage.
#   CELLPOSE_BATCH_SIZE  cellpose batch size (default 16). Only forwarded
#                          when SEGMENTER=cellpose; no effect on stardist.
#   ENVBIN     conda env bin holding wsitrain + torch
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train_tissues_by_slide.sh <tissue,tissue,...> [input_dir] [output_dir]" >&2
  exit 2
fi

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

# Normalise the selector to a comma list: '+' and spaces are accepted too.
TISSUE="$(echo "$1" | tr '+ ' ',,' | tr -s ',' | sed 's/^,//; s/,$//')"
[ -n "$TISSUE" ] || { echo "ERROR: empty tissue list" >&2; exit 2; }
case "$TISSUE" in
  pantissue) echo "ERROR: use train_pantissue_by_slide.sh for all tissues." >&2; exit 2 ;;
  *,*) : ;;
  *) echo "ERROR: '$TISSUE' is a single tissue; use train_one_tissue_by_slide.sh." >&2; exit 2 ;;
esac

INPUT="${2:-$ROOT/data/xenium}"
# Directory-safe scope name; the manifest is keyed per output+tissue, so each
# subset must get its own --output or it will reuse another run's state.
SLUG="$(echo "$TISSUE" | tr ',' '-')"
OUT="${3:-$ROOT/models/${SLUG}_by_slide}"
TASK="${TASK:-pantissue}"
SEGMENTER="${SEGMENTER:-stardist}"
VAL_FRAC="${VAL_FRAC:-0.20}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
GPUS="${GPUS:-all}"

STAGE_FLAGS=()
[ -n "${RUN_SKIP:-}" ] && STAGE_FLAGS=(--run-skip $RUN_SKIP)

FORCE_FLAGS=()
[ -n "${FORCE:-}" ] && FORCE_FLAGS=(--force)

# --- cellpose batch (added by wsitrain redesign, see backup/pre-redesign tag) ---
CP_BATCH_FLAGS=()
if [ "$SEGMENTER" = "cellpose" ]; then
  : "${CELLPOSE_BATCH_SIZE:=16}"
  CP_BATCH_FLAGS=(--cellpose-batch-size "$CELLPOSE_BATCH_SIZE")
fi
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
IFS=',' read -ra _T <<< "$TISSUE"
for t in "${_T[@]}"; do
  [ -d "$INPUT/$t" ] || { echo "ERROR: no such tissue dir: $INPUT/$t" >&2; exit 1; }
done

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
# --output keeps the sample list out of the (often read-only) data tree.
wsitrain check --input "$INPUT" --tissue "$TISSUE" --output "$OUT" || true

echo "== $TISSUE, slide-level split (task=$TASK, segmenter=$SEGMENTER, val_frac=$VAL_FRAC, architecture=$ARCHITECTURE) =="
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
  "${CP_BATCH_FLAGS[@]}" \
  --norm-sample-size "${NORM_SAMPLE_SIZE:-256}" \
  --transform affine \
  --by-slide \
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
echo "  split  : $OUT/report/$TISSUE/ -- confirm mode is 'slide-level', not 'slide-level+hybrid'"
