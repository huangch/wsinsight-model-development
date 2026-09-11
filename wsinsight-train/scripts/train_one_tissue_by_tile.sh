#!/usr/bin/env bash
# Example 2 of 3 -- SINGLE-TISSUE training with a TILE-LEVEL split.
#
# Trains one tissue-specific head. --by-tile matters most here: a single
# tissue usually has only a handful of slides, so a whole-slide holdout would
# hand one sample's entire cell-type profile to validation.
#
# Usage: bash scripts/train_one_tissue_by_tile.sh <tissue> [input_dir] [output_dir]
#   e.g. bash scripts/train_one_tissue_by_tile.sh breast
#
# Tissue names are the directory names under the input tree (bone, brain,
# breast, cervix, colorectal, heart, kidney, liver, lung, lymph_node, ovary,
# pancreas, prostate, skin, tonsil).
#
# Env:
#   TASK       label space (default pantissue). The
#              celltype_assignment_<TASK>_label.csv files already exist under
#              each sample's outs/, so the annotate stage finds nothing to do
#              and returns immediately -- kurtorank is not invoked.
#   SEGMENTER  stardist (default) | cellpose. Only consulted when
#              NUCLEI_SOURCE=he-mask; ignored under the default
#              xenium-coords, where the segment stage is skipped.
#   STARDIST_MODEL_DIR  parent of the csbdeep model folder, e.g.
#              ~/.keras/models/StarDist2D. Auto-detected there when unset.
#   VAL_FRAC   validation fraction (default 0.20)
#   SEED       split seed (default 42)
#   TUNE       auto-tune iterations after training (default 0 = off)
#   RUN_SKIP   run everything except these, e.g. RUN_SKIP="annotate". To run a
#              single stage instead, call it directly: `wsitrain segment ...`
#   FORCE      set to 1 to re-run stages the manifest marks done, discarding
#              the masks the segment stage would otherwise reuse
#   TILE_WORKERS  slides tiled in parallel (default 1). The tile stage is
#              pure CPU (slide reads + numpy + Pillow), so raising this is
#              the cheapest speedup; one slide per worker, no GPU involved.
#   REDO       stage names to wipe and re-run, e.g. REDO=train or
#              REDO="split train". Every later stage is invalidated too, so
#              name the earliest one you want redone. Cheaper than FORCE,
#              which discards the whole run.
#   ARCHITECTURE  'cellvit' | 'hovernet' (default 'cellvit' for end2end). Any
#              torchvision classifier name (resnet50, efficientnet_b0, ...)
#              routes through the cellcls pipeline (--object-detection
#              stardist --architecture <name>) and supports
#              STAIN_NORMALIZATION; end2end runs force it off.
#   STAIN_NORMALIZATION  default 0 (off); set to 1 to Macenko-normalise the
#              training tiles. Honoured only on the cellcls path; the
#              end2end (cellvit/hovernet) path forces it off because
#              CellViT trains on raw H&E and Macenko raises on
#              near-uniform crops.
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
#   TRANSFORM  registration transform handed to the transfer stage:
#              'affine' | 'affine+bspline' (default; bUnwarpJ on top of the
#              affine seed) | 'none'. 'affine+bspline' writes an extra
#              bspline-grid artefact per slide and is slower; it is the
#              default because tissue folds routinely break the
#              affine-only alignment and review expects bUnwarpJ.
#   NUCLEI_SOURCE  where training cell positions come from:
#              'xenium-coords' (default; uses the Xenium cell_id projected
#              to H&E pixels and skips the segment stage entirely) |
#              'he-mask' (legacy path: SEGMENTER runs on H&E, mask
#              nucleus is looked up for each Xenium cell). With the
#              default xenium-coords + affine+bspline the scripts do
#              not need a segmenter at all; SEGMENTER is consulted only
#              when NUCLEI_SOURCE=he-mask is set explicitly.
#   CELLPOSE_BATCH_SIZE  cellpose batch size (default 16). Only forwarded
#                          when NUCLEI_SOURCE=he-mask and SEGMENTER=cellpose.
#   ENVBIN     conda env bin holding wsitrain + torch
#   WSITRAIN_DATA_DIR    input tree (default ../../data/xenium). data/ is far
#              too large for a Git remote, so it is not part of this package;
#              set this when wsinsight-train is checked out on its own.
#   WSITRAIN_MODELS_DIR  parent of the per-run output dirs (default
#              ../../models). Same reason.
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/train_one_tissue_by_tile.sh <tissue> [input_dir] [output_dir]" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
# cellvit-training lives inside this package so wsinsight-train stays
# self-contained when split into its own repository.
PKG="$(cd "$HERE/.." && pwd)"                        # wsinsight-train/
CVT="$PKG/cellvit-training"
# data/ and models/ are too large to ship with the package, so they stay in the
# parent repo; set these when wsinsight-train is checked out on its own.
DATA_ROOT="${WSITRAIN_DATA_DIR:-$(cd "$HERE/../.." && pwd)/data/xenium}"
MODELS_ROOT="${WSITRAIN_MODELS_DIR:-$(cd "$HERE/../.." && pwd)/models}"
# `wsi` is the only env on this host carrying wsitrain + torch + stardist.
ENVBIN="${ENVBIN:-/opt/anaconda3/envs/wsi/bin}"
export PATH="$ENVBIN:$PATH"
export CELLVIT_ROOT="${CELLVIT_ROOT:-$CVT/cellvit/CellViT-plus-plus}"
export TMPDIR="${TMPDIR:-/tmp}"
export CELLPOSE_LOCAL_MODELS_PATH="${CELLPOSE_LOCAL_MODELS_PATH:-/workspace/.cellpose}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.torch}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
# Non-interactive shells leave COLUMNS unset, and a redirected stdout hides the
# width from every child; tput/stty still read the tty, so measure it here.
if [ -z "${COLUMNS:-}" ]; then
  _cols=$(tput cols 2>/dev/null) || _cols=$(stty size </dev/tty 2>/dev/null | cut -d' ' -f2)
  case "${_cols:-}" in ''|*[!0-9]*) _cols=80 ;; esac
  export COLUMNS="$_cols"
fi
mkdir -p "$TMPDIR" "$CELLPOSE_LOCAL_MODELS_PATH" "$TORCH_HOME"

TISSUE="$1"
case "$TISSUE" in
  *,*|*+*|pantissue)
    echo "ERROR: '$TISSUE' is not a single tissue." >&2
    echo "       Use train_tissues_by_tile.sh for a subset, or" >&2
    echo "       train_pantissue_by_tile.sh for all tissues." >&2
    exit 2 ;;
esac

INPUT="${2:-$DATA_ROOT}"
OUT="${3:-$MODELS_ROOT/${TISSUE}_by_tile}"         # own scope: manifest is per output+tissue
TASK="${TASK:-pantissue}"
SEGMENTER="${SEGMENTER:-stardist}"
VAL_FRAC="${VAL_FRAC:-0.20}"
SEED="${SEED:-42}"
TUNE="${TUNE:-0}"
GPUS="${GPUS:-all}"
TRANSFORM="${TRANSFORM:-affine+bspline}"
NUCLEI_SOURCE="${NUCLEI_SOURCE:-xenium-coords}"

# TRANSFORM is one of the wsitrain CLI's transform choices; --nuclei-source
# is its own orthogonal axis. Validate early to give a clean error message
# before wsitrain does its own check.
case "$TRANSFORM" in
  affine|affine+bspline|none) ;;
  *)
    echo "ERROR: TRANSFORM must be 'affine', 'affine+bspline', or 'none' (got '$TRANSFORM')" >&2
    exit 2 ;;
esac
case "$NUCLEI_SOURCE" in
  xenium-coords|he-mask) ;;
  *)
    echo "ERROR: NUCLEI_SOURCE must be 'xenium-coords' or 'he-mask' (got '$NUCLEI_SOURCE')" >&2
    exit 2 ;;
esac

# xenium-coords makes the segment stage early-exit, so no segmenter flag is sent.
SEGMENTER_FLAGS=()
CP_BATCH_FLAGS=()
HEADLINE_SEGMENTER="n/a (xenium-coords)"
if [ "$NUCLEI_SOURCE" = "he-mask" ]; then
  SEGMENTER_FLAGS=(--segmenter "$SEGMENTER")
  HEADLINE_SEGMENTER="$SEGMENTER"
  if [ "$SEGMENTER" = "cellpose" ]; then
    : "${CELLPOSE_BATCH_SIZE:=16}"
    CP_BATCH_FLAGS=(--cellpose-batch-size "$CELLPOSE_BATCH_SIZE")
  fi
fi

STAGE_FLAGS=()
[ -n "${RUN_SKIP:-}" ] && STAGE_FLAGS=(--run-skip $RUN_SKIP)

FORCE_FLAGS=()
[ -n "${FORCE:-}" ] && FORCE_FLAGS=(--force)

# REDO="train" (space- or comma-separated stage names) -> --redo-train.
# wsitrain cascades to the later stages on its own, so naming the earliest
# stage you want redone is enough.
REDO_FLAGS=()
_redo_list="${REDO:-}"
for _s in ${_redo_list//,/ }; do REDO_FLAGS+=("--redo-$_s"); done

TILE_WORKER_FLAGS=()
if [ -n "${TILE_WORKERS:-}" ]; then
  case "$TILE_WORKERS" in
    ''|*[!0-9]*) echo "ERROR: TILE_WORKERS must be a positive integer (got '$TILE_WORKERS')" >&2; exit 2 ;;
  esac
  TILE_WORKER_FLAGS=(--tile-workers "$TILE_WORKERS")
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

# Resolve ARCHITECTURE: default 'cellvit' (or 'hovernet') stays on the end2end
# path; any torchvision classifier name routes to the cellcls pipeline where
# --object-detection stardist feeds cropped nuclei into a torchvision model.
# HoVer-Net is accepted here for forward-compatibility but the end2end path
# today only supports CellViT-plus-plus (set $CELLVIT_ROOT or pick 'cellvit').
ARCHITECTURE="${ARCHITECTURE:-cellvit}"
# Cellcls path additionally requires --patch-size-pixels,
# --patch-spacing-um-px; default 64px at 0.5 µm/pixel for cell-level crops.
# Override via env vars if your Xenium panel needs different geometry.
case "$ARCHITECTURE" in
  cellvit|hovernet)
    OD_FLAGS=()
    ARCH_FLAGS=()
    # CellViT trains on raw H&E, and its Macenko pass raises on a near-uniform
    # crop, so stain normalisation is off on this path regardless of the env.
    if [ "${STAIN_FLAG[0]}" = "--stain-normalization" ]; then
      echo "[note] STAIN_NORMALIZATION is not used on the $ARCHITECTURE (end2end) path; forcing --no-stain-normalization."
    fi
    STAIN_FLAG=(--no-stain-normalization)
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
[ -d "$INPUT/$TISSUE" ] || { echo "ERROR: no such tissue dir: $INPUT/$TISSUE" >&2; exit 1; }
[ -d "$CELLVIT_ROOT" ] || { echo "ERROR: CELLVIT_ROOT not found: $CELLVIT_ROOT" >&2; exit 1; }

echo "== preflight (warnings non-fatal; unaligned samples are skipped) =="
# --output keeps the sample list out of the (often read-only) data tree.
wsitrain check --input "$INPUT" --tissue "$TISSUE" --output "$OUT" || true

echo "== $TISSUE, tile-level split (task=$TASK, nuclei_source=$NUCLEI_SOURCE, segmenter=$HEADLINE_SEGMENTER, transform=$TRANSFORM, val_frac=$VAL_FRAC, architecture=$ARCHITECTURE) =="
# --reset-config: the flags below are the whole story, never a config an
# earlier command happened to leave in $OUT.
wsitrain run \
  --input "$INPUT" \
  --tissue "$TISSUE" \
  --task "$TASK" \
  "${SEGMENTER_FLAGS[@]}" \
  "${OD_FLAGS[@]}" \
  "${ARCH_FLAGS[@]}" \
  "${SD_FLAGS[@]}" \
  "${STAIN_FLAG[@]}" \
  "${CP_BATCH_FLAGS[@]}" \
  --norm-sample-size "${NORM_SAMPLE_SIZE:-256}" \
  --transform "$TRANSFORM" \
  --nuclei-source "$NUCLEI_SOURCE" \
  --by-tile \
  --val-frac "$VAL_FRAC" \
  --seed "$SEED" \
  --output "$OUT" \
  --reset-config \
  "${FORCE_FLAGS[@]}" \
  "${REDO_FLAGS[@]}" \
  "${TILE_WORKER_FLAGS[@]}" \
  "${STAGE_FLAGS[@]}" \
  --tune "$TUNE" \
  --gpus "$GPUS"

echo
echo "Done."
echo "  model  : $OUT/models/$TISSUE/main/"
echo "  report : $OUT/report/$TISSUE/"
