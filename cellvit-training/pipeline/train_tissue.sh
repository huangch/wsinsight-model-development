#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# train.sh -- tissue-agnostic training driver
#
# Runs the four-step CellViT pipeline (train -> pick checkpoint -> validate ->
# TorchScript export) for any tissue whose artifacts live under
#   trainingset/<tissue>/
# with the standard layout:
#   trainingset/<tissue>/train_configs/<BACKBONE>/<FOLD>.yaml
#   trainingset/<tissue>/splits/<FOLD>/val.csv
#   trainingset/<tissue>/label_map.yaml
#
# The <log_comment> baked into the training YAML must match
# "<tissue>-<task>-<backbone-lower>" for checkpoint discovery to work.
#
# Usage:
#   bash train.sh <tissue>                                # defaults: SAM-H-x40, fold_0, hne
#   bash train.sh <tissue> <backbone> <fold> <task>
#
# Examples:
#   bash train.sh colorectal
#   bash train.sh breast     SAM-H-x40 fold_0 hne
#   bash train.sh pantissue  SAM-H-x40 fold_0 pantissue
# -----------------------------------------------------------------------------
set -euo pipefail

TISSUE="${1:?usage: train.sh <tissue> [backbone=SAM-H-x40] [fold=fold_0] [task=hne]}"
BACKBONE="${2:-SAM-H-x40}"
FOLD="${3:-fold_0}"
TASK="${4:-hne}"

# Derive log_comment from tissue + task + backbone (lowercased, e.g. sam-h-x40).
LOG_COMMENT="${TISSUE}-${TASK}-$(echo "${BACKBONE}" | tr '[:upper:]' '[:lower:]')"

# ── Anchor everything to this script's location ──────────────────────────────
# Layout (relative to this script):
#   <PROJECT_ROOT>/cellvit-training/pipeline/train.sh   (this file)
#   <PROJECT_ROOT>/cellvit-training/cellvit/CellViT-plus-plus/
#   <PROJECT_ROOT>/cellvit-training/trainingset/<tissue>/
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLVIT_TRAINING_ROOT="$(cd "${PIPELINE_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CELLVIT_TRAINING_ROOT}/.." && pwd)"
export PROJECT_ROOT CELLVIT_TRAINING_ROOT

LOG_BASE="${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local"
PYTHON="${PYTHON:-/opt/anaconda3/envs/wsinsight/bin/python3}"
CELLVIT_ROOT="${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus"
TRAINSET="${CELLVIT_TRAINING_ROOT}/trainingset/${TISSUE}"

CONFIG="${TRAINSET}/train_configs/${BACKBONE}/${FOLD}.yaml"
VAL_CSV="${TRAINSET}/splits/${FOLD}/val.csv"
LABEL_MAP="${TRAINSET}/label_map.yaml"

for f in "${CONFIG}" "${VAL_CSV}" "${LABEL_MAP}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_BASE}"

# Materialize ${PROJECT_ROOT}/${CELLVIT_TRAINING_ROOT} placeholders in the
# training YAML (CellViT++ does not expand env vars in YAML itself).
command -v envsubst >/dev/null || {
    echo "ERROR: envsubst not found (install gettext)" >&2; exit 1; }
RESOLVED_CONFIG="${TRAINSET}/train_configs/${BACKBONE}/.${FOLD}.resolved.yaml"
envsubst '${PROJECT_ROOT} ${CELLVIT_TRAINING_ROOT}' < "${CONFIG}" > "${RESOLVED_CONFIG}"

echo "[train.sh] tissue=${TISSUE} backbone=${BACKBONE} fold=${FOLD}"
echo "[train.sh] log_comment=${LOG_COMMENT}"
echo "[train.sh] config=${CONFIG} (resolved -> ${RESOLVED_CONFIG})"

# ── Step 1: Train the LinearClassifier head ──────────────────────────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${CELLVIT_ROOT}/cellvit/train_cell_classifier_head.py" \
    --config "${RESOLVED_CONFIG}" \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_train.log"

# ── Step 2: Locate the checkpoint produced by the run just completed ─────────
# Training names its output dir as <TIMESTAMP>_<log_comment>; pick the newest.
RUN_DIR=$(ls -td "${LOG_BASE}/"*"_${LOG_COMMENT}" 2>/dev/null | head -1)
if [[ -z "${RUN_DIR}" ]]; then
    echo "ERROR: Could not find a run directory matching *_${LOG_COMMENT} under ${LOG_BASE}" >&2
    exit 1
fi
CHECKPOINT="${RUN_DIR}/checkpoints/model_best.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Expected checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
echo "Using checkpoint: ${CHECKPOINT}"

# ── Step 3: Validate — confusion matrix + classification report ──────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${PIPELINE_DIR}/validate_classifier.py" \
    --checkpoint "${CHECKPOINT}" \
    --dataset    "${TRAINSET}" \
    --filelist   "${VAL_CSV}" \
    --label-map  "${LABEL_MAP}" \
    --outdir     "${RUN_DIR}/validation" \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_validate.log"

# ── Step 4: Convert to TorchScript ──────────────────────────────────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${CELLVIT_ROOT}/cellvit/cellvit_convert_to_torchscript.py" \
    --checkpoint "${CHECKPOINT}" \
    --height 1024 --width 1024 \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_convert.log"
