#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# validate.sh -- tissue-agnostic validation driver
#
# Re-runs only the validation step of the CellViT pipeline (classification
# report + confusion matrices) against an existing training run.
#
# By default, the newest run matching "*_<tissue>-hne-<backbone-lower>" under
#   cellvit/CellViT-plus-plus/logs_local/
# is used. A specific run directory or checkpoint may also be given.
#
# Usage:
#   bash validate.sh <tissue>                          # defaults: SAM-H-x40, fold_0, latest run
#   bash validate.sh <tissue> <backbone> <fold>
#   bash validate.sh <tissue> <backbone> <fold> <run_dir_or_checkpoint>
#
# Examples:
#   bash validate.sh colorectal
#   bash validate.sh breast SAM-H-x40 fold_0
#   bash validate.sh colorectal SAM-H-x40 fold_0 \
#       /path/to/logs_local/2026-04-20T111253_colorectal-hne-sam-h-x40
# -----------------------------------------------------------------------------
set -euo pipefail

TISSUE="${1:?usage: validate.sh <tissue> [backbone=SAM-H-x40] [fold=fold_0] [run_dir_or_ckpt]}"
BACKBONE="${2:-SAM-H-x40}"
FOLD="${3:-fold_0}"
RUN_OR_CKPT="${4:-}"

LOG_COMMENT="${TISSUE}-hne-$(echo "${BACKBONE}" | tr '[:upper:]' '[:lower:]')"

# ── Anchor everything to this script's location (see train.sh for rationale) ─
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLVIT_TRAINING_ROOT="$(cd "${PIPELINE_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CELLVIT_TRAINING_ROOT}/.." && pwd)"
export PROJECT_ROOT CELLVIT_TRAINING_ROOT

LOG_BASE="${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local"
PYTHON="${PYTHON:-/opt/anaconda3/envs/wsinsight/bin/python3}"
CELLVIT_ROOT="${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus"
TRAINSET="${CELLVIT_TRAINING_ROOT}/trainingset/${TISSUE}"

VAL_CSV="${TRAINSET}/splits/${FOLD}/val.csv"
LABEL_MAP="${TRAINSET}/label_map.yaml"

for f in "${VAL_CSV}" "${LABEL_MAP}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

# ── Resolve checkpoint + run directory ───────────────────────────────────────
if [[ -n "${RUN_OR_CKPT}" ]]; then
    if [[ -f "${RUN_OR_CKPT}" ]]; then
        CHECKPOINT="${RUN_OR_CKPT}"
        RUN_DIR="$(dirname "$(dirname "${CHECKPOINT}")")"
    elif [[ -d "${RUN_OR_CKPT}" ]]; then
        RUN_DIR="${RUN_OR_CKPT}"
        CHECKPOINT="${RUN_DIR}/checkpoints/model_best.pth"
    else
        echo "ERROR: ${RUN_OR_CKPT} is neither a file nor a directory" >&2
        exit 1
    fi
else
    RUN_DIR=$(ls -td "${LOG_BASE}/"*"_${LOG_COMMENT}" 2>/dev/null | head -1)
    if [[ -z "${RUN_DIR}" ]]; then
        echo "ERROR: no run directory matching *_${LOG_COMMENT} under ${LOG_BASE}" >&2
        exit 1
    fi
    CHECKPOINT="${RUN_DIR}/checkpoints/model_best.pth"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

echo "[validate.sh] tissue=${TISSUE} backbone=${BACKBONE} fold=${FOLD}"
echo "[validate.sh] run_dir=${RUN_DIR}"
echo "[validate.sh] checkpoint=${CHECKPOINT}"

# ── Run validation ───────────────────────────────────────────────────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${PIPELINE_DIR}/validate_classifier.py" \
    --checkpoint "${CHECKPOINT}" \
    --dataset    "${TRAINSET}" \
    --filelist   "${VAL_CSV}" \
    --label-map  "${LABEL_MAP}" \
    --outdir     "${RUN_DIR}/validation" \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_validate.log"

echo "[validate.sh] outputs in: ${RUN_DIR}/validation"
