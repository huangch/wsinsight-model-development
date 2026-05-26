#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# validate_tissue.sh -- per-tissue validation driver
#
# Re-runs only the validation step (classification report + confusion matrices)
# against an existing training run for ONE tissue.
#
# By default, the newest run matching "*_<tissue>-hne-<backbone-lower>" under
#   cellvit/CellViT-plus-plus/logs_local/
# is used. A specific run directory or checkpoint may also be given as the
# 4th argument.
#
# Usage:
#   bash validate_tissue.sh <tissue>                          # defaults: SAM-H-x40, fold_0, latest hne run
#   bash validate_tissue.sh <tissue> <backbone> <fold>
#   bash validate_tissue.sh <tissue> <backbone> <fold> <run_dir_or_checkpoint>
#
# Examples:
#   bash validate_tissue.sh colorectal
#   bash validate_tissue.sh breast SAM-H-x40 fold_0
#   bash validate_tissue.sh colorectal SAM-H-x40 fold_0 \
#       /path/to/logs_local/2026-04-20T111253_colorectal-hne-sam-h-x40
#
# See train_tissue.sh (Step 3) which delegates to this script.
# -----------------------------------------------------------------------------
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${PIPELINE_DIR}/_lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,25p' "$0"; exit 0
fi

TISSUE="${1:?usage: validate_tissue.sh <tissue> [backbone=SAM-H-x40] [fold=fold_0] [run_dir_or_ckpt]}"
BACKBONE="${2:-SAM-H-x40}"
FOLD="${3:-fold_0}"
RUN_OR_CKPT="${4:-}"

LOG_COMMENT="$(_lib::log_comment "${TISSUE}" "hne" "${BACKBONE}")"

eval "$(_lib::tissue_paths "${TISSUE}" "${FOLD}" "${BACKBONE}")"

CELLVIT_TRAINING_ROOT="$(_lib::cellvit_training_root)"
PROJECT_ROOT="$(cd "${CELLVIT_TRAINING_ROOT}/.." && pwd)"
export PROJECT_ROOT CELLVIT_TRAINING_ROOT

LOG_BASE="$(_lib::logs_local)"

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
    RUN_DIR="$(_lib::find_latest_run "${LOG_COMMENT}")"
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

echo "[validate_tissue.sh] tissue=${TISSUE} backbone=${BACKBONE} fold=${FOLD}"
echo "[validate_tissue.sh] run_dir=${RUN_DIR}"
echo "[validate_tissue.sh] checkpoint=${CHECKPOINT}"

_lib::run_validate "${TISSUE}" "${FOLD}" "${BACKBONE}" "${RUN_DIR}"

echo "[validate_tissue.sh] outputs in: ${RUN_DIR}/validation"
