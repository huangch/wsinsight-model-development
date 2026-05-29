#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# train_tissue.sh -- per-tissue 4-step CellViT training pipeline
#
# Runs train -> pick checkpoint -> validate -> TorchScript export for ONE
# tissue, with artifacts under trainingset/<tissue>/.
#
# Layout expected:
#   trainingset/<tissue>/train_configs/<BACKBONE>/<FOLD>.yaml
#   trainingset/<tissue>/splits/<FOLD>/val.csv
#   trainingset/<tissue>/label_map.yaml
#
# The <log_comment> baked into the training YAML must equal
#   "<tissue>-<task>-<backbone-lower>"
# for checkpoint discovery to work.
#
# Usage:
#   bash train_tissue.sh <tissue>                                # defaults: SAM-H-x40, fold_0, hne
#   bash train_tissue.sh <tissue> <backbone> <fold> <task>
#
# Examples:
#   bash train_tissue.sh colorectal
#   bash train_tissue.sh breast     SAM-H-x40 fold_0 hne
#   bash train_tissue.sh pantissue  SAM-H-x40 fold_0 pantissue
#
# See train_all_tissues.sh for the loop driver.
# See validate_tissue.sh to re-run Step 3 in isolation.
# -----------------------------------------------------------------------------
set -euo pipefail

# Source shared helpers (path resolution, log_comment, run lookup, validate).
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${PIPELINE_DIR}/_lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,28p' "$0"; exit 0
fi

TISSUE="${1:?usage: train_tissue.sh <tissue> [backbone=SAM-H-x40] [fold=fold_0] [task=hne]}"
BACKBONE="${2:-SAM-H-x40}"
FOLD="${3:-fold_0}"
TASK="${4:-hne}"

# Resolve all per-tissue paths in one call (sets TRAINSET, CONFIG, VAL_CSV, LABEL_MAP).
eval "$(_lib::tissue_paths "${TISSUE}" "${FOLD}" "${BACKBONE}")"
LOG_COMMENT="$(_lib::log_comment "${TISSUE}" "${TASK}" "${BACKBONE}")"

CELLVIT_TRAINING_ROOT="$(_lib::cellvit_training_root)"
PROJECT_ROOT="$(cd "${CELLVIT_TRAINING_ROOT}/.." && pwd)"
export PROJECT_ROOT CELLVIT_TRAINING_ROOT

CELLVIT_ROOT="$(_lib::cellvit_root)"
LOG_BASE="$(_lib::logs_local)"
PYTHON="$(_lib::python)"

for f in "${CONFIG}" "${VAL_CSV}" "${LABEL_MAP}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

mkdir -p "${LOG_BASE}"

# Materialize ${PROJECT_ROOT} / ${CELLVIT_TRAINING_ROOT} placeholders in the
# training YAML (CellViT++ does not expand env vars in YAML itself).
command -v envsubst >/dev/null || {
    echo "ERROR: envsubst not found (install gettext)" >&2; exit 1; }
RESOLVED_CONFIG="${TRAINSET}/train_configs/${BACKBONE}/.${FOLD}.resolved.yaml"
envsubst '${PROJECT_ROOT} ${CELLVIT_TRAINING_ROOT}' < "${CONFIG}" > "${RESOLVED_CONFIG}"

echo "[train_tissue.sh] tissue=${TISSUE} backbone=${BACKBONE} fold=${FOLD}"
echo "[train_tissue.sh] log_comment=${LOG_COMMENT}"
echo "[train_tissue.sh] config=${CONFIG} (resolved -> ${RESOLVED_CONFIG})"

# ── Step 1: Train the LinearClassifier head ──────────────────────────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${CELLVIT_ROOT}/cellvit/train_cell_classifier_head.py" \
    --config "${RESOLVED_CONFIG}" \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_train.log"

# ── Step 2: Locate the checkpoint produced by the run just completed ─────────
RUN_DIR="$(_lib::find_latest_run "${LOG_COMMENT}")"
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

# ── Step 3: Validate — delegated to validate_tissue.sh so logic stays in one place.
# Pass the explicit RUN_DIR as 4th arg to bypass log_comment-based lookup
# (validate_tissue.sh defaults to task=hne when looking up runs by log_comment;
# here we already know the run dir, regardless of task).
bash "${PIPELINE_DIR}/validate_tissue.sh" "${TISSUE}" "${BACKBONE}" "${FOLD}" "${RUN_DIR}"

# ── Step 4: Convert to TorchScript ──────────────────────────────────────────
PYTHONPATH="${CELLVIT_ROOT}" "${PYTHON}" -u \
    "${CELLVIT_ROOT}/cellvit/cellvit_convert_to_torchscript.py" \
    --checkpoint "${CHECKPOINT}" \
    --height 1024 --width 1024 \
    2>&1 | tee "${LOG_BASE}/${TISSUE}_convert.log"
