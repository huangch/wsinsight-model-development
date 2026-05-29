#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# train_all_tissues.sh -- loop the make_splits / make_train_config /
# train_tissue.sh pipeline across every tissue that has exported tiles under
#   trainingset/<tissue>/train/labels/*.csv
#
# Skips:
#   - tissues with no labels yet (export_tiles.groovy hasn't run)
#   - tissues whose train_configs/<backbone>/<fold>.yaml already exists
#     unless --force is given (then make_train_config.py is rerun)
#
# Usage:
#   bash train_all_tissues.sh                            # all tissues, SAM-H-x40, fold_0, task=pantissue
#   bash train_all_tissues.sh --tissues "breast heart"   # subset
#   bash train_all_tissues.sh --backbone SAM-H-x40 --fold fold_0 --task pantissue
#   bash train_all_tissues.sh --dry-run                  # print the plan, don't train
#   bash train_all_tissues.sh --force                    # overwrite existing per-tissue configs
#
# Per-tissue steps:
#   1. python make_splits.py            --tissue <t>
#   2. python make_train_config.py      --tissue <t>  [--force]
#   3. bash   train_tissue.sh           <t> <backbone> <fold> <task>
# -----------------------------------------------------------------------------
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${PIPELINE_DIR}/_lib.sh"
CELLVIT_TRAINING_ROOT="$(_lib::cellvit_training_root)"
PYTHON="$(_lib::python)"

BACKBONE="SAM-H-x40"
FOLD="fold_0"
TASK="pantissue"
TISSUES=""
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tissues)  TISSUES="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --fold)     FOLD="$2"; shift 2 ;;
        --task)     TASK="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --force)    FORCE=1; shift ;;
        -h|--help)
            sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Build tissue list: explicit subset, or every dir under trainingset/ that
# has at least one *.csv under train/labels and is not the pantissue head.
if [[ -z "${TISSUES}" ]]; then
    TISSUES="$(_lib::tissues_with_labels)"
fi

if [[ -z "${TISSUES// }" ]]; then
    echo "[train_all_tissues] No tissues to train: run export_tiles.groovy first." >&2
    exit 1
fi

echo "[train_all_tissues] Tissues: ${TISSUES}"
echo "[train_all_tissues] Backbone=${BACKBONE} Fold=${FOLD} Task=${TASK}"
[[ "${DRY_RUN}" == 1 ]] && { echo "[train_all_tissues] Dry-run, exiting."; exit 0; }

for t in ${TISSUES}; do
    echo
    echo "================================================================"
    echo "  TISSUE: ${t}"
    echo "================================================================"

    CONFIG="${CELLVIT_TRAINING_ROOT}/trainingset/${t}/train_configs/${BACKBONE}/${FOLD}.yaml"

    "${PYTHON}" "${PIPELINE_DIR}/make_splits.py" --tissue "${t}" --fold "${FOLD}"

    if [[ ! -f "${CONFIG}" || "${FORCE}" == 1 ]]; then
        FORCE_FLAG=""
        [[ "${FORCE}" == 1 ]] && FORCE_FLAG="--force"
        "${PYTHON}" "${PIPELINE_DIR}/make_train_config.py" \
            --tissue "${t}" --backbone "${BACKBONE}" --fold "${FOLD}" \
            --task "${TASK}" ${FORCE_FLAG}
    else
        echo "[train_all_tissues] Reusing existing ${CONFIG}"
    fi

    bash "${PIPELINE_DIR}/train_tissue.sh" "${t}" "${BACKBONE}" "${FOLD}" "${TASK}"
done

echo
echo "[train_all_tissues] All tissues completed."
