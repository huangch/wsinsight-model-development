#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# splits_all_tissues.sh -- loop make_splits.py across every tissue that has
# exported tiles under trainingset/<tissue>/train/labels/*.csv.
#
# Mirrors the *_all_tissues.sh convention: per-tissue work is delegated to
# make_splits.py; this driver only handles discovery, looping, and aggregating
# pass/fail status.
#
# Usage:
#   bash splits_all_tissues.sh                            # all tissues, fold_0, val_frac=0.1, per-tile
#   bash splits_all_tissues.sh --tissues "breast heart"   # subset
#   bash splits_all_tissues.sh --fold fold_0 --val-frac 0.1
#   bash splits_all_tissues.sh --by-slide                 # slide-level holdout
#   bash splits_all_tissues.sh --dry-run                  # print the plan only
#   bash splits_all_tissues.sh --force                    # overwrite existing val.csv
#   bash splits_all_tissues.sh --include-pantissue        # also split trainingset/pantissue/
#
# Notes:
#   - Single-slide tissues fall back from --by-slide to per-tile shuffle
#     (with a WARN), per make_splits.py's existing behaviour.
#   - 'pantissue' is excluded by default since it must be aggregated first
#     (bash aggregate_pantissue.sh) before its split makes sense.
# -----------------------------------------------------------------------------
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${PIPELINE_DIR}/_lib.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,24p' "$0"; exit 0
fi

PYTHON="$(_lib::python)"
CELLVIT_TRAINING_ROOT="$(_lib::cellvit_training_root)"

FOLD="fold_0"
VAL_FRAC="0.1"
BY_SLIDE=0
TISSUES=""
DRY_RUN=0
FORCE=0
INCLUDE_PT="exclude"
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tissues)            TISSUES="$2"; shift 2 ;;
        --fold)               FOLD="$2"; shift 2 ;;
        --val-frac)           VAL_FRAC="$2"; shift 2 ;;
        --by-slide)           BY_SLIDE=1; shift ;;
        --dry-run)            DRY_RUN=1; shift ;;
        --force)              FORCE=1; shift ;;
        --include-pantissue)  INCLUDE_PT="include"; shift ;;
        --seed)               SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Discover tissues if not explicit (those with at least one labels CSV).
if [[ -z "${TISSUES}" ]]; then
    TISSUES="$(_lib::tissues_with_labels "${INCLUDE_PT}")"
fi

if [[ -z "${TISSUES// }" ]]; then
    echo "[splits_all_tissues] No tissues with labels: run export_tiles.groovy first." >&2
    exit 1
fi

echo "[splits_all_tissues] Tissues  : ${TISSUES}"
echo "[splits_all_tissues] Fold     : ${FOLD}"
echo "[splits_all_tissues] Val frac : ${VAL_FRAC}"
echo "[splits_all_tissues] By slide : ${BY_SLIDE}"
echo "[splits_all_tissues] Force    : ${FORCE}"

[[ "${DRY_RUN}" == 1 ]] && { echo "[splits_all_tissues] Dry-run, exiting."; exit 0; }

EXTRA_FLAGS=()
[[ "${BY_SLIDE}" == 1 ]] && EXTRA_FLAGS+=("--by-slide")

FAIL=()
for t in ${TISSUES}; do
    echo
    echo "── splits: ${t} ──"
    val_csv="${CELLVIT_TRAINING_ROOT}/trainingset/${t}/splits/${FOLD}/val.csv"
    if [[ -f "${val_csv}" && "${FORCE}" != 1 ]]; then
        echo "[splits_all_tissues] ${t}: ${val_csv} exists, skipping (use --force to overwrite)."
        continue
    fi
    if ! "${PYTHON}" "${PIPELINE_DIR}/make_splits.py" \
            --tissue "${t}" --fold "${FOLD}" \
            --val-frac "${VAL_FRAC}" --seed "${SEED}" \
            "${EXTRA_FLAGS[@]}"; then
        echo "WARN: make_splits.py failed for ${t}; continuing." >&2
        FAIL+=("${t}")
    fi
done

echo
if (( ${#FAIL[@]} > 0 )); then
    echo "[splits_all_tissues] Failed tissues: ${FAIL[*]}"
    exit 2
fi
echo "[splits_all_tissues] All tissues completed."
