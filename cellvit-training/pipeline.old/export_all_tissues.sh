#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# export_all_tissues.sh -- driver for pipeline/qupath/export_tiles.groovy that runs
# the export in two passes:
#
#   Pass 1: all multi-slide tissues at OVERLAP_RATIO = 0.0 (one global batch).
#   Pass 2: the five single-slide tissues at OVERLAP_RATIO = 0.5, one tissue
#           at a time (since the groovy uses args[0]=FORCE_TISSUE / args[1]=
#           OVERLAP_RATIO and skips images whose URI doesn't match).
#
# Requires QuPath >= 0.5 on PATH. The QuPath project is expected at
#   ${QPROJ:-${CELLVIT_TRAINING_ROOT}/../data/qprj/project.qpproj}
#
# Usage:
#   bash export_all_tissues.sh                    # default: pass 1 + pass 2
#   bash export_all_tissues.sh --single-slide     # pass 2 only (5 tissues @ 0.5)
#   bash export_all_tissues.sh --multi-slide      # pass 1 only (all @ 0.0)
#   bash export_all_tissues.sh --tissues "heart"  # explicit subset, overlap 0.5
#   bash export_all_tissues.sh --overlap 0.25     # override overlap for pass 2
#   bash export_all_tissues.sh --dry-run          # print the plan, don't run
#   bash export_all_tissues.sh --qproj <path>     # override QuPath project path
#
# Notes:
# - Pass 1 calls QuPath ONCE over the project; the groovy auto-routes each
#   image to trainingset/<tissue>/ based on the 'data/xenium/<tissue>/' URI.
# - Pass 2 calls QuPath once per single-slide tissue, passing
#   args=[<tissue>, <overlap>], so the groovy skips images whose URI does
#   not contain 'data/xenium/<tissue>/'.
# - export_tiles.groovy is resumable: PNGs that already exist are skipped,
#   so re-running this is cheap.
# -----------------------------------------------------------------------------
set -euo pipefail

# Shared helpers: _lib::cellvit_training_root, etc.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

CELLVIT_TRAINING_ROOT="$(_lib::cellvit_training_root)"
GROOVY="${CELLVIT_TRAINING_ROOT}/pipeline/qupath/export_tiles.groovy"

QPROJ="${QPROJ:-${CELLVIT_TRAINING_ROOT}/../data/qprj/project.qpproj}"
QUPATH="${QUPATH:-QuPath}"

SINGLE_SLIDE_TISSUES=(heart brain cervix prostate lymph_node)
OVERLAP=0.5
MODE="both"
TISSUES=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single-slide) MODE="single"; shift ;;
        --multi-slide)  MODE="multi";  shift ;;
        --tissues)      MODE="single"; TISSUES="$2"; shift 2 ;;
        --overlap)      OVERLAP="$2"; shift 2 ;;
        --qproj)        QPROJ="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${QPROJ}" ]]; then
    echo "ERROR: QuPath project not found: ${QPROJ}" >&2
    echo "       Pass --qproj <path> or export QPROJ=..." >&2
    exit 1
fi
if [[ ! -f "${GROOVY}" ]]; then
    echo "ERROR: groovy not found: ${GROOVY}" >&2
    exit 1
fi

run_qupath() {
    local desc="$1"; shift
    echo
    echo "================================================================"
    echo "  ${desc}"
    echo "    ${QUPATH} script -p ${QPROJ} $* ${GROOVY}"
    echo "================================================================"
    if [[ "${DRY_RUN}" == 1 ]]; then return 0; fi
    "${QUPATH}" script -p "${QPROJ}" "$@" "${GROOVY}"
}

# ── Pass 1: multi-slide tissues, no overlap ──────────────────────────────
if [[ "${MODE}" == "both" || "${MODE}" == "multi" ]]; then
    run_qupath "Pass 1: all multi-slide tissues (auto-route, overlap=0.0)"
fi

# ── Pass 2: single-slide tissues, overlap=${OVERLAP} ─────────────────────
if [[ "${MODE}" == "both" || "${MODE}" == "single" ]]; then
    if [[ -z "${TISSUES}" ]]; then
        TISSUES="${SINGLE_SLIDE_TISSUES[*]}"
    fi
    for t in ${TISSUES}; do
        run_qupath "Pass 2: ${t} (overlap=${OVERLAP})" \
            -a "${t}" -a "${OVERLAP}"
    done
fi

echo
echo "[export_all_tissues] Done."
