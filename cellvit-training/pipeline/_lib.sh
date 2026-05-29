#!/usr/bin/env bash
# =============================================================================
# _lib.sh -- shared helpers for cellvit-training/pipeline/ bash drivers.
#
# Source this from any pipeline script:
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"
#
# All functions are namespaced as `_lib::*` so they cannot conflict with
# caller-defined identifiers. None of them have side effects beyond echoing
# strings to stdout (except `_lib::tissue_paths`, which exports variables).
# =============================================================================

# ── Path resolution ──────────────────────────────────────────────────────────
# These resolve once per shell using the location of this file as the anchor.
# _PIPELINE_DIR is where this file lives. _CELLVIT_TRAINING_ROOT is its parent.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CELLVIT_TRAINING_ROOT="$(cd "${_LIB_DIR}/.." && pwd)"

_lib::pipeline_dir()           { echo "${_LIB_DIR}"; }
_lib::cellvit_training_root()  { echo "${_CELLVIT_TRAINING_ROOT}"; }
_lib::trainingset_root()       { echo "${_CELLVIT_TRAINING_ROOT}/trainingset"; }
_lib::cellvit_root()           { echo "${_CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus"; }
_lib::logs_local()             { echo "${_CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local"; }
_lib::templates_dir()          { echo "${_CELLVIT_TRAINING_ROOT}/templates"; }

# Default python for invoking cellvit/* and validate_classifier.py.
_lib::python() {
    if [[ -n "${PYTHON:-}" ]]; then
        echo "${PYTHON}"
    elif [[ -x "/opt/anaconda3/envs/wsinsight/bin/python3" ]]; then
        echo "/opt/anaconda3/envs/wsinsight/bin/python3"
    else
        command -v python3
    fi
}

# ── Per-tissue path resolution ───────────────────────────────────────────────
# Usage:
#     eval "$(_lib::tissue_paths breast)"        # default fold_0, SAM-H-x40
#     eval "$(_lib::tissue_paths breast fold_0 SAM-H-x40)"
#
# Exports TISSUE, FOLD, BACKBONE, TRAINSET, CONFIG, VAL_CSV, LABEL_MAP for the
# caller. Existence of files is NOT checked here — callers do that themselves.
_lib::tissue_paths() {
    local tissue="${1:?usage: _lib::tissue_paths <tissue> [fold=fold_0] [backbone=SAM-H-x40]}"
    local fold="${2:-fold_0}"
    local backbone="${3:-SAM-H-x40}"
    local trainset="${_CELLVIT_TRAINING_ROOT}/trainingset/${tissue}"
    cat <<EOF
TISSUE='${tissue}'
FOLD='${fold}'
BACKBONE='${backbone}'
TRAINSET='${trainset}'
CONFIG='${trainset}/train_configs/${backbone}/${fold}.yaml'
VAL_CSV='${trainset}/splits/${fold}/val.csv'
LABEL_MAP='${trainset}/label_map.yaml'
EOF
}

# ── Tissue discovery ─────────────────────────────────────────────────────────
# All three echo a space-separated list, sorted alphabetically. They exclude
# 'pantissue' from the result *unless* the caller passes --include-pantissue.
# The aggregated trainingset/pantissue/ is a real tissue for training purposes
# but is special-cased here because most drivers want to loop only over the
# "source" tissues that pantissue is built from.

_lib::_filter_pantissue() {
    # stdin = list of tissues (one per line); arg 1 = "include" | "exclude"
    local mode="${1:-exclude}"
    if [[ "${mode}" == "include" ]]; then cat; else grep -v '^pantissue$' || true; fi
}

# Tissues with at least one *.csv under train/labels/
_lib::tissues_with_labels() {
    local mode="${1:-exclude}"   # exclude | include  (pantissue)
    local root="${_CELLVIT_TRAINING_ROOT}/trainingset"
    [[ -d "${root}" ]] || return 0
    for d in "${root}"/*/; do
        local t="$(basename "${d}")"
        if compgen -G "${d}/train/labels/*.csv" >/dev/null 2>&1; then
            echo "${t}"
        fi
    done | sort -u | _lib::_filter_pantissue "${mode}" | tr '\n' ' ' | sed 's/ $//'
}

# Tissues with a populated splits/<fold>/val.csv
_lib::tissues_with_splits() {
    local fold="${1:-fold_0}"
    local mode="${2:-exclude}"
    local root="${_CELLVIT_TRAINING_ROOT}/trainingset"
    [[ -d "${root}" ]] || return 0
    for d in "${root}"/*/; do
        local t="$(basename "${d}")"
        [[ -f "${d}splits/${fold}/val.csv" ]] && echo "${t}"
    done | sort -u | _lib::_filter_pantissue "${mode}" | tr '\n' ' ' | sed 's/ $//'
}

# Every directory under trainingset/ (whether or not it has data).
_lib::tissues_in_dataset() {
    local mode="${1:-exclude}"
    local root="${_CELLVIT_TRAINING_ROOT}/trainingset"
    [[ -d "${root}" ]] || return 0
    for d in "${root}"/*/; do echo "$(basename "${d}")"; done \
        | sort -u | _lib::_filter_pantissue "${mode}" | tr '\n' ' ' | sed 's/ $//'
}

# ── Training-run identification ──────────────────────────────────────────────
# log_comment = "<tissue>-<task>-<backbone-lower>"
# Used both as the directory suffix under logs_local/ and as the YAML field
# that determines that suffix.
_lib::log_comment() {
    local tissue="${1:?usage: _lib::log_comment <tissue> <task> <backbone>}"
    local task="${2:?missing task}"
    local backbone="${3:?missing backbone}"
    echo "${tissue}-${task}-$(echo "${backbone}" | tr '[:upper:]' '[:lower:]')"
}

# Find the newest run directory matching *_<log_comment> under logs_local/.
# Searches BOTH the top level and one level deeper (since the trainer may
# nest a fresh <ts>_<lc>/ inside a pre-existing parent <ts>_<lc>/ when the
# YAML's `log_dir` points at an older run dir). Returns the newest match by
# checkpoints/model_best.pth mtime; falls back to dir mtime if no ckpt yet.
# Echoes the absolute path, or empty string if none found.
_lib::find_latest_run() {
    local lc="${1:?usage: _lib::find_latest_run <log_comment>}"
    local base="${_CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local"
    # Prefer run dirs that actually contain a finished model_best.pth.
    local ckpt
    ckpt="$(ls -t "${base}/"*"_${lc}/checkpoints/model_best.pth" \
                  "${base}/"*"_${lc}/"*"_${lc}/checkpoints/model_best.pth" \
                  2>/dev/null | head -1)"
    if [[ -n "${ckpt}" ]]; then
        dirname "$(dirname "${ckpt}")"
        return
    fi
    # No checkpoint yet — fall back to newest matching directory.
    ls -td "${base}/"*"_${lc}" "${base}/"*"_${lc}/"*"_${lc}" 2>/dev/null | head -1
}

# ── Validation invocation ────────────────────────────────────────────────────
# Wraps validate_classifier.py with the standard four paths and tee'd logging.
# Args: tissue fold backbone run_dir
_lib::run_validate() {
    local tissue="${1:?usage: _lib::run_validate <tissue> <fold> <backbone> <run_dir>}"
    local fold="${2:?missing fold}"
    local backbone="${3:?missing backbone}"
    local run_dir="${4:?missing run_dir}"
    local trainset="${_CELLVIT_TRAINING_ROOT}/trainingset/${tissue}"
    local val_csv="${trainset}/splits/${fold}/val.csv"
    local label_map="${trainset}/label_map.yaml"
    local checkpoint="${run_dir}/checkpoints/model_best.pth"
    local cellvit_root="${_CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus"
    local logs_dir="${cellvit_root}/logs_local"
    local py; py="$(_lib::python)"

    PYTHONPATH="${cellvit_root}" "${py}" -u \
        "${_LIB_DIR}/validate_classifier.py" \
        --checkpoint "${checkpoint}" \
        --dataset    "${trainset}" \
        --filelist   "${val_csv}" \
        --label-map  "${label_map}" \
        --outdir     "${run_dir}/validation" \
        2>&1 | tee "${logs_dir}/${tissue}_validate.log"
}
