#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# aggregate_pantissue.sh -- build trainingset/pantissue/ as a symlink-union of
# every per-tissue trainingset/<tissue>/train/{images,labels,tile_geometry}/.
#
# All tissues must share the same label_map.yaml (enforced via md5 check).
# SAMPLE_TAG collisions across tissues are auto-prefixed with `<tissue>__`.
#
# Run AFTER export_all_tissues.sh has finished for every tissue.
#
# Usage:
#   bash aggregate_pantissue.sh                          # all tissues found
#   bash aggregate_pantissue.sh --tissues "breast lung"  # explicit subset
#   bash aggregate_pantissue.sh --force                  # wipe pantissue/train/ first
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CELLVIT_TRAINING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAININGSET="${CELLVIT_TRAINING_ROOT}/trainingset"
TEMPLATES="${CELLVIT_TRAINING_ROOT}/templates"

TISSUES=""
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tissues) TISSUES="$2"; shift 2 ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Discover tissues with populated train/labels/ if none given.
if [[ -z "${TISSUES}" ]]; then
    for d in "${TRAININGSET}"/*/; do
        t="$(basename "$d")"
        [[ "$t" == "pantissue" ]] && continue
        if [[ -d "${d}train/labels" ]] && \
           [[ -n "$(ls -A "${d}train/labels" 2>/dev/null)" ]]; then
            TISSUES+=" $t"
        fi
    done
    TISSUES="${TISSUES# }"
fi

if [[ -z "${TISSUES}" ]]; then
    echo "ERROR: no source tissues with populated train/labels/ found under ${TRAININGSET}" >&2
    exit 1
fi

echo "Source tissues : ${TISSUES}"
echo "Target         : ${TRAININGSET}/pantissue"

# Verify all tissues share the same label_map.yaml.
if [[ ! -f "${TEMPLATES}/label_map.yaml" ]]; then
    echo "ERROR: canonical label_map missing: ${TEMPLATES}/label_map.yaml" >&2
    exit 1
fi
CANON_MD5=$(md5sum "${TEMPLATES}/label_map.yaml" | cut -d' ' -f1)
for t in ${TISSUES}; do
    lm="${TRAININGSET}/${t}/label_map.yaml"
    if [[ ! -f "$lm" ]]; then
        echo "ERROR: missing label_map.yaml for tissue '${t}': $lm" >&2; exit 1
    fi
    md5=$(md5sum "$lm" | cut -d' ' -f1)
    if [[ "$md5" != "${CANON_MD5}" ]]; then
        echo "ERROR: label_map.yaml for '${t}' differs from canonical." >&2
        echo "       expected md5=${CANON_MD5}, got md5=${md5}" >&2
        exit 1
    fi
done
echo "label_map.yaml : OK (all match canonical, md5=${CANON_MD5:0:8})"

# (Re)create the pantissue directory.
PANTISSUE="${TRAININGSET}/pantissue"
if [[ "${FORCE}" == 1 && -d "${PANTISSUE}/train" ]]; then
    echo "  --force: wiping ${PANTISSUE}/train"
    rm -rf "${PANTISSUE}/train"
fi
mkdir -p "${PANTISSUE}/train/images" \
         "${PANTISSUE}/train/labels" \
         "${PANTISSUE}/train/tile_geometry" \
         "${PANTISSUE}/splits/fold_0"
cp "${TEMPLATES}/label_map.yaml" "${PANTISSUE}/label_map.yaml"

# Detect SAMPLE_TAG collisions across tissues. SAMPLE_TAG is the file basename
# of tile_geometry/*.json (one per slide).
echo ""
echo "── Checking for SAMPLE_TAG collisions across tissues ──"
COLLISION=0
declare -A SEEN
for t in ${TISSUES}; do
    geom_dir="${TRAININGSET}/${t}/train/tile_geometry"
    if [[ ! -d "${geom_dir}" ]]; then
        echo "WARN: ${t} has no tile_geometry/ (export may have used old groovy)"
        continue
    fi
    for f in "${geom_dir}"/*.json; do
        [[ -e "$f" ]] || continue
        tag="$(basename "$f" .json)"
        if [[ -n "${SEEN[${tag}]:-}" ]]; then
            echo "  collision: '${tag}' in both '${SEEN[${tag}]}' and '${t}'"
            COLLISION=1
        else
            SEEN[${tag}]="${t}"
        fi
    done
done
if [[ "${COLLISION}" == 1 ]]; then
    PREFIX_MODE=1
    echo "  -> auto-prefixing with '<tissue>__' to disambiguate."
else
    PREFIX_MODE=0
    echo "  -> no collisions; symlinks will keep original basenames."
fi

# Symlink everything.
echo ""
echo "── Symlinking ──"
N_IMG=0; N_CSV=0; N_GEOM=0
for t in ${TISSUES}; do
    src_root="${TRAININGSET}/${t}/train"
    for sub in images labels tile_geometry; do
        src_dir="${src_root}/${sub}"
        [[ -d "${src_dir}" ]] || continue
        for f in "${src_dir}"/*; do
            [[ -e "$f" ]] || continue
            base="$(basename "$f")"
            if [[ "${PREFIX_MODE}" == 1 ]]; then
                target="${PANTISSUE}/train/${sub}/${t}__${base}"
            else
                target="${PANTISSUE}/train/${sub}/${base}"
            fi
            # idempotent: remove any prior symlink/file at target
            [[ -L "${target}" || -e "${target}" ]] && rm -f "${target}"
            ln -s "$f" "${target}"
            case "${sub}" in
                images)        N_IMG=$((N_IMG+1)) ;;
                labels)        N_CSV=$((N_CSV+1)) ;;
                tile_geometry) N_GEOM=$((N_GEOM+1)) ;;
            esac
        done
    done
done

echo "  images        : ${N_IMG} symlinks"
echo "  labels        : ${N_CSV} symlinks"
echo "  tile_geometry : ${N_GEOM} sidecars"
echo ""
echo "Done. Next:"
echo "  python ${SCRIPT_DIR}/make_splits.py --tissue pantissue --val-frac 0.1"
