#!/usr/bin/env bash
# conda-setup.sh — create and populate the wsinsight-train conda environment.
#
# Usage:  bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset]
#
#   -n | --name  ENV_NAME   Conda environment to use (default: current active env).
#   -r | --reset            Deactivate, remove, recreate, and activate the env.
#                           Without this flag the script only (re-)installs
#                           packages into the existing env.
#
# Installs: torch + cellpose + kurtorank + wsitrain (+ optional stardist).
# kurtorank is installed editable from the sibling repo (not on PyPI).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KURTORANK_DIR="$(cd "${SCRIPT_DIR}/../kurtorank" && pwd 2>/dev/null || true)"

ENV_NAME="${CONDA_DEFAULT_ENV:-}"
DO_RESET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name) ENV_NAME="${2:?-n requires a name}"; shift 2 ;;
        -r|--reset) DO_RESET=1; shift ;;
        *) echo "Usage: bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset]" >&2; exit 1 ;;
    esac
done

if [[ -z "$ENV_NAME" ]]; then
    echo "Error: no env specified and none active. Use -n ENV_NAME." >&2; exit 1
fi
echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET})"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -z "${CONDA_BASE}" ]]; then
    for _base in /opt/conda /opt/anaconda3; do
        if [[ -f "${_base}/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="${_base}"
            break
        fi
    done
fi
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "Error: cannot locate conda.sh. Activate conda first or set CONDA_BASE." >&2
    exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ "$DO_RESET" -eq 1 ]]; then
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
    conda create -n "${ENV_NAME}" python=3.11 "setuptools<67" -c conda-forge -y
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# Redirect pip cache off NAS to dodge inode quotas (seen on this cluster).
pip cache purge || true
export PIP_CACHE_DIR=/tmp/pip-cache-wsitrain

# Heavy stack first.
pip install torch torchvision nvidia-ml-py
pip install cellpose

# StarDist is the default segmentation backend.
pip install "stardist" tensorflow || echo "WARNING: stardist install failed; use --segmenter cellpose"

# Runtime + test deps not covered by the heavy stack above.
# zarr is what keeps the tile stage from loading whole slides into RAM.
pip install pyarrow pytest zarr

# kurtorank (editable, not on PyPI).
if [[ -n "${KURTORANK_DIR}" && -f "${KURTORANK_DIR}/pyproject.toml" ]]; then
    pip install --no-deps -e "${KURTORANK_DIR}"
else
    echo "WARNING: kurtorank repo not found at ../../kurtorank; install it manually."
fi

# wsitrain itself (deps above already satisfied).
pip install --no-deps -e "${SCRIPT_DIR}"

# Smoke test.
echo "---- smoke test ----"
wsitrain --version
python -m pytest "${SCRIPT_DIR}/tests" -q || echo "WARNING: test suite did not pass"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
echo "Done. Set CELLVIT_ROOT, then: wsitrain check --input <dir> --tissue pantissue"
