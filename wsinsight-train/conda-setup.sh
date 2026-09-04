#!/usr/bin/env bash
# conda-setup.sh — create and populate the wsinsight-train conda environment.
#
# <<<USAGE_START>>>
# Usage:  bash ./conda-setup.sh ENV_NAME [-r|--reset] [-m|--mcp] [-d|--dev] [-h|--help]
#
#   ENV_NAME                (positional, REQUIRED) Conda environment to use/create.
#                           There is NO fallback to the currently-activated conda env:
#                           the name is mandatory so `-r` can never accidentally
#                           destroy a different active environment.
#   -r | --reset            Deactivate, remove, recreate, and activate ENV_NAME.
#                           Without this flag the script only (re-)installs
#                           packages into the existing env.
#   -m | --mcp              Also install fastmcp, which backs the
#                           `wsinsight-train-mcp` console script. Not installed
#                           by default to keep the env lean.
#   -d | --dev              Also install the dev tools (pytest, pytest-cov, ruff,
#                           pre_commit) so the post-install smoke test can run the
#                           real test suite. Without -d the suite is SKIPped if
#                           pytest is missing; with -d it FAILS (you asked for it).
#                           The package itself is always installed editable (-e).
#   -h | --help             Print this help message and exit.
# <<<USAGE_END>>>
#
# Installs: torch + cellpose + kurtorank + wsitrain (+ optional stardist).
# kurtorank is installed editable from the sibling repo (not on PyPI).
# See the StarDist block below for pointing wsitrain at a pre-downloaded model.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KURTORANK_DIR="$(cd "${SCRIPT_DIR}/../kurtorank" && pwd 2>/dev/null || true)"

DO_RESET=0
DO_MCP=0
DO_DEV=0

print_usage() {
    awk '
        /<<<USAGE_START>>>/ {capture=1; next}
        /<<<USAGE_END>>>/   {capture=0}
        capture            {sub(/^# ?/, ""); print}
    ' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) print_usage; exit 0 ;;
        -r|--reset) DO_RESET=1; shift ;;
        -m|--mcp) DO_MCP=1; shift ;;
        -d|--dev) DO_DEV=1; shift ;;
        -*) echo "Unknown option: $1" >&2; echo "Run '${0##*/} --help' for usage." >&2; exit 1 ;;
        *)
            if [[ -n "${ENV_NAME:-}" ]]; then
                echo "Error: only one positional argument (ENV_NAME) is accepted; got '$ENV_NAME' and '$1'." >&2
                echo "Run '${0##*/} --help' for usage." >&2
                exit 1
            fi
            ENV_NAME="$1"; shift ;;
    esac
done

if [[ -z "${ENV_NAME:-}" ]]; then
    echo "Error: ENV_NAME is required." >&2
    echo "       Run '${0##*/} --help' for usage." >&2
    exit 1
fi
echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET}, mcp=${DO_MCP}, dev=${DO_DEV})"

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
# Exported before any purge: `pip cache purge` obeys this variable, so purging
# first wiped the user's global ~/.cache/pip. Shared dir so the sibling repos
# reuse the multi-hundred-MB torch/TF/cuDNN wheels.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-wsinsight-stack}"

# Heavy stack first, pinned to the same versions as wsinsight/sptxinsight so that
# TensorFlow (StarDist) and PyTorch (CellViT) coexist.
CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"
pip install -c "${CONSTRAINTS}" torch torchvision nvidia-ml-py
pip install -c "${CONSTRAINTS}" cellpose

# StarDist is the default segmentation backend.
pip install -c "${CONSTRAINTS}" "numpy<2" stardist tensorflow \
    || echo "WARNING: stardist install failed; use --segmenter cellpose"

# Using an already-downloaded StarDist model (offline / air-gapped hosts).
# Without one, StarDist2D.from_pretrained() fetches the weights at first run.
# A csbdeep model folder is <parent>/<model-name>/ holding config.json,
# thresholds.json and weights_best.h5. wsitrain looks for the PARENT directory
# in this order and takes the first hit:
#
#   1. wsitrain run ... --stardist-model-dir /data/models/StarDist2D
#      (equivalently `stardist_model_dir:` in wsitrain/defaults/run.yaml).
#   2. $WSITRAIN_STARDIST_DIR   -- same meaning, set once per host.
#   3. $KERAS_HOME/models/StarDist2D
#   4. ~/.keras/models/StarDist2D  -- the default csbdeep cache.
#
# Add --stardist-model NAME if the folder is not named 2D_versatile_he.

# Runtime deps not covered by the heavy stack above.
# zarr is what keeps the tile stage from loading whole slides into RAM.
pip install -c "${CONSTRAINTS}" pyarrow zarr pyyaml tifffile pillow tqdm

# Dev tools are installed directly, not via a [dev] extra: the editable install
# below uses --no-deps, which would drop extras too.
if [[ "${DO_DEV}" -eq 1 ]]; then
    pip install -c "${CONSTRAINTS}" pytest pytest-cov ruff pre_commit
fi

# Same reasoning for the [mcp] extra.
if [[ "${DO_MCP}" -eq 1 ]]; then
    pip install -c "${CONSTRAINTS}" fastmcp
fi

# kurtorank (editable, not on PyPI).
if [[ -n "${KURTORANK_DIR}" && -f "${KURTORANK_DIR}/pyproject.toml" ]]; then
    # Its deps are installed explicitly because the editable install below uses
    # --no-deps: letting pip resolve them relaxes the numpy<2 / zarr<3 generation
    # that stardist and the shared wsinsight env depend on.
    pip install -c "${CONSTRAINTS}" \
        anndata scanpy squidpy spatialdata spatialdata-io \
        xarray dask distributed pandas scipy statsmodels matplotlib seaborn click
    # Only kurtorank's rank/marker-* commands need these, and they import them
    # lazily, so a failure here still leaves `kurtorank annotate` working.
    pip install -c "${CONSTRAINTS}" cellxgene-census tiledbsoma \
        || echo "WARNING: cellxgene-census/tiledbsoma failed; kurtorank marker-*/rank unavailable"
    pip install --no-deps -e "${KURTORANK_DIR}"
else
    echo "WARNING: kurtorank repo not found at ${SCRIPT_DIR}/../kurtorank; install it manually."
fi

# wsitrain itself (deps above already satisfied).
pip install --no-deps -e "${SCRIPT_DIR}"

# ── Smoke test ────────────────────────────────────────────────────────────────
# Hard checks are fatal: a half-installed env must not look like a success.
# The test suite is reported but does not fail the setup.
echo "---- smoke test ----"
SMOKE_FAIL=0
smoke() {                       # smoke <label> <command...>
    label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf '  PASS  %s\n' "$label"
    else
        printf '  FAIL  %s\n' "$label"
        SMOKE_FAIL=$((SMOKE_FAIL + 1))
    fi
}

python -c 'import importlib.metadata as m; print("  numpy", m.version("numpy"), "| torch", m.version("torch"))' || true
python -c 'import torch; print("  cuda", torch.cuda.is_available(), torch.cuda.device_count())' || true

smoke "wsitrain on PATH"     command -v wsitrain
smoke "wsitrain --help"      wsitrain --help
smoke "import wsitrain"      python -c 'import wsitrain'
# `kurtorank --help` is the gate that catches a --no-deps install whose
# dependency tree was never populated; a bare `import kurtorank` does not.
smoke "kurtorank on PATH"    command -v kurtorank
smoke "kurtorank --help"     kurtorank --help
smoke "import torch"         python -c 'import torch'
smoke "numpy < 2"            python -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)'
if [[ "${DO_MCP}" -eq 1 ]]; then
    smoke "wsinsight-train-mcp on PATH" command -v wsinsight-train-mcp
    smoke "wsinsight-train-mcp --help"  wsinsight-train-mcp --help
    # kurtorank is installed into this env too, so its server shares the fastmcp.
    if [[ -n "${KURTORANK_DIR}" && -f "${KURTORANK_DIR}/pyproject.toml" ]]; then
        smoke "kurtorank-mcp on PATH"   command -v kurtorank-mcp
        smoke "kurtorank-mcp --help"    kurtorank-mcp --help
    fi
fi
# StarDist is the default --segmenter but its install is tolerated-failure, so
# this warns rather than failing setup; cellpose is the documented fallback.
python -c 'import stardist' >/dev/null 2>&1 \
    && echo "  PASS  stardist importable" \
    || echo "  WARN  stardist unavailable; use --segmenter cellpose (non-fatal)"
# Reported, not enforced: without a local copy the weights are downloaded on the
# first run, which an offline host cannot do.
# wsitrain.segment owns the lookup order, so ask it instead of retyping paths.
STARDIST_DIR=$(python -c 'from wsitrain.segment import _cached_stardist_dir
print(_cached_stardist_dir("2D_versatile_he") or "")' 2>/dev/null)
if [[ -n "${STARDIST_DIR}" ]]; then
    echo "  PASS  2D_versatile_he found in ${STARDIST_DIR}"
else
    echo "  INFO  2D_versatile_he not cached; it downloads on first run."
    echo "        Offline: unpack it to ~/.keras/models/StarDist2D/2D_versatile_he/,"
    echo "        export WSITRAIN_STARDIST_DIR=<parent-of-model-folder>,"
    echo "        or pass --stardist-model-dir <parent-of-model-folder>."
fi

if [[ -d "${SCRIPT_DIR}/tests" ]]; then
    if python -c "import pytest" >/dev/null 2>&1; then
        python -m pytest "${SCRIPT_DIR}/tests" -q \
            && echo "  PASS  test suite" \
            || echo "  WARN  test suite did not pass (non-fatal)"
    elif [[ "${DO_DEV}" -eq 1 ]]; then
        echo "  FAIL  test suite: pytest missing but -d/--dev was requested" >&2
        smoke "pytest importable (dev)" python -c "import pytest"
    else
        echo "  SKIP  test suite (pytest not installed; rerun with -d/--dev)"
    fi
fi

if [[ "${SMOKE_FAIL}" -ne 0 ]]; then
    echo "smoke test: ${SMOKE_FAIL} check(s) FAILED" >&2
    exit 1
fi
echo "smoke test: all checks passed"
echo "Done. Set CELLVIT_ROOT, then: wsitrain check --input <dir> --tissue pantissue"
