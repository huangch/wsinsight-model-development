#!/usr/bin/env bash
# conda-setup.sh — create and populate a standalone kurtorank conda environment.
#
# Usage:  bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-c|--census]
#
#   -n | --name  ENV_NAME   Conda environment to use (default: current active env).
#   -r | --reset            Deactivate, remove, recreate, and activate the env.
#                           Without this flag the script only (re-)installs
#                           packages into the existing env.
#   -c | --census           Also install cellxgene-census + tiledbsoma, needed by
#                           the `rank` and `marker-*` commands. Left out by
#                           default: they are heavy, and every use site imports
#                           them lazily, so `annotate` works without them.
#
# For the shared env used by wsinsight-train, use
# wsinsight-train/conda-setup.sh instead — it installs this package too.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_NAME="${CONDA_DEFAULT_ENV:-}"
DO_RESET=0
DO_CENSUS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name) ENV_NAME="${2:?-n requires a name}"; shift 2 ;;
        -r|--reset) DO_RESET=1; shift ;;
        -c|--census) DO_CENSUS=1; shift ;;
        *) echo "Usage: bash ./conda-setup.sh [-n ENV_NAME] [-r|--reset] [-c|--census]" >&2; exit 1 ;;
    esac
done

if [[ -z "$ENV_NAME" ]]; then
    echo "Error: no env specified and none active. Use -n ENV_NAME." >&2; exit 1
fi
echo "Target conda environment: ${ENV_NAME}  (reset=${DO_RESET} census=${DO_CENSUS})"

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
    # setuptools<67 keeps pkg_resources available for stardist if this env is
    # later shared with wsinsight-train.
    conda create -n "${ENV_NAME}" python=3.11 "setuptools<67" -c conda-forge -y
fi

conda activate "${ENV_NAME}"
pip install --upgrade pip

# Redirect pip cache off NAS to dodge inode quotas (seen on this cluster).
# Exported before any purge: `pip cache purge` obeys this variable, so purging
# first wiped the user's global ~/.cache/pip. Shared dir so the sibling repos
# reuse the multi-hundred-MB torch/TF/cuDNN wheels.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-wsinsight-stack}"

CONSTRAINTS="${SCRIPT_DIR}/constraints.txt"

# torch first so the CUDA 12 wheel set is settled before anything can pull the
# CUDA 13 generation in behind it.
pip install -c "${CONSTRAINTS}" torch torchvision

# Deps are installed explicitly because the editable install below uses
# --no-deps: letting pip resolve them relaxes the numpy<2 / zarr<3 generation
# that this env shares with wsinsight and stardist.
pip install -c "${CONSTRAINTS}" \
    anndata scanpy squidpy spatialdata spatialdata-io \
    numpy zarr xarray dask distributed \
    pandas scipy statsmodels matplotlib seaborn click tqdm pyarrow pytest

if [[ "$DO_CENSUS" -eq 1 ]]; then
    pip install -c "${CONSTRAINTS}" cellxgene-census tiledbsoma \
        || echo "WARNING: cellxgene-census/tiledbsoma install failed; rank and marker-* unavailable"
fi

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

python -c 'import importlib.metadata as m; print("  numpy", m.version("numpy"), "| anndata", m.version("anndata"))' || true

smoke "kurtorank on PATH"    command -v kurtorank
# `--help` rather than `import kurtorank`: it walks the import tree the CLI
# needs, so it catches a --no-deps install whose deps were never populated.
smoke "kurtorank --help"     kurtorank --help
smoke "import anndata"       python -c 'import anndata'
smoke "numpy < 2"            python -c 'import numpy, sys; sys.exit(int(numpy.__version__.split(".")[0]) >= 2)'
if [[ "$DO_CENSUS" -eq 1 ]]; then
    # Census install is tolerated-failure above, so warn rather than fail here.
    python -c 'import cellxgene_census' >/dev/null 2>&1 \
        && echo "  PASS  cellxgene_census importable" \
        || echo "  WARN  cellxgene_census unavailable; rank/marker-* disabled (non-fatal)"
fi

if [[ -d "${SCRIPT_DIR}/tests" ]]; then
    PYTHONPATH="${SCRIPT_DIR}/src" python -m pytest "${SCRIPT_DIR}/tests" -q \
        && echo "  PASS  test suite" \
        || echo "  WARN  test suite did not pass (non-fatal)"
fi

if [[ "${SMOKE_FAIL}" -ne 0 ]]; then
    echo "smoke test: ${SMOKE_FAIL} check(s) FAILED" >&2
    exit 1
fi
echo "smoke test: all checks passed"
echo "Done. Try: kurtorank annotate --xenium-dir <outs> --tissue-type breast --output-dir <outs>"
