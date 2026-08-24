# ====================================
# wsinsight-train — CellViT model training
# CUDA 12.8 + cuDNN + Ubuntu 22.04
# ====================================
#
# Build from the wsinsight-model-development/ parent directory:
#   docker build -f wsinsight-train/Dockerfile -t wsinsight-train .
#
# kurtorank is a sibling package in ../kurtorank; building from the parent
# directory lets COPY include both repos without hacks.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ------------------------------------
# Basic system dependencies
# ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget unzip vim ca-certificates build-essential pkg-config \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------
# Install Miniconda (Python 3.11)
# No GDAL needed — wsinsight-train has no geospatial dependencies.
# ------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p "$CONDA_DIR" \
 && rm /tmp/mc.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# ------------------------------------
# Accept Anaconda TOS (required since Sept 2024)
# ------------------------------------
RUN conda --version && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main -y || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r -y || true)

# ------------------------------------
# Create training environment (pure Python, no conda-forge packages needed)
# ------------------------------------
RUN conda update -n base --yes --override-channels -c conda-forge conda && \
    conda create -y -n wsitrain -c conda-forge python=3.11 "setuptools<67" pip && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=wsitrain
ENV PATH="$CONDA_DIR/envs/wsitrain/bin:$PATH"

RUN python -m pip install --upgrade pip

# ------------------------------------
# Global Conda shell initialization
# ------------------------------------
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'conda activate wsitrain' >> /etc/bash.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/skel/.bashrc && \
    echo 'conda activate wsitrain' >> /etc/skel/.bashrc

# ------------------------------------
# Copy source (context = wsinsight-model-development/)
# ------------------------------------
WORKDIR /app
COPY ./wsinsight-train /app/wsinsight-train
COPY ./kurtorank       /app/kurtorank

# Every pip layer below is constrained. Unconstrained, torch resolves to >=2.9,
# which pulls the CUDA 13 wheel set over nvidia/cudnn/lib and leaves TensorFlow
# (StarDist) loading a cuDNN it was not built against.
# ARG rather than ENV so the path does not leak into the runtime environment.
ARG CONSTR=/app/wsinsight-train/constraints.txt

# ------------------------------------
# Install PyTorch + GPU support
# ------------------------------------
RUN pip install --retries 10 -c "$CONSTR" torch torchvision nvidia-ml-py

# ------------------------------------
# Install cellpose (cell segmentation backbone)
# ------------------------------------
RUN pip install --retries 10 -c "$CONSTR" cellpose

# ------------------------------------
# Install optional StarDist + TensorFlow (default segmentation backend)
# Tolerate failure — wsitrain can fall back to --segmenter cellpose. stderr is
# deliberately NOT suppressed: if this is skipped the build log must show why.
# ------------------------------------
RUN pip install --retries 5 -c "$CONSTR" "numpy<2" stardist tensorflow || \
    echo "WARNING: stardist/tensorflow optional install skipped; use --segmenter cellpose"

# ------------------------------------
# wsitrain's own runtime deps. It is installed --no-deps below, so nothing else
# would pull these in. zarr is what keeps the tile stage from loading whole
# slides into RAM.
# ------------------------------------
RUN pip install --retries 10 -c "$CONSTR" \
        pyarrow pyyaml tifffile pillow tqdm zarr

# ------------------------------------
# kurtorank's runtime deps. These are NOT a subset of the layers above — the
# preceding torch/cellpose/stardist layers pull in none of the AnnData stack.
# ------------------------------------
RUN pip install --retries 10 -c "$CONSTR" \
        anndata scanpy squidpy spatialdata spatialdata-io \
        xarray dask distributed pandas scipy statsmodels matplotlib seaborn click

# Only kurtorank's rank/marker-* commands need these, and every import site is
# lazy, so a failure here still leaves `kurtorank annotate` working.
RUN pip install --retries 5 -c "$CONSTR" cellxgene-census tiledbsoma || \
    echo "WARNING: cellxgene-census/tiledbsoma skipped; kurtorank rank/marker-* unavailable"

# ------------------------------------
# Install kurtorank (unsupervised annotation library, not on PyPI)
# --no-deps so pip cannot relax the numpy<2 / zarr<3 generation pinned above.
# ------------------------------------
RUN pip install --no-deps -e /app/kurtorank

# ------------------------------------
# Install wsitrain itself
# --no-deps for the same reason; its deps are installed above.
# ------------------------------------
RUN pip install --no-deps -e /app/wsinsight-train

# ------------------------------------
# Dev extras (test runner + coverage). Not part of the runtime image — strip
# this layer in production builds by overriding the ``WSITRAIN_INSTALL_DEV``
# build arg.
# ------------------------------------
ARG WSITRAIN_INSTALL_DEV=1
RUN if [ "${WSITRAIN_INSTALL_DEV}" = "1" ]; then \
        pip install --no-deps -e "/app/wsinsight-train[dev]"; \
    fi

# ------------------------------------
# Sanity check (build-time)
# ------------------------------------
RUN python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU?", torch.cuda.is_available())
import wsitrain
print("wsitrain: ok")
import kurtorank
print("kurtorank: ok")
import cellpose
print("cellpose:", cellpose.__version__)
PY

# ENTRYPOINT below is the `wsitrain` console script, so its absence would only
# surface as an exec failure at `docker run` time. Gate it at build time.
# `kurtorank --help` is the gate that catches a --no-deps install whose
# dependency tree was never populated; the bare `import kurtorank` above does not.
RUN command -v wsitrain && wsitrain --help > /dev/null && \
    command -v kurtorank && kurtorank --help > /dev/null && \
    python -c "import numpy; v=numpy.__version__; assert int(v.split('.')[0]) < 2, 'numpy '+v+' >= 2.0 breaks stardist'; print('numpy', v, 'OK')"

# ------------------------------------
# Non-root user
# ------------------------------------
# The container starts as root; the runtime entrypoint (docker-entrypoint.sh)
# remaps the pre-created ``user`` account to the owner of the mounted /workspace
# (or to $HOST_UID/$HOST_GID) and then drops privileges via setpriv. The uid/gid
# baked here is only a throwaway placeholder, immediately overwritten at run time.
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    bash -lc 'echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/user/.bashrc' && \
    bash -lc 'echo "conda activate wsitrain" >> /home/user/.bashrc' && \
    chown -R 1000:1000 /home/user

# Install the runtime uid/gid-remapping entrypoint.
RUN install -m 0755 /app/wsinsight-train/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# ------------------------------------
# Environment variables
# CELLVIT_ROOT: path to CellViT model weights (mount a volume at runtime)
# ------------------------------------
ENV CELLVIT_ROOT=/workspace/cellvit

WORKDIR /workspace
RUN chown -R 1000:1000 /workspace
# NOTE: no ``USER`` here on purpose — the container starts as root so the
# entrypoint can remap ``user`` to the mount owner, then drops privileges via
# setpriv. Passing ``docker run --user ...`` still works: the entrypoint detects
# a non-root start and execs the command unchanged.

SHELL ["/bin/bash", "-lc"]
# `wsitrain` stays baked into the ENTRYPOINT so the CLI surface is unchanged:
# `docker run <img> run --input ...` still means `wsitrain run --input ...`.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh", "wsitrain"]
CMD ["--help"]
