# ====================================
# kurtorank — standalone GPU image
# CUDA 12.8 + cuDNN + Ubuntu 22.04
# ====================================
# Ships the `kurtorank` CLI (unsupervised ensemble subtype annotation for
# gene-limited spatial transcriptomics). Independent of the wsinsight-train
# image, which installs this package from source instead.
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
# Install Miniconda (Python 3.11 base) — gdal from conda-forge because
# spatialdata pulls geopandas, which will not resolve against numpy<2 from PyPI.
# ------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p "$CONDA_DIR" \
 && rm /tmp/mc.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

RUN conda --version && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main -y || true) && \
    (conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r -y || true)

RUN conda update -n base --yes --override-channels -c conda-forge conda && \
    conda create -y --override-channels -n kurtorank -c conda-forge \
        python=3.11 gdal=3.11.3 pip "setuptools<67" && \
    conda clean -afy

# ------------------------------------
# Global Conda initialization (Docker bash doesn't read /etc/profile.d/*)
# ------------------------------------
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/bash.bashrc && \
    echo 'conda activate kurtorank' >> /etc/bash.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /etc/skel/.bashrc && \
    echo 'conda activate kurtorank' >> /etc/skel/.bashrc

ENV CONDA_DEFAULT_ENV=kurtorank
ENV PATH="$CONDA_DIR/envs/kurtorank/bin:$PATH"

RUN python -m pip install --upgrade pip

ARG CONSTR=/app/kurtorank/constraints.txt

# ------------------------------------
# GPU stack. Copied ahead of the source tree so this layer stays constrained
# (unpinned, torch resolves to >=2.9 and pulls the CUDA 13 wheels over
# nvidia/cudnn/lib) and so the cache survives source edits.
# ------------------------------------
COPY constraints.txt /app/kurtorank/constraints.txt
RUN pip install --retries 10 -c "$CONSTR" "numpy<2" torch torchvision

# ------------------------------------
# AnnData stack. Installed explicitly because the package is installed
# --no-deps below, so nothing else would pull these in.
# ------------------------------------
RUN pip install --retries 10 -c "$CONSTR" \
        anndata scanpy squidpy spatialdata spatialdata-io \
        zarr xarray dask distributed \
        pandas scipy statsmodels matplotlib seaborn click tqdm

# Only the rank / marker-* commands need these, and every import site is lazy,
# so a failure here still leaves `kurtorank annotate` working.
RUN pip install --retries 5 -c "$CONSTR" cellxgene-census tiledbsoma || \
    echo "WARNING: cellxgene-census/tiledbsoma skipped; rank/marker-* unavailable"

# ------------------------------------
# Install kurtorank itself
# --no-deps so pip cannot relax the numpy<2 / zarr<3 generation pinned above.
# ------------------------------------
WORKDIR /app/kurtorank
COPY . .
RUN pip install --no-deps . && \
    pip install --retries 10 "numpy<2"

# ------------------------------------
# Sanity check (build-time)
# ------------------------------------
RUN python - <<'PY'
import torch
import kurtorank
print("kurtorank:", getattr(kurtorank, "__version__", "?"),
      "Torch:", torch.__version__, "CUDA:", torch.version.cuda,
      "GPU?", torch.cuda.is_available())
PY

# Fail the build if the console script was not installed. Without this the
# image can ship with an importable package but no `kurtorank` on PATH.
RUN command -v kurtorank && \
    kurtorank --help > /dev/null

# ------------------------------------
# Non-root user
# ------------------------------------
# The container starts as root; the runtime entrypoint (docker-entrypoint.sh)
# remaps the pre-created ``user`` account to the owner of the mounted /workspace
# (or to $HOST_UID/$HOST_GID) and then drops privileges via setpriv. The uid/gid
# baked here is only a throwaway placeholder, immediately overwritten at run
# time, so it is hard-coded (1000) rather than exposed as a build ARG — the
# shared entrypoint looks the account up by the name ``user``.
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    bash -lc 'echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/user/.bashrc' && \
    bash -lc 'echo "conda activate kurtorank" >> /home/user/.bashrc' && \
    chown -R 1000:1000 /home/user

# Install the runtime uid/gid-remapping entrypoint.
RUN install -m 0755 /app/kurtorank/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace
RUN chown -R 1000:1000 /workspace
# NOTE: no ``USER`` here on purpose — the container starts as root so the
# entrypoint can remap ``user`` to the mount owner, then drops privileges via
# setpriv. Passing ``docker run --user ...`` still works: the entrypoint detects
# a non-root start and execs the command unchanged.

SHELL ["/bin/bash","-lc"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
