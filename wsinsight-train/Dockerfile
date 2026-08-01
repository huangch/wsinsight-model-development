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

# ------------------------------------
# Install PyTorch + GPU support
# ------------------------------------
RUN pip install --retries 10 torch torchvision nvidia-ml-py

# ------------------------------------
# Install cellpose (cell segmentation backbone)
# ------------------------------------
RUN pip install --retries 10 cellpose

# ------------------------------------
# Install optional StarDist + TensorFlow (legacy stain normalization)
# Tolerate failure — wsitrain can run without it. stderr is deliberately NOT
# suppressed: if this is skipped the build log must show why.
# ------------------------------------
RUN pip install --retries 5 stardist tensorflow || \
    echo "WARNING: stardist/tensorflow optional install skipped; continuing"

# ------------------------------------
# Install kurtorank (unsupervised annotation library, not on PyPI)
# --no-deps: all of its deps are already installed above.
# ------------------------------------
RUN pip install --no-deps -e /app/kurtorank

# ------------------------------------
# Install wsitrain itself
# --no-deps: torch, cellpose, kurtorank already installed above.
# ------------------------------------
RUN pip install --no-deps -e /app/wsinsight-train

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
RUN command -v wsitrain && wsitrain --help > /dev/null

# ------------------------------------
# Non-root user
# ------------------------------------
RUN groupadd -g 1000 user && \
    useradd -m -u 1000 -g 1000 -s /bin/bash user && \
    bash -lc 'echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/user/.bashrc' && \
    bash -lc 'echo "conda activate wsitrain" >> /home/user/.bashrc'

# ------------------------------------
# Environment variables
# CELLVIT_ROOT: path to CellViT model weights (mount a volume at runtime)
# ------------------------------------
ENV CELLVIT_ROOT=/workspace/cellvit

WORKDIR /workspace
RUN chown -R 1000:1000 /workspace
USER user

SHELL ["/bin/bash", "-lc"]
ENTRYPOINT ["wsitrain"]
CMD ["--help"]
