"""Path resolution — single source of truth for every directory layout fact.

Replaces the path-shape logic scattered across pipeline/_lib.sh and the
SCRIPT_DIR / CELLVIT_TRAINING_ROOT constants duplicated in every old script.
Every other module imports paths from here; no other module hard-codes
directory names or path templates.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# --- Anchors -----------------------------------------------------------------

CELLVIT_TRAINING_ROOT = Path(__file__).resolve().parent.parent
"""``.../wsinsight-model-development/cellvit-training`` (this package's parent)."""

PROJECT_ROOT = CELLVIT_TRAINING_ROOT.parent
"""``.../wsinsight-model-development`` (one level up from cellvit-training)."""

# --- Defaults ---------------------------------------------------------------

TEMPLATE_TISSUE = "pantissue"
"""Tissue whose train_configs/<backbone>/<fold>.yaml is the master template."""

DEFAULT_BACKBONE = "SAM-H-x40"
DEFAULT_FOLD = "fold_0"
DEFAULT_TASK = "hne"

# --- Per-tissue training-set paths ------------------------------------------

def tissue_root(tissue: str) -> Path:
    return CELLVIT_TRAINING_ROOT / "trainingset" / tissue


def labels_dir(tissue: str) -> Path:
    return tissue_root(tissue) / "train" / "labels"


def images_dir(tissue: str) -> Path:
    return tissue_root(tissue) / "train" / "images"


def label_map_path(tissue: str) -> Path:
    return tissue_root(tissue) / "label_map.yaml"


def splits_dir(tissue: str, fold: str = DEFAULT_FOLD) -> Path:
    return tissue_root(tissue) / "splits" / fold


def train_config_path(tissue: str, backbone: str = DEFAULT_BACKBONE,
                      fold: str = DEFAULT_FOLD) -> Path:
    return tissue_root(tissue) / "train_configs" / backbone / f"{fold}.yaml"


def resolved_config_path(tissue: str, backbone: str = DEFAULT_BACKBONE,
                         fold: str = DEFAULT_FOLD) -> Path:
    """envsubst output materialized by train_tissue.sh at run time."""
    return tissue_root(tissue) / "train_configs" / backbone / f".{fold}.resolved.yaml"


def template_config_path(backbone: str = DEFAULT_BACKBONE,
                         fold: str = DEFAULT_FOLD) -> Path:
    return train_config_path(TEMPLATE_TISSUE, backbone, fold)


def val_csv_path(tissue: str, fold: str = DEFAULT_FOLD) -> Path:
    return splits_dir(tissue, fold) / "val.csv"


def train_csv_path(tissue: str, fold: str = DEFAULT_FOLD) -> Path:
    return splits_dir(tissue, fold) / "train.csv"


# --- Upstream data paths (Xenium + QuPath project) --------------------------

DATA_ROOT = PROJECT_ROOT / "data"
XENIUM_ROOT = DATA_ROOT / "xenium"
QPROJ_PATH = DATA_ROOT / "qprj" / "project.qpproj"


def xenium_tissue_dir(tissue: str) -> Path:
    return XENIUM_ROOT / tissue


# --- CellViT runtime --------------------------------------------------------

CELLVIT_ROOT = CELLVIT_TRAINING_ROOT / "cellvit" / "CellViT-plus-plus"
LOGS_LOCAL = CELLVIT_ROOT / "logs_local"
"""Where the upstream trainer writes per-run output directories."""

CELLVIT_BACKBONE_WEIGHTS = (CELLVIT_TRAINING_ROOT / "cellvit" / "models"
                            / f"CellViT-{DEFAULT_BACKBONE}.pth")


# --- Pipeline scripts (v1 — delegated to by the wrappers in this package) ----

PIPELINE_V1_DIR = CELLVIT_TRAINING_ROOT / "pipeline.old"
GROOVY_DIR = PIPELINE_V1_DIR / "qupath"
LOGS_DIR = CELLVIT_TRAINING_ROOT / "_logs"


def groovy_path(name: str) -> Path:
    """Resolve a Groovy script under pipeline/qupath/."""
    p = GROOVY_DIR / name
    if not p.suffix:
        p = p.with_suffix(".groovy")
    return p


# --- External executables ---------------------------------------------------

def python_executable() -> str:
    """Resolve the Python interpreter used to invoke CellViT-plus-plus and
    other Python steps. Matches _lib::python in the v1 bash drivers.
    """
    env = os.environ.get("PYTHON")
    if env:
        return env
    candidate = "/opt/anaconda3/envs/wsinsight/bin/python3"
    if os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("python3")
    if not found:
        raise RuntimeError("Could not locate a python3 interpreter.")
    return found


def qupath_executable() -> str:
    """Resolve the QuPath CLI (`QuPath`) used for headless Groovy scripts.

    Honors $QUPATH if set; otherwise expects `QuPath` on PATH.
    """
    env = os.environ.get("QUPATH")
    if env:
        return env
    found = shutil.which("QuPath") or shutil.which("qupath")
    if not found:
        raise RuntimeError(
            "Could not locate the QuPath CLI. Set $QUPATH or put `QuPath` "
            "on PATH.")
    return found


def kurtorank_executable() -> str:
    """Resolve the `kurtorank` console script. Honors $KURTORANK."""
    env = os.environ.get("KURTORANK")
    if env:
        return env
    found = shutil.which("kurtorank")
    if not found:
        raise RuntimeError(
            "Could not locate the kurtorank console script. Install with "
            "`pip install -e .` from kurtorank/, or set $KURTORANK.")
    return found


# --- Run discovery (logs_local/<timestamp>_<log_comment>) -------------------

def log_comment(tissue: str, task: str = DEFAULT_TASK,
                backbone: str = DEFAULT_BACKBONE) -> str:
    """Reproduce _lib::log_comment from _lib.sh.

    Format: ``<tissue>-<task>-<backbone-lowercase>``. CellViT-plus-plus uses
    this as the suffix of its per-run directory under logs_local/.
    """
    return f"{tissue}-{task}-{backbone.lower()}"


def find_latest_run(comment: str) -> Path | None:
    """Return the newest logs_local/**/<ts>_<comment>/ directory whose
    checkpoints/model_best.pth exists, or None. Recursive — CellViT-plus-plus
    nests a fresh run inside an older same-comment dir when resuming."""
    if not LOGS_LOCAL.is_dir():
        return None
    suffix = f"_{comment}"
    cands = [p for p in LOGS_LOCAL.rglob(f"*{suffix}")
             if p.is_dir() and (p / "checkpoints" / "model_best.pth").exists()]
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime)
    return cands[-1]
