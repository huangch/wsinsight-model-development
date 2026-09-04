"""Path layout for a single run, anchored under the user's --output dir.

Unlike the old in-repo pipeline (anchored at cellvit-training/), every path
here is relative to the run's output directory, so wsinsight-train is fully
portable and never writes into its own install tree. Workdir layout:

    <output>/
      run-<tissue>.yaml      resolved config
      manifest-<tissue>.json per-stage status + provenance
      masks/<tissue>/       per-slide instance masks (.npy)
      nuclei/<tissue>/      per-slide labelled nuclei (.csv)
      trainingset/<tissue>/ label_map.yaml, train/{images,labels}, splits/, train_configs/
      models/<tissue>/      promoted head + side-car
      report/<tissue>/      confusion + classification report
"""
from __future__ import annotations

from pathlib import Path


def _slug(tissue: str) -> str:
    """Filesystem-safe tissue selector; `--tissue` may be a list ('breast,lung')."""
    out = tissue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    # A tissue of '..' would otherwise walk out of --output.
    return out.strip(".") or "unnamed"


def tissue_root(out: Path, tissue: str) -> Path:
    return out / "trainingset" / _slug(tissue)


def masks_dir(out: Path, tissue: str) -> Path:
    return out / "masks" / _slug(tissue)


def nuclei_dir(out: Path, tissue: str) -> Path:
    return out / "nuclei" / _slug(tissue)


def cells_dir(out: Path, tissue: str) -> Path:
    """Per-slide HDF5 crops for the non-end-to-end classifier."""
    return out / "cells" / _slug(tissue)


def train_config_path(out: Path, tissue: str, backbone: str, fold: str) -> Path:
    return tissue_root(out, tissue) / "train_configs" / backbone / f"{fold}.yaml"


def images_dir(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "train" / "images"


def labels_dir(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "train" / "labels"


def label_map_path(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "label_map.yaml"


def splits_dir(out: Path, tissue: str, fold: str) -> Path:
    return tissue_root(out, tissue) / "splits" / fold


def models_dir(out: Path, tissue: str) -> Path:
    return out / "models" / _slug(tissue)


def report_dir(out: Path, tissue: str) -> Path:
    return out / "report" / _slug(tissue)


def logs_dir(out: Path, tissue: str) -> Path:
    return out / "logs" / _slug(tissue)


def manifest_path(out: Path, tissue: str) -> Path:
    # Every other artifact is per-tissue; a shared manifest makes a second
    # tissue in the same --output skip every stage as "already done".
    return out / f"manifest-{_slug(tissue)}.json"


def resolved_config_path(out: Path, tissue: str) -> Path:
    return out / f"run-{_slug(tissue)}.yaml"
