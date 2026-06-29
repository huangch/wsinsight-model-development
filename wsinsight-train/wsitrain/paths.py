"""Path layout for a single run, anchored under the user's --output dir.

Unlike the old in-repo pipeline (anchored at cellvit-training/), every path
here is relative to the run's output directory, so wsinsight-train is fully
portable and never writes into its own install tree. Workdir layout:

    <output>/
      run.yaml              resolved config
      manifest.json         per-stage status + provenance
      trainingset/<tissue>/ label_map.yaml, train/{images,labels}, splits/, train_configs/
      models/<tissue>/      promoted head + side-car
      report/<tissue>/      confusion + classification report
"""
from __future__ import annotations

from pathlib import Path


def tissue_root(out: Path, tissue: str) -> Path:
    return out / "trainingset" / tissue


def images_dir(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "train" / "images"


def labels_dir(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "train" / "labels"


def label_map_path(out: Path, tissue: str) -> Path:
    return tissue_root(out, tissue) / "label_map.yaml"


def splits_dir(out: Path, tissue: str, fold: str) -> Path:
    return tissue_root(out, tissue) / "splits" / fold


def models_dir(out: Path, tissue: str) -> Path:
    return out / "models" / tissue


def report_dir(out: Path, tissue: str) -> Path:
    return out / "report" / tissue


def manifest_path(out: Path) -> Path:
    return out / "manifest.json"


def resolved_config_path(out: Path) -> Path:
    return out / "run.yaml"
