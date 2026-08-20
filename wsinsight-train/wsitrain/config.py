"""Run configuration: shipped defaults + user overrides + CLI flags.

A run is fully described by a ``RunConfig``. The shipped ``defaults/run.yaml``
holds every tunable; users override via ``--config my.yaml`` and/or individual
CLI flags. This keeps the front-door command to two required values (input +
tissue) while remaining fully declarative and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults" / "run.yaml"


@dataclass
class RunConfig:
    # I/O
    input: Path
    tissue: str
    output: Path

    # segmentation
    segmenter: str = "stardist"         # cellpose | stardist
    cellpose_model: str = "cpsam"
    diameter: float | None = None
    cellpose_batch_size: int = 8         # tile batch; lower if GPU OOMs
    cellpose_flow_threshold: float = 0.0 # 0 = skip GPU flow QC (avoids WSI OOM)

    # registration transform
    # bUnwarpJ's elastic field is fitted on 4x-downsampled images with 8 intervals;
    # measured against Cellpose nuclei it costs 10-19 points of hit rate versus the
    # SIFT affine alone on every slide tested, so affine is the default.
    transform: str = "affine"            # affine | affine+bspline | none
    match_radius_px: int = 4             # search window for the nucleus lookup
    min_match_rate: float = 0.35         # drop slides whose registration is unusable
    # 'background' / 'filtered' are cells whose type is unknown, not junk: they turn
    # up at inference too, so keeping them gives the model somewhere to put them.
    drop_labels: tuple[str, ...] = ()

    # tiling (must match the historical export contract)
    tile_px: int = 1024
    mpp: float = 0.25
    min_cells: int = 5
    bg_thresh: float = 240.0
    overlap: float = 0.0

    # splits / weights
    val_frac: float = 0.20
    by_slide: bool = False               # tile-based (stratified per slide) split
    seed: int = 42
    weight_cap: float = 10.0

    # training
    backbone: str = "SAM-H-x40"
    fold: str = "fold_0"
    task: str = "sthelar_full"
    gpus: str = "auto"
    tune: int = 0                        # 0 = single run; N = auto-tune iterations

    # markers
    markers_csv: Path | None = None
    top_k_markers: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


def load_defaults() -> dict[str, Any]:
    return yaml.safe_load(DEFAULTS_PATH.read_text()) or {}


def build_config(input_dir: Path, tissue: str, output: Path | None,
                 user_config: Path | None = None,
                 overrides: dict[str, Any] | None = None) -> RunConfig:
    """Merge shipped defaults < user config file < explicit CLI overrides."""
    merged = load_defaults()
    if user_config is not None:
        merged.update(yaml.safe_load(Path(user_config).read_text()) or {})
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    merged.pop("input", None)
    merged.pop("tissue", None)
    merged.pop("output", None)
    out = output or (Path(input_dir) / "wsinsight_train_out")
    valid = RunConfig.__dataclass_fields__.keys()
    merged = {k: v for k, v in merged.items() if k in valid}
    return RunConfig(input=Path(input_dir), tissue=tissue, output=Path(out), **merged)
