"""Run configuration: shipped defaults + CLI flags.

A run is fully described by a ``RunConfig``. The shipped ``defaults/run.yaml``
holds every tunable; every one of them is also reachable as a CLI flag, so a
run is reproducible from its command line alone.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
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
    stardist_model: str = "2D_versatile_he"
    stardist_model_dir: Path | None = None   # parent of the csbdeep model folder
    stardist_cpu: bool = False               # TF-only; needed without a CUDA toolkit
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

    def __post_init__(self) -> None:
        # YAML gives a list, the CLI gives a tuple; pin one so the manifest
        # comparison and the declared type agree whatever the source.
        if not isinstance(self.drop_labels, tuple):
            self.drop_labels = tuple(self.drop_labels or ())
        # to_dict() stringifies Paths, so a config reloaded from disk would
        # otherwise hand stages a str where they expect a Path.
        for name in ("stardist_model_dir", "markers_csv"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                # The manifest round-trips through JSON, which turns tuples into
                # lists; without this every reload looks like a config change.
                d[k] = list(v)
        return d


def load_defaults() -> dict[str, Any]:
    return yaml.safe_load(DEFAULTS_PATH.read_text()) or {}


def default_output(input_dir: Path, output: Path | None) -> Path:
    # Resolved, because `input` is stringified into the manifest and compared
    # verbatim: ./data and /abs/data would otherwise invalidate every stage.
    base = Path(output) if output else Path(input_dir) / "wsinsight_train_out"
    return base.expanduser().resolve()


def load_resolved(input_dir: Path, tissue: str, output: Path | None) -> dict[str, Any]:
    """Config left behind by an earlier command in the same --output, if any."""
    from .paths import resolved_config_path

    path = resolved_config_path(default_output(input_dir, output), tissue)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def build_config(input_dir: Path, tissue: str, output: Path | None,
                 overrides: dict[str, Any] | None = None,
                 base: dict[str, Any] | None = None) -> RunConfig:
    """Merge shipped defaults < earlier resolved config < explicit CLI overrides.

    ``base`` carries a previous command's choices forward so that a per-stage
    command need not repeat every shared flag; without it an omitted flag would
    silently fall back to the shipped default and invalidate earlier stages.
    """
    merged = load_defaults()
    if base:
        merged.update({k: v for k, v in base.items() if v is not None})
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})

    for key in ("input", "tissue", "output"):
        merged.pop(key, None)
    out = default_output(input_dir, output)
    valid = RunConfig.__dataclass_fields__.keys()
    merged = {k: v for k, v in merged.items() if k in valid}
    return RunConfig(input=Path(input_dir).expanduser().resolve(), tissue=tissue,
                     output=Path(out), **merged)
