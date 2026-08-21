"""Run configuration: shipped defaults + CLI flags.

A run is fully described by a ``RunConfig``. The shipped ``defaults/run.yaml``
holds every tunable; every one of them is also reachable as a CLI flag, so a
run is reproducible from its command line alone.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults" / "run.yaml"

# Always supplied on the command line, never taken from a file.
_IO_FIELDS = ("input", "tissue", "output")

# Shared with argparse so the flag path and the YAML path agree on what is legal.
CHOICES: dict[str, tuple[str, ...]] = {
    "segmenter": ("cellpose", "stardist"),
    "transform": ("affine", "affine+bspline", "none"),
}


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


def load_config_file(path: Path) -> dict[str, Any]:
    """Read a --config file. Unlike the saved record, this one is hand-written,
    so a typo is an error rather than something to tolerate."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f"--config file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise SystemExit(f"--config {path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"--config {path} must be a mapping of setting: value")
    _reject_unknown_keys(loaded, path)
    _reject_bad_values(loaded, path)
    return loaded


def _reject_unknown_keys(values: dict[str, Any], path: Path) -> None:
    known = set(RunConfig.__dataclass_fields__)
    unknown = [k for k in values if k not in known]
    if not unknown:
        return
    lines = [f"--config {path} has settings wsitrain does not recognise:"]
    for key in sorted(unknown):
        near = difflib.get_close_matches(key, sorted(known - set(_IO_FIELDS)), n=1)
        lines.append(f"  {key}" + (f"    did you mean {near[0]}?" if near else ""))
    lines.append("run `wsitrain run --help` for the full list of settings.")
    raise SystemExit("\n".join(lines))


def _reject_bad_values(values: dict[str, Any], path: Path) -> None:
    # The flag path gets this from argparse `choices=`; a YAML path would
    # otherwise reach the stage that consumes it and fail hours later.
    for key, allowed in CHOICES.items():
        if key in values and values[key] not in allowed:
            raise SystemExit(
                f"--config {path}: {key}={values[key]!r} is not one of "
                f"{', '.join(allowed)}")


def resolve_config(input_dir: Path, tissue: str, output: Path | None, *,
                   overrides: dict[str, Any] | None = None,
                   base: dict[str, Any] | None = None,
                   config: dict[str, Any] | None = None,
                   ) -> tuple[RunConfig, dict[str, str]]:
    """Merge the config layers, and report where each field's value came from.

    Lowest priority first: shipped defaults < saved run-<tissue>.yaml <
    --config file < CLI flags. ``config`` patches the saved layer rather than
    replacing it: the saved record is a full dump, so replacing it would revert
    every earlier non-default choice and invalidate the stages that produced
    them.
    """
    merged = load_defaults()
    source = dict.fromkeys(merged, "default")
    known = RunConfig.__dataclass_fields__.keys()

    for values, label in ((base, "saved"), (config, "config"), (overrides, "flag")):
        for key, value in (values or {}).items():
            # input/tissue/output always come from the command line, so a saved
            # record or a config file may carry them but never set them.
            if value is None or key in _IO_FIELDS:
                continue
            if key not in known:
                continue      # only --config is strict; see load_config_file
            merged[key] = value
            source[key] = label

    merged = {k: v for k, v in merged.items() if k in known}
    return (RunConfig(input=Path(input_dir).expanduser().resolve(), tissue=tissue,
                      output=default_output(input_dir, output), **merged),
            {k: source[k] for k in merged})


def build_config(input_dir: Path, tissue: str, output: Path | None,
                 overrides: dict[str, Any] | None = None,
                 base: dict[str, Any] | None = None,
                 config: dict[str, Any] | None = None) -> RunConfig:
    """``resolve_config`` without the provenance, for callers that don't need it."""
    return resolve_config(input_dir, tissue, output, overrides=overrides,
                          base=base, config=config)[0]
