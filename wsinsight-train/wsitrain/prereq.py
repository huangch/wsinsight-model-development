"""Gate that stops a stage running before the stages it depends on."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import paths

# Direct predecessors only: Manifest._stale_stages always invalidates a
# contiguous suffix, so a done-mark on the direct predecessor implies the
# whole prefix before it is done too.
PREREQUISITES: dict[str, tuple[str, ...]] = {
    "transfer": ("annotate", "segment"),
    "tile": ("transfer",),
    "split": ("tile",),
    "train": ("split",),
    "validate": ("train",),
    "export": ("train",),
}

# Stage -> the artifact its successors read. A done-mark alone is not enough:
# the output dir may have been cleaned by hand since. `annotate` writes into the
# input samples rather than --output, so it is not checked here; it already
# fails loudly on its own.
_ARTIFACTS: dict[str, Callable[[Path, str], Path]] = {
    "segment": paths.masks_dir,
    "transfer": paths.nuclei_dir,
    "tile": paths.labels_dir,
    # configrender points CELLVIT_LOGS at this, so a trained run does land
    # under --output even though CellViT itself lives elsewhere.
    "train": paths.logs_dir,
}


def _is_empty(path: Path) -> bool:
    return not path.is_dir() or next(path.iterdir(), None) is None


def check(stage: str, mf, cfg) -> None:
    """Raise SystemExit unless every stage `stage` depends on has really run."""
    for prev in PREREQUISITES.get(stage, ()):
        if not mf.is_done(prev):
            raise SystemExit(
                f"[{stage}] needs the {prev} stage, which has not run in "
                f"{cfg.output}. Run `wsitrain {prev} --input {cfg.input} "
                f"--tissue {cfg.tissue}` first, or `wsitrain run` for the lot.")
        artifact = _ARTIFACTS.get(prev)
        if artifact is not None and _is_empty(artifact(cfg.output, cfg.tissue)):
            raise SystemExit(
                f"[{stage}] the manifest marks {prev} done but its output "
                f"{artifact(cfg.output, cfg.tissue)} is missing or empty. "
                f"Re-run `wsitrain {prev} --input {cfg.input} "
                f"--tissue {cfg.tissue} --force`.")
