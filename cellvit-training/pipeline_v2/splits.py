"""Train / val tile splitting (slide-aware).

Replaces pipeline/make_splits.py. Two modes:

  * per-tile shuffle (default): flat random split across all tiles. Maximises
    training data but allows same-slide tiles on both sides of the split.
  * slide-level holdout (``by_slide=True``): groups tiles by SAMPLE_TAG so a
    whole slide goes to train OR val. Stricter eval, but with N slides the
    actual val fraction is at most (N-1)/N. Single-slide tissues silently
    fall back to per-tile shuffle.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from . import paths

# Matches "_tile_00042" optionally followed by an augmentation tag like
# "_hflip". Everything before is the SAMPLE_TAG.
_TILE_SUFFIX_RE = re.compile(r"_tile_\d+(?:_[a-z0-9]+)?$")


def sample_tag(stem: str) -> str:
    return _TILE_SUFFIX_RE.sub("", stem)


class SplitResult(NamedTuple):
    train: list[str]
    val: list[str]
    mode: str                       # "per-tile" or "slide-level"
    n_slides: int
    train_slides: list[str]         # populated only in slide-level mode
    val_slides: list[str]           # populated only in slide-level mode


def split_tiles(tissue: str, *, val_frac: float = 0.1, by_slide: bool = False,
                seed: int = 42) -> SplitResult:
    """Compute (but do not write) train/val split for one tissue."""
    label_dir = paths.labels_dir(tissue)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label dir not found: {label_dir}")

    tiles = sorted(p.stem for p in label_dir.glob("*.csv"))
    if not tiles:
        raise ValueError(f"no .csv files under {label_dir}")

    slides: dict[str, list[str]] = defaultdict(list)
    for s in tiles:
        slides[sample_tag(s)].append(s)
    slide_names = sorted(slides)

    rng = random.Random(seed)
    use_slide_level = by_slide and len(slide_names) >= 2

    if use_slide_level:
        shuffled = list(slide_names)
        rng.shuffle(shuffled)
        n_val_g = max(1, min(int(round(len(shuffled) * val_frac)),
                             len(shuffled) - 1))
        val_g = set(shuffled[:n_val_g])
        train_g = set(shuffled[n_val_g:])
        val = sorted(t for g in val_g for t in slides[g])
        train = sorted(t for g in train_g for t in slides[g])
        return SplitResult(train=train, val=val, mode="slide-level",
                           n_slides=len(slide_names),
                           train_slides=sorted(train_g),
                           val_slides=sorted(val_g))

    # Per-tile shuffle (default, or single-slide fallback).
    shuffled = list(tiles)
    rng.shuffle(shuffled)
    n_val = max(1, min(int(round(len(shuffled) * val_frac)),
                       len(shuffled) - 1))
    return SplitResult(train=shuffled[n_val:], val=shuffled[:n_val],
                       mode="per-tile", n_slides=len(slide_names),
                       train_slides=[], val_slides=[])


def write_split(res: SplitResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.csv").write_text("\n".join(res.train) + "\n")
    (out_dir / "val.csv").write_text("\n".join(res.val) + "\n")
