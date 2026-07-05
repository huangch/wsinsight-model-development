"""Train / val tile splitting. Ported from pipeline.old/splits.py.

Two modes:
  * per-tile (default): hold out ``val_frac`` of tiles from EACH slide, so
    every slide/tissue appears in both train and val. Robust to an
    imbalanced slide mix (e.g. 9 breast + 1 lung) that would otherwise send a
    whole tissue into val under slide-level holdout.
  * slide-level (``by_slide=True``): hold whole slides (SAMPLE_TAG) out to one
    side. Use only when leakage between train/val tiles of the same slide must
    be avoided AND the slide/tissue mix is balanced.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

_TILE_SUFFIX_RE = re.compile(r"_tile_\d+(?:_[a-z0-9]+)?$")


def sample_tag(stem: str) -> str:
    return _TILE_SUFFIX_RE.sub("", stem)


class SplitResult(NamedTuple):
    train: list[str]
    val: list[str]
    mode: str
    n_slides: int
    train_slides: list[str]
    val_slides: list[str]


def split_tiles(label_dir: Path, *, val_frac: float = 0.1,
                by_slide: bool = True, seed: int = 42) -> SplitResult:
    label_dir = Path(label_dir)
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

    if by_slide and len(slide_names) >= 2:
        shuffled = list(slide_names)
        rng.shuffle(shuffled)
        n_val = max(1, min(int(round(len(shuffled) * val_frac)), len(shuffled) - 1))
        val_g, train_g = set(shuffled[:n_val]), set(shuffled[n_val:])
        return SplitResult(
            train=sorted(t for g in train_g for t in slides[g]),
            val=sorted(t for g in val_g for t in slides[g]),
            mode="slide-level", n_slides=len(slide_names),
            train_slides=sorted(train_g), val_slides=sorted(val_g))

    # Per-tile split, stratified by slide: hold out val_frac of tiles from
    # EACH slide. Every slide (and therefore every tissue) is represented in
    # both train and val, so an imbalanced slide mix (e.g. 9 breast + 1 lung)
    # can never push a whole tissue entirely into the validation set.
    train: list[str] = []
    val: list[str] = []
    for g in slide_names:
        group = list(slides[g])
        rng.shuffle(group)
        if len(group) >= 2:
            n_val = max(1, min(int(round(len(group) * val_frac)), len(group) - 1))
        else:
            n_val = 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return SplitResult(train=sorted(train), val=sorted(val),
                       mode="per-tile-stratified", n_slides=len(slide_names),
                       train_slides=sorted(slide_names),
                       val_slides=sorted(slide_names))


def write_split(res: SplitResult, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.csv").write_text("\n".join(res.train) + "\n")
    (out_dir / "val.csv").write_text("\n".join(res.val) + "\n")
