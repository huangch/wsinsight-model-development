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


def _slide_class_counts(label_dir: Path, slides: dict[str, list[str]]) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for g, tiles in slides.items():
        c: dict[int, int] = defaultdict(int)
        for t in tiles:
            for line in (label_dir / f"{t}.csv").read_text().splitlines():
                if line:
                    c[int(line.rsplit(",", 1)[1])] += 1
        counts[g] = dict(c)
    return counts


def _repair_class_coverage(val_g: set[str], pool: list[str],
                           counts: dict[str, dict[int, int]],
                           classes: set[int]) -> tuple[set[str], list[str]]:
    """Move slides across the split until every class is on both sides.

    Rare types cluster in a handful of slides, so an unrepaired slide-level
    holdout leaves classes with zero training or zero validation cells; those
    score 0 no matter how good the model is. Only classes carried by two or more
    slides are repairable here -- the caller tile-splits the rest.
    """
    notes: list[str] = []
    seen: set[tuple[frozenset, int]] = set()
    for _ in range(4 * len(pool) + 8):
        train_g = set(pool) - val_g
        missing_tr = {k for k in classes if not any(counts[g].get(k) for g in train_g)}
        missing_va = {k for k in classes if not any(counts[g].get(k) for g in val_g)}
        if not (missing_tr or missing_va):
            break
        k = sorted(missing_tr or missing_va)[0]
        to_train = bool(missing_tr)
        key = (frozenset(val_g), k)
        if key in seen:                    # already tried this state, would loop
            notes.append(f"class {k}: no slide assignment covers both sides")
            classes.discard(k)
            continue
        seen.add(key)
        src = val_g if to_train else train_g
        movable = [g for g in src if counts[g].get(k)]
        if not movable or len(src) <= 1:
            notes.append(f"class {k}: cannot be covered on both sides")
            classes.discard(k)
            continue
        # Give up the least of everything else.
        g = min(movable, key=lambda g: sum(counts[g].values()) - counts[g][k])
        val_g = (val_g - {g}) if to_train else (val_g | {g})
        notes.append(f"moved {g!r} to {'train' if to_train else 'val'} to cover class {k}")
    return val_g, notes


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
        counts = _slide_class_counts(label_dir, slides)
        carriers: dict[int, set[str]] = defaultdict(set)
        for g, c in counts.items():
            for k, n in c.items():
                if n:
                    carriers[k].add(g)
        # A class living on a single slide cannot be on both sides of a whole-slide
        # holdout, so that slide is tile-split instead of assigned wholesale.
        hybrid = {next(iter(v)) for v in carriers.values() if len(v) == 1}
        pool = [g for g in slide_names if g not in hybrid]
        for g in sorted(hybrid):
            sole = sorted(k for k, v in carriers.items() if v == {g})
            print(f"[split] tile-splitting {g!r}: sole carrier of class(es) {sole}")
        if len(pool) < 2:                  # nothing left to hold out whole
            pool = list(slide_names)
            hybrid = set()

        # Hold out slides tissue by tissue, and never from a tissue that has only
        # one slide: an unstratified draw turns those into leave-tissue-out cases
        # the head cannot possibly score on.
        by_tissue: dict[str, list[str]] = defaultdict(list)
        for g in pool:
            by_tissue[g.split("__", 1)[0]].append(g)
        val_g: set[str] = set()
        for _tissue, names in sorted(by_tissue.items()):
            if len(names) < 2:
                continue
            shuffled = list(names)
            rng.shuffle(shuffled)
            n_val = max(1, min(int(round(len(shuffled) * val_frac)), len(shuffled) - 1))
            val_g.update(shuffled[:n_val])
        if not val_g:                      # every tissue has a single slide
            shuffled = list(pool)
            rng.shuffle(shuffled)
            n_val = max(1, min(int(round(len(shuffled) * val_frac)), len(shuffled) - 1))
            val_g = set(shuffled[:n_val])
        val_g, notes = _repair_class_coverage(
            val_g, pool, counts, {k for k, v in carriers.items() if len(v) >= 2})
        for n in notes:
            print(f"[split] {n}")
        train_g = set(pool) - val_g

        train = [t for g in train_g for t in slides[g]]
        val = [t for g in val_g for t in slides[g]]
        for g in sorted(hybrid):
            group = list(slides[g])
            rng.shuffle(group)
            n_val = max(1, min(int(round(len(group) * val_frac)), len(group) - 1)) \
                if len(group) >= 2 else 0
            val.extend(group[:n_val])
            train.extend(group[n_val:])
        return SplitResult(
            train=sorted(train), val=sorted(val),
            mode="slide-level" + ("+hybrid" if hybrid else ""),
            n_slides=len(slide_names),
            train_slides=sorted(train_g | hybrid), val_slides=sorted(val_g | hybrid))

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
    for name, rows in (("train.csv", res.train), ("val.csv", res.val)):
        (out_dir / name).write_text("".join(f"{r}\n" for r in rows))
