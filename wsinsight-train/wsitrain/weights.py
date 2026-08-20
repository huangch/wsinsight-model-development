"""Budget-preserving inverse-frequency class weights. Ported from
pipeline.old/weights.py (paths now passed in, not derived from repo tree).

    raw_i = min(cap / pct_i, cap);  w_i = raw_i * n_classes / sum(raw)
sum(weight) == n_classes, so boosting one class lowers the others.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import NamedTuple


class WeightReport(NamedTuple):
    weights: list[float]
    pct: list[float]
    n_total: int
    label_map: dict[int, str]
    capped_classes: list[int]


def load_label_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in Path(path).read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        key, _, val = raw.partition(":")
        try:
            idx = int(key.strip())
        except ValueError:
            continue
        val = val.strip()
        if val[:1] in ('"', "'"):
            # Quoted names may legitimately contain ':' or '#'.
            quote, end = val[0], val.find(val[0], 1)
            val = val[1:end] if end > 0 else val[1:]
        else:
            val = val.split("#", 1)[0].strip()
        out[idx] = val
    return out


def tally_labels(label_dir: Path) -> Counter:
    counts: Counter = Counter()
    for csv in Path(label_dir).glob("*.csv"):
        with csv.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                _, _, tail = line.rpartition(",")
                try:
                    counts[int(tail)] += 1
                except ValueError:
                    continue
    return counts


def compute_weights(label_map_path: Path, label_dir: Path, *,
                    cap: float = 10.0) -> WeightReport:
    label_map = load_label_map(label_map_path)
    if not label_map:
        raise ValueError(f"label_map is empty: {label_map_path}")
    ids = sorted(label_map)
    if ids != list(range(len(ids))):
        # pct/weights below are positional, so a gap silently drops a class's
        # cells and hands its budget to an id that does not exist.
        raise ValueError(
            f"label_map ids must run 0..{len(ids) - 1} without gaps; "
            f"got {ids} in {label_map_path}")
    counts = tally_labels(label_dir)
    n_total = sum(counts.values())
    if n_total == 0:
        raise ValueError(f"no labels under {label_dir}")
    n_classes = len(label_map)
    pct = [100.0 * counts.get(ci, 0) / n_total for ci in range(n_classes)]
    raw, capped = [], []
    for ci, p in enumerate(pct):
        if p <= 0:
            raw.append(cap); capped.append(ci); continue
        inv = cap / p
        if inv >= cap:
            capped.append(ci)
        raw.append(min(inv, cap))
    scale = n_classes / sum(raw) if sum(raw) > 0 else 1.0
    weights = [round(w * scale, 3) for w in raw]
    return WeightReport(weights, pct, n_total, label_map, capped)
