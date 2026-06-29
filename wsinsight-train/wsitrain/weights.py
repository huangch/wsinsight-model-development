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
        bare = line.split("#", 1)[0].strip()
        if not bare:
            continue
        key, _, val = bare.partition(":")
        try:
            out[int(key.strip())] = val.strip().strip("'\"")
        except ValueError:
            continue
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
