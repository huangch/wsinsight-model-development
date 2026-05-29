"""Budget-preserving inverse-frequency class weights.

Replaces pipeline/compute_class_weights.py. The CLI wrapper there shells out
to itself with subprocess; here we expose a pure function so other modules
can call it directly without re-launching Python.

Formula:
    1.  raw_i = min(cap / pct_i, cap)              # inverse-frequency, capped
    2.  w_i   = raw_i * n_classes / sum(raw)       # rescale to a fixed budget

The rescale step keeps sum(w) == n_classes so the total loss-weighting
budget is invariant under any per-class tweak (boosting one class
proportionally lowers the others).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import NamedTuple

from . import paths


class WeightReport(NamedTuple):
    weights: list[float]                # length == n_classes
    pct: list[float]                    # per-class % of total detections
    n_total: int
    label_map: dict[int, str]
    capped_classes: list[int]           # classes where raw was clipped


def load_label_map(path: Path) -> dict[int, str]:
    """Parse a label_map.yaml of the form ``  0: lymphoid`` (one per line)."""
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        bare = line.split("#", 1)[0].strip()
        if not bare:
            continue
        key, _, val = bare.partition(":")
        try:
            out[int(key.strip())] = val.strip().strip("'\"")
        except ValueError:
            continue
    return out


def tally_labels(label_dir: Path) -> Counter[int]:
    """Count class_int occurrences across every <tile>.csv in label_dir.

    Each row is ``x,y,class_int`` (export_tiles.groovy emits no header).
    Uses ``rsplit(',', 1)`` so we tolerate any column count as long as the
    class int is in the last column.
    """
    counts: Counter[int] = Counter()
    for csv in label_dir.glob("*.csv"):
        with csv.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                _, _, tail = line.rpartition(",")
                if not tail:
                    continue
                try:
                    counts[int(tail)] += 1
                except ValueError:
                    continue
    return counts


def compute_weights(tissue: str, *, cap: float = 10.0) -> WeightReport:
    """Tally tissue labels and return budget-preserved weights."""
    label_map = load_label_map(paths.label_map_path(tissue))
    if not label_map:
        raise ValueError(f"label_map for '{tissue}' is empty")

    counts = tally_labels(paths.labels_dir(tissue))
    n_total = sum(counts.values())
    if n_total == 0:
        raise ValueError(f"no labels under {paths.labels_dir(tissue)}")

    n_classes = len(label_map)
    pct = [100.0 * counts.get(ci, 0) / n_total for ci in range(n_classes)]
    raw: list[float] = []
    capped: list[int] = []
    for ci, p in enumerate(pct):
        if p <= 0:
            raw.append(cap)
            capped.append(ci)
            continue
        inv = cap / p
        if inv >= cap:
            capped.append(ci)
        raw.append(min(inv, cap))

    raw_sum = sum(raw)
    scale = n_classes / raw_sum if raw_sum > 0 else 1.0
    weights = [round(w * scale, 3) for w in raw]
    return WeightReport(weights=weights, pct=pct, n_total=n_total,
                        label_map=label_map, capped_classes=capped)


def format_report_comments(rep: WeightReport, *, cap: float = 10.0) -> list[str]:
    """Return the comment lines we embed in the train YAML, one per class.

    Used by config.render_train_yaml to reproduce the same self-documenting
    block the old make_train_config.py emitted.
    """
    n_classes = len(rep.weights)
    width = max((len(n) for n in rep.label_map.values()), default=10)
    lines = [
        f"Inverse-frequency weights, rescaled so sum(weight) == {n_classes} "
        f"(mean weight 1.0); per-class cap before rescale = {cap}.",
    ]
    for ci in range(n_classes):
        name = rep.label_map.get(ci, "?")
        mark = "  (capped pre-rescale)" if ci in rep.capped_classes else ""
        lines.append(
            f"class {ci:>2}  {name:<{width}}  {rep.pct[ci]:6.2f}%  ->  "
            f"weight {rep.weights[ci]:5.3f}{mark}"
        )
    lines.append(f"(sum of weights = {sum(rep.weights):.3f}, target = {n_classes})")
    return lines
