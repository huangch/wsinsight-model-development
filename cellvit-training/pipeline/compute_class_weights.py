"""
compute_class_weights.py
------------------------
Tally per-tile cell label CSVs under
    trainingset/<tissue>/train/labels/*.csv
and emit inverse-frequency class weights, rescaled so the **sum equals the
number of classes** (i.e. the average weight is 1.0).

Rationale -- the loss weight is a *fixed attention budget*: boosting a rare
class must come at the cost of others, otherwise the gradient norm balloons
and the model is pulled away from a good initialization. Keeping sum(w) == N
lets us tune relative emphasis without changing total signal strength.

Formula:
    1.  raw = min(cap / class_percent, cap)             # inverse-frequency, capped
    2.  weight = raw * N_classes / sum(raw)             # rescale to budget

Prints a comment block (one line per class, matching the pantissue config
style) and a Python-list `weight_list` ready to paste into
    trainingset/<tissue>/train_configs/<backbone>/fold_*.yaml.

Usage:
    python compute_class_weights.py --tissue breast
    python compute_class_weights.py --tissue breast --json out/weights.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent


def _load_label_map(label_map_yaml: Path) -> dict[int, str]:
    """Parse the canonical int → label-name yaml (same parser as
    export_tiles.groovy)."""
    out: dict[int, str] = {}
    with label_map_yaml.open() as fp:
        for line in fp:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            k, _, v = line.partition(":")
            try:
                ci = int(k.strip())
            except ValueError:
                continue
            out[ci] = v.strip().strip('"').strip("'")
    return out


def _tally_labels(label_dir: Path) -> Counter:
    """Count class_int occurrences across every <tile>.csv in label_dir.
    Each row is `x,y,class_int` (export_tiles.groovy emits no header)."""
    counts: Counter = Counter()
    for csv_file in label_dir.glob("*.csv"):
        with csv_file.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) != 2:
                    continue
                try:
                    counts[int(parts[1])] += 1
                except ValueError:
                    continue
    return counts


def _weight(pct: float) -> float:
    if pct <= 0:
        return 10.0
    return min(10.0 / pct, 10.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tissue", required=True)
    p.add_argument("--cap", type=float, default=10.0,
                   help="Upper bound on per-class weight (default 10.0).")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional path; write {label_map, counts, percents, "
                        "weight_list} as JSON.")
    args = p.parse_args()

    tissue_root = CELLVIT_TRAINING_ROOT / "trainingset" / args.tissue
    label_dir = tissue_root / "train" / "labels"
    label_map_yaml = tissue_root / "label_map.yaml"

    if not label_dir.is_dir():
        print(f"ERROR: {label_dir} does not exist.", file=sys.stderr)
        return 1
    if not label_map_yaml.exists():
        print(f"ERROR: {label_map_yaml} does not exist.", file=sys.stderr)
        return 1

    label_map = _load_label_map(label_map_yaml)
    counts = _tally_labels(label_dir)
    total = sum(counts.values())
    if total == 0:
        print(f"ERROR: no cell labels found under {label_dir}.", file=sys.stderr)
        return 1

    n_classes = max(label_map.keys()) + 1 if label_map else 0
    width = max((len(n) for n in label_map.values()), default=0)

    print(f"# Tissue: {args.tissue}")
    print(f"# Label dir: {label_dir}")
    print(f"# Total detections: {total:,}")
    print(f"# Inverse-frequency weights, rescaled so sum(weight) == {n_classes} "
          f"(mean weight 1.0); per-class cap before rescale = {args.cap}.")
    raw: list[float] = []
    pcts: list[float] = []
    for ci in range(n_classes):
        n = counts.get(ci, 0)
        pct = 100.0 * n / total if total else 0.0
        pcts.append(pct)
        w = _weight(pct) if args.cap == 10.0 else (
            min(args.cap / pct, args.cap) if pct > 0 else args.cap
        )
        raw.append(w)

    # ── Rescale to a fixed budget: sum(weights) == n_classes ──────────────
    raw_sum = sum(raw)
    scale = n_classes / raw_sum if raw_sum > 0 else 1.0
    weights: list[float] = [round(w * scale, 3) for w in raw]

    for ci in range(n_classes):
        name = label_map.get(ci, "?")
        cap_mark = ("  (capped pre-rescale)"
                    if pcts[ci] > 0 and (args.cap / pcts[ci]) > args.cap else "")
        print(f"# class {ci:>2}  {name:<{width}}  {pcts[ci]:6.2f}%  ->  "
              f"weight {weights[ci]:5.3f}{cap_mark}")
    print(f"# (sum of weights = {sum(weights):.3f}, target = {n_classes})")

    print()
    print(f"weight_list: {weights}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as fp:
            json.dump(
                {
                    "tissue": args.tissue,
                    "total": total,
                    "label_map": label_map,
                    "counts": {str(k): counts.get(k, 0) for k in range(n_classes)},
                    "percents": {str(k): 100.0 * counts.get(k, 0) / total for k in range(n_classes)},
                    "weight_list": weights,
                },
                fp,
                indent=2,
            )
        print(f"# JSON written: {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
