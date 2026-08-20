"""Per-class cell counts in the train vs val split, to see which classes the
model never saw during training."""
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

root = Path(sys.argv[1])                      # trainingset/<tissue>
labels = root / "train" / "labels"
lmap = yaml.safe_load((root / "label_map.yaml").read_text())
names = {int(k): v for k, v in lmap.items()}

counts = {}
for split in ("train", "val"):
    ids = [r[0] for r in csv.reader((root / "splits" / "fold_0" / f"{split}.csv").open())]
    c = Counter()
    for tid in ids:
        f = labels / f"{tid}.csv"
        if not f.exists():
            continue
        with f.open() as fh:
            rd = csv.DictReader(fh)
            col = "class_int" if "class_int" in (rd.fieldnames or []) else (rd.fieldnames or [None])[-1]
            for row in rd:
                c[int(row[col])] += 1
    counts[split] = c

print(f"{'class':32s} {'train':>9s} {'val':>9s}")
for i in sorted(names):
    tr, va = counts["train"].get(i, 0), counts["val"].get(i, 0)
    flag = "  <-- absent in train" if va and not tr else ""
    print(f"{names[i]:32s} {tr:9d} {va:9d}{flag}")
print(f"{'TOTAL':32s} {sum(counts['train'].values()):9d} {sum(counts['val'].values()):9d}")
