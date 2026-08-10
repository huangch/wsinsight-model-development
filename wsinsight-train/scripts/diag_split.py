import sys
from collections import defaultdict
from pathlib import Path

import yaml

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.splits import split_tiles  # noqa: E402

root = Path(sys.argv[1])
res = split_tiles(root / "train/labels", val_frac=0.2, by_slide=True, seed=42)
lm = yaml.safe_load((root / "label_map.yaml").read_text())

tr = defaultdict(int)
va = defaultdict(int)
for names, acc in ((res.train, tr), (res.val, va)):
    for t in names:
        for line in (root / "train/labels" / f"{t}.csv").read_text().splitlines():
            if line:
                acc[int(line.rsplit(",", 1)[1])] += 1

print(f"\nmode={res.mode} slides={res.n_slides} "
      f"val_slides={len(res.val_slides)} tiles train={len(res.train)} val={len(res.val)}")
print(f"{'class':30s} {'train':>10s} {'val':>10s}")
for k in sorted(set(tr) | set(va)):
    flag = "  <-- EMPTY" if not tr[k] or not va[k] else ""
    print(f"{str(lm.get(k, k)):30s} {tr[k]:10d} {va[k]:10d}{flag}")
