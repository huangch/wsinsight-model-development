"""Is the epithelial/neoplastic call consistent across slides?

Both are epithelial-lineage; whether a cluster is 'neoplastic' is a slide-level
judgement, so a whole-slide holdout can put the same morphology on opposite
sides of the label boundary.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.dataset import discover_samples  # noqa: E402

val = set()
vp = R / "models/pannuke/trainingset/pantissue/splits/fold_0/val.csv"
if vp.exists():
    import re
    rx = re.compile(r"_tile_\d+(?:_[a-z0-9]+)?$")
    val = {rx.sub("", l.strip()) for l in vp.read_text().splitlines() if l.strip()}

print(f"{'':4s} {'tissue':12s} {'bg':>5s} {'conn':>5s} {'epi':>5s} {'infl':>5s} {'neo':>5s}  slide")
for s in discover_samples(R / "data/xenium", "pantissue"):
    f = Path(s.outs) / "celltype_assignment_pannuke_label.csv"
    if not f.exists():
        continue
    c = Counter(pd.read_csv(f)["cell_type"])
    tis, _, name = s.sample_id.partition("__")
    tag = "VAL " if s.sample_id in val else ""
    print(f"{tag:4s} {tis:12s} {c['background']:5d} {c['connective']:5d} "
          f"{c['epithelial']:5d} {c['inflammatory']:5d} {c['neoplastic']:5d}  {name[:50]}")
