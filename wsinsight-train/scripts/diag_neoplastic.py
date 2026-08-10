"""Does the neoplastic call agree with what the slide actually is?

If 'neoplastic' were a morphology call it would track the sample: cancer slides
have it, non-diseased slides do not. Disagreement here means the label, not the
image, is the problem.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.dataset import discover_samples  # noqa: E402

NORMAL = re.compile(r"non-diseased|normal|healthy|hyperplasia|reactive", re.I)
CANCER = re.compile(r"cancer|carcinoma|adenocarcinoma|melanoma|leukemia|tumou?r|"
                    r"\bIDC\b|\bPRCC\b|\bPDAC\b", re.I)

rows = []
for s in discover_samples(R / "data/xenium", "pantissue"):
    outs = Path(s.outs)
    f = outs / "celltype_assignment_pannuke_label.csv"
    if not f.exists():
        continue
    assign = pd.read_csv(f)
    cl = pd.read_csv(outs / "analysis/clustering/gene_expression_graphclust/clusters.csv")
    cl = cl.rename(columns={"Cluster": "classification"})
    m = cl.merge(assign, on="classification")
    n = len(m)
    epi = int((m.cell_type == "epithelial").sum())
    neo = int((m.cell_type == "neoplastic").sum())
    lin = epi + neo
    is_cancer = bool(CANCER.search(s.sample_id))
    is_normal = bool(NORMAL.search(s.sample_id)) and not is_cancer
    rows.append((s.sample_id, is_cancer, is_normal, n, epi, neo,
                 (neo / lin) if lin else float("nan")))

print(f"{'':7s} {'epi cells':>10s} {'neo cells':>10s} {'neo%':>6s}  slide")
for sid, c, nm, n, epi, neo, frac in sorted(rows, key=lambda r: -(r[6] if r[6] == r[6] else -1)):
    tag = "CANCER " if c else ("NORMAL " if nm else "?      ")
    bad = ""
    if c and frac == frac and frac < 0.10:
        bad = "  <-- cancer slide, almost no neoplastic"
    if nm and frac == frac and frac > 0.50:
        bad = "  <-- normal slide, mostly neoplastic"
    fs = f"{frac*100:5.1f}" if frac == frac else "   NA"
    print(f"{tag} {epi:10d} {neo:10d} {fs}%  {sid[:66]}{bad}")

ok = [r for r in rows if r[6] == r[6] and (r[1] or r[2])]
canc = [r[6] for r in ok if r[1]]
norm = [r[6] for r in ok if r[2]]
print(f"\nneoplastic fraction of epithelial-lineage cells")
print(f"  cancer slides (n={len(canc)}): median {pd.Series(canc).median()*100:.1f}%  "
      f"range {min(canc)*100:.1f}-{max(canc)*100:.1f}%")
print(f"  normal slides (n={len(norm)}): median {pd.Series(norm).median()*100:.1f}%  "
      f"range {min(norm)*100:.1f}-{max(norm)*100:.1f}%")
