import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "wsinsight-train")
from wsitrain.dataset import discover_samples

s = [x for x in discover_samples(Path("data/xenium"), "pantissue")
     if "Tissue sample 1 (IDC)" in x.sample_id][0]
o = Path(s.outs)
print(o)
a = pd.read_csv(o / "celltype_assignment_pantissue_label.csv")
print("assign", a.dtypes.to_dict())
print(a.head(3))
c = pd.read_csv(o / "analysis/clustering/gene_expression_graphclust/clusters.csv")
print("clusters", c.dtypes.to_dict())
print(c.head(3))
cells = pd.read_parquet(o / "cells.parquet")[["cell_id"]]
print("cells", cells.dtypes.to_dict())
print(cells.head(3))
print("merge cells<->clusters:", len(cells.merge(
    c.rename(columns={"Barcode": "cell_id"}), on="cell_id")))
