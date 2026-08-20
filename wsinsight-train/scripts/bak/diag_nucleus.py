"""Cell centroid vs nucleus centroid as the point used for label transfer.

cells.parquet x_centroid/y_centroid is the *cell* centroid. Where cells are much
larger than their nucleus (hepatocytes, cardiomyocytes, adipose) that point sits
in cytoplasm, so an exact mask lookup fails for reasons that have nothing to do
with registration.

usage: diag_nucleus.py <slide-substring> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.bunwarp import map_cells  # noqa: E402
from wsitrain.dataset import discover_samples  # noqa: E402

samples = discover_samples(R / "data/xenium", "pantissue")


def rate(mask, um, outs, Wd, H):
    xy = map_cells(um, outs / "registration_params.json",
                   outs / "direct_transf.txt", "affine", target_wh=(Wd, H))
    x = np.round(xy[:, 0]).astype(np.int64)
    y = np.round(xy[:, 1]).astype(np.int64)
    ok = (x >= 0) & (x < Wd) & (y >= 0) & (y < H)
    hit = np.zeros(len(x), bool)
    hit[ok] = mask[y[ok], x[ok]] > 0
    return hit.mean()


for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    mask = np.load(R / "models/masks/pantissue" / f"{s.sample_id}.npy")
    H, Wd = mask.shape
    cov = float((mask > 0).mean())

    cells = pd.read_parquet(outs / "cells.parquet")
    nb = pd.read_parquet(outs / "nucleus_boundaries.parquet")
    nb.columns = [c.lower() for c in nb.columns]
    xc = [c for c in nb.columns if c.endswith("vertex_x")][0]
    yc = [c for c in nb.columns if c.endswith("vertex_y")][0]
    nuc = nb.groupby("cell_id")[[xc, yc]].mean()

    cid = cells["cell_id"].map(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
    nuc.index = pd.Index([v.decode() if isinstance(v, (bytes, bytearray)) else v
                          for v in nuc.index])
    j = pd.DataFrame({"cell_id": cid,
                      "cx": cells["x_centroid"].to_numpy(),
                      "cy": cells["y_centroid"].to_numpy()}).join(nuc, on="cell_id").dropna()

    d = np.hypot(j["cx"] - j[xc], j["cy"] - j[yc])
    r_cell = rate(mask, j[["cx", "cy"]].to_numpy(), outs, Wd, H)
    r_nuc = rate(mask, j[[xc, yc]].to_numpy(), outs, Wd, H)

    print(f"\n{s.sample_id}")
    print(f"  nucleus pixels = {cov*100:5.2f}% of slide (random-hit floor)")
    print(f"  cell vs nucleus centroid distance: p50={d.median():.2f} p90="
          f"{d.quantile(0.9):.2f} um")
    if "cell_area" in cells and "nucleus_area" in cells:
        ratio = (cells["nucleus_area"] / cells["cell_area"].replace(0, np.nan)).median()
        print(f"  median nucleus_area / cell_area = {ratio*100:.1f}%")
    print(f"  hit rate using CELL    centroid: {r_cell*100:5.1f}%")
    print(f"  hit rate using NUCLEUS centroid: {r_nuc*100:5.1f}%")
