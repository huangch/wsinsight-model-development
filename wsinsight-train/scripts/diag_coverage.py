"""Where do Xenium cells disappear? Split the loss into merge vs. mask-lookup,
and render a coverage map of kept (green) vs dropped (red) cells.

usage: diag_coverage.py <slide-substring> [...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

R = Path("/workspace/wsinsight/wsinsight-model-development")
O = R / "models"
DIAG = O / "diag"
DIAG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.bunwarp import map_cells  # noqa: E402
from wsitrain.dataset import discover_samples  # noqa: E402

TISSUE, TASK = "pantissue", "pantissue"
samples = discover_samples(R / "data/xenium", TISSUE)

for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    cells = pd.read_parquet(outs / "cells.parquet")[["cell_id", "x_centroid", "y_centroid"]]
    cl = pd.read_csv(outs / "analysis/clustering/gene_expression_graphclust/clusters.csv")
    cl = cl.rename(columns={"Barcode": "cell_id", "Cluster": "classification"})
    assign = pd.read_csv(outs / f"celltype_assignment_{TASK}_label.csv")
    m1 = cells.merge(cl, on="cell_id")
    m2 = m1.merge(assign, on="classification")

    mask = np.load(O / "masks" / TISSUE / f"{s.sample_id}.npy", mmap_mode="r")
    H, Wd = mask.shape
    xy = map_cells(m2[["x_centroid", "y_centroid"]].to_numpy(),
                   outs / "registration_params.json",
                   outs / "direct_transf.txt", "affine+bspline", target_wh=(Wd, H))
    xr = np.round(xy[:, 0]).astype(np.int64)
    yr = np.round(xy[:, 1]).astype(np.int64)
    inb = (xr >= 0) & (xr < Wd) & (yr >= 0) & (yr < H)
    xc = np.clip(xr, 0, Wd - 1)
    yc = np.clip(yr, 0, H - 1)

    # Chunked mask lookup: fancy-indexing a memmap at random rows is the slow part.
    order = np.argsort(yc)
    hit = np.zeros(len(xc), bool)
    for i in range(0, len(order), 200_000):
        idx = order[i:i + 200_000]
        y0, y1 = yc[idx].min(), yc[idx].max() + 1
        blk = np.asarray(mask[y0:y1])
        hit[idx] = blk[yc[idx] - y0, xc[idx]] > 0
    kept = hit & inb

    print(f"\n{s.sample_id}")
    print(f"  cells.parquet          {len(cells):9d}")
    print(f"  after join clusters.csv{len(m1):9d}  ({100*len(m1)/len(cells):5.1f}%)")
    print(f"  after join assignment  {len(m2):9d}  ({100*len(m2)/len(cells):5.1f}%)")
    print(f"  inside H&E bounds      {int(inb.sum()):9d}  ({100*inb.mean():5.1f}% of merged)")
    print(f"  landed on a nucleus    {int(kept.sum()):9d}  ({100*kept.mean():5.1f}% of merged, "
          f"{100*kept.sum()/len(cells):5.1f}% of all)")
    print(f"  of the in-bounds ones  {100*hit[inb].mean():5.1f}% hit a nucleus")

    # coverage map at ~2000 px wide
    S = max(Wd, H) / 2000.0
    ih, iw = int(H / S) + 1, int(Wd / S) + 1
    img = np.zeros((ih, iw, 3), np.uint8)
    for sel, col in ((~kept, (220, 40, 40)), (kept, (40, 200, 40))):
        px = np.clip((xc[sel] / S).astype(int), 0, iw - 1)
        py = np.clip((yc[sel] / S).astype(int), 0, ih - 1)
        img[py, px] = col
    name = re.sub(r"[^A-Za-z0-9]+", "_", s.sample_id)[:60]
    Image.fromarray(img).save(DIAG / f"coverage_{name}.png")
    print(f"  -> coverage_{name}.png   (green=kept, red=dropped)")
