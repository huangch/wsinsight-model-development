"""Is the bUnwarpJ displacement applied in the right direction?

Compares nucleus hit rate for affine, affine+bspline, and affine with the
B-spline displacement negated. If the negated variant wins, direct_transf.txt
is target->source and we are applying it backwards.

usage: diag_direction.py <slide-substring> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain import bunwarp as B  # noqa: E402
from wsitrain.dataset import discover_samples  # noqa: E402

samples = discover_samples(R / "data/xenium", "pantissue")


def rate(mask, xy):
    H, Wd = mask.shape
    x = np.round(xy[:, 0]).astype(np.int64)
    y = np.round(xy[:, 1]).astype(np.int64)
    ok = (x >= 0) & (x < Wd) & (y >= 0) & (y < H)
    return float((mask[y[ok], x[ok]] > 0).mean())


for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    mask = np.load(R / "models/masks/pantissue" / f"{s.sample_id}.npy")
    H, Wd = mask.shape
    cells = pd.read_parquet(outs / "cells.parquet")[["x_centroid", "y_centroid"]]
    rng = np.random.default_rng(0)
    um = cells.iloc[rng.choice(len(cells), 20000, replace=False)].to_numpy()

    p = B.load_params(outs / "registration_params.json")
    axy = B._apply_source_transform(um / p["pxl"], p)
    intervals, cx, cy = B.load_elastic(outs / "direct_transf.txt")
    tgt = p["tgt_scale"]
    bxy = B._bspline(axy, intervals, cx, cy,
                     int(Wd / tgt + 0.5) - 1, int(H / tgt + 0.5) - 1)
    d = bxy - axy

    print(f"\n{s.sample_id}")
    for tag, v in (("affine        ", axy),
                   ("affine+bspline", bxy),
                   ("affine-bspline", axy - d),
                   ("bspline x0.5  ", axy + 0.5 * d)):
        print(f"  {tag}  {rate(mask, v * tgt) * 100:5.1f}%")
