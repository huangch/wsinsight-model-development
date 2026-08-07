"""Is the transfer loss a systematic sub-cellular offset?

Grid-searches a global (dx, dy) shift that maximises the nucleus hit rate, and
compares affine-only against affine+bspline. A large jump at a non-zero shift
means a coordinate-convention bug, not a registration failure.

usage: diag_shift.py <slide-substring> [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("/workspace/wsinsight/wsinsight-model-development")
O = R / "models"
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.bunwarp import map_cells  # noqa: E402
from wsitrain.dataset import discover_samples  # noqa: E402

TISSUE, TASK = "pantissue", "pantissue"
N = 20000
RADII = (0, 2, 4, 8, 16)
samples = discover_samples(R / "data/xenium", TISSUE)


def hit_rate(mask, x, y, dx=0, dy=0, r=0):
    H, Wd = mask.shape
    xs, ys = x + dx, y + dy
    ok = (xs >= r) & (xs < Wd - r) & (ys >= r) & (ys < Wd * 0 + H - r)
    xs, ys = xs[ok], ys[ok]
    if r == 0:
        return float((mask[ys, xs] > 0).mean()), int(ok.sum())
    got = np.zeros(len(xs), bool)
    for oy in range(-r, r + 1, max(1, r // 2)):
        for ox in range(-r, r + 1, max(1, r // 2)):
            got |= mask[ys + oy, xs + ox] > 0
    return float(got.mean()), int(ok.sum())


for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    cells = pd.read_parquet(outs / "cells.parquet")[["cell_id", "x_centroid", "y_centroid"]]
    rng = np.random.default_rng(0)
    sub = cells.iloc[rng.choice(len(cells), min(N, len(cells)), replace=False)]
    um = sub[["x_centroid", "y_centroid"]].to_numpy()

    mask = np.load(O / "masks" / TISSUE / f"{s.sample_id}.npy")  # full read; needed for random access
    H, Wd = mask.shape
    print(f"\n{s.sample_id}\n  mask {mask.shape}  nuclei={mask.max()}  sampled={len(um)}")

    for mode in ("affine", "affine+bspline"):
        xy = map_cells(um, outs / "registration_params.json",
                       outs / "direct_transf.txt", mode, target_wh=(Wd, H))
        x = np.round(xy[:, 0]).astype(np.int64)
        y = np.round(xy[:, 1]).astype(np.int64)
        base, n = hit_rate(mask, x, y)
        rr = [f"r={r}:{hit_rate(mask, x, y, r=r)[0]*100:5.1f}%" for r in RADII]
        print(f"  {mode:14s} in-bounds={n:6d}  " + "  ".join(rr))

        best = (base, 0, 0)
        for dy in range(-64, 65, 8):
            for dx in range(-64, 65, 8):
                h, _ = hit_rate(mask, x, y, dx, dy)
                if h > best[0]:
                    best = (h, dx, dy)
        print(f"  {'':14s} best shift dx={best[1]:+4d} dy={best[2]:+4d} -> "
              f"{best[0]*100:5.1f}%  (no shift {base*100:5.1f}%)")
