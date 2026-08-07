"""How far does the B-spline push a point, and in what pattern?

usage: diag_bspline.py <slide-substring> [...]
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

for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    mask = np.load(R / "models/masks/pantissue" / f"{s.sample_id}.npy", mmap_mode="r")
    H, Wd = mask.shape
    cells = pd.read_parquet(outs / "cells.parquet")[["x_centroid", "y_centroid"]]
    rng = np.random.default_rng(0)
    um = cells.iloc[rng.choice(len(cells), 20000, replace=False)].to_numpy()

    p = B.load_params(outs / "registration_params.json")
    axy = B._apply_source_transform(um / p["pxl"], p)
    intervals, cx, cy = B.load_elastic(outs / "direct_transf.txt")
    tgt = p["tgt_scale"]
    dw = int(Wd / tgt + 0.5) - 1
    dh = int(H / tgt + 0.5) - 1
    bxy = B._bspline(axy, intervals, cx, cy, dw, dh)
    d = bxy - axy

    print(f"\n{s.sample_id}")
    print(f"  target {Wd}x{H}  tgt_scale={tgt}  src_scale={p['src_scale']}  "
          f"intervals={intervals}  lattice={cx.shape}")
    print(f"  affine out  x[{axy[:,0].min():9.1f},{axy[:,0].max():9.1f}] "
          f"y[{axy[:,1].min():9.1f},{axy[:,1].max():9.1f}]  (lattice extent {dw} x {dh})")
    print(f"  bspline out x[{bxy[:,0].min():9.1f},{bxy[:,0].max():9.1f}] "
          f"y[{bxy[:,1].min():9.1f},{bxy[:,1].max():9.1f}]")
    mag = np.hypot(d[:, 0], d[:, 1])
    q = np.percentile(mag, [50, 90, 99])
    print(f"  |displacement| @TargetScale  p50={q[0]:.2f} p90={q[1]:.2f} p99={q[2]:.2f} px"
          f"   -> full-res p50={q[0]*tgt:.1f} p90={q[1]*tgt:.1f} px")
    print(f"  mean dx={d[:,0].mean():+.2f} dy={d[:,1].mean():+.2f} @TargetScale "
          f"({d[:,0].mean()*tgt:+.1f}, {d[:,1].mean()*tgt:+.1f} full-res)")
    print(f"  cx range [{cx.min():.1f},{cx.max():.1f}]  cy range [{cy.min():.1f},{cy.max():.1f}]")
