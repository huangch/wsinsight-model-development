"""Full-resolution overlay computed live from registration_params.json.

Unlike diag_overlay.py this does not need the transfer stage to have run, so it
works right after re-registering a slide.

usage: diag_recheck.py <slide-substring> [...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

R = Path("/workspace/wsinsight/wsinsight-model-development")
DIAG = R / "models/diag"
DIAG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.bunwarp import map_cells  # noqa: E402
from wsitrain.dataset import discover_samples  # noqa: E402

W = 1024
samples = discover_samples(R / "data/xenium", "pantissue")

for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    outs = Path(s.outs)
    mask = np.load(R / "models/masks/pantissue" / f"{s.sample_id}.npy")
    H, Wd = mask.shape
    nz = mask > 0
    cov = float(nz.mean())
    n_nuc = int(mask.max())
    print(f"\n{s.sample_id}")
    print(f"  mask {Wd}x{H}  nuclei={n_nuc}  nucleus pixels={cov*100:.2f}% of slide "
          f"(random-hit floor)  mean area={cov*Wd*H/max(n_nuc,1):.0f} px")

    cells = pd.read_parquet(outs / "cells.parquet")[["x_centroid", "y_centroid"]]
    xy = map_cells(cells.to_numpy(), outs / "registration_params.json",
                   outs / "direct_transf.txt", "affine", target_wh=(Wd, H))
    x = np.round(xy[:, 0]).astype(np.int64)
    y = np.round(xy[:, 1]).astype(np.int64)
    inb = (x >= 0) & (x < Wd) & (y >= 0) & (y < H)
    hit = np.zeros(len(x), bool)
    hit[inb] = mask[y[inb], x[inb]] > 0
    print(f"  cells={len(x)}  in-bounds={inb.mean()*100:.1f}%  "
          f"on-nucleus={hit.mean()*100:.1f}%  (floor {cov*100:.2f}%)")
    print(f"  mapped bbox x[{x.min()},{x.max()}] y[{y.min()},{y.max()}]  slide {Wd}x{H}")

    # Densest tissue region, so the crop is not empty background.
    S = 512
    hh = np.zeros((H // S + 1, Wd // S + 1), int)
    np.add.at(hh, (np.clip(y // S, 0, hh.shape[0] - 1),
                   np.clip(x // S, 0, hh.shape[1] - 1)), 1)
    cy, cx = np.unravel_index(hh.argmax(), hh.shape)
    x0 = max(0, min(cx * S - W // 2, Wd - W))
    y0 = max(0, min(cy * S - W // 2, H - W))

    with tifffile.TiffFile(str(s.he)) as tf:
        ser = tf.series[0]
        z = zarr.open(ser.aszarr(), mode="r")
        full = z if isinstance(z, zarr.Array) else z[min(z.array_keys(), key=int)]
        cf = ser.axes.startswith(("C", "S"))
        crop = np.asarray(full[:, y0:y0 + W, x0:x0 + W]) if cf \
            else np.asarray(full[y0:y0 + W, x0:x0 + W])
    if cf:
        crop = np.moveaxis(crop, 0, -1)
    crop = np.atleast_3d(crop)[..., :3]
    if crop.shape[-1] == 1:
        crop = np.repeat(crop, 3, -1)
    if crop.dtype != np.uint8:
        crop = (crop.astype(np.float32) / max(float(crop.max()), 1) * 255).astype(np.uint8)

    sub = mask[y0:y0 + W, x0:x0 + W]
    edge = np.zeros(sub.shape, bool)
    edge[:-1, :] |= sub[:-1, :] != sub[1:, :]
    edge[:, :-1] |= sub[:, :-1] != sub[:, 1:]
    vis = crop.copy()
    vis[edge] = (0, 255, 0)
    sel = (x >= x0) & (x < x0 + W) & (y >= y0) & (y < y0 + W)
    for px, py, ok in zip(x[sel], y[sel], hit[sel]):
        vis[max(0, py - y0 - 2):py - y0 + 3, max(0, px - x0 - 2):px - x0 + 3] = \
            (0, 0, 255) if ok else (255, 0, 0)
    name = re.sub(r"[^A-Za-z0-9]+", "_", s.sample_id)[:60]
    Image.fromarray(vis).save(DIAG / f"recheck_{name}.png")
    print(f"  crop@({x0},{y0}) cells_in_crop={int(sel.sum())} "
          f"-> recheck_{name}.png  (green=nucleus outline, blue=hit, red=miss)")
