"""Diagnostics for the pantissue run.

Writes to models/diag/:
  tiles_contact_sheet.png   20 random training tiles as saved on disk
  overlay_<slide>.png       H&E crop + segmentation outlines + transferred centroids
  (stdout)                  per-slide match rate, image axes/dtype, cluster-id sanity
"""
from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tifffile
import zarr
from PIL import Image

R = Path("/workspace/wsinsight/wsinsight-model-development")
O = R / "models"
DIAG = O / "diag"
DIAG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.dataset import discover_samples  # noqa: E402

TISSUE = "pantissue"
samples = discover_samples(R / "data/xenium", TISSUE)
mask_dir = O / "masks" / TISSUE
nuc_dir = O / "nuclei" / TISSUE

# ---------------------------------------------------------------- 1. tiles
img_dir = O / "trainingset" / TISSUE / "train" / "images"
pngs = sorted(img_dir.glob("*.png"))
random.Random(0).shuffle(pngs)
pick = pngs[:20]
print(f"=== 1. tile PNGs ({len(pngs)} total) ===")
print(f"{'tile':44s} {'shape':16s} {'dtype':7s} {'min':>5s} {'max':>5s} {'mean':>7s}")
thumbs = []
for p in pick:
    a = np.asarray(Image.open(p))
    print(f"{p.stem[:42]:44s} {str(a.shape):16s} {str(a.dtype):7s} "
          f"{a.min():5d} {a.max():5d} {a.mean():7.1f}")
    thumbs.append(np.asarray(Image.open(p).convert("RGB").resize((256, 256))))
if thumbs:
    grid = [np.hstack(thumbs[i:i + 5]) for i in range(0, len(thumbs) - len(thumbs) % 5, 5)]
    if grid:
        Image.fromarray(np.vstack(grid)).save(DIAG / "tiles_contact_sheet.png")
        print(f"-> {DIAG / 'tiles_contact_sheet.png'}")

# ------------------------------------------------- 2. per-slide match rates
print("\n=== 2. image axes / mask density / transfer match rate ===")
print(f"{'slide':34s} {'series0 shape':22s} {'axes':6s} {'dtype':8s} "
      f"{'nuclei':>9s} {'xenium':>9s} {'matched':>9s} {'%':>6s}")
rows = []
for s in samples:
    with tifffile.TiffFile(str(s.he)) as tf:
        ser = tf.series[0]
        shape, axes, dtype = ser.shape, ser.axes, ser.dtype
    mp = mask_dir / f"{s.sample_id}.npy"
    # Strided read: a full .max() on a whole-slide int32 mask is several GB per slide.
    if mp.exists():
        m = np.load(mp, mmap_mode="r")
        nuclei = int(np.asarray(m[::16, ::16]).max())
        mshape = m.shape
    else:
        nuclei, mshape = -1, None
    n_xen = pq.ParquetFile(Path(s.outs) / "cells.parquet").metadata.num_rows
    nz = nuc_dir / f"{s.sample_id}.csv"
    matched = (sum(1 for _ in nz.open()) - 1) if nz.exists() else 0
    pct = 100.0 * matched / max(n_xen, 1)
    rows.append((s, pct))
    print(f"{s.sample_id[:32]:34s} {str(shape):22s} {axes:6s} {str(dtype):8s} "
          f"{nuclei:9d} {n_xen:9d} {matched:9d} {pct:6.1f}", flush=True)
med = float(np.median([p for _, p in rows])) if rows else 0.0
print(f"\nmedian match rate: {med:.1f}%   (<20% => registration/matching is broken)")

# ------------------------------------------- 3. cluster-id consistency check
print("\n=== 3. assignment CSV vs current clusters.csv ===")
for s in samples[:12]:
    outs = Path(s.outs)
    a = outs / f"celltype_assignment_{TISSUE}_label.csv"
    c = outs / "analysis/clustering/gene_expression_graphclust/clusters.csv"
    if not (a.exists() and c.exists()):
        print(f"{s.sample_id[:40]:42s} MISSING")
        continue
    ar = list(csv.DictReader(a.open()))
    acol = "classification" if "classification" in ar[0] else list(ar[0])[0]
    aset = {r[acol] for r in ar}
    cset = {r["Cluster"] for r in csv.DictReader(c.open())}
    miss = cset - aset
    print(f"{s.sample_id[:40]:42s} assign={len(aset):3d} clusters={len(cset):3d} "
          f"unmapped={len(miss):3d}{'  <-- MISMATCH' if miss else ''}")

# ------------------------------------------------- 4. overlay for 2 slides
print("\n=== 4. overlays ===")
ranked = sorted(rows, key=lambda t: t[1])
for s, pct in [ranked[-1], ranked[0]]:
    mp = mask_dir / f"{s.sample_id}.npy"
    nz = nuc_dir / f"{s.sample_id}.csv"
    if not (mp.exists() and nz.exists()):
        continue
    mask = np.load(mp, mmap_mode="r")
    pts = np.loadtxt(nz, delimiter=",", skiprows=1, ndmin=2)
    if pts.size == 0:
        print(f"{s.sample_id[:40]}: no transferred cells")
        continue
    W = 1024
    cx, cy = int(np.median(pts[:, 0])), int(np.median(pts[:, 1]))
    x0 = max(0, min(cx - W // 2, mask.shape[1] - W))
    y0 = max(0, min(cy - W // 2, mask.shape[0] - W))
    with tifffile.TiffFile(str(s.he)) as tf:
        arr = zarr.open(tf.series[0].aszarr(), mode="r")
        crop = np.asarray(arr[y0:y0 + W, x0:x0 + W])
    if crop.ndim == 3 and crop.shape[0] in (3, 4) and crop.shape[-1] not in (3, 4):
        crop = np.moveaxis(crop, 0, -1)
    crop = np.atleast_3d(crop)[..., :3]
    if crop.shape[-1] == 1:
        crop = np.repeat(crop, 3, -1)
    if crop.dtype != np.uint8:
        crop = (crop.astype(np.float32) / max(float(crop.max()), 1) * 255).astype(np.uint8)
    sub = np.asarray(mask[y0:y0 + W, x0:x0 + W])
    edge = np.zeros(sub.shape, bool)
    edge[:-1, :] |= sub[:-1, :] != sub[1:, :]
    edge[:, :-1] |= sub[:, :-1] != sub[:, 1:]
    vis = crop.copy()
    vis[edge] = (0, 255, 0)
    sel = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x0 + W) &
              (pts[:, 1] >= y0) & (pts[:, 1] < y0 + W)]
    for px, py in sel[:, :2].astype(int):
        yy, xx = py - y0, px - x0
        vis[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3] = (255, 0, 0)
    name = re.sub(r"[^A-Za-z0-9]+", "_", s.sample_id)[:60]
    Image.fromarray(vis).save(DIAG / f"overlay_{name}.png")
    print(f"{s.sample_id[:40]:42s} match={pct:5.1f}% cells_in_crop={len(sel):5d} "
          f"-> overlay_{name}.png")

