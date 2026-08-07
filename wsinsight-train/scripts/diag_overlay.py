"""Overlay H&E crop + segmentation outlines (green) + transferred centroids (red).

usage: diag_overlay.py <slide-substring> [<slide-substring> ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
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
W = 1024
samples = discover_samples(R / "data/xenium", TISSUE)

for pat in sys.argv[1:]:
    hits = [s for s in samples if pat.lower() in s.sample_id.lower()]
    if not hits:
        print(f"no slide matches {pat!r}")
        continue
    s = hits[0]
    mp = O / "masks" / TISSUE / f"{s.sample_id}.npy"
    nz = O / "nuclei" / TISSUE / f"{s.sample_id}.csv"
    mask = np.load(mp, mmap_mode="r")
    pts = np.loadtxt(nz, delimiter=",", skiprows=1, ndmin=2) if nz.exists() else np.empty((0, 3))

    with tifffile.TiffFile(str(s.he)) as tf:
        ser = tf.series[0]
        z = zarr.open(ser.aszarr(), mode="r")
        full = z if isinstance(z, zarr.Array) else z[min(z.array_keys(), key=int)]
        chan_first = ser.axes.startswith("C") or ser.axes.startswith("S")
        H, Wd = (full.shape[1], full.shape[2]) if chan_first else full.shape[:2]

        if len(pts):
            cx, cy = int(np.median(pts[:, 0])), int(np.median(pts[:, 1]))
        else:
            cx, cy = Wd // 2, H // 2
        x0 = max(0, min(cx - W // 2, Wd - W))
        y0 = max(0, min(cy - W // 2, H - W))
        crop = np.asarray(full[:, y0:y0 + W, x0:x0 + W]) if chan_first \
            else np.asarray(full[y0:y0 + W, x0:x0 + W])

    if chan_first:
        crop = np.moveaxis(crop, 0, -1)
    crop = np.atleast_3d(crop)[..., :3]
    if crop.shape[-1] == 1:
        crop = np.repeat(crop, 3, -1)
    if crop.dtype != np.uint8:
        crop = (crop.astype(np.float32) / max(float(crop.max()), 1) * 255).astype(np.uint8)

    sub = np.asarray(mask[y0:y0 + W, x0:x0 + W]) if mask.ndim == 2 else np.zeros((W, W), int)
    edge = np.zeros(sub.shape, bool)
    edge[:-1, :] |= sub[:-1, :] != sub[1:, :]
    edge[:, :-1] |= sub[:, :-1] != sub[:, 1:]
    vis = crop.copy()
    vis[edge] = (0, 255, 0)

    sel = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x0 + W) &
              (pts[:, 1] >= y0) & (pts[:, 1] < y0 + W)] if len(pts) else pts
    for px, py in sel[:, :2].astype(int):
        yy, xx = py - y0, px - x0
        vis[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3] = (255, 0, 0)

    name = re.sub(r"[^A-Za-z0-9]+", "_", s.sample_id)[:60]
    Image.fromarray(vis).save(DIAG / f"overlay_{name}.png")
    print(f"{s.sample_id[:50]:52s} axes={ser.axes} mask={mask.shape} "
          f"crop@({x0},{y0}) cells={len(sel)} -> overlay_{name}.png")
