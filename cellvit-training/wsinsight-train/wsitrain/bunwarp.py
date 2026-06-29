"""Apply ST2WSI (bUnwarpJ + SIFT) registration to Xenium cell coordinates.

ST2WSI writes per pair:
  registration_params.json  — affine matrix, DAPI pixel size, flip/rotate, scales
  direct_transf.txt         — bUnwarpJ elastic B-spline grid (source->target)

Pipeline mapping a Xenium centroid (microns) to H&E pixels:
  µm -> DAPI px (÷pxlSz, flip/rotate) -> affine -> [B-spline] -> H&E px (×scale)
Mode 'affine' stops before the spline; 'affine+bspline' applies both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_params(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    m = [float(v) for v in d["xnumAnnotImgRegParamSiftMatrix"]]  # a,c,e,b,d,f
    return {
        "affine": np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]]], float),
        "pxl": float(d["xnumAnnotImgRegParamDapiImgPxlSize"]),
        "flip_h": bool(d["xnumAnnotImgRegParamFlipHori"]),
        "rot": str(d["xnumAnnotImgRegParamRotation"]),
        "src_w": int(d["xnumAnnotImgRegParamSrcImgWidth"]),
        "src_h": int(d["xnumAnnotImgRegParamSrcImgHeight"]),
        "src_scale": int(d["xnumAnnotImgRegParamSourceScale"]),
        "tgt_scale": int(d["xnumAnnotImgRegParamTargetScale"]),
    }


def load_elastic(path: Path):
    """Read bUnwarpJ saveElasticTransformation: intervals + (I+3)x(I+3) cx,cy."""
    toks = Path(path).read_text().split()
    intervals = int(toks[toks.index("Intervals=") + 1]) if "Intervals=" in toks else int(
        next(t for t in toks if t.isdigit()))
    nums = [float(t) for t in toks if _is_float(t)]
    n = (intervals + 3) ** 2
    cx = np.array(nums[:n]).reshape(intervals + 3, intervals + 3)
    cy = np.array(nums[n:2 * n]).reshape(intervals + 3, intervals + 3)
    return intervals, cx, cy


def _is_float(t: str) -> bool:
    try:
        float(t); return True
    except ValueError:
        return False


def _pre_affine(xy: np.ndarray, p: dict) -> np.ndarray:
    """flip/rotate DAPI px coords, then SIFT affine -> target px."""
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    w, h = p["src_w"], p["src_h"]
    if p["flip_h"]:
        x = w - x
    r = p["rot"]
    if r == "90":
        x, y = h - y, x
    elif r == "180":
        x, y = w - x, h - y
    elif r == "270":
        x, y = y, w - x
    a = p["affine"]
    tx = a[0, 0] * x + a[0, 1] * y + a[0, 2]
    ty = a[1, 0] * x + a[1, 1] * y + a[1, 2]
    return np.stack([tx, ty], 1)


def _bspline(xy, intervals, cx, cy, w, h):
    out = np.empty_like(xy)
    for i, (x, y) in enumerate(xy):
        u = (x / w) * intervals + 1.0
        v = (y / h) * intervals + 1.0
        iu, iv = int(np.floor(u)), int(np.floor(v))
        sx = sy = 0.0
        for l in range(iv - 1, iv + 3):
            for k in range(iu - 1, iu + 3):
                if 0 <= k < intervals + 3 and 0 <= l < intervals + 3:
                    b = _b3(u - k) * _b3(v - l)
                    sx += cx[l, k] * b; sy += cy[l, k] * b
        out[i] = (sx, sy)
    return out


def _b3(t):
    t = abs(t)
    if t < 1: return 2/3 - t*t + 0.5*t**3
    if t < 2: return ((2 - t) ** 3) / 6
    return 0.0


def map_cells(xy_um, params, elastic=None, mode="affine+bspline"):
    p = load_params(params)
    xy = np.asarray(xy_um, float) / p["pxl"]
    xy = _pre_affine(xy, p)
    if mode == "affine+bspline" and elastic is not None:
        intervals, cx, cy = load_elastic(elastic)
        xy = _bspline(xy, intervals, cx, cy, p["src_w"], p["src_h"])
    return xy * p["tgt_scale"]
