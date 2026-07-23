"""Unit tests for wsitrain.bunwarp coordinate transform (ST2WSI affine + B-spline).

These pin the scale handling that was previously wrong:
  * source coords are divided by SourceScale *before* the affine,
  * the result is multiplied by TargetScale to reach full-resolution target px,
  * the B-spline lattice is defined over the target image at TargetScale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsitrain.bunwarp import map_cells


def _write_params(tmp: Path, *, affine, pxl=1.0, flip_h=False, flip_v=False,
                  rot="0", src_w=100, src_h=100, src_scale=1, tgt_scale=1) -> Path:
    p = tmp / "registration_params.json"
    p.write_text(json.dumps({
        "xnumAnnotImgRegParamSiftMatrix": list(map(float, affine)),  # a,c,e,b,d,f
        "xnumAnnotImgRegParamDapiImgPxlSize": pxl,
        "xnumAnnotImgRegParamFlipHori": flip_h,
        "xnumAnnotImgRegParamFlipVert": flip_v,
        "xnumAnnotImgRegParamRotation": rot,
        "xnumAnnotImgRegParamSrcImgWidth": src_w,
        "xnumAnnotImgRegParamSrcImgHeight": src_h,
        "xnumAnnotImgRegParamSourceScale": src_scale,
        "xnumAnnotImgRegParamTargetScale": tgt_scale,
    }))
    return p


def _write_elastic(tmp: Path, intervals: int, cx_val: float, cy_val: float) -> Path:
    """direct_transf.txt with a constant coefficient grid (cx=cx_val, cy=cy_val)."""
    n = (intervals + 3) ** 2
    p = tmp / "direct_transf.txt"
    toks = [f"Intervals={intervals}"] + [f"{cx_val}"] * n + [f"{cy_val}"] * n
    p.write_text(" ".join(toks))
    return p


IDENTITY = [1, 0, 0, 0, 1, 0]  # a,c,e,b,d,f -> [[1,0,0],[0,1,0]]


def test_affine_applies_source_and_target_scale(tmp_path):
    # pxl=1 -> DAPI px == µm; no flip/rot; src_scale=2, tgt_scale=4; identity affine.
    # (10,20) -> /2 = (5,10) -> affine identity -> *4 = (20,40)
    params = _write_params(tmp_path, affine=IDENTITY, src_scale=2, tgt_scale=4)
    out = map_cells([[10.0, 20.0]], params, mode="affine")
    np.testing.assert_allclose(out, [[20.0, 40.0]])


def test_rotation_180(tmp_path):
    # rot 180 on full-res dims (100x100): (10,20)->(90,80); /2=(45,40); *4=(180,160)
    params = _write_params(tmp_path, affine=IDENTITY, rot="180",
                           src_w=100, src_h=100, src_scale=2, tgt_scale=4)
    out = map_cells([[10.0, 20.0]], params, mode="affine")
    np.testing.assert_allclose(out, [[180.0, 160.0]])


def test_flip_horizontal(tmp_path):
    # flip_h on width 100: x -> 100 - x; (10,20)->(90,20); src=tgt=1 identity
    params = _write_params(tmp_path, affine=IDENTITY, flip_h=True, src_w=100, src_h=100)
    out = map_cells([[10.0, 20.0]], params, mode="affine")
    np.testing.assert_allclose(out, [[90.0, 20.0]])


def test_bspline_partition_of_unity(tmp_path):
    # A constant coefficient grid makes the cubic B-spline output that constant for
    # any interior point (basis is a partition of unity). So affine+bspline maps an
    # interior point to (cx_val, cy_val) in target/TargetScale space, then *TargetScale.
    tgt = 4
    intervals = 8
    W = H = 9 * tgt                       # -> int(W/tgt+0.5)-1 = 8 = intervals
    params = _write_params(tmp_path, affine=IDENTITY, src_scale=1, tgt_scale=tgt,
                           src_w=1000, src_h=1000)
    elastic = _write_elastic(tmp_path, intervals, cx_val=100.0, cy_val=200.0)
    # input (4,4): pxl=1 -> (4,4); /1 -> (4,4); affine identity -> ax=ay=4 -> bu=bv=4
    # u = 4*8/8 + 1 = 5 (interior of the 11x11 lattice) -> spline == (100,200)
    out = map_cells([[4.0, 4.0]], params, elastic, mode="affine+bspline",
                    target_wh=(W, H))
    np.testing.assert_allclose(out, [[100.0 * tgt, 200.0 * tgt]], atol=1e-6)


def test_affine_bspline_requires_elastic_and_target(tmp_path):
    params = _write_params(tmp_path, affine=IDENTITY)
    elastic = _write_elastic(tmp_path, 8, 1.0, 1.0)
    with pytest.raises(ValueError):
        map_cells([[1.0, 1.0]], params, None, mode="affine+bspline", target_wh=(36, 36))
    with pytest.raises(ValueError):
        map_cells([[1.0, 1.0]], params, elastic, mode="affine+bspline", target_wh=None)


def test_unknown_mode_raises(tmp_path):
    params = _write_params(tmp_path, affine=IDENTITY)
    with pytest.raises(ValueError):
        map_cells([[1.0, 1.0]], params, mode="bogus")
