import json
from pathlib import Path

import numpy as np

from wsitrain import bunwarp, splits, weights, dataset


def test_split_per_tile(tmp_path):
    d = tmp_path / "labels"; d.mkdir()
    for i in range(10):
        (d / f"s1_tile_{i:05d}.csv").write_text("0,0,1\n")
    r = splits.split_tiles(d, val_frac=0.2, by_slide=False, seed=1)
    assert len(r.train) + len(r.val) == 10 and r.mode == "per-tile"


def test_split_by_slide(tmp_path):
    d = tmp_path / "labels"; d.mkdir()
    for s in ("a", "b", "c"):
        for i in range(5):
            (d / f"{s}_tile_{i:05d}.csv").write_text("0,0,1\n")
    r = splits.split_tiles(d, by_slide=True, seed=1)
    assert r.mode == "slide-level" and set(r.train_slides) & set(r.val_slides) == set()


def test_weights_budget(tmp_path):
    (tmp_path / "lm.yaml").write_text('0: "a"\n1: "b"\n')
    d = tmp_path / "labels"; d.mkdir()
    (d / "t_tile_0.csv").write_text("0,0,0\n" * 90 + "0,0,1\n" * 10)
    rep = weights.compute_weights(tmp_path / "lm.yaml", d, cap=10)
    assert abs(sum(rep.weights) - 2) < 0.01


def test_bunwarp_affine(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({
        "xnumAnnotImgRegParamSiftMatrix": ["1", "0", "0", "0", "1", "0"],
        "xnumAnnotImgRegParamDapiImgPxlSize": "1.0", "xnumAnnotImgRegParamFlipHori": False,
        "xnumAnnotImgRegParamRotation": "0", "xnumAnnotImgRegParamSrcImgWidth": "100",
        "xnumAnnotImgRegParamSrcImgHeight": "100", "xnumAnnotImgRegParamSourceScale": 1,
        "xnumAnnotImgRegParamTargetScale": 1}))
    xy = bunwarp.map_cells([[10, 20]], p, None, "affine")
    assert np.allclose(xy, [[10, 20]])


def test_discovery_pooling(tmp_path):
    s = tmp_path / "breast/d1/outs"; s.mkdir(parents=True)
    (s / "cells.parquet").write_text("x")
    (s.parent / "x_he_image.ome.tif").write_text("x")
    assert len(dataset.discover_samples(tmp_path, "pantissue")) == 1
    assert len(dataset.discover_samples(tmp_path, "lung")) == 0


def test_discovery_subset(tmp_path):
    for t in ("breast", "lung", "skin"):
        s = tmp_path / f"{t}/d/outs"; s.mkdir(parents=True)
        (s / "cells.parquet").write_text("x"); (s.parent / "x_he_image.ome.tif").write_text("x")
    assert len(dataset.discover_samples(tmp_path, "breast,lung")) == 2
    assert len(dataset.discover_samples(tmp_path, "pantissue")) == 3
