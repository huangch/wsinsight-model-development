"""WSI reading and the tiling stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile

from wsitrain import paths
from wsitrain.dataset import Sample
from wsitrain.stages import read_he_rgb, tile


def _write_nuclei(out, tissue, sample_id, rows):
    d = out / "nuclei" / tissue
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["x_px", "y_px", "class_int"]).to_csv(
        d / f"{sample_id}.csv", index=False)


def _slide(tmp_path, arr, name="s1", tissue="breast", **kw):
    p = tmp_path / f"{name}_he_image.ome.tif"
    tifffile.imwrite(p, arr, **kw)
    return Sample(f"{tissue}__{name}", tissue, tmp_path / "outs", p, True)


# --------------------------------------------------------------------------
# read_he_rgb
# --------------------------------------------------------------------------

def test_reads_plain_rgb(tmp_path):
    arr = np.full((8, 12, 3), 30, np.uint8)
    tifffile.imwrite(tmp_path / "a.tif", arr)
    assert read_he_rgb(tmp_path / "a.tif").shape == (8, 12, 3)


def test_expands_grayscale_to_three_channels(tmp_path):
    tifffile.imwrite(tmp_path / "g.tif", np.full((8, 12), 30, np.uint8))
    out = read_he_rgb(tmp_path / "g.tif")
    assert out.shape == (8, 12, 3)
    assert (out[..., 0] == out[..., 2]).all()


def test_channel_first_is_transposed(tmp_path):
    arr = np.full((3, 8, 12), 30, np.uint8)
    tifffile.imwrite(tmp_path / "c.tif", arr, photometric="rgb", planarconfig="separate")
    assert read_he_rgb(tmp_path / "c.tif").shape == (8, 12, 3)


def test_uint16_is_rescaled_to_uint8(tmp_path):
    arr = np.full((8, 8, 3), 65535, np.uint16)
    tifffile.imwrite(tmp_path / "u.tif", arr)
    out = read_he_rgb(tmp_path / "u.tif")
    assert out.dtype == np.uint8 and out.max() == 255


def test_extra_channels_are_truncated(tmp_path):
    arr = np.full((8, 8, 4), 30, np.uint8)
    tifffile.imwrite(tmp_path / "rgba.tif", arr)
    assert read_he_rgb(tmp_path / "rgba.tif").shape[-1] == 3


# --------------------------------------------------------------------------
# tile
# --------------------------------------------------------------------------

def test_tiles_and_labels_are_written(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(1, 1, 0), (9, 9, 1)])

    info = tile(cfg, [s], cfg.output)

    assert info["tiles"] == 2
    assert len(list(paths.images_dir(cfg.output, cfg.tissue).glob("*.png"))) == 2


def test_label_coordinates_are_tile_local(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(9, 10, 3)])

    tile(cfg, [s], cfg.output)

    csv = next(paths.labels_dir(cfg.output, cfg.tissue).glob("*.csv"))
    assert csv.read_text() == "1,2,3\n"


def test_labels_are_headerless(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(1, 1, 0)])

    tile(cfg, [s], cfg.output)

    csv = next(paths.labels_dir(cfg.output, cfg.tissue).glob("*.csv"))
    assert "class_int" not in csv.read_text()


def test_background_tiles_are_skipped(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=240.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 255, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(1, 1, 0), (9, 9, 1)])

    assert tile(cfg, [s], cfg.output)["tiles"] == 0


def test_sparse_tiles_are_skipped(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=5, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(1, 1, 0), (2, 2, 0)])

    assert tile(cfg, [s], cfg.output)["tiles"] == 0


def test_slide_without_nuclei_csv_is_ignored(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))

    assert tile(cfg, [s], cfg.output)["tiles"] == 0


def test_overlap_increases_tile_count(tmp_path, cfg_factory):
    cells = [(x, y, 0) for x in range(1, 16, 2) for y in range(1, 16, 2)]

    plain = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s1 = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8), name="a")
    _write_nuclei(plain.output, plain.tissue, s1.sample_id, cells)
    n_plain = tile(plain, [s1], plain.output)["tiles"]

    lapped = cfg_factory(tile_px=8, overlap=0.5, min_cells=1, bg_thresh=250.0)
    s2 = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8), name="b")
    _write_nuclei(lapped.output, lapped.tissue, s2.sample_id, cells)
    n_lapped = tile(lapped, [s2], lapped.output)["tiles"]

    assert n_lapped > n_plain


def test_full_overlap_does_not_hang(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=1.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((16, 16, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id, [(1, 1, 0)])

    assert tile(cfg, [s], cfg.output)["tiles"] >= 1


def test_tile_stems_are_unique(tmp_path, cfg_factory):
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    s = _slide(tmp_path, np.full((32, 32, 3), 10, np.uint8))
    _write_nuclei(cfg.output, cfg.tissue, s.sample_id,
                  [(x, y, 0) for x in range(1, 32, 8) for y in range(1, 32, 8)])

    written = tile(cfg, [s], cfg.output)["tiles"]
    stems = {p.stem for p in paths.images_dir(cfg.output, cfg.tissue).glob("*.png")}
    assert len(stems) == written
