"""SlideReader windowed access and its full-read fallback."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import tifffile

from wsitrain.stages import SlideReader, read_he_rgb


@pytest.fixture
def no_zarr(monkeypatch):
    """Force the non-lazy path regardless of what is installed."""
    monkeypatch.setitem(sys.modules, "zarr", None)


def _gradient(h=12, w=16):
    arr = np.arange(h * w, dtype=np.uint16).reshape(h, w) % 251
    return np.repeat(arr[..., None], 3, axis=-1).astype(np.uint8)


def test_reports_slide_dimensions(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "a.tif", _gradient(12, 16))
    with SlideReader(tmp_path / "a.tif") as r:
        assert (r.height, r.width) == (12, 16)


def test_fallback_marks_itself_not_lazy(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "a.tif", _gradient())
    with SlideReader(tmp_path / "a.tif") as r:
        assert r.lazy is False


def test_window_matches_full_read(tmp_path, no_zarr):
    arr = _gradient(12, 16)
    tifffile.imwrite(tmp_path / "a.tif", arr)
    full = read_he_rgb(tmp_path / "a.tif")
    with SlideReader(tmp_path / "a.tif") as r:
        np.testing.assert_array_equal(r.window(4, 8, 4, 4), full[4:8, 8:12])


def test_window_shape_is_requested_size(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "a.tif", _gradient())
    with SlideReader(tmp_path / "a.tif") as r:
        assert r.window(0, 0, 5, 7).shape == (5, 7, 3)


def test_grayscale_window_expands_channels(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "g.tif", np.full((12, 16), 40, np.uint8))
    with SlideReader(tmp_path / "g.tif") as r:
        assert r.window(0, 0, 4, 4).shape == (4, 4, 3)


def test_channel_first_dimensions(tmp_path, no_zarr):
    arr = np.full((3, 12, 16), 40, np.uint8)
    tifffile.imwrite(tmp_path / "c.tif", arr, photometric="rgb", planarconfig="separate")
    with SlideReader(tmp_path / "c.tif") as r:
        assert (r.height, r.width) == (12, 16)
        assert r.window(0, 0, 4, 4).shape == (4, 4, 3)


def test_uint16_window_is_rescaled(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "u.tif", np.full((12, 16, 3), 65535, np.uint16))
    with SlideReader(tmp_path / "u.tif") as r:
        w = r.window(0, 0, 4, 4)
        assert w.dtype == np.uint8 and w.max() == 255


def test_close_is_idempotent(tmp_path, no_zarr):
    tifffile.imwrite(tmp_path / "a.tif", _gradient())
    r = SlideReader(tmp_path / "a.tif")
    r.close()
    r.close()


def test_lazy_path_agrees_with_fallback(tmp_path):
    """When zarr is present the windowed read must match the full read exactly."""
    pytest.importorskip("zarr")
    arr = _gradient(12, 16)
    tifffile.imwrite(tmp_path / "a.tif", arr)
    full = read_he_rgb(tmp_path / "a.tif")
    with SlideReader(tmp_path / "a.tif") as r:
        np.testing.assert_array_equal(r.window(4, 8, 4, 4), full[4:8, 8:12])
        assert (r.height, r.width) == (12, 16)


class _FakeArray:
    """Minimal stand-in for a zarr array: shape plus slicing."""

    def __init__(self, a):
        self._a = a

    @property
    def shape(self):
        return self._a.shape

    def __getitem__(self, key):
        return self._a[key]


def _force_lazy(monkeypatch, array):
    monkeypatch.setattr(SlideReader, "_open_lazy",
                        lambda self, series: _FakeArray(array))


def test_lazy_branch_windows_match_full_read(tmp_path, monkeypatch):
    arr = _gradient(12, 16)
    tifffile.imwrite(tmp_path / "a.tif", arr)
    _force_lazy(monkeypatch, arr)

    full = read_he_rgb(tmp_path / "a.tif")
    with SlideReader(tmp_path / "a.tif") as r:
        assert r.lazy is True
        np.testing.assert_array_equal(r.window(4, 8, 4, 4), full[4:8, 8:12])


def test_lazy_branch_handles_channel_first(tmp_path, monkeypatch):
    arr = np.arange(3 * 12 * 16, dtype=np.uint8).reshape(3, 12, 16)
    tifffile.imwrite(tmp_path / "c.tif", arr, photometric="rgb", planarconfig="separate")
    _force_lazy(monkeypatch, arr)

    with SlideReader(tmp_path / "c.tif") as r:
        assert r.lazy is True
        assert (r.height, r.width) == (12, 16)
        win = r.window(2, 3, 4, 5)
        assert win.shape == (4, 5, 3)
        np.testing.assert_array_equal(win[..., 0], arr[0, 2:6, 3:8])


def test_lazy_branch_picks_full_resolution_level(tmp_path):
    full_res = _gradient(12, 16)

    class _FakeGroup:
        def array_keys(self):
            return ["1", "0"]

        def __getitem__(self, key):
            return _FakeArray(full_res if key == "0" else _gradient(6, 8))

    assert SlideReader._full_res(_FakeGroup()).shape == full_res.shape


def test_full_res_passes_through_plain_array():
    arr = _FakeArray(_gradient(12, 16))
    assert SlideReader._full_res(arr) is arr


def test_missing_zarr_falls_back_cleanly(tmp_path, monkeypatch):
    arr = _gradient(12, 16)
    tifffile.imwrite(tmp_path / "a.tif", arr)

    def boom(self, series):
        raise ModuleNotFoundError("No module named 'zarr'")

    monkeypatch.setattr(SlideReader, "_open_lazy", boom)

    full = read_he_rgb(tmp_path / "a.tif")
    with SlideReader(tmp_path / "a.tif") as r:
        assert r.lazy is False
        np.testing.assert_array_equal(r.window(4, 8, 4, 4), full[4:8, 8:12])


def test_tile_uses_windows_from_lazy_reader(tmp_path, cfg_factory, monkeypatch):
    """The tile stage must produce identical output on the lazy path."""
    import pandas as pd

    from wsitrain import paths, stages
    from wsitrain.dataset import Sample

    arr = np.full((16, 16, 3), 10, np.uint8)
    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, arr)
    _force_lazy(monkeypatch, arr)

    s = Sample("breast__s1", "breast", tmp_path, he, True)
    nuc = cfg.output / "nuclei" / cfg.tissue
    nuc.mkdir(parents=True)
    pd.DataFrame([(1, 1, 0), (9, 9, 1)], columns=["x_px", "y_px", "class_int"]).to_csv(
        nuc / f"{s.sample_id}.csv", index=False)

    assert stages.tile(cfg, [s], cfg.output)["tiles"] == 2
    assert len(list(paths.images_dir(cfg.output, cfg.tissue).glob("*.png"))) == 2


def test_tile_skips_pixel_decode_for_sparse_tiles(tmp_path, cfg_factory, monkeypatch):
    """Tiles that cannot meet min_cells must never be decoded."""
    import pandas as pd

    from wsitrain import stages
    from wsitrain.dataset import Sample

    cfg = cfg_factory(tile_px=8, overlap=0.0, min_cells=1, bg_thresh=250.0)
    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((16, 16, 3), 10, np.uint8))
    s = Sample("breast__s1", "breast", tmp_path, he, True)

    nuc = cfg.output / "nuclei" / cfg.tissue
    nuc.mkdir(parents=True)
    # One cell, so only the top-left tile of the 2x2 grid qualifies.
    pd.DataFrame([(1, 1, 0)], columns=["x_px", "y_px", "class_int"]).to_csv(
        nuc / f"{s.sample_id}.csv", index=False)

    windows = []
    original = stages.SlideReader.window

    def counting_window(self, y0, x0, h, w):
        windows.append((y0, x0))
        return original(self, y0, x0, h, w)

    monkeypatch.setattr(stages.SlideReader, "window", counting_window)
    stages.tile(cfg, [s], cfg.output)

    assert windows == [(0, 0)]
