"""Regressions for defects found in review: flag semantics, staleness, degenerate splits."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from wsitrain import paths, segment as segment_mod, stages
from wsitrain.configrender import _gpu_id
from wsitrain.config import build_config
from wsitrain.dataset import Sample
from wsitrain.manifest import Manifest


@pytest.fixture
def gpu_probe(tmp_path, monkeypatch):
    """Run the segment stage and report the gpu flag it requested."""
    seen = {}

    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            return np.ones(he_rgb.shape[:2], np.int32)

    def spy(name, **kw):
        seen.update(kw)
        return Fake()

    monkeypatch.setattr(segment_mod, "get_segmenter", spy)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((8, 8, 3), 10, np.uint8))
    sample = Sample("breast__s1", "breast", tmp_path, he, True)

    def _run(cfg):
        stages.segment(cfg, [sample], cfg.output)
        return seen["gpu"]
    return _run


# --------------------------------------------------------------------------
# --gpus means the same thing in every stage
# --------------------------------------------------------------------------

def test_gpu_zero_selects_device_not_cpu(cfg_factory, gpu_probe):
    """'0' is device 0 for CellViT, so it must not disable the segmenter's GPU."""
    assert gpu_probe(cfg_factory(gpus="0")) is True


def test_gpu_index_keeps_gpu_enabled(cfg_factory, gpu_probe):
    assert gpu_probe(cfg_factory(gpus="2")) is True


def test_auto_keeps_gpu_enabled(cfg_factory, gpu_probe):
    assert gpu_probe(cfg_factory(gpus="auto")) is True


@pytest.mark.parametrize("raw", ["cpu", "none", "no", "false", ""])
def test_cpu_aliases_disable_gpu(cfg_factory, gpu_probe, raw):
    assert gpu_probe(cfg_factory(gpus=raw)) is False


def test_gpu_zero_agrees_across_stages(tmp_path, cfg_factory, gpu_probe):
    cfg = cfg_factory(gpus="0")
    assert _gpu_id(cfg) == "0"
    assert gpu_probe(cfg) is True


# --------------------------------------------------------------------------
# manifest staleness
# --------------------------------------------------------------------------

def test_mpp_change_reruns_segmentation(tmp_path):
    """mpp drives the StarDist rescale, so masks are stale when it changes."""
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"mpp": 0.25})
    for stage in ("annotate", "segment", "tile"):
        mf.mark(stage, "done")

    fresh = Manifest.load_or_new(p, {"mpp": 0.5})

    assert fresh.is_done("annotate")
    assert not fresh.is_done("segment")


def test_marker_panel_change_reruns_annotate(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"top_k_markers": 25})
    mf.mark("annotate", "done")

    assert not Manifest.load_or_new(p, {"top_k_markers": 50}).is_done("annotate")


def test_markers_csv_change_reruns_annotate(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"markers_csv": "a.csv"})
    mf.mark("annotate", "done")

    assert not Manifest.load_or_new(p, {"markers_csv": "b.csv"}).is_done("annotate")


# --------------------------------------------------------------------------
# degenerate splits
# --------------------------------------------------------------------------

def _one_tile_tree(cfg, n_tiles):
    labels = paths.labels_dir(cfg.output, cfg.tissue)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n_tiles):
        (labels / f"breast__s1_tile_{i:05d}.csv").write_text("0,0,0\n1,1,1\n")
    paths.tissue_root(cfg.output, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "a"\n1: "b"\n')


def test_split_rejects_empty_validation_set(cfg_factory):
    cfg = cfg_factory(by_slide=False)
    _one_tile_tree(cfg, 1)
    with pytest.raises(RuntimeError, match="both sides must be non-empty"):
        stages.split(cfg, [], cfg.output)


def test_split_accepts_a_usable_set(cfg_factory, monkeypatch):
    cfg = cfg_factory(by_slide=False, val_frac=0.5)
    _one_tile_tree(cfg, 4)
    monkeypatch.setenv("CELLVIT_ROOT", str(cfg.output / "cv"))

    info = stages.split(cfg, [], cfg.output)

    assert info["n_train"] >= 1 and info["n_val"] >= 1
