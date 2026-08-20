"""Segmentation backends: GPU selection, mpp handling, resampling."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from wsitrain import segment


class _DummyCellpose:
    def __init__(self, gpu, model_type):
        self.gpu = gpu
        self.model_type = model_type
        self.calls = []

    def eval(self, he_rgb, diameter=None, channels=None, batch_size=None,
             flow_threshold=None):
        self.calls.append({"diameter": diameter, "batch_size": batch_size,
                           "flow_threshold": flow_threshold,
                           "shape": he_rgb.shape})
        return np.zeros(he_rgb.shape[:2], dtype=np.uint16), None, None


@pytest.fixture
def fake_cellpose(monkeypatch):
    made = []

    def factory(gpu, model_type):
        m = _DummyCellpose(gpu, model_type)
        made.append(m)
        return m

    mod = types.ModuleType("cellpose")
    mod.models = types.SimpleNamespace(CellposeModel=factory)
    monkeypatch.setitem(sys.modules, "cellpose", mod)
    return made


def test_default_backend_is_stardist():
    assert segment.get_segmenter("stardist").name == "stardist"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown segmenter"):
        segment.get_segmenter("instanseg")


@pytest.mark.parametrize("gpu", [True, False])
def test_cellpose_honours_gpu_flag(fake_cellpose, gpu):
    seg = segment.get_segmenter("cellpose", gpu=gpu)
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert fake_cellpose[0].gpu is gpu


def test_cellpose_diameter_is_microns(fake_cellpose):
    seg = segment.get_segmenter("cellpose", diameter=10.0)
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.5)
    assert fake_cellpose[0].calls[0]["diameter"] == 20.0


def test_cellpose_auto_diameter_stays_none(fake_cellpose):
    seg = segment.get_segmenter("cellpose")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.5)
    assert fake_cellpose[0].calls[0]["diameter"] is None


def test_cellpose_model_instantiated_once(fake_cellpose):
    seg = segment.get_segmenter("cellpose")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert len(fake_cellpose) == 1


def test_cellpose_returns_int32(fake_cellpose):
    seg = segment.get_segmenter("cellpose")
    assert seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25).dtype == np.int32


def test_rgb_resample_is_noop_near_unit_scale():
    arr = np.zeros((10, 20, 3), np.uint8)
    assert segment._resample_rgb(arr, 1.0) is arr
    assert segment._resample_rgb(arr, 1.01) is arr


def test_rgb_resample_scales_both_axes():
    arr = np.zeros((10, 20, 3), np.uint8)
    assert segment._resample_rgb(arr, 2.0).shape == (20, 40, 3)
    assert segment._resample_rgb(arr, 0.5).shape == (5, 10, 3)


def test_rgb_resample_never_degenerates():
    assert segment._resample_rgb(np.zeros((4, 4, 3), np.uint8), 0.01).shape[:2] == (1, 1)


def test_label_resample_restores_original_shape():
    labels = np.zeros((4, 4), np.int32)
    labels[1:3, 1:3] = 7
    restored = segment._resample_labels(segment._resample_labels(labels, (8, 8)), (4, 4))
    assert restored.shape == (4, 4)


def test_label_resample_invents_no_ids():
    labels = np.array([[0, 1], [2, 0]], np.int32)
    up = segment._resample_labels(labels, (6, 6))
    assert set(np.unique(up)).issubset({0, 1, 2})


def test_label_resample_noop_when_shape_matches():
    labels = np.zeros((4, 4), np.int32)
    assert segment._resample_labels(labels, (4, 4)) is labels


def test_stardist_rescales_to_native_mpp(monkeypatch):
    seen = {}

    class FakeModel:
        @staticmethod
        def from_pretrained(name):
            return FakeModel()

        def predict_instances(self, img):
            seen["shape"] = img.shape
            return np.zeros(img.shape[:2], np.int32), None

    monkeypatch.setitem(sys.modules, "stardist", types.ModuleType("stardist"))
    sd_models = types.ModuleType("stardist.models")
    sd_models.StarDist2D = FakeModel
    monkeypatch.setitem(sys.modules, "stardist.models", sd_models)

    csb = types.ModuleType("csbdeep")
    csb_utils = types.ModuleType("csbdeep.utils")
    csb_utils.normalize = lambda x: x
    monkeypatch.setitem(sys.modules, "csbdeep", csb)
    monkeypatch.setitem(sys.modules, "csbdeep.utils", csb_utils)

    seg = segment.StarDistSegmenter()
    he = np.zeros((10, 10, 3), np.uint8)
    # 0.5 um/px is half the native resolution, so the image is upsampled 2x.
    mask = seg.segment(he, mpp=0.5)
    assert seen["shape"][:2] == (20, 20)
    assert mask.shape == (10, 10)


def test_segment_stage_runs_without_torch(tmp_path, cfg_factory, monkeypatch):
    """StarDist is the default backend, so torch must not be a hard import."""
    import tifffile

    from wsitrain.dataset import Sample
    from wsitrain.stages import segment as segment_stage

    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            return np.ones(he_rgb.shape[:2], np.int32)

    monkeypatch.setattr(segment, "get_segmenter", lambda *a, **k: Fake())
    monkeypatch.setitem(sys.modules, "torch", None)

    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((8, 8, 3), 10, np.uint8))
    s = Sample("breast__s1", "breast", tmp_path, he, True)
    cfg = cfg_factory()

    info = segment_stage(cfg, [s], cfg.output)

    assert info["nuclei_per_sample"][s.sample_id] == 1


def test_segment_stage_reuses_cached_masks(tmp_path, cfg_factory, monkeypatch):
    import tifffile

    from wsitrain.dataset import Sample
    from wsitrain.stages import segment as segment_stage

    calls = []

    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            calls.append(1)
            return np.ones(he_rgb.shape[:2], np.int32)

    monkeypatch.setattr(segment, "get_segmenter", lambda *a, **k: Fake())
    monkeypatch.setitem(sys.modules, "torch", None)

    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((8, 8, 3), 10, np.uint8))
    s = Sample("breast__s1", "breast", tmp_path, he, True)
    cfg = cfg_factory()

    segment_stage(cfg, [s], cfg.output)
    segment_stage(cfg, [s], cfg.output)

    assert calls == [1]
