"""Segmentation backends: GPU selection, mpp handling, resampling."""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from wsitrain import segment


@pytest.fixture(autouse=True)
def _clear_stardist_env(monkeypatch):
    """A developer's own KERAS_HOME must not decide where these tests look."""
    monkeypatch.delenv("WSITRAIN_STARDIST_DIR", raising=False)
    monkeypatch.delenv("KERAS_HOME", raising=False)


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


def test_stardist_loads_from_a_local_folder(monkeypatch):
    """An offline model dir must bypass from_pretrained entirely."""
    seen = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            seen["name"] = name
            seen["basedir"] = basedir

        @staticmethod
        def from_pretrained(name):
            seen["downloaded"] = True
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)

    seg = segment.get_segmenter("stardist", stardist_model="2D_versatile_he",
                                stardist_model_dir="/models/StarDist2D")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["name"] == "2D_versatile_he"
    assert seen["basedir"] == "/models/StarDist2D"
    assert "downloaded" not in seen


def test_stardist_downloads_when_no_folder_given(monkeypatch, tmp_path):
    seen = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            pass

        @staticmethod
        def from_pretrained(name):
            seen["downloaded"] = name
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    # Point the cache probe at an empty dir so the download path is taken.
    monkeypatch.setattr(segment, "STARDIST_CACHE", tmp_path / "empty")

    seg = segment.get_segmenter("stardist")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert seen["downloaded"] == "2D_versatile_he"


def test_stardist_prefers_the_unpacked_cache(monkeypatch, tmp_path):
    """csbdeep re-downloads even when the folder exists; use it directly."""
    seen = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            seen["basedir"] = basedir
            seen["name"] = name

        @staticmethod
        def from_pretrained(name):
            seen["downloaded"] = True
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    cache = tmp_path / "StarDist2D"
    (cache / "2D_versatile_he").mkdir(parents=True)
    (cache / "2D_versatile_he" / "config.json").write_text("{}")
    monkeypatch.setattr(segment, "STARDIST_CACHE", cache)

    seg = segment.get_segmenter("stardist")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["basedir"] == str(cache)
    assert "downloaded" not in seen


def test_explicit_dir_beats_the_cache(monkeypatch, tmp_path):
    seen = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            seen["basedir"] = basedir

        @staticmethod
        def from_pretrained(name):
            seen["downloaded"] = True
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    cache = tmp_path / "StarDist2D"
    (cache / "2D_versatile_he").mkdir(parents=True)
    (cache / "2D_versatile_he" / "config.json").write_text("{}")
    monkeypatch.setattr(segment, "STARDIST_CACHE", cache)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/explicit")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert seen["basedir"] == "/explicit"


def _stardist_probe(monkeypatch, tmp_path):
    """Fake StarDist2D that records the basedir it was handed."""
    seen = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            seen["basedir"] = basedir

        @staticmethod
        def from_pretrained(name):
            seen["downloaded"] = True
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    monkeypatch.setattr(segment, "STARDIST_CACHE", tmp_path / "empty")
    return seen


def _unpack_model(root, name="2D_versatile_he"):
    (root / name).mkdir(parents=True)
    (root / name / "config.json").write_text("{}")
    return root


def test_wsitrain_stardist_dir_env_is_used(monkeypatch, tmp_path):
    """Offline hosts keep the model outside the Keras cache."""
    seen = _stardist_probe(monkeypatch, tmp_path)
    models = _unpack_model(tmp_path / "models")
    monkeypatch.setenv("WSITRAIN_STARDIST_DIR", str(models))

    segment.get_segmenter("stardist").segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["basedir"] == str(models)
    assert "downloaded" not in seen


def test_keras_home_env_is_used(monkeypatch, tmp_path):
    """Moving the Keras cache off $HOME must not force a re-download."""
    seen = _stardist_probe(monkeypatch, tmp_path)
    models = _unpack_model(tmp_path / "keras" / "models" / "StarDist2D")
    monkeypatch.setenv("KERAS_HOME", str(tmp_path / "keras"))

    segment.get_segmenter("stardist").segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["basedir"] == str(models)
    assert "downloaded" not in seen


def test_wsitrain_env_beats_keras_home(monkeypatch, tmp_path):
    seen = _stardist_probe(monkeypatch, tmp_path)
    explicit = _unpack_model(tmp_path / "explicit")
    _unpack_model(tmp_path / "keras" / "models" / "StarDist2D")
    monkeypatch.setenv("WSITRAIN_STARDIST_DIR", str(explicit))
    monkeypatch.setenv("KERAS_HOME", str(tmp_path / "keras"))

    segment.get_segmenter("stardist").segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["basedir"] == str(explicit)


def test_flag_beats_the_env_vars(monkeypatch, tmp_path):
    seen = _stardist_probe(monkeypatch, tmp_path)
    monkeypatch.setenv("WSITRAIN_STARDIST_DIR", str(_unpack_model(tmp_path / "env")))

    seg = segment.get_segmenter("stardist", stardist_model_dir="/explicit")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["basedir"] == "/explicit"


def test_env_pointing_nowhere_falls_back_to_download(monkeypatch, tmp_path):
    seen = _stardist_probe(monkeypatch, tmp_path)
    monkeypatch.setenv("WSITRAIN_STARDIST_DIR", str(tmp_path / "missing"))

    segment.get_segmenter("stardist").segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)

    assert seen["downloaded"] is True


def test_stardist_cpu_hides_gpus_from_tensorflow_only(monkeypatch, tmp_path):
    hidden = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            pass

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.config = types.SimpleNamespace(
        set_visible_devices=lambda devs, kind: hidden.update({"kind": kind, "devs": devs}))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/m", stardist_cpu=True)
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert hidden == {"kind": "GPU", "devs": []}


def test_stardist_gpu_mode_does_not_touch_tensorflow_devices(monkeypatch):
    hidden = {}

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            pass

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)
    fake_tf = types.ModuleType("tensorflow")
    fake_tf.config = types.SimpleNamespace(
        set_visible_devices=lambda devs, kind: hidden.update({"called": True}))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/m")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert hidden == {}


def test_missing_cuda_toolkit_gives_an_actionable_error(monkeypatch):
    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            pass

        def predict_instances(self, img):
            raise RuntimeError("libdevice not found at ./libdevice.10.bc")

    _install_fake_stardist(monkeypatch, FakeModel)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/m")
    with pytest.raises(RuntimeError, match="--stardist-cpu"):
        seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)


def test_unrelated_errors_are_not_swallowed(monkeypatch):
    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            pass

        def predict_instances(self, img):
            raise ValueError("something else entirely")

    _install_fake_stardist(monkeypatch, FakeModel)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/m")
    with pytest.raises(ValueError, match="something else"):
        seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)


# --------------------------------------------------------------------------
# XLA CUDA shim
# --------------------------------------------------------------------------

def _fake_triton(monkeypatch, tmp_path, *, complete=True):
    nvidia = tmp_path / "triton" / "backends" / "nvidia"
    (nvidia / "bin").mkdir(parents=True)
    (nvidia / "lib").mkdir(parents=True)
    (nvidia / "bin" / "ptxas").write_text("#!/bin/true\n")
    if complete:
        (nvidia / "lib" / "libdevice.10.bc").write_bytes(b"\0")
    mod = types.ModuleType("triton")
    mod.__file__ = str(tmp_path / "triton" / "__init__.py")
    monkeypatch.setitem(sys.modules, "triton", mod)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.delenv("XLA_FLAGS", raising=False)


def test_shim_exposes_the_layout_xla_expects(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    shim = segment.configure_tensorflow_cuda()
    assert shim is not None
    root = __import__("pathlib").Path(shim)
    assert (root / "bin" / "ptxas").exists()
    assert (root / "nvvm" / "libdevice" / "libdevice.10.bc").exists()


def test_shim_is_added_to_xla_flags(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    shim = segment.configure_tensorflow_cuda()
    assert f"--xla_gpu_cuda_data_dir={shim}" in os.environ["XLA_FLAGS"]


def test_memory_growth_is_enabled(monkeypatch, tmp_path):
    """TF reserves most of the GPU on init; torch needs it for the train stage."""
    _fake_triton(monkeypatch, tmp_path)
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    segment.configure_tensorflow_cuda()
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"


def test_memory_growth_respects_an_override(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "false")
    segment.configure_tensorflow_cuda()
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "false"


def test_existing_xla_flags_are_preserved(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    monkeypatch.setenv("XLA_FLAGS", "--foo=1")
    segment.configure_tensorflow_cuda()
    assert os.environ["XLA_FLAGS"].startswith("--foo=1")


def test_user_supplied_cuda_dir_is_not_overridden(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/mine")
    assert segment.configure_tensorflow_cuda() is None
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_cuda_data_dir=/mine"


def test_shim_is_skipped_when_triton_is_incomplete(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path, complete=False)
    assert segment.configure_tensorflow_cuda() is None


def test_shim_is_skipped_without_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    assert segment.configure_tensorflow_cuda() is None


def test_shim_is_idempotent(monkeypatch, tmp_path):
    _fake_triton(monkeypatch, tmp_path)
    first = segment.configure_tensorflow_cuda()
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    assert segment.configure_tensorflow_cuda() == first


def test_shim_repairs_a_dangling_symlink(monkeypatch, tmp_path):
    """A triton reinstall can leave the old link pointing at nothing."""
    import pathlib

    _fake_triton(monkeypatch, tmp_path)
    shim = pathlib.Path(segment.configure_tensorflow_cuda())

    ptxas = shim / "bin" / "ptxas"
    ptxas.unlink()
    ptxas.symlink_to(tmp_path / "gone" / "ptxas")
    assert ptxas.is_symlink() and not ptxas.exists()

    monkeypatch.delenv("XLA_FLAGS", raising=False)
    segment.configure_tensorflow_cuda()

    assert ptxas.exists()


def test_stardist_model_is_loaded_once(monkeypatch):
    loads = []

    class FakeModel:
        def __init__(self, config, name=None, basedir=None):
            loads.append(1)

        @staticmethod
        def from_pretrained(name):
            return FakeModel(None)

        def predict_instances(self, img):
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)

    seg = segment.get_segmenter("stardist", stardist_model_dir="/models")
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    seg.segment(np.zeros((8, 8, 3), np.uint8), mpp=0.25)
    assert len(loads) == 1


def _install_fake_stardist(monkeypatch, model_cls):
    monkeypatch.setitem(sys.modules, "stardist", types.ModuleType("stardist"))
    sd_models = types.ModuleType("stardist.models")
    sd_models.StarDist2D = model_cls
    monkeypatch.setitem(sys.modules, "stardist.models", sd_models)
    csb = types.ModuleType("csbdeep")
    csb_utils = types.ModuleType("csbdeep.utils")
    csb_utils.normalize = lambda x, pmin=3, pmax=99.8: x
    monkeypatch.setitem(sys.modules, "csbdeep", csb)
    monkeypatch.setitem(sys.modules, "csbdeep.utils", csb_utils)


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
        def __init__(self, config, name=None, basedir=None):
            pass

        @staticmethod
        def from_pretrained(name):
            return FakeModel(None)

        def predict_instances(self, img):
            seen["shape"] = img.shape
            return np.zeros(img.shape[:2], np.int32), None

    _install_fake_stardist(monkeypatch, FakeModel)

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
