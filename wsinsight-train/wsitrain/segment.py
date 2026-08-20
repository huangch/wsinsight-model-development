"""Pluggable nucleus segmentation backends.

A ``Segmenter`` takes an H&E RGB array and returns an int32 instance mask
(0 = background). Cellpose is the default; StarDist is kept for parity with
the legacy QuPath StarDist labels. Future models (InstanSeg, CellViT-seg)
implement the same protocol. Heavy deps are imported lazily so the package
imports without a GPU stack.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

# Where csbdeep unpacks pretrained StarDist models.
STARDIST_CACHE = Path(os.path.expanduser("~/.keras/models/StarDist2D"))


def _cached_stardist_dir(name: str) -> Path | None:
    """csbdeep re-downloads the zip even when the folder is already unpacked."""
    return STARDIST_CACHE if (STARDIST_CACHE / name / "config.json").is_file() else None


# XLA needs a full CUDA toolkit (ptxas + libdevice), which a driver-only host lacks.
_TF_GPU_HINTS = ("ptx", "libdevice", "nvvm")


def _triton_cuda_dir() -> Path | None:
    """torch's triton ships ptxas + libdevice; XLA just wants them laid out its way."""
    try:
        import triton
    except ImportError:
        return None
    nvidia = Path(triton.__file__).parent / "backends" / "nvidia"
    ptxas, libdevice = nvidia / "bin" / "ptxas", nvidia / "lib" / "libdevice.10.bc"
    if not (ptxas.is_file() and libdevice.is_file()):
        return None

    shim = Path(os.environ.get("TMPDIR", "/tmp")) / "wsitrain-xla-cuda"
    (shim / "bin").mkdir(parents=True, exist_ok=True)
    (shim / "nvvm" / "libdevice").mkdir(parents=True, exist_ok=True)
    for src, dst in ((ptxas, shim / "bin" / "ptxas"),
                     (libdevice, shim / "nvvm" / "libdevice" / "libdevice.10.bc")):
        # A triton reinstall can leave the old link dangling; exists() would
        # report False for it and symlink_to would then raise.
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return shim


def configure_tensorflow_cuda() -> str | None:
    """Point XLA at a usable CUDA data dir. Must run before TensorFlow loads."""
    # TensorFlow reserves most of the GPU on init, leaving little for the
    # torch/CellViT stage that follows in the same run.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    if "--xla_gpu_cuda_data_dir" in os.environ.get("XLA_FLAGS", ""):
        return None
    shim = _triton_cuda_dir()
    if shim is None:
        return None
    flags = os.environ.get("XLA_FLAGS", "").strip()
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={shim}".strip()
    return str(shim)


class Segmenter(Protocol):
    name: str

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        """Return int32 instance mask, shape (H, W)."""
        ...


def _resample_rgb(arr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 0.02:
        return arr
    from PIL import Image
    h, w = arr.shape[:2]
    size = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    return np.asarray(Image.fromarray(arr).resize(size, Image.BILINEAR))


def _resample_labels(labels: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if tuple(labels.shape[:2]) == tuple(hw):
        return labels
    from PIL import Image
    h, w = hw
    im = Image.fromarray(labels.astype(np.int32))
    return np.asarray(im.resize((w, h), Image.NEAREST))


class CellposeSegmenter:
    name = "cellpose"

    def __init__(self, model: str = "cpsam", diameter: float | None = None,
                 batch_size: int = 8, flow_threshold: float = 0.0,
                 *, gpu: bool = False):
        self.model_name = model
        self.diameter = diameter
        self.batch_size = batch_size
        self.flow_threshold = flow_threshold
        self.gpu = gpu
        self._model = None  # lazy: instantiated once on first call, reused across slides

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        from cellpose import models  # lazy
        if self._model is None:
            self._model = models.CellposeModel(gpu=self.gpu, model_type=self.model_name)
        # diameter is configured in microns; cellpose expects pixels.
        diameter_px = (self.diameter / float(mpp)) if self.diameter else None
        masks, *_ = self._model.eval(he_rgb, diameter=diameter_px, channels=[0, 0],
                                     batch_size=self.batch_size,
                                     flow_threshold=self.flow_threshold)
        return masks.astype("int32")


class StarDistSegmenter:
    name = "stardist"
    # 2D_versatile_he was trained on H&E at roughly this resolution.
    native_mpp = 0.25
    # Above this side length the slide is predicted block-wise. A whole slide in
    # one pass asks for a single convolution over the full frame: a 37k x 57k
    # slide needs a 24 GiB activation, which OOMs even an 80 GB A100.
    big_px = 4096
    # StarDist warns below ~94 px for this model's receptive field.
    big_context = 128
    big_min_overlap = 128

    def __init__(self, model: str = "2D_versatile_he", native_mpp: float | None = None,
                 model_dir=None, cpu: bool = False):
        self.model_name = model
        self.model_dir = model_dir
        self.cpu = cpu
        if native_mpp:
            self.native_mpp = float(native_mpp)
        self._model = None  # lazy: loaded once, reused across slides

    def _load(self):
        if self.cpu:
            import tensorflow as tf
            # Scoped to TensorFlow: torch (and the CellViT subprocess) keep the GPU.
            tf.config.set_visible_devices([], "GPU")
        else:
            configure_tensorflow_cuda()
        from stardist.models import StarDist2D  # lazy
        basedir = self.model_dir or _cached_stardist_dir(self.model_name)
        if basedir:
            # csbdeep layout: <basedir>/<name>/{config.json,weights_best.h5}
            return StarDist2D(None, name=self.model_name, basedir=str(basedir))
        return StarDist2D.from_pretrained(self.model_name)

    def _n_tiles(self, h: int, w: int) -> tuple[int, int, int]:
        """Tile count that keeps one network pass near a training batch.

        Mirrors StarDist's own heuristic but off the public config rather than
        the private _guess_n_tiles. Falls back to StarDist's 512 px training
        patch for any model that exposes no usable config.
        """
        cfg = getattr(self._model, "config", None)
        try:
            b = float(cfg.train_batch_size) ** (1.0 / int(cfg.n_dim))
            py = float(cfg.train_patch_size[0])
            px = float(cfg.train_patch_size[1])
        except Exception:
            b, py, px = 1.0, 512.0, 512.0
        return (max(int(np.ceil(h / (py * b))), 1),
                max(int(np.ceil(w / (px * b))), 1), 1)

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        from csbdeep.utils import normalize  # lazy
        if self._model is None:
            self._model = self._load()
        scale = float(mpp) / self.native_mpp
        img = _resample_rgb(he_rgb, scale)
        norm = normalize(img)
        h, w = norm.shape[:2]
        try:
            if max(h, w) > self.big_px:
                # Block-wise with overlap/context so nuclei on block seams are
                # not cut; returns one full-size label array.
                labels, _ = self._model.predict_instances_big(
                    norm, axes="YXC", block_size=self.big_px,
                    min_overlap=self.big_min_overlap, context=self.big_context,
                    n_tiles=self._n_tiles(self.big_px, self.big_px),
                    show_progress=False, show_tile_progress=False)
            else:
                n_tiles = self._n_tiles(h, w)
                if n_tiles == (1, 1, 1):
                    # One tile is already the default; omitting the kwargs keeps
                    # this working against any predict_instances signature.
                    labels, _ = self._model.predict_instances(norm)
                else:
                    labels, _ = self._model.predict_instances(
                        norm, n_tiles=n_tiles, show_tile_progress=False)
        except Exception as exc:
            if not self.cpu and any(h_ in str(exc).lower() for h_ in _TF_GPU_HINTS):
                raise RuntimeError(
                    "StarDist could not compile for the GPU: this host has no CUDA "
                    "toolkit (ptxas/libdevice). Re-run with --stardist-cpu, or use "
                    "--segmenter cellpose, which runs on the GPU through torch."
                ) from exc
            raise
        return _resample_labels(labels, he_rgb.shape[:2]).astype("int32")


def get_segmenter(name: str, *, cellpose_model: str = "cpsam",
                  diameter: float | None = None, batch_size: int = 8,
                  flow_threshold: float = 0.0, gpu: bool = False,
                  model_type: str | None = None,
                  stardist_model: str = "2D_versatile_he",
                  stardist_model_dir=None,
                  stardist_cpu: bool = False) -> Segmenter:
    if name == "cellpose":
        return CellposeSegmenter(
            model_type or cellpose_model,
            diameter,
            batch_size,
            flow_threshold,
            gpu=gpu,
        )
    if name == "stardist":
        return StarDistSegmenter(stardist_model, model_dir=stardist_model_dir,
                                 cpu=stardist_cpu)
    raise ValueError(f"unknown segmenter: {name!r} (cellpose | stardist)")
