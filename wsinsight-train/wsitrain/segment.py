"""Pluggable nucleus segmentation backends.

A ``Segmenter`` takes an H&E RGB array and returns an int32 instance mask
(0 = background). Cellpose is the default; StarDist is kept for parity with
the legacy QuPath StarDist labels. Future models (InstanSeg, CellViT-seg)
implement the same protocol. Heavy deps are imported lazily so the package
imports without a GPU stack.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


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

    def __init__(self, model: str = "2D_versatile_he", native_mpp: float | None = None):
        self.model_name = model
        if native_mpp:
            self.native_mpp = float(native_mpp)

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        from stardist.models import StarDist2D  # lazy
        from csbdeep.utils import normalize
        scale = float(mpp) / self.native_mpp
        img = _resample_rgb(he_rgb, scale)
        model = StarDist2D.from_pretrained(self.model_name)
        labels, _ = model.predict_instances(normalize(img))
        return _resample_labels(labels, he_rgb.shape[:2]).astype("int32")


def get_segmenter(name: str, *, cellpose_model: str = "cpsam",
                  diameter: float | None = None, batch_size: int = 8,
                  flow_threshold: float = 0.0, gpu: bool = False,
                  model_type: str | None = None) -> Segmenter:
    if name == "cellpose":
        return CellposeSegmenter(
            model_type or cellpose_model,
            diameter,
            batch_size,
            flow_threshold,
            gpu=gpu,
        )
    if name == "stardist":
        return StarDistSegmenter()
    raise ValueError(f"unknown segmenter: {name!r} (cellpose | stardist)")
