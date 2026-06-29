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


class CellposeSegmenter:
    name = "cellpose"

    def __init__(self, model: str = "cpsam", diameter: float | None = None):
        self.model_name = model
        self.diameter = diameter

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        from cellpose import models  # lazy
        model = models.CellposeModel(gpu=True, model_type=self.model_name)
        masks, *_ = model.eval(he_rgb, diameter=self.diameter, channels=[0, 0])
        return masks.astype("int32")


class StarDistSegmenter:
    name = "stardist"

    def __init__(self, model: str = "2D_versatile_he"):
        self.model_name = model

    def segment(self, he_rgb: np.ndarray, *, mpp: float) -> np.ndarray:
        from stardist.models import StarDist2D  # lazy
        from csbdeep.utils import normalize
        model = StarDist2D.from_pretrained(self.model_name)
        labels, _ = model.predict_instances(normalize(he_rgb))
        return labels.astype("int32")


def get_segmenter(name: str, *, cellpose_model: str = "cpsam",
                  diameter: float | None = None) -> Segmenter:
    if name == "cellpose":
        return CellposeSegmenter(cellpose_model, diameter)
    if name == "stardist":
        return StarDistSegmenter()
    raise ValueError(f"unknown segmenter: {name!r} (cellpose | stardist)")
