"""Shared fixtures: synthetic Xenium/H&E samples, masks and run configs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wsitrain.config import build_config
from wsitrain.dataset import Sample

CLUSTER_REL = "analysis/clustering/gene_expression_graphclust/clusters.csv"


@pytest.fixture
def cfg_factory(tmp_path):
    """RunConfig anchored at tmp_path, with attributes overridable per test."""
    def _make(tissue: str = "breast", **overrides):
        cfg = build_config(tmp_path / "input", tissue, tmp_path / "out")
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"RunConfig has no field {key!r}")
            setattr(cfg, key, value)
        return cfg
    return _make


@pytest.fixture
def sample_factory(tmp_path):
    """Write a minimal Xenium `outs/` tree and return the matching Sample.

    ``cells`` is [(cell_id, x_um, y_um)], ``clusters`` is [(cell_id, cluster)],
    ``assign`` is [(cluster, cell_type)].
    """
    def _make(sample_id: str = "s1", tissue: str = "breast", *,
              cells, clusters, assign, task: str = "sthelar_full",
              nucleus_offset: tuple[float, float] | None = None) -> Sample:
        outs = tmp_path / "input" / tissue / sample_id / "outs"
        (outs / CLUSTER_REL).parent.mkdir(parents=True, exist_ok=True)

        cell_df = pd.DataFrame(cells, columns=["cell_id", "x_centroid", "y_centroid"])
        cell_df.to_parquet(outs / "cells.parquet")
        pd.DataFrame(clusters, columns=["Barcode", "Cluster"]).to_csv(
            outs / CLUSTER_REL, index=False)
        pd.DataFrame(assign, columns=["classification", "cell_type"]).to_csv(
            outs / f"celltype_assignment_{task}_label.csv", index=False)

        if nucleus_offset is not None:
            dx, dy = nucleus_offset
            rows = []
            for cid, x, y in cells:
                rows += [(cid, x + dx, y + dy), (cid, x + dx + 1, y + dy),
                         (cid, x + dx, y + dy + 1)]
            pd.DataFrame(rows, columns=["cell_id", "vertex_x", "vertex_y"]).to_parquet(
                outs / "nucleus_boundaries.parquet")

        he = outs.parent / f"{sample_id}_he_image.ome.tif"
        he.write_bytes(b"")
        return Sample(f"{tissue}__{sample_id}", tissue, outs, he, True)
    return _make


@pytest.fixture
def mask_factory(tmp_path):
    """Write an instance mask where each entry is (nucleus_id, y0, x0, size)."""
    def _make(sample: Sample, out: Path, tissue: str = "breast", *,
              shape: tuple[int, int] = (24, 24), blobs=((1, 4, 4, 3), (2, 10, 10, 3))):
        mask = np.zeros(shape, dtype=np.int32)
        for nid, y0, x0, size in blobs:
            mask[y0:y0 + size, x0:x0 + size] = nid
        dst = out / "masks" / tissue / f"{sample.sample_id}.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, mask)
        return mask
    return _make


@pytest.fixture
def label_dir_factory(tmp_path):
    """Create a directory of headerless tile label CSVs: {stem: [class_int, ...]}."""
    def _make(tiles: dict[str, list[int]], name: str = "labels") -> Path:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        for stem, classes in tiles.items():
            (d / f"{stem}.csv").write_text(
                "".join(f"{i},{i},{c}\n" for i, c in enumerate(classes)))
        return d
    return _make
