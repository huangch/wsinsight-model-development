"""End-to-end chain: segment -> transfer -> tile -> split, on realistic names.

The per-stage tests use tidy identifiers. Real 10x sample directories contain
spaces, commas, parentheses and colons, and those names flow into tile stems,
CSV filenames and split files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

pytest.importorskip("pyarrow")

from wsitrain import paths, segment as segment_mod, splits, weights
from wsitrain.config import build_config
from wsitrain.dataset import discover_samples
from wsitrain.stages import segment, split, tile, transfer

CLUSTER_REL = "analysis/clustering/gene_expression_graphclust/clusters.csv"

# Modelled on the real data: spaces, a comma, parentheses and a colon.
SAMPLE_NAMES = [
    "FFPE Human Breast with Pre-designed Panel/Tissue sample 1 (IDC)",
    "Post-Xenium Technical Note: v1 and Prime 5K/Experiment 1, v1",
]

LABELS = {1: "Tumor, invasive", 2: "T cell: CD8"}


@pytest.fixture
def dataset(tmp_path):
    """Two breast samples with cells on a grid, plus dark H&E images."""
    root = tmp_path / "input"
    for name in SAMPLE_NAMES:
        outs = root / "breast" / name / "outs"
        (outs / CLUSTER_REL).parent.mkdir(parents=True, exist_ok=True)

        cells, clusters = [], []
        cid = 0
        for y in range(4, 60, 6):
            for x in range(4, 60, 6):
                cells.append((f"c{cid}", float(x), float(y)))
                clusters.append((f"c{cid}", 2 if cid % 3 == 0 else 1))
                cid += 1
        pd.DataFrame(cells, columns=["cell_id", "x_centroid", "y_centroid"]).to_parquet(
            outs / "cells.parquet")
        pd.DataFrame(clusters, columns=["Barcode", "Cluster"]).to_csv(
            outs / CLUSTER_REL, index=False)
        pd.DataFrame(sorted(LABELS.items()), columns=["classification", "cell_type"]).to_csv(
            outs / "celltype_assignment_pantissue_label.csv", index=False)

        tifffile.imwrite(outs.parent / "x_he_image.ome.tif",
                         np.full((64, 64, 3), 20, np.uint8))
    return root


@pytest.fixture
def cfg(dataset, tmp_path):
    return build_config(dataset, "breast", tmp_path / "out",
                        overrides={"task": "pantissue", "transform": "none",
                                   "mpp": 1.0, "match_radius_px": 2,
                                   "min_match_rate": 0.0, "tile_px": 16,
                                   "min_cells": 1, "bg_thresh": 250.0,
                                   "overlap": 0.0, "val_frac": 0.5,
                                   "by_slide": False, "segmenter": "fake"})


@pytest.fixture
def pipeline(cfg, dataset, monkeypatch):
    """Run segment->transfer->tile->split with a deterministic fake segmenter."""
    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            # One 3x3 nucleus per cell position, ids from 1.
            mask = np.zeros(he_rgb.shape[:2], np.int32)
            nid = 0
            for y in range(4, 60, 6):
                for x in range(4, 60, 6):
                    nid += 1
                    mask[y - 1:y + 2, x - 1:x + 2] = nid
            return mask

    monkeypatch.setattr(segment_mod, "get_segmenter", lambda *a, **k: Fake())
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    samples = discover_samples(dataset, "breast")
    results = {"samples": samples}
    results["segment"] = segment(cfg, samples, cfg.output)
    results["transfer"] = transfer(cfg, samples, cfg.output)
    results["tile"] = tile(cfg, samples, cfg.output)
    results["split"] = split(cfg, samples, cfg.output)
    return results


def test_both_samples_discovered(pipeline):
    assert len(pipeline["samples"]) == 2


def test_segment_produces_masks(pipeline):
    assert all(v > 0 for v in pipeline["segment"]["nuclei_per_sample"].values())


def test_transfer_keeps_both_slides(pipeline):
    assert len(pipeline["transfer"]["cells_per_sample"]) == 2


def test_transfer_registers_both_classes(pipeline):
    assert pipeline["transfer"]["n_classes"] == 2


def test_label_map_survives_colon_and_comma(cfg, pipeline):
    names = set(weights.load_label_map(
        paths.label_map_path(cfg.output, cfg.tissue)).values())
    assert names == set(LABELS.values())


def test_tiles_written(pipeline):
    assert pipeline["tile"]["tiles"] > 0


def test_every_tile_has_a_label_file(cfg, pipeline):
    imgs = {p.stem for p in paths.images_dir(cfg.output, cfg.tissue).glob("*.png")}
    labs = {p.stem for p in paths.labels_dir(cfg.output, cfg.tissue).glob("*.csv")}
    assert imgs == labs


def test_tile_coordinates_are_inside_the_tile(cfg, pipeline):
    for csv in paths.labels_dir(cfg.output, cfg.tissue).glob("*.csv"):
        for line in csv.read_text().splitlines():
            x, y, _ = line.split(",")
            assert 0 <= int(x) < cfg.tile_px
            assert 0 <= int(y) < cfg.tile_px


def test_tile_classes_are_within_the_label_map(cfg, pipeline):
    n = len(weights.load_label_map(paths.label_map_path(cfg.output, cfg.tissue)))
    seen = set(weights.tally_labels(paths.labels_dir(cfg.output, cfg.tissue)))
    assert seen and max(seen) < n


def test_sample_tag_recovers_slide_from_tile_stem(cfg, pipeline):
    slides = {splits.sample_tag(p.stem)
              for p in paths.labels_dir(cfg.output, cfg.tissue).glob("*.csv")}
    assert len(slides) == 2
    assert all(s.startswith("breast__") for s in slides)


def test_split_is_non_empty_on_both_sides(pipeline):
    assert pipeline["split"]["n_train"] > 0 and pipeline["split"]["n_val"] > 0


def test_split_covers_every_slide(cfg, pipeline):
    sd = paths.splits_dir(cfg.output, cfg.tissue, cfg.fold)
    train = sd.joinpath("train.csv").read_text().splitlines()
    val = sd.joinpath("val.csv").read_text().splitlines()
    assert {splits.sample_tag(t) for t in val} == {splits.sample_tag(t) for t in train}


def test_split_files_have_no_blank_entries(cfg, pipeline):
    sd = paths.splits_dir(cfg.output, cfg.tissue, cfg.fold)
    for name in ("train.csv", "val.csv"):
        lines = sd.joinpath(name).read_text().splitlines()
        assert all(line.strip() for line in lines)


def test_split_entries_resolve_to_real_files(cfg, pipeline):
    sd = paths.splits_dir(cfg.output, cfg.tissue, cfg.fold)
    lab = paths.labels_dir(cfg.output, cfg.tissue)
    img = paths.images_dir(cfg.output, cfg.tissue)
    for name in ("train.csv", "val.csv"):
        for stem in sd.joinpath(name).read_text().splitlines():
            assert (lab / f"{stem}.csv").exists()
            assert (img / f"{stem}.png").exists()


def test_weights_sum_to_class_count(cfg, pipeline):
    rep = weights.compute_weights(paths.label_map_path(cfg.output, cfg.tissue),
                                  paths.labels_dir(cfg.output, cfg.tissue),
                                  cap=cfg.weight_cap)
    assert sum(rep.weights) == pytest.approx(len(rep.label_map), abs=0.01)


def test_rendered_config_is_valid_yaml(cfg, pipeline):
    import yaml

    path = (paths.tissue_root(cfg.output, cfg.tissue) / "train_configs"
            / cfg.backbone / f"{cfg.fold}.yaml")
    doc = yaml.safe_load(path.read_text())
    assert doc["data"]["num_classes"] == 2
    assert set(doc["data"]["label_map"].values()) == set(LABELS.values())
