"""Transfer stage: label join, registration, nucleus lookup, QC and label_map."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from wsitrain import paths
from wsitrain.stages import transfer

TASK = "sthelar_full"


def _nuclei(out, cfg, sample):
    return pd.read_csv(out / "nuclei" / cfg.tissue / f"{sample.sample_id}.csv")


def _base_cfg(cfg_factory, **over):
    # mpp=1.0 makes microns and pixels interchangeable, so the expected nucleus
    # hits can be read straight off the mask fixture.
    defaults = dict(mpp=1.0, transform="none", match_radius_px=0,
                    min_match_rate=0.0, task=TASK)
    defaults.update(over)
    return cfg_factory(**defaults)


def test_matched_cells_written_unmatched_dropped(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 11, 11), ("c3", 20, 20)],
        clusters=[("c1", 1), ("c2", 2), ("c3", 1)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    info = transfer(cfg, [s], cfg.output)

    assert info["cells_per_sample"][s.sample_id] == 2
    assert info["match_rate"][s.sample_id] == pytest.approx(2 / 3, abs=1e-4)
    assert len(_nuclei(cfg.output, cfg, s)) == 2


def test_label_map_is_sorted_and_zero_based(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 11, 11)],
        clusters=[("c1", 1), ("c2", 2)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    transfer(cfg, [s], cfg.output)

    text = paths.label_map_path(cfg.output, cfg.tissue).read_text()
    assert text == '0: "immune"\n1: "tumor"\n'


def test_agreeing_duplicates_collapse_to_one_row(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    # c1 and c1b both land on nucleus 1 carrying the same label.
    s = sample_factory(
        cells=[("c1", 5, 5), ("c1b", 5, 6), ("c2", 11, 11)],
        clusters=[("c1", 1), ("c1b", 1), ("c2", 2)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    info = transfer(cfg, [s], cfg.output)

    assert info["cells_per_sample"][s.sample_id] == 2
    assert info["conflicting_cells"][s.sample_id] == 0


def test_conflicting_duplicates_are_discarded(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    # Two disagreeing labels on nucleus 1; nucleus 2 stays clean.
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 5, 6), ("c3", 11, 11)],
        clusters=[("c1", 1), ("c2", 2), ("c3", 1)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    info = transfer(cfg, [s], cfg.output)

    assert info["conflicting_cells"][s.sample_id] == 2
    assert info["cells_per_sample"][s.sample_id] == 1


def test_match_rate_excludes_dedup_losses(cfg_factory, sample_factory, mask_factory):
    """Every cell hits a nucleus, so the rate is 1.0 even though dedup removes rows."""
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 5, 6), ("c3", 11, 11)],
        clusters=[("c1", 1), ("c2", 2), ("c3", 1)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    info = transfer(cfg, [s], cfg.output)

    assert info["match_rate"][s.sample_id] == 1.0


def test_radius_search_recovers_near_misses(cfg_factory, sample_factory, mask_factory):
    blobs = ((1, 6, 6, 3),)
    args = dict(cells=[("c1", 5, 5)], clusters=[("c1", 1)], assign=[(1, "tumor")])

    strict = _base_cfg(cfg_factory, match_radius_px=0)
    s1 = sample_factory("strict", **args)
    mask_factory(s1, strict.output, blobs=blobs)
    assert transfer(strict, [s1], strict.output)["match_rate"][s1.sample_id] == 0.0

    loose = _base_cfg(cfg_factory, match_radius_px=3)
    s2 = sample_factory("loose", **args)
    mask_factory(s2, loose.output, blobs=blobs)
    assert transfer(loose, [s2], loose.output)["match_rate"][s2.sample_id] == 1.0


def test_slide_below_min_match_rate_is_dropped(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory, min_match_rate=0.9)
    good = sample_factory("good", cells=[("a", 5, 5)], clusters=[("a", 1)],
                          assign=[(1, "tumor")])
    bad = sample_factory("bad", cells=[("b", 21, 21)], clusters=[("b", 1)],
                         assign=[(1, "tumor")])
    mask_factory(good, cfg.output)
    mask_factory(bad, cfg.output)

    info = transfer(cfg, [good, bad], cfg.output)

    assert info["dropped_slides"] == [bad.sample_id]
    assert good.sample_id in info["cells_per_sample"]


def test_all_slides_dropped_raises(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory, min_match_rate=0.9)
    s = sample_factory(cells=[("a", 21, 21)], clusters=[("a", 1)], assign=[(1, "tumor")])
    mask_factory(s, cfg.output)

    with pytest.raises(RuntimeError, match="registration is broken"):
        transfer(cfg, [s], cfg.output)


def test_classes_only_on_dropped_slides_are_compacted(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory, min_match_rate=0.9)
    good = sample_factory("good", cells=[("a", 5, 5)], clusters=[("a", 1)],
                          assign=[(1, "tumor")])
    # 'rare' exists only on the slide that fails QC and must leave the label map.
    bad = sample_factory("bad", cells=[("b", 21, 21)], clusters=[("b", 1)],
                         assign=[(1, "rare")])
    mask_factory(good, cfg.output)
    mask_factory(bad, cfg.output)

    info = transfer(cfg, [good, bad], cfg.output)

    assert info["n_classes"] == 1
    assert paths.label_map_path(cfg.output, cfg.tissue).read_text() == '0: "tumor"\n'
    assert _nuclei(cfg.output, cfg, good)["class_int"].tolist() == [0]


def test_drop_labels_removes_requested_types(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory, drop_labels=("Immune",))
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 11, 11)],
        clusters=[("c1", 1), ("c2", 2)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    info = transfer(cfg, [s], cfg.output)

    assert info["n_classes"] == 1
    assert paths.label_map_path(cfg.output, cfg.tissue).read_text() == '0: "tumor"\n'


def test_missing_registration_raises_when_transform_requested(
        cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory, transform="affine")
    s = sample_factory(cells=[("c1", 5, 5)], clusters=[("c1", 1)], assign=[(1, "tumor")])
    mask_factory(s, cfg.output)

    with pytest.raises(RuntimeError, match="registration_params.json"):
        transfer(cfg, [s], cfg.output)


def test_stale_cluster_assignment_raises(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    # Cluster 9 has no row in the assignment CSV.
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 11, 11)],
        clusters=[("c1", 1), ("c2", 9)],
        assign=[(1, "tumor")])
    mask_factory(s, cfg.output)

    with pytest.raises(RuntimeError, match="stale relative to clusters.csv"):
        transfer(cfg, [s], cfg.output)


def test_barcode_mismatch_raises(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(
        cells=[("c1", 5, 5), ("c2", 11, 11)],
        clusters=[("zz1", 1), ("zz2", 2)],
        assign=[(1, "tumor"), (2, "immune")])
    mask_factory(s, cfg.output)

    with pytest.raises(RuntimeError, match="do not line up"):
        transfer(cfg, [s], cfg.output)


def test_nucleus_boundaries_override_cell_centroid(cfg_factory, sample_factory, mask_factory):
    """The cell centroid sits off-nucleus; the nucleus centroid rescues the match."""
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(
        cells=[("c1", 1, 1)], clusters=[("c1", 1)], assign=[(1, "tumor")],
        nucleus_offset=(4, 4))
    mask_factory(s, cfg.output)

    assert transfer(cfg, [s], cfg.output)["match_rate"][s.sample_id] == 1.0


def test_out_of_bounds_points_are_not_clamped(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(cells=[("c1", 500, 500)], clusters=[("c1", 1)],
                       assign=[(1, "tumor")])
    mask_factory(s, cfg.output)

    assert transfer(cfg, [s], cfg.output)["match_rate"][s.sample_id] == 0.0


def test_written_columns_match_tile_contract(cfg_factory, sample_factory, mask_factory):
    cfg = _base_cfg(cfg_factory)
    s = sample_factory(cells=[("c1", 5, 5)], clusters=[("c1", 1)], assign=[(1, "tumor")])
    mask_factory(s, cfg.output)

    transfer(cfg, [s], cfg.output)

    assert list(_nuclei(cfg.output, cfg, s).columns) == ["x_px", "y_px", "class_int"]
