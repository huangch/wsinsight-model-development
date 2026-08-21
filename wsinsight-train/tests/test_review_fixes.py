"""Regressions for the review fixes: cache staleness, invalidation targets,
path safety, config carry-over, and the tqdm terminal hardening.
"""
from __future__ import annotations

import io
import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import tifffile

import wsitrain  # noqa: F401  -- installs the tqdm hardening on import
from wsitrain import dag, paths, segment as segment_mod, stages
from wsitrain.cli import main
from wsitrain.config import build_config
from wsitrain.dataset import Sample, _find_he
from wsitrain.manifest import Manifest
from wsitrain.splits import sample_tag
from wsitrain.stages import _segment_recipe, _segment_recipe_path, reset_cache


@pytest.fixture
def cohort(tmp_path):
    """Minimal sample tree that discover_samples() accepts."""
    outs = tmp_path / "cohort" / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    return tmp_path / "cohort"


@pytest.fixture
def seen_cfg(monkeypatch):
    """Stub the DAG out and capture the RunConfig the CLI assembled."""
    got = {}

    def fake_run(cfg, **kw):
        got["cfg"] = cfg
        got.update(kw)
        return 0

    monkeypatch.setattr(dag, "run", fake_run)
    return got


# --------------------------------------------------------------------------
# segment: mask reuse must be keyed on how the mask was made
# --------------------------------------------------------------------------

@pytest.fixture
def segment_probe(tmp_path, monkeypatch):
    """Run the segment stage against a fake backend; report per-slide calls."""
    calls = []

    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            calls.append(mpp)
            return np.ones(he_rgb.shape[:2], np.int32)

    monkeypatch.setattr(segment_mod, "get_segmenter", lambda name, **kw: Fake())
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((8, 8, 3), 10, np.uint8))
    sample = Sample("breast__s1", "breast", tmp_path, he, True)

    def _run(cfg):
        before = len(calls)
        stages.segment(cfg, [sample], cfg.output)
        return len(calls) - before

    return _run


def test_masks_are_reused_when_the_settings_match(cfg_factory, segment_probe):
    cfg = cfg_factory(segmenter="stardist")
    assert segment_probe(cfg) == 1
    assert segment_probe(cfg) == 0


def test_changing_segmenter_resegments_instead_of_reusing_masks(cfg_factory, segment_probe):
    """A cellpose run must not silently adopt StarDist masks."""
    assert segment_probe(cfg_factory(segmenter="stardist")) == 1
    assert segment_probe(cfg_factory(segmenter="cellpose")) == 1


def test_changing_mpp_resegments(cfg_factory, segment_probe):
    assert segment_probe(cfg_factory(segmenter="stardist", mpp=0.25)) == 1
    assert segment_probe(cfg_factory(segmenter="stardist", mpp=0.5)) == 1


def test_the_other_backends_knobs_do_not_resegment(cfg_factory, segment_probe):
    """Re-running a stardist cohort because --cellpose-model moved costs hours."""
    assert segment_probe(cfg_factory(segmenter="stardist", cellpose_model="cpsam")) == 1
    assert segment_probe(cfg_factory(segmenter="stardist", cellpose_model="nuclei")) == 0


def test_recipe_records_only_the_selected_backend(cfg_factory):
    stardist = _segment_recipe(cfg_factory(segmenter="stardist"))
    cellpose = _segment_recipe(cfg_factory(segmenter="cellpose"))
    assert "cellpose_model" not in stardist and "stardist_model" in stardist
    assert "stardist_model" not in cellpose and "cellpose_model" in cellpose


def test_recipe_sidecar_sits_beside_the_mask_dir(cfg_factory):
    """A lone sidecar inside masks/ would satisfy prereq's non-empty check."""
    mask_dir = paths.masks_dir(cfg_factory().output, "breast")
    assert _segment_recipe_path(mask_dir).parent == mask_dir.parent


def test_force_drops_the_masks_segment_would_reuse(cfg_factory):
    cfg = cfg_factory()
    mask_dir = paths.masks_dir(cfg.output, cfg.tissue)
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "breast__s1.npy", np.zeros((4, 4), np.int32))
    _segment_recipe_path(mask_dir).write_text("{}")

    reset_cache("segment", cfg, cfg.output)

    assert list(mask_dir.glob("*.npy")) == []
    assert not _segment_recipe_path(mask_dir).exists()


def test_force_leaves_other_stages_alone(cfg_factory):
    cfg = cfg_factory()
    mask_dir = paths.masks_dir(cfg.output, cfg.tissue)
    mask_dir.mkdir(parents=True)
    np.save(mask_dir / "breast__s1.npy", np.zeros((4, 4), np.int32))

    reset_cache("tile", cfg, cfg.output)

    assert list(mask_dir.glob("*.npy"))


def test_reset_cache_tolerates_a_missing_mask_dir(cfg_factory):
    reset_cache("segment", cfg_factory(), cfg_factory().output)


# --------------------------------------------------------------------------
# manifest: each key must invalidate the stage that actually writes its output
# --------------------------------------------------------------------------

def _manifest_with(tmp_path, config, done):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, config)
    for stage in done:
        mf.mark(stage, "done")
    return p


ALL_DONE = ("annotate", "segment", "transfer", "tile", "split", "train")


@pytest.mark.parametrize("key,old,new", [("backbone", "SAM-H-x40", "SAM-B-x40"),
                                         ("fold", "fold_0", "fold_1")])
def test_model_identity_invalidates_split_not_train(tmp_path, key, old, new):
    """split writes splits/<fold>/ and train_configs/<backbone>/<fold>.yaml."""
    base = {"backbone": "SAM-H-x40", "fold": "fold_0"}
    p = _manifest_with(tmp_path, base, ALL_DONE)

    fresh = Manifest.load_or_new(p, {**base, key: new})

    assert fresh.is_done("tile")
    assert not fresh.is_done("split")
    assert not fresh.is_done("train")


def test_transform_invalidates_segment(tmp_path):
    """transform decides which samples every stage sees, segmentation included."""
    p = _manifest_with(tmp_path, {"transform": "affine"}, ALL_DONE)

    fresh = Manifest.load_or_new(p, {"transform": "none"})

    assert fresh.is_done("annotate")
    assert not fresh.is_done("segment")
    assert not fresh.is_done("transfer")


# --------------------------------------------------------------------------
# config: identity of a run
# --------------------------------------------------------------------------

def test_relative_and_absolute_input_are_one_run(tmp_path, monkeypatch):
    """`input` is compared verbatim in the manifest, so the spelling must not matter."""
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)

    rel = build_config(Path("data"), "breast", None)
    absolute = build_config(tmp_path / "data", "breast", None)

    assert rel.to_dict()["input"] == absolute.to_dict()["input"]
    assert rel.output == absolute.output


def test_saved_config_is_inherited_by_a_later_command(tmp_path, cohort, seen_cfg):
    out = tmp_path / "out"
    out.mkdir()
    paths.resolved_config_path(out, "breast").write_text("segmenter: cellpose\n")

    main(["segment", "--input", str(cohort), "--tissue", "breast", "--output", str(out)])

    assert seen_cfg["cfg"].segmenter == "cellpose"


def test_reset_config_ignores_the_saved_config(tmp_path, cohort, seen_cfg):
    out = tmp_path / "out"
    out.mkdir()
    paths.resolved_config_path(out, "breast").write_text("segmenter: cellpose\n")

    main(["segment", "--input", str(cohort), "--tissue", "breast",
          "--output", str(out), "--reset-config"])

    assert seen_cfg["cfg"].segmenter == "stardist"


def test_no_stardist_cpu_takes_back_a_saved_true(tmp_path, cohort, seen_cfg):
    """A bare store_true could never be unset once it reached the saved config."""
    out = tmp_path / "out"
    out.mkdir()
    paths.resolved_config_path(out, "breast").write_text("stardist_cpu: true\n")

    main(["segment", "--input", str(cohort), "--tissue", "breast",
          "--output", str(out), "--no-stardist-cpu"])

    assert seen_cfg["cfg"].stardist_cpu is False


# --------------------------------------------------------------------------
# dag: a doomed invocation must leave nothing behind
# --------------------------------------------------------------------------

def test_empty_cohort_writes_no_resolved_config(tmp_path):
    cfg = build_config(tmp_path / "empty", "breast", tmp_path / "out")
    (tmp_path / "empty").mkdir()

    with pytest.raises(SystemExit):
        dag.run(cfg)

    assert not paths.resolved_config_path(cfg.output, cfg.tissue).exists()


def test_empty_cohort_does_not_invalidate_the_manifest(tmp_path):
    (tmp_path / "empty").mkdir()
    old = build_config(tmp_path / "empty", "breast", tmp_path / "out",
                       overrides={"segmenter": "stardist"})
    mpath = paths.manifest_path(old.output, old.tissue)
    mf = Manifest.load_or_new(mpath, old.to_dict())
    mf.mark("segment", "done")

    new = build_config(tmp_path / "empty", "breast", tmp_path / "out",
                       overrides={"segmenter": "cellpose"})
    with pytest.raises(SystemExit):
        dag.run(new)

    assert json.loads(mpath.read_text())["stages"]["segment"]["status"] == "done"


# --------------------------------------------------------------------------
# paths: --tissue is a path component
# --------------------------------------------------------------------------

@pytest.mark.parametrize("helper", ["tissue_root", "masks_dir", "nuclei_dir",
                                    "models_dir", "report_dir", "logs_dir"])
@pytest.mark.parametrize("tissue", ["../escape", "a/b", "breast lung"])
def test_tissue_cannot_escape_the_output_dir(tmp_path, helper, tissue):
    p = getattr(paths, helper)(tmp_path, tissue)
    assert ".." not in p.parts
    assert tmp_path in p.parents


def test_manifest_and_tree_agree_on_the_slug(tmp_path):
    """A tissue spelled one way in the tree and another in the manifest name
    makes a second run look untouched."""
    tissue = "breast lung"
    assert paths.tissue_root(tmp_path, tissue).name in paths.manifest_path(
        tmp_path, tissue).name


# --------------------------------------------------------------------------
# tile stems and their inverse
# --------------------------------------------------------------------------

def test_tile_stem_carries_row_and_column_separately():
    """`ti * 10000 + tj` collided once a row exceeded 10000 tiles."""
    assert sample_tag("breast__s1_tile_00003_10001") == "breast__s1"
    assert sample_tag("breast__s1_tile_00003_00001") == "breast__s1"


# --------------------------------------------------------------------------
# transfer: nucleus lookup near the slide border, and its failure messages
# --------------------------------------------------------------------------

def _transfer_cfg(cfg_factory, **over):
    defaults = dict(mpp=1.0, transform="none", match_radius_px=4,
                    min_match_rate=0.0, task="sthelar_full")
    defaults.update(over)
    return cfg_factory(**defaults)


def test_a_cell_within_the_radius_of_the_edge_still_matches(
        cfg_factory, sample_factory, mask_factory):
    """Excluding these points lost real matches all along the slide border."""
    pytest.importorskip("pyarrow")
    cfg = _transfer_cfg(cfg_factory)
    s = sample_factory(cells=[("c1", 1, 1)], clusters=[("c1", 1)],
                       assign=[(1, "tumor")])
    mask_factory(s, cfg.output, blobs=((1, 0, 0, 3),))

    info = stages.transfer(cfg, [s], cfg.output)

    assert info["cells_per_sample"][s.sample_id] == 1


def test_missing_mask_names_the_slide_and_the_cause(
        cfg_factory, sample_factory):
    pytest.importorskip("pyarrow")
    cfg = _transfer_cfg(cfg_factory)
    s = sample_factory(cells=[("c1", 5, 5)], clusters=[("c1", 1)],
                       assign=[(1, "tumor")])
    paths.masks_dir(cfg.output, cfg.tissue).mkdir(parents=True)

    with pytest.raises(SystemExit, match="no mask for"):
        stages.transfer(cfg, [s], cfg.output)


def test_affine_bspline_without_the_elastic_file_is_explained(
        cfg_factory, sample_factory, mask_factory):
    """bunwarp's own ValueError named neither the slide nor the way out."""
    pytest.importorskip("pyarrow")
    cfg = _transfer_cfg(cfg_factory, transform="affine+bspline")
    s = sample_factory(cells=[("c1", 5, 5)], clusters=[("c1", 1)],
                       assign=[(1, "tumor")])
    mask_factory(s, cfg.output)
    (s.outs / "registration_params.json").write_text("{}")

    with pytest.raises(RuntimeError, match="direct_transf.txt"):
        stages.transfer(cfg, [s], cfg.output)


# --------------------------------------------------------------------------
# discovery + run selection
# --------------------------------------------------------------------------

def test_named_he_wins_over_a_loose_match(tmp_path):
    (tmp_path / "cache.ome.tif").write_bytes(b"")
    (tmp_path / "s1_he_image.ome.tif").write_bytes(b"")
    assert _find_he(tmp_path).name == "s1_he_image.ome.tif"


def test_run_dir_matches_a_mixed_case_tissue(tmp_path, cfg_factory, monkeypatch):
    """_run_tag lowercases only the backbone, but the path is compared lowercased."""
    cellvit = tmp_path / "CellViT"
    cfg = cfg_factory(tissue="Breast")
    tag = f"{cfg.tissue}-{cfg.task}-{cfg.backbone.lower()}"
    run = cellvit / "logs_local" / tag / "checkpoints"
    run.mkdir(parents=True)
    (run / "model_best.pth").write_text("w")
    monkeypatch.setenv("CELLVIT_ROOT", str(cellvit))

    assert stages._find_run_dir(cfg, cfg.output) == run.parent


def test_validate_says_so_when_nothing_was_scored(cfg_factory, monkeypatch, capsys):
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    cfg = cfg_factory()
    run = paths.logs_dir(cfg.output, cfg.tissue) / "run" / "checkpoints"
    run.mkdir(parents=True)
    (run / "model_best.pth").write_text("w")

    stages.validate(cfg, [], cfg.output)

    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------
# tqdm: bar style and terminal resize
# --------------------------------------------------------------------------

def _winch_handler():
    handler = signal.getsignal(signal.SIGWINCH)
    assert callable(handler), "wsitrain import did not install a SIGWINCH handler"
    return handler


def test_bars_default_to_the_ascii_style():
    """Third-party bars (cellpose, stardist, torch) drew unicode blocks."""
    from tqdm import tqdm

    with tqdm(range(10), file=io.StringIO()) as bar:
        bar.update(3)
        assert bar.ascii == " ="
        assert bar.dynamic_ncols
        assert "=" in str(bar)


def test_explicit_bar_style_is_not_overridden():
    from tqdm import tqdm

    with tqdm(range(10), file=io.StringIO(), ascii=True) as bar:
        assert bar.ascii is True


def test_resize_repaints_a_live_bar():
    from tqdm import tqdm

    buf = io.StringIO()
    with tqdm(range(10), file=buf) as bar:
        bar.update(3)
        buf.seek(0)
        buf.truncate(0)
        _winch_handler()(signal.SIGWINCH, None)
        painted = buf.getvalue()

    # Erase-to-end-of-line, not tqdm's clear(), which pads to the OLD width.
    assert "\x1b[K" in painted
    assert "30%" in painted


def test_resize_drops_a_stale_terminal_size(monkeypatch):
    """tqdm falls back to COLUMNS/LINES when the ioctl fails."""
    monkeypatch.setenv("COLUMNS", "999")
    monkeypatch.setenv("LINES", "999")

    _winch_handler()(signal.SIGWINCH, None)

    assert "COLUMNS" not in os.environ
    assert "LINES" not in os.environ


def test_a_disabled_bar_does_not_cost_the_others_their_repaint():
    from tqdm import tqdm

    buf = io.StringIO()
    with tqdm(range(10), file=buf, disable=True), tqdm(range(10), file=buf) as live:
        live.update(5)
        buf.seek(0)
        buf.truncate(0)
        _winch_handler()(signal.SIGWINCH, None)
        painted = buf.getvalue()

    assert "50%" in painted
