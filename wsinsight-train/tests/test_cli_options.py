"""Every RunConfig tunable must be reachable from the command line."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from wsitrain import dag
from wsitrain.cli import _labels, main
from wsitrain.config import RunConfig

# Supplied positionally on every run, not as tunables.
IO_FIELDS = {"input", "tissue", "output"}


@pytest.fixture
def captured(tmp_path, monkeypatch):
    """Run the CLI with the DAG stubbed out and return the resolved RunConfig."""
    seen = {}

    def fake_run(cfg, **kw):
        seen["cfg"] = cfg
        seen.update(kw)
        return 0

    monkeypatch.setattr(dag, "run", fake_run)
    outs = tmp_path / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")

    def _run(*extra):
        main(["run", "--input", str(tmp_path), "--tissue", "breast", *extra])
        return seen["cfg"]
    return _run


def test_config_option_is_gone(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--config", "x.yaml"])


def test_every_tunable_has_a_flag(captured):
    """Guard against a field being added to RunConfig without a CLI flag."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
        main(["run", "--help"])
    help_text = buf.getvalue()

    missing = [f.name for f in fields(RunConfig)
               if f.name not in IO_FIELDS
               and f"--{f.name.replace('_', '-')}" not in help_text]
    assert missing == []


# --------------------------------------------------------------------------
# individual flags reach the config
# --------------------------------------------------------------------------

@pytest.mark.parametrize("flag,value,attr,expected", [
    ("--task", "pannuke", "task", "pannuke"),
    ("--top-k-markers", "13", "top_k_markers", 13),
    ("--cellpose-model", "cyto3", "cellpose_model", "cyto3"),
    ("--diameter", "12.5", "diameter", 12.5),
    ("--mpp", "0.5", "mpp", 0.5),
    ("--tile-px", "512", "tile_px", 512),
    ("--min-cells", "9", "min_cells", 9),
    ("--bg-thresh", "200", "bg_thresh", 200.0),
    ("--overlap", "0.5", "overlap", 0.5),
    ("--val-frac", "0.3", "val_frac", 0.3),
    ("--seed", "7", "seed", 7),
    ("--weight-cap", "5", "weight_cap", 5.0),
    ("--backbone", "256-x40", "backbone", "256-x40"),
    ("--fold", "fold_2", "fold", "fold_2"),
    ("--match-radius-px", "8", "match_radius_px", 8),
    ("--min-match-rate", "0.6", "min_match_rate", 0.6),
])
def test_flag_reaches_config(captured, flag, value, attr, expected):
    assert getattr(captured(flag, value), attr) == expected


def test_markers_csv_becomes_a_path(captured):
    assert captured("--markers-csv", "/tmp/m.csv").markers_csv == Path("/tmp/m.csv")


def test_defaults_apply_without_flags(captured):
    cfg = captured()
    assert cfg.segmenter == "stardist" and cfg.tile_px == 1024 and cfg.by_slide is False


# --------------------------------------------------------------------------
# boolean pair
# --------------------------------------------------------------------------

def test_by_slide_enables(captured):
    assert captured("--by-slide").by_slide is True


def test_by_tile_disables(captured):
    assert captured("--by-tile").by_slide is False


def test_by_slide_absent_uses_default(captured):
    assert captured().by_slide is False


def test_split_modes_are_mutually_exclusive(captured):
    with pytest.raises(SystemExit):
        captured("--by-tile", "--by-slide")


def test_no_by_slide_is_gone(captured):
    with pytest.raises(SystemExit):
        captured("--no-by-slide")


# --------------------------------------------------------------------------
# drop_labels list handling
# --------------------------------------------------------------------------

def test_drop_labels_space_separated(captured):
    assert captured("--drop-labels", "background", "filtered").drop_labels == (
        "background", "filtered")


def test_drop_labels_comma_separated(captured):
    assert captured("--drop-labels", "background,filtered").drop_labels == (
        "background", "filtered")


def test_drop_labels_mixed_forms(captured):
    assert captured("--drop-labels", "a,b", "c").drop_labels == ("a", "b", "c")


def test_drop_labels_preserves_spaces_in_names(captured):
    assert captured("--drop-labels", "T cell", "B cell").drop_labels == (
        "T cell", "B cell")


def test_drop_labels_absent_is_empty(captured):
    assert captured().drop_labels == ()


@pytest.mark.parametrize("raw,expected", [
    (None, None),
    (["a"], ("a",)),
    (["a,b"], ("a", "b")),
    (["a, b ", "c"], ("a", "b", "c")),
    ([",,"], ()),
])
def test_label_parsing(raw, expected):
    assert _labels(raw) == expected


# --------------------------------------------------------------------------
# stage control still works
# --------------------------------------------------------------------------

def _skip_seen(monkeypatch, tmp_path, *extra):
    seen = {}
    monkeypatch.setattr(dag, "run", lambda cfg, **kw: seen.update(kw) or 0)
    main(["run", "--input", str(tmp_path), "--tissue", "breast", *extra])
    return seen


def test_run_skip_is_forwarded(captured, tmp_path, monkeypatch):
    seen = _skip_seen(monkeypatch, tmp_path, "--run-skip", "train", "validate", "--force")
    assert seen["skip"] == ["train", "validate"]
    assert seen["force"] is True


def test_run_skip_accepts_commas(captured, tmp_path, monkeypatch):
    seen = _skip_seen(monkeypatch, tmp_path, "--run-skip", "train,validate")
    assert seen["skip"] == ["train", "validate"]


def test_run_skip_can_be_repeated(captured, tmp_path, monkeypatch):
    seen = _skip_seen(monkeypatch, tmp_path, "--run-skip", "train", "--run-skip", "export")
    assert seen["skip"] == ["train", "export"]


def test_nothing_skipped_by_default(captured, tmp_path, monkeypatch):
    assert _skip_seen(monkeypatch, tmp_path)["skip"] == []


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--run-skip", "bogus"])


def test_stage_only_and_stage_skip_are_gone(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--stage-only", "tile"])
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--stage-skip", "train"])


def test_from_and_to_are_gone(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--from", "tile"])
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--to", "tile"])


def test_bare_skip_is_gone(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--skip", "train"])
