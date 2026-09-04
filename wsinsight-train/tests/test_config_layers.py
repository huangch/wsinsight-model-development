"""Config layering: provenance, validation, and the --config file.

Layers, lowest first: shipped defaults < saved run-<tissue>.yaml < --config
file < CLI flags. --config patches the saved layer rather than replacing it.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest
import yaml

from wsitrain import STAGES, dag, paths
from wsitrain.cli import _fields_for, main
from wsitrain.config import RunConfig, load_defaults, resolve_config


@pytest.fixture
def cohort(tmp_path):
    outs = tmp_path / "cohort" / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    return tmp_path / "cohort"


@pytest.fixture
def seen_cfg(monkeypatch):
    got = {}
    monkeypatch.setattr(dag, "run", lambda cfg, **kw: got.update(cfg=cfg) or 0)
    return got


@pytest.fixture
def saved(tmp_path):
    """Write a saved run-<tissue>.yaml, as dag.run would."""
    def _write(**values):
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)
        paths.resolved_config_path(out, "breast").write_text(yaml.safe_dump(values))
        return out
    return _write


def _run(cohort, out, *extra, tissue="breast"):
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["segment", "--input", str(cohort), "--tissue", tissue,
              "--output", str(out), *extra])
    return buf.getvalue()


# --------------------------------------------------------------------------
# which settings does each command own
# --------------------------------------------------------------------------

def test_every_tunable_belongs_to_some_command():
    """A field no command exposes is unreachable; a second hand-written list
    would be the thing that drifts, so this is derived from the parsers."""
    owned = set().union(*(_fields_for(s) for s in STAGES), _fields_for("run"))
    tunable = set(RunConfig.__dataclass_fields__) - {"input", "tissue", "output"}
    assert owned == tunable


def test_run_owns_every_stage_field():
    for stage in STAGES:
        assert _fields_for(stage) <= _fields_for("run")


def test_stage_owns_only_its_own_fields():
    assert "val_frac" not in _fields_for("segment")
    assert "segmenter" in _fields_for("segment")
    assert "val_frac" in _fields_for("split")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_each_layer_is_labelled(tmp_path):
    cfg, source = resolve_config(
        tmp_path, "breast", tmp_path / "o",
        base={"mpp": 0.5}, config={"tile_px": 512}, overrides={"seed": 7})

    assert source["mpp"] == "saved"
    assert source["tile_px"] == "config"
    assert source["seed"] == "flag"
    assert source["bg_thresh"] == "default"
    assert (cfg.mpp, cfg.tile_px, cfg.seed) == (0.5, 512, 7)


def test_higher_layers_win(tmp_path):
    cfg, source = resolve_config(
        tmp_path, "breast", tmp_path / "o",
        base={"mpp": 0.5}, config={"mpp": 0.75}, overrides={"mpp": 1.0})
    assert cfg.mpp == 1.0 and source["mpp"] == "flag"

    cfg, source = resolve_config(
        tmp_path, "breast", tmp_path / "o", base={"mpp": 0.5}, config={"mpp": 0.75})
    assert cfg.mpp == 0.75 and source["mpp"] == "config"


def test_provenance_covers_every_field(tmp_path):
    cfg, source = resolve_config(tmp_path, "breast", tmp_path / "o")
    assert set(source) == set(load_defaults())


def test_printout_names_the_source(cohort, saved, seen_cfg):
    out = saved(mpp=0.5)
    text = _run(cohort, out, "--segmenter", "cellpose")

    assert "mpp" in text and "run-breast.yaml" in text
    assert "--segmenter" in text


def test_printout_lists_settings_other_commands_own(cohort, saved, seen_cfg):
    out = saved(val_frac=0.33)
    text = _run(cohort, out)
    assert "carried for later stages" in text
    assert "val_frac" in text.split("carried for later stages")[1]


def test_show_config_lists_defaults_too(cohort, saved, seen_cfg):
    out = saved()
    assert "mpp" in _run(cohort, out, "--show-config")


# --------------------------------------------------------------------------
# --config: stacking, not replacing
# --------------------------------------------------------------------------

def test_config_patches_the_saved_layer(tmp_path, cohort, saved, seen_cfg):
    """Replacing would revert mpp to the shipped default and invalidate the
    segment stage that produced the masks."""
    out = saved(mpp=0.5, segmenter="cellpose")
    f = tmp_path / "tweak.yaml"
    f.write_text("tile_px: 512\n")

    _run(cohort, out, "--config", str(f))

    cfg = seen_cfg["cfg"]
    assert cfg.tile_px == 512      # from the file
    assert cfg.mpp == 0.5          # still from the saved record
    assert cfg.segmenter == "cellpose"


def test_reset_config_with_config_drops_the_saved_layer(tmp_path, cohort, saved,
                                                        seen_cfg):
    out = saved(mpp=0.5, segmenter="cellpose")
    f = tmp_path / "exact.yaml"
    f.write_text("tile_px: 512\n")

    _run(cohort, out, "--config", str(f), "--reset-config")

    cfg = seen_cfg["cfg"]
    assert cfg.tile_px == 512
    assert cfg.mpp == 0.25         # shipped default, saved layer ignored
    assert cfg.segmenter == "stardist"


def test_flags_beat_the_config_file(tmp_path, cohort, saved, seen_cfg):
    out = saved()
    f = tmp_path / "c.yaml"
    f.write_text("mpp: 0.75\n")

    text = _run(cohort, out, "--config", str(f), "--mpp", "1.0")

    assert seen_cfg["cfg"].mpp == 1.0
    assert "--mpp" in text


def test_a_setting_another_command_owns_still_reaches_that_command(
        tmp_path, cohort, saved, seen_cfg):
    """The whole point of B: segment does not read val_frac, but must not drop it."""
    f = tmp_path / "c.yaml"
    f.write_text("val_frac: 0.33\n")
    out = tmp_path / "out"
    out.mkdir()

    _run(cohort, out, "--config", str(f))

    assert seen_cfg["cfg"].val_frac == 0.33


def test_a_saved_config_round_trips(tmp_path, cohort, saved, seen_cfg):
    out = saved(**{**load_defaults(), "mpp": 0.5, "backbone": "SAM-B-x40"})
    dump = tmp_path / "dump.yaml"
    dump.write_text(paths.resolved_config_path(out, "breast").read_text())

    _run(cohort, tmp_path / "fresh", "--config", str(dump), "--reset-config")

    cfg = seen_cfg["cfg"]
    assert cfg.mpp == 0.5 and cfg.backbone == "SAM-B-x40"


def test_io_keys_in_a_config_file_are_ignored(tmp_path, cohort, seen_cfg):
    f = tmp_path / "c.yaml"
    f.write_text("input: /somewhere/else\ntissue: lung\nmpp: 0.75\n")
    out = tmp_path / "out"

    _run(cohort, out, "--config", str(f))

    cfg = seen_cfg["cfg"]
    assert cfg.tissue == "breast"
    assert str(cfg.input) == str(cohort.resolve())
    assert cfg.mpp == 0.75


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_unknown_key_is_rejected_with_a_suggestion(tmp_path, cohort):
    f = tmp_path / "c.yaml"
    # Not "tile_size": that is now closer to batch_size than to tile_px.
    f.write_text("tile_pixels: 512\n")

    with pytest.raises(SystemExit, match="tile_px"):
        _run(cohort, tmp_path / "out", "--config", str(f))


def test_bad_choice_is_rejected_before_anything_runs(tmp_path, cohort):
    f = tmp_path / "c.yaml"
    f.write_text("segmenter: stardist2\n")

    with pytest.raises(SystemExit, match="stardist2"):
        _run(cohort, tmp_path / "out", "--config", str(f))


def test_missing_config_file_is_named(tmp_path, cohort):
    with pytest.raises(SystemExit, match="not found"):
        _run(cohort, tmp_path / "out", "--config", str(tmp_path / "nope.yaml"))


def test_a_config_file_that_is_not_a_mapping_is_rejected(tmp_path, cohort):
    f = tmp_path / "c.yaml"
    f.write_text("- just\n- a list\n")

    with pytest.raises(SystemExit, match="mapping"):
        _run(cohort, tmp_path / "out", "--config", str(f))


def test_a_stale_key_in_the_saved_record_is_tolerated(tmp_path, cohort, saved,
                                                      seen_cfg):
    """The saved file is machine-written and may predate a rename; only the
    hand-written --config file is held to a strict schema."""
    out = saved(mpp=0.5, retired_setting=1)

    _run(cohort, out)

    assert seen_cfg["cfg"].mpp == 0.5
