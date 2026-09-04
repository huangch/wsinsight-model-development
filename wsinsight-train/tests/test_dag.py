"""DAG orchestration, stage gating and CellViT run resolution."""
from __future__ import annotations

import pytest

from wsitrain import dag, stages
from wsitrain.config import build_config


@pytest.fixture
def runnable(tmp_path):
    """An input tree with one discoverable sample, plus a config pointing at it."""
    outs = tmp_path / "input" / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    return build_config(tmp_path / "input", "breast", tmp_path / "out")


# --------------------------------------------------------------------------
# stage selection
# --------------------------------------------------------------------------

def test_all_stages_run_by_default(runnable, monkeypatch):
    ran = []
    for stage in dag.STAGE_FUNCS:
        monkeypatch.setitem(dag.STAGE_FUNCS, stage,
                            lambda c, s, o, _n=stage: ran.append(_n) or _nonempty(_n))
    dag.run(runnable)
    assert ran == list(dag.STAGES)


def _nonempty(stage):
    return {"nuclei_per_sample": {"a": 1}, "cells_per_sample": {"a": 1},
            "tiles": 1, "cells": 1}


def _only(*stages):
    """Skip list that leaves only ``stages`` running."""
    return [s for s in dag.STAGES if s not in stages]


def test_skipped_stage_does_not_run(runnable, monkeypatch):
    ran = []
    for stage in dag.STAGE_FUNCS:
        monkeypatch.setitem(dag.STAGE_FUNCS, stage,
                            lambda c, s, o, _n=stage: ran.append(_n) or _nonempty(_n))
    dag.run(runnable, skip=["train", "validate"])
    assert "train" not in ran and "validate" not in ran
    assert "tile" in ran and "export" in ran


def test_stage_order_is_preserved(runnable, monkeypatch):
    ran = []
    for stage in dag.STAGE_FUNCS:
        monkeypatch.setitem(dag.STAGE_FUNCS, stage,
                            lambda c, s, o, _n=stage: ran.append(_n) or _nonempty(_n))
    dag.run(runnable, skip=["segment"])
    assert ran == [s for s in dag.STAGES if s != "segment"]


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def test_run_executes_requested_stages(runnable, monkeypatch):
    seen = []
    for stage in dag.STAGE_FUNCS:
        monkeypatch.setitem(dag.STAGE_FUNCS, stage,
                            lambda c, s, o, _n=stage: seen.append(_n) or _nonempty(_n))
    assert dag.run(runnable, skip=[s for s in dag.STAGES if s != "split"]) == 0
    assert seen == ["split"]


def test_run_skips_listed_stages(runnable, monkeypatch):
    seen = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: seen.append("split"))
    dag.run(runnable, skip=list(dag.STAGES))
    assert seen == []


def test_run_writes_resolved_config(runnable, monkeypatch):
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: {})
    dag.run(runnable, skip=_only("split"))
    assert (runnable.output / "run-breast.yaml").exists()


def test_manifest_filename_is_scoped_to_tissue(runnable, monkeypatch):
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: {})
    dag.run(runnable, skip=_only("split"))
    assert (runnable.output / "manifest-breast.json").exists()


def _two_tissue_input(tmp_path):
    for tissue in ("breast", "lung"):
        outs = tmp_path / "input" / tissue / "s1" / "outs"
        outs.mkdir(parents=True)
        (outs / "cells.parquet").write_bytes(b"")
        (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    return tmp_path / "input"


def test_tissues_do_not_share_a_manifest(tmp_path, monkeypatch):
    """A second tissue in the same --output must not inherit 'done' stages."""
    src = _two_tissue_input(tmp_path)
    out = tmp_path / "out"
    ran = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "split",
                        lambda cfg, s, o: ran.append(cfg.tissue))

    for tissue in ("breast", "lung"):
        dag.run(build_config(src, tissue, out), skip=_only("split"))

    assert ran == ["breast", "lung"]


def test_comma_tissue_scope_gets_its_own_manifest(tmp_path, monkeypatch):
    src = _two_tissue_input(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: {})
    dag.run(build_config(src, "breast,lung", out), skip=_only("split"))

    assert (out / "manifest-breast,lung.json").exists()


def test_completed_stage_is_not_rerun(runnable, monkeypatch):
    calls = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: calls.append(1))
    dag.run(runnable, skip=_only("split"))
    dag.run(runnable, skip=_only("split"))
    assert len(calls) == 1


def test_force_reruns_completed_stage(runnable, monkeypatch):
    calls = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "split", lambda c, s, o: calls.append(1))
    dag.run(runnable, skip=_only("split"))
    dag.run(runnable, skip=_only("split"), force=True)
    assert len(calls) == 2


def test_empty_required_output_aborts(runnable, monkeypatch):
    monkeypatch.setitem(dag.STAGE_FUNCS, "tile", lambda c, s, o: {"tiles": 0})
    with pytest.raises(SystemExit, match="produced no tiles"):
        dag.run(runnable, skip=_only("tile"))


def test_empty_required_output_is_not_marked_done(runnable, monkeypatch):
    monkeypatch.setitem(dag.STAGE_FUNCS, "tile", lambda c, s, o: {"tiles": 0})
    with pytest.raises(SystemExit):
        dag.run(runnable, skip=_only("tile"))

    calls = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "tile",
                        lambda c, s, o: (calls.append(1), {"tiles": 3})[1])
    dag.run(runnable, skip=_only("tile"))
    assert calls == [1]


def test_not_implemented_stage_stops_cleanly(runnable, monkeypatch):
    def boom(c, s, o):
        raise NotImplementedError("todo")

    monkeypatch.setitem(dag.STAGE_FUNCS, "split", boom)
    assert dag.run(runnable, skip=_only("split")) == 0


def test_missing_samples_aborts_data_stages(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = build_config(empty, "breast", tmp_path / "out")
    with pytest.raises(SystemExit, match="no samples found"):
        dag.run(cfg, skip=_only("segment"))


def test_unaligned_samples_are_filtered(tmp_path, monkeypatch):
    outs = tmp_path / "input" / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_unaligned_image.ome.tif").write_bytes(b"")
    cfg = build_config(tmp_path / "input", "breast", tmp_path / "out")

    with pytest.raises(SystemExit, match="no samples found"):
        dag.run(cfg, skip=_only("segment"))


def test_transform_none_keeps_unaligned_samples(tmp_path, monkeypatch):
    outs = tmp_path / "input" / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_unaligned_image.ome.tif").write_bytes(b"")
    cfg = build_config(tmp_path / "input", "breast", tmp_path / "out",
                       overrides={"transform": "none"})

    got = []
    monkeypatch.setitem(dag.STAGE_FUNCS, "segment",
                        lambda c, s, o: (got.extend(s), {"nuclei_per_sample": {"a": 1}})[1])
    dag.run(cfg, skip=_only("segment"))
    assert len(got) == 1


# --------------------------------------------------------------------------
# _find_run_dir
# --------------------------------------------------------------------------

def _checkpoint(root, name):
    d = root / name / "checkpoints"
    d.mkdir(parents=True)
    (d / "model_best.pth").write_text("x")
    return d.parent


def test_run_dir_prefers_per_tissue_logs(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = build_config(tmp_path / "in", "breast", out)
    mine = _checkpoint(out / "logs" / "breast", "run1")
    _checkpoint(tmp_path / "cv" / "logs_local", "lung-other-run")
    monkeypatch.setenv("CELLVIT_ROOT", str(tmp_path / "cv"))

    assert stages._find_run_dir(cfg, out) == mine


def test_run_dir_falls_back_to_tagged_legacy_run(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = build_config(tmp_path / "in", "breast", out)
    tag = f"{cfg.tissue}-{cfg.task}-{cfg.backbone.lower()}"
    legacy = _checkpoint(tmp_path / "cv" / "logs_local", f"2024_{tag}")
    monkeypatch.setenv("CELLVIT_ROOT", str(tmp_path / "cv"))

    assert stages._find_run_dir(cfg, out) == legacy


def test_run_dir_ignores_untagged_legacy_run(tmp_path, monkeypatch):
    out = tmp_path / "out"
    cfg = build_config(tmp_path / "in", "breast", out)
    _checkpoint(tmp_path / "cv" / "logs_local", "lung-sthelar_full-sam-h-x40")
    monkeypatch.setenv("CELLVIT_ROOT", str(tmp_path / "cv"))

    assert stages._find_run_dir(cfg, out, required=False) is None


def test_run_dir_raises_when_required(tmp_path, monkeypatch):
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    cfg = build_config(tmp_path / "in", "breast", tmp_path / "out")
    with pytest.raises(RuntimeError, match="no trained run"):
        stages._find_run_dir(cfg, tmp_path / "out")


def test_run_dir_picks_newest_within_tissue(tmp_path, monkeypatch):
    import os
    import time

    out = tmp_path / "out"
    cfg = build_config(tmp_path / "in", "breast", out)
    old = _checkpoint(out / "logs" / "breast", "old")
    new = _checkpoint(out / "logs" / "breast", "new")
    past = time.time() - 500
    os.utime(old / "checkpoints" / "model_best.pth", (past, past))

    assert stages._find_run_dir(cfg, out) == new
