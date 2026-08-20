"""Auto-tune levers and the outer loop (previously untested end to end)."""
from __future__ import annotations

import json
import subprocess

import pytest

from wsitrain import paths, tuning
from wsitrain.tuning import (DROP_STEP, MIN_IMPROVEMENT, WEIGHT_BOOST,
                             apply_lever, read_macro_f1, run_tune, weakest_class)


# --------------------------------------------------------------------------
# apply_lever
# --------------------------------------------------------------------------

def test_weight_lever_boosts_the_weak_class():
    w, d, lr = apply_lever("weight", [1.0, 2.0], 0.1, 1e-4, weak=1)
    assert w == [1.0, 2.0 * WEIGHT_BOOST] and d == 0.1 and lr == 1e-4


def test_weight_lever_does_nothing_without_a_weak_class():
    before = [1.0, 2.0]
    w, d, lr = apply_lever("weight", before, 0.1, 1e-4, weak=None)
    assert w == before and d == 0.1 and lr == 1e-4


def test_weight_lever_does_not_mutate_the_caller_list():
    before = [1.0, 2.0]
    apply_lever("weight", before, 0.1, 1e-4, weak=0)
    assert before == [1.0, 2.0]


def test_drop_lever_steps_up():
    _, d, _ = apply_lever("drop", [1.0], 0.1, 1e-4, weak=None)
    assert d == pytest.approx(0.1 + DROP_STEP)


def test_drop_lever_is_capped():
    _, d, _ = apply_lever("drop", [1.0], 0.5, 1e-4, weak=None)
    assert d == 0.5


def test_lr_lever_halves():
    _, _, lr = apply_lever("lr", [1.0], 0.1, 1e-4, weak=None)
    assert lr == pytest.approx(5e-5)


def test_unknown_lever_is_inert():
    assert apply_lever("bogus", [1.0], 0.1, 1e-4, weak=0) == ([1.0], 0.1, 1e-4)


# --------------------------------------------------------------------------
# read_macro_f1 / weakest_class
# --------------------------------------------------------------------------

def _run_with_scores(tmp_path, value):
    vr = tmp_path / "run" / "val_results"
    vr.mkdir(parents=True)
    (vr / "scores.json").write_text(json.dumps({"F1-Score/Validation": value}))
    return tmp_path / "run"


def test_missing_val_results_reports_nothing(tmp_path):
    assert read_macro_f1(tmp_path) == (None, None)


def test_scores_json_is_reported_as_micro(tmp_path):
    assert read_macro_f1(_run_with_scores(tmp_path, 0.75)) == (0.75, "micro")


def test_scores_json_without_the_key(tmp_path):
    vr = tmp_path / "run" / "val_results"
    vr.mkdir(parents=True)
    (vr / "scores.json").write_text("{}")
    assert read_macro_f1(tmp_path / "run") == (None, None)


def test_tensors_are_reported_as_macro(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    vr = tmp_path / "run" / "val_results"
    vr.mkdir(parents=True)
    torch.save(torch.tensor([0, 1, 1]), vr / "predictions.pt")
    torch.save(torch.tensor([0, 1, 0]), vr / "gt.pt")
    value, source = read_macro_f1(tmp_path / "run")
    assert source == "macro" and 0.0 <= value <= 1.0


def test_weakest_class_is_none_without_tensors(tmp_path):
    assert weakest_class(_run_with_scores(tmp_path, 0.5)) is None


def test_weakest_class_picks_the_worst(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(tuning, "_per_class_f1",
                        lambda run_dir: ([0, 2, 5], np.array([0.9, 0.1, 0.8])))
    assert weakest_class(tmp_path) == 2


def test_weakest_class_ignores_healthy_classes(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(tuning, "_per_class_f1",
                        lambda run_dir: ([0, 1], np.array([0.95, 0.9])))
    assert weakest_class(tmp_path) is None


# --------------------------------------------------------------------------
# run_tune
# --------------------------------------------------------------------------

MIN_IMPROVEMENT = MIN_IMPROVEMENT
BASELINE_F1 = 0.10


@pytest.fixture
def tune_env(tmp_path, cfg_factory, monkeypatch):
    """cfg plus a fake trainer whose reported F1 comes from a scripted list."""
    cfg = cfg_factory(tune=3, weight_cap=10.0)
    lab = paths.labels_dir(cfg.output, cfg.tissue)
    lab.mkdir(parents=True)
    (lab / "s_tile_00000.csv").write_text("0,0,0\n" * 8 + "0,0,1\n" * 2)
    sd = paths.splits_dir(cfg.output, cfg.tissue, cfg.fold)
    sd.mkdir(parents=True)
    (sd / "train.csv").write_text("s_tile_00000\n")
    (sd / "val.csv").write_text("s_tile_00000\n")
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "a"\n1: "b"\n')
    monkeypatch.setenv("CELLVIT_ROOT", str(tmp_path / "cv"))

    state = {"scores": [], "renders": [], "runs": 0}

    # run_tune always follows an initial train stage, so a baseline run exists.
    base = paths.logs_dir(cfg.output, cfg.tissue) / "run_base"
    (base / "checkpoints").mkdir(parents=True)
    (base / "checkpoints" / "model_best.pth").write_text("base")
    (base / "val_results").mkdir()
    (base / "val_results" / "scores.json").write_text(
        json.dumps({"F1-Score/Validation": BASELINE_F1}))

    from wsitrain import configrender as cr
    original_render = cr.render_config

    def fake_render(c, o, *, drop_rate, lr, weights):
        state["renders"].append({"drop": drop_rate, "lr": lr, "weights": list(weights)})
        return original_render(c, o, drop_rate=drop_rate, lr=lr, weights=weights)

    def fake_train(cmd, **kw):
        i = state["runs"]
        state["runs"] += 1
        score = state["scores"][i] if i < len(state["scores"]) else None
        run = paths.logs_dir(cfg.output, cfg.tissue) / f"run{i}"
        (run / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run / "checkpoints" / "model_best.pth").write_text(f"w{i}")
        if score is not None:
            vr = run / "val_results"
            vr.mkdir(parents=True, exist_ok=True)
            (vr / "scores.json").write_text(
                json.dumps({"F1-Score/Validation": score}))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cr, "render_config", fake_render)
    monkeypatch.setattr(subprocess, "run", fake_train)
    # Without a weak class the weight lever is inert; pin one so levers fire.
    monkeypatch.setattr(tuning, "weakest_class", lambda run_dir: 0)
    return cfg, state


def _tune(cfg, scores, state):
    state["scores"] = scores
    return run_tune(cfg, cfg.output, str(cfg.output / "cv"),
                    base_config=None, py="python")


def test_improving_runs_are_accepted(tune_env):
    cfg, state = tune_env
    info = _tune(cfg, [0.50, 0.60, 0.70], state)
    assert info["best_f1"] == pytest.approx(0.70)


def test_stops_after_two_consecutive_rejections(tune_env):
    cfg, state = tune_env
    cfg.tune = 6
    _tune(cfg, [0.80, 0.10, 0.10, 0.10], state)
    assert state["runs"] == 3


def test_rejected_lever_is_not_compounded(tune_env):
    """A rejected lever must not leave its change in place for the next one."""
    cfg, state = tune_env
    cfg.tune = 3
    _tune(cfg, [0.80, 0.10, 0.10], state)
    drops = [r["drop"] for r in state["renders"]]
    # Only one drop lever fired, from the original 0.1 -> 0.15; a compounded
    # value (0.2) would mean the rejected iteration was carried forward.
    assert max(drops) == pytest.approx(0.15)


def test_rejected_weight_boost_is_discarded(tune_env):
    cfg, state = tune_env
    cfg.tune = 3
    _tune(cfg, [0.80, 0.10, 0.10], state)
    first = state["renders"][0]["weights"][0]
    # Iteration 2's boost was rejected, so iteration 3 starts from iteration 1's
    # accepted weights rather than the rejected ones.
    assert state["renders"][2]["weights"][0] == pytest.approx(first)


def test_unmeasurable_run_is_never_accepted(tune_env):
    cfg, state = tune_env
    cfg.tune = 2
    info = _tune(cfg, [], state)          # trainer writes no scores.json
    assert info["best_f1"] == pytest.approx(BASELINE_F1)
    assert all(not e["accepted"] for e in _log(cfg))


def test_metric_sources_are_not_mixed(tune_env, monkeypatch):
    """A macro score must not be judged an improvement over a micro baseline."""
    cfg, state = tune_env
    cfg.tune = 1
    calls = {"n": 0}

    def fake_read(run_dir):
        calls["n"] += 1
        return (BASELINE_F1, "micro") if calls["n"] == 1 else (0.99, "macro")

    monkeypatch.setattr(tuning, "read_macro_f1", fake_read)
    info = _tune(cfg, [0.99], state)
    assert info["best_f1"] == pytest.approx(BASELINE_F1)


def test_best_checkpoint_becomes_the_newest(tune_env):
    """export() takes the newest checkpoint, so the best run must be touched."""
    import pathlib

    from wsitrain.stages import _find_run_dir

    cfg, state = tune_env
    cfg.tune = 3
    info = _tune(cfg, [0.80, 0.10, 0.10], state)
    assert _find_run_dir(cfg, cfg.output) == pathlib.Path(info["best_run"])


def test_tuning_log_is_one_json_object_per_line(tune_env):
    cfg, state = tune_env
    _tune(cfg, [0.50, 0.60, 0.70], state)
    for entry in _log(cfg):
        assert {"iter", "lever", "accepted"} <= set(entry)


def _log(cfg):
    text = (paths.report_dir(cfg.output, cfg.tissue) / "tuning_log.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_no_op_lever_does_not_retrain(tune_env, monkeypatch):
    """With no weak class the weight lever changes nothing; skip the training run."""
    cfg, state = tune_env
    cfg.tune = 1
    monkeypatch.setattr(tuning, "weakest_class", lambda run_dir: None)
    _tune(cfg, [0.5], state)
    entries = _log(cfg)
    assert entries[0]["lever"] == "weight"
    assert entries[0].get("note") == "lever had no effect"
    assert state["runs"] == 0
