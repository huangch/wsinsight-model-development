"""Stages that shell out: annotate (kurtorank) and train/validate/export (CellViT)."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from wsitrain import paths
from wsitrain.dataset import Sample
from wsitrain.stages import annotate, export, train, validate


@pytest.fixture
def recorder(monkeypatch):
    """Capture subprocess.run invocations instead of executing them."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append({"cmd": [str(c) for c in cmd], **kw})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def kurtorank_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/kurtorank" if name == "kurtorank" else "/usr/bin/python3")


def _sample(tmp_path, tissue="breast", name="s1") -> Sample:
    outs = tmp_path / "input" / tissue / name / "outs"
    outs.mkdir(parents=True, exist_ok=True)
    he = outs.parent / f"{name}_he_image.ome.tif"
    he.write_bytes(b"")
    return Sample(f"{tissue}__{name}", tissue, outs, he, True)


# --------------------------------------------------------------------------
# annotate
# --------------------------------------------------------------------------

def test_annotate_requires_kurtorank(tmp_path, cfg_factory, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="kurtorank not on PATH"):
        annotate(cfg_factory(), [_sample(tmp_path)], tmp_path / "out")


def test_annotate_requires_samples(cfg_factory, kurtorank_on_path, tmp_path):
    with pytest.raises(RuntimeError, match="no samples discovered"):
        annotate(cfg_factory(), [], tmp_path / "out")


def test_annotate_skips_existing_assignment(tmp_path, cfg_factory, kurtorank_on_path, recorder):
    cfg = cfg_factory()
    s = _sample(tmp_path)
    (s.outs / f"celltype_assignment_{cfg.task}_label.csv").write_text("classification,cell_type\n")

    info = annotate(cfg, [s], tmp_path / "out")

    assert recorder == []
    assert info["n_samples"] == 1


def test_annotate_builds_expected_command(tmp_path, cfg_factory, kurtorank_on_path, recorder):
    cfg = cfg_factory(top_k_markers=13)
    s = _sample(tmp_path)

    def fake_run(cmd, **kw):
        recorder.append({"cmd": [str(c) for c in cmd]})
        (s.outs / f"celltype_assignment_{cfg.task}_label.csv").write_text("x\n")
        return subprocess.CompletedProcess(cmd, 0)

    subprocess.run = fake_run
    annotate(cfg, [s], tmp_path / "out")

    cmd = recorder[0]["cmd"]
    assert cmd[1] == "annotate"
    assert "--xenium-dir" in cmd and str(s.outs) in cmd
    assert cmd[cmd.index("--tissue-type") + 1] == "breast"
    assert cmd[cmd.index("--use-top-k-markers") + 1] == "13"


def test_annotate_passes_markers_csv(tmp_path, cfg_factory, kurtorank_on_path, recorder):
    markers = tmp_path / "m.csv"
    markers.write_text("gene\n")
    cfg = cfg_factory(markers_csv=markers)
    s = _sample(tmp_path)

    def fake_run(cmd, **kw):
        recorder.append({"cmd": [str(c) for c in cmd]})
        (s.outs / f"celltype_assignment_{cfg.task}_label.csv").write_text("x\n")
        return subprocess.CompletedProcess(cmd, 0)

    subprocess.run = fake_run
    annotate(cfg, [s], tmp_path / "out")

    assert str(markers) in recorder[0]["cmd"]


def test_annotate_fails_when_no_csv_produced(tmp_path, cfg_factory, kurtorank_on_path, recorder):
    """kurtorank exiting 0 without writing the CSV must not be reported as success."""
    with pytest.raises(RuntimeError, match="did not write"):
        annotate(cfg_factory(), [_sample(tmp_path)], tmp_path / "out")


def test_annotate_error_names_the_tissue(tmp_path, cfg_factory, kurtorank_on_path, recorder):
    with pytest.raises(RuntimeError, match="'breast'"):
        annotate(cfg_factory(), [_sample(tmp_path)], tmp_path / "out")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def test_train_requires_cellvit_root(tmp_path, cfg_factory, monkeypatch):
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="CELLVIT_ROOT"):
        train(cfg_factory(), [], tmp_path / "out")


def test_train_requires_rendered_config(tmp_path, cfg_factory, monkeypatch):
    root = tmp_path / "cv"
    root.mkdir()
    monkeypatch.setenv("CELLVIT_ROOT", str(root))
    with pytest.raises(RuntimeError, match="missing train config"):
        train(cfg_factory(), [], tmp_path / "out")


def test_train_invokes_trainer_with_config(tmp_path, cfg_factory, monkeypatch, recorder):
    root = tmp_path / "cv"
    root.mkdir()
    monkeypatch.setenv("CELLVIT_ROOT", str(root))
    cfg = cfg_factory()
    cfg_path = paths.tissue_root(cfg.output, cfg.tissue) / "train_configs" / cfg.backbone / f"{cfg.fold}.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("gpu: 0\n")

    train(cfg, [], cfg.output)

    call = recorder[0]
    assert str(cfg_path) in call["cmd"]
    assert call["cmd"][1].endswith("train_cell_classifier_head.py")
    assert call["env"]["PYTHONPATH"] == str(root)


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

@pytest.fixture
def exportable(tmp_path, cfg_factory, monkeypatch):
    root = tmp_path / "cv"
    root.mkdir()
    monkeypatch.setenv("CELLVIT_ROOT", str(root))
    cfg = cfg_factory()
    ckpt = paths.logs_dir(cfg.output, cfg.tissue) / "run" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "model_best.pth").write_text("weights")
    paths.tissue_root(cfg.output, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "immune"\n1: "tumor"\n')
    return cfg


def test_export_writes_config_json(exportable, recorder):
    cfg = exportable
    export(cfg, [], cfg.output)

    doc = json.loads((paths.models_dir(cfg.output, cfg.tissue) / "main" / "config.json").read_text())
    assert doc["num_classes"] == 2
    assert doc["class_names"] == ["immune", "tumor"]
    assert doc["architecture"] == "cellvit"


def test_export_records_geometry_from_config(exportable, recorder):
    cfg = exportable
    cfg.tile_px = 512
    cfg.mpp = 0.5
    export(cfg, [], cfg.output)

    doc = json.loads((paths.models_dir(cfg.output, cfg.tissue) / "main" / "config.json").read_text())
    assert doc["patch_size_pixels"] == 512
    assert doc["spacing_um_px"] == 0.5


def test_export_copies_label_map(exportable, recorder):
    cfg = exportable
    export(cfg, [], cfg.output)
    dst = paths.models_dir(cfg.output, cfg.tissue) / "main" / "label_map.yaml"
    assert dst.read_text() == '0: "immune"\n1: "tumor"\n'


def test_export_converts_the_selected_checkpoint(exportable, recorder):
    cfg = exportable
    export(cfg, [], cfg.output)

    cmd = recorder[0]["cmd"]
    assert cmd[1].endswith("cellvit_convert_to_torchscript.py")
    assert cmd[cmd.index("--checkpoint") + 1].endswith("model_best.pth")
    assert cmd[cmd.index("--height") + 1] == str(cfg.tile_px)


def test_export_returns_class_names(exportable, recorder):
    cfg = exportable
    assert export(cfg, [], cfg.output)["classes"] == ["immune", "tumor"]


def test_export_requires_a_trained_run(tmp_path, cfg_factory, monkeypatch, recorder):
    root = tmp_path / "cv"
    root.mkdir()
    monkeypatch.setenv("CELLVIT_ROOT", str(root))
    cfg = cfg_factory()
    paths.tissue_root(cfg.output, cfg.tissue).mkdir(parents=True)
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "a"\n')

    with pytest.raises(RuntimeError, match="no trained run"):
        export(cfg, [], cfg.output)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def test_validate_requires_a_run(tmp_path, cfg_factory, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="no trained run"):
        validate(cfg_factory(), [], tmp_path / "out")


def test_validate_surfaces_scores(tmp_path, cfg_factory, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    cfg = cfg_factory()
    run = paths.logs_dir(cfg.output, cfg.tissue) / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "model_best.pth").write_text("w")
    (run / "val_results").mkdir()
    (run / "val_results" / "scores.json").write_text('{"F1-Score/Validation": 0.5}')

    info = validate(cfg, [], cfg.output)

    assert info["metrics"]["F1-Score/Validation"] == 0.5
    assert (paths.report_dir(cfg.output, cfg.tissue) / "scores.json").exists()


def test_validate_tolerates_missing_val_results(tmp_path, cfg_factory, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    cfg = cfg_factory()
    run = paths.logs_dir(cfg.output, cfg.tissue) / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "model_best.pth").write_text("w")

    assert validate(cfg, [], cfg.output)["metrics"] == {}


def test_validate_plots_confusion_matrix(tmp_path, cfg_factory, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("CELLVIT_ROOT", raising=False)
    cfg = cfg_factory()
    run = paths.logs_dir(cfg.output, cfg.tissue) / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "model_best.pth").write_text("w")
    vr = run / "val_results"
    vr.mkdir()
    torch.save(torch.tensor([0, 1, 1]), vr / "predictions.pt")
    torch.save(torch.tensor([0, 1, 0]), vr / "gt.pt")
    paths.tissue_root(cfg.output, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "a"\n1: "b"\n')

    validate(cfg, [], cfg.output)

    assert (paths.report_dir(cfg.output, cfg.tissue) / "confusion_matrix.png").exists()
