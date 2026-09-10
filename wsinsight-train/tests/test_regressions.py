"""Regressions for defects found in review: flag semantics, staleness, degenerate splits."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from wsitrain import paths, segment as segment_mod, stages
from wsitrain.configrender import _gpu_id
from wsitrain.dataset import Sample
from wsitrain.manifest import Manifest


@pytest.fixture
def gpu_probe(tmp_path, monkeypatch):
    """Run the segment stage and report the gpu flag it requested."""
    seen = {}

    class Fake:
        name = "fake"

        def segment(self, he_rgb, *, mpp):
            return np.ones(he_rgb.shape[:2], np.int32)

    def spy(name, **kw):
        seen.update(kw)
        return Fake()

    monkeypatch.setattr(segment_mod, "get_segmenter", spy)
    monkeypatch.setitem(__import__("sys").modules, "torch", None)

    he = tmp_path / "s1_he_image.ome.tif"
    tifffile.imwrite(he, np.full((8, 8, 3), 10, np.uint8))
    sample = Sample("breast__s1", "breast", tmp_path, he, True)

    def _run(cfg):
        stages.segment(cfg, [sample], cfg.output)
        return seen["gpu"]
    return _run


# --------------------------------------------------------------------------
# --gpus means the same thing in every stage
# --------------------------------------------------------------------------

def test_gpu_zero_selects_device_not_cpu(cfg_factory, gpu_probe):
    """'0' is device 0 for CellViT, so it must not disable the segmenter's GPU."""
    assert gpu_probe(cfg_factory(gpus="0", nuclei_source="he-mask")) is True


def test_gpu_index_keeps_gpu_enabled(cfg_factory, gpu_probe):
    assert gpu_probe(cfg_factory(gpus="2", nuclei_source="he-mask")) is True


def test_auto_keeps_gpu_enabled(cfg_factory, gpu_probe):
    assert gpu_probe(cfg_factory(gpus="auto", nuclei_source="he-mask")) is True


@pytest.mark.parametrize("raw", ["cpu", "none", "no", "false", ""])
def test_cpu_aliases_disable_gpu(cfg_factory, gpu_probe, raw):
    assert gpu_probe(cfg_factory(gpus=raw, nuclei_source="he-mask")) is False


def test_gpu_zero_agrees_across_stages(tmp_path, cfg_factory, gpu_probe):
    cfg = cfg_factory(gpus="0", nuclei_source="he-mask")
    assert _gpu_id(cfg) == "0"
    assert gpu_probe(cfg) is True


# --------------------------------------------------------------------------
# manifest staleness
# --------------------------------------------------------------------------

def test_mpp_change_reruns_segmentation(tmp_path):
    """mpp drives the StarDist rescale, so masks are stale when it changes."""
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"mpp": 0.25})
    for stage in ("annotate", "segment", "tile"):
        mf.mark(stage, "done")

    fresh = Manifest.load_or_new(p, {"mpp": 0.5})

    assert fresh.is_done("annotate")
    assert not fresh.is_done("segment")


def test_marker_panel_change_reruns_annotate(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"top_k_markers": 25})
    mf.mark("annotate", "done")

    assert not Manifest.load_or_new(p, {"top_k_markers": 50}).is_done("annotate")


def test_markers_csv_change_reruns_annotate(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"markers_csv": "a.csv"})
    mf.mark("annotate", "done")

    assert not Manifest.load_or_new(p, {"markers_csv": "b.csv"}).is_done("annotate")


# --------------------------------------------------------------------------
# degenerate splits
# --------------------------------------------------------------------------

def _one_tile_tree(cfg, n_tiles):
    labels = paths.labels_dir(cfg.output, cfg.tissue)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n_tiles):
        (labels / f"breast__s1_tile_{i:05d}.csv").write_text("0,0,0\n1,1,1\n")
    paths.tissue_root(cfg.output, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(cfg.output, cfg.tissue).write_text('0: "a"\n1: "b"\n')


def test_split_rejects_empty_validation_set(cfg_factory):
    cfg = cfg_factory(by_slide=False)
    _one_tile_tree(cfg, 1)
    with pytest.raises(RuntimeError, match="both sides must be non-empty"):
        stages.split(cfg, [], cfg.output)


def test_split_accepts_a_usable_set(cfg_factory, monkeypatch):
    cfg = cfg_factory(by_slide=False, val_frac=0.5)
    _one_tile_tree(cfg, 4)
    monkeypatch.setenv("CELLVIT_ROOT", str(cfg.output / "cv"))

    info = stages.split(cfg, [], cfg.output)

    assert info["n_train"] >= 1 and info["n_val"] >= 1


def test_stale_multi_gpu_config_is_refused_before_cellvit_runs(tmp_path):
    # A config rendered before the gpu fix reached CellViT as "cuda:0,1".
    cfgp = tmp_path / "fold_0.yaml"
    cfgp.write_text("seed: 42\ngpu: 0,1\nbackbone: SAM-H-x40\n")
    with pytest.raises(RuntimeError, match="--redo-split"):
        stages._check_rendered_gpu(cfgp)


def test_single_gpu_config_is_accepted(tmp_path):
    cfgp = tmp_path / "fold_0.yaml"
    cfgp.write_text("seed: 42\ngpu: 0\nbackbone: SAM-H-x40\n")
    stages._check_rendered_gpu(cfgp)


def test_wiping_train_keeps_the_config_split_owns(cfg_factory):
    # --redo-train used to delete train_configs/, leaving split marked done
    # with its output gone -> "missing train config" on the next run.
    from wsitrain import paths
    cfg = cfg_factory()
    cfgp = paths.train_config_path(cfg.output, cfg.tissue, cfg.backbone, cfg.fold)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text("gpu: 0\n")

    stages._STAGE_WIPES["train"](cfg.output, cfg)

    assert cfgp.exists()


def test_wiping_split_does_remove_its_own_config(cfg_factory):
    from wsitrain import paths
    cfg = cfg_factory()
    cfgp = paths.train_config_path(cfg.output, cfg.tissue, cfg.backbone, cfg.fold)
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    cfgp.write_text("gpu: 0\n")

    stages._STAGE_WIPES["split"](cfg.output, cfg)

    assert not cfgp.exists()


def test_no_wipe_reaches_an_upstream_stages_artefacts(cfg_factory):
    """A wipe may clear its own or a later stage's output, never an earlier one.

    Cascade re-runs later stages, so deleting their files is safe; an earlier
    stage stays marked done, so deleting its files strands the run (--redo-train
    used to delete the config split renders).
    """
    from wsitrain import STAGES, paths
    from wsitrain.stages import _STAGE_WIPES

    def seed(out, cfg):
        owned: dict[str, list] = {}
        def put(stage, d, name):
            d.mkdir(parents=True, exist_ok=True)
            (d / name).write_text("x")
            owned.setdefault(stage, []).append(d / name)
        put("segment", paths.masks_dir(out, cfg.tissue), "s.npy")
        put("transfer", paths.nuclei_dir(out, cfg.tissue), "n.csv")
        put("tile", paths.images_dir(out, cfg.tissue), "t.png")
        put("crop", paths.cells_dir(out, cfg.tissue), "c.h5")
        put("split", paths.splits_dir(out, cfg.tissue, cfg.fold), "train.csv")
        put("split", paths.train_config_path(
            out, cfg.tissue, cfg.backbone, cfg.fold).parent, "fold_0.yaml")
        put("train", paths.tissue_root(out, cfg.tissue) / "cache", "c.h5")
        put("validate", paths.report_dir(out, cfg.tissue), "scores.json")
        put("export", paths.models_dir(out, cfg.tissue) / "main", "m.pth")
        put("report", paths.report_dir(out, cfg.tissue), "summary.txt")
        return owned

    for target, wipe in _STAGE_WIPES.items():
        cfg = cfg_factory()
        owned = seed(cfg.output, cfg)
        wipe(cfg.output, cfg)
        upstream = [s for s, ps in owned.items()
                    if STAGES.index(s) < STAGES.index(target)
                    and any(not p.exists() for p in ps)]
        assert not upstream, f"_wipe_{target} deleted upstream artefacts: {upstream}"


def test_an_aborted_run_does_not_wipe_artefacts(cfg_factory):
    """--redo used to wipe before the run was known to be viable.

    dag.run defers its lasting writes until after the sample checks; the wipe
    is the most destructive of them and skipped that guard, so a typo'd
    --input deleted artefacts the aborted run could never rebuild.
    """
    from wsitrain import dag, paths
    cfg = cfg_factory()
    (cfg.input / cfg.tissue).mkdir(parents=True, exist_ok=True)  # no samples
    d = paths.tissue_root(cfg.output, cfg.tissue) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    (d / "precious.h5").write_text("expensive to recompute")

    with pytest.raises(SystemExit):
        dag.run(cfg, redo={"train"})

    assert (d / "precious.h5").exists()
