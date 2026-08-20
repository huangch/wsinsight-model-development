"""Config merging, manifest invalidation and CellViT config rendering."""
from __future__ import annotations

import yaml
import pytest

from wsitrain import paths
from wsitrain.config import build_config, load_defaults
from wsitrain.configrender import _backbone_weights, _gpu_id, render_config
from wsitrain.manifest import Manifest


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def test_defaults_are_loadable():
    assert load_defaults()["segmenter"] == "stardist"


def test_default_segmenter_is_stardist(tmp_path):
    assert build_config(tmp_path, "breast", tmp_path / "o").segmenter == "stardist"


def test_output_defaults_beside_input(tmp_path):
    cfg = build_config(tmp_path, "breast", None)
    assert cfg.output == tmp_path / "wsinsight_train_out"


def test_cli_overrides_beat_defaults(tmp_path):
    cfg = build_config(tmp_path, "breast", tmp_path / "o",
                       overrides={"segmenter": "cellpose"})
    assert cfg.segmenter == "cellpose"


def test_none_overrides_are_ignored(tmp_path):
    cfg = build_config(tmp_path, "breast", tmp_path / "o",
                       overrides={"segmenter": None})
    assert cfg.segmenter == "stardist"


def test_user_config_beats_defaults_and_loses_to_cli(tmp_path):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({"tile_px": 512, "seed": 7}))
    cfg = build_config(tmp_path, "breast", tmp_path / "o", user_config=user,
                       overrides={"seed": 99})
    assert cfg.tile_px == 512 and cfg.seed == 99


def test_unknown_config_keys_are_dropped(tmp_path):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({"not_a_field": 1}))
    assert build_config(tmp_path, "breast", tmp_path / "o", user_config=user)


def test_io_fields_cannot_be_overridden_by_file(tmp_path):
    user = tmp_path / "u.yaml"
    user.write_text(yaml.safe_dump({"tissue": "lung", "input": "/elsewhere"}))
    cfg = build_config(tmp_path, "breast", tmp_path / "o", user_config=user)
    assert cfg.tissue == "breast" and cfg.input == tmp_path


def test_to_dict_is_yaml_safe(tmp_path):
    cfg = build_config(tmp_path, "breast", tmp_path / "o")
    assert isinstance(yaml.safe_dump(cfg.to_dict()), str)


# --------------------------------------------------------------------------
# gpu / weights resolution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("auto", "0"), ("", "0"), ("all", "0"), ("cpu", "0"),
    ("0", "0"), ("1", "1"), ("3", "3"), ("2,3", "2"), ("nonsense", "0"),
])
def test_gpu_id_resolution(tmp_path, raw, expected):
    cfg = build_config(tmp_path, "breast", tmp_path / "o", overrides={"gpus": raw})
    assert _gpu_id(cfg) == expected


def test_backbone_weights_empty_without_root():
    assert _backbone_weights("", "SAM-H-x40") == ""


def test_backbone_weights_prefers_inside_root(tmp_path):
    root = tmp_path / "CellViT"
    (root / "models").mkdir(parents=True)
    (root / "models" / "CellViT-SAM-H-x40.pth").write_text("w")
    assert _backbone_weights(str(root), "SAM-H-x40").startswith(str(root / "models"))


def test_backbone_weights_falls_back_to_sibling(tmp_path):
    root = tmp_path / "checkout" / "CellViT"
    root.mkdir(parents=True)
    sibling = tmp_path / "checkout" / "models"
    sibling.mkdir()
    (sibling / "CellViT-SAM-H-x40.pth").write_text("w")
    assert _backbone_weights(str(root), "SAM-H-x40") == str(
        sibling / "CellViT-SAM-H-x40.pth")


# --------------------------------------------------------------------------
# render_config
# --------------------------------------------------------------------------

@pytest.fixture
def rendered_tree(tmp_path):
    """Minimal label/split tree so render_config can compute weights."""
    def _make(tissue="breast", classes=('a', 'b')):
        out = tmp_path / "out"
        labels = paths.labels_dir(out, tissue)
        labels.mkdir(parents=True, exist_ok=True)
        (labels / "s_tile_00000.csv").write_text("".join(
            f"0,0,{i}\n" for i in range(len(classes))))
        sd = paths.splits_dir(out, tissue, "fold_0")
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "train.csv").write_text("s_tile_00000\n")
        (sd / "val.csv").write_text("s_tile_00000\n")
        paths.label_map_path(out, tissue).write_text(
            "".join(f'{i}: "{c}"\n' for i, c in enumerate(classes)))
        return out
    return _make


def test_render_writes_requested_gpu(tmp_path, rendered_tree):
    out = rendered_tree()
    cfg = build_config(tmp_path, "breast", out, overrides={"gpus": "3"})
    assert "gpu: 3" in render_config(cfg, out).read_text()


def test_render_emits_all_classes(tmp_path, rendered_tree):
    out = rendered_tree(classes=("a", "b", "c"))
    cfg = build_config(tmp_path, "breast", out)
    body = render_config(cfg, out).read_text()
    assert "num_classes: 3" in body
    assert "0: a" in body and "2: c" in body


def test_render_uses_per_tissue_log_dir(tmp_path, rendered_tree):
    out = rendered_tree()
    cfg = build_config(tmp_path, "breast", out)
    body = render_config(cfg, out).read_text()
    assert str(paths.logs_dir(out, "breast")) in body
    assert paths.logs_dir(out, "breast").is_dir()


def test_render_hash_tracks_split_contents(tmp_path, rendered_tree):
    out = rendered_tree()
    cfg = build_config(tmp_path, "breast", out)
    first = render_config(cfg, out).read_text()
    (paths.splits_dir(out, "breast", "fold_0") / "train.csv").write_text("other\n")
    second = render_config(cfg, out).read_text()
    assert first != second


def test_render_is_valid_yaml(tmp_path, rendered_tree):
    out = rendered_tree()
    cfg = build_config(tmp_path, "breast", out)
    doc = yaml.safe_load(render_config(cfg, out).read_text())
    assert doc["data"]["num_classes"] == 2


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def test_manifest_round_trips(tmp_path):
    mf = Manifest.load_or_new(tmp_path / "m.json", {"task": "t"})
    mf.mark("annotate", "done", n=1)
    assert Manifest.load_or_new(tmp_path / "m.json", {"task": "t"}).is_done("annotate")


def test_manifest_only_done_counts(tmp_path):
    mf = Manifest.load_or_new(tmp_path / "m.json", {})
    mf.mark("annotate", "failed")
    assert not mf.is_done("annotate")


def test_manifest_invalidates_from_changed_key(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"segmenter": "stardist", "tile_px": 1024})
    for stage in ("annotate", "segment", "transfer", "tile"):
        mf.mark(stage, "done")

    fresh = Manifest.load_or_new(p, {"segmenter": "cellpose", "tile_px": 1024})

    assert fresh.is_done("annotate")
    assert not fresh.is_done("segment")
    assert not fresh.is_done("tile")


def test_manifest_survives_unrelated_change(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"gpus": "auto", "tile_px": 1024})
    mf.mark("tile", "done")
    assert Manifest.load_or_new(p, {"gpus": "1", "tile_px": 1024}).is_done("tile")


def test_manifest_uses_earliest_changed_stage(tmp_path):
    p = tmp_path / "m.json"
    mf = Manifest.load_or_new(p, {"task": "a", "seed": 1})
    for stage in ("annotate", "split"):
        mf.mark(stage, "done")

    fresh = Manifest.load_or_new(p, {"task": "b", "seed": 2})

    assert not fresh.is_done("annotate") and not fresh.is_done("split")
