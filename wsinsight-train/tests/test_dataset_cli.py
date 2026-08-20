"""Sample discovery, manifest I/O, splits, weights and the CLI front door."""
from __future__ import annotations

import pytest

from wsitrain import dataset, splits, weights
from wsitrain.cli import main


def _sample_tree(root, tissue, name, *, he="{name}_he_image.ome.tif", reg=False):
    outs = root / tissue / name / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / he.format(name=name)).write_bytes(b"")
    if reg:
        (outs / "direct_transf.txt").write_text("Intervals=8")
        (outs / "registration_params.json").write_text("{}")
    return outs


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def test_requires_cells_parquet(tmp_path):
    outs = tmp_path / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    assert dataset.discover_samples(tmp_path) == []


def test_requires_he_image(tmp_path):
    outs = tmp_path / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    assert dataset.discover_samples(tmp_path) == []


def test_sample_id_encodes_relative_path(tmp_path):
    _sample_tree(tmp_path, "breast", "s1")
    assert dataset.discover_samples(tmp_path)[0].sample_id == "breast__s1"


def test_tissue_comes_from_top_level_dir(tmp_path):
    _sample_tree(tmp_path, "lung", "s1")
    assert dataset.discover_samples(tmp_path)[0].tissue == "lung"


def test_plus_separated_tissue_list(tmp_path):
    for t in ("breast", "lung", "skin"):
        _sample_tree(tmp_path, t, "s1")
    assert len(dataset.discover_samples(tmp_path, "breast+lung")) == 2


def test_registration_files_mark_sample_aligned(tmp_path):
    _sample_tree(tmp_path, "breast", "s1", he="{name}_he_unaligned_image.ome.tif", reg=True)
    assert dataset.discover_samples(tmp_path)[0].aligned is True


def test_unaligned_filename_without_registration(tmp_path):
    _sample_tree(tmp_path, "breast", "s1", he="{name}_he_unaligned_image.ome.tif")
    assert dataset.discover_samples(tmp_path)[0].aligned is False


def test_aligned_he_preferred_over_unaligned(tmp_path):
    outs = _sample_tree(tmp_path, "breast", "s1")
    (outs.parent / "s1_he_unaligned_image.ome.tif").write_bytes(b"")
    assert "unaligned" not in dataset.discover_samples(tmp_path)[0].he.name


def test_manifest_round_trip(tmp_path):
    _sample_tree(tmp_path, "breast", "s1")
    found = dataset.discover_samples(tmp_path)
    dataset.write_manifest(found, tmp_path / "m.csv")
    back = dataset.read_manifest(tmp_path / "m.csv")
    assert [s.sample_id for s in back] == [s.sample_id for s in found]
    assert back[0].aligned == found[0].aligned


def test_validate_reports_missing_input(tmp_path):
    problems = dataset.validate_input(tmp_path / "nope", "breast")
    assert any("does not exist" in p for p in problems)


def test_validate_reports_unaligned(tmp_path):
    _sample_tree(tmp_path, "breast", "s1", he="{name}_he_unaligned_image.ome.tif")
    assert any("UNALIGNED" in p for p in dataset.validate_input(tmp_path, "breast"))


def test_validate_clean_tree_has_no_problems(tmp_path):
    _sample_tree(tmp_path, "breast", "s1", reg=True)
    assert dataset.validate_input(tmp_path, "breast") == []


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def test_split_requires_existing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        splits.split_tiles(tmp_path / "missing")


def test_split_requires_tiles(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    with pytest.raises(ValueError, match="no .csv"):
        splits.split_tiles(d)


def test_sample_tag_strips_tile_suffix():
    assert splits.sample_tag("breast__s1_tile_00042") == "breast__s1"


def test_per_tile_split_is_deterministic(label_dir_factory):
    d = label_dir_factory({f"s1_tile_{i:05d}": [0] for i in range(10)})
    a = splits.split_tiles(d, by_slide=False, seed=5)
    b = splits.split_tiles(d, by_slide=False, seed=5)
    assert a.train == b.train and a.val == b.val


def test_per_tile_split_covers_every_slide(label_dir_factory):
    tiles = {f"{s}_tile_{i:05d}": [0] for s in ("a", "b") for i in range(10)}
    res = splits.split_tiles(label_dir_factory(tiles), by_slide=False, val_frac=0.2)
    assert {splits.sample_tag(t) for t in res.val} == {"a", "b"}


def test_single_tile_slide_stays_in_train(label_dir_factory):
    res = splits.split_tiles(label_dir_factory({"solo_tile_00000": [0]}), by_slide=False)
    assert res.val == [] and res.train == ["solo_tile_00000"]


def test_no_tile_appears_on_both_sides(label_dir_factory):
    tiles = {f"{s}_tile_{i:05d}": [0] for s in ("a", "b", "c") for i in range(6)}
    res = splits.split_tiles(label_dir_factory(tiles), by_slide=False)
    assert set(res.train).isdisjoint(res.val)


def test_slide_level_split_keeps_slides_whole(label_dir_factory):
    tiles = {f"breast__{s}_tile_{i:05d}": [0, 1] for s in ("a", "b", "c", "d")
             for i in range(4)}
    res = splits.split_tiles(label_dir_factory(tiles), by_slide=True, val_frac=0.25)
    assert set(res.train_slides).isdisjoint(res.val_slides)


def test_sole_carrier_slide_is_tile_split(label_dir_factory):
    tiles = {f"breast__{s}_tile_{i:05d}": [0] for s in ("a", "b", "c") for i in range(4)}
    tiles["breast__rare_tile_00000"] = [9]
    tiles["breast__rare_tile_00001"] = [9]
    res = splits.split_tiles(label_dir_factory(tiles), by_slide=True)
    assert "hybrid" in res.mode


def test_write_split_creates_both_files(tmp_path):
    res = splits.SplitResult(["a"], ["b"], "m", 2, ["a"], ["b"])
    splits.write_split(res, tmp_path / "sp")
    assert (tmp_path / "sp" / "train.csv").read_text() == "a\n"
    assert (tmp_path / "sp" / "val.csv").read_text() == "b\n"


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def test_label_map_parsing_handles_comments_and_quotes(tmp_path):
    p = tmp_path / "lm.yaml"
    p.write_text('# header\n0: "a"\n1: b   # trailing\n\n')
    assert weights.load_label_map(p) == {0: "a", 1: "b"}


def test_empty_label_map_raises(tmp_path, label_dir_factory):
    p = tmp_path / "lm.yaml"
    p.write_text("# nothing\n")
    with pytest.raises(ValueError, match="label_map is empty"):
        weights.compute_weights(p, label_dir_factory({"t_tile_0": [0]}))


def test_no_labels_raises(tmp_path):
    p = tmp_path / "lm.yaml"
    p.write_text('0: "a"\n')
    empty = tmp_path / "labels"
    empty.mkdir()
    with pytest.raises(ValueError, match="no labels"):
        weights.compute_weights(p, empty)


def test_weight_budget_equals_class_count(tmp_path, label_dir_factory):
    p = tmp_path / "lm.yaml"
    p.write_text('0: "a"\n1: "b"\n2: "c"\n')
    d = label_dir_factory({"t_tile_0": [0] * 80 + [1] * 15 + [2] * 5})
    rep = weights.compute_weights(p, d, cap=10)
    assert sum(rep.weights) == pytest.approx(3, abs=0.01)


def test_rare_class_outweighs_common(tmp_path, label_dir_factory):
    p = tmp_path / "lm.yaml"
    p.write_text('0: "a"\n1: "b"\n')
    d = label_dir_factory({"t_tile_0": [0] * 90 + [1] * 10})
    rep = weights.compute_weights(p, d, cap=10)
    assert rep.weights[1] > rep.weights[0]


def test_absent_class_is_capped(tmp_path, label_dir_factory):
    p = tmp_path / "lm.yaml"
    p.write_text('0: "a"\n1: "ghost"\n')
    rep = weights.compute_weights(p, label_dir_factory({"t_tile_0": [0] * 10}), cap=10)
    assert 1 in rep.capped_classes


def test_tally_ignores_blank_lines(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    (d / "t.csv").write_text("0,0,1\n\n0,0,1\n")
    assert weights.tally_labels(d)[1] == 2


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_version_exits_zero():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])


def test_invalid_stage_name_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--from", "bogus"])


def test_invalid_segmenter_rejected(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "--input", str(tmp_path), "--segmenter", "instanseg"])


def test_check_reports_problems_for_empty_input(tmp_path, capsys):
    assert main(["check", "--input", str(tmp_path)]) == 1
    assert "PROBLEMS" in capsys.readouterr().out


def test_check_writes_sample_manifest(tmp_path):
    _sample_tree(tmp_path, "breast", "s1", reg=True)
    main(["check", "--input", str(tmp_path), "--tissue", "breast"])
    assert (tmp_path / "wsitrain_samples.csv").exists()
