"""Per-stage commands: dispatch, option surface, and the prerequisite gate."""
from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout

import pytest
import yaml

from wsitrain import STAGES, dag, paths, prereq
from wsitrain.cli import main
from wsitrain.manifest import Manifest


@pytest.fixture
def cohort(tmp_path):
    """Minimal sample tree that discover_samples() accepts."""
    outs = tmp_path / "breast" / "s1" / "outs"
    outs.mkdir(parents=True)
    (outs / "cells.parquet").write_bytes(b"")
    (outs.parent / "s1_he_image.ome.tif").write_bytes(b"")
    return tmp_path


@pytest.fixture
def seen(monkeypatch):
    """Stub the DAG out and capture the cfg + kwargs it was called with."""
    got = {}

    def fake_run(cfg, **kw):
        got["cfg"] = cfg
        got.update(kw)
        return 0

    monkeypatch.setattr(dag, "run", fake_run)
    return got


def _help(*argv) -> str:
    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
        main([*argv, "--help"])
    return buf.getvalue()


def _offered(*argv) -> set[str]:
    """Option strings a command accepts, read off its usage line.

    Only the usage block is scanned: help prose and the epilog also mention
    flags, and those are not part of the command's surface.
    """
    usage = _help(*argv).split("\n\n", 1)[0]
    return set(re.findall(r"--[a-z][a-z0-9-]*", usage))


# --------------------------------------------------------------------------
# every stage is a command, and it runs only that stage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stage", STAGES)
def test_stage_command_runs_only_that_stage(stage, cohort, seen):
    main([stage, "--input", str(cohort), "--tissue", "breast"])
    assert seen["only"] == stage
    assert seen["skip"] == []


@pytest.mark.parametrize("stage", STAGES)
def test_stage_command_takes_force(stage, cohort, seen):
    main([stage, "--input", str(cohort), "--tissue", "breast", "--force"])
    assert seen["force"] is True


def test_run_skip_is_rejected_on_a_stage_command(cohort):
    with pytest.raises(SystemExit):
        main(["tile", "--input", str(cohort), "--run-skip", "train"])


# --------------------------------------------------------------------------
# option surface: each command offers exactly the flags its stage reads
# --------------------------------------------------------------------------

EXPECTED_FLAGS = {
    "annotate": {"--task", "--markers-csv", "--top-k-markers"},
    "segment": {"--segmenter", "--cellpose-model", "--stardist-model",
                "--stardist-model-dir", "--stardist-cpu", "--no-stardist-cpu",
                "--diameter", "--cellpose-batch-size", "--cellpose-flow-threshold",
                "--transform", "--mpp", "--gpus",
                "--stardist-normalization-pmin", "--stardist-normalization-pmax"},
    "transfer": {"--transform", "--match-radius-px", "--min-match-rate",
                 "--drop-labels", "--task", "--mpp"},
    "tile": {"--min-cells", "--bg-thresh", "--overlap", "--tile-px"},
    "crop": {"--object-detection", "--architecture", "--patch-size-pixels",
             "--patch-spacing-um-px", "--stain-normalization",
             "--no-stain-normalization", "--norm-sample-size",
             "--stardist-normalization-pmin", "--stardist-normalization-pmax",
             "--mpp"},
    "split": {"--val-frac", "--by-tile", "--by-slide", "--seed", "--weight-cap",
              "--backbone", "--fold", "--task", "--gpus"},
    "train": {"--tune", "--backbone", "--fold", "--gpus",
              "--epochs", "--batch-size", "--lr", "--weight-decay",
              "--pretrained", "--num-workers"},
    "validate": {"--backbone", "--fold"},
    "export": {"--backbone", "--fold", "--tile-px"},
    "report": set(),
}

# -h is the only spelling argparse puts in the usage line, so --help is not here.
COMMON_FLAGS = {"--input", "--tissue", "--output", "--force", "--config",
                "--reset-config", "--show-config"}


@pytest.mark.parametrize("stage", STAGES)
def test_stage_option_surface(stage):
    assert _offered(stage) == EXPECTED_FLAGS[stage] | COMMON_FLAGS


def test_segment_does_not_offer_training_flags():
    text = _help("segment")
    assert "--backbone" not in text and "--fold" not in text


def test_run_still_offers_everything():
    text = _help("run")
    for flags in EXPECTED_FLAGS.values():
        for flag in flags:
            assert flag in text
    assert "--run-skip" in text


# --------------------------------------------------------------------------
# prerequisite gate
# --------------------------------------------------------------------------

def _manifest(out, tissue, done):
    out.mkdir(parents=True, exist_ok=True)
    path = paths.manifest_path(out, tissue)
    path.write_text(json.dumps({
        "config": {},
        "stages": {s: {"status": "done", "ts": 0} for s in done},
    }))
    return Manifest.load_or_new(path, {})


class _Cfg:
    def __init__(self, out, tissue="breast", input="/in"):
        self.output, self.tissue, self.input = out, tissue, input


@pytest.mark.parametrize("stage,missing", [
    ("transfer", "annotate"),
    ("tile", "transfer"),
    ("split", "tile"),
    ("train", "split"),
    ("validate", "train"),
    ("export", "train"),
])
def test_stage_refuses_to_run_without_its_predecessor(stage, missing, tmp_path):
    out = tmp_path / "out"
    mf = _manifest(out, "breast", done=())
    with pytest.raises(SystemExit) as e:
        prereq.check(stage, mf, _Cfg(out))
    assert missing in str(e.value)


@pytest.mark.parametrize("stage", ["annotate", "segment", "report"])
def test_entry_stages_have_no_prerequisites(stage, tmp_path):
    out = tmp_path / "out"
    prereq.check(stage, _manifest(out, "breast", done=()), _Cfg(out))


def test_prereq_passes_once_predecessors_are_done(tmp_path):
    out = tmp_path / "out"
    mf = _manifest(out, "breast", done=("annotate", "segment"))
    (paths.masks_dir(out, "breast")).mkdir(parents=True)
    (paths.masks_dir(out, "breast") / "s1.npy").write_bytes(b"")
    prereq.check("transfer", mf, _Cfg(out))


def test_prereq_rejects_a_done_mark_whose_output_is_gone(tmp_path):
    """The manifest can outlive a hand-cleaned output dir."""
    out = tmp_path / "out"
    mf = _manifest(out, "breast", done=("annotate", "segment"))
    with pytest.raises(SystemExit) as e:
        prereq.check("transfer", mf, _Cfg(out))
    assert "missing or empty" in str(e.value)


def test_prereq_rejects_an_empty_output_dir(tmp_path):
    out = tmp_path / "out"
    mf = _manifest(out, "breast", done=("annotate", "segment"))
    paths.masks_dir(out, "breast").mkdir(parents=True)
    with pytest.raises(SystemExit):
        prereq.check("transfer", mf, _Cfg(out))


def test_every_stage_is_covered_by_the_prereq_table():
    """A new stage must be given prerequisites, even if empty."""
    entry = {"annotate", "segment", "report"}
    assert set(prereq.PREREQUISITES) | entry == set(STAGES)


# --------------------------------------------------------------------------
# shared options survive from one command to the next
# --------------------------------------------------------------------------

def test_shared_option_carries_over_to_the_next_command(cohort, seen):
    out = cohort / "out"
    main(["segment", "--input", str(cohort), "--tissue", "breast",
          "--output", str(out), "--mpp", "0.5"])
    # dag.run is stubbed, so write the resolved config it would have left behind.
    out.mkdir(parents=True, exist_ok=True)
    paths.resolved_config_path(out, "breast").write_text(
        yaml.safe_dump(seen["cfg"].to_dict()))

    main(["transfer", "--input", str(cohort), "--tissue", "breast",
          "--output", str(out)])
    assert seen["cfg"].mpp == 0.5


def test_typed_flag_beats_the_saved_config(cohort, seen):
    out = cohort / "out"
    out.mkdir(parents=True, exist_ok=True)
    paths.resolved_config_path(out, "breast").write_text("mpp: 0.5\n")

    main(["segment", "--input", str(cohort), "--tissue", "breast",
          "--output", str(out), "--mpp", "0.25"])
    assert seen["cfg"].mpp == 0.25


def test_unset_flag_falls_back_to_the_shipped_default(cohort, seen):
    main(["segment", "--input", str(cohort), "--tissue", "breast"])
    assert seen["cfg"].mpp == 0.25
