"""wsitrain CLI: `check`, `run`, and one command per pipeline stage.

Every stage command is a thin alias for `run` restricted to that stage, so all
of them share one config/manifest/sample-discovery path through `dag.run`.
Each command offers only the flags its stage actually reads; anything the stage
needs but you did not type is carried over from the resolved config an earlier
command left in --output.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__, STAGES
from .config import build_config, load_resolved
from .dataset import discover_samples, validate_input


def _cmd_check(args) -> int:
    problems = validate_input(Path(args.input), args.tissue)
    samples = discover_samples(Path(args.input), args.tissue)
    print(f"input: {args.input}  tissue: {args.tissue}")
    print(f"samples: {len(samples)}")
    for s in samples:
        print(f"  - {s.sample_id} [{s.tissue}] aligned={s.aligned}")
    if samples:
        from .config import default_output
        from .dataset import write_manifest
        # Not --input: that is often a read-only mount, and this file is a
        # report about the run, not part of the dataset.
        mpath = default_output(Path(args.input), args.output) / "wsitrain_samples.csv"
        try:
            write_manifest(samples, mpath)
            print(f"samples listed in: {mpath}")
        except OSError as e:
            print(f"(could not write {mpath}: {e})")
    if shutil.which("kurtorank") is None:
        problems.append("kurtorank console script not found on PATH")
    try:
        import torch  # noqa
        print(f"cuda: {torch.cuda.is_available()} ({torch.cuda.device_count()} gpu)")
    except Exception:
        problems.append("torch not importable (GPU stack missing)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nOK — ready to run.")
    return 0


def _labels(values) -> tuple[str, ...] | None:
    """Accept both ``--drop-labels a b`` and ``--drop-labels a,b``."""
    if values is None:
        return None
    out: list[str] = []
    for v in values:
        out += [part.strip() for part in str(v).split(",") if part.strip()]
    return tuple(out)


def _stages(values) -> list[str]:
    """Accept both ``a b`` and ``a,b`` forms."""
    out: list[str] = []
    for v in values or ():
        out += [part.strip() for part in str(v).split(",") if part.strip()]
    unknown = [s for s in out if s not in STAGES]
    if unknown:
        raise SystemExit(
            f"unknown stage(s) {', '.join(unknown)}; choose from {', '.join(STAGES)}")
    return out


# ---------------------------------------------------------------- flag groups

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", required=True)
    p.add_argument("--tissue", default="pantissue")
    p.add_argument("--output", default=None)
    p.add_argument("--force", action="store_true",
                   help="re-run stages the manifest already marks done")
    p.add_argument("--reset-config", action="store_true",
                   help="ignore the config an earlier command saved in --output "
                        "and start from the shipped defaults plus these flags")


def _add_labelspace(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default=None,
                   help="label space, e.g. pantissue | hne | pannuke")


def _add_annotate(p: argparse.ArgumentParser) -> None:
    p.add_argument("--markers-csv", type=Path, default=None,
                   help="marker panel CSV (default: kurtorank bundled)")
    p.add_argument("--top-k-markers", type=int, default=None,
                   help="keep the K most specific genes per subtype")


def _add_mpp(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mpp", type=float, default=None,
                   help="microns per pixel of the H&E")


def _add_tile_px(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tile-px", type=int, default=None, help="tile edge in pixels")


def _add_model_id(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backbone", default=None, help="CellViT backbone, e.g. SAM-H-x40")
    p.add_argument("--fold", default=None, help="fold name, e.g. fold_0")


def _add_gpus(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpus", default=None, help="device index, or 'cpu' to disable")


def _add_segment(p: argparse.ArgumentParser) -> None:
    p.add_argument("--segmenter", default=None, choices=["cellpose", "stardist"])
    p.add_argument("--cellpose-model", default=None, help="cellpose model name")
    p.add_argument("--stardist-model", default=None, help="stardist model name")
    p.add_argument("--stardist-model-dir", type=Path, default=None,
                   help="parent of the stardist model folder; avoids downloading "
                        "(env fallback: WSITRAIN_STARDIST_DIR, then KERAS_HOME)")
    # A bare store_true could never be taken back once saved into the config.
    cpu = p.add_mutually_exclusive_group()
    cpu.add_argument("--stardist-cpu", dest="stardist_cpu", action="store_const",
                     const=True,
                     help="run StarDist on CPU (no CUDA toolkit); torch keeps the GPU")
    cpu.add_argument("--no-stardist-cpu", dest="stardist_cpu", action="store_const",
                     const=False, help="run StarDist on the GPU (default)")
    p.set_defaults(stardist_cpu=None)
    p.add_argument("--diameter", type=float, default=None,
                   help="nucleus diameter in MICRONS (omit for auto)")
    p.add_argument("--cellpose-batch-size", type=int, default=None,
                   help="cellpose tile batch; lower (4/2/1) if GPU OOMs")
    p.add_argument("--cellpose-flow-threshold", type=float, default=None,
                   help="cellpose flow QC; 0 skips GPU flow check (avoids WSI OOM)")


def _add_transform(p: argparse.ArgumentParser) -> None:
    # Not transfer-only: dag drops unaligned samples from every stage, so
    # segment has to be told the same thing or it segments a different cohort.
    p.add_argument("--transform", default=None,
                   choices=["affine", "affine+bspline", "none"])


def _add_transfer(p: argparse.ArgumentParser) -> None:
    p.add_argument("--match-radius-px", type=int, default=None,
                   help="nucleus lookup search radius in H&E px (0 = exact pixel)")
    p.add_argument("--min-match-rate", type=float, default=None,
                   help="drop slides whose registration matches fewer than this fraction")
    p.add_argument("--drop-labels", nargs="+", default=None, metavar="LABEL",
                   help="cell types to exclude; space- or comma-separated")


def _add_tile(p: argparse.ArgumentParser) -> None:
    p.add_argument("--min-cells", type=int, default=None,
                   help="drop tiles with fewer cells")
    p.add_argument("--bg-thresh", type=float, default=None,
                   help="drop tiles whose mean RGB exceeds this")
    p.add_argument("--overlap", type=float, default=None,
                   help="tile overlap fraction (0-1)")


def _add_split(p: argparse.ArgumentParser) -> None:
    p.add_argument("--val-frac", type=float, default=None, help="validation fraction")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--by-tile", dest="by_slide", action="store_const", const=False,
                      help="hold out a fraction of tiles from every slide (default)")
    mode.add_argument("--by-slide", dest="by_slide", action="store_const", const=True,
                      help="hold out whole slides")
    p.set_defaults(by_slide=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--weight-cap", type=float, default=None,
                   help="cap on inverse-frequency class weights")


def _add_train(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tune", type=int, default=None,
                   help="auto-tune iterations (0=off)")


# Flag groups per stage command. A shared group is repeated under every stage
# that reads it: split and train render the CellViT config, so they need the
# label space and the device on top of the model identity.
_STAGE_FLAGS = {
    "annotate": (_add_labelspace, _add_annotate),
    "segment": (_add_segment, _add_transform, _add_mpp, _add_gpus),
    "transfer": (_add_transfer, _add_transform, _add_labelspace, _add_mpp),
    "tile": (_add_tile, _add_tile_px),
    "split": (_add_split, _add_model_id, _add_labelspace, _add_gpus),
    "train": (_add_train, _add_model_id, _add_gpus),
    "validate": (_add_model_id,),
    "export": (_add_model_id, _add_tile_px),
    "report": (),
}

_STAGE_HELP = {
    "annotate": "Label cells with kurtorank.",
    "segment": "Segment nuclei on each H&E.",
    "transfer": "Join cell labels onto the H&E nuclei.",
    "tile": "Cut labelled slides into training tiles.",
    "split": "Build train/val lists and the CellViT config.",
    "train": "Train the CellViT classifier head.",
    "validate": "Score the trained head and write the confusion matrix.",
    "export": "Promote the best checkpoint to a deployable model.",
    "report": "Summarise the run.",
}

_OVERRIDE_FIELDS = (
    "segmenter", "gpus", "transform", "tune", "task", "match_radius_px",
    "min_match_rate", "cellpose_batch_size", "cellpose_flow_threshold",
    "cellpose_model", "stardist_model", "stardist_model_dir", "stardist_cpu",
    "diameter", "tile_px", "mpp", "min_cells", "bg_thresh", "overlap",
    "val_frac", "by_slide", "seed", "weight_cap", "backbone", "fold",
    "markers_csv", "top_k_markers",
)


def _cmd_run(args) -> int:
    # Absent on a stage command that does not expose the flag; build_config
    # then falls back to the saved config, and only then to the shipped default.
    overrides = {name: getattr(args, name, None) for name in _OVERRIDE_FIELDS}
    overrides["drop_labels"] = _labels(getattr(args, "drop_labels", None))

    output = Path(args.output) if args.output else None
    base = ({} if getattr(args, "reset_config", False)
            else load_resolved(Path(args.input), args.tissue, output))
    cfg = build_config(Path(args.input), args.tissue, output, overrides=overrides,
                       base=base)
    from . import dag
    return dag.run(cfg, only=getattr(args, "only", None),
                   skip=_stages(getattr(args, "run_skip", None)), force=args.force)


def main(argv=None) -> int:
    p = argparse.ArgumentParser("wsitrain", description="Train WSInsight CellViT heads end-to-end.")
    p.add_argument("--version", action="version", version=f"wsitrain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="Preflight: validate input + environment.")
    c.add_argument("--input", required=True)
    c.add_argument("--tissue", default="pantissue")
    c.add_argument("--output", default=None)
    c.set_defaults(fn=_cmd_check)

    r = sub.add_parser("run", help="Run the end-to-end training pipeline.")
    _add_common(r)
    for add in (_add_labelspace, _add_annotate, _add_segment, _add_mpp, _add_transform,
                _add_transfer, _add_tile, _add_tile_px, _add_split, _add_train,
                _add_model_id, _add_gpus):
        add(r)
    r.add_argument("--run-skip", action="extend", nargs="+", default=[],
                   metavar="STAGE", help="run everything EXCEPT these stages")
    r.epilog = ("stages run in this order: " + " \u2192 ".join(STAGES)
                + ". --run-skip takes space- or comma-separated names; each stage "
                  "is also a command of its own.")
    r.set_defaults(fn=_cmd_run)

    for stage in STAGES:
        sp = sub.add_parser(stage, help=_STAGE_HELP[stage])
        _add_common(sp)
        for add in _STAGE_FLAGS[stage]:
            add(sp)
        sp.epilog = ("flags this stage does not take are carried over from the "
                     "config an earlier command wrote into --output; the stages "
                     "this one depends on must already be done.")
        # A stage command is `run` narrowed to one stage.
        sp.set_defaults(fn=_cmd_run, only=stage)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
