"""argparse dispatcher for ``python -m pipeline <subcommand>``.

Single place where the command-line surface is defined. Each entry binds a
subcommand name to ``(add_arguments_fn, run_fn, help_text)``. All wrapper
modules expose ``add_arguments(parser)`` + ``run(args) -> int``; the three
modules that bundle multiple subcommands (``qupath`` and ``delegators``)
expose namespaced helpers instead.
"""
from __future__ import annotations

import argparse
import sys

from . import delegators, panel, prepare, qupath, run as run_mod, train, validate

# name -> (add_arguments, run, help)
_SUBCOMMANDS = {
    "panel": (panel.add_arguments, panel.run,
              "Annotate Xenium samples with kurtorank."),
    "detect": (qupath.add_detect_arguments, qupath.run_detect,
               "QuPath: tissue detection + StarDist + Xenium cluster transfer."),
    "label": (qupath.add_label_arguments, qupath.run_label,
              "QuPath: cluster_id -> label-name remap on detections."),
    "export": (qupath.add_export_arguments, qupath.run_export,
               "QuPath: detections -> tile PNG + per-tile CSV."),
    "prepare": (prepare.add_arguments, prepare.run,
                "Splits + class weights + train YAML for one tissue."),
    "train": (train.add_arguments, train.run,
              "Train head + TorchScript export (delegates to train_tissue.sh)."),
    "validate": (validate.add_arguments, validate.run,
                 "Re-run validation against a finished training run."),
    "audit": (delegators.add_audit_arguments, delegators.run_audit,
              "Train/val pixel and cell-label reuse audit."),
    "aggregate": (delegators.add_aggregate_arguments, delegators.run_aggregate,
                  "Build trainingset/pantissue/ from per-tissue splits."),
    "orchestrate": (delegators.add_orchestrate_arguments,
                    delegators.run_orchestrate,
                    "Outer-loop agent: iteratively tune weights / lr / drop_rate."),
    "run": (run_mod.add_arguments, run_mod.run,
            "End-to-end driver: panel -> detect -> label -> export -> "
            "prepare -> train -> validate -> audit. Use --from/--to to "
            "bound the range."),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="cellvit-training pipeline (clean rewrite).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, (add_fn, run_fn, help_text) in _SUBCOMMANDS.items():
        sp = sub.add_parser(name, help=help_text, description=help_text)
        add_fn(sp)
        sp.set_defaults(_run=run_fn)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
