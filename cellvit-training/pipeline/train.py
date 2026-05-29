"""``train`` subcommand — delegates to the v1 ``train_tissue.sh`` driver.

The bash driver runs four steps (train head → locate checkpoint → validate →
TorchScript export) which themselves invoke CellViT-plus-plus. v2 wraps it
unchanged because re-implementing that lifecycle in Python would duplicate
~200 lines of subprocess plumbing for no functional gain.
"""
from __future__ import annotations

import argparse

from . import paths
from ._run import sh


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--backbone", default=paths.DEFAULT_BACKBONE)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--task", default="pantissue",
                   help="<log_comment> suffix; must match what "
                        "make_train_config produced (default pantissue).")
    p.add_argument("--dry-run", action="store_true")


def run(args: argparse.Namespace) -> int:
    script = paths.PIPELINE_V1_DIR / "train_tissue.sh"
    log_path = (paths.LOGS_DIR
                / f"train_{args.tissue}_{args.backbone}_{args.fold}.log")
    sh(["bash", script, args.tissue, args.backbone, args.fold, args.task],
       log_path=log_path, dry=args.dry_run)
    return 0
