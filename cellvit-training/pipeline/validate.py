"""``validate`` subcommand — delegates to the v1 ``validate_tissue.sh``."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import paths
from ._run import sh


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--backbone", default=paths.DEFAULT_BACKBONE)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--task", default=paths.DEFAULT_TASK,
                   help="Task suffix used in log_comment for run discovery "
                        "(default 'hne'; use 'pantissue' for the pantissue "
                        "cohort).")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Specific logs_local/<run>/ to validate against. "
                        "Default: newest run matching <tissue>-<task>-<backbone>.")
    p.add_argument("--dry-run", action="store_true")


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir
    if run_dir is None:
        comment = paths.log_comment(args.tissue, args.task, args.backbone)
        run_dir = paths.find_latest_run(comment)
        if run_dir is None and not args.dry_run:
            raise SystemExit(
                f"[validate] no run directory matching *_{comment} under "
                f"{paths.LOGS_LOCAL}. Pass --run-dir explicitly or --task "
                f"matching the trained log_comment.")
    cmd = ["bash", str(paths.PIPELINE_V1_DIR / "validate_tissue.sh"),
           args.tissue, args.backbone, args.fold]
    if run_dir is not None:
        cmd.append(str(run_dir))
    log_path = paths.LOGS_DIR / f"validate_{args.tissue}.log"
    sh(cmd, log_path=log_path, dry=args.dry_run)
    return 0
