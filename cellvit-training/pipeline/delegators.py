"""``audit``, ``aggregate``, ``orchestrate`` — thin delegators to v1 scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import paths
from ._run import sh


# --- audit ------------------------------------------------------------------

def add_audit_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: pipeline/audit_outputs/).")
    p.add_argument("--dry-run", action="store_true")


def run_audit(args: argparse.Namespace) -> int:
    py = "python3" if args.dry_run else paths.python_executable()
    cmd: list[str] = [py,
                      str(paths.PIPELINE_V1_DIR / "audit_split_reuse.py"),
                      "--tissue", args.tissue, "--fold", args.fold]
    if args.out is not None:
        cmd += ["--out", str(args.out)]
    sh(cmd, log_path=paths.LOGS_DIR / f"audit_{args.tissue}.log",
       dry=args.dry_run)
    return 0


# --- aggregate --------------------------------------------------------------

def add_aggregate_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissues", default=None,
                   help='Space-separated subset, e.g. "breast lung". '
                        "Default: every tissue with exported labels.")
    p.add_argument("--force", action="store_true",
                   help="Wipe trainingset/pantissue/train/ first.")
    p.add_argument("--dry-run", action="store_true")


def run_aggregate(args: argparse.Namespace) -> int:
    cmd: list[str] = ["bash",
                      str(paths.PIPELINE_V1_DIR / "aggregate_pantissue.sh")]
    if args.tissues:
        cmd += ["--tissues", args.tissues]
    if args.force:
        cmd += ["--force"]
    sh(cmd, log_path=paths.LOGS_DIR / "aggregate.log", dry=args.dry_run)
    return 0


# --- orchestrate ------------------------------------------------------------

def add_orchestrate_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--backbone", default=paths.DEFAULT_BACKBONE)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--baseline-run-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")


def run_orchestrate(args: argparse.Namespace) -> int:
    py = "python3" if args.dry_run else paths.python_executable()
    cmd: list[str] = [py,
                      str(paths.PIPELINE_V1_DIR / "agent_orchestrator.py"),
                      "--tissue", args.tissue,
                      "--backbone", args.backbone,
                      "--fold", args.fold]
    if args.max_iter is not None:
        cmd += ["--max-iter", str(args.max_iter)]
    if args.baseline_run_dir is not None:
        cmd += ["--baseline-run-dir", str(args.baseline_run_dir)]
    if args.dry_run:
        cmd += ["--dry-run"]
    sh(cmd, log_path=paths.LOGS_DIR / f"orchestrate_{args.tissue}.log",
       dry=args.dry_run)
    return 0
