"""QuPath wrappers — ``detect``, ``label``, ``export`` subcommands.

Each runs a single ``QuPath script -p <project> [-a ...] <script.groovy>``
invocation over the global QuPath project. All three Groovy scripts live
under ``pipeline/qupath/``; v2 does not duplicate them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import paths
from ._run import sh


def _qupath_cmd(groovy: Path, project: Path,
                script_args: list[str], *, dry: bool = False) -> list[str | Path]:
    qupath = "QuPath" if dry else paths.qupath_executable()
    cmd: list[str | Path] = [qupath, "script", "-s", "-p", project]
    for a in script_args:
        cmd += ["-a", a]
    cmd.append(groovy)
    return cmd


# --- detect -----------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qproj", type=Path, default=paths.QPROJ_PATH,
                   help=f"QuPath project (default {paths.QPROJ_PATH}).")
    p.add_argument("--dry-run", action="store_true")


def add_detect_arguments(p: argparse.ArgumentParser) -> None:
    _add_common(p)


def run_detect(args: argparse.Namespace) -> int:
    cmd = _qupath_cmd(paths.groovy_path("run_qust_pipeline"),
                      args.qproj, [], dry=args.dry_run)
    sh(cmd, log_path=paths.LOGS_DIR / "qupath" / "detect.log",
       dry=args.dry_run)
    return 0


# --- label ------------------------------------------------------------------

def add_label_arguments(p: argparse.ArgumentParser) -> None:
    _add_common(p)
    p.add_argument("--assignment-csv", type=Path, required=True,
                   help="Absolute path to celltype_assignment_*_label.csv "
                        "produced by `panel`.")


def run_label(args: argparse.Namespace) -> int:
    cmd = _qupath_cmd(paths.groovy_path("load_mapping"),
                      args.qproj, [str(args.assignment_csv.resolve())],
                      dry=args.dry_run)
    sh(cmd, log_path=paths.LOGS_DIR / "qupath" / "label.log",
       dry=args.dry_run)
    return 0


# --- export -----------------------------------------------------------------

def add_export_arguments(p: argparse.ArgumentParser) -> None:
    _add_common(p)
    p.add_argument("--tissue", default=None,
                   help="Restrict export to one tissue "
                        "(maps to args[0]=FORCE_TISSUE in the groovy). "
                        "Omit to export every tissue in one pass.")
    p.add_argument("--overlap", type=float, default=None,
                   help="Tile overlap fraction in [0.0, 1.0); maps to "
                        "args[1] in the groovy. Use 0.5 for single-slide "
                        "tissues.")


def run_export(args: argparse.Namespace) -> int:
    script_args: list[str] = []
    if args.tissue is not None:
        script_args.append(args.tissue)
        if args.overlap is not None:
            script_args.append(f"{args.overlap:.3f}")
    elif args.overlap is not None:
        raise SystemExit("[export] --overlap requires --tissue.")
    cmd = _qupath_cmd(paths.groovy_path("export_tiles"),
                      args.qproj, script_args, dry=args.dry_run)
    log_name = f"export_{args.tissue or 'all'}.log"
    sh(cmd, log_path=paths.LOGS_DIR / "qupath" / log_name,
       dry=args.dry_run)
    return 0
