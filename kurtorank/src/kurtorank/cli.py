"""Unified `kurtorank` CLI with `annotate` and `rank-markers` subcommands."""
from __future__ import annotations

import os
import sys

import click


# ---------------------------------------------------------------------------
# annotate subcommand (click-native, already defined in annotate.main)
# ---------------------------------------------------------------------------
from kurtorank.annotate.main import annotate_cmd
from kurtorank.seed.main import build_panel_cmd


# ---------------------------------------------------------------------------
# rank-markers subcommand (passthrough to the argparse-based module)
# ---------------------------------------------------------------------------
@click.command(
    "rank-markers",
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": [],  # let the argparse layer handle --help
    },
)
@click.argument("argv", nargs=-1, type=click.UNPROCESSED)
def rank_markers_cmd(argv: tuple[str, ...]) -> None:
    """Rerank a markers CSV against a CELLxGENE Census atlas.

    All flags are forwarded to ``kurtorank.rank.main``. Run
    ``kurtorank rank-markers --help`` for the full list.
    """
    from kurtorank.rank.main import rank_markers_main

    old_argv = sys.argv
    sys.argv = ["kurtorank rank-markers", *argv]
    try:
        rc = rank_markers_main()
    finally:
        sys.argv = old_argv
    if isinstance(rc, int) and rc != 0:
        sys.exit(rc)


# ---------------------------------------------------------------------------
# root group
# ---------------------------------------------------------------------------
@click.group(
    help="KurtoRank v3 — unsupervised ensemble subtype annotation "
         "for gene-limited spatial transcriptomics.",
)
@click.version_option(package_name="kurtorank", prog_name="kurtorank")
def cli() -> None:
    pass


# Register subcommands.
cli.add_command(annotate_cmd, name="annotate")
cli.add_command(rank_markers_cmd)
cli.add_command(build_panel_cmd)


def _main() -> None:
    # Keep multiprocessing / BLAS settings consistent with the original
    # monolithic entry point.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import multiprocessing as _mp
        _mp.set_start_method("spawn", force=True)
    except (RuntimeError, ValueError):
        pass
    try:
        import torch
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    cli()


if __name__ == "__main__":
    _main()
