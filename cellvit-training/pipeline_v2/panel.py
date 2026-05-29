"""``panel`` subcommand — annotate Xenium samples with kurtorank.

Discovers every ``outs/`` directory under ``data/xenium/<tissue>/`` and
runs ``kurtorank annotate`` once per sample, writing the per-sample
``annotated.h5ad`` + ``celltype_assignment_*.csv`` next to the Xenium data.

This is step 0 of the chain: produces the CSVs that
``load_mapping.groovy`` consumes to remap QuPath ``cluster_id`` → label name.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import paths
from ._run import sh


def _discover_samples(tissue_dir: Path) -> list[Path]:
    """Return every ``outs/`` directory anywhere under ``tissue_dir``."""
    if not tissue_dir.is_dir():
        return []
    return sorted(p for p in tissue_dir.rglob("outs") if p.is_dir())


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True,
                   help="Tissue under data/xenium/ to annotate.")
    p.add_argument("--tissue-type",
                   help="kurtorank --tissue-type value. Defaults to --tissue.")
    p.add_argument("--top-k", type=int, default=25,
                   help="--use-top-k-markers (default 25).")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--markers-csv", type=Path, default=None,
                   help="Override panel CSV (defaults to kurtorank bundled).")
    p.add_argument("--overwrite", action="store_true",
                   help="Pass --overwrite to kurtorank.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only annotate the first N discovered samples.")
    p.add_argument("--dry-run", action="store_true")


def run(args: argparse.Namespace) -> int:
    tissue_dir = paths.xenium_tissue_dir(args.tissue)
    samples = _discover_samples(tissue_dir)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print(f"[panel] no outs/ directories under {tissue_dir}")
        return 1
    print(f"[panel] tissue={args.tissue}  samples={len(samples)}")

    krk = "kurtorank" if args.dry_run else paths.kurtorank_executable()
    tissue_type = args.tissue_type or args.tissue
    log_dir = paths.LOGS_DIR / "panel"

    for i, outs in enumerate(samples, 1):
        sample_name = outs.parent.name.replace(" ", "_")
        cmd: list[str | Path] = [
            krk, "annotate",
            "--xenium-dir", outs,
            "--tissue-type", tissue_type,
            "--output-dir", outs,
            "--use-graphclust",
            "--use-top-k-markers", str(args.top_k),
            "--n-jobs", str(args.n_jobs),
        ]
        if args.markers_csv:
            cmd += ["--markers-csv", args.markers_csv]
        if args.overwrite:
            cmd += ["--overwrite"]
        log_path = log_dir / f"{args.tissue}_{i:03d}_{sample_name}.log"
        print(f"[panel] ({i}/{len(samples)}) {outs}")
        sh(cmd, log_path=log_path, dry=args.dry_run)
    return 0
