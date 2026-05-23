#!/usr/bin/env python3
"""
reannotate_xenium.py
--------------------
Drive `kurtorank annotate` across the full Xenium training tree under
`model-development/data/xenium/<tissue>/`.

Why: when the bundled marker panel `markers-v3.csv` changes (e.g. after the
DISCO-driven `build-panel` improvement), every previously-computed
`celltype_assignment_hne_label.csv` is stale, and the CellViT-training
pipeline (`build_cell_labels.py`) will train on outdated labels.

This driver:
  1. Walks `data/xenium/<tissue>/SOURCES.yaml` for each tissue dir.
  2. Resolves each sample's `outs/` path.
  3. Skips samples whose `annotated.h5ad` is newer than `markers-v3.csv`
     (unless `--force` is given).
  4. Runs `kurtorank annotate --xenium-dir <outs> --tissue-type <tissue>
     --overwrite` for the remaining samples.

Usage:
    python scripts/reannotate_xenium.py --dry-run
    python scripts/reannotate_xenium.py --tissues breast,colorectal
    python scripts/reannotate_xenium.py --parallel 1 --n-jobs 8 \
        --log-dir logs/reannotate

Notes:
- Each sample is a separate subprocess so a crash in one does not abort the
  whole sweep. The driver itself is serial across samples by default; set
  `--parallel N` to run N samples concurrently (each child still uses
  `--n-jobs` workers internally).
- This script never edits or deletes data. It only invokes the kurtorank
  CLI, which writes back into the sample's `outs/` directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
KURTORANK_ROOT = SCRIPT_DIR.parent
MODEL_DEV_ROOT = KURTORANK_ROOT.parent
DEFAULT_XENIUM_ROOT = MODEL_DEV_ROOT / "data" / "xenium"
DEFAULT_MARKERS_CSV = (
    KURTORANK_ROOT / "src" / "kurtorank" / "markers" / "data" / "markers-v3.csv"
)


def discover_samples(xenium_root: Path, tissues: list[str] | None) -> list[dict]:
    """Walk SOURCES.yaml manifests and return a flat sample list."""
    samples: list[dict] = []
    tissue_dirs = sorted(p for p in xenium_root.iterdir() if p.is_dir())
    for tdir in tissue_dirs:
        tissue = tdir.name
        if tissues and tissue not in tissues:
            continue
        manifest = tdir / "SOURCES.yaml"
        if not manifest.is_file():
            print(f"[warn] {tissue}: SOURCES.yaml missing, skipping", file=sys.stderr)
            continue
        with manifest.open() as fh:
            cfg = yaml.safe_load(fh) or {}
        for entry in cfg.get("samples", []):
            rel_outs = entry.get("relative_outs")
            if not rel_outs:
                continue
            outs = (tdir / rel_outs).resolve()
            samples.append(
                {
                    "tissue": tissue,
                    "outs": outs,
                    "name": entry.get("dataset_name", str(outs)),
                }
            )
    return samples


def needs_reannotation(outs: Path, markers_csv: Path) -> tuple[bool, str]:
    annotated = outs / "annotated.h5ad"
    if not outs.is_dir():
        return False, "outs/ missing"
    if not annotated.is_file():
        return True, "no annotated.h5ad"
    if not markers_csv.is_file():
        return True, "markers-v3.csv missing (rebuild anyway)"
    if annotated.stat().st_mtime < markers_csv.stat().st_mtime:
        return True, "annotated.h5ad older than markers-v3.csv"
    return False, "up-to-date"


def run_one(sample: dict, n_jobs: int, log_dir: Path | None, extra: list[str]) -> dict:
    tissue = sample["tissue"]
    outs = sample["outs"]
    cmd = [
        "kurtorank",
        "annotate",
        "--xenium-dir",
        str(outs),
        "--tissue-type",
        tissue,
        "--overwrite",
        "--n-jobs",
        str(n_jobs),
        *extra,
    ]
    started = time.time()
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe = "__".join([tissue, outs.parent.name]).replace("/", "_").replace(" ", "_")
        log_path = log_dir / f"{safe}.log"
        log_fh = log_path.open("w")
    else:
        log_fh = None
    try:
        proc = subprocess.run(
            cmd,
            stdout=log_fh or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        rc = proc.returncode
        tail = (proc.stdout or "").splitlines()[-5:] if log_fh is None else []
    finally:
        if log_fh is not None:
            log_fh.close()
    return {
        "sample": sample,
        "rc": rc,
        "elapsed_s": time.time() - started,
        "log_path": str(log_path) if log_path else None,
        "tail": tail,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xenium-root", type=Path, default=DEFAULT_XENIUM_ROOT)
    ap.add_argument("--markers-csv", type=Path, default=DEFAULT_MARKERS_CSV,
                    help="Reference panel CSV; samples older than this are re-annotated.")
    ap.add_argument("--tissues", default="",
                    help="Comma-separated tissue subset (e.g. breast,colorectal). Default: all.")
    ap.add_argument("--force", action="store_true",
                    help="Re-annotate every sample regardless of mtime.")
    ap.add_argument("--dry-run", action="store_true",
                    help="List planned work and exit without invoking kurtorank.")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Number of samples to run concurrently (each uses --n-jobs internally).")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="Passed to `kurtorank annotate --n-jobs` for each sample.")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Write per-sample logs here. Default: stream to stdout.")
    ap.add_argument("kurtorank_args", nargs=argparse.REMAINDER,
                    help="Extra args forwarded verbatim to `kurtorank annotate` "
                         "(prefix with `--`, e.g. `-- --use-leiden --no-generate-plots`).")
    args = ap.parse_args()

    extra: list[str] = []
    if args.kurtorank_args:
        # argparse keeps the leading `--` separator; drop it.
        extra = [a for a in args.kurtorank_args if a != "--"]

    tissues = [t.strip() for t in args.tissues.split(",") if t.strip()] or None

    samples = discover_samples(args.xenium_root, tissues)
    if not samples:
        print("No samples discovered. Check --xenium-root / --tissues.", file=sys.stderr)
        return 2

    plan: list[dict] = []
    skipped: list[dict] = []
    for s in samples:
        need, reason = needs_reannotation(s["outs"], args.markers_csv)
        if args.force or need:
            plan.append({**s, "reason": "forced" if args.force else reason})
        else:
            skipped.append({**s, "reason": reason})

    print(f"Discovered {len(samples)} samples; {len(plan)} need re-annotation, "
          f"{len(skipped)} up-to-date.")
    print(f"Reference markers: {args.markers_csv} "
          f"(mtime={time.ctime(args.markers_csv.stat().st_mtime) if args.markers_csv.exists() else 'missing'})")
    print()
    by_tissue: dict[str, int] = {}
    for s in plan:
        by_tissue[s["tissue"]] = by_tissue.get(s["tissue"], 0) + 1
    for t, n in sorted(by_tissue.items()):
        print(f"  {t:14s} {n} sample(s) queued")
    print()

    if args.dry_run:
        print("--dry-run: not invoking kurtorank.")
        for s in plan:
            print(f"  [{s['tissue']}] {s['outs']}  ({s['reason']})")
        return 0

    if not plan:
        print("Nothing to do.")
        return 0

    failures: list[dict] = []
    if args.parallel <= 1:
        for i, s in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] {s['tissue']}: {s['outs']}")
            res = run_one(s, args.n_jobs, args.log_dir, extra)
            status = "OK" if res["rc"] == 0 else f"FAIL rc={res['rc']}"
            print(f"    {status} in {res['elapsed_s']:.0f}s"
                  + (f" -> {res['log_path']}" if res["log_path"] else ""))
            if res["rc"] != 0:
                failures.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(run_one, s, args.n_jobs, args.log_dir, extra): s for s in plan}
            done = 0
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                s = res["sample"]
                status = "OK" if res["rc"] == 0 else f"FAIL rc={res['rc']}"
                print(f"[{done}/{len(plan)}] {s['tissue']}: {s['outs']} -- {status} "
                      f"in {res['elapsed_s']:.0f}s"
                      + (f" -> {res['log_path']}" if res["log_path"] else ""))
                if res["rc"] != 0:
                    failures.append(res)

    print()
    print(f"Done. {len(plan) - len(failures)} ok, {len(failures)} failed.")
    for res in failures:
        s = res["sample"]
        print(f"  FAIL [{s['tissue']}] {s['outs']} rc={res['rc']} log={res['log_path']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
