"""``run`` — end-to-end driver that chains every subcommand for one tissue.

Step order matches the README's "End-to-end pipeline (per tissue)" table:

    0. panel       — kurtorank annotate over every data/xenium/<tissue> sample
    1. detect      — QuPath QuST tissue + nucleus detection + cluster transfer
    2. label       — QuPath cluster_id → label-name remap (one CSV per sample)
    3. export      — QuPath detections → tile PNG + per-tile CSV
    4. prepare     — splits + class weights + fold YAML
    5. train       — train classifier head + TorchScript export
    6. validate    — confusion + classification report (already run by train.sh)
    7. audit       — split-reuse leakage check

``--from`` / ``--to`` bound the range. ``--skip`` removes individual steps.
``label`` is skipped automatically unless ``--assignment-csv`` is given since
the v1 chain pre-runs it offline; pass ``--label-from-panel`` to auto-locate
the CSV emitted by step 0 inside each sample's outs/ directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import delegators, paths, prepare, qupath, train, validate
from . import panel as panel_mod

_STEPS: tuple[str, ...] = (
    "panel", "detect", "label", "export",
    "prepare", "train", "validate", "audit",
)


def _step_range(start: str, end: str) -> list[str]:
    i, j = _STEPS.index(start), _STEPS.index(end)
    if i > j:
        raise SystemExit(f"--from {start} comes after --to {end}")
    return list(_STEPS[i : j + 1])


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--from", dest="from_step", choices=_STEPS,
                   default="panel")
    p.add_argument("--to", dest="to_step", choices=_STEPS, default="audit")
    p.add_argument("--skip", action="append", default=[], choices=_STEPS,
                   help="Skip a step (repeatable).")
    p.add_argument("--backbone", default=paths.DEFAULT_BACKBONE)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--task", default="pantissue")
    p.add_argument("--assignment-csv", type=Path, default=None,
                   help="Override CSV for the label step.")
    p.add_argument("--dry-run", action="store_true")


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def run(args: argparse.Namespace) -> int:
    todo = [s for s in _step_range(args.from_step, args.to_step)
            if s not in args.skip]
    print(f"[run] tissue={args.tissue}  steps={todo}")

    dry = args.dry_run

    if "panel" in todo:
        panel_mod.run(_ns(tissue=args.tissue, tissue_type=None, top_k=25,
                          n_jobs=8, markers_csv=None, overwrite=False,
                          limit=None, dry_run=dry))
    if "detect" in todo:
        qupath.run_detect(_ns(qproj=paths.QPROJ_PATH, dry_run=dry))
    if "label" in todo:
        if args.assignment_csv is None:
            print("[run] skipping `label` — pass --assignment-csv to enable.")
        else:
            qupath.run_label(_ns(qproj=paths.QPROJ_PATH,
                                 assignment_csv=args.assignment_csv,
                                 dry_run=dry))
    if "export" in todo:
        qupath.run_export(_ns(qproj=paths.QPROJ_PATH, tissue=args.tissue,
                              overlap=None, dry_run=dry))
    if "prepare" in todo:
        prepare.run(_ns(tissue=args.tissue, backbone=args.backbone,
                        fold=args.fold, val_frac=0.10, by_slide=False,
                        seed=42, task=args.task, cap=10.0,
                        skip_splits=False, force=True))
    if "train" in todo:
        train.run(_ns(tissue=args.tissue, backbone=args.backbone,
                      fold=args.fold, task=args.task, dry_run=dry))
    if "validate" in todo:
        validate.run(_ns(tissue=args.tissue, backbone=args.backbone,
                         fold=args.fold, task=args.task, run_dir=None,
                         dry_run=dry))
    if "audit" in todo:
        delegators.run_audit(_ns(tissue=args.tissue, fold=args.fold,
                                 out=None, dry_run=dry))
    return 0
