"""`prepare` subcommand -- ready a tissue for training in one call.

Consolidates the old three-step Python flow:
    make_splits.py + compute_class_weights.py + make_train_config.py
into a single function with a single argparse namespace. Writes:

    trainingset/<tissue>/splits/<fold>/{train,val}.csv
    trainingset/<tissue>/train_configs/<backbone>/<fold>.yaml

The class-weight comments are embedded in the YAML, so the human-readable
"weight 0.524 (capped pre-rescale)" report lives next to the value that
actually drives training.
"""
from __future__ import annotations

import argparse

from . import config, paths, splits, weights


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tissue", required=True)
    p.add_argument("--backbone", default=paths.DEFAULT_BACKBONE)
    p.add_argument("--fold", default=paths.DEFAULT_FOLD)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--by-slide", action="store_true",
                   help="Slide-level holdout instead of per-tile shuffle.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task", default=paths.TEMPLATE_TISSUE,
                   help="Task tag embedded in log_comment.")
    p.add_argument("--cap", type=float, default=10.0,
                   help="Per-class weight cap before budget rescale.")
    p.add_argument("--skip-splits", action="store_true",
                   help="Reuse existing splits/<fold>/{train,val}.csv.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing train_configs/.../<fold>.yaml.")


def run(args: argparse.Namespace) -> int:
    # 1. Class weights (also gives us the label_map for the YAML render).
    rep = weights.compute_weights(args.tissue, cap=args.cap)
    print(f"[weights] n_total={rep.n_total:,}  n_classes={len(rep.weights)}  "
          f"sum={sum(rep.weights):.3f}")
    for line in weights.format_report_comments(rep, cap=args.cap):
        print(f"  # {line}")

    # 2. Splits.
    if args.skip_splits:
        sp_dir = paths.splits_dir(args.tissue, args.fold)
        if not (sp_dir / "train.csv").is_file():
            print(f"ERROR: --skip-splits set but {sp_dir}/train.csv missing")
            return 1
        print(f"[splits] reusing {sp_dir}")
    else:
        sp = splits.split_tiles(args.tissue, val_frac=args.val_frac,
                                by_slide=args.by_slide, seed=args.seed)
        sp_dir = paths.splits_dir(args.tissue, args.fold)
        splits.write_split(sp, sp_dir)
        print(f"[splits] mode={sp.mode}  train={len(sp.train)}  val={len(sp.val)}"
              f"  slides={sp.n_slides} -> {sp_dir}")
        if sp.mode == "slide-level":
            print(f"  train slides: {sp.train_slides}")
            print(f"  val slides  : {sp.val_slides}")

    # 3. Train config.
    out_path = paths.train_config_path(args.tissue, args.backbone, args.fold)
    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} exists; pass --force to overwrite")
        return 1
    rendered = config.render(args.tissue, backbone=args.backbone,
                             fold=args.fold, task=args.task,
                             weight_report=rep)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"[config] wrote {out_path}")
    return 0
