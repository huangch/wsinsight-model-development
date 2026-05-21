"""
make_splits.py
--------------
Build train/val splits from exported tiles.

Run AFTER export_tiles.groovy has finished for all
samples of the target tissue.

Usage:
    python make_splits.py --tissue breast
    python make_splits.py --tissue colorectal --fold fold_0 --val-frac 0.2
"""

import argparse
import os
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tissue", required=True, help="Tissue name (e.g. breast, colorectal)")
    p.add_argument("--fold", default="fold_0", help="Fold name (default: fold_0)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tissue_root = CELLVIT_TRAINING_ROOT / "trainingset" / args.tissue
    label_dir = tissue_root / "train" / "labels"
    splits_dir = tissue_root / "splits" / args.fold

    if not label_dir.is_dir():
        raise SystemExit(f"ERROR: label dir not found: {label_dir}")

    tiles = sorted(
        f.replace(".csv", "") for f in os.listdir(label_dir) if f.endswith(".csv")
    )
    print(f"Total tiles: {len(tiles)}")

    random.seed(args.seed)
    random.shuffle(tiles)
    n_val = int(len(tiles) * args.val_frac)
    val_tiles = tiles[:n_val]
    train_tiles = tiles[n_val:]

    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("\n".join(train_tiles) + "\n")
    (splits_dir / "val.csv").write_text("\n".join(val_tiles) + "\n")

    print(f"train: {len(train_tiles)}  val: {len(val_tiles)}")
    print(f"Written to {splits_dir}/")


if __name__ == "__main__":
    main()
