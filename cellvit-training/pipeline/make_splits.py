"""
make_splits.py
--------------
Build train/val splits from exported tiles, grouped by slide (SAMPLE_TAG).

Tiles produced by qupath/export_tiles.groovy are named
    <SAMPLE_TAG>_tile_<NNNNN>[ _<aug>].png/.csv
where SAMPLE_TAG identifies the source slide. We split at the SAMPLE_TAG
level (slide-level holdout), not the tile level, so tiles from one slide
never appear in both train and val. This matters especially when overlap
is enabled at export time.

Single-slide tissues (heart, brain, cervix, prostate, lymph_node) fall back
to per-tile shuffle with a WARN: there is no true slide-level holdout when
only one slide exists.

Run AFTER export_tiles.groovy has finished for all samples of the target
tissue.

Usage:
    python make_splits.py --tissue breast
    python make_splits.py --tissue colorectal --fold fold_0 --val-frac 0.2
"""

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent

# Tile-suffix pattern: matches "_tile_00042" optionally followed by an
# augmentation tag like "_hflip", "_rot90". Everything before this suffix is
# the SAMPLE_TAG.
_TILE_SUFFIX_RE = re.compile(r"_tile_\d+(?:_[a-z0-9]+)?$")


def _sample_tag(stem: str) -> str:
    return _TILE_SUFFIX_RE.sub("", stem)


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
        f[:-4] for f in os.listdir(label_dir) if f.endswith(".csv")
    )
    if not tiles:
        raise SystemExit(f"ERROR: no .csv files under {label_dir}")
    print(f"Total tiles: {len(tiles)}")

    # Group tiles by SAMPLE_TAG (slide stem).
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in tiles:
        groups[_sample_tag(stem)].append(stem)
    group_names = sorted(groups)
    print(f"Slides (SAMPLE_TAGs): {len(group_names)}")

    rng = random.Random(args.seed)

    if len(group_names) <= 1:
        # Single-slide tissue → no true slide-level holdout possible.
        # Fall back to per-tile shuffle so val gets non-empty spatial holdout.
        print(
            f"WARN: only {len(group_names)} slide for tissue '{args.tissue}'. "
            f"Falling back to per-tile shuffle — val is same-slide spatial "
            f"holdout, NOT a true slide-level holdout."
        )
        shuffled = list(tiles)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * args.val_frac))
        val_tiles = shuffled[:n_val]
        train_tiles = shuffled[n_val:]
    else:
        shuffled_groups = list(group_names)
        rng.shuffle(shuffled_groups)
        n_val_groups = max(1, int(round(len(shuffled_groups) * args.val_frac)))
        # Guarantee at least one train group remains.
        n_val_groups = min(n_val_groups, len(shuffled_groups) - 1)
        val_groups = set(shuffled_groups[:n_val_groups])
        train_groups = set(shuffled_groups[n_val_groups:])

        val_tiles = sorted(t for g in val_groups for t in groups[g])
        train_tiles = sorted(t for g in train_groups for t in groups[g])
        print(
            f"Group split: train_groups={len(train_groups)} "
            f"val_groups={len(val_groups)}"
        )
        print(f"  train slides: {sorted(train_groups)}")
        print(f"  val slides  : {sorted(val_groups)}")

    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.csv").write_text("\n".join(train_tiles) + "\n")
    (splits_dir / "val.csv").write_text("\n".join(val_tiles) + "\n")

    print(f"train: {len(train_tiles)}  val: {len(val_tiles)}")
    print(f"Written to {splits_dir}/")


if __name__ == "__main__":
    main()
