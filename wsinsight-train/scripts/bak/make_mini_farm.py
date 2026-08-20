"""Build a symlink farm holding N slides per tissue.

Relative paths are preserved so wsitrain derives the same sample_id and reuses
masks already segmented under the full-cohort output directory.

Usage:
    python scripts/make_mini_farm.py --input DIR --farm DIR [--per-tissue 1]
                                     [--masks DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wsitrain.dataset import discover_samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--farm", required=True, type=Path)
    ap.add_argument("--per-tissue", type=int, default=1)
    ap.add_argument("--masks", type=Path, default=None,
                    help="mask dir; slides already segmented there are preferred")
    args = ap.parse_args()

    if args.farm.exists():
        raise SystemExit(f"farm already exists, remove it first: {args.farm}")

    have = set()
    if args.masks and args.masks.is_dir():
        have = {p.stem for p in args.masks.glob("*.npy")}

    by_tissue: dict[str, list] = {}
    for s in discover_samples(args.input, "pantissue"):
        if s.aligned:
            by_tissue.setdefault(s.tissue, []).append(s)

    picked = []
    for tissue in sorted(by_tissue):
        # Already-segmented first, then disease slides, so the subset stays free
        # of extra GPU work while keeping a malignant case per tissue.
        cand = sorted(by_tissue[tissue],
                      key=lambda s: (s.sample_id not in have,
                                     "ancer" not in s.sample_id,
                                     s.sample_id))
        picked.extend(cand[:args.per_tissue])

    for s in picked:
        src = Path(s.outs).parent
        rel = src.relative_to(args.input)
        dst = args.farm / rel
        # The sample dir must be real: Path.rglob() refuses to descend into
        # symlinked directories, so only its contents are linked.
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            (dst / child.name).symlink_to(child)
        print(f"{'reuse' if s.sample_id in have else 'NEW  '}  {rel}")

    print(f"\n{len(picked)} slides across {len(by_tissue)} tissues -> {args.farm}")
    print(f"already segmented: {sum(s.sample_id in have for s in picked)}")


main()
