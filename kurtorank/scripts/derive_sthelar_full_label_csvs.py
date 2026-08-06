"""Derive celltype_assignment_sthelar_full_label.csv from each sample's
celltype_assignment_subtype.csv using the markers-v6.csv crosswalk.

`kurtorank annotate` now writes the STHELAR label CSVs itself, so this script is
only needed to backfill samples annotated before that change (re-running
annotate on them would be far slower).

The per-cluster `subtype` -> `sthelar_full_label` mapping is 1:1 in
markers-v6.csv (unlike hne_label, which fans out for lymphocyte /
hematologic_blast), so we remap the subtype CSV's `cell_type` column through it.
Cluster-level assignment is unchanged; only the label vocabulary is collapsed.

Usage:
    python derive_sthelar_full_label_csvs.py --data-dir DIR [--markers-csv CSV]
                                             [--exclude heart] [--overwrite]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_PKG_MARKERS = (Path(__file__).resolve().parent.parent
                / "src/kurtorank/markers/data/markers-v6.csv")


def build_subtype_to_sthelar_map(markers_csv: Path) -> dict[str, str]:
    df = pd.read_csv(markers_csv)
    for col in ("subtype", "sthelar_full_label"):
        if col not in df.columns:
            raise SystemExit(f"{markers_csv} lacks required column {col!r}")
    pairs = df[["subtype", "sthelar_full_label"]].dropna().drop_duplicates()
    dup = pairs["subtype"].value_counts()
    bad = dup[dup > 1]
    if len(bad):
        raise SystemExit(
            f"subtype maps to multiple sthelar_full_label values: {bad.to_dict()}")
    return dict(zip(pairs["subtype"], pairs["sthelar_full_label"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Root to recurse for celltype_assignment_subtype.csv files.")
    ap.add_argument("--markers-csv", type=Path, default=_PKG_MARKERS,
                    help="markers-v6.csv with the subtype->sthelar_full_label crosswalk.")
    ap.add_argument("--exclude", action="append", default=[],
                    help="Top-level tissue folder name(s) to skip (repeatable).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite existing celltype_assignment_sthelar_full_label.csv.")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    mapping = build_subtype_to_sthelar_map(args.markers_csv)
    print(f"Loaded {len(mapping)} subtype -> sthelar_full_label entries "
          f"from {args.markers_csv.name}")

    excluded = {e for e in args.exclude}
    src_files = sorted(data_dir.rglob("celltype_assignment_subtype.csv"))
    if excluded:
        def _keep(p: Path) -> bool:
            rel = p.relative_to(data_dir).as_posix()
            return rel.split("/", 1)[0] not in excluded
        src_files = [p for p in src_files if _keep(p)]
    print(f"Found {len(src_files)} celltype_assignment_subtype.csv file(s) "
          f"under {data_dir}")

    unknown_total: dict[str, int] = {}
    written = skipped = 0
    for src in src_files:
        dst = src.with_name("celltype_assignment_sthelar_full_label.csv")
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        df = pd.read_csv(src)
        unknown = sorted(set(df["cell_type"]) - set(mapping))
        for u in unknown:
            unknown_total[u] = unknown_total.get(u, 0) + 1
        df["cell_type"] = df["cell_type"].map(mapping).fillna(df["cell_type"])
        df.to_csv(dst, index=False)
        written += 1
        rel = src.relative_to(data_dir).parent
        print(f"  [{written:>3}] {rel}  ({len(df)} clusters)")

    print(f"\nWrote {written} file(s); skipped {skipped} existing "
          f"(use --overwrite to rebuild).")
    if unknown_total:
        print("\nWARNING: subtype values not in markers-v6.csv (left unmapped):")
        for k, v in sorted(unknown_total.items()):
            print(f"  {k!r} appeared in {v} file(s)")


if __name__ == "__main__":
    main()
