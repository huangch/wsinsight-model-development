"""Derive celltype_assignment_hne_label.csv from existing
celltype_assignment_subtype.csv files using the markers-v3_2.csv mapping.

This avoids re-running kurtorank annotate when only the (subtype -> hne_label)
mapping in markers-v3_2.csv has been updated. Per-cluster subtype assignment
is preserved on disk by kurtorank.annotate; we just re-collapse the label
vocabulary subtype -> hne_label.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/wsinsight/model-development")
MARKERS_CSV = ROOT / "kurtorank/src/kurtorank/markers/data/markers-v3_2.csv"


def build_subtype_to_hne_label_map() -> dict[str, str]:
    df = pd.read_csv(MARKERS_CSV)
    pairs = df[["subtype", "hne_label"]].drop_duplicates()
    dup = pairs["subtype"].value_counts()
    bad = dup[dup > 1]
    if len(bad):
        raise SystemExit(f"subtype maps to multiple hne_label values: {bad.to_dict()}")
    return dict(zip(pairs["subtype"], pairs["hne_label"]))


def main() -> None:
    mapping = build_subtype_to_hne_label_map()
    print(f"Loaded {len(mapping)} subtype -> hne_label entries from markers-v3_2.csv")

    src_files = sorted(ROOT.glob("data/**/celltype_assignment_subtype.csv"))
    print(f"Found {len(src_files)} celltype_assignment_subtype.csv files")

    unknown_total: dict[str, int] = {}
    written = 0
    for src in src_files:
        df = pd.read_csv(src)
        unknown = sorted(set(df["cell_type"]) - set(mapping))
        for u in unknown:
            unknown_total[u] = unknown_total.get(u, 0) + 1
        df["cell_type"] = df["cell_type"].map(mapping).fillna(df["cell_type"])
        dst = src.with_name("celltype_assignment_hne_label.csv")
        df.to_csv(dst, index=False)
        written += 1
        rel = src.relative_to(ROOT)
        print(f"  [{written:>2}/{len(src_files)}] {rel.parent}  ({len(df)} clusters)")

    print(f"\nWrote {written} celltype_assignment_hne_label.csv files.")
    if unknown_total:
        print("\nWARNING: subtype values not in markers-v3_2.csv (left unmapped):")
        for k, v in sorted(unknown_total.items()):
            print(f"  {k!r} appeared in {v} file(s)")


if __name__ == "__main__":
    main()
