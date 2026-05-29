#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# audit_all_tissues.sh -- run audit_split_reuse.py for every tissue that has a
# populated splits/<fold>/ directory under trainingset/. Aggregates per-tissue
# JSON reports and writes a dataset-wide summary CSV.
#
# Usage:
#   bash audit_all_tissues.sh                       # fold_0, default out dir
#   bash audit_all_tissues.sh --fold fold_0
#   bash audit_all_tissues.sh --tissues "breast colorectal"
#   bash audit_all_tissues.sh --out audit_outputs/
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "${SCRIPT_DIR}/_lib.sh"
DEFAULT_ROOT="$(_lib::trainingset_root)"
TRAININGSET="${DEFAULT_ROOT}"

FOLD="fold_0"
OUT="${SCRIPT_DIR}/audit_outputs"
TISSUES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold)              FOLD="$2"; shift 2 ;;
        --out)               OUT="$2"; shift 2 ;;
        --tissues)           TISSUES="$2"; shift 2 ;;
        --trainingset-root)  TRAININGSET="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${OUT}"

# Discover tissues if not explicit.
if [[ -z "${TISSUES}" ]]; then
    if [[ "${TRAININGSET}" == "${DEFAULT_ROOT}" ]]; then
        TISSUES="$(_lib::tissues_with_splits "${FOLD}" include)"
    else
        # Custom trainingset root (e.g. synthetic-audit fixture) -- enumerate inline.
        TISSUES=""
        for d in "${TRAININGSET}"/*/; do
            t="$(basename "$d")"
            if [[ -d "${d}splits/${FOLD}" && -f "${d}splits/${FOLD}/val.csv" ]]; then
                TISSUES+=" $t"
            fi
        done
        TISSUES="${TISSUES# }"
    fi
fi

if [[ -z "${TISSUES}" ]]; then
    echo "ERROR: no tissues with populated splits/${FOLD}/ found under ${TRAININGSET}" >&2
    exit 1
fi

echo "Tissues : ${TISSUES}"
echo "Fold    : ${FOLD}"
echo "Out dir : ${OUT}"
echo ""

# Run per-tissue audit. Tolerate per-tissue failure (e.g. missing geometry).
for t in ${TISSUES}; do
    echo "── Auditing ${t} ──"
    if ! python "${SCRIPT_DIR}/audit_split_reuse.py" --tissue "${t}" --fold "${FOLD}" --out "${OUT}" --trainingset-root "${TRAININGSET}"; then
        echo "WARN: audit failed for ${t}; continuing." >&2
    fi
    echo ""
done

# Aggregate per-tissue JSONs → one summary CSV.
SUMMARY="${OUT}/split_reuse_summary.csv"
python - "${OUT}" "${FOLD}" "${SUMMARY}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
fold = sys.argv[2]
summary_path = Path(sys.argv[3])

rows = []
totals = {"pixel_total_val": 0, "pixel_total_reused": 0,
          "cell_total_val_unique": 0, "cell_total_reused": 0}
for jf in sorted(out_dir.glob(f"split_reuse_*_{fold}.json")):
    with jf.open() as f:
        d = json.load(f)
    rows.append({
        "tissue": d["tissue"],
        "fold": d["fold"],
        "overlap_ratio": d["overlap_ratio_used"],
        "n_samples": d["n_samples_in_split"],
        "n_train_tiles": d["n_train_stems_total"],
        "n_val_tiles": d["n_val_stems_total"],
        "pixel_val": d["pixel_total_val"],
        "pixel_reused": d["pixel_total_reused"],
        "pct_pixel": round(d["pct_pixel_reused"], 3),
        "cell_val_unique": d["cell_total_val_unique"],
        "cell_reused": d["cell_total_reused"],
        "pct_cell": round(d["pct_cell_reused"], 3),
    })
    for k in totals:
        v = d.get(k, 0)
        if isinstance(v, (int, float)):
            totals[k] += v

# Sort by pct_cell descending so worst-affected surfaces first.
rows.sort(key=lambda r: r["pct_cell"], reverse=True)

# Append dataset-wide aggregate row.
pct_pixel_total = (100.0 * totals["pixel_total_reused"] / totals["pixel_total_val"]) \
    if totals["pixel_total_val"] else 0.0
pct_cell_total = (100.0 * totals["cell_total_reused"] / totals["cell_total_val_unique"]) \
    if totals["cell_total_val_unique"] else 0.0
rows.append({
    "tissue": "__DATASET_TOTAL__",
    "fold": fold,
    "overlap_ratio": "—",
    "n_samples": "",
    "n_train_tiles": "",
    "n_val_tiles": "",
    "pixel_val": totals["pixel_total_val"],
    "pixel_reused": totals["pixel_total_reused"],
    "pct_pixel": round(pct_pixel_total, 3),
    "cell_val_unique": totals["cell_total_val_unique"],
    "cell_reused": totals["cell_total_reused"],
    "pct_cell": round(pct_cell_total, 3),
})

if rows:
    with summary_path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n=== Summary ({len(rows)-1} tissues) ===")
    # Print a compact table.
    cols = ["tissue", "overlap_ratio", "n_train_tiles", "n_val_tiles",
            "pct_pixel", "pct_cell"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"\nWritten: {summary_path}")
else:
    print("No per-tissue JSON reports found; summary skipped.")
PY
