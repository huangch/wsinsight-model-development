"""
audit_split_reuse.py
--------------------
Audit the train/val split written by `make_splits.py` for label/pixel reuse.

Quantifies, per tissue and dataset-wide:

  * Cell reuse:  # of (centroid, label) pairs that appear in BOTH a train
                 tile's CSV and a val tile's CSV. With overlap > 0 the
                 export_tiles.groovy bucketing rule writes a single cell
                 into up to 4 tiles' CSVs (cells in the overlap margin),
                 so per-tile-shuffle splits CAN duplicate the same
                 supervised pair across the train/val boundary.

  * Pixel reuse: area of every val tile's slide-coord bbox that is also
                 covered by at least one train tile's bbox on the same
                 slide.

Cell reuse is the textbook label-leakage criterion. Pixel reuse is the
weaker "context overlap" criterion. Both are computed exactly from the
artifacts (`tile_geometry/*.json`, `splits/<fold>/{train,val}.csv`,
per-tile label CSVs) without any model inference.

Usage:
    python audit_split_reuse.py --tissue breast
    python audit_split_reuse.py --tissue breast --fold fold_0
    python audit_split_reuse.py --tissue breast --out audit_outputs/

Prerequisites:
  * `train/tile_geometry/<sample>.json` must exist for every sample
    appearing in `splits/<fold>/{train,val}.csv`. If missing, run
    `qupath/dump_tile_geometry.groovy` once to backfill.

Output:
  * `<out>/split_reuse_<tissue>_<fold>.json` — full per-class breakdown.
  * Human-readable summary table to stdout.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent

# Tile filename:  <SAMPLE_TAG>_tile_<NNNNN>[ _<aug>]
# Augmented variants share the same slide-coord bbox as their base tile,
# but their cell coordinates have been transformed (flip/rot) — they
# would inflate cell counts if matched naively, so we exclude them from
# the audit. The base tile of every aug variant is in the same split as
# the variant (make_splits.py shuffles all tile stems together), so this
# is a representative-sampling choice, not a coverage loss.
_BASE_STEM_RE = re.compile(r"^(?P<sample>.+)_tile_(?P<idx>\d{5})$")
_AUG_SUFFIX_RE = re.compile(r"_tile_\d{5}_[a-z0-9]+$")

# Cell-identity rounding: slide-µm coords are rounded to this many decimal
# places before being used as a hashable key. 0.5 µm is well below
# typical nucleus-to-nucleus distance (~5-10 µm), so this won't merge
# distinct cells. It absorbs the ±0.125 µm rounding noise that
# export_tiles.groovy's `Math.round((c.x - ox) / 0.25)` introduces.
_CELL_ROUND_UM = 0.5  # round to nearest 0.5 µm


def _parse_stem(stem: str):
    """Return (sample_tag, tile_idx_int) for a base tile stem, or None."""
    if _AUG_SUFFIX_RE.search(stem):
        return None  # augmented variant; skip
    m = _BASE_STEM_RE.match(stem)
    if m is None:
        return None
    return m.group("sample"), int(m.group("idx"))


def _load_geometry(tissue_root: Path) -> dict:
    geom_dir = tissue_root / "train" / "tile_geometry"
    if not geom_dir.is_dir():
        raise SystemExit(
            f"ERROR: {geom_dir} not found. Run qupath/dump_tile_geometry.groovy "
            f"to backfill geometry sidecars for already-exported slides."
        )
    geom = {}
    for jf in sorted(geom_dir.glob("*.json")):
        with jf.open() as f:
            data = json.load(f)
        geom[data["sample_tag"]] = data
    if not geom:
        raise SystemExit(f"ERROR: no *.json under {geom_dir}")
    return geom


def _tile_bbox_slide_px(tile_idx: int, g: dict) -> tuple:
    """Return (x0, y0, x1, y1) slide-pixel bbox for the given tileIdx
    (1-based, as emitted by export_tiles.groovy).

    Mirrors the groovy:
        for row in 0..nRows-1:
          for col in 0..nCols-1:
            tileIdx += 1     # incremented BEFORE skip checks
            ox_µm = col * STRIDE_UM
            oy_µm = row * STRIDE_UM
            rx_slide_px = round(ox_µm / slideMPP)
            ry_slide_px = round(oy_µm / slideMPP)
            rw = round(TILE_UM / slideMPP)
        with edge tiles clamped to slide extent.
    """
    n_cols = g["n_cols"]
    slide_mpp = g["slide_mpp"]
    export_mpp = g["export_mpp"]
    tile_px = g["tile_px"]
    stride_px = g["stride_px"]
    stride_um = stride_px * export_mpp
    tile_um = tile_px * export_mpp

    idx0 = tile_idx - 1
    row = idx0 // n_cols
    col = idx0 % n_cols

    ox_um = col * stride_um
    oy_um = row * stride_um
    rx = round(ox_um / slide_mpp)
    ry = round(oy_um / slide_mpp)
    rw = round(tile_um / slide_mpp)
    rh = round(tile_um / slide_mpp)

    # Clamp to slide extent (matches groovy edge-tile handling).
    x1 = min(rx + rw, g["slide_width_px"])
    y1 = min(ry + rh, g["slide_height_px"])
    x0 = min(rx, g["slide_width_px"])
    y0 = min(ry, g["slide_height_px"])
    return (x0, y0, x1, y1)


def _tile_origin_um(tile_idx: int, g: dict) -> tuple:
    """Return (ox_µm, oy_µm) for the given tileIdx — needed to lift
    tile-local cell coords back to slide-µm coords."""
    n_cols = g["n_cols"]
    stride_um = g["stride_px"] * g["export_mpp"]
    idx0 = tile_idx - 1
    row = idx0 // n_cols
    col = idx0 % n_cols
    return col * stride_um, row * stride_um


def _read_split(splits_dir: Path, name: str) -> list:
    f = splits_dir / f"{name}.csv"
    if not f.exists():
        raise SystemExit(f"ERROR: {f} not found")
    stems = [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]
    return stems


def _load_cells_for_stem(label_dir: Path, stem: str, ox_um: float,
                         oy_um: float, export_mpp: float) -> list:
    """Return list of (x_µm, y_µm, cls) for cells in this tile, in
    SLIDE-COORDS. Returns [] if the CSV is missing or empty."""
    csv_path = label_dir / f"{stem}.csv"
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                cpx = int(row[0]); cpy = int(row[1]); cls = int(row[2])
            except ValueError:
                continue
            x_um = ox_um + cpx * export_mpp
            y_um = oy_um + cpy * export_mpp
            out.append((x_um, y_um, cls))
    return out


def _round_key(x_um: float, y_um: float, cls: int) -> tuple:
    q = _CELL_ROUND_UM
    return (round(x_um / q) * q, round(y_um / q) * q, cls)


def _rect_union_area(rects: list) -> int:
    """Area of the union of axis-aligned rectangles via the
    sweep-line + coordinate-compression algorithm. Inputs are
    (x0, y0, x1, y1) ints. Returns total covered pixel area as int."""
    if not rects:
        return 0
    # Coordinate compression on Y.
    ys = sorted({y for r in rects for y in (r[1], r[3])})
    y_to_idx = {y: i for i, y in enumerate(ys)}

    # For each Y-strip, sweep X events and sum covered length.
    total = 0
    for yi in range(len(ys) - 1):
        y0 = ys[yi]; y1 = ys[yi + 1]
        strip_h = y1 - y0
        if strip_h <= 0:
            continue
        # Collect [x0, x1) intervals for rects covering this strip.
        ivs = [(r[0], r[2]) for r in rects if r[1] <= y0 and r[3] >= y1]
        if not ivs:
            continue
        ivs.sort()
        covered = 0
        cur_x0, cur_x1 = ivs[0]
        for a, b in ivs[1:]:
            if a > cur_x1:
                covered += cur_x1 - cur_x0
                cur_x0, cur_x1 = a, b
            else:
                cur_x1 = max(cur_x1, b)
        covered += cur_x1 - cur_x0
        total += covered * strip_h
    return total


def _rect_intersection_area(rects_a: list, rects_b: list) -> int:
    """Area of (union of A) ∩ (union of B). Computed as
    area(A) + area(B) − area(A ∪ B) using the inclusion-exclusion
    identity: |A ∩ B| = |A| + |B| − |A ∪ B|."""
    return (_rect_union_area(rects_a)
            + _rect_union_area(rects_b)
            - _rect_union_area(rects_a + rects_b))


def audit(tissue: str, fold: str, out_dir: Path,
          trainingset_root: Path = None) -> dict:
    root = trainingset_root or (CELLVIT_TRAINING_ROOT / "trainingset")
    tissue_root = root / tissue
    splits_dir = tissue_root / "splits" / fold
    label_dir = tissue_root / "train" / "labels"

    if not splits_dir.is_dir():
        raise SystemExit(f"ERROR: {splits_dir} not found. Run make_splits.py first.")
    if not label_dir.is_dir():
        raise SystemExit(f"ERROR: {label_dir} not found.")

    geom = _load_geometry(tissue_root)
    train_stems = _read_split(splits_dir, "train")
    val_stems = _read_split(splits_dir, "val")
    n_train_total = len(train_stems)
    n_val_total = len(val_stems)

    # Group stems by sample_tag, drop augmented variants.
    train_by_sample: dict[str, list[int]] = defaultdict(list)
    val_by_sample: dict[str, list[int]] = defaultdict(list)
    n_train_aug_dropped = 0
    n_val_aug_dropped = 0
    for stem in train_stems:
        p = _parse_stem(stem)
        if p is None:
            n_train_aug_dropped += 1
            continue
        train_by_sample[p[0]].append(p[1])
    for stem in val_stems:
        p = _parse_stem(stem)
        if p is None:
            n_val_aug_dropped += 1
            continue
        val_by_sample[p[0]].append(p[1])

    # ────────── Pixel-reuse computation ──────────
    pixel_total_val = 0
    pixel_total_reused = 0
    per_sample_pixel = {}
    overlap_used = None  # detect mixed-overlap exports

    all_samples = sorted(set(train_by_sample) | set(val_by_sample))
    for sample in all_samples:
        if sample not in geom:
            print(f"WARN: no geometry sidecar for sample '{sample}' — skipped.")
            continue
        g = geom[sample]
        if overlap_used is None:
            overlap_used = g["overlap_ratio"]
        elif overlap_used != g["overlap_ratio"]:
            overlap_used = "mixed"

        train_rects = [_tile_bbox_slide_px(i, g) for i in train_by_sample.get(sample, [])]
        val_rects = [_tile_bbox_slide_px(i, g) for i in val_by_sample.get(sample, [])]

        a_val = _rect_union_area(val_rects)
        a_reused = _rect_intersection_area(train_rects, val_rects) if val_rects and train_rects else 0
        pixel_total_val += a_val
        pixel_total_reused += a_reused
        per_sample_pixel[sample] = {
            "val_px": a_val,
            "reused_px": a_reused,
            "n_train_tiles": len(train_rects),
            "n_val_tiles": len(val_rects),
        }

    pct_pixel = (100.0 * pixel_total_reused / pixel_total_val) if pixel_total_val else 0.0

    # ────────── Cell-reuse computation ──────────
    train_cells_by_sample: dict[str, set] = defaultdict(set)
    val_cells_per_sample_list: dict[str, list] = defaultdict(list)
    val_cells_count_by_cls: dict[int, int] = defaultdict(int)
    val_cells_total = 0

    for sample in all_samples:
        if sample not in geom:
            continue
        g = geom[sample]
        export_mpp = g["export_mpp"]
        for tidx in train_by_sample.get(sample, []):
            ox, oy = _tile_origin_um(tidx, g)
            for x_um, y_um, cls in _load_cells_for_stem(
                    label_dir, f"{sample}_tile_{tidx:05d}", ox, oy, export_mpp):
                train_cells_by_sample[sample].add(_round_key(x_um, y_um, cls))
        for tidx in val_by_sample.get(sample, []):
            ox, oy = _tile_origin_um(tidx, g)
            for x_um, y_um, cls in _load_cells_for_stem(
                    label_dir, f"{sample}_tile_{tidx:05d}", ox, oy, export_mpp):
                val_cells_per_sample_list[sample].append((x_um, y_um, cls))

    # Deduplicate val cells per-sample (same cell can be in up to 4 val
    # tiles' CSVs due to bucketing) and count reuse against train set
    # of the same sample.
    cell_total_val_unique = 0
    cell_total_reused = 0
    reused_by_cls: dict[int, int] = defaultdict(int)
    unique_val_by_cls: dict[int, int] = defaultdict(int)

    for sample, cells in val_cells_per_sample_list.items():
        seen = set()
        for x_um, y_um, cls in cells:
            k = _round_key(x_um, y_um, cls)
            if k in seen:
                continue
            seen.add(k)
            cell_total_val_unique += 1
            unique_val_by_cls[cls] += 1
            val_cells_count_by_cls[cls] += 1
            if k in train_cells_by_sample.get(sample, ()):
                cell_total_reused += 1
                reused_by_cls[cls] += 1

    # Also count raw (non-deduplicated) val cell row total — useful for
    # cross-checking against `wc -l` on val CSVs.
    val_cells_raw_rows = sum(len(v) for v in val_cells_per_sample_list.values())

    pct_cell = (100.0 * cell_total_reused / cell_total_val_unique) if cell_total_val_unique else 0.0

    # ────────── Assemble report ──────────
    result = {
        "tissue": tissue,
        "fold": fold,
        "overlap_ratio_used": overlap_used,
        "n_train_stems_total": n_train_total,
        "n_val_stems_total": n_val_total,
        "n_train_aug_dropped": n_train_aug_dropped,
        "n_val_aug_dropped": n_val_aug_dropped,
        "n_samples_in_split": len(all_samples),
        "n_samples_in_train": len(train_by_sample),
        "n_samples_in_val": len(val_by_sample),
        "pixel_total_val": pixel_total_val,
        "pixel_total_reused": pixel_total_reused,
        "pct_pixel_reused": pct_pixel,
        "cell_total_val_unique": cell_total_val_unique,
        "cell_total_val_raw_rows": val_cells_raw_rows,
        "cell_total_reused": cell_total_reused,
        "pct_cell_reused": pct_cell,
        "per_class_breakdown": [
            {
                "cls": cls,
                "val_unique": unique_val_by_cls.get(cls, 0),
                "reused": reused_by_cls.get(cls, 0),
                "pct_reused": (100.0 * reused_by_cls.get(cls, 0) / unique_val_by_cls[cls])
                              if unique_val_by_cls.get(cls, 0) else 0.0,
            }
            for cls in sorted(unique_val_by_cls)
        ],
        "per_sample_pixel": per_sample_pixel,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"split_reuse_{tissue}_{fold}.json"
    out_file.write_text(json.dumps(result, indent=2))

    # ────────── Stdout summary ──────────
    print(f"=== audit_split_reuse: {tissue} / {fold} ===")
    print(f"overlap_ratio     : {overlap_used}")
    print(f"samples           : {len(all_samples)} "
          f"(train={len(train_by_sample)} val={len(val_by_sample)})")
    print(f"tiles  train/val  : {n_train_total} / {n_val_total}  "
          f"(aug dropped: {n_train_aug_dropped}/{n_val_aug_dropped})")
    print(f"")
    print(f"PIXEL REUSE")
    print(f"  val pixels      : {pixel_total_val:>15,d}")
    print(f"  reused pixels   : {pixel_total_reused:>15,d}")
    print(f"  % reused        : {pct_pixel:>14.2f} %")
    print(f"")
    print(f"CELL REUSE  (textbook label-leakage criterion)")
    print(f"  val cells raw   : {val_cells_raw_rows:>15,d}  (row count across val CSVs)")
    print(f"  val cells unique: {cell_total_val_unique:>15,d}  (after de-dup by slide-µm key)")
    print(f"  reused cells    : {cell_total_reused:>15,d}  (also in some train CSV of same sample)")
    print(f"  % reused        : {pct_cell:>14.2f} %")
    if result["per_class_breakdown"]:
        print(f"")
        print(f"  by class:")
        print(f"    cls  val_unique   reused   %")
        for row in result["per_class_breakdown"]:
            print(f"    {row['cls']:>3d}  {row['val_unique']:>10,d}  {row['reused']:>7,d}  "
                  f"{row['pct_reused']:>6.2f}")
    print(f"")
    print(f"Written: {out_file}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tissue", required=True)
    p.add_argument("--fold", default="fold_0")
    p.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "audit_outputs"),
        help="Output directory for the JSON report.",
    )
    p.add_argument(
        "--trainingset-root",
        default=None,
        help="Override the default <repo>/cellvit-training/trainingset path. "
             "Useful for auditing a copy of the trainingset at a non-standard "
             "location.",
    )
    args = p.parse_args()
    audit(args.tissue, args.fold, Path(args.out),
          Path(args.trainingset_root) if args.trainingset_root else None)


if __name__ == "__main__":
    main()
