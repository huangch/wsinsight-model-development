"""
build_cell_labels.py
--------------------
Tissue-agnostic builder. For each Xenium sample listed in the tissue config,
join:
    cells.csv.gz + clusters.csv + celltype_assignment_hne_label.csv
into a flat per-cell CSV:
    cell_id, x_um, y_um, class_int

Also emits label_map.yaml (int -> label-name) derived from the same label_map
in the tissue config, so it cannot drift from the training class_int.

Tissue-specific parameters (xenium_base, out_dir, label_map, samples) live in
tissue_configs/<tissue>.yaml.

Usage:
    . ~/local/conda_init.sh && conda activate spatial
    python build_cell_labels.py --tissue colorectal
    python build_cell_labels.py --tissue breast
    python build_cell_labels.py --config path/to/custom.yaml
"""

import argparse
import csv
import gzip
import os
import sys
from collections import Counter
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CELLVIT_TRAINING_ROOT.parent
DEFAULT_CONFIG_DIR = CELLVIT_TRAINING_ROOT / "tissue_configs"

# Make ${PROJECT_ROOT} / ${CELLVIT_TRAINING_ROOT} available for YAML expansion.
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("CELLVIT_TRAINING_ROOT", str(CELLVIT_TRAINING_ROOT))


def load_config(tissue: str | None, config_path: str | None) -> dict:
    if config_path:
        path = Path(config_path)
    else:
        if not tissue:
            raise SystemExit("ERROR: must provide either --tissue or --config")
        path = DEFAULT_CONFIG_DIR / f"{tissue}.yaml"
    if not path.is_file():
        raise SystemExit(f"ERROR: tissue config not found: {path}")

    # Expand ${VAR} tokens (e.g. ${PROJECT_ROOT}) before YAML parses; this lets
    # tissue configs stay portable across checkouts / folder renames.
    text = os.path.expandvars(path.read_text())
    cfg = yaml.safe_load(text)

    for key in ("xenium_base", "out_dir", "label_map", "samples"):
        if key not in cfg:
            raise SystemExit(f"ERROR: '{key}' missing from {path}")

    # Validate label_map ints are a contiguous 0..N-1 permutation.
    label_ints = sorted(cfg["label_map"].values())
    if label_ints != list(range(len(label_ints))):
        raise SystemExit(
            f"ERROR: label_map ints in {path} must be a contiguous "
            f"0..{len(label_ints) - 1} set; got {label_ints}"
        )

    cfg["_path"] = str(path)
    return cfg


def build_cell_label_map(outs_dir, label_to_int, label_file="celltype_assignment_hne_label.csv"):
    """Returns dict: cell_id -> {"x_um": float, "y_um": float, "class_int": int}"""

    # 1. cluster id (string) -> cell_type string
    ct_map = {}
    with open(os.path.join(outs_dir, label_file)) as f:
        for row in csv.DictReader(f):
            ct_map[row["classification"]] = row["cell_type"]

    # 2. cell_id (Barcode) -> cluster id
    cluster_map = {}
    clusters_path = os.path.join(
        outs_dir, "analysis", "clustering",
        "gene_expression_graphclust", "clusters.csv"
    )
    with open(clusters_path) as f:
        for row in csv.DictReader(f):
            cluster_map[row["Barcode"]] = row["Cluster"]

    # 3. cells.csv.gz -> cell_id, x_centroid, y_centroid
    result = {}
    skipped_no_cluster = 0
    skipped_no_ct = 0
    skipped_no_int = 0

    with gzip.open(os.path.join(outs_dir, "cells.csv.gz"), "rt") as f:
        for row in csv.DictReader(f):
            cid = row["cell_id"]
            cluster = cluster_map.get(cid)
            if cluster is None:
                skipped_no_cluster += 1
                continue
            ct = ct_map.get(cluster)
            if ct is None:
                skipped_no_ct += 1
                continue
            class_int = label_to_int.get(ct)
            if class_int is None:
                skipped_no_int += 1
                continue
            result[cid] = {
                "x_um":      float(row["x_centroid"]),
                "y_um":      float(row["y_centroid"]),
                "class_int": class_int,
            }

    if skipped_no_cluster:
        print(f"  WARNING: {skipped_no_cluster} cells skipped — no cluster entry")
    if skipped_no_ct:
        print(f"  WARNING: {skipped_no_ct} cells skipped — cluster not in label file")
    if skipped_no_int:
        print(f"  WARNING: {skipped_no_int} cells skipped — unknown cell_type string")

    return result


def write_label_csv(cell_map, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "x_um", "y_um", "class_int"])
        for cid, v in cell_map.items():
            writer.writerow([cid, v["x_um"], v["y_um"], v["class_int"]])


def write_label_map_yaml(label_to_int, out_path):
    """Emit label_map.yaml as int -> label-name, derived from label_to_int.
    Consumed by validate_classifier.py (--label-map) and mirrors the label_map
    block embedded in train_configs/.../fold_0.yaml.
    """
    int_to_label = {v: k for k, v in label_to_int.items()}
    with open(out_path, "w") as f:
        for ci in sorted(int_to_label):
            f.write(f'{ci}: "{int_to_label[ci]}"\n')


def main():
    parser = argparse.ArgumentParser(description="Build per-cell label CSVs for one tissue.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tissue", help="Tissue name; loads tissue_configs/<tissue>.yaml")
    group.add_argument("--config", help="Explicit path to a tissue YAML config")
    args = parser.parse_args()

    cfg = load_config(args.tissue, args.config)
    xenium_base = cfg["xenium_base"]
    out_dir     = cfg["out_dir"]
    label_to_int = dict(cfg["label_map"])
    samples      = [tuple(s) for s in cfg["samples"]]

    print(f"[config] {cfg['_path']}")
    print(f"[config] {len(label_to_int)} classes, {len(samples)} samples")

    os.makedirs(out_dir, exist_ok=True)

    label_map_path = os.path.join(out_dir, "label_map.yaml")
    write_label_map_yaml(label_to_int, label_map_path)
    print(f"[label_map] wrote {label_map_path}")

    int_to_label = {v: k for k, v in label_to_int.items()}

    for sample_tag, rel_path in samples:
        outs_dir = os.path.join(xenium_base, rel_path)
        out_path = os.path.join(out_dir, f"cell_labels_{sample_tag}.csv")

        if not os.path.isdir(outs_dir):
            print(f"[SKIP] {sample_tag}: outs dir not found: {outs_dir}")
            continue

        if os.path.exists(out_path):
            print(f"[SKIP] {sample_tag}: output already exists: {out_path}")
            continue

        print(f"[{sample_tag}] Building label map from {outs_dir} ...")
        try:
            cell_map = build_cell_label_map(outs_dir, label_to_int)
            write_label_csv(cell_map, out_path)
            print(f"  -> {len(cell_map):,} cells written to {out_path}")

            counts = Counter(v["class_int"] for v in cell_map.values())
            for ci in sorted(counts):
                pct = 100 * counts[ci] / len(cell_map)
                print(f"     class {ci:2d} ({int_to_label[ci]:25s}): {counts[ci]:6,}  ({pct:.1f}%)")

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise

    print("\nDone.")


if __name__ == "__main__":
    main()
