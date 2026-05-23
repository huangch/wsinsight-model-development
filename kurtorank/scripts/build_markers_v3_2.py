"""Build markers-v3_2.csv from markers-v3_1.csv by adding pantissue_type / pantissue_label columns.

Run once; idempotent (overwrites markers-v3_2.csv).
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "kurtorank" / "markers" / "data"
SRC = DATA_DIR / "markers-v3_1.csv"
DST = DATA_DIR / "markers-v3_2.csv"

# pantissue_label: lowercase snake_case, disease-agnostic.
# Mapping: hne_label -> pantissue_label
HNE_TO_PANTISSUE_LABEL = {
    # tumor / neoplastic
    "malignant_epithelial":     "tumor",
    "malignant_melanocytic":    "tumor",
    "malignant_endothelial":    "tumor",
    "malignant_mixed":          "tumor",
    "malignant_muscle":         "tumor",
    "malignant_neuroendocrine": "tumor",
    "embryonal_tumor":          "tumor",

    # normal epithelial (incl. specialized epithelial lineages)
    "epithelial":          "epithelial",
    "melanocyte":          "epithelial",
    "neuroendocrine":      "epithelial",   # non-malignant NE cells (islets etc.)
    "basaloid_progenitor": "epithelial",   # basal keratinocyte / cervical basal
    "ependymal":           "epithelial",
    "mesothelial":         "epithelial",
    "germ_cell":           "epithelial",
    "podocyte":            "epithelial",

    # lymphoid
    "lymphocyte":          "lymphoid",

    # plasma cell
    "plasma_cell":         "plasma",

    # myeloid (mononuclear)
    "macrophage_like":     "myeloid",

    # granulocytes (multilobed / granular)
    "neutrophil":          "granulocyte",
    "eosinophil":          "granulocyte",
    "basophil":            "granulocyte",
    "mast_cell":           "granulocyte",

    # hematopoietic blast / immature
    "hematologic_blast":   "blast",

    # stromal / mesenchymal
    "fibroblast_like":     "stromal",
    "smooth_muscle":       "stromal",
    "pericyte":            "stromal",
    "perivascular":        "stromal",
    "adipocyte":           "stromal",
    "chondrocyte":         "stromal",
    "osteoblast":          "stromal",
    "osteocyte":           "stromal",
    "osteoclast":          "stromal",   # morphologically grouped with bone cells on H&E
    "cardiomyocyte":       "stromal",
    "mesangial":           "stromal",

    # endothelial
    "endothelial":         "endothelial",

    # neural
    "neuron":              "neural",
    "glial":               "neural",
    "schwann":             "neural",

    # filtered (anucleate or excluded from training)
    "red_blood":           "filtered",
    "platelet":            "filtered",
}

# pantissue_type: human-readable form (Title Case with separators)
PANTISSUE_LABEL_TO_TYPE = {
    "tumor":       "Tumor / neoplastic",
    "epithelial":  "Epithelial",
    "lymphoid":    "Lymphoid",
    "plasma":      "Plasma cell",
    "myeloid":     "Myeloid",
    "granulocyte": "Granulocyte",
    "blast":       "Hematopoietic blast",
    "stromal":     "Stromal / mesenchymal",
    "endothelial": "Endothelial",
    "neural":      "Neural",
    "filtered":    "Filtered (anucleate)",
}


def main() -> None:
    df = pd.read_csv(SRC)

    missing = sorted(set(df["hne_label"].dropna().unique()) - set(HNE_TO_PANTISSUE_LABEL.keys()))
    if missing:
        raise SystemExit(f"Unmapped hne_label values: {missing}")

    df["pantissue_label"] = df["hne_label"].map(HNE_TO_PANTISSUE_LABEL)
    df["pantissue_type"]  = df["pantissue_label"].map(PANTISSUE_LABEL_TO_TYPE)

    # Re-order columns so pantissue_{type,label} sit between hne_label and markers,
    # matching the ascending-granularity convention in the existing schema.
    cols = list(df.columns)
    for new_col in ("pantissue_type", "pantissue_label"):
        cols.remove(new_col)
    idx = cols.index("hne_label") + 1
    cols[idx:idx] = ["pantissue_type", "pantissue_label"]
    df = df[cols]

    df.to_csv(DST, index=False)

    # Summary
    print(f"Wrote {DST}  ({len(df)} rows, {len(df.columns)} cols)")
    print()
    print("Header:")
    print("  " + ",".join(df.columns))
    print()
    print("pantissue_label distribution (row count):")
    print(df["pantissue_label"].value_counts().to_string())
    print()
    print(f"Unique hne_label   : {df['hne_label'].nunique()}")
    print(f"Unique pantissue   : {df['pantissue_label'].nunique()}")


if __name__ == "__main__":
    main()
