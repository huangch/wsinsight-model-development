"""Build markers-v5.csv from markers-v4_2.csv by adding STHELAR crosswalk columns.

v5 adds six columns that map every subtype row onto the three label spaces used
by the STHELAR / CellViT-for-STHELAR dataset (Giraud-Sauveur et al., Sci. Data
2026):

    sthelar_full_type / sthelar_full_label                 (9-class)
    sthelar_coarse_type / sthelar_coarse_label             (5-class)
    sthelar_cancer_normal_type / sthelar_cancer_normal_label (binary)

Convention (matches the existing hne_/pantissue_ pairs):
  *_label  = STHELAR's verbatim machine token (e.g. ``T_NK``, ``B_Plasma``),
             so it can be fed to the STHELAR YAML configs unchanged.
  *_type   = human-readable form (e.g. ``T / NK lymphocytes``).

Mapping is driven by ``hne_label`` + the ``malignant`` flag, with a ``subtype``
keyword override where STHELAR splits finer than the kurtorank hne taxonomy
(lymphoid B vs T/NK, and the hematologic_blast leukemia/lymphoma cases).

Run once; idempotent (overwrites markers-v5.csv).
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "kurtorank" / "markers" / "data"
SRC = DATA_DIR / "markers-v4_2.csv"
DST = DATA_DIR / "markers-v5.csv"

# STHELAR 9-class full label space (configs/preprocessing_sthelar.yaml):
#   Epithelial, Blood_vessel, Fibroblast_Myofibroblast, Myeloid, B_Plasma,
#   T_NK, Melanocyte, Specialized, Other
#
# Two hne_label values resolve per-row from the subtype string instead of a
# fixed bucket; they are flagged with the sentinels below.
_LYMPHOCYTE = "<lymphocyte-split>"
_BLAST = "<blast-split>"

# hne_label -> STHELAR full label.
HNE_TO_STHELAR_FULL = {
    # epithelial lineages (incl. specialized secretory epithelium)
    "malignant_epithelial":     "Epithelial",
    "epithelial":               "Epithelial",
    "basaloid_progenitor":      "Epithelial",
    "neuroendocrine":           "Epithelial",
    "malignant_neuroendocrine": "Epithelial",
    "ependymal":                "Epithelial",
    "mesothelial":              "Epithelial",
    "germ_cell":                "Epithelial",

    # melanocytic
    "melanocyte":            "Melanocyte",
    "malignant_melanocytic": "Melanocyte",

    # endothelial (vascular + lymphatic)
    "endothelial":          "Blood_vessel",
    "malignant_endothelial": "Blood_vessel",

    # fibroblast / myofibroblast
    "fibroblast_like":      "Fibroblast_Myofibroblast",

    # specialized parenchymal / mural / muscle / neural / skeletal
    "smooth_muscle":        "Specialized",
    "pericyte":             "Specialized",
    "perivascular":         "Specialized",
    "malignant_muscle":     "Specialized",
    "cardiomyocyte":        "Specialized",
    "mesangial":            "Specialized",
    "adipocyte":            "Specialized",
    "chondrocyte":          "Specialized",
    "osteoblast":           "Specialized",
    "osteocyte":            "Specialized",
    "osteoclast":           "Specialized",
    "podocyte":             "Specialized",
    "neuron":               "Specialized",
    "glial":                "Specialized",
    "schwann":              "Specialized",

    # myeloid (mononuclear + granulocytes; STHELAR groups granulocytes as myeloid)
    "macrophage_like":      "Myeloid",
    "neutrophil":           "Myeloid",
    "eosinophil":           "Myeloid",
    "basophil":             "Myeloid",
    "mast_cell":            "Myeloid",

    # plasma cell
    "plasma_cell":          "B_Plasma",

    # subtype-resolved
    "lymphocyte":           _LYMPHOCYTE,
    "hematologic_blast":    _BLAST,

    # other / non-cell
    "malignant_mixed":      "Other",
    "embryonal_tumor":      "Other",
    "red_blood":            "Other",
    "platelet":             "Other",
}

# STHELAR full label -> human-readable type.
STHELAR_FULL_TYPE = {
    "Epithelial":               "Epithelial",
    "Blood_vessel":             "Blood vessel / endothelial",
    "Fibroblast_Myofibroblast": "Fibroblast / myofibroblast",
    "Myeloid":                  "Myeloid",
    "B_Plasma":                 "B / plasma cells",
    "T_NK":                     "T / NK lymphocytes",
    "Melanocyte":               "Melanocyte",
    "Specialized":              "Specialized",
    "Other":                    "Other",
}

# STHELAR's own 5-class collapse (configs/examples/preprocessing_sthelar20x_5class.yaml).
STHELAR_FULL_TO_COARSE = {
    "T_NK":                     "Immune",
    "B_Plasma":                 "Immune",
    "Myeloid":                  "Immune",
    "Blood_vessel":             "Stromal",
    "Fibroblast_Myofibroblast": "Stromal",
    "Epithelial":               "Epithelial",
    "Melanocyte":               "Other",
    "Specialized":              "Other",
    "Other":                    "Other",
}

STHELAR_COARSE_TYPE = {
    "Immune":     "Immune",
    "Stromal":    "Stromal",
    "Epithelial": "Epithelial",
    "Other":      "Other",
}

STHELAR_CANCER_NORMAL_TYPE = {
    "Cancer": "Cancer",
    "Normal": "Normal",
}


def _split_lymphocyte(subtype: str) -> str:
    """Resolve an hne_label=lymphocyte row to STHELAR B_Plasma / T_NK / Myeloid."""
    s = subtype.lower()
    # Dendritic cells occasionally carry the 'lymphocyte' hne_label; STHELAR
    # groups DCs with the myeloid lineage.
    if "dendritic" in s or "dc" in s.split():
        return "Myeloid"
    b_keys = (
        "b cell", "b-cell", "b cells", "germinal", "mantle",
        "marginal zone", "breg", "b-regulatory", "b regulatory", "plasma",
    )
    if any(k in s for k in b_keys):
        return "B_Plasma"
    return "T_NK"


def _split_blast(subtype: str) -> str:
    """Resolve an hne_label=hematologic_blast row to STHELAR B_Plasma / T_NK / Myeloid / Other."""
    s = subtype.lower()
    if "myeloid" in s or "suppressor" in s or "mdsc" in s:
        return "Myeloid"
    b_keys = (
        "myeloma", "b-cell", "b cell", "mantle", "follicular lymphoma",
        "marginal zone", "hodgkin", "reed-sternberg", "dlbcl",
        "diffuse large b", "burkitt", "lymphoplasmacytic",
    )
    if any(k in s for k in b_keys):
        return "B_Plasma"
    t_keys = (
        "t-cell lymphoma", "t cell lymphoma", "t-lymphoblastic",
        "t lymphoblastic", "anaplastic large", "alcl", "peripheral t",
    )
    if any(k in s for k in t_keys):
        return "T_NK"
    # Leukemias, HSPCs and unspecified blasts have no STHELAR equivalent.
    return "Other"


def _full_label(row: pd.Series) -> str:
    mapped = HNE_TO_STHELAR_FULL[row["hne_label"]]
    if mapped == _LYMPHOCYTE:
        return _split_lymphocyte(str(row["subtype"]))
    if mapped == _BLAST:
        return _split_blast(str(row["subtype"]))
    return mapped


def main() -> None:
    df = pd.read_csv(SRC)

    missing = sorted(set(df["hne_label"].dropna().unique()) - set(HNE_TO_STHELAR_FULL))
    if missing:
        raise SystemExit(f"Unmapped hne_label values: {missing}")

    # full (9-class)
    df["sthelar_full_label"] = df.apply(_full_label, axis=1)
    df["sthelar_full_type"] = df["sthelar_full_label"].map(STHELAR_FULL_TYPE)

    # coarse (5-class), derived from full via STHELAR's documented collapse
    df["sthelar_coarse_label"] = df["sthelar_full_label"].map(STHELAR_FULL_TO_COARSE)
    df["sthelar_coarse_type"] = df["sthelar_coarse_label"].map(STHELAR_COARSE_TYPE)

    # cancer / normal (binary), driven by the malignant flag
    cn = df["malignant"].astype(bool).map({True: "Cancer", False: "Normal"})
    df["sthelar_cancer_normal_label"] = cn
    df["sthelar_cancer_normal_type"] = cn.map(STHELAR_CANCER_NORMAL_TYPE)

    # Sanity: nothing left unmapped.
    for col in ("sthelar_full_type", "sthelar_coarse_type", "sthelar_cancer_normal_type"):
        if df[col].isna().any():
            bad = df.loc[df[col].isna(), ["tissue_type", "subtype", "hne_label"]]
            raise SystemExit(f"Unmapped rows for {col}:\n{bad.to_string()}")

    # Re-order so the six new columns sit between pantissue_label and markers,
    # matching the ascending-granularity convention of the existing schema
    # (type before label within each pair).
    new_cols = [
        "sthelar_full_type", "sthelar_full_label",
        "sthelar_coarse_type", "sthelar_coarse_label",
        "sthelar_cancer_normal_type", "sthelar_cancer_normal_label",
    ]
    cols = [c for c in df.columns if c not in new_cols]
    idx = cols.index("pantissue_label") + 1
    cols[idx:idx] = new_cols
    df = df[cols]

    df.to_csv(DST, index=False)

    # Summary
    print(f"Wrote {DST}  ({len(df)} rows, {len(df.columns)} cols)")
    print()
    print("Header:")
    print("  " + ",".join(df.columns))
    print()
    for col in ("sthelar_full_label", "sthelar_coarse_label", "sthelar_cancer_normal_label"):
        print(f"{col} distribution:")
        print(df[col].value_counts().to_string())
        print()
    print("Subtype-resolved rows (lymphocyte / hematologic_blast):")
    split_rows = df[df["hne_label"].isin(("lymphocyte", "hematologic_blast"))]
    print(
        split_rows[["tissue_type", "subtype", "hne_label", "sthelar_full_label"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
