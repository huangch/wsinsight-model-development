#!/usr/bin/env python
"""
Build a PanNuke-style marker table from DISCO using DISCOtoolkit.

Columns:
    tissue_type  - organ / conceptual site (e.g. bladder, brain, immune)
    common       - 'common' or 'rare' (based on frequency threshold)
    normal       - 'normal' or 'neoplastic' (heuristic from cell_type name)
    major        - broad cell class (e.g. epithelial, T cell, fibroblast)
    subtype      - more specific cell type (DISCO cell_type string)
    pannuke_type - one of: neoplastic, epithelial, connective,
                              inflammatory, dead, other
    markers      - comma-separated marker gene list (CSV-quoted)

Dependencies:
    pip install discotoolkit scanpy anndata pandas
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import discotoolkit as dt
import scanpy as sc
import anndata as ad
import pandas as pd


# --------- configuration ---------

# Your desired tissue_type labels
TARGET_TISSUES = [
    "bladder", "brain", "bone", "cervix", "heart", "kidney", "breast",
    "colorectal", "prostate", "liver", "lung", "ovary", "skin", "pancreas",
    "tonsil", "lymph_node",
    "immune",        # migratory immune cells (from blood)
    "circulating",   # circulating blood cells (RBC/platelets, etc.)
]

# Map your tissue_type → DISCO "tissue" string
# (adjust the right-hand side to match the exact DISCO tissue names)
DISCO_TISSUE_MAPPING: Dict[str, str] = {
    "bladder": "Bladder",
    "brain": "Brain",
    "bone": "Bone marrow",       # includes bone marrow
    "cervix": "Cervix",
    "heart": "Heart",
    "kidney": "Kidney",
    "breast": "Breast",
    "colorectal": "Colon",       # you can add "Rectum" separately if needed
    "prostate": "Prostate",
    "liver": "Liver",
    "lung": "Lung",
    "ovary": "Ovary",
    "skin": "Skin",
    "pancreas": "Pancreas",
    "tonsil": "Tonsil",
    "lymph_node": "Lymph node",

    # for immune / circulating we will both pull from Blood
    "immune": "Blood",
    "circulating": "Blood",
}

# how many markers per cell type
N_MARKERS_PER_CELLTYPE = 50

# if a cell type has >= this fraction of cells in that tissue → "common"
COMMON_FREQ_THRESHOLD = 0.01   # 1% of all cells in that tissue

# directory where downloaded DISCO h5ad will be stored (per tissue)
DATA_ROOT = Path("disco_by_tissue")


# --------- helpers: string classification ---------

TUMOR_KEYWORDS = [
    "tumor", "cancer", "carcinoma", "adenocarcinoma", "malignant",
    "neoplastic", "HCC", "HNSCC", "glioblastoma", "GBM"
]

IMMUNE_MAJOR_KEYWORDS = [
    "t cell", "b cell", "nk cell", "macrophage", "monocyte",
    "dendritic", "dc", "neutrophil", "eosinophil", "basophil",
    "mast cell", "myeloid"
]

CIRCULATING_MAJOR_KEYWORDS = [
    "erythrocyte", "rbc", "red blood", "platelet", "megakaryocyte"
]

CONNECTIVE_MAJOR_KEYWORDS = [
    "fibroblast", "stromal", "mesenchymal", "smooth muscle",
    "pericyte", "endothelial"
]

EPITHELIAL_MAJOR_KEYWORDS = [
    "epithelial", "epithelium", "keratinocyte", "hepatocyte",
    "enterocyte", "ductal", "acinar", "alveolar", "urothelial",
    "secretory cell"
]


def looks_neoplastic(cell_type: str) -> bool:
    name = cell_type.lower()
    return any(kw in name for kw in TUMOR_KEYWORDS)


def guess_major(cell_type: str) -> str:
    """
    Heuristic mapping from DISCO cell_type string → broad "major" label.
    You can extend or refine this as needed.
    """
    name = cell_type.lower()

    if any(kw in name for kw in IMMUNE_MAJOR_KEYWORDS):
        return "immune"

    if any(kw in name for kw in CIRCULATING_MAJOR_KEYWORDS):
        return "circulating"

    if any(kw in name for kw in CONNECTIVE_MAJOR_KEYWORDS):
        return "connective"

    if any(kw in name for kw in EPITHELIAL_MAJOR_KEYWORDS):
        return "epithelial"

    if "neur" in name or "oligodendro" in name:
        return "neural"

    if "glia" in name or "astrocyte" in name:
        return "glial"

    return "other"


def map_to_pannuke(major: str, normal_status: str, cell_type: str) -> str:
    """
    Map to PanNuke-style categories:
        neoplastic, epithelial, connective, inflammatory, dead, other
    """

    name = cell_type.lower()

    # explicit tumor call has priority
    if normal_status == "neoplastic" or looks_neoplastic(cell_type):
        return "neoplastic"

    # dead / apoptotic / debris
    if any(x in name for x in ["apoptotic", "dead", "dying", "debris"]):
        return "dead"

    # immune → inflammatory
    if major == "immune":
        return "inflammatory"

    if major == "connective":
        return "connective"

    if major == "epithelial":
        return "epithelial"

    return "other"


def classify_common(cell_counts: pd.Series, cell_type: str) -> str:
    """
    Decide 'common' vs 'rare' based on frequency of this cell_type
    relative to all cells in the tissue AnnData.
    """
    total = cell_counts.sum()
    freq = cell_counts.get(cell_type, 0) / max(total, 1)
    return "common" if freq >= COMMON_FREQ_THRESHOLD else "rare"


# --------- core functions ---------

def download_tissue_data(tissue_type: str) -> Path:
    """
    Download DISCO data for a given tissue_type label.
    Returns directory path where h5ad files are stored.
    """
    disco_tissue = DISCO_TISSUE_MAPPING[tissue_type]
    out_dir = DATA_ROOT / tissue_type
    out_dir.mkdir(parents=True, exist_ok=True)

    filt = dt.Filter(
        # sample=None,
        # project=None,
        tissue=disco_tissue,
        disease=None,
        platform=None,            # grab both normal & disease
        cell_type=None,               # all cell types
        cell_type_confidence="medium",
        include_cell_type_children=True,
        min_cell_per_sample=100,
    )

    metadata = dt.filter_disco_metadata(filt)
    if metadata is None or metadata.empty:
        raise RuntimeError(f"No DISCO samples found for tissue={disco_tissue}")

    dt.download_disco_data(metadata, output_dir=str(out_dir))
    return out_dir


def load_and_concat(dir_path: Path) -> ad.AnnData:
    """
    Load all .h5ad files from dir_path and concatenate.
    """
    import anndata as ad_  # local alias to avoid shadowing

    adata_list: List[ad_.AnnData] = []

    for fn in os.listdir(dir_path):
        if not fn.endswith(".h5ad"):
            continue
        fp = dir_path / fn
        a = sc.read(fp)

        # sanitize .obs / .var column names
        for col in list(a.obs.columns):
            new_col = re.sub(r"\.", "_", col)
            if new_col != col:
                a.obs.rename(columns={col: new_col}, inplace=True)
        for col in list(a.var.columns):
            new_col = re.sub(r"\.", "_", col)
            if new_col != col:
                a.var.rename(columns={col: new_col}, inplace=True)

        # DISCO often bundles a .raw layer; clear it to save RAM before concat
        if a.raw is not None:
            a.raw = None

        adata_list.append(a)

    if not adata_list:
        raise RuntimeError(f"No .h5ad files in {dir_path}")

    adata = ad_.concat(adata_list)
    adata.obs_names_make_unique()
    return adata


def compute_markers(adata: ad.AnnData, groupby: str = "cell_type") -> Dict[str, List[str]]:
    """
    Run Scanpy rank_genes_groups and return
    cell_type → list of top markers (gene names).
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"{groupby} not found in adata.obs")

    # basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True, inplace=True)

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon",
        n_genes=N_MARKERS_PER_CELLTYPE,
        use_raw=False,
    )

    r = adata.uns["rank_genes_groups"]
    groups = r["names"].dtype.names

    result: Dict[str, List[str]] = {}
    for g in groups:
        genes = list(r["names"][g])
        # drop Nones and pad/truncate
        clean_genes = [x for x in genes if isinstance(x, str)]
        result[g] = clean_genes[:N_MARKERS_PER_CELLTYPE]

    return result


def build_rows_for_tissue(tissue_type: str) -> List[Dict]:
    """
    Main worker: for one tissue_type,
    return a list of dict rows ready for CSV.
    """
    print(f"Processing tissue_type={tissue_type}")

    data_dir = download_tissue_data(tissue_type)
    adata = load_and_concat(data_dir)

    # cell counts per cell_type (for 'common' flag)
    cell_counts = adata.obs["cell_type"].value_counts()

    markers_dict = compute_markers(adata, groupby="cell_type")

    rows: List[Dict] = []

    for cell_type, markers in markers_dict.items():
        major = guess_major(cell_type)
        normal_status = "neoplastic" if looks_neoplastic(cell_type) else "normal"
        pannuke_type = map_to_pannuke(major, normal_status, cell_type)
        common_flag = classify_common(cell_counts, cell_type)

        # SPECIAL CASE: immune vs circulating pseudo-tissues
        effective_tissue = tissue_type
        if tissue_type in ("immune", "circulating"):
            # both pulled from Blood; split by major type
            if tissue_type == "immune" and major != "immune":
                continue
            if tissue_type == "circulating" and major != "circulating":
                continue

        markers_str = ",".join(markers)  # CSV will quote the field

        row = {
            "tissue_type": effective_tissue,
            "common": common_flag,
            "normal": normal_status,
            "major": major,
            "subtype": cell_type,
            "pannuke_type": pannuke_type,
            "markers": markers_str,
        }
        rows.append(row)

    return rows


def main(output_csv: str = "disco_pan_tissue_markers.csv") -> None:
    all_rows: List[Dict] = []

    for tissue in TARGET_TISSUES:
        rows = build_rows_for_tissue(tissue)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
        "tissue_type", "common", "normal", "major",
        "subtype", "pannuke_type", "markers",
    ])
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    main()

