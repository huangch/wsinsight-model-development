from __future__ import annotations

import csv
from pathlib import Path


SRC = Path(
    "/workspace/wsinsight/wsinsight-model-development/"
    "kurtorank/src/kurtorank/markers/data/markers-v5.csv"
)
DEST = Path(
    "/workspace/wsinsight/wsinsight-model-development/"
    "kurtorank/src/kurtorank/markers/data/markers-v6.csv"
)


LABEL_MAP = {
    "Parenchymal cell": 1,
    "Endothelial cell": 2,
    "Fibroblast": 3,
    "Macrophage": 4,
    "cDC": 5,
    "CD4 T": 6,
    "CD8 T": 7,
    "B": 8,
    "Plasma cell": 9,
    "NK": 10,
    "pDC": 11,
    "Mast cell": 12,
    "Other": 0,
}


INSERT_BEFORE = "markers"
NEW_COLUMNS = ["lcp_type", "lcp_label"]


def norm(value: str | None) -> str:
    return (value or "").strip().lower()


def map_row(row: dict[str, str]) -> str:
    subtype = norm(row.get("subtype"))
    major = norm(row.get("major_type"))
    hne = norm(row.get("hne_label"))
    pantissue = norm(row.get("pantissue_label"))
    sthelar_full = norm(row.get("sthelar_full_label"))

    if pantissue == "filtered" or "erythrocyte" in subtype or "platelet" in subtype:
        return "Other"

    if "plasmacytoid tumor" in subtype:
        return "Parenchymal cell"

    if "plasmacytoid dendritic" in subtype:
        return "pDC"

    if (
        ("plasma" in subtype and "plasmacytoid" not in subtype)
        or hne == "plasma_cell"
        or pantissue == "plasma"
    ):
        return "Plasma cell"

    if "basophil" in subtype:
        return "Mast cell"

    if "mast" in subtype or hne == "mast_cell":
        return "Mast cell"

    if "langerhans" in subtype:
        return "cDC"

    if (
        "dendritic" in subtype
        or "cdc" in subtype
        or "dc3" in subtype
        or "pre-dc" in subtype
    ):
        if "follicular dendritic" in subtype:
            return "Fibroblast"
        return "cDC"

    if (
        "macrophage" in subtype
        or "microglia" in subtype
        or "kupffer" in subtype
        or "monocyte" in subtype
        or "osteoclast" in subtype
        or "myeloid-derived suppressor" in subtype
    ):
        return "Macrophage"

    if subtype.startswith("nkt") or "natural killer t" in subtype or "mait" in subtype:
        return "CD8 T"

    if subtype.startswith("ilc2") or subtype.startswith("ilc3") or "ilc3-like" in subtype:
        return "CD4 T"

    if subtype.startswith("ilc1"):
        return "NK"

    if "natural killer" in subtype or "hepatic pit nk" in subtype:
        return "NK"

    if (
        "cd8" in subtype
        or "temra" in subtype
        or "exhausted" in subtype
        or "resident-memory t" in subtype
        or "gamma-delta t" in subtype
        or "effector-memory t" in subtype
    ):
        return "CD8 T"

    if (
        "cd4" in subtype
        or "treg" in subtype
        or "t follicular helper" in subtype
        or "central-memory t" in subtype
        or "memory t cells" in subtype
        or "paracortical t cells" in subtype
    ):
        return "CD4 T"

    if (
        subtype.endswith(" b cells")
        or "b-regulatory" in subtype
        or "b-cell lymphoma" in subtype
        or "mantle cell lymphoma" in subtype
        or "follicular lymphoma" in subtype
        or "marginal zone lymphoma" in subtype
        or major == "germinal center cells"
    ):
        return "B"

    if sthelar_full == "b_plasma" and pantissue != "plasma":
        return "B"

    if (
        pantissue == "endothelial"
        or hne == "endothelial"
        or "endothelial" in subtype
        or major in {"vascular cells", "vascular progenitor cells"}
    ):
        return "Endothelial cell"

    if "juxtaglomerular renin" in subtype:
        return "Fibroblast"

    if (
        "fibroblast" in subtype
        or "myofibroblast" in subtype
        or "pericyte" in subtype
        or major == "stromal cells"
    ):
        return "Fibroblast"

    if (
        pantissue in {"tumor", "epithelial", "neural"}
        or "malignant" in major
        or major
        in {
            "epithelial cells",
            "mesothelial cells",
            "neuronal cells",
            "glial cells",
            "neuroendocrine cells",
            "cardiomyocytes",
            "glomerular cells",
            "ependymal cells",
            "endocrine cells",
            "exocrine cells",
            "progenitor cells",
        }
    ):
        return "Parenchymal cell"

    return "Other"


def main() -> None:
    with SRC.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            paper_type = map_row(row)
            row["lcp_type"] = paper_type
            row["lcp_label"] = str(LABEL_MAP[paper_type])
            rows.append(row)

    base_fields = [
        field for field in fieldnames if field not in {"paper_type", "paper12_label", *NEW_COLUMNS}
    ]
    if INSERT_BEFORE in base_fields:
        idx = base_fields.index(INSERT_BEFORE)
        new_fields = base_fields[:idx] + NEW_COLUMNS + base_fields[idx:]
    else:
        new_fields = base_fields + NEW_COLUMNS

    with DEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {DEST} with {len(rows)} rows")


if __name__ == "__main__":
    main()