"""Build markers-v4_2.csv from markers-v4_1.csv.

v4_2 is a *panel-aware* curation of the head_neck epithelial + malignant
subtypes only. Every other row is copied verbatim from v4_1.

Motivation
----------
The 10x Xenium head_neck GEO panels (~476 genes, immune/TME-focused) do NOT
measure the classic squamous keratins (KRT5/14/6A/17), TP63, SFN or
desmosomal genes that the v4_1 epithelial/tumor signatures lead with. As a
result:
  - "Mucosal stratified squamous epithelial cells" had 0/31 markers in panel,
  - "Oral basal keratinocytes" had 4/44,
both fell below the >=3 detectable-marker filter and were dropped, leaving the
single epithelial cluster with no benign competitor -> it defaulted to "tumor".

Fix
---
Prepend genuine squamous/basal epithelial anchor genes that ARE present in the
panel (and are biologically valid squamous markers) to the epithelial and
malignant rows, and lead the malignant rows with the proliferation/oncogenic
genes that distinguish tumor from benign epithelium. All original genes are
preserved (reordered), so keratin-bearing panels (e.g. H&E) are unaffected.

NOTE: these are biological judgement calls and should be reviewed by a domain
expert before publication use.
"""
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "markers-v4_1.csv"
DST = HERE / "markers-v4_2.csv"

# Squamous/basal epithelial anchors that exist in the Xenium head_neck panel
# and are biologically valid epithelial markers (shared by normal + malignant
# epithelium -> establishes "this is epithelium").
EPI_ANCHORS = ["EPCAM", "COL17A1", "SERPINB3", "SERPINB2",
               "GPRC5A", "FGFBP1", "CLCA2"]

# Discriminators that separate malignant from benign epithelium on this panel:
# proliferation + HNSCC oncogenes / immune-evasion.
TUMOR_LEAD = ["MKI67", "TOP2A", "PCNA", "UBE2C",
              "SOX2", "EGFR", "CD274", "MYC"]

# Lymphoid / checkpoint / IFN-immune genes that the v4_1 HPV+ row inherited but
# that are NOT tumor-intrinsic (they mark infiltrating T/NK cells and the
# inflamed microenvironment, not the malignant epithelium). Removing them keeps
# the tumor signature about the tumor cell, so KurtoRank does not score a tumor
# cluster on the basis of how many lymphocytes sit next to it.
#   CD8A,GZMB           cytotoxic T / NK
#   CXCL9,CXCL10,CXCL11 IFN-induced T-cell chemokines (myeloid/stroma)
#   PDCD1,LAG3,HAVCR2   T-cell checkpoints (tumor expresses CD274, not these)
#   IDO1,STAT1          IFN-response, ubiquitous
#   CXCL13              TLS / Tfh-B follicle
# CD274 (PD-L1) is kept: it is genuinely tumor-cell-expressed immune evasion.
# S100A8/S100A9 are kept: also expressed by inflamed/malignant squamous
# epithelium, not lymphoid-exclusive.
DROP_IMMUNE = ["CD8A", "GZMB", "CXCL9", "CXCL10", "CXCL11",
               "PDCD1", "IDO1", "STAT1", "LAG3", "HAVCR2", "CXCL13"]


def reorder(markers_str, lead, drop=()):
    """Return marker list with `lead` genes first (deduped, order-preserving),
    followed by the original genes not already in `lead`, with any gene in
    `drop` removed entirely."""
    drop = set(drop)
    orig = [g.strip() for g in markers_str.split(",") if g.strip()]
    seen, out = set(), []
    for g in lead + orig:
        if g and g not in seen and g not in drop:
            seen.add(g)
            out.append(g)
    return ",".join(out)


df = pd.read_csv(SRC)

# (subtype, lead-gene list) for the head_neck epithelial/tumor rows.
edits = {
    # malignant: epithelial anchors + tumor discriminators lead
    "Head and neck squamous cell carcinoma cells": TUMOR_LEAD + EPI_ANCHORS,
    "HPV-positive oropharyngeal squamous cell carcinoma cells": TUMOR_LEAD + EPI_ANCHORS,
    # benign epithelium: epithelial anchors lead, NO proliferation genes added
    "Mucosal stratified squamous epithelial cells": EPI_ANCHORS,
    "Oral basal keratinocytes": EPI_ANCHORS,
}

# Per-subtype genes to strip out (non-tumor-intrinsic immune infiltrate).
drops = {
    "HPV-positive oropharyngeal squamous cell carcinoma cells": DROP_IMMUNE,
}

mask_hn = df.tissue_type == "head_neck"
n_changed = 0
for subtype, lead in edits.items():
    sel = mask_hn & (df.subtype == subtype)
    assert sel.sum() == 1, f"expected exactly one row for {subtype!r}, got {sel.sum()}"
    idx = df.index[sel][0]
    drop = drops.get(subtype, ())
    df.at[idx, "markers"] = reorder(df.at[idx, "markers"], lead, drop=drop)
    df.at[idx, "rank_source"] = "v4_2_panel_aware"
    old_comment = str(df.at[idx, "comment"])
    df.at[idx, "comment"] = (old_comment +
        " | v4_2: led with panel-present epithelial anchors"
        " (EPCAM/COL17A1/SERPINB3/SERPINB2/GPRC5A/FGFBP1/CLCA2)"
        + ("; tumor leads with proliferation+oncogenic (MKI67/TOP2A/PCNA/UBE2C/SOX2/EGFR/CD274/MYC)."
           if "carcinoma" in subtype.lower() else "; no proliferation genes (benign).")
        + (" Removed non-tumor-intrinsic immune-infiltrate genes"
           " (CD8A/GZMB/CXCL9-11/PDCD1/IDO1/STAT1/LAG3/HAVCR2/CXCL13); kept"
           " CD274 (tumor PD-L1)." if drop else ""))
    n_changed += 1

assert n_changed == 4
assert len(df) == len(pd.read_csv(SRC)), "row count changed!"
df.to_csv(DST, index=False)
print(f"Wrote {DST} ({len(df)} rows, {n_changed} head_neck rows augmented).")
