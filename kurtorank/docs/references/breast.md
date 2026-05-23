# Breast References

## 1. The Cancer Genome Atlas Network. *Comprehensive molecular portraits of human breast tumours.* Nature 490, 61-70 (2012)

- Foundational TCGA BRCA study describing intrinsic subtypes (Luminal A/B, HER2-enriched, Basal-like) with subtype-specific drivers, copy-number events, and expression modules.
- Supports luminal markers (ESR1, PGR, FOXA1, GATA3, SCUBE2, MAPT, GREB1), HER2 amplicon elements (ERBB2, GRB7, PGAP3, STARD3, CDK12), and basal EMT/immune features (KRT5/6/14/17, EGFR, FOXC1, ITGA6/ITGB4, PDPN).

## 2. Curtis, C. et al. *The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups.* Nature 486, 346-352 (2012) (METABRIC)

- Provides large-scale expression and CNA correlations that refine Luminal A versus Luminal B genes (CITED1, TOX3, SLC34A2, BAG4), HER2+ kinase/amplification targets (PTK6, GRIA2, ERBB3/4), and basal immune/stromal modules.
- Informs the added Luminal A/B and HER2 marker extensions in `markers.csv`.

## 3. Wu, S.Z. et al. *Single-cell and spatially resolved atlases of human breast cancers.* Nature Genetics 53, 1334-1347 (2021)

- scRNA-seq plus spatial profiling across luminal, HER2-amplified, basal-like, and claudin-low tumors showing epithelial–stromal niche interactions and rare populations (neuroendocrine states, myoepithelial progenitors, adipocytes).
- Provides high-confidence cell-type markers for fibroblasts (FAP, POSTN, COL14A1), pericytes (PDGFRB, RGS5, CSPG4), endothelial/lymphatic clusters (PROX1, APLN, ANGPTL4), immune-rich claudin-low signatures, and adipocyte regulators (ADIPOQ, CIDEC, ADIG, ANGPTL8).

## 4. Pal, B. et al. *Single-cell RNA sequencing reveals novel markers of mammary stem and luminal progenitors.* Nature Communications 8, 16069 (2017)

- Captures normal ductal/lobular epithelial diversity, defining luminal progenitor markers (KRT8/18/19, SLC39A6, SPDEF, SCUBE2, SLC7A8), mature ductal markers (MUC1, CLDN3/4/7, EPCAM), and basal/myoepithelial programs (TP63, ACTA2, KRT14, TAGLN).
- Underpins the normal ductal and myoepithelial sections plus the added luminal genes (KRT20, SLC34A2, CYP24A1, DSTYK, PDE7B) that top up Luminal A.

## 5. Lawson, D.A. et al. *Single-cell analysis reveals a stem-cell program in human metastatic breast cancer cells.* Nature 526, 131-135 (2015)

- Identifies rare neuroendocrine-like and stem-like populations expressing ASCL1, INSM1, DLL3, GRP, PEG10, and stress-response genes within metastatic lesions, supporting the neuroendocrine marker set in the CSV.
- Confirms claudin-low/EMT features with AXL, ITGA5, LOX, S100A4, and immune chemokines, reinforcing their inclusion.

## 6. Tabula Sapiens Consortium. *The Tabula Sapiens: A multiple-organ single-cell transcriptomic atlas of the human body.* Science 376, eabl4896 (2022)

- Provides reference profiles for normal breast stromal/immune compartments, including adipocytes (PLIN1/4, ADIG, ANGPTL8, C10orf10), fibroblasts (COL1A1/2, VCAN, IGFBP5), endothelial subsets (PECAM1, PROCR, CLEC14A), and lymphatic ECs (PROX1, PODOPLANIN).
- Used to justify the adipocyte top-up genes (ADIG, CIDEC-AS1, ANGPTL8, C10orf10, CISH) and to confirm pan-endothelial and pericyte markers.

## 7. Uhlén, M. et al. *The Human Protein Atlas in 2023.* Nucleic Acids Research 51, D1057-D1073 (2023). Available at <https://www.proteinatlas.org/> (tissue: breast)

- IHC validation of luminal (ESR1, SCUBE2, MAPT), HER2+ (ERBB2, GRB7, PTK6), basal (KRT5/6/14/17, EGFR), myoepithelial (ACTA2, TP63, TAGLN), adipocyte (ADIPOQ, LEP, CIDEC) and stromal markers (FAP, POSTN, PDGFRB, RGS5) at the protein level.
- Confirms expression/localization of the genes newly appended to the breast block.
