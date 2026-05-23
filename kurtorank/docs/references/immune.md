# Immune References

## 1. Regev, A. et al. *The Human Cell Atlas white blood cell (WB) pilot: single-cell transcriptomics of peripheral blood.* BioRxiv 2017. doi:10.1101/176578

- scRNA-seq of >68k PBMCs defining naïve/central-memory/effector-memory CD4⁺ and CD8⁺ T cells, MAIT, γδ T, NK, B cell maturation states, plasmablasts, dendritic cell subtypes, monocytes, eosinophils, basophils, and mast cells.
- Provides marker combinations for circulating subsets (CCR7/SELL/IL7R vs. CX3CR1/KLRG1 TEMRA, PRF1/GZMB cytotoxic programs, MS4A1/CD79A B-lineages, CLEC9A vs. CLEC10A DC splits) used in `markers.csv`.

## 2. Tabula Sapiens Consortium. *The Tabula Sapiens: A multiple-organ single-cell transcriptomic atlas of the human body.* Science 376, eabl4896 (2022)

- Includes matched blood and tissue immune cells, capturing migratory NK, NKT, ILC1/2/3, plasmacytoid DCs, inflammatory DC3, and myeloid-derived suppressor cells.
- Validates the inclusion of trafficking chemokines (CCR5, CXCR3/4/6), cytotoxic effectors (GNLY, FGFBP2), interferon-stimulated genes (ISG15, MX1/2), and suppressive mediators (ARG1, IDO1) in the immune block.

## 3. Monaco, G. et al. *RNA-Seq Signatures Normalized by mRNA Abundance Allow Absolute Deconvolution of Human Immune Cell Types.* Cell Reports 26, 1627-1640 (2019)

- Bulk RNA-seq of 29 purified immune populations (CD4⁺ subsets, Tfh, Tregs, CD8⁺, NK, B cell subsets, plasmablasts, monocytes, dendritic cells, mast cells, neutrophils, eosinophils, basophils).
- Supplies high-confidence genes (FOXP3, IKZF2, LRRC32 for Tregs; CXCL13, BCL6 for Tfh; CLEC9A vs. CLEC10A vs. CD1C splits for DCs) supporting the curated immune marker sets.

## 4. ImmGen Consortium. *The Immunological Genome Project.* Nature Immunology 9, 1091-1094 (2008) and updates

- Microarray/RNA-seq compendium of mouse and human immune populations highlighting transcription factor networks for effector-memory, exhausted, and innate subsets.
- Used to cross-reference transcriptional regulators (TOX, PRDM1, BATF3, ID2/3, ZBTB46, RUNX3, BHLHE40) assigned to migratory T, NK, ILC, and dendritic populations.

## 5. Villani, A.-C. et al. *Single-cell RNA-seq reveals new types of human blood dendritic cells, monocytes, and progenitors.* Science 356, eaah4573 (2017)

- Discovers circulating pre-DC, cDC2, DC3, and classical/non-classical monocyte subsets, including defining markers (LILRA4, TCF4, CLEC4C for pDC; AXL/SIGLEC6 pre-DC; CX3CR1/LILRB1 for non-classical monocytes) reflected in the updated immune block.
- Provides rationale for the added `Precursor dendritic cells (pre-DC)` row and the marker compositions for DC1/DC2/DC3/monocyte subsets.

## 6. Zheng, G.X.Y. et al. *Massively parallel digital transcriptional profiling of single cells.* Nature Communications 8, 14049 (2017)

- 10x Genomics PBMC benchmark dataset establishing robust gene signatures for NK (NKG7, PRF1, CST7), B cells (MS4A1, CD74, CD79A/B), plasmablasts (MZB1, XBP1, PRDM1), and cytotoxic/exhausted CD8⁺ T cells (LAG3, PDCD1, HAVCR2).
- Serves as an orthogonal validation source for the circulating immune gene panels, including exhausted CD8⁺, NK, and MAIT marker combinations.

## 7. Finak, G. et al. *MAIT cells in human peripheral blood support tissue T-cell homeostasis through IL-17 family cytokines.* Nature Communications 6, 6288 (2015)

- Characterizes human MAIT transcriptional programs (KLRB1, IL7R, CCR6, RORC, SLC4A10, ZBTB16) and effector molecules (IL17A/F, TNFSF11), supporting the MAIT-specific markers retained in the immune block.
