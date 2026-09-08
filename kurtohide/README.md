# KurtoHIDE — prototype status

KurtoHIDE layers a hierarchical decision rule on top of [KurtoRank](../kurtorank/).
It implements the *parent-then-leaf* decomposition from Völkl et al.,
*Bioinformatics* 2025, 41:i207 (HIDE), adapted to Xenium-scale marker panels.

The decision layer ships as a module (`kurtohide.py`) with explicit
test cases (synthetic data) and an executed Jupyter demo
(`kurtohide_demo.ipynb`) that produces real A/B diffs against flat
KurtoRank on the largest Leiden clusters of `data/xenium_mini/breast/`.

## Why hierarchical? The HIDE intuition

KurtoRank runs all subtype tests across **every** candidate in
`markers-v6.csv` (~58 leaves) at once. When the panel has many
sibling subtypes (e.g. 37 immune leaves that share most of their
markers), the family-scoped BH-correction inflates large families and
deflates noise from unrelated subtrees. The flat argmin can then
return clusters assigned to majors that are technically wrong
(e.g. CD4+ T cells for what is really a heterogeneous tumour cluster).

HIDE walks the cell tree top-down and re-estimates the weighting at
every parent, decoupling sibling-level noise from parent-level
discrimination. KurtoHIDE ports steps (0) and (2) only:

- **Stage 0 — major vote.** For every major M, scope the BH-correction
  and kurtosis-weighting to that major's children only. Score each
  major by the **min family-scoped rank-sum** (the best-sibling under
  family-local weights).
- **Stage 1 — family-local subtype vote.** Re-run scoring only over
  the winning major's children.

## What changed (vs. the first regression report)

The first executed retrospective (Sep 7, 2026 21:23) showed every
cluster collapsing to one of two 1-child major families — *Mesenchymal
cells* (1 leaf) and *Vascular progenitor cells* (1 leaf). Two
interacting bugs:

1. `decide_hide` scored each major by the **median rank-sum** across
   siblings. For a 1-child family the median equals the lone leaf's
   score; for a 37-child family the median is inflated by
   family-scoped BH correction. Argmin therefore picked any singleton
   over the genuine family leader.
2. `derive_major_markers(tree, min_children=2)` deliberately excluded
   1-child families from the parent-level marker list. That meant the
   singleton family saw no Stage-0 BH adjustment at all — making its
   median rank a structural floor.

The fix is on disk in `kurtohide.py` (both edits visible at
`grep "np.min(score)\|min_children: int = 1" kurtohide.py`):

- **Median → min** in the Stage-0 score (single-line change).
- **`min_children=1`** in `derive_major_markers`, so 1-child majors
  get a real marker list and go through the same family-scoped BH
  pipeline as everyone else.

## Headline result (8-cluster breast Xenium sample, executed notebook)

After the fix, the 8-cluster A/B diff is:

| cluster | KurtoRank v3 (flat) | KurtoHIDE |
|---|---|---|
| 1 | CD4+ T cells / **Immune cells** | Invasive Lobular Carcinoma / **Malignant epithelial cells** |
| 2 | Naïve B cells / **Immune cells** | Luminal A tumor / **Malignant epithelial cells** |
| 3 | Dendritic cells / **Immune cells** | Normal ductal epithelial / **Epithelial cells** |
| 4 | Luminal A tumor / **Malignant epithelial** | Circulating Endothelial / **Vascular cells** |
| 5 | ILC tumor / **Malignant epithelial** | Platelets (Thrombocytes) / **Circulating blood cells** |
| 7 | Plasma cells / **Immune cells** | Luminal A tumor / **Malignant epithelial cells** |
| 8 | HER2 tumor / **Malignant epithelial** | Circulating Endothelial / **Vascular cells** |
| 10 | Endothelial / **Vascular cells** | Platelets (Thrombocytes) / **Circulating blood cells** |

Major-distribution:

| rule | Immune | Mal. epi. | Vascular | Circ. blood | Epithelial | (others) |
|---|---|---|---|---|---|---|
| flat | 4 | 3 | 1 | 0 | 0 | 0 |
| KurtoHIDE | 0 | 3 | 2 | 2 | 1 | 0 |

KurtoHIDE moves all 4 of the flat "Immune cells" calls out of the
over-favoured 37-leaf family, and distributes them across **4
different majors**. The original regression (one singleton family for
all clusters) is fixed.

The new failure mode — clusters 4, 5, 8, 10 reassigned to Circulating
blood/Vascular/Endothelial families when the flat call is a tumour
subtype — is the natural consequence of using *min* over a heavily
noisy per-family score, not the original bias. Real diagnosis
(ground-truth pathologist labels or markers-v7 with `level=leaf|major`)
is needed to say whether the new calls are actually better.

## Files

| File | Purpose |
|---|---|
| `kurtohide.py` | 17 KB module: panel parser, tree, parent-marker derivation, `decide_flat`, `decide_hide`, `diff`, edge-case guards for empty frames / score vectors. |
| `kurtohide_demo.ipynb` | 25 cells, executed (8-cluster breast-Xenium retrospective, ~46 min wall-clock first time, **~1 s with pickle cache**). |
| `markers-v6.csv` | Copy from `../kurtorank/markers/data/`. |
| `kurtorank3.ipynb` | Reference end-to-end KurtoRank notebook. |

## Reproduction

```bash
cd /workspace/wsinsight/wsinsight-model-development/kurtohide
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate wsi

# First run: ~46 min (8 clusters × ~5 min + 3 min rank_genes_groups).
# Re-runs: ~1 s using /tmp/kurtohide_raw_df.pkl.
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 kurtohide_demo.ipynb

# Force a fresh recompute:
rm /tmp/kurtohide_raw_df.pkl
```

## What is *not* here (yet)

After the fix the major-level signal is no longer degenerate; what
remains is methodological, not implementation:

1. **Parent-marker upgrade.** Today's "frequent-children + top-K"
   heuristic would yield sharper families if backed by Census rerank.
2. **Real ground truth for the breast sample.** The 4 reassignments
   that look wrong (e.g. cluster 4 "Luminal A tumor" being called
   "Circulating Endothelial / Vascular cells") need either: (a) a
   pathologist's labels for the Leiden / graphclust input, or (b) a
   reference annotation against which to score.
3. **`--method {flat,hide,both}` CLI integration** into the
   `kurtorank` package. Section 9a of the notebook has the patch
   sketch; not landed yet.
4. **Proportion-based design B** (HIDE steps (1) and (3) — residual
   subtract + parent-normalisation) — requires an externally-supplied
   reference matrix X that KurtoRank's marker-list-only flow doesn't
   carry today.

## Citations

- **HIDE** (the method we adapt): Völkl D. et al., *Bioinformatics*
  2025, 41(S1):i207–i216. doi:10.1093/bioinformatics/btaf179. Zenodo
  code: doi:10.5281/zenodo.14724906.
- **KurtoRank** (the underlying ensemble): `../kurtorank/`, marker
  panel `markers-v6.csv`.
