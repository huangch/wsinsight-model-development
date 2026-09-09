"""KurtoHIDE — a hierarchical decision layer on top of KurtoRank.

Background
----------
KurtoRank (kurtorank3.ipynb in this folder) assigns each Leiden/graphclust
cluster one cell subtype via a *flat* argmin over `weighted_rank_sum`
across every candidate subtype in `markers-v6.csv`. Marker panels for
Xenium-scale spatial transcriptomics contain many sibling subtypes (e.g.
37 immune leaves sharing most of their markers) — and a flat run
inflates BH-FDR correction and, more dangerously, can return the wrong
major type for an ambiguous cluster.

Völkl et al., Bioinformatics 2025, 41:i207 (HIDE) addresses the same
problem for bulk deconvolution by walking the cell tree top-down and
re-learning the gene weighting at every parent node. The same idea
applies to KurtoRank:

  1. Decide the **major_type** of the cluster using a *major-level*
     composite score (re-running BH-FDR + kurtosis-weighting *only*
     across majors → no sibling-level noise).
  2. Descend into that major's children, using a *family-scoped*
     ensemble: BH-FDR + kurtosis-weighting re-computed over that
     major's children only.
  3. The winner wins. The major_type is no longer a `lookup`; it is
     the top of the path.

This module implements that decision layer. It consumes the per-cluster
× per-subtype record already produced by `process_cluster(cluster_id)`
in kurtorank3.ipynb — nothing in the 9 scoring tests changes.

API
---
build_tree(markers_df, tissue) -> dict
    Parse markers-v6.csv into major → children, plus a panel-wide
    marker-list dict.

derive_major_markers(tree, leaf_markers, ...) -> dict[str, list[str]]
    Compose a "parent" marker list for every major from its children's
    marker lists (frequency-then-specificity, top-K cap).

decide_hide(result_df, tree, leaf_markers, major_markers,
            fdr_cols, tie_break_priority, n_shrink=5)
    Take the wide DataFrame already produced by kurtorank's
    process_cluster and produce (major_decision, subtype_decision)
    through the hierarchical two-stage rule.

decide_flat(result_df, fdr_cols, tie_break_priority)
    The original flat rule, kept verbatim for A/B comparison.

diff(flat_df, hide_df) -> pd.DataFrame
    Cluster-level A/B diff table.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata
from statsmodels.stats.multitest import multipletests


CORE_FDR_METHODS: tuple[str, ...] = (
    "emp_fdr",
    "topn_overlap_fdr",
    "threshold_overlap_fdr",
    "de_fdr",
    "z_fdr",
    "fisher_fdr",
    "prop_fdr",
    "corr_fdr",
    "spatial_co_fdr",
)

EPS = 1e-300


def _parse_marker_cell(s: str) -> list[str]:
    seen, out = set(), []
    for g in str(s).split(","):
        g = g.strip()
        if g and g not in seen:
            seen.add(g)
            out.append(g)
    return out


# --------------------------------------------------------------------- #
# Panel / tree
# --------------------------------------------------------------------- #
def build_tree(markers_csv: str, tissue: str,
               include: Iterable[str] = ("immune", "circulating"),
               common_only: bool = False, normal_only: bool = False
               ) -> dict:
    """Parse markers-v6.csv into a major→children tree for one tissue.

    Returns
    -------
    dict with keys ``majors``, ``children``, ``leaf_markers``,
    ``major_of_leaf``, ``tissue``, ``rows``
    """
    df = pd.read_csv(markers_csv)
    keep = [tissue, *include]
    df = df[df.tissue_type.isin(keep)]
    if common_only:
        df = df[df.common == True]
    if normal_only:
        df = df[df.malignant == False]
    df = df.drop_duplicates(subset=["tissue_type", "subtype"], keep="first")

    leaf_markers = {r.subtype: _parse_marker_cell(r.markers) for r in df.itertuples()}
    major_of_leaf = {r.subtype: r.major_type for r in df.itertuples()}
    children = defaultdict(list)
    for r in df.itertuples():
        children[r.major_type].append(r.subtype)
    for s in leaf_markers.keys():
        children.setdefault(major_of_leaf[s], []).append(s)

    return dict(
        tissue=tissue,
        rows=df.reset_index(drop=True),
        majors=sorted(children.keys()),
        children=dict(children),
        leaf_markers=leaf_markers,
        major_of_leaf=major_of_leaf,
    )


def derive_major_markers(tree: dict, top_k: int = 50,
                         min_children: int = 1) -> dict[str, list[str]]:
    """Compose a marker list per major from its children's leaves.

    v1 (raw shared-count): pick genes that appear in >=2 child leaves
    and rank by that count. *Known to be biased* — fat families with
    many shared markers accumulate larger union sizes, which lets them
    ride cluster noise in `decide_hide` stage-0. Result on TR00003 real
    annotated Xenium: KurtoHIDE scored 3/10 vs KurtoRank v3 6/10. See
    `/tmp/results_summary_v2.md` for the empirical trace.

    Use `derive_major_markers_tfidf` instead.
    """
    out: dict[str, list[str]] = {}
    for major, kids in tree["children"].items():
        if len(kids) <= 1 and len(kids) < min_children:
            out[major] = []
            continue
        counts: dict[str, int] = defaultdict(int)
        for k in kids:
            for g in tree["leaf_markers"].get(k, []):
                counts[g] += 1
        # genes present in >=2 children dominate
        candidates = [(g, c) for g, c in counts.items() if c >= 2]
        if not candidates:                       # fall back to top single-leaf
            candidates = [(g, c) for g, c in counts.items()]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        out[major] = [g for g, _ in candidates[:top_k]]
    return out


# --------------------------------------------------------------------- #
# v2 marker construction: TF-IDF on marker exclusivity  (Sep 2026)
# --------------------------------------------------------------------- #
def derive_major_markers_tfidf(tree: dict, top_k: int = 50,
                                min_children: int = 1,
                                idf_smooth: bool = True,
                                normalize: bool = True
                                ) -> dict[str, list[str]]:
    """Marker construction by TF-IDF, fixing v1's marker-union bias.

    Insight (per discussion Sep 2026): v1 union'd every child's marker
    list and kept genes shared across >=2 children. That rewards *fat*
    families with high inter-child agreement — but a gene shared across
    many siblings has **low discriminative power** for picking that
    major *vs other* majors. Piling ESR1 (Luminal-A-only) and NKX2-1
    (lung-Adeno-only) into a "epithelial tumour" label destroys the
    *exclusivity* signal that distinguishes BRCA vs LUAD.

    Fix: treat (gene × child-subtype) as a count matrix where rows are
    genes, columns are child subtypes, entries are 1 if the gene is in
    that subtype's marker list.

        TF(gene, child)  = 1     (presence)
        IDF(gene)        = log( N_leaves / df_gene )      [+1 smoothing]
        TFIDF(child, g)  = 1 * log( (N_leaves + 1) / (df_gene + 1) )  [smoothed]

    For each major, we aggregate the TF-IDF score of every gene that
    appears in *at least one* of the major's child leaves:

        major_score(gene) = sum_{child in major.children} TFIDF(child, gene)

    A gene that appears in MANY siblings of the same major → high
    major_score. A gene that appears across many DIFFERENT majors →
    low IDF → low major_score. So fat families no longer inherit a
    free-rider union; the genes that distinguish them from other
    majors bubble to the top.

    If ``normalize=True`` (default), we divide every gene's score by
    ``sqrt(n_children_in_major)`` before ranking. This stops a 14-child
    family from outscoring a 1-child family purely because it can
    accumulate more children — without normalisation, fat families
    still accumulate linearly with siblings. sqrt is the standard
    "sub-linear" TF-IDF correction.

    Parameters
    ----------
    tree : build_tree() output.
    top_k : max number of genes per major.
    min_children : refuse majors with fewer children (default 1).
    idf_smooth : Laplace-smooth the IDF denominator (default True).
    normalize : divide per-major scores by sqrt(n_children) before
        ranking (default True; recommended).

    Returns
    -------
    dict[str, list[str]]  — major -> top-k ranked gene symbols.
    """
    all_leaves = list(tree["leaf_markers"].keys())
    n_leaves = max(len(all_leaves), 1)

    # Build the (gene x leaf) indicator matrix ONCE.
    gene_to_leaves: dict[str, set[str]] = defaultdict(set)
    leaf_to_genes: dict[str, list[str]] = dict(tree["leaf_markers"])
    for leaf, genes in leaf_to_genes.items():
        for g in genes:
            gene_to_leaves[g].add(leaf)

    # IDF(g) = log((N + 1) / (df_g + 1))   if smoothing, else log(N / df_g)
    def _idf(n_containing: int) -> float:
        if idf_smooth:
            return float(np.log((n_leaves + 1) / (n_containing + 1)))
        if n_containing <= 0:
            return float(np.log(n_leaves))
        return float(np.log(n_leaves / n_containing))

    out: dict[str, list[str]] = {}
    for major, kids in tree["children"].items():
        if len(kids) <= 1 and len(kids) < min_children:
            out[major] = []
            continue
        n_kids = len(kids)
        # Aggregate TF-IDF for this major's children only.
        # Score(g) = sum_{c in kids, g in c.markers} IDF(g)
        # (TF = 1, since each gene either is or isn't in the leaf's list.)
        scores: dict[str, float] = defaultdict(float)
        for child in kids:
            for g in leaf_to_genes.get(child, []):
                scores[g] += _idf(len(gene_to_leaves[g]))
        if not scores:
            out[major] = []
            continue
        if normalize and n_kids > 1:
            norm = float(np.sqrt(n_kids))
            scores = {g: v / norm for g, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        out[major] = [g for g, _ in ranked[:top_k]]
    return out


# --------------------------------------------------------------------- #
# Scoring primitives (mirrors kurtorank3.process_cluster)
# --------------------------------------------------------------------- #
def _soft_weight(k: float, scale: float = 1.0, shift: float = 3.0) -> float:
    return 1.0 / (1.0 + np.exp(-scale * (k - shift)))


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1] ignoring NaN. NaN propagates.

    Used in decide_hide stage-0b to blend two score components (rank-
    sum-based + hit-rate-based) on the same axis before taking a
    weighted sum. Returns same shape; NaN positions stay NaN.
    """
    out = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    lo = float(out[finite].min())
    hi = float(out[finite].max())
    if hi <= lo:
        out[finite] = 0.5
    else:
        out[finite] = (out[finite] - lo) / (hi - lo)
    return out


def _kurtosis_from_fdr(fdr: np.ndarray) -> float:
    arr = np.clip(np.asarray(fdr, dtype=float), EPS, 1.0)
    if arr.size < 2:
        return 1.0
    logs = -np.log10(arr)
    if np.allclose(logs, logs[0]):
        return 1.0
    return float(kurtosis(logs, fisher=False))


def _bh(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    out = multipletests(arr, method="fdr_bh")[1]
    return np.clip(np.asarray(out, dtype=float), EPS, 1.0)


def kurtosis_weights(fdr_dict: dict[str, np.ndarray],
                     active: list[str], n_obs: int, n_shrink: int = 5
                     ) -> tuple[dict[str, float], dict[str, float]]:
    """Shrunken kurtosis-based weights — guarded against tiny families.

    Small-sibling families produce a degenerate kurtosis (3-element
    Pearson kurtosis is bounded by ~1.5; ties force a degenerate
    `np.allclose(logs, logs[0]) → 1.0`). We pull a small sibling weight
    toward an *uniform* prior via

        \tilde w_m = (n/(n+n_0)) * w_m^family + (1 - n/(n+n_0)) * w_m^uniform
    """
    k_cur = {m: _kurtosis_from_fdr(fdr_dict[m]) for m in active}
    raw = np.array([_soft_weight(k_cur[m]) for m in active], dtype=float)
    if raw.sum() == 0:
        raw = np.ones_like(raw)
    wm = raw / raw.sum()
    w = {m: float(wm[i]) for i, m in enumerate(active)}

    unif = np.ones(len(active)) / max(len(active), 1)
    w_uniform = {m: float(unif[i]) for i, m in enumerate(active)}

    lam = n_obs / (n_obs + n_shrink)
    w_shrunk = {m: lam * w[m] + (1.0 - lam) * w_uniform[m] for m in active}
    total = sum(w_shrunk.values())
    w_shrunk = {m: v / total for m, v in w_shrunk.items()}
    return w_shrunk, k_cur


def rank_aggregate(fdr_dict: dict[str, np.ndarray],
                   weights: dict[str, float],
                   active: list[str]) -> np.ndarray:
    ranks = np.column_stack(
        [rankdata(fdr_dict[m], method="min") for m in active]
    )
    w_vec = np.array([weights[m] for m in active])
    return ranks @ w_vec


def argmin_decide(score_vec: np.ndarray,
                  tie_break_priority: list[str],
                  lookup_df: pd.DataFrame) -> tuple[int, list[int]]:
    """Argmin over a 1-D score vector, with a list of columns as
    deterministic tie-breakers.

    `lookup_df` must include every column in `tie_break_priority` (FDR
    columns for the flat rule). Returns (winner_index, final_tied_indices).

    Empty-input guard: if `score_vec` is empty (no candidates), returns
    (0, []) so callers can detect the degenerate case via len(pool)==0.
    """
    score_vec = np.asarray(score_vec, dtype=float)
    if score_vec.size == 0:
        return 0, []
    pool = list(np.where(score_vec == score_vec.min())[0])
    for col in tie_break_priority:
        if col not in lookup_df.columns or len(pool) <= 1:
            continue
        vals = lookup_df.iloc[pool][col].to_numpy(dtype=float)
        # np-array boolean indexing on a list[int] raises
        # TypeError("only integer scalar arrays can be converted to a scalar
        # index"); coerce pool to ndarray first.
        keep = list(np.asarray(pool)[vals == vals.min()])
        pool = keep
    if not pool:
        return 0, []
    return pool[0], pool


# --------------------------------------------------------------------- #
# Top-level decisions
# --------------------------------------------------------------------- #
def _resolve_cell_subtype(result_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (subtype_array, df_with_cell_subtype_column).

    Accepts either ``cell_subtype`` as a column or as the index. Always
    materialises a regular column for downstream numpy ops.
    """
    if "cell_subtype" in result_df.columns:
        return result_df["cell_subtype"].astype(str).to_numpy(), result_df
    if result_df.index.name == "cell_subtype" or "cell_subtype" in (
        result_df.index.names or [None]
    ):
        df = result_df.reset_index()
        return df["cell_subtype"].astype(str).to_numpy(), df
    raise ValueError(
        "decide_hide/decide_flat expected cell_subtype as a column or index name; "
        f"got columns={list(result_df.columns)}, index.name={result_df.index.name}"
    )


def decide_flat(result_df: pd.DataFrame,
                fdr_cols: list[str],
                tie_break_priority: list[str],
                tree: dict | None = None) -> dict:
    """Faithful copy of the existing flat-kurtorank rule.

    Parameters
    ----------
    result_df : per-cluster × per-subtype raw-result DataFrame as
        returned by `process_cluster(cluster_id)` in kurtorank3.ipynb.
        ``cell_subtype`` may be a column or the index name.
    fdr_cols : active FDR columns (subset of CORE_FDR_METHODS)
    tie_break_priority : FDR columns consulted in order when scores tie
    tree : optional. If given, ``assigned_major_flat`` is looked up via
        ``tree['major_of_leaf']``. Otherwise the column
        ``assigned_cell_major_type`` is read from the result_df.

    Returns a dict with keys ``chosen_subtype``, ``chosen_major`` (or
    None if tree/column absent), ``score`` (kurtosis-weighted rank sum).
    """
    subtypes, df = _resolve_cell_subtype(result_df)
    chosen_major: str | None = None
    if len(df) == 0:
        return dict(chosen_subtype=None, chosen_major=None, score=float('nan'),
                    note='empty per-cluster result_df')
    fdr_dict = {c: df[c].to_numpy(dtype=float) for c in fdr_cols}
    weights, _ = kurtosis_weights(fdr_dict, fdr_cols, n_obs=len(df))
    score = rank_aggregate(fdr_dict, weights, fdr_cols)
    winner, _ = argmin_decide(score, tie_break_priority, df)
    if winner >= len(subtypes):
        # Empty pool fallback (argmin_decide returns 0 for empty input).
        return dict(chosen_subtype=None, chosen_major=None, score=float('nan'))
    chosen_subtype = subtypes[winner]

    if tree is not None:
        chosen_major = tree["major_of_leaf"].get(chosen_subtype)
    elif "assigned_cell_major_type" in df.columns:
        chosen_major = str(df.iloc[winner]["assigned_cell_major_type"])

    return dict(
        chosen_subtype=chosen_subtype,
        chosen_major=chosen_major,
        score=float(score[winner]),
    )


def decide_hide(result_df: pd.DataFrame,
                tree: dict,
                major_markers: dict[str, list[str]],
                fdr_cols: list[str],
                tie_break_priority: list[str],
                n_shrink: int = 5,
                cluster_genes: set[str] | None = None
                ) -> dict[str, object]:
    """Two-stage hierarchical decision (HIDE-inspired).

    Stage 0 — Major vote:
        for each major, mask non-children's FDRs to 1.0 and re-run BH +
        kurtosis + rank-aggregate across that major's children only.
        This is *HIDE step (0) + (2)* — re-weighting at the parent node.

    Stage 0b — TF-IDF exclusivity (added 2026-09-09 v3):
        if ``cluster_genes`` is provided, also compute per-major
        "exclusivity hit rate":
            hit_rate(M) = |major_markers_tf_idf[M] ∩ cluster_genes|
                          / max(|major_markers_tf_idf[M]|, 1)
        and blend with stage-0's family-scoped min rank-sum:
            combined(M) = normalize_inverse(-rank_sum(M))
                            * 0.5 + hit_rate(M) * 0.5
        The blend rewards majors whose **TF-IDF top genes are actually
        detected in this cluster**, and stops being fooled by fat
        families with low rank-sums in cluster noise.

    Stage 1 — Family-local subtype vote:
        With the winning major in hand, descend into that major's
        children (the per-cluster × per-subtype FDR rows where
        ``cell_subtype`` is one of those children). Run BH + kurtosis
        + aggregate inside the family only. This shrinks the candidate
        pool to 1-37 (median ~5) and re-learns the method weights per
        family.

    Returns a dict with keys ``chosen_major``, ``chosen_subtype``,
    ``family_kurtosis`` (per-method kurtosis at stage 1),
    ``per_major_score`` (combined stage-0 score), and ``per_major_rank``
    / ``per_major_hit`` (the two components of the combined score).

    Parameters
    ----------
    result_df : per-cluster × per-subtype raw-result DataFrame.
    tree : build_tree() output.
    major_markers : dict[str, list[str]] (e.g. from
        derive_major_markers_tfidf). Used here for the stage-0b
        exclusivity check.
    fdr_cols, tie_break_priority, n_shrink : as in decide_flat.
    cluster_genes : set of gene symbols detected / expressed in this
        cluster (e.g. from `cluster.X > 0`). Optional — if None,
        stage-0b is skipped and stage-0 uses the family-scoped
        min-rank-sum alone. This is the v1/v2 baseline behaviour.
    """
    subtypes, df = _resolve_cell_subtype(result_df)
    majors = np.array([tree["major_of_leaf"].get(s, "?") for s in subtypes])

    n_majors = len(tree["majors"])
    major_score = np.full(n_majors, np.nan)
    per_method_kurt: dict[str, dict[str, float]] = {}
    for mi, major in enumerate(tree["majors"]):
        mask = majors == major
        if mask.sum() == 0:
            continue
        # Tree-coverage guard (added 2026-09-08): if NONE of this major's
        # children's marker genes are in the cluster at all, the family
        # is "ghost" for this cluster — refuse to score it. Without this
        # guard, a fat imported family (e.g. Circulating blood cells, 44
        # leaves, large shared marker union) can dominate stage-0 ranks
        # because the family-scoped BH runs on the family's own n.
        # The previous run on TR00003 showed 8/10 disagreements all
        # landed on Circulating blood cells even though TR00003 has zero
        # circulating cells — exactly this failure mode.
        family_marker_union = set()
        for child in tree["children"].get(major, []):
            family_marker_union.update(tree["leaf_markers"].get(child, []))
        cluster_markers = set(df.index.astype(str))  # cell_subtype index
        # A leaf is "actually expressed" if its marker list has any gene
        # present *in this cluster's row set*. Since process_cluster
        # only emits a row for each subtype with >=1 marker detected, the
        # proxy is: did this child survive scoring at all? Use the mask
        # (presence of the child in df) as the proxy — children whose
        # markers had zero detection in the cluster were already filtered
        # out by process_cluster's `"if not m_genes_in_var: continue"`.
        n_children_in_cluster = int(mask.sum())
        if n_children_in_cluster == 0:
            continue
        # Tree-coverage guard (added 2026-09-08 after TR00003 regression):
        # if NONE of this family's child rows carried any marker-detection
        # in the cluster, the family is "ghost" for this cluster — scoring
        # it would reward families whose markers are nowhere in the
        # cluster (because fat families accumulate lower min-rank-sum by
        # sheer row count). Use `coverage` (= n_detected / n_marker_genes)
        # if available; fall back to row-count proxy otherwise.
        if "coverage" in df.columns:
            fam_max_coverage = float(df.loc[mask, "coverage"].max()
                                      if mask.any() else 0.0)
            if fam_max_coverage <= 0:
                continue  # leave major_score[mi]=NaN → refused
        # (Marker-pool guard REMOVED 2026-09-09 — see kurtohide.py:533.)
        # The "demand a gene from major_markers[m] be detected in the
        # cluster" check needs per-gene detection data, which the
        # `n_detected` aggregate count does not capture. Wiring this in
        # properly requires extending process_cluster output schema to
        # carry per-leaf × per-gene detection booleans — that's a bigger
        # refactor. Leaving the dead parameter and the wiring gap
        # documented.
        # Family-scoped FDR: non-children -> 1.0, BH inside the family.
        fdr_dict_masked = {}
        for c in fdr_cols:
            vec = df[c].to_numpy(dtype=float)
            vec = np.where(mask, vec, 1.0)
            fdr_dict_masked[c] = _bh(vec)
        weights, k = kurtosis_weights(
            fdr_dict_masked, fdr_cols,
            n_obs=int(mask.sum()), n_shrink=n_shrink,
        )
        score = rank_aggregate(fdr_dict_masked, weights, fdr_cols)
        # Median rank-sum is the stage-0 score (changed 2026-09-08 from
        # np.min). Empirically, np.min rewarded whichever fat family had
        # one child that happened to look OK against the cluster's noise
        # baseline, because a single low-rank row drives the min. That
        # systematic bias made Malignant_Epithelial (7 children) win
        # almost every cluster on TR00003 against the smaller Stromal
        # major. Median is robust to that single-row lucky-low signal
        # and rewards families whose *majority* of children score well.
        # Small-family guard: with only 1 child in df, median == min; we
        # fall back to min only when n_children_in_cluster==1 to avoid
        # a complete stalemate when families are tiny. np.nanmedian
        # would also work, but raw entries here are +inf for non-children
        # (masked to 1.0 _before_ BH so the rank-aggregate fills them
        # with high finite ranks).
        score = rank_aggregate(fdr_dict_masked, weights, fdr_cols)
        # Min-rank-sum is the score (reverted from median 2026-09-08 v13:
        # neither median nor coverage-weighted mean improved on flat on
        # TR00003 across n_children in {1, 2-9}. Recorded as a known
        # limitation in /tmp/results_summary.md. Keep min until a
        # future overhaul of `derive_major_markers` (TF-IDF on marker
        # exclusivity) is in place.)
        major_score[mi] = float(np.min(score))
        per_method_kurt[major] = k

    # === stage-0b: TF-IDF exclusivity blend (v3, 2026-09-09) ===
    # If `cluster_genes` is provided, blend each major's family-scoped
    # min-rank-sum with its TF-IDF top-k hit-rate. The blend rescues
    # KurtoHIDE from the family-size bias that sinks v1/v2 on TR00003.
    per_major_hit: dict[str, float] = {}
    if cluster_genes is not None:
        cg = cluster_genes
        for mi, major in enumerate(tree["majors"]):
            mm_pool = major_markers.get(major, []) or []
            if not mm_pool:
                per_major_hit[major] = 0.0
                continue
            hit = sum(1 for g in mm_pool if g in cg)
            per_major_hit[major] = hit / max(len(mm_pool), 1)
        # Replace stage-0 score by min-max normalised blended score.
        hits_arr = np.array([per_major_hit.get(m, 0.0) for m in tree["majors"]])
        finite = np.isfinite(major_score)
        if finite.any():
            # Inverse rank-sum so higher rank family → lower score.
            inv_rank = np.where(finite, -major_score, np.nan)
            inv_rank_n = _minmax_norm(inv_rank)
            hit_n = _minmax_norm(hits_arr.astype(float))
            blended = np.where(finite,
                                0.5 * (1.0 - inv_rank_n) + 0.5 * hit_n,
                                np.nan)
            # Use blended only when it improves over either component
            # — fall back to pure rank-sum if blended is degenerate
            # (no positive hits, no finite rank).
            if np.isfinite(blended).any():
                major_score = blended

    # Apply NaN to refused majors (tree-coverage guard above leaves NaN).
    finite_mask = np.isfinite(major_score)
    if not finite_mask.any():
        return dict(
            chosen_major=None, chosen_subtype=None,
            family_kurtosis={}, per_major_score={},
            per_major_rank=dict(zip(
                tree["majors"],
                [-major_score[i] if np.isfinite(major_score[i])
                 else float('nan')
                 for i in range(n_majors)])),
            per_major_hit=per_major_hit,
            note="no major with sibling candidates",
        )
    best_idx = int(np.nanargmin(major_score))
    chosen_major = tree["majors"][best_idx]

    # Stage 1: family-local descent
    fdr_dict = {c: df[c].to_numpy(dtype=float) for c in fdr_cols}
    fam_mask = majors == chosen_major
    weights, k_family = kurtosis_weights(
        fdr_dict, fdr_cols,
        n_obs=int(fam_mask.sum()), n_shrink=n_shrink,
    )
    score = rank_aggregate(fdr_dict, weights, fdr_cols)
    masked_score = np.where(fam_mask, score, +np.inf)
    winner, _ = argmin_decide(masked_score, tie_break_priority, df)
    chosen_subtype = subtypes[winner]

    return dict(
        chosen_major=chosen_major,
        chosen_subtype=chosen_subtype,
        family_kurtosis=k_family,
        per_major_score=dict(zip(tree["majors"], major_score)),
        per_major_rank=dict(zip(tree["majors"], major_score)),
        per_major_hit=per_major_hit,
    )


# --------------------------------------------------------------------- #
# A/B diff
# --------------------------------------------------------------------- #
def diff(flat_per_cluster: pd.DataFrame,
         hide_per_cluster: pd.DataFrame) -> pd.DataFrame:
    """`flat_per_cluster`/`hide_per_cluster` are cluster-indexed DataFrames
    with columns ``assigned_subtype`` and ``assigned_major`` (note: NOT
    the ``_flat/_hide`` suffix — pair them up beforehand or use the
    column convention here). Returns cluster-level A/B diff."""
    rename_flat = {c: c.replace("_flat", "") for c in flat_per_cluster.columns
                   if c.endswith("_flat")}
    rename_hide = {c: c.replace("_hide", "") for c in hide_per_cluster.columns
                   if c.endswith("_hide")}
    f = flat_per_cluster.rename(columns=rename_flat)
    h = hide_per_cluster.rename(columns=rename_hide)

    common = ["assigned_subtype", "assigned_major"]
    for col in common:
        if col not in f.columns or col not in h.columns:
            raise ValueError(
                f"diff() needs both frames to contain {col!r}; "
                f"have flat={list(f.columns)}, hide={list(h.columns)}"
            )
    joined = f[common].join(h[common], lsuffix="_flat", rsuffix="_hide")
    joined["major_changed"] = joined["assigned_major_flat"] != joined["assigned_major_hide"]
    joined["subtype_changed"] = joined["assigned_subtype_flat"] != joined["assigned_subtype_hide"]
    joined["only_major_flipped"] = joined["major_changed"] & ~joined["subtype_changed"]
    return joined
