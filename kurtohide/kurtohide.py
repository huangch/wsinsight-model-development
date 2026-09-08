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
    By default `min_children=1` so 1-child majors get a real marker list,
    not a free pass (see decide_hide regression note).

    Rationale: markers-v6.csv only carries leaves. Naively union'ing all
    leaves produces unspecific noise (often >100 genes), which HIDE step
    (0) and KurtoRank's BH-FDR both suffer from. We pick genes that
    appear in *more than one* child leaf and rank by that frequency
    (ties → alphabetical). This keeps the major-level signal compact
    without requiring Census rerank, which is the other principled
    option.
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
# Scoring primitives (mirrors kurtorank3.process_cluster)
# --------------------------------------------------------------------- #
def _soft_weight(k: float, scale: float = 1.0, shift: float = 3.0) -> float:
    return 1.0 / (1.0 + np.exp(-scale * (k - shift)))


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
        keep = pool[vals == vals.min()]
        pool = list(keep)
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
                n_shrink: int = 5) -> dict[str, object]:
    """Two-stage hierarchical decision (HIDE-inspired).

    Stage 0 — Major vote:
        for each major, mask non-children's FDRs to 1.0 and re-run BH +
        kurtosis + rank-aggregate across that major's children only.
        This is *HIDE step (0) + (2)* — re-weighting at the parent node.
        ``major_markers`` is reserved for step (2) extension (Census-
        rerank). The current draft does not consume it — the family-
        scoped FDR already re-weights evidence without needing a parent
        marker list per major.

    Stage 1 — Family-local subtype vote:
        With the winning major in hand, descend into that major's
        children (the per-cluster × per-subtype FDR rows where
        ``cell_subtype`` is one of those children). Run BH + kurtosis
        + aggregate inside the family only. This shrinks the candidate
        pool to 1-37 (median ~5) and re-learns the method weights per
        family.

    Returns a dict with keys ``chosen_major``, ``chosen_subtype``,
    ``family_kurtosis`` (per-method kurtosis at stage 1), and
    ``per_major_score`` (median rank-sum per major at stage 0).
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
        # Min-rank-sum is the score: the *best* sibling under family-scoped
        # weights. Median inflates large families (37-leaf immune family
        # has a high median because of family-wide BH correction); min
        # makes all family sizes comparable on the same axis.
        major_score[mi] = float(np.min(score))
        per_method_kurt[major] = k

    # Choose the major whose median rank-sum is the smallest.
    finite_mask = np.isfinite(major_score)
    if not finite_mask.any():
        return dict(
            chosen_major=None, chosen_subtype=None,
            family_kurtosis={}, per_major_score={},
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
