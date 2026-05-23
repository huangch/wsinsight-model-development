"""
rank_markers.py
---------------
Rerank marker-gene lists in markers-v3.csv by atlas-derived specificity
scores using the CELLxGENE Census (human reference).

Policy (from chat discussion):
  - Ranking source: CELLxGENE Census (streaming, no bulk download).
  - Composite score = 0.4*AUC + 0.3*min(log2FC/3, 1) + 0.2*(pct_in - pct_out)
                      + 0.1*lit_score   (lit_score: small prior from hne_label
                      agreement; not a literature DB in this run).
  - Adaptive cutoff: keep genes with composite >= tau (default 0.30),
    with floor = 5 and ceiling = 50 per subtype.
  - Rows where Census yields too few target cells fall back to
    `rank_source = "v3_curated"` (keep original order, trimmed to ceiling).

Outputs:
  - markers-v3.csv         : overwritten with reordered `markers` column plus
                             two new columns: `rank_source`, `low_support`.
  - markers-v3_qc.csv      : one row per (subtype, gene) with component scores
                             and assigned rank, to audit later.

Usage:
    python rank_markers.py              # run on all 337 rows
    python rank_markers.py --tissue breast   # restrict to one tissue (debug)
    python rank_markers.py --dry-run    # compute but do not overwrite v3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Silence the anndata "ImplicitModificationWarning: Transforming to str index"
# that Census get_anndata emits per query; it polluted the log so badly that
# it looked like the program was waiting for input.
try:
    from anndata import ImplicitModificationWarning
    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
except Exception:
    warnings.filterwarnings("ignore", message="Transforming to str index")


# ---------------------------------------------------------------------------
# 0) Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("rank_markers")

def _setup_logging(verbose: bool, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    # Reset handlers so repeated calls (e.g. in spawned workers) don't stack.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)


# A dedicated output stream for the live status line. When stderr is a pipe
# (e.g. ``2>&1 | tee``) the carriage-return trick breaks because tee buffers
# the byte stream and the terminal sees every update as a new line. To avoid
# that entirely we open /dev/tty directly when available; otherwise we fall
# back to stderr.
_STATUS_STREAM = None

def _status_stream():
    global _STATUS_STREAM
    if _STATUS_STREAM is not None:
        return _STATUS_STREAM
    try:
        _STATUS_STREAM = open("/dev/tty", "w", buffering=1)
    except Exception:
        _STATUS_STREAM = sys.stderr
    return _STATUS_STREAM


# ---------------------------------------------------------------------------
# 1) v3 tissue -> Census `tissue_general` mapping (verified 2025-11-08 release)
# ---------------------------------------------------------------------------

TISSUE_MAP: dict[str, list[str]] = {
    "bladder":     ["urinary bladder", "bladder organ"],
    "bone":        ["bone marrow"],  # no standalone bone in Census
    "brain":       ["brain", "central nervous system"],
    "breast":      ["breast"],
    "cervix":      ["uterus"],  # no cervix label; use uterus
    "circulating": ["blood", "bone marrow"],
    "colorectal":  ["colon", "large intestine", "small intestine"],
    "heart":       ["heart"],
    "immune":      ["lymph node", "blood", "bone marrow", "spleen", "thymus"],
    "kidney":      ["kidney"],
    "liver":       ["liver"],
    "lung":        ["lung", "respiratory system"],
    "lymph_node":  ["lymph node"],
    "ovary":       ["ovary"],
    "pancreas":    ["pancreas"],
    "prostate":    ["prostate gland"],
    "skin":        ["skin of body"],
    "tonsil":      ["lymph node"],  # no tonsil label; lymph node is closest
}

# Min/max target cells in Census for scoring to proceed.
MIN_TARGET_CELLS = 100
MIN_BG_CELLS = 300
TARGET_SAMPLE_CAP = 8_000
# 20k was gratuitous; Mann-Whitney on 5k vs 8k is indistinguishable at this
# gene count and 4x faster per query.
BG_SAMPLE_CAP = 5_000

# Adaptive cutoff parameters.
TAU = 0.30
FLOOR_K = 5
CEILING_K = 50


# ---------------------------------------------------------------------------
# 2) Subtype -> Census `cell_type` keyword hints
# ---------------------------------------------------------------------------
# Rules:
#   - Tokenized keyword match is case-insensitive on the Census cell_type name
#     AFTER normalization (lowercase, strip punctuation).
#   - A Census cell_type matches if ALL `must` tokens are present and NONE of
#     the `must_not` tokens are present.
#   - If a subtype has no entry below, we derive keywords from its `hne_label`
#     and `major_type` automatically (HNE_LABEL_KEYS + MAJOR_TYPE_KEYS).
#
# These aliases cover the ambiguous / tricky cases; the automatic derivation
# handles the long tail.

HNE_LABEL_KEYS: dict[str, dict] = {
    "fibroblast_like":       {"must_any": ["fibroblast"]},
    "endothelial":           {"must_any": ["endothelial"]},
    "lymphocyte":            {"must_any": ["t cell", "b cell", "lymphocyte",
                                           "nk cell", "natural killer",
                                           "cd4", "cd8"]},
    "macrophage_like":       {"must_any": ["macrophage", "monocyte",
                                           "microglia", "kupffer",
                                           "dendritic cell"]},
    "plasma_cell":           {"must_any": ["plasma cell"]},
    "mast_cell":             {"must_any": ["mast cell"]},
    "neutrophil":            {"must_any": ["neutrophil"]},
    "basophil":              {"must_any": ["basophil"]},
    "eosinophil":            {"must_any": ["eosinophil"]},
    "platelet":              {"must_any": ["platelet", "megakaryocyte"]},
    "red_blood":             {"must_any": ["erythrocyte", "erythroid",
                                           "red blood"]},
    "hematologic_blast":     {"must_any": ["blast", "hematopoietic stem",
                                           "progenitor"]},
    "basaloid_progenitor":   {"must_any": ["basal cell", "progenitor"]},
    "pericyte":              {"must_any": ["pericyte"]},
    "perivascular":          {"must_any": ["perivascular"]},
    "smooth_muscle":         {"must_any": ["smooth muscle"]},
    "cardiomyocyte":         {"must_any": ["cardiomyocyte", "myocardial"]},
    "adipocyte":             {"must_any": ["adipocyte"]},
    "osteoblast":            {"must_any": ["osteoblast"]},
    "osteoclast":            {"must_any": ["osteoclast"]},
    "osteocyte":             {"must_any": ["osteocyte"]},
    "chondrocyte":           {"must_any": ["chondrocyte"]},
    "schwann":               {"must_any": ["schwann"]},
    "glial":                 {"must_any": ["glial", "astrocyte",
                                           "oligodendrocyte"]},
    "ependymal":             {"must_any": ["ependymal"]},
    "neuron":                {"must_any": ["neuron"]},
    "neuroendocrine":        {"must_any": ["neuroendocrine",
                                           "enteroendocrine"]},
    "mesothelial":           {"must_any": ["mesothelial"]},
    "podocyte":              {"must_any": ["podocyte"]},
    "mesangial":             {"must_any": ["mesangial"]},
    "germ_cell":             {"must_any": ["germ cell", "oocyte",
                                           "spermatocyte"]},
    "melanocyte":            {"must_any": ["melanocyte"]},
    "epithelial":            {"must_any": ["epithelial", "luminal", "basal",
                                           "ductal", "club", "goblet",
                                           "enterocyte", "hepatocyte",
                                           "acinar", "tubule",
                                           "urothelial"]},
    # Malignant labels collapse to "malignant cell" in Census.
    "malignant_epithelial":  {"must_any": ["malignant"]},
    "malignant_neuroendocrine": {"must_any": ["malignant"]},
    "malignant_endothelial": {"must_any": ["malignant"]},
    "malignant_muscle":      {"must_any": ["malignant"]},
    "malignant_melanocytic": {"must_any": ["malignant"]},
    "malignant_mixed":       {"must_any": ["malignant"]},
    "embryonal_tumor":       {"must_any": ["malignant"]},
}

# Subtype-name overrides for cases where the hne_label bucket is too coarse
# and we want to steer the scorer to the matched normal lineage even for
# malignant rows (so we get a meaningful AUC rather than a "malignant vs
# everything else" signal that is uninformative for reranking).
SUBTYPE_OVERRIDES: dict[str, dict] = {
    # Breast
    "Luminal A tumor cells":            {"must_any": ["luminal"]},
    "Luminal B tumor cells":            {"must_any": ["luminal"]},
    "HER2-enriched tumor cells":        {"must_any": ["luminal"]},
    "Basal-like tumor cells":           {"must_any": ["basal"]},
    "Claudin-low tumor cells":          {"must_any": ["basal",
                                                      "myoepithelial"]},
    "Invasive Lobular Carcinoma (ILC) tumor cells":
                                        {"must_any": ["luminal"]},
    "Normal ductal epithelial cells":   {"must_any": ["ductal", "luminal"]},
    "Myoepithelial cells":              {"must_any": ["myoepithelial",
                                                      "basal"]},

    # Bladder
    "Luminal-papillary tumor cells":    {"must_any": ["urothelial",
                                                      "luminal"]},
    "Luminal-infiltrated tumor cells":  {"must_any": ["urothelial",
                                                      "luminal"]},
    "Luminal-unstable tumor cells":     {"must_any": ["urothelial",
                                                      "luminal"]},
    "Umbrella urothelial cells":        {"must_any": ["urothelial",
                                                      "umbrella"]},
    "Intermediate urothelial cells":    {"must_any": ["urothelial"]},
    "Basal urothelial cells":           {"must_any": ["basal"]},

    # CRC
    "Enterocytes":                      {"must_any": ["enterocyte"]},
    "Goblet cells":                     {"must_any": ["goblet"]},
    "Paneth cells":                     {"must_any": ["paneth"]},
    "Tuft cells":                       {"must_any": ["tuft"]},
    "Enteroendocrine cells":            {"must_any": ["enteroendocrine"]},

    # Liver / biliary
    "Hepatocytes":                      {"must_any": ["hepatocyte"]},
    "Cholangiocytes":                   {"must_any": ["cholangiocyte",
                                                      "bile duct"]},

    # Kidney
    "Proximal tubule cells":            {"must_any": ["proximal"]},
    "Distal tubule cells":              {"must_any": ["distal"]},
    "Podocytes":                        {"must_any": ["podocyte"]},
    "Mesangial cells":                  {"must_any": ["mesangial"]},

    # Pancreas
    "Acinar cells":                     {"must_any": ["acinar"]},
    "Ductal cells":                     {"must_any": ["ductal"]},
    "Alpha cells":                      {"must_any": ["alpha cell",
                                                      "pancreatic a"]},
    "Beta cells":                       {"must_any": ["beta cell",
                                                      "pancreatic b"]},

    # Prostate
    "Luminal epithelial cells":         {"must_any": ["luminal"]},
    "Basal epithelial cells":           {"must_any": ["basal"]},

    # Lung
    "Alveolar type 1 cells":            {"must_any": ["type i pneumocyte",
                                                      "alveolar type 1",
                                                      "alveolar type i"]},
    "Alveolar type 2 cells":            {"must_any": ["type ii pneumocyte",
                                                      "alveolar type 2",
                                                      "alveolar type ii"]},
    "Club cells":                       {"must_any": ["club"]},
    "Ciliated cells":                   {"must_any": ["ciliated"]},

    # Immune subtypes
    "CD4+ T cells":                     {"must_any": ["cd4"]},
    "CD8+ T cells":                     {"must_any": ["cd8"]},
    "Regulatory T cells":               {"must_any": ["regulatory t",
                                                      "treg"]},
    "B cells":                          {"must_any": ["b cell"]},
    "NK cells":                         {"must_any": ["natural killer",
                                                      "nk cell"]},
    "Dendritic cells":                  {"must_any": ["dendritic cell"]},
    "Macrophages":                      {"must_any": ["macrophage"]},
    "Monocytes":                        {"must_any": ["monocyte"]},
    "Plasma cells":                     {"must_any": ["plasma cell"]},
    "Mast cells":                       {"must_any": ["mast cell"]},
    "Neutrophils":                      {"must_any": ["neutrophil"]},
}


# ---------------------------------------------------------------------------
# 3) Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.lower()

def _match_cell_types(
    available: Iterable[str], hint: dict
) -> list[str]:
    """Return the subset of `available` cell_type labels matching `hint`."""
    must_any = [_norm(k) for k in hint.get("must_any", [])]
    must_not = [_norm(k) for k in hint.get("must_not", [])]
    out = []
    for ct in available:
        n = _norm(ct)
        if must_any and not any(k in n for k in must_any):
            continue
        if must_not and any(k in n for k in must_not):
            continue
        out.append(ct)
    return out


def resolve_hint(row: pd.Series) -> dict | None:
    """Pick the best keyword hint for a v3 row."""
    sub = row["subtype"]
    if sub in SUBTYPE_OVERRIDES:
        return SUBTYPE_OVERRIDES[sub]
    hne = row.get("hne_label")
    if isinstance(hne, str) and hne in HNE_LABEL_KEYS:
        return HNE_LABEL_KEYS[hne]
    return None


@dataclass
class RowScore:
    subtype: str
    tissue_type: str
    tissues_censored: tuple[str, ...]
    matched_cell_types: tuple[str, ...]
    n_target: int
    n_background: int
    rank_source: str            # "census" | "v3_curated"
    low_support: bool
    gene_scores: list[dict]     # per-gene scoring dicts (reranked)
    new_markers: list[str]


# ---------------------------------------------------------------------------
# 4) Scoring primitives
# ---------------------------------------------------------------------------

def _auc_per_gene(X_in: np.ndarray, X_out: np.ndarray) -> np.ndarray:
    """Vectorized ROC AUC per gene.

    AUC == (U / (n_in * n_out)) where U is the Mann-Whitney U statistic
    (one-sided, `greater` alternative). Using average-ranks, this reduces to:
        AUC_j = (mean(rank_in_j) - (n_in + 1) / 2) / n_out
    Computed once per gene column with scipy.stats.rankdata.
    """
    from scipy.stats import rankdata
    n_in = X_in.shape[0]
    n_out = X_out.shape[0]
    if n_in == 0 or n_out == 0:
        return np.full(X_in.shape[1], np.nan, dtype=np.float32)
    X = np.vstack([X_in, X_out])
    # rankdata along axis=0 per column, average ties.
    ranks = np.apply_along_axis(lambda c: rankdata(c, method="average"), 0, X)
    rank_in = ranks[:n_in, :]
    sum_in = rank_in.sum(axis=0)
    u = sum_in - n_in * (n_in + 1) / 2.0  # U for "in > out"
    auc = u / (n_in * n_out)
    # Genes constant in both populations -> AUC undefined; coerce to 0.5.
    same = (X.max(axis=0) == X.min(axis=0))
    auc = np.where(same, 0.5, auc)
    return auc.astype(np.float32)


def _score_matrix(
    X_in: np.ndarray,
    X_out: np.ndarray,
    genes: list[str],
) -> pd.DataFrame:
    """Compute pct_in, pct_out, log2fc, auc per gene."""
    pct_in = (X_in > 0).mean(axis=0)
    pct_out = (X_out > 0).mean(axis=0)
    # log2fc based on mean expression, with pseudocount.
    mean_in = X_in.mean(axis=0) + 1e-6
    mean_out = X_out.mean(axis=0) + 1e-6
    log2fc = np.log2(mean_in / mean_out)
    auc = _auc_per_gene(X_in, X_out)
    return pd.DataFrame({
        "gene": genes,
        "pct_in": np.asarray(pct_in).ravel(),
        "pct_out": np.asarray(pct_out).ravel(),
        "log2fc": np.asarray(log2fc).ravel(),
        "auc": auc,
    })


def _composite(df: pd.DataFrame) -> pd.Series:
    """Composite specificity score in [0, 1] (approx)."""
    auc_t = (df["auc"].clip(0.5, 1.0) - 0.5) * 2.0       # 0..1
    lfc_t = (df["log2fc"].clip(0, 3) / 3.0)               # 0..1
    pct_t = (df["pct_in"] - df["pct_out"]).clip(0, 1)     # 0..1
    return 0.40 * auc_t + 0.30 * lfc_t + 0.20 * pct_t + 0.10 * 0.0
    # lit_score left at 0.0 in this run (no external DB wired).


# ---------------------------------------------------------------------------
# 5) Census scoring per row
# ---------------------------------------------------------------------------

def _open_census(version: str | None = None, uri: str | None = None):
    import cellxgene_census
    if uri:
        log.info("opening local Census SOMA at %s...", uri)
        t = time.time()
        c = cellxgene_census.open_soma(uri=uri)
        log.info("census opened in %.1fs (local)", time.time() - t)
        return c
    log.info("opening CELLxGENE Census (version=%s)...", version or "latest")
    t = time.time()
    c = cellxgene_census.open_soma(census_version=version) if version \
        else cellxgene_census.open_soma()
    log.info("census opened in %.1fs (remote)", time.time() - t)
    return c


# Per-tissue cache of available cell_type labels (avoids re-running obs.read).
_TISSUE_CT_CACHE: dict[tuple[str, ...], list[str]] = {}


def _available_cell_types(census, tissues: list[str]) -> list[str]:
    key = tuple(sorted(tissues))
    if key in _TISSUE_CT_CACHE:
        return _TISSUE_CT_CACHE[key]
    t = time.time()
    obs = census["census_data"]["homo_sapiens"].obs
    tg_sql = ",".join(f"'{t_}'" for t_ in tissues)
    df = obs.read(
        column_names=["cell_type"],
        value_filter=f"tissue_general in [{tg_sql}]",
    ).concat().to_pandas()
    out = sorted(df["cell_type"].dropna().unique().tolist())
    _TISSUE_CT_CACHE[key] = out
    log.info("cached %d cell_types for tissues=%s (%.1fs)",
             len(out), tissues, time.time() - t)
    return out


def _query_anndata(
    census,
    tissue_general: list[str],
    cell_type: list[str] | None,
    gene_symbols: list[str],
    sample_cap: int,
    rng: np.random.Generator,
    label: str = "",
):
    """Query Census for an AnnData with given filters, capped by sampling."""
    import cellxgene_census
    tg_sql = ",".join(f"'{t}'" for t in tissue_general)
    obs_filter = f"tissue_general in [{tg_sql}]"
    if cell_type:
        ct_sql = ",".join(f"'{t}'" for t in cell_type)
        obs_filter += f" and cell_type in [{ct_sql}]"
    t = time.time()
    log.debug("  [%s] querying Census: %s (n_genes=%d)",
              label, obs_filter[:120], len(gene_symbols))
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter=obs_filter,
        var_value_filter=(
            "feature_name in [" +
            ",".join(f"'{g}'" for g in gene_symbols) + "]"
        ),
    )
    n0 = adata.n_obs
    if adata.n_obs > sample_cap:
        idx = rng.choice(adata.n_obs, size=sample_cap, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    log.debug("  [%s] fetched %d cells (sampled to %d) in %.1fs",
              label, n0, adata.n_obs, time.time() - t)
    return adata


def score_row(
    row: pd.Series,
    census,
    rng: np.random.Generator,
) -> RowScore:
    tissue_type = row["tissue_type"]
    tissues = TISSUE_MAP.get(tissue_type, [])
    markers = [g.strip() for g in row["markers"].split(",") if g.strip()]
    hint = resolve_hint(row)
    if hint is None or not tissues:
        return RowScore(
            subtype=row["subtype"], tissue_type=tissue_type,
            tissues_censored=tuple(tissues),
            matched_cell_types=(),
            n_target=0, n_background=0,
            rank_source="v3_curated",
            low_support=False,
            gene_scores=[], new_markers=markers[:CEILING_K],
        )

    # Discover available cell_type labels in these tissues, then filter by hint.
    available = _available_cell_types(census, tissues)
    matched = _match_cell_types(available, hint)
    log.debug("  matched %d/%d cell_types: %s",
              len(matched), len(available),
              matched[:6] + (["..."] if len(matched) > 6 else []))

    if not matched:
        return RowScore(
            subtype=row["subtype"], tissue_type=tissue_type,
            tissues_censored=tuple(tissues),
            matched_cell_types=(),
            n_target=0, n_background=0,
            rank_source="v3_curated",
            low_support=True,
            gene_scores=[], new_markers=markers[:CEILING_K],
        )

    try:
        ad_in = _query_anndata(
            census, tissues, matched, markers, TARGET_SAMPLE_CAP, rng,
            label="target",
        )
        ad_out = _query_anndata(
            census, tissues,
            [c for c in available if c not in set(matched)],
            markers, BG_SAMPLE_CAP, rng,
            label="bg",
        )
    except Exception as e:
        log.error("census query failed for %r: %s", row["subtype"], e)
        return RowScore(
            subtype=row["subtype"], tissue_type=tissue_type,
            tissues_censored=tuple(tissues),
            matched_cell_types=tuple(matched),
            n_target=0, n_background=0,
            rank_source="v3_curated",
            low_support=True,
            gene_scores=[], new_markers=markers[:CEILING_K],
        )

    n_in, n_out = ad_in.n_obs, ad_out.n_obs
    if n_in < MIN_TARGET_CELLS or n_out < MIN_BG_CELLS:
        return RowScore(
            subtype=row["subtype"], tissue_type=tissue_type,
            tissues_censored=tuple(tissues),
            matched_cell_types=tuple(matched),
            n_target=n_in, n_background=n_out,
            rank_source="v3_curated",
            low_support=True,
            gene_scores=[], new_markers=markers[:CEILING_K],
        )

    # Align gene ordering to `markers`, inserting zeros for genes not present.
    var_names = list(ad_in.var["feature_name"])
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    cols = [gene_to_idx.get(g, -1) for g in markers]
    X_in = ad_in.X.toarray() if hasattr(ad_in.X, "toarray") else np.asarray(ad_in.X)
    X_out = ad_out.X.toarray() if hasattr(ad_out.X, "toarray") else np.asarray(ad_out.X)

    # Pad missing genes with zero columns so _score_matrix still works.
    def _reorder(X):
        out = np.zeros((X.shape[0], len(markers)), dtype=np.float32)
        for k, c in enumerate(cols):
            if c >= 0:
                out[:, k] = X[:, c]
        return out

    X_in_r = _reorder(X_in)
    X_out_r = _reorder(X_out)
    scored = _score_matrix(X_in_r, X_out_r, markers)
    scored["present_in_census"] = [c >= 0 for c in cols]
    scored["composite"] = _composite(scored)
    # Genes not present in Census get nan -> composite 0; these go to tail.
    scored.loc[~scored["present_in_census"], "composite"] = 0.0

    scored = scored.sort_values("composite", ascending=False).reset_index(drop=True)

    # Adaptive filter: keep composite >= TAU, with floor/ceiling guards.
    kept = scored[scored["composite"] >= TAU]
    if len(kept) < FLOOR_K:
        kept = scored.head(FLOOR_K)
    if len(kept) > CEILING_K:
        kept = kept.head(CEILING_K)

    low_support = (scored["composite"] >= TAU).sum() < FLOOR_K

    return RowScore(
        subtype=row["subtype"], tissue_type=tissue_type,
        tissues_censored=tuple(tissues),
        matched_cell_types=tuple(matched),
        n_target=n_in, n_background=n_out,
        rank_source="census",
        low_support=bool(low_support),
        gene_scores=kept.to_dict(orient="records"),
        new_markers=kept["gene"].tolist(),
    )


# ---------------------------------------------------------------------------
# 5b) Tissue-cached batch scoring (1 Census query per tissue, reuse in memory)
# ---------------------------------------------------------------------------

def _score_tissue_group(
    tissue_type: str,
    rows_records: list[dict],
    census_uri: str | None,
    census_version: str | None,
    seed: int,
    verbose: bool,
    progress_queue=None,
) -> tuple[list[dict], list[dict]]:
    """Process all rows of one tissue_type with a single preloaded AnnData.

    Returns (rowscore_dicts, qc_dicts). Uses dicts (not dataclasses) so the
    result is trivially picklable across processes.

    If ``progress_queue`` is given, worker emits live events:
      ("preload_start", tissue, n_rows, n_genes)
      ("preload_done",  tissue, n_cells, elapsed)
      ("row_done",      tissue, subtype, n_kept, elapsed)
      ("tissue_done",   tissue, n_rows, elapsed)
      ("tissue_error",  tissue, str(exc))
    """
    _setup_logging(verbose)
    # If a progress queue is attached, suppress worker-side INFO logs (they
    # collide with the tqdm bar in the parent terminal). Main process still
    # reports preload/row/tissue events via the bar + summary lines.
    if progress_queue is not None:
        logging.getLogger().setLevel(logging.WARNING)
        log.setLevel(logging.WARNING)
    rng = np.random.default_rng(seed)
    tissues = TISSUE_MAP.get(tissue_type, [])
    _t_tissue = time.time()

    def _emit(evt):
        if progress_queue is not None:
            try:
                progress_queue.put(evt)
            except Exception:
                pass

    def _as_rowscore(
        subtype, matched, n_in, n_out, rank_source, low_support,
        gene_scores, new_markers,
    ) -> dict:
        return {
            "subtype": subtype,
            "tissue_type": tissue_type,
            "tissues_censored": tuple(tissues),
            "matched_cell_types": tuple(matched),
            "n_target": int(n_in),
            "n_background": int(n_out),
            "rank_source": rank_source,
            "low_support": bool(low_support),
            "gene_scores": gene_scores,
            "new_markers": new_markers,
        }

    # No tissue mapping -> all rows fall back to v3_curated.
    if not tissues:
        out = []
        for rec in rows_records:
            markers = [g.strip() for g in str(rec["markers"]).split(",") if g.strip()]
            out.append(_as_rowscore(
                rec["subtype"], (), 0, 0, "v3_curated", False, [], markers[:CEILING_K],
            ))
            _emit(("row_done", tissue_type, rec["subtype"], 0, 0.0))
        _emit(("tissue_done", tissue_type, len(rows_records),
               time.time() - _t_tissue))
        return out, []

    # Union of marker genes across this tissue's rows.
    gene_union: set[str] = set()
    for rec in rows_records:
        for g in str(rec["markers"]).split(","):
            g = g.strip()
            if g:
                gene_union.add(g)
    all_genes = sorted(gene_union)

    import cellxgene_census
    census = _open_census(census_version, census_uri)
    try:
        t_load = time.time()
        available = _available_cell_types(census, tissues)

        tg_sql = ",".join(f"'{t}'" for t in tissues)
        var_sql = ",".join(f"'{g}'" for g in all_genes)
        log.info("[%s] preloading AnnData (tissues=%s, %d genes)...",
                 tissue_type, tissues, len(all_genes))
        _emit(("preload_start", tissue_type, len(rows_records),
               len(all_genes)))
        adata = cellxgene_census.get_anndata(
            census, organism="Homo sapiens",
            obs_value_filter=f"tissue_general in [{tg_sql}]",
            var_value_filter=f"feature_name in [{var_sql}]",
        )
        _preload_elapsed = time.time() - t_load
        log.info("[%s] loaded %d cells x %d genes in %.1fs",
                 tissue_type, adata.n_obs, adata.n_vars, _preload_elapsed)
        _emit(("preload_done", tissue_type, adata.n_obs, _preload_elapsed))

        # Prepare shared structures
        X = adata.X
        if hasattr(X, "tocsr"):
            X = X.tocsr()
        obs_ct = adata.obs["cell_type"].astype(str).to_numpy()
        gene_names = adata.var["feature_name"].astype(str).to_numpy()
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        results: list[dict] = []
        qc_rows: list[dict] = []
        for rec in rows_records:
            t_row = time.time()
            subtype = rec["subtype"]
            markers = [g.strip() for g in str(rec["markers"]).split(",") if g.strip()]
            hint = resolve_hint(pd.Series(rec))

            if hint is None:
                results.append(_as_rowscore(
                    subtype, (), 0, 0, "v3_curated", False, [], markers[:CEILING_K],
                ))
                _emit(("row_done", tissue_type, subtype,
                       len(markers[:CEILING_K]), time.time() - t_row))
                continue

            matched = _match_cell_types(available, hint)
            if not matched:
                results.append(_as_rowscore(
                    subtype, (), 0, 0, "v3_curated", True, [], markers[:CEILING_K],
                ))
                _emit(("row_done", tissue_type, subtype,
                       len(markers[:CEILING_K]), time.time() - t_row))
                continue

            matched_set = set(matched)
            mask_in = np.isin(obs_ct, list(matched_set))
            mask_out = ~mask_in
            n_in_tot = int(mask_in.sum())
            n_out_tot = int(mask_out.sum())

            if n_in_tot < MIN_TARGET_CELLS or n_out_tot < MIN_BG_CELLS:
                results.append(_as_rowscore(
                    subtype, matched, n_in_tot, n_out_tot,
                    "v3_curated", True, [], markers[:CEILING_K],
                ))
                _emit(("row_done", tissue_type, subtype,
                       len(markers[:CEILING_K]), time.time() - t_row))
                continue

            idx_in_all = np.flatnonzero(mask_in)
            idx_out_all = np.flatnonzero(mask_out)
            if len(idx_in_all) > TARGET_SAMPLE_CAP:
                idx_in = rng.choice(idx_in_all, size=TARGET_SAMPLE_CAP, replace=False)
            else:
                idx_in = idx_in_all
            if len(idx_out_all) > BG_SAMPLE_CAP:
                idx_out = rng.choice(idx_out_all, size=BG_SAMPLE_CAP, replace=False)
            else:
                idx_out = idx_out_all
            idx_in.sort()
            idx_out.sort()

            cols = [gene_to_idx.get(g, -1) for g in markers]
            present_cols = [c for c in cols if c >= 0]
            if not present_cols:
                results.append(_as_rowscore(
                    subtype, matched, len(idx_in), len(idx_out),
                    "v3_curated", True, [], markers[:CEILING_K],
                ))
                _emit(("row_done", tissue_type, subtype,
                       len(markers[:CEILING_K]), time.time() - t_row))
                continue

            # Slice sparse matrix, then densify only the small sub-block.
            X_in_sub = np.asarray(X[idx_in][:, present_cols].todense()) \
                if hasattr(X, "todense") else np.asarray(X[idx_in][:, present_cols])
            X_out_sub = np.asarray(X[idx_out][:, present_cols].todense()) \
                if hasattr(X, "todense") else np.asarray(X[idx_out][:, present_cols])

            # Pad missing genes with zero columns.
            X_in_r = np.zeros((len(idx_in), len(markers)), dtype=np.float32)
            X_out_r = np.zeros((len(idx_out), len(markers)), dtype=np.float32)
            present_k = 0
            for k, c in enumerate(cols):
                if c >= 0:
                    X_in_r[:, k] = X_in_sub[:, present_k]
                    X_out_r[:, k] = X_out_sub[:, present_k]
                    present_k += 1

            scored = _score_matrix(X_in_r, X_out_r, markers)
            scored["present_in_census"] = [c >= 0 for c in cols]
            scored["composite"] = _composite(scored)
            scored.loc[~scored["present_in_census"], "composite"] = 0.0
            scored = scored.sort_values("composite", ascending=False).reset_index(drop=True)

            kept = scored[scored["composite"] >= TAU]
            if len(kept) < FLOOR_K:
                kept = scored.head(FLOOR_K)
            if len(kept) > CEILING_K:
                kept = kept.head(CEILING_K)

            low_support = (scored["composite"] >= TAU).sum() < FLOOR_K
            gene_scores = kept.to_dict(orient="records")
            new_markers = kept["gene"].tolist()

            results.append(_as_rowscore(
                subtype, matched, len(idx_in), len(idx_out),
                "census", low_support, gene_scores, new_markers,
            ))
            for gs in gene_scores:
                qc_rows.append({"tissue_type": tissue_type, "subtype": subtype, **gs})

            _row_elapsed = time.time() - t_row
            log.info(
                "[%s] %-55s source=%-7s n_in=%5d n_out=%5d n_kept=%3d "
                "low_support=%s (%.1fs)",
                tissue_type, subtype[:55], "census",
                len(idx_in), len(idx_out), len(new_markers),
                low_support, _row_elapsed,
            )
            _emit(("row_done", tissue_type, subtype,
                   len(new_markers), _row_elapsed))

        _emit(("tissue_done", tissue_type, len(results),
               time.time() - _t_tissue))
        return results, qc_rows
    finally:
        census.close()


# ---------------------------------------------------------------------------
# 6) Driver
# ---------------------------------------------------------------------------

def _write_checkpoint(
    path: str, df: pd.DataFrame, results: list, qc_rows: list[dict]
) -> None:
    """Append-style checkpoint so you can kill the job and resume by hand."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by_subtype = {r.subtype: r for r in results}
    rows = []
    for _, row in df.iterrows():
        r = by_subtype.get(row["subtype"])
        rows.append({
            "tissue_type": row["tissue_type"],
            "subtype": row["subtype"],
            "rank_source": r.rank_source if r else "pending",
            "low_support": r.low_support if r else None,
            "n_target": r.n_target if r else None,
            "n_background": r.n_background if r else None,
            "n_kept": len(r.new_markers) if r else None,
            "markers": ",".join(r.new_markers) if r else row["markers"],
        })
    pd.DataFrame(rows).to_csv(p, index=False)


def _run_rerank(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Core reranking pipeline. ``args`` is a namespace (argparse or SimpleNamespace)
    providing: input, output, qc_output, tissue, tissues, dry_run, seed,
    census_version, census_uri, parallel, verbose, checkpoint, log_file.

    Returns (df_out, qc_df). Writes to ``args.output`` / ``args.qc_output``
    only when those are truthy.
    """
    df = pd.read_csv(args.input)
    process_mask = pd.Series(True, index=df.index)
    if args.tissues:
        wanted = [t.strip() for t in args.tissues.split(",") if t.strip()]
        process_mask = df["tissue_type"].isin(wanted)
        log.info("processing subset: %s (%d/%d rows); other rows kept unchanged",
                 wanted, int(process_mask.sum()), len(df))
    elif args.tissue:
        process_mask = df["tissue_type"] == args.tissue
        log.info("processing subset: %s (%d/%d rows); other rows kept unchanged",
                 [args.tissue], int(process_mask.sum()), len(df))
    df = df.reset_index(drop=True)
    process_mask = process_mask.reset_index(drop=True)
    log.info("input: %s (%d rows, %d tissues)",
             args.input, len(df), df["tissue_type"].nunique())

    # Build tissue groups for the rows we intend to process.
    to_process = df[process_mask].copy()
    tissue_groups: dict[str, list[dict]] = {}
    for _, row in to_process.iterrows():
        tissue_groups.setdefault(row["tissue_type"], []).append(row.to_dict())
    log.info("tissue groups to process: %s",
             {k: len(v) for k, v in tissue_groups.items()})

    all_results: list[dict] = []
    all_qc: list[dict] = []

    # Rows NOT in process_mask -> emit "skipped" RowScores preserving markers.
    for _, row in df[~process_mask].iterrows():
        markers = [g.strip() for g in str(row["markers"]).split(",") if g.strip()]
        all_results.append({
            "subtype": row["subtype"],
            "tissue_type": row["tissue_type"],
            "tissues_censored": tuple(TISSUE_MAP.get(row["tissue_type"], ())),
            "matched_cell_types": (),
            "n_target": 0, "n_background": 0,
            "rank_source": "skipped", "low_support": False,
            "gene_scores": [], "new_markers": markers,
        })

    t0 = time.time()
    total_to_process = sum(len(v) for v in tissue_groups.values())

    # Live progress state shared with the drain thread.
    import threading
    from collections import OrderedDict
    tissue_state: "OrderedDict[str, str]" = OrderedDict(
        (tt, "queued") for tt in tissue_groups
    )
    state_lock = threading.Lock()
    pbar = None  # created after all setup logging is emitted

    def _active_postfix() -> str:
        with state_lock:
            active = [f"{tt}:{st}" for tt, st in tissue_state.items()
                      if st not in ("queued", "done")]
            shown = active[:4]
            more = len(active) - len(shown)
        s = " | ".join(shown) if shown else "(none)"
        if more > 0:
            s += f" (+{more})"
        return s

    def _drain_loop(q, stop_evt):
        while not stop_evt.is_set():
            try:
                evt = q.get(timeout=0.5)
            except Exception:
                evt = None

            if evt is None:
                pbar.set_postfix_str(_active_postfix(), refresh=False)
                continue

            kind = evt[0]
            if kind == "preload_start":
                _, tt, n_rows, n_genes = evt
                with state_lock:
                    tissue_state[tt] = f"preload({n_rows}r,{n_genes}g)"
            elif kind == "preload_done":
                _, tt, n_cells, _elapsed = evt
                with state_lock:
                    tissue_state[tt] = f"scoring(0/?, {n_cells}c)"
            elif kind == "row_done":
                _, tt, _sub, _nkept, _el = evt
                total_rows = len(tissue_groups.get(tt, []))
                with state_lock:
                    prev = tissue_state.get(tt, "")
                    done_count = 0
                    if prev.startswith("scoring("):
                        try:
                            done_count = int(prev.split("(")[1].split("/")[0])
                        except Exception:
                            done_count = 0
                    done_count += 1
                    tissue_state[tt] = f"scoring({done_count}/{total_rows})"
                pbar.update(1)
            elif kind == "tissue_done":
                _, tt, _nr, _el = evt
                with state_lock:
                    tissue_state[tt] = "done"
            elif kind == "tissue_error":
                _, tt, _msg = evt
                with state_lock:
                    tissue_state[tt] = "ERROR"

            pbar.set_postfix_str(_active_postfix(), refresh=False)

    if args.parallel and args.parallel > 1 and len(tissue_groups) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as _mp
        n_workers = min(args.parallel, len(tissue_groups))
        log.info("dispatching %d tissue(s) across %d workers...",
                 len(tissue_groups), n_workers)
        # Use spawn to avoid workers inheriting any main-process state
        # (e.g. open file handles, lingering tqdm instances).
        ctx = _mp.get_context("spawn")
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        stop_evt = threading.Event()
        # Create the tqdm bar now, after all setup log lines have been
        # printed, so nothing scrolls above/below the live bar.
        pbar = tqdm(
            total=total_to_process, file=_status_stream(),
            dynamic_ncols=True, unit="row", desc="ranking",
            mininterval=0.3, leave=True,
        )
        drain_thread = threading.Thread(
            target=_drain_loop, args=(progress_queue, stop_evt), daemon=True,
        )
        drain_thread.start()
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futures = {
                    ex.submit(
                        _score_tissue_group,
                        tt, recs, args.census_uri, args.census_version,
                        args.seed, args.verbose, progress_queue,
                    ): tt
                    for tt, recs in tissue_groups.items()
                }
                for fut in as_completed(futures):
                    tt = futures[fut]
                    try:
                        res, qc = fut.result()
                    except Exception as e:
                        log.error("[%s] worker failed: %s", tt, e)
                        # Fallback: emit v3_curated for this tissue's rows.
                        for rec in tissue_groups[tt]:
                            markers = [g.strip() for g in str(rec["markers"]).split(",") if g.strip()]
                            all_results.append({
                                "subtype": rec["subtype"],
                                "tissue_type": tt,
                                "tissues_censored": tuple(TISSUE_MAP.get(tt, ())),
                                "matched_cell_types": (),
                                "n_target": 0, "n_background": 0,
                                "rank_source": "v3_curated", "low_support": True,
                                "gene_scores": [], "new_markers": markers[:CEILING_K],
                            })
                        with state_lock:
                            tissue_state[tt] = "ERROR"
                        continue
                    all_results.extend(res)
                    all_qc.extend(qc)
                    if args.checkpoint:
                        _write_checkpoint_from_records(
                            args.checkpoint, df, all_results,
                        )
        finally:
            stop_evt.set()
            try:
                progress_queue.put(None)
            except Exception:
                pass
            drain_thread.join(timeout=2.0)
    else:
        # Sequential path uses a plain in-process queue + drain thread.
        import queue as _queue
        progress_queue = _queue.Queue()
        pbar = tqdm(
            total=total_to_process, file=_status_stream(),
            dynamic_ncols=True, unit="row", desc="ranking",
            mininterval=0.3, leave=True,
        )
        stop_evt = threading.Event()
        drain_thread = threading.Thread(
            target=_drain_loop, args=(progress_queue, stop_evt), daemon=True,
        )
        drain_thread.start()
        try:
            for tt, recs in tissue_groups.items():
                try:
                    res, qc = _score_tissue_group(
                        tt, recs, args.census_uri, args.census_version,
                        args.seed, args.verbose, progress_queue,
                    )
                except Exception as e:
                    log.error("[%s] failed: %s", tt, e)
                    for rec in recs:
                        markers = [g.strip() for g in str(rec["markers"]).split(",") if g.strip()]
                        all_results.append({
                            "subtype": rec["subtype"],
                            "tissue_type": tt,
                            "tissues_censored": tuple(TISSUE_MAP.get(tt, ())),
                            "matched_cell_types": (),
                            "n_target": 0, "n_background": 0,
                            "rank_source": "v3_curated", "low_support": True,
                            "gene_scores": [], "new_markers": markers[:CEILING_K],
                        })
                    with state_lock:
                        tissue_state[tt] = "ERROR"
                    continue
                all_results.extend(res)
                all_qc.extend(qc)
                if args.checkpoint:
                    _write_checkpoint_from_records(args.checkpoint, df, all_results)
        finally:
            stop_evt.set()
            try:
                progress_queue.put(None)
            except Exception:
                pass
            drain_thread.join(timeout=2.0)

    elapsed = time.time() - t0
    pbar.close()
    log.info("scored %d rows in %.1f min", len(all_results), elapsed / 60)

    # Merge back into df (preserving original row order).
    by_subtype = {r["subtype"]: r for r in all_results}
    df_out = df.copy()
    df_out["markers"] = df_out["subtype"].map(
        lambda s: ",".join(by_subtype[s]["new_markers"])
        if s in by_subtype else df_out.loc[df_out["subtype"] == s, "markers"].iloc[0]
    )
    df_out["rank_source"] = df_out["subtype"].map(
        lambda s: by_subtype[s]["rank_source"] if s in by_subtype else "v3_curated"
    )
    df_out["low_support"] = df_out["subtype"].map(
        lambda s: by_subtype[s]["low_support"] if s in by_subtype else False
    )

    qc_df = pd.DataFrame(all_qc)

    if args.dry_run:
        log.info("dry-run: not writing outputs.")
        log.info("preview:\n%s",
                 df_out[["tissue_type","subtype","rank_source","low_support"]].head().to_string())
        return df_out, qc_df

    if args.output:
        out_path = Path(args.output)
        df_out.to_csv(out_path, index=False)
        log.info("wrote %s (%d rows)", out_path, len(df_out))
    if args.qc_output and len(qc_df):
        qc_df.to_csv(args.qc_output, index=False)
        log.info("wrote %s (%d gene rows)", args.qc_output, len(qc_df))

    log.info("rank_source distribution:\n%s",
             df_out["rank_source"].value_counts().to_string())
    log.info("new marker-count distribution:\n%s",
             df_out["markers"].str.count(",").add(1).describe().round(1).to_string())
    low = df_out[df_out["low_support"]]
    log.info("low-support subtypes: %d", len(low))
    for _, r in low.iterrows():
        log.info("  - %-12s / %s", r["tissue_type"], r["subtype"])

    return df_out, qc_df


def rerank_markers(
    *,
    input_csv: str | Path = "markers-v3.csv",
    output_csv: str | Path | None = None,
    qc_output: str | Path | None = None,
    tissue: str | None = None,
    tissues: str | Iterable[str] | None = None,
    dry_run: bool = False,
    seed: int = 1234,
    census_version: str | None = None,
    census_uri: str | None = None,
    parallel: int = 4,
    verbose: bool = False,
    checkpoint: str | Path | None = None,
    log_file: str | Path | None = None,
    configure_logging: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerank marker lists in ``input_csv`` using the CELLxGENE Census.

    This is the Python API behind ``kurtorank rank-markers``. Unlike the
    CLI, outputs are **not** written to disk unless ``output_csv`` /
    ``qc_output`` are provided.

    Parameters
    ----------
    input_csv : path to markers CSV (schema matching ``markers-v3.csv``).
    output_csv : if set, write the reranked CSV here.
    qc_output : if set, write the per-gene QC CSV here.
    tissue : limit processing to this single ``tissue_type``.
    tissues : iterable (or comma-separated string) of tissue types to
        process. Takes precedence over ``tissue``. Other rows are kept
        unchanged with ``rank_source='skipped'``.
    dry_run : compute but skip writing outputs.
    seed : RNG seed for the Census scoring.
    census_version : Census release tag (``None`` = latest).
    census_uri : path to a local Census SOMA (recommended for speed).
    parallel : number of parallel tissue workers (``<=1`` = sequential).
    verbose : log per-query Census timing.
    checkpoint : path to incremental checkpoint CSV.
    log_file : append logs to this file.
    configure_logging : if ``True``, install the package's logging
        handlers. Set to ``False`` when calling from a host that already
        configures logging (e.g. a notebook).

    Returns
    -------
    (df_out, qc_df) : pandas DataFrames with the reranked panel and the
    per-gene QC table.
    """
    from types import SimpleNamespace
    if isinstance(tissues, str):
        tissues_s = tissues
    elif tissues is None:
        tissues_s = None
    else:
        tissues_s = ",".join(str(t) for t in tissues)
    args = SimpleNamespace(
        input=str(input_csv),
        output=str(output_csv) if output_csv else None,
        qc_output=str(qc_output) if qc_output else None,
        tissue=tissue,
        tissues=tissues_s,
        dry_run=bool(dry_run),
        seed=int(seed),
        census_version=census_version,
        census_uri=census_uri,
        parallel=int(parallel),
        verbose=bool(verbose),
        checkpoint=str(checkpoint) if checkpoint else None,
        log_file=str(log_file) if log_file else None,
    )
    if configure_logging:
        _setup_logging(args.verbose, args.log_file)
    return _run_rerank(args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="markers-v3.csv")
    ap.add_argument("--output", default="markers-v3.csv",
                    help="Where to write the reranked CSV (in-place by default).")
    ap.add_argument("--qc-output", default="markers-v3_qc.csv")
    ap.add_argument("--tissue", default=None,
                    help="Limit to one tissue_type (debug).")
    ap.add_argument("--tissues", default=None,
                    help="Comma-separated list of tissue_types to process "
                         "(takes precedence over --tissue). Rows from other "
                         "tissues are kept unchanged (rank_source='skipped').")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--census-version", default=None,
                    help="Census release tag; default = latest.")
    ap.add_argument("--census-uri", default=None,
                    help="Path to a local Census SOMA (e.g. "
                         "/workspace/census/2025-11-08/soma). Avoids "
                         "streaming over HTTPS; ~20-30x faster per row.")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Number of parallel tissue workers (0 or 1 = "
                         "sequential). Default 4.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Show per-query Census timing.")
    ap.add_argument("--checkpoint", default=None,
                    help="Write results so far to this path after every tissue.")
    ap.add_argument("--log-file", default=None,
                    help="Append logs to this file (in addition to stderr). "
                         "Use this instead of `| tee` so the live status "
                         "line renders correctly.")
    args = ap.parse_args()

    _setup_logging(args.verbose, args.log_file)
    _run_rerank(args)
    return 0


def _write_checkpoint_from_records(
    path: str, df: pd.DataFrame, results: list[dict]
) -> None:
    """Checkpoint using dict-records (new tissue-batched format)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    by_subtype = {r["subtype"]: r for r in results}
    rows = []
    for _, row in df.iterrows():
        r = by_subtype.get(row["subtype"])
        rows.append({
            "tissue_type": row["tissue_type"],
            "subtype": row["subtype"],
            "rank_source": r["rank_source"] if r else "pending",
            "low_support": r["low_support"] if r else None,
            "n_target": r["n_target"] if r else None,
            "n_background": r["n_background"] if r else None,
            "n_kept": len(r["new_markers"]) if r else None,
            "markers": ",".join(r["new_markers"]) if r else row["markers"],
        })
    pd.DataFrame(rows).to_csv(p, index=False)


# Alias used by the unified `kurtorank` CLI.
rank_markers_main = main


if __name__ == "__main__":
    sys.exit(main())
