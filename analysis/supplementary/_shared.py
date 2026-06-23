"""
Shared helpers for the H-Plot HNSCC supplementary (reviewer-response) analyses.

These functions are lifted *verbatim in maths* from the inline helper cells of
``data/results/head_neck/hplot_comprehensive_analysis_v2.ipynb`` (Parts A, B and
C) and refactored to take their inputs as arguments instead of reading notebook
globals.  Nothing here changes a threshold or an estimator relative to the main
analysis; the supplementary scripts import these so the numbers are directly
comparable to the notebook.

Packages that already provide a function (``hplot.core.HPlot``,
``hplot.stats._adjust_pvalues``, the ``sptxinsight`` graph/CCI builders) are
imported, not re-implemented.  The cluster-mass + permutation-FDR screens are
only defined inline in the notebook, so they live here.
"""
from __future__ import annotations

import os
import glob
import gzip

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, chi2

from hplot.core import HPlot
from hplot.stats import _adjust_pvalues  # noqa: F401  (re-exported for scripts)

# --------------------------------------------------------------------------- #
#  Paths (identical roots to the notebook)                                     #
# --------------------------------------------------------------------------- #
REPO_ROOT  = "/workspace/wsinsight/wsinsight-model-development"
DATA_ROOT  = os.path.join(REPO_ROOT, "data", "xenium", "head_neck")
WORK_DIR   = os.path.join(REPO_ROOT, "data", "results", "head_neck")
CACHE_DIR  = os.path.join(WORK_DIR, "cache")
SUPP_DIR   = os.path.join(WORK_DIR, "supplementary")
SERIES_MTX = os.path.join(DATA_ROOT, "GSE300147_series_matrix.txt.gz")
H5AD_NAME  = "annotated.h5ad"

TCGA_DIR   = os.path.join(REPO_ROOT, "data", "tcga", "hnsc")
CSV        = os.path.join(TCGA_DIR, "results", "objects", "hplot-outputs.csv")
LABELS_DIR = os.path.join(TCGA_DIR, "labels")

os.makedirs(SUPP_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Thresholds (a priori, matched to the main analysis)                         #
# --------------------------------------------------------------------------- #
PLOT_LAYERS          = (-7, 14)
GRID                 = np.arange(int(PLOT_LAYERS[0]), int(PLOT_LAYERS[1]) + 1)
MIN_PER_GROUP        = 3
MIN_CLUSTER_W        = 2

CLUSTER_ALPHA_XENIUM = 0.10
FDR_Q_XENIUM         = 0.10
B_XENIUM             = 1000

CLUSTER_ALPHA_TCGA   = 0.05
FDR_Q_TCGA           = 0.05
B_TCGA               = 2000

# Xenium per-sample recompute constants (Part A cell 7), needed to re-layer the
# graph at a different bin/edge width for the parameter-robustness analysis.
MPP          = 1.0
MAX_EDGE_UM  = 25.0     # Delaunay edge pruning (microns) -- the spatial bin knob
HPLOT_K      = 2
HPLOT_N      = 10
HPLOT_R      = 0.2
CCI_KERNEL   = "exponential"
CCI_LAM_UM   = 25.0
PAIR_CHUNK   = 128
BASE_COL     = "cancer_associated"

HL_COLORS = {"low": "#1f77b4", "high": "#d62728"}
HL_ORDER  = ["low", "high"]


class MissingInput(RuntimeError):
    """Raised when a required input file/column is absent (fail loudly)."""


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise MissingInput(f"required input not found: {path}")
    return path


# ========================================================================== #
#  XENIUM side (Part A)                                                       #
# ========================================================================== #
def parse_p16(path: str = SERIES_MTX):
    """p16 status + sample title per GSM, from the GEO series matrix."""
    _require(path)
    rows, gsm, title = {}, None, None
    with gzip.open(path, "rt") as f:
        for line in f:
            cells = [c.strip().strip('"') for c in line.rstrip("\n").split("\t")]
            if line.startswith("!Sample_geo_accession"):
                gsm = cells[1:]
            elif line.startswith("!Sample_title"):
                title = cells[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                rows.setdefault(cells[1].split(":")[0].strip(), cells[1:])
    p16 = {gsm[i]: rows["p16 status"][i].split(":")[-1].strip()
           for i in range(len(gsm))}
    ttl = {gsm[i]: title[i] for i in range(len(gsm))}
    return p16, ttl


def load_xenium_results():
    """Load the per-run cached H-plot results (joblib) and a label table.

    Returns (results, label_table) where ``results`` is keyed by sample id and
    each value is the per-run dict the notebook cached (``gex_mean``,
    ``out_mean``, ``layers``, ``hpv`` ...).  ``label_table`` adds the patient id
    (the run suffix ", run N" is stripped) so runs can be collapsed per patient.
    """
    import joblib
    caches = sorted(glob.glob(os.path.join(CACHE_DIR, "*.joblib")))
    if not caches:
        raise MissingInput(
            f"no cached Xenium run results in {CACHE_DIR}; run Part A of the "
            "notebook once to populate the cache.")
    P16, TITLE = parse_p16()

    def gsm_of(sid):
        return sid.split("_")[0]

    results, rows = {}, []
    for c in caches:
        sid = os.path.basename(c)[:-7]
        r = joblib.load(c)
        results[sid] = r
        g = gsm_of(sid)
        title = TITLE.get(g, "?")
        patient = title.split(", run")[0].strip()          # "Patient 9, run 2" -> "Patient 9"
        rows.append(dict(sample_id=sid, gsm=g, patient=patient,
                         p16=P16.get(g, "N/A"), hpv=r["hpv"]))
    label_table = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    return results, label_table


def reconstruct_panel():
    """Rebuild the common gene panel + LR-pair labels exactly as Part A cell 11.

    Returns (GENES, GPOS, PAIR_LABELS, PAIR_POS).  Column ``j`` of a run's
    ``gex_mean`` is ``GENES[j]``; column ``j`` of ``out_mean`` is
    ``PAIR_LABELS[j]``.
    """
    p = build_panel()
    return p["GENES"], p["GPOS"], p["PAIR_LABELS"], p["PAIR_POS"]


def build_panel():
    """Full Part A cell-11 panel: genes + LR pairs + CCI index arrays.

    Returns a dict with GENES, GPOS, GUP, PAIRS, PAIR_LABELS, PAIR_POS, USED,
    USED_COL, LIG_IDX, REC_IDX, nGenes, nP and the sample dirs.  Everything
    ``compute_run_tables`` needs to re-derive a run's gex/CCI tables.
    """
    import scanpy as sc
    from sptxinsight.insightlib.cci_generation import (
        load_lr_pairs, filter_pairs_to_panel)

    sample_dirs = sorted(
        d for d in glob.glob(os.path.join(DATA_ROOT, "*"))
        if os.path.isfile(os.path.join(d, H5AD_NAME)))
    if not sample_dirs:
        raise MissingInput(f"no annotated.h5ad runs under {DATA_ROOT}")
    common = None
    for d in sample_dirs:
        v = set(map(str, sc.read_h5ad(os.path.join(d, H5AD_NAME),
                                      backed="r").var_names))
        common = v if common is None else (common & v)
    GENES = sorted(common)
    GPOS = {g: i for i, g in enumerate(GENES)}
    GUP = {g.upper(): i for i, g in enumerate(GENES)}

    PAIRS = filter_pairs_to_panel(load_lr_pairs(), GENES)
    PAIR_LABELS = [f"{l}->{r}" for l, r in PAIRS]
    PAIR_POS = {p: i for i, p in enumerate(PAIR_LABELS)}
    USED = sorted({g.upper() for pr in PAIRS for g in pr} & set(GUP))
    USED_POS = {g: i for i, g in enumerate(USED)}
    USED_COL = np.array([GUP[g] for g in USED], dtype=np.int64)
    LIG_IDX = np.array([USED_POS[l.upper()] for l, r in PAIRS], dtype=np.int64)
    REC_IDX = np.array([USED_POS[r.upper()] for l, r in PAIRS], dtype=np.int64)
    return dict(GENES=GENES, GPOS=GPOS, GUP=GUP, PAIRS=PAIRS,
                PAIR_LABELS=PAIR_LABELS, PAIR_POS=PAIR_POS, USED=USED,
                USED_COL=USED_COL, LIG_IDX=LIG_IDX, REC_IDX=REC_IDX,
                nGenes=len(GENES), nP=len(PAIRS), sample_dirs=sample_dirs)


def compute_run_tables(adata, panel, max_edge_um=MAX_EDGE_UM):
    """Re-derive one run's per-layer gex/CCI tables (Part A cell 13 body).

    Parameterised by ``max_edge_um`` (the spatial bin / Delaunay-edge knob) so
    the parameter-robustness script can re-layer the graph at a different bin
    width.  Returns dict(layers, gex_mean, out_mean, in_mean).
    """
    import scipy.sparse as sp
    from sptxinsight.insightlib.insight_helpers import (
        delaunay_triangulation, k_hop_neighbors,
        identify_region_by_cell_function_enrichment,
        identify_border_cells, calculate_distance_to_border, compute_hplot)
    from sptxinsight.insightlib.cci_generation import build_weight_matrices

    GENES = panel["GENES"]; USED_COL = panel["USED_COL"]
    LIG_IDX = panel["LIG_IDX"]; REC_IDX = panel["REC_IDX"]; nP = panel["nP"]

    n = adata.n_obs
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    is_base = adata.obs[BASE_COL].astype(bool).to_numpy()
    nodes = pd.DataFrame({
        "center_x": coords[:, 0], "center_y": coords[:, 1],
        "is_base_type": is_base, "is_target_type": False, "target_value": 0.0})

    edges = delaunay_triangulation(coords, max_edge_um * MPP)
    neigh, A, Mk = k_hop_neighbors(n, edges, HPLOT_K)
    nodes = identify_region_by_cell_function_enrichment(
        neigh, nodes, HPLOT_N, HPLOT_R, Mk_sparse=Mk)
    nodes = identify_border_cells(nodes, {}, A_sparse=A)
    nodes = calculate_distance_to_border(nodes, {}, A_sparse=A)

    map_df = compute_hplot(nodes, edges).sort_values("layer")
    layers = map_df["layer"].to_numpy(dtype=float)
    nL = len(layers)

    sd = nodes["signed_distance_to_border"].to_numpy(dtype=float)
    layer_to_row = {float(l): i for i, l in enumerate(layers)}
    row = pd.Series(sd).map(layer_to_row)
    sel = np.where(row.notna().to_numpy())[0]
    rows = row.to_numpy()
    Lmat = sp.coo_matrix(
        (np.ones(sel.size, dtype=np.float32),
         (rows[sel].astype(np.int64), sel)), shape=(nL, n)).tocsr()
    cpl = np.asarray(Lmat.sum(axis=1)).ravel()
    inv_cpl = (1.0 / np.maximum(cpl, 1.0)).astype(np.float32)[:, None]

    var_idx = {g: i for i, g in enumerate(map(str, adata.var_names))}
    col = np.array([var_idx[g] for g in GENES], dtype=np.int64)
    X = adata.X.tocsr()
    E = np.nan_to_num(np.asarray(X[:, col].todense(), dtype=np.float32))
    gex_mean = (Lmat @ E) * inv_cpl

    E_used = E[:, USED_COL]
    _, W_mean = build_weight_matrices(edges, n, CCI_KERNEL, CCI_LAM_UM / MPP)
    NM = (W_mean @ E_used).astype(np.float32)
    out_mean = np.zeros((nL, nP), dtype=np.float32)
    in_mean = np.zeros((nL, nP), dtype=np.float32)
    for s0 in range(0, nP, PAIR_CHUNK):
        s1 = min(s0 + PAIR_CHUNK, nP)
        lb, rb = LIG_IDX[s0:s1], REC_IDX[s0:s1]
        out_mean[:, s0:s1] = (Lmat @ (E_used[:, lb] * NM[:, rb])) * inv_cpl
        in_mean[:, s0:s1] = (Lmat @ (E_used[:, rb] * NM[:, lb])) * inv_cpl
    return dict(layers=layers, gex_mean=gex_mean,
                out_mean=out_mean, in_mean=in_mean)


# ---- cluster-mass screen (Part A cell 17, de-globalised) ------------------ #
def _tie_corr(M, N):
    U = M.shape[1]
    C = np.ones(U)
    denom = N ** 3 - N
    if denom == 0:
        return C
    for j in range(U):
        _, cnt = np.unique(M[:, j], return_counts=True)
        t = cnt[cnt > 1]
        if t.size:
            C[j] = 1.0 - (t ** 3 - t).sum() / denom
    return np.clip(C, 1e-12, None)


def _kw_H(R, lab, k, N):
    H = np.zeros(R.shape[1])
    for g in range(k):
        m = lab == g
        ng = int(m.sum())
        if ng == 0:
            return None
        H += R[m].sum(0) ** 2 / ng
    return 12.0 / (N * (N + 1)) * H - 3.0 * (N + 1)


def _best_cluster_mass(Hmat, thr, min_w):
    L, U = Hmat.shape
    run_mass = np.zeros(U); run_len = np.zeros(U, dtype=int)
    best_mass = np.zeros(U); best_len = np.zeros(U, dtype=int)
    supra = Hmat > thr
    for l in range(L):
        s = supra[l]
        run_mass = np.where(s, run_mass + Hmat[l], 0.0)
        run_len = np.where(s, run_len + 1, 0)
        ok = (run_len >= min_w) & (run_mass > best_mass)
        best_mass = np.where(ok, run_mass, best_mass)
        best_len = np.where(ok, run_len, best_len)
    return best_mass, best_len


def _perm_fdr(obs, null):
    B, U = null.shape
    obs = np.asarray(obs, float)
    order = np.argsort(-obs)
    obs_sorted = obs[order]
    flat = np.sort(null.ravel())
    ge_null = flat.size - np.searchsorted(flat, obs_sorted, side="left")
    EV = ge_null / B
    R = np.arange(1, U + 1)
    fdr_sorted = np.minimum(EV / R, 1.0)
    fdr_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]
    q = np.empty(U); q[order] = fdr_sorted
    q[obs <= 0] = 1.0
    return q


def screen_cluster(results, grid, key, n_units, arm_lists,
                   B=B_XENIUM, seed=0, alpha=CLUSTER_ALPHA_XENIUM,
                   min_w=MIN_CLUSTER_W, min_per_group=MIN_PER_GROUP):
    """Two/k-group cluster-mass spatial screen (Part A cell 17), de-globalised.

    ``results`` : dict ``unit_id -> {'layers':..., key:(nL x n_units)...}``.
    ``arm_lists``: list of ``(arm_name, [unit_ids])``.
    Returns the same dict the notebook returns (mass, band_lo/hi, top_arm,
    perm_p, fdr_global, ...).
    """
    def _layer_vec(sid, ell):
        r = results[sid]
        idx = np.where(r["layers"].astype(int) == ell)[0]
        return None if idx.size == 0 else r[key][idx[0]]

    k = len(arm_lists)
    arm_names = [a for a, _ in arm_lists]
    arm_sids = [s for _, s in arm_lists]
    runs = [s for sids in arm_sids for s in sids]
    g0 = np.concatenate([[gi] * len(arm_sids[gi]) for gi in range(k)])
    run_pos = {s: i for i, s in enumerate(runs)}
    thr = float(chi2.ppf(1 - alpha, df=k - 1))

    tested_layers, layer_cache, H_obs_rows, gm_rows = [], [], [], []
    for ell in grid:
        per = [[_layer_vec(s, ell) for s in sids] for sids in arm_sids]
        present = [[(s, v) for s, v in zip(sids, lst) if v is not None]
                   for sids, lst in zip(arm_sids, per)]
        if any(len(p) < min_per_group for p in present):
            continue
        rows, sid_order = [], []
        for gi, plist in enumerate(present):
            for s, v in plist:
                rows.append(v); sid_order.append(s)
        M = np.vstack(rows); N = M.shape[0]
        R = rankdata(M, axis=0); C = _tie_corr(M, N)
        pos = np.array([run_pos[s] for s in sid_order])
        H = _kw_H(R, g0[pos], k, N) / C
        tested_layers.append(ell); layer_cache.append((R, C, N, pos))
        H_obs_rows.append(H)
        gm_rows.append(np.vstack([np.vstack([v for _, v in present[gi]]).mean(0)
                                  for gi in range(k)]))

    if not tested_layers:
        raise MissingInput("no layer met the min-per-group requirement; "
                           "cannot screen (check group sizes).")
    tested_layers = np.asarray(tested_layers)
    H_obs = np.vstack(H_obs_rows)
    L = H_obs.shape[0]
    obs_mass, obs_len = _best_cluster_mass(H_obs, thr, min_w)

    supra = H_obs > thr
    peak_layer = np.full(n_units, np.nan)
    band_lo = np.full(n_units, np.nan); band_hi = np.full(n_units, np.nan)
    top_arm = np.empty(n_units, dtype=object); top_arm[:] = ""
    peak_mean = np.zeros((n_units, k))
    for u in range(n_units):
        if obs_mass[u] <= 0:
            continue
        best = rm = 0.0; rl = 0; start = 0; bstart = bend = -1
        for l in range(L):
            if supra[l, u]:
                if rl == 0:
                    start = l
                rm += H_obs[l, u]; rl += 1
                if rl >= min_w and rm > best:
                    best = rm; bstart = start; bend = l
            else:
                rm = 0.0; rl = 0
        if bend < 0:
            continue
        band_lo[u] = tested_layers[bstart]; band_hi[u] = tested_layers[bend]
        pk = bstart + int(np.argmax(H_obs[bstart:bend + 1, u]))
        peak_layer[u] = tested_layers[pk]
        gmk = gm_rows[pk][:, u]; peak_mean[u] = gmk
        top_arm[u] = arm_names[int(np.argmax(gmk))]

    rng = np.random.default_rng(seed)
    null = np.empty((B, n_units), dtype=np.float32)
    for b in range(B):
        gp = rng.permutation(g0)
        Hn = np.empty((L, n_units))
        for li, (R, C, N, pos) in enumerate(layer_cache):
            Hn[li] = _kw_H(R, gp[pos], k, N) / C
        null[b], _ = _best_cluster_mass(Hn, thr, min_w)

    perm_p = np.maximum((null >= obs_mass[None, :]).mean(0), 1.0 / B)
    fdr_global = _perm_fdr(obs_mass, null)
    return dict(mass=obs_mass, width=obs_len, peak_layer=peak_layer,
                band_lo=band_lo, band_hi=band_hi, top_arm=top_arm,
                peak_mean=peak_mean, perm_p=perm_p, fdr_global=fdr_global,
                arm_names=arm_names, thr=thr, tested_layers=tested_layers)


def collapse_runs_to_patients(results, label_table, mode="mean"):
    """Build a patient-keyed results dict from the run-level cache.

    ``mode='mean'`` averages each per-layer vector across a patient's runs
    (aligned by integer layer); ``mode='first'`` keeps only the first run.
    Returns (patient_results, patient_arm_lists) where arm_lists mirrors the
    HPV+/Rest grouping used by ``screen_cluster``.
    """
    lt = label_table.copy()
    patient_results = {}
    for pat, sub in lt.groupby("patient"):
        sids = sub["sample_id"].tolist()
        hpv = sub["hpv"].iloc[0]
        if mode == "first" or len(sids) == 1:
            r0 = results[sids[0]]
            patient_results[pat] = dict(
                layers=r0["layers"], gex_mean=r0["gex_mean"],
                out_mean=r0["out_mean"], in_mean=r0["in_mean"], hpv=hpv)
            continue
        # mode == "mean": align by integer layer, average present runs
        runs = [results[s] for s in sids]
        keys = ["gex_mean", "out_mean", "in_mean"]
        all_layers = sorted({int(l) for r in runs for l in r["layers"]})
        stacks = {kk: [] for kk in keys}
        for ell in all_layers:
            for kk in keys:
                vecs = []
                for r in runs:
                    idx = np.where(r["layers"].astype(int) == ell)[0]
                    if idx.size:
                        vecs.append(r[kk][idx[0]])
                stacks[kk].append(np.mean(vecs, axis=0))
        patient_results[pat] = dict(
            layers=np.asarray(all_layers, dtype=float),
            gex_mean=np.vstack(stacks["gex_mean"]),
            out_mean=np.vstack(stacks["out_mean"]),
            in_mean=np.vstack(stacks["in_mean"]), hpv=hpv)

    arm_lists = []
    for arm in ("HPV+", "Rest"):
        pats = [p for p in patient_results if patient_results[p]["hpv"] == arm]
        arm_lists.append((arm, pats))
    return patient_results, arm_lists


# ========================================================================== #
#  TCGA side (Part B)                                                         #
# ========================================================================== #
def _norm(s):
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# scalar attribute matchers (Part B cell 52)
ATTR_MATCH = {
    "mutation_count":          lambda n: n == "mutationcount",
    "tmb":                     lambda n: "tmbnonsynonymous" in n or n == "tmb",
    "fga":                     lambda n: n == "fractiongenomealtered",
    "aneuploidy":              lambda n: "aneuploidy" in n,
    "ploidy":                  lambda n: n == "ploidy",
    "msi_mantis":              lambda n: "mantis" in n,
    "msisensor":               lambda n: "msisensor" in n or ("msi" in n and "sensor" in n),
    "purity":                  lambda n: "purity" in n,
    "leukocyte_fraction":      lambda n: "leukocytefraction" in n,
    "til_fraction":            lambda n: "tilregionalfraction" in n or ("til" in n and "fraction" in n),
    "lymphocyte_infiltration": lambda n: "lymphocyteinfiltration" in n,
    "ifn_gamma":               lambda n: "ifngamma" in n,
    "tgf_beta":                lambda n: "tgfbeta" in n,
    "wound_healing":           lambda n: "woundhealing" in n,
    "macrophage_regulation":   lambda n: "macrophageregulation" in n,
    "proliferation":           lambda n: n == "proliferation",
    "snv_neoantigens":         lambda n: "snvneoantigen" in n,
    "indel_neoantigens":       lambda n: "indelneoantigen" in n,
    "tcr_shannon":             lambda n: "tcrshannon" in n,
    "tcr_richness":            lambda n: "tcrrichness" in n,
    "bcr_shannon":             lambda n: "bcrshannon" in n,
    "stromal_score":           lambda n: "stromalfraction" in n or "stromalscore" in n,
    "th1_cells":               lambda n: n == "th1cells",
    "th2_cells":               lambda n: n == "th2cells",
    "th17_cells":              lambda n: n == "th17cells",
    "csf1_response":           lambda n: "csf1" in n,
    "stat1_signaling":         lambda n: n == "stat1score" or "stat1signaling" in n,
    "cd8_t_cells":             lambda n: n == "cd8tcells",
    "cytotoxic_cells":         lambda n: "cytotoxic" in n,
    "nk_cells":                lambda n: n == "nkcells",
    "b_cells":                 lambda n: n == "bcells",
    "treg_cells":              lambda n: n == "tregcells",
    "macrophages":             lambda n: n == "macrophages",
    "neutrophils":             lambda n: n == "neutrophils",
    "mast_cells":              lambda n: n == "mastcells",
    "macrophage_m1":           lambda n: "macrophagem1" in n,
    "macrophage_m2":           lambda n: "macrophagem2" in n,
    "icr_score":               lambda n: n == "icrscore",
    "angiogenesis":            lambda n: n == "angiogenesis",
    "antigen_presentation":    lambda n: n == "apm1",
}

CHECKPOINT_GENES = {
    "PDCD1": "PD-1 (PDCD1) expr", "CD274": "PD-L1 (CD274) expr",
    "PDCD1LG2": "PD-L2 (PDCD1LG2) expr", "CTLA4": "CTLA-4 expr",
    "LAG3": "LAG-3 expr", "HAVCR2": "TIM-3 (HAVCR2) expr", "TIGIT": "TIGIT expr",
    "IDO1": "IDO1 expr", "CD8A": "CD8A expr", "GZMB": "GZMB expr",
    "CXCL9": "CXCL9 expr", "FOXP3": "FOXP3 expr", "CXCL10": "CXCL10 expr",
    "EGFR": "EGFR expr", "VCAN": "VCAN expr", "TNC": "TNC expr",
}
GEP_GENES = ["CCL5", "CD27", "CD274", "CD276", "CD8A", "CMKLR1", "CXCL9",
             "CXCR6", "HLA-DQA1", "HLA-DRB1", "HLA-E", "IDO1", "LAG3", "NKG7",
             "PDCD1LG2", "PSMB10", "STAT1", "TIGIT"]

# display names for the continuous panel (Part B cell 56)
CONT_LABELS = [
    ("mutation_count", "Mutation Count"), ("tmb", "TMB (nonsynonymous)"),
    ("fga", "Fraction Genome Altered"), ("aneuploidy", "Aneuploidy Score"),
    ("msi_mantis", "MSI MANTIS Score"), ("msisensor", "MSIsensor Score"),
    ("purity", "Tumour purity"), ("ploidy", "Ploidy"),
    ("leukocyte_fraction", "Leukocyte Fraction"),
    ("til_fraction", "TIL Regional Fraction"),
    ("lymphocyte_infiltration", "Lymphocyte Infiltration"),
    ("ifn_gamma", "IFN-gamma Response"), ("tgf_beta", "TGF-beta Response"),
    ("wound_healing", "Wound Healing"),
    ("macrophage_regulation", "Macrophage Regulation"),
    ("proliferation", "Proliferation"), ("snv_neoantigens", "SNV Neoantigens"),
    ("indel_neoantigens", "Indel Neoantigens"), ("tcr_shannon", "TCR Shannon"),
    ("tcr_richness", "TCR Richness"), ("bcr_shannon", "BCR Shannon"),
    ("stromal_score", "Stromal Fraction"), ("th1_cells", "Th1 cells"),
    ("th2_cells", "Th2 cells"), ("th17_cells", "Th17 cells"),
    ("cd8_t_cells", "CD8 T cells (Bindea)"),
    ("cytotoxic_cells", "Cytotoxic cells (Bindea)"),
    ("nk_cells", "NK cells (Bindea)"), ("b_cells", "B cells (Bindea)"),
    ("treg_cells", "Tregs (Bindea)"), ("macrophages", "Macrophages (Bindea)"),
    ("neutrophils", "Neutrophils (Bindea)"), ("mast_cells", "Mast cells (Bindea)"),
    ("macrophage_m1", "Macrophage M1 (CIBERSORT)"),
    ("macrophage_m2", "Macrophage M2 (CIBERSORT)"),
    ("csf1_response", "CSF1 response (TAM)"),
    ("stat1_signaling", "STAT1 signalling"), ("icr_score", "ICR score"),
    ("angiogenesis", "Angiogenesis"),
    ("antigen_presentation", "Antigen-presentation (APM1)"),
    ("gep_inflamed", "T-cell-inflamed GEP"),
]
CONT_LABELS += [("gex_" + g, disp) for g, disp in CHECKPOINT_GENES.items()]

# "cold" markers (immune-low where the marker is high): genomic instability /
# stromal programmes that align with the Xenium HPV-negative stromal-wall story.
COLD_MARKERS = ["tgf_beta", "wound_healing", "fga", "aneuploidy"]


def load_tcga_curves():
    """Per-patient immune-infiltration curve matrix (Part B cell 48)."""
    _require(CSV)
    raw = pd.read_csv(CSV)
    raw["patient"] = raw["id"].str[:12]
    raw["sample_code"] = raw["id"].str[13:15]
    raw = raw[raw["sample_code"] == "01"].copy()
    core = raw[(raw["layer"] >= PLOT_LAYERS[0]) & (raw["layer"] <= PLOT_LAYERS[1])]
    cov = core.groupby("id")["layer"].nunique()
    good_ids = cov[cov >= 15].index
    raw = raw[raw["id"].isin(good_ids) & (raw["all_count"] >= 3)].copy()
    long = (raw.groupby(["patient", "layer"])
               .agg(target_prop=("target_prop", "mean"),
                    distance=("distance", "mean")).reset_index())
    long = long[long["layer"].between(GRID[0], GRID[-1])].copy()
    mat = (long.pivot(index="patient", columns="layer", values="target_prop")
              .reindex(columns=GRID))
    return mat, long


def load_tcga_labels(mat):
    """Per-patient biomarker label frame (Part B cell 52), reindexed to `mat`."""
    if not os.path.isdir(LABELS_DIR):
        raise MissingInput(f"labels dir not found: {LABELS_DIR}")
    labels = {}

    def _set(pat, key, val):
        labels.setdefault(pat, {})[key] = val

    def _patient_col(df):
        prefs = ["patientbarcode", "tcgaparticipantbarcode", "participantbarcode",
                 "patientid", "participant", "sampleid", "samplebarcode",
                 "barcode", "sample", "array"]
        norm = {c: _norm(c) for c in df.columns}
        for want in prefs:
            for c, n in norm.items():
                if n == want:
                    return c
        for c, n in norm.items():
            if "barcode" in n or "participant" in n:
                return c
        return None

    def _iter_tables(f):
        fl = f.lower()
        try:
            if fl.endswith((".xlsx", ".xls")):
                for _, d in pd.read_excel(f, sheet_name=None, dtype=str).items():
                    yield d
            elif fl.endswith((".txt", ".tsv")):
                yield pd.read_csv(f, sep="\t", comment="#", dtype=str)
            else:
                yield pd.read_csv(f, dtype=str)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), "->", e)

    def _is_expression(f):
        n = _norm(os.path.basename(f))
        return any(k in n for k in ("rsem", "mrna", "expression", "rnaseq"))

    files = sorted(f for f in glob.glob(os.path.join(LABELS_DIR, "*"))
                   if os.path.isfile(f))

    for f in files:
        if _is_expression(f):
            continue
        for df in _iter_tables(f):
            df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
            pcol = _patient_col(df)
            if pcol is None:
                continue
            pat = df[pcol].astype(str).str[:12]
            for key, test in ATTR_MATCH.items():
                col = next((c for c in df.columns if test(_norm(c))), None)
                if col is None:
                    continue
                vals = pd.to_numeric(df[col], errors="coerce")
                for p, v in zip(pat, vals):
                    if pd.notna(v):
                        _set(p, key, float(v))
            sc_ = next((c for c in df.columns
                        if "immunesubtype" in _norm(c) or _norm(c) == "subtype"), None)
            if sc_ is not None:
                sv = df[sc_].astype(str).str.extract(r"(C[1-6])")[0]
                for p, v in zip(pat, sv):
                    if isinstance(v, str):
                        _set(p, "immune_subtype", v)
            hc = next((c for c in df.columns if "hpv" in _norm(c)), None)
            if hc is not None:
                for p, raw in zip(pat, df[hc].astype(str)):
                    n = _norm(raw)
                    if n in ("positive", "pos", "1", "yes", "hpvpositive") or "hpvpos" in n:
                        _set(p, "hpv_status", "HPV+")
                    elif n in ("negative", "neg", "0", "no", "hpvnegative") or "hpvneg" in n:
                        _set(p, "hpv_status", "HPV-")

    def _load_expression(f):
        try:
            df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), "->", e); return
        hugo = next((c for c in df.columns
                     if _norm(c) in ("hugosymbol", "genesymbol", "hugo", "gene")),
                    df.columns[0])
        samp_cols = [c for c in df.columns if str(c).upper().startswith("TCGA")]
        if not samp_cols:
            return
        pats = [str(c)[:12] for c in samp_cols]
        genes_wanted = {g.upper() for g in (set(CHECKPOINT_GENES) | set(GEP_GENES))}
        sub = df[df[hugo].astype(str).str.upper().isin(genes_wanted)]
        if sub.empty:
            return
        X = sub[samp_cols].apply(pd.to_numeric, errors="coerce")
        X.index = sub[hugo].astype(str).str.upper().values
        if np.nanmax(X.to_numpy()) > 50:
            X = np.log2(X.clip(lower=0) + 1.0)
        for g in CHECKPOINT_GENES:
            if g.upper() in X.index:
                row = X.loc[g.upper()]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                for p, v in zip(pats, np.asarray(row, float)):
                    if np.isfinite(v):
                        _set(p, "gex_" + g, float(v))
        avail = [g for g in GEP_GENES if g.upper() in X.index]
        if avail:
            Z = X.loc[[g.upper() for g in avail]]
            if isinstance(Z, pd.Series):
                Z = Z.to_frame().T
            Z = Z.sub(Z.mean(axis=1), axis=0).div(
                Z.std(axis=1).replace(0, np.nan), axis=0)
            gep = Z.mean(axis=0)
            for p, v in zip(pats, np.asarray(gep, float)):
                if np.isfinite(v):
                    _set(p, "gep_inflamed", float(v))

    for f in files:
        if _is_expression(f):
            _load_expression(f)

    return pd.DataFrame.from_dict(labels, orient="index").reindex(mat.index)


# ---- TCGA cluster-mass screen (Part B cell 54, de-globalised) ------------- #
def _best_band(H, thr, min_w):
    L = len(H); supra = np.where(np.isnan(H), False, H > thr)
    best = 0.0; bs = be = -1; rm = 0.0; rl = 0; start = 0
    for l in range(L):
        if supra[l]:
            if rl == 0:
                start = l
            rm += H[l]; rl += 1
            if rl >= min_w and rm > best:
                best = rm; bs = start; be = l
        else:
            rm = 0.0; rl = 0
    if be < 0:
        return 0.0, (-1, -1, -1)
    peak = bs + int(np.nanargmax(H[bs:be + 1]))
    return best, (bs, be, peak)


def binarize(series, min_per_group=MIN_PER_GROUP):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    g = np.full(x.size, -1); ok = ~np.isnan(x)
    if ok.sum() < 2 * min_per_group:
        return g
    g[ok] = (x[ok] > np.nanmedian(x)).astype(int)
    return g


def screen_label(mat, group_of, k, grid=GRID, B=B_TCGA, seed=0,
                 alpha=CLUSTER_ALPHA_TCGA, min_w=MIN_CLUSTER_W,
                 min_per_group=MIN_PER_GROUP):
    """Single-biomarker cluster-mass screen on the immune curve (Part B cell 54)."""
    M = mat.values
    labeled = group_of >= 0
    idx_lab = np.where(labeled)[0]
    pos = -np.ones(M.shape[0], int); pos[idx_lab] = np.arange(idx_lab.size)
    cache = []
    for li in range(len(grid)):
        col = M[:, li]; present = labeled & ~np.isnan(col); gi = np.where(present)[0]
        if gi.size == 0:
            cache.append(None); continue
        vals = col[gi]; R = rankdata(vals); N = vals.size
        _, cnt = np.unique(vals, return_counts=True); t = cnt[cnt > 1]
        C = 1.0 - (t ** 3 - t).sum() / (N ** 3 - N) if N > 1 else 1.0
        cache.append((R, max(C, 1e-12), N, pos[gi], vals))
    g_lab = group_of[idx_lab]
    thr = float(chi2.ppf(1 - alpha, df=k - 1))

    def _H_from(R, C, N, grp):
        s = 0.0
        for g in range(k):
            m = grp == g; ng = int(m.sum())
            if ng < min_per_group:
                return np.nan
            s += R[m].sum() ** 2 / ng
        return (12.0 / (N * (N + 1)) * s - 3.0 * (N + 1)) / C

    H_obs = np.full(len(grid), np.nan)
    grp_means = np.full((len(grid), k), np.nan)
    for li, c in enumerate(cache):
        if c is None:
            continue
        R, C, N, p, vals = c; grp = g_lab[p]
        H_obs[li] = _H_from(R, C, N, grp)
        for g in range(k):
            m = grp == g
            if m.sum():
                grp_means[li, g] = vals[m].mean()
    mass, (bs, be, pk) = _best_band(H_obs, thr, min_w)
    rng = np.random.default_rng(seed); null = np.empty(B)
    for b in range(B):
        gp = rng.permutation(g_lab); Hn = np.full(len(grid), np.nan)
        for li, c in enumerate(cache):
            if c is None:
                continue
            R, C, N, p, _v = c; Hn[li] = _H_from(R, C, N, gp[p])
        null[b], _ = _best_band(Hn, thr, min_w)
    perm_p = max(float((null >= mass).mean()), 1.0 / B) if mass > 0 else 1.0
    return dict(thr=thr, H_obs=H_obs, grp_means=grp_means, mass=mass,
                band=(bs, be, pk), perm_p=perm_p,
                group_sizes=[int((g_lab == g).sum()) for g in range(k)])


# ---- small H-plot drawing helper (uses the hplot package) ----------------- #
def kw_layer_pvalues(df, groups, group_col, min_n=MIN_PER_GROUP):
    rows = []
    for layer, gdf in df.groupby("layer"):
        arrs = [gdf.loc[gdf[group_col] == g, "target_prop"].dropna().to_numpy()
                for g in groups]
        dist = gdf["distance"].mean()
        if all(a.size >= min_n for a in arrs) and len(arrs) >= 2:
            try:
                stat, p = kruskal(*arrs)
            except ValueError:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        rows.append(dict(layer=layer, distance=dist, p_value=p, stat=stat))
    out = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    out["p_adj"] = _adjust_pvalues(out["p_value"].to_numpy(), "fdr_bh")
    return out


def plot_curve(long, patient_to_group, gnames, name, ax, color_map, band=None):
    """Draw one immune-infiltration H-plot grouped by `patient_to_group`."""
    grp = pd.Series(patient_to_group).reindex(long["patient"].values).to_numpy()
    df = long.assign(grp=grp).dropna(subset=["grp"])
    df = df[df["layer"].between(GRID[0], GRID[-1])]
    hp = HPlot()
    hp.fit(df, targets="target_prop", layer="layer", group="grp",
           distance="distance", unit="ring", color_map=color_map,
           legend_order=gnames, legend_title=name)
    hp.pvalue_test = "kruskal"
    hp.layer_pvalues_ = kw_layer_pvalues(df, gnames, "grp")
    ax = hp.plot(ax=ax, ci_show=False, display_base_type="tumour",
                 value_kind="proportion", display_target_type="immune cells",
                 pvalue_show=True, pvalue_label=f"KW p ({name})")
    if band is not None and np.isfinite(band[0]) and np.isfinite(band[1]):
        ax.axvspan(band[0], band[1], color="0.6", alpha=0.12, zorder=0)
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_title(name)
    return ax


# ---- TCGA survival frame (Part C border-infiltration Cox covariates) ------ #
def build_survival_frame(band_lo=-5, band_hi=0):
    """Reproduce the Part C survival frame exactly (script 03 / notebook Part C).

    Returns one row per TCGA-HNSC patient with:
      interior_infiltration : mean immune fraction over layers L in [band_lo, band_hi]
      infiltration_z        : cohort-standardised interior_infiltration
      HPV_pos, AGE, late_stage, OS_t (months), OS_e (event)
    Raises ``MissingInput`` if any required Part C file is absent.
    """
    cur_f = _require(f"{TCGA_DIR}/results/objects/hplot-outputs.csv")
    cur = pd.read_csv(cur_f)
    cur = cur[(cur.layer >= -10) & (cur.layer <= 20)].copy()
    cur["patient"] = cur["id"].str[:12]
    mat = cur.groupby(["patient", "layer"])["target_prop"].mean().unstack("layer")
    band = [L for L in mat.columns if band_lo <= L <= band_hi]
    if not band:
        raise MissingInput(
            f"no layers in L in [{band_lo},{band_hi}] for the infiltration covariate")
    geom = (mat[band].mean(axis=1).rename("interior_infiltration")
            .dropna().reset_index())

    sv_f = _require(f"{TCGA_DIR}/labels/data_clinical_patient_survival.csv")
    sv = pd.read_csv(sv_f).rename(columns={"patientId": "patient"})
    for c in ("OS_STATUS", "OS_MONTHS"):
        if c not in sv.columns:
            raise MissingInput(f"survival file missing column '{c}'")
    sv["OS_e"] = sv.OS_STATUS.astype(str).str.startswith("1").astype(float)
    sv["OS_t"] = pd.to_numeric(sv.OS_MONTHS, errors="coerce")
    d = geom.merge(sv[["patient", "OS_e", "OS_t"]], on="patient").query("OS_t > 0").dropna()

    cov_f = _require(f"{TCGA_DIR}/labels/data_clinical_patient_covariates.csv")
    cov = pd.read_csv(cov_f).rename(columns={"patientId": "patient"})
    cov["AGE"] = pd.to_numeric(cov.AGE, errors="coerce")
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" not in cov.columns:
        raise MissingInput("covariate file missing AJCC_PATHOLOGIC_TUMOR_STAGE")
    cov["late_stage"] = cov.AJCC_PATHOLOGIC_TUMOR_STAGE.astype(str).str.contains(
        "III|IV", regex=True).astype(float)

    samp_f = _require(f"{TCGA_DIR}/labels/data_clinical_sample.txt")
    samp = pd.read_csv(samp_f, sep="\t", comment="#")
    if "HPV_STATUS" not in samp.columns:
        raise MissingInput("sample file missing HPV_STATUS column")
    samp["HPV_pos"] = samp.HPV_STATUS.astype(str).str.lower().str.startswith(
        "pos").astype(float)
    hpv = samp.groupby("PATIENT_ID")["HPV_pos"].max().rename_axis("patient").reset_index()

    mv = (d.merge(cov[["patient", "AGE", "late_stage"]], on="patient", how="left")
            .merge(hpv, on="patient", how="left"))
    mv["infiltration_z"] = (mv.interior_infiltration - mv.interior_infiltration.mean()) \
        / mv.interior_infiltration.std()
    mv = mv[["patient", "OS_t", "OS_e", "interior_infiltration", "infiltration_z",
             "HPV_pos", "AGE", "late_stage"]].dropna()
    return mv
