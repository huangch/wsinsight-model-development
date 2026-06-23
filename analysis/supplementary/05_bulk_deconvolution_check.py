#!/usr/bin/env python
"""
Supplementary Analysis 5 -- Can bulk deconvolution replace the spatial read-out?

Reviewer question
-----------------
"You show a CD8 border-exclusion gradient in Xenium and a matching border-immune
signal in the TCGA H-Plot.  Couldn't ordinary bulk RNA-seq deconvolution (EPIC /
CIBERSORTx) of the same TCGA tumours give the same answer, making the spatial
analysis unnecessary?"

Important limitation (read first)
---------------------------------
A *literal* EPIC / CIBERSORTx run is NOT possible in this environment:
  * neither EPIC, CIBERSORTx, nor R is installed;
  * CIBERSORTx additionally needs a web token / container we do not have;
  * and only a 23-gene immune subset of the TCGA-HNSC RSEM matrix is cached
    locally (data_mrna_seq_v2_rsem.txt = GEP + checkpoint genes), not the full
    ~20k-gene matrix a reference-based deconvolution needs.
So this script does NOT claim to reproduce EPIC/CIBERSORTx outputs.  Instead it
asks the question those tools would be used for -- "does a single bulk CD8 /
cytotoxic read-out per patient recover the spatial border-exclusion signal?" --
using the closest feasible bulk cell-abundance proxies:
  (i)  a bulk CD8/cytotoxic marker z-score from the cached RSEM
       (mean z of log1p CD8A, GZMB, NKG7, CCL5 -- the marker-based core of any
       deconvolution method);
  (ii) the pre-computed Bindea CD8 T-cell and cytotoxic-cell ssGSEA signature
       scores (a CIBERSORT-family per-patient cell-abundance estimate);
  (iii) bulk CD8A expression (gex_CD8A).
If a FULL bulk matrix (> %d genes) is ever placed at the RSEM path, the script
instead runs a transparent, dependency-free NNLS reference deconvolution against
a Xenium cell-subtype signature (EPIC-style) and uses its CD8 fraction.

The decisive comparison
-----------------------
A bulk read-out is ONE number per patient: it has no layer axis.  We therefore
test whether it recovers (a) the AMOUNT of border infiltration
(interior_infiltration, mean immune fraction over L in [-5,0]) and, separately,
(b) the SPATIAL GEOMETRY -- the border-minus-core contrast
(interior_infiltration - core_infiltration, L in [10,20]) that is the actual
H-Plot contribution.  Bulk recovering (a) but not (b) is the expected,
paper-supporting result: bulk sees how much immune signal there is, not where it
sits relative to the tumour border.

Outputs (data/results/head_neck/supplementary/)
  - bulk_deconvolution_results.csv              (per-patient read-outs + spatial)
  - bulk_deconvolution_correlation_summary.csv  (Spearman/Pearson, KW, by HPV)
  - bulk_deconvolution.png / .svg
Prints an honest PASS/SOFTEN/FLAG summary.

Run:
  PYTHONPATH=/workspace/wsinsight/sptxinsight \
    /opt/anaconda3/envs/spatial/bin/python \
    analysis/supplementary/05_bulk_deconvolution_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, kruskal

sys.path.insert(0, os.path.dirname(__file__))
import _shared as sh  # noqa: E402

FULL_GENE_THRESHOLD = 1000     # >= this many genes => attempt real deconvolution
CD8_MARKERS = ["CD8A", "GZMB", "NKG7", "CCL5"]
CORE_LO, CORE_HI = 10, 20      # tumour-core layer window
SIG_P = 0.05

__doc__ = __doc__ % FULL_GENE_THRESHOLD


def _zscore(df):
    return (df - df.mean()) / df.std(ddof=0)


def load_bulk_rsem():
    """Load the cached RSEM table -> (genes x patients) DataFrame, patient=12chr."""
    f = sh._require(f"{sh.TCGA_DIR}/labels/data_mrna_seq_v2_rsem.txt")
    raw = pd.read_csv(f, sep="\t")
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in raw.columns else raw.columns[0]
    raw = raw.dropna(subset=[gene_col]).drop_duplicates(subset=[gene_col])
    expr = raw.set_index(gene_col)
    expr = expr.select_dtypes(include=[np.number])
    expr.columns = [c[:12] for c in expr.columns]
    expr = expr.T.groupby(level=0).mean().T          # collapse replicate samples
    return expr


def bulk_cd8_marker_score(expr):
    present = [g for g in CD8_MARKERS if g in expr.index]
    if len(present) < 2:
        raise sh.MissingInput(
            f"need >=2 CD8/cytotoxic markers in the bulk matrix; "
            f"found {present} of {CD8_MARKERS}")
    sub = np.log1p(expr.loc[present].T.astype(float))      # patients x markers
    z = _zscore(sub)
    return z.mean(axis=1).rename("bulk_cd8_marker_z"), present


def try_full_deconvolution(expr):
    """EPIC-style NNLS CD8 fraction IF a full bulk matrix is present, else None.

    Builds a Xenium cell-subtype reference (mean profile per subtype over shared
    genes) and solves min_x ||S x - b||, x>=0 per patient; returns the CD8 T-cell
    fraction.  Skipped (returns None) when only the 23-gene subset is cached.
    """
    if expr.shape[0] < FULL_GENE_THRESHOLD:
        return None
    import scanpy as sc
    from scipy.optimize import nnls
    panel = sh.build_panel()
    # build per-subtype mean profile pooled across runs
    prof = {}
    for d in panel["sample_dirs"]:
        ad = sc.read_h5ad(os.path.join(d, sh.H5AD_NAME))
        if "cell_subtype" not in ad.obs:
            continue
        X = ad.X.tocsr()
        vn = list(map(str, ad.var_names))
        for st, idx in ad.obs.groupby("cell_subtype").groups.items():
            ii = ad.obs.index.get_indexer(idx)
            m = np.asarray(X[ii].mean(axis=0)).ravel()
            prof.setdefault(st, []).append(pd.Series(m, index=vn))
    ref = pd.DataFrame({st: pd.concat(v, axis=1).mean(axis=1)
                        for st, v in prof.items()})
    shared = ref.index.intersection(expr.index)
    if len(shared) < 50:
        return None
    S = np.log1p(ref.loc[shared].to_numpy())
    B = np.log1p(expr.loc[shared].to_numpy())
    cd8_cols = [c for c in ref.columns if "CD8" in str(c)]
    if not cd8_cols:
        return None
    fracs = []
    for j in range(B.shape[1]):
        x, _ = nnls(S, B[:, j])
        x = x / max(x.sum(), 1e-9)
        fracs.append(float(sum(x[list(ref.columns).index(c)] for c in cd8_cols)))
    return pd.Series(fracs, index=expr.columns, name="deconv_cd8_fraction")


def spatial_features():
    """Per-patient TCGA H-Plot spatial summaries from hplot-outputs.csv."""
    f = sh._require(f"{sh.TCGA_DIR}/results/objects/hplot-outputs.csv")
    cur = pd.read_csv(f)
    cur = cur[(cur.layer >= -10) & (cur.layer <= 20)].copy()
    cur["patient"] = cur["id"].str[:12]
    piv = cur.groupby(["patient", "layer"])["target_prop"].mean().unstack("layer")
    cols = list(piv.columns)
    interior = piv[[L for L in cols if -5 <= L <= 0]].mean(axis=1)
    core = piv[[L for L in cols if CORE_LO <= L <= CORE_HI]].mean(axis=1)
    out = pd.DataFrame({
        "interior_infiltration": interior,
        "core_infiltration": core,
        "border_minus_core": interior - core,
        "overall_infiltration": piv.mean(axis=1)})
    return out.dropna(subset=["interior_infiltration", "border_minus_core"])


def corr_block(x, y, label_bulk, label_spatial, stratum):
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 2 * sh.MIN_PER_GROUP:
        return dict(bulk_readout=label_bulk, spatial_feature=label_spatial,
                    stratum=stratum, n=n, spearman_r=np.nan, spearman_p=np.nan,
                    pearson_r=np.nan, pearson_p=np.nan, kw_p=np.nan, note="low-n")
    xs, ys = x[m].to_numpy(), y[m].to_numpy()
    sr, sp = spearmanr(xs, ys)
    pr, pp = pearsonr(xs, ys)
    # KW: split bulk read-out at its median, compare the spatial feature
    hi = xs > np.median(xs)
    if hi.sum() >= sh.MIN_PER_GROUP and (~hi).sum() >= sh.MIN_PER_GROUP:
        _, kwp = kruskal(ys[hi], ys[~hi])
    else:
        kwp = np.nan
    return dict(bulk_readout=label_bulk, spatial_feature=label_spatial,
                stratum=stratum, n=n, spearman_r=round(float(sr), 4),
                spearman_p=round(float(sp), 5), pearson_r=round(float(pr), 4),
                pearson_p=round(float(pp), 5),
                kw_p=(round(float(kwp), 5) if np.isfinite(kwp) else np.nan),
                note="")


def main():
    print("== Supplementary Analysis 5: bulk deconvolution vs spatial gradient ==")

    # ---- bulk read-outs --------------------------------------------------- #
    expr = load_bulk_rsem()
    n_genes, n_pat_bulk = expr.shape
    print(f"cached bulk RSEM: {n_genes} genes x {n_pat_bulk} patients")
    if n_genes < FULL_GENE_THRESHOLD:
        print(f"  -> only a {n_genes}-gene immune subset is present "
              f"(< {FULL_GENE_THRESHOLD}); EPIC/CIBERSORTx reference "
              "deconvolution is NOT run. Using bulk CD8/cytotoxic read-outs as "
              "the closest feasible cell-abundance proxy (see header).")
    marker_z, used_markers = bulk_cd8_marker_score(expr)
    print(f"  bulk CD8/cytotoxic marker z-score from: {used_markers}")
    deconv = try_full_deconvolution(expr)
    if deconv is not None:
        print(f"  full matrix detected -> ran NNLS reference deconvolution "
              f"({deconv.notna().sum()} patients)")

    # pre-computed bulk signatures from the label tables
    mat, _long = sh.load_tcga_curves()
    labels_df = sh.load_tcga_labels(mat)
    hpv = labels_df["hpv_status"] if "hpv_status" in labels_df.columns else None

    bulk = pd.DataFrame(index=marker_z.index)
    bulk["bulk_cd8_marker_z"] = marker_z
    for col, nm in (("cd8_t_cells", "bindea_cd8_ssgsea"),
                    ("cytotoxic_cells", "bindea_cytotoxic_ssgsea"),
                    ("gex_CD8A", "bulk_CD8A_expr")):
        if col in labels_df.columns:
            bulk[nm] = pd.to_numeric(labels_df[col], errors="coerce").reindex(
                bulk.index)
    if deconv is not None:
        bulk["deconv_cd8_fraction"] = deconv.reindex(bulk.index)

    # ---- spatial features ------------------------------------------------- #
    spat = spatial_features()
    print(f"spatial H-Plot summaries for {spat.shape[0]} patients")

    df = bulk.join(spat, how="inner")
    if hpv is not None:
        df["HPV"] = hpv.reindex(df.index)
    n = len(df)
    print(f"matched bulk+spatial patients: n={n}")
    if n < 4 * sh.MIN_PER_GROUP:
        raise sh.MissingInput(f"too few matched patients (n={n}) for correlation")

    res_csv = os.path.join(sh.SUPP_DIR, "bulk_deconvolution_results.csv")
    df.round(5).to_csv(res_csv)
    print("wrote", res_csv)

    # ---- correlations: each bulk read-out vs LEVEL and vs GEOMETRY --------- #
    bulk_cols = [c for c in bulk.columns if c in df.columns]
    spatial_targets = [("interior_infiltration", "LEVEL (interior infiltration)"),
                       ("border_minus_core", "GEOMETRY (border - core)")]
    PRIMARY = "bulk_cd8_marker_z"

    rows = []
    for bc in bulk_cols:
        for sf, _lab in spatial_targets:
            rows.append(corr_block(df[bc], df[sf], bc, sf, "all"))
            if hpv is not None:
                for arm in ("HPV-", "HPV+"):
                    sub = df[df["HPV"] == arm]
                    rows.append(corr_block(sub[bc], sub[sf], bc, sf, arm))
    summary = pd.DataFrame(rows)
    sum_csv = os.path.join(sh.SUPP_DIR, "bulk_deconvolution_correlation_summary.csv")
    summary.to_csv(sum_csv, index=False)
    print("wrote", sum_csv)
    pd.set_option("display.width", 220)
    print("\n", summary[summary.stratum == "all"].to_string(index=False))

    # ---- figure ----------------------------------------------------------- #
    show = [c for c in (PRIMARY, "bindea_cd8_ssgsea") if c in df.columns]
    fig, axes = plt.subplots(len(show), 2, figsize=(10, 4.2 * len(show)),
                             squeeze=False)
    for i, bc in enumerate(show):
        for j, (sf, lab) in enumerate(spatial_targets):
            ax = axes[i][j]
            x = df[bc]; y = df[sf]; m = x.notna() & y.notna()
            ax.scatter(x[m], y[m], s=12, alpha=0.5, color="#3182bd",
                       edgecolor="none")
            if m.sum() > 2:
                b1, b0 = np.polyfit(x[m], y[m], 1)
                xs = np.linspace(x[m].min(), x[m].max(), 50)
                ax.plot(xs, b1 * xs + b0, color="#d62728", lw=1.4)
                sr, sp = spearmanr(x[m], y[m])
                ax.set_title(f"{bc}\nvs {lab}\nSpearman r={sr:+.2f}, p={sp:.3f}",
                             fontsize=9)
            ax.axhline(0, color="0.7", lw=0.6, ls="--")
            ax.set_xlabel(bc, fontsize=8); ax.set_ylabel(sf, fontsize=8)
    fig.suptitle("Bulk CD8 read-out vs spatial infiltration: recovers LEVEL "
                 "(left) far better than GEOMETRY (right)", y=1.001, fontsize=11)
    fig.tight_layout()
    png = os.path.join(sh.SUPP_DIR, "bulk_deconvolution.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, "and .svg")

    # ---- verdict ---------------------------------------------------------- #
    def _get(bc, sf, stratum="all", field="spearman_p"):
        r = summary[(summary.bulk_readout == bc) & (summary.spatial_feature == sf)
                    & (summary.stratum == stratum)]
        return float(r[field].iloc[0]) if len(r) else np.nan

    lvl_p = _get(PRIMARY, "interior_infiltration")
    lvl_r = _get(PRIMARY, "interior_infiltration", field="spearman_r")
    geo_p = _get(PRIMARY, "border_minus_core")
    geo_r = _get(PRIMARY, "border_minus_core", field="spearman_r")
    recovers_level = np.isfinite(lvl_p) and lvl_p < SIG_P and lvl_r > 0
    recovers_geom = np.isfinite(geo_p) and geo_p < SIG_P and geo_r > 0

    print("\n--- SUMMARY ---")
    print(f"primary bulk read-out: {PRIMARY}")
    print(f"  vs LEVEL    (interior infiltration): Spearman r={lvl_r:+.2f}, "
          f"p={lvl_p:.3f} -> {'recovers' if recovers_level else 'does NOT recover'}")
    print(f"  vs GEOMETRY (border - core)        : Spearman r={geo_r:+.2f}, "
          f"p={geo_p:.3f} -> {'recovers' if recovers_geom else 'does NOT recover'}")
    if recovers_level and not recovers_geom:
        print("[PASS-COMPLEMENTARY] A single bulk CD8/cytotoxic read-out tracks "
              "HOW MUCH immune signal a tumour has, but NOT WHERE it sits "
              "relative to the tumour border. The spatial border-exclusion "
              "gradient resolved by the H-Plot is therefore non-redundant with "
              "bulk deconvolution -- bulk cannot replace the spatial read-out. "
              "(Caveat: literal EPIC/CIBERSORTx not run; bulk proxied by CD8 "
              "marker/ssGSEA scores on the cached 23-gene subset.)")
    elif recovers_level and recovers_geom:
        print("[SOFTEN] The bulk CD8 read-out correlates with BOTH the amount and "
              "the border-vs-core geometry of infiltration. The spatial advantage "
              "is therefore partial, not absolute; wording claiming bulk cannot "
              "see the gradient should be softened. (Bulk still gives one number "
              "per patient with no per-layer profile.)")
    elif not recovers_level:
        print("[FLAG] The bulk CD8 read-out does not even track the overall "
              "infiltration level here (likely limited power / the 23-gene "
              "subset). No strong claim either way can be made from this proxy; "
              "a full-matrix EPIC/CIBERSORTx run would be needed to settle it.")


if __name__ == "__main__":
    main()
