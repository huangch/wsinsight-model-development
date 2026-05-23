
#!/usr/bin/env python
# coding: utf-8
"""
kurtorank.annotate.main
-----------------------
KurtoRank v3 annotate pipeline. Ported from kurtorank3.ipynb, derived from
kurtorank2-cli.py.

v3 changes relative to v2:
  - Consumes markers-v3.csv (atlas-reranked by rank_markers.py). Tolerates
    the extra `rank_source` and `low_support` columns; malignant filtering
    uses the `malignant` column rather than v2's `normal`.
  - Optional `--use-top-k-markers K` truncation: keep only the K most
    specific genes per subtype (CSV order reflects atlas specificity).
  - QC/SNR: fallback-aware. `negative_probe_counts` is preferred; when
    absent, `control_probe_counts` is used and the choice is recorded in
    `adata.uns["snr_source"]`.
  - Cell filtering uses `sc.pp.filter_cells(min_genes=...)` instead of
    `min_counts=...` (semantic fix).
  - Explicit per-array FDR clipping (v2 used `locals()[...] = ...` which
    silently no-op'd inside a function).
  - Tie-breaking only considers tie_break_priority entries that are active
    in `method_switch` AND present as columns; disabled methods have
    FDR==1.0 and cannot discriminate.
"""

from __future__ import annotations

import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message="Transforming to str index.",
)

import os
import gc
import math
import logging
import signal
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Mapping, Optional, Sequence
import click


def _default_markers_csv() -> Path:
    """Path to the markers-v3_2.csv bundled with this package."""
    try:
        from importlib.resources import files as _resource_files
        return Path(str(_resource_files("kurtorank.markers") / "data" / "markers-v3_2.csv"))
    except Exception:
        # Fallback for editable installs / older Python.
        return Path(__file__).resolve().parent.parent / "markers" / "data" / "markers-v3_2.csv"


import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving only

# Heavy scientific stack — imported lazily so `kurtorank annotate --help`
# works in a minimal environment. Actual invocation of the annotate pipeline
# still requires the full dependency set declared in pyproject.toml.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import spatialdata as sd
    from spatialdata_io import xenium
    import anndata as ad
    import scanpy as sc
    import squidpy as sq
    import torch
    from torch.distributions import Chi2
    from scipy.stats import (
        norm,
        combine_pvalues,
        hypergeom,
        zscore,
        rankdata,
        kurtosis,
        fisher_exact,
        ks_2samp,
        pearsonr,
        chi2,
        spearmanr,
    )
    from scipy.spatial import cKDTree
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.contingency_tables import Table2x2
    from tqdm.auto import tqdm
except ImportError as _e:  # pragma: no cover — defer import errors until run-time
    _KURTO_IMPORT_ERROR = _e
else:
    _KURTO_IMPORT_ERROR = None

TISSUE_TYPES: tuple[str, ...] = (
    "bladder",
    "bone",
    "brain",
    "breast",
    "cervix",
    "circulating",
    "colorectal",
    "heart",
    "immune",
    "kidney",
    "liver",
    "lung",
    "lymph_node",
    "ovary",
    "pancreas",
    "prostate",
    "skin",
    "tonsil",
)

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

METHOD_LABELS: dict[str, str] = {
    "emp_fdr": "Empirical",
    "topn_overlap_fdr": "Top-N Overlap",
    "threshold_overlap_fdr": "Threshold Overlap",
    "de_fdr": "DE (Geom. Mean)",
    "z_fdr": "DE Z-score",
    "fisher_fdr": "Fisher",
    "prop_fdr": "Proportion",
    "corr_fdr": "Correlation",
    "spatial_co_fdr": "Spatial Co-localization",
}

DEFAULT_METHOD_SWITCH: dict[str, bool] = {
    "emp_fdr": True,
    "topn_overlap_fdr": True,
    "threshold_overlap_fdr": False,
    "de_fdr": True,
    "z_fdr": False,
    "fisher_fdr": True,
    "prop_fdr": False,
    "corr_fdr": True,
    "spatial_co_fdr": True,
}


# ---------------------- logging helpers ----------------------


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("scanpy").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# Global plotting configuration; controlled via CLI options in main().
PLOT_DPI: int = 600
PLOT_FORMAT: str = "png"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def cluster_sort_key(val):
    # Cluster labels are sometimes numeric strings, sometimes text; keep ordering deterministic.
    """Sort numerically when possible, otherwise lexicographically."""
    try:
        return (0, int(val))
    except (TypeError, ValueError):
        return (1, str(val))


def obs_as_str_with_unknown(series: pd.Series, placeholder: str = "unknown") -> pd.Series:
    """Return a string series, ensuring placeholder is a valid category when needed."""
    ser = series.copy()
    if pd.api.types.is_categorical_dtype(ser):
        current = ser.cat.categories
        if placeholder not in current:
            ser = ser.cat.add_categories([placeholder])
    return ser.fillna(placeholder).astype(str)


# ---------------------- signal handling ----------------------


_ORIGINAL_SIGINT_HANDLER = signal.SIG_DFL
_LAST_SIGINT_TIME = 0.0
_DOUBLE_SIGINT_WINDOW = 5.0


def install_double_sigint_handler(window: float = 5.0):
    # Give users a chance to cancel accidental interrupts and only exit on a quick double Ctrl-C.
    """Require two Ctrl-C presses within `window` seconds before exiting."""
    global _ORIGINAL_SIGINT_HANDLER, _DOUBLE_SIGINT_WINDOW, _LAST_SIGINT_TIME

    _ORIGINAL_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
    _DOUBLE_SIGINT_WINDOW = max(0.1, float(window))
    _LAST_SIGINT_TIME = 0.0

    def _handler(signum, frame):
        global _LAST_SIGINT_TIME
        now = time.monotonic()
        if _LAST_SIGINT_TIME and (now - _LAST_SIGINT_TIME) <= _DOUBLE_SIGINT_WINDOW:
            logger.warning("Second Ctrl-C received; exiting now.")
            signal.signal(signal.SIGINT, _ORIGINAL_SIGINT_HANDLER or signal.SIG_DFL)
            handler = _ORIGINAL_SIGINT_HANDLER
            if callable(handler):
                handler(signum, frame)
            else:
                raise KeyboardInterrupt
        else:
            _LAST_SIGINT_TIME = now
            logger.warning(
                "Ctrl-C detected. Press again within %.1f seconds to terminate." % _DOUBLE_SIGINT_WINDOW
            )

    signal.signal(signal.SIGINT, _handler)


def save_fig(fig, out_dir: Path, filename: str, dpi: int | None = None, plot_format: str | None = None):
    """Save a Matplotlib figure with the global/default plot settings.

    The filename's extension is replaced with the requested format
    (default "png"), so passing "foo.png" with plot_format="svg" will
    produce "foo.svg" on disk.
    """
    global PLOT_DPI, PLOT_FORMAT
    effective_dpi = int(dpi) if dpi is not None else int(PLOT_DPI)
    fmt = (plot_format or PLOT_FORMAT or "png").lower()

    out_path = out_dir / filename
    # Always enforce the chosen format as the file extension.
    out_path = out_path.with_suffix(f".{fmt}")

    fig.savefig(out_path, dpi=effective_dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


def save_current_fig(out_dir: Path, filename: str, dpi: int | None = None, plot_format: str | None = None):
    fig = plt.gcf()
    save_fig(fig, out_dir, filename, dpi=dpi, plot_format=plot_format)


def resolve_method_switch(selected_methods: Sequence[str]) -> dict[str, bool]:
    """Return a method switch map, honoring CLI selections and defaults."""
    switch = DEFAULT_METHOD_SWITCH.copy()
    if selected_methods:
        normalized = {m.lower() for m in selected_methods}
        switch = {method: (method in normalized) for method in CORE_FDR_METHODS}
    if not any(switch.values()):
        raise click.BadParameter("At least one annotation method must remain enabled.", param_hint="--method")
    return switch


# ---------------------- data loading ----------------------


def load_or_build_adata(xenium_dir: Path) -> ad.AnnData:
    raw_path = xenium_dir / "raw.h5ad"

    if raw_path.exists():
        logger.info(f"Loading existing raw data: {raw_path}")
        adata = sc.read_h5ad(raw_path)
        return adata

    logger.info(f"Loading Xenium spatial data from: {xenium_dir}")
    sdata = xenium(str(xenium_dir), cells_as_circles=True)
    adata = sdata.tables["table"].copy()
    logger.info(f"Loaded adata: {adata}")

    graphclust_csv = xenium_dir / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
    if graphclust_csv.exists():
        logger.info(f"Loading Xenium graphclust: {graphclust_csv}")
        graphclust_df = pd.read_csv(graphclust_csv, dtype={"Cluster": str})
        graphclust_dict = dict(zip(graphclust_df["Barcode"], graphclust_df["Cluster"]))
        adata.obs["graphclust"] = adata.obs.cell_id.map(graphclust_dict).astype("category")
    else:
        logger.warning("graphclust CSV not found; 'graphclust' column will be missing.")

    logger.info(f"Saving raw.h5ad to: {raw_path}")
    adata.write_h5ad(raw_path)
    return adata


def ensure_graphclust(adata: ad.AnnData, xenium_dir: Path):
    if "graphclust" in adata.obs.columns:
        return
    graphclust_csv = xenium_dir / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
    if graphclust_csv.exists():
        logger.info(f"Adding 'graphclust' from: {graphclust_csv}")
        graphclust_df = pd.read_csv(graphclust_csv, dtype={"Cluster": str})
        graphclust_dict = dict(zip(graphclust_df["Barcode"], graphclust_df["Cluster"]))
        adata.obs["graphclust"] = adata.obs.cell_id.map(graphclust_dict).astype("category")
    else:
        logger.warning("graphclust CSV not found; cannot add 'graphclust'.")


# ---------------------- QC and clustering ----------------------


def run_qc(
    adata: ad.AnnData,
    min_genes: int,
    min_cells: int,
    lower_percentile: float,
    upper_percentile: float,
    n_top_genes: int,
    generate_plots: bool,
    out_dir: Path,
):
    logger.info("Calculating QC metrics.")
    sc.pp.calculate_qc_metrics(
        adata,
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    if "nucleus_area" not in adata.obs:
        logger.warning("nucleus_area not found; some Xenium metrics may be missing.")

    # v3: fallback-aware SNR source. Only override negative_probe_counts with
    # control_probe_counts when the former is absent; record the source in
    # adata.uns for downstream audit.
    if "negative_probe_counts" in adata.obs.columns:
        adata.uns["snr_source"] = "negative_probe_counts"
    elif "control_probe_counts" in adata.obs.columns:
        adata.obs["negative_probe_counts"] = adata.obs["control_probe_counts"]
        adata.uns["snr_source"] = "control_probe_counts (fallback)"
    else:
        adata.obs["negative_probe_counts"] = 0.0
        adata.uns["snr_source"] = "none (zeros)"

    if "control_probe_counts" in adata.obs and "total_counts" in adata.obs:
        adata.obs["control_probe_ratio"] = adata.obs["control_probe_counts"] / adata.obs["total_counts"]
    else:
        adata.obs["control_probe_ratio"] = 0.0

    if "nucleus_area" in adata.obs and "cell_area" in adata.obs:
        adata.obs["nucleus_area_ratio"] = adata.obs["nucleus_area"] / adata.obs["cell_area"]
    else:
        adata.obs["nucleus_area_ratio"] = np.nan

    if "transcript_counts" in adata.obs and "negative_probe_counts" in adata.obs:
        adata.obs["signal_to_noise"] = (
            (adata.obs["transcript_counts"] - adata.obs["negative_probe_counts"]) /
            adata.obs["transcript_counts"]
        ).fillna(0)
    else:
        adata.obs["signal_to_noise"] = 0.0

    if "control_probe_counts" in adata.obs and "total_counts" in adata.obs:
        cprobes = adata.obs["control_probe_counts"].sum() / adata.obs["total_counts"].sum() * 100
        logger.info(f"Negative DNA probe count %: {cprobes:.3f}")
    if "control_codeword_counts" in adata.obs and "total_counts" in adata.obs:
        cwords = adata.obs["control_codeword_counts"].sum() / adata.obs["total_counts"].sum() * 100
        logger.info(f"Negative decoding count %: {cwords:.3f}")

    available_obs = [
        c for c in [
            "total_counts",
            "n_genes_by_counts",
            "nucleus_area",
            "cell_area",
            "nucleus_area_ratio",
            "transcript_counts",
            "control_probe_counts",
            "control_codeword_counts",
            "signal_to_noise",
            "control_probe_ratio",
        ]
        if c in adata.obs.columns
    ]

    if generate_plots and available_obs:
        logger.info("Plotting QC histograms.")
        fig_cols = int(np.ceil(np.sqrt(len(available_obs))))
        fig_rows = int(np.ceil(len(available_obs) / fig_cols))
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))
        axes = np.array(axes).reshape(fig_rows, fig_cols)

        for ax in axes.flat:
            ax.set_visible(False)

        for i, obs in enumerate(available_obs):
            r = i // fig_cols
            c = i % fig_cols
            sns.histplot(adata.obs[obs], kde=True, ax=axes[r, c])
            axes[r, c].set_title(f"Distribution of {obs}")
            axes[r, c].set_visible(True)

        plt.tight_layout()
        save_fig(fig, out_dir, "qc_histograms.png")

        logger.info("Plotting QC spatial maps.")
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))
        axes = np.array(axes).reshape(fig_rows, fig_cols)
        for ax in axes.flat:
            ax.set_visible(False)

        for i, obs in enumerate(available_obs):
            r = i // fig_cols
            c = i % fig_cols
            fig_sc = sc.pl.spatial(
                adata,
                color=obs,
                size=10,
                spot_size=1,
                show=False,
                ax=axes[r, c],
                title=obs,
            )
            axes[r, c].set_visible(True)

        plt.tight_layout()
        save_fig(fig, out_dir, "qc_spatial.png")

    logger.info(f"Filtering cells/genes: min_genes={min_genes}, min_cells={min_cells}")
    adata.raw = adata.copy()
    original_cells = adata.n_obs

    # v3: use min_genes (minimum genes detected per cell) rather than
    # min_counts (total UMI). Semantics match the v3 notebook.
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    if lower_percentile > 0 or upper_percentile < 100:
        if "total_counts" in adata.obs:
            count_min = np.percentile(adata.obs["total_counts"], lower_percentile)
            count_max = np.percentile(adata.obs["total_counts"], upper_percentile)
            logger.info(f"Transcript count range: {count_min:.1f} - {count_max:.1f}")
            adata._inplace_subset_obs(
                (adata.obs["total_counts"] >= count_min) &
                (adata.obs["total_counts"] <= count_max)
            )

        if "n_genes_by_counts" in adata.obs:
            genes_min = np.percentile(adata.obs["n_genes_by_counts"], lower_percentile)
            genes_max = np.percentile(adata.obs["n_genes_by_counts"], upper_percentile)
            logger.info(f"Genes detected range: {genes_min:.1f} - {genes_max:.1f}")
            adata._inplace_subset_obs(
                (adata.obs["n_genes_by_counts"] >= genes_min) &
                (adata.obs["n_genes_by_counts"] <= genes_max)
            )

        if "nucleus_area" in adata.obs:
            adata = adata[adata.obs["nucleus_area"].notna()].copy()
            area_min = np.percentile(adata.obs["nucleus_area"], lower_percentile)
            area_max = np.percentile(adata.obs["nucleus_area"], upper_percentile)
            logger.info(f"Nucleus area range: {area_min:.1f} - {area_max:.1f}")
            adata._inplace_subset_obs(
                (adata.obs["nucleus_area"] >= area_min) &
                (adata.obs["nucleus_area"] <= area_max)
            )

    cells_kept = adata.n_obs
    cells_removed = original_cells - cells_kept
    percent_removed = cells_removed / max(original_cells, 1) * 100
    logger.info(
        f"Cells before filtering: {original_cells}; "
        f"after: {cells_kept}; removed: {cells_removed} ({percent_removed:.1f}%)"
    )

    logger.info("Normalization and HVG selection (n_top_genes=%d).", n_top_genes)
    adata.layers["counts"] = adata.X.copy()

    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
     
    # sc.pp.highly_variable_genes(
    #     adata,
    #     flavor="seurat",
    #     n_top_genes=n_top_genes,
    # )
       
    #
    # seurat_v3 needs to be done on dataraw counts w
    #
     
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    
    if generate_plots:
        logger.info("Plotting highly variable genes.")
        sc.pl.highly_variable_genes(adata, show=False)
        save_current_fig(out_dir, "hvg.png")        

    logger.info("Running PCA and neighbors.")
    if "X_pca" not in adata.obsm:
        adata_proc = adata[:, adata.var.highly_variable].copy()
        sc.pp.scale(adata_proc, max_value=10)
        sc.tl.pca(adata_proc, svd_solver="arpack")
        adata.obsm["X_pca"] = adata_proc.obsm["X_pca"]
        adata.uns["pca"] = adata_proc.uns["pca"]

    if generate_plots:
        sc.pl.pca_variance_ratio(adata, log=True, show=False)
        save_current_fig(out_dir, "pca_variance_ratio.png")

    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30, use_rep="X_pca")
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    # if generate_plots:
    #     sc.pl.umap(
    #         adata,
    #         color=["total_counts", "n_genes_by_counts"],
    #         use_raw=False,
    #         color_map="viridis",
    #         show=False,
    #     )
    #     save_current_fig(out_dir, "umap_qc.png")

    return adata


def run_leiden_scan(
    adata: ad.AnnData,
    use_graphclust: bool,
    chosen_leiden_res: float,
    generate_plots: bool,
    out_dir: Path,
):
    leiden_resolutions = [0.8, 0.5, 1.2]
    if not use_graphclust:
        logger.info("Running Leiden clustering for candidate resolutions.")
        for res in leiden_resolutions:
            key = f"leiden_res_{res}"
            if key not in adata.obs:
                sc.tl.leiden(adata, resolution=res, key_added=key)
        if generate_plots:
            sc.pl.umap(
                adata,
                color=[f"leiden_res_{res}" for res in leiden_resolutions],
                show=False,
            )
            save_current_fig(out_dir, "umap_leiden_resolutions.png")

    if use_graphclust:
        primary_cluster = "graphclust"
    else:
        primary_cluster = f"leiden_res_{chosen_leiden_res}"

    if primary_cluster not in adata.obs:
        raise ValueError(f"Primary cluster column '{primary_cluster}' not found in adata.obs")

    adata.obs["clusters"] = adata.obs[primary_cluster]
    logger.info(f"Chosen primary cluster: {primary_cluster}")

    if generate_plots:
        sc.pl.umap(
            adata,
            color=["total_counts", "n_genes_by_counts", primary_cluster],
            # color=[primary_cluster],
            legend_loc="on data",
            legend_fontsize=8,
            legend_fontoutline=2,
            frameon=False,
            show=False,
        )
        save_current_fig(out_dir, "umap_primary_cluster.png")

        sc.pl.spatial(
            adata,
            color=["total_counts", "n_genes_by_counts", primary_cluster],
            # color=[primary_cluster],
            size=10,
            show=False,
            spot_size=1,
            title="Spatial Distribution of Clusters",
        )
        save_current_fig(out_dir, "spatial_primary_cluster.png")

    return primary_cluster
# ---------------------- KurtoRank core ----------------------


def get_least_used_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n_gpu = torch.cuda.device_count()
    memory_allocated = [torch.cuda.memory_allocated(i) for i in range(n_gpu)]
    best_gpu = int(np.argmin(memory_allocated))
    return torch.device(f"cuda:{best_gpu}")


def torch_empirical_p(adata_sub: ad.AnnData, gene_list, n_perm=1000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    device = get_least_used_gpu()

    var_names = list(adata_sub.var_names)
    gene_indices = [var_names.index(g) for g in gene_list if g in var_names]
    if len(gene_indices) < 3:
        return 1.0, None

    X = adata_sub.X
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(
            X.A if hasattr(X, "A") else X.toarray(),
            dtype=torch.float32,
        ).to(device)
    else:
        X = X.to(device)

    marker_expr = X[:, gene_indices]
    marker_stat = torch.median(marker_expr, dim=1).values.mean()

    n_genes = X.shape[1]
    perm_stats = []
    for _ in range(n_perm):
        rand_indices = torch.randperm(n_genes)[: len(gene_indices)]
        rand_expr = X[:, rand_indices]
        rand_stat = torch.median(rand_expr, dim=1).values.mean()
        perm_stats.append(rand_stat)

    perm_stats = torch.stack(perm_stats)
    p_emp = (perm_stats >= marker_stat).float().mean().item()
    return p_emp, marker_stat


def geometric_mean_p(pvals):
    pvals = np.clip(pvals, 1e-300, 1.0)
    return float(np.exp(np.mean(np.log(pvals))))


def soft_weight(k, scale=1.0, shift=3.0):
    return 1 / (1 + np.exp(-scale * (k - shift)))


_CLUSTER_CONTEXT = None


def _init_cluster_context(context):
    global _CLUSTER_CONTEXT
    _CLUSTER_CONTEXT = context


def _process_cluster_worker(cluster_id):
    if _CLUSTER_CONTEXT is None:
        raise RuntimeError("Cluster worker context not initialized.")

    ctx = _CLUSTER_CONTEXT
    adata = ctx["adata"]
    primary_cluster = ctx["primary_cluster"]
    cell_subtypes = ctx["cell_subtypes"]
    all_markers = ctx["all_markers"]
    method_switch = ctx["method_switch"]
    n_perm = ctx["n_perm"]
    seed = ctx["seed"]
    top_n_de_genes = ctx["top_n_de_genes"]
    de_pval_threshold = ctx["de_pval_threshold"]
    de_logfc_threshold = ctx["de_logfc_threshold"]
    background_gene_count = ctx["background_gene_count"]
    all_major_types = ctx["all_major_types"]
    all_pannuke_labels = ctx["all_pannuke_labels"]
    all_hne_types = ctx["all_hne_types"]
    all_hne_labels = ctx["all_hne_labels"]
    all_pantissue_labels = ctx["all_pantissue_labels"]
    all_pantissue_types = ctx["all_pantissue_types"]
    all_malignant_indicators = ctx["all_malignant_indicators"]
    tie_break_priority = ctx["tie_break_priority"]
    var_names_all = ctx["var_names_all"]

    cluster_cells = adata.obs[primary_cluster] == cluster_id
    adata_sub = adata[cluster_cells, :].copy()
    adata_bg = adata[~cluster_cells, :].copy()

    try:
        names_all = adata.uns["rank_genes_groups"]["names"][cluster_id]
        pvals_all = adata.uns["rank_genes_groups"]["pvals"][cluster_id]
        logfc_all = adata.uns["rank_genes_groups"]["logfoldchanges"][cluster_id]
        scores_all = adata.uns["rank_genes_groups"]["scores"][cluster_id]
    except Exception:
        names_all = []
        pvals_all = []
        logfc_all = []
        scores_all = []

    try:
        names_top = adata.uns["rank_genes_groups"]["names"][cluster_id][:top_n_de_genes]
        topn_de_gene_set = set(names_top)
    except Exception:
        topn_de_gene_set = set()

    threshold_de_gene_set = set()
    if method_switch.get("threshold_overlap_fdr", False):
        try:
            for g, pval, logfc in zip(names_all, pvals_all, logfc_all):
                if pval < de_pval_threshold and abs(logfc) > de_logfc_threshold:
                    threshold_de_gene_set.add(g)
        except Exception:
            pass

    de_gene_pvals = {}
    try:
        for g, p in zip(names_all, pvals_all):
            de_gene_pvals[g] = p
    except Exception:
        pass

    rg_name_score = {}
    try:
        rg_name_score = dict(zip(names_all, scores_all))
    except Exception:
        pass

    emp_p_list = []
    de_p_list = []
    z_p_list = []
    topn_overlap_p_list = []
    threshold_overlap_p_list = []
    n_topn_overlap_list = []
    n_threshold_overlap_list = []
    marker_coverage = []

    fisher_p_list = []
    prop_p_list = []
    corr_p_list = []
    spatial_co_p_list = []

    has_spatial = "spatial" in adata_sub.obsm
    n_perm_corr = min(500, n_perm)
    n_perm_spatial = min(500, n_perm)

    for ct in cell_subtypes:
        markers = [g for g in all_markers[ct] if g in adata.var_names]
        if len(markers) < 3:
            emp_p_list.append(1.0)
            de_p_list.append(1.0)
            z_p_list.append(1.0)
            topn_overlap_p_list.append(1.0)
            threshold_overlap_p_list.append(1.0)
            n_topn_overlap_list.append(0)
            n_threshold_overlap_list.append(0)
            marker_coverage.append(0.0)
            fisher_p_list.append(1.0)
            prop_p_list.append(1.0)
            corr_p_list.append(1.0)
            spatial_co_p_list.append(1.0)
            continue

        if method_switch.get("emp_fdr", True):
            emp_p, _ = torch_empirical_p(adata_sub, markers, n_perm=n_perm, seed=seed)
            emp_p = np.clip(emp_p, 1e-300, 1.0)
        else:
            emp_p = 1.0
        emp_p_list.append(emp_p)

        if method_switch.get("de_fdr", True):
            marker_pvals = [de_gene_pvals[g] for g in markers if g in de_gene_pvals]
            if len(marker_pvals) >= 2:
                de_p = geometric_mean_p(marker_pvals)
            else:
                de_p = 1.0
            de_p = np.clip(de_p, 1e-300, 1.0)
        else:
            de_p = 1.0
        de_p_list.append(de_p)

        if method_switch.get("z_fdr", True):
            marker_z_scores = [rg_name_score[g] for g in markers if g in rg_name_score]
            if len(marker_z_scores) >= 2:
                avg_z = np.mean(marker_z_scores)
                z_p = 2 * norm.sf(abs(avg_z))
            else:
                z_p = 1.0
            z_p = np.clip(z_p, 1e-300, 1.0)
        else:
            z_p = 1.0
        z_p_list.append(z_p)

        if method_switch.get("topn_overlap_fdr", True):
            topn_overlap_genes = set(markers) & topn_de_gene_set
            n_topn_overlap = len(topn_overlap_genes)
            if len(topn_de_gene_set) > 0:
                topn_overlap_p = hypergeom.sf(
                    n_topn_overlap - 1,
                    background_gene_count,
                    len(markers),
                    len(topn_de_gene_set),
                )
            else:
                topn_overlap_p = 1.0
            topn_overlap_p = np.clip(topn_overlap_p, 1e-300, 1.0)
        else:
            n_topn_overlap = 0
            topn_overlap_p = 1.0
        topn_overlap_p_list.append(topn_overlap_p)
        n_topn_overlap_list.append(n_topn_overlap)

        if method_switch.get("threshold_overlap_fdr", False):
            threshold_overlap_genes = set(markers) & threshold_de_gene_set
            n_threshold_overlap = len(threshold_overlap_genes)
            if len(threshold_de_gene_set) > 0:
                threshold_overlap_p = hypergeom.sf(
                    n_threshold_overlap - 1,
                    background_gene_count,
                    len(markers),
                    len(threshold_de_gene_set),
                )
            else:
                threshold_overlap_p = 1.0
            threshold_overlap_p = np.clip(threshold_overlap_p, 1e-300, 1.0)
        else:
            n_threshold_overlap = 0
            threshold_overlap_p = 1.0
        threshold_overlap_p_list.append(threshold_overlap_p)
        n_threshold_overlap_list.append(n_threshold_overlap)

        hit = [g for g in markers if g in topn_de_gene_set]
        coverage = len(hit) / len(markers) if markers else 0.0
        marker_coverage.append(coverage)

        pos_c = pos_b = None
        a = b = c = d = None
        if (
            method_switch.get("fisher_fdr", True)
            or method_switch.get("prop_fdr", True)
            or method_switch.get("spatial_co_fdr", True)
        ):
            try:
                X_c = adata_sub[:, markers].X
                X_b = adata_bg[:, markers].X
                if hasattr(X_c, "toarray"):
                    X_c = X_c.toarray()
                if hasattr(X_b, "toarray"):
                    X_b = X_b.toarray()
                mean_c = X_c.mean(axis=1)
                mean_b = X_b.mean(axis=1)
                pos_c = mean_c > 0
                pos_b = mean_b > 0
                a = pos_c.sum()
                b = (~pos_c).sum()
                c = pos_b.sum()
                d = (~pos_b).sum()
            except Exception:
                pos_c = pos_b = None

        if method_switch.get("fisher_fdr", True) and (pos_c is not None):
            try:
                if (a + b == 0) or (c + d == 0) or (a + c == 0):
                    fisher_p = 1.0
                else:
                    table = np.array([[a, b], [c, d]], dtype=float)
                    if np.any(table == 0):
                        table = table + 0.5
                    _, fisher_p = fisher_exact(table, alternative="greater")
                    fisher_p = float(fisher_p)
            except Exception:
                fisher_p = 1.0
        else:
            fisher_p = 1.0
        fisher_p = np.clip(fisher_p, 1e-300, 1.0)
        fisher_p_list.append(fisher_p)

        if method_switch.get("prop_fdr", True) and (pos_c is not None):
            try:
                n1 = a + b
                n2 = c + d
                if n1 == 0 or n2 == 0:
                    prop_p = 1.0
                else:
                    p1 = a / n1
                    p2 = c / n2
                    p_pool = (a + c) / (n1 + n2)
                    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
                    if se == 0:
                        prop_p = 1.0
                    else:
                        z = (p1 - p2) / se
                        prop_p = 2 * norm.sf(abs(z))
            except Exception:
                prop_p = 1.0
        else:
            prop_p = 1.0
        prop_p = np.clip(prop_p, 1e-300, 1.0)
        prop_p_list.append(prop_p)

        if method_switch.get("corr_fdr", True):
            try:
                X_m = adata_sub[:, markers].X
                if hasattr(X_m, "toarray"):
                    X_m = X_m.toarray()
                if X_m.shape[0] < 3 or X_m.shape[1] < 2:
                    corr_p = 1.0
                else:
                    var_col = X_m.var(axis=0)
                    valid_cols = var_col > 0
                    if valid_cols.sum() < 2:
                        corr_p = 1.0
                    else:
                        X_m_valid = X_m[:, valid_cols]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            C = np.corrcoef(X_m_valid, rowvar=False)
                        iu = np.triu_indices_from(C, k=1)
                        obs_corr = np.nanmean(C[iu])

                        perm_corrs = []
                        for _ in range(n_perm_corr):
                            rand_genes = np.random.choice(
                                var_names_all, size=valid_cols.sum(), replace=False
                            )
                            X_rand = adata_sub[:, rand_genes].X
                            if hasattr(X_rand, "toarray"):
                                X_rand = X_rand.toarray()
                            if X_rand.shape[1] < 2:
                                continue
                            var_col_r = X_rand.var(axis=0)
                            valid_r = var_col_r > 0
                            if valid_r.sum() < 2:
                                continue
                            X_rand_valid = X_rand[:, valid_r]
                            with np.errstate(divide="ignore", invalid="ignore"):
                                C_rand = np.corrcoef(X_rand_valid, rowvar=False)
                            iu2 = np.triu_indices_from(C_rand, k=1)
                            perm_corrs.append(np.nanmean(C_rand[iu2]))
                        if len(perm_corrs) < 10:
                            corr_p = 1.0
                        else:
                            perm_corrs = np.array(perm_corrs)
                            corr_p = ((perm_corrs >= obs_corr).sum() + 1) / (len(perm_corrs) + 1)
            except Exception:
                corr_p = 1.0
        else:
            corr_p = 1.0
        corr_p = np.clip(corr_p, 1e-300, 1.0)
        corr_p_list.append(corr_p)

        if (
            method_switch.get("spatial_co_fdr", True)
            and has_spatial
            and (pos_c is not None)
        ):
            try:
                coords = adata_sub.obsm["spatial"]
                if coords.shape[0] != adata_sub.n_obs:
                    spatial_p = 1.0
                else:
                    coords_pos = coords[pos_c]
                    n_pos = coords_pos.shape[0]
                    if n_pos < 3:
                        spatial_p = 1.0
                    else:
                        def mean_nn_dist(xy):
                            tree = cKDTree(xy)
                            dists, _ = tree.query(xy, k=2)
                            return float(dists[:, 1].mean())

                        obs_nn = mean_nn_dist(coords_pos)
                        n_cells_cluster = coords.shape[0]
                        perm_nn = []
                        for _ in range(n_perm_spatial):
                            idx = np.random.choice(n_cells_cluster, size=n_pos, replace=False)
                            perm_xy = coords[idx]
                            perm_nn.append(mean_nn_dist(perm_xy))
                        perm_nn = np.array(perm_nn)
                        spatial_p = ((perm_nn <= obs_nn).sum() + 1) / (len(perm_nn) + 1)
            except Exception:
                spatial_p = 1.0
        else:
            spatial_p = 1.0
        spatial_p = np.clip(spatial_p, 1e-300, 1.0)
        spatial_co_p_list.append(spatial_p)

    emp_fdr = multipletests(emp_p_list, method="fdr_bh")[1]
    de_fdr = multipletests(de_p_list, method="fdr_bh")[1]
    z_fdr = multipletests(z_p_list, method="fdr_bh")[1]
    topn_overlap_fdr = multipletests(topn_overlap_p_list, method="fdr_bh")[1]
    threshold_overlap_fdr = multipletests(threshold_overlap_p_list, method="fdr_bh")[1]
    fisher_fdr = multipletests(fisher_p_list, method="fdr_bh")[1]
    prop_fdr = multipletests(prop_p_list, method="fdr_bh")[1]
    corr_fdr = multipletests(corr_p_list, method="fdr_bh")[1]
    spatial_co_fdr = multipletests(spatial_co_p_list, method="fdr_bh")[1]

    # v3: explicit clipping. `locals()[name] = ...` is a no-op inside a
    # function and silently left the FDR arrays un-clipped in v2.
    emp_fdr = np.clip(emp_fdr, 1e-300, 1.0)
    de_fdr = np.clip(de_fdr, 1e-300, 1.0)
    z_fdr = np.clip(z_fdr, 1e-300, 1.0)
    topn_overlap_fdr = np.clip(topn_overlap_fdr, 1e-300, 1.0)
    threshold_overlap_fdr = np.clip(threshold_overlap_fdr, 1e-300, 1.0)
    fisher_fdr = np.clip(fisher_fdr, 1e-300, 1.0)
    prop_fdr = np.clip(prop_fdr, 1e-300, 1.0)
    corr_fdr = np.clip(corr_fdr, 1e-300, 1.0)
    spatial_co_fdr = np.clip(spatial_co_fdr, 1e-300, 1.0)

    def kurtosis_score_from_fdr(fdr_arr):
        fdr_arr = np.clip(fdr_arr, 1e-300, 1.0)
        logs = -np.log10(fdr_arr)
        if np.allclose(logs, logs[0]):
            return 1.0
        return float(kurtosis(logs, fisher=False))

    core_methods = [m for m in CORE_FDR_METHODS if method_switch.get(m, True)]

    fdr_dict = {
        "emp_fdr": emp_fdr,
        "topn_overlap_fdr": topn_overlap_fdr,
        "threshold_overlap_fdr": threshold_overlap_fdr,
        "de_fdr": de_fdr,
        "z_fdr": z_fdr,
        "fisher_fdr": fisher_fdr,
        "prop_fdr": prop_fdr,
        "corr_fdr": corr_fdr,
        "spatial_co_fdr": spatial_co_fdr,
    }

    kurt_scores = {m: kurtosis_score_from_fdr(fdr_dict[m]) for m in core_methods}
    raw_weights = np.array([soft_weight(kurt_scores[m]) for m in core_methods])
    raw_weights = raw_weights / raw_weights.sum()
    weights = {m: w for m, w in zip(core_methods, raw_weights)}

    ranks = {m: rankdata(fdr_dict[m], method="min") for m in core_methods}
    weighted_rank_sum = np.zeros_like(emp_fdr, dtype=float)
    for m in core_methods:
        weighted_rank_sum += ranks[m] * weights[m]

    result_df = pd.DataFrame(
        {
            "cluster": cluster_id,
            "cell_subtype": cell_subtypes,
            "empirical_p": emp_p_list,
            "de_p": de_p_list,
            "z_p": z_p_list,
            "topn_overlap_p": topn_overlap_p_list,
            "threshold_overlap_p": threshold_overlap_p_list,
            "fisher_p": fisher_p_list,
            "prop_p": prop_p_list,
            "corr_p": corr_p_list,
            "spatial_co_p": spatial_co_p_list,
            "emp_fdr": emp_fdr,
            "de_fdr": de_fdr,
            "z_fdr": z_fdr,
            "topn_overlap_fdr": topn_overlap_fdr,
            "threshold_overlap_fdr": threshold_overlap_fdr,
            "fisher_fdr": fisher_fdr,
            "prop_fdr": prop_fdr,
            "corr_fdr": corr_fdr,
            "spatial_co_fdr": spatial_co_fdr,
            "weighted_rank_sum": weighted_rank_sum,
            "marker_coverage": marker_coverage,
            "n_topn_overlap_genes": n_topn_overlap_list,
            "n_threshold_overlap_genes": n_threshold_overlap_list,
        }
    )

    for m in CORE_FDR_METHODS:
        col = m + "_weight"
        if m in core_methods:
            result_df[col] = weights[m]
        else:
            result_df[col] = 0.0

    min_score = result_df["weighted_rank_sum"].min()
    tie_df = result_df[result_df["weighted_rank_sum"] == min_score].copy()
    # v3: restrict tie-break priority to methods that are (a) enabled in
    # method_switch and (b) actually present as columns in result_df.
    # Disabled methods have FDR==1.0 everywhere and cannot discriminate.
    active_priority = [
        k for k in tie_break_priority
        if method_switch.get(k, True) and k in tie_df.columns
    ]
    if not active_priority:
        active_priority = [k for k in tie_break_priority if k in tie_df.columns]
    tie_df = tie_df.sort_values(by=active_priority, ascending=True)
    top_fdr = tie_df.iloc[0][active_priority]
    final_tie = tie_df[np.logical_and.reduce([tie_df[k] == top_fdr[k] for k in active_priority])]

    chosen = final_tie.iloc[0]["cell_subtype"]
    tied = [ct for ct in final_tie["cell_subtype"] if ct != chosen]

    result_df["assigned_cell_subtype"] = chosen
    result_df["assigned_cell_major_type"] = all_major_types[chosen]
    result_df["assigned_cell_pannuke_label"] = all_pannuke_labels[chosen]
    result_df["assigned_cell_hne_type"] = all_hne_types[chosen]
    result_df["assigned_cell_hne_label"] = all_hne_labels[chosen]
    result_df["assigned_cell_pantissue_type"] = all_pantissue_types[chosen]
    result_df["assigned_cell_pantissue_label"] = all_pantissue_labels[chosen]
    result_df["other_tied_cell_subtypes"] = "/".join([ct for ct in tied if ct != chosen])
    result_df["cancer_associated"] = all_malignant_indicators[chosen]

    return result_df


def run_kurtorank(
    adata: ad.AnnData,
    primary_cluster: str,
    markers_csv: Path,
    tissue_type: str,
    common_only: bool,
    normal_only: bool,
    method_switch: Mapping[str, bool],
    n_perm: int,
    n_jobs: int,
    generate_plots: bool,
    out_dir: Path,
    use_top_k_markers: Optional[int] = None,
    allow_cuda_parallel: bool = False,
):
    method_switch = dict(method_switch)
    logger.info(f"Reading markers from: {markers_csv}")
    all_markers_df = pd.read_csv(markers_csv)

    all_markers_df = all_markers_df[
        (all_markers_df.tissue_type == tissue_type)
        | (all_markers_df.tissue_type == "immune")
        | (all_markers_df.tissue_type == "circulating")
    ]

    if common_only:
        all_markers_df = all_markers_df[all_markers_df.common == True]
    if normal_only:
        all_markers_df = all_markers_df[all_markers_df.malignant == False]

    all_markers_df.reset_index(drop=True, inplace=True)

    required_cols = {"hne_type", "hne_label", "pannuke_label", "major_type"}
    missing_cols = required_cols.difference(all_markers_df.columns)
    if missing_cols:
        raise ValueError(
            "Markers CSV is missing required columns for H&E/PanNuke labeling: "
            + ", ".join(sorted(missing_cols))
        )

    cancer_markers = all_markers_df[all_markers_df.malignant == True]

    all_markers = all_markers_df.set_index("subtype")["markers"].apply(lambda x: x.split(",")).to_dict()
    # v3: optionally truncate each subtype's marker list to top-K genes.
    # markers-v3.csv stores genes in atlas-derived specificity order, so
    # the first K are the most discriminative per Census scoring.
    if use_top_k_markers is not None and use_top_k_markers > 0:
        k = int(use_top_k_markers)
        all_markers = {
            st: [g for g in genes if g][:k]
            for st, genes in all_markers.items()
        }
        logger.info("Truncated each subtype's marker list to top %d genes.", k)
    all_major_types = all_markers_df.set_index("subtype")["major_type"].to_dict()
    all_pannuke_labels = all_markers_df.set_index("subtype")["pannuke_label"].to_dict()
    all_hne_types = all_markers_df.set_index("subtype")["hne_type"].to_dict()
    all_hne_labels = all_markers_df.set_index("subtype")["hne_label"].to_dict()
    # v3_2: pantissue_{type,label} columns are optional. If absent (older marker
    # CSVs), fall back to hne_type/hne_label so downstream code paths still work.
    if "pantissue_label" in all_markers_df.columns:
        all_pantissue_labels = all_markers_df.set_index("subtype")["pantissue_label"].to_dict()
    else:
        all_pantissue_labels = dict(all_hne_labels)
    if "pantissue_type" in all_markers_df.columns:
        all_pantissue_types = all_markers_df.set_index("subtype")["pantissue_type"].to_dict()
    else:
        all_pantissue_types = dict(all_hne_types)
    all_malignant_indicators = all_markers_df.set_index("subtype")["malignant"].astype(bool).to_dict()

    marker_genes = list({g for m in all_markers.values() for g in m})
    available_markers = [gene for gene in marker_genes if gene in adata.var_names]

    if len(available_markers) > 0:
        logger.info("Computing Moran's I for marker genes.")
        if "spatial_connectivities" not in adata.obsp:
            logger.info("Building spatial neighbor graph for Xenium coordinates.")
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",      # for Xenium coordinates in adata.obsm["spatial"]
                spatial_key="spatial",
                key_added="spatial",       # will create 'spatial_connectivities' & 'spatial_distances'
            )

         # 1) Ensure obs[primary_cluster] is string and categorical with numeric order
        clusters_as_str = adata.obs[primary_cluster].astype(str)

        cluster_order = sorted(pd.unique(clusters_as_str), key=cluster_sort_key)
        adata.obs[primary_cluster] = pd.Categorical(
            clusters_as_str,
            categories=cluster_order,
            ordered=True,
        )
        logger.info(f"Cluster order for neighborhood enrichment: {cluster_order}")

        # 2) Neighborhood enrichment using the categorical ordering
        sq.gr.nhood_enrichment(adata, cluster_key=primary_cluster)

        # 3) Plot with consistent cluster order and save to disk when requested
        if generate_plots:
            ax = sq.pl.nhood_enrichment(
                adata,
                cluster_key=primary_cluster,
                figsize=(5, 5),
                show=False,
            )
            fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
            save_fig(fig, out_dir, "nhood_enrichment.png")

        # now Moran's I
        sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            genes=available_markers,
        )

        if generate_plots and "moranI" in adata.uns:
            moran_scores = adata.uns["moranI"]
            top_genes = moran_scores.sort_values("I", ascending=False).head(5)
            top_gene_names = top_genes.index.tolist()
            fig, axes = plt.subplots(1, len(top_gene_names), figsize=(5 * len(top_gene_names), 5))
            if len(top_gene_names) == 1:
                axes = [axes]
            for i, gene in enumerate(tqdm(top_gene_names, dynamic_ncols=True)):
                sc.pl.spatial(
                    adata,
                    color=gene,
                    size=10,
                    img_key=None,
                    spot_size=1,
                    show=False,
                    ax=axes[i],
                )
            plt.tight_layout()
            save_fig(fig, out_dir, "moran_top_markers.png")

    filtered_markers = {}
    sum_marker_count = 0
    logger.info("Filtering marker genes by presence in adata.")
    for i, (cell_subtype, markers) in enumerate(all_markers.items()):
        existing = [g for g in markers if g in adata.var_names]
        if len(existing) >= 2:
            filtered_markers[cell_subtype] = existing
            sum_marker_count += len(existing)
            logger.info(f"{i+1}/{len(all_markers)} - {cell_subtype}: {len(existing)}/{len(markers)} markers found")
        else:
            logger.info(f"{i+1}/{len(all_markers)} - Not enough markers for {cell_subtype}.")
    if len(filtered_markers) == 0:
        raise ValueError("No marker sets with >=2 genes found in dataset.")

    avg_marker_num = sum_marker_count / max(len(all_markers), 1)
    logger.info(f"Average marker number after filtering: {avg_marker_num:.2f}")
    all_markers = filtered_markers
    cell_subtypes = list(all_markers.keys())

    # --- v3 pre-flight sanity checks (marker redundancy + GPU/joblib safety) ---
    # 1. Pairwise Jaccard of marker sets restricted to the panel. Pairs with
    #    high overlap are intrinsically hard for KurtoRank to separate; we
    #    flag them so users can revisit their marker lists.
    try:
        import itertools as _itertools
        panel_genes = set(adata.var_names)
        sets_in_panel = {
            ct: set(all_markers[ct]) & panel_genes
            for ct in all_markers
            if len(set(all_markers[ct]) & panel_genes) >= 3
        }
        jacs = []
        for a, b in _itertools.combinations(sets_in_panel.keys(), 2):
            sa, sb = sets_in_panel[a], sets_in_panel[b]
            u = len(sa | sb)
            jacs.append((a, b, (len(sa & sb) / u) if u else 0.0))
        if jacs:
            j_df = pd.DataFrame(jacs, columns=["subtype_a", "subtype_b", "jaccard"])
            high = j_df[j_df.jaccard >= 0.4].sort_values("jaccard", ascending=False)
            logger.info(
                f"Subtype pairs with Jaccard >= 0.4 (after restricting to panel): "
                f"{len(high)}"
            )
            if len(high):
                for row in high.head(15).itertuples(index=False):
                    logger.info(
                        f"  {row.subtype_a}  <->  {row.subtype_b}  "
                        f"(jaccard={row.jaccard:.2f})"
                    )
                logger.info(
                    "  -> These pairs are statistically hard to separate by KurtoRank."
                )
            adata.uns["marker_jaccard_high"] = high.to_dict(orient="list")
    except Exception as _exc:  # pragma: no cover — diagnostic only
        logger.warning(f"Jaccard redundancy check failed: {_exc}")

    # 2. Joblib / CUDA safety. `torch_empirical_p` allocates on the
    #    least-used GPU; running it under multiple workers can trigger
    #    CUDA context duplication and OOM. Cap to 1 unless the user
    #    explicitly opts in via `--allow-cuda-parallel`.
    if (
        method_switch.get("emp_fdr", True)
        and torch.cuda.is_available()
        and n_jobs > 1
        and not allow_cuda_parallel
    ):
        logger.warning(
            f"v3 safety: CUDA is visible and emp_fdr is ON; capping n_jobs "
            f"from {n_jobs} to 1 to avoid GPU context contention. "
            f"Pass --allow-cuda-parallel to override."
        )
        n_jobs = 1

    cluster_counts = adata.obs[primary_cluster].value_counts()
    min_cluster_size = 50
    small_clusters = cluster_counts[cluster_counts < min_cluster_size].index
    logger.info(f"Removing {len(small_clusters)} small clusters (min size {min_cluster_size}).")
    keep_mask = ~adata.obs[primary_cluster].isin(small_clusters)
    adata._inplace_subset_obs(keep_mask)

    seed = 1234
    de_logfc_threshold = 1.5
    de_pval_threshold = 0.001
    tie_break_priority = [
        "emp_fdr",
        "topn_overlap_fdr",
        "de_fdr",
        "z_fdr",
        "fisher_fdr",
        "prop_fdr",
        "corr_fdr",
        "spatial_co_fdr",
        "threshold_overlap_fdr",
    ]
    top_n_de_genes = 50
    background_gene_count = len(adata.var_names)

    logger.info("Running rank_genes_groups (DE) once for all clusters.")
    sc.tl.rank_genes_groups(
        adata,
        groupby=primary_cluster,
        method="wilcoxon",
        use_raw=False,
        pts=True,
    )

    active_methods = [m for m, enabled in method_switch.items() if enabled]
    if not active_methods:
        raise ValueError("At least one annotation method must be enabled.")
    logger.info("Active FDR methods: %s", ", ".join(active_methods))
    adata.uns["method_switch"] = method_switch.copy()

    cluster_labels = adata.obs[primary_cluster].unique()
    var_names_all = np.array(adata.var_names)

    context = {
        "adata": adata,
        "primary_cluster": primary_cluster,
        "cell_subtypes": cell_subtypes,
        "all_markers": all_markers,
        "method_switch": method_switch,
        "n_perm": n_perm,
        "seed": seed,
        "top_n_de_genes": top_n_de_genes,
        "de_pval_threshold": de_pval_threshold,
        "de_logfc_threshold": de_logfc_threshold,
        "background_gene_count": background_gene_count,
        "all_major_types": all_major_types,
        "all_pannuke_labels": all_pannuke_labels,
        "all_hne_types": all_hne_types,
        "all_hne_labels": all_hne_labels,
        "all_pantissue_labels": all_pantissue_labels,
        "all_pantissue_types": all_pantissue_types,
        "all_malignant_indicators": all_malignant_indicators,
        "tie_break_priority": tie_break_priority,
        "var_names_all": var_names_all,
    }

    def run_all_clusters(cluster_labels, n_jobs):
        # Share immutable state via a global context so both serial and parallel paths stay in sync.
        cluster_list = list(cluster_labels)
        if n_jobs is None or n_jobs < 1:
            n_jobs = 1

        if n_jobs == 1:
            results = []
            previous_context = _CLUSTER_CONTEXT
            _init_cluster_context(context)
            try:
                for cl in tqdm(cluster_list, desc="Predicting cell subtypes", dynamic_ncols=True):
                    results.append(_process_cluster_worker(cl))
            finally:
                _init_cluster_context(previous_context)
            return results

        results = []
        previous_context = _CLUSTER_CONTEXT
        _init_cluster_context(context)
        try:
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_cluster_context,
                initargs=(context,),
            ) as executor:
                future_to_cluster = {
                    executor.submit(_process_cluster_worker, cl): cl for cl in cluster_list
                }
                for future in tqdm(
                    as_completed(future_to_cluster),
                    total=len(future_to_cluster),
                    desc="Predicting cell subtypes",
                    dynamic_ncols=True,
                ):
                    cl = future_to_cluster[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.error(f"KurtoRank worker failed for cluster {cl}: {exc}")
                        raise
        finally:
            _init_cluster_context(previous_context)
        return results

    logger.info(f"Running KurtoRank per cluster with n_jobs={n_jobs}")
    results = run_all_clusters(cluster_labels, n_jobs)
    final_df = pd.concat(results, ignore_index=True)

    cluster_to_celltype = final_df.groupby("cluster")["assigned_cell_subtype"].first().to_dict()
    adata.obs["cell_subtype"] = adata.obs[primary_cluster].map(cluster_to_celltype)

    cluster_to_major_type = final_df.groupby("cluster")["assigned_cell_major_type"].first().to_dict()
    adata.obs["cell_major_type"] = adata.obs[primary_cluster].map(cluster_to_major_type)

    cluster_to_pannuke_label = final_df.groupby("cluster")["assigned_cell_pannuke_label"].first().to_dict()
    adata.obs["cell_pannuke_label"] = adata.obs[primary_cluster].map(cluster_to_pannuke_label)

    cluster_to_hne_type = final_df.groupby("cluster")["assigned_cell_hne_type"].first().to_dict()
    adata.obs["cell_hne_type"] = adata.obs[primary_cluster].map(cluster_to_hne_type)

    cluster_to_hne_label = final_df.groupby("cluster")["assigned_cell_hne_label"].first().to_dict()
    adata.obs["cell_hne_label"] = adata.obs[primary_cluster].map(cluster_to_hne_label)

    cluster_to_pantissue_type = final_df.groupby("cluster")["assigned_cell_pantissue_type"].first().to_dict()
    adata.obs["cell_pantissue_type"] = adata.obs[primary_cluster].map(cluster_to_pantissue_type)

    cluster_to_pantissue_label = final_df.groupby("cluster")["assigned_cell_pantissue_label"].first().to_dict()
    adata.obs["cell_pantissue_label"] = adata.obs[primary_cluster].map(cluster_to_pantissue_label)

    cancer_associated = final_df.groupby("cluster")["cancer_associated"].first().to_dict()
    adata.obs["cancer_associated"] = adata.obs[primary_cluster].map(cancer_associated)

    chosen_map = final_df.groupby("cluster")["assigned_cell_subtype"].first()
    rows = []
    for cluster, ct in chosen_map.items():
        match = final_df[(final_df["cluster"] == cluster) & (final_df["cell_subtype"] == ct)]
        if not match.empty:
            rows.append(match.iloc[0])
    summary_df = pd.DataFrame(rows)
    adata.uns["cell_subtype_prediction_summary"] = summary_df
    # Persist the full per-cluster result table for downstream visualizations/exports.
    adata.uns["kurtorank_results"] = final_df

    logger.info("KurtoRank annotation completed.")
    return adata, final_df


# ---------------------- Visualization / exports ----------------------


def visualize_annotation(
    adata: ad.AnnData,
    primary_cluster: str,
    generate_plots: bool,
    out_dir: Path,
    method_switch: Optional[Mapping[str, bool]] = None,
):
    """Create CLI-friendly diagnostic plots for KurtoRank annotations."""
    if not generate_plots:
        return

    if "cell_hne_combo" in adata.obs.columns:
        logger.debug("Dropping legacy cell_hne_combo column from annotated data.")
        adata.obs.drop(columns=["cell_hne_combo"], inplace=True)

    # Core FDR profiles for each active annotation method across clusters.


    summary_df = adata.uns.get("cell_subtype_prediction_summary")
    final_df = adata.uns.get("kurtorank_results")
    if method_switch is None:
        method_switch = dict(adata.uns.get("method_switch", DEFAULT_METHOD_SWITCH))
    else:
        method_switch = dict(method_switch)

    active_fdr_cols: list[str] = []
    if summary_df is None or summary_df.empty:
        logger.warning("No cell subtype prediction summary available; skipping advanced annotation plots.")
        summary_df = None
    else:
        summary_df = summary_df.copy()
        summary_df["cluster"] = summary_df["cluster"].astype(str)
        available_methods = [m for m in CORE_FDR_METHODS if m in summary_df.columns]
        active_fdr_cols = [m for m in available_methods if method_switch.get(m, True)]
        if not active_fdr_cols:
            active_fdr_cols = available_methods

    def _ordered_clusters(df: Optional[pd.DataFrame]) -> list[str]:
        """Prefer the AnnData cluster order; fall back to numeric ordering."""
        if df is None or df.empty:
            return []
        cluster_series = adata.obs[primary_cluster]
        if pd.api.types.is_categorical_dtype(cluster_series):
            preferred = [str(c) for c in cluster_series.cat.categories]
        else:
            preferred = sorted(cluster_series.astype(str).unique(), key=cluster_sort_key)
        ordered = [c for c in preferred if c in df["cluster"].values]
        if not ordered:
            ordered = sorted(df["cluster"].unique(), key=cluster_sort_key)
        return ordered

    cluster_order: list[str] = _ordered_clusters(summary_df) if summary_df is not None else []

    if summary_df is not None and active_fdr_cols and cluster_order:
        clusters = cluster_order
        fig_cols = min(3, len(active_fdr_cols)) or 1
        fig_rows = math.ceil(len(active_fdr_cols) / fig_cols)
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(5 * fig_cols, 4 * fig_rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        x_pos = np.arange(len(clusters))
        logger.info("Plotting FDR overview per cluster.")
        for idx, col in enumerate(active_fdr_cols):
            ax = axes[idx]
            vals = [
                summary_df.loc[summary_df["cluster"] == cluster_label, col].mean()
                for cluster_label in clusters
            ]
            ax.bar(x_pos, vals, color="steelblue")
            ax.set_title(col.replace("_", " ").title())
            ax.set_xlabel("Cluster")
            ax.set_ylabel("FDR")
            ax.axhline(0.05, color="red", linestyle="--", alpha=0.7)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(clusters, rotation=0)

        for ax in axes[len(active_fdr_cols):]:
            ax.set_visible(False)

        fig.tight_layout()
        save_fig(fig, out_dir, "fdr_overview.png")
    elif summary_df is not None:
        logger.warning("No active FDR columns available for plotting.")

    if summary_df is not None and {"weighted_rank_sum", "marker_coverage", "assigned_cell_subtype"}.issubset(summary_df.columns):
        clusters_np = summary_df["cluster"].astype(str).values
        subtypes_np = summary_df["assigned_cell_subtype"].astype(str).values

        wrs_values = summary_df["weighted_rank_sum"].to_numpy(dtype=float)
        wrs_order = np.argsort(-wrs_values)
        wrs_sorted = wrs_values[wrs_order]
        clusters_wrs = clusters_np[wrs_order]
        subtypes_wrs = subtypes_np[wrs_order]

        mc_values = summary_df["marker_coverage"].to_numpy(dtype=float)
        mc_order = np.argsort(mc_values)
        mc_sorted = mc_values[mc_order]
        clusters_mc = clusters_np[mc_order]
        subtypes_mc = subtypes_np[mc_order]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].barh(clusters_wrs, wrs_sorted)
        axes[0].set_title("Weighted Rank Sum by Cluster")
        axes[0].set_xlabel("Weighted Rank (lower = better)")
        axes[0].set_ylabel("Cluster")
        if len(wrs_sorted):
            xmax_wrs = wrs_sorted.max() * 1.2
            axes[0].set_xlim(0, xmax_wrs)
            pad = wrs_sorted.max() * 0.02
            for y, v, st in zip(clusters_wrs, wrs_sorted, subtypes_wrs):
                axes[0].text(v + pad, y, st, va="center", ha="left", fontsize=8)

        axes[1].barh(clusters_mc, mc_sorted)
        axes[1].set_title("Marker Coverage by Cluster")
        axes[1].set_xlabel("Coverage (higher = better)")
        axes[1].set_ylabel("Cluster")
        if len(mc_sorted):
            xmax_mc = mc_sorted.max() * 1.2
            axes[1].set_xlim(0, xmax_mc)
            pad2 = mc_sorted.max() * 0.02
            for y, v, st in zip(clusters_mc, mc_sorted, subtypes_mc):
                axes[1].text(v + pad2, y, st, va="center", ha="left", fontsize=8)

        # Normalize weighted_rank_sum to [0, 1] and flip so larger = better.
        wrs_raw = summary_df["weighted_rank_sum"].to_numpy(dtype=float)
        wrs_min = float(wrs_raw.min()) if len(wrs_raw) else 0.0
        wrs_max = float(wrs_raw.max()) if len(wrs_raw) else 1.0
        if wrs_max > wrs_min:
            wrs_norm = (wrs_raw - wrs_min) / (wrs_max - wrs_min)
        else:
            wrs_norm = np.zeros_like(wrs_raw)
        scatter_x = 1.0 - wrs_norm

        scatter_y = summary_df["marker_coverage"].to_numpy(dtype=float)
        clusters_sc = summary_df["cluster"].astype(str).to_numpy()
        subtypes_sc = summary_df["assigned_cell_subtype"].astype(str).to_numpy()

        # Scatter using a normalized rank-quality score on x (higher = better)
        # and marker coverage on y, plus a diagonal reference line.
        # Use a single uniform color, moderate marker size, and 50% transparency.
        axes[2].scatter(
            scatter_x,
            scatter_y,
            c="tab:blue",
            s=150,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.5,
        )
        axes[2].set_title("Rank vs Coverage Scatter")
        axes[2].set_xlabel("1 - Normalized Weighted Rank Sum (higher = better)")
        axes[2].set_ylabel("Marker Coverage")
        axes[2].grid(alpha=0.3)
        axes[2].set_xlim(-0.05, 1.05)
        axes[2].set_ylim(-0.05, 1.05)

        # Draw a red dashed diagonal from (0, 0) to (1, 1).
        axes[2].plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="red", linewidth=1.0, alpha=0.8)

        # Place cluster IDs directly on top of each dot so the marker acts as
        # a background for the text.
        for x, y, cl in zip(scatter_x, scatter_y, clusters_sc):
            axes[2].text(
                x,
                y,
                str(cl),
                fontsize=7,
                ha="center",
                va="center",
                color="white",
            )

        # Build mapping text listing each cluster and its corresponding subtype,
        # sorted by numeric-aware cluster order (1, 2, 3, 10, ... becomes
        # 1, 2, 3, 4, ... when clusters are numeric strings).
        cluster_subtype_pairs = list(zip(clusters_sc, subtypes_sc))
        cluster_subtype_pairs.sort(key=lambda cs: cluster_sort_key(cs[0]))
        mapping_lines = [
            f"{cl}: {st}" for cl, st in cluster_subtype_pairs
        ]
        mapping_str = "\n".join(mapping_lines)
        axes[2].text(
            1.02,
            1.0,
            mapping_str,
            transform=axes[2].transAxes,
            fontsize=7,
            ha="left",
            va="top",
        )

        fig.tight_layout()
        save_fig(fig, out_dir, "rank_marker_overview.png")

    # Confidence heuristics turn per-method FDR tables into comparable 0-1 scores.
    def calculate_unweighted_confidence(df: pd.DataFrame, fdr_cols: list[str]) -> np.ndarray:
        fdr_matrix = df[fdr_cols].to_numpy(dtype=float)
        med = np.median(fdr_matrix, axis=1)
        abs_conf = 1.0 - np.clip(med, 0.0, 1.0)
        ranges = fdr_matrix.max(axis=1) - fdr_matrix.min(axis=1)
        max_range = ranges.max() if ranges.max() > 0 else 1.0
        agr_conf = 1.0 - (ranges / max_range)
        return np.clip(abs_conf * agr_conf, 0.0, 1.0)

    def calculate_weighted_confidence(df: pd.DataFrame, fdr_cols: list[str]) -> np.ndarray:
        scores = []
        for _, row in df.iterrows():
            fdr_vals = np.array([row[c] for c in fdr_cols], dtype=float)
            weights = []
            for c in fdr_cols:
                w_col = c + "_weight"
                weights.append(row[w_col] if w_col in df.columns else 0.0)
            weights = np.array(weights, dtype=float)
            if np.allclose(weights, 0):
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            mean_fdr = float(np.sum(fdr_vals * weights))
            abs_conf = 1.0 - np.clip(mean_fdr, 0.0, 1.0)
            var = float(np.sum(weights * (fdr_vals - mean_fdr) ** 2))
            std = np.sqrt(var)
            std_upper = 0.5
            agr_conf = 1.0 - min(std, std_upper) / std_upper
            scores.append(max(0.0, min(1.0, abs_conf * agr_conf)))
        return np.array(scores)

    if summary_df is not None and active_fdr_cols and cluster_order:
        summary_df_viz = summary_df.set_index("cluster").loc[cluster_order].reset_index()
        summary_df_viz["unweighted_confidence"] = calculate_unweighted_confidence(summary_df_viz, active_fdr_cols)
        summary_df_viz["weighted_confidence"] = calculate_weighted_confidence(summary_df_viz, active_fdr_cols)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        x_pos = np.arange(len(cluster_order))
        width = 0.35
        axes[0].bar(x_pos - width / 2, summary_df_viz["unweighted_confidence"], width, label="Unweighted", alpha=0.7)
        axes[0].bar(x_pos + width / 2, summary_df_viz["weighted_confidence"], width, label="Weighted", alpha=0.7)
        axes[0].set_title("Consensus Score Comparison")
        axes[0].set_xlabel("Cluster")
        axes[0].set_ylabel("Consensus Score")
        axes[0].set_xticks(x_pos, cluster_order, rotation=0)
        axes[0].set_ylim(0.0, 1.05)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

        perf = pd.DataFrame({"cluster": cluster_order})
        for m in active_fdr_cols:
            conf = 1.0 - summary_df_viz[m].values.astype(float)
            weights_series = summary_df_viz.get(m + "_weight")
            weights = weights_series.to_numpy(dtype=float) if weights_series is not None else np.zeros_like(conf)
            perf[METHOD_LABELS.get(m, m)] = conf * weights

        pivot = perf.set_index("cluster")
        pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab10")
        axes[1].set_title("Weighted Confidence Contributions by Method")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Weighted Confidence")
        axes[1].set_xticklabels(cluster_order, rotation=0)
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

        fig.tight_layout()
        save_fig(fig, out_dir, "consensus_scores.png")

    if summary_df is not None and active_fdr_cols and cluster_order:
        heatmap_data = summary_df.set_index("cluster").loc[cluster_order, active_fdr_cols].astype(float)
        heatmap_data = heatmap_data.rename(columns=lambda m: METHOD_LABELS.get(m, m))
        fig, ax = plt.subplots(figsize=(9, 9))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2e",
            cmap="YlGnBu",
            cbar_kws={"label": "FDR"},
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("FDR Heatmap: Active Methods")
        ax.set_ylabel("Cluster")
        ax.set_xlabel("Method")
        fig.tight_layout()
        save_fig(fig, out_dir, "fdr_heatmap.png")

        logger.info("Annotated clusters: %d", len(summary_df))
        logger.info("Unique cell subtypes: %d", summary_df["assigned_cell_subtype"].nunique())
        for m in active_fdr_cols:
            mean_val = summary_df[m].mean()
            median_val = summary_df[m].median()
            min_val = summary_df[m].min()
            sig = int((summary_df[m] < 0.05).sum())
            logger.info(
                "%s — mean: %.4f | median: %.4f | min: %.4f | clusters <0.05: %d",
                METHOD_LABELS.get(m, m),
                mean_val,
                median_val,
                min_val,
                sig,
            )

    # --- Cross-method correlation heatmap ---
    if summary_df is not None and len(active_fdr_cols) > 1:
        corr_mat = np.zeros((len(active_fdr_cols), len(active_fdr_cols)))
        for i, m1 in enumerate(active_fdr_cols):
            for j, m2 in enumerate(active_fdr_cols):
                x = summary_df[m1].values.astype(float)
                y = summary_df[m2].values.astype(float)
                rho, _ = spearmanr(x, y)
                corr_mat[i, j] = rho

        corr_df = pd.DataFrame(
            corr_mat,
            index=[METHOD_LABELS.get(m, m) for m in active_fdr_cols],
            columns=[METHOD_LABELS.get(m, m) for m in active_fdr_cols],
        )
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            corr_df,
            annot=True,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            cbar_kws={"label": "Spearman correlation (FDR across clusters)"},
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_title("Correlation Between Methods")
        ax.set_ylabel("Method")
        ax.set_xlabel("Method")
        fig.tight_layout()
        save_fig(fig, out_dir, "method_correlation_heatmap.png")

    # --- Agreement vs support summary ---
    enabled_methods = [m for m in CORE_FDR_METHODS if summary_df is not None and method_switch.get(m, True) and m in summary_df.columns]
    if summary_df is not None and final_df is not None and enabled_methods:
        ensemble_map = summary_df.set_index("cluster")["assigned_cell_subtype"].to_dict()
        clusters_sorted = sorted(ensemble_map.keys(), key=cluster_sort_key)

        agree_matrix = pd.DataFrame(index=clusters_sorted, columns=enabled_methods, dtype=float)
        rank_matrix = pd.DataFrame(index=clusters_sorted, columns=enabled_methods, dtype=float)

        working_df = final_df.copy()
        working_df["cluster"] = working_df["cluster"].astype(str)

        for m in enabled_methods:
            sub = working_df[["cluster", "cell_subtype", m]].copy()
            sub = sub[sub["cluster"].isin(clusters_sorted)]
            for cl in clusters_sorted:
                ens_ct = ensemble_map[cl]
                sub_cl = sub[sub["cluster"] == cl]
                if sub_cl.empty:
                    agree_matrix.loc[cl, m] = np.nan
                    rank_matrix.loc[cl, m] = np.nan
                    continue
                sub_cl = sub_cl.sort_values(by=m, ascending=True, kind="mergesort")
                ordered_subtypes = list(sub_cl["cell_subtype"])
                method_best = ordered_subtypes[0]
                agree_matrix.loc[cl, m] = 1.0 if method_best == ens_ct else 0.0
                rank_matrix.loc[cl, m] = ordered_subtypes.index(ens_ct) + 1 if ens_ct in ordered_subtypes else np.nan

        method_agree_rate = agree_matrix.mean(axis=0)
        agree_hm = agree_matrix.rename(columns=lambda m: METHOD_LABELS.get(m, m))
        rank_hm = rank_matrix.rename(columns=lambda m: METHOD_LABELS.get(m, m))

        support_counts = []
        for _, row in summary_df.iterrows():
            vals = [row[m] for m in enabled_methods if m in row]
            support_counts.append(int(np.sum(np.array(vals) < 0.05)))

        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        x_labels = [METHOD_LABELS.get(m, m) for m in enabled_methods]
        axes[0].bar(range(len(enabled_methods)), method_agree_rate.values)
        axes[0].set_xticks(range(len(enabled_methods)))
        axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
        axes[0].set_ylabel("Agreement Rate with KurtoRank")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Method vs Ensemble Agreement")

        # Use rank (position of the KurtoRank-chosen subtype in each
        # method's own ranking) as the heat level. Lower ranks (1 = best)
        # are shown as darker cells. Rank 1 cells are additionally
        # highlighted with an outline.
        rank_plot = rank_hm.astype(float)
        # Determine color scale bounds from observed ranks.
        if np.isfinite(rank_plot.values).any():
            vmin = 1.0
            vmax = np.nanmax(rank_plot.values)
        else:
            vmin, vmax = 0.0, 1.0

        hm = sns.heatmap(
            rank_plot,
            annot=rank_hm,
            fmt=".0f",
            cmap="Greens",
            vmin=vmin,
            vmax=vmax,
            cbar=True,
            linewidths=0.5,
            ax=axes[1],
        )

        # Outline cells where the rank is 1 (method's top call matches the
        # KurtoRank choice).
        for i, cl in enumerate(rank_plot.index):
            for j, m in enumerate(rank_plot.columns):
                val = rank_plot.loc[cl, m]
                if np.isfinite(val) and int(val) == 1:
                    axes[1].add_patch(
                        plt.Rectangle(
                            (j, i),
                            1,
                            1,
                            fill=False,
                            edgecolor="red",
                            lw=1.2,
                            clip_on=False,
                        )
                    )
        axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
        axes[1].set_title("Cluster-level Rank of KurtoRank Choice")
        axes[1].set_ylabel("Cluster")
        axes[1].set_xlabel("Method")

        axes[2].hist(support_counts, bins=range(0, len(enabled_methods) + 2), align="left", rwidth=0.8)
        axes[2].set_xlabel("# Methods with FDR < 0.05 (support)")
        axes[2].set_ylabel("Number of Clusters")
        axes[2].set_title("Support Strength for Ensemble Choice")
        axes[2].set_xticks(range(0, len(enabled_methods) + 1))

        fig.tight_layout()
        save_fig(fig, out_dir, "agreement_support_summary.png")
    elif summary_df is not None and final_df is None:
        logger.warning("Full KurtoRank results missing; skipping agreement/support summary plots.")

    # # --- Method entropy vs ensemble confidence ---
    # if summary_df is not None:
    #     weight_cols = [c for c in summary_df.columns if c.endswith("_weight")]
    #     if weight_cols:
    #         weight_matrix = summary_df[weight_cols].to_numpy(dtype=float)
    #         entropies = []
    #         for row in weight_matrix:
    #             w = row.copy()
    #             if w.sum() <= 0:
    #                 entropies.append(0.0)
    #                 continue
    #             w = w / w.sum()
    #             entropy = -np.sum([p * np.log2(p) for p in w if p > 0])
    #             max_entropy = np.log2(len(w)) if len(w) > 0 else 1.0
    #             entropies.append(float(entropy / max_entropy if max_entropy > 0 else 0.0))

    #         summary_df["method_entropy"] = entropies
    #         entropy_order = cluster_order or summary_df["cluster"].tolist()
    #         if "weighted_confidence" not in summary_df.columns and active_fdr_cols:
    #             summary_df["weighted_confidence"] = calculate_weighted_confidence(summary_df, active_fdr_cols)

    #         summary_df_ord = summary_df.set_index("cluster").loc[entropy_order].reset_index()

    #         fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    #         axes[0].bar(range(len(entropy_order)), summary_df_ord["method_entropy"])
    #         axes[0].set_xticks(range(len(entropy_order)))
    #         axes[0].set_xticklabels(entropy_order, rotation=45, ha="right")
    #         axes[0].set_xlabel("Cluster")
    #         axes[0].set_ylabel("Normalized Method Entropy")
    #         axes[0].set_ylim(0, 1.05)
    #         axes[0].set_title("Method Entropy per Cluster")

    #         if "weighted_confidence" in summary_df_ord.columns:
    #             axes[1].scatter(summary_df_ord["method_entropy"], summary_df_ord["weighted_confidence"])
    #             axes[1].set_xlabel("Method Entropy")
    #             axes[1].set_ylabel("Weighted Consensus Score")
    #             axes[1].set_title("Entropy vs Ensemble Confidence")
    #             axes[1].grid(alpha=0.3)
    #             axes[1].set_xlim(-0.05, 1.05)
    #             axes[1].set_ylim(-0.05, 1.05)
    #         else:
    #             axes[1].axis("off")

    #         fig.tight_layout()
    #         save_fig(fig, out_dir, "method_entropy.png")

    logger.info("Plotting UMAP and spatial annotation.")
    sc.pl.umap(
        adata,
        color=[
            primary_cluster,
            "cell_subtype",
            "cancer_associated",
            "cell_major_type",
            "cell_pannuke_label",
            "cell_hne_type",
        ],
        show=False,
        ncols=2,
        title=[
            primary_cluster,
            "cell_subtype",
            "cancer_associated",
            "cell_major_type",
            "cell_pannuke_label",
            "cell_hne_type",
        ],
        legend_loc="on data",
        legend_fontsize=8,
        legend_fontoutline=2,
        frameon=False,
    )
    save_current_fig(out_dir, "umap_annotation.png")

    sc.pl.spatial(
        adata,
        color=[
            primary_cluster,
            "cell_subtype",
            "cancer_associated",
            "cell_major_type",
            "cell_pannuke_label",
            "cell_hne_type",
        ],
        size=10,
        ncols=2,
        show=False,
        spot_size=1,
        title=[
            primary_cluster,
            "cell_subtype",
            "cancer_associated",
            "cell_major_type",
            "cell_pannuke_label",
            "cell_hne_type",
        ],
    )
    save_current_fig(out_dir, "spatial_annotation.png")

    logger.info("Computing dotplot markers per cell_subtype.")
    sc.tl.rank_genes_groups(
        adata,
        groupby="cell_subtype",
        method="wilcoxon",
        use_raw=False,
        key_added="dotplot",
    )
    top_markers = []
    for group in adata.obs["cell_subtype"].cat.categories:
        markers = sc.get.rank_genes_groups_df(adata, group=group, key="dotplot").head(10)["names"].tolist()
        top_markers.extend(markers)
    top_markers = list(dict.fromkeys(top_markers))

    sc.tl.dendrogram(adata, groupby="cell_subtype")

    dot = sc.pl.dotplot(
        adata,
        var_names=top_markers,
        groupby="cell_subtype",
        standard_scale="var",
        dendrogram=True,
        return_fig=True,
        show=False,
    )
    
    axes_dict = dot.get_axes()
    # prefer the main plot axis if present
    if isinstance(axes_dict, dict):
        main_ax = axes_dict.get("mainplot_ax", list(axes_dict.values())[0])
    else:
        # older scanpy versions: list of axes
        main_ax = axes_dict[0] if isinstance(axes_dict, (list, tuple)) else axes_dict
    
    for label in main_ax.get_xticklabels():
        label.set_fontsize(10)
    
    save_fig(dot.fig, out_dir, "dotplot_cell_subtype.png")

    logger.info("Plotting mean expression violins.")
    # Use sparse-aware mean to avoid densifying the raw matrix when computing overview violins.
    raw_matrix = adata.raw.X if adata.raw is not None else adata.X
    mean_expression = np.asarray(raw_matrix.mean(axis=1)).ravel()
    adata.obs["mean_expression"] = mean_expression

    fig, axes = plt.subplots(1, 4, figsize=(16, 6), sharey=True)
    sns.violinplot(
        x=adata.obs["cell_subtype"],
        y=adata.obs["mean_expression"],
        ax=axes[0],
    )
    
    axes[0].set_title("Mean Exp. by Cell Type")
    axes[0].tick_params(axis="x", labelrotation=45)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    axes[0].set_xlabel("")

    sns.violinplot(
        x=adata.obs["cell_major_type"],
        y=adata.obs["mean_expression"],
        ax=axes[1],
    )
    axes[1].set_title("Mean Exp. by Major Type")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_xlabel("")

    sns.violinplot(
        x=adata.obs["cell_pannuke_label"],
        y=adata.obs["mean_expression"],
        ax=axes[2],
    )
    axes[2].set_title("Mean Exp. by PanNuke Label")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_xlabel("")

    sns.violinplot(
        x=adata.obs["cell_hne_type"],
        y=adata.obs["mean_expression"],
        ax=axes[3],
    )
    axes[3].set_title("Mean Exp. by H&E Type")
    axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45, ha="right")
    axes[3].set_xlabel("")

    plt.tight_layout()
    save_fig(fig, out_dir, "violin_mean_expression.png")

    logger.info("Plotting composition bar plots.")
    cell_counts = adata.obs["cell_subtype"].value_counts()
    cell_percent = (cell_counts / len(adata)) * 100
    composition_df = pd.DataFrame({"Count": cell_counts, "Percentage": cell_percent})

    cat_counts = adata.obs["cell_major_type"].value_counts()
    cat_percent = (cat_counts / len(adata)) * 100
    category_df = pd.DataFrame({"Count": cat_counts, "Percentage": cat_percent})

    pannuke_counts = adata.obs["cell_pannuke_label"].value_counts()
    pannuke_percent = (pannuke_counts / len(adata)) * 100
    pannuke_df = pd.DataFrame({"Count": pannuke_counts, "Percentage": pannuke_percent})

    hne_type_counts = adata.obs["cell_hne_type"].value_counts()
    hne_type_percent = (hne_type_counts / len(adata)) * 100
    hne_type_df = pd.DataFrame({"Count": hne_type_counts, "Percentage": hne_type_percent})

    subtype_bars = len(composition_df)
    majortype_bars = len(category_df)
    pannuke_bars = len(pannuke_df)
    hne_type_bars = len(hne_type_df)
    width_ratios = [subtype_bars, majortype_bars, pannuke_bars, hne_type_bars]

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), gridspec_kw={"width_ratios": width_ratios})
    composition_df["Percentage"].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Cell Subtype Composition")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    category_df["Percentage"].plot(kind="bar", ax=axes[1])
    axes[1].set_title("Major Cell Type Composition")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    pannuke_df["Percentage"].plot(kind="bar", ax=axes[2])
    axes[2].set_title("PanNuke Label Composition")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_xlabel("")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")

    hne_type_df["Percentage"].plot(kind="bar", ax=axes[3])
    axes[3].set_title("H&E Type Composition")
    axes[3].set_ylabel("Percentage (%)")
    axes[3].set_xlabel("")
    axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    save_fig(fig, out_dir, "composition_bars.png")

    logger.info("Plotting cluster vs cell type heatmaps.")
    def create_heatmap_data(adata, cluster_col, celltype_col, significant_threshold=0.0):
        # Build a normalized contingency table (cluster vs cell type) for the requested obs fields.
        cluster_categories = adata.obs[cluster_col].unique()
        celltype_categories = adata.obs[celltype_col].unique()
        heatmap_data = pd.DataFrame(
            index=sorted(cluster_categories, key=cluster_sort_key),
            columns=sorted(celltype_categories),
        )
        for cluster in cluster_categories:
            subset = adata[adata.obs[cluster_col] == cluster]
            norm_counts = subset.obs[celltype_col].value_counts(normalize=True)
            for celltype, value in norm_counts.items():
                heatmap_data.loc[cluster, celltype] = value
        heatmap_data = heatmap_data.fillna(0)
        columns_sums = heatmap_data.sum()
        significant_columns = columns_sums[columns_sums >= significant_threshold].index
        heatmap_data = heatmap_data[significant_columns]
        return heatmap_data

    subtype_data = create_heatmap_data(adata, "graphclust", "cell_subtype")
    majortype_data = create_heatmap_data(adata, "graphclust", "cell_major_type")
    pannuke_label_data = create_heatmap_data(adata, "graphclust", "cell_pannuke_label")
    hne_type_data = create_heatmap_data(adata, "graphclust", "cell_hne_type")

    common_clusters = sorted(
        set(subtype_data.index)
        & set(majortype_data.index)
        & set(pannuke_label_data.index)
        & set(hne_type_data.index),
        key=cluster_sort_key,
    )
    subtype_data = subtype_data.loc[common_clusters]
    majortype_data = majortype_data.loc[common_clusters]
    pannuke_label_data = pannuke_label_data.loc[common_clusters]
    hne_type_data = hne_type_data.loc[common_clusters]

    global_min = min(
        subtype_data.min().min(),
        majortype_data.min().min(),
        pannuke_label_data.min().min(),
        hne_type_data.min().min(),
    )
    global_max = max(
        subtype_data.max().max(),
        majortype_data.max().max(),
        pannuke_label_data.max().max(),
        hne_type_data.max().max(),
    )

    subtype_cols = subtype_data.shape[1]
    majortype_cols = majortype_data.shape[1]
    pannuke_cols = pannuke_label_data.shape[1]
    hne_type_cols = hne_type_data.shape[1]
    colorbar_compensation = 1.2
    width_ratios = [subtype_cols, majortype_cols, pannuke_cols, hne_type_cols * colorbar_compensation]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": width_ratios},
    )

    sns.heatmap(
        subtype_data,
        annot=False,
        cmap="viridis",
        linewidths=0.5,
        ax=axes[0],
        cbar=False,
        vmin=global_min,
        vmax=global_max,
    )
    sns.heatmap(
        majortype_data,
        annot=False,
        cmap="viridis",
        linewidths=0.5,
        ax=axes[1],
        cbar=False,
        vmin=global_min,
        vmax=global_max,
    )
    sns.heatmap(
        pannuke_label_data,
        annot=False,
        cmap="viridis",
        linewidths=0.5,
        ax=axes[2],
        cbar=False,
        vmin=global_min,
        vmax=global_max,
    )
    sns.heatmap(
        hne_type_data,
        annot=False,
        cmap="viridis",
        linewidths=0.5,
        ax=axes[3],
        cbar=True,
        cbar_kws={"label": "Proportion", "shrink": 0.8},
        vmin=global_min,
        vmax=global_max,
    )

    def add_highlights(ax, data, highlight_color="red", highlight_linewidth=2, highlight_threshold=0.0):
        for i, row_idx in enumerate(data.index):
            row_max = np.max(data.loc[row_idx].values)
            for j, col_idx in enumerate(data.columns):
                value = data.loc[row_idx, col_idx]
                if value == row_max and value > highlight_threshold:
                    ax.add_patch(
                        plt.Rectangle(
                            (j, i),
                            1,
                            1,
                            fill=False,
                            edgecolor=highlight_color,
                            lw=highlight_linewidth,
                            clip_on=False,
                        )
                    )

    add_highlights(axes[0], subtype_data)
    add_highlights(axes[1], majortype_data)
    add_highlights(axes[2], pannuke_label_data)
    add_highlights(axes[3], hne_type_data)

    axes[0].set_title("Graph Clusters vs Cell Subtypes", fontsize=12, pad=20)
    axes[0].set_ylabel("Graph Cluster", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    axes[1].set_title("Graph Clusters vs Major Cell Types", fontsize=12, pad=20)
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_yticklabels([])

    axes[2].set_title("Graph Clusters vs PanNuke Labels", fontsize=12, pad=20)
    axes[2].set_ylabel("")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_yticklabels([])

    axes[3].set_title("Graph Clusters vs H&E Types", fontsize=12, pad=20)
    axes[3].set_ylabel("")
    axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45, ha="right")
    axes[3].set_yticklabels([])

    plt.tight_layout()
    save_fig(fig, out_dir, "cluster_vs_celltype_heatmaps.png")


def export_qust_csvs(adata: ad.AnnData, xenium_dir: Path, out_dir: Path):
    subtype_data = adata.obs.groupby("graphclust")["cell_subtype"].value_counts(normalize=True).unstack(fill_value=0)
    majortype_data = adata.obs.groupby("graphclust")["cell_major_type"].value_counts(normalize=True).unstack(fill_value=0)
    pannuke_label_data = adata.obs.groupby("graphclust")["cell_pannuke_label"].value_counts(normalize=True).unstack(fill_value=0)
    hne_type_data = adata.obs.groupby("graphclust")["cell_hne_type"].value_counts(normalize=True).unstack(fill_value=0)
    hne_label_data = adata.obs.groupby("graphclust")["cell_hne_label"].value_counts(normalize=True).unstack(fill_value=0)
    pantissue_label_data = adata.obs.groupby("graphclust")["cell_pantissue_label"].value_counts(normalize=True).unstack(fill_value=0)

    subtype_assignment = pd.DataFrame(
        {
            "classification": subtype_data.idxmax(axis=1).index.tolist(),
            "cell_type": subtype_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")
    major_assignment = pd.DataFrame(
        {
            "classification": majortype_data.idxmax(axis=1).index.tolist(),
            "cell_type": majortype_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")
    pannuke_label_assignment = pd.DataFrame(
        {
            "classification": pannuke_label_data.idxmax(axis=1).index.tolist(),
            "cell_type": pannuke_label_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")
    hne_label_assignment = pd.DataFrame(
        {
            "classification": hne_label_data.idxmax(axis=1).index.tolist(),
            "cell_type": hne_label_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")
    hne_type_assignment = pd.DataFrame(
        {
            "classification": hne_type_data.idxmax(axis=1).index.tolist(),
            "cell_type": hne_type_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")
    pantissue_label_assignment = pd.DataFrame(
        {
            "classification": pantissue_label_data.idxmax(axis=1).index.tolist(),
            "cell_type": pantissue_label_data.idxmax(axis=1).tolist(),
        }
    ).set_index("classification")

    out_sub = out_dir / "celltype_assignment_subtype.csv"
    out_major = out_dir / "celltype_assignment_major.csv"
    out_pannuke = out_dir / "celltype_assignment_pannuke_label.csv"
    out_hne_type = out_dir / "celltype_assignment_hne_type.csv"
    out_hne = out_dir / "celltype_assignment_hne_label.csv"
    out_pantissue = out_dir / "celltype_assignment_pantissue_label.csv"

    subtype_assignment.to_csv(out_sub)
    major_assignment.to_csv(out_major)
    pannuke_label_assignment.to_csv(out_pannuke)
    hne_type_assignment.to_csv(out_hne_type)
    hne_label_assignment.to_csv(out_hne)
    pantissue_label_assignment.to_csv(out_pantissue)

    logger.info(f"Saved QuST subtype assignment: {out_sub}")
    logger.info(f"Saved QuST major assignment: {out_major}")
    logger.info(f"Saved QuST pannuke label assignment: {out_pannuke}")
    logger.info(f"Saved QuST H&E type assignment: {out_hne_type}")
    logger.info(f"Saved QuST H&E label assignment: {out_hne}")
    logger.info(f"Saved QuST pantissue label assignment: {out_pantissue}")


# ---------------------- CLI ----------------------


@click.command()
@click.option(
    "--xenium-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Xenium 'outs' directory.",
)
@click.option(
    "--markers-csv",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=lambda: _default_markers_csv(),
    show_default="bundled markers-v3_2.csv",
    help="Marker gene CSV file. Defaults to the markers-v3_2.csv shipped with "
         "this package; pass a path to override.",
)
@click.option(
    "--tissue-type",
    "tissue_type",
    type=click.Choice(TISSUE_TYPES, case_sensitive=False),
    required=True,
    help="Tissue type for marker filtering.",
)
@click.option(
    "--output-dir",
    type=click.Path(dir_okay=True, file_okay=False, path_type=Path),
    default=None,
    help="Output directory. Default: xenium-path.",
)
@click.option(
    "--common-only/--no-common-only",
    default=True,
    show_default=True,
    help="Use only common cell types.",
)
@click.option(
    "--normal-only/--include-cancer",
    default=False,
    show_default=True,
    help="Include only non-malignant (normal) cell types.",
)
@click.option(
    "--use-graphclust/--use-leiden",
    default=True,
    show_default=True,
    help="Use Xenium graphclust or Leiden for primary clustering.",
)
@click.option(
    "--chosen-leiden-res",
    default=0.5,
    show_default=True,
    type=float,
    help="Leiden resolution if use-leiden is selected.",
)
@click.option(
    "--min-genes",
    default=10,
    show_default=True,
    type=int,
    help="Minimum counts per cell.",
)
@click.option(
    "--min-cells",
    default=5,
    show_default=True,
    type=int,
    help="Minimum cells per gene.",
)
@click.option(
    "--lower-percentile",
    default=1.0,
    show_default=True,
    type=float,
    help="Lower percentile for QC filtering.",
)
@click.option(
    "--upper-percentile",
    default=99.0,
    show_default=True,
    type=float,
    help="Upper percentile for QC filtering.",
)
@click.option(
    "--n-top-genes",
    default=3500,
    show_default=True,
    type=int,
    help="Number of highly variable genes to select during QC (Seurat v3).",
)
@click.option(
    "--n-perm",
    default=1000,
    show_default=True,
    type=int,
    help="Number of permutations for empirical/correlation/spatial tests.",
)
@click.option(
    "--n-jobs",
    default=32,
    show_default=True,
    type=int,
    help="Worker processes for per-cluster annotation (set to 1 to disable parallelism).",
)
@click.option(
    "--allow-cuda-parallel/--no-allow-cuda-parallel",
    "allow_cuda_parallel",
    default=False,
    show_default=True,
    help="Allow n_jobs > 1 when CUDA is visible and emp_fdr is enabled. "
         "Off by default because torch_empirical_p allocates on the least-used "
         "GPU and parallel workers can duplicate the CUDA context and OOM.",
)
@click.option(
    "--method",
    "enabled_methods",
    multiple=True,
    type=click.Choice(CORE_FDR_METHODS, case_sensitive=False),
    help="Annotation methods to enable (repeat flag to select multiple). Defaults to built-in switch.",
)
@click.option(
    "--generate-plots/--no-generate-plots",
    default=True,
    show_default=True,
    help="Generate and save plots instead of showing.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Re-run pipeline even if annotated.h5ad already exists.",
)
@click.option(
    "--regenerate-plots/--no-regenerate-plots",
    default=False,
    show_default=True,
    help="When annotated.h5ad exists reuse annotations but recreate plots/CSVs.",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    show_default=True,
    help="Verbose logging.",
)
@click.option(
    "--plot-format",
    type=click.Choice(["png", "svg"], case_sensitive=False),
    default="png",
    show_default=True,
    help="Image format for saved plots (e.g. png or svg).",
)
@click.option(
    "--plot-dpi",
    default=600,
    show_default=True,
    type=int,
    help="DPI for rasterized plots (used mainly for PNG).",
)
@click.option(
    "--use-top-k-markers",
    "use_top_k_markers",
    type=int,
    default=None,
    show_default=True,
    help="Truncate each subtype's marker list to the top-K most specific "
         "genes (order in markers-v3.csv). Requires an atlas-reranked CSV "
         "produced by rank_markers.py. Leave unset to use the full list.",
)
def main(
    xenium_dir,
    markers_csv,
    tissue_type,
    output_dir,
    common_only,
    normal_only,
    use_graphclust,
    chosen_leiden_res,
    min_genes,
    min_cells,
    lower_percentile,
    upper_percentile,
    n_top_genes,
    n_perm,
    n_jobs,
    allow_cuda_parallel,
    enabled_methods,
    generate_plots,
    overwrite,
    regenerate_plots,
    verbose,
    plot_format,
    plot_dpi,
    use_top_k_markers,
):
    """KurtoRank CLI: QC + annotation for Xenium data."""
    if _KURTO_IMPORT_ERROR is not None:
        raise click.ClickException(
            "kurtorank.annotate requires the scientific stack "
            "(spatialdata, scanpy, squidpy, torch, scipy, statsmodels, ...) "
            f"but importing it failed: {_KURTO_IMPORT_ERROR}. "
            "Install with `pip install -e pan-tissue/devel/kurtorank` "
            "in an environment that provides those packages."
        )
    setup_logging(verbose)
    # Configure global plotting defaults from CLI options.
    global PLOT_DPI, PLOT_FORMAT
    PLOT_DPI = int(plot_dpi)
    PLOT_FORMAT = plot_format.lower()
    install_double_sigint_handler()
    xenium_dir = xenium_dir.resolve()
    if output_dir is None:
        output_dir = xenium_dir
    else:
        output_dir = output_dir.resolve()
    ensure_dir(output_dir)
    
    annotated_path = output_dir / "annotated.h5ad"
    expected_csvs = [
        output_dir / "celltype_assignment_subtype.csv",
        output_dir / "celltype_assignment_major.csv",
        output_dir / "celltype_assignment_pannuke_label.csv",
        output_dir / "celltype_assignment_hne_type.csv",
        output_dir / "celltype_assignment_hne_label.csv",
    ]

    annotated_exists = annotated_path.exists()
    csv_missing = [path for path in expected_csvs if not path.exists()]

    if annotated_exists and not overwrite:
        force_regen = regenerate_plots or bool(csv_missing)
        if force_regen:
            if csv_missing:
                missing_str = ", ".join(str(p) for p in csv_missing)
                logger.info(
                    "Annotated dataset exists but required CSVs missing (%s); regenerating plots/CSVs.",
                    missing_str,
                )
            else:
                logger.info(
                    "Annotated dataset already exists at %s; regenerating plots/CSVs per user request.",
                    annotated_path,
                )

            adata = sc.read_h5ad(annotated_path)
            primary_cluster = adata.uns.get("primary_cluster_key")
            if primary_cluster is None or primary_cluster not in adata.obs.columns:
                if "clusters" in adata.obs.columns:
                    primary_cluster = "clusters"
                    logger.warning(
                        "primary_cluster_key not stored; defaulting to 'clusters' column for visualization/export."
                    )
                else:
                    raise ValueError(
                        "Annotated dataset lacks primary cluster information; rerun with --overwrite to rebuild."
                    )

            export_qust_csvs(adata, xenium_dir, output_dir)
            visualize_annotation(adata, primary_cluster, generate_plots, output_dir)
            logger.info("Finished regenerating outputs from existing annotated dataset.")
        else:
            logger.info(
                "Annotated dataset already exists at %s and all CSV outputs are present; skipping all steps (use --regenerate-plots or --overwrite to rerun).",
                annotated_path,
            )
        return

    logger.info(f"Xenium dir: {xenium_dir}")
    logger.info(f"Markers CSV: {markers_csv}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Tissue type: {tissue_type}")
    logger.info(f"Common only: {common_only}")
    logger.info(f"Normal only (exclude malignant markers): {normal_only}")
    logger.info(f"Use graphclust: {use_graphclust}")
    logger.info(f"Highly variable gene count: {n_top_genes}")
    logger.info(f"Generate plots: {generate_plots}")
    logger.info(f"Regenerate plots flag: {regenerate_plots}")
    logger.info(f"Regenerate plots if annotated exists: {regenerate_plots}")

    sc.settings.autoshow = False

    method_switch = resolve_method_switch(enabled_methods)

    qced_path = xenium_dir / "qced.h5ad"
    if qced_path.exists():
        logger.info(f"Loading existing QC'ed data: {qced_path}")
        adata = sc.read_h5ad(qced_path)
        qc_needed = False
    else:
        adata = load_or_build_adata(xenium_dir)
        qc_needed = True

    ensure_graphclust(adata, xenium_dir)

    if qc_needed:
        logger.info("QC'ed data not found; running QC workflow.")
        adata = run_qc(
            adata,
            min_genes=min_genes,
            min_cells=min_cells,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            n_top_genes=n_top_genes,
            generate_plots=generate_plots,
            out_dir=output_dir,
        )
        logger.info(f"Saving QC'ed data to: {qced_path}")
        adata.write_h5ad(qced_path)
    else:
        logger.info("QC already completed; using existing qced.h5ad.")

    primary_cluster = run_leiden_scan(
        adata,
        use_graphclust=use_graphclust,
        chosen_leiden_res=chosen_leiden_res,
        generate_plots=generate_plots,
        out_dir=output_dir,
    )

    adata, final_df = run_kurtorank(
        adata,
        primary_cluster=primary_cluster,
        markers_csv=markers_csv,
        tissue_type=tissue_type,
        common_only=common_only,
        normal_only=normal_only,
        method_switch=method_switch,
        n_perm=n_perm,
        n_jobs=n_jobs,
        generate_plots=generate_plots,
        out_dir=output_dir,
        use_top_k_markers=use_top_k_markers,
        allow_cuda_parallel=allow_cuda_parallel,
    )

    adata.uns["primary_cluster_key"] = primary_cluster

    visualize_annotation(adata, primary_cluster, generate_plots, output_dir, method_switch)
    export_qust_csvs(adata, xenium_dir, output_dir)

    logger.info(f"Saving annotated data to: {annotated_path}")
    adata.write_h5ad(annotated_path)

    logger.info("KurtoRank pipeline finished.")


# Exposed as a click command; registered as a subcommand by kurtorank.cli.
annotate_cmd = main
