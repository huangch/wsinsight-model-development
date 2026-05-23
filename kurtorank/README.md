# KurtoRank — Pan-Tissue Xenium Annotation

**KurtoRank** is an unsupervised ensemble strategy for cell-subtype annotation
in gene-limited spatial transcriptomics (10x Xenium and similar panels, ~300
targeted genes). It combines ~9 complementary per-subtype tests (permutation,
DE overlap, z-score, Fisher exact, proportion, correlation, spatial
co-occurrence, …) and weights each test by the **excess kurtosis** of its
across-subtype score distribution — decisive (peaked) tests up-weighted,
flat (uninformative) tests down-weighted. The weighted ensemble is
rank-aggregated into a final subtype call per Leiden / graphclust cluster.

This is the installable `kurtorank` Python package (v3.0.0).

## Package layout

```
kurtorank/
├── pyproject.toml
├── README.md
├── SKILL.md
└── src/kurtorank/
    ├── __init__.py        # exports __version__ + rerank_markers
    ├── __main__.py        # `python -m kurtorank`
    ├── cli.py             # click group: annotate + rank-markers
    ├── annotate/main.py   # annotate pipeline
    ├── rank/main.py       # Census reranker
    └── markers/
        ├── __init__.py    # default_markers_csv()
        └── data/markers-v3.csv
```

The curated panel (`markers-v3.csv`) is bundled as package data, so the
annotate CLI has a sensible default and no extra files need to travel with
the install.

---

## 1. Install

Any Python environment with `pip` will work — conda, venv, mamba, pixi,
uv, etc. Developed and tested against **Python 3.12**; Python 3.10+ is
supported.

```bash
# pick / create an env, e.g.:
#   conda create -n kurtorank python=3.12 -y && conda activate kurtorank
#   python -m venv .venv && source .venv/bin/activate
# then, from the repository root (the directory containing pyproject.toml):
pip install -e .
```

Core deps (pulled in automatically): `scanpy`, `squidpy`, `spatialdata`,
`spatialdata-io`, `anndata`, `torch`, `scipy`, `statsmodels`, `pandas`,
`numpy`, `click`, `tqdm`, `cellxgene-census`, `tiledbsoma`.

After installation you get a `kurtorank` console script and an importable
`kurtorank` Python package.

---

## 2. CLI overview

```bash
kurtorank --help
kurtorank --version            # kurtorank, version 3.0.0
kurtorank annotate --help
kurtorank rank-markers --help
kurtorank build-panel --help
```

Three subcommands:

| Subcommand | Purpose |
| --- | --- |
| `kurtorank annotate` | QC + ensemble annotation of a Xenium sample. |
| `kurtorank rank-markers` | Rerank a markers CSV against a CELLxGENE Census atlas. |
| `kurtorank build-panel` | Produce a skeleton marker panel from DISCO atlases. |

---

## 3. Annotate a Xenium sample

```bash
kurtorank annotate \
  --xenium-dir /path/to/xenium/outs \
  --tissue-type breast \
  --output-dir ./out/breast_sample1 \
  --use-graphclust \
  --use-top-k-markers 25 \
  --n-jobs 8
```

### Common flags

| Flag | Meaning |
| --- | --- |
| `--xenium-dir` | Path to Xenium `outs/` directory. **Required**. |
| `--tissue-type` | `bladder, bone, brain, breast, cervix, circulating, colorectal, heart, immune, kidney, liver, lung, lymph_node, ovary, pancreas, prostate, skin, tonsil`. **Required**. |
| `--markers-csv` | Panel CSV. Defaults to the bundled `markers-v3.csv`; pass a path to override. |
| `--output-dir` | Where to write `annotated.h5ad`, plots, CSVs (defaults to `--xenium-dir`). |
| `--common-only / --no-common-only` | Keep only `common==True` rows. |
| `--normal-only / --include-cancer` | Exclude / include malignant subtypes. |
| `--use-graphclust / --use-leiden` | Primary clustering backend. |
| `--chosen-leiden-res` | Leiden resolution when `--use-leiden`. |
| `--use-top-k-markers K` | Truncate each subtype's marker list to the top-K most specific genes (CSV order reflects atlas specificity). Leave unset to use the full list. |
| `--min-genes`, `--min-cells` | Low-level QC thresholds. |
| `--n-perm` | Permutations for empirical p-values (default 1000). |
| `--n-jobs` | Parallel workers for per-cluster annotation. |
| `--allow-cuda-parallel / --no-allow-cuda-parallel` | Permit `--n-jobs > 1` when CUDA is visible and `emp_fdr` is on. Off by default (GPU context duplication risk). |
| `--generate-plots / --no-generate-plots` | Save QC and annotation figures. |
| `--overwrite` | Rerun even if `annotated.h5ad` exists. |
| `--regenerate-plots` | Keep existing annotations, only re-draw. |
| `--plot-format {png,svg}` / `--plot-dpi` | Figure output controls. |

Full list: `kurtorank annotate --help`.

### Outputs (in `--output-dir`)

- `annotated.h5ad` — AnnData with:
  - `obs["kurtorank_cell_subtype"]` — final per-cell subtype call.
  - `obs["kurtorank_major_type"]`, `obs["kurtorank_hne_type"]`,
    `obs["kurtorank_pannuke_label"]` — downstream rollups.
  - `uns["kurtorank_results"]` — per-cluster score / weight table.
  - `uns["snr_source"]` — whether SNR used `negative_probe_counts` or the
    `control_probe_counts` fallback.
  - `uns["marker_jaccard_high"]` — subtype pairs with panel-restricted
    Jaccard ≥ 0.4 (diagnostic; these pairs are hard to separate).
- `*.csv` — per-cluster subtype assignments + QuST-compatible exports.
- `*.png` / `*.svg` — QC histograms, spatial cluster map, agreement plots.

---

## 4. Rerank the marker panel (optional)

Needed only when (a) adding/removing subtypes, (b) refreshing against a
newer Census release, or (c) customizing the panel to a different tissue
mix. The bundled `markers-v3.csv` ships a curated + already-reranked panel
for 18 tissues.

### CLI

```bash
kurtorank rank-markers \
  --input  markers-v3.csv \
  --output markers-v3.csv \
  --qc-output markers-v3_qc.csv \
  --census-uri /path/to/census-soma \
  --tissues breast,colorectal,immune,circulating \
  --parallel 4 \
  --checkpoint checkpoint.csv \
  --log-file rank.log
```

Key flags:

- `--census-uri` — path to a local Census SOMA store (20-30× faster than
  streaming). **Omit** to stream over HTTPS from the hosted Census.
- `--census-version` — pin to a specific Census release tag (e.g.
  `2025-11-08`). Default is `latest`, which can drift between runs.
- `--tissues t1,t2,…` — restrict to specific tissue groups; other rows are
  kept unchanged with `rank_source="skipped"`. **Omit both `--tissues` and
  `--tissue` to rerank every tissue in the input CSV.**
- `--parallel N` — one worker per tissue group (effective cap is
  `n_tissues`). Use 2–4 when streaming online; up to `n_tissues` (18 for
  the bundled panel) with a local SOMA.
- `--checkpoint` — CSV written after each tissue finishes. Survives
  interruptions.
- `--log-file` — append logs here instead of piping through `tee` (keeps
  the live progress bar intact).
- `--dry-run` — compute but do not write outputs.

### Rerank every tissue (local Census)

```bash
kurtorank rank-markers \
  --input  markers-v3.csv \
  --output markers-v3.csv \
  --census-uri /path/to/census-soma \
  --parallel 18 \
  --checkpoint checkpoint.csv \
  --log-file rank_all.log
```

Expected wall-clock on local SOMA: ~25–40 min, dominated by the slowest
tissue preload (`immune` > `colorectal` > `lung` > …).

#### Downloading a local Census SOMA

A local SOMA is ~1.5 TB (full Census release) but cuts rank-markers
wall-clock by 20–30× vs. streaming. Recommended for any repeated use.
Census releases live in the public S3 bucket
`s3://cellxgene-census-public-us-west-2/cell-census/<release>/soma/`.

The official SDK offers `cellxgene_census.download_source_h5ad()`, but for
a full SOMA mirror we recommend `s5cmd` (or `awscli`), which parallelize
multipart S3 downloads:

```bash
# one-time install
pip install awscli                        # slower but ubiquitous
# -- or --
conda install -c conda-forge s5cmd        # recommended; 5-10× faster

# pick a release tag (check https://cellxgene.cziscience.com/census/)
RELEASE=2025-11-08
DEST=./census-soma/$RELEASE

# s5cmd (preferred):
s5cmd --no-sign-request \
  cp "s3://cellxgene-census-public-us-west-2/cell-census/$RELEASE/soma/*" \
     "$DEST/"

# awscli fallback:
aws s3 sync --no-sign-request \
  "s3://cellxgene-census-public-us-west-2/cell-census/$RELEASE/soma/" \
  "$DEST/"
```

Then pass `--census-uri "$DEST"` to `kurtorank rank-markers`. The bucket
is public; `--no-sign-request` skips AWS credentials.

### Rerank online (streamed Census, no local snapshot)

```bash
kurtorank rank-markers \
  --input  markers-v3.csv \
  --output markers-v3.csv \
  --census-version 2025-11-08 \
  --parallel 4 \
  --checkpoint checkpoint.csv \
  --log-file rank_online.log
```

Notes:

- **Drop `--census-uri`** — the script falls back to
  `cellxgene_census.open_soma()` against the hosted S3-backed store.
- **Pin `--census-version`** for reproducibility (default `latest`
  drifts).
- **Keep `--parallel` to 2–4.** Streaming is I/O-bound; more workers
  thrash the shared connection and trigger S3 throttling.
- Budget **10–20 h for the full 18-tissue run** (versus ~30 min local).
  Always use `--checkpoint` so transient errors don't cost a full restart.

Outputs:

- `--output` CSV — updated panel with columns: `markers` (reordered),
  `rank_source` (`census` / `v3_curated` / `skipped`), `low_support`.
- `--qc-output` CSV — per-gene audit: AUC, log2FC, pct_in/out, composite.

### Python API

```python
from kurtorank import rerank_markers

df_out, qc_df = rerank_markers(
    input_csv="markers-v3.csv",
    tissues=["breast", "colorectal"],
    census_uri="/path/to/census-soma",
    parallel=4,
    dry_run=True,     # don't touch disk unless output_csv is set
)
print(df_out["rank_source"].value_counts())
```

Signature: `rerank_markers(*, input_csv, output_csv=None, qc_output=None,
tissue=None, tissues=None, dry_run=False, seed=1234, census_version=None,
census_uri=None, parallel=4, verbose=False, checkpoint=None, log_file=None,
configure_logging=True) -> (df_out, qc_df)`.

Outputs are written **only** when `output_csv` / `qc_output` are provided.

---

## 5. Build a seed panel from DISCO (optional)

`kurtorank build-panel` fetches cell-type differentially-expressed gene
(DEG) tables from the public [DISCO atlas](https://immunesinglecell.com)
and emits a **skeleton** marker CSV suitable as a starting point for a
new tissue. The resulting CSV is *not* a drop-in replacement for
`markers-v3.csv` — the biology columns consumed by `annotate`
(`major_type`, `pannuke_label`, `hne_type`, `hne_label`, `common`,
`malignant`) must be filled in manually after curation.

### 5.1. Discover atlases

```bash
kurtorank build-panel --list-atlases
# atlas                      type        tissue              cells   cts
# blood                      tissue      Blood             169,686    25
# adipose_cell               tissue      Adipose           190,492    36
# COVID-19_blood             disease     Blood             283,898    19
# ...
```

Atlases are tagged `tissue`, `disease`, or `cell type`. By default only
`tissue` atlases are included; add `--include-disease` or
`--include-celltype` to opt in.

### 5.2. Build a panel

```bash
# one or more explicit atlases
kurtorank build-panel \
  --atlases blood,lung,liver \
  --output seed.csv

# every tissue atlas (~26 atlases; downloads a few MB each)
kurtorank build-panel --all-atlases --output seed.csv
```

Output columns: `tissue_type, subtype, markers, n_cells, source,
added_at`. `source` is `disco:<atlas>:v<ver>:<release>`.

Default DEG filter is **Seurat-standard**: `logfc >= 1.0` and
`pct1 >= 0.25`. Override with `--logfc-min` / `--pct1-min`, or cap the
per-cell-type marker count with `--max-markers 50`.

Responses are cached in `~/.kurtorank/disco/<release>/<atlas>__<md5>.json`
and reused until DISCO bumps the atlas checksum.

### 5.3. Typical workflow

1. Run `build-panel` to produce `seed.csv`.
2. Open `seed.csv` in a spreadsheet; add/curate the biology columns
   (`major_type`, `pannuke_label`, `hne_type`, `hne_label`, `common`,
   `malignant`) that `annotate` consumes.
3. Optionally rerank the curated panel against Census with
   `kurtorank rank-markers` (see §6).
4. Pass the final CSV to `kurtorank annotate --markers-csv ...`.

### 5.4. Python API

```python
from pathlib import Path
from kurtorank import build_panel

build_panel(
    atlases=["blood", "lung"],
    output=Path("seed.csv"),
    logfc_min=1.0,
    pct1_min=0.25,
)
```

---

## 6. End-to-end example

```bash
# (once) rerank the panel against a new Census release:
kurtorank rank-markers \
  --input  markers-v3.csv \
  --output markers-v3.ranked.csv \
  --census-uri /path/to/census-soma \
  --parallel 8

# (per slide) annotate:
kurtorank annotate \
  --xenium-dir /data/slides/sample_A/outs \
  --markers-csv markers-v3.ranked.csv \
  --tissue-type breast \
  --use-top-k-markers 30 \
  --output-dir /data/results/sample_A
```

---

## 7. What changed from v2?

| | v2 | v3 |
| --- | --- | --- |
| Distribution | Loose scripts | Installable `kurtorank` package |
| Marker CSV | `markers-v2.csv` (hand-curated) | `markers-v3.csv` (atlas-reranked) |
| Marker order | Literature order | Atlas specificity (composite AUC + log2FC + pct_in − pct_out) |
| Marker truncation | None | `--use-top-k-markers K` |
| SNR source | Forced `control_probe_counts` | Prefers `negative_probe_counts`; fallback recorded in `uns` |
| Cell filter | `filter_cells(min_counts=…)` | `filter_cells(min_genes=…)` (semantic fix) |
| FDR clipping | `locals()[name] = …` (silent no-op) | Explicit per-array `np.clip` |
| Tie-break | All priority columns | Only methods active in `method_switch` AND present as columns |
| Malignant filter column | `normal` (bool) | `malignant` (bool) |
| CUDA × joblib safety | Implicit | Capped to `n_jobs=1` unless `--allow-cuda-parallel` |
| Marker redundancy | Not checked | Pairwise Jaccard report (stored in `uns["marker_jaccard_high"]`) |

---

## 8. Troubleshooting

- **`ModuleNotFoundError: kurtorank`** — run `pip install -e .` from the
  repository root (the directory containing `pyproject.toml`).
- **`No module named 'cellxgene_census'` / `'tiledbsoma'`** — your install
  was done with `--no-deps`. Reinstall with deps: `pip install -e .`
  (without `--no-deps`), or install them directly:
  `pip install cellxgene-census tiledbsoma`.
- **Rank-markers progress bar looks stuck at 0%** — tissues preload large
  AnnData slices from Census (multi-minute TileDB decompression). Check
  RSS in `htop`; workers should be growing memory. The bar advances once
  preloads complete.
- **Many subtypes flagged `low_support`** — either the Census tissue has
  too few cells matching the subtype's keyword hint, or the panel has too
  few marker genes present. Check the `--qc-output` CSV for the per-gene
  composite scores of those subtypes.
- **CUDA OOM / context errors during annotate** — the default CUDA safety
  cap is active. Leave `--n-jobs > 1` and `--no-allow-cuda-parallel` only
  when you have confirmed your GPU can host multiple contexts; otherwise
  just use `--n-jobs 1`.
- **Status line corrupted when piping** — don't pipe stderr through `tee`.
  Use `--log-file` for rank-markers; for annotate redirect stderr to a
  file if you need persistent logs.

---

## 9. Citation / contact

Author: CH Huang (`huangch.tw@gmail.com`).

If you use `kurtorank build-panel`, please also cite DISCO:

- Li M *et al.* *DISCO: a database of Deeply Integrated human Single-Cell
  Omics data.* Nucleic Acids Research, 2022.
  <https://doi.org/10.1093/nar/gkab1020>
- *Rediscovering publicly available single-cell data with the DISCO
  platform.* Nucleic Acids Research, 2025.
  <https://doi.org/10.1093/nar/gkae1108>
