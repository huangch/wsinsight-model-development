---
description: KurtoRank installable package (pan-tissue Xenium annotation + Census marker reranking + DISCO seed-panel builder). Invoke via the `kurtorank` CLI or the `rerank_markers` / `build_panel` Python APIs.
applyTo: "**/kurtorank/**"
---

# SKILL: kurtorank (v3.0.0)

## Purpose

An installable Python package (`kurtorank`). Three entry points, one
shared codebase:

1. **`kurtorank annotate`** — KurtoRank CLI. Consumes a Xenium `outs/`
   directory + a marker panel CSV, produces `annotated.h5ad` and
   diagnostic plots. Implementation in `src/kurtorank/annotate/main.py`.
2. **`kurtorank rank-markers`** — reranks the curated marker panel
   using CELLxGENE Census atlas statistics. Writes `markers` (reordered),
   `rank_source`, `low_support` columns back in place, plus a per-gene
   QC audit. Implementation in `src/kurtorank/rank/main.py`, also
   importable as `kurtorank.rerank_markers`.
3. **`kurtorank build-panel`** — fetches cell-type DEG tables from the
   DISCO atlas (https://immunesinglecell.com) and emits a **skeleton**
   marker CSV (`tissue_type, subtype, markers, n_cells, source,
   added_at`). Biology columns required by `annotate` (`major_type`,
   `pannuke_label`, `hne_type`, `hne_label`, `common`, `malignant`) are
   **not** produced here and must be filled in manually. Implementation
   in `src/kurtorank/seed/main.py` + `src/kurtorank/seed/disco.py`, also
   importable as `kurtorank.build_panel`.

The bundled panel is `src/kurtorank/markers/data/markers-v3.csv`, exposed
via `kurtorank.markers.default_markers_csv()` and used as the default for
`kurtorank annotate --markers-csv`.

## Install

Any Python environment with `pip` (developed against Python 3.12; 3.10+
supported).

```bash
# from the repository root (directory containing pyproject.toml):
pip install -e .
```

Console script: `kurtorank`. Python import: `import kurtorank`.

## Invariants agents must preserve

- **Distribution name is `kurtorank`**, **import name is `kurtorank`**,
  version **3.0.0**. The "3" lives in the version, not the package name.
- Annotate, rank-markers, and build-panel are separate Click subcommands;
  **do not** fold them. They have distinct runtime profiles (annotate =
  GPU / scanpy; rank-markers = Census / tiledbsoma; build-panel = HTTP
  only, stdlib).
- `cellxgene-census` and `tiledbsoma` are **core** dependencies so that
  `kurtorank rank-markers` works out of the box. Do not move them into an
  optional extra — users were hitting `ModuleNotFoundError` at runtime.
- `markers-v3.csv` is bundled via `[tool.setuptools.package-data]`. Keep
  `rank_source` and `low_support` columns — downstream consumers rely on
  their presence.
- `build-panel` output is a **skeleton**, not a drop-in replacement for
  `markers-v3.csv`. Do not auto-merge DISCO output into the bundled
  panel; that requires hand-curation of biology columns.
- DISCO atlas identifier is the **slug** (e.g. `blood`, `adipose_cell`),
  not the display tissue label. One tissue can span multiple atlases
  (`adipose_cell` vs `adipose_nucleus`) — merging is the user's call.
- `build-panel` default filter is Seurat-standard (`logfc >= 1.0` AND
  `pct1 >= 0.25`). The default atlas filter is `type == "tissue"`;
  disease / cell-type atlases require explicit flags.
- `seed/disco.py` uses only stdlib (`urllib.request`). Do not add
  `requests` / `httpx` as a dependency — `build-panel` must remain
  dependency-free beyond what `annotate` already pulls in.
- `rank-markers` uses **one worker per tissue group**; effective count is
  `min(--parallel, n_tissues)`. Raising `--parallel` past `n_tissues`
  doesn't help — bottleneck is per-tissue TileDB preload.
- Workers MUST use `mp.get_context("spawn")`. Fork inheritance broke
  tqdm + torch state in earlier iterations.
- Live progress writes to `/dev/tty` when available (see `_status_stream()`
  in `src/kurtorank/rank/main.py`). Do **not** pipe rank-markers stderr
  through `tee`; use `--log-file` for persistent logs.
- Worker INFO logs are silenced when `progress_queue` is attached
  (`logging.getLogger().setLevel(logging.WARNING)`). Preserve — flooding
  stderr breaks the bar.
- Annotate's heavy imports (`spatialdata`, `scanpy`, `squidpy`, `torch`,
  `scipy`, `statsmodels`) are **lazy** at the module top. Keep them
  wrapped in the `try / except ImportError` block so `kurtorank --help`
  works in minimal envs.
- CUDA safety cap: when `method_switch["emp_fdr"]` is on, CUDA is
  visible, and `n_jobs > 1`, `run_kurtorank` caps `n_jobs=1` unless
  `allow_cuda_parallel=True`. Don't remove.

## Canonical commands

### Rerank markers against a local Census (subset of tissues)

```bash
kurtorank rank-markers \
  --census-uri /path/to/census-soma \
  --tissues breast,colorectal,immune,circulating \
  --parallel 4 \
  --checkpoint checkpoint.csv \
  --log-file rank.log
```

- Wall-clock is dominated by the slowest tissue preload
  (`immune` > `colorectal` > `circulating` > `breast`).
- Expect 15–25 min for a 4-tissue subset on local SOMA.
- `--dry-run` computes without writing.

### Rerank every tissue (local Census, full panel)

Omit `--tissues` / `--tissue` to process every tissue in the input CSV:

```bash
kurtorank rank-markers \
  --census-uri /path/to/census-soma \
  --parallel 18 \
  --checkpoint checkpoint.csv \
  --log-file rank_all.log
```

- Effective parallelism caps at `n_tissues` (18 for the bundled panel).
- Expect 25–40 min total on local SOMA.

### Download a Census SOMA snapshot (recommended)

Local SOMA is ~1.5 TB but makes rank-markers 20–30× faster. Bucket is
public (`--no-sign-request`). Use `s5cmd` (5-10× faster than awscli) or
`awscli` as a fallback:

```bash
conda install -c conda-forge s5cmd       # recommended
# or: pip install awscli

RELEASE=2025-11-08
DEST=./census-soma/$RELEASE
s5cmd --no-sign-request \
  cp "s3://cellxgene-census-public-us-west-2/cell-census/$RELEASE/soma/*" \
     "$DEST/"
```

Then pass `--census-uri "$DEST"`. Neither `s5cmd` nor `awscli` is a
package dependency; they are only needed for the one-time mirror.

### Rerank online (streamed Census)

Drop `--census-uri` to fall back to hosted Census over HTTPS:

```bash
kurtorank rank-markers \
  --census-version 2025-11-08 \
  --parallel 4 \
  --checkpoint checkpoint.csv \
  --log-file rank_online.log
```

- **Keep `--parallel` low (2–4).** Streaming is I/O-bound; more workers
  thrash the shared HTTPS connection and trigger S3 throttling.
- **Pin `--census-version`** for reproducibility; default `latest` drifts.
- Expect **20–30× slower** than local (10–20 h for full 18-tissue run).
- `--checkpoint` is essential — transient S3 errors resume cleanly.

### Annotate a Xenium sample

```bash
kurtorank annotate \
  --xenium-dir /path/to/outs \
  --tissue-type breast \
  --output-dir ./out/sample1 \
  --common-only --include-cancer \
  --use-graphclust \
  --use-top-k-markers 25 \
  --n-jobs 8
```

Omit `--markers-csv` to use the bundled `markers-v3.csv`.

`--use-top-k-markers K` is v3-specific: the CSV stores genes in
atlas-specificity order, so top-K means "K most discriminative genes".

### Python API (rerank only)

```python
from kurtorank import rerank_markers
df_out, qc_df = rerank_markers(
    input_csv="markers-v3.csv",
    tissues=["breast", "colorectal"],
    census_uri="/path/to/census-soma",
    parallel=4,
    dry_run=True,
)
```

## Key internal structure

### `src/kurtorank/rank/main.py`

- `TISSUE_MAP` (~line 100) — `tissue_type -> list[Census tissue_general]`.
  Update when adding a tissue.
- `HNE_LABEL_KEYS` / `SUBTYPE_OVERRIDES` — keyword hints that select
  Census `cell_type` labels matching a marker-CSV row. Per-subtype
  overrides take priority over hne_label hints.
- `_score_tissue_group(tissue_type, rows, …, progress_queue=…)` — worker
  entry point. Preloads ONE AnnData per tissue then scores every row
  in memory. Emits: `preload_start`, `preload_done`, `row_done`,
  `tissue_done`, `tissue_error`.
- Scoring: `composite = 0.4·AUC + 0.3·min(log2FC/3,1) + 0.2·(pct_in−pct_out) + 0.1·lit`.
  `TAU=0.30`, `FLOOR_K=5`, `CEILING_K=50`, `TARGET_SAMPLE_CAP=8000`,
  `BG_SAMPLE_CAP=5000`.
- Two entry points: `_run_rerank(args)` (pure pipeline, returns
  `(df_out, qc_df)`, only writes when args say so) and
  `rerank_markers(**kwargs)` (typed Python API). `main()` is the argparse
  CLI — do not inline the pipeline back into it.
- Main driver uses `ProcessPoolExecutor(mp_context=spawn)` + a
  `Manager.Queue` drained by a daemon thread that updates a single tqdm
  bar.

### `src/kurtorank/annotate/main.py`

- `_process_cluster_worker(cluster_id)` — per-cluster worker; reads from
  a module-level `_CLUSTER_CONTEXT` set by an initializer.
- `CORE_FDR_METHODS` — the 9 methods the kurtosis weighting operates on.
  Names include the `_fdr` suffix (e.g. `emp_fdr`, `de_fdr`).
- `run_kurtorank(..., use_top_k_markers=None, allow_cuda_parallel=False)`
  — top-level pipeline. Accepts the CUDA-safety override.
- Jaccard redundancy check runs after `all_markers` is finalized; pairs
  with Jaccard ≥ 0.4 (panel-restricted) are logged and stashed in
  `adata.uns["marker_jaccard_high"]`.
- Tie-break: `active_priority = [k for k in tie_break_priority if
  method_switch.get(k, True) and k in tie_df.columns]`. Required when a
  method is disabled via `--method`.

### `src/kurtorank/cli.py`

- Root click group `cli()` with two subcommands:
  - `annotate` = `kurtorank.annotate.main.annotate_cmd` (click-native).
  - `rank-markers` = passthrough; forwards `argv` to the argparse-based
    `rank_markers_main()`.
- `_main()` (the console-script entry point) sets BLAS thread caps and
  `mp.set_start_method("spawn", force=True)` before calling `cli()`.

## Frequently-needed patches

- **Add a new tissue**: edit `TISSUE_MAP` in `src/kurtorank/rank/main.py`,
  add rows with the new `tissue_type` to `markers-v3.csv` (columns:
  `common`, `malignant`, `hne_type`, `hne_label`, `pannuke_label`,
  `major_type`, `markers`), optionally add `SUBTYPE_OVERRIDES`, then run
  `kurtorank rank-markers --tissues <new>`.
- **Add a new KurtoRank test**: extend `CORE_FDR_METHODS`, add it to
  `method_switch` defaults, emit the FDR array in
  `_process_cluster_worker`, and append to `tie_break_priority`.
- **Ship an updated panel**: overwrite
  `src/kurtorank/markers/data/markers-v3.csv`. The editable install
  picks it up automatically; wheels need a rebuild.

## Things that look like bugs but are not

- `tqdm` is imported top-level in `rank/main.py` but only instantiated in
  the main process — workers deliberately do not touch it. Removing the
  import will break the bar.
- `checkpoint.csv` is written after every tissue completion. Safe to
  delete after a successful run.
- `rank-markers` prints `[INFO] dispatching …` *before* creating the
  tqdm bar so the bar has a stable final line. Preserve that order.

## Pitfalls that have bitten us

- **Piping rank-markers stderr through `tee` destroys the status line.**
  Use `--log-file`.
- **`ProcessPoolExecutor` default fork context** inherits tqdm state
  across workers and causes garbled output. Must use spawn.
- **Per-row Census queries are ~80s each.** Always use the tissue
  preload pattern; never re-query per row.
- **`locals()[name] = np.clip(...)`** is a silent no-op inside a
  function. If you see that pattern, replace with explicit assignments.
- **CUDA + joblib**: `torch_empirical_p` allocates on the least-used GPU.
  Running with `n_jobs > 1` duplicates the CUDA context and OOMs. The
  `allow_cuda_parallel` flag is opt-in only.

## Related

- Census snapshot directory: any local SOMA mirror (typically 1.5 TB,
  TileDB). See README §"Downloading a local Census SOMA".
- Runtime: any Python 3.10+ environment with `pip`; dev target is
  Python 3.12.
