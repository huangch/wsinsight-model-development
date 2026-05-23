# data/

This folder holds the **raw inputs** consumed by the two model-development
workflows. **Nothing here is committed to git** — every dataset is publicly
available under its original license; this README records exactly where to
fetch each one and where to place it on disk so the pipeline finds it.

After cloning the repo, recreate this tree locally by following the
per-source instructions below.

## Layout (after fetching)

```
data/
├── xenium/         (~1.7 TB)   raw 10x Xenium output bundles, per tissue
└── census/         (~1.5 TB)   local CELLxGENE Census SOMA mirror
```

The on-disk paths must match what the pipeline expects:

- Each Xenium sample lives at
  `data/xenium/<tissue>/<dataset_folder_name>/` with `outs/` plus the H&E
  `*.ome.tif` next to it (see §1 below).
- The canonical per-sample manifest is the **QuPath project** at
  `data/qprj/project.qpproj`: every image opened in it becomes an entry
  the headless QuPath wrappers iterate over. The QuPath project is
  git-ignored (large `.qpdata` files).
- The int ↔ label-name table consumed at training time is
  `cellvit-training/trainingset/<tissue>/label_map.yaml` (hand-authored,
  tracked in git).
- KurtoRank → `data/census/<YYYY-MM-DD>/` (a SOMA mirror at a pinned date).

If your filesystem layout differs, set the `PROJECT_ROOT` environment
variable before invoking the wrappers — every script is anchored on it.

## Sources

### 1. Xenium — `data/xenium/`

**Source:** [10x Genomics — Xenium Datasets](https://www.10xgenomics.com/datasets?menu%5Bproducts.name%5D=Xenium)
**License:** [10x Genomics License](https://www.10xgenomics.com/legal/end-user-software-license-agreement) (free for non-commercial research)
**Per-sample manifests:** [`data/xenium/<tissue>/SOURCES.yaml`](xenium/) (where present)

Each sample on the 10x site provides an `outs.zip` bundle plus the H&E
`*_he_image.ome.tif` (or `*_he_unaligned_image.ome.tif`). Unzip the bundle
into an `outs/` subdir of the sample folder and place the H&E `.ome.tif`
file alongside `outs/` (one level up, sharing the directory).

Final layout per sample:

```
data/xenium/<tissue>/<dataset_folder_name>/
├── <he_image_stem>.ome.tif
└── outs/
    ├── cells.csv.gz
    ├── analysis/clustering/gene_expression_graphclust/clusters.csv
    ├── celltype_assignment_<tissue>_label.csv  # see Note below (e.g. pantissue)
    └── …  (other 10x outputs)
```

> **Note on `celltype_assignment_<tissue>_label.csv`:** This file is *not*
> part of
> the standard 10x release. It is produced by the `kurtorank annotate` CLI
> (see [`kurtorank/`](../kurtorank/README.md)) from `cell_feature_matrix.h5`.
> Either run KurtoRank on each sample, or obtain the pre-computed file from
> the project owner.

See [`data/xenium/README.md`](xenium/README.md) for the per-tissue table
plus an example `wget`/`curl` pattern.

### 2. CELLxGENE Census — `data/census/`

**Source:** [CELLxGENE Discover Census](https://chanzuckerberg.github.io/cellxgene-census/)
**License:** [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (per-dataset terms apply)

KurtoRank's marker-ranking step (`kurtorank rank-markers`) reads from a
local SOMA mirror of a pinned Census release. The mirror is large
(~1.5 TB) but only one date snapshot is needed.

See [`data/census/README.md`](census/README.md) for the `cellxgene-census`
download recipe.

## Verifying the layout

After fetching, confirm the pipeline can discover every sample by opening
the QuPath project and listing its images:

```bash
QuPath script -p data/qprj/project.qpproj -e 'getQuPath().getProject().getImageList().each { println it.getImageName() }'
```

Every `*.ome.tif` you placed under `data/xenium/<tissue>/` should appear
in the output. The headless wrappers
(`cellvit-training/qupath/run_qust_pipeline.groovy`,
`load_mapping.groovy`, `export_tiles.groovy`) discover each sample's
`outs/` directory from the image URI — no separate manifest file is
needed.
