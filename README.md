# wsinsight-model-development

This repository hosts the H&E classifier-head training workflow used to
produce the site-specific Model Zoo heads consumed by
[WSInsight](https://github.com/huangch/wsinsight). It is **not** required
to run WSInsight inference; it is provided so that third parties with
their own Xenium-paired cohort can reproduce or extend the lineage-resolved
single-cell heads.

## Sub-areas

- [`cellvit-training/`](cellvit-training/) — H&E classifier-head training
  workflow that fine-tunes CellViT-SAM-H-x40 from QuST-derived Xenium labels.
  Driven by the tissue-agnostic [`pipeline/train.sh`](cellvit-training/pipeline/train.sh)
  and [`pipeline/validate.sh`](cellvit-training/pipeline/validate.sh) scripts;
  per-tissue config in [`tissue_configs/`](cellvit-training/tissue_configs/).

## Data

- [`data/`](data/) — bulk reference data (not committed; see `.gitignore`):
  - `data/xenium/` — raw 10x Xenium output bundles.

  Recreate the data tree locally by following `data/README.md` before
  running the training pipeline.
