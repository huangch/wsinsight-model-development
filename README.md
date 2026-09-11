# wsinsight-model-development

This repository hosts the H&E classifier-head training workflow used to
produce the site-specific Model Zoo heads consumed by
[WSInsight](https://github.com/huangch/wsinsight). It is **not** required
to run WSInsight inference; it is provided so that third parties with
their own Xenium-paired cohort can reproduce or extend the lineage-resolved
single-cell heads.

## Sub-areas

- [`kurtorank/`](kurtorank/) — installable Python package that produces the
  per-cluster cell-type assignment CSV (`celltype_assignment_<tissue>_label.csv`)
  consumed by the QuPath label-transfer step. Ranks markers against
  CELLxGENE Census and annotates Xenium samples; runs upstream of the
  training pipeline.

- [`wsinsight-train/cellvit-training/`](wsinsight-train/cellvit-training/) — the
  vendored CellViT trainer plus its base checkpoints. It is not a standalone
  package: `wsitrain` puts it on `PYTHONPATH` and runs it as a subprocess,
  locating it through `$CELLVIT_ROOT`. It lives inside `wsinsight-train/` so
  that package stays self-contained.

  Promoted heads land under `models/` with a small yaml side-car; the `.pth`
  checkpoints are git-ignored and published separately on Hugging Face Hub.

## Data

- [`data/`](data/) — bulk reference data (not committed; see `.gitignore`):
  - `data/xenium/` — raw 10x Xenium output bundles.

  Recreate the data tree locally by following [`data/README.md`](data/README.md)
  before running the training pipeline.
