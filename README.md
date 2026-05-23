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
  and [`pipeline/validate.sh`](cellvit-training/pipeline/validate.sh) scripts.
  Per-tissue label-name table lives in
  [`trainingset/<tissue>/label_map.yaml`](cellvit-training/trainingset/);
  the per-fold YAML lives in
  [`trainingset/<tissue>/train_configs/<backbone>/fold_*.yaml`](cellvit-training/trainingset/).
  The QuPath project that anchors per-sample annotations is at
  `data/qprj/project.qpproj` (machine-local).

  Currently set up: **`pantissue`** (12-class pan-tissue head). Promoted
  heads land under [`cellvit-training/models/`](cellvit-training/) with a
  small yaml side-car; the `.pth` checkpoint is git-ignored and published
  separately on Hugging Face Hub.

## Data

- [`data/`](data/) — bulk reference data (not committed; see `.gitignore`):
  - `data/xenium/` — raw 10x Xenium output bundles.

  Recreate the data tree locally by following [`data/README.md`](data/README.md)
  before running the training pipeline.
