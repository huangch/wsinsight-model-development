---
name: wsinsight-train
description: Install and run wsitrain to build wsinsight-ready CellViT models from Xenium+H&E cohorts
---

# wsinsight-train — Agentic AI Skill File

## Purpose

Use this skill to install and operate wsinsight-train (command: wsitrain), a headless training CLI that builds wsinsight-ready CellViT model folders from paired Xenium and H&E data.

## Quick install

```bash
bash conda-setup.sh -n wsitrain -r
conda activate wsitrain
export CELLVIT_ROOT=/path/to/CellViT-plus-plus
```

## Core commands

```bash
wsitrain check --input /path/to/cohort --tissue pantissue
wsitrain run --input /path/to/cohort --tissue breast
wsitrain --version
```

Each pipeline stage is also a command: `annotate`, `segment`, `transfer`,
`tile`, `split`, `train`, `validate`, `export`, `report`. They take the same
`--input/--tissue/--output` and refuse to start until the stages they depend on
are done, so you can step through the pipeline one command at a time. `run`
drops stages with `--run-skip a b`.

A stage command only offers the flags its own stage reads; everything else is
inherited from `<output>/run-<tissue>.yaml`, written by the previous command.
Use `--reset-config` to ignore that and fall back to the shipped defaults.

## Data contract

Per sample:

- outs/cells.parquet
- outs/analysis/clustering/gene_expression_graphclust/clusters.csv
- outs/celltype_assignment_<task>_label.csv (from kurtorank annotate)
- outs/registration_params.json (+ optional outs/direct_transf.txt)
- sibling H&E file matching *_he_*image.ome.tif

## Notes

- Registration is consumed, not generated. Produce ST2WSI outputs before training.
- Use --transform affine+bspline, affine, or none based on alignment state; it
  decides which samples every stage sees, so segment and transfer must agree.
- --gpus cpu only turns off segmentation on the GPU; CellViT training needs a
  device index and the split stage rejects it.
- stardist is the default segmenter; its tensorflow install is optional, and
  --segmenter cellpose is the fallback when it is skipped.
