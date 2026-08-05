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
wsitrain train --input /path/to/cohort --tissue breast
wsitrain --version
```

## Data contract

Per sample:

- outs/cells.parquet
- outs/analysis/clustering/gene_expression_graphclust/clusters.csv
- outs/celltype_assignment_<task>_label.csv (from kurtorank annotate)
- outs/registration_params.json (+ optional outs/direct_transf.txt)
- sibling H&E file matching *_he_*image.ome.tif

## Notes

- Registration is consumed, not generated. Produce ST2WSI outputs before training.
- Use --transform affine+bspline, affine, or none based on alignment state.
- Optional stardist/tensorflow install may be skipped; cellpose path remains supported.
