# wsinsight-train

End-to-end, **headless** CLI to train [WSInsight](https://github.com/huangch/wsinsight)
CellViT cell-classification heads from paired **10x Xenium + H&E** — no GUI, no
QuPath required. Distribution `wsinsight-train`; package + command `wsitrain`.

KurtoRank labels Xenium cells, nuclei are segmented on H&E (Cellpose default,
StarDist optional), labels transfer by ST2WSI registration (affine + B-spline),
tiles train a CellViT head, and the result is exported as a wsinsight-ready
model folder.

---

## Install

```bash
bash conda-setup.sh -n wsitrain -r          # torch + cellpose + kurtorank + wsitrain
conda activate wsitrain
export CELLVIT_ROOT=/path/to/CellViT-plus-plus
```

Editable dev install: `pip install -e .` (kurtorank is editable from `../kurtorank`).

## Input data contract

Auto-discovered recursively under `--input`; a sample = any dir containing
`outs/cells.parquet` + a sibling H&E `*_he_*image.ome.tif`. Per-sample
registration files live in `outs/`:

```
<sample>/
  outs/
    cells.parquet
    analysis/clustering/gene_expression_graphclust/clusters.csv
    celltype_assignment_<task>_label.csv        # from kurtorank annotate
    registration_params.json + direct_transf.txt # from ST2WSI (affine+bspline)
  *_he_image.ome.tif
```

A sample is "aligned" if registration files exist (or the H&E is already
pixel-aligned). `wsitrain check` writes an editable `wsitrain_samples.csv`.

## Commands

```bash
wsitrain check --input DIR [--tissue T]     # preflight: samples + env + GPU
wsitrain run --input DIR --tissue T ...      # full pipeline
wsitrain --version
```

Training scope via `--tissue`: one (`breast`), subset (`breast,lung`), or all
(`pantissue`). Stages: `annotate → segment → transfer → tile → split → train →
validate → export`; choose what to run with `--stage-only` / `--stage-skip`, resume via
`manifest.json`.

Key flags: `--task` (label space: `sthelar_full|sthelar_coarse|sthelar_cancer_normal|
hne|pantissue|pannuke|lcp`, default `sthelar_full`), `--segmenter cellpose|stardist`,
`--transform affine+bspline|affine|none`, `--tune N` (auto-tune iters), `--gpus auto`.

## Scripts

```bash
bash scripts/train_one_tissue.sh data/xenium breast    # + cellpose/stardist parity
bash scripts/train_multi_tissue.sh data/xenium breast,lung   # pooled subset
bash scripts/train_pantissue.sh                        # all tissues pooled (with OOM guards + auto-tune)
```
Override: `TASK=hne ENVBIN=... CELLVIT_ROOT=... bash scripts/...`.
Caches default to `/workspace/.cellpose` + `/workspace/.torch`; tmp to `/tmp`.

## Outputs (under `--output`)

```
models/<tissue>/main/   config.json + torchscript/.pth + label_map.yaml  (wsinsight-ready)
report/<tissue>/        confusion + classification report + tuning_log
trainingset/<tissue>/   tiles, label_map, splits, train_configs
masks/, nuclei/         intermediate; run.yaml + manifest.json
```

## Pan-cancer vs subset

`--tissue pantissue` pools every sample; `--tissue breast,lung` pools a chosen
subset (replaces the old per-tissue aggregate). All share one label_map.

## Notes

- Registration is **applied**, not generated — produce it with ST2WSI first.
- `--tune N` retrains, keeping changes that improve macro-F1 (one lever/iter).
- The `wsinsight` model folder loads directly like `zoo/.../CellViT-SAM-H-x40/main`.
