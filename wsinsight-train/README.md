# wsinsight-train

End-to-end, **headless** CLI to train [WSInsight](https://github.com/huangch/wsinsight)
CellViT cell-classification heads from paired **10x Xenium + H&E** — no GUI, no
QuPath required. Distribution `wsinsight-train`; package + command `wsitrain`.

KurtoRank labels Xenium cells, nuclei are segmented on H&E (StarDist default,
Cellpose optional), labels transfer by ST2WSI registration (SIFT affine by
default, optional bUnwarpJ B-spline), tiles train a CellViT head, and the
result is exported as a wsinsight-ready model folder.

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
    registration_params.json                    # from ST2WSI (SIFT affine)
    direct_transf.txt                           # optional; only --transform affine+bspline
  *_he_image.ome.tif
```

A sample is "aligned" if registration files exist (or the H&E is already
pixel-aligned). `wsitrain check` lists what it found in
`<output>/wsitrain_samples.csv`; that file is a report, not an input.

## Commands

```bash
wsitrain check --input DIR [--tissue T] [--output DIR]  # preflight: samples + env + GPU
wsitrain run --input DIR --tissue T ...      # full pipeline
wsitrain <stage> --input DIR --tissue T ...  # one stage, e.g. wsitrain segment
wsitrain --version
```

Training scope via `--tissue`: one (`breast`), subset (`breast,lung`), or all
(`pantissue`). Stages: `annotate → segment → transfer → tile → split → train →
validate → export → report`, and each is also a command of its own.

Run the lot with `wsitrain run` (drop stages with `--run-skip a b`), or drive the
pipeline one command at a time. A stage command refuses to start unless the
stages it depends on are marked done in the manifest and their output is still
there, so `wsitrain tile` on a fresh `--output` tells you to run `transfer`
first rather than producing an empty tile set. Each stage command offers only
the flags its own stage reads; anything else it needs is carried over from the
config the previous command wrote into `--output`. Completed stages are skipped
on re-run (`--force` overrides, and also discards the masks segment would
otherwise reuse).

### Where a setting's value comes from

Four layers, lowest priority first:

```
defaults/run.yaml  <  <output>/run-<tissue>.yaml  <  --config FILE  <  CLI flags
```

Every command prints which layer each of its settings came from; `--show-config`
lists the ones sitting at their defaults too. Settings a command does not read
(`--val-frac` during `segment`, say) are still carried forward for the stages
that do, and reported as such.

`--config` **patches** the saved record rather than replacing it. The saved file
is a full dump, so replacing it would revert every earlier non-default choice
and invalidate the stages that produced them. The two flags are independent:

| | base layers | use for |
|---|---|---|
| `--config F` | defaults + saved + F | change a few settings on an existing run |
| `--reset-config` | defaults | start clean; what the scripts do |
| `--config F --reset-config` | defaults + F | reproduce a run exactly from its `run-<tissue>.yaml` |

A `--config` file is hand-written, so it is checked strictly: an unrecognised
setting is an error with a suggestion, and values are validated against the same
choices the flags accept. `input`, `tissue` and `output` in the file are ignored
— they always come from the command line, so a saved record can be fed back
verbatim.

Key flags: `--task` (label space: `sthelar_full|sthelar_coarse|sthelar_cancer_normal|
hne|pantissue|pannuke|lcp`, default `sthelar_full`), `--segmenter cellpose|stardist`,
`--transform affine+bspline|affine|none`, `--tune N` (auto-tune iters), `--gpus auto`.
`--gpus cpu` disables the GPU for segmentation only; CellViT training needs a
device index, so the split stage refuses it.

## Scripts

```bash
bash scripts/train_one_tissue_by_tile.sh breast [input_dir] [output_dir]
bash scripts/train_tissues_by_tile.sh breast,lung [input_dir] [output_dir]  # pooled subset
bash scripts/train_pantissue_by_tile.sh [input_dir] [output_dir]            # all tissues pooled
```
Env overrides: `TASK`, `SEGMENTER`, `STARDIST_MODEL_DIR`, `VAL_FRAC`, `SEED`,
`TUNE`, `RUN_SKIP`, `FORCE`, `GPUS`, `ENVBIN`, `CELLVIT_ROOT`.
Each script passes `--reset-config`, so its flags alone define the run.
Caches default to `/workspace/.cellpose` + `/workspace/.torch`; tmp to `/tmp`.

## Outputs (under `--output`)

```
models/<tissue>/main/   config.json + torchscript_model.pt + label_map.yaml  (wsinsight-ready)
report/<tissue>/        confusion_matrix.png, scores.json, summary.txt, tuning_log.jsonl
trainingset/<tissue>/   tiles, label_map.yaml, splits/, train_configs/
logs/<tissue>/          CellViT run dirs (checkpoints + val_results)
masks/, nuclei/         intermediates
run-<tissue>.yaml       resolved config, inherited by the next command
manifest-<tissue>.json  per-stage status
```

## Pan-cancer vs subset

`--tissue pantissue` pools every sample; `--tissue breast,lung` pools a chosen
subset (replaces the old per-tissue aggregate). All share one label_map.

## Notes

- Registration is **applied**, not generated — produce it with ST2WSI first.
- `--tune N` retrains, keeping changes that improve macro-F1 (one lever/iter).
- The `wsinsight` model folder loads directly like `zoo/.../CellViT-SAM-H-x40/main`.
