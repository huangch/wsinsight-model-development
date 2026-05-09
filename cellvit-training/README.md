# cellvit-training/

H&E classifier-head training workflow that fine-tunes a CellViT-SAM-H-x40
backbone using QuST-derived Xenium cell-type labels. Produces the per-tissue
TorchScript heads consumed by WSInsight's Model Zoo.

The workflow is **tissue-agnostic**: anything that varies per tissue lives in
[`tissue_configs/<tissue>.yaml`](tissue_configs/) and
[`trainingset/<tissue>/`](trainingset/). The same wrappers train every tissue.

## Layout

```
cellvit-training/
├── pipeline/                # tissue-agnostic Python + bash drivers
│   ├── build_cell_labels.py # Xenium outs/ → per-cell label CSVs
│   ├── make_splits.py       # exported tiles → train/val splits
│   ├── train.sh             # 4-step training wrapper
│   ├── validate.sh          # re-run validation against a finished run
│   └── validate_classifier.py
├── tissue_configs/          # per-tissue inventory + label_map (source of truth)
│   ├── breast.yaml
│   └── colorectal.yaml
├── qupath/                  # QuPath Groovy helpers (tile + label export)
│   ├── export_tiles.groovy
│   └── load_mapping.groovy
├── trainingset/             # exported tiles + per-fold training configs
│   ├── breast/
│   └── colorectal/
├── cellvit/                 # vendored CellViT-plus-plus + base weights
│   ├── CellViT-plus-plus/
│   └── models/CellViT-SAM-H-x40.pth
├── models/legacy/           # archived classifier checkpoints
├── qprj/                    # QuPath projects used for label curation
└── TRAININGSET_PREPARATION.md
```

## Path portability

Everything is anchored on script location — no path is hard-coded to
`/workspace/...`. The wrappers derive:

```
PIPELINE_DIR           = dirname(BASH_SOURCE)
CELLVIT_TRAINING_ROOT  = PIPELINE_DIR/..
PROJECT_ROOT           = CELLVIT_TRAINING_ROOT/..
```

and export `${PROJECT_ROOT}` and `${CELLVIT_TRAINING_ROOT}` so that
`envsubst` can materialize the placeholders inside the training YAML before
handing it to CellViT++. The whole tree can be moved or renamed with no
edits — set `PYTHON=<...>` to override the interpreter if needed.

## End-to-end pipeline (per tissue)

For a tissue named `<tissue>` (e.g. `breast`, `colorectal`):

```bash
# 0. (one-time) curate cell-type labels in QuPath; export per-sample
#    cell_id → cell_type CSVs into the tissue's outs/ folder.

# 1. Build per-cell label CSVs from Xenium outs/ + tissue_configs/<tissue>.yaml
python pipeline/build_cell_labels.py --tissue <tissue>

# 2. (in QuPath) open each <sample>_he_image.ome.tif and run
#    qupath/export_tiles.groovy with OUTPUT_ROOT pointing at
#    trainingset/<tissue>/   to produce 1024×1024 PNG tiles + label CSVs.

# 3. Build train/val splits from the exported tiles
python pipeline/make_splits.py --tissue <tissue>

# 4. Train head + auto-validate + export TorchScript (4 steps in one)
bash pipeline/train.sh <tissue>                     # SAM-H-x40, fold_0
bash pipeline/train.sh <tissue> SAM-H-x40 fold_0    # explicit form

# 5. Re-run only validation against a finished training run
bash pipeline/validate.sh <tissue>
```

`train.sh` runs four steps in sequence: train the `LinearClassifier` head,
locate `model_best.pth` under
`cellvit/CellViT-plus-plus/logs_local/<TIMESTAMP>_<tissue>-hne-<backbone>/checkpoints/`,
emit a confusion matrix + classification report, and convert the checkpoint
to TorchScript at 1024×1024.

## Adding a new tissue

1. Create `tissue_configs/<tissue>.yaml` with `xenium_base`, `out_dir`,
   `label_map` (frozen 0..N-1 integer assignment), and `samples` (list of
   `[he_image_stem, relative_outs_path]` pairs). Use `${PROJECT_ROOT}` for
   any path reference.
2. Curate cell types in QuPath; ensure each sample's `outs/` contains
   `celltype_assignment_hne_label.csv`, `clusters.csv`, and `cells.csv.gz`.
3. Run `pipeline/build_cell_labels.py --tissue <tissue>` to produce the
   per-cell CSVs and `trainingset/<tissue>/label_map.yaml`.
4. Open each sample in QuPath and run `qupath/export_tiles.groovy` (set
   `OUTPUT_ROOT` to `trainingset/<tissue>/`).
5. Author `trainingset/<tissue>/train_configs/SAM-H-x40/fold_0.yaml`
   (copy from breast or colorectal; keep the `${CELLVIT_TRAINING_ROOT}`
   tokens; set `num_classes`, `label_map`, and `weight_list` to match the
   tissue's class distribution).
6. `pipeline/make_splits.py --tissue <tissue>` then
   `pipeline/train.sh <tissue>`.

## Tissue roadmap

The eventual target is the following 15 tissues. **breast** and
**colorectal** are currently set up; the rest are planned:

| Folder | Disease | Abbrev | Source |
|--------|---------|--------|--------|
| bone | Acute Lymphoblastic Leukemia | ALL | Standard hematology |
| brain | Brain Cancer / Glioblastoma | GBM | TCGA code |
| breast | Breast Invasive Carcinoma | BRCA | TCGA code |
| cervix | Cervical Cancer | CESC | TCGA code |
| colorectal | Colorectal Cancer | CRC | Universal |
| kidney | Renal Cell Carcinoma | RCC | Universal |
| liver | Hepatocellular Carcinoma | HCC | Universal |
| lung | Lung Cancer | LUAD | TCGA code |
| lymph_node | Lymph Node | LN | Anatomical |
| ovary | Ovarian Cancer | OV | TCGA code |
| pancreas | Pancreatic Ductal Adenocarcinoma | PDAC | Universal |
| prostate | Prostate Adenocarcinoma | PRAD | TCGA code |
| skin | Melanoma | SKCM | TCGA code |
| tonsil | Follicular Lymphoid Hyperplasia | FLH | Clinical |
| heart | Non-diseased | HRT | N/A |

## Detailed dataset spec

See [TRAININGSET_PREPARATION.md](TRAININGSET_PREPARATION.md) for the full
tile + label CSV format specification (input files, coordinate frames,
output directory layout, validation checklist).
