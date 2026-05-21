# cellvit-training/

H&E classifier-head training workflow that fine-tunes a CellViT-SAM-H-x40
backbone using QuST-derived Xenium cell-type labels. Produces the per-tissue
TorchScript heads consumed by WSInsight's Model Zoo.

The workflow is **tissue-agnostic**: anything that varies per tissue lives in
and [`trainingset/<tissue>/label_map.yaml`](trainingset/). The same wrappers train every tissue.

## Layout

```
cellvit-training/
├── pipeline/                # tissue-agnostic Python + bash drivers
│   ├── make_splits.py       # exported tiles → train/val splits
│   ├── train.sh             # 4-step training wrapper
│   ├── validate.sh          # re-run validation against a finished run
│   └── validate_classifier.py
├── qupath/                  # QuPath Groovy helpers (CLI-batch capable)
│   ├── load_mapping.groovy  # cluster_id → label-name on detections
│   └── export_tiles.groovy  # detections → tile PNG + per-tile CSV
├── trainingset/             # exported tiles + per-fold training configs
│   └── <tissue>/
│       ├── label_map.yaml   # canonical int ↔ label-name table
│       ├── train/{images,labels}/
│       ├── splits/fold_0/{train,val}.txt
│       └── train_configs/SAM-H-x40/fold_0.yaml
├── cellvit/                 # vendored CellViT-plus-plus + base weights
│   ├── CellViT-plus-plus/
│   └── models/CellViT-SAM-H-x40.pth
├── models/legacy/           # archived classifier checkpoints
└── TRAININGSET_PREPARATION.md
```

The QuPath project that holds the per-sample annotations (foreground,
StarDist nuclei, Xenium cluster IDs) lives at `../data/qprj/project.qpproj`.
It's the single source of truth for the H&E-anchored cell labels consumed
by `export_tiles.groovy`.

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

For a tissue named `<tissue>` (e.g. `pantissue`):

```bash
# 0. (one-time) curate cell-type labels with kurtorank to produce
#    celltype_assignment_pantissue_label.csv in each sample's outs/.
#    Hand-author trainingset/<tissue>/label_map.yaml (int ↔ label-name).

# 1. (in QuPath) For each H&E image in data/qprj/project.qpproj:
#      a. QuST → PetesSimpleTissueDetection (foreground annotation)
#      b. QuST → StarDistCellNucleusDetection (nuclei in tissue mask)
#      c. QuST → XeniumAnnotation (assign Xenium cluster_id to each detection)
#    Then save the project.

# 2. Remap cluster_id → pantissue label on every detection (CLI batch):
QuPath script -s -p ../data/qprj/project.qpproj \
    -a /abs/path/celltype_assignment_pantissue_label.csv \
    qupath/load_mapping.groovy

# 3. Export tiles + per-tile cell CSVs (CLI batch):
#    Edit OUTPUT_ROOT at top of export_tiles.groovy first.
QuPath script -p ../data/qprj/project.qpproj qupath/export_tiles.groovy

# 4. Build train/val splits from the exported tiles
python pipeline/make_splits.py --tissue <tissue>

# 5. Train head + auto-validate + export TorchScript (4 steps in one)
bash pipeline/train.sh <tissue>                       # SAM-H-x40, fold_0
bash pipeline/train.sh <tissue> SAM-H-x40 fold_0 <task>   # explicit form

# 6. Re-run only validation against a finished training run
bash pipeline/validate.sh <tissue>
```

`train.sh` runs four steps in sequence: train the `LinearClassifier` head,
locate `model_best.pth` under
`cellvit/CellViT-plus-plus/logs_local/<TIMESTAMP>_<tissue>-<task>-<backbone>/checkpoints/`,
emit a confusion matrix + classification report, and convert the checkpoint
to TorchScript at 1024×1024.

## Adding a new tissue

1. Create `trainingset/<tissue>/label_map.yaml` (int ↔ label-name, 0..N-1
   contiguous). This is the canonical mapping.
2. Add the tissue's H&E images to `../data/qprj/project.qpproj`.
3. Curate labels in QuPath (QuST tissue detection → StarDist → XeniumAnnotation →
   `load_mapping.groovy` with the sample's
   `celltype_assignment_<label_col>.csv`).
4. Set `OUTPUT_ROOT` at the top of `qupath/export_tiles.groovy` to
   `trainingset/<tissue>/`, then run it via the QuPath CLI.
5. Author `trainingset/<tissue>/train_configs/SAM-H-x40/fold_0.yaml`
   (copy from an existing tissue; set `num_classes`, `label_map`, and
   `weight_list` to match the tissue's class distribution).
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
