# CellViT++ Training Set Preparation (QuPath/Groovy)

This document is the complete specification for preparing a CellViT++ classifier
training set from 10x Xenium H&E data using QuPath. It is tissue-agnostic;
breast is used throughout as the running example.

The surrounding workflow (Python helpers, training wrappers, tissue config
format) is described in [README.md](README.md). This file is the format spec
for the dataset itself.

---

## 1. Overview

**Goal:** Export 1024×1024 px image tiles and matching cell-detection label CSVs
from H&E whole-slide images (Xenium `morphology.ome.tif`), with cell centroids
labelled by integer class id from the Xenium cell-type assignment.

**Model to train:** `CellViT-SAM-H-x40`
**Dataset class used in training code:** `DetectionDataset`
**Framework:** CellViT-plus-plus (`cellvit/CellViT-plus-plus/`)

---

## 2. Input Files (per Xenium sample)

All input files are under each sample's `outs/` folder.

| File | Description |
|---|---|
| `morphology.ome.tif` | Full H&E whole-slide image (OME-TIFF). **This is the image to open in QuPath.** Do NOT use `morphology_focus/` files (those are stain channels). |
| `cells.csv.gz` | Per-cell: `cell_id`, `x_centroid` (µm), `y_centroid` (µm), plus QC columns. Coordinates are in the Xenium tissue coordinate space (microns). |
| `analysis/clustering/gene_expression_graphclust/clusters.csv` | Per-cell: `Barcode` (= `cell_id`), `Cluster` (integer 1-based). |
| `celltype_assignment_hne_label.csv` | Per-cluster: `classification` (cluster int), `cell_type` (string). Used to map Cluster → cell type label. |

### Coordinate system
- `x_centroid` and `y_centroid` in `cells.csv.gz` are in **microns** in the Xenium slide coordinate space.
- QuPath, when the slide is opened with correct pixel calibration, also works in microns.
- To convert microns to pixels at the export resolution: `pixel = micron / MPP`.

---

## 3. Target Export Specification

| Parameter | Value |
|---|---|
| **Export MPP** | **0.25 µm/pixel** (40× equivalent) |
| **Tile size** | **1024 × 1024 pixels** |
| **Tile overlap** | **64 pixels** (on each edge, i.e. export stride = 960 px = 240 µm) |
| **Image format** | **PNG, RGB (3-channel)** |
| **Coordinate frame for labels** | Pixel coordinates **relative to the tile's top-left corner** at 0.25 MPP |

> **Why 0.25 MPP?** The CellViT-SAM-H-x40 network was trained at 0.25 µm/px. The
> inference pipeline enforces this — any other resolution requires rescaling and
> degrades performance.

---

## 4. Cell-Type Label Mapping (Breast)

### 4.1 Recommended label scheme: `hne_label` (11 classes)

This is the primary label set (`celltype_assignment_hne_label.csv`).
Use this for training unless stated otherwise. The integer assignment is
fixed and global — do not renumber per sample.

| Integer (`class_int`) | `cell_type` string |
|---|---|
| 0 | `lymphocyte` |
| 1 | `malignant_epithelial` |
| 2 | `epithelial` |
| 3 | `plasma_cell` |
| 4 | `macrophage_like` |
| 5 | `fibroblast_like` |
| 6 | `pericyte` |
| 7 | `endothelial` |
| 8 | `basophil` |
| 9 | `adipocyte` |
| 10 | `mast_cell` |

The same mapping is the source of truth in
[`tissue_configs/breast.yaml`](tissue_configs/breast.yaml) (`label_map:`)
and is mirrored to `trainingset/breast/label_map.yaml` by
`build_cell_labels.py`. Per-class frequencies (used to set `weight_list:`)
are computed across the actual exported tiles and recorded as comments in
[`trainingset/breast/train_configs/SAM-H-x40/fold_0.yaml`](trainingset/breast/train_configs/SAM-H-x40/fold_0.yaml).

> Class integers must be **0-indexed** and **consecutive**. The value in the label
> CSV is always the integer, not the string.

### 4.2 Alternative label scheme: `pannuke_label` (4 classes)

Provided in `celltype_assignment_pannuke_label.csv`. Compatible with the existing
`checkpoints/classifier/sam-h/pannuke.pth` classifier head.

| Integer | `cell_type` string | Breast frequency |
|---|---|---|
| 0 | `neoplastic` | 22.7% |
| 1 | `inflammatory` | 43.6% |
| 2 | `connective` | 18.1% |
| 3 | `epithelial` | 15.6% |

### 4.3 Full label inventory per sample (breast, `hne_label`)

The union of all cell type labels seen across all 8 breast Xenium samples:

```
adipocyte, basophil, endothelial, epithelial, fibroblast_like,
lymphocyte, macrophage_like, malignant_epithelial, mast_cell,
pericyte, plasma_cell
```

Not every label appears in every sample. The label map integer assignment
above (Section 4.1) is **fixed and global** — do not renumber per-sample.

---

## 5. Label Join Pipeline (Python pre-processing)

Before QuPath exports the label CSVs, build a `cell_id → (x_um, y_um, class_int)`
lookup table from the three source files. This is implemented by
[`pipeline/build_cell_labels.py`](pipeline/build_cell_labels.py):

```bash
python pipeline/build_cell_labels.py --tissue breast
python pipeline/build_cell_labels.py --tissue colorectal
python pipeline/build_cell_labels.py --config path/to/custom_tissue.yaml
```

It reads the tissue's [`tissue_configs/<tissue>.yaml`](tissue_configs/) for
the sample list and `label_map`, joins the three Xenium files per sample,
and emits one flat CSV per sample (`cell_id, x_um, y_um, class_int`) plus
the canonical `trainingset/<tissue>/label_map.yaml`. The `${PROJECT_ROOT}`
token in the config is expanded automatically.

---

## 6. Output Directory Structure

```
trainingset/breast/
├── train/
│   ├── images/
│   │   ├── breast_5k_tile_0042.png
│   │   ├── breast_5k_tile_0043.png
│   │   └── ...
│   └── labels/
│       ├── breast_5k_tile_0042.csv
│       ├── breast_5k_tile_0043.csv
│       └── ...
├── test/
│   ├── images/
│   └── labels/
├── splits/
│   └── fold_0/
│       ├── train.csv
│       └── val.csv
└── label_map.yaml
```

**The `split` directory name** (`train` or `test`) is passed to `DetectionDataset`
as the `split` parameter in the training config.

---

## 7. Tile Image Format

- **Format:** PNG
- **Color:** RGB (3 channels). If the OME-TIFF has more channels, export only the H&E channel composite (or channel 0 as RGB).
- **Size:** exactly 1024 × 1024 pixels
- **Bit depth:** 8-bit per channel (uint8)
- **MPP at export:** 0.25 µm/px

### Tile naming convention

```
{sample_tag}_tile_{tile_index:04d}.png
```

Examples of `sample_tag` for the 8 breast samples:

| Sample folder | Suggested `sample_tag` |
|---|---|
| `FFPE Human Breast Cancer with 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes` | `breast_5k` |
| `FFPE Human Breast using the Entire Sample Area/Replicate 1` | `breast_esa_r1` |
| `FFPE Human Breast using the Entire Sample Area/Replicate 2` | `breast_esa_r2` |
| `FFPE Human Breast with Custom Add-on Panel/Tissue sample 1` | `breast_custom_s1` |
| `FFPE Human Breast with Custom Add-on Panel/Tissue sample 2` | `breast_custom_s2` |
| `FFPE Human Breast with Pre-designed Panel/Tissue sample 1` | `breast_pre_s1` |
| `FFPE Human Breast with Pre-designed Panel/Tissue sample 2` | `breast_pre_s2` |
| `Xenium FFPE Human Breast with Custom Add-on Panel/Tissue sample 1 (IDC)` | `breast_idc_s1` |

The tile image and its label CSV must share the **same stem** (filename without
extension).

---

## 8. Label CSV Format

One CSV file per tile. **No header row.** Three columns: `x, y, class_int`.

```
x_pixel, y_pixel, class_int
```

- `x_pixel`: integer, column of the cell centroid **within this tile** in pixels at 0.25 MPP. Origin = tile top-left corner. Range: [0, 1023].
- `y_pixel`: integer, row of the cell centroid **within this tile** in pixels at 0.25 MPP.
- `class_int`: integer class index from Section 4.1.

**Example `breast_5k_tile_0042.csv`:**
```
46,7,2
191,100,0
108,191,1
146,173,5
233,117,4
```

**Rules:**
- Cells that fall outside [0, 1023] in either axis must be **excluded** from that tile's CSV.
- If the tile has **zero cells**, still export an empty image file, but **do not create a label CSV** (or create an empty one — the DataLoader skips tiles not in the filelist anyway).
- Cells at tile boundaries (overlap region): include a cell in a tile if its centroid falls in [tile_x_start, tile_x_start + 1024) and [tile_y_start, tile_y_start + 1024) at 0.25 MPP.

### Coordinate conversion (Xenium µm → tile pixel)

```
tile_origin_x_um = tile_col_index * stride_um    # stride_um = 240.0 µm (960 px × 0.25)
tile_origin_y_um = tile_row_index * stride_um

x_pixel = round((cell_x_um - tile_origin_x_um) / 0.25)
y_pixel = round((cell_y_um - tile_origin_y_um) / 0.25)
```

---

## 9. Splits Files

`splits/fold_0/train.csv` and `splits/fold_0/val.csv` are plain text files, one
tile stem per line, **no header**:

```
breast_5k_tile_0001
breast_5k_tile_0002
breast_5k_tile_0005
...
```

**Split strategy:**
- Allocate ~80% of tiles to `train.csv`, ~20% to `val.csv`.
- Split by **spatial region** (e.g. top 80% of slide rows → train, bottom 20% → val), not randomly, to avoid tile overlap leakage from the 64 px overlap.
- Keep tiles from different samples together in the same pool (no per-sample split).

---

## 10. Label Map File

`label_map.yaml` — a YAML file at the dataset root:

```yaml
0: "lymphocyte"
1: "malignant_epithelial"
2: "epithelial"
3: "plasma_cell"
4: "macrophage_like"
5: "fibroblast_like"
6: "pericyte"
7: "endothelial"
8: "basophil"
9: "adipocyte"
10: "mast_cell"
```

---

## 11. Training Config Template

After the dataset is prepared, training is launched with the tissue-agnostic
wrapper:

```bash
bash pipeline/train.sh breast               # SAM-H-x40, fold_0
bash pipeline/train.sh breast SAM-H-x40 fold_0
```

The wrapper exports `${PROJECT_ROOT}` and `${CELLVIT_TRAINING_ROOT}`,
materializes the tokenized YAML at
`trainingset/<tissue>/train_configs/<backbone>/.<fold>.resolved.yaml` via
`envsubst`, then invokes `cellvit/train_cell_classifier_head.py`.

The authoring template (used by both breast and colorectal) is:

```yaml
logging:
  mode: offline
  project: cellvit-pan-tissue
  notes: breast-hne-label-11class-SAM-H-x40
  log_comment: breast-hne-sam-h-x40         # MUST be "<tissue>-hne-<backbone-lower>"
  wandb_dir: ${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local
  log_dir:   ${CELLVIT_TRAINING_ROOT}/cellvit/CellViT-plus-plus/logs_local
  level: Debug

random_seed: 42
gpu: 0

data:
  dataset: DetectionDataset
  dataset_path:   ${CELLVIT_TRAINING_ROOT}/trainingset/breast
  normalize_stains_train: false
  normalize_stains_val: false
  num_classes: 11
  train_filelist: ${CELLVIT_TRAINING_ROOT}/trainingset/breast/splits/fold_0/train.csv
  val_filelist:   ${CELLVIT_TRAINING_ROOT}/trainingset/breast/splits/fold_0/val.csv
  label_map:
    0: lymphocyte
    1: malignant_epithelial
    2: epithelial
    3: plasma_cell
    4: macrophage_like
    5: fibroblast_like
    6: pericyte
    7: endothelial
    8: basophil
    9: adipocyte
    10: mast_cell

cellvit_path: ${CELLVIT_TRAINING_ROOT}/cellvit/models/CellViT-SAM-H-x40.pth

model:
  hidden_dim: 256

training:
  cache_cell_dataset: true
  batch_size: 64
  epochs: 50
  drop_rate: 0.1
  optimizer: AdamW
  optimizer_hyperparameter:
    betas: [0.85, 0.9]
    lr: 0.0003
    weight_decay: 0.00002
  early_stopping_patience: 20
  mixed_precision: true
  eval_every: 1
  weighted_sampling: true        # important: class imbalance is severe
  # Inverse-frequency weights (weight ≈ 10 / class_percent), capped at 10.
  # Re-derive per tissue from the actual exported-tile class distribution.
  weight_list: [1.1, 0.3, 0.4, 4.7, 1.4, 0.5, 4.1, 3.4, 7.7, 10.0, 8.0]
  scheduler:
    scheduler_type: exponential
    gamma: 0.95
```

Notes:
- All paths use `${CELLVIT_TRAINING_ROOT}` (a sibling of the project root) so
  the tree is portable across folder renames.
- `log_comment` must equal `<tissue>-hne-<backbone-lower>`; the wrapper uses
  this string to glob for the run directory after training finishes.
- `num_classes` and `label_map` must match `tissue_configs/<tissue>.yaml`
  exactly (the same `label_map` is also written to
  `trainingset/<tissue>/label_map.yaml` by `build_cell_labels.py`).

---

## 12. QuPath Groovy Script Requirements

The Groovy script must perform the following steps:

### 12.1 Inputs
- Path to `morphology.ome.tif` (opened as the current QuPath project entry)
- Path to the pre-built cell label CSV: `cell_id, x_um, y_um, class_int`
- Output directory root (e.g. `trainingset/breast/`)
- `sample_tag` string (e.g. `breast_5k`)
- Split: `"train"` or `"test"`

### 12.2 Export parameters
```
EXPORT_MPP    = 0.25       // µm/px
TILE_SIZE_PX  = 1024       // pixels
OVERLAP_PX    = 64         // pixels on each side
STRIDE_PX     = 960        // = TILE_SIZE_PX - OVERLAP_PX
STRIDE_UM     = 240.0      // = STRIDE_PX * EXPORT_MPP
TILE_SIZE_UM  = 256.0      // = TILE_SIZE_PX * EXPORT_MPP
```

### 12.3 Tile export loop (pseudocode)
```
for row_index in 0..n_rows:
    for col_index in 0..n_cols:
        tile_x_um = col_index * STRIDE_UM
        tile_y_um = row_index * STRIDE_UM

        // Export image region
        region = RegionRequest(tile_x_um, tile_y_um, TILE_SIZE_UM, TILE_SIZE_UM, downsample)
        image = server.readRegion(region)  // rendered as RGB
        save(image, "{output_dir}/{split}/images/{sample_tag}_tile_{idx:04d}.png")

        // Find cells in this tile
        cells_in_tile = [c for c in cells
                         if tile_x_um <= c.x_um < tile_x_um + TILE_SIZE_UM
                         and tile_y_um <= c.y_um < tile_y_um + TILE_SIZE_UM]

        if len(cells_in_tile) == 0: continue  // skip empty tiles

        // Write label CSV
        with open("{output_dir}/{split}/labels/{sample_tag}_tile_{idx:04d}.csv", "w") as f:
            for c in cells_in_tile:
                px = round((c.x_um - tile_x_um) / EXPORT_MPP)
                py = round((c.y_um - tile_y_um) / EXPORT_MPP)
                f.write(f"{px},{py},{c.class_int}\n")
```

### 12.4 QuPath downsample factor
QuPath uses `downsample` = `EXPORT_MPP / slide_pixel_size_um`.
Get the slide pixel size from `server.getPixelCalibration().getPixelWidthMicrons()`.

```groovy
double exportMPP = 0.25
double slideMPP  = server.getPixelCalibration().getPixelWidthMicrons()
double downsample = exportMPP / slideMPP
```

### 12.5 Tissue masking (recommended)
Only export tiles that contain tissue. A simple check: skip tiles where the mean
pixel brightness > 240 (mostly white background).

### 12.6 Minimum cell count per tile (recommended)
Skip tiles with fewer than **5 cells** (after filtering to assigned cells).
These contribute little signal and inflate the dataset with near-empty labels.

---

## 13. Validation Checklist

Before handing the dataset to training, verify:

- [ ] All PNG files are exactly 1024×1024 px, RGB, uint8
- [ ] Every PNG in `train/images/` has a matching CSV in `train/labels/` with the same stem
- [ ] All CSV files have no header row, exactly 3 integer columns per row
- [ ] All `x_pixel` and `y_pixel` values are in range [0, 1023]
- [ ] All `class_int` values are in range [0, 10]
- [ ] `splits/fold_0/train.csv` and `val.csv` contain only stems that exist in `train/images/`
- [ ] `label_map.yaml` is present at the dataset root
- [ ] No tile stem appears in both train and val

---

## 14. Reference Implementations

- Python label-join: [`pipeline/build_cell_labels.py`](pipeline/build_cell_labels.py)
- QuPath tile/label export: [`qupath/export_tiles.groovy`](qupath/export_tiles.groovy)
- Train/val splits: [`pipeline/make_splits.py`](pipeline/make_splits.py)
- Tissue YAML examples: [`tissue_configs/breast.yaml`](tissue_configs/breast.yaml),
  [`tissue_configs/colorectal.yaml`](tissue_configs/colorectal.yaml)
- Worked training configs:
  [`trainingset/breast/train_configs/SAM-H-x40/fold_0.yaml`](trainingset/breast/train_configs/SAM-H-x40/fold_0.yaml),
  [`trainingset/colorectal/train_configs/SAM-H-x40/fold_0.yaml`](trainingset/colorectal/train_configs/SAM-H-x40/fold_0.yaml)

