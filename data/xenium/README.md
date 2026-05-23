# data/xenium/

Raw 10x Xenium output bundles, organized one folder per tissue.
**Not committed to git** (~1.7 TB). See [../README.md](../README.md) for the
overall data policy.

## Source

**10x Genomics — Xenium Datasets**: <https://www.10xgenomics.com/datasets?menu%5Bproducts.name%5D=Xenium>

License: [10x Genomics License](https://www.10xgenomics.com/legal/end-user-software-license-agreement) (free for non-commercial research).

## Per-sample manifests

Every tissue folder contains a `SOURCES.yaml` listing every sample with
its 10x dataset name, the `he_image` stem, and the on-disk relative path.
These files are the canonical machine-readable sample list. The QuPath
project at `data/qprj/project.qpproj` is the runtime manifest — every
image opened in it becomes an entry the headless wrappers iterate over.
The int ↔ label-name mapping consumed at training time lives in
[`cellvit-training/trainingset/<tissue>/label_map.yaml`](../../cellvit-training/trainingset/).

| Tissue | Samples |
|--------|--------:|
| [bone](bone/SOURCES.yaml)             | 3 |
| [brain](brain/SOURCES.yaml)           | 1 |
| [breast](breast/SOURCES.yaml)         | 8 |
| [cervix](cervix/SOURCES.yaml)         | 1 |
| [colorectal](colorectal/SOURCES.yaml) | 3 |
| [heart](heart/SOURCES.yaml)           | 1 |
| [kidney](kidney/SOURCES.yaml)         | 3 |
| [liver](liver/SOURCES.yaml)           | 2 |
| [lung](lung/SOURCES.yaml)             | 4 |
| [lymph_node](lymph_node/SOURCES.yaml) | 1 |
| [ovary](ovary/SOURCES.yaml)           | 2 |
| [pancreas](pancreas/SOURCES.yaml)     | 3 |
| [prostate](prostate/SOURCES.yaml)     | 1 |
| [skin](skin/SOURCES.yaml)             | 5 |
| [tonsil](tonsil/SOURCES.yaml)         | 2 |

The currently trained head (`pantissue`) consumes every sample listed
above. To add a new tissue to the pipeline, follow the parent
[README](../../cellvit-training/README.md#adding-a-new-tissue).

## Tissue roadmap

| Folder | Disease | Abbrev | Source convention |
|--------|---------|--------|--------|
| bone | Acute Lymphoblastic Leukemia | ALL | Standard hematology |
| brain | Brain Cancer / Glioblastoma | GBM | TCGA code |
| breast | Breast Invasive Carcinoma | BRCA | TCGA code |
| cervix | Cervical Cancer | CESC | TCGA code |
| colorectal | Colorectal Cancer | CRC | Universal |
| heart | Non-diseased | HRT | N/A |
| kidney | Renal Cell Carcinoma | RCC | Universal |
| liver | Hepatocellular Carcinoma | HCC | Universal |
| lung | Lung Cancer | LUAD | TCGA code |
| lymph_node | Lymph Node | LN | Anatomical |
| ovary | Ovarian Cancer | OV | TCGA code |
| pancreas | Pancreatic Ductal Adenocarcinoma | PDAC | Universal |
| prostate | Prostate Adenocarcinoma | PRAD | TCGA code |
| skin | Melanoma | SKCM | TCGA code |
| tonsil | Follicular Lymphoid Hyperplasia | FLH | Clinical |

## On-disk layout per sample

```
data/xenium/<tissue>/<10x_dataset_name>/
├── <he_image_stem>.ome.tif      # full-resolution H&E (whole-slide)
└── outs/
    ├── cells.csv.gz                                                     # required
    ├── analysis/clustering/gene_expression_graphclust/clusters.csv     # required
    ├── celltype_assignment_<tissue>_label.csv                       # produced by `kurtorank annotate`
    └── ...                                                              # other standard 10x outputs
```

## Fetch recipe

```bash
# Example: one breast sample
TISSUE=breast
DATASET="FFPE Human Breast Cancer with 5K Human Pan Tissue and Pathways Panel plus 100 Custom Genes"
mkdir -p "data/xenium/${TISSUE}/${DATASET}"
cd       "data/xenium/${TISSUE}/${DATASET}"

# 1. Download the H&E and outs.zip from the 10x dataset page (see SOURCES.yaml).
#    Replace <URL_*> with the URL listed on the 10x page for this dataset.
curl -L -o "Xenium_Prime_Breast_Cancer_FFPE_he_image.ome.tif"  "<URL_HE>"
curl -L -o outs.zip                                            "<URL_OUTS>"

# 2. Unzip the outs/ bundle in place.
unzip outs.zip && rm outs.zip

# 3. (per sample) annotate cell types — produces outs/celltype_assignment_<tissue>_label.csv
kurtorank annotate --xenium-dir outs/ --tissue-type ${TISSUE}
```

The 10x download URLs are not pinned here because 10x rotates CDN paths;
look up each sample by its dataset name on the 10x Datasets page (linked
above).

