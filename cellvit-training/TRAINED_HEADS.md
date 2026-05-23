# Trained heads

Per-tissue CellViT-SAM-H-x40 heads produced by this training pipeline are
**not committed to this repository**. Each released head is multi-GB, and
they are versioned independently of the training source.

## Where to download

Pretrained WSInsight heads are published on the Hugging Face Hub at
<https://huggingface.co/huangch>. Look for repositories named
`<tissue>-CellViT-SAM-H-x40` (e.g. `breast-CellViT-SAM-H-x40`,
`colorectal-CellViT-SAM-H-x40`, `pantissue-CellViT-SAM-H-x40`).

The corresponding patch-level heads inherited from WSInfer are at
<https://huggingface.co/kaczmarj>.

## Currently trained heads

| Head | Classes | Backbone | Best val AUROC | Notes |
|------|---------|----------|----------------|-------|
| `pantissue-12cls-SAM-H-x40_2026-05-22` | 12 | CellViT-SAM-H-x40 | 0.852 | Pan-tissue head; `blast` split into `lymphoid`/`myeloid`/`other_stromal_mesenchymal`. See `trainingset/pantissue/label_map.yaml`. |

## Local promotion path

After `train.sh` finishes, the chosen head is promoted to
`cellvit-training/models/<head-name>/` containing:

- `model_best.pth` — checkpoint (git-ignored; publish to HF Hub separately)
- `config.yaml`    — training config used to produce it (tracked)
- `label_map.yaml` — int ↔ label-name mapping (tracked)

The `.pth` file is excluded by `.gitignore`; only the small yaml side-cars
are committed so the head is reproducible.

## Where the pipeline writes them

Local training runs write checkpoints under
`cellvit-training/cellvit/CellViT-plus-plus/logs_local/<run_id>/checkpoints/`
and W&B run logs under `.../logs_local/wandb/`. Both paths are git-ignored.

After training, copy the chosen `.pth` checkpoint into the WSInsight Model
Zoo (or push it to your own Hugging Face Hub repository) following the
contribution recipe in the WSInsight repository.
