# Trained heads

Per-tissue CellViT-SAM-H-x40 heads produced by this training pipeline are
**not committed to this repository**. Each released head is multi-GB, and
they are versioned independently of the training source.

## Where to download

Pretrained WSInsight heads are published on the Hugging Face Hub at
<https://huggingface.co/huangch>. Look for repositories named
`<tissue>-CellViT-SAM-H-x40` (e.g. `breast-CellViT-SAM-H-x40`,
`colorectal-CellViT-SAM-H-x40`).

The corresponding patch-level heads inherited from WSInfer are at
<https://huggingface.co/kaczmarj>.

## Where the pipeline writes them

Local training runs write checkpoints under
`cellvit-training/cellvit/CellViT-plus-plus/logs_local/<run_id>/checkpoints/`
and W&B run logs under `.../logs_local/wandb/`. Both paths are git-ignored.

After training, copy the chosen `.pth` checkpoint into the WSInsight Model
Zoo (or push it to your own Hugging Face Hub repository) following the
contribution recipe in the WSInsight repository.
