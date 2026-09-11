# cellvit/models/ — pretrained CellViT backbones (not in git)

This directory holds the **pretrained CellViT nuclei segmentation/classification
backbones** that `wsitrain` fine-tunes a cell-type head on top of. The weights
were trained upstream on **PanNuke**; nothing here is produced by this repo.

**The `.pth` files are not committed** — 22 GB total, far past what a Git remote
will take. They are matched by the `models/` rule in the repo root
[`.gitignore`](../../../../.gitignore); only this README is allowlisted, so the
directory still materialises on a fresh clone with an explanation of what is
missing. After cloning you must download the weights yourself before training.

## What belongs here

| File | Bytes | sha256 (first 16) | Used by this repo |
|---|---:|---|---|
| `CellViT-SAM-H-x40.pth` | 2,799,315,941 | `b324c10fddb0f80f` | **yes — every training config** |
| `CellViT-SAM-H-x40-AMP.pth` | 2,799,319,650 | `356418f19d9d478f` | no |
| `CellViT-SAM-H-x20.pth` | 2,799,315,877 | `5d1bf9c7f2b5651a` | no |
| `CellViT-Virchow-x40-AMP.pth` | 2,776,227,988 | `645e1eb3f37cc9e8` | no |
| `CellViT-256-x40.pth` | 187,224,155 | `ee3986922fc50035` | no |
| `CellViT-256-x40-AMP.pth` | 187,226,632 | `c58d356c92d3fd21` | no |
| `CellViT-256-x20.pth` | 187,224,155 | `457d25eca225cbe0` | no |

`CellViT-256-x20.pth` and `CellViT-256-x40.pth` are byte-identical in size but
are different files; compare the hash, not `ls -l`.

**Only `CellViT-SAM-H-x40.pth` is required.** It is the `cellvit_path` in every
`train_configs/SAM-H-x40/fold_0.yaml` and in every archived run `config.yaml`
under the top-level `models/`. The other six are kept for backbone comparisons
and can be skipped on a fresh install.

Naming: `SAM-H` / `256` is the encoder (SAM ViT-H vs ViT-256), `x40` / `x20` the
magnification the model was trained at, `-AMP` a mixed-precision variant.

### `jit/`

Also ignored, and also optional — TorchScript exports of the same seven models
(`*.torchscript_optimized.pt` plus a `.json` and a `.graph.txt` per model, ~22 GB).
Used for inference without the Python model definition. The training path does
not read them; regenerate them from the `.pth` files if needed.

## Where to download

Upstream publishes the checkpoints on Google Drive. From the
[CellViT++ README](../CellViT-plus-plus/README.md) ("Model Checkpoints"):

> Checkpoints can be downloaded here from
> [Google-Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing).
> […] Unfortunately, we cannot share all checkpoints due to their license.

That caveat is why the table above records sizes and hashes: some files may not
be obtainable from the public folder, and the hash is the only way to confirm a
copy sourced elsewhere is the same one these runs used.

Projects:

- CellViT++ (current, vendored at `../CellViT-plus-plus/`) —
  <https://github.com/TIO-IKIM/CellViT-plus-plus>
- CellViT (original, where these PanNuke backbones come from) —
  <https://github.com/TIO-IKIM/CellViT>

Place the files directly in this directory, unrenamed:

```
cellvit-training/cellvit/models/CellViT-SAM-H-x40.pth
```

Verify before training:

```sh
sha256sum CellViT-SAM-H-x40.pth   # expect b324c10fddb0f80f...
```

## If the path is wrong

The training scripts resolve this directory from `$CELLVIT_ROOT`, which
`scripts/train_*.sh` default to `<wsinsight-train>/cellvit-training/cellvit/CellViT-plus-plus`.
A missing backbone surfaces as a `FileNotFoundError` on `cellvit_path` at the
start of the train stage, not as a silent fallback.
