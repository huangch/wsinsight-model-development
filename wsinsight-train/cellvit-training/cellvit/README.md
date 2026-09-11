# cellvit-training/cellvit/

Vendored upstream + base model weights. **Not committed to git** — both
items are large and freely re-downloadable. The pipeline expects this
layout to exist on disk after a clone.

```
cellvit-training/cellvit/
├── CellViT-plus-plus/         # upstream checkout, tracked as a git submodule
└── models/                    # 7 pretrained backbones, 22 GB; see models/README.md
    └── CellViT-SAM-H-x40.pth  # the only one the training configs need (~2.8 GB)
```

## CellViT-plus-plus checkout

**Source:** <https://github.com/TIO-IKIM/CellViT-plus-plus>
**License:** see upstream repo (Apache-2.0 at time of writing).

This is a git submodule, so a clone of the parent repo already pins the commit:

```bash
git submodule update --init --recursive
```

The training wrappers (`cellvit-training/pipeline/train_tissue.sh`,
`validate_tissue.sh`) point at `cellvit/CellViT-plus-plus/cellvit/...` for the
training/conversion entry points; nothing else in this repo depends on
upstream internals.

## CellViT-SAM-H-x40.pth (base weights)

Download location, the full seven-file inventory, and sha256 checksums are in
[`models/README.md`](models/README.md). Place the file at
`cellvit-training/cellvit/models/CellViT-SAM-H-x40.pth`.

Configs reference it via `${CELLVIT_TRAINING_ROOT}/cellvit/models/CellViT-SAM-H-x40.pth`
(see [`trainingset/<tissue>/train_configs/SAM-H-x40/fold_0.yaml`](../trainingset/)).

## Why this is not vendored

- ~few hundred MB upstream code + ~2 GB base weights — too large for a
  source-only repo.
- Upstream is actively maintained; pinning a commit hash here keeps the
  reference reproducible without forking.
