# cellvit-training/cellvit/

Vendored upstream + base model weights. **Not committed to git** — both
items are large and freely re-downloadable. The pipeline expects this
layout to exist on disk after a clone.

```
cellvit-training/cellvit/
├── CellViT-plus-plus/         # upstream repo checkout (~few hundred MB)
└── models/
    └── CellViT-SAM-H-x40.pth  # base backbone (~2 GB)
```

## CellViT-plus-plus checkout

**Source:** <https://github.com/TIO-IKIM/CellViT-plus-plus>
**License:** see upstream repo (Apache-2.0 at time of writing).

```bash
cd cellvit-training/cellvit
git clone https://github.com/TIO-IKIM/CellViT-plus-plus.git
# Pin to the commit you want to use:
cd CellViT-plus-plus && git checkout <commit_or_tag> && cd ..
```

The training wrappers (`cellvit-training/pipeline/train.sh`,
`validate.sh`) point at `cellvit/CellViT-plus-plus/cellvit/...` for the
training/conversion entry points; nothing else in this repo depends on
upstream internals.

## CellViT-SAM-H-x40.pth (base weights)

**Source:** CellViT-plus-plus release page (HuggingFace / GitHub Releases).
Look for `CellViT-SAM-H-x40.pth` and place it at
`cellvit-training/cellvit/models/CellViT-SAM-H-x40.pth`.

Configs reference it via `${CELLVIT_TRAINING_ROOT}/cellvit/models/CellViT-SAM-H-x40.pth`
(see [`trainingset/<tissue>/train_configs/SAM-H-x40/fold_0.yaml`](../trainingset/)).

## Why this is not vendored

- ~few hundred MB upstream code + ~2 GB base weights — too large for a
  source-only repo.
- Upstream is actively maintained; pinning a commit hash here keeps the
  reference reproducible without forking.
