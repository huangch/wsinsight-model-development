# wsinsight-patches/

WSInsight-specific patches to the vendored upstream
[`CellViT-plus-plus`](https://github.com/TIO-IKIM/CellViT-plus-plus) checkout
that lives next to this folder.

These patches are required to make CellViT-plus-plus work with the
WSInsight training pipeline (`pipeline/train.sh`, `pipeline/validate.sh`).
They are tracked here so the patches are backed up on GitHub even though
the `CellViT-plus-plus/` working tree is itself a separate git checkout
that points at the upstream remote.

## Contents

- `cellvit-plus-plus.patch` — `git diff` of the 4 modified files in the
  upstream checkout (small, ~14 LOC of net edits).
- `cellvit_convert_to_torchscript.py` — new script (988 LOC) that converts
  a trained `LinearClassifier` head checkpoint into a TorchScript module
  at 1024×1024. Invoked by `pipeline/train.sh` step 4 and lives at
  `cellvit/CellViT-plus-plus/cellvit/cellvit_convert_to_torchscript.py`
  in the working tree.

## Applying the patches to a fresh checkout

```bash
cd cellvit-training/cellvit/CellViT-plus-plus
git apply ../wsinsight-patches/cellvit-plus-plus.patch
cp ../wsinsight-patches/cellvit_convert_to_torchscript.py \
   cellvit/cellvit_convert_to_torchscript.py
```

## Regenerating the patch (after editing the upstream checkout)

```bash
cd cellvit-training/cellvit/CellViT-plus-plus
git diff > ../wsinsight-patches/cellvit-plus-plus.patch
cp cellvit/cellvit_convert_to_torchscript.py \
   ../wsinsight-patches/cellvit_convert_to_torchscript.py
```
