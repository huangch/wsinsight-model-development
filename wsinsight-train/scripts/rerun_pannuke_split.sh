#!/usr/bin/env bash
# Re-run PanNuke training from the split stage, reusing the existing tiles.
# Usage: scripts/rerun_pannuke_split.sh
set -euo pipefail

ROOT=/workspace/wsinsight/wsinsight-model-development

export PATH=/opt/anaconda3/envs/wsinsight/bin:$PATH
export LD_LIBRARY_PATH=/opt/anaconda3/envs/wsinsight/lib
export PYTHONUNBUFFERED=1
export CELLVIT_ROOT="$ROOT/cellvit-training/cellvit/CellViT-plus-plus"
export CELLPOSE_LOCAL_MODELS_PATH=/workspace/.cellpose
export TORCH_HOME=/workspace/.torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$ROOT"

CFG="$(mktemp -t wsitrain-pannuke-XXXXXX.yaml)"
trap 'rm -f "$CFG"' EXIT
printf 'by_slide: true\n' > "$CFG"

# --force because the change lives in the train config template, which the
# manifest does not track.
wsitrain run \
  --input "$ROOT/data/xenium" \
  --tissue pantissue \
  --task pannuke \
  --config "$CFG" \
  --transform affine \
  --output "$ROOT/models/pannuke" \
  --from split --to report --force \
  --tune 0 --gpus auto

echo "Done. Report: $ROOT/models/pannuke/report/pantissue/"
