# pipeline/ — bash + python drivers for CellViT cell-classifier training

This directory holds every driver used to turn QuPath-exported tiles into a
trained CellViT cell-classification head. All scripts are tissue-agnostic.

## Naming convention

| Suffix | Scope | Examples |
|--------|-------|----------|
| `*_tissue.sh` | per-tissue operation | `train_tissue.sh`, `validate_tissue.sh` |
| `*_all_tissues.sh` | loops a per-tissue script across the dataset | `train_all_tissues.sh`, `audit_all_tissues.sh`, `export_all_tissues.sh` |
| `_lib.sh` | shared bash helpers (sourced, never executed) | `_lib.sh` |
| `*.py` | python utilities invoked from the shell drivers | `make_splits.py`, `audit_split_reuse.py` |

## Layered structure

```
┌──────────────────────────────────────────────────────────┐
│   driver scripts  (run by the user)                      │
│   ── train_all_tissues.sh                                │
│   ── audit_all_tissues.sh                                │
│   ── export_all_tissues.sh                               │
└────────────────────┬─────────────────────────────────────┘
                     │ loops & invokes per-tissue scripts
                     ▼
┌──────────────────────────────────────────────────────────┐
│   per-tissue scripts (operate on one tissue)             │
│   ── train_tissue.sh    (4-step train → ckpt → validate │
│                          → torchscript)                  │
│   ── validate_tissue.sh (re-run Step 3 only)             │
└────────────────────┬─────────────────────────────────────┘
                     │ both source _lib.sh
                     ▼
┌──────────────────────────────────────────────────────────┐
│   _lib.sh           (shared bash API)                    │
│   ── _lib::tissue_paths       _lib::log_comment          │
│   ── _lib::find_latest_run    _lib::run_validate         │
│   ── _lib::tissues_with_labels   _lib::tissues_with_splits │
└──────────────────────────────────────────────────────────┘
```

## File one-liners

| File | Purpose |
|------|---------|
| `_lib.sh` | Shared bash helpers (path resolution, tissue discovery, run lookup, validate wrapper). Sourced by every other `.sh` here. |
| `train_tissue.sh` | 4-step pipeline for ONE tissue: train head → locate `model_best.pth` → call `validate_tissue.sh` → TorchScript convert. |
| `validate_tissue.sh` | Re-run validation only (classification report + confusion matrices) for ONE tissue. Delegated to by `train_tissue.sh` Step 3. |
| `train_all_tissues.sh` | Loops splits → config → `train_tissue.sh` across every tissue with exported tiles. `--dry-run`, `--force`, `--tissues "a b c"` supported. |
| `audit_all_tissues.sh` | Loops `audit_split_reuse.py` across every tissue with a populated `splits/<fold>/`. Writes `audit_outputs/split_reuse_summary.csv`. |
| `export_all_tissues.sh` | Wrapper that drives the QuPath `export_tiles.groovy` batch (long-running, GPU-free). |
| `aggregate_pantissue.sh` | Builds `trainingset/pantissue/` by symlinking every per-tissue split. Verifies `label_map.yaml` MD5 matches across all tissues; auto-prefixes `<tissue>__` if a `SAMPLE_TAG` collision is detected. |
| `make_splits.py` | Slide-aware (`SAMPLE_TAG`-keyed) train/val split for ONE tissue. `--val-frac` defaults to 0.1. |
| `make_train_config.py` | Renders `trainingset/<tissue>/train_configs/<backbone>/<fold>.yaml` from the pantissue template, injecting class weights, `log_comment`, and resolved paths. |
| `compute_class_weights.py` | Tally `train/labels/*.csv` → inverse-frequency weights consumed by `make_train_config.py`. |
| `audit_split_reuse.py` | Per-tissue pixel/cell reuse check between train and val (catches overlap-driven leakage). |
| `validate_classifier.py` | Pure-python validation: emits per-class precision/recall/F1, AUROC, confusion matrix PDFs + JSON reports. Called by `_lib::run_validate`. Outputs: `classification_report.{txt,json}` + `confusion_matrix.{json,png,svg}`. |

## Common workflows

### Fresh dataset → all heads trained
```bash
bash export_all_tissues.sh        # 1. QuPath export (hours, no GPU)
bash audit_all_tissues.sh         # 2. sanity: ~0% reuse expected
bash aggregate_pantissue.sh       # 3. build trainingset/pantissue/
bash train_all_tissues.sh         # 4. loop per-tissue training
```

### Single tissue (manual)
```bash
python make_splits.py             --tissue colorectal --val-frac 0.1
python make_train_config.py       --tissue colorectal
bash   train_tissue.sh            colorectal
```

### Re-run validation against a finished training run
```bash
bash validate_tissue.sh           colorectal
# or pin an explicit run dir:
bash validate_tissue.sh           colorectal SAM-H-x40 fold_0 \
    /path/to/logs_local/2026-04-20T111253_colorectal-hne-sam-h-x40
```

## Bash helpers (`_lib.sh`) — public API

All helpers are namespaced `_lib::*`. None mutate global state; the only
function that "exports" data is `_lib::tissue_paths`, which echoes
`VAR=value` lines for `eval`.

| Function | Purpose |
|----------|---------|
| `_lib::pipeline_dir` | Absolute path to this directory. |
| `_lib::cellvit_training_root` | `<repo>/cellvit-training`. |
| `_lib::trainingset_root` | `<repo>/cellvit-training/trainingset`. |
| `_lib::cellvit_root` | `<repo>/cellvit-training/cellvit/CellViT-plus-plus`. |
| `_lib::logs_local` | Training-output base dir. |
| `_lib::templates_dir` | `<repo>/cellvit-training/templates`. |
| `_lib::python` | Resolves `$PYTHON` → wsinsight conda env → system `python3`. |
| `_lib::tissue_paths <t> [fold] [backbone]` | Echoes `TISSUE/FOLD/BACKBONE/TRAINSET/CONFIG/VAL_CSV/LABEL_MAP` for `eval "$(...)"`. |
| `_lib::log_comment <t> <task> <backbone>` | `"<t>-<task>-<backbone-lower>"`; the YAML field that determines the run-dir suffix. |
| `_lib::find_latest_run <log_comment>` | Newest `*_<log_comment>/` under `logs_local/`, or empty. |
| `_lib::run_validate <t> <fold> <backbone> <run_dir>` | Invokes `validate_classifier.py` with the four standard paths, tees log. |
| `_lib::tissues_with_labels [include\|exclude]` | Space-separated list of tissues with ≥1 `train/labels/*.csv`. Excludes `pantissue` by default. |
| `_lib::tissues_with_splits <fold> [include\|exclude]` | Tissues with a populated `splits/<fold>/val.csv`. |
| `_lib::tissues_in_dataset [include\|exclude]` | Every subdirectory of `trainingset/`. |

## Conventions

- **Anchoring.** Every script resolves paths from the location of
  `${BASH_SOURCE[0]}`, never from `$PWD`. You can `bash <abs path>/script.sh`
  from anywhere without `cd`-ing first.
- **Python interpreter.** Resolves via `_lib::python` — set `PYTHON=...`
  to override; defaults to `/opt/anaconda3/envs/wsinsight/bin/python3` if
  present, else system `python3`.
- **`pantissue` tissue.** A first-class tissue under `trainingset/pantissue/`
  built from symlinked per-tissue splits. Most loop drivers exclude it by
  default to avoid double-training (the per-tissue heads see the same data).
