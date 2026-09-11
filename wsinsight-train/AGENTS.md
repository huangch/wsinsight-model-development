# wsitrain (wsinsight-train) — Agent Guide

Headless CLI that trains WSInsight cell-classification heads from paired
Xenium + H&E. Produces a `models/<tissue>/main/` folder wsinsight loads with
`--zoo-model-dir`. Python >=3.11.

## Environment (read first)

- `bash ./conda-setup.sh wsitrain -r` installs torch + cellpose + kurtorank
  + wsitrain. It installs kurtorank's dep tree explicitly before the editable
  install; do not "simplify" that back to `--no-deps`.
- **The CellViT path needs `$CELLVIT_ROOT`** pointing at a CellViT-plus-plus
  checkout. `train` and `export` raise without it. The crop-classifier path
  does not use it at all.
- StarDist runs on TensorFlow, so it needs a PTX toolchain (`ptxas` +
  `libdevice`). Without one, `segment` fails with "No PTX compilation provider
  is available"; `--stardist-cpu` is the escape hatch, or install
  `nvidia-cuda-nvcc-cu12`.
- Tests that spawn children (`tqdmshim`) only pass with the package
  **installed**: `subproc.child_env` resets `PYTHONPATH`, so a
  `PYTHONPATH=.` test run cannot import wsitrain in the child.

## Two model shapes, one pipeline

`--object-detection` is the switch, matching the field wsinsight reads out of
`config.json`:

| | `end2end` (default) | `stardist` |
|---|---|---|
| cutting stage | `tile` (1024 px tiles) | `crop` (one crop per cell) |
| trainer | CellViT via `$CELLVIT_ROOT` | any `torchvision` classifier |
| exported config | `object_detection: {name: end2end}` | `{name: stardist, ...}` |

The stage that does not apply returns `{"skipped": ...}`; `dag._REQUIRED_OUTPUT`
must keep tolerating that or the pipeline fails on the unused branch.

`--architecture`, `--patch-size-pixels`, `--patch-spacing-um-px` and
`--stain-normalization` have **no defaults on purpose**: they define the
deployed model's geometry, and a wrong guess trains a model that scores well
here and misclassifies under wsinsight. `RunConfig.__post_init__` refuses to
build without them.

## Where training cells come from

`--nuclei-source` is orthogonal to `--object-detection`:

| | `xenium-coords` (default) | `he-mask` |
|---|---|---|
| `segment` | early-returns `{"skipped": ...}` | runs the segmenter |
| nucleus id | the Xenium `cell_id` (a string, never cast to int) | mask label under the projected centroid |
| `--segmenter` | **not passed at all** by `scripts/` | honoured |
| cost | seconds | ~25 min GPU per cohort |

Because `segment` is skipped, `prereq` must not demand it on this path, and the
multi-GPU `segment` fan-out in `dag.run` only engages for `he-mask`.

## Stages

`annotate → segment → transfer → tile | crop → split → train → validate →
export → report`. Each is also a command. Completed stages are skipped on
re-run (manifest-based); settings are inherited from the `run-<tissue>.yaml`
the previous command wrote into `--output`.

**Adding a stage touches eight places** — miss one and a test fails:
`__init__.STAGES`, `stages.STAGE_FUNCS`, `stages._STAGE_WIPES`,
`cli._STAGE_HELP`, `cli._STAGE_FLAGS`, `prereq.PREREQUISITES` (+ `_ARTIFACTS`),
`defaults/run.yaml`, and `tests/test_stage_commands.EXPECTED_FLAGS`.
`defaults/run.yaml` is the source of the provenance map, so a field missing
there `KeyError`s `_print_config`.

## Re-running: `--redo-<stage>` and the wipe rules

Three invariants, each of which has already been violated once:

1. **A wipe deletes only what its own stage produces, or a later stage's.**
   Cascade re-runs later stages, so clearing their files is safe; an earlier
   stage stays marked done, so clearing its files strands the run.
   `_wipe_train` used to delete the config `split` renders → `missing train
   config`. Locked by `test_no_wipe_reaches_an_upstream_stages_artefacts`.
2. **Redo cascades, and unmarks every affected stage** — not just the ones this
   invocation runs. `--run-skip` or an interrupt otherwise leaves the manifest
   claiming output the wipe deleted.
3. **Wiping waits until the run is known viable.** `dag.run` defers its lasting
   writes past the sample checks; the wipe is the most destructive of them and
   must not run before a typo'd `--input` has been rejected.

`annotate` deliberately has **no wipe**: its `celltype_assignment_*.csv` live in
the user's data tree, not under `--output`.

## Stain normalization

`crop` Macenko-normalises through the **same HistomicsTK calls wsinsight runs at
inference**, one matrix per slide estimated from `--norm-sample-size` (256)
cells. Estimation is linear in pixels — 256 cells 0.16 s vs 12 s for all of
them — which is why it samples. Applying costs ~1 ms/cell.

Each `cells/<tissue>/<sample>.h5` keeps **both** `images` (normalised, what the
model trains on) and `images_raw`, plus the matrices as attributes. Re-reading a
slide costs ~5 ms/cell against ~1 ms to re-normalise, so the raw pixels earn
their space.

`--stardist-normalization-pmin/pmax` feed **both** `segment`'s `normalize()` call
and the exported config, so detection cannot drift between training and
inference. They default to csbdeep's own 3.0 / 99.8 — not the 1.0 the
QuST-produced `hne_cell_classification` model happens to declare.

## MCP server (`wsinsight-train-mcp`)

- Entry point `wsitrain.mcp.__main__:main`; extra `mcp = ["fastmcp>=4.0,<5"]`.
  stdio by default; `--http HOST:PORT` has **no default port** — the docs use
  8768 (after wsinsight 8765 / sptxinsight 8766 / hplot 8767).
- `mcp/schema.py` is **reflected from `cli.parser_for()`**, the same parser the
  CLI runs. It used to be hand-written and had drifted to naming thirteen flags
  argparse rejects while omitting the required `--tissue` everywhere.
  `tests/test_mcp_schema_parity.py` fails if the two diverge.
- `--no-stain-normalization` is recorded as an `off_flag`: the setting defaults
  to unset, so "off" cannot be expressed by omission.
- Every command except `check` runs as a background job.

## The CellViT config template is the only channel to the trainer

`defaults/train_config_template.yaml` is rendered by `split`, read by `train`.
A setting hardcoded there is a **flag that silently lies**: `epochs`, `lr`,
`weight_decay` and `normalize_stains_*` were all fixed values, so `--epochs 50`
trained for 10 and `--no-stain-normalization` still ran Macenko. Anything with
a `RunConfig` field must be substituted, and
`test_every_template_placeholder_is_supplied` fails on an orphan either way.

Two consequences:

- **A newly-substituted key becomes load-bearing**, so it must join
  `manifest._INVALIDATES` (`epochs`/`lr`/`weight_decay` → `split`, which renders
  the config). The map's "perf-only knobs are deliberately absent" comment was
  true only while they were inert.
- **What CellViT cannot honour must fail loudly**, not be dropped:
  `config.check_effective` refuses a run whose `--batch-size`/`--num-workers`/
  `--pretrained` the end2end path would ignore (CellViT hardcodes its
  DataLoader to 8). It fires only when the value came from a flag/config file
  *and* differs from the shipped default, so saved dumps still resume.

`gpu:` is a scalar. `_gpu_ids()` resolves `--gpus`; `_gpu_id()` returns the
first, because CellViT builds `f"cuda:{gpu}"` and has no DataParallel path — a
comma-joined list reaches torch as the invalid device `cuda:0,1`.

## Tests & lint

- `python -m pytest tests/` (492 tests; 8 in `test_review_fixes.py` +
  `test_subprocess_stages.py` fail wherever `shutil.which("python3")` resolves
  to an interpreter without `wandb` — environmental, not a regression). ruff is
  clean; keep it that way.
- Unit tests over the wipe/manifest helpers are not enough: every redo bug so
  far only appeared once the manifest, `--run-skip` and stage ordering
  interacted. `tests/test_pipeline_e2e.py` runs real stages with a fake
  segmenter — put redo/force coverage there.
- Input contract: a sample is any dir with `outs/cells.parquet` and a sibling
  `*_he_*image.ome.tif`. `wsitrain check` reports what it found.

## Sibling repos (same ecosystem)

- `kurtorank` — installed editable from `../kurtorank`; supplies the
  `celltype_assignment_<task>[_label].csv` this reads. The `_label` suffix is
  inconsistent by design; use `stages.assignment_csv`.
- `wsinsight` — deploys the exported model; the golden standard for pins.
- `ST2WSI_Registration` — produces the `registration_params.json` `transfer`
  needs.
