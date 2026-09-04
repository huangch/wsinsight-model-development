# KurtoRank — Agent Guide

Unsupervised ensemble subtype annotation for gene-limited spatial
transcriptomics (Xenium-scale panels). Turns a marker table plus a Xenium
sample into a per-cluster cell-type assignment CSV. Python >=3.11, v3.1.0.

## Environment (read first)

- Co-installable with the shared `wsinsight` conda env; that is the intended
  setup. In a shared env install with `pip install --no-deps -e .` so pip
  cannot move the locked `numpy<2` / `zarr<3` / `anndata<0.13` generation.
- Standalone: `bash ./conda-setup.sh -n kurtorank [-r|--reset] [-c|--census]`.
- **`cellxgene-census` + `tiledbsoma` are optional and their install is allowed
  to fail.** Only `rank-markers` and the `marker-*` paths touch Census, and
  every import site is lazy. `conda-setup.sh` gates them behind `-c/--census`
  and only warns. A missing Census is not a broken install.
- `fastmcp` is likewise an extra (`kurtorank[mcp]`), not a default.
- The setup script has no `set -e`: a failed pip step is silent, and the
  `kurtorank --help` smoke gate is what catches it. A bare `import kurtorank`
  does not.

## CLI

Entry point `kurtorank` (Click). Three commands:

- `annotate` — the main pass. 25 options; writes the assignment CSVs.
- `build-panel` — assemble a marker panel from DISCO atlases. Note the output
  flag is `--output` / `-o`, **not** `--output-dir`.
- `rank-markers` — a variadic **positional passthrough** into the argparse
  layer in `rank/main.py`. It takes no Click options of its own.

## Assignment-CSV naming (the wsitrain contract)

`annotate` is deliberately inconsistent about the suffix, and the writer's own
comment says not to tidy it:

- **with** `_label`: `pantissue`, `hne`, `pannuke`, `sthelar_*`, `lcp`
- **without**: `subtype`, `major`, `hne_type` → `celltype_assignment_subtype.csv`

`wsitrain` resolves both via `wsitrain.stages.assignment_csv(outs, task)`,
which tries `_label` first then the bare name. Renaming the writer would strip
the already-annotated files on disk of their names.

## MCP server (`kurtorank-mcp`)

- Entry point `kurtorank.mcp.__main__:main`; extra `mcp = ["fastmcp>=2.0"]`.
  stdio by default; `--http HOST:PORT` has **no default port** — the docs use
  8769 (after wsinsight 8765 / sptxinsight 8766 / hplot 8767 / wsitrain 8768).
- `mcp/schema.py` is **reflected from the Click commands**, not hand-written.
  It used to be enumerated by hand and had drifted to offering `--output-dir`
  where `build-panel` takes `--output`, plus an `annotate --panel` that never
  existed. `tests/test_mcp_schema_parity.py` fails if the two diverge.
- Flags whose default is on (`--common-only`, `--normal-only`, ...) carry an
  `off_flag` so an agent can turn them off; omission cannot express "off".
- All three commands run as background jobs (`job_id` + `job_status` /
  `job_logs` / `cancel_job`).

## Tests & lint

- `PYTHONPATH=src python -m pytest tests/ -q` (22 tests).
- ruff has ~50 **pre-existing** findings, concentrated in
  `annotate/main.py` (E402 imports below a runtime-ordering block), plus unused
  imports in `scripts/` and one notebook that fails the nbformat schema. They
  predate the MCP work; do not treat a clean ruff run as the bar here.

## Sibling repos (same ecosystem)

- `wsinsight-train` (`wsitrain`) — consumes the assignment CSVs; installs this
  package editable from `../kurtorank`.
- `wsinsight` — the WSI pipeline the trained heads are deployed into, and the
  golden standard for package pins where repos disagree.
- `sptxinsight`, `hplot`, `clawsight`, `clawpyter` — the rest of the stack.
