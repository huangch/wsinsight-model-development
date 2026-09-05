# KurtoRank — Agent Guide

Unsupervised ensemble subtype annotation for gene-limited spatial
transcriptomics (Xenium-scale panels). Turns a marker table plus a Xenium
sample into a per-cluster cell-type assignment CSV. Python >=3.11, v3.1.0.

**This file is for *developing* this package.** To *use* the CLI, read
`SKILL.md` (full option reference, decision guide, troubleshooting). For the
method and worked examples, read `README.md`. Do not duplicate the option
tables from those files here.

## Environment (read first)

- Co-installable with the shared `wsinsight` conda env; that is the intended
  setup. In a shared env install with `pip install --no-deps -e .` so pip
  cannot move the locked `numpy<2` / `zarr<3` / `anndata<0.13` generation.
- Standalone: `bash ./conda-setup.sh kurtorank [-r|--reset] [-c|--census] [-d|--dev]`. ENV_NAME is a required positional; run `./conda-setup.sh --help` for the full CLI.
- **`cellxgene-census` + `tiledbsoma` are optional and their install is allowed
  to fail.** Only `rank-markers` and the `marker-*` paths touch Census, and
  every import site is lazy. `conda-setup.sh` gates them behind `-c/--census`
  and only warns. A missing Census is not a broken install.
- `fastmcp` is likewise an extra (`kurtorank[mcp]`), not a default.
- The setup script has no `set -e`: a failed pip step is silent, and the
  `kurtorank --help` smoke gate is what catches it. A bare `import kurtorank`
  does not.

## CLI

Entry point `kurtorank` (Click). Four commands (`annotate`, `build-panel`,
`rank-markers`, `schema`); options are documented in `SKILL.md` §3. What
matters when changing them:

- `annotate` — the main pass; the only command with `--output-dir`.
- `build-panel` — the output flag is `--output` / `-o`, **not** `--output-dir`.
- `rank-markers` — a variadic **positional passthrough** into the argparse
  layer in `rank/main.py`. It takes no Click options of its own, so adding a
  Click option here would shadow, not extend, the argparse surface.
- `schema` — emits `{"schema_version": 1, "commands": {...}}`. Every engine in
  the family exposes this, so downstream tools need no per-project casing.
  Regenerate consumers after any option change.
- The bundled panel (`markers/data/markers-v6.csv`) is 347 rows over 19 tissue
  types, and `TISSUE_MAP` in `rank/main.py` has a Census mapping for all 19 —
  keep the two in step when adding a tissue, or `rank-markers` silently skips
  it.

## Assignment-CSV naming (the wsitrain contract)

`annotate` is deliberately inconsistent about the suffix, and the writer's own
comment says not to tidy it:

- **with** `_label`: `pantissue`, `hne`, `pannuke`, `sthelar_*`, `lcp`
- **without**: `subtype`, `major`, `hne_type` → `celltype_assignment_subtype.csv`

`wsitrain` resolves both via `wsitrain.stages.assignment_csv(outs, task)`,
which tries `_label` first then the bare name. Renaming the writer would strip
the already-annotated files on disk of their names.

## MCP server (`kurtorank-mcp`)

- Entry point `kurtorank.mcp.__main__:main`; extra `mcp = ["fastmcp>=4.0,<5"]`.
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

## Doc maintenance

Three files, three audiences — keep a fact in exactly one of them:

| File | Audience | Owns |
| --- | --- | --- |
| `README.md` | humans | the method, install, worked examples |
| `SKILL.md` | an agent using the package | every CLI option, decision guide, troubleshooting |
| `AGENTS.md` | an agent developing the package | repo layout, internal contracts, tests, lint baseline |

Option defaults and the tissue/method lists appear in `SKILL.md`; when you
change a CLI signature, update `SKILL.md` and only mention it here if the
change carries a development constraint.
