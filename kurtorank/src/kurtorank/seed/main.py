"""``kurtorank build-panel`` \u2014 produce a skeleton marker panel from DISCO.

Output schema (skeleton CSV for manual biology curation afterwards):

    tissue_type, subtype, markers, n_cells, source, added_at

where:

* ``tissue_type`` is the DISCO atlas slug (unique key, e.g. ``blood``,
  ``adipose_cell``). Users typically rename these during curation.
* ``subtype`` is the DISCO cell-type name verbatim.
* ``markers`` is a comma-separated gene list, sorted by logfc descending,
  filtered by ``--logfc-min`` and ``--pct1-min``.
* ``source`` encodes provenance: ``disco:<atlas>:v<ver>:<release>``.
* ``added_at`` is the ISO timestamp when the row was produced.

Downstream biology columns used by ``kurtorank annotate``
(``major_type``, ``pannuke_label``, ``hne_type``, ``hne_label``,
``common``, ``malignant``, ``rank_source``, ``low_support``) are **not**
produced here \u2014 they must be filled in manually before the panel is
useful to ``annotate``.
"""
from __future__ import annotations

import csv
import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from kurtorank.seed import disco as _disco


OUTPUT_COLUMNS = (
    "tissue_type",
    "subtype",
    "markers",
    "n_cells",
    "source",
    "added_at",
)


@dataclass
class BuildSummary:
    atlases_requested: list[str]
    atlases_built: list[str]
    atlases_skipped: list[tuple[str, str]]  # (atlas, reason)
    rows_written: int
    output_path: Path


def _did_you_mean(name: str, choices: list[str]) -> list[str]:
    import difflib

    return difflib.get_close_matches(name, choices, n=3, cutoff=0.5)


def _select_atlases(
    all_atlases: list[_disco.AtlasInfo],
    *,
    names: list[str] | None,
    select_all: bool,
    include_types: set[str],
    skip_missing: bool,
) -> list[_disco.AtlasInfo]:
    by_slug = {a.atlas: a for a in all_atlases}

    if select_all:
        return [a for a in all_atlases if a.type in include_types]

    selected: list[_disco.AtlasInfo] = []
    for n in names or []:
        if n in by_slug:
            selected.append(by_slug[n])
            continue
        suggestions = _did_you_mean(n, list(by_slug))
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        msg = f"Unknown atlas {n!r}.{hint}"
        if skip_missing:
            click.echo(f"[warn] {msg} Skipping.", err=True)
            continue
        raise click.ClickException(msg)
    return selected


def build_panel(
    *,
    atlases: list[str] | None = None,
    all_atlases: bool = False,
    include_disease: bool = False,
    include_celltype: bool = False,
    logfc_min: float = 1.0,
    pct1_min: float = 0.25,
    max_markers_per_cell_type: int | None = None,
    output: Path,
    cache_dir: Path | None = None,
    skip_missing: bool = False,
    timeout: int = _disco.DEFAULT_TIMEOUT,
) -> BuildSummary:
    """Python API entry point. See :func:`build_panel_cmd` for the CLI."""

    include_types: set[str] = {"tissue"}
    if include_disease:
        include_types.add("disease")
    if include_celltype:
        include_types.add("cell type")

    meta = _disco.list_atlases(timeout=timeout)
    release = _disco.get_release_tag(timeout=timeout)

    targets = _select_atlases(
        meta,
        names=atlases,
        select_all=all_atlases,
        include_types=include_types,
        skip_missing=skip_missing,
    )
    if not targets:
        raise click.ClickException(
            "No atlases selected. Use --atlases or --all-atlases."
        )

    now_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    rows_written = 0
    built: list[str] = []
    skipped: list[tuple[str, str]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for atlas in targets:
            click.echo(
                f"[disco] fetching {atlas.atlas} "
                f"({atlas.type}, {atlas.cell_type_count} cell types, "
                f"{atlas.cell_count:,} cells)",
                err=True,
            )
            try:
                deg_rows = _disco.load_or_fetch_deg(
                    atlas, release, cache_dir, timeout=timeout
                )
            except _disco.DiscoError as e:
                skipped.append((atlas.atlas, str(e)))
                click.echo(f"[warn] skipping {atlas.atlas}: {e}", err=True)
                continue

            markers = _disco.deg_rows_to_markers(
                deg_rows,
                logfc_min=logfc_min,
                pct1_min=pct1_min,
                max_markers_per_cell_type=max_markers_per_cell_type,
            )
            if not markers:
                skipped.append((atlas.atlas, "no cell types passed filter"))
                click.echo(
                    f"[warn] {atlas.atlas}: 0 cell types passed filter", err=True
                )
                continue

            source_tag = (
                f"disco:{atlas.atlas}:v{atlas.version}:{release}"
            )
            for ct in sorted(markers):
                gene_list = ",".join(markers[ct])
                writer.writerow(
                    {
                        "tissue_type": atlas.atlas,
                        "subtype": ct,
                        "markers": gene_list,
                        "n_cells": atlas.cell_count,
                        "source": source_tag,
                        "added_at": now_iso,
                    }
                )
                rows_written += 1
            built.append(atlas.atlas)

    return BuildSummary(
        atlases_requested=[a.atlas for a in targets],
        atlases_built=built,
        atlases_skipped=skipped,
        rows_written=rows_written,
        output_path=output,
    )


# ---------------------------------------------------------------------------
# click subcommand
# ---------------------------------------------------------------------------
@click.command("build-panel")
@click.option(
    "--atlases",
    "atlases_csv",
    default=None,
    help="Comma-separated DISCO atlas slugs (e.g. blood,lung,adipose_cell).",
)
@click.option(
    "--all-atlases",
    is_flag=True,
    default=False,
    help="Fetch every atlas matching the type filter (default: type==tissue).",
)
@click.option(
    "--list-atlases",
    is_flag=True,
    default=False,
    help="Print the DISCO atlas catalog and exit (no download).",
)
@click.option(
    "--include-disease",
    is_flag=True,
    default=False,
    help="Also include atlases where type=='disease' (default: excluded).",
)
@click.option(
    "--include-celltype",
    is_flag=True,
    default=False,
    help="Also include atlases where type=='cell type' (default: excluded).",
)
@click.option(
    "--logfc-min",
    type=float,
    default=1.0,
    show_default=True,
    help="Minimum log fold-change to keep a DEG row as a marker.",
)
@click.option(
    "--pct1-min",
    type=float,
    default=0.25,
    show_default=True,
    help="Minimum fraction of cells in the target cell type that must express the gene.",
)
@click.option(
    "--max-markers",
    type=int,
    default=None,
    help="Optional per-cell-type cap on marker count (top-N by logfc).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output CSV path. Required unless --list-atlases is given.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.home() / ".kurtorank" / "disco",
    show_default=True,
    help="Directory for the DISCO HTTP response cache.",
)
@click.option(
    "--skip-missing",
    is_flag=True,
    default=False,
    help="Skip (warn) unknown atlas names instead of failing.",
)
@click.option(
    "--timeout",
    type=int,
    default=_disco.DEFAULT_TIMEOUT,
    show_default=True,
    help="HTTP timeout in seconds per request.",
)
def build_panel_cmd(
    atlases_csv: str | None,
    all_atlases: bool,
    list_atlases: bool,
    include_disease: bool,
    include_celltype: bool,
    logfc_min: float,
    pct1_min: float,
    max_markers: int | None,
    output: Path | None,
    cache_dir: Path,
    skip_missing: bool,
    timeout: int,
) -> None:
    """Build a skeleton marker panel CSV from DISCO atlases.

    The output lists ``(tissue_type, subtype, markers)`` plus provenance.
    Biology columns required by ``kurtorank annotate`` (``major_type``,
    ``pannuke_label``, ``hne_type``, ``hne_label``, ``common``, ``malignant``)
    are **not** produced here and must be filled in manually.
    """
    if list_atlases:
        meta = _disco.list_atlases(timeout=timeout)
        click.echo(f"{'atlas':<40}{'type':<12}{'tissue':<30}{'cells':>12}  {'cts':>4}")
        for a in sorted(meta, key=lambda x: (x.type, x.atlas)):
            click.echo(
                f"{a.atlas:<40}{a.type:<12}{a.tissue:<30}"
                f"{a.cell_count:>12,}  {a.cell_type_count:>4}"
            )
        return

    if (atlases_csv is None) == (not all_atlases):
        # neither OR both
        if atlases_csv is None and not all_atlases:
            raise click.ClickException(
                "Provide either --atlases NAMES or --all-atlases (or --list-atlases)."
            )
        if atlases_csv is not None and all_atlases:
            raise click.ClickException(
                "--atlases and --all-atlases are mutually exclusive."
            )

    if output is None:
        raise click.ClickException("--output is required when building a panel.")

    if output.exists():
        raise click.ClickException(
            f"Refusing to overwrite existing file {output}. "
            f"Delete it first or choose a new path."
        )

    atlas_names = (
        [s.strip() for s in atlases_csv.split(",") if s.strip()]
        if atlases_csv
        else None
    )

    summary = build_panel(
        atlases=atlas_names,
        all_atlases=all_atlases,
        include_disease=include_disease,
        include_celltype=include_celltype,
        logfc_min=logfc_min,
        pct1_min=pct1_min,
        max_markers_per_cell_type=max_markers,
        output=output,
        cache_dir=cache_dir,
        skip_missing=skip_missing,
        timeout=timeout,
    )

    # summary banner (stderr so it doesn't contaminate downstream pipes)
    click.echo(
        f"\n[done] wrote {summary.rows_written} rows from "
        f"{len(summary.atlases_built)} atlases to {summary.output_path}",
        err=True,
    )
    if summary.atlases_skipped:
        click.echo(
            f"[done] skipped {len(summary.atlases_skipped)} atlas(es): "
            + ", ".join(f"{a} ({r})" for a, r in summary.atlases_skipped),
            err=True,
        )
    click.echo(
        "[note] biology columns (major_type, pannuke_label, hne_type, "
        "hne_label, common, malignant) must be curated manually before "
        "feeding this panel to `kurtorank annotate`.",
        err=True,
    )


if __name__ == "__main__":
    build_panel_cmd()  # pragma: no cover
