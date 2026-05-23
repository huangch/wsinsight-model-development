"""Tests for ``kurtorank.seed`` \u2014 DISCO seed-panel builder.

Network is fully mocked; these tests must pass offline.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from kurtorank.seed import disco as _disco
from kurtorank.seed.main import build_panel, build_panel_cmd


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
FAKE_STATS = {
    "sampleCount": 22743,
    "cellCount": 144831560,
    "atlasCount": 2,
    "lastUpdate": "2025-04",
}

FAKE_ATLASES = [
    {
        "atlas": "blood",
        "tissue": "Blood",
        "type": "tissue",
        "cell_type": 20,
        "cell": 100000,
        "version": 1.0,
        "rds_md5": "abc123",
    },
    {
        "atlas": "COVID-19_blood",
        "tissue": "Blood",
        "type": "disease",
        "cell_type": 15,
        "cell": 50000,
        "version": 1.0,
        "rds_md5": "def456",
    },
]


def _fake_deg_rows():
    # Two cell types; some genes pass the filter, some don't.
    return [
        # T cell: 3 markers pass
        {"gene": "CD3D", "cell_type": "T cell", "logfc": 3.5, "pct1": 0.9, "pct2": 0.05, "pvalue": 0.0, "type": "Cell Type DEG"},
        {"gene": "CD3E", "cell_type": "T cell", "logfc": 2.1, "pct1": 0.85, "pct2": 0.1, "pvalue": 0.0, "type": "Cell Type DEG"},
        {"gene": "CCL5", "cell_type": "T cell", "logfc": 1.2, "pct1": 0.30, "pct2": 0.2, "pvalue": 1e-10, "type": "Cell Type DEG"},
        # T cell: filtered out (low logfc)
        {"gene": "ACTB", "cell_type": "T cell", "logfc": 0.3, "pct1": 0.99, "pct2": 0.99, "pvalue": 0.001, "type": "Cell Type DEG"},
        # T cell: filtered out (low pct1)
        {"gene": "RARE", "cell_type": "T cell", "logfc": 5.0, "pct1": 0.10, "pct2": 0.01, "pvalue": 0.0, "type": "Cell Type DEG"},
        # B cell: 2 markers pass
        {"gene": "MS4A1", "cell_type": "B cell", "logfc": 4.0, "pct1": 0.8, "pct2": 0.01, "pvalue": 0.0, "type": "Cell Type DEG"},
        {"gene": "CD79A", "cell_type": "B cell", "logfc": 2.5, "pct1": 0.7, "pct2": 0.05, "pvalue": 0.0, "type": "Cell Type DEG"},
        # duplicate gene within same cell type \u2014 should collapse
        {"gene": "CD79A", "cell_type": "B cell", "logfc": 2.4, "pct1": 0.7, "pct2": 0.05, "pvalue": 0.0, "type": "Cell Type DEG"},
    ]


def _fake_get_json(path, *, timeout=60, **params):
    if path == "/getStatistics":
        return FAKE_STATS
    if path == "/atlas/getAtlasMeta":
        return FAKE_ATLASES
    if path == "/atlas/getDeg":
        rows = _fake_deg_rows()
        return {"total": len(rows), "page": 1, "size": 500, "data": rows}
    raise AssertionError(f"Unexpected path: {path}")


@pytest.fixture
def patch_http(monkeypatch):
    monkeypatch.setattr(_disco, "_get_json", _fake_get_json)


# ---------------------------------------------------------------------------
# filter / grouping logic (pure, no HTTP)
# ---------------------------------------------------------------------------
def test_deg_rows_to_markers_applies_filter_and_orders_by_logfc():
    markers = _disco.deg_rows_to_markers(_fake_deg_rows(), logfc_min=1.0, pct1_min=0.25)
    assert markers["T cell"] == ["CD3D", "CD3E", "CCL5"]
    assert markers["B cell"] == ["MS4A1", "CD79A"]


def test_deg_rows_to_markers_honors_max_cap():
    markers = _disco.deg_rows_to_markers(
        _fake_deg_rows(), logfc_min=1.0, pct1_min=0.25, max_markers_per_cell_type=2
    )
    assert markers["T cell"] == ["CD3D", "CD3E"]


def test_deg_rows_to_markers_permissive_filter_includes_more():
    markers = _disco.deg_rows_to_markers(_fake_deg_rows(), logfc_min=0.2, pct1_min=0.05)
    assert "ACTB" in markers["T cell"]  # would be rejected under Seurat default


# ---------------------------------------------------------------------------
# build_panel (integration)
# ---------------------------------------------------------------------------
def test_build_panel_writes_expected_schema(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    summary = build_panel(
        atlases=["blood"],
        output=out,
        cache_dir=None,
    )
    assert summary.rows_written == 2  # T cell + B cell
    assert summary.atlases_built == ["blood"]
    assert summary.atlases_skipped == []

    with out.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert {r["subtype"] for r in rows} == {"T cell", "B cell"}
    r = next(r for r in rows if r["subtype"] == "T cell")
    assert r["tissue_type"] == "blood"
    assert r["markers"] == "CD3D,CD3E,CCL5"
    assert r["source"].startswith("disco:blood:v1.0:2025-04")
    assert r["n_cells"] == "100000"
    assert "added_at" in r and r["added_at"]


def test_build_panel_default_excludes_disease_atlas(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    summary = build_panel(all_atlases=True, output=out, cache_dir=None)
    # Only the tissue atlas should be built; disease atlas excluded by default.
    assert summary.atlases_built == ["blood"]


def test_build_panel_include_disease_expands(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    summary = build_panel(
        all_atlases=True, include_disease=True, output=out, cache_dir=None
    )
    assert set(summary.atlases_built) == {"blood", "COVID-19_blood"}


def test_build_panel_unknown_atlas_fails(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    with pytest.raises(Exception):
        build_panel(atlases=["bloodd"], output=out, cache_dir=None)


def test_build_panel_skip_missing(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    summary = build_panel(
        atlases=["bloodd", "blood"], output=out, cache_dir=None, skip_missing=True
    )
    assert summary.atlases_built == ["blood"]


def test_build_panel_uses_disk_cache(tmp_path: Path, patch_http):
    cache = tmp_path / "cache"
    out1 = tmp_path / "p1.csv"
    out2 = tmp_path / "p2.csv"

    build_panel(atlases=["blood"], output=out1, cache_dir=cache)
    # cache file should exist keyed by release and rds_md5
    cache_files = list((cache / "2025-04").glob("blood__*.json"))
    assert len(cache_files) == 1
    assert "abc123" in cache_files[0].name

    # Break the HTTP layer \u2014 second run must still succeed via cache.
    with mock.patch.object(
        _disco, "_get_json",
        side_effect=lambda path, **kw: (
            FAKE_STATS if path == "/getStatistics"
            else FAKE_ATLASES if path == "/atlas/getAtlasMeta"
            else (_ for _ in ()).throw(AssertionError("HTTP must not be called"))
        ),
    ):
        summary = build_panel(atlases=["blood"], output=out2, cache_dir=cache)
    assert summary.rows_written == 2


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------
def test_cli_list_atlases(patch_http):
    runner = CliRunner()
    result = runner.invoke(build_panel_cmd, ["--list-atlases"])
    assert result.exit_code == 0, result.output
    assert "blood" in result.output
    assert "COVID-19_blood" in result.output


def test_cli_refuses_to_overwrite(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    out.write_text("existing\n")
    runner = CliRunner()
    result = runner.invoke(
        build_panel_cmd, ["--atlases", "blood", "--output", str(out)]
    )
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output


def test_cli_requires_selector(patch_http):
    runner = CliRunner()
    result = runner.invoke(build_panel_cmd, ["--output", "/tmp/x.csv"])
    assert result.exit_code != 0
    assert "--atlases" in result.output or "--all-atlases" in result.output


def test_cli_end_to_end(tmp_path: Path, patch_http):
    out = tmp_path / "panel.csv"
    runner = CliRunner()
    result = runner.invoke(
        build_panel_cmd,
        ["--atlases", "blood", "--output", str(out), "--cache-dir", str(tmp_path / "c")],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
