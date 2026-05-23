"""DISCO v3 API client for per-atlas cell-type DEG markers.

Base URL: https://immunesinglecell.com/disco_v3_api

Used endpoints (confirmed 2026-04):

* ``GET /getStatistics`` \u2014 release metadata (``lastUpdate``).
* ``GET /atlas/getAtlasMeta`` \u2014 list of atlases with ``tissue``, ``type``,
  ``cell``, ``version``, ``rds_md5`` fields.
* ``GET /atlas/getDeg?atlas=X&type=Cell%20Type%20DEG`` \u2014 paginated DEG
  table. Row schema: ``{gene, cell_type, logfc, pct1, pct2, pvalue}``.

No external runtime dependencies: uses ``urllib.request`` from the stdlib.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_URL = "https://immunesinglecell.com/disco_v3_api"
USER_AGENT = "kurtorank-seed/1.0 (+https://github.com/)"
DEFAULT_TIMEOUT = 60
DEFAULT_PAGE_SIZE = 500


@dataclass(frozen=True)
class AtlasInfo:
    """Subset of ``/atlas/getAtlasMeta`` fields kurtorank uses."""

    atlas: str           # unique slug, e.g. "blood", "adipose_cell"
    tissue: str          # display label, e.g. "Blood", "Adipose"
    type: str            # "tissue" | "disease" | "cell type"
    cell_type_count: int
    cell_count: int
    version: float
    rds_md5: str


class DiscoError(RuntimeError):
    """Raised on HTTP or schema failures when talking to DISCO."""


# ---------------------------------------------------------------------------
# low-level HTTP
# ---------------------------------------------------------------------------
def _get_json(path: str, *, timeout: int = DEFAULT_TIMEOUT, **params: str) -> object:
    url = f"{BASE_URL}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise DiscoError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise DiscoError(f"Network error contacting {url}: {e.reason}") from e
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw[:200].decode(errors="replace")
        raise DiscoError(f"Non-JSON response from {url}: {snippet!r}") from e


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------
def get_release_tag(*, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Return the DISCO release tag (e.g. ``'2025-04'``) for provenance."""
    stats = _get_json("/getStatistics", timeout=timeout)
    if not isinstance(stats, dict) or "lastUpdate" not in stats:
        raise DiscoError(f"Unexpected /getStatistics payload: {stats!r}")
    return str(stats["lastUpdate"])


def list_atlases(*, timeout: int = DEFAULT_TIMEOUT) -> list[AtlasInfo]:
    """Return metadata for all atlases exposed by DISCO."""
    data = _get_json("/atlas/getAtlasMeta", timeout=timeout)
    if not isinstance(data, list):
        raise DiscoError(f"Unexpected /atlas/getAtlasMeta payload: {type(data)}")
    out: list[AtlasInfo] = []
    for row in data:
        try:
            out.append(
                AtlasInfo(
                    atlas=str(row["atlas"]),
                    tissue=str(row.get("tissue", "")).strip(),
                    type=str(row.get("type", "")).strip(),
                    cell_type_count=int(row.get("cell_type", 0)),
                    cell_count=int(row.get("cell", 0)),
                    version=float(row.get("version", 0.0)),
                    rds_md5=str(row.get("rds_md5", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            raise DiscoError(f"Malformed atlas row {row!r}: {e}") from e
    return out


def fetch_deg(
    atlas: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = 3,
) -> list[dict]:
    """Fetch the full Cell Type DEG table for one atlas, following pagination.

    Returns a list of raw DEG rows (dicts with ``gene``, ``cell_type``,
    ``logfc``, ``pct1``, ``pct2``, ``pvalue``, ``type``).
    """
    rows: list[dict] = []
    page = 1
    while True:
        params = {
            "atlas": atlas,
            "type": "Cell Type DEG",
            "page": str(page),
            "size": str(page_size),
        }
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                payload = _get_json("/atlas/getDeg", timeout=timeout, **params)
                break
            except DiscoError as e:
                last_err = e
                time.sleep(2 ** attempt)
        else:
            raise DiscoError(
                f"Failed to fetch DEG page {page} for atlas {atlas!r}: {last_err}"
            )
        if not isinstance(payload, dict):
            raise DiscoError(f"Unexpected getDeg payload for {atlas}: {type(payload)}")
        data = payload.get("data") or []
        rows.extend(data)
        total = int(payload.get("total", 0))
        if len(rows) >= total or not data:
            break
        page += 1
    return rows


# ---------------------------------------------------------------------------
# filtering + grouping
# ---------------------------------------------------------------------------
def deg_rows_to_markers(
    rows: Iterable[dict],
    *,
    logfc_min: float = 1.0,
    pct1_min: float = 0.25,
    max_markers_per_cell_type: int | None = None,
) -> dict[str, list[str]]:
    """Apply Seurat-style filter and group into ``{cell_type: [gene, ...]}``.

    Genes within each cell type are ordered by ``logfc`` descending. The
    filter keeps rows where ``logfc >= logfc_min`` **and**
    ``pct1 >= pct1_min``. Duplicate genes within a cell type are collapsed
    (first occurrence kept).
    """
    by_ct: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        try:
            logfc = float(r["logfc"])
            pct1 = float(r["pct1"])
            gene = str(r["gene"]).strip()
            cell_type = str(r["cell_type"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if logfc < logfc_min or pct1 < pct1_min or not gene or not cell_type:
            continue
        by_ct[cell_type].append((logfc, gene))

    out: dict[str, list[str]] = {}
    for ct, pairs in by_ct.items():
        pairs.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        ordered: list[str] = []
        for _, g in pairs:
            if g in seen:
                continue
            seen.add(g)
            ordered.append(g)
            if max_markers_per_cell_type and len(ordered) >= max_markers_per_cell_type:
                break
        out[ct] = ordered
    return out


# ---------------------------------------------------------------------------
# disk cache (keyed by release + atlas + rds_md5)
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: Path, release: str, atlas: str, rds_md5: str) -> Path:
    return cache_dir / release / f"{atlas}__{rds_md5 or 'nomd5'}.json"


def load_or_fetch_deg(
    atlas: AtlasInfo,
    release: str,
    cache_dir: Path | None,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Return DEG rows for an atlas, using an on-disk cache when provided."""
    if cache_dir is None:
        return fetch_deg(atlas.atlas, page_size=page_size, timeout=timeout)

    path = _cache_path(cache_dir, release, atlas.atlas, atlas.rds_md5)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)

    rows = fetch_deg(atlas.atlas, page_size=page_size, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows
