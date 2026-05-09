# data/census/

Local SOMA mirror of a pinned [CELLxGENE Discover Census](https://chanzuckerberg.github.io/cellxgene-census/)
release. Used by `kurtorank rank-markers` to compute the marker-bank
rankings that ship with the [`kurtorank/`](../../kurtorank/README.md)
package.

**Not committed to git** (~1.5 TB). See [../README.md](../README.md) for
the overall data policy.

## Source

- **Census release index**: <https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_data_release_info.html>
- **Python API**: <https://chanzuckerberg.github.io/cellxgene-census/python-api.html>

License: per-dataset terms apply (most contributors release CC-BY-4.0).
The Census project itself is [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## On-disk layout

```
data/census/
└── <YYYY-MM-DD>/      # the Census release date you pinned
    ├── soma/          # SOMA collection
    └── ...
```

The release date is part of the path so multiple Census versions can
coexist; KurtoRank picks the directory it was configured against.

## Fetch recipe

```python
# Mirror a specific Census release locally. This downloads ~1.5 TB and
# takes hours; do it once on a host with the disk and bandwidth.
import cellxgene_census

CENSUS_DATE = "2025-11-08"          # match data/census/<DATE>/ on disk
TARGET_DIR  = f"./data/census/{CENSUS_DATE}"

cellxgene_census.download_source_h5ad  # see API for staged download paths
# or:
with cellxgene_census.open_soma(census_version=CENSUS_DATE) as census:
    census.to_disk(TARGET_DIR)        # local SOMA mirror
```

The exact API for materializing a local mirror has evolved across
`cellxgene-census` versions; consult the Python API page above for the
recipe matching the version pinned in
[`kurtorank/pyproject.toml`](../../kurtorank/pyproject.toml).

## Pinning

The Census release date is pinned in KurtoRank's package metadata so that
the marker rankings are reproducible. If you only need to *use* the
pre-computed marker bank shipped with KurtoRank
(`kurtorank/src/kurtorank/markers/data/markers-v3.csv`), you do **not**
need this folder at all — it is only required to *re-rank* markers from
scratch.
