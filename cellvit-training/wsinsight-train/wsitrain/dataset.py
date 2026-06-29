"""Input discovery + sample manifest.

Mechanism (fallback order):
  1. auto-discover: recurse from --input; any dir with outs/cells.parquet is a
     sample; pair the nearest sibling *_he_*image.ome.tif. Flag 'unaligned'.
  2. manifest: a samples.csv (sample_id,tissue,outs,he,aligned) that
     overrides/edits discovery. `check` writes one to eyeball.
  3. BYO: skip discovery, enter at --from tiling.

Tissue is a pooling bucket, not a cell-type filter:
  --tissue breast    -> only samples under a breast/ path
  --tissue pantissue -> pool every discovered sample (pan-cancer head)

Registration is assumed done unless the H&E filename says 'unaligned'.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    sample_id: str
    tissue: str
    outs: Path
    he: Path
    aligned: bool


def _find_he(sample_dir: Path) -> Path | None:
    cands = (sorted(sample_dir.glob("*_he_*image.ome.tif"))
             + sorted(sample_dir.glob("*he*.ome.tif")))
    if not cands:
        return None
    aligned = [p for p in cands if "unaligned" not in p.name]
    return aligned[0] if aligned else cands[0]


def _tissue_of(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else rel


def discover_samples(input_dir: Path, tissue: str = "pantissue") -> list[Sample]:
    """Discover samples. tissue may be 'pantissue' (all), one tissue, or a
    comma list ('breast,lung') to pool a chosen subset."""
    input_dir = Path(input_dir)
    wanted = None if tissue == "pantissue" else {t.strip() for t in tissue.replace("+", ",").split(",")}
    samples: list[Sample] = []
    seen: set[Path] = set()
    for outs in sorted(input_dir.rglob("outs")):
        if not (outs / "cells.parquet").exists() or outs.parent in seen:
            continue
        seen.add(outs.parent)
        he = _find_he(outs.parent)
        if he is None:
            continue
        rel = outs.parent.relative_to(input_dir).as_posix()
        tis = _tissue_of(rel)
        if wanted is not None and tis not in wanted:
            continue
        sid = rel.replace("/", "__")
        samples.append(Sample(sid, tis, outs, he, "unaligned" not in he.name))
    return samples


def write_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["sample_id", "tissue", "outs", "he", "aligned"])
        for s in samples:
            w.writerow([s.sample_id, s.tissue, s.outs, s.he, int(s.aligned)])


def read_manifest(path: Path) -> list[Sample]:
    with Path(path).open() as fp:
        return [Sample(r["sample_id"], r["tissue"], Path(r["outs"]),
                       Path(r["he"]), bool(int(r["aligned"])))
                for r in csv.DictReader(fp)]


def validate_input(input_dir: Path, tissue: str = "pantissue") -> list[str]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return [f"input dir does not exist: {input_dir}"]
    samples = discover_samples(input_dir, tissue)
    problems: list[str] = []
    if not samples:
        problems.append(f"no samples (need outs/cells.parquet + H&E) for tissue={tissue}")
    for s in samples:
        if not s.aligned:
            problems.append(f"{s.sample_id}: H&E UNALIGNED ({s.he.name}) — register first")
    return problems
