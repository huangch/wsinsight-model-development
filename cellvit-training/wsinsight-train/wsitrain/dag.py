"""End-to-end DAG driver with manifest-based resume."""
from __future__ import annotations

from pathlib import Path

from . import STAGES
from .config import RunConfig
from .dataset import discover_samples
from .manifest import Manifest
from .paths import resolved_config_path, manifest_path
from .stages import STAGE_FUNCS

import yaml


def step_range(start: str, end: str) -> list[str]:
    i, j = STAGES.index(start), STAGES.index(end)
    if i > j:
        raise SystemExit(f"--from {start} comes after --to {end}")
    return list(STAGES[i:j + 1])


def run(cfg: RunConfig, *, from_step: str = "annotate", to_step: str = "report",
        skip: list[str] | None = None, force: bool = False) -> int:
    skip = skip or []
    cfg.output.mkdir(parents=True, exist_ok=True)
    resolved_config_path(cfg.output).write_text(yaml.safe_dump(cfg.to_dict()))
    mf = Manifest.load_or_new(manifest_path(cfg.output), cfg.to_dict())
    samples = discover_samples(cfg.input, cfg.tissue)
    todo = [s for s in step_range(from_step, to_step) if s not in skip]
    print(f"[run] tissue={cfg.tissue} samples={len(samples)} steps={todo}")

    for stage in todo:
        if not force and mf.is_done(stage):
            print(f"[{stage}] up-to-date — skipping")
            continue
        print(f"[{stage}] running…")
        try:
            info = STAGE_FUNCS[stage](cfg, samples, cfg.output)
            mf.mark(stage, "done", **(info or {}))
        except NotImplementedError as e:
            mf.mark(stage, "pending", note=str(e))
            print(f"[{stage}] not yet implemented: {e}")
            return 0
    return 0
