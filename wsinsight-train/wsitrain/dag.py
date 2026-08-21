"""End-to-end DAG driver with manifest-based resume."""
from __future__ import annotations

from . import STAGES
from .config import RunConfig
from .dataset import discover_samples
from .manifest import Manifest
from .paths import resolved_config_path, manifest_path
from .stages import STAGE_FUNCS, reset_cache
from . import prereq

import yaml


# Stage -> key in its return dict that must be non-empty; a stage that produced
# nothing must not be recorded as done or later runs resume on missing files.
_REQUIRED_OUTPUT = {
    "segment": "nuclei_per_sample",
    "transfer": "cells_per_sample",
    "tile": "tiles",
}


def run(cfg: RunConfig, *, only: str | None = None, skip: list[str] | None = None,
        force: bool = False) -> int:
    """Run the pipeline.

    ``only`` names a single stage (a stage command). Because the stages it
    depends on are not part of this invocation, they are checked against the
    manifest first. A full or ``skip``-ed run needs no such check: it executes
    the stages in order and aborts as soon as one fails.
    """
    skipped = set(skip or [])
    cfg.output.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(cfg.input, cfg.tissue)
    if cfg.transform != "none":
        kept = [s for s in samples if s.aligned]
        dropped = len(samples) - len(kept)
        if dropped:
            print(f"[run] skipping {dropped} unaligned sample(s) (transform={cfg.transform}); "
                  f"register them or use --transform none")
        samples = kept
    todo = [only] if only else [s for s in STAGES if s not in skipped]
    print(f"[run] tissue={cfg.tissue} samples={len(samples)} steps={todo}")

    if not samples and {"annotate", "segment", "transfer", "tile"}.intersection(todo):
        raise SystemExit(
            f"[run] no samples found for tissue={cfg.tissue} under {cfg.input} — "
            "check --input and that samples live in <input>/<tissue>/<sample>/outs/")

    # Only once the invocation is known to be viable. Both of these have lasting
    # effects -- the config is the base every later command inherits, and loading
    # the manifest can invalidate stages -- so a doomed run must not reach them.
    resolved_config_path(cfg.output, cfg.tissue).write_text(yaml.safe_dump(cfg.to_dict()))
    mf = Manifest.load_or_new(manifest_path(cfg.output, cfg.tissue), cfg.to_dict())

    for stage in todo:
        if not force and mf.is_done(stage):
            print(f"[{stage}] up-to-date — skipping")
            continue
        if only:
            prereq.check(stage, mf, cfg)
        if force:
            reset_cache(stage, cfg, cfg.output)
        print(f"[{stage}] running…")
        try:
            info = STAGE_FUNCS[stage](cfg, samples, cfg.output)
            key = _REQUIRED_OUTPUT.get(stage)
            if key and not (info or {}).get(key):
                mf.mark(stage, "failed", **(info or {}))
                raise SystemExit(f"[{stage}] produced no {key}; refusing to mark it done")
            mf.mark(stage, "done", **(info or {}))
        except NotImplementedError as e:
            mf.mark(stage, "pending", note=str(e))
            print(f"[{stage}] not yet implemented: {e}")
            return 0
    return 0
