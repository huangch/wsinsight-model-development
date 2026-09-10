"""End-to-end DAG driver with manifest-based resume."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import STAGES
from .config import RunConfig
from .configrender import _gpu_ids
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
    "crop": "cells",
}



def _chunked(items, k):
    """Split ``items`` into at most ``k`` roughly-equal lists (order preserved)."""
    if k <= 0:
        return [items]
    n = len(items)
    base, rem = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        size = base + (1 if j < rem else 0)
        out.append(items[i:i + size])
        i += size
    return [c for c in out if c]


def _fan_out_segment(cfg, samples, out, gpu_ids):
    """Run the segment stage in parallel across ``len(gpu_ids)`` GPUs.

    Each worker is a child ``wsitrain segment`` subprocess pinned to one
    device via ``CUDA_VISIBLE_DEVICES=<id>``. The samples are split into
    contiguous chunks; outputs land in the shared
    ``<output>/masks/<tissue>/<sample>.npy`` so the next stage (transfer)
    picks them up transparently. Returns the merged ``{sample_id: nuclei_count}``
    dict that the regular ``segment`` returns.
    """
    chunks = _chunked([s.sample_id for s in samples], len(gpu_ids))
    procs = []
    for gid, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gid
        # Reuse the currently-running Python + wsitrain entry point so the
        # child inherits the install path and uses the same code. Each
        # individual sample_id is one argv element; argparse --samples has
        # ``nargs="+"`` so it accepts the whole list verbatim. Comma inside a
        # sample_id is intentional (Xenium panel names) and is preserved.
        cmd = [sys.executable, "-m", "wsitrain.cli", "segment",
               "--input", str(cfg.input),
               "--tissue", cfg.tissue,
               "--output", str(out),
               "--gpus", gid,
               "--samples", *chunk,
               "--force"]
        print(f"[dag] fan-out segment -> cuda:{gid} for {len(chunk)} slide(s): {chunk}")
        # Stream the worker's stderr to the parent's stderr so its traceback
        # is visible immediately if it fails. stdout is buffered.
        procs.append((gid, subprocess.Popen(cmd, env=env,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.PIPE)))
    counts = {}
    for gid, p in procs:
        stderr_out, _ = p.communicate()
        if p.returncode != 0:
            tail = stderr_out.decode("utf-8", "replace").splitlines()[-30:] if stderr_out else []
            sys.stderr.write(
                f"[dag] segment worker on cuda:{gid} failed "
                f"(exit={p.returncode}); last 30 lines of stderr:\n"
                + "\n".join(tail) + "\n")
            raise SystemExit(
                f"[dag] segment worker on cuda:{gid} exited with code {p.returncode}; aborting.")
    # Read the produced masks back to compute the per-sample nuclei count
    # (segment normally returns this; we reproduce it here so the manifest
    # gets the same shape).
    import numpy as _np
    from . import paths as _paths
    for s in samples:
        mpath = _paths.masks_dir(out, cfg.tissue) / f"{s.sample_id}.npy"
        if mpath.is_file():
            counts[s.sample_id] = int(_np.load(mpath, mmap_mode="r").max())
    return {"segmenter": "(fan-out)", "nuclei_per_sample": counts}


def run(cfg: RunConfig, *, only: str | None = None, skip: list[str] | None = None,
        force: bool = False, samples: list[str] | None = None) -> int:
    """Run the pipeline.

    ``only`` names a single stage (a stage command). Because the stages it
    depends on are not part of this invocation, they are checked against the
    manifest first. A full or ``skip``-ed run needs no such check: it executes
    the stages in order and aborts as soon as one fails.

    ``samples`` filters the discovered samples to the given subset AFTER the
    aligned/unaligned filter. The intent is to let the operator shard work
    across GPUs/CPU pools by launching N parallel ``wsitrain`` processes with
    disjoint ``--samples`` lists; the per-process manifest entries merge
    naturally because each process writes its own outputs.
    """
    skipped = set(skip or [])
    cfg.output.mkdir(parents=True, exist_ok=True)
    samples_discovered = discover_samples(cfg.input, cfg.tissue)
    if cfg.transform != "none":
        kept = [s for s in samples_discovered if s.aligned]
        dropped = len(samples_discovered) - len(kept)
        if dropped:
            print(f"[run] skipping {dropped} unaligned sample(s) (transform={cfg.transform}); "
                  f"register them or use --transform none")
        samples_discovered = kept
    if samples:
        wanted = set(samples)
        before = len(samples_discovered)
        samples_kept = [s for s in samples_discovered if s.sample_id in wanted]
        missing = sorted(wanted - {s.sample_id for s in samples_kept})
        if missing:
            raise SystemExit(
                f"[run] --samples requested {len(missing)} id(s) not discovered for "
                f"tissue={cfg.tissue} (e.g. {missing[:3]}); check spelling or "
                "drop --samples to discover them.")
        if len(samples_kept) != before:
            print(f"[run] --samples narrowed {before} -> {len(samples_kept)} sample(s)")
        samples = samples_kept
    else:
        samples = samples_discovered
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
            gpu_ids = _gpu_ids(cfg)
            use_fanout = (
                stage == "segment"
                and len(gpu_ids) > 1
                and len(samples) > 1
                and getattr(cfg, "nuclei_source", "xenium-coords") == "he-mask"
            )
            if use_fanout:
                info = _fan_out_segment(cfg, samples, cfg.output, gpu_ids)
            else:
                info = STAGE_FUNCS[stage](cfg, samples, cfg.output)
            key = _REQUIRED_OUTPUT.get(stage)
            # Only one of tile/crop applies to a given model; the other reports
            # itself skipped and owes no output.
            if key and not (info or {}).get("skipped") and not (info or {}).get(key):
                mf.mark(stage, "failed", **(info or {}))
                raise SystemExit(f"[{stage}] produced no {key}; refusing to mark it done")
            mf.mark(stage, "done", **(info or {}))
        except NotImplementedError as e:
            mf.mark(stage, "pending", note=str(e))
            print(f"[{stage}] not yet implemented: {e}")
            return 0
    return 0
