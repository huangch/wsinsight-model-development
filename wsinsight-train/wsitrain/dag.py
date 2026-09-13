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

# Stage -> a light disk-presence probe for its product. The manifest can claim
# a stage is done while its product is gone (a failed export marked done on the
# checkpoint_ok fallback, a wiped results dir), and resume would then silently
# skip it. train's checkpoint directory name is the trainer's decision, so its
# probe is the per-tissue log root having a checkpoints/ leaf; the other two are
# deterministic file names wsinsight itself writes.
def _product_missing(cfg: RunConfig, out: Path, stage: str) -> str | None:
    import glob as _glob

    from .paths import models_dir, report_dir

    if stage == "train":
        root = out / "logs" / _slug(cfg.tissue)
        if not (root.is_dir() and any(root.glob("*/checkpoints"))
                or any(_glob.glob(str(root / "checkpoints")))):
            return str(root / "<run>/checkpoints")
        return None
    if stage == "export":
        p = models_dir(out, cfg.tissue) / "main" / "torchscript_model.pt"
        return None if p.is_file() else str(p)
    if stage == "report":
        p = report_dir(out, cfg.tissue) / "scores.json"
        return None if p.is_file() else str(p)
    return None


def _slug(tissue: str) -> str:
    out = tissue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return out.strip(".") or "unnamed"



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


def _fan_out_segment(cfg, samples, out, gpu_ids,
                     per_slide_timeout_s: int = 1800):
    """Run the segment stage in parallel across ``len(gpu_ids)`` GPUs.

    Each worker is a child ``wsitrain segment`` subprocess pinned to one
    device via ``CUDA_VISIBLE_DEVICES=<id>``. The samples are split into
    contiguous chunks; outputs land in the shared
    ``<output>/masks/<tissue>/<sample>.npy`` so the next stage (transfer)
    picks them up transparently. Returns the merged ``{sample_id: nuclei_count}``
    dict that the regular ``segment`` returns.

    ``per_slide_timeout_s`` bounds each worker at (N slides in chunk *
    per_slide_timeout_s) seconds. A hung worker (e.g. deadlocked cellpose
    TF init on a shared host) is killed and the failure is reported, so the
    remaining fast workers do not block forever.
    """
    chunks = _chunked([s.sample_id for s in samples], len(gpu_ids))
    workers = []
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
        deadline = per_slide_timeout_s * len(chunk)
        print(f"[dag] fan-out segment -> cuda:{gid} for {len(chunk)} slide(s); "
              f"timeout {deadline}s")
        # Stream stderr line-by-line so we see hangs as they happen. stdout is
        # buffered (segment uses tqdm which redraws via \r, see tile stage).
        workers.append((gid, chunk, deadline,
                        subprocess.Popen(cmd, env=env,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         bufsize=0)))
    counts = {}
    import select
    for gid, chunk, deadline, proc in workers:
        tail_lines: list[str] = []
        exited = False
        # Read output as the process runs so the operator sees progress.
        import time as _time
        start = _time.time()
        fd = proc.stdout.fileno()
        os.set_blocking(fd, False)
        buf = b""
        while True:
            if _time.time() - start > deadline:
                proc.kill()
                proc.wait()
                raise SystemExit(
                    f"[dag] segment worker on cuda:{gid} exceeded "
                    f"{deadline}s timeout; killed. last output: "
                    + "\n".join(tail_lines[-10:]))
            if proc.poll() is not None:
                exited = True
                # Drain remaining stdout.
                remainder = proc.stdout.read()
                if remainder:
                    for ln in remainder.decode("utf-8", "replace").splitlines():
                        tail_lines.append(ln)
                break
            r, _, _ = select.select([fd], [], [], 1.0)
            if not r:
                continue
            try:
                chunk_bytes = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk_bytes:
                continue
            buf += chunk_bytes
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                decoded = line.decode("utf-8", "replace")
                # Mirror to parent's stderr so the operator sees it live.
                sys.stderr.write(decoded + "\n")
                sys.stderr.flush()
                tail_lines.append(decoded)
        if proc.returncode != 0:
            raise SystemExit(
                f"[dag] segment worker on cuda:{gid} exited with code "
                f"{proc.returncode} (after {len(tail_lines)} log lines); aborting. "
                f"last output:\n" + "\n".join(tail_lines[-10:]))
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
        force: bool = False, samples: list[str] | None = None,
        redo: set[str] | None = None) -> int:
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

    ``redo`` is a set of stage names whose on-disk artefacts must be wiped
    BEFORE the stage runs, regardless of what the manifest says. Use a single
    ``--redo-<stage>`` flag (repeatable) or pass the set programmatically.
    Naming a stage invalidates every later stage too, since their output was
    derived from it. ``--force`` is the legacy "wipe everything" switch.
    Both unmark the affected stages in the manifest so the re-run is allowed,
    and both wait until the invocation is known to be viable.
    """
    skipped = set(skip or [])
    cfg.output.mkdir(parents=True, exist_ok=True)

    from .stages import _STAGE_WIPES as _wipes
    redo = set(redo or ())
    if redo:
        # A redone stage makes every later stage's output stale, so cascade the
        # way a config change does in Manifest._stale_stages.
        earliest = min(STAGES.index(s) for s in redo if s in STAGES)
        redo |= set(STAGES[earliest:])
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

    # Every redone stage is unmarked, not just the ones this invocation will
    # run or can wipe: --run-skip or an interrupt would otherwise leave the
    # manifest claiming output the wipe deleted, and annotate has no wipe at
    # all (its CSVs live in the user's data tree). Wiping waits until here for
    # the same reason the writes above do -- a doomed run must not delete
    # artefacts it will never rebuild.
    if force:
        mf = Manifest(manifest_path(cfg.output, cfg.tissue))
    for stage in (list(_wipes) if force else sorted(redo & set(_wipes))):
        n = _wipes[stage](cfg.output, cfg)
        if n:
            print(f"[dag] wiped {n} artefact(s) for stage {stage!r}")
    if not force:
        for s in sorted(redo & set(STAGES), key=STAGES.index):
            mf.mark(s, "pending")
        if redo:
            print(f"[dag] invalidated {sorted(redo, key=STAGES.index)}")
    for stage in todo:
        if not (force or stage in redo) and mf.is_done(stage):
            # A done status can outlive its product (failed export marked done
            # on the checkpoint_ok fallback, a wiped results dir). Re-run such
            # a stage instead of silently skipping it.
            gone = _product_missing(cfg, cfg.output, stage)
            if gone:
                mf.mark(stage, "pending", note=f"product missing: {gone}")
                print(f"[{stage}] marked done but its product is gone ({gone}); re-running")
            else:
                print(f"[{stage}] up-to-date — skipping")
                continue
        if only:
            prereq.check(stage, mf, cfg)
        print(f"[{stage}] running…")
        try:
            # Resolved lazily: --gpus cpu raises, and only segment fan-out
            # needs a device, so the documented CPU escape hatch
            # (--gpus cpu --run-skip split train validate export) still runs.
            gpu_ids = (
                _gpu_ids(cfg)
                if stage == "segment"
                and len(samples) > 1
                and getattr(cfg, "nuclei_source", "xenium-coords") == "he-mask"
                else []
            )
            if len(gpu_ids) > 1:
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
