"""wsitrain CLI: `run`, `check`, `version`. Minimal front door."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from . import __version__, STAGES
from .config import build_config
from .dataset import discover_samples, validate_input


def _cmd_check(args) -> int:
    problems = validate_input(Path(args.input), args.tissue)
    samples = discover_samples(Path(args.input), args.tissue)
    print(f"input: {args.input}  tissue: {args.tissue}")
    print(f"samples: {len(samples)}")
    for s in samples:
        print(f"  - {s.sample_id} [{s.tissue}] aligned={s.aligned}")
    if samples:
        from .dataset import write_manifest
        mpath = Path(args.input) / "wsitrain_samples.csv"
        write_manifest(samples, mpath)
        print(f"manifest: {mpath}")
    if shutil.which("kurtorank") is None:
        problems.append("kurtorank console script not found on PATH")
    try:
        import torch  # noqa
        print(f"cuda: {torch.cuda.is_available()} ({torch.cuda.device_count()} gpu)")
    except Exception:
        problems.append("torch not importable (GPU stack missing)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nOK — ready to run.")
    return 0


def _cmd_run(args) -> int:
    cfg = build_config(Path(args.input), args.tissue,
                       Path(args.output) if args.output else None,
                       user_config=args.config,
                       overrides={"segmenter": args.segmenter, "gpus": args.gpus,
                                  "transform": args.transform, "tune": args.tune,
                                  "task": args.task,
                                  "match_radius_px": args.match_radius_px,
                                  "min_match_rate": args.min_match_rate,
                                  "cellpose_batch_size": args.cellpose_batch_size,
                                  "cellpose_flow_threshold": args.cellpose_flow_threshold})
    from . import dag
    return dag.run(cfg, from_step=args.from_step, to_step=args.to_step,
                   skip=args.skip, force=args.force)


def main(argv=None) -> int:
    p = argparse.ArgumentParser("wsitrain", description="Train WSInsight CellViT heads end-to-end.")
    p.add_argument("--version", action="version", version=f"wsitrain {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="Preflight: validate input + environment.")
    c.add_argument("--input", required=True)
    c.add_argument("--tissue", default="pantissue")
    c.set_defaults(fn=_cmd_check)

    r = sub.add_parser("run", help="Run the end-to-end training pipeline.")
    r.add_argument("--input", required=True)
    r.add_argument("--tissue", default="pantissue")
    r.add_argument("--output", default=None)
    r.add_argument("--config", default=None)
    r.add_argument("--segmenter", default=None, choices=["cellpose", "stardist"])
    r.add_argument("--cellpose-batch-size", type=int, default=None,
                   help="cellpose tile batch; lower (4/2/1) if GPU OOMs")
    r.add_argument("--cellpose-flow-threshold", type=float, default=None,
                   help="cellpose flow QC; 0 skips GPU flow check (avoids WSI OOM)")
    r.add_argument("--transform", default=None, choices=["affine", "affine+bspline", "none"])
    r.add_argument("--match-radius-px", type=int, default=None,
                   help="nucleus lookup search radius in H&E px (0 = exact pixel)")
    r.add_argument("--min-match-rate", type=float, default=None,
                   help="drop slides whose registration matches fewer than this fraction")
    r.add_argument("--gpus", default=None)
    r.add_argument("--tune", type=int, default=None, help="auto-tune iterations (0=off)")
    r.add_argument("--task", default=None)
    r.add_argument("--from", dest="from_step", default="annotate", choices=STAGES)
    r.add_argument("--to", dest="to_step", default="report", choices=STAGES)
    r.add_argument("--skip", action="append", default=[], choices=STAGES)
    r.add_argument("--force", action="store_true")
    r.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
