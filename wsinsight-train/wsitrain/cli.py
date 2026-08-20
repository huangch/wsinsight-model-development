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


def _labels(values) -> tuple[str, ...] | None:
    """Accept both ``--drop-labels a b`` and ``--drop-labels a,b``."""
    if values is None:
        return None
    out: list[str] = []
    for v in values:
        out += [part.strip() for part in str(v).split(",") if part.strip()]
    return tuple(out)


def _stages(values) -> list[str]:
    """Accept both ``a b`` and ``a,b`` forms."""
    out: list[str] = []
    for v in values or ():
        out += [part.strip() for part in str(v).split(",") if part.strip()]
    unknown = [s for s in out if s not in STAGES]
    if unknown:
        raise SystemExit(
            f"unknown stage(s) {', '.join(unknown)}; choose from {', '.join(STAGES)}")
    return out


def _skipped_stages(args) -> list[str]:
    if args.stage_only:
        keep = _stages(args.stage_only)
        return [s for s in STAGES if s not in keep]
    return _stages(args.stage_skip)


def _cmd_run(args) -> int:
    cfg = build_config(Path(args.input), args.tissue,
                       Path(args.output) if args.output else None,
                       overrides={"segmenter": args.segmenter, "gpus": args.gpus,
                                  "transform": args.transform, "tune": args.tune,
                                  "task": args.task,
                                  "match_radius_px": args.match_radius_px,
                                  "min_match_rate": args.min_match_rate,
                                  "cellpose_batch_size": args.cellpose_batch_size,
                                  "cellpose_flow_threshold": args.cellpose_flow_threshold,
                                  "cellpose_model": args.cellpose_model,
                                  "stardist_model": args.stardist_model,
                                  "stardist_model_dir": args.stardist_model_dir,
                                  "stardist_cpu": args.stardist_cpu,
                                  "diameter": args.diameter,
                                  "drop_labels": _labels(args.drop_labels),
                                  "tile_px": args.tile_px,
                                  "mpp": args.mpp,
                                  "min_cells": args.min_cells,
                                  "bg_thresh": args.bg_thresh,
                                  "overlap": args.overlap,
                                  "val_frac": args.val_frac,
                                  "by_slide": args.by_slide,
                                  "seed": args.seed,
                                  "weight_cap": args.weight_cap,
                                  "backbone": args.backbone,
                                  "fold": args.fold,
                                  "markers_csv": args.markers_csv,
                                  "top_k_markers": args.top_k_markers})
    from . import dag
    return dag.run(cfg, skip=_skipped_stages(args), force=args.force)


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

    a = r.add_argument_group("annotate")
    a.add_argument("--task", default=None, help="label space, e.g. pantissue | hne | pannuke")
    a.add_argument("--markers-csv", type=Path, default=None,
                   help="marker panel CSV (default: kurtorank bundled)")
    a.add_argument("--top-k-markers", type=int, default=None,
                   help="keep the K most specific genes per subtype")

    g = r.add_argument_group("segment")
    g.add_argument("--segmenter", default=None, choices=["cellpose", "stardist"])
    g.add_argument("--cellpose-model", default=None, help="cellpose model name")
    g.add_argument("--stardist-model", default=None, help="stardist model name")
    g.add_argument("--stardist-model-dir", type=Path, default=None,
                   help="parent of the stardist model folder; avoids downloading")
    g.add_argument("--stardist-cpu", action="store_true", default=None,
                   help="run StarDist on CPU (no CUDA toolkit); torch keeps the GPU")
    g.add_argument("--diameter", type=float, default=None,
                   help="nucleus diameter in MICRONS (omit for auto)")
    g.add_argument("--cellpose-batch-size", type=int, default=None,
                   help="cellpose tile batch; lower (4/2/1) if GPU OOMs")
    g.add_argument("--cellpose-flow-threshold", type=float, default=None,
                   help="cellpose flow QC; 0 skips GPU flow check (avoids WSI OOM)")
    g.add_argument("--mpp", type=float, default=None,
                   help="microns per pixel of the H&E")

    t = r.add_argument_group("transfer")
    t.add_argument("--transform", default=None, choices=["affine", "affine+bspline", "none"])
    t.add_argument("--match-radius-px", type=int, default=None,
                   help="nucleus lookup search radius in H&E px (0 = exact pixel)")
    t.add_argument("--min-match-rate", type=float, default=None,
                   help="drop slides whose registration matches fewer than this fraction")
    t.add_argument("--drop-labels", nargs="+", default=None, metavar="LABEL",
                   help="cell types to exclude; space- or comma-separated")

    ti = r.add_argument_group("tile")
    ti.add_argument("--tile-px", type=int, default=None, help="tile edge in pixels")
    ti.add_argument("--min-cells", type=int, default=None, help="drop tiles with fewer cells")
    ti.add_argument("--bg-thresh", type=float, default=None,
                   help="drop tiles whose mean RGB exceeds this")
    ti.add_argument("--overlap", type=float, default=None, help="tile overlap fraction (0-1)")

    s = r.add_argument_group("split")
    s.add_argument("--val-frac", type=float, default=None, help="validation fraction")
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--by-tile", dest="by_slide", action="store_const", const=False,
                      help="hold out a fraction of tiles from every slide (default)")
    mode.add_argument("--by-slide", dest="by_slide", action="store_const", const=True,
                      help="hold out whole slides")
    s.set_defaults(by_slide=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--weight-cap", type=float, default=None,
                   help="cap on inverse-frequency class weights")

    tr = r.add_argument_group("train")
    tr.add_argument("--backbone", default=None, help="CellViT backbone, e.g. SAM-H-x40")
    tr.add_argument("--fold", default=None, help="fold name, e.g. fold_0")
    tr.add_argument("--gpus", default=None, help="device index, or 'cpu' to disable")
    tr.add_argument("--tune", type=int, default=None, help="auto-tune iterations (0=off)")

    d = r.add_argument_group("stage control")
    sel = d.add_mutually_exclusive_group()
    sel.add_argument("--stage-only", action="extend", nargs="+", default=[],
                     metavar="STAGE", help="run ONLY these stages")
    sel.add_argument("--stage-skip", action="extend", nargs="+", default=[],
                     metavar="STAGE", help="run everything EXCEPT these stages")
    d.add_argument("--force", action="store_true",
                   help="re-run stages the manifest already marks done")
    r.epilog = ("stages run in this order: " + " \u2192 ".join(STAGES)
                + ". --stage-only/--stage-skip take space- or comma-separated names.")
    r.set_defaults(fn=_cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
