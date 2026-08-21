"""Deterministic auto-tune levers for the train stage (opt-in via cfg.tune>0).

Each iteration reads validation macro-F1 + per-class F1 from CellViT++ JSON
artifacts, applies one lever, retrains, and keeps the change only if macro-F1
improves >= MIN_IMPROVEMENT. Levers rotate on rejection. No LLM; rule-based.
"""
from __future__ import annotations

import json
from pathlib import Path

MIN_IMPROVEMENT = 0.005
WEAK_F1 = 0.60
WEIGHT_BOOST = 1.5
DROP_STEP = 0.05
LEVERS = ("weight", "drop", "lr")


def _per_class_f1(run_dir: Path):
    """(class ids, per-class F1) from the trainer's raw validation tensors."""
    pred_pt = run_dir / "val_results" / "predictions.pt"
    gt_pt   = run_dir / "val_results" / "gt.pt"
    if not pred_pt.exists() or not gt_pt.exists():
        return None
    try:
        import torch
        from sklearn.metrics import f1_score
        preds = torch.load(pred_pt, map_location="cpu").numpy()
        gts   = torch.load(gt_pt,   map_location="cpu").numpy()
        classes = sorted(set(gts.tolist()))
        return classes, f1_score(gts, preds, labels=classes, average=None, zero_division=0)
    except Exception:
        return None


def read_macro_f1(run_dir: Path):
    """``(value, source)`` where source is 'macro' (tensors) or 'micro' (scores.json).

    scores.json's "F1-Score/Validation" is micro-F1 (== accuracy), which the
    majority class dominates. The two are not comparable, so the source is
    returned alongside the value and the caller refuses to mix them.
    """
    res = _per_class_f1(run_dir)
    if res is not None and len(res[1]):
        return float(res[1].mean()), "macro"
    scores = run_dir / "val_results" / "scores.json"
    if scores.exists():
        value = json.loads(scores.read_text()).get("F1-Score/Validation")
        if value is not None:
            return float(value), "micro"
    return None, None


def weakest_class(run_dir: Path) -> int | None:
    res = _per_class_f1(run_dir)
    if res is None:
        return None
    classes, per_class = res
    i = int(per_class.argmin())
    # Index back through `classes`: absent classes shift positional indices.
    return int(classes[i]) if per_class[i] < WEAK_F1 else None


def apply_lever(lever, weights, drop, lr, weak):
    if lever == "weight" and weak is not None and weights is not None:
        weights = list(weights)
        weights[weak] *= WEIGHT_BOOST
    elif lever == "drop":
        drop = round(min(drop + DROP_STEP, 0.5), 3)
    elif lever == "lr":
        lr = lr * 0.5
    return weights, drop, lr


def run_tune(cfg, out, cellvit, *, base_config, py):
    """Outer loop: retrain with one lever/iter, keep if macro-F1 improves."""
    import os
    import subprocess
    from pathlib import Path
    from . import configrender, paths, weights as weights_mod
    from .stages import _find_run_dir

    run_dir = _find_run_dir(cfg, out, required=False)
    best_f1, best_src = read_macro_f1(run_dir) if run_dir else (None, None)
    best_run = run_dir
    # Seed with the default inverse-frequency weights so the "weight" lever has a
    # real per-class vector to boost (was None -> list(None) TypeError on iter 1).
    _rep = weights_mod.compute_weights(
        paths.label_map_path(out, cfg.tissue),
        paths.labels_dir(out, cfg.tissue), cap=cfg.weight_cap)
    drop, lr, weights = 0.1, 0.000075, list(_rep.weights)
    li, rejects, log = 0, 0, []
    for it in range(cfg.tune):
        lever = LEVERS[li % len(LEVERS)]
        weak = weakest_class(run_dir) if run_dir else None
        cand = apply_lever(lever, weights, drop, lr, weak)
        if cand == (weights, drop, lr):
            # Retraining an identical config cannot teach us anything.
            log.append({"iter": it, "lever": lever, "f1": None, "accepted": False,
                        "note": "lever had no effect"})
            li += 1
            continue
        cand_w, cand_drop, cand_lr = cand
        cp = configrender.render_config(cfg, out, drop_rate=cand_drop, lr=cand_lr,
                                        weights=cand_w)
        _env = os.environ.copy()
        _env["PYTHONPATH"] = cellvit
        subprocess.run(
            [py, str(Path(cellvit) / "cellvit" / "train_cell_classifier_head.py"),
             "--config", str(cp)],
            cwd=cellvit, env=_env, check=True,
        )
        run_dir = _find_run_dir(cfg, out, required=False)
        f1, src = read_macro_f1(run_dir) if run_dir else (None, None)
        # An unmeasurable run, or one scored on a different metric, is not
        # evidence of improvement.
        comparable = f1 is not None and (best_src is None or src == best_src)
        ok = comparable and (best_f1 is None or f1 - best_f1 >= MIN_IMPROVEMENT)
        log.append({"iter": it, "lever": lever, "f1": f1, "source": src,
                    "accepted": bool(ok)})
        if ok:
            best_f1, best_src, best_run = f1, src, run_dir
            weights, drop, lr = cand_w, cand_drop, cand_lr
            rejects = 0
        else:
            # Keep the previous config: a rejected lever must not compound. The
            # run dir goes back too, or the next weakest_class() would read the
            # class balance of the run we just threw away.
            run_dir = best_run
            rejects += 1
            li += 1
            if rejects >= 2:
                break
    if best_run is not None:
        # export() takes the newest checkpoint by mtime, which would otherwise be
        # the last (possibly rejected) run rather than the best one.
        ckpt = best_run / "checkpoints" / "model_best.pth"
        if ckpt.exists():
            # Stamping "now" can tie with a run that finished in the same clock
            # tick; the sort is stable, so the tie resolves by scan order and the
            # rejected run wins. Step past the newest peer instead.
            import time

            peers = [p.stat().st_mtime
                     for p in paths.logs_dir(out, cfg.tissue).rglob(
                         "checkpoints/model_best.pth")
                     if p != ckpt]
            stamp = max([time.time()] + [m + 1.0 for m in peers])
            os.utime(ckpt, (stamp, stamp))
    (paths.report_dir(out, cfg.tissue)).mkdir(parents=True, exist_ok=True)
    (paths.report_dir(out, cfg.tissue) / "tuning_log.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in log))
    return {"best_f1": best_f1, "iters": len(log),
            "best_run": str(best_run) if best_run else None}
