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


def read_macro_f1(run_dir: Path) -> float | None:
    js = sorted(run_dir.rglob("*classification_report*.json"))
    if not js:
        return None
    d = json.loads(js[-1].read_text())
    return d.get("macro avg", {}).get("f1-score")


def weakest_class(run_dir: Path) -> int | None:
    js = sorted(run_dir.rglob("*classification_report*.json"))
    if not js:
        return None
    d = json.loads(js[-1].read_text())
    cls = {int(k): v["f1-score"] for k, v in d.items() if k.isdigit()}
    if not cls:
        return None
    c = min(cls, key=cls.get)
    return c if cls[c] < WEAK_F1 else None


def apply_lever(lever, weights, drop, lr, weak):
    if lever == "weight" and weak is not None:
        weights = list(weights); weights[weak] *= WEIGHT_BOOST
    elif lever == "drop":
        drop = round(min(drop + DROP_STEP, 0.5), 3)
    elif lever == "lr":
        lr = lr * 0.5
    return weights, drop, lr


def run_tune(cfg, out, cellvit, *, base_config, py):
    """Outer loop: retrain with one lever/iter, keep if macro-F1 improves."""
    import subprocess
    from pathlib import Path
    from . import configrender, paths

    logs = Path(cellvit) / "logs_local"
    rd = sorted(logs.rglob("checkpoints/model_best.pth"), key=lambda p: p.stat().st_mtime)
    best_f1 = read_macro_f1(rd[-1].parent.parent) if rd else None
    drop, lr, weights = 0.1, 0.0003, None
    li, rejects, log = 0, 0, []
    for it in range(cfg.tune):
        lever = LEVERS[li % len(LEVERS)]
        weak = weakest_class(rd[-1].parent.parent) if rd else None
        weights, drop, lr = apply_lever(lever, weights, drop, lr, weak)
        cp = configrender.render_config(cfg, out, drop_rate=drop, lr=lr, weights=weights)
        subprocess.run([py, str(Path(cellvit) / "train.py"), "--config", str(cp)], cwd=cellvit, check=True)
        rd = sorted(logs.rglob("checkpoints/model_best.pth"), key=lambda p: p.stat().st_mtime)
        f1 = read_macro_f1(rd[-1].parent.parent)
        ok = best_f1 is None or (f1 and f1 - best_f1 >= MIN_IMPROVEMENT)
        log.append({"iter": it, "lever": lever, "f1": f1, "accepted": bool(ok)})
        if ok: best_f1, rejects = f1, 0
        else:
            rejects += 1; li += 1
            if rejects >= 2: break
    (paths.report_dir(out, cfg.tissue)).mkdir(parents=True, exist_ok=True)
    import json
    (paths.report_dir(out, cfg.tissue) / "tuning_log.jsonl").write_text(
        "\n".join(json.dumps(x) for x in log))
    return {"best_f1": best_f1, "iters": len(log)}
