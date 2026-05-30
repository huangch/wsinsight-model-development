"""
agent_orchestrator.py
---------------------
Outer-loop agent that iteratively improves a CellViT classifier head by
launching multiple training runs with adjusted hyperparameters and reading
back the JSON validation artifacts (classification_report.json,
confusion_matrix.json) emitted by validate_classifier.py.

The agent is **deterministic** (rule-based, no LLM): each iteration it
inspects per-class F1 + confusion off-diagonals and picks one lever:

    Lever A  ── class re-weighting          (boost weight_list[c] of weakest)
    Lever C  ── regularization              (increase training.drop_rate)
    Lever D  ── learning-rate decay         (drop optimizer.lr by 0.5×)

Each iteration writes a new YAML to
    trainingset/<tissue>/train_configs/<backbone>/<fold>__iterN.yaml
runs train_tissue.sh with that config, parses the validation JSONs from
the resulting run_dir, and compares macro-F1 against the previous best.

A run is *accepted* if macro-F1 improves by >= MIN_IMPROVEMENT (default
0.005). After two consecutive rejections, the lever rotates. After
MAX_ITER rejections in a row, the loop stops.

All decisions are appended to
    pipeline/agent_runs/<tissue>_<timestamp>/decision_log.jsonl

Usage:
    python agent_orchestrator.py --tissue colorectal
    python agent_orchestrator.py --tissue pantissue --max-iter 6
    python agent_orchestrator.py --tissue heart --dry-run     # no GPU spent

Pre-conditions:
    * trainingset/<tissue>/train_configs/<backbone>/<fold>.yaml exists
      (generate via make_train_config.py)
    * trainingset/<tissue>/splits/<fold>/{train,val}.csv exist
    * trainingset/<tissue>/label_map.yaml exists
    * One CUDA GPU is available.

This script does *not* invoke the inner training loop directly; it shells
out to bash pipeline/train_tissue.sh and parses its artifacts. So the
agent and the trainer remain decoupled.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent
TRAININGSET = CELLVIT_TRAINING_ROOT / "trainingset"
LOGS_LOCAL = CELLVIT_TRAINING_ROOT / "cellvit" / "CellViT-plus-plus" / "logs_local"

# ── Policy thresholds (tuneable; document the *why* if you change these) ──
MIN_IMPROVEMENT = 0.005      # macro-F1 delta required to accept a new run
WEAK_F1_THRESHOLD = 0.60     # below this → class is "weak", up-weight it
WEIGHT_BOOST = 1.5           # multiply weight_list[c] by this, then rescale
WEIGHT_CAP = 20.0            # never let any weight exceed this (post-rescale)
WEIGHT_BUDGET_RATIO = 1.0    # sum(weights) target = N_classes * this ratio
DROP_RATE_STEP = 0.05        # additive bump per Lever-C iteration
DROP_RATE_CAP = 0.30         # cap to avoid destroying learning
LR_DECAY = 0.5               # multiplicative cut per Lever-D iteration
LR_FLOOR = 1e-5              # cap to avoid stalling

LEVERS = ("A", "C", "D")


# ── YAML helpers (no PyYAML dep — operate as line-edits to preserve order) ──
# The training config is parsed by upstream code expecting specific keys at
# specific paths; we use targeted regex edits instead of YAML round-trip so
# comments + key order survive intact.

def _read(path: Path) -> str:
    return path.read_text()


def _write(path: Path, txt: str) -> None:
    path.write_text(txt)


def _get_weight_list(yaml_text: str) -> list[float]:
    m = re.search(r"^  weight_list:\s*\[(.+?)\]\s*$", yaml_text, re.M)
    if m:
        return [float(x) for x in m.group(1).split(",")]
    # Fall back to multi-line block form (one '- value' per line).
    m = re.search(r"^  weight_list:\s*\n((?:  - [-\d.eE+]+\n)+)", yaml_text, re.M)
    if not m:
        raise RuntimeError("weight_list not found in YAML")
    return [float(line.strip("- \n")) for line in m.group(1).splitlines()]


def _set_weight_list(yaml_text: str, weights: list[float]) -> str:
    new_line = "  weight_list: [" + ", ".join(f"{w:g}" for w in weights) + "]"
    if re.search(r"^  weight_list:\s*\[", yaml_text, re.M):
        return re.sub(r"^  weight_list:.*$", new_line, yaml_text, count=1, flags=re.M)
    # Replace multi-line block with inline form.
    return re.sub(
        r"^  weight_list:\s*\n(?:  - [-\d.eE+]+\n)+",
        new_line + "\n",
        yaml_text, count=1, flags=re.M,
    )


def _get_float_field(yaml_text: str, key_path: list[str]) -> float:
    # key_path = ["training", "drop_rate"] → match "  drop_rate: 0.1" after a
    # "training:" header. We use depth-based indent heuristic.
    flat = ".".join(key_path)
    if flat == "training.drop_rate":
        m = re.search(r"^  drop_rate:\s*([-\d.eE+]+)\s*$", yaml_text, re.M)
    elif flat == "training.optimizer_hyperparameter.lr":
        m = re.search(r"^    lr:\s*([-\d.eE+]+)\s*$", yaml_text, re.M)
    else:
        raise ValueError(f"Unsupported key_path: {flat}")
    if not m:
        raise RuntimeError(f"Field not found: {flat}")
    return float(m.group(1))


def _set_float_field(yaml_text: str, key_path: list[str], value: float) -> str:
    flat = ".".join(key_path)
    if flat == "training.drop_rate":
        return re.sub(
            r"^(  drop_rate:\s*)[-\d.eE+]+(\s*)$",
            lambda m: f"{m.group(1)}{value:g}{m.group(2)}",
            yaml_text, count=1, flags=re.M,
        )
    if flat == "training.optimizer_hyperparameter.lr":
        return re.sub(
            r"^(    lr:\s*)[-\d.eE+]+(\s*)$",
            lambda m: f"{m.group(1)}{value:g}{m.group(2)}",
            yaml_text, count=1, flags=re.M,
        )
    raise ValueError(f"Unsupported key_path: {flat}")


def _set_log_comment(yaml_text: str, new_comment: str) -> str:
    """Rewrite logging.log_comment so logs_local/ gets a unique suffix per iter
    (otherwise _lib::find_latest_run would re-pick old runs)."""
    return re.sub(
        r"^  log_comment:.*$",
        f"  log_comment: {new_comment}",
        yaml_text, count=1, flags=re.M,
    )


# ── Validation-output reader ─────────────────────────────────────────────────

def _find_run_dir(log_comment: str) -> Path | None:
    """Return the newest *actual* run dir for ``log_comment`` under logs_local/.

    The trainer nests the real run dir one level deep, e.g.
    ``logs_local/<outer-ts>_<log_comment>/<run-ts>_<log_comment>/``. Both the
    outer wrapper and the inner run dir share the ``_<log_comment>`` suffix, so a
    plain ``glob('*_<log_comment>')`` can return the wrapper (which has no
    ``validation/``). Search recursively and prefer dirs that actually contain
    the validation report; fall back to the newest matching dir otherwise.
    """
    if not LOGS_LOCAL.is_dir():
        return None
    matches = sorted(LOGS_LOCAL.rglob(f"*_{log_comment}"), key=os.path.getmtime)
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        return None
    with_validation = [
        m for m in matches
        if (m / "validation" / "classification_report.json").is_file()
    ]
    if with_validation:
        return with_validation[-1]
    return matches[-1]


def _read_validation(run_dir: Path) -> dict[str, Any]:
    val_dir = run_dir / "validation"
    rep_path = val_dir / "classification_report.json"
    cm_path = val_dir / "confusion_matrix.json"
    if not rep_path.is_file() or not cm_path.is_file():
        raise FileNotFoundError(
            f"Missing validation JSONs under {val_dir}\n"
            f"  expected: {rep_path}\n  expected: {cm_path}\n"
            "Make sure validate_classifier.py finished (and is the JSON-emitting version)."
        )
    report = json.loads(rep_path.read_text())
    cm = json.loads(cm_path.read_text())
    return {"report": report, "confusion": cm, "run_dir": str(run_dir)}


# ── Decision policy ──────────────────────────────────────────────────────────

def _pick_weakest_class(report: dict) -> tuple[int, str, float] | None:
    """Return (class_idx, name, f1) of the worst-performing class with
    F1 < WEAK_F1_THRESHOLD, or None if all classes meet the bar."""
    candidates = []
    for k, v in report.items():
        if not isinstance(v, dict) or "f1-score" not in v:
            continue
        if k in ("accuracy", "macro avg", "weighted avg", "_summary"):
            continue
        # Class name is k; we need its index. The label_map in the YAML is the
        # source of truth, but report's `target_names` come from class_names
        # which matches label_map order. Caller resolves index via mapping.
        candidates.append((k, float(v["f1-score"]), int(v.get("support", 0))))
    if not candidates:
        return None
    # Skip classes with zero support (no val examples — can't fix).
    candidates = [c for c in candidates if c[2] > 0]
    if not candidates:
        return None
    name, f1, _ = min(candidates, key=lambda c: c[1])
    if f1 >= WEAK_F1_THRESHOLD:
        return None
    return (-1, name, f1)   # idx resolved later


def _resolve_class_idx_by_name(yaml_text: str, name: str) -> int:
    """Find class index in YAML's label_map block by looking up its name."""
    block = re.search(r"^  label_map:\n((?:    \d+:\s*\S.*\n)+)", yaml_text, re.M)
    if not block:
        raise RuntimeError("label_map block not found in YAML")
    for line in block.group(1).splitlines():
        m = re.match(r"\s*(\d+):\s*(\S+)\s*$", line)
        if m and m.group(2) == name:
            return int(m.group(1))
    raise RuntimeError(f"Class name '{name}' not found in label_map")


def _apply_lever_A(yaml_text: str, report: dict, log: dict) -> str | None:
    """Boost weight_list[c] for the weakest class, then **rescale all weights
    so the total budget (sum) is preserved**. This makes Lever A a
    redistribution -- when one class goes up, the others drop proportionally
    -- instead of an unbounded expansion of total attention.

    Returns new YAML or None if no class qualifies (then caller should try a
    different lever)."""
    weakest = _pick_weakest_class(report)
    if weakest is None:
        log["reason"] = "no weak class (all F1 >= threshold)"
        return None
    _, name, f1 = weakest
    idx = _resolve_class_idx_by_name(yaml_text, name)
    w = _get_weight_list(yaml_text)
    n_classes = len(w)
    budget = n_classes * WEIGHT_BUDGET_RATIO

    old = w[idx]
    boosted = w.copy()
    boosted[idx] = min(old * WEIGHT_BOOST, WEIGHT_CAP)
    # Rescale so sum(boosted) == budget. This redistributes the cost of the
    # boost across all other classes proportionally.
    s = sum(boosted)
    if s <= 0:
        log["reason"] = "weight sum is non-positive; cannot rescale"
        return None
    scale = budget / s
    new_w = [round(min(x * scale, WEIGHT_CAP), 4) for x in boosted]
    if abs(new_w[idx] - old) < 1e-4:
        log["reason"] = f"class '{name}' weight unchanged after rescale (at cap)"
        return None
    log["action"] = {
        "lever": "A",
        "class_idx": idx,
        "class_name": name,
        "class_f1": f1,
        "weight_old": old,
        "weight_new": new_w[idx],
        "budget": budget,
        "budget_used": round(sum(new_w), 4),
    }
    return _set_weight_list(yaml_text, new_w)


def _apply_lever_C(yaml_text: str, report: dict, log: dict) -> str | None:
    """Increase drop_rate (regularization). Use only when train acc >> val
    acc, but we don't have train acc handy from validate_classifier.py — so
    apply unconditionally as a fallback when Lever A is exhausted."""
    cur = _get_float_field(yaml_text, ["training", "drop_rate"])
    new = min(cur + DROP_RATE_STEP, DROP_RATE_CAP)
    if abs(new - cur) < 1e-6:
        log["reason"] = f"drop_rate already at cap ({DROP_RATE_CAP})"
        return None
    log["action"] = {"lever": "C", "drop_rate_old": cur, "drop_rate_new": new}
    return _set_float_field(yaml_text, ["training", "drop_rate"], new)


def _apply_lever_D(yaml_text: str, report: dict, log: dict) -> str | None:
    """Halve learning rate. Useful when the loss plateaued."""
    cur = _get_float_field(yaml_text, ["training", "optimizer_hyperparameter", "lr"])
    new = max(cur * LR_DECAY, LR_FLOOR)
    if abs(new - cur) < 1e-9:
        log["reason"] = f"lr already at floor ({LR_FLOOR})"
        return None
    log["action"] = {"lever": "D", "lr_old": cur, "lr_new": new}
    return _set_float_field(yaml_text, ["training", "optimizer_hyperparameter", "lr"], new)


LEVER_FNS = {"A": _apply_lever_A, "C": _apply_lever_C, "D": _apply_lever_D}


# ── Training-run launcher ────────────────────────────────────────────────────

def _launch_training(tissue: str, backbone: str, fold: str, task: str,
                     config_path: Path, dry_run: bool) -> None:
    """Run train_tissue.sh with an overridden CONFIG environment variable.
    train_tissue.sh itself derives CONFIG from tissue/backbone/fold, so the
    cleanest swap is to copy the iterN.yaml *onto* the canonical name just
    before invocation, train, then restore. But a simpler approach: write
    the canonical fold_0.yaml *to point at* the iterN content by replacing
    the file. We snapshot the original first.

    For v1 simplicity: we overwrite the canonical YAML in place for the
    duration of the call, then restore. Iter YAMLs are kept as a record."""
    canonical = (TRAININGSET / tissue / "train_configs" / backbone / f"{fold}.yaml")
    if not canonical.is_file():
        raise FileNotFoundError(f"canonical YAML missing: {canonical}")
    backup = canonical.with_suffix(".yaml.bak")
    shutil.copy2(canonical, backup)
    try:
        shutil.copy2(config_path, canonical)
        cmd = ["bash", str(SCRIPT_DIR / "train_tissue.sh"),
               tissue, backbone, fold, task]
        print(f"\n  $ {' '.join(cmd)}")
        if dry_run:
            print("  [dry-run] not actually launching")
            return
        result = subprocess.run(cmd, cwd=str(CELLVIT_TRAINING_ROOT))
        if result.returncode != 0:
            raise RuntimeError(f"train_tissue.sh failed (exit {result.returncode})")
    finally:
        shutil.copy2(backup, canonical)
        backup.unlink(missing_ok=True)


def _parse_task_from_log_comment(log_comment: str, tissue: str,
                                  backbone: str) -> str:
    """log_comment = '<tissue>-<task>-<backbone-lower>'  →  return <task>.
    Falls back to 'hne' if parsing fails."""
    suffix = f"-{backbone.lower()}"
    if log_comment.startswith(f"{tissue}-") and log_comment.endswith(suffix):
        return log_comment[len(tissue) + 1: -len(suffix)]
    return "hne"


# ── Main loop ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tissue", required=True)
    ap.add_argument("--backbone", default="SAM-H-x40")
    ap.add_argument("--fold", default="fold_0")
    ap.add_argument("--max-iter", type=int, default=4,
                    help="max outer-loop iterations after the baseline (default: 4)")
    ap.add_argument("--baseline-run-dir", default=None,
                    help="skip baseline training and start from this run_dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="emit YAMLs + decisions but do not launch training")
    args = ap.parse_args()

    tissue, backbone, fold = args.tissue, args.backbone, args.fold
    canonical = TRAININGSET / tissue / "train_configs" / backbone / f"{fold}.yaml"
    if not canonical.is_file():
        print(f"ERROR: missing {canonical}", file=sys.stderr)
        print(f"  generate with: python pipeline/make_train_config.py --tissue {tissue}",
              file=sys.stderr)
        return 1

    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_root = SCRIPT_DIR / "agent_runs" / f"{tissue}_{stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "decision_log.jsonl"

    def _log(event: dict) -> None:
        event["ts"] = _dt.datetime.now().isoformat(timespec="seconds")
        with log_path.open("a") as fp:
            fp.write(json.dumps(event) + "\n")
        print(f"  ← log: {event.get('phase', '?')}  {event.get('note', '')}")

    _log({"phase": "init", "tissue": tissue, "backbone": backbone, "fold": fold,
          "thresholds": {"weak_f1": WEAK_F1_THRESHOLD,
                         "min_improvement": MIN_IMPROVEMENT}})

    base_yaml = _read(canonical)
    base_log_comment = re.search(r"^  log_comment:\s*(\S+)\s*$",
                                 base_yaml, re.M).group(1)
    task = _parse_task_from_log_comment(base_log_comment, tissue, backbone)

    # ── Baseline ──────────────────────────────────────────────────────────
    if args.baseline_run_dir:
        baseline_run = Path(args.baseline_run_dir)
        print(f"\n[baseline] using existing run: {baseline_run}")
    else:
        iter_yaml = run_root / f"{fold}__iter0_baseline.yaml"
        _write(iter_yaml, base_yaml)
        print(f"\n[iter 0] BASELINE  (log_comment={base_log_comment})")
        _launch_training(tissue, backbone, fold, task, iter_yaml, args.dry_run)
        baseline_run = _find_run_dir(base_log_comment)
        if baseline_run is None and not args.dry_run:
            raise RuntimeError(f"baseline run dir not found for {base_log_comment}")

    if args.dry_run and not args.baseline_run_dir:
        _log({"phase": "baseline_skipped_dry_run"})
        print("\n[dry-run] stopping after baseline plan")
        return 0

    baseline = _read_validation(baseline_run)
    best_f1 = baseline["report"]["_summary"]["f1_macro"]
    _log({"phase": "baseline_done", "run_dir": str(baseline_run),
          "f1_macro": best_f1,
          "acc": baseline["report"]["_summary"]["accuracy"]})
    print(f"\n[baseline] macro-F1 = {best_f1:.4f}")

    # ── Outer loop ────────────────────────────────────────────────────────
    cur_yaml = base_yaml
    cur_report = baseline["report"]
    lever_idx = 0
    consecutive_rejections = 0

    for it in range(1, args.max_iter + 1):
        lever = LEVERS[lever_idx % len(LEVERS)]
        decision: dict = {"phase": "iter", "iter": it, "lever_tried": lever}
        print(f"\n[iter {it}] trying lever {lever}")

        new_yaml = LEVER_FNS[lever](cur_yaml, cur_report, decision)
        if new_yaml is None:
            print(f"  lever {lever} not applicable: {decision.get('reason')}")
            _log(decision | {"applied": False})
            lever_idx += 1
            continue

        new_log_comment = f"{base_log_comment}__iter{it}_{lever.lower()}"
        new_yaml = _set_log_comment(new_yaml, new_log_comment)
        iter_yaml = run_root / f"{fold}__iter{it}_{lever.lower()}.yaml"
        _write(iter_yaml, new_yaml)
        _log(decision | {"applied": True, "yaml": str(iter_yaml),
                         "log_comment": new_log_comment})

        _launch_training(tissue, backbone, fold, task, iter_yaml, args.dry_run)
        run_dir = _find_run_dir(new_log_comment)
        if run_dir is None:
            print(f"  ERROR: no run_dir for {new_log_comment}; aborting")
            _log({"phase": "iter_failed", "iter": it,
                  "note": "no run_dir found"})
            return 2

        val = _read_validation(run_dir)
        new_f1 = val["report"]["_summary"]["f1_macro"]
        delta = new_f1 - best_f1
        accepted = delta >= MIN_IMPROVEMENT
        _log({"phase": "iter_result", "iter": it, "lever": lever,
              "run_dir": str(run_dir), "f1_macro": new_f1, "delta": delta,
              "accepted": accepted})
        print(f"  result: macro-F1 = {new_f1:.4f}   delta = {delta:+.4f}   "
              f"{'ACCEPTED' if accepted else 'rejected'}")

        if accepted:
            best_f1 = new_f1
            cur_yaml = new_yaml
            cur_report = val["report"]
            consecutive_rejections = 0
        else:
            consecutive_rejections += 1
            lever_idx += 1
            if consecutive_rejections >= 3:
                print("\n[stop] three consecutive rejections; plateau reached.")
                _log({"phase": "stop", "reason": "plateau"})
                break

    print(f"\n[done] best macro-F1 = {best_f1:.4f}")
    _log({"phase": "done", "best_f1": best_f1})
    return 0


if __name__ == "__main__":
    sys.exit(main())
