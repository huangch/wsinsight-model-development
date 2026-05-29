"""
make_train_config.py
--------------------
Emit trainingset/<tissue>/train_configs/<backbone>/<fold>.yaml from the
pantissue template, substituting tissue-specific fields.

Per-tissue substitutions:
    - dataset_path, train_filelist, val_filelist (path stems)
    - num_classes (= len(label_map))
    - label_map block (from trainingset/<tissue>/label_map.yaml)
    - weight_list (computed by compute_class_weights.py)
    - logging.project / notes / log_comment

Usage:
    python make_train_config.py --tissue breast
    python make_train_config.py --tissue heart --backbone SAM-H-x40 --fold fold_0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CELLVIT_TRAINING_ROOT = SCRIPT_DIR.parent
TEMPLATE_TISSUE = "pantissue"


def _load_label_map(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open() as fp:
        for line in fp:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            k, _, v = line.partition(":")
            try:
                ci = int(k.strip())
            except ValueError:
                continue
            out[ci] = v.strip().strip('"').strip("'")
    return out


def _compute_weights(tissue: str) -> list[float]:
    """Re-uses compute_class_weights.py so we never duplicate the formula."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "compute_class_weights.py"),
         "--tissue", tissue],
        capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("weight_list:"):
            inner = line.split(":", 1)[1].strip().lstrip("[").rstrip("]")
            return [float(x) for x in inner.split(",")]
    raise RuntimeError("compute_class_weights.py produced no weight_list line.")


def _format_label_map_block(label_map: dict[int, str], indent: str = "    ") -> str:
    lines = []
    for ci in sorted(label_map):
        lines.append(f"{indent}{ci}: {label_map[ci]}")
    return "\n".join(lines)


def _build(template: str, *, tissue: str, label_map: dict[int, str],
           weights: list[float], backbone: str, task: str) -> str:
    out = template

    # Logging block: project/notes/log_comment hard-coded to pantissue.
    out = re.sub(
        r"^  project:.*$",
        f"  project: cellvit-{tissue.replace('_', '-')}",
        out, count=1, flags=re.M,
    )
    out = re.sub(
        r"^  notes:.*$",
        f"  notes: {tissue}-{task}-{len(label_map)}class-{backbone}",
        out, count=1, flags=re.M,
    )
    out = re.sub(
        r"^  log_comment:.*$",
        f"  log_comment: {tissue}-{task}-{backbone.lower()}",
        out, count=1, flags=re.M,
    )

    # Replace all "trainingset/pantissue/..." path stems with the target tissue.
    out = out.replace(
        f"trainingset/{TEMPLATE_TISSUE}",
        f"trainingset/{tissue}",
    )

    # num_classes
    out = re.sub(
        r"^(  num_classes:).*$",
        f"\\1 {len(label_map)}",
        out, count=1, flags=re.M,
    )

    # label_map block: replace everything from "  label_map:" up to the next
    # top-level key (a line beginning with a non-space character or "cellvit_path:").
    new_label_block = "  label_map:\n" + _format_label_map_block(label_map, "    ")
    out = re.sub(
        r"^  label_map:\n(    .+\n)+",
        new_label_block + "\n",
        out, count=1, flags=re.M,
    )

    # weight_list: replace the previous comment block AND the weight_list line.
    weight_line = "  weight_list: [" + ", ".join(f"{w:g}" for w in weights) + "]"
    # Remove all consecutive "  # ..." lines immediately above the existing
    # weight_list line, plus the weight_list line itself; replace with a
    # short banner + the new weight_line. Use compute_class_weights.py
    # output as the comment block.
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "compute_class_weights.py"),
         "--tissue", tissue],
        capture_output=True, text=True, check=True,
    )
    comment_block = "\n".join(
        "  " + ln for ln in proc.stdout.splitlines()
        if ln.startswith("#") and "weight" in ln.lower() or ln.startswith("# class")
    )
    # Handle BOTH inline (`weight_list: [...]`) and multi-line (`weight_list:\n  - x\n...`)
    # forms. Strip any preceding consecutive "  # ..." comment lines, then the
    # weight_list itself (whichever form), and replace with a fresh comment
    # banner + inline weight_line.
    inline_pat = r"(  # [^\n]*\n)*  weight_list:[ \t]*\[[^\]]*\][ \t]*$"
    block_pat = r"(  # [^\n]*\n)*  weight_list:[ \t]*\n(?:  - [-\d.eE+]+\n)+"
    if re.search(inline_pat, out, flags=re.M):
        out = re.sub(inline_pat, comment_block + "\n" + weight_line,
                     out, count=1, flags=re.M)
    elif re.search(block_pat, out, flags=re.M):
        out = re.sub(block_pat, comment_block + "\n" + weight_line + "\n",
                     out, count=1, flags=re.M)
    else:
        raise RuntimeError("Could not locate weight_list in template")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tissue", required=True)
    p.add_argument("--backbone", default="SAM-H-x40")
    p.add_argument("--fold", default="fold_0")
    p.add_argument("--task", default="pantissue",
                   help="Task tag used inside log_comment so it matches "
                        "train_tissue.sh's `<tissue>-<task>-<backbone>` pattern.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing config (default refuses).")
    args = p.parse_args()

    tissue_root = CELLVIT_TRAINING_ROOT / "trainingset" / args.tissue
    template_path = (CELLVIT_TRAINING_ROOT / "trainingset" / TEMPLATE_TISSUE
                     / "train_configs" / args.backbone / f"{args.fold}.yaml")
    out_path = (tissue_root / "train_configs" / args.backbone
                / f"{args.fold}.yaml")

    if not template_path.exists():
        print(f"ERROR: template {template_path} missing.", file=sys.stderr)
        return 1
    if not (tissue_root / "label_map.yaml").exists():
        print(f"ERROR: {tissue_root / 'label_map.yaml'} missing.", file=sys.stderr)
        return 1
    if out_path.exists() and not args.force:
        print(f"ERROR: {out_path} already exists; pass --force to overwrite.",
              file=sys.stderr)
        return 1

    label_map = _load_label_map(tissue_root / "label_map.yaml")
    weights = _compute_weights(args.tissue)

    if len(weights) != len(label_map):
        print(
            f"ERROR: weight vector length ({len(weights)}) != label_map size "
            f"({len(label_map)}); rerun compute_class_weights.py.",
            file=sys.stderr,
        )
        return 1

    template = template_path.read_text()
    rendered = _build(template, tissue=args.tissue, label_map=label_map,
                      weights=weights, backbone=args.backbone, task=args.task)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
