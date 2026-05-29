"""Render train_configs/<backbone>/<fold>.yaml from the pantissue template.

Replaces pipeline/make_train_config.py. Uses targeted line edits (not full
YAML round-trip) so comments and key order survive intact, matching what the
upstream trainer expects.
"""
from __future__ import annotations

import re

from . import paths, weights as _weights


def _replace_line(text: str, pattern: str, repl: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n == 0:
        raise RuntimeError(f"could not find line matching: {pattern}")
    return new


def _replace_weight_list(text: str, weights: list[float],
                         comment_lines: list[str]) -> str:
    """Replace any pre-existing weight_list (inline or block form) with a
    fresh comment banner + inline weight_list."""
    weight_line = "  weight_list: [" + ", ".join(f"{w:g}" for w in weights) + "]"
    banner = "\n".join(f"  # {ln}" for ln in comment_lines)
    inline_pat = r"(  # [^\n]*\n)*  weight_list:[ \t]*\[[^\]]*\][ \t]*$"
    block_pat = r"(  # [^\n]*\n)*  weight_list:[ \t]*\n(?:  - [-\d.eE+]+\n)+"
    if re.search(inline_pat, text, flags=re.M):
        return re.sub(inline_pat, banner + "\n" + weight_line,
                      text, count=1, flags=re.M)
    if re.search(block_pat, text, flags=re.M):
        return re.sub(block_pat, banner + "\n" + weight_line + "\n",
                      text, count=1, flags=re.M)
    raise RuntimeError("could not locate existing weight_list in template")


def _format_label_map_block(label_map: dict[int, str], indent: str = "    ") -> str:
    return "\n".join(f"{indent}{ci}: {label_map[ci]}" for ci in sorted(label_map))


def render(tissue: str, *, backbone: str = paths.DEFAULT_BACKBONE,
           fold: str = paths.DEFAULT_FOLD, task: str = paths.TEMPLATE_TISSUE,
           weight_report: _weights.WeightReport | None = None) -> str:
    """Build a tissue-specific train YAML by substituting into the template.

    If ``weight_report`` is None, weights are recomputed from labels.
    """
    template_path = paths.template_config_path(backbone, fold)
    if not template_path.is_file():
        raise FileNotFoundError(f"template not found: {template_path}")

    rep = weight_report or _weights.compute_weights(tissue)
    label_map = rep.label_map
    if len(rep.weights) != len(label_map):
        raise ValueError(
            f"weights ({len(rep.weights)}) != label_map ({len(label_map)})"
        )

    out = template_path.read_text()
    out = _replace_line(out, r"^  project:.*$",
                        f"  project: cellvit-{tissue.replace('_', '-')}")
    out = _replace_line(out, r"^  notes:.*$",
                        f"  notes: {tissue}-{task}-{len(label_map)}class-{backbone}")
    out = _replace_line(out, r"^  log_comment:.*$",
                        f"  log_comment: {tissue}-{task}-{backbone.lower()}")
    out = out.replace(f"trainingset/{paths.TEMPLATE_TISSUE}",
                      f"trainingset/{tissue}")
    out = _replace_line(out, r"^(  num_classes:).*$",
                        f"\\1 {len(label_map)}")
    out = re.sub(r"^  label_map:\n(    .+\n)+",
                 "  label_map:\n" + _format_label_map_block(label_map) + "\n",
                 out, count=1, flags=re.M)
    out = _replace_weight_list(out, rep.weights,
                               _weights.format_report_comments(rep))
    return out
