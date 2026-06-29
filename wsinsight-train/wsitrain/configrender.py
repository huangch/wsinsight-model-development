"""Render the CellViT++ train config from the shipped template + class weights."""
from __future__ import annotations

import os
from pathlib import Path
from string import Template

from . import paths, weights as _weights

TEMPLATE = Path(__file__).resolve().parent / "defaults" / "train_config_template.yaml"


def render_config(cfg, out: Path, *, drop_rate: float = 0.1, lr: float = 0.0003,
                  weights: list[float] | None = None) -> Path:
    rep = _weights.compute_weights(paths.label_map_path(out, cfg.tissue),
                                   paths.labels_dir(out, cfg.tissue), cap=cfg.weight_cap)
    label_map = rep.label_map
    wlist = weights or rep.weights
    cellvit = os.environ.get("CELLVIT_ROOT", "")
    body = Template(TEMPLATE.read_text()).substitute(
        TISSUE=cfg.tissue, TASK=cfg.task, BACKBONE=cfg.backbone,
        BACKBONE_LC=cfg.backbone.lower(), NUM_CLASSES=len(label_map), SEED=cfg.seed,
        FOLD=cfg.fold, TISSUE_ROOT=str(paths.tissue_root(out, cfg.tissue)),
        CELLVIT_LOGS=str(Path(cellvit) / "logs_local") if cellvit else "logs_local",
        CELLVIT_WEIGHTS=str(Path(cellvit) / "models" / f"CellViT-{cfg.backbone}.pth") if cellvit else "",
        DROP_RATE=drop_rate, LR=lr,
        LABEL_MAP="\n".join(f"    {i}: {label_map[i]}" for i in sorted(label_map)),
        WEIGHTS="[" + ", ".join(f"{w:g}" for w in rep.weights) + "]")
    dst = paths.tissue_root(out, cfg.tissue) / "train_configs" / cfg.backbone / f"{cfg.fold}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(body)
    return dst
