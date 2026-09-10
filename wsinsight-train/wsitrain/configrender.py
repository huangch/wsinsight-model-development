"""Render the CellViT++ train config from the shipped template + class weights."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from string import Template

from . import paths, weights as _weights

TEMPLATE = Path(__file__).resolve().parent / "defaults" / "train_config_template.yaml"


def _split_hash(out: Path, cfg) -> str:
    """Content hash of the train/val split files. Injected into the CellViT
    config as ``data.hash_info`` so the extracted-cell cache is keyed on the
    actual tile membership. Without this the cache is keyed only on the split
    *filename* (which never changes), so a re-split silently reloads stale
    cells."""
    hasher = hashlib.sha256()
    splits_dir = paths.splits_dir(out, cfg.tissue, cfg.fold)
    for name in ("train.csv", "val.csv"):
        p = splits_dir / name
        hasher.update(name.encode("utf-8"))
        if p.exists():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()[:16]


def _backbone_weights(cellvit: str, backbone: str) -> str:
    if not cellvit:
        return ""
    name = f"CellViT-{backbone}.pth"
    root = Path(cellvit)
    for cand in (root / "models" / name, root.parent / "models" / name):
        if cand.exists():
            return str(cand)
    return str(root / "models" / name)


def _gpu_ids(cfg) -> list[str]:
    """Resolve --gpus into CUDA device ids.

    Accepted spellings (Docker-style):
        "all"                    -> torch.cuda.device_count() devices
        "0", "0,1", "1,3,5"      -> explicit list
        "cpu","none","false","no" -> fail-fast (train stage requires a GPU)

    Note: "auto" was removed; "all" is the Docker-style synonym.
    """
    raw = str(cfg.gpus).strip().lower()
    if raw == "all" or raw == "":
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n = 0
        if n == 0:
            raise SystemExit(
                "--gpus all requested but no CUDA devices are visible. Pass "
                "--gpus cpu (and --run-skip train validate export) if the host "
                "has no GPU, or check the driver install.")
        return [str(i) for i in range(n)]
    if raw in {"cpu", "none", "false", "no"}:
        # The template renders this into `gpu:`, a CUDA device index. Returning
        # "0" here would train on the GPU the same flag just told segment to
        # avoid, so refuse rather than pick a device the user ruled out.
        raise SystemExit(
            f"--gpus {cfg.gpus!r} turns the GPU off, but CellViT training needs "
            "a CUDA device. Pass --gpus <index> (e.g. --gpus 0), or stop before "
            "the split stage with `--run-skip split train validate export`.")
    # "0,1" / "0,1,3" etc -- validate each token is an integer.
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        return [str(int(p)) for p in parts]
    except ValueError:
        raise SystemExit(
            f"--gpus {cfg.gpus!r}: expected 'all', 'cpu', or a comma-separated "
            "list of device indices like '0' or '0,1'.")


def _gpu_id(cfg) -> str:
    """The one device the train stage uses.

    CellViT's cell-classifier builds ``f"cuda:{run_conf['gpu']}"`` and has no
    DataParallel path, so the config's ``gpu:`` field must be a single index;
    a comma-separated list reaches torch as the invalid device "cuda:0,1".
    """
    ids = _gpu_ids(cfg)
    if len(ids) > 1:
        print(f"[config] --gpus {cfg.gpus!r} -> {len(ids)} devices; CellViT "
              f"trains on one, using cuda:{ids[0]}")
    return ids[0]


def render_config(cfg, out: Path, *, drop_rate: float = 0.1, lr: float | None = None,
                  weights: list[float] | None = None) -> Path:
    rep = _weights.compute_weights(paths.label_map_path(out, cfg.tissue),
                                   paths.labels_dir(out, cfg.tissue), cap=cfg.weight_cap)
    label_map = rep.label_map
    wlist = weights or rep.weights
    cellvit = os.environ.get("CELLVIT_ROOT", "")
    weights_path = _backbone_weights(cellvit, cfg.backbone)
    # Per-tissue log root so validate/export cannot pick up another tissue's run.
    log_dir = paths.logs_dir(out, cfg.tissue)
    log_dir.mkdir(parents=True, exist_ok=True)
    body = Template(TEMPLATE.read_text()).substitute(
        TISSUE=cfg.tissue, TASK=cfg.task, BACKBONE=cfg.backbone,
        BACKBONE_LC=cfg.backbone.lower(), NUM_CLASSES=len(label_map), SEED=cfg.seed,
        FOLD=cfg.fold, TISSUE_ROOT=str(paths.tissue_root(out, cfg.tissue)),
        CELLVIT_LOGS=str(log_dir),
        CELLVIT_WEIGHTS=weights_path,
        NORMALIZE_STAINS="true" if cfg.stain_normalization else "false",
        DROP_RATE=drop_rate, LR=(cfg.lr if lr is None else lr),
        EPOCHS=cfg.epochs, WEIGHT_DECAY=cfg.weight_decay,
        HASH_INFO=_split_hash(out, cfg), GPU_ID=_gpu_id(cfg),
        # json.dumps gives a valid YAML double-quoted scalar, so ':' and '#' in a
        # cell-type name cannot break or silently truncate the config.
        LABEL_MAP="\n".join(f"    {i}: {json.dumps(label_map[i])}" for i in sorted(label_map)),
        WEIGHTS="[" + ", ".join(f"{w:g}" for w in wlist) + "]")
    dst = paths.train_config_path(out, cfg.tissue, cfg.backbone, cfg.fold)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(body)
    return dst
