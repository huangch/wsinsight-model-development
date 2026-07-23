"""Pipeline stages. Each stage is ``run(cfg, samples, out) -> dict`` and is
idempotent: the DAG skips it when the manifest marks it done.

Heavy stages (segment/transfer/tile/train/export) are scaffolded with their
contracts and raise NotImplementedError until the porting from pipeline.old's
Groovy + train_tissue.sh is complete. annotate, split and report are wired.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths, splits as splits_mod, weights as weights_mod


def annotate(cfg, samples, out: Path) -> dict[str, Any]:
    """KurtoRank annotate over every sample's outs/ → celltype_assignment CSV."""
    import shutil
    import subprocess

    exe = shutil.which("kurtorank")
    if exe is None:
        raise RuntimeError("kurtorank not on PATH; pip install kurtorank")
    if not samples:
        raise RuntimeError(f"no samples discovered under {cfg.input}")

    done = []
    for s in samples:
        pattern = f"celltype_assignment_*_label.csv"
        existing = list(Path(s.outs).glob(pattern))
        if existing:
            done.append(existing[0].name)
            continue
        cmd = [exe, "annotate", "--xenium-dir", str(s.outs),
               "--tissue-type", cfg.tissue, "--output-dir", str(s.outs),
               "--use-graphclust", "--use-top-k-markers", str(cfg.top_k_markers)]
        if cfg.markers_csv:
            cmd += ["--markers-csv", str(cfg.markers_csv)]
        subprocess.run(cmd, check=True)
        hits = list(Path(s.outs).glob(pattern))
        done.append(hits[0].name if hits else "MISSING")
    return {"n_samples": len(samples), "assignments": done}


def segment(cfg, samples, out: Path) -> dict[str, Any]:
    """Segment nuclei on each H&E (Cellpose/StarDist) → instance masks .npy."""
    import numpy as np
    import tifffile

    from ..segment import get_segmenter

    seg = get_segmenter(cfg.segmenter, cellpose_model=cfg.cellpose_model,
                        diameter=cfg.diameter, batch_size=cfg.cellpose_batch_size,
                        flow_threshold=cfg.cellpose_flow_threshold)
    mask_dir = out / "masks" / cfg.tissue
    mask_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    from tqdm import tqdm
    for s in tqdm(samples, desc="segment", unit="slide", ascii=" =", dynamic_ncols=True):
        dst = mask_dir / f"{s.sample_id}.npy"
        if dst.exists():
            counts[s.sample_id] = int(np.load(dst).max())
            continue
        he = tifffile.imread(str(s.he))
        mask = seg.segment(he, mpp=cfg.mpp)
        np.save(dst, mask)
        counts[s.sample_id] = int(mask.max())
        del he, mask  # free large arrays before next slide
    return {"segmenter": seg.name, "nuclei_per_sample": counts}


def transfer(cfg, samples, out: Path) -> dict[str, Any]:
    """Coordinate-join KurtoRank cell labels onto H&E nuclei (replaces QuPath).

    Reads each sample's annotated.h5ad (KurtoRank output) for per-cell
    centroids (microns) + assigned label, converts µm→pixels via cfg.mpp,
    looks up the H&E nucleus at that pixel, and writes a per-nucleus table
    (nucleus_id, x_px, y_px, class_int). Builds a canonical label_map.yaml.
    Assumes H&E is registered to Xenium space.
    """
    import numpy as np
    import pandas as pd

    mask_dir = out / "masks" / cfg.tissue
    nuc_dir = out / "nuclei" / cfg.tissue
    nuc_dir.mkdir(parents=True, exist_ok=True)

    def _per_cell(s) -> pd.DataFrame:
        outs = Path(s.outs)
        cells = pd.read_parquet(outs / "cells.parquet")[["cell_id", "x_centroid", "y_centroid"]]
        assign = pd.read_csv(outs / f"celltype_assignment_{cfg.task}_label.csv")
        cl = pd.read_csv(outs / "analysis/clustering/gene_expression_graphclust/clusters.csv")
        cl = cl.rename(columns={"Barcode": "cell_id", "Cluster": "classification"})
        m = cells.merge(cl, on="cell_id").merge(assign, on="classification")
        return m.rename(columns={"x_centroid": "x_um", "y_centroid": "y_um", "cell_type": "label"})

    def _to_px(s, df, mask):
        from ..bunwarp import map_cells
        params = Path(s.outs) / "registration_params.json"
        elastic = Path(s.outs) / "direct_transf.txt"
        if cfg.transform == "none" or not params.exists():
            xpx = (df["x_um"] / cfg.mpp).to_numpy()
            ypx = (df["y_um"] / cfg.mpp).to_numpy()
        else:
            # target_wh = full-res H&E (target) dims; the nucleus mask is at that
            # resolution, so its shape supplies the bUnwarpJ lattice extent.
            xy = map_cells(df[["x_um", "y_um"]].to_numpy(), params,
                           elastic if elastic.exists() else None, cfg.transform,
                           target_wh=(mask.shape[1], mask.shape[0]))
            xpx, ypx = xy[:, 0], xy[:, 1]
        xpx = np.clip(np.round(xpx).astype(int), 0, mask.shape[1] - 1)
        ypx = np.clip(np.round(ypx).astype(int), 0, mask.shape[0] - 1)
        return xpx, ypx

    # Pass 1: label vocabulary across samples.
    from tqdm import tqdm
    frames, labels = {}, set()
    for s in tqdm(samples, desc="transfer:read", unit="slide", ascii=" =", dynamic_ncols=True):
        df = _per_cell(s)
        frames[s.sample_id] = df
        labels.update(df["label"].unique())
    label_map = {i: n for i, n in enumerate(sorted(labels))}
    name_to_int = {v: k for k, v in label_map.items()}
    paths.tissue_root(out, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(out, cfg.tissue).write_text(
        "\n".join(f'{i}: "{n}"' for i, n in label_map.items()) + "\n")

    counts = {}
    for s in tqdm(samples, desc="transfer:join", unit="slide", ascii=" =", dynamic_ncols=True):
        df = frames.pop(s.sample_id)  # release each frame as we go
        mask = np.load(mask_dir / f"{s.sample_id}.npy")
        xpx, ypx = _to_px(s, df, mask)
        df = df.assign(x_px=xpx, y_px=ypx, class_int=df["label"].map(name_to_int))
        df = df[mask[ypx, xpx] > 0]
        df[["x_px", "y_px", "class_int"]].to_csv(nuc_dir / f"{s.sample_id}.csv", index=False)
        counts[s.sample_id] = len(df)
        del mask, df  # free mask + joined frame before next slide
    return {"n_classes": len(label_map), "cells_per_sample": counts}




def tile(cfg, samples, out: Path) -> dict[str, Any]:
    """Emit tile_px PNG + sibling x,y,class_int CSV (legacy CellViT contract).

    Cells live in per-sample nuclei CSVs (x_px,y_px,class_int at H&E pixel
    resolution). Tiles are cut on a stride grid; tiles with < min_cells or
    mostly-background mean RGB > bg_thresh are dropped. Coordinates in each CSV
    are tile-local pixels.
    """
    import numpy as np
    import pandas as pd
    import tifffile
    from PIL import Image

    nuc_dir = out / "nuclei" / cfg.tissue
    img_dir = paths.images_dir(out, cfg.tissue)
    lab_dir = paths.labels_dir(out, cfg.tissue)
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stride = int(cfg.tile_px * (1 - cfg.overlap))
    written = 0
    from tqdm import tqdm
    for s in tqdm(samples, desc="tile", unit="slide", ascii=" =", dynamic_ncols=True):
        he = tifffile.imread(str(s.he))
        cells = pd.read_csv(nuc_dir / f"{s.sample_id}.csv")
        h, w = he.shape[:2]
        for ti, y0 in enumerate(range(0, h - cfg.tile_px + 1, stride)):
            for tj, x0 in enumerate(range(0, w - cfg.tile_px + 1, stride)):
                patch = he[y0:y0 + cfg.tile_px, x0:x0 + cfg.tile_px]
                if float(np.asarray(patch).mean()) > cfg.bg_thresh:
                    continue
                sub = cells[(cells.x_px >= x0) & (cells.x_px < x0 + cfg.tile_px) &
                            (cells.y_px >= y0) & (cells.y_px < y0 + cfg.tile_px)]
                if len(sub) < cfg.min_cells:
                    continue
                stem = f"{s.sample_id}_tile_{ti * 10000 + tj:05d}"
                Image.fromarray(np.asarray(patch)[..., :3].astype("uint8")).save(img_dir / f"{stem}.png")
                sub.assign(x=sub.x_px - x0, y=sub.y_px - y0)[["x", "y", "class_int"]].to_csv(
                    lab_dir / f"{stem}.csv", header=False, index=False)
                written += 1
        del he  # free WSI array after all tiles for this slide
    return {"tiles": written}



def split(cfg, samples, out: Path) -> dict[str, Any]:
    label_dir = paths.labels_dir(out, cfg.tissue)
    res = splits_mod.split_tiles(label_dir, val_frac=cfg.val_frac,
                                 by_slide=cfg.by_slide, seed=cfg.seed)
    splits_mod.write_split(res, paths.splits_dir(out, cfg.tissue, cfg.fold))
    wr = weights_mod.compute_weights(paths.label_map_path(out, cfg.tissue),
                                     label_dir, cap=cfg.weight_cap)
    from ..configrender import render_config
    cfgp = render_config(cfg, out)
    return {"mode": res.mode, "n_train": len(res.train), "n_val": len(res.val),
            "weights": wr.weights, "config": str(cfgp)}


def train(cfg, samples, out: Path) -> dict[str, Any]:
    """Render the fold config + invoke CellViT++. Locate the trainer via
    $CELLVIT_ROOT (the vendored CellViT-plus-plus checkout)."""
    import os
    import shutil
    import subprocess

    cellvit = os.environ.get("CELLVIT_ROOT")
    if not cellvit or not Path(cellvit).is_dir():
        raise RuntimeError("set $CELLVIT_ROOT to the CellViT-plus-plus checkout")
    cfg_path = (paths.tissue_root(out, cfg.tissue) / "train_configs"
                / cfg.backbone / f"{cfg.fold}.yaml")
    if not cfg_path.exists():
        raise RuntimeError(f"missing train config: {cfg_path}")
    py = shutil.which("python3") or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = cellvit  # CellViT++ requires its root on sys.path
    subprocess.run([py, str(Path(cellvit) / "cellvit" / "train_cell_classifier_head.py"),
                    "--config", str(cfg_path)],
                   cwd=cellvit, env=env, check=True)
    if cfg.tune and cfg.tune > 0:
        from ..tuning import run_tune
        return run_tune(cfg, out, cellvit, base_config=cfg_path, py=py)
    return {"config": str(cfg_path)}


def validate(cfg, samples, out: Path) -> dict[str, Any]:
    """Read val_results produced by the trainer and generate a confusion-matrix PNG.

    The trainer stores val_results/scores.json (scalar metrics) and
    val_results/predictions.pt + val_results/gt.pt (raw tensors) whenever a new
    best checkpoint is saved.  We surface those into the report directory and
    build a confusion-matrix figure from the raw tensors.
    """
    import json
    import os
    import shutil

    import numpy as np

    cellvit = os.environ.get("CELLVIT_ROOT")
    logs = Path(cellvit) / "logs_local" if cellvit else None
    if not logs or not logs.is_dir():
        raise RuntimeError("no logs_local under $CELLVIT_ROOT; run train first")
    runs = sorted(logs.rglob("checkpoints/model_best.pth"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise RuntimeError("no trained run; train first")
    run_dir = runs[-1].parent.parent
    val_results_dir = run_dir / "val_results"

    rd = paths.report_dir(out, cfg.tissue)
    rd.mkdir(parents=True, exist_ok=True)

    metrics = {}
    # --- scores.json ---------------------------------------------------------
    scores_src = val_results_dir / "scores.json"
    if scores_src.exists():
        shutil.copy2(scores_src, rd / "scores.json")
        with open(scores_src) as f:
            metrics = json.load(f)

    # --- confusion matrix from stored tensors --------------------------------
    import torch
    pred_pt = val_results_dir / "predictions.pt"
    gt_pt   = val_results_dir / "gt.pt"
    if pred_pt.exists() and gt_pt.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        from ..weights import load_label_map
        label_map = load_label_map(paths.label_map_path(out, cfg.tissue))
        class_names = [label_map[i] for i in sorted(label_map)]

        preds = torch.load(pred_pt, map_location="cpu").numpy()
        gts   = torch.load(gt_pt,   map_location="cpu").numpy()
        cm    = confusion_matrix(gts, preds, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, mat, title in zip(axes, [cm, cm_norm], ["Counts", "Normalised"]):
            disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=class_names)
            disp.plot(ax=ax, colorbar=False, xticks_rotation="vertical")
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(rd / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    return {"run_dir": str(run_dir), "metrics": metrics}


def export(cfg, samples, out: Path) -> dict[str, Any]:
    """Convert best checkpoint to TorchScript via cellvit_convert_to_torchscript.py,
    then assemble a wsinsight-ready model folder under models/<tissue>/main/."""
    import json
    import os
    import shutil
    import subprocess

    from ..weights import load_label_map

    cellvit = os.environ.get("CELLVIT_ROOT")
    if not cellvit or not Path(cellvit).is_dir():
        raise RuntimeError("set $CELLVIT_ROOT to the CellViT-plus-plus checkout")
    logs = Path(cellvit) / "logs_local"
    if not logs.is_dir():
        raise RuntimeError("no logs_local under $CELLVIT_ROOT; run train first")
    cands = sorted(logs.rglob("checkpoints/model_best.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise RuntimeError("no model_best.pth found; training incomplete")
    best = cands[-1]
    dst = paths.models_dir(out, cfg.tissue) / "main"
    dst.mkdir(parents=True, exist_ok=True)

    ts_out = dst / "torchscript_model.pt"
    py = shutil.which("python3") or "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = cellvit
    subprocess.run(
        [py,
         str(Path(cellvit) / "cellvit" / "cellvit_convert_to_torchscript.py"),
         "--checkpoint", str(best),
         "--output",     str(ts_out),
         "--height",     str(cfg.tile_px),
         "--width",      str(cfg.tile_px)],
        cwd=cellvit, env=env, check=True,
    )

    label_map = load_label_map(paths.label_map_path(out, cfg.tissue))
    class_names = [label_map[i] for i in sorted(label_map)]
    config = {
        "spec_version": "1.0", "architecture": "cellvit",
        "num_classes": len(class_names), "class_names": class_names,
        "patch_size_pixels": cfg.tile_px, "halo_size_pixels": 0,
        "spacing_um_px": cfg.mpp,
        "backbone": cfg.backbone,
        "stain_normalization": False, "object_based": True,
        "mixed_precision": False, "object_detection": {"name": "end2end"},
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(paths.label_map_path(out, cfg.tissue), dst / "label_map.yaml")
    return {"model_dir": str(dst), "classes": class_names}


def report(cfg, samples, out: Path) -> dict[str, Any]:
    rd = paths.report_dir(out, cfg.tissue)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "summary.txt").write_text(
        f"tissue: {cfg.tissue}\nsamples: {len(samples)}\nsegmenter: {cfg.segmenter}\n"
        f"transform: {cfg.transform}\n")
    return {"report_dir": str(rd)}


STAGE_FUNCS = {
    "annotate": annotate, "segment": segment, "transfer": transfer,
    "tile": tile, "split": split, "train": train, "validate": validate,
    "export": export, "report": report,
}
