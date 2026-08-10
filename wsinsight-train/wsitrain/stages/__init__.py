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

CONFUSION_CMAP = "Blues"


def read_he_rgb(path) -> "Any":
    """Read a WSI as (H, W, 3) uint8 regardless of how the vendor stored the axes.

    Several Xenium H&E exports are CYX, so ``shape[:2]`` silently yields (3, H).
    """
    import numpy as np
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        ser = tf.series[0]
        axes = ser.axes
        arr = ser.asarray()
    if axes.startswith("C") or axes.startswith("S"):
        arr = np.moveaxis(arr, 0, -1)
    arr = np.atleast_3d(arr)[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, -1)
    if arr.dtype != np.uint8:
        info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
        hi = float(info.max) if info else float(arr.max() or 1)
        arr = (arr.astype(np.float32) / hi * 255).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


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
        # Must be the current task's vocabulary; another task's CSV is not a substitute.
        wanted = f"celltype_assignment_{cfg.task}_label.csv"
        if (Path(s.outs) / wanted).exists():
            done.append(wanted)
            continue
        # kurtorank needs a concrete tissue; cfg.tissue may be "pantissue" or a comma list.
        cmd = [exe, "annotate", "--xenium-dir", str(s.outs),
               "--tissue-type", s.tissue, "--output-dir", str(s.outs),
               "--use-graphclust", "--use-top-k-markers", str(cfg.top_k_markers)]
        if cfg.markers_csv:
            cmd += ["--markers-csv", str(cfg.markers_csv)]
        subprocess.run(cmd, check=True)
        done.append(wanted if (Path(s.outs) / wanted).exists() else "MISSING")
    return {"n_samples": len(samples), "assignments": done}


def segment(cfg, samples, out: Path) -> dict[str, Any]:
    """Segment nuclei on each H&E (Cellpose/StarDist) → instance masks .npy."""
    import numpy as np
    import torch

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
        he = read_he_rgb(s.he)
        mask = seg.segment(he, mpp=cfg.mpp)
        np.save(dst, mask)
        counts[s.sample_id] = int(mask.max())
        del he, mask  # free large arrays before next slide
        # Cellpose leaves tens of GB reserved; without this the next slide OOMs.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    def _nucleus_xy(outs: Path, cells: pd.DataFrame) -> pd.DataFrame:
        """Replace the cell centroid with the nucleus centroid where available.

        x_centroid/y_centroid is the centroid of the *cell*. In tissues where the
        nucleus is a small part of the cell (heart 5%, liver 15%) that point sits
        in cytoplasm, and an exact nucleus lookup fails for reasons unrelated to
        registration -- heart goes from 19% to 73% matched with this swap.
        """
        nbp = outs / "nucleus_boundaries.parquet"
        if not nbp.exists():
            return cells
        nb = pd.read_parquet(nbp)
        nb.columns = [c.lower() for c in nb.columns]
        xs = [c for c in nb.columns if c.endswith("vertex_x")]
        ys = [c for c in nb.columns if c.endswith("vertex_y")]
        if not (xs and ys and "cell_id" in nb.columns):
            return cells
        nb["cell_id"] = nb["cell_id"].map(
            lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
        nuc = nb.groupby("cell_id")[[xs[0], ys[0]]].mean()
        nuc.columns = ["nx", "ny"]
        out = cells.join(nuc, on="cell_id")
        keep = out["nx"].notna()
        out.loc[keep, "x_centroid"] = out.loc[keep, "nx"]
        out.loc[keep, "y_centroid"] = out.loc[keep, "ny"]
        return out.drop(columns=["nx", "ny"])

    def _per_cell(s) -> pd.DataFrame:
        outs = Path(s.outs)
        cells = pd.read_parquet(outs / "cells.parquet")[["cell_id", "x_centroid", "y_centroid"]]
        # Some Xenium releases store cell_id as bytes; clusters.csv is always text.
        cells["cell_id"] = cells["cell_id"].map(
            lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)
        cells = _nucleus_xy(outs, cells)
        assign = pd.read_csv(outs / f"celltype_assignment_{cfg.task}_label.csv")
        cl = pd.read_csv(outs / "analysis/clustering/gene_expression_graphclust/clusters.csv")
        cl = cl.rename(columns={"Barcode": "cell_id", "Cluster": "classification"})
        m = cells.merge(cl, on="cell_id")
        if len(m) < 0.5 * len(cells):
            raise RuntimeError(
                f"{s.sample_id}: only {len(m)}/{len(cells)} cells matched clusters.csv "
                f"on cell_id — the barcode keys do not line up.")
        n_pre = len(m)
        m = m.merge(assign, on="classification")
        # The join key is a cluster *number*. A few unassigned tail clusters are
        # normal (kurtorank skips tiny ones); losing a large share instead means
        # clustering was rerun and the numbers no longer mean the same thing.
        lost = 1 - len(m) / max(n_pre, 1)
        if lost > 0.05:
            orphan = sorted(set(cl["classification"]) - set(assign["classification"]))
            raise RuntimeError(
                f"{s.sample_id}: {lost:.1%} of cells fall in clusters {orphan} with no "
                f"entry in celltype_assignment_{cfg.task}_label.csv. The assignment is "
                f"stale relative to clusters.csv — rerun `wsitrain run --from annotate`.")
        if lost:
            print(f"[transfer] {s.sample_id}: {lost:.2%} of cells in unassigned clusters")
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
        return np.round(xpx).astype(np.int64), np.round(ypx).astype(np.int64)

    def _lookup(mask, xpx, ypx, radius: int):
        """Nucleus id under each point, searching outward to ``radius``.

        Out-of-bounds points are reported as 0 rather than clipped onto the
        border, which would manufacture matches along the slide edge.
        """
        h, w = mask.shape
        r = max(int(radius), 0)
        inb = (xpx >= r) & (xpx < w - r) & (ypx >= r) & (ypx < h - r)
        nid = np.zeros(len(xpx), mask.dtype)
        xi, yi = xpx[inb], ypx[inb]
        got = mask[yi, xi]
        offsets = [(0, 0)] if r == 0 else sorted(
            ((dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
             if dy * dy + dx * dx <= r * r and (dy or dx)),
            key=lambda t: t[0] ** 2 + t[1] ** 2)
        for dy, dx in offsets:                      # nearest ring first
            todo = got == 0
            if not todo.any():
                break
            got[todo] = mask[yi[todo] + dy, xi[todo] + dx]
        nid[inb] = got
        return nid

    # Pass 1: label vocabulary across samples.
    from tqdm import tqdm
    drop = {str(d).lower() for d in (cfg.drop_labels or ())}
    frames, labels = {}, set()
    for s in tqdm(samples, desc="transfer:read", unit="slide", ascii=" =", dynamic_ncols=True):
        df = _per_cell(s)
        if drop:
            df = df[~df["label"].str.lower().isin(drop)]
        frames[s.sample_id] = df
        labels.update(df["label"].unique())
    label_map = {i: n for i, n in enumerate(sorted(labels))}
    name_to_int = {v: k for k, v in label_map.items()}
    paths.tissue_root(out, cfg.tissue).mkdir(parents=True, exist_ok=True)
    paths.label_map_path(out, cfg.tissue).write_text(
        "\n".join(f'{i}: "{n}"' for i, n in label_map.items()) + "\n")

    counts, rates, dropped = {}, {}, []
    for s in tqdm(samples, desc="transfer:join", unit="slide", ascii=" =", dynamic_ncols=True):
        df = frames.pop(s.sample_id)  # release each frame as we go
        mask = np.load(mask_dir / f"{s.sample_id}.npy")
        xpx, ypx = _to_px(s, df, mask)
        nid = _lookup(mask, xpx, ypx, cfg.match_radius_px)
        df = df.assign(x_px=xpx, y_px=ypx, nucleus_id=nid,
                       class_int=df["label"].map(name_to_int))
        df = df[df.nucleus_id > 0]
        # Several Xenium cells can land on one nucleus with conflicting labels;
        # an arbitrary winner would teach the model contradictions.
        df = df.drop_duplicates("nucleus_id", keep=False)
        rate = len(df) / max(len(nid), 1)
        rates[s.sample_id] = round(rate, 4)
        if rate < cfg.min_match_rate:
            # Below this the surviving matches are mostly chance collisions with
            # whatever nucleus happens to sit under a mis-registered point.
            dropped.append(s.sample_id)
            print(f"[transfer] DROP {s.sample_id}: match rate {rate:.1%} "
                  f"< min_match_rate {cfg.min_match_rate:.0%}")
            del mask, df
            continue
        df[["x_px", "y_px", "class_int"]].to_csv(nuc_dir / f"{s.sample_id}.csv", index=False)
        counts[s.sample_id] = len(df)
        del mask, df  # free mask + joined frame before next slide
    if not counts:
        raise RuntimeError("every slide fell below min_match_rate; registration is broken")
    print(f"[transfer] kept {len(counts)}/{len(samples)} slides, "
          f"median match rate {float(np.median(list(rates.values()))):.1%}")
    return {"n_classes": len(label_map), "cells_per_sample": counts,
            "match_rate": rates, "dropped_slides": dropped}




def tile(cfg, samples, out: Path) -> dict[str, Any]:
    """Emit tile_px PNG + sibling x,y,class_int CSV (legacy CellViT contract).

    Cells live in per-sample nuclei CSVs (x_px,y_px,class_int at H&E pixel
    resolution). Tiles are cut on a stride grid; tiles with < min_cells or
    mostly-background mean RGB > bg_thresh are dropped. Coordinates in each CSV
    are tile-local pixels.
    """
    import numpy as np
    import pandas as pd
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
        csv = nuc_dir / f"{s.sample_id}.csv"
        if not csv.exists():        # dropped by the transfer QC
            continue
        he = read_he_rgb(s.he)
        cells = pd.read_csv(csv)
        h, w = he.shape[:2]
        for ti, y0 in enumerate(range(0, h - cfg.tile_px + 1, stride)):
            for tj, x0 in enumerate(range(0, w - cfg.tile_px + 1, stride)):
                patch = he[y0:y0 + cfg.tile_px, x0:x0 + cfg.tile_px]
                if float(patch.mean()) > cfg.bg_thresh:
                    continue
                sub = cells[(cells.x_px >= x0) & (cells.x_px < x0 + cfg.tile_px) &
                            (cells.y_px >= y0) & (cells.y_px < y0 + cfg.tile_px)]
                if len(sub) < cfg.min_cells:
                    continue
                stem = f"{s.sample_id}_tile_{ti * 10000 + tj:05d}"
                Image.fromarray(patch).save(img_dir / f"{stem}.png")
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
        for ax, mat, title, fmt in zip(axes, [cm, cm_norm],
                                       ["Counts", "Normalised"], ["d", ".2f"]):
            disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=class_names)
            disp.plot(ax=ax, colorbar=False, xticks_rotation="vertical",
                      cmap=CONFUSION_CMAP, values_format=fmt)
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
