"""Pipeline stages. Each stage is ``run(cfg, samples, out) -> dict`` and is
idempotent: the DAG skips it when the manifest marks it done.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths, splits as splits_mod, subproc, weights as weights_mod

CONFUSION_CMAP = "Blues"


def _write_label_map(path: Path, label_map: dict) -> None:
    import json

    path.write_text(
        "".join(f"{i}: {json.dumps(label_map[i])}\n" for i in sorted(label_map)))


def _run_tag(cfg) -> str:
    return f"{cfg.tissue}-{cfg.task}-{cfg.backbone.lower()}"


def _find_run_dir(cfg, out: Path, *, required: bool = True):
    """Newest CellViT run belonging to THIS tissue/task/backbone.

    The shared ``$CELLVIT_ROOT/logs_local`` tree holds every run ever trained, so
    picking the globally newest checkpoint pairs one tissue's weights with
    another's label_map. Prefer the per-tissue log root; fall back to the shared
    tree only for runs whose directory carries this run's log_comment.
    """
    import os

    tag = _run_tag(cfg)
    roots = [(paths.logs_dir(out, cfg.tissue), False)]
    cellvit = os.environ.get("CELLVIT_ROOT")
    if cellvit:
        roots.append((Path(cellvit) / "logs_local", True))
    for root, needs_tag in roots:
        if not root.is_dir():
            continue
        cands = sorted(root.rglob("checkpoints/model_best.pth"),
                       key=lambda p: p.stat().st_mtime)
        if needs_tag:
            # _run_tag only lowercases the backbone; the path is compared
            # lowercased, so a capitalised --tissue/--task would never match.
            cands = [p for p in cands if tag.lower() in str(p).lower()]
        if cands:
            return cands[-1].parent.parent
    if required:
        raise RuntimeError(f"no trained run for {tag!r}; run the train stage first")
    return None


def _to_rgb8(arr, axes: str):
    """Normalise a raw array (or window of one) to (H, W, 3) uint8."""
    import numpy as np

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


def read_he_rgb(path) -> "Any":
    """Read a WSI as (H, W, 3) uint8 regardless of how the vendor stored the axes.

    Several Xenium H&E exports are CYX, so ``shape[:2]`` silently yields (3, H).
    """
    import tifffile

    with tifffile.TiffFile(str(path)) as tf:
        ser = tf.series[0]
        return _to_rgb8(ser.asarray(), ser.axes)


class SlideReader:
    """Windowed WSI access.

    Tiling only ever needs one tile at a time, but a full-resolution slide is
    tens of gigabytes once decoded. When zarr is available the pixels stay on
    disk and each window is decoded on demand; otherwise this degrades to a
    single full read, matching the previous behaviour.
    """

    def __init__(self, path):
        import tifffile

        self._tf = tifffile.TiffFile(str(path))
        series = self._tf.series[0]
        self.axes = series.axes
        self._store = None
        self._arr = None
        try:
            self._z = self._open_lazy(series)
        except Exception:
            self._z = None
        if self._z is None:
            self._arr = _to_rgb8(series.asarray(), self.axes)
            self._close_handles()

        shape = self._arr.shape if self._z is None else self._z.shape
        if self._z is not None and self._channel_first:
            self.height, self.width = int(shape[1]), int(shape[2])
        else:
            self.height, self.width = int(shape[0]), int(shape[1])

    def _open_lazy(self, series):
        """Full-resolution sliceable array backed by the file, or None.

        Isolated so the windowing logic can be exercised without zarr; note
        ``series.aszarr()`` imports zarr itself, so both halves need it.
        """
        import zarr

        self._store = series.aszarr()
        return self._full_res(zarr.open(self._store, mode="r"))

    @staticmethod
    def _full_res(node):
        """Level 0 of a pyramidal OME-TIFF group, or the node itself."""
        if hasattr(node, "shape"):
            return node
        levels = sorted(node.array_keys(), key=lambda k: int(k))
        return node[levels[0]]

    @property
    def _channel_first(self) -> bool:
        return self.axes.startswith("C") or self.axes.startswith("S")

    @property
    def lazy(self) -> bool:
        return self._z is not None

    def window(self, y0: int, x0: int, h: int, w: int):
        import numpy as np

        if self._z is None:
            return self._arr[y0:y0 + h, x0:x0 + w]
        if self._channel_first:
            raw = self._z[:, y0:y0 + h, x0:x0 + w]
        else:
            raw = self._z[y0:y0 + h, x0:x0 + w]
        return _to_rgb8(np.asarray(raw), self.axes)

    def _close_handles(self):
        for attr in ("_store", "_tf"):
            handle = getattr(self, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def close(self):
        self._z = None
        self._arr = None
        self._close_handles()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def assignment_csv(outs: Path, task: str) -> Path | None:
    """Locate a sample's cell-type assignment CSV, or None if absent.

    kurtorank is inconsistent about the suffix: it writes ``pantissue``,
    ``hne``, ``pannuke`` and the ``sthelar_*`` tasks as
    ``celltype_assignment_<task>_label.csv`` but ``subtype``, ``major`` and
    ``hne_type`` as ``celltype_assignment_<task>.csv``. Looking only for the
    ``_label`` form made those three tasks unreachable even though every
    sample had the data.

    ``_label`` is tried first: where both exist (``pannuke`` on older runs)
    the un-suffixed file is the legacy one.
    """
    for name in (f"celltype_assignment_{task}_label.csv",
                 f"celltype_assignment_{task}.csv"):
        p = Path(outs) / name
        if p.exists():
            return p
    return None


def annotate(cfg, samples, out: Path) -> dict[str, Any]:
    """KurtoRank annotate over every sample's outs/ → celltype_assignment CSV."""
    import shutil
    import subprocess

    if not samples:
        raise RuntimeError(f"no samples discovered under {cfg.input}")

    # Resolved lazily: a dataset that is already annotated must not require
    # kurtorank on PATH just to confirm there is nothing to do.
    exe = None

    done = []
    for s in samples:
        # Must be the current task's vocabulary; another task's CSV is not a substitute.
        wanted = f"celltype_assignment_{cfg.task}_label.csv"
        found = assignment_csv(s.outs, cfg.task)
        if found is not None:
            done.append(found.name)
            continue
        if exe is None:
            exe = shutil.which("kurtorank")
            if exe is None:
                raise RuntimeError(
                    f"{s.sample_id}: {wanted} is missing and kurtorank is not on "
                    f"PATH. Install it (pip install kurtorank) or skip this stage "
                    f"with --run-skip annotate if the CSVs are provided another way.")
        # kurtorank needs a concrete tissue; cfg.tissue may be "pantissue" or a comma list.
        cmd = [exe, "annotate", "--xenium-dir", str(s.outs),
               "--tissue-type", s.tissue, "--output-dir", str(s.outs),
               "--use-graphclust", "--use-top-k-markers", str(cfg.top_k_markers)]
        if cfg.markers_csv:
            cmd += ["--markers-csv", str(cfg.markers_csv)]
        subprocess.run(cmd, check=True)
        found = assignment_csv(s.outs, cfg.task)
        if found is None:
            # Silently continuing marks the stage done and resurfaces as a
            # FileNotFoundError in transfer, far from the real cause.
            raise RuntimeError(
                f"{s.sample_id}: kurtorank annotate exited 0 but did not write "
                f"{wanted} into {s.outs}. Check that --tissue-type {s.tissue!r} "
                f"is a tissue kurtorank supports.")
        done.append(found.name)
    return {"n_samples": len(samples), "assignments": done}


def _segment_recipe(cfg) -> dict[str, Any]:
    """The settings that decide what a mask contains.

    Only the chosen backend's knobs count: re-segmenting a stardist cohort
    because --cellpose-model moved would cost hours for an identical result.
    """
    recipe: dict[str, Any] = {"segmenter": cfg.segmenter, "mpp": cfg.mpp}
    if cfg.segmenter == "cellpose":
        recipe.update(cellpose_model=cfg.cellpose_model,
                      cellpose_flow_threshold=cfg.cellpose_flow_threshold,
                      diameter=cfg.diameter)
    else:
        recipe.update(
            stardist_model=cfg.stardist_model,
            stardist_normalization_pmin=cfg.stardist_normalization_pmin,
            stardist_normalization_pmax=cfg.stardist_normalization_pmax,
            stardist_model_dir=(str(cfg.stardist_model_dir)
                                if cfg.stardist_model_dir else None))
    return recipe


def _segment_recipe_path(mask_dir: Path) -> Path:
    # Beside the mask dir, not inside it: prereq treats a non-empty mask dir as
    # proof that segmentation ran, and a lone sidecar is not that.
    return mask_dir.parent / f"{mask_dir.name}.recipe.json"


def reset_cache(stage: str, cfg, out: Path) -> None:
    """Drop the per-file caches `--force` is expected to invalidate.

    Marking a stage not-done is not enough for stages that skip work per input
    file; segment reuses any mask already on disk, so --force would be a no-op.
    """
    if stage != "segment":
        return
    mask_dir = paths.masks_dir(out, cfg.tissue)
    for p in mask_dir.glob("*.npy"):
        p.unlink()
    _segment_recipe_path(mask_dir).unlink(missing_ok=True)


def segment(cfg, samples, out: Path) -> dict[str, Any]:
    """Segment nuclei on each H&E (Cellpose/StarDist) → instance masks .npy."""
    import json

    import numpy as np

    from ..segment import get_segmenter

    try:  # only needed to release the cellpose allocator between slides
        import torch
    except ImportError:
        torch = None

    # "0" selects device 0, matching configrender._gpu_id; only explicit
    # cpu-ish values turn the GPU off.
    gpu_mode = str(cfg.gpus).strip().lower() not in {"cpu", "none", "false", "no", ""}
    seg = get_segmenter(cfg.segmenter, cellpose_model=cfg.cellpose_model,
                        diameter=cfg.diameter, batch_size=cfg.cellpose_batch_size,
                        flow_threshold=cfg.cellpose_flow_threshold,
                        gpu=gpu_mode,
                        stardist_model=cfg.stardist_model,
                        stardist_model_dir=cfg.stardist_model_dir,
                        stardist_cpu=cfg.stardist_cpu,
                        stardist_norm_pmin=cfg.stardist_normalization_pmin,
                        stardist_norm_pmax=cfg.stardist_normalization_pmax)
    mask_dir = paths.masks_dir(out, cfg.tissue)
    mask_dir.mkdir(parents=True, exist_ok=True)
    # The mask filename carries no segmenter, so without this a cellpose run
    # would silently adopt (and report as its own) masks StarDist wrote.
    recipe = _segment_recipe(cfg)
    recipe_path = _segment_recipe_path(mask_dir)
    previous = json.loads(recipe_path.read_text()) if recipe_path.exists() else None
    stale = previous is not None and previous != recipe
    if stale:
        print(f"[segment] settings changed since the existing masks "
              f"({previous} -> {recipe}); re-segmenting")
    recipe_path.write_text(json.dumps(recipe, sort_keys=True))
    counts = {}
    from tqdm import tqdm
    for s in tqdm(samples, desc="segment", unit="slide", ascii=" =", dynamic_ncols=True):
        dst = mask_dir / f"{s.sample_id}.npy"
        if dst.exists() and not stale:
            counts[s.sample_id] = int(np.load(dst, mmap_mode="r").max())
            continue
        he = read_he_rgb(s.he)
        mask = seg.segment(he, mpp=cfg.mpp)
        np.save(dst, mask)
        counts[s.sample_id] = int(mask.max())
        del he, mask  # free large arrays before next slide
        # Cellpose leaves tens of GB reserved; without this the next slide OOMs.
        if torch is not None and torch.cuda.is_available():
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

    mask_dir = paths.masks_dir(out, cfg.tissue)
    nuc_dir = paths.nuclei_dir(out, cfg.tissue)
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
        # Centroids may arrive as ints; pandas>=2 refuses the float assignment below.
        out = out.astype({"x_centroid": "float64", "y_centroid": "float64"})
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
        assign_path = assignment_csv(outs, cfg.task)
        if assign_path is None:
            raise RuntimeError(
                f"{s.sample_id}: no celltype_assignment_{cfg.task}[_label].csv in "
                f"{outs}. Run the annotate stage, or pick a --task whose CSV exists.")
        assign = pd.read_csv(assign_path)
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
                f"entry in {assign_path.name}. The assignment is "
                f"stale relative to clusters.csv — rerun `wsitrain run` with "
                f"--force.")
        if lost:
            print(f"[transfer] {s.sample_id}: {lost:.2%} of cells in unassigned clusters")
        return m.rename(columns={"x_centroid": "x_um", "y_centroid": "y_um", "cell_type": "label"})

    def _to_px(s, df, mask):
        from ..bunwarp import map_cells
        params = Path(s.outs) / "registration_params.json"
        elastic = Path(s.outs) / "direct_transf.txt"
        if cfg.transform == "none":
            xpx = (df["x_um"] / cfg.mpp).to_numpy()
            ypx = (df["y_um"] / cfg.mpp).to_numpy()
        else:
            if not params.exists():
                raise RuntimeError(
                    f"{s.sample_id}: --transform {cfg.transform} needs {params}, "
                    f"which is missing. Register with ST2WSI first, or pass "
                    f"--transform none to scale by mpp alone.")
            if cfg.transform == "affine+bspline" and not elastic.exists():
                raise RuntimeError(
                    f"{s.sample_id}: --transform affine+bspline needs {elastic}, "
                    f"which is missing. Pass --transform affine to use the SIFT "
                    f"affine alone.")
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
        border, which would manufacture matches along the slide edge. Points
        that are themselves inside stay eligible: it is the *neighbour* offset
        that is clamped, so a cell within ``radius`` of the edge is not lost.
        """
        h, w = mask.shape
        r = max(int(radius), 0)
        inb = (xpx >= 0) & (xpx < w) & (ypx >= 0) & (ypx < h)
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
            got[todo] = mask[np.clip(yi[todo] + dy, 0, h - 1),
                             np.clip(xi[todo] + dx, 0, w - 1)]
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
    _write_label_map(paths.label_map_path(out, cfg.tissue), label_map)

    counts, rates, dropped = {}, {}, []
    conflicts, used = {}, set()
    for s in tqdm(samples, desc="transfer:join", unit="slide", ascii=" =", dynamic_ncols=True):
        df = frames.pop(s.sample_id)  # release each frame as we go
        # Only scattered points are read, so the mask never needs to be resident.
        mask_path = mask_dir / f"{s.sample_id}.npy"
        if not mask_path.exists():
            raise SystemExit(
                f"[transfer] no mask for {s.sample_id}: {mask_path} is missing. "
                "The segment stage ran over a different sample set (most often "
                "because --transform changed); re-run `wsitrain segment` with "
                "the transform this command is using.")
        mask = np.load(mask_path, mmap_mode="r")
        xpx, ypx = _to_px(s, df, mask)
        nid = _lookup(mask, xpx, ypx, cfg.match_radius_px)
        df = df.assign(x_px=xpx, y_px=ypx, nucleus_id=nid,
                       class_int=df["label"].map(name_to_int))
        # Registration quality is how many cells landed on a nucleus at all; the
        # dedup below is nucleus density, and must not be charged against it.
        rate = int((nid > 0).sum()) / max(len(nid), 1)
        rates[s.sample_id] = round(rate, 4)
        if rate < cfg.min_match_rate:
            # Below this the surviving matches are mostly chance collisions with
            # whatever nucleus happens to sit under a mis-registered point.
            dropped.append(s.sample_id)
            print(f"[transfer] DROP {s.sample_id}: match rate {rate:.1%} "
                  f"< min_match_rate {cfg.min_match_rate:.0%}")
            del mask, df
            continue
        df = df[df.nucleus_id > 0]
        # Several Xenium cells can land on one nucleus. Only genuinely conflicting
        # labels are unusable; agreeing ones collapse to a single row.
        agree = df.groupby("nucleus_id")["class_int"].transform("nunique").eq(1)
        conflicts[s.sample_id] = int((~agree).sum())
        df = df[agree].drop_duplicates("nucleus_id", keep="first")
        df[["x_px", "y_px", "class_int"]].to_csv(nuc_dir / f"{s.sample_id}.csv", index=False)
        counts[s.sample_id] = len(df)
        used.update(int(v) for v in df["class_int"].unique())
        del mask, df  # free mask + joined frame before next slide
    if not counts:
        raise RuntimeError("every slide fell below min_match_rate; registration is broken")
    # Classes carried only by dropped slides would otherwise sit in label_map with
    # zero tiles and collect the full inverse-frequency cap.
    if used and len(used) < len(label_map):
        remap = {old: new for new, old in enumerate(sorted(used))}
        ghosts = [label_map[k] for k in sorted(set(label_map) - used)]
        label_map = {new: label_map[old] for old, new in remap.items()}
        _write_label_map(paths.label_map_path(out, cfg.tissue), label_map)
        for sid in counts:
            p = nuc_dir / f"{sid}.csv"
            t = pd.read_csv(p)
            t["class_int"] = t["class_int"].map(remap)
            t.to_csv(p, index=False)
        print(f"[transfer] dropped {len(ghosts)} class(es) with no surviving cells: {ghosts}")
    print(f"[transfer] kept {len(counts)}/{len(samples)} slides, "
          f"median match rate {float(np.median(list(rates.values()))):.1%}")
    return {"n_classes": len(label_map), "cells_per_sample": counts,
            "match_rate": rates, "dropped_slides": dropped,
            "conflicting_cells": conflicts}




def tile(cfg, samples, out: Path) -> dict[str, Any]:
    """Emit tile_px PNG + sibling x,y,class_int CSV (legacy CellViT contract).

    Cells live in per-sample nuclei CSVs (x_px,y_px,class_int at H&E pixel
    resolution). Tiles are cut on a stride grid; tiles with < min_cells or
    mostly-background mean RGB > bg_thresh are dropped. Coordinates in each CSV
    are tile-local pixels.
    """
    import pandas as pd
    from PIL import Image

    if _is_cellcls(cfg):
        return {"skipped": "non-end2end model; cells are cut by the crop stage"}

    nuc_dir = paths.nuclei_dir(out, cfg.tissue)
    img_dir = paths.images_dir(out, cfg.tissue)
    lab_dir = paths.labels_dir(out, cfg.tissue)
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    stride = max(int(cfg.tile_px * (1 - cfg.overlap)), 1)
    written = 0
    from tqdm import tqdm
    for s in tqdm(samples, desc="tile", unit="slide", ascii=" =", dynamic_ncols=True):
        csv = nuc_dir / f"{s.sample_id}.csv"
        if not csv.exists():        # dropped by the transfer QC
            continue
        cells = pd.read_csv(csv)
        with SlideReader(s.he) as reader:
            h, w = reader.height, reader.width
            for ti, y0 in enumerate(range(0, h - cfg.tile_px + 1, stride)):
                for tj, x0 in enumerate(range(0, w - cfg.tile_px + 1, stride)):
                    # Cheap cell filter first: most tiles fail it, and decoding
                    # their pixels only to discard them dominates the stage.
                    sub = cells[(cells.x_px >= x0) & (cells.x_px < x0 + cfg.tile_px) &
                                (cells.y_px >= y0) & (cells.y_px < y0 + cfg.tile_px)]
                    if len(sub) < cfg.min_cells:
                        continue
                    patch = reader.window(y0, x0, cfg.tile_px, cfg.tile_px)
                    if float(patch.mean()) > cfg.bg_thresh:
                        continue
                    stem = f"{s.sample_id}_tile_{ti:05d}_{tj:05d}"
                    Image.fromarray(patch).save(img_dir / f"{stem}.png")
                    sub.assign(x=sub.x_px - x0, y=sub.y_px - y0)[["x", "y", "class_int"]].to_csv(
                        lab_dir / f"{stem}.csv", header=False, index=False)
                    written += 1
    return {"tiles": written}



def _is_cellcls(cfg) -> bool:
    """True when an external detector supplies the cells and we only classify."""
    return getattr(cfg, "object_detection", "end2end") != "end2end"


def crop(cfg, samples, out: Path) -> dict[str, Any]:
    """Emit one HDF5 of centred cell crops per slide (non-end-to-end contract).

    Reads the same nuclei CSVs as `tile`, but cuts a patch_size_pixels window
    around each cell resampled to patch_spacing_um_px.
    """
    import pandas as pd
    from tqdm import tqdm

    from .. import cellcls

    if not _is_cellcls(cfg):
        return {"skipped": "end2end model; tiles are cut by the tile stage"}

    nuc_dir = paths.nuclei_dir(out, cfg.tissue)
    dst_dir = paths.cells_dir(out, cfg.tissue)
    dst_dir.mkdir(parents=True, exist_ok=True)
    read_px = cellcls.read_px_for(cfg.patch_size_pixels, cfg.patch_spacing_um_px, cfg.mpp)
    total, files = 0, []
    for s in tqdm(samples, desc="crop", unit="slide", ascii=" =", dynamic_ncols=True):
        csv = nuc_dir / f"{s.sample_id}.csv"
        if not csv.exists():        # dropped by the transfer QC
            continue
        cells = pd.read_csv(csv)
        with SlideReader(s.he) as reader:
            n = cellcls.crop_cells_to_h5(
                reader, cells, dst_dir / f"{s.sample_id}.h5",
                patch_px=cfg.patch_size_pixels, read_px=read_px,
                sample_id=s.sample_id,
                patch_spacing_um_px=cfg.patch_spacing_um_px,
                stain_normalization=bool(cfg.stain_normalization),
                norm_sample_size=cfg.norm_sample_size, seed=cfg.seed)
        if n:
            files.append(s.sample_id)
        total += n
    if total == 0:
        raise RuntimeError(
            f"crop wrote no cells; check that {nuc_dir} holds transfer output.")
    return {"cells": total, "slides": len(files), "read_px": read_px}


def _split_cells(cfg, out: Path) -> dict[str, Any]:
    """Whole-slide split: a cell must never appear on both sides."""
    import random

    files = sorted(paths.cells_dir(out, cfg.tissue).glob("*.h5"))
    if len(files) < 2:
        raise RuntimeError(
            f"need at least 2 slides to split, found {len(files)} in "
            f"{paths.cells_dir(out, cfg.tissue)}")
    order = list(files)
    random.Random(cfg.seed).shuffle(order)
    n_val = max(int(round(len(order) * cfg.val_frac)), 1)
    val, train = order[:n_val], order[n_val:]
    if not train:
        raise RuntimeError("val_frac left no training slides")
    d = paths.splits_dir(out, cfg.tissue, cfg.fold)
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.txt").write_text("\n".join(str(p) for p in train) + "\n")
    (d / "val.txt").write_text("\n".join(str(p) for p in val) + "\n")
    return {"mode": "slide", "n_train_slides": len(train), "n_val_slides": len(val)}


def _cell_split_files(cfg, out: Path) -> tuple[list[Path], list[Path]]:
    d = paths.splits_dir(out, cfg.tissue, cfg.fold)
    def _read(name):
        p = d / name
        if not p.exists():
            raise RuntimeError(f"missing {p}; run the split stage first")
        return [Path(x) for x in p.read_text().split() if x]
    return _read("train.txt"), _read("val.txt")


def _cellcls_run_dir(cfg, out: Path) -> Path:
    return paths.logs_dir(out, cfg.tissue) / f"cellcls_{cfg.fold}"


def _cellcls_classes(cfg, out: Path) -> list[str]:
    from ..weights import load_label_map

    label_map = load_label_map(paths.label_map_path(out, cfg.tissue))
    return [label_map[i] for i in sorted(label_map)]


def _train_cells(cfg, out: Path) -> dict[str, Any]:
    import json

    import numpy as np

    from .. import cellcls

    train_files, val_files = _cell_split_files(cfg, out)
    class_names = _cellcls_classes(cfg, out)
    # Statistics come from the training slides only; the validation set would
    # leak into the exported config's Normalize.
    mean, std = cellcls.compute_norm_stats(train_files, seed=cfg.seed)

    counts = np.bincount(cellcls.CellH5Dataset(train_files).labels(),
                         minlength=len(class_names)).astype(float)
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    weights = np.minimum(weights / max(weights[weights > 0].min(), 1e-9), cfg.weight_cap)

    run_dir = _cellcls_run_dir(cfg, out)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "norm_stats.json").write_text(
        json.dumps({"mean": mean, "std": std}, indent=2))

    res = cellcls.train_classifier(
        train_files, val_files, architecture=cfg.architecture,
        patch_px=cfg.patch_size_pixels, num_classes=len(class_names),
        mean=mean, std=std, out_dir=run_dir, epochs=cfg.epochs,
        batch_size=cfg.batch_size, lr=cfg.lr, weight_decay=cfg.weight_decay,
        num_workers=cfg.num_workers, pretrained=cfg.pretrained,
        class_weights=weights.tolist())
    res["run_dir"] = str(run_dir)
    return res


def _validate_cells(cfg, out: Path) -> dict[str, Any]:
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from torch.utils.data import DataLoader

    from .. import cellcls

    _, val_files = _cell_split_files(cfg, out)
    class_names = _cellcls_classes(cfg, out)
    run_dir = _cellcls_run_dir(cfg, out)
    stats = json.loads((run_dir / "norm_stats.json").read_text())

    model, _ = cellcls.load_checkpoint(run_dir / "model_best.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loader = DataLoader(
        cellcls.CellH5Dataset(val_files, stats["mean"], stats["std"], train=False),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    acc, preds, gts = cellcls.evaluate(model, loader, device)

    rd = paths.report_dir(out, cfg.tissue)
    rd.mkdir(parents=True, exist_ok=True)
    metrics = {"val_accuracy": acc, "n_val_cells": int(len(gts))}
    (rd / "scores.json").write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(gts, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names).plot(
        ax=ax, colorbar=True, xticks_rotation=45, cmap=CONFUSION_CMAP,
        values_format=".2f", text_kw={"fontsize": 8})
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.set_title("Normalised")
    fig.tight_layout()
    fig.savefig(rd / "confusion_matrix.png", dpi=600)
    plt.close(fig)
    return {"run_dir": str(run_dir), "metrics": metrics}


def _export_cells(cfg, out: Path) -> dict[str, Any]:
    import json
    import shutil

    from .. import cellcls

    class_names = _cellcls_classes(cfg, out)
    run_dir = _cellcls_run_dir(cfg, out)
    stats = json.loads((run_dir / "norm_stats.json").read_text())
    dst = paths.models_dir(out, cfg.tissue) / "main"
    dst.mkdir(parents=True, exist_ok=True)
    cellcls.export_torchscript(run_dir / "model_best.pth",
                               dst / "torchscript_model.pt", cfg.patch_size_pixels)

    config = {
        "spec_version": "1.0",
        "architecture": cfg.architecture,
        "num_classes": len(class_names),
        "class_names": class_names,
        "patch_size_pixels": cfg.patch_size_pixels,
        "spacing_um_px": cfg.patch_spacing_um_px,
        "transform": [
            {"name": "Resize", "arguments": {"size": cfg.patch_size_pixels}},
            {"name": "ToTensor"},
            {"name": "Normalize",
             "arguments": {"mean": stats["mean"], "std": stats["std"]}},
        ],
        "stain_normalization": cfg.stain_normalization,
        "object_based": True,
        "object_detection": {
            "name": cfg.object_detection,
            "normalization_pmin": cfg.stardist_normalization_pmin,
            "normalization_pmax": cfg.stardist_normalization_pmax,
        },
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy2(paths.label_map_path(out, cfg.tissue), dst / "label_map.yaml")
    return {"model_dir": str(dst), "classes": class_names}


def split(cfg, samples, out: Path) -> dict[str, Any]:
    if _is_cellcls(cfg):
        return _split_cells(cfg, out)
    label_dir = paths.labels_dir(out, cfg.tissue)
    res = splits_mod.split_tiles(label_dir, val_frac=cfg.val_frac,
                                 by_slide=cfg.by_slide, seed=cfg.seed)
    if not res.train or not res.val:
        # CellViT picks its best checkpoint on the validation set, so an empty
        # side trains a model that can never be scored or early-stopped.
        raise RuntimeError(
            f"split produced {len(res.train)} train / {len(res.val)} val tiles; "
            f"both sides must be non-empty. Adjust val_frac or tile more slides.")
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

    if _is_cellcls(cfg):
        return _train_cells(cfg, out)

    cellvit = os.environ.get("CELLVIT_ROOT")
    if not cellvit or not Path(cellvit).is_dir():
        raise RuntimeError("set $CELLVIT_ROOT to the CellViT-plus-plus checkout")
    cfg_path = paths.train_config_path(out, cfg.tissue, cfg.backbone, cfg.fold)
    if not cfg_path.exists():
        raise RuntimeError(f"missing train config: {cfg_path}")
    py = shutil.which("python3") or "python"
    env = subproc.child_env(cellvit)
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
    import shutil

    if _is_cellcls(cfg):
        return _validate_cells(cfg, out)

    run_dir = _find_run_dir(cfg, out)
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
    pred_pt = val_results_dir / "predictions.pt"
    gt_pt   = val_results_dir / "gt.pt"
    if pred_pt.exists() and gt_pt.exists():
        import torch
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

        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
        disp.plot(ax=ax, colorbar=True, xticks_rotation=45,
                  cmap=CONFUSION_CMAP, values_format=".2f",
                  text_kw={"fontsize": 8})
        # sklearn only sets the rotation, which leaves 45-degree labels centred
        # under the tick rather than ending at it.
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
        ax.set_title("Normalised")
        fig.tight_layout()
        fig.savefig(rd / "confusion_matrix.png", dpi=600)
        plt.close(fig)

    if not metrics and not (pred_pt.exists() and gt_pt.exists()):
        # Returning quietly here reads as "validated, nothing to report".
        print(f"[validate] WARNING: no scores.json or predictions.pt under "
              f"{val_results_dir}; the trainer writes them only when it saves a "
              f"new best checkpoint. Nothing was scored — check the train log.")
    return {"run_dir": str(run_dir), "metrics": metrics}


def export(cfg, samples, out: Path) -> dict[str, Any]:
    """Convert best checkpoint to TorchScript via cellvit_convert_to_torchscript.py,
    then assemble a wsinsight-ready model folder under models/<tissue>/main/."""
    import json
    import os
    import shutil
    import subprocess

    from ..weights import load_label_map

    if _is_cellcls(cfg):
        return _export_cells(cfg, out)

    cellvit = os.environ.get("CELLVIT_ROOT")
    if not cellvit or not Path(cellvit).is_dir():
        raise RuntimeError("set $CELLVIT_ROOT to the CellViT-plus-plus checkout")
    best = _find_run_dir(cfg, out) / "checkpoints" / "model_best.pth"
    dst = paths.models_dir(out, cfg.tissue) / "main"
    dst.mkdir(parents=True, exist_ok=True)

    ts_out = dst / "torchscript_model.pt"
    py = shutil.which("python3") or "python"
    env = subproc.child_env(cellvit)
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
        # Required by wsinsight's model-config schema; mirrors the shipped
        # zoo/huangch/CellViT-SAM-H-x40 config.
        "transform": [
            {"name": "Resize", "arguments": {"size": cfg.tile_px}},
            {"name": "ToTensor"},
            {"name": "Normalize",
             "arguments": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        ],
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
    "tile": tile, "crop": crop, "split": split, "train": train,
    "validate": validate, "export": export, "report": report,
}
