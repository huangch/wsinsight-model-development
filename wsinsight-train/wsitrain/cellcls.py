"""Non-end-to-end cell classifier: crop cells, train a torchvision backbone.

The end-to-end path (CellViT/HoVer-Net) predicts detection and class from a
large tile. Here an external detector supplies the nuclei and this model only
classifies a small crop centred on each one, which is what wsinsight runs when
``object_detection.name`` is ``stardist`` rather than ``end2end``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Cells are stored per slide so a split can keep whole slides on one side.
IMAGES = "images"
IMAGES_RAW = "images_raw"
LABELS = "labels"
X_PX = "x_px"
Y_PX = "y_px"


def build_model(architecture: str, num_classes: int, patch_px: int,
                pretrained: bool = False):
    """Any torchvision classifier, by name."""
    from torchvision.models import get_model

    weights = "DEFAULT" if pretrained else None
    try:
        return get_model(architecture, weights=weights, num_classes=num_classes)
    except TypeError:
        # ViT/Swin fix their input resolution at construction time.
        return get_model(architecture, weights=weights, num_classes=num_classes,
                         image_size=patch_px)


def read_px_for(patch_px: int, patch_spacing_um_px: float, slide_mpp: float) -> int:
    """Window to read so that, once resized to patch_px, one pixel is patch_spacing."""
    if slide_mpp <= 0:
        raise ValueError(f"slide mpp must be > 0, got {slide_mpp}")
    return max(int(round(patch_px * patch_spacing_um_px / slide_mpp)), 1)


def stain_target_matrix():
    """The reference H&E matrix wsinsight normalises towards at inference."""
    import histomicstk as htk
    import numpy as np

    smap = htk.preprocessing.color_deconvolution.stain_color_map
    # Column order matches run_inference.py; swapping it changes the result.
    return np.array([smap[s] for s in ("eosin", "hematoxylin", "null")]).T


def estimate_stain_matrix(crops):
    """Macenko source matrix from a stack of crops, as one tall image."""
    import histomicstk as htk
    import numpy as np

    stacked = np.asarray(crops, dtype=np.float64).reshape(-1, crops.shape[2], 3)
    return htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(
        stacked + 1e-6, 255)


def _normalise(patch, w_source, w_target):
    import histomicstk as htk
    import numpy as np

    out = htk.preprocessing.color_normalization.deconvolution_based_normalization(
        patch.astype(np.float64) + 1e-6, W_source=w_source, W_target=w_target)
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_cells_to_h5(reader, cells, dst: Path, *, patch_px: int, read_px: int,
                     sample_id: str, patch_spacing_um_px: float,
                     stain_normalization: bool = False,
                     norm_sample_size: int = 256, seed: int = 0) -> int:
    """Write one HDF5 of centred crops. Returns the number of cells written.

    Cells whose window would leave the slide are dropped: a padded crop would
    teach the model that a class lives against a black edge.
    """
    import h5py
    import numpy as np
    from PIL import Image

    half = read_px // 2
    keep = cells[(cells.x_px >= half) & (cells.x_px < reader.width - half) &
                 (cells.y_px >= half) & (cells.y_px < reader.height - half)]
    n = len(keep)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if n == 0:
        return 0

    def _read(row):
        y0, x0 = int(row.y_px) - half, int(row.x_px) - half
        patch = reader.window(y0, x0, read_px, read_px)
        if patch.shape[0] != read_px or patch.shape[1] != read_px:
            return None
        if read_px != patch_px:
            patch = np.asarray(
                Image.fromarray(patch).resize((patch_px, patch_px), Image.BILINEAR))
        return patch

    w_source = w_target = None
    if stain_normalization:
        # Estimating on every cell is linear in pixels and needlessly slow;
        # a sample is what the inference path uses too.
        rng = np.random.default_rng(seed)
        take = min(norm_sample_size, n)
        idx = rng.choice(n, size=take, replace=False)
        sample = [p for p in (_read(keep.iloc[int(i)]) for i in idx) if p is not None]
        if not sample:
            raise RuntimeError(f"{sample_id}: no readable crops to estimate stain matrix")
        w_source = estimate_stain_matrix(np.stack(sample))
        w_target = stain_target_matrix()

    with h5py.File(dst, "w") as f:
        def _dset(name):
            return f.create_dataset(
                name, shape=(n, patch_px, patch_px, 3), dtype="uint8",
                maxshape=(None, patch_px, patch_px, 3),
                chunks=(1, patch_px, patch_px, 3), compression="lzf")

        images = _dset(IMAGES)
        # Re-reading the slide costs ~5 ms/cell against ~1 ms to re-normalise,
        # so the untouched pixels are kept for a second normalisation pass.
        raw = _dset(IMAGES_RAW) if w_source is not None else None
        written = 0
        xs, ys, ls = [], [], []
        for row in keep.itertuples(index=False):
            patch = _read(row)
            if patch is None:
                continue
            if raw is not None:
                raw[written] = patch
                patch = _normalise(patch, w_source, w_target)
            images[written] = patch
            xs.append(int(row.x_px))
            ys.append(int(row.y_px))
            ls.append(int(row.class_int))
            written += 1
        images.resize((written, patch_px, patch_px, 3))
        if raw is not None:
            raw.resize((written, patch_px, patch_px, 3))
        f.create_dataset(LABELS, data=np.asarray(ls, dtype="int64"))
        f.create_dataset(X_PX, data=np.asarray(xs, dtype="int32"))
        f.create_dataset(Y_PX, data=np.asarray(ys, dtype="int32"))
        f.attrs["sample_id"] = sample_id
        f.attrs["patch_size_pixels"] = patch_px
        f.attrs["spacing_um_px"] = patch_spacing_um_px
        f.attrs["stain_normalization"] = bool(stain_normalization)
        if w_source is not None:
            # Kept so a stored crop can be traced back to how it was normalised.
            f.attrs["stain_w_source"] = w_source
            f.attrs["stain_w_target"] = w_target
            f.attrs["norm_sample_size"] = int(min(norm_sample_size, n))
    return written


class CellH5Dataset:
    """Crops from a list of per-slide HDF5 files.

    The file handles are opened on first access inside each worker: an h5py
    handle inherited across a fork is not safe to read from.
    """

    def __init__(self, files: list[Path], mean=None, std=None, train: bool = False):
        import numpy as np

        self.files = [Path(f) for f in files]
        self.mean = mean
        self.std = std
        self.train = train
        self._handles: dict[int, Any] = {}

        counts = []
        for f in self.files:
            import h5py
            with h5py.File(f, "r") as h:
                counts.append(len(h[LABELS]))
        self._counts = counts
        self._offsets = np.cumsum([0] + counts)
        self._transform = self._build_transform()

    def _build_transform(self):
        import torchvision.transforms as T

        steps = []
        if self.train:
            steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(degrees=(0, 90))]
        steps.append(T.ToTensor())
        if self.mean is not None and self.std is not None:
            steps.append(T.Normalize(self.mean, self.std))
        return T.Compose(steps)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        import bisect

        fi = bisect.bisect_right(self._offsets, index) - 1
        return fi, index - int(self._offsets[fi])

    def _handle(self, fi: int):
        import h5py

        h = self._handles.get(fi)
        if h is None:
            h = h5py.File(self.files[fi], "r")
            self._handles[fi] = h
        return h

    def __getitem__(self, index: int):
        from PIL import Image

        fi, local = self._locate(index)
        h = self._handle(fi)
        img = Image.fromarray(h[IMAGES][local])
        return self._transform(img), int(h[LABELS][local])

    def labels(self):
        import h5py
        import numpy as np

        out = []
        for f in self.files:
            with h5py.File(f, "r") as h:
                out.append(np.asarray(h[LABELS]))
        return np.concatenate(out) if out else np.asarray([], dtype="int64")


def compute_norm_stats(files: list[Path], max_cells: int = 50000,
                       seed: int = 0) -> tuple[list[float], list[float]]:
    """Per-channel mean/std over a sample of TRAINING crops only."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(seed)
    chunks = []
    per_file = max(max_cells // max(len(files), 1), 1)
    for f in files:
        with h5py.File(f, "r") as h:
            n = len(h[LABELS])
            if n == 0:
                continue
            idx = np.sort(rng.choice(n, size=min(per_file, n), replace=False))
            chunks.append(np.asarray(h[IMAGES][idx], dtype="float64") / 255.0)
    if not chunks:
        raise RuntimeError("no crops to compute normalisation statistics from")
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=(0, 1, 2))
    std = stacked.std(axis=(0, 1, 2))
    return mean.tolist(), std.tolist()


def train_classifier(train_files, val_files, *, architecture: str, patch_px: int,
                     num_classes: int, mean, std, out_dir: Path,
                     epochs: int = 50, batch_size: int = 128, lr: float = 1e-4,
                     weight_decay: float = 1e-4, num_workers: int = 8,
                     pretrained: bool = False, class_weights=None,
                     device: str | None = None) -> dict[str, Any]:
    """Train and keep the checkpoint with the best validation accuracy."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = CellH5Dataset(train_files, mean, std, train=True)
    va = CellH5Dataset(val_files, mean, std, train=False)
    # Workers each reopen the HDF5 files, so persist them across epochs.
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=num_workers > 0)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=num_workers > 0)

    model = build_model(architecture, num_classes, patch_px, pretrained).to(device)
    w = None if class_weights is None else torch.tensor(
        class_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", patience=5)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "model_best.pth"
    best_acc, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(dl_tr, desc=f"epoch {epoch + 1}/{epochs}", unit="batch",
                         ascii=" =", dynamic_ncols=True, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(y)
        acc, _, _ = evaluate(model, dl_va, device)
        scheduler.step(acc)
        history.append({"epoch": epoch + 1, "train_loss": total / max(len(tr), 1),
                        "val_acc": acc})
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "architecture": architecture,
                        "num_classes": num_classes,
                        "patch_size_pixels": patch_px}, best_path)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"best_val_acc": best_acc, "checkpoint": str(best_path), "epochs": epochs}


def evaluate(model, loader, device: str):
    """Accuracy plus the raw predictions and ground truth."""
    import numpy as np
    import torch

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            preds.append(logits.argmax(1).cpu().numpy())
            gts.append(y.numpy())
    if not preds:
        return 0.0, np.asarray([]), np.asarray([])
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    return float((preds == gts).mean()), preds, gts


def load_checkpoint(path: Path, device: str = "cpu"):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["architecture"], ckpt["num_classes"],
                        ckpt["patch_size_pixels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_torchscript(checkpoint: Path, dst: Path, patch_px: int) -> Path:
    """Trace to the (B,3,P,P) -> (B,num_classes) logits contract wsinsight loads."""
    import torch

    model, _ = load_checkpoint(checkpoint)
    example = torch.zeros(1, 3, patch_px, patch_px)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    dst.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(dst))
    return dst
