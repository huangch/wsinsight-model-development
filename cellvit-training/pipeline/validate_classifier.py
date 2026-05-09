"""
validate_classifier.py
----------------------
Tissue-agnostic validator. Runs inference with a trained LinearClassifier
checkpoint (or fused CellViTWithLinearClassifier checkpoint) on a
validation split and produces a classification report plus confusion
matrices saved as 600 DPI PNG and SVG.

Two ways to invoke:

1) By tissue (recommended) — auto-derives --dataset / --filelist / --label-map
   from `tissue_configs/<tissue>.yaml`:

    python validate_classifier.py \
        --tissue colorectal \
        --checkpoint <run_dir>/checkpoints/model_best.pth

2) By explicit paths (back-compat):

    python validate_classifier.py \
        --checkpoint <path/to/model_best.pth> \
        --dataset    <trainingset_root> \
        --filelist   <trainingset_root>/splits/fold_0/val.csv \
        --label-map  <trainingset_root>/label_map.yaml \
        --outdir     <run_dir>/validation

The script accepts both checkpoint types supported by
cellvit_convert_to_torchscript.py:
  (a) LinearClassifier checkpoint  (arch = LinearClassifier)
  (b) Full CellViT checkpoint       (arch = CellViTSAM / CellViT256 / etc.)
"""

import argparse
import os
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

current_dir = os.path.dirname(os.path.abspath(__file__))
cellvit_root = os.path.join(current_dir, "cellvit", "CellViT-plus-plus")
sys.path.insert(0, cellvit_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

from cellvit.training.datasets.detection_dataset import DetectionDataset
from cellvit.inference.postprocessing_cupy import DetectionCellPostProcessorCupy


# ── Re-use the same loader from cellvit_convert_to_torchscript.py ────────────

def unflatten_dict(d: dict, sep: str = ".") -> dict:
    output_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        cur = output_dict
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
    return output_dict


def load_models(checkpoint_path: str, device: torch.device,
                cellvit_path_override: str | None = None):
    """
    Returns (cellvit_model, classifier, num_classes, label_map_from_ckpt)
    where classifier may be None if the checkpoint is a full CellViT (no head).

    If `cellvit_path_override` is given, it replaces the path baked into the
    LinearClassifier checkpoint's config. Otherwise, when the embedded
    cellvit_path no longer resolves, we look for a same-basename file under
    `<cellvit-training>/cellvit/models/` (derived from this script's location)
    so checkpoints stay loadable after folder renames or repo moves.
    """
    from cellvit.models.cell_segmentation.cellvit import CellViT
    from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    from cellvit.models.cell_segmentation.cellvit_uni import CellViTUNI
    from cellvit.models.cell_segmentation.cellvit_virchow import CellViTVirchow
    from cellvit.models.classifier.linear_classifier import LinearClassifier

    def _build_cellvit(run_conf, arch):
        if arch == "CellViT":
            return CellViT(
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                embed_dim=run_conf["model"]["embed_dim"],
                input_channels=run_conf["model"].get("input_channels", 3),
                depth=run_conf["model"]["depth"],
                num_heads=run_conf["model"]["num_heads"],
                extract_layers=run_conf["model"]["extract_layers"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif arch == "CellViT256":
            return CellViT256(
                model256_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif arch == "CellViTSAM":
            return CellViTSAM(
                model_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
                vit_structure=run_conf["model"]["backbone"],
                regression_loss=run_conf["model"].get("regression_loss", False),
            )
        elif arch == "CellViTUNI":
            return CellViTUNI(
                model_uni_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            )
        elif arch == "CellViTVirchow":
            return CellViTVirchow(
                model_virchow_path=None,
                num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
                num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            )
        raise NotImplementedError(f"Unknown arch: {arch}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]

    if arch == "LinearClassifier":
        cls_conf = unflatten_dict(ckpt["config"], ".")
        cellvit_path = cls_conf["cellvit_path"]
        if cellvit_path_override:
            cellvit_path = cellvit_path_override
        elif not os.path.exists(cellvit_path):
            # Checkpoints embed an absolute cellvit_path captured at training
            # time. If that path no longer exists (folder renamed, repo moved,
            # etc.), look for a same-basename file under the in-tree models dir
            # `<cellvit-training>/cellvit/models/`.
            from pathlib import Path as _Path
            cellvit_training_root = _Path(__file__).resolve().parent.parent
            candidate = cellvit_training_root / "cellvit" / "models" / os.path.basename(cellvit_path)
            if candidate.is_file():
                print(f"  Remapping stale cellvit_path -> {candidate}")
                cellvit_path = str(candidate)
        print(f"  Detected LinearClassifier checkpoint.")
        print(f"  Loading base CellViT from: {cellvit_path}")

        base_ckpt = torch.load(cellvit_path, map_location="cpu", weights_only=False)
        run_conf = unflatten_dict(base_ckpt["config"], ".")
        cellvit = _build_cellvit(run_conf, base_ckpt["arch"])
        cellvit.load_state_dict(base_ckpt["model_state_dict"])
        cellvit.eval().to(device)

        embed_dim = ckpt["model_state_dict"]["fc1.weight"].shape[1]
        num_classes = cls_conf["data"]["num_classes"]
        label_map = cls_conf["data"].get("label_map", {})

        classifier = LinearClassifier(
            embed_dim=embed_dim,
            hidden_dim=cls_conf["model"].get("hidden_dim", 100),
            num_classes=num_classes,
            drop_rate=0,
        )
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.eval().to(device)

    else:
        run_conf = unflatten_dict(ckpt["config"], ".")
        cellvit = _build_cellvit(run_conf, arch)
        cellvit.load_state_dict(ckpt["model_state_dict"])
        cellvit.eval().to(device)
        num_classes = run_conf["data"]["num_nuclei_classes"]
        label_map = run_conf["data"].get("label_map", {})
        classifier = None

    return cellvit, classifier, num_classes, label_map


# ── Inference ────────────────────────────────────────────────────────────────

def run_inference(cellvit, classifier, dataloader, device, num_classes):
    """
    Returns:
        all_preds  (np.ndarray, int): predicted class index per cell
        all_gt     (np.ndarray, int): ground-truth class index per cell
    """
    postprocessor = DetectionCellPostProcessorCupy(wsi=None, nr_types=6)

    all_preds = []
    all_gt    = []

    with torch.no_grad():
        for images, cell_gt_batch, types_batch, image_names in tqdm.tqdm(
            dataloader, desc="Inference"
        ):
            images = images.to(device)

            # ── CellViT forward
            predictions = cellvit(images, retrieve_tokens=True)

            # ── Postprocess to get cell detections + per-cell tokens
            predictions = _apply_softmax_reorder(predictions)
            batch_cells = _extract_cells(
                cell_gt_batch, types_batch, predictions, images.shape[2:],
                postprocessor, device
            )

            if not batch_cells:
                continue

            tokens = torch.stack([c["token"] for c in batch_cells]).to(device)
            gt_types = torch.tensor([c["type"] for c in batch_cells], dtype=torch.long)

            if classifier is not None:
                logits = classifier(tokens)
            else:
                # nuclei_type_map from CellViT itself — not used in this flow
                raise RuntimeError(
                    "Full CellViT checkpoint without a LinearClassifier head: "
                    "nuclei_type_map validation is not supported by this script. "
                    "Please pass a LinearClassifier checkpoint."
                )

            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_gt.append(gt_types)

    if not all_preds:
        raise RuntimeError("No cells were detected during inference.")

    return torch.cat(all_preds).numpy(), torch.cat(all_gt).numpy()


def _apply_softmax_reorder(predictions: dict) -> dict:
    """Apply softmax and permute to (B, H, W, C) — mirrors trainer.apply_softmax_reorder."""
    predictions["nuclei_binary_map"] = F.softmax(predictions["nuclei_binary_map"], dim=1)
    predictions["nuclei_type_map"] = F.softmax(predictions["nuclei_type_map"], dim=1)
    predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(0, 2, 3, 1)
    predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(0, 2, 3, 1)
    predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
    return predictions


def _extract_cells(cell_gt_batch, types_batch, predictions, image_shape,
                   postprocessor, device):
    """
    Match CellViT detections to ground-truth annotations and return cells
    with tokens assigned.  Mirrors CellViTHeadTrainer.get_cellvit_result().
    """
    from cellvit.training.utils.tools import pair_coordinates

    h, w = image_shape
    batch_cells = []

    # Run postprocessor on the entire batch at once.
    # predictions maps are already permuted to (B, H, W, C) by _apply_softmax_reorder.
    _, cell_pred_dicts = postprocessor.post_process_batch(predictions)

    for i in range(len(cell_gt_batch)):
        gt_raw = cell_gt_batch[i]   # list of (x, y[, ...]) tuples
        gt_types = types_batch[i]   # list of ints

        if len(gt_raw) == 0:
            continue

        true_centroids = np.array(gt_raw, dtype=np.float32)[:, :2]  # (N, 2) x,y

        cell_pred_dict = cell_pred_dicts[i]
        if len(cell_pred_dict) == 0:
            continue

        # centroid from postprocessor is [m10/m00, m01/m00] = [x, y]
        pred_centroids = np.array(
            [v["centroid"] for v in cell_pred_dict.values()], dtype=np.float32
        )  # (M, 2) x,y

        # pair_coordinates(setA, setB) → pairing[:,0] = A indices, [:,1] = B indices
        pairing, _, _ = pair_coordinates(true_centroids, pred_centroids, 15)

        if len(pairing) == 0:
            continue

        tokens_map = predictions["tokens"][i]  # (D, H_t, W_t)
        D, Ht, Wt = tokens_map.shape
        patch_size = h // Ht  # e.g. 1024 // 64 = 16

        cell_list = list(cell_pred_dict.values())
        for true_idx, pred_idx in pairing:
            c  = cell_list[int(pred_idx)]
            cx = float(c["centroid"][0])   # x (column)
            cy = float(c["centroid"][1])   # y (row)
            tx = min(int(cx // patch_size), Wt - 1)
            ty = min(int(cy // patch_size), Ht - 1)
            token = tokens_map[:, ty, tx].detach().cpu()
            batch_cells.append({
                "token": token,
                "type":  int(gt_types[int(true_idx)]),
            })

    return batch_cells


# ── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, outdir: Path, normalize: bool = True):
    """Save confusion matrix as 600 DPI PNG and SVG."""
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
        fmt = ".2f"
        title = "Confusion Matrix (row-normalised)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    n = len(class_names)
    figsize = max(10, n * 1.1)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Font sizes: scale gently with n but keep them readable.
    tick_fs  = max(12, 20 - n // 3)
    label_fs = max(14, 22 - n // 3)
    title_fs = max(16, 24 - n // 3)
    cell_fs  = max(10, 18 - n // 3)

    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=(1.0 if normalize else None))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=tick_fs)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_title(title, fontsize=title_fs)
    ax.set_ylabel("True label", fontsize=label_fs)
    ax.set_xlabel("Predicted label", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_plot[i, j]:{fmt}}"
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=cell_fs)

    fig.tight_layout()

    suffix = "norm" if normalize else "counts"
    png_path = outdir / f"confusion_matrix_{suffix}.png"
    svg_path = outdir / f"confusion_matrix_{suffix}.svg"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {svg_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate a CellViT LinearClassifier and produce confusion matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="LinearClassifier or CellViT checkpoint (.pth).")
    parser.add_argument("--tissue", default=None, metavar="NAME",
                        help="Tissue name (e.g. breast, colorectal). If given, "
                             "--dataset / --filelist / --label-map are auto-derived "
                             "from tissue_configs/<tissue>.yaml (unless also set "
                             "explicitly, in which case the explicit value wins).")
    parser.add_argument("--fold", default="fold_0",
                        help="Split fold used when --tissue is given.")
    parser.add_argument("--dataset", default=None, metavar="PATH",
                        help="Dataset root directory (same as dataset_path in the train YAML).")
    parser.add_argument("--filelist", default=None, metavar="PATH",
                        help="CSV filelist for the validation split. "
                             "If omitted, all images in <dataset>/val/images/ are used.")
    parser.add_argument("--label-map", default=None, metavar="PATH",
                        help="YAML file mapping int → class name "
                             "(e.g. trainingset/<tissue>/label_map.yaml). "
                             "If omitted, class indices are used as labels.")
    parser.add_argument("--cellvit-path", default=None, metavar="PATH",
                        help="Override the cellvit_path baked into a "
                             "LinearClassifier checkpoint's config (useful when "
                             "the original backbone path has moved).")
    parser.add_argument("--outdir", default=None, metavar="PATH",
                        help="Output directory for confusion matrices. "
                             "Defaults to <checkpoint_dir>/validation/.")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. 'cuda' or 'cpu'. Defaults to auto.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers.")
    args = parser.parse_args()

    # ── Resolve paths from --tissue (if provided)
    if args.tissue:
        tissue_cfg_path = Path(current_dir) / "tissue_configs" / f"{args.tissue}.yaml"
        if not tissue_cfg_path.exists():
            parser.error(f"tissue config not found: {tissue_cfg_path}")
        with open(tissue_cfg_path) as f:
            tcfg = yaml.safe_load(f)
        out_dir = Path(tcfg["out_dir"])
        if args.dataset is None:
            args.dataset = str(out_dir)
        if args.label_map is None:
            args.label_map = str(out_dir / "label_map.yaml")
        if args.filelist is None:
            args.filelist = str(out_dir / "splits" / args.fold / "val.csv")
        print(f"Tissue     : {args.tissue}")
        print(f"  dataset  : {args.dataset}")
        print(f"  filelist : {args.filelist}")
        print(f"  label-map: {args.label_map}")

    if args.dataset is None:
        parser.error("--dataset is required when --tissue is not given")

    # ── Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Output dir
    cp = Path(args.checkpoint)
    outdir = Path(args.outdir) if args.outdir else cp.parent / "validation"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {outdir}")

    # ── Label map
    class_names = None
    if args.label_map and Path(args.label_map).exists():
        with open(args.label_map) as f:
            lm = yaml.safe_load(f)
        # YAML is either {0: "name", 1: "name"} or {"0": "name"}
        lm = {int(k): v for k, v in lm.items()}
        class_names = [lm[i] for i in sorted(lm)]
        print(f"Classes    : {class_names}")

    # ── Load models
    print(f"\nLoading checkpoint: {cp}")
    cellvit, classifier, num_classes, ckpt_label_map = load_models(
        str(cp), device, cellvit_path_override=args.cellvit_path,
    )

    # Fall back to label map embedded in checkpoint if no external file given
    if class_names is None and ckpt_label_map:
        lm = {int(k): v for k, v in ckpt_label_map.items()}
        class_names = [lm[i] for i in sorted(lm)]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # ── Dataset / DataLoader
    transforms = A.Compose(
        [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
    dataset = DetectionDataset(
        dataset_path=args.dataset,
        split="train",
        filelist_path=args.filelist,
        transforms=transforms,
        normalize_stains=False,
    )
    dataset.cache_dataset()
    print(f"\nVal tiles  : {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=DetectionDataset.collate_batch,
    )

    # ── Inference
    print("\nRunning inference ...")
    preds, gt = run_inference(cellvit, classifier, dataloader, device, num_classes)
    print(f"  Total matched cells: {len(preds):,}")

    # ── Metrics
    acc = accuracy_score(gt, preds)
    f1  = f1_score(gt, preds, average="macro", zero_division=0)
    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 (macro): {f1:.4f}")

    report = classification_report(
        gt, preds,
        labels=list(range(num_classes)),
        target_names=class_names[:num_classes],
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)

    report_path = outdir / "classification_report.txt"
    report_path.write_text(
        f"Accuracy : {acc:.4f}\nF1 (macro): {f1:.4f}\n\n{report}"
    )
    print(f"  Saved: {report_path}")

    # ── Confusion matrices
    cm = confusion_matrix(gt, preds, labels=list(range(num_classes)))
    print("\nGenerating confusion matrices ...")
    plot_confusion_matrix(cm, class_names[:num_classes], outdir, normalize=True)
    plot_confusion_matrix(cm, class_names[:num_classes], outdir, normalize=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
