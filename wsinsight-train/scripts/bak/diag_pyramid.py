"""Pyramid downsample factors vs what ST2WSI rounded them to.

ST2WSI's getScalingFactor() returns an int, so a non-integer pyramid step is
silently rounded and the coordinate chain inherits the error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import tifffile

R = Path("/workspace/wsinsight/wsinsight-model-development")
sys.path.insert(0, str(R / "wsinsight-train"))
from wsitrain.dataset import discover_samples  # noqa: E402

samples = discover_samples(R / "data/xenium", "pantissue")
pats = sys.argv[1:] or [""]


def levels(path):
    with tifffile.TiffFile(str(path)) as tf:
        out = []
        for s in tf.series:
            sh = s.shape
            if s.axes.startswith(("C", "S")):
                h, w = sh[1], sh[2]
            elif len(sh) >= 2:
                h, w = sh[0], sh[1]
            else:
                continue
            out.append((w, h))
    return out


for pat in pats:
    for s in samples:
        if pat.lower() not in s.sample_id.lower():
            continue
        p = json.loads((Path(s.outs) / "registration_params.json").read_text())
        src_scale = int(p["xnumAnnotImgRegParamSourceScale"])
        tgt_scale = int(p["xnumAnnotImgRegParamTargetScale"])
        src_w = int(p["xnumAnnotImgRegParamSrcImgWidth"])
        src_h = int(p["xnumAnnotImgRegParamSrcImgHeight"])

        dapi = next((q for q in (Path(s.outs) / "morphology_focus").glob("*.ome.tif")), None)
        print(f"\n{s.sample_id}")
        print(f"  params: SourceScale={src_scale} TargetScale={tgt_scale} "
              f"SrcImg={src_w}x{src_h}")

        for tag, path, scale in (("DAPI  ", dapi, src_scale), ("H&E   ", s.he, tgt_scale)):
            if path is None:
                print(f"  {tag} <missing>")
                continue
            lv = levels(path)
            w0, h0 = lv[0]
            desc = "  ".join(f"s{i}:{w}x{h}({(w0/w + h0/h)/2:.2f}x)"
                             for i, (w, h) in enumerate(lv))
            print(f"  {tag} {desc}")
            # which series did ST2WSI use? the one whose rounded factor == scale
            hits = [i for i, (w, h) in enumerate(lv)
                    if int(0.5 + (w0 / w + h0 / h) / 2) == scale]
            for i in hits:
                w, h = lv[i]
                true = (w0 / w + h0 / h) / 2
                err = abs(true - scale) / scale
                mark = "  <-- ROUNDED" if err > 0.01 else ""
                print(f"         used s{i}: true={true:.4f} stored={scale} "
                      f"err={err*100:.2f}%{mark}")
        # SrcImgWidth should equal the DAPI full-res width
        if dapi is not None:
            dw, dh = levels(dapi)[0]
            if abs(src_w - dw) > 2 or abs(src_h - dh) > 2:
                print(f"  !! SrcImg {src_w}x{src_h} != DAPI full-res {dw}x{dh}")
