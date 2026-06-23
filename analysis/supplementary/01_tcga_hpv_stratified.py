#!/usr/bin/env python
"""
Supplementary Analysis 1 -- TCGA cold/hot biomarker pattern, stratified by HPV.

Question
--------
The Xenium data say the versican/tenascin -> EGFR "stromal-wall" mechanism that
keeps immune cells out of the tumour is an HPV-NEGATIVE story.  If that is right,
the matching TCGA pattern -- "immune-hot inside, immune-cold outside", driven by
the cold (genomic-instability / TGF-beta / wound-healing) markers -- should be
cleaner in the HPV-negative patients than in the HPV-positive ones.

What this script does
---------------------
Re-runs the exact Part B cluster-mass + permutation-FDR screen (same thresholds:
CLUSTER_ALPHA_TCGA=0.05, FDR_Q_TCGA=0.05, MIN_CLUSTER_W=2, B=2000) for every
biomarker, but SEPARATELY inside the HPV- and HPV+ subgroups, and compares the
band strength side by side.

Outputs (data/results/head_neck/supplementary/)
  - tcga_hpv_stratified_results.csv
  - tcga_hpv_stratified_coldmarkers.png / .svg   (top cold markers, HPV- vs HPV+)
Prints a PASS/FLAG summary line.

Run:
  PYTHONPATH=/workspace/wsinsight/sptxinsight \
    /opt/anaconda3/envs/spatial/bin/python \
    analysis/supplementary/01_tcga_hpv_stratified.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import _shared as sh  # noqa: E402

LOW_POWER_N = 30   # flag any HPV arm with fewer than this many patients


def main():
    print("== Supplementary Analysis 1: TCGA biomarker pattern by HPV status ==")
    mat, long = sh.load_tcga_curves()
    labels_df = sh.load_tcga_labels(mat)
    print(f"loaded {mat.shape[0]} patients x {mat.shape[1]} layers; "
          f"{labels_df.shape[1]} label columns")

    if "hpv_status" not in labels_df.columns:
        raise sh.MissingInput(
            "no 'hpv_status' column was recovered from the TCGA label files; "
            "cannot stratify by HPV (check data_clinical_sample.txt HPV_STATUS).")

    hpv = labels_df["hpv_status"]
    pats_neg = mat.index[hpv.eq("HPV-").to_numpy()]
    pats_pos = mat.index[hpv.eq("HPV+").to_numpy()]
    n_neg, n_pos = len(pats_neg), len(pats_pos)
    print(f"HPV- patients: {n_neg} | HPV+ patients: {n_pos}")
    if n_neg < 2 * sh.MIN_PER_GROUP or n_pos < 2 * sh.MIN_PER_GROUP:
        raise sh.MissingInput(
            f"too few patients to median-split within an HPV arm "
            f"(HPV-={n_neg}, HPV+={n_pos}); need >= {2 * sh.MIN_PER_GROUP} each.")

    low_power = (n_neg // 2) < LOW_POWER_N or (n_pos // 2) < LOW_POWER_N
    if low_power:
        print(f"  [low-power note] within an HPV arm the marker is median-split "
              f"into low/high halves (~n/2 each): HPV- ~{n_neg // 2}/half, "
              f"HPV+ ~{n_pos // 2}/half. Halves < {LOW_POWER_N} are "
              f"under-powered, so a NULL in the smaller (HPV+) arm is weak "
              f"evidence of absence, not proof the pattern is HPV-specific.")

    mat_neg = mat.loc[pats_neg]
    mat_pos = mat.loc[pats_pos]

    def screen_arm(mat_arm, series_arm):
        """Median-split a biomarker WITHIN the arm, then run the Part B screen."""
        g = sh.binarize(series_arm)
        n0, n1 = int((g == 0).sum()), int((g == 1).sum())
        if n0 < sh.MIN_PER_GROUP or n1 < sh.MIN_PER_GROUP:
            return None
        R = sh.screen_label(mat_arm, g, 2)
        bs, be, pk = R["band"]
        if pk >= 0:
            top = "high" if np.nanargmax(R["grp_means"][pk]) == 1 else "low"
            band = (int(sh.GRID[bs]), int(sh.GRID[be]), int(sh.GRID[pk]))
        else:
            top, band = "", (np.nan, np.nan, np.nan)
        return dict(mass=R["mass"], p=R["perm_p"], band_lo=band[0],
                    band_hi=band[1], peak=band[2], top=top, n0=n0, n1=n1, _R=R)

    rows = []
    detail = {}
    for col, disp in sh.CONT_LABELS:
        if col not in labels_df.columns:
            continue
        rn = screen_arm(mat_neg, labels_df.loc[pats_neg, col])
        rp = screen_arm(mat_pos, labels_df.loc[pats_pos, col])
        if rn is None and rp is None:
            continue
        detail[col] = (disp, rn, rp)
        mass_neg = rn["mass"] if rn else np.nan
        mass_pos = rp["mass"] if rp else np.nan
        p_neg = rn["p"] if rn else np.nan
        p_pos = rp["p"] if rp else np.nan
        sig_neg = bool(rn and rn["p"] < sh.FDR_Q_TCGA)
        sig_pos = bool(rp and rp["p"] < sh.FDR_Q_TCGA)
        if sig_neg and not sig_pos:
            stronger = "HPV-"
        elif sig_pos and not sig_neg:
            stronger = "HPV+"
        elif sig_neg and sig_pos:
            stronger = "HPV-" if (mass_neg or 0) >= (mass_pos or 0) else "HPV+"
        else:
            stronger = "neither"
        rows.append(dict(
            marker=disp, key=col, is_cold_marker=col in sh.COLD_MARKERS,
            band_mass_HPVneg=mass_neg, p_HPVneg=p_neg,
            top_group_HPVneg=(rn["top"] if rn else ""),
            band_HPVneg=(f"L{rn['band_lo']}..{rn['band_hi']}" if rn and rn["peak"] is not np.nan else ""),
            band_mass_HPVpos=mass_pos, p_HPVpos=p_pos,
            top_group_HPVpos=(rp["top"] if rp else ""),
            band_HPVpos=(f"L{rp['band_lo']}..{rp['band_hi']}" if rp and rp["peak"] is not np.nan else ""),
            stronger_in=stronger,
            n_HPVneg=(f"{rn['n0']}/{rn['n1']}" if rn else "NA"),
            n_HPVpos=(f"{rp['n0']}/{rp['n1']}" if rp else "NA"),
            low_power=low_power))

    res = pd.DataFrame(rows)
    # rank: cold markers first, then by HPV- band strength
    res = res.sort_values(["is_cold_marker", "band_mass_HPVneg"],
                          ascending=[False, False]).reset_index(drop=True)
    out_csv = os.path.join(sh.SUPP_DIR, "tcga_hpv_stratified_results.csv")
    res.to_csv(out_csv, index=False)
    print("wrote", out_csv)
    pd.set_option("display.width", 200)
    print(res.drop(columns=["low_power"]).to_string(index=False))

    # --- figure: side-by-side H-plots for the top 3 cold markers ----------- #
    cold_present = [c for c in sh.COLD_MARKERS if c in detail]
    if not cold_present:
        print("[FLAG] none of the cold markers "
              f"{sh.COLD_MARKERS} were available; skipping cold-marker figure.")
    else:
        top_cold = (res[res.is_cold_marker]
                    .sort_values("band_mass_HPVneg", ascending=False)
                    .head(3))
        cm = {"low": "#1f77b4", "high": "#d62728"}
        fig, axes = plt.subplots(len(top_cold), 2,
                                 figsize=(11, 3.4 * len(top_cold)),
                                 squeeze=False)
        for i, (_, r) in enumerate(top_cold.iterrows()):
            col = r["key"]; disp = r["marker"]
            for j, (arm, pats, R) in enumerate(
                    [("HPV-", pats_neg, detail[col][1]),
                     ("HPV+", pats_pos, detail[col][2])]):
                ax = axes[i][j]
                if R is None:
                    ax.text(0.5, 0.5, f"{disp}\n{arm}: not screenable",
                            ha="center", va="center"); ax.axis("off"); continue
                g = sh.binarize(labels_df.loc[pats, col])
                pmap = {p: (sh.HL_ORDER[int(gi)] if gi >= 0 else None)
                        for p, gi in zip(pats, g)}
                lsub = long[long["patient"].isin(pats)]
                band = (R["band_lo"], R["band_hi"])
                sh.plot_curve(lsub, pmap, sh.HL_ORDER, f"{disp}", ax, cm, band)
                ax.set_title(f"{disp} -- {arm}  (mass={R['mass']:.0f}, "
                             f"p={R['p']:.3g})", fontsize=9)
        fig.suptitle("Cold-marker immune geometry: HPV-negative vs HPV-positive "
                     "TCGA-HNSC", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        png = os.path.join(sh.SUPP_DIR, "tcga_hpv_stratified_coldmarkers.png")
        fig.savefig(png, dpi=180, bbox_inches="tight")
        fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight")
        print("wrote", png)

    # --- summary line ------------------------------------------------------ #
    cold = res[res.is_cold_marker]
    neg_only = cold[(cold.p_HPVneg < sh.FDR_Q_TCGA)
                    & ~(cold.p_HPVpos < sh.FDR_Q_TCGA)]
    tag = "FLAG-LOWPOWER" if low_power else "PASS"
    print(f"\n[{tag}] cold markers significant in HPV- but NOT HPV+: "
          f"{neg_only.marker.tolist() or 'none'} "
          f"(HPV- n={n_neg}, HPV+ n={n_pos}; median-split halves ~{n_neg // 2} "
          f"vs ~{n_pos // 2}). The HPV+ arm is under-powered, so read this as "
          f"'consistent with an HPV-negative-specific cold pattern', not proof.")


if __name__ == "__main__":
    main()
