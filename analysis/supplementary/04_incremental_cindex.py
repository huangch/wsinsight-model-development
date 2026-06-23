#!/usr/bin/env python
"""
Supplementary Analysis 4 -- Incremental predictive value of border infiltration.

Question
--------
The headline survival claim is that immune infiltration just inside the tumour
border (interior_infiltration, mean immune fraction over layers L in [-5,0]) adds
prognostic information on top of the standard clinical variables (HPV, age,
stage).  A borderline hazard ratio (HR ~ 0.87, p ~ 0.06) is not, by itself,
evidence that the variable *improves prediction*.  This script answers the
sharper question: does adding the border-infiltration covariate measurably and
reproducibly raise the model's discrimination (C-index)?

What this script does
---------------------
Fits three nested Cox models on the exact Part C survival frame:
  Model 0 (clinical) :  OS ~ HPV + age + stage
  Model 1 (+infil)   :  ... + infiltration_z
  Model 2 (+interact):  ... + infiltration_z:HPV
For each model it reports the apparent (train) C-index, an honest 5-fold
cross-validated C-index (seed-controlled, implemented here so the fold split is
reproducible), the partial AIC and log-likelihood, and a likelihood-ratio test
against the previous (nested) model.  It then puts a 95 % bootstrap confidence
interval on the Model 0 -> Model 1 gain in apparent C-index by resampling
patients and refitting both models.

Verdict logic (house style -- do not overclaim a borderline result):
  GENUINE gain  only if the bootstrap CI for Delta-C excludes 0 AND the
                Model0->Model1 likelihood-ratio p < 0.05;
  otherwise     reported as "directionally positive but not statistically
                robust".

Outputs (data/results/head_neck/supplementary/)
  - incremental_cindex_results.csv
  - incremental_cindex.png / .svg
Prints the per-model C-index table, the Delta-C bootstrap CI and the verdict.

Run:
  PYTHONPATH=/workspace/wsinsight/sptxinsight \
    /opt/anaconda3/envs/spatial/bin/python \
    analysis/supplementary/04_incremental_cindex.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from scipy.stats import chi2
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.dirname(__file__))
import _shared as sh  # noqa: E402

N_FOLDS = 5
N_BOOT = 500
LR_ALPHA = 0.05           # nested-model LR significance threshold

MODELS = [
    ("Model 0: clinical (HPV+age+stage)", ["HPV_pos", "AGE", "late_stage"]),
    ("Model 1: + infiltration", ["HPV_pos", "AGE", "late_stage", "infiltration_z"]),
    ("Model 2: + infiltration:HPV",
     ["HPV_pos", "AGE", "late_stage", "infiltration_z", "infil_x_HPV"]),
]


def _fit(df, cols):
    cph = CoxPHFitter()
    cph.fit(df[["OS_t", "OS_e"] + cols], "OS_t", "OS_e")
    return cph


def _cv_cindex(df, cols, k=N_FOLDS, seed=0):
    """Seed-controlled k-fold cross-validated C-index (held-out folds)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    folds = np.array_split(idx, k)
    scores = []
    for f in folds:
        te = df.iloc[f]
        tr = df.iloc[np.setdiff1d(idx, f, assume_unique=False)]
        if te["OS_e"].sum() < 1 or tr["OS_e"].sum() < 1:
            continue
        cph = CoxPHFitter()
        cph.fit(tr[["OS_t", "OS_e"] + cols], "OS_t", "OS_e")
        risk = cph.predict_partial_hazard(te).to_numpy().ravel()
        # higher hazard -> shorter survival; concordance_index wants a score
        # that is larger for longer survival, so negate the hazard.
        scores.append(concordance_index(te["OS_t"], -risk, te["OS_e"]))
    return float(np.mean(scores)) if scores else np.nan


def main():
    print("== Supplementary Analysis 4: incremental C-index of border infiltration ==")
    mv = sh.build_survival_frame().reset_index(drop=True)
    mv["infil_x_HPV"] = mv.infiltration_z * mv.HPV_pos
    n = len(mv)
    print(f"survival frame: n={n} | events={int(mv.OS_e.sum())} | "
          f"HPV+ n={int(mv.HPV_pos.sum())}, HPV- n={int((mv.HPV_pos == 0).sum())}")

    fits, rows = [], []
    prev = None
    for name, cols in MODELS:
        cph = _fit(mv, cols)
        fits.append((name, cols, cph))
        c_train = float(cph.concordance_index_)
        c_cv = _cv_cindex(mv, cols, seed=0)
        ll = float(cph.log_likelihood_)
        aic = float(cph.AIC_partial_)
        if prev is None:
            lr_stat = lr_p = np.nan
        else:
            d_df = len(cols) - len(prev[1])
            lr_stat = 2.0 * (ll - prev[2])
            lr_p = float(chi2.sf(lr_stat, df=max(d_df, 1)))
        rows.append(dict(
            model=name, n=n, n_params=len(cols),
            concordance_index_train=round(c_train, 4),
            concordance_index_cv5=round(c_cv, 4),
            AIC=round(aic, 3), loglik=round(ll, 3),
            LR_vs_previous=(round(lr_stat, 3) if np.isfinite(lr_stat) else np.nan),
            LR_p_vs_previous=(round(lr_p, 4) if np.isfinite(lr_p) else np.nan)))
        prev = (name, cols, ll)

    c_train = {r["model"]: r["concordance_index_train"] for r in rows}
    c_cv = {r["model"]: r["concordance_index_cv5"] for r in rows}
    m0, m1, m2 = (MODELS[0][0], MODELS[1][0], MODELS[2][0])
    cols0, cols1 = MODELS[0][1], MODELS[1][1]

    dC_train = c_train[m1] - c_train[m0]
    dC_cv = c_cv[m1] - c_cv[m0]
    lr_p_01 = next(r["LR_p_vs_previous"] for r in rows if r["model"] == m1)

    # ---- bootstrap CI for the Model0 -> Model1 apparent C-index gain --------
    rng = np.random.default_rng(0)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        samp = mv.iloc[rng.integers(0, n, n)].reset_index(drop=True)
        if samp["OS_e"].sum() < 2:
            boot[b] = np.nan
            continue
        try:
            c0 = float(_fit(samp, cols0).concordance_index_)
            c1 = float(_fit(samp, cols1).concordance_index_)
            boot[b] = c1 - c0
        except Exception:
            boot[b] = np.nan
    boot = boot[np.isfinite(boot)]
    b_lo, b_hi = np.percentile(boot, [2.5, 97.5])
    b_mean = float(boot.mean())
    frac_gt0 = float((boot > 0).mean())

    ci_excludes_0 = (b_lo > 0) or (b_hi < 0)
    lr_sig = np.isfinite(lr_p_01) and lr_p_01 < LR_ALPHA
    genuine = ci_excludes_0 and lr_sig

    # append bootstrap summary onto the +infiltration row in the CSV
    out = pd.DataFrame(rows)
    out["deltaC_train_vs_clinical"] = [np.nan, round(dC_train, 4),
                                       round(c_train[m2] - c_train[m0], 4)]
    out["deltaC_cv5_vs_clinical"] = [np.nan, round(dC_cv, 4),
                                     round(c_cv[m2] - c_cv[m0], 4)]
    out["boot_deltaC_mean"] = [np.nan, round(b_mean, 4), np.nan]
    out["boot_deltaC_lo95"] = [np.nan, round(float(b_lo), 4), np.nan]
    out["boot_deltaC_hi95"] = [np.nan, round(float(b_hi), 4), np.nan]
    out["boot_frac_gt0"] = [np.nan, round(frac_gt0, 4), np.nan]

    out_csv = os.path.join(sh.SUPP_DIR, "incremental_cindex_results.csv")
    out.to_csv(out_csv, index=False)
    print("wrote", out_csv)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))

    # ---- figure ------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ["clinical", "+infil", "+infil:HPV"]
    xs = np.arange(3)
    tr = [c_train[m] for m in (m0, m1, m2)]
    cv = [c_cv[m] for m in (m0, m1, m2)]
    axL.bar(xs - 0.18, tr, width=0.36, label="train", color="#9ecae1")
    axL.bar(xs + 0.18, cv, width=0.36, label="5-fold CV", color="#3182bd")
    for x, (a, b) in enumerate(zip(tr, cv)):
        axL.text(x - 0.18, a + 0.002, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        axL.text(x + 0.18, b + 0.002, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    axL.axhline(0.5, color="0.6", lw=0.8, ls="--")
    axL.set_xticks(xs); axL.set_xticklabels(labels)
    axL.set_ylim(0.5, max(max(tr), max(cv)) + 0.03)
    axL.set_ylabel("Harrell C-index"); axL.set_title("Discrimination by model")
    axL.legend(frameon=False, fontsize=9)

    axR.hist(boot, bins=40, color="#9ecae1", edgecolor="white")
    axR.axvline(0, color="#d62728", lw=1.4, ls="--", label="no gain")
    axR.axvline(b_lo, color="0.3", lw=1.0); axR.axvline(b_hi, color="0.3", lw=1.0)
    axR.axvline(b_mean, color="#08519c", lw=1.4, label="mean")
    axR.set_xlabel(r"$\Delta$C-index (Model1 $-$ Model0), bootstrap")
    axR.set_ylabel("bootstrap draws")
    axR.set_title(f"+infiltration gain\n95% CI [{b_lo:.3f}, {b_hi:.3f}], "
                  f"P(>0)={frac_gt0:.2f}")
    axR.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    png = os.path.join(sh.SUPP_DIR, "incremental_cindex.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", png, "and .svg")

    # ---- verdict -----------------------------------------------------------
    print("\n--- SUMMARY ---")
    print(f"train C-index : clinical={c_train[m0]:.3f} -> +infil={c_train[m1]:.3f} "
          f"(Delta={dC_train:+.3f})")
    print(f"5-fold CV     : clinical={c_cv[m0]:.3f} -> +infil={c_cv[m1]:.3f} "
          f"(Delta={dC_cv:+.3f})")
    print(f"Model0->Model1 LR test p = {lr_p_01:.4f}")
    print(f"bootstrap Delta-C (apparent): mean={b_mean:+.4f}, "
          f"95% CI [{b_lo:+.4f}, {b_hi:+.4f}], P(Delta>0)={frac_gt0:.2f}")
    if genuine:
        print("[PASS] Adding border infiltration gives a statistically robust "
              "improvement in discrimination (bootstrap CI excludes 0 AND nested "
              "LR p<0.05).")
    else:
        why = []
        if not ci_excludes_0:
            why.append("bootstrap CI for Delta-C includes 0")
        if not lr_sig:
            why.append(f"nested LR p={lr_p_01:.3f} >= {LR_ALPHA}")
        print("[FLAG-WEAK] Border infiltration is directionally positive but NOT a "
              "statistically robust gain in prediction (" + "; ".join(why) + "). "
              "The covariate is reported as a borderline, hypothesis-consistent "
              "signal, not as an independently validated prognostic predictor.")


if __name__ == "__main__":
    main()
