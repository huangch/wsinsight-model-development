#!/usr/bin/env python
"""
Supplementary Analysis 3 -- Cox model: HPV x border-immune-gradient interaction.

Question
--------
In the main analysis, higher immune infiltration just inside the tumour border is
a borderline protective factor for overall survival (HR ~ 0.87 per +1 SD,
p ~ 0.058).  The Xenium mechanism (the stromal wall that excludes immune cells)
is HPV-negative-specific.  So the survival benefit of being "infiltrated" should
be concentrated in HPV-negative patients.  This script tests that with an
interaction term.

What this script does
---------------------
Rebuilds the exact Part C border-region infiltration covariate
(interior_infiltration = mean immune fraction over L in [-5, 0]) and the same
covariates (HPV, age, late stage), then:
  1. baseline   Cox:  OS ~ infiltration + HPV + age + stage
  2. interaction Cox:  ... + infiltration:HPV
  3. likelihood-ratio test (1 df) baseline vs interaction
  4. stratified HR/SD for infiltration within HPV- and within HPV+
  5. incremental C-index across nested models

Outputs (data/results/head_neck/supplementary/)
  - cox_hpv_interaction_results.csv
  - cox_hpv_interaction_forest.png / .svg
Prints the interaction LR p-value and whether C-index improves.

Run:
  PYTHONPATH=/workspace/wsinsight/sptxinsight \
    /opt/anaconda3/envs/spatial/bin/python \
    analysis/supplementary/03_cox_hpv_interaction.py
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

sys.path.insert(0, os.path.dirname(__file__))
import _shared as sh  # noqa: E402

TC = sh.TCGA_DIR
INTERACTION_ALPHA = 0.10   # exploratory threshold for the interaction term


def _need(path):
    if not os.path.exists(path):
        raise sh.MissingInput(f"required Part C input not found: {path}")
    return path


def _zcol(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


def build_frame():
    """Reproduce the Part C survival frame: infiltration + HPV + age + stage."""
    return sh.build_survival_frame()


def fit(df, cols):
    cph = CoxPHFitter()
    cph.fit(df[["OS_t", "OS_e"] + cols], "OS_t", "OS_e")
    return cph


def main():
    print("== Supplementary Analysis 3: HPV x infiltration interaction (Cox) ==")
    mv = build_frame()
    n = len(mv)
    print(f"survival frame: n={n} | events={int(mv.OS_e.sum())} | "
          f"HPV+ n={int(mv.HPV_pos.sum())}, HPV- n={int((mv.HPV_pos == 0).sum())}")

    base_cols = ["infiltration_z", "HPV_pos", "AGE", "late_stage"]
    clin_cols = ["HPV_pos", "AGE", "late_stage"]

    m_clin = fit(mv, clin_cols)
    m_base = fit(mv, base_cols)

    mv_int = mv.copy()
    mv_int["infil_x_HPV"] = mv_int.infiltration_z * mv_int.HPV_pos
    int_cols = base_cols + ["infil_x_HPV"]
    m_int = fit(mv_int, int_cols)

    # likelihood-ratio test: baseline vs interaction (1 df)
    lr_stat = 2.0 * (m_int.log_likelihood_ - m_base.log_likelihood_)
    lr_p = float(chi2.sf(lr_stat, df=1))

    def hr_p(model, term):
        return (float(np.exp(model.params_[term])),
                float(model.summary.loc[term, "p"]))

    base_hr, base_p = hr_p(m_base, "infiltration_z")
    int_hr, int_p = hr_p(m_int, "infil_x_HPV")
    infil_main_hr, infil_main_p = hr_p(m_int, "infiltration_z")

    # stratified HR/SD within each HPV arm (re-standardise within arm)
    strat = {}
    for arm, mask in (("HPV-", mv.HPV_pos == 0), ("HPV+", mv.HPV_pos == 1)):
        sub = mv[mask].copy()
        sub["infiltration_z"] = _zcol(sub.interior_infiltration) \
            if "interior_infiltration" in sub else _zcol(sub.infiltration_z)
        sub = sub.dropna(subset=["infiltration_z"])
        if len(sub) < 2 * sh.MIN_PER_GROUP or sub.OS_e.sum() < 5:
            strat[arm] = dict(n=len(sub), HR=np.nan, lo=np.nan, hi=np.nan,
                              p=np.nan, note="low-N")
            continue
        cph = CoxPHFitter()
        cph.fit(sub[["OS_t", "OS_e", "infiltration_z", "AGE", "late_stage"]],
                "OS_t", "OS_e")
        s = cph.summary.loc["infiltration_z"]
        strat[arm] = dict(n=len(sub), HR=float(np.exp(s["coef"])),
                          lo=float(s["exp(coef) lower 95%"]),
                          hi=float(s["exp(coef) upper 95%"]),
                          p=float(s["p"]), note="")

    # incremental C-index
    c_clin = m_clin.concordance_index_
    c_base = m_base.concordance_index_
    c_int = m_int.concordance_index_

    rows = [
        dict(model="clinical (HPV+age+stage)", n=n, infiltration_HR=np.nan,
             infiltration_p=np.nan, interaction_HR=np.nan, interaction_p=np.nan,
             C_index=round(c_clin, 4)),
        dict(model="+ infiltration", n=n, infiltration_HR=round(base_hr, 4),
             infiltration_p=round(base_p, 4), interaction_HR=np.nan,
             interaction_p=np.nan, C_index=round(c_base, 4)),
        dict(model="+ infiltration:HPV", n=n,
             infiltration_HR=round(infil_main_hr, 4),
             infiltration_p=round(infil_main_p, 4),
             interaction_HR=round(int_hr, 4), interaction_p=round(int_p, 4),
             C_index=round(c_int, 4)),
    ]
    for arm in ("HPV-", "HPV+"):
        s = strat[arm]
        rows.append(dict(model=f"stratified: infiltration within {arm}",
                         n=s["n"], infiltration_HR=(round(s["HR"], 4)
                                                    if np.isfinite(s["HR"]) else np.nan),
                         infiltration_p=(round(s["p"], 4)
                                         if np.isfinite(s["p"]) else np.nan),
                         interaction_HR=np.nan, interaction_p=np.nan,
                         C_index=np.nan))
    out = pd.DataFrame(rows)
    out_csv = os.path.join(sh.SUPP_DIR, "cox_hpv_interaction_results.csv")
    out.to_csv(out_csv, index=False)
    print("wrote", out_csv)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nLR test (baseline vs +interaction, 1 df): "
          f"chi2={lr_stat:.3f}, p={lr_p:.4f}")

    # ---- forest plot: HR/SD infiltration within HPV- vs HPV+ ------------- #
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ys = {"HPV-": 1.0, "HPV+": 0.0}
    cols = {"HPV-": "#c44e52", "HPV+": "#4c72b0"}
    for arm, y in ys.items():
        s = strat[arm]
        if not np.isfinite(s["HR"]):
            ax.text(1.0, y, f"{arm}: n={s['n']} (low-N)", va="center", fontsize=9)
            continue
        ax.plot([s["lo"], s["hi"]], [y, y], color=cols[arm], lw=2)
        ax.plot(s["HR"], y, "o", color=cols[arm], ms=8)
        ax.text(s["hi"] * 1.02, y,
                f"{arm}: HR/SD={s['HR']:.2f} "
                f"[{s['lo']:.2f}-{s['hi']:.2f}], p={s['p']:.3f}, n={s['n']}",
                va="center", fontsize=8)
    ax.axvline(1.0, color="0.5", ls="--", lw=0.9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["HPV+", "HPV-"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("HR per +1 SD interior infiltration (HR<1 = better survival)")
    ax.set_title(f"Border-immune infiltration vs OS by HPV status "
                 f"(interaction p={lr_p:.3f})")
    fig.tight_layout()
    png = os.path.join(sh.SUPP_DIR, "cox_hpv_interaction_forest.png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".svg"), bbox_inches="tight")
    print("wrote", png)

    # ---- summary line ---------------------------------------------------- #
    dC = c_int - c_clin
    sig = lr_p < INTERACTION_ALPHA
    tag = "PASS-INTERACTION" if sig else "NULL-INTERACTION"
    print(f"\n[{tag}] interaction LR p={lr_p:.4f} "
          f"({'<' if sig else '>='} {INTERACTION_ALPHA:g}); "
          f"C-index clinical={c_clin:.3f} -> +infiltration={c_base:.3f} -> "
          f"+interaction={c_int:.3f} (delta={dC:+.3f}, "
          f"{'meaningful' if dC > 0.01 else 'not meaningful'} >0.01).")


if __name__ == "__main__":
    main()
