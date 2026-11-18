"""
CLI study runner for Sub-Goal 1: Refined 1D Spheroid ODE/PDE Radiotherapy Model.

Generates:
  1. outputs/ef5_rt_model/spheroid/fig1_alex_byrne_validation.png
     (Validates our spherical O2 PDE solver against Helen Byrne's exact analytical Greenspan Eqs. 44-50
      and reproduces Alex Fletcher's untreated growth dynamics.)
  2. outputs/ef5_rt_model/spheroid/fig2_plos_one_regimens.png
     (2x3 grid comparing Control vs EF5+ vs EF5- across 1x10, 1x20, 1x40, 2x10, 4x5, and 4x10 Gy.)
  3. outputs/ef5_rt_model/spheroid/fig3_reoxygenation_profiles.png
     (Radial pO2, viable/damaged/necrotic fractions, and virtual EF5 T/M pre-RT, post-RT, Day +2, Day +13, Day +32.)
  4. outputs/ef5_rt_model/spheroid/fig4_legacy_vs_refined_and_first_study.png
     (Compares legacy 2013 MATLAB port vs refined PDE model and overlays First EF5 RT Study 20 Gy / Control data.)
  5. results/ef5_rt_model/spheroid_regimen_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modeling.legacy_spheroid import (
    analytical_byrne_greenspan_oxygen,
    run_alex_fletcher_growth,
    run_legacy_spheroid_2013,
    solve_spherical_oxygen_bvp_step,
)
from src.modeling.physics import RadiobiologyParams
from src.modeling.spheroid_rt_model import SpheroidRTModel, make_plos_one_scenarios


def run_spheroid_study(root: Path = Path(".")) -> pd.DataFrame:
    out_dir = root / "outputs" / "ef5_rt_model" / "spheroid"
    res_dir = root / "results" / "ef5_rt_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Verify 1D spherical O2 solver vs Byrne analytical + Alex growth
    # ------------------------------------------------------------------
    print("1. Validating spherical oxygen PDE vs Byrne analytical Greenspan & Alex Fletcher's CodeForBobby...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=130)
    radii_check = [250.0, 500.0, 850.0]
    max_rel_errs = []
    for R_um, col in zip(radii_check, ["#1f77b4", "#ff7f0e", "#d62728"]):
        r_num, c_num = solve_spherical_oxygen_bvp_step(R_um)
        ana = analytical_byrne_greenspan_oxygen(R_um)
        rel_err = float(np.max(np.abs(c_num - ana["c_mM"])) / 0.21)
        max_rel_errs.append(rel_err)
        axes[0].plot(ana["r_um"], ana["c_mM"], "-", color=col, lw=2.2, label=f"Byrne R={R_um:.0f} um")
        axes[0].plot(r_num[::6], c_num[::6], "o", color=col, ms=4, alpha=0.8)
    axes[0].axhline(0.18, color="#666666", ls="--", lw=1.0, label="c_H = 0.18 mM")
    axes[0].axhline(0.16, color="#222222", ls=":", lw=1.2, label="c_N = 0.16 mM")
    axes[0].set_xlabel("Radius r (um)")
    axes[0].set_ylabel("Oxygen concentration c (mM)")
    axes[0].set_title(f"Steady O2 PDE vs Analytical\n(max rel err = {max(max_rel_errs):.2e})")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(alpha=0.25)

    alex = run_alex_fletcher_growth()
    axes[1].plot(alex["time_days"], alex["R_um"], "k-", lw=2.2, label="Outer radius R")
    axes[1].plot(alex["time_days"], alex["R_H_um"], "r--", lw=1.8, label="Hypoxic radius R_H")
    axes[1].plot(alex["time_days"], alex["R_N_um"], "b:", lw=2.0, label="Necrotic radius R_N")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel("Radius (um)")
    axes[1].set_title("Alex Fletcher Untreated Growth\n(CodeForBobby.m, bug-fixed)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(alex["time_days"], alex["V_normalised"], color="#2ca02c", lw=2.2)
    axes[2].set_xlabel("Time (days)")
    axes[2].set_ylabel("Normalized Volume V / V0")
    axes[2].set_title("Self-Limiting Spheroid Growth\n(No ad-hoc Vt brake)")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_alex_byrne_validation.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Sub-Goal 1 PLOS ONE Regimen Sweep (Refined Spheroid RT Model)
    # ------------------------------------------------------------------
    print("2. Running refined coupled spheroid RT model across PLOS ONE regimens...")
    params = RadiobiologyParams()
    model = SpheroidRTModel(params)

    # Daily fractionation spacing = 1.0 day (standard preclinical daily fractions)
    regimens: List[Tuple[str, List[Tuple[float, float]]]] = [
        ("1 x 10 Gy", [(0.0, 10.0)]),
        ("1 x 20 Gy", [(0.0, 20.0)]),
        ("1 x 40 Gy", [(0.0, 40.0)]),
        ("2 x 10 Gy", [(0.0, 10.0), (1.0, 10.0)]),
        ("4 x 5 Gy", [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]),
        ("4 x 10 Gy", [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0), (3.0, 10.0)]),
    ]

    all_results: Dict[str, Dict[str, object]] = {}
    summary_rows = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), dpi=130, sharex=True, sharey=True)
    for ax, (label, sched) in zip(axes.ravel(), regimens):
        scenarios = make_plos_one_scenarios(sched, label, t_end_days=35.0)
        res_map = {k: model.run(sc) for k, sc in scenarios.items()}
        all_results[label] = res_map

        ax.plot(res_map["Control"].t_days, res_map["Control"].rel_volume, color="#2ca02c", lw=2.2, label="Control (0 Gy)")
        ax.plot(
            res_map["EF5+"].t_days,
            res_map["EF5+"].rel_volume,
            color="#e41a1c",
            lw=2.4,
            label=f"EF5+ (pre T/M={res_map['EF5+'].mean_ef5_tm[0]:.2f})",
        )
        ax.plot(
            res_map["EF5-"].t_days,
            res_map["EF5-"].rel_volume,
            color="#111111",
            lw=2.4,
            label=f"EF5- (pre T/M={res_map['EF5-'].mean_ef5_tm[0]:.2f})",
        )
        ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 35)
        ax.set_ylim(0.3, 3.6)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

        for grp in ["Control", "EF5+", "EF5-"]:
            r = res_map[grp]
            idx13 = int(np.argmin(np.abs(r.t_days - 13.0)))
            idx32 = int(np.argmin(np.abs(r.t_days - 32.0)))
            summary_rows.append(
                {
                    "regimen": label,
                    "group": grp,
                    "total_dose_gy": sum(d for _, d in sched) if grp != "Control" else 0.0,
                    "n_fractions": len(sched) if grp != "Control" else 0,
                    "pre_rt_po2_mean_mmHg": round(float(r.mean_po2_mmHg[0]), 2),
                    "pre_rt_ef5_tm_mean": round(float(r.mean_ef5_tm[0]), 3),
                    "pre_rt_hypoxic_frac": round(float(r.hypoxic_frac[0]), 3),
                    "day13_rel_vol": round(float(r.rel_volume[idx13]), 3),
                    "day32_rel_vol": round(float(r.rel_volume[idx32]), 3),
                    "day35_rel_vol": round(float(r.rel_volume[-1]), 3),
                    "day13_ef5_tm_mean": round(float(r.mean_ef5_tm[idx13]), 3),
                    "day32_ef5_tm_mean": round(float(r.mean_ef5_tm[idx32]), 3),
                    "day13_po2_mean_mmHg": round(float(r.mean_po2_mmHg[idx13]), 2),
                    "day32_po2_mean_mmHg": round(float(r.mean_po2_mmHg[idx32]), 2),
                    "min_rel_vol": round(float(np.min(r.rel_volume)), 3),
                }
            )

    for ax in axes[1, :]:
        ax.set_xlabel("Days Post-Treatment")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized Tumour Volume (V / V0)")
    fig.suptitle(
        "Refined Coupled Oxygen-PDE Spheroid RT Model — PLOS ONE Regimen Benchmarks (A549)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "fig2_plos_one_regimens.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Radial Oxygenation, Reoxygenation & Virtual EF5 Profiles
    # ------------------------------------------------------------------
    print("3. Plotting radial pO2, cell-compartment, and emergent reoxygenation dynamics...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), dpi=130)
    r_hyp_4x5 = all_results["4 x 5 Gy"]["EF5+"]
    r_hyp_1x20 = all_results["1 x 20 Gy"]["EF5+"]
    r_ctrl = all_results["1 x 20 Gy"]["Control"]

    # Row 0: Temporal reoxygenation & EF5 T/M across 1x20 Gy vs 4x5 Gy vs Control
    axes[0, 0].plot(r_ctrl.t_days, r_ctrl.mean_po2_mmHg, color="#2ca02c", lw=2.0, label="Control")
    axes[0, 0].plot(r_hyp_1x20.t_days, r_hyp_1x20.mean_po2_mmHg, color="#e41a1c", lw=2.2, label="EF5+ 1x20 Gy")
    axes[0, 0].plot(r_hyp_4x5.t_days, r_hyp_4x5.mean_po2_mmHg, color="#377eb8", lw=2.2, label="EF5+ 4x5 Gy")
    axes[0, 0].set_title("Emergent Post-RT Reoxygenation\n(Mean Tumour pO2)")
    axes[0, 0].set_xlabel("Days Post-RT")
    axes[0, 0].set_ylabel("Mean pO2 (mmHg)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(r_ctrl.t_days, r_ctrl.mean_ef5_tm, color="#2ca02c", lw=2.0, label="Control")
    axes[0, 1].plot(r_hyp_1x20.t_days, r_hyp_1x20.mean_ef5_tm, color="#e41a1c", lw=2.2, label="EF5+ 1x20 Gy")
    axes[0, 1].plot(r_hyp_4x5.t_days, r_hyp_4x5.mean_ef5_tm, color="#377eb8", lw=2.2, label="EF5+ 4x5 Gy")
    axes[0, 1].set_title("Predicted Virtual EF5 PET T/M\n(Drops Post-RT vs Control)")
    axes[0, 1].set_xlabel("Days Post-RT")
    axes[0, 1].set_ylabel("Simulated Mean EF5 T/M")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    axes[0, 2].plot(r_hyp_1x20.t_days, r_hyp_1x20.viable_frac, color="#1f77b4", lw=2.2, label="Viable (V)")
    axes[0, 2].plot(r_hyp_1x20.t_days, r_hyp_1x20.damaged_frac, color="#ff7f0e", lw=2.2, label="RT-Damaged (D1+D2)")
    axes[0, 2].plot(r_hyp_1x20.t_days, r_hyp_1x20.necrotic_frac, color="#444444", lw=2.2, label="Necrotic/Fibrotic (N)")
    axes[0, 2].set_title("EF5+ 1x20 Gy Cellular Composition")
    axes[0, 2].set_xlabel("Days Post-RT")
    axes[0, 2].set_ylabel("Tumour Volume Fraction")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(alpha=0.25)

    # Row 1: Radial snapshots for EF5+ 4x5 Gy (inter-fraction reoxygenation) & 1x20 Gy
    snap_styles = [("pre_rt", "Pre-RT", "#111111", "-"), ("day2", "Day +2", "#e41a1c", "--"), ("day13", "Day +13", "#377eb8", "-."), ("day32", "Day +32", "#2ca02c", ":")]
    for tag, lbl, col, ls in snap_styles:
        if tag in r_hyp_4x5.snapshots:
            s = r_hyp_4x5.snapshots[tag]
            axes[1, 0].plot(s["r_mm"], s["p_o2"], color=col, ls=ls, lw=2.0, label=lbl)
            axes[1, 1].plot(s["r_mm"], s["ef5_tm"], color=col, ls=ls, lw=2.0, label=lbl)
        if tag in r_hyp_1x20.snapshots:
            s2 = r_hyp_1x20.snapshots[tag]
            axes[1, 2].plot(s2["r_mm"], s2["p_o2"], color=col, ls=ls, lw=2.0, label=lbl)

    axes[1, 0].axhline(params.p_hyp, color="#888888", ls=":", lw=1.0)
    axes[1, 0].set_title("Radial pO2 Profile — EF5+ 4x5 Gy")
    axes[1, 0].set_xlabel("Radius r (mm)")
    axes[1, 0].set_ylabel("pO2 (mmHg)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].set_title("Radial Virtual EF5 T/M — EF5+ 4x5 Gy")
    axes[1, 1].set_xlabel("Radius r (mm)")
    axes[1, 1].set_ylabel("Virtual EF5 T/M")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    axes[1, 2].axhline(params.p_hyp, color="#888888", ls=":", lw=1.0)
    axes[1, 2].set_title("Radial pO2 Profile — EF5+ 1x20 Gy")
    axes[1, 2].set_xlabel("Radius r (mm)")
    axes[1, 2].set_ylabel("pO2 (mmHg)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_reoxygenation_profiles.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Legacy 2013 Port vs Refined Model + First EF5 RT Study 20 Gy Overlay
    # ------------------------------------------------------------------
    print("4. Comparing legacy 2013 port vs refined model & First EF5 RT Study trajectories...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=130)

    # Panel A: Legacy 2013 1x6 Gy (faithful vs OER^2 bug fix)
    leg_h_raw = run_legacy_spheroid_2013(6.0, 1, hypoxic=True, preserve_legacy_typos=True)
    leg_n_raw = run_legacy_spheroid_2013(6.0, 1, hypoxic=False, preserve_legacy_typos=True)
    leg_h_fix = run_legacy_spheroid_2013(6.0, 1, hypoxic=True, preserve_legacy_typos=False)
    axes[0].plot(leg_h_raw.t_post_rt, leg_h_raw.rel_volume, "r-", lw=2.0, label="Legacy 2013 Hypoxic (raw)")
    axes[0].plot(leg_n_raw.t_post_rt, leg_n_raw.rel_volume, "k-", lw=2.0, label="Legacy 2013 Normoxic (raw)")
    axes[0].plot(leg_h_fix.t_post_rt, leg_h_fix.rel_volume, "r--", lw=1.8, label="Legacy Hypoxic (r & OER^2 fixed)")
    axes[0].set_xlim(0, 25)
    axes[0].set_ylim(0.5, 2.2)
    axes[0].set_xlabel("Days Post-RT")
    axes[0].set_ylabel("Relative Volume")
    axes[0].set_title("2013 Legacy ODE Port (1x6 Gy)\nShowing Typo Sensitivity")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)

    # Panel B: EF5+ vs EF5- separation summary across single vs fractionated regimens (Day 32)
    df = pd.DataFrame(summary_rows)
    reg_labels = [r[0] for r in regimens]
    gaps = []
    for lbl in reg_labels:
        sub = df[df["regimen"] == lbl].set_index("group")
        gaps.append(float(sub.loc["EF5+", "day32_rel_vol"] - sub.loc["EF5-", "day32_rel_vol"]))
    bars = axes[1].bar(range(len(reg_labels)), gaps, color=["#e41a1c", "#fb6a4a", "#fcae91", "#6baed6", "#3182bd", "#08519c"])
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(reg_labels)))
    axes[1].set_xticklabels(reg_labels, rotation=28, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Day +32 Delta Rel. Vol (EF5+ minus EF5-)")
    axes[1].set_title("Hypoxia Resistance Gap Collapses\nat High Dose & Multi-Fraction")
    axes[1].grid(axis="y", alpha=0.25)
    for b, g in zip(bars, gaps):
        axes[1].text(b.get_x() + b.get_width() / 2.0, g + 0.02, f"{g:+.2f}", ha="center", va="bottom", fontsize=8)

    # Panel C: Overlay refined 1D model (1x20 Gy & 0 Gy) with First EF5 RT Study PET/CT volumes
    wide_csv = root / "results" / "first_ef5_rt_study" / "tumour_master_wide.csv"
    if wide_csv.exists():
        wdf = pd.read_csv(wide_csv)
        for _, row in wdf.iterrows():
            v0 = row["d_minus1_pet_vol_cm3"]
            if pd.isna(v0) or v0 <= 0:
                continue
            ts = [0.0]
            rvs = [1.0]
            if not pd.isna(row["d13_pet_vol_cm3"]):
                ts.append(13.0)
                rvs.append(row["d13_pet_vol_cm3"] / v0)
            if not pd.isna(row["d32_pet_vol_cm3"]):
                ts.append(32.0)
                rvs.append(row["d32_pet_vol_cm3"] / v0)
            if len(ts) > 1:
                col = "#2ca02c" if row["treatment"] == "Control_0Gy" else "#d62728"
                axes[2].plot(ts, rvs, "o-", color=col, alpha=0.35, lw=1.2, ms=4)

    r_20 = all_results["1 x 20 Gy"]
    axes[2].plot(r_20["Control"].t_days, r_20["Control"].rel_volume, color="#2ca02c", lw=2.5, label="Model Control (0 Gy)")
    axes[2].plot(r_20["EF5+"].t_days, r_20["EF5+"].rel_volume, color="#e41a1c", lw=2.5, label="Model EF5+ (20 Gy)")
    axes[2].plot(r_20["EF5-"].t_days, r_20["EF5-"].rel_volume, color="#111111", lw=2.5, label="Model EF5- (20 Gy)")
    axes[2].set_xlim(0, 35)
    axes[2].set_ylim(0.3, 4.6)
    axes[2].set_xlabel("Days Post-RT")
    axes[2].set_ylabel("Relative Volume (V / V_Day-1)")
    axes[2].set_title("1D Model vs First EF5 RT Study\n(Faint lines = 6-mouse PET volumes)")
    axes[2].legend(fontsize=7.5, loc="upper left")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_legacy_vs_refined_and_first_study.png", dpi=130)
    plt.close(fig)

    df.to_csv(res_dir / "spheroid_regimen_summary.csv", index=False)
    print(f"Saved Sub-Goal 1 summary CSV to {res_dir / 'spheroid_regimen_summary.csv'}")
    print(df[["regimen", "group", "pre_rt_ef5_tm_mean", "day13_rel_vol", "day32_rel_vol", "day32_ef5_tm_mean"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_spheroid_study()
