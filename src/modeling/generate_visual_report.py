"""
Generates comprehensive publication-quality visual comparisons for both Sub-Goal 1 (1D ODE/PDE)
and Sub-Goal 2 (3D CT/EF5-PET image-initialised voxel model):

  1. outputs/ef5_rt_model/visual_report/01_1d_plos_one_real_vs_2013_vs_refined.png
     (3-column comparison: Real PLOS ONE Data | Original 2013 Spheroid ODE | Refined Oxygen-PDE Model
      across 1x10 Gy, 1x40 Gy, 4x5 Gy, and 4x10 Gy.)
  2. outputs/ef5_rt_model/visual_report/02_1d_first_study_and_7gy_real_vs_models.png
     (Real A549 20 Gy First Study + 7 Gy rt_data_a549.mat curves overlaid against Original 2013 ODE
      and Refined Oxygen-PDE Model, plus quantitative comparison panels.)
  3. outputs/ef5_rt_model/visual_report/03a_3d_multislice_c2_M1R_RT_20Gy.png
     outputs/ef5_rt_model/visual_report/03b_3d_multislice_c2_M1R_Ctrl_0Gy.png
     outputs/ef5_rt_model/visual_report/03c_3d_multislice_c2_M1L1R_RT_20Gy.png
     (5-slice stacks spanning each tumour ROI at Day -1, Day +13, Day +32 comparing Real PET/CT Overlay
      vs Simulated 3D Virtual EF5 T/M vs Simulated 3D pO2 mmHg.)
  4. outputs/ef5_rt_model/visual_report/04_3d_simulated_vs_real_growth_curves_all_mice.png
     (2x5 grid of all 5 longitudinal mice: absolute 3D PET/CT volume + normalized caliper/equivalent
      diameter growth curves for paired RT_20Gy vs Control_0Gy.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio

from src.inspect_set1 import make_pet_overlay_cmap
from src.modeling.legacy_spheroid import run_legacy_spheroid_2013
from src.modeling.physics import RadiobiologyParams
from src.modeling.spheroid_rt_model import SpheroidRTModel, make_plos_one_scenarios
from src.modeling.voxel_initializer import initialize_tumour_from_npz
from src.modeling.voxel_rt_model import VoxelRTModel, VoxelSimulationConfig, VoxelSimulationResult


LONGITUDINAL_MICE = ["c2_M1R", "c2_M0", "c2_M1L", "c2_M1L1R", "c2_M2R"]
TIMEPOINT_FILES = {
    -1: "day-1_2010-12-02",
    13: "day13_2010-12-15",
    32: "day32_2011-01-03",
}
PET_OVERLAY_CMAP = make_pet_overlay_cmap("inferno", max_alpha=0.80)


def _bundle_path(root: Path, mouse_id: str, day: int) -> Path:
    return root / "outputs" / "first_ef5_rt_study" / "voxel_bundles" / mouse_id / f"{mouse_id}_{TIMEPOINT_FILES[day]}.npz"


# ---------------------------------------------------------------------------
# Figure 1: 1D PLOS ONE Real vs Original 2013 vs Refined Oxygen-PDE
# ---------------------------------------------------------------------------
def build_fig1_1d_plos_one_comparison(root: Path, out_dir: Path) -> None:
    print("1. Building 1D PLOS ONE 3-column comparison (Real | Original 2013 | Refined O2-PDE)...")
    srt_img = np.asarray(Image.open(root / "images" / "set4" / "past_work" / "Reports" / "expt_vs_sim_srt.png"))
    frt_img = np.asarray(Image.open(root / "images" / "set4" / "past_work" / "Reports" / "expt_vs_sim_frt.png"))

    # Exact crop boxes (y0:y1, x0:x1) separating the 4 sub-panels in srt (714x1068) and frt (731x1075)
    crops = [
        ("1 x 10 Gy", [(0.0, 10.0)], srt_img[0:355, 0:510], srt_img[0:355, 520:1068]),
        ("1 x 40 Gy", [(0.0, 40.0)], srt_img[355:714, 0:510], srt_img[355:714, 520:1068]),
        ("4 x 5 Gy", [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)], frt_img[0:365, 0:515], frt_img[0:365, 525:1075]),
        ("4 x 10 Gy", [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0), (3.0, 10.0)], frt_img[365:731, 0:515], frt_img[365:731, 525:1075]),
    ]

    prm = RadiobiologyParams()
    model = SpheroidRTModel(prm)

    fig, axes = plt.subplots(4, 3, figsize=(16.5, 16.0), dpi=130, facecolor="white")
    col_titles = [
        "Real Experimental Data\n(PLOS ONE 2015, 80 A549 Tumours)",
        "Original 2013 Spheroid ODE\n(Binary rim r2, ad-hoc Vt brake & OERh=1)",
        "New Refined Oxygen-PDE Model\n(Continuous O2 PDE + AHF OER + Reoxygenation)",
    ]

    for row_idx, (label, sched, expt_crop, orig_crop) in enumerate(crops):
        # Col 0: Real experimental plot
        ax0 = axes[row_idx, 0]
        ax0.imshow(expt_crop)
        ax0.set_xticks([])
        ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor("#CCCCCC")
        ax0.set_ylabel(label, fontsize=11.5, fontweight="bold", labelpad=10)

        # Col 1: Original 2013 simulated plot
        ax1 = axes[row_idx, 1]
        ax1.imshow(orig_crop)
        ax1.set_xticks([])
        ax1.set_yticks([])
        for sp in ax1.spines.values():
            sp.set_edgecolor("#CCCCCC")

        # Col 2: New Refined Oxygen-PDE simulation
        ax2 = axes[row_idx, 2]
        scs = make_plos_one_scenarios(sched, label, t_end_days=35.0)
        res = {k: model.run(sc) for k, sc in scs.items()}
        ax2.plot(res["Control"].t_days, res["Control"].rel_volume, color="#2ca02c", lw=2.5, label="Control (0 Gy)")
        ax2.plot(
            res["EF5+"].t_days,
            res["EF5+"].rel_volume,
            color="#e41a1c",
            lw=2.6,
            label=f"EF5+ (T/M {res['EF5+'].mean_ef5_tm[0]:.2f} -> {res['EF5+'].mean_ef5_tm[-1]:.2f})",
        )
        ax2.plot(
            res["EF5-"].t_days,
            res["EF5-"].rel_volume,
            color="#111111",
            lw=2.6,
            label=f"EF5- (T/M {res['EF5-'].mean_ef5_tm[0]:.2f} -> {res['EF5-'].mean_ef5_tm[-1]:.2f})",
        )
        ax2.axhline(1.0, color="#999999", ls=":", lw=1.0)
        ax2.set_xlim(0, 35)
        ax2.set_ylim(0.4, 3.8)
        ax2.set_title(f"Refined {label}", fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("Normalized Volume (V / V0)", fontsize=9.5)
        if row_idx == 3:
            ax2.set_xlabel("Days Post-Treatment", fontsize=9.5)
        ax2.legend(fontsize=8.2, loc="upper left", frameon=True)
        ax2.grid(alpha=0.25)

        if row_idx == 0:
            for c_idx, ct in enumerate(col_titles):
                axes[0, c_idx].set_title(ct, fontsize=11.5, fontweight="bold", pad=12)

    fig.suptitle(
        "1D Spheroid Model Comparison: Real PLOS ONE Data vs Original 2013 ODE vs Refined Oxygen-PDE",
        fontsize=14,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_dir / "01_1d_plos_one_real_vs_2013_vs_refined.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Real 20 Gy (First Study) & 7 Gy (rt_data_a549.mat) vs Original & Refined 1D
# ---------------------------------------------------------------------------
def build_fig2_1d_first_study_and_7gy(root: Path, out_dir: Path) -> None:
    print("2. Building 1D real-data overlay (First Study 20 Gy + Legacy 7 Gy vs Original 2013 & Refined)...")
    cal_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "caliper_growth_curves.csv")
    wide_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "tumour_master_wide.csv")
    mat7 = sio.loadmat(root / "images" / "set4" / "past_work" / "Code" / "02. Scheidegger" / "rt_data_a549.mat")

    prm = RadiobiologyParams()
    model = SpheroidRTModel(prm)

    # Run refined 1D for 1x20 Gy and 1x7 Gy
    res_20 = {k: model.run(sc) for k, sc in make_plos_one_scenarios([(0.0, 20.0)], "1x20 Gy", t_end_days=40.0).items()}
    res_7 = {k: model.run(sc) for k, sc in make_plos_one_scenarios([(0.0, 7.0)], "1x7 Gy", t_end_days=45.0).items()}

    # Run original 2013 legacy ODE at 1x20 Gy and 1x7 Gy (rescaled to 2013 dose scale & raw dose scale)
    leg_20_ctrl = run_legacy_spheroid_2013(0.0, 1, hypoxic=False, preserve_legacy_typos=True)
    leg_20_h = run_legacy_spheroid_2013(20.0, 1, hypoxic=True, preserve_legacy_typos=True)
    leg_20_n = run_legacy_spheroid_2013(20.0, 1, hypoxic=False, preserve_legacy_typos=True)

    leg_7_ctrl = run_legacy_spheroid_2013(0.0, 1, hypoxic=False, preserve_legacy_typos=True)
    leg_7_h = run_legacy_spheroid_2013(7.0, 1, hypoxic=True, preserve_legacy_typos=True)
    leg_7_n = run_legacy_spheroid_2013(7.0, 1, hypoxic=False, preserve_legacy_typos=True)

    # Stratify First EF5 RT Study 20 Gy caliper trajectories into EF5+ (T/M > 2.5) vs EF5- (T/M <= 2.5)
    post_cal = cal_df[cal_df["days_post_rt"] >= -1].copy()
    post_cal["rel_vol_cal"] = post_cal["norm_diameter_vs_day_minus1"] ** 3
    post_cal["group"] = np.where(
        post_cal["treatment"] == "Control_0Gy",
        "Control",
        np.where(post_cal["pre_rt_tm_mean"] > 2.5, "EF5+ (T/M > 2.5)", "EF5- (T/M <= 2.5)"),
    )
    grp_cal = post_cal.groupby(["group", "days_post_rt"])["rel_vol_cal"].agg(["mean", "sem"]).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), dpi=130, facecolor="white")

    # Panel A (0,0): First EF5 RT Study 20 Gy Real Caliper (D/D0)^3 vs Refined 1D Model
    ax = axes[0, 0]
    styles = [("Control", "#2ca02c", "^"), ("EF5+ (T/M > 2.5)", "#e41a1c", "o"), ("EF5- (T/M <= 2.5)", "#111111", "s")]
    for grp, col, mk in styles:
        sub = grp_cal[grp_cal["group"] == grp].sort_values("days_post_rt")
        ax.errorbar(sub["days_post_rt"], sub["mean"], yerr=sub["sem"].fillna(0), fmt=mk + "--", color=col, lw=1.4, ms=5.5, capsize=3, alpha=0.8, label=f"Real {grp}")
    ax.plot(res_20["Control"].t_days, res_20["Control"].rel_volume, color="#2ca02c", lw=2.8, label="Refined 1D Control")
    ax.plot(res_20["EF5+"].t_days, res_20["EF5+"].rel_volume, color="#e41a1c", lw=2.8, label="Refined 1D EF5+")
    ax.plot(res_20["EF5-"].t_days, res_20["EF5-"].rel_volume, color="#111111", lw=2.8, label="Refined 1D EF5-")
    ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
    ax.set_xlim(-2, 40)
    ax.set_ylim(0.2, 4.2)
    ax.set_title("First EF5 RT Study (1 x 20 Gy): Real Calipers\nvs Refined Oxygen-PDE Model", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Days Post-RT")
    ax.set_ylabel("Normalized Volume (D/D0)^3 or V/V0")
    ax.legend(fontsize=7.3, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)

    # Panel B (0,1): Same Real 20 Gy Data vs Original 2013 MATLAB ODE
    ax = axes[0, 1]
    for grp, col, mk in styles:
        sub = grp_cal[grp_cal["group"] == grp].sort_values("days_post_rt")
        ax.errorbar(sub["days_post_rt"], sub["mean"], yerr=sub["sem"].fillna(0), fmt=mk + "--", color=col, lw=1.4, ms=5.5, capsize=3, alpha=0.65, label=f"Real {grp}")
    ax.plot(leg_20_ctrl.t_post_rt, leg_20_ctrl.rel_volume, color="#2ca02c", lw=2.6, label="Orig 2013 Control")
    ax.plot(leg_20_h.t_post_rt, leg_20_h.rel_volume, color="#e41a1c", lw=2.6, label="Orig 2013 EF5+ (20 Gy)")
    ax.plot(leg_20_n.t_post_rt, leg_20_n.rel_volume, color="#111111", lw=2.6, label="Orig 2013 EF5- (20 Gy)")
    ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.2, 4.2)
    ax.set_title("First EF5 RT Study (1 x 20 Gy): Real Calipers\nvs Original 2013 MATLAB ODE", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Days Post-RT")
    ax.set_ylabel("Normalized Volume (V / V0)")
    ax.legend(fontsize=7.3, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)

    # Panel C (0,2): Legacy 7 Gy A549 dataset (rt_data_a549.mat) vs Refined 1D vs Original 2013
    ax = axes[0, 2]
    t7 = mat7["a549_7Gy_norm_t"].ravel()
    L7 = mat7["a549_7Gy_norm_length"]  # rows: 0=Ctrl, 1=T/M>2.5 (EF5+), 2=T/M<2.5 (EF5-)
    ax.plot(t7, L7[0] ** 3, "^--", color="#2ca02c", lw=1.5, ms=5.5, alpha=0.8, label="Real 7 Gy Control")
    ax.plot(t7, L7[1] ** 3, "o--", color="#e41a1c", lw=1.5, ms=5.5, alpha=0.8, label="Real 7 Gy EF5+ (T/M>2.5)")
    ax.plot(t7, L7[2] ** 3, "s--", color="#111111", lw=1.5, ms=5.5, alpha=0.8, label="Real 7 Gy EF5- (T/M<2.5)")
    ax.plot(res_7["Control"].t_days, res_7["Control"].rel_volume, color="#2ca02c", lw=2.6, label="Refined 1D Control")
    ax.plot(res_7["EF5+"].t_days, res_7["EF5+"].rel_volume, color="#e41a1c", lw=2.6, label="Refined 1D EF5+ (7 Gy)")
    ax.plot(res_7["EF5-"].t_days, res_7["EF5-"].rel_volume, color="#111111", lw=2.6, label="Refined 1D EF5- (7 Gy)")
    ax.plot(leg_7_h.t_post_rt, leg_7_h.rel_volume, ":", color="#e41a1c", lw=2.0, label="Orig 2013 EF5+ (7 Gy)")
    ax.plot(leg_7_n.t_post_rt, leg_7_n.rel_volume, ":", color="#111111", lw=2.0, label="Orig 2013 EF5- (7 Gy)")
    ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
    ax.set_xlim(0, 45)
    ax.set_ylim(0.4, 3.6)
    ax.set_title("Legacy A549 7 Gy Data (rt_data_a549.mat)\nvs Refined 1D (solid) & Orig 2013 (dotted)", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Days Post-RT")
    ax.set_ylabel("Normalized Volume (L/L0)^3 or V/V0")
    ax.legend(fontsize=7.0, loc="upper left", ncol=2)
    ax.grid(alpha=0.25)

    # Panel D (1,0): Original 2013 vs Refined — Hypoxia Resistance Gap across PLOS ONE regimens
    ax = axes[1, 0]
    regs = [
        ("1x10", [(0.0, 10.0)], 10.0, 1),
        ("1x20", [(0.0, 20.0)], 20.0, 1),
        ("1x40", [(0.0, 40.0)], 40.0, 1),
        ("2x10", [(0.0, 10.0), (1.0, 10.0)], 10.0, 2),
        ("4x5", [(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)], 5.0, 4),
        ("4x10", [(0.0, 10.0), (1.0, 10.0), (2.0, 10.0), (3.0, 10.0)], 10.0, 4),
    ]
    gaps_ref, gaps_leg = [], []
    for lbl, sched, d_f, n_f in regs:
        r_ref = {k: model.run(sc) for k, sc in make_plos_one_scenarios(sched, lbl, t_end_days=32.0).items()}
        gaps_ref.append(float(r_ref["EF5+"].rel_volume[-1] - r_ref["EF5-"].rel_volume[-1]))
        lh = run_legacy_spheroid_2013(d_f, n_f, hypoxic=True, preserve_legacy_typos=True)
        ln = run_legacy_spheroid_2013(d_f, n_f, hypoxic=False, preserve_legacy_typos=True)
        idx_h = int(np.argmin(np.abs(lh.t_post_rt - 32.0)))
        idx_n = int(np.argmin(np.abs(ln.t_post_rt - 32.0)))
        gaps_leg.append(float(lh.rel_volume[idx_h] - ln.rel_volume[idx_n]))
    x = np.arange(len(regs))
    w = 0.36
    ax.bar(x - w / 2, gaps_leg, w, color="#999999", label="Original 2013 ODE (unscaled Gy)")
    ax.bar(x + w / 2, gaps_ref, w, color="#e41a1c", label="Refined Oxygen-PDE Model")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in regs])
    ax.set_ylabel("Day +32 Rel. Vol Gap (EF5+ minus EF5-)")
    ax.set_title("Hypoxia Resistance Separation at Actual Gy Doses\n(Orig 2013 breaks at real Gy; Refined tracks PLOS ONE)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # Panel E (1,1): Emergent Reoxygenation Prediction (New Capability vs Real First Study EF5 PET)
    ax = axes[1, 1]
    # Compute observed First Study paired RT/Control EF5 T/M ratios at Day -1, +13, +32
    obs_rat = {}
    for d_lbl, col in [(-1, "d_minus1_tm_mean"), (13, "d13_tm_mean"), (32, "d32_tm_mean")]:
        vals = []
        for mid in LONGITUDINAL_MICE:
            rt_v = wide_df[(wide_df["mouse_id"] == mid) & (wide_df["treatment"] == "RT_20Gy")][col].iloc[0]
            ct_v = wide_df[(wide_df["mouse_id"] == mid) & (wide_df["treatment"] == "Control_0Gy")][col].iloc[0]
            if not pd.isna(rt_v) and not pd.isna(ct_v):
                vals.append(float(rt_v / ct_v))
        obs_rat[d_lbl] = (float(np.mean(vals)), float(np.std(vals)))
    t_mod = res_20["EF5+"].t_days
    sim_rat_h = res_20["EF5+"].mean_ef5_tm / res_20["Control"].mean_ef5_tm
    sim_rat_n = res_20["EF5-"].mean_ef5_tm / res_20["Control"].mean_ef5_tm
    ax.plot(t_mod, sim_rat_h / sim_rat_h[0], color="#e41a1c", lw=2.5, label="Refined 1D EF5+ (20 Gy / Ctrl)")
    ax.plot(t_mod, sim_rat_n / sim_rat_n[0], color="#111111", lw=2.2, ls="-.", label="Refined 1D EF5- (20 Gy / Ctrl)")
    obs_t = [-1, 13, 32]
    obs_m = [obs_rat[d][0] / obs_rat[-1][0] for d in obs_t]
    obs_s = [obs_rat[d][1] / obs_rat[-1][0] for d in obs_t]
    ax.errorbar(obs_t, obs_m, yerr=obs_s, fmt="ko--", lw=2.0, ms=7, capsize=4, label="Real First Study PET (RT/Ctrl, norm.)")
    ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
    ax.set_xlim(-2, 35)
    ax.set_ylim(0.2, 1.3)
    ax.set_title("New Capability: Virtual EF5 PET Reoxygenation\n(Original 2013 ODE had no O2/EF5 readout)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Days Post-RT")
    ax.set_ylabel("Normalized RT / Control EF5 T/M")
    ax.legend(fontsize=7.8)
    ax.grid(alpha=0.25)

    # Panel F (1,2): Clean Structured Improvement Scorecard
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        "Key Improvements: Original 2013 ODE -> Refined O2-PDE\n"
        "-----------------------------------------------------\n"
        "1. Oxygen Profile:\n"
        "   - 2013: Fixed binary rim width r2 = 0.45\n"
        "   - New:  Continuous quasi-steady spherical O2 PDE\n"
        "           (matches Byrne analytical to < 1.5e-3)\n\n"
        "2. Growth Saturation:\n"
        "   - 2013: Ad-hoc exponential brake above Vt = 500\n"
        "   - New:  Emergent from O2 diffusion & angiogenic lag\n\n"
        "3. Radiobiology & Doses:\n"
        "   - 2013: Rescaled doses (2,6,12 Gy), b/OER typo,\n"
        "           OERh=1 hack for normoxic tumours\n"
        "   - New:  True physical Gy (7-40 Gy), Alper-Howard-\n"
        "           Flanders OER(pO2), corrected b/OER^2\n\n"
        "4. Post-RT Swelling & Reoxygenation:\n"
        "   - 2013: Manual kd3=0->0.05 switch; r2 halved on RT\n"
        "   - New:  Smooth 2-stage mitotic clearance (D1->D2->N)\n"
        "           & emergent inter-fraction O2 rise + virtual EF5"
    )
    ax.text(
        0.02, 0.96, summary_text,
        va="top", ha="left", family="monospace", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F9FC", edgecolor="#CCCCCC"),
        transform=ax.transAxes,
    )

    fig.suptitle(
        "1D Spheroid Model Validation on Real A549 Data (20 Gy & 7 Gy) & Comparison vs Original 2013 Model",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_dir / "02_1d_first_study_and_7gy_real_vs_models.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: 3D Multi-Slice ROI Stacks (Real PET/CT Overlay vs Sim EF5 & Sim pO2)
# ---------------------------------------------------------------------------
def _extract_obs_multislice(
    npz_path: Path, side: str, rel_offsets_mm: List[float]
) -> Optional[Dict[str, object]]:
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    if not bool(d[f"has_{side}_roi"].item()):
        return None
    mask_pet = np.asarray(d[f"mask_{side}_pet"], dtype=bool)
    if not np.any(mask_pet):
        return None
    pet_tm = np.asarray(d["pet_tm"], dtype=np.float64)
    ct_avail = bool(d["ct_available"].item())
    ct_on_pet = np.asarray(d["ct_on_pet_grid"], dtype=np.float64) if ct_avail else None

    y_mm = np.asarray(d["pet_y_mm"], dtype=np.float64)
    z_mm = np.asarray(d["pet_z_mm"], dtype=np.float64)
    x_mm = np.asarray(d["pet_x_mm"], dtype=np.float64)

    iy, iz, ix = np.nonzero(mask_pet)
    y_c = float(np.mean(y_mm[iy]))
    z_c = float(np.mean(z_mm[iz]))
    x_c = float(np.mean(x_mm[ix]))

    # Sort Y and X ascending once for clean display
    y_rel = y_mm - y_c
    x_rel = x_mm - x_c
    flip_y = y_rel[0] > y_rel[-1]
    flip_x = x_rel[0] > x_rel[-1]
    if flip_y:
        y_rel = y_rel[::-1]
    if flip_x:
        x_rel = x_rel[::-1]
    extent = [float(x_rel[0]), float(x_rel[-1]), float(y_rel[0]), float(y_rel[-1])]

    ct_clim = None
    if ct_on_pet is not None:
        med = float(np.median(ct_on_pet[::2, ::2, ::2]))
        ct_clim = (0.0, 2200.0) if med > -200.0 else (-500.0, 800.0)

    slices = []
    for dz in rel_offsets_mm:
        z_target = z_c + dz
        zi = int(np.argmin(np.abs(z_mm - z_target)))
        sl_tm = pet_tm[:, zi, :]
        sl_mk = mask_pet[:, zi, :]
        sl_ct = ct_on_pet[:, zi, :] if ct_on_pet is not None else None
        if flip_y:
            sl_tm = sl_tm[::-1, :]
            sl_mk = sl_mk[::-1, :]
            if sl_ct is not None:
                sl_ct = sl_ct[::-1, :]
        if flip_x:
            sl_tm = sl_tm[:, ::-1]
            sl_mk = sl_mk[:, ::-1]
            if sl_ct is not None:
                sl_ct = sl_ct[:, ::-1]
        slices.append({"dz_actual": float(z_mm[zi] - z_c), "tm": sl_tm, "mask": sl_mk, "ct": sl_ct})

    return {
        "slices": slices,
        "extent": extent,
        "ct_avail": ct_avail,
        "ct_clim": ct_clim,
        "vol_cm3": float(mask_pet.sum() * np.prod(d["pet_spacing_mm"]) / 1000.0),
        "mean_tm": float(np.mean(pet_tm[mask_pet])),
    }


def _extract_sim_multislice(
    res: VoxelSimulationResult, snap_day: float, rel_offsets_mm: List[float]
) -> Dict[str, object]:
    snap = res.snapshots[snap_day]
    tm = snap["ef5_tm"]
    po2 = snap["p_o2"]
    phi = snap["phi"]
    mask = snap["roi_mask"]
    st = res.init_state
    if np.any(mask):
        iy, iz, ix = np.nonzero(mask)
        y_c = float(np.mean(st.y_mm[iy]))
        z_c = float(np.mean(st.z_mm[iz]))
        x_c = float(np.mean(st.x_mm[ix]))
        mean_tm = float(np.mean(tm[mask]))
        mean_po2 = float(np.mean(po2[mask]))
    else:
        y_c, z_c, x_c = float(np.mean(st.y_mm)), float(np.mean(st.z_mm)), float(np.mean(st.x_mm))
        mean_tm = float(res.params.tm_min)
        mean_po2 = float(st.p_bdry_mmHg)

    extent = [
        float(st.x_mm[0] - x_c),
        float(st.x_mm[-1] - x_c),
        float(st.y_mm[0] - y_c),
        float(st.y_mm[-1] - y_c),
    ]
    idx_t = int(np.argmin(np.abs(res.t_sim_days - snap_day)))
    slices = []
    for dz in rel_offsets_mm:
        zi = int(np.argmin(np.abs(st.z_mm - (z_c + dz))))
        slices.append(
            {
                "dz_actual": float(st.z_mm[zi] - z_c),
                "tm": tm[:, zi, :],
                "po2": po2[:, zi, :],
                "phi": phi[:, zi, :],
                "mask": mask[:, zi, :],
            }
        )
    return {
        "slices": slices,
        "extent": extent,
        "vol_cm3": float(res.volume_cm3[idx_t]),
        "mean_tm": mean_tm,
        "mean_po2": mean_po2,
    }


def _adaptive_z_offsets(
    npz_path: Path, side: str, res: VoxelSimulationResult, snap_day: float
) -> List[float]:
    """
    Computes 5 evenly spaced coronal slice offsets spanning [-span, +span] around the tumour
    centroid at a given timepoint, adapting to tumour growth so enlarged Day +13/+32 control
    tumours do not clip out peripheral hypoxic lobes.
    """
    z_half = 3.4
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=False)
        if bool(d[f"has_{side}_roi"].item()):
            mk = np.asarray(d[f"mask_{side}_pet"], dtype=bool)
            if np.any(mk):
                z_mm = np.asarray(d["pet_z_mm"], dtype=np.float64)
                iz = np.nonzero(mk)[1]
                z_half = 0.5 * float(np.max(z_mm[iz]) - np.min(z_mm[iz]))
    else:
        mask_s = res.snapshots[snap_day]["roi_mask"]
        if np.any(mask_s):
            iz = np.nonzero(mask_s)[1]
            z_mm = res.init_state.z_mm
            z_half = 0.5 * float(np.max(z_mm[iz]) - np.min(z_mm[iz]))
    span = round(float(np.clip(0.68 * z_half, 2.4, 5.2)), 1)
    return [round(span * f, 1) for f in (-1.0, -0.5, 0.0, 0.5, 1.0)]


def build_fig3_multislice_panel(
    root: Path,
    out_path: Path,
    mouse_id: str,
    side: str,
    res: VoxelSimulationResult,
) -> None:
    """
    Renders a 9-row x 5-column multi-slice ROI visual stack:
      3 timepoints (Day -1, Day +13, Day +32) x 3 modalities (Real PET/CT Overlay, Sim EF5 T/M, Sim pO2)
      across 5 coronal slices spanning each timepoint's tumour ROI z-extent.
    """
    tp_specs = [(-1, 0.0, "Day -1 (Pre-RT)"), (13, 14.0, "Day +13 (Post-RT)"), (32, 33.0, "Day +32 (Post-RT)")]
    col_headers = ["Slice 1 (-span)", "Slice 2 (-span/2)", "Center (z = 0)", "Slice 4 (+span/2)", "Slice 5 (+span)"]
    contour_col = "#FF4081" if side == "rt" else "#00E5FF"
    trt_lbl = "Irradiated (20 Gy)" if side == "rt" else "Control (0 Gy)"
    lim_mm = 10.5
    tm_clim = (0.7, 4.5)
    po2_clim = (0.0, 35.0)

    cmap_tm = plt.get_cmap("inferno").copy()
    cmap_tm.set_bad("#0A0812")
    cmap_po2 = plt.get_cmap("viridis").copy()
    cmap_po2.set_bad("#141414")

    fig, axes = plt.subplots(9, 5, figsize=(14.5, 22.0), dpi=120, facecolor="#111111")
    fig.subplots_adjust(left=0.11, right=0.93, top=0.918, bottom=0.03, wspace=0.08, hspace=0.26)

    im_tm_ref = None
    im_po2_ref = None

    for tp_i, (d_study, sk, tp_title) in enumerate(tp_specs):
        npz_tp = _bundle_path(root, mouse_id, d_study)
        offsets_mm = _adaptive_z_offsets(npz_tp, side, res, sk)
        obs = _extract_obs_multislice(npz_tp, side, offsets_mm)
        sim = _extract_sim_multislice(res, sk, offsets_mm)
        r_obs = tp_i * 3
        r_ef5 = tp_i * 3 + 1
        r_po2 = tp_i * 3 + 2

        if r_obs == 0:
            mod_tag = "PET/CT" if (obs and obs["ct_avail"]) else "PET-only"
            obs_tag = f"Obs Vol={obs['vol_cm3']:.2f} cc, T/M={obs['mean_tm']:.2f}" if obs else "No Obs Scan"
            fig.text(
                0.52, 0.941,
                f"--- {tp_title} ({mod_tag}: {obs_tag} | Sim Vol={sim['vol_cm3']:.2f} cc, T/M={sim['mean_tm']:.2f}) ---",
                color="#FFD740", fontsize=9.2, fontweight="bold", ha="center", va="bottom",
            )

        for c_i, dz in enumerate(offsets_mm):
            # Row A: Real PET/CT Overlay
            ax = axes[r_obs, c_i]
            ax.set_facecolor("black")
            if obs is None:
                ax.text(0.5, 0.5, "No scan", color="#888888", ha="center", va="center", transform=ax.transAxes, fontsize=8.5)
            else:
                sl = obs["slices"][c_i]
                ext = obs["extent"]
                if obs["ct_avail"] and sl["ct"] is not None:
                    ax.imshow(sl["ct"], cmap="gray", vmin=obs["ct_clim"][0], vmax=obs["ct_clim"][1], extent=ext, origin="lower", interpolation="bilinear", aspect="equal")
                    ax.imshow(sl["tm"], cmap=PET_OVERLAY_CMAP, vmin=tm_clim[0], vmax=tm_clim[1], extent=ext, origin="lower", interpolation="bilinear", aspect="equal")
                else:
                    ax.imshow(sl["tm"], cmap="inferno", vmin=tm_clim[0], vmax=tm_clim[1], extent=ext, origin="lower", interpolation="bilinear", aspect="equal")
                if np.any(sl["mask"]):
                    ax.contour(sl["mask"].astype(float), levels=[0.5], colors=[contour_col], linewidths=1.5, extent=ext, origin="lower")
                ax.set_xlim(-lim_mm, lim_mm)
                ax.set_ylim(-lim_mm, lim_mm)
                ax.text(
                    0.04, 0.94, f"z={dz:+.1f} mm",
                    color="white", fontsize=7.5, fontweight="bold", ha="left", va="top",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.65, edgecolor="none"),
                )
            if r_obs == 0:
                ax.set_title(col_headers[c_i], color="white", fontsize=9.2, fontweight="bold", pad=4)
            elif c_i == 2:
                mod_tag = "PET/CT" if (obs and obs["ct_avail"]) else "PET-only"
                obs_tag = f"Obs Vol={obs['vol_cm3']:.2f} cc, T/M={obs['mean_tm']:.2f}" if obs else "No Obs Scan"
                ax.set_title(f"--- {tp_title} ({mod_tag}: {obs_tag} | Sim Vol={sim['vol_cm3']:.2f} cc, T/M={sim['mean_tm']:.2f}) ---", color="#FFD740", fontsize=9.2, fontweight="bold", pad=6)
            if c_i == 0:
                ax.set_ylabel(f"{tp_title.split()[0]} {tp_title.split()[1]}\nReal PET/CT", color="white", fontsize=8.5, fontweight="bold")

            # Row B: Simulated 3D Virtual EF5 T/M (masked outside tumour active footprint for clean contrast)
            ax = axes[r_ef5, c_i]
            ax.set_facecolor("#0A0812")
            sl_s = sim["slices"][c_i]
            ext_s = sim["extent"]
            tm_masked = np.where(sl_s["phi"] >= 0.05, sl_s["tm"], np.nan)
            im_tm_ref = ax.imshow(tm_masked, cmap=cmap_tm, vmin=tm_clim[0], vmax=tm_clim[1], extent=ext_s, origin="lower", interpolation="bilinear", aspect="equal")
            if np.any(sl_s["mask"]):
                ax.contour(sl_s["mask"].astype(float), levels=[0.5], colors=[contour_col], linewidths=1.5, extent=ext_s, origin="lower")
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            if c_i == 0:
                ax.set_ylabel("Sim 3D\nEF5 T/M", color=contour_col, fontsize=8.5, fontweight="bold")

            # Row C: Simulated 3D Local pO2 (mmHg) — exterior host stroma masked dark so intra-tumour pO2 is unambiguous
            ax = axes[r_po2, c_i]
            ax.set_facecolor("#141414")
            po2_masked = np.where(sl_s["phi"] >= 0.05, sl_s["po2"], np.nan)
            im_po2_ref = ax.imshow(po2_masked, cmap=cmap_po2, vmin=po2_clim[0], vmax=po2_clim[1], extent=ext_s, origin="lower", interpolation="bilinear", aspect="equal")
            if np.any(sl_s["mask"]):
                ax.contour(sl_s["mask"].astype(float), levels=[0.5], colors=["white"], linewidths=1.3, extent=ext_s, origin="lower")
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            if c_i == 0:
                ax.set_ylabel("Sim 3D\npO2 (mmHg)", color="#80CBC4", fontsize=8.5, fontweight="bold")

            for row_ax in (axes[r_obs, c_i], axes[r_ef5, c_i], axes[r_po2, c_i]):
                row_ax.set_xticks([])
                row_ax.set_yticks([])
                for sp in row_ax.spines.values():
                    sp.set_edgecolor("#444444")

    # Add shared colorbars on the right margin
    cax1 = fig.add_axes([0.945, 0.55, 0.014, 0.32])
    cb1 = fig.colorbar(im_tm_ref, cax=cax1)
    cb1.set_label("EF5 PET T/M Ratio", color="white", fontsize=9)
    cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    cax2 = fig.add_axes([0.945, 0.14, 0.014, 0.32])
    cb2 = fig.colorbar(im_po2_ref, cax=cax2)
    cb2.set_label("Simulated pO2 (mmHg)", color="white", fontsize=9)
    cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)

    fig.suptitle(
        f"3D Multi-Slice ROI Comparison: Real PET/CT vs Simulated Voxel Maps — {mouse_id} {trt_lbl}\n"
        f"(5 Coronal Slices through Tumour ROI | Day -1 Pre-RT, Day +13 & Day +32 Post-RT | ±10.5 mm FOV)",
        color="white",
        fontsize=12.5,
        fontweight="bold",
        y=0.982,
    )
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: 3D Simulated vs Real Growth Curves Across All 5 Longitudinal Mice
# ---------------------------------------------------------------------------
def build_fig4_3d_growth_curves_all_mice(
    root: Path,
    out_dir: Path,
    results_by_mouse: Dict[str, Dict[str, VoxelSimulationResult]],
) -> None:
    print("4. Building 2x5 cohort 3D simulated vs real growth curves figure...")
    wide_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "tumour_master_wide.csv")
    cal_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "caliper_growth_curves.csv")

    fig, axes = plt.subplots(2, 5, figsize=(19.0, 8.2), dpi=130, facecolor="white", sharex=True)

    for col_i, mid in enumerate(LONGITUDINAL_MICE):
        res_rt = results_by_mouse[mid]["rt"]
        res_ct = results_by_mouse[mid]["ctrl"]
        m_wide = wide_df[wide_df["mouse_id"] == mid]
        m_cal = cal_df[cal_df["mouse_id"] == mid]

        rt_tm0 = float(m_wide[m_wide["treatment"] == "RT_20Gy"]["d_minus1_tm_mean"].iloc[0])
        ct_tm0 = float(m_wide[m_wide["treatment"] == "Control_0Gy"]["d_minus1_tm_mean"].iloc[0])

        # Row 0: Absolute 3D Tumour Volume (cm^3)
        ax0 = axes[0, col_i]
        ax0.plot(res_ct.days_post_rt, res_ct.volume_cm3, color="#2ca02c", lw=2.4, label="Sim 3D Ctrl (0 Gy)")
        ax0.plot(res_rt.days_post_rt, res_rt.volume_cm3, color="#e41a1c", lw=2.4, label="Sim 3D RT (20 Gy)")
        for trt, col, mk, lbl in [("Control_0Gy", "#1b7837", "s", "Real PET/CT Ctrl"), ("RT_20Gy", "#b2182b", "o", "Real PET/CT RT")]:
            r = m_wide[m_wide["treatment"] == trt].iloc[0]
            ts, vs = [], []
            for d_val, cname in [(-1, "d_minus1_pet_vol_cm3"), (13, "d13_pet_vol_cm3"), (32, "d32_pet_vol_cm3")]:
                if not pd.isna(r[cname]):
                    ts.append(d_val)
                    vs.append(float(r[cname]))
            ax0.plot(ts, vs, mk + "--", color=col, lw=1.6, ms=6.5, label=lbl)
        ax0.axvline(0.0, color="#666666", ls=":", lw=1.0)
        ax0.set_title(f"{mid}\n(Pre-RT T/M: RT={rt_tm0:.2f}, Ctrl={ct_tm0:.2f})", fontsize=10, fontweight="bold")
        if col_i == 0:
            ax0.set_ylabel("3D Tumour Volume (cm^3)", fontsize=9.5, fontweight="bold")
            ax0.legend(fontsize=7.3, loc="upper left")
        ax0.grid(alpha=0.25)

        # Row 1: Relative Linear Diameter (Calipers + PET/CT (V/V0)^(1/3) vs Sim 3D Eq. Diameter)
        ax1 = axes[1, col_i]
        ax1.plot(res_ct.days_post_rt, res_ct.equiv_diam_mm / res_ct.equiv_diam_mm[0], color="#2ca02c", lw=2.4, label="Sim 3D Ctrl Eq.D")
        ax1.plot(res_rt.days_post_rt, res_rt.equiv_diam_mm / res_rt.equiv_diam_mm[0], color="#e41a1c", lw=2.4, label="Sim 3D RT Eq.D")
        for trt, col, mk, lbl in [("Control_0Gy", "#1b7837", "s", "Real Caliper Ctrl"), ("RT_20Gy", "#b2182b", "o", "Real Caliper RT")]:
            sc = m_cal[(m_cal["treatment"] == trt) & (m_cal["days_post_rt"] >= -1)].sort_values("days_post_rt")
            if not sc.empty:
                ax1.plot(sc["days_post_rt"], sc["norm_diameter_vs_day_minus1"], mk + "--", color=col, lw=1.4, ms=5.5, alpha=0.85, label=lbl)
            # Also plot PET/CT (V/V0)^(1/3) as hollow diamonds
            r = m_wide[m_wide["treatment"] == trt].iloc[0]
            v0 = float(r["d_minus1_pet_vol_cm3"])
            ts, rds = [-1], [1.0]
            for d_val, cname in [(13, "d13_pet_vol_cm3"), (32, "d32_pet_vol_cm3")]:
                if not pd.isna(r[cname]):
                    ts.append(d_val)
                    rds.append((float(r[cname]) / v0) ** (1.0 / 3.0))
            ax1.plot(ts, rds, "D", mfc="none", mec=col, mew=1.4, ms=6.5, label=f"PET/CT (V/V0)^1/3" if col_i == 0 and trt == "RT_20Gy" else None)

        ax1.axhline(1.0, color="#888888", ls=":", lw=0.9)
        ax1.axvline(0.0, color="#666666", ls=":", lw=1.0)
        ax1.set_xlim(-2, 40)
        ax1.set_ylim(0.45, 2.05)
        ax1.set_xlabel("Days Post-RT", fontsize=9)
        if col_i == 0:
            ax1.set_ylabel("Normalized Diameter (D / D_Day-1)", fontsize=9.5, fontweight="bold")
            ax1.legend(fontsize=7.0, loc="upper left")
        ax1.grid(alpha=0.25)

    fig.suptitle(
        "3D Image-Initialised Voxel Model vs Real Longitudinal Growth Curves — All 5 Bilateral A549 Mice (Shared Parameters)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(out_dir / "04_3d_simulated_vs_real_growth_curves_all_mice.png", dpi=130)
    plt.close(fig)


def main() -> None:
    root = Path(".")
    out_dir = root / "outputs" / "ef5_rt_model" / "visual_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 1: 1D comparison figures
    build_fig1_1d_plos_one_comparison(root, out_dir)
    build_fig2_1d_first_study_and_7gy(root, out_dir)

    # Part 2: Run 3D voxel model for all 5 longitudinal mice
    print("3. Running 3D voxel simulations for multi-slice ROI stacks and cohort growth curves...")
    prm = RadiobiologyParams()
    model_3d = VoxelRTModel(prm)
    results_by_mouse: Dict[str, Dict[str, VoxelSimulationResult]] = {}

    for mid in LONGITUDINAL_MICE:
        npz_m1 = _bundle_path(root, mid, -1)
        results_by_mouse[mid] = {}
        for side in ["rt", "ctrl"]:
            st = initialize_tumour_from_npz(npz_m1, side, params=prm, dx_mm=0.5, pad_mm=8.5)
            sched = [(1.0, 20.0)] if side == "rt" else []
            cfg = VoxelSimulationConfig(t_end_days=33.0, dt_days=0.5, dose_schedule=sched, snapshot_days=[0.0, 14.0, 33.0])
            results_by_mouse[mid][side] = model_3d.run(st, cfg)

    # Multi-slice ROI stacks for showcase mice (c2_M1R full PET+CT at all 3 timepoints; c2_M1L1R low-hypoxia responder)
    print("   Rendering multi-slice ROI stacks...")
    build_fig3_multislice_panel(root, out_dir / "03a_3d_multislice_c2_M1R_RT_20Gy.png", "c2_M1R", "rt", results_by_mouse["c2_M1R"]["rt"])
    build_fig3_multislice_panel(root, out_dir / "03b_3d_multislice_c2_M1R_Ctrl_0Gy.png", "c2_M1R", "ctrl", results_by_mouse["c2_M1R"]["ctrl"])
    build_fig3_multislice_panel(root, out_dir / "03c_3d_multislice_c2_M1L1R_RT_20Gy.png", "c2_M1L1R", "rt", results_by_mouse["c2_M1L1R"]["rt"])

    # Cohort-wide 3D simulated vs real growth curves
    build_fig4_3d_growth_curves_all_mice(root, out_dir, results_by_mouse)

    # Sync report figures into the git-tracked reports/figures folder
    import shutil
    rep_fig_dir = root / "outputs" / "ef5_rt_model" / "reports" / "figures"
    rep_fig_dir.mkdir(parents=True, exist_ok=True)
    for png in out_dir.glob("*.png"):
        shutil.copy2(png, rep_fig_dir / png.name)
    print(f"All visual report figures saved under {out_dir} and synced to {rep_fig_dir}")


if __name__ == "__main__":
    main()
