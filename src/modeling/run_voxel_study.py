"""
CLI study runner for Sub-Goal 2: Tumour-Specific 3D Image-Initialised Voxel RT Model.

Evaluates the 3D continuum oxygen-PDE + Darcy-pressure voxel model on the curated
First EF5 RT Study (Nov 2010 - Jan 2011, A549 bilateral subcutaneous tumours):

  1. Pre-RT Untreated Growth Check (Day -15 -> Day -1) on c2_M0 (RT & Ctrl) and c2_M1L1R (Ctrl).
  2. Primary Paired Bilateral Study (Day -1 -> Day +13 -> Day +32) across all 5 longitudinal mice
     (c2_M1R, c2_M0, c2_M1L, c2_M1L1R, c2_M2R; 10 tumours: 5 RT_20Gy + 5 Control_0Gy) using ONE
     shared RadiobiologyParams parameter set.

Outputs:
  - results/ef5_rt_model/voxel_study_comparison.csv
  - results/ef5_rt_model/voxel_early_growth_check.csv
  - outputs/ef5_rt_model/voxel/figures/cohort_summary.png
  - outputs/ef5_rt_model/voxel/figures/<mouse_id>_observed_vs_simulated.png
  - outputs/ef5_rt_model/voxel/nifti/<mouse_id>/<tumour_id>_day{day}_{ef5_tm,po2,mask}.nii.gz
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from src.modeling.physics import RadiobiologyParams
from src.modeling.voxel_initializer import VoxelTumourInitialState, initialize_tumour_from_npz
from src.modeling.voxel_rt_model import VoxelRTModel, VoxelSimulationConfig, VoxelSimulationResult


LONGITUDINAL_MICE = ["c2_M1R", "c2_M0", "c2_M1L", "c2_M1L1R", "c2_M2R"]

TIMEPOINT_FILES = {
    -15: "day-15_2010-11-18",
    -1: "day-1_2010-12-02",
    13: "day13_2010-12-15",
    32: "day32_2011-01-03",
}


def _bundle_path(root: Path, mouse_id: str, day: int) -> Path:
    slug = TIMEPOINT_FILES[day]
    return root / "outputs" / "first_ef5_rt_study" / "voxel_bundles" / mouse_id / f"{mouse_id}_{slug}.npz"


def _save_sim_nifti(
    out_path: Path, arr_yzx: np.ndarray, init_state: VoxelTumourInitialState, dtype=np.float32
) -> None:
    """
    Exports a 3D simulation field on canonical (Y, Z, X) ascending mm grid to NIfTI-1 RAS+
    matching the convention of outputs/first_ef5_rt_study/nifti/ (DICOM LPS -> RAS+ via x=-x, y=-y).
    """
    # Reorder from (Y, Z, X) -> (X, Y, Z)
    arr_xyz = np.transpose(arr_yzx, (2, 0, 1))
    # Flip X and Y to convert ascending DICOM LPS (x_mm, y_mm) to ascending RAS+ (-x_mm, -y_mm)
    arr_ras = arr_xyz[::-1, ::-1, :].astype(dtype)
    dx = init_state.dx_mm
    affine = np.array(
        [
            [dx, 0.0, 0.0, -float(init_state.x_mm[-1])],
            [0.0, dx, 0.0, -float(init_state.y_mm[-1])],
            [0.0, 0.0, dx, float(init_state.z_mm[0])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr_ras, affine), str(out_path))


def _load_obs_coronal_slice(
    npz_path: Path, side: str
) -> Optional[Tuple[np.ndarray, np.ndarray, List[float], float]]:
    """Extracts mid-tumour coronal EF5 PET T/M slice and binary ROI contour from an observed .npz bundle."""
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    has_key = f"has_{side}_roi"
    if has_key in d and not bool(d[has_key].item()):
        return None
    mask = np.asarray(d[f"mask_{side}_pet"], dtype=bool)
    if not np.any(mask):
        return None
    pet_tm = np.asarray(d["pet_tm"], dtype=np.float64)
    y_mm = np.asarray(d["pet_y_mm"], dtype=np.float64)
    z_mm = np.asarray(d["pet_z_mm"], dtype=np.float64)
    x_mm = np.asarray(d["pet_x_mm"], dtype=np.float64)

    # Pick z-index through tumour centroid (axis 1 = Z)
    iy, iz, ix = np.nonzero(mask)
    z_mid = int(round(float(np.mean(iz))))
    y_c = float(np.mean(y_mm[iy]))
    x_c = float(np.mean(x_mm[ix]))

    sl_tm = pet_tm[:, z_mid, :]
    sl_mask = mask[:, z_mid, :]
    # Sort axes ascending around centroid (relative mm)
    y_rel = y_mm - y_c
    x_rel = x_mm - x_c
    if y_rel[0] > y_rel[-1]:
        y_rel = y_rel[::-1]
        sl_tm = sl_tm[::-1, :]
        sl_mask = sl_mask[::-1, :]
    if x_rel[0] > x_rel[-1]:
        x_rel = x_rel[::-1]
        sl_tm = sl_tm[:, ::-1]
        sl_mask = sl_mask[:, ::-1]

    extent = [float(x_rel[0]), float(x_rel[-1]), float(y_rel[0]), float(y_rel[-1])]
    mean_tm = float(np.mean(pet_tm[mask]))
    return sl_tm, sl_mask, extent, mean_tm


def _extract_sim_coronal_slice(
    res: VoxelSimulationResult, snap_day: float
) -> Tuple[np.ndarray, np.ndarray, List[float], float]:
    """Extracts mid-tumour coronal simulated EF5 T/M slice and ROI mask at snap_day."""
    snap = res.snapshots[snap_day]
    tm = snap["ef5_tm"]
    mask = snap["roi_mask"]
    st = res.init_state
    if np.any(mask):
        iy, iz, ix = np.nonzero(mask)
        z_mid = int(round(float(np.mean(iz))))
        y_c = float(np.mean(st.y_mm[iy]))
        x_c = float(np.mean(st.x_mm[ix]))
        mean_tm = float(np.mean(tm[mask]))
    else:
        z_mid = tm.shape[1] // 2
        y_c = float(np.mean(st.y_mm))
        x_c = float(np.mean(st.x_mm))
        mean_tm = float(res.params.tm_min)
    sl_tm = tm[:, z_mid, :]
    sl_mask = mask[:, z_mid, :]
    extent = [
        float(st.x_mm[0] - x_c),
        float(st.x_mm[-1] - x_c),
        float(st.y_mm[0] - y_c),
        float(st.y_mm[-1] - y_c),
    ]
    return sl_tm, sl_mask, extent, mean_tm


def _render_mouse_comparison(
    out_path: Path,
    root: Path,
    mouse_id: str,
    res_rt: VoxelSimulationResult,
    res_ctrl: VoxelSimulationResult,
    wide_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> None:
    """
    Renders a rich 4-row x 4-column per-mouse comparison montage:
      Rows 0-1: Observed vs Simulated RT_20Gy mid-tumour coronal EF5 T/M + ROI contour (Day -1, +13, +32)
      Rows 2-3: Observed vs Simulated Control_0Gy mid-tumour coronal EF5 T/M + ROI contour (Day -1, +13, +32)
      Rightmost column (Col 3): Quantitative Volume, Caliper Diameter, and EF5 T/M trajectory overlays!
    """
    fig = plt.figure(figsize=(16.5, 12.2), dpi=120, facecolor="#111111")
    gs = fig.add_gridspec(4, 4, left=0.05, right=0.97, top=0.92, bottom=0.06, wspace=0.26, hspace=0.34, width_ratios=[1, 1, 1, 1.35])

    days_study = [-1, 13, 32]
    snap_keys = [0.0, 14.0, 33.0]
    clim_tm = (0.5, 4.8)
    lim_mm = 11.5

    row_configs = [
        (0, "Observed RT (20 Gy)", "rt", True, res_rt, "#FF4081"),
        (1, "Simulated RT (20 Gy)", "rt", False, res_rt, "#FF4081"),
        (2, "Observed Ctrl (0 Gy)", "ctrl", True, res_ctrl, "#00E5FF"),
        (3, "Simulated Ctrl (0 Gy)", "ctrl", False, res_ctrl, "#00E5FF"),
    ]

    for row_idx, row_lbl, side, is_obs, res, contour_col in row_configs:
        for col_idx, (d_study, sk) in enumerate(zip(days_study, snap_keys)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("black")
            if is_obs:
                obs = _load_obs_coronal_slice(_bundle_path(root, mouse_id, d_study), side)
                if obs is None:
                    ax.text(0.5, 0.5, "No scan\n(died / N/A)", color="#999999", ha="center", va="center", transform=ax.transAxes, fontsize=9.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    sl_tm, sl_mask, ext, mean_tm = obs
                    ax.imshow(sl_tm, cmap="inferno", vmin=clim_tm[0], vmax=clim_tm[1], extent=ext, origin="lower", interpolation="bilinear", aspect="equal")
                    ax.contour(sl_mask.astype(float), levels=[0.5], colors=[contour_col], linewidths=1.6, extent=ext, origin="lower")
                    ax.set_xlim(-lim_mm, lim_mm)
                    ax.set_ylim(-lim_mm, lim_mm)
                    ax.set_title(f"Day {d_study:+d} | T/M={mean_tm:.2f}", color="white", fontsize=8.8)
            else:
                sl_tm, sl_mask, ext, mean_tm = _extract_sim_coronal_slice(res, sk)
                ax.imshow(sl_tm, cmap="inferno", vmin=clim_tm[0], vmax=clim_tm[1], extent=ext, origin="lower", interpolation="bilinear", aspect="equal")
                if np.any(sl_mask):
                    ax.contour(sl_mask.astype(float), levels=[0.5], colors=[contour_col], linewidths=1.6, extent=ext, origin="lower")
                ax.set_xlim(-lim_mm, lim_mm)
                ax.set_ylim(-lim_mm, lim_mm)
                ax.set_title(f"Day {d_study:+d} | T/M={mean_tm:.2f}", color="white", fontsize=8.8)
            if col_idx == 0:
                ax.set_ylabel(row_lbl, color=contour_col, fontsize=9, fontweight="bold")
            ax.tick_params(colors="#CCCCCC", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#444444")

    # Right column: quantitative trajectories (Volume, Relative Caliper/Diameter, EF5 T/M)
    m_wide = wide_df[wide_df["mouse_id"] == mouse_id]
    m_cal = cal_df[cal_df["mouse_id"] == mouse_id]

    # Plot 1 (rows 0-1 col 3): Absolute PET/CT volume vs Simulated 3D Volume
    ax_v = fig.add_subplot(gs[0:2, 3])
    ax_v.set_facecolor("#181818")
    ax_v.plot(res_rt.days_post_rt, res_rt.volume_cm3, color="#FF4081", lw=2.3, label="Sim RT (20 Gy)")
    ax_v.plot(res_ctrl.days_post_rt, res_ctrl.volume_cm3, color="#00E5FF", lw=2.3, label="Sim Ctrl (0 Gy)")
    for trt, col, mk in [("RT_20Gy", "#FF4081", "o"), ("Control_0Gy", "#00E5FF", "s")]:
        sub = m_wide[m_wide["treatment"] == trt]
        if not sub.empty:
            r = sub.iloc[0]
            ts, vs = [], []
            for d_val, c_name in [(-1, "d_minus1_pet_vol_cm3"), (13, "d13_pet_vol_cm3"), (32, "d32_pet_vol_cm3")]:
                if not pd.isna(r[c_name]):
                    ts.append(d_val)
                    vs.append(float(r[c_name]))
            ax_v.plot(ts, vs, mk + "--", color=col, ms=7, lw=1.4, label=f"Obs {trt.split('_')[0]}")
    ax_v.axvline(0.0, color="#FFD740", ls=":", lw=1.2)
    ax_v.set_title("3D Tumour Volume (cm^3)", color="white", fontsize=10, fontweight="bold")
    ax_v.set_xlabel("Days Post-RT", color="white", fontsize=8.5)
    ax_v.set_ylabel("Volume (cm^3)", color="white", fontsize=8.5)
    ax_v.tick_params(colors="white", labelsize=8)
    ax_v.legend(fontsize=7.5, facecolor="#222222", edgecolor="#555555", labelcolor="white")
    ax_v.grid(alpha=0.2)

    # Plot 2 (row 2 col 3): Relative diameter vs observed calipers
    ax_c = fig.add_subplot(gs[2, 3])
    ax_c.set_facecolor("#181818")
    ax_c.plot(res_rt.days_post_rt, res_rt.equiv_diam_mm / res_rt.equiv_diam_mm[0], color="#FF4081", lw=2.2, label="Sim RT Eq.Diam")
    ax_c.plot(res_ctrl.days_post_rt, res_ctrl.equiv_diam_mm / res_ctrl.equiv_diam_mm[0], color="#00E5FF", lw=2.2, label="Sim Ctrl Eq.Diam")
    for trt, col, mk in [("RT_20Gy", "#FF4081", "o"), ("Control_0Gy", "#00E5FF", "s")]:
        sc = m_cal[(m_cal["treatment"] == trt) & (m_cal["days_post_rt"] >= -1)].sort_values("days_post_rt")
        if not sc.empty:
            ax_c.plot(sc["days_post_rt"], sc["norm_diameter_vs_day_minus1"], mk + "--", color=col, ms=5, lw=1.2, alpha=0.85)
    ax_c.axhline(1.0, color="#777777", ls=":", lw=0.9)
    ax_c.axvline(0.0, color="#FFD740", ls=":", lw=1.0)
    ax_c.set_title("Relative Diameter vs Calipers", color="white", fontsize=9.5, fontweight="bold")
    ax_c.set_ylabel("D / D_Day-1", color="white", fontsize=8.5)
    ax_c.tick_params(colors="white", labelsize=8)
    ax_c.grid(alpha=0.2)

    # Plot 3 (row 3 col 3): Mean EF5 T/M trajectory (Observed vs Simulated)
    ax_e = fig.add_subplot(gs[3, 3])
    ax_e.set_facecolor("#181818")
    ax_e.plot(res_rt.days_post_rt, res_rt.mean_ef5_tm, color="#FF4081", lw=2.2, label="Sim RT T/M")
    ax_e.plot(res_ctrl.days_post_rt, res_ctrl.mean_ef5_tm, color="#00E5FF", lw=2.2, label="Sim Ctrl T/M")
    for trt, col, mk in [("RT_20Gy", "#FF4081", "o"), ("Control_0Gy", "#00E5FF", "s")]:
        sub = m_wide[m_wide["treatment"] == trt]
        if not sub.empty:
            r = sub.iloc[0]
            ts, tms = [], []
            for d_val, c_name in [(-1, "d_minus1_tm_mean"), (13, "d13_tm_mean"), (32, "d32_tm_mean")]:
                if not pd.isna(r[c_name]):
                    ts.append(d_val)
                    tms.append(float(r[c_name]))
            ax_e.plot(ts, tms, mk + "--", color=col, ms=6, lw=1.3)
    ax_e.axvline(0.0, color="#FFD740", ls=":", lw=1.0)
    ax_e.set_title("Mean EF5 PET T/M (Reoxygenation)", color="white", fontsize=9.5, fontweight="bold")
    ax_e.set_xlabel("Days Post-RT", color="white", fontsize=8.5)
    ax_e.set_ylabel("EF5 T/M", color="white", fontsize=8.5)
    ax_e.tick_params(colors="white", labelsize=8)
    ax_e.grid(alpha=0.2)

    fig.suptitle(
        f"3D Image-Initialised Voxel Model vs Observed EF5 PET/CT — {mouse_id} "
        f"(Init Geom: RT={res_rt.init_state.geometry_source}, Ctrl={res_ctrl.init_state.geometry_source}, dx={res_rt.init_state.dx_mm} mm)",
        color="white",
        fontsize=12.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_voxel_study(root: Path = Path("."), dx_mm: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_fig_dir = root / "outputs" / "ef5_rt_model" / "voxel" / "figures"
    out_nii_dir = root / "outputs" / "ef5_rt_model" / "voxel" / "nifti"
    res_dir = root / "results" / "ef5_rt_model"
    for d in [out_fig_dir, out_nii_dir, res_dir]:
        d.mkdir(parents=True, exist_ok=True)

    wide_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "tumour_master_wide.csv")
    cal_df = pd.read_csv(root / "results" / "first_ef5_rt_study" / "caliper_growth_curves.csv")

    # Shared radiobiology parameter set across ALL mice (zero per-mouse tuning!)
    params = RadiobiologyParams()
    model = VoxelRTModel(params)

    # ------------------------------------------------------------------
    # 1. Early Untreated Growth Check (Day -15 -> Day -1, 14 days, 0 Gy)
    # ------------------------------------------------------------------
    print("1. Running pre-RT untreated 3D early-growth check (Day -15 -> Day -1)...")
    early_cases = [("c2_M0", "rt"), ("c2_M0", "ctrl"), ("c2_M1L1R", "ctrl")]
    early_rows = []
    for mid, side in early_cases:
        npz_m15 = _bundle_path(root, mid, -15)
        st = initialize_tumour_from_npz(npz_m15, side, params=params, dx_mm=dx_mm, pad_mm=7.0, dose_gy_override=0.0)
        cfg = VoxelSimulationConfig(t_end_days=14.0, dt_days=0.5, dose_schedule=[], snapshot_days=[0.0, 14.0])
        res = model.run(st, cfg)
        tid = st.tumour_id
        wrow = wide_df[wide_df["tumour_id"] == tid].iloc[0]
        early_rows.append(
            {
                "tumour_id": tid,
                "geometry_source": st.geometry_source,
                "obs_dm15_vol_cm3": round(float(wrow["d_minus15_pet_vol_cm3"]), 4),
                "sim_dm15_vol_cm3": round(float(res.volume_cm3[0]), 4),
                "obs_dm1_vol_cm3": round(float(wrow["d_minus1_pet_vol_cm3"]), 4),
                "sim_dm1_vol_cm3": round(float(res.volume_cm3[-1]), 4),
                "obs_fold_growth": round(float(wrow["d_minus1_pet_vol_cm3"] / wrow["d_minus15_pet_vol_cm3"]), 2),
                "sim_fold_growth": round(float(res.rel_volume[-1]), 2),
                "obs_dm15_tm_mean": round(float(wrow["d_minus15_tm_mean"]), 3),
                "obs_dm1_tm_mean": round(float(wrow["d_minus1_tm_mean"]), 3),
                "sim_dm15_tm_mean": round(float(res.mean_ef5_tm[0]), 3),
                "sim_dm1_tm_mean": round(float(res.mean_ef5_tm[-1]), 3),
            }
        )
    early_df = pd.DataFrame(early_rows)
    early_df.to_csv(res_dir / "voxel_early_growth_check.csv", index=False)
    print(early_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Primary Paired Bilateral Study (Day -1 -> Day +13 -> Day +32)
    # ------------------------------------------------------------------
    print("\n2. Running 3D image-initialised voxel model across all 5 longitudinal mice (10 tumours)...")
    comp_rows = []
    results_by_mouse: Dict[str, Dict[str, VoxelSimulationResult]] = {}

    for mid in LONGITUDINAL_MICE:
        npz_m1 = _bundle_path(root, mid, -1)
        results_by_mouse[mid] = {}

        for side in ["rt", "ctrl"]:
            t0 = time.perf_counter()
            st = initialize_tumour_from_npz(npz_m1, side, params=params, dx_mm=dx_mm, pad_mm=8.5)
            sched = [(1.0, 20.0)] if side == "rt" else []  # Day -1 is t=0 -> RT Day 0 is t=1.0 day
            cfg = VoxelSimulationConfig(
                t_end_days=33.0,
                dt_days=0.5,
                dose_schedule=sched,
                snapshot_days=[0.0, 14.0, 33.0],
            )
            res = model.run(st, cfg)
            elapsed = time.perf_counter() - t0
            results_by_mouse[mid][side] = res

            # Export simulated 3D NIfTI maps for Day -1, +13, +32
            for s_key, d_lbl in [(0.0, "day-1"), (14.0, "day13"), (33.0, "day32")]:
                snap = res.snapshots[s_key]
                base = out_nii_dir / mid / f"{st.tumour_id}_sim_{d_lbl}"
                _save_sim_nifti(Path(f"{base}_ef5_tm.nii.gz"), snap["ef5_tm"], st, dtype=np.float32)
                _save_sim_nifti(Path(f"{base}_po2_mmHg.nii.gz"), snap["p_o2"], st, dtype=np.float32)
                _save_sim_nifti(Path(f"{base}_mask.nii.gz"), snap["roi_mask"], st, dtype=np.uint8)

            # Extract comparison metrics at Day -1 (t=0), Day +13 (t=14), Day +32 (t=33)
            idx13 = int(np.argmin(np.abs(res.days_post_rt - 13.0)))
            idx32 = int(np.argmin(np.abs(res.days_post_rt - 32.0)))
            wrow = wide_df[wide_df["tumour_id"] == st.tumour_id].iloc[0]

            # Spatial correlation between observed pre-RT EF5 T/M and initial t=0 simulated EF5 T/M inside ROI
            roi0 = res.snapshots[0.0]["roi_mask"]
            r_init = float(np.corrcoef(st.pet_tm0[roi0], res.snapshots[0.0]["ef5_tm"][roi0])[0, 1]) if roi0.sum() > 10 else np.nan

            obs_v0 = float(wrow["d_minus1_pet_vol_cm3"])
            obs_v13 = float(wrow["d13_pet_vol_cm3"]) if not pd.isna(wrow["d13_pet_vol_cm3"]) else np.nan
            obs_v32 = float(wrow["d32_pet_vol_cm3"]) if not pd.isna(wrow["d32_pet_vol_cm3"]) else np.nan

            comp_rows.append(
                {
                    "tumour_id": st.tumour_id,
                    "mouse_id": mid,
                    "treatment": st.treatment,
                    "dose_gy": st.dose_gy,
                    "geometry_source": st.geometry_source,
                    "grid_shape": "x".join(map(str, st.shape)),
                    "runtime_s": round(elapsed, 2),
                    "init_ef5_spatial_r": round(r_init, 3),
                    "obs_d_minus1_vol_cm3": round(obs_v0, 4),
                    "sim_d_minus1_vol_cm3": round(float(res.volume_cm3[0]), 4),
                    "obs_d13_vol_cm3": round(obs_v13, 4) if not np.isnan(obs_v13) else np.nan,
                    "sim_d13_vol_cm3": round(float(res.volume_cm3[idx13]), 4),
                    "obs_d32_vol_cm3": round(obs_v32, 4) if not np.isnan(obs_v32) else np.nan,
                    "sim_d32_vol_cm3": round(float(res.volume_cm3[idx32]), 4),
                    "obs_d13_rel_vol": round(obs_v13 / obs_v0, 3) if not np.isnan(obs_v13) else np.nan,
                    "sim_d13_rel_vol": round(float(res.rel_volume[idx13]), 3),
                    "obs_d32_rel_vol": round(obs_v32 / obs_v0, 3) if not np.isnan(obs_v32) else np.nan,
                    "sim_d32_rel_vol": round(float(res.rel_volume[idx32]), 3),
                    "obs_d_minus1_tm_mean": round(float(wrow["d_minus1_tm_mean"]), 3),
                    "sim_d_minus1_tm_mean": round(float(res.mean_ef5_tm[0]), 3),
                    "obs_d13_tm_mean": round(float(wrow["d13_tm_mean"]), 3) if not pd.isna(wrow["d13_tm_mean"]) else np.nan,
                    "sim_d13_tm_mean": round(float(res.mean_ef5_tm[idx13]), 3),
                    "obs_d32_tm_mean": round(float(wrow["d32_tm_mean"]), 3) if not pd.isna(wrow["d32_tm_mean"]) else np.nan,
                    "sim_d32_tm_mean": round(float(res.mean_ef5_tm[idx32]), 3),
                    "sim_d_minus1_po2_mmHg": round(float(res.mean_po2_mmHg[0]), 2),
                    "sim_d13_po2_mmHg": round(float(res.mean_po2_mmHg[idx13]), 2),
                    "sim_d32_po2_mmHg": round(float(res.mean_po2_mmHg[idx32]), 2),
                    "sim_d32_viable_frac": round(float(res.viable_frac[idx32]), 3),
                    "sim_d32_necrotic_frac": round(float(res.necrotic_frac[idx32]), 3),
                }
            )
            print(
                f"  {st.tumour_id:13s} ({st.geometry_source:3s}, {st.shape}, {elapsed:.1f}s) | "
                f"Vol: {res.volume_cm3[0]:.2f} -> {res.volume_cm3[idx13]:.2f} -> {res.volume_cm3[idx32]:.2f} cm^3 "
                f"(obs: {obs_v0:.2f} -> {obs_v13:.2f} -> {obs_v32 if not np.isnan(obs_v32) else -1:.2f}) | "
                f"Sim T/M: {res.mean_ef5_tm[0]:.2f} -> {res.mean_ef5_tm[idx13]:.2f} -> {res.mean_ef5_tm[idx32]:.2f}"
            )

        _render_mouse_comparison(
            out_fig_dir / f"{mid}_observed_vs_simulated.png",
            root,
            mid,
            results_by_mouse[mid]["rt"],
            results_by_mouse[mid]["ctrl"],
            wide_df,
            cal_df,
        )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(res_dir / "voxel_study_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Cohort-Wide Summary Comparison Figure
    # ------------------------------------------------------------------
    print("\n3. Rendering cohort-wide 3D voxel summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8), dpi=130)

    # Panel A: Observed vs Simulated Relative Volume trajectories (RT vs Control)
    for mid in LONGITUDINAL_MICE:
        r_rt = results_by_mouse[mid]["rt"]
        r_ct = results_by_mouse[mid]["ctrl"]
        axes[0, 0].plot(r_ct.days_post_rt, r_ct.rel_volume, color="#2ca02c", lw=1.8, alpha=0.75)
        axes[0, 0].plot(r_rt.days_post_rt, r_rt.rel_volume, color="#d62728", lw=1.8, alpha=0.75)
    # Overlay mean observed points
    for trt, col, lbl in [("Control_0Gy", "#1b7837", "Obs Control (mean +/- sd)"), ("RT_20Gy", "#b2182b", "Obs RT 20 Gy (mean +/- sd)")]:
        sub = comp_df[comp_df["treatment"] == trt]
        obs_means = [1.0, float(sub["obs_d13_rel_vol"].mean()), float(sub["obs_d32_rel_vol"].mean())]
        obs_sds = [0.0, float(sub["obs_d13_rel_vol"].std()), float(sub["obs_d32_rel_vol"].std())]
        axes[0, 0].errorbar([-1, 13, 32], obs_means, yerr=obs_sds, fmt="o--", color=col, lw=2.2, ms=7, capsize=4, label=lbl)
    axes[0, 0].plot([], [], color="#2ca02c", lw=2.0, label="Sim 3D Control (5 mice)")
    axes[0, 0].plot([], [], color="#d62728", lw=2.0, label="Sim 3D RT 20 Gy (5 mice)")
    axes[0, 0].axvline(0.0, color="#666666", ls=":", lw=1.0)
    axes[0, 0].set_title("Cohort Relative Volume: Simulated 3D vs Observed PET/CT")
    axes[0, 0].set_xlabel("Days Post-RT")
    axes[0, 0].set_ylabel("Normalized Volume (V / V_Day-1)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[0, 0].grid(alpha=0.25)

    # Panel B: Scatter of Observed vs Simulated Absolute Volume across Day +13 & Day +32
    for trt, col, mk in [("Control_0Gy", "#2ca02c", "s"), ("RT_20Gy", "#d62728", "o")]:
        sub = comp_df[comp_df["treatment"] == trt]
        axes[0, 1].scatter(sub["obs_d13_vol_cm3"], sub["sim_d13_vol_cm3"], c=col, marker=mk, s=55, alpha=0.75, label=f"{trt.split('_')[0]} Day +13")
        sub32 = sub.dropna(subset=["obs_d32_vol_cm3"])
        axes[0, 1].scatter(sub32["obs_d32_vol_cm3"], sub32["sim_d32_vol_cm3"], c=col, marker=mk, edgecolor="black", lw=1.2, s=75, label=f"{trt.split('_')[0]} Day +32")
    lim_v = max(float(comp_df[["obs_d32_vol_cm3", "sim_d32_vol_cm3"]].max().max()), 1.8) * 1.08
    axes[0, 1].plot([0, lim_v], [0, lim_v], "k--", lw=1.0, alpha=0.6)
    axes[0, 1].set_xlim(0, lim_v)
    axes[0, 1].set_ylim(0, lim_v)
    axes[0, 1].set_xlabel("Observed PET/CT Volume (cm^3)")
    axes[0, 1].set_ylabel("Simulated 3D Volume (cm^3)")
    axes[0, 1].set_title("Observed vs Simulated Absolute Tumour Volumes")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.25)

    # Panel C: Paired Within-Mouse RT / Control EF5 T/M Ratio (robust to Day +32 muscle background shift!)
    ratios_obs_m1, ratios_obs_13, ratios_obs_32 = [], [], []
    ratios_sim_m1, ratios_sim_13, ratios_sim_32 = [], [], []
    for mid in LONGITUDINAL_MICE:
        r_rt = comp_df[(comp_df["mouse_id"] == mid) & (comp_df["treatment"] == "RT_20Gy")].iloc[0]
        r_ct = comp_df[(comp_df["mouse_id"] == mid) & (comp_df["treatment"] == "Control_0Gy")].iloc[0]
        ratios_obs_m1.append(r_rt["obs_d_minus1_tm_mean"] / r_ct["obs_d_minus1_tm_mean"])
        ratios_sim_m1.append(r_rt["sim_d_minus1_tm_mean"] / r_ct["sim_d_minus1_tm_mean"])
        if not pd.isna(r_rt["obs_d13_tm_mean"]):
            ratios_obs_13.append(r_rt["obs_d13_tm_mean"] / r_ct["obs_d13_tm_mean"])
            ratios_sim_13.append(r_rt["sim_d13_tm_mean"] / r_ct["sim_d13_tm_mean"])
        if not pd.isna(r_rt["obs_d32_tm_mean"]):
            ratios_obs_32.append(r_rt["obs_d32_tm_mean"] / r_ct["obs_d32_tm_mean"])
            ratios_sim_32.append(r_rt["sim_d32_tm_mean"] / r_ct["sim_d32_tm_mean"])
    x_pos = np.array([0, 1, 2])
    w = 0.34
    axes[1, 0].bar(x_pos - w / 2, [np.mean(ratios_obs_m1), np.mean(ratios_obs_13), np.mean(ratios_obs_32)], w, yerr=[np.std(ratios_obs_m1), np.std(ratios_obs_13), np.std(ratios_obs_32)], capsize=4, color="#555555", label="Observed PET")
    axes[1, 0].bar(x_pos + w / 2, [np.mean(ratios_sim_m1), np.mean(ratios_sim_13), np.mean(ratios_sim_32)], w, yerr=[np.std(ratios_sim_m1), np.std(ratios_sim_13), np.std(ratios_sim_32)], capsize=4, color="#e41a1c", label="Simulated 3D")
    axes[1, 0].axhline(1.0, color="black", ls=":", lw=1.0)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(["Day -1\n(Pre-RT)", "Day +13\n(Post-RT)", "Day +32\n(Post-RT)"])
    axes[1, 0].set_ylabel("Paired EF5 T/M Ratio (RT / Control)")
    axes[1, 0].set_title("Within-Mouse Post-RT Reoxygenation\n(RT / Contralateral Control EF5 Ratio)")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(axis="y", alpha=0.25)

    # Panel D: Pre-RT Hypoxia (EF5 T/M) vs Day +32 20 Gy Response across Mice
    rt_sub = comp_df[comp_df["treatment"] == "RT_20Gy"].copy()
    axes[1, 1].scatter(rt_sub["obs_d_minus1_tm_mean"], rt_sub["sim_d32_rel_vol"], c="#e41a1c", s=85, label="Sim 3D Day +32 Rel. Vol")
    obs_cal_d27 = []
    for _, r in rt_sub.iterrows():
        c_row = cal_df[(cal_df["tumour_id"] == r["tumour_id"]) & (cal_df["days_post_rt"] == 27)]
        val = float(c_row["norm_diameter_vs_day_minus1"].iloc[0]) ** 3 if not c_row.empty else np.nan
        obs_cal_d27.append(val)
        axes[1, 1].annotate(r["mouse_id"], (r["obs_d_minus1_tm_mean"] + 0.04, r["sim_d32_rel_vol"] + 0.03), fontsize=8.5, color="#b2182b")
    axes[1, 1].scatter(rt_sub["obs_d_minus1_tm_mean"], obs_cal_d27, c="#111111", marker="^", s=75, label="Obs Caliper Day +27 (D/D0)^3")
    axes[1, 1].axhline(1.0, color="#777777", ls=":", lw=1.0)
    axes[1, 1].set_xlabel("Pre-RT EF5 PET Mean T/M (Day -1)")
    axes[1, 1].set_ylabel("Post-RT Relative Volume")
    axes[1, 1].set_title("Pre-RT Hypoxia Stratifies 20 Gy Response Across Mice")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        "Sub-Goal 2: 3D CT/EF5-PET Image-Initialised Voxel Model — First EF5 RT Study (5 Bilateral Mice)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_fig_dir / "cohort_summary.png", dpi=130)
    plt.close(fig)

    print(f"Saved Sub-Goal 2 comparison CSV to {res_dir / 'voxel_study_comparison.csv'}")
    return comp_df, early_df


if __name__ == "__main__":
    run_voxel_study()
