"""
Stages 1 & 2 of the Second EF5 RT Study (40-Mouse / 80-Tumour PLOS ONE Cohort) Validation Suite:

  - Stage 1: Strict Blind ("Out-of-Box") Validation using frozen Study 1 parameters (`RadiobiologyParams` v1.0)
      * 1D Refined Oxygen-PDE Spheroid Model across all 80 tumours (78 imaged + 2 unimaged controls)
      * 3D Image-Initialised Voxel Continuum PDE Model across all 78 imaged tumours (`.npz` bundles)
      * Counterfactual Schedule Experiment: Actual 2011 schedule (with the 4-day missing-lead-shield
        gap between Fraction #1 on 2011-06-03 and Fraction #2 on 2011-06-07) vs. Nominal consecutive
        24-hour daily fractionation.
      * Clinical & PLOS ONE endpoint evaluation: Trajectory RMSE/MAE/r, Specific Growth Rate (SGR),
        Time-to-Doubling (TT2x), Time-to-Tripling (TT3x), Responder (R) vs. Non-Responder (NR) ROC-AUC,
        and EF5+ (hypoxic) vs. EF5- (oxygenated) stratification across single and fractionated regimens.

  - Stage 2: Pre-RT Personalised Growth Anchoring & Imaging/Caliper Modality Ablation
      * Strictly Pre-RT Growth Calibration (`Days 23, 30, 35, 43` / `2011-05-11` to `2011-05-31`):
        estimates tumour-specific baseline proliferation rate `kp_max_i` with ZERO leakage of post-RT data.
      * Baseline Hypoxia ROI & Scanner Normalization Ablation:
        (a) Full-Bed CT/Combo ROI T/M (PLOS ONE Sheet2 standard, cutoff 1.5),
        (b) Anatomical CT ROI T/M (cutoff 1.5),
        (c) PET Hotspot Sub-ROI T/M (Technical Report standard, cutoff 2.5),
        (d) Scan-Time-Corrected T/M (normalising post-injection uptake drift to the cohort median 3.45 hr).
      * Caliper Volume Formula Ablation:
        `0.5 * X * Y^2` (PLOS ONE / modified ellipsoid) vs. `pi/6 * X * Y * Z` (3-axis ellipsoid).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.modeling.physics import RadiobiologyParams, po2_to_ef5_tm
from src.modeling.spheroid_rt_model import SpheroidRTModel, SpheroidScenario, SpheroidSimulationResult
from src.modeling.voxel_initializer import initialize_tumour_from_npz
from src.modeling.voxel_rt_model import VoxelRTModel, VoxelSimulationConfig


# Canonical calendar dates in the Second EF5 RT Study (May - July 2011)
BASELINE_CALIPER_DATE = datetime.date(2011, 5, 31)  # Day 43 post-implant (nominal Day -3)
PLOS_END_DATE = datetime.date(2011, 6, 30)          # Day 73 post-implant (nominal Day +27, t = 30d post-baseline)
FINAL_CALIPER_DATE = datetime.date(2011, 7, 26)     # Day 99 post-implant (nominal Day +53, t = 56d post-baseline)


@dataclass
class TumourStudy2Record:
    """Complete pre-RT imaging, schedule, and longitudinal caliper record for one Study 2 tumour."""

    tumour_id: str
    plos_tumour_code: str
    mouse_id: str
    cage_num: int
    mouse_num: int
    side: str                       # "L" or "R"
    treatment: str                  # "0Gy", "1x10Gy", "1x20Gy", "1x40Gy", "2x5Gy", "2x10Gy", "4x5Gy", "4x10Gy"
    total_dose_gy: float
    num_fractions: int
    fraction_dose_gy: float
    imaged: bool
    scan_date: str
    time_to_image_hr: float
    blue_score: int
    khalila_note: str
    survival_status: str
    # Volume & T/M metrics from the different ROI conventions
    fullbed_ct_vol_cm3: float
    fullbed_pet_vol_cm3: float
    plos_combo_tm: float            # Full-bed CT/combo ROI T/M used in PLOS ONE Sheet2 (cutoff ~1.5)
    ct_roi_tm: float                # Anatomical CT ROI T/M from 10 PET and CT Tumor ROI stats.xlsx
    pet_hotspot_tm: float           # PET hotspot sub-ROI T/M used in Technical Report (cutoff ~2.5)
    time_corrected_tm: float        # Scan-time-normalized T/M (corrected to median 3.45 hr post-injection)
    # Baseline (2011-05-31, Day 43) caliper volumes in mm^3 and cm^3
    base_cal_05xy2_mm3: float
    base_cal_ellipsoid_mm3: float
    # Pre-RT caliper observations (Days 23, 30, 35, 43 -> t_rel_base in [-20, -13, -8, 0] days)
    pre_t_days: np.ndarray          # Calendar days relative to 2011-05-31 (<= 0)
    pre_vol_05xy2_mm3: np.ndarray
    pre_vol_ellipsoid_mm3: np.ndarray
    # Post-baseline caliper observations (Days 43..99 -> t_rel_base in [0, 6, 13, 21, 30, 36, 50, 56] days)
    post_t_days: np.ndarray         # Calendar days relative to 2011-05-31 (>= 0)
    post_days_post_inj: np.ndarray  # Study days [43, 49, 56, 64, 73, 79, 84, 99]
    post_days_post_rt_nom: np.ndarray  # Nominal days post-RT [-3, 3, 10, 18, 27, 33, 37, 53]
    post_vol_05xy2_mm3: np.ndarray
    post_vol_ellipsoid_mm3: np.ndarray
    post_rel_05xy2: np.ndarray      # V(t) / V(2011-05-31) using 0.5*X*Y^2
    post_rel_ellipsoid: np.ndarray  # V(t) / V(2011-05-31) using pi/6*X*Y*Z
    # Radiation schedules relative to 2011-05-31 (t = 0) and relative to scan_date (for 3D voxel sim)
    actual_schedule_from_base: List[Tuple[float, float]]
    nominal_schedule_from_base: List[Tuple[float, float]]
    actual_schedule_from_scan: List[Tuple[float, float]]
    nominal_schedule_from_scan: List[Tuple[float, float]]
    scan_offset_from_base_days: float  # (scan_date - 2011-05-31).days (1.0 for Day 1, 2.0 for Day 2)


def build_rt_schedules(
    cage_num: int,
    num_fractions: int,
    fraction_dose_gy: float,
    scan_date_str: str,
) -> Tuple[
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    List[Tuple[float, float]],
    float,
]:
    """
    Constructs exact 2011 calendar radiation schedules and counterfactual nominal 24h daily schedules:
      - Cages 1-3 (1x10, 1x20, 1x40 Gy): Single fraction delivered on 2011-06-02 (2.0 days after 2011-05-31).
      - Cage 4 (0 Gy Control): No radiation.
      - Cages 5-8 (2x5, 2x10, 4x5, 4x10 Gy):
          * Actual 2011 schedule: Fraction #1 on 2011-06-03 (t = 3.0d after 2011-05-31), followed by the
            4-day missing-lead-shield weekend gap, then Fraction #2 on 2011-06-07 (t = 7.0d),
            Fraction #3 on 2011-06-08 (t = 8.0d), Fraction #4 on 2011-06-09 (t = 9.0d).
          * Nominal counterfactual schedule (no missing shield): Fractions on consecutive days
            starting 2011-06-03 (t = 3.0, 4.0, 5.0, 6.0d after 2011-05-31).
    """
    scan_dt = datetime.date.fromisoformat(scan_date_str) if scan_date_str else datetime.date(2011, 6, 1)
    scan_offset = float((scan_dt - BASELINE_CALIPER_DATE).days)

    if num_fractions <= 0 or fraction_dose_gy <= 0.0 or cage_num == 4:
        return [], [], [], [], scan_offset

    if num_fractions == 1:
        # Delivered 2011-06-02 (t = 2.0 days after 2011-05-31)
        act_base = [(2.0, float(fraction_dose_gy))]
        nom_base = [(2.0, float(fraction_dose_gy))]
    elif num_fractions == 2:
        # Actual: June 3 (t=3.0) and June 7 (t=7.0); Nominal: June 3 (t=3.0) and June 4 (t=4.0)
        act_base = [(3.0, float(fraction_dose_gy)), (7.0, float(fraction_dose_gy))]
        nom_base = [(3.0, float(fraction_dose_gy)), (4.0, float(fraction_dose_gy))]
    elif num_fractions == 4:
        # Actual: June 3 (t=3.0), June 7 (t=7.0), June 8 (t=8.0), June 9 (t=9.0)
        act_base = [
            (3.0, float(fraction_dose_gy)),
            (7.0, float(fraction_dose_gy)),
            (8.0, float(fraction_dose_gy)),
            (9.0, float(fraction_dose_gy)),
        ]
        nom_base = [
            (3.0, float(fraction_dose_gy)),
            (4.0, float(fraction_dose_gy)),
            (5.0, float(fraction_dose_gy)),
            (6.0, float(fraction_dose_gy)),
        ]
    else:
        raise ValueError(f"Unexpected num_fractions={num_fractions}")

    act_scan = [(max(t - scan_offset, 0.0), d) for t, d in act_base]
    nom_scan = [(max(t - scan_offset, 0.0), d) for t, d in nom_base]
    return act_base, nom_base, act_scan, nom_scan, scan_offset


def load_study2_cohort(root: Path) -> List[TumourStudy2Record]:
    """Loads and joins all 80 Study 2 tumours with their 11 caliper dates and pre-RT PET/CT metrics."""
    tables_dir = root / "results" / "second_ef5_rt_study"
    master_df = pd.read_csv(tables_dir / "tumour_master_wide.csv")
    cal_df = pd.read_csv(tables_dir / "caliper_growth_curves.csv")

    # Compute scan-time-corrected T/M (Technical Report Slide 15 notes slight positive drift of T/M
    # with injection-to-scan waiting time `time_to_image_hr`, slope ~0.18 hr^-1 around median 3.45 hr)
    valid_times = master_df["time_to_image_hr"].dropna()
    med_time_hr = float(valid_times.median()) if len(valid_times) > 0 else 3.45
    # Fit empirical linear trend of plos_sheet2_combo_tm vs time_to_image_hr across imaged tumours
    imaged_mask = master_df["imaged"].astype(bool) & master_df["plos_sheet2_combo_tm"].notna() & master_df["time_to_image_hr"].notna()
    if imaged_mask.sum() > 5:
        t_vals = master_df.loc[imaged_mask, "time_to_image_hr"].to_numpy(dtype=float)
        tm_vals = master_df.loc[imaged_mask, "plos_sheet2_combo_tm"].to_numpy(dtype=float)
        slope, _ = np.polyfit(t_vals - med_time_hr, tm_vals, 1)
        # Keep gentle physiological correction bounded
        drift_slope = float(np.clip(slope, -0.35, 0.35))
    else:
        drift_slope = 0.15

    # Cohort median T/M fallback for the single unimaged mouse (c4_M1R_L, c4_M1R_R)
    med_combo_tm = float(master_df["plos_sheet2_combo_tm"].dropna().median())
    med_ct_tm = float(master_df["report_ct_roi_tm"].dropna().median())
    med_hot_tm = float(master_df["report_pet_hotspot_tm"].dropna().median())
    med_ct_vol = float(master_df["fullbed_ct_vol_cm3"].dropna().median())

    records: List[TumourStudy2Record] = []
    for _, row in master_df.iterrows():
        tid = str(row["tumour_id"])
        imaged = bool(row["imaged"])
        scan_date = str(row["scan_date"]) if pd.notna(row["scan_date"]) and str(row["scan_date"]).strip() else "2011-06-01"
        t_img_hr = float(row["time_to_image_hr"]) if pd.notna(row["time_to_image_hr"]) else med_time_hr

        combo_tm = float(row["plos_sheet2_combo_tm"]) if pd.notna(row["plos_sheet2_combo_tm"]) else (
            float(row["fullbed_tm_mean"]) if pd.notna(row["fullbed_tm_mean"]) else med_combo_tm
        )
        ct_tm = float(row["report_ct_roi_tm"]) if pd.notna(row["report_ct_roi_tm"]) else combo_tm
        hot_tm = float(row["report_pet_hotspot_tm"]) if pd.notna(row["report_pet_hotspot_tm"]) else max(combo_tm * 1.45, med_hot_tm)
        time_corr_tm = float(np.clip(combo_tm - 0.5 * drift_slope * (t_img_hr - med_time_hr), 0.75, 4.50))

        ct_vol = float(row["fullbed_ct_vol_cm3"]) if pd.notna(row["fullbed_ct_vol_cm3"]) else med_ct_vol
        pet_vol = float(row["fullbed_pet_vol_cm3"]) if pd.notna(row["fullbed_pet_vol_cm3"]) else ct_vol

        sub_cal = cal_df[cal_df["tumour_id"] == tid].sort_values("days_post_injection")
        cal_dates = [datetime.date.fromisoformat(str(d)) for d in sub_cal["date"]]
        t_rel_base = np.array([(d - BASELINE_CALIPER_DATE).days for d in cal_dates], dtype=float)
        d_inj = sub_cal["days_post_injection"].to_numpy(dtype=int)
        d_rt_nom = sub_cal["days_post_rt_nominal"].to_numpy(dtype=int)
        v_05xy2 = sub_cal["vol_05xy2_mm3"].to_numpy(dtype=float)
        v_ellip = sub_cal["vol_ellipsoid_xyz_mm3"].to_numpy(dtype=float)

        # Baseline Day 43 (2011-05-31, t_rel_base == 0)
        b_idx = np.where(d_inj == 43)[0]
        base_05 = float(v_05xy2[b_idx[0]]) if len(b_idx) > 0 and v_05xy2[b_idx[0]] > 0 else float(ct_vol * 1000.0)
        base_el = float(v_ellip[b_idx[0]]) if len(b_idx) > 0 and v_ellip[b_idx[0]] > 0 else float(ct_vol * 1000.0 * (np.pi / 6.0) / 0.5)

        # Pre-RT valid points (d_inj <= 43 and volume > 0)
        pre_mask = (d_inj <= 43) & (v_05xy2 > 0.0)
        pre_t = t_rel_base[pre_mask]
        pre_v05 = v_05xy2[pre_mask]
        pre_vel = np.where(v_ellip[pre_mask] > 0.0, v_ellip[pre_mask], pre_v05 * (np.pi / 6.0) / 0.5)

        # Post-baseline valid points (d_inj >= 43 and volume > 0)
        # Note: in the spreadsheets, 0.0 after Day 73/79/84 indicates mouse sacrifice or tumour excision,
        # except for Cage 3 (1x40 Gy) where tumours shrank to a tiny residual scab/floor before sacrifice.
        post_mask = (d_inj >= 43) & (v_05xy2 > 0.0)
        post_t = t_rel_base[post_mask]
        post_dinj = d_inj[post_mask]
        post_drt = d_rt_nom[post_mask]
        post_v05 = v_05xy2[post_mask]
        post_vel = np.where(v_ellip[post_mask] > 0.0, v_ellip[post_mask], post_v05 * (np.pi / 6.0) / 0.5)

        act_base, nom_base, act_scan, nom_scan, scan_off = build_rt_schedules(
            int(row["cage_num"]),
            int(row["num_fractions"]),
            float(row["fraction_dose_gy"]),
            scan_date,
        )

        records.append(
            TumourStudy2Record(
                tumour_id=tid,
                plos_tumour_code=str(row["plos_tumour_code"]),
                mouse_id=str(row["mouse_id"]),
                cage_num=int(row["cage_num"]),
                mouse_num=int(row["mouse_num"]),
                side=str(row["side"]),
                treatment=str(row["treatment"]),
                total_dose_gy=float(row["total_dose_gy"]),
                num_fractions=int(row["num_fractions"]),
                fraction_dose_gy=float(row["fraction_dose_gy"]),
                imaged=imaged,
                scan_date=scan_date,
                time_to_image_hr=t_img_hr,
                blue_score=int(row["blue_score"]) if pd.notna(row["blue_score"]) else 0,
                khalila_note=str(row["khalila_note_2011_07_20"]) if pd.notna(row["khalila_note_2011_07_20"]) else "",
                survival_status=str(row["survival_status"]) if pd.notna(row["survival_status"]) else "",
                fullbed_ct_vol_cm3=ct_vol,
                fullbed_pet_vol_cm3=pet_vol,
                plos_combo_tm=combo_tm,
                ct_roi_tm=ct_tm,
                pet_hotspot_tm=hot_tm,
                time_corrected_tm=time_corr_tm,
                base_cal_05xy2_mm3=base_05,
                base_cal_ellipsoid_mm3=base_el,
                pre_t_days=pre_t,
                pre_vol_05xy2_mm3=pre_v05,
                pre_vol_ellipsoid_mm3=pre_vel,
                post_t_days=post_t,
                post_days_post_inj=post_dinj,
                post_days_post_rt_nom=post_drt,
                post_vol_05xy2_mm3=post_v05,
                post_vol_ellipsoid_mm3=post_vel,
                post_rel_05xy2=post_v05 / max(base_05, 1e-6),
                post_rel_ellipsoid=post_vel / max(base_el, 1e-6),
                actual_schedule_from_base=act_base,
                nominal_schedule_from_base=nom_base,
                actual_schedule_from_scan=act_scan,
                nominal_schedule_from_scan=nom_scan,
                scan_offset_from_base_days=scan_off,
            )
        )
    return records


# ---------------------------------------------------------------------------
# 1D Spheroid Tumour-Specific Initialiser & Runner
# ---------------------------------------------------------------------------
def anchor_1d_spheroid_scenario(
    rec: TumourStudy2Record,
    params: RadiobiologyParams,
    tm_mode: str = "plos_combo_tm",
    vol_mode: str = "cal_05xy2",
    schedule_mode: str = "actual",
    t_end_days: float = 56.0,
    dt_days: float = 0.2,
    n_shells: int = 60,
) -> SpheroidScenario:
    """
    Creates a tumour-specific 1D SpheroidScenario anchored to the tumour's measured pre-RT volume
    and pre-RT EF5 PET T/M (analogous to the 3D voxel initialiser).
    """
    if vol_mode == "ct_roi":
        v0_cm3 = rec.fullbed_ct_vol_cm3
    elif vol_mode == "cal_ellipsoid":
        v0_cm3 = rec.base_cal_ellipsoid_mm3 / 1000.0
    else:
        v0_cm3 = rec.base_cal_05xy2_mm3 / 1000.0
    v0_cm3 = float(np.clip(v0_cm3, 0.03, 1.50))
    R0_mm = float(((3.0 * v0_cm3 * 1000.0) / (4.0 * np.pi)) ** (1.0 / 3.0))

    if tm_mode == "ct_roi_tm":
        obs_tm = rec.ct_roi_tm
    elif tm_mode == "pet_hotspot_tm":
        # PET hotspot T/M measures the hypoxic peak (~2.5 cutoff vs 1.5 whole-tumour cutoff);
        # map onto whole-tumour equivalent scale centered at 1.5/2.5
        obs_tm = 1.0 + (rec.pet_hotspot_tm - 1.0) * (1.55 / 2.50)
    elif tm_mode == "time_corrected_tm":
        obs_tm = rec.time_corrected_tm
    else:
        obs_tm = rec.plos_combo_tm

    # Map observed T/M smoothly to (h_vasc_scale, core_vasc_floor, gamma_scale)
    # At T/M <= 1.15 -> well-perfused normoxic (h_vasc_scale ~ 3.6, core_vasc_floor ~ 0.85, gamma_scale ~ 0.85)
    # At T/M ~ 1.55 -> intermediate cutoff (h_vasc_scale ~ 2.0, core_vasc_floor ~ 0.68, gamma_scale ~ 1.00)
    # At T/M >= 2.30 -> hypoxic EF5+ (h_vasc_scale ~ 0.95, core_vasc_floor ~ 0.52, gamma_scale ~ 1.22)
    # Also compensate slightly for R0_mm so a small tumour with high T/M still achieves its measured hypoxia
    r_fac = (R0_mm / 4.0) ** 1.35
    tm_norm = float(np.clip((obs_tm - 1.10) / (2.40 - 1.10), 0.0, 1.35))
    h_vasc_scale = float(np.clip((3.60 * np.exp(-1.15 * tm_norm)) * r_fac, 0.45, 5.50))
    core_vasc_floor = float(np.clip(0.86 - 0.28 * tm_norm, 0.35, 0.90))
    gamma_scale = float(np.clip(0.84 + 0.32 * tm_norm, 0.75, 1.35))

    sched = rec.actual_schedule_from_base if schedule_mode == "actual" else rec.nominal_schedule_from_base

    return SpheroidScenario(
        name=f"{rec.tumour_id}_{rec.treatment}_{tm_mode}",
        R0_mm=R0_mm,
        p_bdry_mmHg=params.p_host_bdry,
        h_vasc_scale=h_vasc_scale,
        core_vasc_floor=core_vasc_floor,
        gamma_scale=gamma_scale,
        dose_schedule=list(sched),
        t_pre_eq_days=10.0,
        t_end_days=t_end_days,
        dt_days=dt_days,
        n_shells=n_shells,
    )


def fit_prert_tumour_kp_max(
    rec: TumourStudy2Record,
    base_params: RadiobiologyParams,
    vol_mode: str = "cal_05xy2",
    prior_weight: float = 0.25,
) -> float:
    """
    Stage 2: Estimates tumour-specific baseline proliferation rate `kp_max_i` strictly from the
    4 pre-RT caliper measurements (Days 23, 30, 35, 43 -> t in [-20, -13, -8, 0] days).
    Uses zero post-RT data (strictly blind to post-treatment response).

    Notably, by Day 43 (`2011-05-31`), tumours have already decelerated from their early
    Day 23->35 exponential phase into their late pre-RT (Day 35->43) regime. We weight the
    recent pre-RT velocity (Day 30/35 -> Day 43) and the tumour's Day 43 baseline size
    so that fast-decelerating large baseline tumours are not artificially extrapolated at
    their Day 23 infant growth speed.
    """
    t_pre = rec.pre_t_days
    v_pre = rec.pre_vol_ellipsoid_mm3 if vol_mode == "cal_ellipsoid" else rec.pre_vol_05xy2_mm3
    if len(t_pre) < 2 or v_pre[-1] <= 0:
        return base_params.kp_max

    log_v = np.log(np.maximum(v_pre, 1.0))
    # Overall pre-RT log-linear slope (Days 23..43)
    sgr_all, _ = np.polyfit(t_pre, log_v, 1)
    # Late pre-RT slope (last 2-3 pre-RT points leading into Day 43 baseline)
    if len(t_pre) >= 3 and (t_pre[-1] - t_pre[-2]) >= 4.0:
        sgr_late = float((log_v[-1] - log_v[-2]) / max(t_pre[-1] - t_pre[-2], 1.0))
        sgr_pre_eff = 0.45 * float(sgr_all) + 0.55 * sgr_late
    else:
        sgr_pre_eff = float(sgr_all)
    sgr_pre_eff = float(np.clip(sgr_pre_eff, 0.010, 0.140))

    # Cohort reference pre-RT SGR at Day 43 transition is ~0.072 day^-1 for kp_max = 0.115..0.130
    ref_sgr_pre = 0.078
    # Larger tumours at Day 43 (v0 > 250 mm^3) experience stronger subcutaneous solid stress going forward
    v0_mm3 = float(v_pre[-1])
    size_damp = (240.0 / max(v0_mm3, 120.0)) ** 0.22
    kp_est = base_params.kp_max * ((sgr_pre_eff / ref_sgr_pre) ** 0.55) * size_damp

    n_pts = len(t_pre)
    w_prior = prior_weight + (0.20 if n_pts < 4 else 0.0)
    kp_shrunk = (1.0 - w_prior) * kp_est + w_prior * (0.90 * base_params.kp_max)
    return float(np.clip(kp_shrunk, 0.065, 0.185))


def simulate_1d_tumour(
    rec: TumourStudy2Record,
    params: RadiobiologyParams,
    tm_mode: str = "plos_combo_tm",
    vol_mode: str = "cal_05xy2",
    schedule_mode: str = "actual",
    t_end_days: float = 56.0,
    dt_days: float = 0.2,
    n_shells: int = 60,
) -> Tuple[SpheroidSimulationResult, np.ndarray]:
    """Runs the 1D Spheroid model for a single Study 2 tumour and interpolates at post-RT caliper dates."""
    scen = anchor_1d_spheroid_scenario(
        rec,
        params,
        tm_mode=tm_mode,
        vol_mode=vol_mode,
        schedule_mode=schedule_mode,
        t_end_days=t_end_days,
        dt_days=dt_days,
        n_shells=n_shells,
    )
    model = SpheroidRTModel(params)
    res = model.run(scen)
    pred_rel_at_obs = np.interp(rec.post_t_days, res.t_days, res.rel_volume)
    return res, pred_rel_at_obs


# ---------------------------------------------------------------------------
# 3D Voxel Continuum Parallel Worker
# ---------------------------------------------------------------------------
def _run_single_3d_voxel_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function for parallel 3D Voxel PDE simulation of a single imaged tumour."""
    npz_path = Path(task["npz_path"])
    side = str(task["side"])
    params: RadiobiologyParams = task["params"]
    tm_scale = float(task.get("tm_scale", 1.0))
    dose_schedule: List[Tuple[float, float]] = task["dose_schedule"]
    scan_offset_days = float(task["scan_offset_days"])
    post_t_days = np.asarray(task["post_t_days"], dtype=float)
    t_end_from_base = float(task.get("t_end_from_base", 56.0))
    dx_mm = float(task.get("dx_mm", 0.62))
    dt_days = float(task.get("dt_days", 0.5))

    init_state = initialize_tumour_from_npz(
        npz_path,
        side=side,
        params=params,
        dx_mm=dx_mm,
        pad_mm=5.5,
        tm_scale=tm_scale,
    )
    t_sim_end = max(t_end_from_base - scan_offset_days, 35.0)
    cfg = VoxelSimulationConfig(
        t_end_days=t_sim_end,
        dt_days=dt_days,
        dose_schedule=dose_schedule,
        snapshot_days=[0.0, 12.0, 29.0],
    )
    vmod = VoxelRTModel(params)
    res = vmod.run(init_state, cfg)

    # Convert simulation time (relative to scan_date) back to calendar days relative to 2011-05-31
    t_from_base = res.t_sim_days + scan_offset_days
    # Extrapolate backward 1-2 days from scan_date (June 1/2) to baseline caliper date (May 31)
    # using pre-RT growth rate so V(2011-05-31) = 1.0
    sgr_pre = max((params.kp_max * 0.55 - params.ka_basal), 0.03)
    v_at_base_cm3 = float(res.volume_cm3[0]) * np.exp(-sgr_pre * scan_offset_days)
    rel_vol_curve = res.volume_cm3 / max(v_at_base_cm3, 1e-6)

    t_full = np.concatenate([[0.0], t_from_base])
    rel_full = np.concatenate([[1.0], rel_vol_curve])
    pred_rel_at_obs = np.interp(post_t_days, t_full, rel_full)

    # Also interpolate uniform daily grid t = 0..56 for cohort curves
    t_grid = np.arange(0.0, t_end_from_base + 0.5, 1.0)
    rel_grid = np.interp(t_grid, t_full, rel_full)
    vfrac_grid = np.interp(t_grid, t_from_base, res.viable_frac)
    nfrac_grid = np.interp(t_grid, t_from_base, res.necrotic_frac)
    po2_grid = np.interp(t_grid, t_from_base, res.mean_po2_mmHg)
    ef5_grid = np.interp(t_grid, t_from_base, res.mean_ef5_tm)

    return {
        "tumour_id": task["tumour_id"],
        "obs_vol_cm3": init_state.obs_volume_cm3,
        "obs_tm_mean": init_state.obs_tm_mean,
        "obs_tm_p95": init_state.obs_tm_p95,
        "sim_tm0_mean": float(res.mean_ef5_tm[0]),
        "pred_rel_at_obs": pred_rel_at_obs,
        "t_grid": t_grid,
        "rel_grid": rel_grid,
        "vfrac_grid": vfrac_grid,
        "nfrac_grid": nfrac_grid,
        "po2_grid": po2_grid,
        "ef5_grid": ef5_grid,
    }


def run_3d_voxel_cohort(
    root: Path,
    records: Sequence[TumourStudy2Record],
    params_by_tumour: Dict[str, RadiobiologyParams],
    schedule_mode: str = "actual",
    tm_mode: str = "plos_combo_tm",
    dx_mm: float = 0.62,
    dt_days: float = 0.5,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Runs the 3D Voxel PDE simulator across all 78 imaged Study 2 tumours in parallel."""
    bundles_dir = root / "outputs" / "second_ef5_rt_study" / "voxel_bundles"
    tasks: List[Dict[str, Any]] = []
    for rec in records:
        if not rec.imaged:
            continue
        npz_path = bundles_dir / rec.mouse_id / f"{rec.mouse_id}_pre_rt.npz"
        if not npz_path.exists():
            continue
        prm = params_by_tumour.get(rec.tumour_id, RadiobiologyParams())
        sched = rec.actual_schedule_from_scan if schedule_mode == "actual" else rec.nominal_schedule_from_scan

        if tm_mode == "time_corrected_tm" and rec.plos_combo_tm > 0:
            tm_scale = rec.time_corrected_tm / rec.plos_combo_tm
        elif tm_mode == "pet_hotspot_tm" and rec.plos_combo_tm > 0:
            obs_equiv = 1.0 + (rec.pet_hotspot_tm - 1.0) * (1.55 / 2.50)
            tm_scale = obs_equiv / rec.plos_combo_tm
        else:
            tm_scale = 1.0

        tasks.append(
            {
                "tumour_id": rec.tumour_id,
                "npz_path": str(npz_path),
                "side": rec.side.lower(),
                "params": prm,
                "tm_scale": tm_scale,
                "dose_schedule": list(sched),
                "scan_offset_days": rec.scan_offset_from_base_days,
                "post_t_days": rec.post_t_days,
                "t_end_from_base": 56.0,
                "dx_mm": dx_mm,
                "dt_days": dt_days,
            }
        )

    n_workers = max_workers or min(8, max(1, (os.cpu_count() or 4) - 1))
    results_map: Dict[str, Dict[str, Any]] = {}
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for out in ex.map(_run_single_3d_voxel_worker, tasks):
                results_map[out["tumour_id"]] = out
    else:
        for t in tasks:
            out = _run_single_3d_voxel_worker(t)
            results_map[out["tumour_id"]] = out
    return results_map


# ---------------------------------------------------------------------------
# Clinical & Trajectory Evaluation Metrics
# ---------------------------------------------------------------------------
def compute_clinical_endpoints(
    t_days: np.ndarray,
    rel_vol: np.ndarray,
    plos_window_max_day: float = 30.0,
) -> Dict[str, float]:
    """
    Computes PLOS ONE clinical endpoints from a relative volume trajectory V(t)/V(0):
      1. Specific Growth Rate (SGR, day^-1) inside the PLOS ONE 30-day window (Days 43..73)
      2. Relative volume at the primary PLOS ONE endpoint (t = 30d / Day 73 / +27d post-RT)
      3. Time to 2x baseline volume (TT2x, days; right-censored at max observed time if not reached)
      4. Time to 3x baseline volume (TT3x, days)
    """
    w_mask = t_days <= (plos_window_max_day + 1.0)
    t_w = t_days[w_mask]
    v_w = np.maximum(rel_vol[w_mask], 0.05)

    if len(t_w) >= 2 and (t_w[-1] - t_w[0]) >= 3.0:
        sgr_plos, _ = np.polyfit(t_w, np.log(v_w), 1)
    else:
        sgr_plos = 0.0

    v_d30 = float(np.interp(30.0, t_days, rel_vol)) if t_days[-1] >= 18.0 else float(rel_vol[-1])

    # Time to 2x and 3x baseline volume via linear crossing interpolation
    def _crossing_time(thresh: float) -> Tuple[float, bool]:
        idx = np.where(rel_vol >= thresh)[0]
        if len(idx) == 0:
            return float(t_days[-1]), False
        k = int(idx[0])
        if k == 0:
            return 0.0, True
        v0, v1 = float(rel_vol[k - 1]), float(rel_vol[k])
        t0, t1 = float(t_days[k - 1]), float(t_days[k])
        frac = (thresh - v0) / max(v1 - v0, 1e-9)
        return float(t0 + np.clip(frac, 0.0, 1.0) * (t1 - t0)), True

    tt2x, reached_2x = _crossing_time(2.0)
    tt3x, reached_3x = _crossing_time(3.0)

    return {
        "sgr_plos_day_inv": float(sgr_plos),
        "rel_vol_d30": float(v_d30),
        "tt2x_days": float(tt2x),
        "reached_2x": bool(reached_2x),
        "tt3x_days": float(tt3x),
        "reached_3x": bool(reached_3x),
    }


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact Wilcoxon-Mann-Whitney ROC-AUC without external sklearn dependency."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diffs = pos[:, None] - neg[None, :]
    auc = float(np.mean(diffs > 0) + 0.5 * np.mean(diffs == 0))
    return auc


def evaluate_cohort_predictions(
    records: Sequence[TumourStudy2Record],
    preds_by_tumour: Dict[str, np.ndarray],
    vol_mode: str = "cal_05xy2",
    stage_label: str = "Stage1_1D_Blind",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates per-tumour and per-regimen trajectory accuracy and PLOS ONE clinical endpoints.
    """
    point_rows: List[Dict[str, Any]] = []
    tumour_rows: List[Dict[str, Any]] = []

    for rec in records:
        if rec.tumour_id not in preds_by_tumour:
            continue
        pred_rel = np.asarray(preds_by_tumour[rec.tumour_id], dtype=float)
        obs_rel = rec.post_rel_ellipsoid if vol_mode == "cal_ellipsoid" else rec.post_rel_05xy2
        t_post = rec.post_t_days
        d_inj = rec.post_days_post_inj

        for k in range(len(t_post)):
            if t_post[k] == 0.0:
                continue  # t=0 is normalized baseline 1.0 by definition
            in_plos = bool(d_inj[k] <= 73)
            o_val = float(obs_rel[k])
            p_val = float(pred_rel[k])
            point_rows.append(
                {
                    "stage": stage_label,
                    "tumour_id": rec.tumour_id,
                    "mouse_id": rec.mouse_id,
                    "cage_num": rec.cage_num,
                    "treatment": rec.treatment,
                    "total_dose_gy": rec.total_dose_gy,
                    "num_fractions": rec.num_fractions,
                    "t_days_from_base": float(t_post[k]),
                    "days_post_injection": int(d_inj[k]),
                    "in_plos_window": in_plos,
                    "pre_rt_tm": rec.plos_combo_tm,
                    "hypoxic_stratum": "EF5+ (Hypoxic)" if rec.plos_combo_tm > 1.50 else "EF5- (Oxygenated)",
                    "obs_rel_vol": o_val,
                    "pred_rel_vol": p_val,
                    "err_rel_vol": p_val - o_val,
                    "obs_log2_rel": float(np.log2(max(o_val, 0.05))),
                    "pred_log2_rel": float(np.log2(max(p_val, 0.05))),
                    "err_log2_rel": float(np.log2(max(p_val, 0.05)) - np.log2(max(o_val, 0.05))),
                }
            )

        obs_ep = compute_clinical_endpoints(t_post, obs_rel)
        pred_ep = compute_clinical_endpoints(t_post, pred_rel)

        post_only = t_post > 0
        plos_only = (t_post > 0) & (d_inj <= 73)
        rmse_all = float(np.sqrt(np.mean((pred_rel[post_only] - obs_rel[post_only]) ** 2))) if np.any(post_only) else 0.0
        rmse_plos = float(np.sqrt(np.mean((pred_rel[plos_only] - obs_rel[plos_only]) ** 2))) if np.any(plos_only) else 0.0
        log2_rmse_plos = (
            float(
                np.sqrt(
                    np.mean(
                        (np.log2(np.maximum(pred_rel[plos_only], 0.05)) - np.log2(np.maximum(obs_rel[plos_only], 0.05)))
                        ** 2
                    )
                )
            )
            if np.any(plos_only)
            else 0.0
        )

        # PLOS ONE responder definition: tumour fails to double by Day 73 (+27d post-RT) or SGR <= 0.020 day^-1
        obs_responder = int((obs_ep["rel_vol_d30"] < 2.0) or (obs_ep["sgr_plos_day_inv"] < 0.020))
        pred_responder_score = -pred_ep["sgr_plos_day_inv"]
        pred_responder = int((pred_ep["rel_vol_d30"] < 2.0) or (pred_ep["sgr_plos_day_inv"] < 0.020))

        tumour_rows.append(
            {
                "stage": stage_label,
                "tumour_id": rec.tumour_id,
                "plos_tumour_code": rec.plos_tumour_code,
                "mouse_id": rec.mouse_id,
                "cage_num": rec.cage_num,
                "treatment": rec.treatment,
                "total_dose_gy": rec.total_dose_gy,
                "num_fractions": rec.num_fractions,
                "imaged": rec.imaged,
                "pre_rt_tm": rec.plos_combo_tm,
                "pet_hotspot_tm": rec.pet_hotspot_tm,
                "hypoxic_stratum": "EF5+ (Hypoxic)" if rec.plos_combo_tm > 1.50 else "EF5- (Oxygenated)",
                "blue_score": rec.blue_score,
                "khalila_note": rec.khalila_note,
                "rmse_rel_plos": round(rmse_plos, 4),
                "rmse_rel_full11": round(rmse_all, 4),
                "rmse_log2_plos": round(log2_rmse_plos, 4),
                "obs_sgr_plos": round(obs_ep["sgr_plos_day_inv"], 5),
                "pred_sgr_plos": round(pred_ep["sgr_plos_day_inv"], 5),
                "obs_rel_vol_d30": round(obs_ep["rel_vol_d30"], 4),
                "pred_rel_vol_d30": round(pred_ep["rel_vol_d30"], 4),
                "obs_tt2x_days": round(obs_ep["tt2x_days"], 2),
                "pred_tt2x_days": round(pred_ep["tt2x_days"], 2),
                "obs_reached_2x": obs_ep["reached_2x"],
                "pred_reached_2x": pred_ep["reached_2x"],
                "obs_responder": obs_responder,
                "pred_responder": pred_responder,
                "pred_responder_score": round(pred_responder_score, 5),
            }
        )

    pts_df = pd.DataFrame(point_rows)
    tum_df = pd.DataFrame(tumour_rows)

    plos_pts = pts_df[pts_df["in_plos_window"]]
    overall_r_plos = float(np.corrcoef(plos_pts["obs_rel_vol"], plos_pts["pred_rel_vol"])[0, 1]) if len(plos_pts) > 2 else 0.0
    overall_r_log2_plos = float(np.corrcoef(plos_pts["obs_log2_rel"], plos_pts["pred_log2_rel"])[0, 1]) if len(plos_pts) > 2 else 0.0
    overall_r_full = float(np.corrcoef(pts_df["obs_rel_vol"], pts_df["pred_rel_vol"])[0, 1]) if len(pts_df) > 2 else 0.0
    overall_r_log2_full = float(np.corrcoef(pts_df["obs_log2_rel"], pts_df["pred_log2_rel"])[0, 1]) if len(pts_df) > 2 else 0.0

    sgr_r = float(np.corrcoef(tum_df["obs_sgr_plos"], tum_df["pred_sgr_plos"])[0, 1]) if len(tum_df) > 2 else 0.0
    tt2x_r = float(np.corrcoef(tum_df["obs_tt2x_days"], tum_df["pred_tt2x_days"])[0, 1]) if len(tum_df) > 2 else 0.0
    resp_acc = float(np.mean(tum_df["obs_responder"] == tum_df["pred_responder"]))
    resp_auc = _roc_auc_binary(
        tum_df["obs_responder"].to_numpy(dtype=int),
        tum_df["pred_responder_score"].to_numpy(dtype=float),
    )

    summary = {
        "stage": stage_label,
        "n_tumours": len(tum_df),
        "n_post_observations_full11": len(pts_df),
        "n_post_observations_plos": len(plos_pts),
        "rmse_rel_plos": round(float(np.sqrt(np.mean(plos_pts["err_rel_vol"] ** 2))), 4),
        "mae_rel_plos": round(float(np.mean(np.abs(plos_pts["err_rel_vol"]))), 4),
        "rmse_log2_plos": round(float(np.sqrt(np.mean(plos_pts["err_log2_rel"] ** 2))), 4),
        "pearson_r_rel_plos": round(overall_r_plos, 4),
        "pearson_r_log2_plos": round(overall_r_log2_plos, 4),
        "rmse_rel_full11": round(float(np.sqrt(np.mean(pts_df["err_rel_vol"] ** 2))), 4),
        "mae_rel_full11": round(float(np.mean(np.abs(pts_df["err_rel_vol"]))), 4),
        "rmse_log2_full11": round(float(np.sqrt(np.mean(pts_df["err_log2_rel"] ** 2))), 4),
        "pearson_r_rel_full11": round(overall_r_full, 4),
        "pearson_r_log2_full11": round(overall_r_log2_full, 4),
        "sgr_pearson_r": round(sgr_r, 4),
        "tt2x_pearson_r": round(tt2x_r, 4),
        "responder_accuracy": round(resp_acc, 4),
        "responder_roc_auc": round(resp_auc, 4),
    }
    return pts_df, tum_df, summary


# ---------------------------------------------------------------------------
# End-to-End Runner for Stages 1 & 2
# ---------------------------------------------------------------------------
def run_stages_1_and_2(root: Optional[Path] = None, run_3d: bool = True) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    out_res_dir = root / "results" / "study2_validation"
    out_res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("SECOND EF5 RT STUDY (40 MICE / 80 TUMOURS): STAGES 1 & 2 VALIDATION")
    print("=" * 78)

    records = load_study2_cohort(root)
    base_params = RadiobiologyParams()  # Frozen v1.0-study1-model parameters

    # -----------------------------------------------------------------------
    # 1. Stage 1A: 1D Spheroid Blind Out-of-Box Validation (Frozen v1.0)
    # -----------------------------------------------------------------------
    print("-> Running Stage 1A: 1D Refined Spheroid Blind Out-of-Box Validation (80 tumours)...")
    s1_1d_preds: Dict[str, np.ndarray] = {}
    s1_1d_curves: Dict[str, SpheroidSimulationResult] = {}
    for rec in records:
        res, pred_obs = simulate_1d_tumour(rec, base_params, tm_mode="plos_combo_tm", schedule_mode="actual")
        s1_1d_preds[rec.tumour_id] = pred_obs
        s1_1d_curves[rec.tumour_id] = res

    pts_s1_1d, tum_s1_1d, sum_s1_1d = evaluate_cohort_predictions(
        records, s1_1d_preds, stage_label="Stage1_1D_Blind_Frozen"
    )
    print(
        f"   [Stage 1A 1D Blind] PLOS Window: log2-RMSE={sum_s1_1d['rmse_log2_plos']:.3f}, "
        f"r={sum_s1_1d['pearson_r_log2_plos']:.3f}, SGR r={sum_s1_1d['sgr_pearson_r']:.3f}, "
        f"Responder AUC={sum_s1_1d['responder_roc_auc']:.3f}"
    )

    # -----------------------------------------------------------------------
    # 2. Stage 1B: 3D Voxel Continuum Blind Out-of-Box Validation (78 imaged tumours)
    # -----------------------------------------------------------------------
    s1_3d_results: Dict[str, Dict[str, Any]] = {}
    if run_3d:
        print("-> Running Stage 1B: 3D Image-Initialised Voxel PDE Blind Validation (78 imaged tumours)...")
        frozen_map = {rec.tumour_id: base_params for rec in records}
        s1_3d_results = run_3d_voxel_cohort(root, records, frozen_map, schedule_mode="actual")
        s1_3d_preds = {tid: d["pred_rel_at_obs"] for tid, d in s1_3d_results.items()}
        pts_s1_3d, tum_s1_3d, sum_s1_3d = evaluate_cohort_predictions(
            records, s1_3d_preds, stage_label="Stage1_3D_Voxel_Blind_Frozen"
        )
        print(
            f"   [Stage 1B 3D Blind] PLOS Window: log2-RMSE={sum_s1_3d['rmse_log2_plos']:.3f}, "
            f"r={sum_s1_3d['pearson_r_log2_plos']:.3f}, SGR r={sum_s1_3d['sgr_pearson_r']:.3f}, "
            f"Responder AUC={sum_s1_3d['responder_roc_auc']:.3f}"
        )
    else:
        pts_s1_3d, tum_s1_3d, sum_s1_3d = pd.DataFrame(), pd.DataFrame(), {}

    # -----------------------------------------------------------------------
    # 3. Stage 1C: Counterfactual Missing-Shield Experiment (Actual 4-Day Gap vs Nominal 24h)
    # -----------------------------------------------------------------------
    print("-> Running Stage 1C: Counterfactual Missing-Lead-Shield Schedule Analysis (Cages 5-8)...")
    s1_1d_nom_preds: Dict[str, np.ndarray] = {}
    shield_gap_rows: List[Dict[str, Any]] = []
    for rec in records:
        res_nom, pred_nom = simulate_1d_tumour(rec, base_params, tm_mode="plos_combo_tm", schedule_mode="nominal")
        s1_1d_nom_preds[rec.tumour_id] = pred_nom
        if rec.num_fractions > 1:
            res_act = s1_1d_curves[rec.tumour_id]
            v_act_d30 = float(np.interp(30.0, res_act.t_days, res_act.rel_volume))
            v_nom_d30 = float(np.interp(30.0, res_nom.t_days, res_nom.rel_volume))
            v_act_d13 = float(np.interp(13.0, res_act.t_days, res_act.rel_volume))
            v_nom_d13 = float(np.interp(13.0, res_nom.t_days, res_nom.rel_volume))
            obs_ep = compute_clinical_endpoints(rec.post_t_days, rec.post_rel_05xy2)
            shield_gap_rows.append(
                {
                    "tumour_id": rec.tumour_id,
                    "treatment": rec.treatment,
                    "num_fractions": rec.num_fractions,
                    "fraction_dose_gy": rec.fraction_dose_gy,
                    "pre_rt_tm": rec.plos_combo_tm,
                    "hypoxic_stratum": "EF5+ (Hypoxic)" if rec.plos_combo_tm > 1.50 else "EF5- (Oxygenated)",
                    "obs_rel_vol_d30": round(obs_ep["rel_vol_d30"], 4),
                    "pred_actual_4d_gap_d13": round(v_act_d13, 4),
                    "pred_nominal_24h_d13": round(v_nom_d13, 4),
                    "pred_actual_4d_gap_d30": round(v_act_d30, 4),
                    "pred_nominal_24h_d30": round(v_nom_d30, 4),
                    "gap_volume_penalty_pct_d30": round(100.0 * (v_act_d30 - v_nom_d30) / max(v_nom_d30, 1e-6), 2),
                    "abs_err_actual_d30": round(abs(v_act_d30 - obs_ep["rel_vol_d30"]), 4),
                    "abs_err_nominal_d30": round(abs(v_nom_d30 - obs_ep["rel_vol_d30"]), 4),
                }
            )
    shield_gap_df = pd.DataFrame(shield_gap_rows)
    shield_gap_df.to_csv(out_res_dir / "stage1_counterfactual_shield_gap.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Stage 2A: Pre-RT Growth Anchoring (Days 23..43 only, ZERO post-RT leakage)
    # -----------------------------------------------------------------------
    print("-> Running Stage 2A: Strictly Pre-RT Personalised Growth Anchoring (Days 23..43 only)...")
    prert_params_map: Dict[str, RadiobiologyParams] = {}
    s2_1d_preds: Dict[str, np.ndarray] = {}
    s2_1d_curves: Dict[str, SpheroidSimulationResult] = {}
    kp_rows: List[Dict[str, Any]] = []

    for rec in records:
        kp_i = fit_prert_tumour_kp_max(rec, base_params, vol_mode="cal_05xy2")
        # Scale hypoxic proliferation floor proportionally
        kp_hyp_i = base_params.kp_hyp_floor * (kp_i / base_params.kp_max)
        prm_i = replace(base_params, kp_max=kp_i, kp_hyp_floor=kp_hyp_i)
        prert_params_map[rec.tumour_id] = prm_i

        res_i, pred_i = simulate_1d_tumour(rec, prm_i, tm_mode="plos_combo_tm", schedule_mode="actual")
        s2_1d_preds[rec.tumour_id] = pred_i
        s2_1d_curves[rec.tumour_id] = res_i
        kp_rows.append(
            {
                "tumour_id": rec.tumour_id,
                "mouse_id": rec.mouse_id,
                "treatment": rec.treatment,
                "n_pre_rt_points": len(rec.pre_t_days),
                "pre_rt_vol_d23_mm3": float(rec.pre_vol_05xy2_mm3[0]) if len(rec.pre_vol_05xy2_mm3) > 0 else np.nan,
                "pre_rt_vol_d43_mm3": rec.base_cal_05xy2_mm3,
                "pop_kp_max": base_params.kp_max,
                "prert_fitted_kp_max": round(kp_i, 5),
                "kp_ratio_vs_pop": round(kp_i / base_params.kp_max, 4),
            }
        )
    pd.DataFrame(kp_rows).to_csv(out_res_dir / "stage2_prert_fitted_growth_rates.csv", index=False)

    pts_s2_1d, tum_s2_1d, sum_s2_1d = evaluate_cohort_predictions(
        records, s2_1d_preds, stage_label="Stage2_1D_PreRT_Growth_Anchored"
    )
    print(
        f"   [Stage 2A 1D Pre-RT Growth Anchored] PLOS Window: log2-RMSE={sum_s2_1d['rmse_log2_plos']:.3f}, "
        f"r={sum_s2_1d['pearson_r_log2_plos']:.3f}, SGR r={sum_s2_1d['sgr_pearson_r']:.3f}, "
        f"Responder AUC={sum_s2_1d['responder_roc_auc']:.3f}"
    )

    s2_3d_results: Dict[str, Dict[str, Any]] = {}
    if run_3d:
        print("-> Running Stage 2A (3D): 3D Voxel PDE with Pre-RT Personalised Growth Rates...")
        s2_3d_results = run_3d_voxel_cohort(root, records, prert_params_map, schedule_mode="actual")
        s2_3d_preds = {tid: d["pred_rel_at_obs"] for tid, d in s2_3d_results.items()}
        pts_s2_3d, tum_s2_3d, sum_s2_3d = evaluate_cohort_predictions(
            records, s2_3d_preds, stage_label="Stage2_3D_Voxel_PreRT_Growth_Anchored"
        )
        print(
            f"   [Stage 2A 3D Pre-RT Growth Anchored] PLOS Window: log2-RMSE={sum_s2_3d['rmse_log2_plos']:.3f}, "
            f"r={sum_s2_3d['pearson_r_log2_plos']:.3f}, SGR r={sum_s2_3d['sgr_pearson_r']:.3f}, "
            f"Responder AUC={sum_s2_3d['responder_roc_auc']:.3f}"
        )
    else:
        pts_s2_3d, tum_s2_3d, sum_s2_3d = pd.DataFrame(), pd.DataFrame(), {}

    # -----------------------------------------------------------------------
    # 5. Stage 2B: ROI Convention, Scan-Time Drift & Caliper Formula Ablations
    # -----------------------------------------------------------------------
    print("-> Running Stage 2B: ROI Convention, Scan-Time Normalization & Caliper Formula Ablations...")
    ablation_configs = [
        ("Stage1_1D_FullBed_Combo_TM", base_params, "plos_combo_tm", "cal_05xy2", "actual"),
        ("Stage1_1D_Nominal_NoShieldGap", base_params, "plos_combo_tm", "cal_05xy2", "nominal"),
        ("Stage2_1D_PreRT_Combo_TM", prert_params_map, "plos_combo_tm", "cal_05xy2", "actual"),
        ("Stage2_1D_PreRT_CT_ROI_TM", prert_params_map, "ct_roi_tm", "cal_05xy2", "actual"),
        ("Stage2_1D_PreRT_PET_Hotspot_TM", prert_params_map, "pet_hotspot_tm", "cal_05xy2", "actual"),
        ("Stage2_1D_PreRT_ScanTimeCorr_TM", prert_params_map, "time_corrected_tm", "cal_05xy2", "actual"),
        ("Stage2_1D_PreRT_Ellipsoid_XYZ", prert_params_map, "plos_combo_tm", "cal_ellipsoid", "actual"),
    ]
    ablation_summaries: List[Dict[str, Any]] = [sum_s1_1d]
    if sum_s1_3d:
        ablation_summaries.append(sum_s1_3d)
    ablation_summaries.append(sum_s2_1d)
    if sum_s2_3d:
        ablation_summaries.append(sum_s2_3d)

    for label, prm_src, tm_m, vol_m, sched_m in ablation_configs:
        preds_ab: Dict[str, np.ndarray] = {}
        for rec in records:
            p_use = prm_src[rec.tumour_id] if isinstance(prm_src, dict) else prm_src
            _, pred_v = simulate_1d_tumour(rec, p_use, tm_mode=tm_m, vol_mode=vol_m, schedule_mode=sched_m)
            preds_ab[rec.tumour_id] = pred_v
        _, _, sum_ab = evaluate_cohort_predictions(records, preds_ab, vol_mode=vol_m, stage_label=label)
        ablation_summaries.append(sum_ab)

    ablation_df = pd.DataFrame(ablation_summaries)
    ablation_df.to_csv(out_res_dir / "stage1_stage2_ablation_summary.csv", index=False)

    # Save per-tumour and per-observation tables
    all_tum_dfs = [df for df in [tum_s1_1d, tum_s1_3d, tum_s2_1d, tum_s2_3d] if len(df) > 0]
    all_pts_dfs = [df for df in [pts_s1_1d, pts_s1_3d, pts_s2_1d, pts_s2_3d] if len(df) > 0]
    pd.concat(all_tum_dfs, ignore_index=True).to_csv(out_res_dir / "stage1_stage2_tumour_metrics.csv", index=False)
    pd.concat(all_pts_dfs, ignore_index=True).to_csv(out_res_dir / "stage1_stage2_pointwise_predictions.csv", index=False)

    # Build regimen-level summary table comparing Observed vs Stage 1 (1D & 3D) vs Stage 2 (1D & 3D)
    reg_rows: List[Dict[str, Any]] = []
    for (cnum, trt), grp in tum_s1_1d.groupby(["cage_num", "treatment"]):
        tids = grp["tumour_id"].tolist()
        s2_grp = tum_s2_1d[tum_s2_1d["tumour_id"].isin(tids)]
        s1_3d_grp = tum_s1_3d[tum_s1_3d["tumour_id"].isin(tids)] if len(tum_s1_3d) > 0 else pd.DataFrame()
        s2_3d_grp = tum_s2_3d[tum_s2_3d["tumour_id"].isin(tids)] if len(tum_s2_3d) > 0 else pd.DataFrame()

        reg_rows.append(
            {
                "cage_num": int(cnum),
                "treatment": str(trt),
                "n_tumours": len(grp),
                "mean_pre_rt_tm": round(float(grp["pre_rt_tm"].mean()), 3),
                "obs_mean_rel_vol_d30": round(float(grp["obs_rel_vol_d30"].mean()), 3),
                "obs_sd_rel_vol_d30": round(float(grp["obs_rel_vol_d30"].std(ddof=1)), 3),
                "s1_1d_mean_rel_vol_d30": round(float(grp["pred_rel_vol_d30"].mean()), 3),
                "s1_3d_mean_rel_vol_d30": round(float(s1_3d_grp["pred_rel_vol_d30"].mean()), 3) if len(s1_3d_grp) > 0 else np.nan,
                "s2_1d_mean_rel_vol_d30": round(float(s2_grp["pred_rel_vol_d30"].mean()), 3),
                "s2_3d_mean_rel_vol_d30": round(float(s2_3d_grp["pred_rel_vol_d30"].mean()), 3) if len(s2_3d_grp) > 0 else np.nan,
                "obs_mean_sgr_plos": round(float(grp["obs_sgr_plos"].mean()), 4),
                "s1_1d_mean_sgr_plos": round(float(grp["pred_sgr_plos"].mean()), 4),
                "s2_1d_mean_sgr_plos": round(float(s2_grp["pred_sgr_plos"].mean()), 4),
                "s1_1d_rmse_log2_plos": round(float(grp["rmse_log2_plos"].mean()), 3),
                "s2_1d_rmse_log2_plos": round(float(s2_grp["rmse_log2_plos"].mean()), 3),
                "s1_3d_rmse_log2_plos": round(float(s1_3d_grp["rmse_log2_plos"].mean()), 3) if len(s1_3d_grp) > 0 else np.nan,
                "s2_3d_rmse_log2_plos": round(float(s2_3d_grp["rmse_log2_plos"].mean()), 3) if len(s2_3d_grp) > 0 else np.nan,
            }
        )
    reg_df = pd.DataFrame(reg_rows).sort_values("cage_num")
    reg_df.to_csv(out_res_dir / "stage1_stage2_regimen_summary.csv", index=False)

    return {
        "records": records,
        "base_params": base_params,
        "prert_params_map": prert_params_map,
        "s1_1d_curves": s1_1d_curves,
        "s2_1d_curves": s2_1d_curves,
        "s1_3d_results": s1_3d_results,
        "s2_3d_results": s2_3d_results,
        "shield_gap_df": shield_gap_df,
        "ablation_df": ablation_df,
        "regimen_df": reg_df,
        "tum_s1_1d": tum_s1_1d,
        "tum_s1_3d": tum_s1_3d,
        "tum_s2_1d": tum_s2_1d,
        "tum_s2_3d": tum_s2_3d,
        "pts_s1_1d": pts_s1_1d,
        "pts_s1_3d": pts_s1_3d,
        "pts_s2_1d": pts_s2_1d,
        "pts_s2_3d": pts_s2_3d,
    }


if __name__ == "__main__":
    run_stages_1_and_2()
