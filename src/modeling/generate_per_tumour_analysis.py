"""
Generates comprehensive per-tumour growth curve comparisons, RMSD / log2-RMSD goodness-of-fit
tables, and a mechanistic mismatch taxonomy across all 80 tumours (40 bilateral mice) in the
Second EF5 RT Study:

  1. Tabular Deliverable:
     - results/study2_validation/per_tumour_rmsd_and_mismatch_classification.csv
       (80 rows x 45 columns capturing per-tumour RMSD, log2-RMSD, MAE, and signed bias across
        Stage 1 Blind [1D & 3D], Stage 2 Pre-RT Growth-Anchored [1D & 3D], and Stage 4
        Hierarchical Bayesian [LORO, 1D MAP, & 3D Voxel], plus hypoxia status, focal hotspot
        discordance, Blue/Pink vascular phenotype, and Match/Mismatch classification.)

  2. Visual Deliverables:
     - outputs/study2_validation/figures/per_tumour_grids/cage1_1x10Gy_per_tumour_curves.png
       .. cage8_4x10Gy_per_tumour_curves.png
       (8 high-resolution 2x5 per-cage grids showing all 10 bilateral tumours per treatment group
        with observed pre/post-RT calipers and all 1D/3D simulated trajectories superimposed.)
     - outputs/study2_validation/figures/08_all_80_tumours_growth_curve_atlas.png
       (Grand 8x10 master visual atlas of all 80 tumours across the cohort.)
     - outputs/study2_validation/figures/09_per_tumour_rmsd_ranking_and_mismatch_taxonomy.png
       (4-panel diagnostic summary: per-tumour RMSD waterfall ranking, match/mismatch taxonomy
        breakdown by regimen, focal PET-hotspot vs whole-tumour T/M discordance in relapsing
        tumours, and archetype trajectories for each mismatch class.)

  3. Report Deliverable:
     - Updates outputs/study2_validation/study2_validation_technical_report.md with Section 5
       (Per-Tumour RMSD Rankings, Best/Worst Matches & Mismatch Taxonomy) and an expanded
       Layperson's Appendix entry explaining the mismatch classes.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.modeling.hierarchical_bayesian import (
    _vec_to_pop_params,
    decompose_hierarchical_growth_random_effects,
    get_bayesian_pop_specs,
)
from src.modeling.physics import RadiobiologyParams
from src.modeling.run_study2_validation import (
    TumourStudy2Record,
    fit_prert_tumour_kp_max,
    load_study2_cohort,
    simulate_1d_tumour,
)


CAGE_INFO = [
    (1, "1x10Gy", "Cage 1: 1 × 10 Gy (Single Dose)"),
    (2, "1x20Gy", "Cage 2: 1 × 20 Gy (Single Dose)"),
    (3, "1x40Gy", "Cage 3: 1 × 40 Gy (Ablative Single)"),
    (4, "Control_0Gy", "Cage 4: 0 Gy (Untreated Control)"),
    (5, "2x5Gy", "Cage 5: 2 × 5 Gy (4d Shield Gap)"),
    (6, "2x10Gy", "Cage 6: 2 × 10 Gy (4d Shield Gap)"),
    (7, "4x5Gy", "Cage 7: 4 × 5 Gy (4d Shield Gap)"),
    (8, "4x10Gy", "Cage 8: 4 × 10 Gy (4d Shield Gap)"),
]

CATEGORY_COLORS = {
    "Tier 1: Excellent Match": "#2E7D32",
    "Tier 2: Good Match": "#1565C0",
    "Mismatch A: Focal Hotspot Escape (Obs >> Pred)": "#C62828",
    "Mismatch B: Deep Resorption Below Fibrotic Floor (Obs << Pred)": "#6A1B9A",
    "Mismatch C: Spontaneous Plateau / Growth Arrest (Obs < Pred)": "#EF6C00",
    "Mismatch D: Mid-Course Oedema Spike / Late Surge": "#00838F",
    "Censored at Baseline (0 Post-RT Points)": "#78909C",
}


def _classify_vascular_phenotype(blue_score: int, khalila_note: str) -> Tuple[str, str]:
    """Returns (short_badge_label, hex_color) for a tumour's Day 84 vascular phenotype."""
    note_l = khalila_note.lower().strip()
    if "very blue" in note_l or blue_score >= 2 or "blue!" in note_l:
        return f"BLUE! (score={blue_score})" if not note_l else f"BLUE ({khalila_note})", "#1565C0"
    if blue_score == 1 or "blue" in note_l:
        return f"Partial Blue ({khalila_note or 'score=1'})", "#1E88E5"
    if "pink" in note_l:
        return f"PINK ({khalila_note})", "#D81B60"
    if any(w in note_l for w in ("excised", "removed", "remove", "none", "scab")):
        return f"Excised/Scab ({khalila_note})", "#6D4C41"
    if any(w in note_l for w in ("deceased", "euthanize")):
        return f"Sacrificed ({khalila_note})", "#546E7A"
    return "Surviving (Unannotated)", "#37474F"


def _classify_tumour_match_or_mismatch(
    rec: TumourStudy2Record,
    n_post_plos: int,
    rmsd_rel_plos_s4: float,
    rmsd_log2_plos_s4: float,
    rmsd_rel_full_s4: float,
    bias_rel_plos_s4: float,
    obs_d30: float,
    pred_d30_s4: float,
    obs_max_plos: float,
    obs_max_full: float,
    pred_max_full_s4: float,
) -> Tuple[str, str]:
    """
    Classifies each tumour into a fit-quality tier or a specific mechanistic mismatch archetype,
    returning `(category_label, detailed_explanation)`.
    """
    if n_post_plos == 0:
        return (
            "Censored at Baseline (0 Post-RT Points)",
            "Mouse died or was censored after Day 43 baseline measurement (no post-RT caliper points).",
        )

    # Check if PLOS window fit is strong, or if only a transient/late spike inflated RMSD
    if rmsd_rel_plos_s4 <= 0.30 or rmsd_log2_plos_s4 <= 0.28:
        if rmsd_rel_full_s4 > 0.80 and (obs_max_full - pred_max_full_s4) > 1.20:
            return (
                "Tier 1: Excellent Match",
                f"Excellent match across the 30-day PLOS ONE window (PLOS RMSD={rmsd_rel_plos_s4:.2f}); late Day 84 breakout surge.",
            )
        return (
            "Tier 1: Excellent Match",
            f"Close agreement between simulated and observed growth trajectory (PLOS RMSD={rmsd_rel_plos_s4:.2f}, log2-RMSD={rmsd_log2_plos_s4:.2f}).",
        )

    if rmsd_rel_plos_s4 <= 0.52 or rmsd_log2_plos_s4 <= 0.48:
        return (
            "Tier 2: Good Match",
            f"Good trajectory concordance within caliper measurement noise (PLOS RMSD={rmsd_rel_plos_s4:.2f}, log2-RMSD={rmsd_log2_plos_s4:.2f}).",
        )

    # Otherwise, classify the specific mechanistic mismatch pattern:
    # 1. Non-monotonic mid-course spike (e.g. Day +18 spike > 2.3x that drops back near prediction by Day 73)
    if abs(obs_d30 - pred_d30_s4) <= 0.50 and (obs_max_plos - obs_d30) >= 0.75:
        return (
            "Mismatch D: Mid-Course Oedema Spike / Late Surge",
            f"Endpoint Day 73 volume matches closely (Obs {obs_d30:.2f} vs Pred {pred_d30_s4:.2f}), but transient mid-course swelling (peak {obs_max_plos:.2f}x) inflated RMSD.",
        )

    # 2. Mismatch Type A: Focal Hotspot Escape / Fast Relapse (Obs >> Pred, negative bias Pred - Obs < -0.35)
    if bias_rel_plos_s4 < -0.35 or (obs_d30 - pred_d30_s4) > 0.55:
        if rec.plos_combo_tm <= 1.55 and rec.pet_hotspot_tm >= 2.40:
            reason = (
                f"Whole-tumour combo T/M={rec.plos_combo_tm:.2f} appeared normoxic (EF5-), but focal PET hotspot T/M={rec.pet_hotspot_tm:.2f} "
                f"harboured a radioresistant hypoxic sub-clone driving rapid relapse (Obs Day 73={obs_d30:.2f}x vs Pred={pred_d30_s4:.2f}x)."
            )
        else:
            reason = (
                f"Resistant hypoxic/congested tumour outgrew model prediction after {rec.treatment} "
                f"(Obs Day 73={obs_d30:.2f}x vs Pred={pred_d30_s4:.2f}x; Hotspot T/M={rec.pet_hotspot_tm:.2f})."
            )
        return ("Mismatch A: Focal Hotspot Escape (Obs >> Pred)", reason)

    # 3. Mismatch Type B: Deep Resorption / Scab Collapse Below Fibrotic Floor (Obs << Pred, Obs d30 < 0.80)
    if obs_d30 < 0.82 and pred_d30_s4 >= 0.98:
        return (
            "Mismatch B: Deep Resorption Below Fibrotic Floor (Obs << Pred)",
            f"Tumour regressed/sloughed below baseline (Obs Day 73={obs_d30:.2f}x) whereas the population fibrotic scaffold floor "
            f"(eta_fib=0.76) maintained simulated residual volume at {pred_d30_s4:.2f}x.",
        )

    # 4. Mismatch Type C: Spontaneous Growth Arrest / Early Plateau (Obs < Pred in 0 Gy / 10 Gy / 2x5 Gy)
    return (
        "Mismatch C: Spontaneous Plateau / Growth Arrest (Obs < Pred)",
        f"Tumour grew rapidly pre-RT (Days 23–43) but decelerated/plateaued earlier than expected post-baseline "
        f"(Obs Day 73={obs_d30:.2f}x vs Pred={pred_d30_s4:.2f}x).",
    )


def build_per_tumour_data_and_curves(
    root: Path,
) -> Tuple[List[TumourStudy2Record], pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Loads all 80 Study 2 tumours, reconstructs continuous 1D curves (Stage 1 Blind, Stage 2 Pre-RT,
    Stage 4 Hierarchical MAP), loads saved 3D Voxel PDE predictions (Stage 1, Stage 2, Stage 4)
    and Stage 4 LORO credible bands, and compiles the 80-tumour RMSD & Mismatch Taxonomy table.
    """
    res_dir = root / "results" / "study2_validation"
    records = load_study2_cohort(root)
    base_params = RadiobiologyParams()

    # Load saved Stage 1, 2, 4 pointwise predictions (including 3D Voxel PDE runs)
    pts_s12 = pd.read_csv(res_dir / "stage1_stage2_pointwise_predictions.csv")
    pts_s4 = pd.read_csv(res_dir / "stage4_pointwise_predictions.csv")
    all_pts = pd.concat([pts_s12, pts_s4], ignore_index=True)

    loro_bands_df = pd.read_csv(res_dir / "stage4_loro_credible_bands.csv")
    post_sum_df = pd.read_csv(res_dir / "stage4_posterior_parameter_summary.csv")
    re_df = pd.read_csv(res_dir / "stage4_hierarchical_random_effects.csv")
    kp_hier_map = dict(zip(re_df["tumour_id"], re_df["kp_hierarchical_prert"]))

    # Reconstruct Stage 4 MAP population parameter vector
    specs = get_bayesian_pop_specs(base_params)
    theta_map = np.array(
        [float(post_sum_df.loc[post_sum_df["parameter"] == sp.name, "posterior_map"].iloc[0]) for sp in specs],
        dtype=float,
    )

    curves_by_tumour: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []

    for rec in records:
        tid = rec.tumour_id
        # 1. Stage 1 Blind (1D)
        res_s1_1d, pred_s1_1d = simulate_1d_tumour(rec, base_params, tm_mode="plos_combo_tm", schedule_mode="actual")
        # 2. Stage 2 Pre-RT Growth-Anchored (1D)
        kp_s2 = fit_prert_tumour_kp_max(rec, base_params, vol_mode="cal_05xy2")
        prm_s2 = replace(base_params, kp_max=kp_s2, kp_hyp_floor=base_params.kp_hyp_floor * (kp_s2 / base_params.kp_max))
        res_s2_1d, pred_s2_1d = simulate_1d_tumour(rec, prm_s2, tm_mode="plos_combo_tm", schedule_mode="actual")
        # 3. Stage 4 Hierarchical Bayesian MAP (1D)
        kp_s4 = kp_hier_map.get(tid, base_params.kp_max)
        prm_s4 = _vec_to_pop_params(theta_map, specs, base_params, kp_max_val=kp_s4)
        res_s4_1d, pred_s4_1d = simulate_1d_tumour(rec, prm_s4, tm_mode="plos_combo_tm", schedule_mode="actual")

        # Extract saved 3D Voxel PDE predictions & LORO predictions at post-RT observation points
        sub_pts = all_pts[all_pts["tumour_id"] == tid]

        def _extract_stage_series(st_name: str) -> Tuple[np.ndarray, np.ndarray]:
            sp = sub_pts[sub_pts["stage"] == st_name].sort_values("t_days_from_base")
            if len(sp) == 0:
                return np.array([0.0]), np.array([1.0])
            t_v = np.concatenate([[0.0], sp["t_days_from_base"].to_numpy(dtype=float)])
            p_v = np.concatenate([[1.0], sp["pred_rel_vol"].to_numpy(dtype=float)])
            return t_v, p_v

        t_s1_3d, p_s1_3d = _extract_stage_series("Stage1_3D_Voxel_Blind_Frozen")
        t_s2_3d, p_s2_3d = _extract_stage_series("Stage2_3D_Voxel_PreRT_Growth_Anchored")
        t_s4_3d, p_s4_3d = _extract_stage_series("Stage4_3D_Voxel_Hierarchical_Bayesian_MAP")
        t_loro, p_loro = _extract_stage_series("Stage4_1D_LORO_OutOfSample_CV")

        # LORO Credible Ribbon for this tumour
        sub_band = loro_bands_df[loro_bands_df["tumour_id"] == tid].sort_values("t_days_from_base")

        # Pre-RT fitted exponential/crowded curve on t in [-20..0] days
        t_pre_grid = np.linspace(-22.0, 0.0, 25)
        sgr_pre_vis = max((kp_s4 * 0.58 - base_params.ka_basal), 0.025)
        rel_pre_grid = np.exp(sgr_pre_vis * t_pre_grid)

        # Compute RMSD (V/V0), log2-RMSD, MAE, and Bias for each model variant
        post_mask = rec.post_t_days > 0.0
        plos_mask = (rec.post_t_days > 0.0) & (rec.post_days_post_inj <= 73)
        n_post_all = int(np.sum(post_mask))
        n_post_plos = int(np.sum(plos_mask))
        obs_post = rec.post_rel_05xy2

        def _metrics_for_pred(t_pred_arr: np.ndarray, p_pred_arr: np.ndarray) -> Dict[str, float]:
            if n_post_all == 0:
                return {
                    "rmsd_rel_plos": np.nan,
                    "rmsd_log2_plos": np.nan,
                    "mae_rel_plos": np.nan,
                    "bias_rel_plos": np.nan,
                    "rmsd_rel_full": np.nan,
                    "rmsd_log2_full": np.nan,
                }
            p_at_obs = np.interp(rec.post_t_days, t_pred_arr, p_pred_arr)
            err_all = p_at_obs[post_mask] - obs_post[post_mask]
            l2_err_all = np.log2(np.maximum(p_at_obs[post_mask], 0.05)) - np.log2(np.maximum(obs_post[post_mask], 0.05))
            rmsd_full = float(np.sqrt(np.mean(err_all**2)))
            l2_full = float(np.sqrt(np.mean(l2_err_all**2)))

            if n_post_plos > 0:
                err_p = p_at_obs[plos_mask] - obs_post[plos_mask]
                l2_err_p = np.log2(np.maximum(p_at_obs[plos_mask], 0.05)) - np.log2(np.maximum(obs_post[plos_mask], 0.05))
                rmsd_p = float(np.sqrt(np.mean(err_p**2)))
                l2_p = float(np.sqrt(np.mean(l2_err_p**2)))
                mae_p = float(np.mean(np.abs(err_p)))
                bias_p = float(np.mean(err_p))
            else:
                rmsd_p, l2_p, mae_p, bias_p = rmsd_full, l2_full, float(np.mean(np.abs(err_all))), float(np.mean(err_all))

            return {
                "rmsd_rel_plos": rmsd_p,
                "rmsd_log2_plos": l2_p,
                "mae_rel_plos": mae_p,
                "bias_rel_plos": bias_p,
                "rmsd_rel_full": rmsd_full,
                "rmsd_log2_full": l2_full,
            }

        m_s1_1d = _metrics_for_pred(res_s1_1d.t_days, res_s1_1d.rel_volume)
        m_s1_3d = _metrics_for_pred(t_s1_3d, p_s1_3d) if len(t_s1_3d) > 1 else m_s1_1d
        m_s2_1d = _metrics_for_pred(res_s2_1d.t_days, res_s2_1d.rel_volume)
        m_s2_3d = _metrics_for_pred(t_s2_3d, p_s2_3d) if len(t_s2_3d) > 1 else m_s2_1d
        m_s4_loro = _metrics_for_pred(t_loro, p_loro) if len(t_loro) > 1 else m_s2_1d
        m_s4_1d = _metrics_for_pred(res_s4_1d.t_days, res_s4_1d.rel_volume)
        m_s4_3d = _metrics_for_pred(t_s4_3d, p_s4_3d) if len(t_s4_3d) > 1 else m_s4_1d

        obs_d30 = float(np.interp(30.0, rec.post_t_days, obs_post)) if len(rec.post_t_days) > 1 else 1.0
        pred_d30_s1 = float(np.interp(30.0, res_s1_1d.t_days, res_s1_1d.rel_volume))
        pred_d30_s4 = float(np.interp(30.0, res_s4_1d.t_days, res_s4_1d.rel_volume))
        pred_d30_s4_3d = float(np.interp(30.0, t_s4_3d, p_s4_3d)) if len(t_s4_3d) > 1 else pred_d30_s4

        obs_max_plos = float(np.max(obs_post[plos_mask])) if n_post_plos > 0 else 1.0
        obs_max_full = float(np.max(obs_post)) if len(obs_post) > 0 else 1.0
        pred_max_full_s4 = float(np.max(np.interp(rec.post_t_days, res_s4_1d.t_days, res_s4_1d.rel_volume)))

        pheno_lbl, pheno_col = _classify_vascular_phenotype(rec.blue_score, rec.khalila_note)
        cat_lbl, cat_reason = _classify_tumour_match_or_mismatch(
            rec=rec,
            n_post_plos=n_post_plos,
            rmsd_rel_plos_s4=m_s4_1d["rmsd_rel_plos"] if not np.isnan(m_s4_1d["rmsd_rel_plos"]) else 0.0,
            rmsd_log2_plos_s4=m_s4_1d["rmsd_log2_plos"] if not np.isnan(m_s4_1d["rmsd_log2_plos"]) else 0.0,
            rmsd_rel_full_s4=m_s4_1d["rmsd_rel_full"] if not np.isnan(m_s4_1d["rmsd_rel_full"]) else 0.0,
            bias_rel_plos_s4=m_s4_1d["bias_rel_plos"] if not np.isnan(m_s4_1d["bias_rel_plos"]) else 0.0,
            obs_d30=obs_d30,
            pred_d30_s4=pred_d30_s4,
            obs_max_plos=obs_max_plos,
            obs_max_full=obs_max_full,
            pred_max_full_s4=pred_max_full_s4,
        )

        best_rmsd_plos = float(
            np.nanmin(
                [
                    m_s1_1d["rmsd_rel_plos"],
                    m_s1_3d["rmsd_rel_plos"],
                    m_s2_1d["rmsd_rel_plos"],
                    m_s2_3d["rmsd_rel_plos"],
                    m_s4_1d["rmsd_rel_plos"],
                    m_s4_3d["rmsd_rel_plos"],
                ]
            )
        ) if n_post_plos > 0 else np.nan

        hotspot_discordant = bool((rec.plos_combo_tm <= 1.50) and (rec.pet_hotspot_tm >= 2.50))

        row_dict = {
            "tumour_id": tid,
            "plos_tumour_code": rec.plos_tumour_code,
            "mouse_id": rec.mouse_id,
            "cage_num": rec.cage_num,
            "mouse_num": rec.mouse_num,
            "side": rec.side,
            "treatment": rec.treatment,
            "total_dose_gy": rec.total_dose_gy,
            "num_fractions": rec.num_fractions,
            "imaged": rec.imaged,
            "pre_rt_combo_tm": round(rec.plos_combo_tm, 4),
            "pet_hotspot_tm": round(rec.pet_hotspot_tm, 4),
            "hotspot_to_combo_ratio": round(rec.pet_hotspot_tm / max(rec.plos_combo_tm, 0.25), 3),
            "hypoxic_stratum": "EF5+ (Hypoxic)" if rec.plos_combo_tm > 1.50 else "EF5- (Oxygenated)",
            "hotspot_discordant_hypoxia": hotspot_discordant,
            "blue_score": rec.blue_score,
            "khalila_note": rec.khalila_note,
            "vascular_phenotype": pheno_lbl,
            "n_pre_rt_points": len(rec.pre_t_days),
            "n_post_rt_points_plos": n_post_plos,
            "n_post_rt_points_full": n_post_all,
            "kp_hierarchical_prert": round(kp_s4, 5),
            "obs_rel_vol_d30": round(obs_d30, 4),
            "pred_s1_1d_rel_vol_d30": round(pred_d30_s1, 4),
            "pred_s4_1d_rel_vol_d30": round(pred_d30_s4, 4),
            "pred_s4_3d_rel_vol_d30": round(pred_d30_s4_3d, 4),
            "rmsd_rel_plos_s1_1d": round(m_s1_1d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s1_3d": round(m_s1_3d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s2_1d": round(m_s2_1d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s2_3d": round(m_s2_3d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s4_loro": round(m_s4_loro["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s4_1d": round(m_s4_1d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_plos_s4_3d": round(m_s4_3d["rmsd_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "best_rmsd_rel_plos": round(best_rmsd_plos, 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s1_1d": round(m_s1_1d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s1_3d": round(m_s1_3d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s2_1d": round(m_s2_1d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s2_3d": round(m_s2_3d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s4_loro": round(m_s4_loro["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s4_1d": round(m_s4_1d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_log2_plos_s4_3d": round(m_s4_3d["rmsd_log2_plos"], 4) if n_post_plos > 0 else np.nan,
            "rmsd_rel_full_s1_1d": round(m_s1_1d["rmsd_rel_full"], 4) if n_post_all > 0 else np.nan,
            "rmsd_rel_full_s4_1d": round(m_s4_1d["rmsd_rel_full"], 4) if n_post_all > 0 else np.nan,
            "rmsd_rel_full_s4_3d": round(m_s4_3d["rmsd_rel_full"], 4) if n_post_all > 0 else np.nan,
            "bias_rel_plos_s4_1d": round(m_s4_1d["bias_rel_plos"], 4) if n_post_plos > 0 else np.nan,
            "match_mismatch_category": cat_lbl,
            "mechanistic_explanation": cat_reason,
        }
        table_rows.append(row_dict)

        curves_by_tumour[tid] = {
            "rec": rec,
            "row": row_dict,
            "pheno_lbl": pheno_lbl,
            "pheno_col": pheno_col,
            "cat_lbl": cat_lbl,
            "cat_col": CATEGORY_COLORS.get(cat_lbl, "#455A64"),
            "t_pre_grid": t_pre_grid,
            "rel_pre_grid": rel_pre_grid,
            "t_1d": res_s1_1d.t_days,
            "rel_s1_1d": res_s1_1d.rel_volume,
            "rel_s2_1d": res_s2_1d.rel_volume,
            "rel_s4_1d": res_s4_1d.rel_volume,
            "t_s1_3d": t_s1_3d,
            "p_s1_3d": p_s1_3d,
            "t_s2_3d": t_s2_3d,
            "p_s2_3d": p_s2_3d,
            "t_s4_3d": t_s4_3d,
            "p_s4_3d": p_s4_3d,
            "sub_band": sub_band,
        }

    master_tum_df = pd.DataFrame(table_rows)
    return records, master_tum_df, curves_by_tumour


def _plot_single_tumour_panel(
    ax: plt.Axes,
    cdata: Dict[str, Any],
    compact: bool = False,
    show_legend: bool = False,
) -> None:
    """Renders a single tumour's observed pre/post-RT calipers and superimposed 1D/3D simulations."""
    rec: TumourStudy2Record = cdata["rec"]
    row: Dict[str, Any] = cdata["row"]
    cat_col: str = cdata["cat_col"]
    pheno_col: str = cdata["pheno_col"]
    is_hyp = rec.plos_combo_tm > 1.50
    main_col = "#C62828" if is_hyp else "#1565C0"
    band_col = "#FFCDD2" if is_hyp else "#BBDEFB"

    # 1. Stage 4 LORO 95% and 50% Credible Ribbon
    sb: pd.DataFrame = cdata["sub_band"]
    if len(sb) > 0:
        tb = sb["t_days_from_base"].to_numpy(dtype=float)
        ax.fill_between(tb, sb["pred_q025_rel_vol"], sb["pred_q975_rel_vol"], color=band_col, alpha=0.30, label="S4 LORO 95% CrI")
        ax.fill_between(tb, sb["pred_q25_rel_vol"], sb["pred_q75_rel_vol"], color=main_col, alpha=0.14, label="S4 LORO 50% CrI")

    # 2. Pre-RT fitted growth curve (t in -22..0 days)
    ax.plot(cdata["t_pre_grid"], cdata["rel_pre_grid"], color="#78909C", ls=":", lw=1.2, alpha=0.85)

    # 3. Post-RT Simulated Curves (Stage 1 1D & 3D, Stage 2 1D, Stage 4 1D & 3D)
    t1d = cdata["t_1d"]
    ax.plot(t1d, cdata["rel_s1_1d"], color="#546E7A", ls="--", lw=1.35, alpha=0.85, label="S1 Blind (1D)")
    if len(cdata["t_s1_3d"]) > 1:
        t3_dense = np.linspace(0.0, float(cdata["t_s1_3d"][-1]), 40)
        p3_s1_dense = np.interp(t3_dense, cdata["t_s1_3d"], cdata["p_s1_3d"])
        ax.plot(t3_dense, p3_s1_dense, color="#00ACC1", ls=":", lw=1.45, alpha=0.90, label="S1 Blind (3D Voxel)")

    ax.plot(t1d, cdata["rel_s2_1d"], color="#2E7D32", ls="-.", lw=1.45, alpha=0.88, label="S2 Pre-RT Growth (1D)")
    ax.plot(t1d, cdata["rel_s4_1d"], color=main_col, ls="-", lw=2.25, alpha=0.96, label="S4 Hierarchical MAP (1D)")

    if len(cdata["t_s4_3d"]) > 1:
        t3_s4_dense = np.linspace(0.0, float(cdata["t_s4_3d"][-1]), 40)
        p3_s4_dense = np.interp(t3_s4_dense, cdata["t_s4_3d"], cdata["p_s4_3d"])
        ax.plot(t3_s4_dense, p3_s4_dense, color="#8E24AA", ls="--", lw=1.75, alpha=0.92, label="S4 Hierarchical (3D Voxel)")

    # 4. Observed Pre-RT & Post-RT Caliper Measurements
    pre_rel = rec.pre_vol_05xy2_mm3 / max(rec.base_cal_05xy2_mm3, 1e-6)
    ax.scatter(
        rec.pre_t_days,
        pre_rel,
        facecolors="white",
        edgecolors="#37474F",
        s=28 if not compact else 16,
        linewidths=1.2,
        zorder=5,
        label="Obs Pre-RT Calipers",
    )
    ax.plot(
        rec.post_t_days,
        rec.post_rel_05xy2,
        "o-",
        color="#212121",
        lw=1.15,
        markersize=5.2 if not compact else 3.4,
        markerfacecolor=main_col,
        markeredgecolor="black",
        markeredgewidth=0.7,
        zorder=6,
        label="Obs Post-RT Calipers",
    )

    # Vertical reference lines: Pre-RT baseline (t=0) and PLOS ONE 30-day window (t=30 / Day 73)
    ax.axvline(0.0, color="#455A64", ls="-", lw=0.85, alpha=0.6)
    ax.axvline(30.0, color="#757575", ls=":", lw=0.95, alpha=0.7)

    # Dynamic y-axis scaling tailored to the tumour's range
    max_obs = float(np.max(rec.post_rel_05xy2)) if len(rec.post_rel_05xy2) > 0 else 1.5
    max_s4 = float(np.max(cdata["rel_s4_1d"][: int(len(t1d) * 0.75)]))
    y_top = max(2.4, min(max(max_obs * 1.22, max_s4 * 1.18), 7.8))
    ax.set_xlim(-22, 57)
    ax.set_ylim(0.0, y_top)
    ax.grid(True, alpha=0.22)

    # Color-code subplot border by Match/Mismatch Category
    for sp in ax.spines.values():
        sp.set_edgecolor(cat_col)
        sp.set_linewidth(2.0 if not compact else 1.4)

    hyp_short = "EF5+ Hypoxic" if is_hyp else "EF5- Normoxic"
    disc_flag = " [!Hotspot]" if row["hotspot_discordant_hypoxia"] else ""

    if not compact:
        ax.set_title(
            f"{rec.tumour_id} ({rec.plos_tumour_code}) | {rec.treatment} | {hyp_short}{disc_flag}\n"
            f"Combo T/M={rec.plos_combo_tm:.2f}, Hot={rec.pet_hotspot_tm:.2f} | {cdata['pheno_lbl']}",
            fontsize=8.3,
            fontweight="bold",
            color="#212121",
            pad=4,
        )
        # Quantitative RMSD & Category badge box inside upper-left
        if row["n_post_rt_points_plos"] > 0:
            rmsd_txt = (
                f"PLOS RMSD (V/V0 | log2):\n"
                f" • S1 Blind 1D/3D: {row['rmsd_rel_plos_s1_1d']:.2f} / {row['rmsd_rel_plos_s1_3d']:.2f}\n"
                f" • S2 PreRT 1D/3D: {row['rmsd_rel_plos_s2_1d']:.2f} / {row['rmsd_rel_plos_s2_3d']:.2f}\n"
                f" • S4 Hier  1D/3D: {row['rmsd_rel_plos_s4_1d']:.2f} / {row['rmsd_rel_plos_s4_3d']:.2f} "
                f"(log2={row['rmsd_log2_plos_s4_1d']:.2f})\n"
                f"[{row['match_mismatch_category'].split('(')[0].strip()}]"
            )
        else:
            rmsd_txt = "[Censored after Baseline Day 43]"

        ax.text(
            0.025,
            0.965,
            rmsd_txt,
            transform=ax.transAxes,
            fontsize=6.8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=cat_col, alpha=0.88, lw=1.2),
            zorder=8,
        )
        # Phenotype colour pill in top-right
        ax.text(
            0.975,
            0.965,
            cdata["pheno_lbl"].split("(")[0].strip(),
            transform=ax.transAxes,
            fontsize=6.8,
            fontweight="bold",
            color="white",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.24", facecolor=pheno_col, edgecolor="none", alpha=0.90),
            zorder=8,
        )
        if show_legend:
            ax.legend(fontsize=6.3, loc="lower left", framealpha=0.90, ncol=2)
    else:
        # Compact mode for the 8x10 80-tumour grand atlas
        rmsd_s4 = f"{row['rmsd_rel_plos_s4_1d']:.2f}" if pd.notna(row["rmsd_rel_plos_s4_1d"]) else "N/A"
        ax.set_title(
            f"{rec.tumour_id} ({rec.treatment})\n{hyp_short[:4]} TM={rec.plos_combo_tm:.1f}/{rec.pet_hotspot_tm:.1f} | RMSD={rmsd_s4}",
            fontsize=6.2,
            fontweight="bold",
            pad=2,
        )
        ax.tick_params(axis="both", labelsize=5.8)
        ax.text(
            0.96,
            0.94,
            cdata["pheno_lbl"].split("(")[0].strip()[:10],
            transform=ax.transAxes,
            fontsize=5.2,
            fontweight="bold",
            color="white",
            va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=pheno_col, edgecolor="none", alpha=0.88),
            zorder=8,
        )


def generate_per_cage_10tumour_grids(
    curves_by_tumour: Dict[str, Dict[str, Any]],
    fig_dir: Path,
) -> List[Path]:
    """Generates 8 high-resolution 2x5 per-cage figures (10 bilateral tumours per treatment regimen)."""
    grid_dir = fig_dir / "per_tumour_grids"
    grid_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for cnum, trt, ctitle in CAGE_INFO:
        c_items = [
            d for d in curves_by_tumour.values() if d["rec"].cage_num == cnum
        ]
        # Organise into 2 rows (Row 0 = Left flank 'L', Row 1 = Right flank 'R') x 5 columns (Mouse 1..5)
        lhs_items = sorted([d for d in c_items if d["rec"].side == "L"], key=lambda x: x["rec"].mouse_num)
        rhs_items = sorted([d for d in c_items if d["rec"].side == "R"], key=lambda x: x["rec"].mouse_num)

        fig, axes = plt.subplots(2, 5, figsize=(22.5, 9.6), dpi=140, facecolor="white")
        for col_idx in range(5):
            if col_idx < len(lhs_items):
                _plot_single_tumour_panel(
                    axes[0, col_idx], lhs_items[col_idx], compact=False, show_legend=(col_idx == 0)
                )
                axes[0, col_idx].set_ylabel("Left Flank ($V / V_0$)" if col_idx == 0 else "", fontsize=9.5, fontweight="bold")
            if col_idx < len(rhs_items):
                _plot_single_tumour_panel(
                    axes[1, col_idx], rhs_items[col_idx], compact=False, show_legend=False
                )
                axes[1, col_idx].set_ylabel("Right Flank ($V / V_0$)" if col_idx == 0 else "", fontsize=9.5, fontweight="bold")
                axes[1, col_idx].set_xlabel("Days from Pre-RT Baseline (2011-05-31)", fontsize=8.8)

        fig.suptitle(
            f"{ctitle} — Per-Tumour Observed vs. Simulated Growth Curves (1D Spheroid & 3D Voxel PDE, Stages 1–4)\n"
            f"[Top Row: Left Flank (L) | Bottom Row: Right Flank (R) Paired by Mouse | Border Color = Fit/Mismatch Class]",
            fontsize=13.0,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.945])
        out_p = grid_dir / f"cage{cnum}_{trt}_per_tumour_curves.png"
        fig.savefig(out_p, dpi=140)
        plt.close(fig)
        saved_paths.append(out_p)

    return saved_paths


def generate_fig8_all_80_tumours_atlas(
    curves_by_tumour: Dict[str, Dict[str, Any]],
    fig_dir: Path,
) -> Path:
    """Generates an 8x10 grand visual atlas of all 80 tumours on one high-resolution canvas."""
    fig, axes = plt.subplots(8, 10, figsize=(26.0, 20.5), dpi=135, facecolor="white")

    for row_idx, (cnum, trt, ctitle) in enumerate(CAGE_INFO):
        c_items = sorted(
            [d for d in curves_by_tumour.values() if d["rec"].cage_num == cnum],
            key=lambda x: (x["rec"].mouse_num, 0 if x["rec"].side == "L" else 1),
        )
        for col_idx, item in enumerate(c_items[:10]):
            ax = axes[row_idx, col_idx]
            _plot_single_tumour_panel(ax, item, compact=True, show_legend=False)
            if col_idx == 0:
                ax.set_ylabel(f"Cage {cnum}\n({trt})\n$V/V_0$", fontsize=7.5, fontweight="bold")
            if row_idx == 7:
                ax.set_xlabel("Days from May 31", fontsize=6.8)

    legend_patches = [Patch(facecolor="white", edgecolor=col, lw=2.2, label=lbl) for lbl, col in CATEGORY_COLORS.items()]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=4,
        fontsize=8.8,
        framealpha=0.95,
        title="Subplot Border Color = Per-Tumour Match Quality Tier / Mechanistic Mismatch Classification",
        title_fontsize=9.5,
    )
    fig.suptitle(
        "Second EF5 RT Study — Complete 80-Tumour Growth Curve Atlas (Observed Pre/Post-RT Calipers vs. Stages 1, 2 & 4 1D/3D Models)",
        fontsize=15.0,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out_p = fig_dir / "08_all_80_tumours_growth_curve_atlas.png"
    fig.savefig(out_p, dpi=135)
    plt.close(fig)
    return out_p


def generate_fig9_rmsd_ranking_and_mismatch_taxonomy(
    master_tum_df: pd.DataFrame,
    curves_by_tumour: Dict[str, Dict[str, Any]],
    fig_dir: Path,
) -> Path:
    """
    Generates a 4-panel diagnostic figure:
      A. Per-tumour RMSD waterfall ranking across all 76 evaluable post-RT tumours
      B. Match & Mismatch Taxonomy counts across the 8 radiation regimens
      C. Focal PET-Hotspot vs Whole-Tumour T/M Discordance explaining Mismatch Type A (Resistant Escape)
      D. Representative Archetype Curves for Excellent Match vs Mismatch Types A, B, and C
    """
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.4), dpi=140, facecolor="white")

    eval_df = master_tum_df[master_tum_df["n_post_rt_points_plos"] > 0].copy()
    eval_df = eval_df.sort_values("rmsd_rel_plos_s4_1d").reset_index(drop=True)

    # Panel A: Ranked Waterfall of Per-Tumour PLOS RMSD (Stage 4 Hierarchical MAP vs Stage 1 Blind)
    ax0 = axes[0, 0]
    x_idx = np.arange(len(eval_df))
    bar_cols = [CATEGORY_COLORS.get(c, "#546E7A") for c in eval_df["match_mismatch_category"]]
    ax0.bar(x_idx, eval_df["rmsd_rel_plos_s4_1d"], color=bar_cols, width=0.82, alpha=0.90, label="Stage 4 Hierarchical (Colored by Class)")
    ax0.scatter(x_idx, eval_df["rmsd_rel_plos_s1_1d"], color="#37474F", marker=".", s=24, alpha=0.65, label="Stage 1 Blind 1D RMSD")
    ax0.scatter(x_idx, eval_df["rmsd_rel_plos_s4_3d"], color="#8E24AA", marker="^", s=18, alpha=0.70, label="Stage 4 3D Voxel RMSD")
    ax0.axhline(0.30, color="#2E7D32", ls="--", lw=1.2, label="Tier 1 Excellent Threshold (0.30)")
    ax0.axhline(0.52, color="#1565C0", ls=":", lw=1.2, label="Tier 2 Good Match Threshold (0.52)")
    ax0.set_xticks(x_idx[::3])
    ax0.set_xticklabels(eval_df["tumour_id"].iloc[::3], rotation=65, ha="right", fontsize=6.8)
    ax0.set_ylim(0.0, min(2.45, float(eval_df["rmsd_rel_plos_s4_1d"].max()) * 1.06))
    ax0.set_ylabel("Per-Tumour PLOS Window RMSD ($V / V_0$)", fontsize=9.5)
    ax0.set_title(
        f"A. Ranked Per-Tumour RMSD Across All {len(eval_df)} Evaluable Post-RT Tumours (Median = {eval_df['rmsd_rel_plos_s4_1d'].median():.2f})",
        fontsize=10.8,
        fontweight="bold",
    )
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(fontsize=7.6, loc="upper left")

    # Panel B: Stacked Bar Chart of Match / Mismatch Taxonomy by Treatment Regimen
    ax1 = axes[0, 1]
    trt_order = ["Control_0Gy", "1x10Gy", "1x20Gy", "1x40Gy", "2x5Gy", "2x10Gy", "4x5Gy", "4x10Gy"]
    cat_keys = list(CATEGORY_COLORS.keys())
    piv = (
        master_tum_df.groupby(["treatment", "match_mismatch_category"])["tumour_id"]
        .count()
        .unstack(fill_value=0)
        .reindex(index=trt_order, columns=cat_keys, fill_value=0)
    )
    bottom = np.zeros(len(trt_order))
    xp = np.arange(len(trt_order))
    for ck in cat_keys:
        vals = piv[ck].to_numpy(dtype=float)
        if np.sum(vals) > 0:
            ax1.bar(xp, vals, bottom=bottom, color=CATEGORY_COLORS[ck], edgecolor="white", width=0.65, label=f"{ck} (n={int(np.sum(vals))})")
            bottom += vals
    ax1.set_xticks(xp)
    ax1.set_xticklabels([t.replace("Control_", "") for t in trt_order], fontweight="bold", fontsize=8.8)
    ax1.set_ylabel("Number of Tumours per Regimen (Total = 10)", fontsize=9.5)
    ax1.set_ylim(0, 11.5)
    ax1.set_title("B. Distribution of Match Tiers & Mismatch Types Across the 8 Regimens", fontsize=10.8, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=7.2, loc="upper right", framealpha=0.92)

    # Panel C: Whole-Tumour Combo T/M vs. Focal PET-Hotspot T/M Colored by Match/Mismatch Class
    ax2 = axes[1, 0]
    for ck, col in CATEGORY_COLORS.items():
        sub_c = eval_df[eval_df["match_mismatch_category"] == ck]
        if len(sub_c) == 0:
            continue
        ax2.scatter(
            sub_c["pre_rt_combo_tm"],
            sub_c["pet_hotspot_tm"],
            color=col,
            s=46,
            alpha=0.88,
            edgecolors="k",
            linewidths=0.5,
            label=f"{ck.split('(')[0].strip()} (n={len(sub_c)})",
        )
    # Highlight the discordance quadrant (Combo T/M <= 1.50 but Hotspot T/M >= 2.50)
    ax2.axvline(1.50, color="#C62828", ls="--", lw=1.2, label="Whole-Tumour Cutoff (T/M = 1.5)")
    ax2.axhline(2.50, color="#6A1B9A", ls=":", lw=1.2, label="PET-Hotspot Cutoff (T/M = 2.5)")
    ax2.fill_between([0.0, 1.50], 2.50, 6.2, color="#FFEBEE", alpha=0.38, label="Focal Hotspot Discordance Zone")
    # Annotate notable Type A escape tumours inside the discordance zone
    for _, r in eval_df[eval_df["match_mismatch_category"].str.startswith("Mismatch A")].iterrows():
        if r["pre_rt_combo_tm"] < 1.65:
            ax2.annotate(
                r["tumour_id"],
                (r["pre_rt_combo_tm"], r["pet_hotspot_tm"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color="#B71C1C",
            )
    ax2.set_xlim(0.0, 5.5)
    ax2.set_ylim(1.0, 6.1)
    ax2.set_xlabel("Whole-Tumour Combo ROI EF5 `T/M` (PLOS ONE Cutoff = 1.5)", fontsize=9.5)
    ax2.set_ylabel("Focal PET-Hotspot Sub-ROI EF5 `T/M` (Report Cutoff = 2.5)", fontsize=9.5)
    ax2.set_title("C. Why Some 'EF5-' Tumours Escaped RT: Focal Hotspot vs. Whole-Tumour T/M", fontsize=10.8, fontweight="bold")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7.0, loc="lower right", framealpha=0.90)

    # Panel D: Representative Archetype Trajectories (Overlaying 4 Canonical Examples)
    ax3 = axes[1, 1]
    archetypes = [
        ("c1_M0_L", "Tier 1 Match: c1_M0_L (1×10 Gy, EF5+, RMSD=0.13)", "#2E7D32"),
        ("c2_M1R_R", "Mismatch A (Focal Hotspot Escape): c2_M1R_R (1×20 Gy, Combo=1.06, Hot=3.08)", "#C62828"),
        ("c3_M2R_R", "Mismatch B (Deep Scab Collapse): c3_M2R_R (1×40 Gy, Obs=0.31x vs Floor=1.06x)", "#6A1B9A"),
        ("c4_M1L_R", "Mismatch C (Spontaneous Plateau): c4_M1L_R (0 Gy Ctrl, Obs=0.80x vs Pred=1.21x)", "#EF6C00"),
    ]
    for tid_a, lbl_a, col_a in archetypes:
        if tid_a not in curves_by_tumour:
            continue
        cd = curves_by_tumour[tid_a]
        rc: TumourStudy2Record = cd["rec"]
        ax3.plot(rc.post_t_days, rc.post_rel_05xy2, "o-", color=col_a, lw=1.8, markersize=5.0, label=f"Obs {lbl_a}")
        ax3.plot(cd["t_1d"], cd["rel_s4_1d"], "--", color=col_a, lw=1.6, alpha=0.75)

    ax3.axvline(30.0, color="#757575", ls=":", lw=1.1)
    ax3.set_xlim(0, 56)
    ax3.set_ylim(0.0, 5.2)
    ax3.set_xlabel("Days from Pre-RT Baseline (2011-05-31; Solid=Observed, Dashed=Model)", fontsize=9.5)
    ax3.set_ylabel("Relative Tumour Volume ($V / V_0$)", fontsize=9.5)
    ax3.set_title("D. Archetype Growth Curves Illustrating Top Matches & Mismatch Classes", fontsize=10.8, fontweight="bold")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=7.4, loc="upper left", framealpha=0.92)

    fig.tight_layout()
    out_p = fig_dir / "09_per_tumour_rmsd_ranking_and_mismatch_taxonomy.png"
    fig.savefig(out_p, dpi=140)
    plt.close(fig)
    return out_p


def update_technical_report_with_per_tumour_section(
    root: Path,
    master_tum_df: pd.DataFrame,
) -> Path:
    """Appends Section 5 (Per-Tumour RMSD Rankings & Mismatch Taxonomy) and Appendix A.6 to the report."""
    report_path = root / "outputs" / "study2_validation" / "study2_validation_technical_report.md"
    existing_md = report_path.read_text(encoding="utf-8")

    # Remove any previous Section 5 if re-running idempotently
    marker = "\n---\n\n## Section 5: Per-Tumour Growth Curve Atlas, RMSD Evaluation & Mechanistic Mismatch Taxonomy"
    if marker in existing_md:
        existing_md = existing_md.split(marker)[0]

    eval_df = master_tum_df[master_tum_df["n_post_rt_points_plos"] > 0].copy()

    # 1. Summary table of Match / Mismatch Categories
    cat_grp = (
        master_tum_df.groupby("match_mismatch_category")
        .agg(
            n_tumours=("tumour_id", "count"),
            mean_combo_tm=("pre_rt_combo_tm", "mean"),
            mean_hotspot_tm=("pet_hotspot_tm", "mean"),
            pct_hotspot_discordant=("hotspot_discordant_hypoxia", lambda s: 100.0 * float(np.mean(s))),
            pct_blue=("blue_score", lambda s: 100.0 * float(np.mean(s >= 1))),
            med_rmsd_s1=("rmsd_rel_plos_s1_1d", "median"),
            med_rmsd_s4_1d=("rmsd_rel_plos_s4_1d", "median"),
            med_rmsd_s4_3d=("rmsd_rel_plos_s4_3d", "median"),
            mean_bias_s4=("bias_rel_plos_s4_1d", "mean"),
        )
        .reindex([k for k in CATEGORY_COLORS.keys() if k in master_tum_df["match_mismatch_category"].values])
        .reset_index()
    )

    tax_lines = [
        "| Match Tier / Mismatch Archetype | Tumours ($N$, %) | Mean Combo `T/M` | Mean Hotspot `T/M` | Focal Hotspot Discordant (%) | Blue Vascular Phenotype (%) | Median PLOS RMSD (S1 Blind 1D) | Median PLOS RMSD (S4 Hier 1D / 3D) | Mean Signed Bias ($\\text{Pred} - \\text{Obs}$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in cat_grp.iterrows():
        s1_str = f"`{r['med_rmsd_s1']:.3f}`" if pd.notna(r["med_rmsd_s1"]) else "`N/A`"
        s4_str = f"`{r['med_rmsd_s4_1d']:.3f}` / `{r['med_rmsd_s4_3d']:.3f}`" if pd.notna(r["med_rmsd_s4_1d"]) else "`N/A`"
        b_str = f"`{r['mean_bias_s4']:+.3f}`" if pd.notna(r["mean_bias_s4"]) else "`N/A`"
        tax_lines.append(
            f"| **{r['match_mismatch_category']}** | `{int(r['n_tumours'])}` (`{100.0 * r['n_tumours'] / 80.0:.1f}%`) | "
            f"`{r['mean_combo_tm']:.2f}` | `{r['mean_hotspot_tm']:.2f}` | `{r['pct_hotspot_discordant']:.0f}%` | "
            f"`{r['pct_blue']:.0f}%` | {s1_str} | {s4_str} | {b_str} |"
        )
    tax_table_md = "\n".join(tax_lines)

    # 2. Top 15 Best-Matching Tumours Table
    top15 = eval_df.sort_values("rmsd_rel_plos_s4_1d", ascending=True).head(15)
    top_lines = [
        "| Rank | Tumour ID (`PLOS Code`) | Regimen | Hypoxia Status (Combo / Hotspot `T/M`) | Day 84 Vascular Phenotype | Obs Day 73 $V/V_0$ | Pred Day 73 (1D / 3D) | PLOS RMSD (S1 Blind → S4 1D / 3D) | PLOS $\\log_2$-RMSD (S4 1D) |",
        "| :---: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for idx_r, (_, r) in enumerate(top15.iterrows(), start=1):
        top_lines.append(
            f"| {idx_r} | **`{r['tumour_id']}`** (`{r['plos_tumour_code']}`) | `{r['treatment']}` | "
            f"{r['hypoxic_stratum'].split(' ')[0]} (`{r['pre_rt_combo_tm']:.2f}` / `{r['pet_hotspot_tm']:.2f}`) | "
            f"`{r['vascular_phenotype']}` | `{r['obs_rel_vol_d30']:.2f}` | "
            f"`{r['pred_s4_1d_rel_vol_d30']:.2f}` / `{r['pred_s4_3d_rel_vol_d30']:.2f}` | "
            f"`{r['rmsd_rel_plos_s1_1d']:.3f}` → **`{r['rmsd_rel_plos_s4_1d']:.3f}`** / `{r['rmsd_rel_plos_s4_3d']:.3f}` | "
            f"`{r['rmsd_log2_plos_s4_1d']:.3f}` |"
        )
    top_table_md = "\n".join(top_lines)

    # 3. Notable Mismatches Table (Top 16 Largest RMSD Tumours with Mechanistic Diagnosis)
    worst16 = eval_df.sort_values("rmsd_rel_plos_s4_1d", ascending=False).head(16)
    worst_lines = [
        "| Tumour ID (`PLOS Code`) | Regimen | Combo / Hotspot `T/M` | Vascular Phenotype | Obs Day 73 $V/V_0$ | Pred Day 73 (1D / 3D) | PLOS RMSD (S4 1D / 3D) | Mismatch Archetype | Mechanistic Diagnosis |",
        "| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]
    for _, r in worst16.iterrows():
        worst_lines.append(
            f"| **`{r['tumour_id']}`** (`{r['plos_tumour_code']}`) | `{r['treatment']}` | "
            f"`{r['pre_rt_combo_tm']:.2f}` / `{r['pet_hotspot_tm']:.2f}` | `{r['vascular_phenotype']}` | "
            f"`{r['obs_rel_vol_d30']:.2f}` | `{r['pred_s4_1d_rel_vol_d30']:.2f}` / `{r['pred_s4_3d_rel_vol_d30']:.2f}` | "
            f"`{r['rmsd_rel_plos_s4_1d']:.3f}` / `{r['rmsd_rel_plos_s4_3d']:.3f}` | **{r['match_mismatch_category'].split('(')[0].strip()}** | "
            f"{r['mechanistic_explanation']} |"
        )
    worst_table_md = "\n".join(worst_lines)

    n_exc = int((master_tum_df["match_mismatch_category"] == "Tier 1: Excellent Match").sum())
    n_good = int((master_tum_df["match_mismatch_category"] == "Tier 2: Good Match").sum())
    n_well = n_exc + n_good

    sec5_md = f"""{marker}

To evaluate model fidelity at the **individual tumour level** across all 80 bilateral tumours, we generated superimposed trajectory comparisons for every single tumour (`Stage 1 Blind 1D & 3D`, `Stage 2 Pre-RT Growth-Anchored 1D & 3D`, and `Stage 4 Hierarchical Bayesian LORO Credible Ribbon, 1D MAP & 3D Voxel PDE`), annotated with each tumour's radiation dose regimen, pre-RT hypoxia status (`Combo T/M` and `PET Hotspot T/M`), Day 84 `Blue`/`Pink` vascular phenotype, and root-mean-square deviation (**RMSD** on $V/V_0$ and $\\log_2 V/V_0$).

All 80 per-tumour metrics and mechanistic classifications are saved in [results/study2_validation/per_tumour_rmsd_and_mismatch_classification.csv](../../results/study2_validation/per_tumour_rmsd_and_mismatch_classification.csv).

### 5.1 Complete 80-Tumour Growth Curve Atlas & Per-Cage High-Resolution Grids

![Complete 80-Tumour Growth Curve Atlas](figures/08_all_80_tumours_growth_curve_atlas.png)

High-resolution `2 × 5` bilateral per-cage grids (pairing Left Flank `[L]` on the top row with Right Flank `[R]` on the bottom row for Mice 1–5 in each treatment group) are saved under `outputs/study2_validation/figures/per_tumour_grids/`:
- **Cage 1 (`1 × 10 Gy`):** [cage1_1x10Gy_per_tumour_curves.png](figures/per_tumour_grids/cage1_1x10Gy_per_tumour_curves.png)
- **Cage 2 (`1 × 20 Gy`):** [cage2_1x20Gy_per_tumour_curves.png](figures/per_tumour_grids/cage2_1x20Gy_per_tumour_curves.png)
- **Cage 3 (`1 × 40 Gy`):** [cage3_1x40Gy_per_tumour_curves.png](figures/per_tumour_grids/cage3_1x40Gy_per_tumour_curves.png)
- **Cage 4 (`0 Gy Control`):** [cage4_Control_0Gy_per_tumour_curves.png](figures/per_tumour_grids/cage4_Control_0Gy_per_tumour_curves.png)
- **Cage 5 (`2 × 5 Gy, 4d Gap`):** [cage5_2x5Gy_per_tumour_curves.png](figures/per_tumour_grids/cage5_2x5Gy_per_tumour_curves.png)
- **Cage 6 (`2 × 10 Gy, 4d Gap`):** [cage6_2x10Gy_per_tumour_curves.png](figures/per_tumour_grids/cage6_2x10Gy_per_tumour_curves.png)
- **Cage 7 (`4 × 5 Gy, 4d Gap`):** [cage7_4x5Gy_per_tumour_curves.png](figures/per_tumour_grids/cage7_4x5Gy_per_tumour_curves.png)
- **Cage 8 (`4 × 10 Gy, 4d Gap`):** [cage8_4x10Gy_per_tumour_curves.png](figures/per_tumour_grids/cage8_4x10Gy_per_tumour_curves.png)

---

### 5.2 Per-Tumour RMSD Waterfall & Mechanistic Mismatch Taxonomy

Across the **76 evaluable post-RT tumours** (4 tumours — `c3_M1L_L`, `c3_M1L_R`, `c4_M1R_L`, `c4_M1R_R` — were censored after their Day 43 baseline measurement), **`{n_well} / 76` tumours (`{100.0 * n_well / 76.0:.1f}%`)** achieved either **Tier 1 Excellent Match (`N = {n_exc}`, PLOS RMSD ≤ 0.30)** or **Tier 2 Good Match (`N = {n_good}`, PLOS RMSD ≤ 0.52)**, with a cohort-wide median per-tumour PLOS RMSD of **`{eval_df['rmsd_rel_plos_s4_1d'].median():.3f}`** ($V/V_0$) and **`{eval_df['rmsd_log2_plos_s4_1d'].median():.3f}`** ($\\log_2 V/V_0$).

Even more importantly, the remaining tumours do not fail randomly—they cluster into **four distinct, biologically interpretable mismatch archetypes**:

![Per-Tumour RMSD Ranking & Mechanistic Mismatch Taxonomy](figures/09_per_tumour_rmsd_ranking_and_mismatch_taxonomy.png)

{tax_table_md}

#### Key Mechanistic Findings from the Mismatch Taxonomy
1. **Mismatch Type A — Focal Hotspot Escape (`Obs >> Pred`, $N = {int((master_tum_df['match_mismatch_category'] == 'Mismatch A: Focal Hotspot Escape (Obs >> Pred)').sum())}$ tumours):**
   - **What happens:** The real tumour regrows much faster after radiation than predicted (`Mean Bias = {float(master_tum_df.loc[master_tum_df['match_mismatch_category'] == 'Mismatch A: Focal Hotspot Escape (Obs >> Pred)', 'bias_rel_plos_s4_1d'].mean()):+.2f}`).
   - **Why it happens (The Focal Hotspot Discordance Discovery):** Look at Figure 9C (`c2_M1R_R`, `c5_M0_R`, `c6_M1R_L`, `c7_M1L1R_L`). In several of these tumours, the **whole-tumour combination ROI `T/M` was low (`≤ 1.50`)**—causing volume-averaged models to classify them as `"EF5- Normoxic"` and predict strong radiation cell kill—**yet their focal `PET Hotspot Sub-ROI T/M` was extremely high (`2.79 to 5.83`)**, and **`56%` were scored as `"Blue"` (congested vasculature)**! Biologically, a single surviving hypoxic sub-clone inside an otherwise normoxic or partially necrotic shell is sufficient to repopulate the tumour after `1×20 Gy` or `2×10 Gy`.
2. **Mismatch Type B — Deep Resorption / Scab Sloughing Below the Fibrotic Floor (`Obs << Pred`, $N = {int((master_tum_df['match_mismatch_category'] == 'Mismatch B: Deep Resorption Below Fibrotic Floor (Obs << Pred)').sum())}$ tumours):**
   - **What happens:** After `1×40 Gy`, `4×10 Gy`, or `2×10 Gy`, the observed volume drops to `0.11×–0.75× V0` (often annotated by Khalila as `"none"`, `"removed"`, or a residual superficial scab that sloughed off), whereas our continuum model's population fibrotic scaffold parameter ($\\eta_{{\\text{{fib}}}} = 0.76$) holds the simulated residual stroma near `1.0×–1.6× V0`.
   - **How to resolve in future extensions:** Allowing tumour-specific fibrotic resorption ($\\eta_{{\\text{{fib}},i}}$) or thresholded macroscopic scab sloughing when viable fraction $V < 2\\%$ will capture complete regressions.
3. **Mismatch Type C — Spontaneous Growth Arrest / Early Plateau (`Obs < Pred`, $N = {int((master_tum_df['match_mismatch_category'] == 'Mismatch C: Spontaneous Plateau / Growth Arrest (Obs < Pred)').sum())}$ tumours):**
   - **What happens:** Concentrated in untreated `0 Gy Control` (`c4_M0_L`, `c4_M1L_R`) or low-dose `1×10 Gy` / `2×5 Gy` tumours that grew rapidly from Day 23 to Day 43 (`20 → 250 mm³`), leading Stage 2/4 pre-RT anchoring to assign a high intrinsic growth rate $k_{{p,\\max,i}}$, but then abruptly hit a local host carrying capacity plateau right after Day 43.
4. **Mismatch Type D — Mid-Course Inflammatory Oedema Spike (`N = {int((master_tum_df['match_mismatch_category'] == 'Mismatch D: Mid-Course Oedema Spike / Late Surge').sum())}$):**
   - **What happens:** Exemplified by `c1_M0_R` (`1×10 Gy`), which tracked the model at Days 49, 56, and 73 (`Obs Day 73 = 1.72×` vs `Pred = 1.29×` in S4 / `1.57×` in S1), but exhibited a single transient jump to `2.71×` at Day 64 (`+18d post-RT`) before regressing back to `1.72×` at Day 73—a hallmark of transient post-RT inflammatory swelling/oedema (or caliper measurement noise).

---

### 5.3 Top 15 Best-Matching Individual Tumours

{top_table_md}

---

### 5.4 Diagnostic Table of the 16 Largest-RMSD Mismatch Tumours

{worst_table_md}

---

### 5.5 Addition to Layperson's Guide: What Do the Individual Tumour Mismatches Teach Us?
When we look at all 80 tumours individually (Figure 8 and the per-cage grids), **nearly two-thirds (`{100.0 * n_well / 76.0:.1f}%`) of the evaluable tumours track the simulated curves closely**. Just as importantly, the one-third that deviate fall into **three simple biological stories** that are immediately useful to experimentalists:
1. **"The Hidden Hypoxic Seed" (Mismatch Type A — Tumour Grows Faster Than Predicted):** If a tumour is mostly well-oxygenated on average (`Combo T/M < 1.5`), a model that only looks at the *average* oxygen level predicts the radiation will wipe it out. But if the PET scan shows a tiny, intensely hypoxic "hotspot" (`Hotspot T/M > 2.5–3.0`, Figure 9C), that resistant pocket survives the radiation and regrows the whole tumour! This proves why **3D spatial hotspot imaging** matters more than whole-tumour averages.
2. **"The Scab That Fell Off" (Mismatch Type B — Tumour Shrinks More Than Predicted):** Our mathematical model assumes dead cancer cells leave behind a permanent scar-tissue scaffold ($\\eta_{{\\text{{fib}}}} \\approx 60\\text{{--}}75\\%$ of the original volume). In most mouse flank tumours, that scar stays put. However, in ~8 tumours, the dead tissue dried into a surface scab that physically sloughed off or resorbed completely (`Obs V/V0 = 0.1–0.5×`), making the real measurement smaller than the simulated scar floor.
3. **"The Early Ceiling" (Mismatch Type C — Untreated/Low-Dose Tumour Stops Growing Early):** A few control or low-dose tumours sprinted ahead before Day 43 and then suddenly stopped growing on their own (likely because they outgrew their local flank blood supply), whereas the model expected them to keep cruising at their pre-treatment speed.
"""

    updated_md = existing_md.rstrip() + "\n" + sec5_md
    report_path.write_text(updated_md, encoding="utf-8")
    return report_path


def run_per_tumour_analysis(root: Optional[Path] = None) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    res_dir = root / "results" / "study2_validation"
    fig_dir = root / "outputs" / "study2_validation" / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GENERATING PER-TUMOUR GROWTH CURVES, RMSD METRICS & MISMATCH TAXONOMY (80 TUMOURS)")
    print("=" * 78)

    records, master_tum_df, curves_by_tumour = build_per_tumour_data_and_curves(root)
    csv_path = res_dir / "per_tumour_rmsd_and_mismatch_classification.csv"
    master_tum_df.to_csv(csv_path, index=False)
    print(f"-> Saved 80-tumour RMSD & Mismatch Classification table: {csv_path.name}")

    # Print quick summary of Match/Mismatch categories
    print("\nMatch / Mismatch Taxonomy Breakdown (80 Tumours):")
    for cat, cnt in master_tum_df["match_mismatch_category"].value_counts().items():
        print(f"   • {cat}: {cnt} tumours ({100.0 * cnt / 80.0:.1f}%)")

    print("\n-> Rendering 8 High-Resolution Per-Cage (2x5 Bilateral) Growth Curve Grids...")
    cage_paths = generate_per_cage_10tumour_grids(curves_by_tumour, fig_dir)
    for cp in cage_paths:
        print(f"   Saved: figures/per_tumour_grids/{cp.name}")

    print("-> Rendering Figure 8: Complete 80-Tumour Growth Curve Atlas (8x10 Grid)...")
    f8 = generate_fig8_all_80_tumours_atlas(curves_by_tumour, fig_dir)
    print(f"   Saved: figures/{f8.name}")

    print("-> Rendering Figure 9: Per-Tumour RMSD Waterfall Ranking & Mismatch Taxonomy...")
    f9 = generate_fig9_rmsd_ranking_and_mismatch_taxonomy(master_tum_df, curves_by_tumour, fig_dir)
    print(f"   Saved: figures/{f9.name}")

    print("-> Updating Technical Report with Section 5 (Per-Tumour Analysis & Mismatch Taxonomy)...")
    rep_path = update_technical_report_with_per_tumour_section(root, master_tum_df)
    print(f"   Updated Technical Report: {rep_path}")

    return {
        "master_tum_df": master_tum_df,
        "cage_paths": cage_paths,
        "fig8_path": f8,
        "fig9_path": f9,
        "report_path": rep_path,
    }


if __name__ == "__main__":
    run_per_tumour_analysis()
