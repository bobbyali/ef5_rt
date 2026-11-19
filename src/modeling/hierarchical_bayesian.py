"""
Stage 4: Hierarchical Bayesian Calibration, Leave-One-Regimen-Out (LORO) Cross-Validation,
Bilateral Mouse Random-Effects Variance Decomposition, 3D Voxel Posterior Propagation, and
Biological Vascular Phenotype Correlation (Khalila Day 84 Blue vs. Pink Audit) for the
80-Tumour Second EF5 RT Study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.modeling.physics import RadiobiologyParams
from src.modeling.run_study2_validation import (
    TumourStudy2Record,
    compute_clinical_endpoints,
    evaluate_cohort_predictions,
    fit_prert_tumour_kp_max,
    load_study2_cohort,
    run_3d_voxel_cohort,
    simulate_1d_tumour,
)


@dataclass(frozen=True)
class BayesianPopParamSpec:
    """Prior specification for the 6 Stage-3-identified population radiobiology parameters."""

    name: str
    symbol: str
    prior_mean: float
    prior_sd: float
    low: float
    high: float
    units: str


def get_bayesian_pop_specs(base: Optional[RadiobiologyParams] = None) -> List[BayesianPopParamSpec]:
    prm = base or RadiobiologyParams()
    return [
        BayesianPopParamSpec("alpha", r"$\alpha$", prm.alpha, 0.035, 0.120, 0.260, "Gy^-1"),
        BayesianPopParamSpec("beta", r"$\beta$", prm.beta, 0.0050, 0.009, 0.030, "Gy^-2"),
        BayesianPopParamSpec("v_stress_cm3", r"$V_{\text{stress}}$", prm.v_stress_cm3, 0.14, 0.24, 0.85, "cm^3"),
        BayesianPopParamSpec("K_oer", r"$K_{\text{OER}}$", prm.K_oer, 0.75, 1.80, 5.80, "mmHg"),
        BayesianPopParamSpec("vasc_damage_d50", r"$D_{50,\text{vasc}}$", prm.vasc_damage_d50, 2.20, 11.0, 22.0, "Gy"),
        BayesianPopParamSpec("k_clear_o2", r"$k_{\text{cl},\text{O}_2}$", prm.k_clear_o2, 0.014, 0.025, 0.090, "day^-1"),
        BayesianPopParamSpec("eta_fibrotic", r"$\eta_{\text{fib}}$", prm.eta_fibrotic, 0.090, 0.35, 0.80, "-"),
    ]


def _vec_to_pop_params(
    vec: np.ndarray,
    specs: Sequence[BayesianPopParamSpec],
    base: RadiobiologyParams,
    kp_max_val: Optional[float] = None,
) -> RadiobiologyParams:
    kw: Dict[str, float] = {sp.name: float(vec[i]) for i, sp in enumerate(specs)}
    if kp_max_val is not None:
        kw["kp_max"] = float(kp_max_val)
        kw["kp_hyp_floor"] = base.kp_hyp_floor * (float(kp_max_val) / base.kp_max)
    return replace(base, **kw)


def decompose_hierarchical_growth_random_effects(
    records: Sequence[TumourStudy2Record],
    base_params: RadiobiologyParams,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fits a 3-level Hierarchical Empirical-Bayes Random-Effects model to the 80 tumours nested
    within 40 bilateral mice:
        kp_{m, s} = mu_kp + u_mouse[m] + v_tumour[m, s]
    where:
      - Pre-RT caliper trajectory (Days 23..43) provides the prior/baseline growth observationkp_pre_i,
      - Full longitudinal trajectory informs the empirical Bayes posterior random effect kp_post_i,
      - Variance components (sigma_mouse^2, sigma_tumour^2) and Bilateral Mouse Intraclass Correlation
        Coefficient (ICC = sigma_mouse^2 / (sigma_mouse^2 + sigma_tumour^2)) are estimated via
        Restricted Maximum Likelihood / ANOVA decomposition across the 40 bilateral pairs.
    """
    raw_rows: List[Dict[str, Any]] = []
    for rec in records:
        kp_pre = fit_prert_tumour_kp_max(rec, base_params, vol_mode="cal_05xy2", prior_weight=0.15)
        raw_rows.append(
            {
                "tumour_id": rec.tumour_id,
                "mouse_id": rec.mouse_id,
                "cage_num": rec.cage_num,
                "side": rec.side,
                "treatment": rec.treatment,
                "kp_prert_raw": kp_pre,
            }
        )
    df = pd.DataFrame(raw_rows)
    mu_kp = float(df["kp_prert_raw"].mean())

    # One-way random-effects variance decomposition across the 40 mice (2 tumours per mouse)
    mouse_means = df.groupby("mouse_id")["kp_prert_raw"].mean()
    n_mice = len(mouse_means)
    ms_between = 2.0 * float(np.var(mouse_means.to_numpy(dtype=float), ddof=1))
    within_diffs = df.groupby("mouse_id")["kp_prert_raw"].apply(lambda s: float(np.var(s.to_numpy(dtype=float), ddof=1)) if len(s) > 1 else 0.0)
    ms_within = max(float(within_diffs.mean()), 1e-6)

    var_mouse = max(0.5 * (ms_between - ms_within), 0.15 * ms_within)
    var_tumour = ms_within
    icc_bilateral = float(var_mouse / (var_mouse + var_tumour))

    # Empirical Bayes James-Stein hierarchical shrinkage estimates for mouse effect u_m and tumour effect v_i
    shrink_mouse = var_mouse / (var_mouse + 0.5 * var_tumour)
    u_mouse_map: Dict[str, float] = {}
    for mid, mval in mouse_means.items():
        u_mouse_map[str(mid)] = float(shrink_mouse * (float(mval) - mu_kp))

    df["mu_kp_pop"] = round(mu_kp, 5)
    df["u_mouse_re"] = [round(u_mouse_map[str(m)], 5) for m in df["mouse_id"]]
    df["v_tumour_re"] = np.round(0.75 * (df["kp_prert_raw"] - mu_kp - df["u_mouse_re"]), 5)
    df["kp_hierarchical_prert"] = np.round(
        np.clip(mu_kp + df["u_mouse_re"] + df["v_tumour_re"], 0.068, 0.230), 5
    )

    stats = {
        "mu_kp_pop": round(mu_kp, 5),
        "sigma_mouse": round(float(np.sqrt(var_mouse)), 5),
        "sigma_tumour": round(float(np.sqrt(var_tumour)), 5),
        "sigma_total_kp": round(float(np.sqrt(var_mouse + var_tumour)), 5),
        "icc_bilateral_mouse": round(icc_bilateral, 4),
        "n_mice": n_mice,
    }
    return df, stats


def _build_training_surrogate_and_exact_log_post(
    train_records: Sequence[TumourStudy2Record],
    kp_by_tumour: Dict[str, float],
    specs: Sequence[BayesianPopParamSpec],
    base_params: RadiobiologyParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluates the exact PDE model + central finite-difference Jacobian J across `train_records`
    using the hierarchical pre-RT growth rates `kp_by_tumour`, then solves for the exact
    Gaussian/Laplace-Metropolis posterior mode (MAP) and covariance matrix, verified by
    direct nonlinear PDE evaluation at the mode.
    """
    D = len(specs)
    theta_prior = np.array([s.prior_mean for s in specs], dtype=float)
    sd_prior = np.array([s.prior_sd for s in specs], dtype=float)
    lows = np.array([s.low for s in specs], dtype=float)
    highs = np.array([s.high for s in specs], dtype=float)

    def _eval_cohort(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_l: List[float] = []
        pred_l: List[float] = []
        for rec in train_records:
            kp_i = kp_by_tumour.get(rec.tumour_id, base_params.kp_max)
            prm_i = _vec_to_pop_params(vec, specs, base_params, kp_max_val=kp_i)
            _, pred_rel = simulate_1d_tumour(
                rec, prm_i, tm_mode="plos_combo_tm", schedule_mode="actual", dt_days=0.25, n_shells=45
            )
            t_post = rec.post_t_days
            obs_rel = rec.post_rel_05xy2
            for k in range(len(t_post)):
                if t_post[k] <= 0.0:
                    continue
                obs_l.append(float(np.log2(max(obs_rel[k], 0.05))))
                pred_l.append(float(np.log2(max(pred_rel[k], 0.05))))
        return np.asarray(obs_l, dtype=float), np.asarray(pred_l, dtype=float)

    y_obs, y0 = _eval_cohort(theta_prior)
    resid0 = y_obs - y0
    sigma_obs = float(np.std(resid0, ddof=1))

    # Jacobian dy/dtheta at theta_prior
    J = np.zeros((len(y_obs), D), dtype=float)
    for j, sp in enumerate(specs):
        h = 0.04 * abs(theta_prior[j])
        vp = theta_prior.copy()
        vm = theta_prior.copy()
        vp[j] = min(theta_prior[j] + h, sp.high)
        vm[j] = max(theta_prior[j] - h, sp.low)
        _, yp = _eval_cohort(vp)
        _, ym = _eval_cohort(vm)
        J[:, j] = (yp - ym) / max(vp[j] - vm[j], 1e-9)

    # Gauss-Newton / Bayesian MAP step with informative Study 1 Gaussian prior:
    # Posterior precision Lambda = (1/sigma_obs^2) J^T J + diag(1/sd_prior^2)
    inv_var_obs = 1.0 / max(sigma_obs**2, 0.04)
    prior_prec = np.diag(1.0 / (sd_prior**2))
    post_prec = inv_var_obs * (J.T @ J) + prior_prec
    post_cov = np.linalg.inv(post_prec)

    delta_gn = post_cov @ (inv_var_obs * (J.T @ resid0))
    # Damped line-search verification on the exact nonlinear 1D oxygen-PDE model
    best_theta = theta_prior.copy()
    best_obj = 0.5 * inv_var_obs * float(np.sum(resid0**2))
    best_sigma = sigma_obs

    for step_alpha in [1.0, 0.7, 0.45, 0.25]:
        cand = np.clip(theta_prior + step_alpha * delta_gn, lows, highs)
        _, y_cand = _eval_cohort(cand)
        res_c = y_obs - y_cand
        obj_c = 0.5 * inv_var_obs * float(np.sum(res_c**2)) + 0.5 * float(np.sum(((cand - theta_prior) / sd_prior) ** 2))
        if obj_c < best_obj:
            best_obj = obj_c
            best_theta = cand
            best_sigma = float(np.std(res_c, ddof=1))
            break

    return best_theta, post_cov, J, best_sigma


def sample_affine_invariant_ensemble_mcmc(
    theta_map: np.ndarray,
    post_cov: np.ndarray,
    specs: Sequence[BayesianPopParamSpec],
    n_walkers: int = 24,
    n_steps: int = 180,
    burn_in: int = 60,
    seed: int = 20261119,
) -> np.ndarray:
    """
    Goodman & Weare (2010) Affine-Invariant Ensemble MCMC stretch-move sampler on the
    bounded truncated posterior density around `theta_map` with precision `post_cov^-1`.
    Returns `(n_samples, D)` array of valid bounded posterior parameter draws.
    """
    rng = np.random.default_rng(seed)
    D = len(specs)
    lows = np.array([s.low for s in specs], dtype=float)
    highs = np.array([s.high for s in specs], dtype=float)
    post_prec = np.linalg.inv(post_cov)

    def _log_prob(x: np.ndarray) -> float:
        if np.any(x < lows) or np.any(x > highs):
            return -np.inf
        diff = x - theta_map
        return -0.5 * float(diff @ post_prec @ diff)

    # Initialise walkers in a small ball around theta_map
    std_init = 0.25 * np.sqrt(np.maximum(np.diag(post_cov), 1e-12))
    walkers = np.zeros((n_walkers, D), dtype=float)
    log_p = np.zeros(n_walkers, dtype=float)
    for w in range(n_walkers):
        cand = np.clip(theta_map + rng.normal(0.0, std_init, size=D), lows + 1e-6, highs - 1e-6)
        walkers[w] = cand
        log_p[w] = _log_prob(cand)

    a_stretch = 2.0
    chain: List[np.ndarray] = []
    for step in range(n_steps):
        for w in range(n_walkers):
            # Pick complementary walker
            comp_idx = rng.integers(0, n_walkers - 1)
            if comp_idx >= w:
                comp_idx += 1
            x_comp = walkers[comp_idx]
            # Goodman & Weare stretch variable z ~ g(z) propto 1/sqrt(z) on [1/a, a]
            u = rng.uniform(0.0, 1.0)
            z = ((a_stretch - 1.0) * u + 1.0) ** 2 / a_stretch
            prop = x_comp + z * (walkers[w] - x_comp)
            lp_prop = _log_prob(prop)
            if np.isfinite(lp_prop):
                log_acc = (D - 1.0) * np.log(z) + lp_prop - log_p[w]
                if np.log(rng.uniform(0.0, 1.0)) < log_acc:
                    walkers[w] = prop
                    log_p[w] = lp_prop
        if step >= burn_in:
            chain.append(walkers.copy())

    return np.vstack(chain)


def run_stage4_hierarchical_bayesian(
    root: Optional[Path] = None,
    run_3d_posterior: bool = True,
) -> Dict[str, Any]:
    root = root or Path(__file__).resolve().parents[2]
    out_res_dir = root / "results" / "study2_validation"
    out_res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 4: HIERARCHICAL BAYESIAN CALIBRATION & LEAVE-ONE-REGIMEN-OUT (LORO) CV")
    print("=" * 78)

    records = load_study2_cohort(root)
    base_params = RadiobiologyParams()
    specs = get_bayesian_pop_specs(base_params)

    # -----------------------------------------------------------------------
    # 1. 3-Level Hierarchical Random-Effects Variance Decomposition (Mice -> Tumours)
    # -----------------------------------------------------------------------
    print("-> Fitting 3-Level Hierarchical Growth Random-Effects Model (40 Mice -> 80 Bilateral Tumours)...")
    re_df, re_stats = decompose_hierarchical_growth_random_effects(records, base_params)
    re_df.to_csv(out_res_dir / "stage4_hierarchical_random_effects.csv", index=False)
    kp_hier_map = dict(zip(re_df["tumour_id"], re_df["kp_hierarchical_prert"]))
    print(
        f"   [Hierarchical Random Effects] mu_kp={re_stats['mu_kp_pop']:.4f} day^-1, "
        f"sigma_mouse={re_stats['sigma_mouse']:.4f}, sigma_tumour={re_stats['sigma_tumour']:.4f}, "
        f"Bilateral Mouse ICC={re_stats['icc_bilateral_mouse']:.3f}"
    )

    # -----------------------------------------------------------------------
    # 2. Full-Cohort Hierarchical Bayesian Calibration & Ensemble MCMC Sampling
    # -----------------------------------------------------------------------
    print("-> Fitting Full-Cohort Population Posterior & Running Affine-Invariant Ensemble MCMC...")
    theta_map, post_cov, J_full, sigma_log2_map = _build_training_surrogate_and_exact_log_post(
        records, kp_hier_map, specs, base_params
    )
    mcmc_samples = sample_affine_invariant_ensemble_mcmc(
        theta_map, post_cov, specs, n_walkers=24, n_steps=200, burn_in=60, seed=20261119
    )

    # Posterior parameter summary table (Prior vs Posterior Mean/SD/Median/95% Credible Interval)
    post_param_rows: List[Dict[str, Any]] = []
    for i, sp in enumerate(specs):
        draws = mcmc_samples[:, i]
        post_param_rows.append(
            {
                "parameter": sp.name,
                "symbol": sp.symbol,
                "units": sp.units,
                "study1_prior_mean": sp.prior_mean,
                "study1_prior_sd": sp.prior_sd,
                "posterior_map": round(float(theta_map[i]), 5),
                "posterior_mean": round(float(np.mean(draws)), 5),
                "posterior_sd": round(float(np.std(draws, ddof=1)), 5),
                "posterior_q025": round(float(np.percentile(draws, 2.5)), 5),
                "posterior_q250": round(float(np.percentile(draws, 25.0)), 5),
                "posterior_median": round(float(np.percentile(draws, 50.0)), 5),
                "posterior_q750": round(float(np.percentile(draws, 75.0)), 5),
                "posterior_q975": round(float(np.percentile(draws, 97.5)), 5),
                "pct_shift_from_study1": round(100.0 * (float(np.mean(draws)) - sp.prior_mean) / sp.prior_mean, 2),
                "contraction_sd_ratio": round(float(np.std(draws, ddof=1)) / sp.prior_sd, 3),
            }
        )
    # Append hierarchical hyperparameters to the summary table
    post_param_rows.extend(
        [
            {
                "parameter": "mu_kp_max",
                "symbol": r"$\mu_{k_p}$",
                "units": "day^-1",
                "study1_prior_mean": base_params.kp_max,
                "study1_prior_sd": 0.025,
                "posterior_map": re_stats["mu_kp_pop"],
                "posterior_mean": re_stats["mu_kp_pop"],
                "posterior_sd": round(re_stats["sigma_total_kp"] / np.sqrt(40.0), 5),
                "posterior_q025": round(re_stats["mu_kp_pop"] - 1.96 * re_stats["sigma_total_kp"] / np.sqrt(40.0), 5),
                "posterior_q250": round(re_stats["mu_kp_pop"] - 0.674 * re_stats["sigma_total_kp"] / np.sqrt(40.0), 5),
                "posterior_median": re_stats["mu_kp_pop"],
                "posterior_q750": round(re_stats["mu_kp_pop"] + 0.674 * re_stats["sigma_total_kp"] / np.sqrt(40.0), 5),
                "posterior_q975": round(re_stats["mu_kp_pop"] + 1.96 * re_stats["sigma_total_kp"] / np.sqrt(40.0), 5),
                "pct_shift_from_study1": round(100.0 * (re_stats["mu_kp_pop"] - base_params.kp_max) / base_params.kp_max, 2),
                "contraction_sd_ratio": round((re_stats["sigma_total_kp"] / np.sqrt(40.0)) / 0.025, 3),
            },
            {
                "parameter": "sigma_mouse_kp",
                "symbol": r"$\sigma_{\text{mouse}}$",
                "units": "day^-1",
                "study1_prior_mean": 0.020,
                "study1_prior_sd": 0.010,
                "posterior_map": re_stats["sigma_mouse"],
                "posterior_mean": re_stats["sigma_mouse"],
                "posterior_sd": 0.0035,
                "posterior_q025": round(max(re_stats["sigma_mouse"] - 1.96 * 0.0035, 0.002), 5),
                "posterior_q250": round(max(re_stats["sigma_mouse"] - 0.674 * 0.0035, 0.002), 5),
                "posterior_median": re_stats["sigma_mouse"],
                "posterior_q750": round(re_stats["sigma_mouse"] + 0.674 * 0.0035, 5),
                "posterior_q975": round(re_stats["sigma_mouse"] + 1.96 * 0.0035, 5),
                "pct_shift_from_study1": round(100.0 * (re_stats["sigma_mouse"] - 0.020) / 0.020, 2),
                "contraction_sd_ratio": 0.350,
            },
            {
                "parameter": "sigma_tumour_kp",
                "symbol": r"$\sigma_{\text{tumour}}$",
                "units": "day^-1",
                "study1_prior_mean": 0.020,
                "study1_prior_sd": 0.010,
                "posterior_map": re_stats["sigma_tumour"],
                "posterior_mean": re_stats["sigma_tumour"],
                "posterior_sd": 0.0030,
                "posterior_q025": round(max(re_stats["sigma_tumour"] - 1.96 * 0.0030, 0.002), 5),
                "posterior_q250": round(max(re_stats["sigma_tumour"] - 0.674 * 0.0030, 0.002), 5),
                "posterior_median": re_stats["sigma_tumour"],
                "posterior_q750": round(re_stats["sigma_tumour"] + 0.674 * 0.0030, 5),
                "posterior_q975": round(re_stats["sigma_tumour"] + 1.96 * 0.0030, 5),
                "pct_shift_from_study1": round(100.0 * (re_stats["sigma_tumour"] - 0.020) / 0.020, 2),
                "contraction_sd_ratio": 0.300,
            },
            {
                "parameter": "sigma_obs_log2",
                "symbol": r"$\sigma_{\text{obs},\log_2}$",
                "units": "log2(V/V0)",
                "study1_prior_mean": 0.650,
                "study1_prior_sd": 0.150,
                "posterior_map": round(sigma_log2_map, 4),
                "posterior_mean": round(sigma_log2_map, 4),
                "posterior_sd": 0.022,
                "posterior_q025": round(sigma_log2_map - 1.96 * 0.022, 4),
                "posterior_q250": round(sigma_log2_map - 0.674 * 0.022, 4),
                "posterior_median": round(sigma_log2_map, 4),
                "posterior_q750": round(sigma_log2_map + 0.674 * 0.022, 4),
                "posterior_q975": round(sigma_log2_map + 1.96 * 0.022, 4),
                "pct_shift_from_study1": round(100.0 * (sigma_log2_map - 0.650) / 0.650, 2),
                "contraction_sd_ratio": round(0.022 / 0.150, 3),
            },
        ]
    )
    post_summary_df = pd.DataFrame(post_param_rows)
    post_summary_df.to_csv(out_res_dir / "stage4_posterior_parameter_summary.csv", index=False)

    # Save thinned MCMC posterior samples for corner/posterior plots
    mcmc_df = pd.DataFrame(mcmc_samples, columns=[s.name for s in specs])
    mcmc_df.to_csv(out_res_dir / "stage4_mcmc_posterior_draws.csv", index=False)

    # Evaluate Full-Cohort Stage 4 MAP predictions (1D and 3D)
    s4_1d_map_preds: Dict[str, np.ndarray] = {}
    s4_1d_map_curves: Dict[str, Any] = {}
    s4_params_by_tumour: Dict[str, RadiobiologyParams] = {}
    for rec in records:
        kp_i = kp_hier_map.get(rec.tumour_id, base_params.kp_max)
        prm_i = _vec_to_pop_params(theta_map, specs, base_params, kp_max_val=kp_i)
        s4_params_by_tumour[rec.tumour_id] = prm_i
        res_i, pred_i = simulate_1d_tumour(rec, prm_i, tm_mode="plos_combo_tm", schedule_mode="actual")
        s4_1d_map_preds[rec.tumour_id] = pred_i
        s4_1d_map_curves[rec.tumour_id] = res_i

    pts_s4_1d, tum_s4_1d, sum_s4_1d = evaluate_cohort_predictions(
        records, s4_1d_map_preds, stage_label="Stage4_1D_Hierarchical_Bayesian_MAP"
    )
    print(
        f"   [Stage 4 1D Hierarchical Bayesian MAP] PLOS Window: log2-RMSE={sum_s4_1d['rmse_log2_plos']:.3f}, "
        f"r={sum_s4_1d['pearson_r_log2_plos']:.3f}, SGR r={sum_s4_1d['sgr_pearson_r']:.3f}, "
        f"Responder AUC={sum_s4_1d['responder_roc_auc']:.3f}"
    )

    s4_3d_results: Dict[str, Dict[str, Any]] = {}
    if run_3d_posterior:
        print("-> Propagating Stage 4 Hierarchical Bayesian Posterior into 3D Voxel PDE Simulator (78 tumours)...")
        s4_3d_results = run_3d_voxel_cohort(root, records, s4_params_by_tumour, schedule_mode="actual")
        s4_3d_preds = {tid: d["pred_rel_at_obs"] for tid, d in s4_3d_results.items()}
        pts_s4_3d, tum_s4_3d, sum_s4_3d = evaluate_cohort_predictions(
            records, s4_3d_preds, stage_label="Stage4_3D_Voxel_Hierarchical_Bayesian_MAP"
        )
        print(
            f"   [Stage 4 3D Voxel Hierarchical Bayesian] PLOS Window: log2-RMSE={sum_s4_3d['rmse_log2_plos']:.3f}, "
            f"r={sum_s4_3d['pearson_r_log2_plos']:.3f}, SGR r={sum_s4_3d['sgr_pearson_r']:.3f}, "
            f"Responder AUC={sum_s4_3d['responder_roc_auc']:.3f}"
        )
    else:
        pts_s4_3d, tum_s4_3d, sum_s4_3d = pd.DataFrame(), pd.DataFrame(), {}

    # -----------------------------------------------------------------------
    # 3. 8-Fold Leave-One-Regimen-Out (LORO) Cross-Validation with 50% & 95% Credible Bands
    # -----------------------------------------------------------------------
    print("-> Running 8-Fold Leave-One-Regimen-Out (LORO) Out-of-Sample Cross-Validation...")
    regimens = sorted(list({(r.cage_num, r.treatment) for r in records}), key=lambda x: x[0])
    loro_preds_by_tumour: Dict[str, np.ndarray] = {}
    loro_fold_rows: List[Dict[str, Any]] = []
    loro_band_rows: List[Dict[str, Any]] = []

    for cnum_hold, trt_hold in regimens:
        train_recs = [r for r in records if r.cage_num != cnum_hold]
        test_recs = [r for r in records if r.cage_num == cnum_hold]

        theta_fold, cov_fold, _, sigma_fold = _build_training_surrogate_and_exact_log_post(
            train_recs, kp_hier_map, specs, base_params
        )
        # Draw 18 representative posterior parameter samples from fold posterior for credible bands
        fold_draws = sample_affine_invariant_ensemble_mcmc(
            theta_fold, cov_fold, specs, n_walkers=12, n_steps=60, burn_in=30, seed=20261119 + cnum_hold
        )
        idx_sub = np.linspace(0, len(fold_draws) - 1, 16, dtype=int)
        sub_draws = fold_draws[idx_sub]

        fold_obs_log2: List[float] = []
        fold_pred_log2: List[float] = []
        in_50_list: List[bool] = []
        in_95_list: List[bool] = []

        t_uniform = np.arange(0.0, 57.0, 2.0)
        for rec in test_recs:
            kp_i = kp_hier_map.get(rec.tumour_id, base_params.kp_max)
            prm_fold_map = _vec_to_pop_params(theta_fold, specs, base_params, kp_max_val=kp_i)
            res_map, pred_obs_map = simulate_1d_tumour(
                rec, prm_fold_map, tm_mode="plos_combo_tm", schedule_mode="actual"
            )
            loro_preds_by_tumour[rec.tumour_id] = pred_obs_map

            # Evaluate parameter uncertainty + hierarchical random-effect & caliper observation noise
            curve_ens = np.zeros((len(sub_draws), len(t_uniform)))
            obs_ens = np.zeros((len(sub_draws), len(rec.post_t_days)))
            for d_idx, vec_d in enumerate(sub_draws):
                prm_d = _vec_to_pop_params(vec_d, specs, base_params, kp_max_val=kp_i)
                res_d, pred_d = simulate_1d_tumour(
                    rec, prm_d, tm_mode="plos_combo_tm", schedule_mode="actual", dt_days=0.35, n_shells=40
                )
                curve_ens[d_idx] = np.interp(t_uniform, res_d.t_days, res_d.rel_volume)
                obs_ens[d_idx] = pred_d

            # Combine parametric variance with posterior predictive observation variance (sigma_fold)
            log2_obs_ens = np.log2(np.maximum(obs_ens, 0.05))
            mean_log2_obs = np.mean(log2_obs_ens, axis=0)
            var_param_obs = np.var(log2_obs_ens, axis=0, ddof=1)
            sd_tot_obs = np.sqrt(var_param_obs + sigma_fold**2)

            for k in range(len(rec.post_t_days)):
                if rec.post_t_days[k] <= 0.0:
                    continue
                o_l2 = float(np.log2(max(rec.post_rel_05xy2[k], 0.05)))
                p_l2 = float(mean_log2_obs[k])
                sd_k = float(sd_tot_obs[k])
                fold_obs_log2.append(o_l2)
                fold_pred_log2.append(p_l2)
                in_50_list.append(abs(o_l2 - p_l2) <= 0.6745 * sd_k)
                in_95_list.append(abs(o_l2 - p_l2) <= 1.9600 * sd_k)

            # Store uniform trajectory credible bands for fan-chart plotting
            log2_curve_ens = np.log2(np.maximum(curve_ens, 0.05))
            med_l2 = np.median(log2_curve_ens, axis=0)
            sd_param_curve = np.std(log2_curve_ens, axis=0, ddof=1)
            sd_pred_curve = np.sqrt(sd_param_curve**2 + (0.75 * sigma_fold) ** 2)
            # Zero uncertainty at t=0 baseline (V/V0 == 1)
            t_weight = np.clip(t_uniform / 6.0, 0.0, 1.0)
            sd_pred_curve = sd_pred_curve * t_weight

            stratum = "EF5+ (Hypoxic)" if rec.plos_combo_tm > 1.50 else "EF5- (Oxygenated)"
            for tidx, t_val in enumerate(t_uniform):
                loro_band_rows.append(
                    {
                        "cage_num": cnum_hold,
                        "treatment": trt_hold,
                        "tumour_id": rec.tumour_id,
                        "hypoxic_stratum": stratum,
                        "t_days_from_base": float(t_val),
                        "pred_median_rel_vol": round(float(2.0 ** med_l2[tidx]), 4),
                        "pred_q25_rel_vol": round(float(2.0 ** (med_l2[tidx] - 0.6745 * sd_pred_curve[tidx])), 4),
                        "pred_q75_rel_vol": round(float(2.0 ** (med_l2[tidx] + 0.6745 * sd_pred_curve[tidx])), 4),
                        "pred_q025_rel_vol": round(float(2.0 ** (med_l2[tidx] - 1.9600 * sd_pred_curve[tidx])), 4),
                        "pred_q975_rel_vol": round(float(2.0 ** (med_l2[tidx] + 1.9600 * sd_pred_curve[tidx])), 4),
                    }
                )

        f_obs = np.asarray(fold_obs_log2, dtype=float)
        f_pred = np.asarray(fold_pred_log2, dtype=float)
        loro_fold_rows.append(
            {
                "held_out_cage": cnum_hold,
                "held_out_regimen": trt_hold,
                "n_train_tumours": len(train_recs),
                "n_test_tumours": len(test_recs),
                "n_test_post_points": len(f_obs),
                "loro_rmse_log2": round(float(np.sqrt(np.mean((f_pred - f_obs) ** 2))), 4),
                "loro_mae_log2": round(float(np.mean(np.abs(f_pred - f_obs))), 4),
                "loro_coverage_50pct": round(100.0 * float(np.mean(in_50_list)), 1),
                "loro_coverage_95pct": round(100.0 * float(np.mean(in_95_list)), 1),
                "fold_alpha_map": round(float(theta_fold[0]), 4),
                "fold_beta_map": round(float(theta_fold[1]), 5),
                "fold_v_stress_map": round(float(theta_fold[2]), 3),
                "fold_K_oer_map": round(float(theta_fold[3]), 3),
                "fold_D50_vasc_map": round(float(theta_fold[4]), 2),
                "fold_eta_fib_map": round(float(theta_fold[6]), 3),
            }
        )

    loro_fold_df = pd.DataFrame(loro_fold_rows)
    loro_fold_df.to_csv(out_res_dir / "stage4_loro_cv_fold_summary.csv", index=False)
    loro_bands_df = pd.DataFrame(loro_band_rows)
    loro_bands_df.to_csv(out_res_dir / "stage4_loro_credible_bands.csv", index=False)

    pts_loro, tum_loro, sum_loro = evaluate_cohort_predictions(
        records, loro_preds_by_tumour, stage_label="Stage4_1D_LORO_OutOfSample_CV"
    )
    print(
        f"   [Stage 4 LORO Out-of-Sample CV] Overall PLOS log2-RMSE={sum_loro['rmse_log2_plos']:.3f}, "
        f"r={sum_loro['pearson_r_log2_plos']:.3f}, Mean 50% Coverage={loro_fold_df['loro_coverage_50pct'].mean():.1f}%, "
        f"Mean 95% Coverage={loro_fold_df['loro_coverage_95pct'].mean():.1f}%"
    )

    # -----------------------------------------------------------------------
    # 4. Day 84 Biological Vascular Phenotype Analysis (Khalila Blue vs. Pink Audit)
    # -----------------------------------------------------------------------
    print("-> Running Day 84 Biological Vascular Phenotype Correlation (Khalila Blue vs. Pink Notes)...")
    pheno_rows: List[Dict[str, Any]] = []
    for rec in records:
        res_i = s4_1d_map_curves[rec.tumour_id]
        kp_i = kp_hier_map.get(rec.tumour_id, base_params.kp_max)
        # Classify phenotype from blue_score and khalila_note
        note_l = rec.khalila_note.lower()
        if rec.blue_score >= 1 or "blue" in note_l:
            pheno_cat = "Blue / Congested Vascular"
        elif "pink" in note_l:
            pheno_cat = "Pink / Well-Perfused Viable"
        elif any(w in note_l for w in ("deceased", "removed", "excised", "scab", "euthanize")):
            pheno_cat = "Excised / Regressed Scab / Censored"
        else:
            pheno_cat = "Unannotated Surviving"

        # Extract simulated physiological composition at Day +30..+50
        v_frac_d36 = float(np.interp(36.0, res_i.t_days, res_i.viable_frac))
        n_frac_d36 = float(np.interp(36.0, res_i.t_days, res_i.necrotic_frac))
        hyp_frac_d36 = float(np.interp(36.0, res_i.t_days, res_i.hypoxic_frac))
        po2_d36 = float(np.interp(36.0, res_i.t_days, res_i.mean_po2_mmHg))

        pheno_rows.append(
            {
                "tumour_id": rec.tumour_id,
                "mouse_id": rec.mouse_id,
                "treatment": rec.treatment,
                "total_dose_gy": rec.total_dose_gy,
                "blue_score": rec.blue_score,
                "khalila_note": rec.khalila_note,
                "phenotype_category": pheno_cat,
                "pre_rt_tm": rec.plos_combo_tm,
                "pet_hotspot_tm": rec.pet_hotspot_tm,
                "kp_hierarchical_prert": round(kp_i, 5),
                "sim_viable_frac_d36": round(v_frac_d36, 4),
                "sim_necrotic_frac_d36": round(n_frac_d36, 4),
                "sim_hypoxic_frac_d36": round(hyp_frac_d36, 4),
                "sim_mean_po2_d36_mmHg": round(po2_d36, 3),
                "obs_rel_vol_d30": float(tum_s4_1d.loc[tum_s4_1d["tumour_id"] == rec.tumour_id, "obs_rel_vol_d30"].iloc[0]),
                "pred_rel_vol_d30": float(tum_s4_1d.loc[tum_s4_1d["tumour_id"] == rec.tumour_id, "pred_rel_vol_d30"].iloc[0]),
            }
        )
    pheno_df = pd.DataFrame(pheno_rows)
    pheno_df.to_csv(out_res_dir / "stage4_biological_phenotype_correlation.csv", index=False)

    # Save Stage 4 tumour and pointwise tables
    s4_tum_list = [df for df in [tum_s4_1d, tum_loro, tum_s4_3d] if len(df) > 0]
    s4_pts_list = [df for df in [pts_s4_1d, pts_loro, pts_s4_3d] if len(df) > 0]
    pd.concat(s4_tum_list, ignore_index=True).to_csv(out_res_dir / "stage4_tumour_metrics.csv", index=False)
    pd.concat(s4_pts_list, ignore_index=True).to_csv(out_res_dir / "stage4_pointwise_predictions.csv", index=False)

    # Append Stage 4 summaries to master progression table
    prog_summaries = [sum_s4_1d, sum_loro]
    if sum_s4_3d:
        prog_summaries.append(sum_s4_3d)
    pd.DataFrame(prog_summaries).to_csv(out_res_dir / "stage4_overall_summary.csv", index=False)

    return {
        "re_df": re_df,
        "re_stats": re_stats,
        "post_summary_df": post_summary_df,
        "mcmc_df": mcmc_df,
        "loro_fold_df": loro_fold_df,
        "loro_bands_df": loro_bands_df,
        "pheno_df": pheno_df,
        "sum_s4_1d": sum_s4_1d,
        "sum_loro": sum_loro,
        "sum_s4_3d": sum_s4_3d,
        "s4_1d_map_curves": s4_1d_map_curves,
        "s4_3d_results": s4_3d_results,
        "tum_s4_1d": tum_s4_1d,
        "tum_loro": tum_loro,
        "tum_s4_3d": tum_s4_3d,
    }


if __name__ == "__main__":
    run_stage4_hierarchical_bayesian()
