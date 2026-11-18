"""
Shared biophysics & radiobiology module (consistent physical units: mm, days, mmHg, Gy).

Used identically by both:
  - Sub-goal 1: 1D radially-symmetric coupled spheroid ODE/PDE RT model
  - Sub-goal 2: 3D CT/EF5-PET-initialised continuum voxel RT model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass
class RadiobiologyParams:
    """
    Physically interpretable radiobiology, oxygen-transport, mechanical-saturation, and
    EF5-binding parameters calibrated for subcutaneous A549 human lung adenocarcinoma xenografts.
    """

    # --- Oxygen scale & cellular thresholds (mmHg) ---
    p_max: float = 40.0         # Max physiological arterial pO2 ceiling (mmHg)
    p_host_bdry: float = 35.0   # Shared physiological host subcutaneous peritumoural pO2 (mmHg)
    p_hyp: float = 0.60         # Half-maximal proliferation pO2 threshold (mmHg; glycolytic mitosis persists in moderate hypoxia)
    p_nec: float = 0.45         # Severe anoxia -> necrosis transition threshold (mmHg)
    hill_p: float = 2.5         # Hill exponent for proliferation switch
    hill_n: float = 3.0         # Hill exponent for necrosis switch

    # --- Cellular growth, mechanical saturation, necrosis & clearance kinetics (day^-1) ---
    kp_max: float = 0.130       # Intrinsic normoxic proliferation rate in small/unconstrained tumours (day^-1)
    kp_hyp_floor: float = 0.055 # Residual glycolytic proliferation in viable chronic hypoxia (day^-1)
    v_stress_cm3: float = 0.55  # Host subcutaneous elastic crowding volume scale (cm^3; drives concave saturation)
    hill_stress: float = 2.0    # Exponent for host solid-stress crowding inhibition
    p_darcy_crit: float = 0.65  # Critical internal Darcy pressure P_crit (mm^2/day) for Byrne stress-inhibited mitosis
    ka_basal: float = 0.010     # Basal viable cell apoptosis / turnover rate (day^-1)
    k_nec_max: float = 0.085    # Max viable -> necrotic conversion rate when pO2 << p_nec (day^-1)

    # 2-stage mitotic catastrophe / p53-WT A549 senescence & delayed clearance for RT-damaged cells:
    #   D1 (early doomed/senescent cells: metabolically active) -> D2 (lysing) -> cleared + fibrotic scaffold N
    k_d1_prolif: float = 0.022  # Mild transient abortive expansion/swelling rate of D1 (day^-1)
    k_d1_to_d2: float = 0.075   # Transit rate from doomed/senescent D1 to clearing D2 (~13.3 day lag, day^-1)
    k_clear_min: float = 0.012  # Baseline clearance rate of D2 in hypoxic/poorly perfused core (day^-1)
    k_clear_o2: float = 0.052   # Additional oxygen/perfusion-assisted clearance rate of D2 (day^-1)
    eta_fibrotic: float = 0.60  # Fraction of clearing D2 leaving persistent fibrotic/stromal scaffold N
    k_nec_clear: float = 0.008  # Slow compaction/clearance of necrotic/fibrotic core N (day^-1)

    # --- Oxygen transport & consumption (physiological Krogh xenograft scale) ---
    D_o2: float = 0.010         # Effective oxygen diffusivity (mm^2 / unit-time; L_o2 ~ 0.24 mm Krogh scale)
    gamma_0: float = 6.0        # Max viable-cell O2 consumption rate (mmHg / unit-time, same scale as D_o2)
    K_m_o2: float = 2.0         # Michaelis-Menten half-saturation pO2 for O2 consumption (mmHg)
    q_d1: float = 0.14          # Relative O2 consumption of early RT-damaged D1 cells (drives inter-fraction reoxygenation)
    q_d2: float = 0.08          # Relative O2 consumption of late lysing/inflammatory D2 compartment
    h_vasc_base: float = 0.10   # Distributed microvascular O2 exchange rate inside perfused tissue
    v_vasc_half_cm3: float = 1.40  # Macroscopic tumour volume scale for mild angiogenic lag (cm^3)

    # --- Linear-Quadratic (LQ) + Alper-Howard-Flanders OER (in-vivo A549 xenografts) ---
    alpha: float = 0.185        # Normoxic in-vivo A549 LQ alpha (Gy^-1)
    beta: float = 0.018         # Normoxic in-vivo A549 LQ beta (Gy^-2), alpha/beta ~ 10.3 Gy
    oer_max: float = 3.0        # Maximum hypoxic Oxygen Enhancement Ratio (m)
    K_oer: float = 3.5          # Alper-Howard-Flanders half-maximal pO2 (mmHg)

    # --- High-dose endothelial/vascular damage (thresholded >= 15-20 Gy; spares 5-10 Gy fractions) ---
    vasc_damage_d50: float = 15.5   # Single-fraction dose for half-maximal endothelial injury (Gy)
    vasc_damage_hill: float = 5.0   # Steep Hill exponent so 5-10 Gy spares vasculature while 20-40 Gy injures it
    vasc_damage_max: float = 0.82   # Max fractional drop in vascular O2 exchange at high single dose
    vasc_recovery_rate: float = 0.012  # Slow microvascular recovery rate post-RT (day^-1)
    high_dose_lysis_boost: float = 0.068  # Extra necrotic/stromal compaction rate driven by acute vascular collapse (day^-1)

    # --- Bioreductive [18F]-EF5 PET <-> pO2 mapping (Koch/Evans kinetics) ---
    tm_min: float = 1.0         # Normoxic muscle-normalized EF5 T/M floor
    tm_max: float = 5.5         # Severely hypoxic viable-cell EF5 T/M ceiling
    K_ef5: float = 5.5          # pO2 (mmHg) for half-maximal bioreductive EF5 binding (~0.7% O2)
    ef5_d1_weight: float = 0.82 # Residual enzymatic EF5 reduction in metabolically active doomed/senescent D1 cells


def oer_alper_howard_flanders(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """
    Continuous Alper-Howard-Flanders Oxygen Enhancement Ratio OER(pO2) in [1, m].

    OER(0) = 1 (maximally radioresistant anoxia), OER(p -> inf) = m (full radiosensitization).
    Note: Radiosensitivity modifier OMF(p) = OER(p) / m in [1/m, 1], so effective dose is
    d_eff = d * OMF(p) = d / OER_rel(p), where OER_rel(p) = m / OER(p) in [1, m].
    """
    p = np.maximum(p_o2, 0.0)
    return (params.oer_max * p + params.K_oer) / (p + params.K_oer)


def omf_oxygen_modifier(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """
    Oxygen Modification Factor OMF(pO2) = OER(pO2) / m in [1/m, 1].
    OMF -> 1 at normoxia (full LQ sensitivity), OMF -> 1/m at anoxia (m-fold dose protection).
    """
    return oer_alper_howard_flanders(p_o2, params) / params.oer_max


def lq_surviving_fraction(dose_gy: float, p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """
    Corrected Linear-Quadratic surviving fraction S(d, pO2) with OER as a dose-modifying factor:

        d_eff = dose_gy * OMF(pO2)
        S(d, pO2) = exp(-alpha * d_eff - beta * d_eff^2)

    Note: Fixes the legacy 2013 MATLAB typo where beta was divided by OER instead of OER^2.
    """
    if dose_gy <= 0.0:
        return np.ones_like(p_o2, dtype=np.float64) if isinstance(p_o2, np.ndarray) else 1.0
    d_eff = dose_gy * omf_oxygen_modifier(p_o2, params)
    return np.exp(-params.alpha * d_eff - params.beta * (d_eff ** 2))


def vascular_damage_fraction(dose_gy: float, params: RadiobiologyParams) -> float:
    """
    Thresholded high-dose single-fraction endothelial/microvascular injury fraction in [0, vasc_damage_max].
    Spares low/moderate fraction sizes (e.g. 5-10 Gy, allowing classic inter-fraction reoxygenation)
    while capturing acute microvascular damage above ~15-20 Gy (Song/Fuks/Kolesnick effect).
    """
    if dose_gy <= 0.0 or params.vasc_damage_max <= 0.0:
        return 0.0
    h = params.vasc_damage_hill
    num = dose_gy ** h
    den = num + (params.vasc_damage_d50 ** h)
    return float(params.vasc_damage_max * (num / max(den, 1e-12)))


def proliferation_rate(
    p_o2: ArrayLike,
    params: RadiobiologyParams,
    vol_cm3: float = 0.0,
    pressure: ArrayLike = 0.0,
) -> ArrayLike:
    """
    Smooth Hill-function viable-cell proliferation rate k_p(pO2, V, P) (day^-1) incorporating
    oxygen limitation and Helen Byrne mechanical/solid-stress growth saturation:

        k_p = k_p_o2(pO2) / (1 + (V / V_stress)^hill_stress + max(P, 0) / P_crit)

    When vol_cm3 == 0 and pressure == 0, reduces to the pure oxygen-limited rate.
    Shrinking post-RT tumours have P <= 0 and small V, so regrowth is not artificially braked.
    """
    p = np.maximum(p_o2, 0.0)
    pn = p ** params.hill_p
    phn = params.p_hyp ** params.hill_p
    kp_o2 = params.kp_hyp_floor + (params.kp_max - params.kp_hyp_floor) * (pn / (pn + phn + 1e-12))

    stress_term: ArrayLike = 0.0
    if vol_cm3 > 0.0 and params.v_stress_cm3 > 0.0:
        stress_term = (vol_cm3 / params.v_stress_cm3) ** params.hill_stress
    if params.p_darcy_crit > 0.0:
        stress_term = stress_term + np.maximum(pressure, 0.0) / params.p_darcy_crit

    return kp_o2 / (1.0 + stress_term)


def necrosis_rate(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """Smooth Hill-function hypoxia-induced necrosis rate k_nec(pO2) (day^-1)."""
    p = np.maximum(p_o2, 0.0)
    pn = p ** params.hill_n
    pnn = params.p_nec ** params.hill_n
    return params.k_nec_max * (pnn / (pn + pnn + 1e-12))


def clearance_rate(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """Perfusion/oxygen-assisted clearance rate k_clear(pO2) for lysing damaged cells D2 (day^-1)."""
    p = np.maximum(p_o2, 0.0)
    return params.k_clear_min + params.k_clear_o2 * (p / (p + params.p_hyp + 1e-12))


def o2_consumption_mm(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """Michaelis-Menten oxygen uptake factor p / (p + K_m) in [0, 1]."""
    p = np.maximum(p_o2, 0.0)
    return p / (p + params.K_m_o2 + 1e-12)


def po2_to_ef5_tm(
    p_o2: ArrayLike,
    viable_frac: ArrayLike,
    params: RadiobiologyParams,
    d1_frac: ArrayLike = 0.0,
) -> ArrayLike:
    """
    Forward bioreductive EF5 PET model: predicts tumour-to-muscle (T/M) ratio from local pO2
    and metabolically active cell fraction.

        TM = tm_min + (tm_max - tm_min) * V_active * K_ef5 / (pO2 + K_ef5)

    Necrotic tissue (V_active = 0) cannot enzymatically reduce 2-nitroimidazoles and therefore
    contributes only background nonspecific signal (tm_min).
    """
    p = np.maximum(p_o2, 0.0)
    v_active = np.clip(viable_frac + params.ef5_d1_weight * d1_frac, 0.0, 1.0)
    binding = params.K_ef5 / (p + params.K_ef5 + 1e-12)
    return params.tm_min + (params.tm_max - params.tm_min) * v_active * binding


def ef5_tm_to_po2(
    tm: ArrayLike,
    params: RadiobiologyParams,
    viable_frac: ArrayLike = 1.0,
    p_min: float = 0.4,
) -> ArrayLike:
    """
    Regularized inverse EF5 PET -> pO2 mapping (mmHg) for initialising spatial hypoxia
    from pre-treatment EF5 PET T/M images.

    Inverts po2_to_ef5_tm analytically with smooth clamping to [p_min, p_max].
    """
    span = np.maximum((params.tm_max - params.tm_min) * np.clip(viable_frac, 0.05, 1.0), 1e-6)
    # Normalized binding fraction b in (0, 1)
    b_raw = (tm - params.tm_min) / span
    # Smoothly clamp b so pO2 stays strictly inside [p_min, p_max]
    b_min = params.K_ef5 / (params.p_max + params.K_ef5)
    b_max = params.K_ef5 / (p_min + params.K_ef5)
    b = np.clip(b_raw, b_min, b_max)
    p = params.K_ef5 * (1.0 - b) / np.maximum(b, 1e-6)
    return np.clip(p, p_min, params.p_max)
