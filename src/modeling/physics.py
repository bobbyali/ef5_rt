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
    Physically interpretable radiobiology, oxygen-transport, and EF5-binding parameters
    calibrated for subcutaneous A549 human lung adenocarcinoma xenografts.
    """

    # --- Oxygen scale & cellular thresholds (mmHg) ---
    p_max: float = 40.0         # Well-perfused arterial/rim capillary pO2 (mmHg)
    p_hyp: float = 4.0          # Half-maximal proliferation pO2 threshold (mmHg; biological hypoxia ~3-5 mmHg)
    p_nec: float = 0.8          # Severe hypoxia -> necrosis transition threshold (mmHg)
    hill_p: float = 2.5         # Hill exponent for proliferation switch
    hill_n: float = 3.0         # Hill exponent for necrosis switch

    # --- Cellular growth, necrosis & clearance kinetics (day^-1) ---
    kp_max: float = 0.075       # Max normoxic rim proliferation rate (day^-1; ~3.2-4.2x over 33d)
    kp_hyp_floor: float = 0.0   # Residual proliferation in severe hypoxia (day^-1)
    ka_basal: float = 0.008     # Basal viable cell apoptosis / turnover rate (day^-1)
    k_nec_max: float = 0.018    # Max viable -> necrotic conversion rate when pO2 << p_nec (day^-1)

    # 2-stage mitotic catastrophe / delayed clearance for RT-damaged cells:
    #   D1 (early doomed cells: abortive division/swelling) -> D2 (lysing) -> cleared (+ residual fibrotic/necrotic)
    k_d1_prolif: float = 0.052  # Transient abortive expansion/swelling rate of D1 (day^-1)
    k_d1_to_d2: float = 0.13    # Transit rate from early doomed D1 to clearing D2 (~7.7 day mean lag, day^-1)
    k_clear_min: float = 0.018  # Baseline clearance rate of D2 in hypoxic/poorly perfused core (day^-1)
    k_clear_o2: float = 0.085   # Additional oxygen/perfusion-assisted clearance rate of D2 (day^-1)
    eta_fibrotic: float = 0.42  # Fraction of clearing D2 leaving persistent necrotic/stromal scaffold N
    k_nec_clear: float = 0.003  # Very slow clearance of necrotic/fibrotic core N (day^-1)

    # --- Oxygen transport & consumption (physiological xenograft tissue scale) ---
    # Characteristic diffusion length L_o2 = sqrt(D_o2 * p_max / gamma_0) ~ 0.60 mm (rim-limited growth!)
    D_o2: float = 0.055         # Effective oxygen diffusivity (mm^2 / unit-time)
    gamma_0: float = 6.0        # Max viable-cell O2 consumption rate (mmHg / unit-time, same scale as D_o2)
    K_m_o2: float = 2.0         # Michaelis-Menten half-saturation pO2 for O2 consumption (mmHg)
    q_d1: float = 0.18          # Relative O2 consumption of early RT-damaged D1 cells (drives reoxygenation)
    q_d2: float = 0.05          # Relative O2 consumption of late lysing D2 cells
    h_vasc_base: float = 0.10   # Distributed microvascular O2 exchange rate inside perfused tissue
    v_vasc_half_cm3: float = 0.85  # Macroscopic tumour volume where vascular efficiency halves (angiogenic lag)

    # --- Linear-Quadratic (LQ) + Alper-Howard-Flanders OER ---
    alpha: float = 0.20         # Normoxic in-vivo A549 LQ alpha (Gy^-1)
    beta: float = 0.020         # Normoxic in-vivo A549 LQ beta (Gy^-2), textbook alpha/beta = 10 Gy
    oer_max: float = 3.0        # Maximum hypoxic Oxygen Enhancement Ratio (m)
    K_oer: float = 3.0          # Alper-Howard-Flanders half-maximal pO2 (mmHg)

    # Optional endothelial/vascular damage after high single-fraction doses (>= 15 Gy)
    vasc_damage_d50: float = 25.0   # Dose for half-maximal transient vascular attenuation (Gy)
    vasc_damage_max: float = 0.0    # Max fractional drop in vascular exchange (0 = disabled by default)
    vasc_recovery_rate: float = 0.05  # Vascular recovery rate (day^-1)

    # --- Bioreductive [18F]-EF5 PET <-> pO2 mapping ---
    tm_min: float = 1.0         # Normoxic muscle-normalized EF5 T/M floor
    tm_max: float = 5.5         # Severely hypoxic viable-cell EF5 T/M ceiling
    K_ef5: float = 3.5          # pO2 (mmHg) for half-maximal bioreductive EF5 binding
    ef5_d1_weight: float = 0.25 # Residual metabolic EF5 reduction in early doomed D1 cells


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


def proliferation_rate(p_o2: ArrayLike, params: RadiobiologyParams) -> ArrayLike:
    """Smooth Hill-function viable-cell proliferation rate k_p(pO2) (day^-1)."""
    p = np.maximum(p_o2, 0.0)
    pn = p ** params.hill_p
    phn = params.p_hyp ** params.hill_p
    return params.kp_hyp_floor + (params.kp_max - params.kp_hyp_floor) * (pn / (pn + phn + 1e-12))


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
