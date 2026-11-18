"""
Python ports and mathematical audit of the 2013 legacy spheroid models from images/set4/past_work/:
  1. Legacy 2013 spheroid ODE model (Code/08. Spheroid Model/my_spheroid_odes_07.m & my_spheroid_rt_07.m)
     - Supports both exact legacy reproduction (preserve_legacy_typos=True) and bug-fixed mode
       (fixes missing cube-root on (3*d)/(4*pi) and fixes bh = b / OERh^2).
  2. Dr. Alex Fletcher's 2013 untreated spheroid model (Code/09. Alex Spheroid Model/CodeForBobby.m)
     - Replaces MATLAB pdepe with a fast, exact 1D spherical finite-volume Newton solver.
     - Verified against Helen Byrne's exact analytical Greenspan Eqs. (44), (47)-(50) in pcmi_proofs.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# 1. Legacy 2013 Spheroid ODE Model (my_spheroid_odes_07 / my_spheroid_rt_07)
# ---------------------------------------------------------------------------
@dataclass
class LegacySpheroidResult:
    t: np.ndarray
    Nv: np.ndarray
    Ndn: np.ndarray
    Ndh: np.ndarray
    volume: np.ndarray
    t_post_rt: np.ndarray
    rel_volume: np.ndarray


def _legacy_normoxic_fraction(Ntotal: float, d: float, r2: float, preserve_legacy_typos: bool) -> Tuple[float, float, float]:
    Ntotal = max(float(Ntotal), 1e-9)
    if preserve_legacy_typos:
        # Exact MATLAB line 23: r = (3*d)/(4*pi)*(Ntotal.^(1/3))
        r = (3.0 * d) / (4.0 * np.pi) * (Ntotal ** (1.0 / 3.0))
    else:
        r = ((3.0 * d * Ntotal) / (4.0 * np.pi)) ** (1.0 / 3.0)
    r1 = max(r - r2, 0.0)
    if r <= r2:
        fn = 1.0
    else:
        num = 3.0 * r1**2 * r2 + 3.0 * r1 * r2**2 + r2**3
        den = r1**3 + num
        fn = float(np.clip(num / max(den, 1e-12), 0.0, 1.0))
    V = (4.0 / 3.0) * np.pi * r**3
    return r, fn, V


def run_legacy_spheroid_2013(
    dose_per_fraction: float,
    n_fractions: int = 1,
    hypoxic: bool = True,
    t_pre_rt: float = 20.0,
    t_interval: float = 5.0,
    t_post_final: float = 30.0,
    preserve_legacy_typos: bool = True,
) -> LegacySpheroidResult:
    """
    Reproduces my_spheroid_rt_07.m / my_spheroid_odes_07.m (Rehan Ali, 22 April 2013).
    """
    d_cell = 0.1
    r2 = 0.45
    kpi = 1.0
    kpj = 0.0 if hypoxic else 1.0
    OERn = 1.0
    OERh = 2.7 if hypoxic else 1.0
    Vt = 500.0
    alpha = 0.3
    beta = 0.03

    an = alpha / OERn
    bn = beta / (OERn if preserve_legacy_typos else OERn**2)
    ah = alpha / OERh
    bh = beta / (OERh if preserve_legacy_typos else OERh**2)
    Sn = float(np.exp(-an * dose_per_fraction - bn * dose_per_fraction**2))
    Sh = float(np.exp(-ah * dose_per_fraction - bh * dose_per_fraction**2))

    def make_rhs(kpi_cur: float, r2_cur: float, kd3_cur: float, kd4_cur: float):
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            Nv, Ndn, Ndh = np.maximum(y, 0.0)
            _, fn, V = _legacy_normoxic_fraction(Nv + Ndn + Ndh, d_cell, r2_cur, preserve_legacy_typos)
            if V < Vt:
                kp1, kp2 = kpi_cur, kpj
            else:
                brake = float(np.exp(-(V - Vt) / 500.0))
                kp1, kp2 = kpi_cur * brake, kpj * brake
            dNv = kp1 * fn * Nv + kp2 * (1.0 - fn) * Nv
            dNdn = -kd3_cur * Ndn + 0.5 * fn * Ndh
            dNdh = -kd4_cur * Ndh - 0.5 * fn * Ndh
            return np.array([dNv, dNdn, dNdh], dtype=np.float64)

        return rhs

    # Pre-RT growth
    y0 = np.array([100.0, 0.0, 0.0], dtype=np.float64)
    t_eval_pre = np.linspace(0.0, t_pre_rt, 101)
    sol = solve_ivp(make_rhs(kpi, r2, 0.0, 0.0), [0.0, t_pre_rt], y0, t_eval=t_eval_pre, rtol=1e-7, atol=1e-9)
    t_all = [sol.t]
    y_all = [sol.y.T]
    t_cur = t_pre_rt
    y_cur = sol.y[:, -1].copy()

    # Record pre-RT volume
    V_pre_rt = float(np.sum(y_cur) * d_cell)

    for _ in range(n_fractions):
        Nv, Ndn, Ndh = y_cur
        _, fn, _ = _legacy_normoxic_fraction(Nv + Ndn + Ndh, d_cell, r2, preserve_legacy_typos)
        d_norm = (1.0 - Sn) * fn * Nv
        d_hyp = (1.0 - Sh) * (1.0 - fn) * Nv
        y_cur = np.array([Nv - d_norm - d_hyp, Ndn + d_norm, Ndh + d_hyp], dtype=np.float64)
        if dose_per_fraction > 0.0:
            kpi /= 1.2
            r2 *= 0.5
        t_seg = np.linspace(t_cur, t_cur + t_interval, 51)
        sol_f = solve_ivp(make_rhs(kpi, r2, 0.0, 0.0), [t_cur, t_cur + t_interval], y_cur, t_eval=t_seg, rtol=1e-7, atol=1e-9)
        t_all.append(sol_f.t[1:])
        y_all.append(sol_f.y.T[1:])
        t_cur += t_interval
        y_cur = sol_f.y[:, -1].copy()

    # Post-RT clearance phase
    t_seg = np.linspace(t_cur, t_cur + t_post_final, 151)
    sol_post = solve_ivp(make_rhs(kpi, r2, 0.05, 0.0), [t_cur, t_cur + t_post_final], y_cur, t_eval=t_seg, rtol=1e-7, atol=1e-9)
    t_all.append(sol_post.t[1:])
    y_all.append(sol_post.y.T[1:])

    t_arr = np.concatenate(t_all)
    y_arr = np.vstack(y_all)
    vol = np.sum(y_arr, axis=1) * d_cell
    mask_post = t_arr >= t_pre_rt
    return LegacySpheroidResult(
        t=t_arr,
        Nv=y_arr[:, 0],
        Ndn=y_arr[:, 1],
        Ndh=y_arr[:, 2],
        volume=vol,
        t_post_rt=t_arr[mask_post] - t_pre_rt,
        rel_volume=vol[mask_post] / max(V_pre_rt, 1e-12),
    )


# ---------------------------------------------------------------------------
# 2. Alex Fletcher's 2013 Untreated Oxygen-Diffusion Spheroid Model (CodeForBobby.m)
#    + Helen Byrne's Exact Analytical Greenspan Solutions (pcmi_proofs.pdf Sec. 4.3)
# ---------------------------------------------------------------------------
def solve_spherical_oxygen_bvp_step(
    R_um: float,
    D_um2_s: float = 1.5e3,
    gamma_mM_s: float = 1.0e-3,
    c_max: float = 0.21,
    c_N: float = 0.16,
    n_nodes: int = 121,
    steepness: float = 1500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast Newton-tridiagonal finite-volume solver for Alex Fletcher's quasi-steady spherical oxygen PDE:
        (D / r^2) d/dr (r^2 dc/dr) = gamma * H_smooth(c - c_N),  0 < r < R
        dc/dr(0) = 0,  c(R) = c_max
    Solves in normalized coordinates xi = r / R, u = c / c_max.
    """
    xi = np.linspace(0.0, 1.0, n_nodes)
    dxi = xi[1] - xi[0]
    u_N = c_N / c_max
    lam = (gamma_mM_s * R_um**2) / (D_um2_s * c_max)

    # Assemble discrete spherical Laplacian L u where (L u)_i = (1/xi_i^2) d/dxi(xi^2 du/dxi)
    # Using conservative finite-volume fluxes at half-nodes xi_{i +- 1/2}:
    # At i = 0 (symmetry, L'Hopital limit = 3 d^2u/dxi^2 -> 6*(u_1 - u_0)/dxi^2)
    main = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)
    lower = np.zeros(n_nodes - 1)

    main[0] = -6.0 / (dxi**2)
    upper[0] = 6.0 / (dxi**2)
    for i in range(1, n_nodes - 1):
        xp = xi[i] + 0.5 * dxi
        xm = xi[i] - 0.5 * dxi
        vol_i = (xp**3 - xm**3) / 3.0
        a_p = (xp**2) / (dxi * vol_i)
        a_m = (xm**2) / (dxi * vol_i)
        lower[i - 1] = a_m
        main[i] = -(a_p + a_m)
        upper[i] = a_p
    # Dirichlet at outer boundary i = n_nodes - 1: u[-1] = 1
    main[-1] = 1.0
    lower[-1] = 0.0

    L = diags([lower, main, upper], offsets=[-1, 0, 1], format="csc")

    # Smooth initial guess (parabolic)
    u = np.clip(1.0 - (lam / 6.0) * (1.0 - xi**2), u_N * 0.5, 1.0)
    u[-1] = 1.0

    # Damped Newton iterations for F(u) = L u - lam * H(u - u_N) = 0
    for _ in range(30):
        arg = np.clip(steepness * (u - u_N), -40.0, 40.0)
        H = 0.5 * (1.0 + np.tanh(arg))
        dH = 0.5 * steepness * (1.0 - np.tanh(arg) ** 2)

        F = L @ u - lam * H
        F[-1] = u[-1] - 1.0
        if np.max(np.abs(F)) < 1e-10:
            break

        J = L - diags(lam * dH, 0, format="csc")
        J = J.tolil()
        J[-1, :] = 0.0
        J[-1, -1] = 1.0
        du = spsolve(J.tocsc(), -F)
        # Backtracking line search to keep u non-negative and monotone
        step = 1.0
        norm_F = float(np.linalg.norm(F))
        for _ls in range(8):
            u_try = np.clip(u + step * du, 0.0, 1.0)
            u_try[-1] = 1.0
            arg_try = np.clip(steepness * (u_try - u_N), -40.0, 40.0)
            F_try = L @ u_try - lam * 0.5 * (1.0 + np.tanh(arg_try))
            F_try[-1] = u_try[-1] - 1.0
            if float(np.linalg.norm(F_try)) < norm_F:
                u = u_try
                break
            step *= 0.5
        else:
            u = np.clip(u + step * du, 0.0, 1.0)
            u[-1] = 1.0

    return xi * R_um, u * c_max


def analytical_byrne_greenspan_oxygen(
    R_um: float,
    D_um2_s: float = 1.5e3,
    gamma_mM_s: float = 1.0e-3,
    c_max: float = 0.21,
    c_H: float = 0.18,
    c_N: float = 0.16,
    n_nodes: int = 121,
) -> Dict[str, object]:
    """
    Exact analytical steady-state oxygen profile and hypoxic/necrotic radii from
    Helen Byrne's PCMI notes (pcmi_proofs.pdf, Section 4.3, Eqs. 44-50).
    """
    r = np.linspace(0.0, R_um, n_nodes)
    G = gamma_mM_s / D_um2_s  # mM / um^2
    c_center_no_nec = c_max - (G / 6.0) * R_um**2

    if c_center_no_nec > c_N:
        R_N = 0.0
        c = c_max - (G / 6.0) * (R_um**2 - r**2)
        if c_center_no_nec >= c_H:
            R_H = 0.0
        else:
            R_H = float(np.sqrt(max(R_um**2 - (6.0 / G) * (c_max - c_H), 0.0)))
    else:
        # Eq. (49): (1 - x)^2 * (1 + 2x) = 6*(c_max - c_N) / (G * R^2), x = R_N / R in (0, 1)
        rhs_n = (6.0 * (c_max - c_N)) / (G * R_um**2)
        x_n = brentq(lambda x: (1.0 - x) ** 2 * (1.0 + 2.0 * x) - rhs_n, 0.0, 1.0)
        R_N = float(x_n * R_um)
        c = np.where(
            r <= R_N,
            c_N,
            c_N + (G / 6.0) * ((r - R_N) ** 2 * (r + 2.0 * R_N)) / np.maximum(r, 1e-12),
        )
        # Eq. (50): find R_H in (R_N, R) where c(R_H) = c_H
        if c_H <= c_N:
            R_H = R_N
        else:
            R_H = float(
                brentq(
                    lambda rh: c_N + (G / 6.0) * ((rh - R_N) ** 2 * (rh + 2.0 * R_N)) / rh - c_H,
                    R_N + 1e-9,
                    R_um,
                )
            )

    return {"r_um": r, "c_mM": c, "R_H_um": R_H, "R_N_um": R_N}


def run_alex_fletcher_growth(
    time_end_days: float = 35.0,
    R0_um: float = 500.0,
    c_H: float = 0.18,
    c_N: float = 0.16,
    c_max: float = 0.21,
    s_per_day: float = 1.0,
    lambda_A: float = 0.0,
    lambda_N: float = 1.0,
    D_um2_s: float = 1.5e3,
    gamma_mM_s: float = 1.0e-3,
    n_steps: int = 201,
) -> Dict[str, np.ndarray]:
    """
    Bug-fixed Python implementation of Alex Fletcher's CodeForBobby.m (10 June 2013).
    Fixes the R_H=0 -> R_N=0 typo (lines 40 & 72) and evaluates oxygen at R(n+1) (line 59).
    """
    time = np.linspace(0.0, time_end_days, n_steps)
    dt = time[1] - time[0]
    R = np.zeros(n_steps)
    R_H = np.zeros(n_steps)
    R_N = np.zeros(n_steps)
    R[0] = R0_um
    V0 = (4.0 / 3.0) * np.pi * R0_um**3

    def eval_profile(R_val: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
        r, c = solve_spherical_oxygen_bvp_step(R_val, D_um2_s=D_um2_s, gamma_mM_s=gamma_mM_s, c_max=c_max, c_N=c_N)
        rh = float(np.interp(c_H, c, r)) if np.min(c) < c_H else 0.0
        rn = float(np.interp(c_N, c, r)) if np.min(c) < c_N else 0.0
        return r, c, rh, rn

    r, c, R_H[0], R_N[0] = eval_profile(R[0])
    for n in range(n_steps - 1):
        prolif = s_per_day * c * 0.5 * (1.0 + np.tanh(1000.0 * (c - c_H)))
        death = s_per_day * lambda_A + s_per_day * lambda_N * 0.5 * (1.0 + np.tanh(1000.0 * (c_N - c)))
        integrand = (prolif - death) * (r**2) / (R[n] ** 2)
        dR = float(np.trapezoid(integrand, r))
        R[n + 1] = max(R[n] + dt * dR, 10.0)
        r, c, R_H[n + 1], R_N[n + 1] = eval_profile(R[n + 1])

    V_norm = ((4.0 / 3.0) * np.pi * R**3) / V0
    return {"time_days": time, "R_um": R, "R_H_um": R_H, "R_N_um": R_N, "V_normalised": V_norm}
