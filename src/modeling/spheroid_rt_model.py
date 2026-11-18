"""
Sub-Goal 1: Coupled 1D Radially-Symmetric Oxygen-Diffusion Spheroid Radiotherapy Model.

Integrates Dr. Alex Fletcher's continuous quasi-steady oxygen reaction-diffusion PDE
(NotesForBobby.pdf / Helen Byrne's PCMI notes Sec. 4) with multi-compartment radiation
response (viable V, early mitotic-catastrophe doomed D1, lysing D2, and necrotic/fibrotic N):

  1. Continuous quasi-steady spherical oxygen PDE on normalized radius xi = r / R(t) in [0, 1]:
         (D_o2 / R^2) (1/xi^2) d/dxi (xi^2 dp/dxi)
         + h_v(xi, t) * (p_bdry - p)
         - gamma_0 * (V + q_d1*D1 + q_d2*D2) * p / (p + K_m) = 0
     with dp/dxi(0) = 0 and p(1) = p_bdry(t).

  2. Exact moving-boundary Greenspan mass-balance ODE for tumour radius R(t):
         dR/dt = R * integral_0^1 S_net(xi, t) * xi^2 dxi
     where local radial cell velocity u(xi) satisfies (1/xi^2) d/dxi(xi^2 u) = R * S_net(xi),
     and volume fractions (V, D1, D2, N) summing identically to 1 are advected conservatively in
     the moving frame with relative velocity w(xi) = (u(xi) - xi * dR/dt) / R(t).

  3. Emergent radiobiology (no ad-hoc Vt brakes or binary OER hacks):
     - Natural growth slowdown as R grows and the hypoxic/necrotic core expands.
     - Continuous Alper-Howard-Flanders OER(pO2) + corrected LQ survival S(d, pO2).
     - Automatic post-RT reoxygenation as killed viable cells drop O2 consumption, abrogating
       EF5+ vs EF5- differences across fractionated schedules (2-4 fractions).
     - Persistent slowly-clearing necrotic/fibrotic core N naturally yields the ~65-75%
       residual volume floor at high single-fraction doses (1 x 40 Gy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from src.modeling.physics import (
    RadiobiologyParams,
    clearance_rate,
    lq_surviving_fraction,
    necrosis_rate,
    o2_consumption_mm,
    po2_to_ef5_tm,
    proliferation_rate,
    vascular_damage_fraction,
)


@dataclass
class SpheroidScenario:
    """Configuration for a single tumour phenotype + radiotherapy regimen."""

    name: str
    R0_mm: float = 4.0                           # Initial spherical radius at t=0 (mm; 4 mm -> 0.27 cm^3)
    p_bdry_mmHg: float = 35.0                    # Outer rim capillary pO2 (mmHg); lower -> more hypoxic EF5+
    h_vasc_scale: float = 1.0                    # Multiplier on internal vascular O2 supply (lower -> EF5+)
    core_vasc_floor: float = 0.15                # Relative core-to-rim perfusion at r=0 (high -> uniform EF5-, low -> rim-only EF5+)
    gamma_scale: float = 1.0                     # Multiplier on metabolic O2 consumption (higher -> EF5+)
    dose_schedule: List[Tuple[float, float]] = field(default_factory=list)  # [(day, dose_Gy), ...]
    t_pre_eq_days: float = 12.0                  # Pre-RT equilibration growth period to establish realistic radial shells
    t_end_days: float = 35.0                     # Days post-RT to simulate
    dt_days: float = 0.1                         # Time step (days)
    n_shells: int = 80                           # Number of radial finite-volume shells


@dataclass
class SpheroidSimulationResult:
    scenario: SpheroidScenario
    params: RadiobiologyParams
    t_days: np.ndarray                           # Time relative to RT Day 0 (days)
    R_mm: np.ndarray                             # Outer tumour radius (mm)
    volume_cm3: np.ndarray                       # Tumour volume (cm^3)
    rel_volume: np.ndarray                       # Volume normalized to Day 0 pre-RT
    viable_frac: np.ndarray                      # Tumour-averaged viable volume fraction
    damaged_frac: np.ndarray                     # Tumour-averaged RT-damaged fraction (D1 + D2)
    necrotic_frac: np.ndarray                    # Tumour-averaged necrotic/fibrotic fraction
    hypoxic_frac: np.ndarray                     # Fraction of tumour volume with pO2 < p_hyp (viable)
    mean_po2_mmHg: np.ndarray                    # Volume-weighted mean pO2 (mmHg)
    mean_ef5_tm: np.ndarray                      # Simulated volume-weighted mean EF5 T/M
    p95_ef5_tm: np.ndarray                       # Simulated 95th-percentile EF5 T/M
    xi_centers: np.ndarray                       # Normalized radial shell centers in (0, 1)
    snapshots: Dict[str, Dict[str, np.ndarray]]  # Radial profiles at key times (pre_rt, post_rt, d13, d32, end)


class SpheroidRTModel:
    """1D spherical finite-volume coupled oxygen-PDE + multi-compartment RT simulator."""

    def __init__(self, params: Optional[RadiobiologyParams] = None) -> None:
        self.params = params or RadiobiologyParams()

    def _build_spherical_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = np.linspace(0.0, 1.0, n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Exact spherical shell weights: integral_{xi_l}^{xi_r} xi^2 dxi = (xi_r^3 - xi_l^3) / 3
        shell_w = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0  # sum(shell_w) == 1/3
        return edges, centers, shell_w

    def solve_oxygen_profile(
        self,
        R_mm: float,
        phi_v: np.ndarray,
        phi_d1: np.ndarray,
        phi_d2: np.ndarray,
        p_bdry: float,
        h_vasc_eff: np.ndarray,
        gamma_scale: float = 1.0,
        p_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solves the quasi-steady spherical reaction-diffusion oxygen PDE on n finite-volume shells:
            (D_o2 / R^2) (1/xi^2) d/dxi (xi^2 dp/dxi)
            + h_vasc_eff * (p_bdry - p)
            - gamma_eff * p / (p + K_m) = 0
        with zero flux at xi = 0 and Dirichlet p = p_bdry at xi = 1.
        """
        prm = self.params
        n = phi_v.size
        edges, centers, shell_w = self._build_spherical_grid(n)
        dxi = edges[1] - edges[0]

        # Diffusion conductance matrix L (finite-volume discretization of (D/R^2) * div(grad p))
        diff_pref = prm.D_o2 / max(R_mm**2, 1e-6)
        main = np.zeros(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)
        rhs_bc = np.zeros(n)

        for i in range(n):
            vol_i = shell_w[i]
            # Inner face (i - 1/2): zero flux for i = 0
            if i > 0:
                xm = edges[i]
                c_m = diff_pref * (xm**2) / (dxi * vol_i)
                lower[i - 1] = c_m
                main[i] -= c_m
            # Outer face (i + 1/2): Dirichlet p = p_bdry at xi = 1 (distance dxi/2 from cell center)
            xp = edges[i + 1]
            if i < n - 1:
                c_p = diff_pref * (xp**2) / (dxi * vol_i)
                upper[i] = c_p
                main[i] -= c_p
            else:
                c_p = diff_pref * (xp**2) / (0.5 * dxi * vol_i)
                main[i] -= c_p
                rhs_bc[i] -= c_p * p_bdry

        L = diags([lower, main, upper], offsets=[-1, 0, 1], format="csc")

        gamma_eff = gamma_scale * prm.gamma_0 * np.maximum(
            phi_v + prm.q_d1 * phi_d1 + prm.q_d2 * phi_d2, 0.0
        )

        p = np.full(n, p_bdry, dtype=np.float64) if p_init is None else np.clip(p_init.copy(), 0.0, p_bdry)
        Km = prm.K_m_o2

        # Damped Newton solve for F(p) = L p - rhs_bc + h_vasc_eff*(p_bdry - p) - gamma_eff * p/(p + Km) = 0
        for _ in range(25):
            mm = p / (p + Km)
            dmm = Km / ((p + Km) ** 2)
            F = (L @ p) - rhs_bc + h_vasc_eff * (p_bdry - p) - gamma_eff * mm
            if np.max(np.abs(F)) < 1e-8:
                break
            diag_react = -(h_vasc_eff + gamma_eff * dmm)
            J = L + diags(diag_react, 0, format="csc")
            dp = spsolve(J, -F)
            step = 1.0
            norm_F = float(np.linalg.norm(F))
            for _ls in range(6):
                p_try = np.clip(p + step * dp, 0.0, p_bdry)
                F_try = (L @ p_try) - rhs_bc + h_vasc_eff * (p_bdry - p_try) - gamma_eff * (p_try / (p_try + Km))
                if float(np.linalg.norm(F_try)) < norm_F:
                    p = p_try
                    break
                step *= 0.5
            else:
                p = np.clip(p + step * dp, 0.0, p_bdry)

        return np.clip(p, 0.0, p_bdry)

    def _advect_spherical_shells(
        self,
        fields: List[np.ndarray],
        S_net: np.ndarray,
        R_mm: float,
        dt: float,
        edges: np.ndarray,
        shell_w: np.ndarray,
    ) -> Tuple[List[np.ndarray], float]:
        """
        Exact conservative finite-volume advection in the moving normalized spherical domain xi = r / R(t).

        In physical coordinates:
            (1/r^2) d/dr (r^2 u) = S_net(r)  =>  xi^2 * (u(xi)/R) = integral_0^xi S_net(s) s^2 ds
            dR/dt = u(R) = R * (3 * integral_0^1 S_net(s) s^2 ds)

        Transforming d/dt(phi) + (1/r^2) d/dr (r^2 u phi) = R_phi to xi = r/R(t):
            d/dt(R^3 phi) + (R^3 / xi^2) d/dxi [ xi^2 * ((u - xi * dR/dt) / R) * phi ] = R^3 * R_phi

        At each interior face xi_f = edges[k], the dimensionless face flux rate is:
            F_face[k] = xi_f^2 * (u(xi_f)/R - xi_f * (dR/dt)/R)
        which vanishes identically at xi = 0 and xi = 1 (closed boundary!).
        """
        n = shell_w.size
        # Cumulative volume source integral_0^{edges[k]} S_net * s^2 ds
        cum_int = np.concatenate([[0.0], np.cumsum(S_net * shell_w)])
        # Exact spherical kinematics:
        #   dR/dt / R = integral_0^1 S_net(xi) xi^2 dxi = cum_int[-1]
        #   dV/dt / V = 3 * dR/dt / R = 3 * cum_int[-1]
        r_dot_over_r = float(cum_int[-1])
        v_dot_over_v = 3.0 * r_dot_over_r
        # Relative radial face flux: xi_f^2 * w_rel(xi_f) = cum_int[k] - edges[k]^3 * cum_int[-1]
        # Notice at k=0: 0 - 0 = 0; at k=n: cum_int[-1] - 1 * cum_int[-1] = 0!
        face_flux_rate = cum_int - (edges**3) * cum_int[-1]

        # CFL sub-stepping for strict positivity & TVD stability
        max_outflow = np.max(np.abs(face_flux_rate[1:-1]) / np.minimum(shell_w[:-1], shell_w[1:])) if n > 1 else 0.0
        n_sub = max(1, int(np.ceil(2.5 * dt * max_outflow)))
        dt_sub = dt / n_sub

        updated = [f.copy() for f in fields]
        for _ in range(n_sub):
            for idx, f in enumerate(updated):
                # First-order upwind face values on interior faces 1..n-1
                q_int = face_flux_rate[1:-1]
                f_upwind = np.where(q_int >= 0.0, f[:-1], f[1:])
                flux = q_int * f_upwind  # length n-1
                div_flux = np.zeros(n)
                div_flux[:-1] += flux / shell_w[:-1]
                div_flux[1:] -= flux / shell_w[1:]
                # Advective update in moving xi frame:
                #   df/dt|_adv = -div_flux + (S_net - 3*dR/dt/R) * f
                # For constant f = 1: div_flux(1) == S_net - 3*dR/dt/R, so d(1)/dt == 0 identically.
                updated[idx] = np.maximum(f + dt_sub * (-div_flux + (S_net - v_dot_over_v) * f), 0.0)

            # Exact renormalization to remove O(1e-15) floating drift
            total = sum(updated)
            total = np.maximum(total, 1e-12)
            updated = [f / total for f in updated]

        R_new = max(R_mm * float(np.exp(r_dot_over_r * dt)), 0.05)
        return updated, R_new

    @staticmethod
    def _solve_spherical_darcy_pressure(
        S_net: np.ndarray,
        R_mm: float,
        edges: np.ndarray,
        centers: np.ndarray,
        shell_w: np.ndarray,
    ) -> np.ndarray:
        """
        Exact 1D spherical Darcy pressure profile solving:
            -(1 / (R^2 xi^2)) d/dxi (xi^2 dP/dxi) = S_net(xi),   P'(0) = 0,  P(1) = 0
        Matches the 3D Darcy equation -laplacian(P) = S_vol identically in spherical symmetry.
        """
        dxi = edges[1] - edges[0]
        cum_int = np.concatenate([[0.0], np.cumsum(S_net * shell_w)])
        # Mid-shell integral_0^{xi_i} S(s) s^2 ds / xi_i^2
        mid_int = 0.5 * (cum_int[:-1] + cum_int[1:])
        dp_dxi_neg = (R_mm**2) * mid_int / np.maximum(centers**2, 1e-12)
        # Integrate inward from xi = 1 (P = 0) to xi = centers[i]
        P_from_rim = np.cumsum(dp_dxi_neg[::-1] * dxi)[::-1] - 0.5 * dp_dxi_neg * dxi
        return P_from_rim

    def _step(
        self,
        R_mm: float,
        phi_v: np.ndarray,
        phi_d1: np.ndarray,
        phi_d2: np.ndarray,
        phi_n: np.ndarray,
        p_o2: np.ndarray,
        P_darcy: np.ndarray,
        vasc_factor: float,
        dt: float,
        edges: np.ndarray,
        centers: np.ndarray,
        shell_w: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prm = self.params
        vol_cm3 = (4.0 / 3.0) * np.pi * (R_mm**3) / 1000.0
        kp = proliferation_rate(p_o2, prm, vol_cm3=vol_cm3, pressure=P_darcy)
        kn = necrosis_rate(p_o2, prm)
        boost = prm.high_dose_lysis_boost * max(1.0 - vasc_factor, 0.0)
        kcl = clearance_rate(p_o2, prm) + 0.5 * boost
        k_nec_eff = prm.k_nec_clear + boost

        # Local cellular reaction rates R_i (volume created/destroyed per unit volume per day)
        R_v = (kp - prm.ka_basal - kn) * phi_v
        R_d1 = (prm.k_d1_prolif - prm.k_d1_to_d2) * phi_d1
        R_d2 = prm.k_d1_to_d2 * phi_d1 - kcl * phi_d2
        R_n = kn * phi_v + prm.eta_fibrotic * kcl * phi_d2 - k_nec_eff * phi_n

        # Net local volume source S_net = sum(R_i)
        S_net = R_v + R_d1 + R_d2 + R_n
        P_new = self._solve_spherical_darcy_pressure(S_net, R_mm, edges, centers, shell_w)

        # Operator-split update:
        # 1) Local reaction update for concentrations in fixed volume:
        #    d(phi_i)/dt = R_i - S_net * phi_i  (preserves sum(phi_i) == 1!)
        phi_v_r = np.maximum(phi_v + dt * (R_v - S_net * phi_v), 0.0)
        phi_d1_r = np.maximum(phi_d1 + dt * (R_d1 - S_net * phi_d1), 0.0)
        phi_d2_r = np.maximum(phi_d2 + dt * (R_d2 - S_net * phi_d2), 0.0)
        phi_n_r = np.maximum(phi_n + dt * (R_n - S_net * phi_n), 0.0)
        tot = np.maximum(phi_v_r + phi_d1_r + phi_d2_r + phi_n_r, 1e-12)
        phi_v_r, phi_d1_r, phi_d2_r, phi_n_r = phi_v_r / tot, phi_d1_r / tot, phi_d2_r / tot, phi_n_r / tot

        # 2) Conservative spherical advection + radius update
        [phi_v_new, phi_d1_new, phi_d2_new, phi_n_new], R_new = self._advect_spherical_shells(
            [phi_v_r, phi_d1_r, phi_d2_r, phi_n_r], S_net, R_mm, dt, edges, shell_w
        )
        return R_new, phi_v_new, phi_d1_new, phi_d2_new, phi_n_new, P_new

    def run(self, scenario: SpheroidScenario) -> SpheroidSimulationResult:
        prm = self.params
        edges, centers, shell_w = self._build_spherical_grid(scenario.n_shells)
        vol_weights = 3.0 * shell_w  # Normalized spherical volume weights summing to 1.0

        # Rim-biased vascular exchange profile (highest near perfused rim, core_vasc_floor at center)
        c_floor = float(np.clip(scenario.core_vasc_floor, 0.0, 1.0))
        h_profile_base = prm.h_vasc_base * scenario.h_vasc_scale * (c_floor + (1.0 - c_floor) * (centers**2))

        def _h_eff(R_cur: float, v_fac: float = 1.0, p_mech: Optional[np.ndarray] = None) -> np.ndarray:
            vol_cm3_cur = (4.0 / 3.0) * np.pi * (R_cur**3) / 1000.0
            vol_atten = 1.0 / (1.0 + vol_cm3_cur / max(prm.v_vasc_half_cm3, 1e-3))
            # Byrne/Jain solid-stress vascular compression: positive Darcy pressure compresses capillaries;
            # post-RT pressure drop (P <= 0) mechanically decompresses vessels -> acute reoxygenation!
            if p_mech is not None and prm.p_darcy_crit > 0.0:
                patency = 1.0 / (1.0 + 3.0 * np.maximum(p_mech, 0.0) / prm.p_darcy_crit)
            else:
                patency = 1.0
            return h_profile_base * vol_atten * v_fac * patency

        # --- Stage A: Pre-RT equilibration to build self-consistent viable/hypoxic/necrotic shells ---
        # Start slightly smaller and grow until reaching ~R0_mm at t = 0
        R = scenario.R0_mm * 0.72
        phi_v = np.ones(scenario.n_shells, dtype=np.float64)
        phi_d1 = np.zeros(scenario.n_shells, dtype=np.float64)
        phi_d2 = np.zeros(scenario.n_shells, dtype=np.float64)
        phi_n = np.zeros(scenario.n_shells, dtype=np.float64)
        P_darcy = np.zeros(scenario.n_shells, dtype=np.float64)
        p_o2 = self.solve_oxygen_profile(R, phi_v, phi_d1, phi_d2, scenario.p_bdry_mmHg, _h_eff(R, 1.0, P_darcy), scenario.gamma_scale)

        n_pre = int(round(scenario.t_pre_eq_days / scenario.dt_days))
        for _ in range(n_pre):
            R, phi_v, phi_d1, phi_d2, phi_n, P_darcy = self._step(
                R, phi_v, phi_d1, phi_d2, phi_n, p_o2, P_darcy, 1.0, scenario.dt_days, edges, centers, shell_w
            )
            p_o2 = self.solve_oxygen_profile(
                R, phi_v, phi_d1, phi_d2, scenario.p_bdry_mmHg, _h_eff(R, 1.0, P_darcy), scenario.gamma_scale, p_init=p_o2
            )

        # Rescale outer radius cleanly to exact scenario.R0_mm at Day 0 while keeping equilibrated shell structure
        R = scenario.R0_mm
        p_o2 = self.solve_oxygen_profile(
            R, phi_v, phi_d1, phi_d2, scenario.p_bdry_mmHg, _h_eff(R, 1.0, P_darcy), scenario.gamma_scale, p_init=p_o2
        )

        # --- Stage B: Post-RT evolution (t = 0 .. t_end_days) ---
        n_steps = int(round(scenario.t_end_days / scenario.dt_days))
        t_arr = np.linspace(0.0, scenario.t_end_days, n_steps + 1)

        R_arr = np.zeros(n_steps + 1)
        v_frac_arr = np.zeros(n_steps + 1)
        d_frac_arr = np.zeros(n_steps + 1)
        n_frac_arr = np.zeros(n_steps + 1)
        hyp_frac_arr = np.zeros(n_steps + 1)
        po2_mean_arr = np.zeros(n_steps + 1)
        ef5_mean_arr = np.zeros(n_steps + 1)
        ef5_p95_arr = np.zeros(n_steps + 1)

        # Prepare dose event index map
        dose_events: Dict[int, float] = {}
        for day_f, dose_f in scenario.dose_schedule:
            if dose_f > 0.0:
                step_idx = int(round(day_f / scenario.dt_days))
                dose_events[step_idx] = dose_events.get(step_idx, 0.0) + dose_f

        vasc_factor = 1.0
        snapshots: Dict[str, Dict[str, np.ndarray]] = {}

        def record_state(idx: int) -> None:
            ef5_prof = po2_to_ef5_tm(p_o2, phi_v, prm, d1_frac=phi_d1)
            R_arr[idx] = R
            v_frac_arr[idx] = float(np.sum(vol_weights * phi_v))
            d_frac_arr[idx] = float(np.sum(vol_weights * (phi_d1 + phi_d2)))
            n_frac_arr[idx] = float(np.sum(vol_weights * phi_n))
            hyp_frac_arr[idx] = float(np.sum(vol_weights * phi_v * (p_o2 < prm.p_hyp)))
            po2_mean_arr[idx] = float(np.sum(vol_weights * p_o2))
            ef5_mean_arr[idx] = float(np.sum(vol_weights * ef5_prof))
            # Volume-weighted 95th percentile of EF5 T/M
            order = np.argsort(ef5_prof)
            cdf = np.cumsum(vol_weights[order])
            ef5_p95_arr[idx] = float(np.interp(0.95, cdf, ef5_prof[order]))

        def save_snapshot(tag: str) -> None:
            snapshots[tag] = {
                "r_mm": centers * R,
                "p_o2": p_o2.copy(),
                "phi_v": phi_v.copy(),
                "phi_d": (phi_d1 + phi_d2).copy(),
                "phi_n": phi_n.copy(),
                "ef5_tm": po2_to_ef5_tm(p_o2, phi_v, prm, d1_frac=phi_d1),
            }

        save_snapshot("pre_rt")
        record_state(0)

        for k in range(n_steps):
            # Apply scheduled RT fraction at start of step k
            if k in dose_events:
                dose_gy = dose_events[k]
                sf = lq_surviving_fraction(dose_gy, p_o2, prm)
                killed = (1.0 - sf) * phi_v
                phi_v = sf * phi_v
                phi_d1 = phi_d1 + killed
                drop = vascular_damage_fraction(dose_gy, prm)
                if drop > 0.0:
                    vasc_factor = max(0.15, vasc_factor * (1.0 - drop))
                p_bdry_eff = scenario.p_bdry_mmHg * (0.35 + 0.65 * vasc_factor)
                # Re-solve quasi-steady oxygen immediately post-RT
                p_o2 = self.solve_oxygen_profile(
                    R, phi_v, phi_d1, phi_d2, p_bdry_eff, _h_eff(R, vasc_factor, P_darcy), scenario.gamma_scale, p_init=p_o2
                )
                if k == 0:
                    save_snapshot("post_rt_day0")

            # Advance cellular dynamics + moving boundary
            R, phi_v, phi_d1, phi_d2, phi_n, P_darcy = self._step(
                R, phi_v, phi_d1, phi_d2, phi_n, p_o2, P_darcy, vasc_factor, scenario.dt_days, edges, centers, shell_w
            )
            # Vascular recovery
            vasc_factor = min(1.0, vasc_factor + prm.vasc_recovery_rate * (1.0 - vasc_factor) * scenario.dt_days)
            p_bdry_eff = scenario.p_bdry_mmHg * (0.35 + 0.65 * vasc_factor)
            p_o2 = self.solve_oxygen_profile(
                R, phi_v, phi_d1, phi_d2, p_bdry_eff, _h_eff(R, vasc_factor, P_darcy), scenario.gamma_scale, p_init=p_o2
            )
            record_state(k + 1)

            t_next = t_arr[k + 1]
            if abs(t_next - 2.0) < 0.5 * scenario.dt_days:
                save_snapshot("day2")
            if abs(t_next - 13.0) < 0.5 * scenario.dt_days:
                save_snapshot("day13")
            if abs(t_next - 32.0) < 0.5 * scenario.dt_days:
                save_snapshot("day32")

        save_snapshot("end")
        vol_cm3 = (4.0 / 3.0) * np.pi * (R_arr**3) / 1000.0
        rel_vol = vol_cm3 / max(vol_cm3[0], 1e-12)

        return SpheroidSimulationResult(
            scenario=scenario,
            params=prm,
            t_days=t_arr,
            R_mm=R_arr,
            volume_cm3=vol_cm3,
            rel_volume=rel_vol,
            viable_frac=v_frac_arr,
            damaged_frac=d_frac_arr,
            necrotic_frac=n_frac_arr,
            hypoxic_frac=hyp_frac_arr,
            mean_po2_mmHg=po2_mean_arr,
            mean_ef5_tm=ef5_mean_arr,
            p95_ef5_tm=ef5_p95_arr,
            xi_centers=centers,
            snapshots=snapshots,
        )


def make_plos_one_scenarios(
    dose_schedule: Sequence[Tuple[float, float]],
    label: str,
    t_end_days: float = 35.0,
) -> Dict[str, SpheroidScenario]:
    """
    Creates matched Control, EF5+ (hypoxic), and EF5- (normoxic) spheroid scenarios for a given
    PLOS ONE radiotherapy regimen.

    All three phenotypes share the same physiological host boundary pO2 (35 mmHg) and differ ONLY
    via internal tumour microvascular perfusion & metabolic demand (no OER hacks!):
      - Control (population average): h_vasc_scale = 2.2, core_vasc_floor = 0.35
      - EF5- (well-perfused core+rim): h_vasc_scale = 3.5, core_vasc_floor = 0.85
      - EF5+ (viable chronic hypoxia): h_vasc_scale = 0.88, core_vasc_floor = 0.58
    """
    sched = list(dose_schedule)
    p_host = RadiobiologyParams().p_host_bdry
    return {
        "Control": SpheroidScenario(
            name=f"{label} Control",
            R0_mm=4.2,
            p_bdry_mmHg=p_host,
            h_vasc_scale=2.2,
            core_vasc_floor=0.35,
            gamma_scale=1.0,
            dose_schedule=[],
            t_end_days=t_end_days,
        ),
        "EF5-": SpheroidScenario(
            name=f"{label} EF5- (Normoxic)",
            R0_mm=4.2,
            p_bdry_mmHg=p_host,
            h_vasc_scale=3.5,
            core_vasc_floor=0.85,
            gamma_scale=0.85,
            dose_schedule=sched,
            t_end_days=t_end_days,
        ),
        "EF5+": SpheroidScenario(
            name=f"{label} EF5+ (Hypoxic)",
            R0_mm=4.2,
            p_bdry_mmHg=p_host,
            h_vasc_scale=1.15,
            core_vasc_floor=0.60,
            gamma_scale=1.18,
            dose_schedule=sched,
            t_end_days=t_end_days,
        ),
    }
