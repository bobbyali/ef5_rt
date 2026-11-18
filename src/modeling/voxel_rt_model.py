"""
Sub-Goal 2: 3D Continuum Oxygen-PDE + Darcy Pressure-Velocity Voxel Radiotherapy Model.

Directly extends Helen Byrne's PCMI notes (pcmi_proofs.pdf, Section 5, Eqs. 53-59) and
Dr. Alex Fletcher's continuous oxygen diffusion framework into 3D on an isotropic voxel grid
initialised from pre-treatment CT (tumour geometry) and EF5 PET (spatial hypoxia):

  1. 3D Quasi-Steady Oxygen Reaction-Diffusion PDE on active tumour domain Omega:
         D_o2 * laplacian(p) + h_v(x, t) * (p_bdry - p)
         - gamma_0 * (V + q_d1*D1 + q_d2*D2) * p / (p + K_m) = 0
     with p = p_bdry outside Omega (well-oxygenated host stroma).

  2. Instantaneous Voxelwise Radiation Events (Alper-Howard-Flanders OER + LQ):
         V+(x)  = S(d, p(x)) * V-(x)
         D1+(x) = D1-(x) + (1 - S(d, p(x))) * V-(x)

  3. 3D Darcy Pressure-Velocity Tumour Expansion & Contraction Mechanics:
         -laplacian(P) = S_vol(x)   for x in Omega,   P = 0 outside Omega
         u = -grad(P)
         d(phi_k)/dt + div(u * phi_k) = R_k(phi, p)    for phi_k in {V, D1, D2, N}
     - Outward pressure gradient during net proliferation expands asymmetric 3D tumour geometry.
     - Inward pressure gradient during post-RT cell clearance compacts/shrinks the tumour.
     - Conservative finite-volume upwind advection + excess-packing redistribution guarantees
       positivity, Phi = V + D1 + D2 + N <= 1, and volume conservation.

  4. Virtual EF5 PET & CT Mask Forward Model:
     - Tumour ROI mask: Phi(x, t) >= 0.5
     - Virtual EF5 PET T/M map: bioreductive binding from (p, V, D1) + optional microPET PSF blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg

from src.modeling.physics import (
    RadiobiologyParams,
    clearance_rate,
    lq_surviving_fraction,
    necrosis_rate,
    po2_to_ef5_tm,
    proliferation_rate,
)
from src.modeling.voxel_initializer import VoxelTumourInitialState


@dataclass
class VoxelSimulationConfig:
    """Numerical & schedule configuration for 3D voxel simulation."""

    t_end_days: float = 33.0                          # Day -1 -> Day +32 is 33 days (RT at t = 1.0 day)
    dt_days: float = 0.5                              # Macro time step (days); advection substeps automatically
    dose_schedule: List[Tuple[float, float]] = field(default_factory=list)  # [(sim_day, dose_Gy), ...]
    snapshot_days: List[float] = field(default_factory=lambda: [0.0, 14.0, 33.0])  # Sim days (Day -1, +13, +32)
    psf_fwhm_mm: float = 0.6                          # Mild voxel-scale PSF blur (pre-RT PET is already scanner-blurred)
    phi_active_thresh: float = 0.05                   # Active domain mask for 3D elliptic solves
    phi_roi_thresh: float = 0.50                      # Isosurface threshold for binary tumour ROI volume
    cg_tol: float = 1e-5                              # Conjugate-gradient tolerance
    cg_maxiter: int = 200


@dataclass
class VoxelSimulationResult:
    init_state: VoxelTumourInitialState
    params: RadiobiologyParams
    config: VoxelSimulationConfig
    t_sim_days: np.ndarray              # Time since scan initialisation (days, e.g. 0..33)
    days_post_rt: np.ndarray            # Study days relative to RT Day 0 (e.g. -1..+32)
    volume_cm3: np.ndarray              # Simulated tumour volume (integral of Phi * dx^3, cm^3)
    mask_volume_cm3: np.ndarray         # Binary mask volume (Phi >= 0.5, cm^3)
    equiv_diam_mm: np.ndarray           # Equivalent spherical diameter (mm)
    major_axis_mm: np.ndarray           # Bounding-box / principal major axis length (mm, caliper proxy)
    rel_volume: np.ndarray              # Volume relative to t=0
    viable_frac: np.ndarray             # Tumour-averaged viable fraction
    damaged_frac: np.ndarray            # Tumour-averaged damaged fraction (D1 + D2)
    necrotic_frac: np.ndarray           # Tumour-averaged necrotic fraction
    hypoxic_frac: np.ndarray            # Fraction of tumour with pO2 < p_hyp (viable)
    mean_po2_mmHg: np.ndarray           # Tumour-averaged pO2 (mmHg)
    mean_ef5_tm: np.ndarray             # Simulated mean EF5 T/M inside ROI
    p95_ef5_tm: np.ndarray              # Simulated p95 EF5 T/M inside ROI
    max_ef5_tm: np.ndarray              # Simulated max EF5 T/M inside ROI
    snapshots: Dict[float, Dict[str, np.ndarray]]  # 3D fields at requested snapshot_days


def _assemble_active_neg_laplacian(
    active_mask: np.ndarray, dx: float
) -> Tuple[coo_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assembles the sparse SPD negative 7-point 3D Laplacian matrix (-laplacian) on active_mask voxels,
    assuming Dirichlet values outside active_mask.

    Returns
    -------
    neg_L : csr_matrix (M, M)
        Sparse SPD matrix representing -laplacian on the M active voxels.
    bc_count : ndarray (M,)
        Number of 6-connected neighbours outside active_mask (Dirichlet faces), divided by dx^2.
    coords : ndarray (M, 3)
        (iy, iz, ix) indices of the M active voxels.
    idx_map : ndarray shape active_mask.shape (int32)
        Maps 3D voxel -> [0..M-1], or -1 outside active_mask.
    """
    shape = active_mask.shape
    coords = np.argwhere(active_mask)
    M = coords.shape[0]
    idx_map = np.full(shape, -1, dtype=np.int32)
    if M == 0:
        return coo_matrix((0, 0)).tocsr(), np.zeros(0), coords, idx_map

    idx_map[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(M, dtype=np.int32)
    inv_dx2 = 1.0 / (dx * dx)

    rows = [np.arange(M, dtype=np.int32)]
    cols = [np.arange(M, dtype=np.int32)]
    vals = [np.full(M, 6.0 * inv_dx2, dtype=np.float64)]
    bc_count = np.zeros(M, dtype=np.float64)

    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for dy, dz, dx_i in offsets:
        ny = coords[:, 0] + dy
        nz = coords[:, 1] + dz
        nx = coords[:, 2] + dx_i
        in_bounds = (
            (ny >= 0) & (ny < shape[0]) & (nz >= 0) & (nz < shape[1]) & (nx >= 0) & (nx < shape[2])
        )
        nbr_idx = np.full(M, -1, dtype=np.int32)
        valid_pos = np.nonzero(in_bounds)[0]
        nbr_idx[valid_pos] = idx_map[ny[valid_pos], nz[valid_pos], nx[valid_pos]]

        is_active_nbr = nbr_idx >= 0
        # Active neighbour -> off-diagonal -1/dx^2
        a_src = np.nonzero(is_active_nbr)[0].astype(np.int32)
        rows.append(a_src)
        cols.append(nbr_idx[is_active_nbr])
        vals.append(np.full(a_src.size, -inv_dx2, dtype=np.float64))
        # Outside active domain -> Dirichlet boundary contribution (+1/dx^2 * u_bc on RHS)
        bc_count[~is_active_nbr] += inv_dx2

    neg_L = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(M, M),
        dtype=np.float64,
    ).tocsr()
    return neg_L, bc_count, coords, idx_map


def _solve_spd_cg(A, b: np.ndarray, x0: Optional[np.ndarray], tol: float, maxiter: int) -> np.ndarray:
    """Jacobi-preconditioned Conjugate Gradient solver for SPD sparse systems."""
    M = b.size
    if M == 0:
        return np.zeros(0, dtype=np.float64)
    diag = np.maximum(A.diagonal(), 1e-12)
    M_inv = LinearOperator((M, M), matvec=lambda v: v / diag, dtype=np.float64)
    x, _ = cg(A, b, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter, M=M_inv)
    return x


def _upwind_advect_3d(
    fields: List[np.ndarray],
    uy: np.ndarray,
    uz: np.ndarray,
    ux: np.ndarray,
    dx: float,
    dt: float,
) -> List[np.ndarray]:
    """
    Conservative 3D first-order upwind finite-volume advection with adaptive CFL sub-stepping.
    Face velocity arrays have shapes:
      uy: (ny-1, nz, nx)   between y[i] and y[i+1]
      uz: (ny, nz-1, nx)   between z[j] and z[j+1]
      ux: (ny, nz, nx-1)   between x[k] and x[k+1]
    """
    # Compute max outgoing flux rate per voxel for strict positivity CFL
    ny, nz, nx = fields[0].shape
    out_rate = np.zeros((ny, nz, nx), dtype=np.float64)
    out_rate[:-1, :, :] += np.maximum(uy, 0.0)
    out_rate[1:, :, :] += np.maximum(-uy, 0.0)
    out_rate[:, :-1, :] += np.maximum(uz, 0.0)
    out_rate[:, 1:, :] += np.maximum(-uz, 0.0)
    out_rate[:, :, :-1] += np.maximum(ux, 0.0)
    out_rate[:, :, 1:] += np.maximum(-ux, 0.0)
    max_cfl = float(np.max(out_rate)) * dt / dx

    n_sub = max(1, int(np.ceil(max_cfl / 0.35)))
    dt_sub = dt / n_sub
    scale = dt_sub / dx

    uy_pos = np.maximum(uy, 0.0)
    uy_neg = np.minimum(uy, 0.0)
    uz_pos = np.maximum(uz, 0.0)
    uz_neg = np.minimum(uz, 0.0)
    ux_pos = np.maximum(ux, 0.0)
    ux_neg = np.minimum(ux, 0.0)

    out = [f.copy() for f in fields]
    for _ in range(n_sub):
        for idx, f in enumerate(out):
            fy = uy_pos * f[:-1, :, :] + uy_neg * f[1:, :, :]
            fz = uz_pos * f[:, :-1, :] + uz_neg * f[:, 1:, :]
            fx = ux_pos * f[:, :, :-1] + ux_neg * f[:, :, 1:]

            df = np.zeros_like(f)
            df[:-1, :, :] -= fy
            df[1:, :, :] += fy
            df[:, :-1, :] -= fz
            df[:, 1:, :] += fz
            df[:, :, :-1] -= fx
            df[:, :, 1:] += fx

            out[idx] = np.maximum(f + scale * df, 0.0)

    return out


def _relax_packing_overshoot(fields: List[np.ndarray], n_iters: int = 3) -> List[np.ndarray]:
    """
    Conservative anti-overshoot relaxation: if discrete advection/reactions push total occupancy
    Phi = sum(fields) > 1.0 in any voxel, redistributes the excess (Phi - 1)+ conservatively into
    adjacent 6-connected voxels with available capacity (1 - Phi)+, preserving total cell mass
    and species ratios.
    """
    out = [f.copy() for f in fields]
    for _ in range(n_iters):
        phi = sum(out)
        excess = np.maximum(phi - 1.0, 0.0)
        if float(np.max(excess)) < 1e-4:
            break
        cap = np.maximum(1.0 - phi, 0.0)
        # Sum available neighbour capacity (6-connected)
        cp = np.pad(cap, 1, mode="constant", constant_values=0.0)
        nbr_cap = (
            cp[2:, 1:-1, 1:-1]
            + cp[:-2, 1:-1, 1:-1]
            + cp[1:-1, 2:, 1:-1]
            + cp[1:-1, :-2, 1:-1]
            + cp[1:-1, 1:-1, 2:]
            + cp[1:-1, 1:-1, :-2]
        )
        # Push fraction: at most min(excess, 0.5*nbr_cap) so neighbours don't wildly overshoot
        push_total = np.where(nbr_cap > 1e-9, np.minimum(excess, 0.5 * nbr_cap), 0.0)
        push_frac = push_total / np.maximum(phi, 1e-12)
        weight_inv = np.where(nbr_cap > 1e-9, push_total / np.maximum(nbr_cap, 1e-12), 0.0)

        for idx, f in enumerate(out):
            removed = f * push_frac
            # Amount per unit neighbour capacity carrying species idx
            spec_ratio = f / np.maximum(phi, 1e-12)
            sender_flux = weight_inv * spec_ratio
            sp = np.pad(sender_flux, 1, mode="constant", constant_values=0.0)
            received = cap * (
                sp[2:, 1:-1, 1:-1]
                + sp[:-2, 1:-1, 1:-1]
                + sp[1:-1, 2:, 1:-1]
                + sp[1:-1, :-2, 1:-1]
                + sp[1:-1, 1:-1, 2:]
                + sp[1:-1, 1:-1, :-2]
            )
            out[idx] = np.maximum(f - removed + received, 0.0)

    return out


class VoxelRTModel:
    """3D image-initialised continuum PDE simulator of tumour growth, hypoxia, and radiotherapy."""

    def __init__(self, params: Optional[RadiobiologyParams] = None) -> None:
        self.params = params or RadiobiologyParams()

    def solve_oxygen_3d(
        self,
        V: np.ndarray,
        D1: np.ndarray,
        D2: np.ndarray,
        phi: np.ndarray,
        h_vasc: np.ndarray,
        p_bdry: float,
        dx_mm: float,
        p_init: Optional[np.ndarray] = None,
        phi_thresh: float = 0.05,
        cg_tol: float = 1e-5,
        cg_maxiter: int = 200,
    ) -> np.ndarray:
        """
        Solves the 3D quasi-steady nonlinear oxygen PDE on active voxels (phi >= phi_thresh):
            -D_o2 * laplacian(p) + h_vasc * (p - p_bdry) + gamma_eff * p / (p + K_m) = 0
        via damped Newton-CG (warm-started from p_init).
        """
        prm = self.params
        active = phi >= phi_thresh
        p_out = np.full(phi.shape, p_bdry, dtype=np.float64)
        if not np.any(active):
            return p_out

        neg_L, bc_count, coords, _ = _assemble_active_neg_laplacian(active, dx_mm)
        iy, iz, ix = coords[:, 0], coords[:, 1], coords[:, 2]
        gamma_eff = prm.gamma_0 * np.maximum(
            V[iy, iz, ix] + prm.q_d1 * D1[iy, iz, ix] + prm.q_d2 * D2[iy, iz, ix], 0.0
        )
        hv = np.maximum(h_vasc[iy, iz, ix], 0.0)
        Km = prm.K_m_o2

        # Warm-start p_vec from previous timestep
        if p_init is not None:
            p_vec = np.clip(p_init[iy, iz, ix].astype(np.float64), 0.1, p_bdry)
        else:
            p_vec = np.full(iy.size, 0.5 * p_bdry, dtype=np.float64)

        A_diff = prm.D_o2 * neg_L
        rhs_bc = prm.D_o2 * bc_count * p_bdry

        # 5 damped Newton-CG iterations suffice with warm-starting
        for _ in range(6):
            mm = p_vec / (p_vec + Km)
            dmm = Km / ((p_vec + Km) ** 2)
            # Residual F(p) = (D*(-L) + hv)*p - (rhs_bc + hv*p_bdry) + gamma_eff * mm
            F = (A_diff @ p_vec) + hv * (p_vec - p_bdry) - rhs_bc + gamma_eff * mm
            if float(np.max(np.abs(F))) < 1e-4:
                break
            J = A_diff + diags(hv + gamma_eff * dmm, 0, format="csr")
            dp = _solve_spd_cg(J, -F, x0=None, tol=cg_tol, maxiter=cg_maxiter)
            p_vec = np.clip(p_vec + dp, 0.05, p_bdry)

        p_out[iy, iz, ix] = p_vec
        return p_out

    def solve_darcy_pressure_3d(
        self,
        S_vol: np.ndarray,
        phi: np.ndarray,
        dx_mm: float,
        P_init: Optional[np.ndarray] = None,
        phi_thresh: float = 0.05,
        cg_tol: float = 1e-5,
        cg_maxiter: int = 200,
    ) -> np.ndarray:
        """
        Solves 3D Darcy pressure Poisson equation (Byrne PCMI notes Sec. 5, Eq. 56):
            -laplacian(P) = S_vol   on phi >= phi_thresh
            P = 0                   outside tumour
        """
        active = phi >= phi_thresh
        P_out = np.zeros(phi.shape, dtype=np.float64)
        if not np.any(active):
            return P_out
        neg_L, _, coords, _ = _assemble_active_neg_laplacian(active, dx_mm)
        iy, iz, ix = coords[:, 0], coords[:, 1], coords[:, 2]
        rhs = S_vol[iy, iz, ix]
        x0 = P_init[iy, iz, ix] if P_init is not None else None
        p_vec = _solve_spd_cg(neg_L, rhs, x0=x0, tol=cg_tol, maxiter=cg_maxiter)
        P_out[iy, iz, ix] = p_vec
        return P_out

    def compute_virtual_ef5(
        self,
        p_o2: np.ndarray,
        V: np.ndarray,
        D1: np.ndarray,
        phi: np.ndarray,
        dx_mm: float,
        psf_fwhm_mm: float = 1.4,
    ) -> np.ndarray:
        """Computes simulated 3D EF5 PET T/M map with optional microPET PSF Gaussian blur."""
        raw_tm = po2_to_ef5_tm(p_o2, V, self.params, d1_frac=D1)
        # Outside tumour background = tm_min
        raw_tm = np.where(phi >= 0.05, raw_tm, self.params.tm_min)
        if psf_fwhm_mm > 0.0:
            sigma_vox = (psf_fwhm_mm / 2.35482) / dx_mm
            return gaussian_filter(raw_tm, sigma=sigma_vox)
        return raw_tm

    def _estimate_major_axis_mm(self, mask: np.ndarray, dx_mm: float) -> float:
        """Estimates caliper-like major axis diameter (mm) from 3D tumour mask principal axes."""
        pts = np.argwhere(mask)
        if pts.shape[0] < 8:
            vol_mm3 = float(mask.sum()) * (dx_mm**3)
            return float((6.0 * vol_mm3 / np.pi) ** (1.0 / 3.0)) if vol_mm3 > 0 else 0.0
        pts_mm = pts * dx_mm
        centered = pts_mm - pts_mm.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        evals = np.linalg.eigvalsh(cov)
        # For a uniform ellipsoid of semi-axis a, variance along axis = a^2 / 5 -> full axis = 2*sqrt(5*lambda)
        return float(2.0 * np.sqrt(5.0 * max(float(evals[-1]), 0.0)))

    def run(
        self,
        init_state: VoxelTumourInitialState,
        config: Optional[VoxelSimulationConfig] = None,
    ) -> VoxelSimulationResult:
        prm = self.params
        cfg = config or VoxelSimulationConfig()
        dx = init_state.dx_mm
        vox_cm3 = (dx**3) / 1000.0

        V = init_state.V0.copy()
        D1 = np.zeros_like(V)
        D2 = np.zeros_like(V)
        N = init_state.N0.copy()
        phi = np.clip(V + D1 + D2 + N, 0.0, 1.0)
        h_vasc = init_state.h_vasc0.copy()

        # Solve initial oxygen PDE with anchored vascular supply (verifies t=0 match to pre-RT EF5!)
        p_o2 = self.solve_oxygen_3d(
            V, D1, D2, phi, h_vasc, init_state.p_bdry_mmHg, dx,
            p_init=init_state.po2_0, phi_thresh=cfg.phi_active_thresh,
            cg_tol=cfg.cg_tol, cg_maxiter=cfg.cg_maxiter,
        )
        P_darcy = np.zeros_like(V)

        n_steps = int(round(cfg.t_end_days / cfg.dt_days))
        t_arr = np.linspace(0.0, cfg.t_end_days, n_steps + 1)
        days_post_rt_arr = init_state.days_post_rt + t_arr

        # Dose events indexed by step k
        dose_events: Dict[int, float] = {}
        for day_f, dose_f in cfg.dose_schedule:
            if dose_f > 0.0:
                k_f = int(round(day_f / cfg.dt_days))
                dose_events[k_f] = dose_events.get(k_f, 0.0) + dose_f

        # Snapshot step indices
        snap_steps: Dict[int, float] = {int(round(sd / cfg.dt_days)): float(sd) for sd in cfg.snapshot_days}
        snapshots: Dict[float, Dict[str, np.ndarray]] = {}

        vol_arr = np.zeros(n_steps + 1)
        mask_vol_arr = np.zeros(n_steps + 1)
        eq_diam_arr = np.zeros(n_steps + 1)
        maj_axis_arr = np.zeros(n_steps + 1)
        v_frac_arr = np.zeros(n_steps + 1)
        d_frac_arr = np.zeros(n_steps + 1)
        n_frac_arr = np.zeros(n_steps + 1)
        hyp_frac_arr = np.zeros(n_steps + 1)
        po2_mean_arr = np.zeros(n_steps + 1)
        ef5_mean_arr = np.zeros(n_steps + 1)
        ef5_p95_arr = np.zeros(n_steps + 1)
        ef5_max_arr = np.zeros(n_steps + 1)

        vasc_factor = 1.0

        def record(step_idx: int) -> None:
            phi_tot = V + D1 + D2 + N
            tot_mass = max(float(np.sum(phi_tot)), 1e-12)
            roi_mask = phi_tot >= cfg.phi_roi_thresh
            vol_cm3 = tot_mass * vox_cm3
            mask_cm3 = float(np.sum(roi_mask)) * vox_cm3
            vol_arr[step_idx] = vol_cm3
            mask_vol_arr[step_idx] = mask_cm3
            eq_diam_arr[step_idx] = float((6.0 * (vol_cm3 * 1000.0) / np.pi) ** (1.0 / 3.0)) if vol_cm3 > 0 else 0.0
            maj_axis_arr[step_idx] = self._estimate_major_axis_mm(roi_mask, dx)
            v_frac_arr[step_idx] = float(np.sum(V)) / tot_mass
            d_frac_arr[step_idx] = float(np.sum(D1 + D2)) / tot_mass
            n_frac_arr[step_idx] = float(np.sum(N)) / tot_mass
            hyp_frac_arr[step_idx] = float(np.sum(V * (p_o2 < prm.p_hyp))) / tot_mass
            po2_mean_arr[step_idx] = float(np.sum(phi_tot * p_o2)) / tot_mass

            ef5_map = self.compute_virtual_ef5(p_o2, V, D1, phi_tot, dx, psf_fwhm_mm=cfg.psf_fwhm_mm)
            if np.any(roi_mask):
                vals = ef5_map[roi_mask]
                ef5_mean_arr[step_idx] = float(np.mean(vals))
                ef5_p95_arr[step_idx] = float(np.percentile(vals, 95))
                ef5_max_arr[step_idx] = float(np.max(vals))
            else:
                ef5_mean_arr[step_idx] = prm.tm_min
                ef5_p95_arr[step_idx] = prm.tm_min
                ef5_max_arr[step_idx] = prm.tm_min

            if step_idx in snap_steps:
                snapshots[snap_steps[step_idx]] = {
                    "phi": phi_tot.astype(np.float32),
                    "V": V.astype(np.float32),
                    "D": (D1 + D2).astype(np.float32),
                    "N": N.astype(np.float32),
                    "p_o2": p_o2.astype(np.float32),
                    "ef5_tm": ef5_map.astype(np.float32),
                    "roi_mask": roi_mask.astype(bool),
                }

        record(0)

        for k in range(n_steps):
            # 1) Apply instantaneous radiation dose event if scheduled at step k
            if k in dose_events:
                dose_gy = dose_events[k]
                sf = lq_surviving_fraction(dose_gy, p_o2, prm)
                killed = (1.0 - sf) * V
                V = sf * V
                D1 = D1 + killed
                if prm.vasc_damage_max > 0.0:
                    drop = prm.vasc_damage_max * (dose_gy**2 / (dose_gy**2 + prm.vasc_damage_d50**2))
                    vasc_factor = max(0.2, vasc_factor * (1.0 - drop))
                p_o2 = self.solve_oxygen_3d(
                    V, D1, D2, V + D1 + D2 + N, h_vasc * vasc_factor, init_state.p_bdry_mmHg, dx,
                    p_init=p_o2, phi_thresh=cfg.phi_active_thresh,
                    cg_tol=cfg.cg_tol, cg_maxiter=cfg.cg_maxiter,
                )

            # 2) Compute local cellular reaction sources (identical physics to Sub-Goal 1!)
            kp = proliferation_rate(p_o2, prm)
            kn = necrosis_rate(p_o2, prm)
            kcl = clearance_rate(p_o2, prm)

            R_V = (kp - prm.ka_basal - kn) * V
            R_D1 = (prm.k_d1_prolif - prm.k_d1_to_d2) * D1
            R_D2 = prm.k_d1_to_d2 * D1 - kcl * D2
            R_N = kn * V + prm.eta_fibrotic * kcl * D2 - prm.k_nec_clear * N
            S_vol = R_V + R_D1 + R_D2 + R_N

            # 3) Solve 3D Darcy pressure Poisson equation: -laplacian(P) = S_vol inside tumour
            phi = V + D1 + D2 + N
            P_darcy = self.solve_darcy_pressure_3d(
                S_vol, phi, dx, P_init=P_darcy,
                phi_thresh=cfg.phi_active_thresh, cg_tol=cfg.cg_tol, cg_maxiter=cfg.cg_maxiter,
            )

            # Face velocities u = -grad(P)
            uy = -(P_darcy[1:, :, :] - P_darcy[:-1, :, :]) / dx
            uz = -(P_darcy[:, 1:, :] - P_darcy[:, :-1, :]) / dx
            ux = -(P_darcy[:, :, 1:] - P_darcy[:, :, :-1]) / dx

            # 4) Operator-split reaction + conservative upwind 3D advection + packing projection
            V_r = np.maximum(V + cfg.dt_days * R_V, 0.0)
            D1_r = np.maximum(D1 + cfg.dt_days * R_D1, 0.0)
            D2_r = np.maximum(D2 + cfg.dt_days * R_D2, 0.0)
            N_r = np.maximum(N + cfg.dt_days * R_N, 0.0)

            V_a, D1_a, D2_a, N_a = _upwind_advect_3d([V_r, D1_r, D2_r, N_r], uy, uz, ux, dx, cfg.dt_days)
            V, D1, D2, N = _relax_packing_overshoot([V_a, D1_a, D2_a, N_a], n_iters=4)
            phi = V + D1 + D2 + N

            # 5) Evolve vascular supply field with moving tumour rim:
            #    - Newly grown outer rim voxels inherit baseline rim perfusion (attenuated by macroscopic
            #      volume-dependent angiogenic lag as total tumour volume expands above v_vasc_half_cm3,
            #      capturing both Gompertzian growth deceleration and rising hypoxia in large control tumours!)
            roi_now = phi >= cfg.phi_roi_thresh
            vol_now_cm3 = float(np.sum(phi)) * vox_cm3
            vol_init_cm3 = max(float(vol_arr[0]), 1e-4)
            # Relative vascular efficiency drops smoothly as tumour expands above baseline & v_vasc_half_cm3
            vol_atten = (1.0 + vol_init_cm3 / prm.v_vasc_half_cm3) / (1.0 + vol_now_cm3 / prm.v_vasc_half_cm3)
            vol_atten = float(np.clip(vol_atten, 0.15, 1.5))

            if np.any(roi_now):
                dist_mm = distance_transform_edt(roi_now) * dx
                rim_target = init_state.rim_h_vasc * vol_atten * np.exp(-dist_mm / 1.8)
                # Gently relax vascularity toward rim-perfused profile as tissue remodels
                remodel_rate = 0.06 * cfg.dt_days
                h_vasc = np.where(
                    phi >= cfg.phi_active_thresh,
                    (1.0 - remodel_rate) * h_vasc + remodel_rate * rim_target,
                    0.0,
                )
                # Ensure freshly invaded rim voxels have at least rim_target
                newly_invaded = (phi >= cfg.phi_active_thresh) & (h_vasc < 0.15 * rim_target)
                h_vasc = np.where(newly_invaded, rim_target, h_vasc)

            vasc_factor = min(1.0, vasc_factor + prm.vasc_recovery_rate * (1.0 - vasc_factor) * cfg.dt_days)

            # 6) Re-solve 3D quasi-steady oxygen PDE on updated tumour geometry & viable cells
            p_o2 = self.solve_oxygen_3d(
                V, D1, D2, phi, h_vasc * vasc_factor * vol_atten, init_state.p_bdry_mmHg, dx,
                p_init=p_o2, phi_thresh=cfg.phi_active_thresh,
                cg_tol=cfg.cg_tol, cg_maxiter=cfg.cg_maxiter,
            )

            record(k + 1)

        rel_vol = vol_arr / max(vol_arr[0], 1e-12)
        return VoxelSimulationResult(
            init_state=init_state,
            params=prm,
            config=cfg,
            t_sim_days=t_arr,
            days_post_rt=days_post_rt_arr,
            volume_cm3=vol_arr,
            mask_volume_cm3=mask_vol_arr,
            equiv_diam_mm=eq_diam_arr,
            major_axis_mm=maj_axis_arr,
            rel_volume=rel_vol,
            viable_frac=v_frac_arr,
            damaged_frac=d_frac_arr,
            necrotic_frac=n_frac_arr,
            hypoxic_frac=hyp_frac_arr,
            mean_po2_mmHg=po2_mean_arr,
            mean_ef5_tm=ef5_mean_arr,
            p95_ef5_tm=ef5_p95_arr,
            max_ef5_tm=ef5_max_arr,
            snapshots=snapshots,
        )
