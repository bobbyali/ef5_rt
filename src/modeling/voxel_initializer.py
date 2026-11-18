"""
Sub-Goal 2: 3D Tumour-Specific Image Initialiser from EF5 PET/CT Voxel Bundles (.npz).

Loads a pre-treatment mouse scan from outputs/first_ef5_rt_study/voxel_bundles/<mouse>/<scan>.npz
(or any future cohort scan following the same schema), extracts a padded isotropic 3D simulation
grid around a single tumour (RT or Control), initialises:
  1. 3D tumour geometry Phi_0(x) from the native high-res CT mask (falling back to PET mask when PET-only),
  2. 3D spatial oxygen pO2_0(x) from the muscle-normalized EF5 PET T/M map,
  3. 3D viable/necrotic split (V_0, N_0), and
  4. A smooth tumour-specific vascular supply field h_v(x) anchored so the 3D quasi-steady oxygen PDE
     reproduces the observed pre-RT spatial pO2 distribution at t = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

from src.modeling.physics import RadiobiologyParams, ef5_tm_to_po2, o2_consumption_mm


@dataclass
class VoxelTumourInitialState:
    """Isotropic 3D initial condition for a single mouse tumour."""

    mouse_id: str
    tumour_id: str
    treatment: str
    dose_gy: float
    scan_date: str
    days_post_rt: int
    source_npz: str
    geometry_source: str              # "CT" or "PET"
    dx_mm: float                      # Isotropic simulation voxel spacing (mm)
    shape: Tuple[int, int, int]       # (ny_sim, nz_sim, nx_sim) matching canonical (Y, Z, X)
    # Physical world coordinates of simulation grid axes (ascending mm)
    y_mm: np.ndarray                  # axis 0
    z_mm: np.ndarray                  # axis 1
    x_mm: np.ndarray                  # axis 2
    # Initial 3D fields on the simulation grid
    phi0: np.ndarray                  # Initial total tumour occupancy in [0, 1]
    V0: np.ndarray                    # Initial viable cell occupancy
    N0: np.ndarray                    # Initial necrotic cell occupancy
    pet_tm0: np.ndarray               # Resampled observed pre-RT EF5 PET T/M map
    pet_idg0: np.ndarray              # Resampled observed pre-RT EF5 PET %ID/g map
    po2_0: np.ndarray                 # Initial 3D pO2 field (mmHg) from EF5 PET
    h_vasc0: np.ndarray               # Anchored 3D vascular O2 supply field
    rim_h_vasc: float                 # Characteristic baseline rim perfusion for newly grown tissue
    p_bdry_mmHg: float                # Effective boundary oxygen concentration (mmHg)
    obs_volume_cm3: float             # Observed initial tumour volume from source ROI (cm^3)
    obs_tm_mean: float                # Observed mean EF5 T/M inside ROI
    obs_tm_p95: float                 # Observed p95 EF5 T/M inside ROI


def _make_interpolator(
    arr: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray, x_coords: np.ndarray, fill_value: float = 0.0
) -> RegularGridInterpolator:
    """Builds a 3D RegularGridInterpolator handling descending RT_Image physical axes cleanly."""
    data = np.asarray(arr, dtype=np.float64)
    yc = np.asarray(y_coords, dtype=np.float64)
    zc = np.asarray(z_coords, dtype=np.float64)
    xc = np.asarray(x_coords, dtype=np.float64)

    if yc.size > 1 and yc[0] > yc[-1]:
        yc = yc[::-1]
        data = data[::-1, :, :]
    if zc.size > 1 and zc[0] > zc[-1]:
        zc = zc[::-1]
        data = data[:, ::-1, :]
    if xc.size > 1 and xc[0] > xc[-1]:
        xc = xc[::-1]
        data = data[:, :, ::-1]

    return RegularGridInterpolator(
        (yc, zc, xc),
        data,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )


def _laplacian_3d(u: np.ndarray, dx: float) -> np.ndarray:
    """Standard 7-point 3D finite-difference Laplacian with zero-Neumann padding at domain boundary."""
    p = np.pad(u, 1, mode="edge")
    return (
        p[2:, 1:-1, 1:-1]
        + p[:-2, 1:-1, 1:-1]
        + p[1:-1, 2:, 1:-1]
        + p[1:-1, :-2, 1:-1]
        + p[1:-1, 1:-1, 2:]
        + p[1:-1, 1:-1, :-2]
        - 6.0 * u
    ) / (dx**2)


def initialize_tumour_from_npz(
    npz_path: Path,
    side: str,
    params: Optional[RadiobiologyParams] = None,
    dx_mm: float = 0.5,
    pad_mm: float = 8.0,
    dose_gy_override: Optional[float] = None,
) -> VoxelTumourInitialState:
    """
    Initialises a 3D tumour simulation grid from a First EF5 RT Study `.npz` voxel bundle.

    Parameters
    ----------
    npz_path : Path
        Path to `<mouse_id>_<timepoint>.npz`.
    side : {"rt", "ctrl"}
        Which tumour to extract from the bilateral mouse scan.
    params : RadiobiologyParams
        Shared radiobiology/oxygen parameters.
    dx_mm : float
        Isotropic simulation voxel spacing in mm (default 0.5 mm).
    pad_mm : float
        Physical padding around initial tumour bounding box in mm to accommodate growth.
    """
    prm = params or RadiobiologyParams()
    side_key = side.lower().strip()
    if side_key not in ("rt", "ctrl"):
        raise ValueError(f"side must be 'rt' or 'ctrl', got {side!r}")

    d = np.load(npz_path, allow_pickle=False)
    mouse_id = str(d["mouse_id"].item())
    scan_date = str(d["scan_date"].item())
    days_post_rt = int(d["days_post_rt"].item())
    ct_avail = bool(d["ct_available"].item())

    has_roi = bool(d["has_rt_roi"].item()) if side_key == "rt" else bool(d["has_ctrl_roi"].item())
    if not has_roi:
        raise ValueError(f"Scan {npz_path.name} does not contain a {side_key} tumour ROI.")

    pet_tm = np.asarray(d["pet_tm"], dtype=np.float64)
    pet_idg = np.asarray(d["pet_idg"], dtype=np.float64)
    mask_pet = np.asarray(d[f"mask_{side_key}_pet"], dtype=bool)
    pet_spacing = np.asarray(d["pet_spacing_mm"], dtype=np.float64)
    pet_y = np.asarray(d["pet_y_mm"], dtype=np.float64)
    pet_z = np.asarray(d["pet_z_mm"], dtype=np.float64)
    pet_x = np.asarray(d["pet_x_mm"], dtype=np.float64)

    # Observed PET statistics directly from native PET ROI
    pet_vals_tm = pet_tm[mask_pet]
    obs_tm_mean = float(np.mean(pet_vals_tm))
    obs_tm_p95 = float(np.percentile(pet_vals_tm, 95))
    obs_pet_vol_cm3 = float(mask_pet.sum() * np.prod(pet_spacing) / 1000.0)

    # Choose high-res CT mask for tumour geometry when available and non-empty
    use_ct_geom = ct_avail and (f"mask_{side_key}_ct" in d) and bool(np.any(d[f"mask_{side_key}_ct"]))
    if use_ct_geom:
        mask_src = np.asarray(d[f"mask_{side_key}_ct"], dtype=bool)
        src_y = np.asarray(d["ct_y_mm"], dtype=np.float64)
        src_z = np.asarray(d["ct_z_mm"], dtype=np.float64)
        src_x = np.asarray(d["ct_x_mm"], dtype=np.float64)
        src_spacing = np.asarray(d["ct_spacing_mm"], dtype=np.float64)
        obs_vol_cm3 = float(mask_src.sum() * np.prod(src_spacing) / 1000.0)
        geom_src = "CT"
    else:
        mask_src = mask_pet
        src_y, src_z, src_x = pet_y, pet_z, pet_x
        obs_vol_cm3 = obs_pet_vol_cm3
        geom_src = "PET"

    # Compute physical bounding box of the tumour in (Y, Z, X)
    idx_y, idx_z, idx_x = np.nonzero(mask_src)
    y_lo, y_hi = float(np.min(src_y[idx_y])), float(np.max(src_y[idx_y]))
    z_lo, z_hi = float(np.min(src_z[idx_z])), float(np.max(src_z[idx_z]))
    x_lo, x_hi = float(np.min(src_x[idx_x])), float(np.max(src_x[idx_x]))

    # Construct ascending isotropic simulation coordinate grid with generous padding
    y_sim = np.arange(y_lo - pad_mm, y_hi + pad_mm + 0.5 * dx_mm, dx_mm, dtype=np.float64)
    z_sim = np.arange(z_lo - pad_mm, z_hi + pad_mm + 0.5 * dx_mm, dx_mm, dtype=np.float64)
    x_sim = np.arange(x_lo - pad_mm, x_hi + pad_mm + 0.5 * dx_mm, dx_mm, dtype=np.float64)
    grid_shape = (y_sim.size, z_sim.size, x_sim.size)
    YY, ZZ, XX = np.meshgrid(y_sim, z_sim, x_sim, indexing="ij")
    pts = np.stack([YY.ravel(), ZZ.ravel(), XX.ravel()], axis=-1)

    # Smoothly resample binary mask onto simulation grid (pre-blur slightly if CT spacing << dx_mm for anti-aliasing)
    mask_float = mask_src.astype(np.float64)
    if use_ct_geom:
        sigma_vox = max(0.35 * dx_mm / float(np.mean(d["ct_spacing_mm"])), 0.5)
        mask_float = gaussian_filter(mask_float, sigma=sigma_vox)
    mask_interp = _make_interpolator(mask_float, src_y, src_z, src_x, fill_value=0.0)
    phi0 = np.clip(mask_interp(pts).reshape(grid_shape), 0.0, 1.0)
    # Sharpen slightly to clean [0, 1] occupancy while matching observed tumour volume
    tumour_bin = phi0 >= 0.5
    phi0 = np.where(tumour_bin, 1.0, 0.0)
    # Mild 0.4-voxel boundary anti-aliasing so gradients are smooth
    phi0 = np.clip(gaussian_filter(phi0, sigma=0.35), 0.0, 1.0)
    # Rescale interior >= 0.5 mask strictly to 1.0 in core
    phi0[distance_transform_edt(tumour_bin) >= 1.5] = 1.0

    # Resample EF5 PET T/M and %ID/g onto the exact same 3D simulation grid
    tm_interp = _make_interpolator(pet_tm, pet_y, pet_z, pet_x, fill_value=prm.tm_min)
    idg_interp = _make_interpolator(pet_idg, pet_y, pet_z, pet_x, fill_value=0.0)
    pet_tm0 = np.maximum(tm_interp(pts).reshape(grid_shape), 0.0)
    pet_idg0 = np.maximum(idg_interp(pts).reshape(grid_shape), 0.0)

    # Map pre-RT EF5 PET T/M -> initial 3D spatial pO2 (mmHg) inside tumour; p_max outside
    # Estimate mild pre-existing central necrotic fraction N0 for very deep core voxels
    dist_from_rim_mm = distance_transform_edt(phi0 >= 0.5) * dx_mm
    # Only a modest pre-existing necrotic fraction deep inside larger tumours (> 2.5 mm from rim)
    nec_frac_core = 0.25 * (1.0 / (1.0 + np.exp(-(dist_from_rim_mm - 2.8) / 0.6))) * (phi0 >= 0.5)
    # If core has unusually low EF5 T/M compared to mid-shell despite being deep, treat as partially necrotic
    if np.any(tumour_bin):
        rim_zone = tumour_bin & (dist_from_rim_mm <= 1.5)
        deep_zone = tumour_bin & (dist_from_rim_mm > 2.5)
        if np.any(deep_zone) and np.any(rim_zone):
            rim_tm_med = float(np.median(pet_tm0[rim_zone]))
            cold_core = deep_zone & (pet_tm0 < 0.85 * rim_tm_med)
            nec_frac_core = np.where(cold_core, np.maximum(nec_frac_core, 0.40), nec_frac_core)

    V0 = phi0 * (1.0 - nec_frac_core)
    N0 = phi0 * nec_frac_core

    # Convert EF5 T/M to initial pO2 inside tumour
    p_from_ef5 = ef5_tm_to_po2(pet_tm0, prm, viable_frac=np.maximum(1.0 - nec_frac_core, 0.35), p_min=0.35)
    # Use consistent physiological host peritumoural boundary pO2 across all mice (inter-tumour
    # hypoxia differences arise from each tumour's internal microvascular supply field h_v(x)).
    rim_mask = (phi0 >= 0.5) & (dist_from_rim_mm <= 1.2)
    p_bdry = float(prm.p_host_bdry)

    po2_0 = phi0 * np.minimum(p_from_ef5, p_bdry) + (1.0 - phi0) * p_bdry
    # Gentle smoothing inside tumour to prevent numerical spikes in Laplacian inversion
    po2_smooth = gaussian_filter(po2_0, sigma=0.45)
    po2_0 = np.where(phi0 >= 0.2, np.clip(po2_smooth, 0.35, p_bdry), p_bdry)

    # Anchor 3D tumour-specific vascular supply field h_v(x) so that at t=0:
    #   D_o2 * laplacian(p0) + h_v * (p_bdry - p0) - gamma_0 * V0 * p0/(p0 + Km) = 0
    lap_p0 = _laplacian_3d(po2_0, dx_mm)
    cons_0 = prm.gamma_0 * V0 * o2_consumption_mm(po2_0, prm)
    denom = np.maximum(p_bdry - po2_0, 1.5)
    h_raw = (cons_0 - prm.D_o2 * lap_p0) / denom
    h_raw = np.clip(h_raw, 0.0, 4.0)
    # Smooth vascular field and set rim perfusion baseline for newly expanding angiogenic tissue
    h_vasc0 = gaussian_filter(h_raw, sigma=0.5)
    h_vasc0 = np.where(phi0 >= 0.1, h_vasc0, 0.0)

    rim_obs = float(np.median(h_vasc0[rim_mask])) if np.any(rim_mask) else prm.h_vasc_base
    # Newly recruited peripheral angiogenic rim blends tumour baseline with well-perfused host capillaries
    rim_h_vasc = float(np.clip(0.35 * rim_obs + 0.65 * (3.2 * prm.h_vasc_base), 0.22, 0.85))

    treatment = "RT_20Gy" if side_key == "rt" else "Control_0Gy"
    dose_gy = (20.0 if side_key == "rt" else 0.0) if dose_gy_override is None else float(dose_gy_override)
    tumour_id = f"{mouse_id}_{'RT' if side_key == 'rt' else 'Ctrl'}"

    return VoxelTumourInitialState(
        mouse_id=mouse_id,
        tumour_id=tumour_id,
        treatment=treatment,
        dose_gy=dose_gy,
        scan_date=scan_date,
        days_post_rt=days_post_rt,
        source_npz=str(npz_path),
        geometry_source=geom_src,
        dx_mm=dx_mm,
        shape=grid_shape,
        y_mm=y_sim,
        z_mm=z_sim,
        x_mm=x_sim,
        phi0=phi0,
        V0=V0,
        N0=N0,
        pet_tm0=pet_tm0,
        pet_idg0=pet_idg0,
        po2_0=po2_0,
        h_vasc0=h_vasc0,
        rim_h_vasc=rim_h_vasc,
        p_bdry_mmHg=p_bdry,
        obs_volume_cm3=obs_vol_cm3,
        obs_tm_mean=obs_tm_mean,
        obs_tm_p95=obs_tm_p95,
    )
