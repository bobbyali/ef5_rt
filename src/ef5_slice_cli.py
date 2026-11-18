"""
Command-line application and Python API for generating scrollable, pixel-aligned
PET, CT, and PET/CT Overlay slices (with superimposed RT_Image ROIs) across all 3 orthogonal axes
at any user-defined slice spacing — down to every individual PET slice and its closest CT slice.

Example CLI usage:
    python -m src.ef5_slice_cli \
        --pet images/set1/c1_M0_PET_mouse.dcm \
        --ct images/set1/c1_M0_CT_mouse.dcm \
        --roi images/set1/rois.ROI \
        --out-dir outputs/set1_cli \
        --step-pet-slices 1 \
        --scope roi
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.rt_image_loader import (
    RTRoi,
    RTImageVolume,
    build_roi_3d_mask,
    compute_roi_statistics,
    load_rt_image_roi,
    load_volume_auto,
)
from src.inspect_set1 import (
    ROI_STYLES,
    get_slice_2d_and_extent,
    make_pet_overlay_cmap,
    plot_rois_on_ax,
)

# Fallback palette if an ROI file has ROI names not in Set 1's default dictionary
FALLBACK_COLORS = [
    "#00E5FF", "#76FF03", "#FF4081", "#FFD740", "#FF9100",
    "#E040FB", "#1DE9B6", "#FF5252", "#40C4FF", "#EEFF41",
]


@dataclass
class SliceExportConfig:
    """Configuration parameters for the multi-axis PET/CT + ROI slice generator."""
    pet_path: Path
    ct_path: Path
    roi_path: Path
    out_dir: Path
    planes: Sequence[str] = ("transaxial", "coronal", "sagittal")
    step_pet_slices: int = 1
    step_mm: Optional[float] = None
    scope: str = "roi"  # 'roi' (slices intersecting ROIs +/- margin_mm) or 'all' (full volume)
    margin_mm: float = 0.0
    modes: Sequence[str] = ("CT", "PET", "Overlay", "Panel")
    ct_clim: Tuple[float, float] = (-500.0, 800.0)
    pet_clim: Tuple[float, float] = (0.0, 2.2)
    overlay_alpha: float = 0.78
    crop_long_axis_to_rois: bool = True
    long_axis_pad_mm: float = 6.0
    dpi: int = 120


def assign_roi_styles(rois: List[RTRoi]) -> Dict[str, Dict[str, object]]:
    """Returns a style dictionary mapping each ROI name to color, linestyle, linewidth, and label."""
    styles: Dict[str, Dict[str, object]] = {}
    for idx, roi in enumerate(rois):
        if roi.name in ROI_STYLES:
            styles[roi.name] = ROI_STYLES[roi.name].copy()
        else:
            color = FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]
            ls = "--" if "ct" in roi.name.lower() else (":" if "bg" in roi.name.lower() else "-")
            styles[roi.name] = {
                "color": color,
                "linestyle": ls,
                "linewidth": 2.0,
                "label": roi.name,
            }
    return styles


def get_axis_Sample_coordinates(
    pet_vol: RTImageVolume,
    rois: List[RTRoi],
    plane: str,
    step_pet_slices: int = 1,
    step_mm: Optional[float] = None,
    scope: str = "roi",
    margin_mm: float = 0.0,
) -> List[Tuple[int, float]]:
    """
    Determines the list of (pet_slice_idx, physical_coord_mm) along the requested plane:
      - 'transaxial': axis 0 (k, Y coordinate, spacing dz)
      - 'coronal':    axis 1 (j, Z coordinate, spacing dy)
      - 'sagittal':   axis 2 (i, X coordinate, spacing dx)
    If `step_mm` is None, steps by `step_pet_slices` integer PET voxel slices so that
    `step_pet_slices=1` visits every single PET slice and its closest equivalent CT slice.
    """
    if plane == "transaxial":
        coords_all = pet_vol.y_coords  # length nz
        coord_axis = 1  # Y in ROI vertices (x, y, z)
    elif plane == "coronal":
        coords_all = pet_vol.z_coords  # length ny
        coord_axis = 2  # Z in ROI vertices (x, y, z)
    elif plane == "sagittal":
        coords_all = pet_vol.x_coords  # length nx
        coord_axis = 0  # X in ROI vertices (x, y, z)
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    n_slices = len(coords_all)

    # Determine physical coordinate bounds if scope == 'roi'
    if scope == "roi" and rois:
        all_pts = np.vstack([r.all_vertices for r in rois if len(r.all_vertices) > 0])
        c_min = float(all_pts[:, coord_axis].min()) - margin_mm
        c_max = float(all_pts[:, coord_axis].max()) + margin_mm
        # Include half a PET voxel tolerance so edge slices are included
        voxel_tol = abs(float(coords_all[1] - coords_all[0])) * 0.55 if n_slices > 1 else 0.5
        valid_indices = [
            idx for idx, c in enumerate(coords_all)
            if (c_min - voxel_tol) <= c <= (c_max + voxel_tol)
        ]
        if not valid_indices:
            valid_indices = list(range(n_slices))
    else:
        valid_indices = list(range(n_slices))

    if step_mm is not None and step_mm > 0:
        start_mm = float(coords_all[valid_indices[0]])
        end_mm = float(coords_all[valid_indices[-1]])
        direction = -1.0 if end_mm < start_mm else 1.0
        target_mms = np.arange(start_mm, end_mm + direction * 1e-4, direction * step_mm)
        result: List[Tuple[int, float]] = []
        for t_mm in target_mms:
            closest_idx = int(np.argmin(np.abs(coords_all - t_mm)))
            result.append((closest_idx, float(t_mm)))
        return result
    else:
        step = max(1, int(step_pet_slices))
        selected_indices = valid_indices[::step]
        return [(idx, float(coords_all[idx])) for idx in selected_indices]


def rois_present_on_slice(
    plane: str,
    coord_mm: float,
    rois: List[RTRoi],
    ct_vol: RTImageVolume,
    roi_masks_ct: Dict[str, np.ndarray],
) -> List[str]:
    """Returns the names of ROIs that intersect the given slice plane at `coord_mm`."""
    dz, dy, dx = ct_vol.spacing
    nz, ny, nx = ct_vol.shape
    ox, oy, oz = ct_vol.origin

    present = []
    if plane == "transaxial":
        k = int(np.clip(round((oy - coord_mm) / dz), 0, nz - 1))
        for roi in rois:
            if np.any(roi_masks_ct[roi.name][k, :, :]) or any(
                abs(c.basis[0, 1]) > 0.5 and abs(c.center[1] - coord_mm) <= (c.thickness * 0.55)
                for c in roi.contours
            ):
                present.append(roi.name)
    elif plane == "coronal":
        j = int(np.clip(round((oz - coord_mm) / dy), 0, ny - 1))
        for roi in rois:
            if np.any(roi_masks_ct[roi.name][:, j, :]):
                present.append(roi.name)
    elif plane == "sagittal":
        i = int(np.clip(round((ox - coord_mm) / dx), 0, nx - 1))
        for roi in rois:
            if np.any(roi_masks_ct[roi.name][:, :, i]):
                present.append(roi.name)
    return present


def run_slice_generator(cfg: SliceExportConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the slice generation pipeline using `cfg`.
    Returns `(stats_df, manifest_df)`.
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PET: {cfg.pet_path}")
    pet_vol = load_volume_auto(cfg.pet_path)
    print(f"Loading CT : {cfg.ct_path}")
    ct_vol = load_volume_auto(cfg.ct_path)
    print(f"Loading ROI: {cfg.roi_path}")
    roi_ver, rois = load_rt_image_roi(cfg.roi_path)

    styles = assign_roi_styles(rois)
    # Temporarily update ROI_STYLES so plot_rois_on_ax uses the assigned styles
    ROI_STYLES.update(styles)

    # 1. Compute and save ROI statistics
    stats_df = compute_roi_statistics([pet_vol, ct_vol], rois)
    bg_rows = stats_df[
        (stats_df["Modality"] == "PT") & (stats_df["ROI Label"].str.lower().str.contains("bg"))
    ]
    if not bg_rows.empty:
        bg_mean = float(bg_rows["Mean"].iloc[0])
        stats_df["TBR (Mean / BG Mean)"] = np.where(
            stats_df["Modality"] == "PT", (stats_df["Mean"] / bg_mean).round(3), np.nan
        )
        stats_df["TBR_max (Max / BG Mean)"] = np.where(
            stats_df["Modality"] == "PT", (stats_df["Maximum"] / bg_mean).round(3), np.nan
        )

    stats_csv = out_dir / "roi_summary_stats.csv"
    stats_tsv = out_dir / "roi_summary_stats.txt"
    stats_df.to_csv(stats_csv, index=False)
    rt_cols = ["ROI Label", "Image Label", "Image Units", "Points", "Volume (cm3)", "Minimum", "Maximum", "Mean", "Std Dev", "Sum"]
    stats_df[rt_cols].rename(columns={"Volume (cm3)": "Volume"}).to_csv(stats_tsv, sep="\t", index=False)

    # 2. Precompute 3D ROI masks on CT grid
    roi_masks_ct = {roi.name: build_roi_3d_mask(ct_vol, roi) for roi in rois}

    # Determine Y-axis view window for coronal/sagittal views if crop_long_axis_to_rois is enabled
    view_ylim_long: Optional[Tuple[float, float]] = None
    if cfg.crop_long_axis_to_rois and rois:
        all_pts = np.vstack([r.all_vertices for r in rois if len(r.all_vertices) > 0])
        y_min = float(all_pts[:, 1].min()) - cfg.long_axis_pad_mm
        y_max = float(all_pts[:, 1].max()) + cfg.long_axis_pad_mm
        view_ylim_long = (y_min, y_max)

    overlay_cmap = make_pet_overlay_cmap("inferno", max_alpha=cfg.overlay_alpha)

    manifest_rows = []
    modes_upper = {m.upper() for m in cfg.modes}

    for plane in cfg.planes:
        plane_lower = plane.lower()
        samples = get_axis_Sample_coordinates(
            pet_vol=pet_vol,
            rois=rois,
            plane=plane_lower,
            step_pet_slices=cfg.step_pet_slices,
            step_mm=cfg.step_mm,
            scope=cfg.scope,
            margin_mm=cfg.margin_mm,
        )
        print(f"Generating {len(samples)} slices for plane '{plane_lower}'...")

        view_ylim = None if plane_lower == "transaxial" else view_ylim_long

        for seq_idx, (pet_slice_idx, coord_mm) in enumerate(samples, start=1):
            ct_2d, ct_ext, ct_slice_idx, ct_actual_mm = get_slice_2d_and_extent(ct_vol, plane_lower, coord_mm)
            pet_2d, pet_ext, pet_slice_idx_actual, pet_actual_mm = get_slice_2d_and_extent(pet_vol, plane_lower, coord_mm)
            active_rois = rois_present_on_slice(plane_lower, coord_mm, rois, ct_vol, roi_masks_ct)

            axis_letter = {"transaxial": "Y", "coronal": "Z", "sagittal": "X"}[plane_lower]
            base_slug = f"{plane_lower}_s{seq_idx:03d}_PET{pet_slice_idx_actual:03d}_CT{ct_slice_idx:03d}_{axis_letter}{coord_mm:+.2f}mm"

            saved_files: Dict[str, str] = {}

            # Render individual scrollable same-size frames (CT, PET, Overlay)
            for mode_order, (mode_key, mode_name) in enumerate([("CT", "CT"), ("PET", "PET"), ("OVERLAY", "Overlay")], start=1):
                if mode_key not in modes_upper:
                    continue

                # Also save into two convenient scrollable layouts:
                # 1) by_slice/ (1_CT, 2_PET, 3_Overlay side-by-side in sort order to flip modalities at fixed slice)
                # 2) by_modality/{CT,PET,Overlay}/ (all slices of one modality in sort order to scroll smoothly through the 3D volume!)
                fig = plt.figure(figsize=(10, 8), dpi=cfg.dpi, facecolor="#111111")
                ax = fig.add_axes([0.10, 0.14, 0.74, 0.74], facecolor="black")
                cax = fig.add_axes([0.86, 0.14, 0.025, 0.74])

                if mode_name == "CT":
                    im = ax.imshow(
                        ct_2d, cmap="gray", vmin=cfg.ct_clim[0], vmax=cfg.ct_clim[1],
                        extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
                    )
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_label("CT Intensity (HU)", color="white", fontsize=10)
                    subtitle = f"CT Only + ROIs | CT Slice {ct_slice_idx} ({ct_actual_mm:.2f} mm) | Closest PET Slice {pet_slice_idx_actual}"
                elif mode_name == "PET":
                    im = ax.imshow(
                        pet_2d, cmap="inferno", vmin=cfg.pet_clim[0], vmax=cfg.pet_clim[1],
                        extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal"
                    )
                    cb = fig.colorbar(im, cax=cax)
                    cb.set_label("EF5 PET Uptake (%IDG)", color="white", fontsize=10)
                    subtitle = f"EF5 PET Only + ROIs | PET Slice {pet_slice_idx_actual} ({pet_actual_mm:.2f} mm) | Closest CT Slice {ct_slice_idx}"
                else:
                    ax.imshow(
                        ct_2d, cmap="gray", vmin=cfg.ct_clim[0], vmax=cfg.ct_clim[1],
                        extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
                    )
                    ax.imshow(
                        pet_2d, cmap=overlay_cmap, vmin=cfg.pet_clim[0], vmax=cfg.pet_clim[1],
                        extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal"
                    )
                    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=cfg.pet_clim[0], vmax=cfg.pet_clim[1]))
                    cb = fig.colorbar(sm, cax=cax)
                    cb.set_label("EF5 PET Uptake (%IDG)", color="white", fontsize=10)
                    subtitle = f"PET/CT Overlay + ROIs | PET Slice {pet_slice_idx_actual} ({pet_actual_mm:.2f} mm) & CT Slice {ct_slice_idx} ({ct_actual_mm:.2f} mm)"

                cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
                for spine in cb.ax.spines.values():
                    spine.set_edgecolor("#555555")

                plot_rois_on_ax(ax, plane_lower, coord_mm, rois, ct_vol, roi_masks_ct)
                ax.set_xlim(ct_ext[0], ct_ext[1])
                ax.set_ylim(*(view_ylim if view_ylim is not None else (ct_ext[2], ct_ext[3])))

                if plane_lower == "transaxial":
                    ax.set_xlabel("Physical X (mm) [Right <---> Left]", color="white", fontsize=11)
                    ax.set_ylabel("Physical Z (mm) [Ventral <---> Dorsal]", color="white", fontsize=11)
                elif plane_lower == "coronal":
                    ax.set_xlabel("Physical X (mm) [Right <---> Left]", color="white", fontsize=11)
                    ax.set_ylabel("Physical Y (mm) [Caudal <---> Cranial]", color="white", fontsize=11)
                else:
                    ax.set_xlabel("Physical Z (mm) [Dorsal <---> Ventral]", color="white", fontsize=11)
                    ax.set_ylabel("Physical Y (mm) [Caudal <---> Cranial]", color="white", fontsize=11)

                ax.set_title(
                    f"{plane_lower.capitalize()} Plane ({axis_letter} = {coord_mm:.2f} mm) — Frame {seq_idx}/{len(samples)}\n{subtitle}",
                    color="white", fontsize=11.5, fontweight="bold", pad=12
                )
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#666666")

                legend_handles = [
                    Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"], label=s["label"])
                    for s in styles.values()
                ]
                fig.legend(
                    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.48, 0.015),
                    ncol=min(3, len(legend_handles)), frameon=True, facecolor="#1E1E1E",
                    edgecolor="#555555", labelcolor="white", fontsize=8.5
                )

                by_slice_path = out_dir / "scrollable_by_slice" / plane_lower / f"{base_slug}_{mode_order}_{mode_name}.png"
                by_mod_path = out_dir / "scrollable_by_modality" / plane_lower / mode_name / f"{base_slug}_{mode_name}.png"
                by_slice_path.parent.mkdir(parents=True, exist_ok=True)
                by_mod_path.parent.mkdir(parents=True, exist_ok=True)

                fig.savefig(by_slice_path, dpi=cfg.dpi, facecolor=fig.get_facecolor())
                fig.savefig(by_mod_path, dpi=cfg.dpi, facecolor=fig.get_facecolor())
                plt.close(fig)

                saved_files[f"{mode_name}_path"] = str(by_mod_path.relative_to(out_dir))

            # Optional 3-panel side-by-side summary
            if "PANEL" in modes_upper:
                panel_path = out_dir / "panels" / plane_lower / f"{base_slug}_Panel.png"
                panel_path.parent.mkdir(parents=True, exist_ok=True)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=cfg.dpi, facecolor="#111111")
                fig.subplots_adjust(left=0.05, right=0.96, bottom=0.16, top=0.84, wspace=0.18)
                titles = [
                    f"CT + ROIs (CT slice {ct_slice_idx}, {ct_actual_mm:.2f} mm)",
                    f"EF5 PET + ROIs (PET slice {pet_slice_idx_actual}, {pet_actual_mm:.2f} mm)",
                    f"Combined PET/CT Overlay + ROIs",
                ]
                for idx_ax, ax in enumerate(axes):
                    ax.set_facecolor("black")
                    if idx_ax == 0:
                        ax.imshow(
                            ct_2d, cmap="gray", vmin=cfg.ct_clim[0], vmax=cfg.ct_clim[1],
                            extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
                        )
                    elif idx_ax == 1:
                        ax.imshow(
                            pet_2d, cmap="inferno", vmin=cfg.pet_clim[0], vmax=cfg.pet_clim[1],
                            extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal"
                        )
                    else:
                        ax.imshow(
                            ct_2d, cmap="gray", vmin=cfg.ct_clim[0], vmax=cfg.ct_clim[1],
                            extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
                        )
                        ax.imshow(
                            pet_2d, cmap=overlay_cmap, vmin=cfg.pet_clim[0], vmax=cfg.pet_clim[1],
                            extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal"
                        )
                    plot_rois_on_ax(ax, plane_lower, coord_mm, rois, ct_vol, roi_masks_ct)
                    ax.set_xlim(ct_ext[0], ct_ext[1])
                    ax.set_ylim(*(view_ylim if view_ylim is not None else (ct_ext[2], ct_ext[3])))
                    ax.set_title(titles[idx_ax], color="white", fontsize=10.5, fontweight="bold")
                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#666666")

                fig.suptitle(
                    f"{plane_lower.capitalize()} Plane ({axis_letter} = {coord_mm:.2f} mm) — PET Slice {pet_slice_idx_actual} / Closest CT Slice {ct_slice_idx}",
                    color="white", fontsize=13.5, fontweight="bold"
                )
                legend_handles = [
                    Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"], label=s["label"])
                    for s in styles.values()
                ]
                fig.legend(
                    handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                    ncol=min(3, len(legend_handles)), frameon=True, facecolor="#1E1E1E",
                    edgecolor="#555555", labelcolor="white", fontsize=9.0
                )
                fig.savefig(panel_path, dpi=cfg.dpi, facecolor=fig.get_facecolor())
                plt.close(fig)
                saved_files["Panel_path"] = str(panel_path.relative_to(out_dir))

            manifest_rows.append(
                {
                    "plane": plane_lower,
                    "sequence_num": seq_idx,
                    "coord_mm": round(coord_mm, 4),
                    "pet_slice_index": pet_slice_idx_actual,
                    "pet_slice_coord_mm": round(pet_actual_mm, 4),
                    "closest_ct_slice_index": ct_slice_idx,
                    "ct_slice_coord_mm": round(ct_actual_mm, 4),
                    "rois_present": "; ".join(active_rois),
                    **saved_files,
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "slice_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"Completed! Generated {len(manifest_df)} slice sets in {out_dir}")
    print(f"  - ROI stats   : {stats_csv}")
    print(f"  - Manifest CSV: {manifest_csv}")
    return stats_df, manifest_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate scrollable, pixel-aligned PET, CT, and PET/CT Overlay slices with superimposed RT_Image ROIs."
    )
    parser.add_argument("--pet", type=Path, required=True, help="Path to the RT_Image PET DICOM volume (.dcm)")
    parser.add_argument("--ct", type=Path, required=True, help="Path to the RT_Image CT DICOM volume (.dcm)")
    parser.add_argument("--roi", type=Path, required=True, help="Path to the RT_Image binary .ROI file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for generated images and CSVs")
    parser.add_argument(
        "--planes", nargs="+", default=["transaxial", "coronal", "sagittal"],
        choices=["transaxial", "coronal", "sagittal"],
        help="Orthogonal planes to render (default: transaxial coronal sagittal)"
    )
    parser.add_argument(
        "--step-pet-slices", type=int, default=1,
        help="Slice step size in integer PET voxels along each axis (1 = every single PET slice + closest CT slice; default: 1)"
    )
    parser.add_argument(
        "--step-mm", type=float, default=None,
        help="Optional slice step size in millimeters (overrides --step-pet-slices if set)"
    )
    parser.add_argument(
        "--scope", choices=["roi", "all"], default="roi",
        help="'roi' = only slices intersecting the ROI bounding box (+/- --margin-mm); 'all' = entire 3D volume (default: roi)"
    )
    parser.add_argument(
        "--margin-mm", type=float, default=0.0,
        help="Extra margin in mm around the ROI bounding box when --scope roi is used (default: 0.0)"
    )
    parser.add_argument(
        "--modes", nargs="+", default=["CT", "PET", "Overlay", "Panel"],
        choices=["CT", "PET", "Overlay", "Panel"],
        help="Which image modes to render for each slice (default: CT PET Overlay Panel)"
    )
    parser.add_argument(
        "--ct-clim", nargs=2, type=float, default=[-500.0, 800.0], metavar=("VMIN", "VMAX"),
        help="CT intensity display window in HU (default: -500 800)"
    )
    parser.add_argument(
        "--pet-clim", nargs=2, type=float, default=[0.0, 2.2], metavar=("VMIN", "VMAX"),
        help="PET intensity display window in %%IDG (default: 0.0 2.2)"
    )
    parser.add_argument(
        "--overlay-alpha", type=float, default=0.78,
        help="Maximum opacity of the PET layer in the fused PET/CT overlay (default: 0.78)"
    )
    parser.add_argument(
        "--full-long-axis", action="store_true",
        help="Show the full cranial-caudal Y axis in coronal/sagittal views instead of zooming around the ROI Y range"
    )
    parser.add_argument(
        "--dpi", type=int, default=120,
        help="Output image resolution in DPI (120 DPI = 1200x960 px for single frames; default: 120)"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = SliceExportConfig(
        pet_path=args.pet,
        ct_path=args.ct,
        roi_path=args.roi,
        out_dir=args.out_dir,
        planes=args.planes,
        step_pet_slices=args.step_pet_slices,
        step_mm=args.step_mm,
        scope=args.scope,
        margin_mm=args.margin_mm,
        modes=args.modes,
        ct_clim=(args.ct_clim[0], args.ct_clim[1]),
        pet_clim=(args.pet_clim[0], args.pet_clim[1]),
        overlay_alpha=args.overlay_alpha,
        crop_long_axis_to_rois=not args.full_long_axis,
        dpi=args.dpi,
    )
    run_slice_generator(cfg)


if __name__ == "__main__":
    main()
