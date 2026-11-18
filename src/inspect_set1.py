"""
Inspects, analyzes, and visualizes Set 1 EF5 PET/CT mouse subcutaneous tumour images and RT_Image ROIs.

Generates:
1. Quantitative ROI statistics (CSV, RT_Image-style TSV, and Markdown report).
2. Scrollable, pixel-aligned same-size PNG images (CT, PET, and PET/CT Overlay with ROIs superimposed)
   across sample slices in all 3 orthogonal planes (Transaxial, Coronal, Sagittal).
3. Multi-panel summary overview figures for quick side-by-side comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from skimage import measure


def make_pet_overlay_cmap(base_cmap_name: str = "inferno", max_alpha: float = 0.78):
    """
    Creates a colormap for PET/CT fusion whose alpha channel ramps smoothly
    from 0.0 (completely transparent at 0 %IDG so CT is never darkened)
    to `max_alpha` for uptake signal.
    """
    base = plt.get_cmap(base_cmap_name, 256)
    colors = base(np.linspace(0, 1, 256))
    # Smoothly ramp alpha from 0.0 at index 0..6 up to max_alpha by index 55 (~20% of full scale)
    alpha = np.clip((np.linspace(0, 1, 256) - 0.02) / 0.18, 0.0, 1.0) * max_alpha
    colors[:, 3] = alpha
    return LinearSegmentedColormap.from_list(f"{base_cmap_name}_alpha", colors)


PET_OVERLAY_CMAP = make_pet_overlay_cmap("inferno", max_alpha=0.78)

from src.rt_image_loader import (
    RTRoi,
    RTImageVolume,
    build_roi_3d_mask,
    compute_roi_statistics,
    load_rt_image_dicom,
    load_rt_image_roi,
)

# Distinct visual styling for each ROI in Set 1 so PET vs CT contours are easy to distinguish
ROI_STYLES: Dict[str, Dict[str, object]] = {
    "c1_M0_L PET ROI": {"color": "#00E5FF", "linestyle": "-",  "linewidth": 2.2, "label": "c1_M0_L PET ROI (Left Tumour - PET)"},
    "c1_M0_L CT ROI":  {"color": "#76FF03", "linestyle": "--", "linewidth": 2.0, "label": "c1_M0_L CT ROI (Left Tumour - CT)"},
    "c1_M0_R PET ROI": {"color": "#FF4081", "linestyle": "-",  "linewidth": 2.2, "label": "c1_M0_R PET ROI (Right Tumour - PET)"},
    "c1_M0_R CT ROI":  {"color": "#FFD740", "linestyle": "--", "linewidth": 2.0, "label": "c1_M0_R CT ROI (Right Tumour - CT)"},
    "M0 bg ROI":       {"color": "#FF9100", "linestyle": ":",  "linewidth": 2.2, "label": "M0 bg ROI (Background Ellipse)"},
}


def get_slice_2d_and_extent(
    vol: RTImageVolume, plane: str, coord_mm: float
) -> Tuple[np.ndarray, List[float], int, float]:
    """
    Extracts a 2D slice from `vol` closest to `coord_mm` along `plane` ('transaxial', 'coronal', 'sagittal')
    and returns (slice_2d, extent_mm [xmin, xmax, ymin, ymax], slice_idx, actual_coord_mm).

    Physical coordinate convention:
      - 'transaxial' (X-Z plane at constant Y, axis 0 = k):
          horizontal = X (cols), vertical = Z (rows)
      - 'coronal' (X-Y plane at constant Z, axis 1 = j):
          horizontal = X (cols), vertical = Y (planes)
      - 'sagittal' (Z-Y plane at constant X, axis 2 = i):
          horizontal = Z (rows), vertical = Y (planes)
    """
    dz, dy, dx = vol.spacing
    nz, ny, nx = vol.shape
    ox, oy, oz = vol.origin

    x_min_edge = ox - nx * dx
    x_max_edge = ox
    y_min_edge = oy - nz * dz
    y_max_edge = oy
    z_min_edge = oz - ny * dy
    z_max_edge = oz

    if plane == "transaxial":
        k = int(np.clip(round((oy - coord_mm) / dz), 0, nz - 1))
        actual_mm = float(oy - k * dz)
        img2d = vol.data[k, :, :]
        extent = [x_max_edge, x_min_edge, z_min_edge, z_max_edge]
        return img2d, extent, k, actual_mm

    elif plane == "coronal":
        j = int(np.clip(round((oz - coord_mm) / dy), 0, ny - 1))
        actual_mm = float(oz - j * dy)
        img2d = vol.data[:, j, :]
        extent = [x_max_edge, x_min_edge, y_min_edge, y_max_edge]
        return img2d, extent, j, actual_mm

    elif plane == "sagittal":
        i = int(np.clip(round((ox - coord_mm) / dx), 0, nx - 1))
        actual_mm = float(ox - i * dx)
        img2d = vol.data[:, :, i]
        extent = [z_max_edge, z_min_edge, y_min_edge, y_max_edge]
        return img2d, extent, i, actual_mm

    else:
        raise ValueError(f"Unknown plane: {plane}")


def plot_rois_on_ax(
    ax: plt.Axes,
    plane: str,
    coord_mm: float,
    rois: List[RTRoi],
    ct_vol: RTImageVolume,
    roi_masks_ct: Dict[str, np.ndarray],
) -> None:
    """
    Superimposes ROI contours onto `ax` in physical millimeter coordinates.
    - For 'transaxial' (the native plane in which RT_Image contours were drawn), plots the exact polygon/ellipse vertices
      whenever a contour lies within half a PET slice thickness (0.40 mm) of `coord_mm`.
    - For 'coronal' and 'sagittal', extracts 2D iso-contours from the high-resolution 3D rasterized ROI mask.
    """
    dz, dy, dx = ct_vol.spacing
    nz, ny, nx = ct_vol.shape
    ox, oy, oz = ct_vol.origin

    if plane == "transaxial":
        k = int(np.clip(round((oy - coord_mm) / dz), 0, nz - 1))
        for roi in rois:
            style = ROI_STYLES.get(roi.name, {"color": "cyan", "linestyle": "-", "linewidth": 2.0})
            # Check if this ROI's contours were drawn directly on transaxial (Y-normal) planes
            has_y_normal = any(abs(c.basis[0, 1]) > 0.5 for c in roi.contours)
            if has_y_normal:
                for c in roi.contours:
                    if abs(c.basis[0, 1]) > 0.5 and abs(c.center[1] - coord_mm) <= (c.thickness * 0.55):
                        pts = np.vstack([c.vertices, c.vertices[0:1]])
                        ax.plot(
                            pts[:, 0],
                            pts[:, 2],
                            color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=style["linewidth"],
                            alpha=0.95,
                        )
            else:
                # Contour was drawn on sagittal/coronal plane (e.g. Set 2 bg ROIs); slice the 3D mask at k
                mask_2d = roi_masks_ct[roi.name][k, :, :]  # shape (ny, nx)
                if np.any(mask_2d):
                    contours_px = measure.find_contours(mask_2d.astype(float), 0.5)
                    for cnt in contours_px:
                        # cnt[:, 0] is j (row -> Z), cnt[:, 1] is i (col -> X)
                        z_phys = oz - (cnt[:, 0] + 0.5) * dy
                        x_phys = ox - (cnt[:, 1] + 0.5) * dx
                        ax.plot(
                            x_phys,
                            z_phys,
                            color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=style["linewidth"],
                            alpha=0.95,
                        )
    elif plane == "coronal":
        j = int(np.clip(round((oz - coord_mm) / dy), 0, ny - 1))
        for roi in rois:
            style = ROI_STYLES.get(roi.name, {"color": "cyan", "linestyle": "-", "linewidth": 2.0})
            mask_2d = roi_masks_ct[roi.name][:, j, :]  # shape (nz, nx)
            if np.any(mask_2d):
                contours_px = measure.find_contours(mask_2d.astype(float), 0.5)
                for cnt in contours_px:
                    # cnt[:, 0] is k (plane -> Y), cnt[:, 1] is i (col -> X)
                    y_phys = oy - (cnt[:, 0] + 0.5) * dz
                    x_phys = ox - (cnt[:, 1] + 0.5) * dx
                    ax.plot(
                        x_phys,
                        y_phys,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=0.95,
                    )
    elif plane == "sagittal":
        i = int(np.clip(round((ox - coord_mm) / dx), 0, nx - 1))
        for roi in rois:
            style = ROI_STYLES.get(roi.name, {"color": "cyan", "linestyle": "-", "linewidth": 2.0})
            mask_2d = roi_masks_ct[roi.name][:, :, i]  # shape (nz, ny)
            if np.any(mask_2d):
                contours_px = measure.find_contours(mask_2d.astype(float), 0.5)
                for cnt in contours_px:
                    # cnt[:, 0] is k (plane -> Y), cnt[:, 1] is j (row -> Z)
                    y_phys = oy - (cnt[:, 0] + 0.5) * dz
                    z_phys = oz - (cnt[:, 1] + 0.5) * dy
                    ax.plot(
                        z_phys,
                        y_phys,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        alpha=0.95,
                    )


def render_single_modality_frame(
    out_path: Path,
    mode: str,  # 'CT', 'PET', or 'Overlay'
    plane: str,
    coord_mm: float,
    slice_label: str,
    pet_vol: RTImageVolume,
    ct_vol: RTImageVolume,
    rois: List[RTRoi],
    roi_masks_ct: Dict[str, np.ndarray],
    ct_clim: Tuple[float, float] = (-500.0, 800.0),
    pet_clim: Tuple[float, float] = (0.0, 2.2),
    view_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Renders a single frame ('CT', 'PET', or 'Overlay') with locked figure size (1200x960 px)
    and locked axes rectangle so the user can flip/scroll between CT, PET, and Overlay seamlessly.
    """
    fig = plt.figure(figsize=(10, 8), dpi=120, facecolor="#111111")
    # Fixed axes geometry [left, bottom, width, height] so the image never shifts between frames
    ax = fig.add_axes([0.10, 0.14, 0.74, 0.74], facecolor="black")
    cax = fig.add_axes([0.86, 0.14, 0.025, 0.74])

    ct_2d, ct_ext, ct_idx, ct_mm = get_slice_2d_and_extent(ct_vol, plane, coord_mm)
    pet_2d, pet_ext, pet_idx, pet_mm = get_slice_2d_and_extent(pet_vol, plane, coord_mm)

    if mode == "CT":
        im = ax.imshow(
            ct_2d,
            cmap="gray",
            vmin=ct_clim[0],
            vmax=ct_clim[1],
            extent=ct_ext,
            origin="upper",
            interpolation="bilinear",
            aspect="equal",
        )
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("CT Intensity (HU)", color="white", fontsize=10)
        mode_title = f"CT Only + ROIs (CT slice {ct_idx})"

    elif mode == "PET":
        im = ax.imshow(
            pet_2d,
            cmap="inferno",
            vmin=pet_clim[0],
            vmax=pet_clim[1],
            extent=pet_ext,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("EF5 PET Uptake (%IDG)", color="white", fontsize=10)
        mode_title = f"EF5 PET Only + ROIs (PET slice {pet_idx})"

    elif mode == "Overlay":
        ax.imshow(
            ct_2d,
            cmap="gray",
            vmin=ct_clim[0],
            vmax=ct_clim[1],
            extent=ct_ext,
            origin="upper",
            interpolation="bilinear",
            aspect="equal",
        )
        ax.imshow(
            pet_2d,
            cmap=PET_OVERLAY_CMAP,
            vmin=pet_clim[0],
            vmax=pet_clim[1],
            extent=pet_ext,
            origin="upper",
            interpolation="bilinear",
            aspect="equal",
        )
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin=pet_clim[0], vmax=pet_clim[1]))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("EF5 PET Uptake (%IDG)", color="white", fontsize=10)
        mode_title = f"Registered PET/CT Overlay + ROIs (CT {ct_idx} / PET {pet_idx})"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    for spine in cb.ax.spines.values():
        spine.set_edgecolor("#555555")

    # Superimpose ROI contours
    plot_rois_on_ax(ax, plane, coord_mm, rois, ct_vol, roi_masks_ct)

    # Lock axis limits to CT physical extent
    ax.set_xlim(ct_ext[0], ct_ext[1])
    if view_ylim is not None:
        ax.set_ylim(view_ylim[0], view_ylim[1])
    else:
        ax.set_ylim(ct_ext[2], ct_ext[3])

    if plane == "transaxial":
        ax.set_xlabel("Physical X (mm) [Right  <--->  Left]", color="white", fontsize=11)
        ax.set_ylabel("Physical Z (mm) [Ventral  <--->  Dorsal]", color="white", fontsize=11)
        plane_desc = f"Transaxial Plane (Y = {coord_mm:.2f} mm)"
    elif plane == "coronal":
        ax.set_xlabel("Physical X (mm) [Right  <--->  Left]", color="white", fontsize=11)
        ax.set_ylabel("Physical Y (mm) [Caudal  <--->  Cranial]", color="white", fontsize=11)
        plane_desc = f"Coronal Plane (Z = {coord_mm:.2f} mm)"
    else:
        ax.set_xlabel("Physical Z (mm) [Dorsal  <--->  Ventral]", color="white", fontsize=11)
        ax.set_ylabel("Physical Y (mm) [Caudal  <--->  Cranial]", color="white", fontsize=11)
        plane_desc = f"Sagittal Plane (X = {coord_mm:.2f} mm)"

    ax.set_title(
        f"Set 1 (c1_M0) — {plane_desc} [{slice_label}]\n{mode_title}",
        color="white",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")

    # Consistent ROI legend at bottom
    legend_handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"], label=s["label"])
        for s in ROI_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.015),
        ncol=3,
        frameon=True,
        facecolor="#1E1E1E",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=8.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_three_panel_comparison(
    out_path: Path,
    plane: str,
    coord_mm: float,
    slice_label: str,
    pet_vol: RTImageVolume,
    ct_vol: RTImageVolume,
    rois: List[RTRoi],
    roi_masks_ct: Dict[str, np.ndarray],
    ct_clim: Tuple[float, float] = (-500.0, 800.0),
    pet_clim: Tuple[float, float] = (0.0, 2.2),
    view_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Renders a 1x3 side-by-side figure (CT | PET | PET/CT Overlay) for a given slice."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=120, facecolor="#111111")
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.16, top=0.84, wspace=0.18)

    ct_2d, ct_ext, ct_idx, _ = get_slice_2d_and_extent(ct_vol, plane, coord_mm)
    pet_2d, pet_ext, pet_idx, _ = get_slice_2d_and_extent(pet_vol, plane, coord_mm)

    titles = [
        f"CT + ROIs (slice {ct_idx})",
        f"EF5 PET + ROIs (slice {pet_idx})",
        "Combined PET/CT Overlay + ROIs",
    ]

    for idx, ax in enumerate(axes):
        ax.set_facecolor("black")
        if idx == 0:
            ax.imshow(
                ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1],
                extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
            )
        elif idx == 1:
            ax.imshow(
                pet_2d, cmap="inferno", vmin=pet_clim[0], vmax=pet_clim[1],
                extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal"
            )
        else:
            ax.imshow(
                ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1],
                extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal"
            )
            ax.imshow(
                pet_2d, cmap=PET_OVERLAY_CMAP, vmin=pet_clim[0], vmax=pet_clim[1],
                extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal"
            )

        plot_rois_on_ax(ax, plane, coord_mm, rois, ct_vol, roi_masks_ct)
        ax.set_xlim(ct_ext[0], ct_ext[1])
        if view_ylim is not None:
            ax.set_ylim(view_ylim[0], view_ylim[1])
        else:
            ax.set_ylim(ct_ext[2], ct_ext[3])

        ax.set_title(titles[idx], color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#666666")

        if plane == "transaxial":
            ax.set_xlabel("X (mm)", color="white")
            if idx == 0:
                ax.set_ylabel("Z (mm)", color="white")
        elif plane == "coronal":
            ax.set_xlabel("X (mm)", color="white")
            if idx == 0:
                ax.set_ylabel("Y (mm)", color="white")
        else:
            ax.set_xlabel("Z (mm)", color="white")
            if idx == 0:
                ax.set_ylabel("Y (mm)", color="white")

    fig.suptitle(
        f"Set 1 (c1_M0) — {plane.capitalize()} Slice at {coord_mm:.2f} mm ({slice_label})",
        color="white",
        fontsize=14,
        fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], linewidth=s["linewidth"], label=s["label"])
        for s in ROI_STYLES.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=True,
        facecolor="#1E1E1E",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=9.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    root = Path("/workspace")
    set1_dir = root / "images" / "set1"
    out_dir = root / "outputs" / "set1"
    notes_dir = root / "agent_notes"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Set 1 DICOM volumes and RT_Image ROI file...")
    pet_vol = load_rt_image_dicom(set1_dir / "c1_M0_PET_mouse.dcm")
    ct_vol = load_rt_image_dicom(set1_dir / "c1_M0_CT_mouse.dcm")
    roi_version, rois = load_rt_image_roi(set1_dir / "rois.ROI")

    print(f"  PET Volume: shape={pet_vol.shape}, spacing={pet_vol.spacing} mm, units='{pet_vol.units}'")
    print(f"  CT  Volume: shape={ct_vol.shape}, spacing={ct_vol.spacing} mm, units='{ct_vol.units}'")
    print(f"  ROI File  : {roi_version}, {len(rois)} ROIs loaded")

    # Compute ROI statistics on both PET and CT
    stats_df = compute_roi_statistics([pet_vol, ct_vol], rois)

    # Compute Tumour-to-Background Ratios (TBR) on PET relative to 'M0 bg ROI'
    bg_row = stats_df[(stats_df["Modality"] == "PT") & (stats_df["ROI Label"] == "M0 bg ROI")]
    bg_mean = float(bg_row["Mean"].iloc[0]) if not bg_row.empty else np.nan
    stats_df["TBR (Mean / BG Mean)"] = np.where(
        stats_df["Modality"] == "PT",
        (stats_df["Mean"] / bg_mean).round(3),
        np.nan,
    )
    stats_df["TBR_max (Max / BG Mean)"] = np.where(
        stats_df["Modality"] == "PT",
        (stats_df["Maximum"] / bg_mean).round(3),
        np.nan,
    )

    csv_path = out_dir / "set1_roi_stats.csv"
    tsv_path = out_dir / "set1_roi_stats.txt"
    stats_df.to_csv(csv_path, index=False)

    # Also save RT_Image-identical tab-separated stats file
    rt_cols = ["ROI Label", "Image Label", "Image Units", "Points", "Volume (cm3)", "Minimum", "Maximum", "Mean", "Std Dev", "Sum"]
    rt_df = stats_df[rt_cols].rename(columns={"Volume (cm3)": "Volume"})
    rt_df.to_csv(tsv_path, sep="\t", index=False)

    print("\n--- Set 1 Quantitative ROI Statistics ---")
    print(stats_df.to_string(index=False))

    # Precompute 3D ROI masks on the CT grid for coronal/sagittal slicing
    roi_masks_ct = {roi.name: build_roi_3d_mask(ct_vol, roi) for roi in rois}

    # Define representative sample slices passing through the tumours in all 3 orthogonal planes:
    # 1. Transaxial (X-Z plane at constant Y):
    #    - Y = -59.70 mm: Right tumour (PET+CT), Left CT ROI, and Background ROI
    #    - Y = -58.11 mm: All 5 ROIs visible simultaneously (Right tumour center + Left tumour caudal + BG center)
    #    - Y = -56.52 mm: All 5 ROIs visible simultaneously (Mid-tumour plane for both Left & Right)
    #    - Y = -54.92 mm: Left tumour core + Right tumour cranial edge
    #    - Y = -52.54 mm: Left tumour cranial region (PET+CT)
    transaxial_samples = [
        (-59.700, "slice01_Y-59.7mm_Right_Tumour_and_BG"),
        (-58.108, "slice02_Y-58.1mm_All_5_ROIs_Core"),
        (-56.516, "slice03_Y-56.5mm_Both_Tumours_Mid"),
        (-54.924, "slice04_Y-54.9mm_Left_Tumour_Core"),
        (-52.536, "slice05_Y-52.5mm_Left_Tumour_Cranial"),
    ]

    # 2. Coronal (X-Y plane at constant Z):
    #    Zoom Y window to [-68, -46] mm around the tumour-bearing region (plus full-body view)
    coronal_samples = [
        (-74.96, "slice01_Z-75.0mm_Right_Tumour_and_BG_Plane"),
        (-73.40, "slice02_Z-73.4mm_Both_Left_and_Right_Tumours"),
        (-71.80, "slice03_Z-71.8mm_Left_Tumour_Mid_Plane"),
    ]

    # 3. Sagittal (Z-Y plane at constant X):
    sagittal_samples = [
        (-56.00, "slice01_X-56.0mm_Left_Tumour_Sagittal"),
        (-62.85, "slice02_X-62.9mm_Background_ROI_Sagittal"),
        (-73.00, "slice03_X-73.0mm_Right_Tumour_Sagittal"),
    ]

    scrollable_dir = out_dir / "scrollable"
    panels_dir = out_dir / "panels"

    print("\nRendering scrollable same-size images and 3-panel comparisons...")

    # Render Transaxial slices
    for y_mm, slug in transaxial_samples:
        for mode_idx, mode in enumerate(["CT", "PET", "Overlay"], start=1):
            fname = f"transaxial_{slug}_{mode_idx}_{mode}.png"
            render_single_modality_frame(
                scrollable_dir / "transaxial" / fname,
                mode=mode,
                plane="transaxial",
                coord_mm=y_mm,
                slice_label=slug.replace("_", " "),
                pet_vol=pet_vol,
                ct_vol=ct_vol,
                rois=rois,
                roi_masks_ct=roi_masks_ct,
            )
        render_three_panel_comparison(
            panels_dir / f"panel_transaxial_{slug}.png",
            plane="transaxial",
            coord_mm=y_mm,
            slice_label=slug.replace("_", " "),
            pet_vol=pet_vol,
            ct_vol=ct_vol,
            rois=rois,
            roi_masks_ct=roi_masks_ct,
        )

    # Render Coronal slices (cropped around Y in [-68, -46] mm so tumours are large and clear, plus full mouse)
    for z_mm, slug in coronal_samples:
        for mode_idx, mode in enumerate(["CT", "PET", "Overlay"], start=1):
            fname = f"coronal_{slug}_{mode_idx}_{mode}.png"
            render_single_modality_frame(
                scrollable_dir / "coronal" / fname,
                mode=mode,
                plane="coronal",
                coord_mm=z_mm,
                slice_label=slug.replace("_", " "),
                pet_vol=pet_vol,
                ct_vol=ct_vol,
                rois=rois,
                roi_masks_ct=roi_masks_ct,
                view_ylim=(-68.0, -46.0),
            )
        render_three_panel_comparison(
            panels_dir / f"panel_coronal_{slug}.png",
            plane="coronal",
            coord_mm=z_mm,
            slice_label=slug.replace("_", " "),
            pet_vol=pet_vol,
            ct_vol=ct_vol,
            rois=rois,
            roi_masks_ct=roi_masks_ct,
            view_ylim=(-68.0, -46.0),
        )

    # Render Sagittal slices (cropped around Y in [-68, -46] mm around tumours)
    for x_mm, slug in sagittal_samples:
        for mode_idx, mode in enumerate(["CT", "PET", "Overlay"], start=1):
            fname = f"sagittal_{slug}_{mode_idx}_{mode}.png"
            render_single_modality_frame(
                scrollable_dir / "sagittal" / fname,
                mode=mode,
                plane="sagittal",
                coord_mm=x_mm,
                slice_label=slug.replace("_", " "),
                pet_vol=pet_vol,
                ct_vol=ct_vol,
                rois=rois,
                roi_masks_ct=roi_masks_ct,
                view_ylim=(-68.0, -46.0),
            )
        render_three_panel_comparison(
            panels_dir / f"panel_sagittal_{slug}.png",
            plane="sagittal",
            coord_mm=x_mm,
            slice_label=slug.replace("_", " "),
            pet_vol=pet_vol,
            ct_vol=ct_vol,
            rois=rois,
            roi_masks_ct=roi_masks_ct,
            view_ylim=(-68.0, -46.0),
        )

    print(f"Done! Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
