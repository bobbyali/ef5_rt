"""
Processes Set 2 (day2_set5) 4-mouse EF5 PET/CT dataset:
1. Loads the registered PET volume (`day2_set5_pet.dcm`), the raw Siemens Inveon CT volume (`i100118_v1.ct.img` + `.hdr`),
   and the 16-ROI file (`mice_tumor_rois.ROI`).
2. Verifies full-bed PET & CT ROI statistics against `day2_set5_stats.txt`.
3. Renders full-bed 4-mouse overview images showing the 4 mouse bounding boxes (`body` ROIs) and tumours.
4. Crops both PET and CT volumes to each of the 4 mouse bounding boxes (`c8 M0`, `c8 M1R`, `c8 M1L`, `c8 M1L1R`),
   saves each mouse's cropped `<mouse>_PET_mouse.dcm`, `<mouse>_CT_mouse.dcm`, and `<mouse>_rois.ROI` to disk
   (matching the exact structure of Set 1), and runs `run_slice_generator` from `src.ef5_slice_cli`
   to generate the scrollable 3-axis slices and statistics for every mouse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

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
    crop_volume_to_body_roi,
    load_inveon_volume,
    load_rt_image_dicom,
    load_rt_image_roi,
    save_rt_image_dicom,
    save_rt_image_roi_subset,
)
from src.ef5_slice_cli import SliceExportConfig, run_slice_generator
from src.inspect_set1 import get_slice_2d_and_extent, make_pet_overlay_cmap, plot_rois_on_ax, ROI_STYLES


# Define the 4 mice in Set 2 and their associated ROI names in mice_tumor_rois.ROI
SET2_MICE: List[Dict[str, object]] = [
    {
        "mouse_id": "c8_M0",
        "display_name": "c8 M0",
        "body_roi": "c8 M0 body",
        "tumor_And_bg_rois": ["c8 M0 LHS Tumor", "c8 M0 RHS Tumor", "c8 M0 bg ROI"],
    },
    {
        "mouse_id": "c8_M1R",
        "display_name": "c8 M1R",
        "body_roi": "c8 M1R body",
        "tumor_And_bg_rois": ["c8 M1R LHS tumor", "c8 M1R RHS Tumor", "c8 M1R bg ROI"],
    },
    {
        "mouse_id": "c8_M1L",
        "display_name": "c8 M1L",
        "body_roi": "c8 M1L body",
        "tumor_And_bg_rois": ["c8 M1L LHS Tumor", "c8 M1L RHS Tumor", "c8 M1L bg ROI"],
    },
    {
        "mouse_id": "c8_M1L1R",
        "display_name": "c8 M1L1R",
        "body_roi": "c8 M1L1R body",
        "tumor_And_bg_rois": ["c8 M1L1R LHS Tumor", "c8 M1L1R RHS Tumor", "c8 M1L1R bg ROI"],
    },
]


def render_full_bed_overview(
    out_dir: Path,
    pet_vol: RTImageVolume,
    ct_vol: RTImageVolume,
    rois: List[RTRoi],
) -> None:
    """
    Renders transaxial and coronal overview slices of the full 4-mouse scanner bed
    showing the 4 mouse bounding boxes ('body' ROIs) and the 8 tumours + 4 background ROIs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_cmap = make_pet_overlay_cmap("inferno", max_alpha=0.78)

    body_colors = {
        "c8 M0 body": "#00E5FF",
        "c8 M1R body": "#76FF03",
        "c8 M1L body": "#FF4081",
        "c8 M1L1R body": "#FFD740",
    }
    for r in rois:
        if r.name in body_colors:
            ROI_STYLES[r.name] = {"color": body_colors[r.name], "linestyle": "--", "linewidth": 1.8, "label": r.name}
        elif "lhs" in r.name.lower():
            ROI_STYLES[r.name] = {"color": "#00E5FF", "linestyle": "-", "linewidth": 2.0, "label": r.name}
        elif "rhs" in r.name.lower():
            ROI_STYLES[r.name] = {"color": "#FF4081", "linestyle": "-", "linewidth": 2.0, "label": r.name}
        else:
            ROI_STYLES[r.name] = {"color": "#FF9100", "linestyle": ":", "linewidth": 2.0, "label": r.name}

    roi_masks_ct = {r.name: build_roi_3d_mask(ct_vol, r) for r in rois}

    # Transaxial slice at Y = -55.72 mm (passes through all 4 mouse body boxes and multiple tumours)
    for plane, coord_mm, tag in [
        ("transaxial", -55.72, "transaxial_Y-55.7mm_All_4_Mice_Bed"),
        ("transaxial", -51.74, "transaxial_Y-51.7mm_All_4_Mice_Bed"),
        ("coronal", -60.09, "coronal_Z-60.1mm_Bottom_Two_Mice_M0_M1R"),
        ("coronal", -31.26, "coronal_Z-31.3mm_Top_Two_Mice_M1L_M1L1R"),
    ]:
        ct_2d, ct_ext, ct_idx, _ = get_slice_2d_and_extent(ct_vol, plane, coord_mm)
        pet_2d, pet_ext, pet_idx, _ = get_slice_2d_and_extent(pet_vol, plane, coord_mm)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), dpi=120, facecolor="#111111")
        fig.subplots_adjust(left=0.05, right=0.96, bottom=0.16, top=0.84, wspace=0.18)
        titles = [
            f"Full-Bed CT + Bounding Boxes & ROIs (slice {ct_idx})",
            f"Full-Bed EF5 PET + Bounding Boxes & ROIs (slice {pet_idx})",
            "Full-Bed PET/CT Overlay + Bounding Boxes & ROIs",
        ]
        for idx_ax, ax in enumerate(axes):
            ax.set_facecolor("black")
            if idx_ax == 0:
                ax.imshow(ct_2d, cmap="gray", vmin=-500, vmax=800, extent=ct_ext, origin="upper", aspect="equal")
            elif idx_ax == 1:
                ax.imshow(pet_2d, cmap="inferno", vmin=0, vmax=2.8, extent=pet_ext, origin="upper", aspect="equal")
            else:
                ax.imshow(ct_2d, cmap="gray", vmin=-500, vmax=800, extent=ct_ext, origin="upper", aspect="equal")
                ax.imshow(pet_2d, cmap=overlay_cmap, vmin=0, vmax=2.8, extent=pet_ext, origin="upper", aspect="equal")

            plot_rois_on_ax(ax, plane, coord_mm, rois, ct_vol, roi_masks_ct)
            ax.set_xlim(ct_ext[0], ct_ext[1])
            ax.set_ylim(ct_ext[2], ct_ext[3])
            ax.set_title(titles[idx_ax], color="white", fontsize=10.5, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#666666")

        fig.suptitle(
            f"Set 2 Full 4-Mouse Scanner Bed — {plane.capitalize()} @ {coord_mm:.2f} mm",
            color="white", fontsize=14, fontweight="bold"
        )
        legend_handles = [
            Line2D([0], [0], color=c, linestyle="--", linewidth=2.0, label=lbl)
            for lbl, c in body_colors.items()
        ] + [
            Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label="LHS Tumour ROIs"),
            Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label="RHS Tumour ROIs"),
            Line2D([0], [0], color="#FF9100", linestyle=":", linewidth=2.2, label="Background ROIs"),
        ]
        fig.legend(
            handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
            ncol=4, frameon=True, facecolor="#1E1E1E", edgecolor="#555555", labelcolor="white", fontsize=9.0
        )
        fig.savefig(out_dir / f"full_bed_{tag}.png", dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    root = Path("/workspace")
    set2_dir = root / "images" / "set2"
    out_set2 = root / "outputs" / "prelim_work" / "set2"
    cropped_root = out_set2 / "cropped_mice"
    slices_root = out_set2 / "mice_slices"
    out_set2.mkdir(parents=True, exist_ok=True)

    print("1. Loading Set 2 full-bed PET (.dcm), raw Inveon CT (.img/.hdr), and mice_tumor_rois.ROI...")
    pet_full = load_rt_image_dicom(set2_dir / "day2_set5_pet.dcm")
    ct_full = load_inveon_volume(set2_dir / "i100118_v1.ct.img", set2_dir / "i100118_v1.ct.img.hdr", mmap=True)
    roi_ver, rois_all = load_rt_image_roi(set2_dir / "mice_tumor_rois.ROI")
    roi_by_name = {r.name: r for r in rois_all}

    print(f"   PET Full Shape: {pet_full.shape}, Spacing: {pet_full.spacing} mm")
    print(f"   CT  Full Shape: {ct_full.shape}, Spacing: {ct_full.spacing} mm")
    print(f"   ROIs Loaded   : {len(rois_all)} ({roi_ver})")

    # 2. Compute full-bed statistics matching day2_set5_stats.txt
    print("\n2. Computing full-bed Set 2 statistics (PET on all 16 ROIs, CT on the 12 tumour/bg ROIs)...")
    pet_stats = compute_roi_statistics([pet_full], rois_all)
    ct_stats = compute_roi_statistics([ct_full], [r for r in rois_all if "body" not in r.name.lower()])
    full_stats = pd.concat([pet_stats, ct_stats], ignore_index=True)

    full_stats_csv = out_set2 / "set2_full_roi_stats.csv"
    full_stats_tsv = out_set2 / "set2_full_roi_stats.txt"
    full_stats.to_csv(full_stats_csv, index=False)
    rt_cols = ["ROI Label", "Image Label", "Image Units", "Points", "Volume (cm3)", "Minimum", "Maximum", "Mean", "Std Dev", "Sum"]
    full_stats[rt_cols].rename(columns={"Volume (cm3)": "Volume"}).to_csv(full_stats_tsv, sep="\t", index=False)
    print(full_stats.to_string(index=False))

    # 3. Render full-bed 4-mouse overview
    print("\n3. Rendering full-bed 4-mouse overview figures...")
    render_full_bed_overview(out_set2 / "full_bed_overview", pet_full, ct_full, rois_all)

    # 4. Crop each mouse using its body bounding box, save cropped DICOMs + ROI file to disk,
    #    and run the Set 1 slice generator (`run_slice_generator`) for each mouse!
    all_mice_stats = []
    for m_info in SET2_MICE:
        mouse_id = str(m_info["mouse_id"])
        display_name = str(m_info["display_name"])
        body_roi_name = str(m_info["body_roi"])
        sub_roi_names = list(m_info["tumor_And_bg_rois"])

        print(f"\n4. Processing mouse '{display_name}' ({mouse_id}) using bounding box '{body_roi_name}'...")
        body_roi = roi_by_name[body_roi_name]
        sub_rois = [roi_by_name[name] for name in sub_roi_names]

        # Crop PET and CT volumes to this mouse's 3D bounding box
        pet_cropped = crop_volume_to_body_roi(pet_full, body_roi)
        ct_cropped = crop_volume_to_body_roi(ct_full, body_roi)

        print(f"   Cropped PET shape={pet_cropped.shape}, origin={pet_cropped.origin.round(4)}")
        print(f"   Cropped CT  shape={ct_cropped.shape}, origin={ct_cropped.origin.round(4)}")

        # Save cropped DICOMs and per-mouse binary .ROI file to disk
        mouse_crop_dir = cropped_root / mouse_id
        pet_dcm_path = save_rt_image_dicom(pet_cropped, mouse_crop_dir / f"{mouse_id}_PET_mouse.dcm")
        ct_dcm_path = save_rt_image_dicom(ct_cropped, mouse_crop_dir / f"{mouse_id}_CT_mouse.dcm")
        roi_sub_path = save_rt_image_roi_subset(roi_ver, sub_rois, mouse_crop_dir / "rois.ROI")

        print(f"   Saved cropped files to {mouse_crop_dir}:")
        print(f"     - {pet_dcm_path.name}")
        print(f"     - {ct_dcm_path.name}")
        print(f"     - {roi_sub_path.name}")

        # Configure distinct colors for LHS Tumor, RHS Tumor, and bg ROI for this mouse
        for r in sub_rois:
            if "lhs" in r.name.lower():
                ROI_STYLES[r.name] = {"color": "#00E5FF", "linestyle": "-", "linewidth": 2.2, "label": f"{r.name} (Left Tumour)"}
            elif "rhs" in r.name.lower():
                ROI_STYLES[r.name] = {"color": "#FF4081", "linestyle": "-", "linewidth": 2.2, "label": f"{r.name} (Right Tumour)"}
            else:
                ROI_STYLES[r.name] = {"color": "#FF9100", "linestyle": ":", "linewidth": 2.2, "label": f"{r.name} (Background)"}

        # Reuse `run_slice_generator` from `src.ef5_slice_cli` on the saved cropped mouse files!
        # Use step_pet_slices=2 across the ROI region so all 4 mice generate quickly with full 3-axis coverage
        mouse_out_dir = slices_root / mouse_id
        cfg = SliceExportConfig(
            pet_path=pet_dcm_path,
            ct_path=ct_dcm_path,
            roi_path=roi_sub_path,
            out_dir=mouse_out_dir,
            planes=("transaxial", "coronal", "sagittal"),
            step_pet_slices=2,
            scope="roi",
            margin_mm=0.0,
            modes=("CT", "PET", "Overlay", "Panel"),
            ct_clim=(-500.0, 800.0),
            pet_clim=(0.0, 2.8),
            overlay_alpha=0.78,
            crop_long_axis_to_rois=True,
            long_axis_pad_mm=6.0,
            dpi=120,
        )
        m_stats_df, _ = run_slice_generator(cfg)
        m_stats_df.insert(0, "Mouse ID", mouse_id)
        all_mice_stats.append(m_stats_df)

    combined_mice_df = pd.concat(all_mice_stats, ignore_index=True)
    combined_mice_df.to_csv(out_set2 / "set2_cropped_mice_roi_stats.csv", index=False)
    print(f"\nAll 4 Set 2 mice cropped and processed! Outputs in {out_set2}")


if __name__ == "__main__":
    main()
