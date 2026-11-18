"""
Consolidated processing pipeline for the First EF5 Radiation Therapy (RT) Study (Dec 2010 - Jan 2011).

Processes the 6 A549 subcutaneous tumour mice (c2_M0, c2_M1L, c2_M1R, c2_M1L1R, c2_M2R, c1_M2R)
across all 3 EF5 PET/CT imaging timepoints:
  - Day -1  (2010-12-02, pre-RT, 20 Gy delivered on 2010-12-03)
  - Day +13 (2010-12-15, post-RT)
  - Day +32 (2011-01-03, post-RT)

Outputs:
  1. Per-mouse, per-timepoint cropped PET DICOM, canonical axis-aligned CT DICOM (when registered CT exists),
     per-mouse binary RT_Image `rois.ROI`, and visual QC overview PNGs under:
       outputs/first_ef5_rt_study/cropped_mice/<mouse_id>/<timepoint>/
  2. Longitudinal montage PNG per mouse under:
       outputs/first_ef5_rt_study/montages/<mouse_id>_longitudinal.png
  3. Curated, publication-ready summary CSV tables (saved to both outputs/first_ef5_rt_study/tables/
     and git-tracked results/first_ef5_rt_study/):
       - mouse_id_mapping.csv
       - longitudinal_imaging_summary.csv
       - caliper_growth_curves.csv
       - tumour_master_wide.csv
       - study_timeline.csv
"""

from __future__ import annotations

import datetime
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform

from src.inspect_set1 import (
    ROI_STYLES,
    get_slice_2d_and_extent,
    make_pet_overlay_cmap,
    plot_rois_on_ax,
)
from src.inspect_study_records import read_xlsx_sheets
from src.rt_image_loader import (
    RTRoi,
    RTImageVolume,
    build_roi_3d_mask,
    crop_volume_to_body_roi,
    load_rt_image_dicom,
    load_rt_image_roi,
    save_rt_image_dicom,
    save_rt_image_roi_subset,
)


# ---------------------------------------------------------------------------
# Definitive study metadata & scan manifest for the 6 A549 RT mice
# ---------------------------------------------------------------------------
RT_DATE = datetime.date(2010, 12, 3)
INJECTION_DATE = datetime.date(2010, 10, 29)

MOUSE_REGISTRY: List[Dict[str, str]] = [
    {
        "mouse_id": "c2_M0",
        "cage": "A549 c2",
        "ear_punch": "0",
        "roi_id_2010": "m1",
        "caliper_key": ("A549 c2", "O"),
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "None (died before 2011-01-03)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Died between 2010-12-30 and 2011-01-03",
        "notes": "Imaged pre-RT (Day -1) and Day +13; last caliper 2010-12-30.",
    },
    {
        "mouse_id": "c2_M1L",
        "cage": "A549 c2",
        "ear_punch": "1L",
        "roi_id_2010": "m2",
        "caliper_key": ("A549 c2", "1L"),
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 2 (PET+CT)",
        "day32_set": "2011-01-03 Set 8 (PET only)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "Complete 3-timepoint PET series; Day +32 Set 8 has no registered CT in RT_Image.",
    },
    {
        "mouse_id": "c2_M1R",
        "cage": "A549 c2",
        "ear_punch": "1R",
        "roi_id_2010": "m3",
        "caliper_key": ("A549 c2", "1R"),
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 2 (PET+CT)",
        "day32_set": "2011-01-03 Set 9 (PET+CT)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "Complete 3-timepoint PET+CT series. Note: Growth Curves A549 Analysis sheet had Post-RT 1R LHS/RHS PET values swapped (fixed here from ROIs).",
    },
    {
        "mouse_id": "c2_M1L1R",
        "cage": "A549 c2",
        "ear_punch": "1L1R",
        "roi_id_2010": "m4",
        "caliper_key": ("A549 c2", "1L1R"),
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "2011-01-03 Set 8 (PET only)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "Complete 3-timepoint PET series; Day +32 Set 8 has no registered CT. Note: 2011 EF5 RT data sheet divided m4 LHS by m3 muscle (1.786 instead of true 1.627; fixed here).",
    },
    {
        "mouse_id": "c2_M2R",
        "cage": "A549 c2",
        "ear_punch": "2R",
        "roi_id_2010": "m5",
        "caliper_key": ("A549 c2", "2R"),
        "day_minus1_set": "2010-12-02 Set 2 (PET only)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "2011-01-03 Set 9 (PET+CT)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "Complete 3-timepoint PET series; Day -1 Set 2 has no registered CT in RT_Image.",
    },
    {
        "mouse_id": "c1_M2R",
        "cage": "A549 c1",
        "ear_punch": "2R",
        "roi_id_2010": "m6",
        "caliper_key": ("A549 c1", "2R"),
        "day_minus1_set": "2010-12-02 Set 2 (PET only)",
        "day13_set": "None (died 2010-12-12)",
        "day32_set": "None (died 2010-12-12)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Died 2010-12-12 (Day +9 post-RT); sole cage 1 survivor of 2010-11-27 flood",
        "notes": "Imaged pre-RT only (Day -1 Set 2). Note: Growth Curves A549 Analysis sheet had c1 2R LHS/RHS pre-RT EF5 swapped (fixed here from ROIs).",
    },
]

MOUSE_BY_ID = {m["mouse_id"]: m for m in MOUSE_REGISTRY}


@dataclass(frozen=True)
class MouseScanSpec:
    mouse_id: str
    timepoint_slug: str
    scan_date: str
    days_post_rt: int
    session_rel_dir: str
    scan_set_label: str
    body_roi: str
    rt_roi: str
    ctrl_roi: str
    muscle_roi: str


SCAN_SPECS: List[MouseScanSpec] = [
    # --- Day -1 (2010-12-02, Pre-RT) ---
    MouseScanSpec("c2_M0",    "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set1", "02 Dec Set 1", "M0",     "M0 RHS RT",     "M0 LHS Control",     "M0 muscle"),
    MouseScanSpec("c2_M1L",   "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set1", "02 Dec Set 1", "M1L",    "M1L RHS RT",    "M1L LHS Control",    "M1L muscle"),
    MouseScanSpec("c2_M1R",   "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set1", "02 Dec Set 1", "M1R",    "M1R RHS RT",    "M1R LHS Control",    "M1R muscle"),
    MouseScanSpec("c2_M1L1R", "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set1", "02 Dec Set 1", "M1L1R",  "M1L1R RHS RT",  "M1L1R LHS Control",  "M1L1R muscle"),
    MouseScanSpec("c2_M2R",   "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set2", "02 Dec Set 2", "M2R",    "M2R RHS RT",    "M2R LHS Control",    "M2R muscle"),
    MouseScanSpec("c1_M2R",   "day-1_2010-12-02", "2010-12-02", -1, "2010-12-02_day-1_pre_rt/set2", "02 Dec Set 2", "c1 M2R", "c1 M2R RHS RT", "c1 M2R LHS Control", "c1 M2R muscle"),
    # --- Day +13 (2010-12-15, Post-RT) ---
    MouseScanSpec("c2_M1L",   "day13_2010-12-15", "2010-12-15", 13, "2010-12-15_day13_post_rt/set2", "15 Dec Set 2", "1L",    "1L RHS RT",     "1L LHS Control",     "1L Muscle"),
    MouseScanSpec("c2_M1R",   "day13_2010-12-15", "2010-12-15", 13, "2010-12-15_day13_post_rt/set2", "15 Dec Set 2", "1R",    "1R RHS RT",     "1R LHS Control",     "1R Muscle"),
    MouseScanSpec("c2_M0",    "day13_2010-12-15", "2010-12-15", 13, "2010-12-15_day13_post_rt/set3", "15 Dec Set 3", "M0",    "M0 RHS RT",     "M0 LHS Control",     "M0 Muscle"),
    MouseScanSpec("c2_M1L1R", "day13_2010-12-15", "2010-12-15", 13, "2010-12-15_day13_post_rt/set3", "15 Dec Set 3", "M1L1R", "M1L1R RHS RT",  "M1L1R LHS Control",  "M1L1R Muscle"),
    MouseScanSpec("c2_M2R",   "day13_2010-12-15", "2010-12-15", 13, "2010-12-15_day13_post_rt/set3", "15 Dec Set 3", "M2R",   "M2R RHS RT",    "M2R LHS Control",    "M2R Muscle"),
    # --- Day +32 (2011-01-03, Post-RT) ---
    MouseScanSpec("c2_M1L1R", "day32_2011-01-03", "2011-01-03", 32, "2011-01-03_day32_post_rt/set8", "03 Jan Set 8", "A549 c2 1L1R", "A549c2 1L1R RHS RT", "A549c2 1L1R LHS Control", "A549c2 1L1R Muscle"),
    MouseScanSpec("c2_M1L",   "day32_2011-01-03", "2011-01-03", 32, "2011-01-03_day32_post_rt/set8", "03 Jan Set 8", "A549 c2 1L",   "A549c2 1L RHS RT",   "A549c2 1L LHS Control",   "A549c2 1L Muscle"),
    MouseScanSpec("c2_M1R",   "day32_2011-01-03", "2011-01-03", 32, "2011-01-03_day32_post_rt/set9", "03 Jan Set 9", "A549c2 1R",    "A549c2 1R RHS RT",   "A549c2 1R LHS Control",   "A549c2 1R Muscle"),
    MouseScanSpec("c2_M2R",   "day32_2011-01-03", "2011-01-03", 32, "2011-01-03_day32_post_rt/set9", "03 Jan Set 9", "A549c2 2R",    "A549c2 2R RHS RT",   "A549c2 2R LHS Control",   "A549c2 2R Muscle"),
]


# ---------------------------------------------------------------------------
# Helpers for Excel date parsing & caliper extraction
# ---------------------------------------------------------------------------
def _uses_1904_dates(xlsx_path: Path) -> bool:
    with zipfile.ZipFile(xlsx_path) as zf:
        wb = zf.read("xl/workbook.xml").decode("utf-8", errors="replace")
        return 'date1904="1"' in wb or 'date1904="true"' in wb.lower()


def _excel_serial_to_date(val: str, is_1904: bool) -> Optional[datetime.date]:
    try:
        n = float(val)
        if 38000 < n < 45000:
            base = datetime.date(1904, 1, 1) if is_1904 else datetime.date(1899, 12, 30)
            return base + datetime.timedelta(days=int(n))
    except Exception:
        pass
    return None


def extract_calipers_and_timeline(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parses the A549 caliper measurements and study timeline from
    `08 EF5 IR Study Growth Curves i.xlsx`.
    """
    is_1904 = _uses_1904_dates(xlsx_path)
    sheets = read_xlsx_sheets(xlsx_path)

    # 1. Timeline (Tumor Sizes R1..R38)
    ts_rows = sheets["Tumor Sizes"]
    timeline_records = []
    last_date_str = ""
    for r in ts_rows[1:38]:
        d_raw = r[0].strip() if len(r) > 0 else ""
        ev = r[1].strip() if len(r) > 1 else ""
        if not ev:
            continue
        dt = _excel_serial_to_date(d_raw, is_1904)
        note = ""
        if dt is not None:
            # Fix year typo in original spreadsheet where Dec 15-30 2010 were entered as Dec 2011
            if dt.year == 2011 and dt.month == 12:
                note = f"Original spreadsheet date {dt.isoformat()} corrected to 2010"
                dt = datetime.date(2010, dt.month, dt.day)
            last_date_str = dt.isoformat()
        date_str = last_date_str
        dt_parsed = datetime.date.fromisoformat(date_str)
        timeline_records.append(
            {
                "date": date_str,
                "days_post_injection": (dt_parsed - INJECTION_DATE).days,
                "days_post_rt": (dt_parsed - RT_DATE).days,
                "event": ev,
                "audit_note": note,
            }
        )
    timeline_df = pd.DataFrame(timeline_records)

    # 2. A549 Caliper Measurements (from `A549 Analysis` R1..R13, cross-verified with `Tumor Sizes`)
    a549_rows = sheets["A549 Analysis"]
    header = a549_rows[0]
    date_cols: List[Tuple[int, datetime.date]] = []
    for ci, cell in enumerate(header):
        dt = _excel_serial_to_date(cell, is_1904)
        if dt is not None:
            date_cols.append((ci, dt))

    caliper_key_to_mouse = {m["caliper_key"]: m for m in MOUSE_REGISTRY}
    caliper_records = []
    cur_cage = ""
    cur_ear = ""
    for r in a549_rows[1:13]:
        if len(r) < 3:
            continue
        if r[0].strip():
            cur_cage = r[0].strip()
        if r[1].strip():
            cur_ear = r[1].strip()
        side_2010 = r[2].strip().upper()
        m_info = caliper_key_to_mouse.get((cur_cage, cur_ear))
        if m_info is None:
            continue
        treatment = "RT_20Gy" if side_2010 == "LHS" else "Control_0Gy"
        roi_side_2013 = "RHS RT" if treatment == "RT_20Gy" else "LHS Control"

        # Find pre-RT (2010-12-02) baseline caliper diameter for normalization
        pre_rt_mm = np.nan
        for ci, dt in date_cols:
            if dt == datetime.date(2010, 12, 2) and ci < len(r) and r[ci].strip():
                try:
                    pre_rt_mm = float(r[ci])
                except ValueError:
                    pass

        for ci, dt in date_cols:
            if ci >= len(r) or not r[ci].strip():
                continue
            try:
                val_mm = float(r[ci])
            except ValueError:
                continue
            caliper_records.append(
                {
                    "mouse_id": m_info["mouse_id"],
                    "cage": m_info["cage"],
                    "ear_punch": m_info["ear_punch"],
                    "roi_id_2010": m_info["roi_id_2010"],
                    "tumour_id": f"{m_info['mouse_id']}_{'RT' if treatment == 'RT_20Gy' else 'Ctrl'}",
                    "treatment": treatment,
                    "dose_gy": 20 if treatment == "RT_20Gy" else 0,
                    "caliper_side_2010": side_2010,
                    "roi_side_2013": roi_side_2013,
                    "date": dt.isoformat(),
                    "days_post_injection": (dt - INJECTION_DATE).days,
                    "days_post_rt": (dt - RT_DATE).days,
                    "diameter_mm": round(val_mm, 2),
                    "norm_diameter_vs_day_minus1": round(val_mm / pre_rt_mm, 4) if not np.isnan(pre_rt_mm) else np.nan,
                    "delta_diameter_vs_day_minus1_mm": round(val_mm - pre_rt_mm, 2) if not np.isnan(pre_rt_mm) else np.nan,
                }
            )

    caliper_df = pd.DataFrame(caliper_records)
    return caliper_df, timeline_df


# ---------------------------------------------------------------------------
# Visualization helpers for QC overviews & per-mouse longitudinal montages
# ---------------------------------------------------------------------------
PET_OVERLAY_CMAP = make_pet_overlay_cmap("inferno", max_alpha=0.78)


def _configure_mouse_roi_styles(rt_name: str, ctrl_name: str, muscle_name: str) -> None:
    ROI_STYLES[rt_name] = {
        "color": "#FF4081",
        "linestyle": "-",
        "linewidth": 2.2,
        "label": f"{rt_name} (Irradiated 20 Gy)",
    }
    ROI_STYLES[ctrl_name] = {
        "color": "#00E5FF",
        "linestyle": "-",
        "linewidth": 2.2,
        "label": f"{ctrl_name} (Control 0 Gy)",
    }
    ROI_STYLES[muscle_name] = {
        "color": "#FFD740",
        "linestyle": ":",
        "linewidth": 2.0,
        "label": f"{muscle_name} (Background Muscle)",
    }


def _ct_display_clim(ct_vol: RTImageVolume) -> Tuple[float, float]:
    """Selects CT grayscale window (Gamma Medica unshifted HU vs standard HU)."""
    med = float(np.median(ct_vol.data[::4, ::4, ::4]))
    if med > -200.0:
        return (0.0, 2200.0)
    return (-500.0, 800.0)


def render_mouse_qc_overview(
    out_path: Path,
    spec: MouseScanSpec,
    pet_vol: RTImageVolume,
    ct_vol: Optional[RTImageVolume],
    sub_rois: List[RTRoi],
    tm_rt: float,
    tm_ctrl: float,
) -> None:
    """
    Renders a clean 2-row QC figure for a cropped mouse timepoint:
      Row 1: Mid-tumour Transaxial slice (CT | PET | Overlay, or PET alone if PET-only)
      Row 2: Mid-tumour Coronal slice (CT | PET | Overlay, or PET alone if PET-only)
    """
    rt_roi, ctrl_roi, muscle_roi = sub_rois[0], sub_rois[1], sub_rois[2]
    _configure_mouse_roi_styles(rt_roi.name, ctrl_roi.name, muscle_roi.name)

    ref_vol = ct_vol if ct_vol is not None else pet_vol
    roi_masks = {r.name: build_roi_3d_mask(ref_vol, r) for r in sub_rois}

    # Mid-tumour transaxial Y and coronal Z coordinates (average of the two tumour centroids)
    y_mid = 0.5 * (float(rt_roi.centroid_mm[1]) + float(ctrl_roi.centroid_mm[1]))
    z_mid = 0.5 * (float(rt_roi.centroid_mm[2]) + float(ctrl_roi.centroid_mm[2]))

    # Coronal Y zoom around tumours + muscle
    all_pts = np.vstack([r.all_vertices for r in sub_rois])
    y_lo = max(float(ref_vol.y_coords.min()), float(all_pts[:, 1].min()) - 8.0)
    y_hi = min(float(ref_vol.y_coords.max()), float(all_pts[:, 1].max()) + 8.0)

    pet_clim = (0.0, 2.8)
    has_ct = ct_vol is not None
    ct_clim = _ct_display_clim(ct_vol) if ct_vol is not None else (0.0, 2200.0)

    if has_ct:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=120, facecolor="#111111")
        fig.subplots_adjust(left=0.06, right=0.96, bottom=0.11, top=0.88, wspace=0.18, hspace=0.26)
        row_specs = [("transaxial", y_mid, None), ("coronal", z_mid, (y_lo, y_hi))]
        for r_idx, (plane, coord, ylim) in enumerate(row_specs):
            ct_2d, ct_ext, ct_i, _ = get_slice_2d_and_extent(ct_vol, plane, coord)
            pet_2d, pet_ext, pet_i, _ = get_slice_2d_and_extent(pet_vol, plane, coord)
            col_titles = [
                f"{plane.capitalize()} CT (slice {ct_i})",
                f"{plane.capitalize()} EF5 PET (slice {pet_i})",
                f"{plane.capitalize()} PET/CT Overlay",
            ]
            for c_idx in range(3):
                ax = axes[r_idx, c_idx]
                ax.set_facecolor("black")
                if c_idx == 0:
                    ax.imshow(ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1], extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal")
                elif c_idx == 1:
                    ax.imshow(pet_2d, cmap="inferno", vmin=pet_clim[0], vmax=pet_clim[1], extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal")
                else:
                    ax.imshow(ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1], extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal")
                    ax.imshow(pet_2d, cmap=PET_OVERLAY_CMAP, vmin=pet_clim[0], vmax=pet_clim[1], extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal")
                plot_rois_on_ax(ax, plane, coord, sub_rois, ref_vol, roi_masks)
                ax.set_xlim(ct_ext[0], ct_ext[1])
                ax.set_ylim(ylim[0], ylim[1]) if ylim is not None else ax.set_ylim(ct_ext[2], ct_ext[3])
                ax.set_title(col_titles[c_idx], color="white", fontsize=10, fontweight="bold")
                ax.tick_params(colors="white", labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#555555")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=120, facecolor="#111111")
        fig.subplots_adjust(left=0.07, right=0.95, bottom=0.18, top=0.83, wspace=0.20)
        for idx, (plane, coord, ylim) in enumerate([("transaxial", y_mid, None), ("coronal", z_mid, (y_lo, y_hi))]):
            ax = axes[idx]
            ax.set_facecolor("black")
            pet_2d, pet_ext, pet_i, _ = get_slice_2d_and_extent(pet_vol, plane, coord)
            ax.imshow(pet_2d, cmap="inferno", vmin=pet_clim[0], vmax=pet_clim[1], extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal")
            plot_rois_on_ax(ax, plane, coord, sub_rois, pet_vol, roi_masks)
            ax.set_xlim(pet_ext[0], pet_ext[1])
            ax.set_ylim(ylim[0], ylim[1]) if ylim is not None else ax.set_ylim(pet_ext[2], pet_ext[3])
            ax.set_title(f"{plane.capitalize()} EF5 PET (slice {pet_i}, PET-only)", color="white", fontsize=10.5, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555555")

    m_info = MOUSE_BY_ID[spec.mouse_id]
    fig.suptitle(
        f"{spec.mouse_id} ({m_info['roi_id_2010']}, {m_info['cage']} ear '{m_info['ear_punch']}') — "
        f"Day {spec.days_post_rt:+d} ({spec.scan_date}, {spec.scan_set_label})\n"
        f"T/M Mean: RT (20 Gy) = {tm_rt:.2f}  |  Control (0 Gy) = {tm_ctrl:.2f}",
        color="white",
        fontsize=12.5,
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label=f"RT 20 Gy ({rt_roi.name})"),
        Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label=f"Control 0 Gy ({ctrl_roi.name})"),
        Line2D([0], [0], color="#FFD740", linestyle=":", linewidth=2.2, label=f"Muscle ({muscle_roi.name})"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=True,
        facecolor="#1E1E1E",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=9,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_longitudinal_montage(
    out_path: Path,
    mouse_id: str,
    entries: List[Tuple[MouseScanSpec, RTImageVolume, Optional[RTImageVolume], List[RTRoi], float, float]],
) -> None:
    """Renders a 1xN longitudinal coronal EF5 PET(/CT) montage across available timepoints for one mouse."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), dpi=120, facecolor="#111111")
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.16, top=0.80, wspace=0.18)

    for ax, (spec, pet_vol, ct_vol, sub_rois, tm_rt, tm_ctrl) in zip(axes, entries):
        rt_roi, ctrl_roi, muscle_roi = sub_rois[0], sub_rois[1], sub_rois[2]
        _configure_mouse_roi_styles(rt_roi.name, ctrl_roi.name, muscle_roi.name)
        ref_vol = ct_vol if ct_vol is not None else pet_vol
        roi_masks = {r.name: build_roi_3d_mask(ref_vol, r) for r in sub_rois}

        z_mid = 0.5 * (float(rt_roi.centroid_mm[2]) + float(ctrl_roi.centroid_mm[2]))
        all_pts = np.vstack([r.all_vertices for r in sub_rois])
        y_lo = max(float(ref_vol.y_coords.min()), float(all_pts[:, 1].min()) - 8.0)
        y_hi = min(float(ref_vol.y_coords.max()), float(all_pts[:, 1].max()) + 8.0)

        ax.set_facecolor("black")
        pet_2d, pet_ext, _, _ = get_slice_2d_and_extent(pet_vol, "coronal", z_mid)
        if ct_vol is not None:
            ct_clim = _ct_display_clim(ct_vol)
            ct_2d, ct_ext, _, _ = get_slice_2d_and_extent(ct_vol, "coronal", z_mid)
            ax.imshow(ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1], extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal")
            ax.imshow(pet_2d, cmap=PET_OVERLAY_CMAP, vmin=0.0, vmax=2.8, extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal")
            ax.set_xlim(ct_ext[0], ct_ext[1])
            mod_tag = "PET/CT"
        else:
            ax.imshow(pet_2d, cmap="inferno", vmin=0.0, vmax=2.8, extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal")
            ax.set_xlim(pet_ext[0], pet_ext[1])
            mod_tag = "PET only"

        plot_rois_on_ax(ax, "coronal", z_mid, sub_rois, ref_vol, roi_masks)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(
            f"Day {spec.days_post_rt:+d} ({spec.scan_date})\n{mod_tag} | T/M: RT={tm_rt:.2f}, Ctrl={tm_ctrl:.2f}",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

    m_info = MOUSE_BY_ID[mouse_id]
    fig.suptitle(
        f"Longitudinal EF5 PET/CT — {mouse_id} ({m_info['roi_id_2010']}, {m_info['cage']} ear '{m_info['ear_punch']}')",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label="Irradiated (20 Gy)"),
        Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label="Control (0 Gy)"),
        Line2D([0], [0], color="#FFD740", linestyle=":", linewidth=2.2, label="Muscle BG"),
    ]
    fig.legend(
        handles=handles,
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


# ---------------------------------------------------------------------------
# Quantitative metrics & main pipeline
# ---------------------------------------------------------------------------
def _vol_stats(vol: RTImageVolume, roi: RTRoi) -> Dict[str, float]:
    mask = build_roi_3d_mask(vol, roi)
    vals = np.asarray(vol.data[mask], dtype=np.float64)
    dz, dy, dx = vol.spacing
    vox_cm3 = (dx * dy * dz) / 1000.0
    n = int(vals.size)
    vol_cm3 = n * vox_cm3
    if n == 0:
        return {
            "points": 0,
            "volume_cm3": 0.0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "sum": np.nan,
        }
    return {
        "points": n,
        "volume_cm3": round(vol_cm3, 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std(ddof=1)) if n > 1 else 0.0, 4),
        "p90": round(float(np.percentile(vals, 90)), 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
        "sum": round(float(vals.mean()) * vol_cm3, 5),
    }


def _quadrant(roi: RTRoi, full_pet: RTImageVolume) -> str:
    x_mid = 0.5 * (float(full_pet.x_coords.min()) + float(full_pet.x_coords.max()))
    z_mid = 0.5 * (float(full_pet.z_coords.min()) + float(full_pet.z_coords.max()))
    cx, _, cz = roi.centroid_mm
    vert = "Top" if cz >= z_mid else "Bottom"
    horiz = "Left" if cx >= x_mid else "Right"
    return f"{vert}-{horiz}"


def save_voxel_bundle_npz(
    out_path: Path,
    spec: MouseScanSpec,
    pet_crop: RTImageVolume,
    ct_crop: Optional[RTImageVolume],
    rt_roi: RTRoi,
    ctrl_roi: RTRoi,
    muscle_roi: RTRoi,
    muscle_mean_idg: float,
    muscle_std_idg: float,
) -> Path:
    """
    Exports a compressed NumPy .npz voxel bundle containing aligned 3D EF5 PET (%ID/g),
    muscle-normalized T/M ratio map, binary tumour/muscle masks, and (when available)
    both native-resolution canonical CT and CT resampled onto the PET voxel grid.
    """
    m_info = MOUSE_BY_ID[spec.mouse_id]
    pet_idg = np.asarray(pet_crop.data, dtype=np.float32)
    pet_tm = (
        (pet_idg / np.float32(muscle_mean_idg)).astype(np.float32)
        if muscle_mean_idg and muscle_mean_idg > 0
        else np.full_like(pet_idg, np.nan, dtype=np.float32)
    )
    mask_rt_pet = build_roi_3d_mask(pet_crop, rt_roi)
    mask_ctrl_pet = build_roi_3d_mask(pet_crop, ctrl_roi)
    mask_muscle_pet = build_roi_3d_mask(pet_crop, muscle_roi)

    payload: Dict[str, object] = {
        "mouse_id": np.array(spec.mouse_id),
        "roi_id_2010": np.array(m_info["roi_id_2010"]),
        "cage": np.array(m_info["cage"]),
        "ear_punch": np.array(m_info["ear_punch"]),
        "scan_date": np.array(spec.scan_date),
        "days_post_rt": np.int32(spec.days_post_rt),
        "scan_set": np.array(spec.scan_set_label),
        "rt_roi_label": np.array(rt_roi.name),
        "ctrl_roi_label": np.array(ctrl_roi.name),
        "muscle_roi_label": np.array(muscle_roi.name),
        "muscle_mean_idg": np.float32(muscle_mean_idg),
        "muscle_std_idg": np.float32(muscle_std_idg),
        "axis_convention": np.array(
            "axis0=Y(origin[1]-k*dz), axis1=Z(origin[2]-j*dy), axis2=X(origin[0]-i*dx)"
        ),
        "pet_idg": pet_idg,
        "pet_tm": pet_tm,
        "mask_rt_pet": mask_rt_pet,
        "mask_ctrl_pet": mask_ctrl_pet,
        "mask_muscle_pet": mask_muscle_pet,
        "pet_spacing_mm": np.array(pet_crop.spacing, dtype=np.float64),
        "pet_origin_mm": np.array(pet_crop.origin, dtype=np.float64),
        "pet_x_mm": pet_crop.x_coords.astype(np.float64),
        "pet_y_mm": pet_crop.y_coords.astype(np.float64),
        "pet_z_mm": pet_crop.z_coords.astype(np.float64),
        "ct_available": np.bool_(ct_crop is not None),
    }

    if ct_crop is not None:
        ct_hu = np.asarray(ct_crop.data, dtype=np.float32)
        dz_p, dy_p, dx_p = pet_crop.spacing
        dz_c, dy_c, dx_c = ct_crop.spacing
        ox_p, oy_p, oz_p = pet_crop.origin
        ox_c, oy_c, oz_c = ct_crop.origin

        scale = np.array([dz_p / dz_c, dy_p / dy_c, dx_p / dx_c], dtype=np.float64)
        offset = np.array(
            [(oy_c - oy_p) / dz_c, (oz_c - oz_p) / dy_c, (ox_c - ox_p) / dx_c],
            dtype=np.float64,
        )
        cval = -1000.0 if float(np.median(ct_hu[::4, ::4, ::4])) < -200.0 else 0.0
        ct_on_pet = affine_transform(
            ct_hu,
            matrix=scale,
            offset=offset,
            output_shape=pet_crop.shape,
            order=1,
            mode="constant",
            cval=cval,
        ).astype(np.float32)

        payload.update(
            {
                "ct_on_pet_grid": ct_on_pet,
                "ct_hu": ct_hu,
                "mask_rt_ct": build_roi_3d_mask(ct_crop, rt_roi),
                "mask_ctrl_ct": build_roi_3d_mask(ct_crop, ctrl_roi),
                "mask_muscle_ct": build_roi_3d_mask(ct_crop, muscle_roi),
                "ct_spacing_mm": np.array(ct_crop.spacing, dtype=np.float64),
                "ct_origin_mm": np.array(ct_crop.origin, dtype=np.float64),
                "ct_x_mm": ct_crop.x_coords.astype(np.float64),
                "ct_y_mm": ct_crop.y_coords.astype(np.float64),
                "ct_z_mm": ct_crop.z_coords.astype(np.float64),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    root = Path("/workspace")
    cons_dir = root / "images" / "set3" / "consolidated_a549"
    rep_dir = root / "images" / "set3" / "reports"
    out_root = root / "outputs" / "first_ef5_rt_study"
    cropped_root = out_root / "cropped_mice"
    voxel_root = out_root / "voxel_bundles"
    montage_root = out_root / "montages"
    out_tables = out_root / "tables"
    git_results = root / "results" / "first_ef5_rt_study"

    for d in [cropped_root, voxel_root, montage_root, out_tables, git_results]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Extract calipers & timeline from original study spreadsheet
    print("1. Extracting A549 caliper growth curves & study timeline...")
    caliper_df, timeline_df = extract_calipers_and_timeline(
        rep_dir / "08 EF5 IR Study Growth Curves i.xlsx"
    )
    print(f"   Extracted {len(caliper_df)} caliper rows and {len(timeline_df)} timeline events.")

    # Helper to look up closest caliper measurement for a (mouse_id, treatment, days_post_rt)
    def nearest_caliper(mouse_id: str, treatment: str, target_day: int) -> Tuple[str, Optional[int], Optional[float]]:
        sub = caliper_df[(caliper_df["mouse_id"] == mouse_id) & (caliper_df["treatment"] == treatment)]
        if sub.empty:
            return ("", None, None)
        idx = (sub["days_post_rt"] - target_day).abs().idxmin()
        row = sub.loc[idx]
        if abs(int(row["days_post_rt"]) - target_day) > 7:
            return ("", None, None)
        return (str(row["date"]), int(row["days_post_rt"]), float(row["diameter_mm"]))

    # 2. Load each consolidated full-bed session once (avoid reloading 262 MB CTs multiple times)
    print("\n2. Processing all 15 mouse-timepoint scans (cropping PET/CT, ROIs, stats, QC images)...")
    specs_by_session: Dict[str, List[MouseScanSpec]] = {}
    for sp in SCAN_SPECS:
        specs_by_session.setdefault(sp.session_rel_dir, []).append(sp)

    imaging_rows: List[Dict[str, object]] = []
    montage_cache: Dict[str, List[Tuple[MouseScanSpec, RTImageVolume, Optional[RTImageVolume], List[RTRoi], float, float]]] = {}

    for sess_rel, sess_specs in specs_by_session.items():
        sdir = cons_dir / sess_rel
        print(f"\n   Loading session: {sess_rel} ({len(sess_specs)} study mice)")
        pet_full = load_rt_image_dicom(sdir / "image1.dcm")
        ct_path = sdir / "image2.dcm"
        ct_full = load_rt_image_dicom(ct_path) if ct_path.exists() else None
        roi_ver, rois_all = load_rt_image_roi(sdir / "rois.ROI")
        roi_map = {r.name: r for r in rois_all}

        if ct_full is not None:
            print(f"     PET={pet_full.shape} {pet_full.spacing} | CT={ct_full.shape} {ct_full.spacing} axis_aligned={ct_full.is_axis_aligned}")
        else:
            print(f"     PET={pet_full.shape} {pet_full.spacing} | CT=None (PET-only)")

        for sp in sess_specs:
            m_info = MOUSE_BY_ID[sp.mouse_id]
            body_roi = roi_map[sp.body_roi]
            rt_roi = roi_map[sp.rt_roi]
            ctrl_roi = roi_map[sp.ctrl_roi]
            muscle_roi = roi_map[sp.muscle_roi]
            sub_rois = [rt_roi, ctrl_roi, muscle_roi]
            quad = _quadrant(body_roi, pet_full)

            # Crop PET & CT to union of mouse body box + all tumour/muscle ROIs (+ 2 mm pad so no tumour clips)
            pet_crop = crop_volume_to_body_roi(pet_full, body_roi, extra_rois=sub_rois, pad_mm=2.0)
            ct_crop = (
                crop_volume_to_body_roi(
                    ct_full, body_roi, ref_pet_vol=pet_crop, extra_rois=sub_rois, pad_mm=2.0
                )
                if ct_full is not None
                else None
            )

            # Save cropped binaries + ROI subset
            mdir = cropped_root / sp.mouse_id / sp.timepoint_slug
            pet_out = save_rt_image_dicom(pet_crop, mdir / f"{sp.mouse_id}_{sp.timepoint_slug}_PET.dcm")
            ct_out = (
                save_rt_image_dicom(ct_crop, mdir / f"{sp.mouse_id}_{sp.timepoint_slug}_CT.dcm")
                if ct_crop is not None
                else None
            )
            roi_out = save_rt_image_roi_subset(roi_ver, sub_rois, mdir / "rois.ROI")

            # Compute PET & CT stats on cropped volumes
            pet_musc = _vol_stats(pet_crop, muscle_roi)
            ct_musc = _vol_stats(ct_crop, muscle_roi) if ct_crop is not None else None

            pet_rt = _vol_stats(pet_crop, rt_roi)
            pet_ctrl = _vol_stats(pet_crop, ctrl_roi)
            ct_rt = _vol_stats(ct_crop, rt_roi) if ct_crop is not None else None
            ct_ctrl = _vol_stats(ct_crop, ctrl_roi) if ct_crop is not None else None

            musc_mean = pet_musc["mean"]
            tm_rt = round(pet_rt["mean"] / musc_mean, 4) if musc_mean and musc_mean > 0 else np.nan
            tm_ctrl = round(pet_ctrl["mean"] / musc_mean, 4) if musc_mean and musc_mean > 0 else np.nan

            # Render QC overview PNG
            qc_out = mdir / f"{sp.mouse_id}_{sp.timepoint_slug}_qc.png"
            render_mouse_qc_overview(qc_out, sp, pet_crop, ct_crop, sub_rois, tm_rt, tm_ctrl)

            # Export simulator-ready compressed NumPy .npz voxel bundle
            npz_out = save_voxel_bundle_npz(
                voxel_root / sp.mouse_id / f"{sp.mouse_id}_{sp.timepoint_slug}.npz",
                spec=sp,
                pet_crop=pet_crop,
                ct_crop=ct_crop,
                rt_roi=rt_roi,
                ctrl_roi=ctrl_roi,
                muscle_roi=muscle_roi,
                muscle_mean_idg=float(pet_musc["mean"]),
                muscle_std_idg=float(pet_musc["std"]),
            )
            montage_cache.setdefault(sp.mouse_id, []).append(
                (sp, pet_crop, ct_crop, sub_rois, tm_rt, tm_ctrl)
            )

            for treatment, roi_obj, p_st, c_st, tm_val in [
                ("RT_20Gy", rt_roi, pet_rt, ct_rt, tm_rt),
                ("Control_0Gy", ctrl_roi, pet_ctrl, ct_ctrl, tm_ctrl),
            ]:
                cal_date, cal_day, cal_mm = nearest_caliper(sp.mouse_id, treatment, sp.days_post_rt)
                imaging_rows.append(
                    {
                        "mouse_id": sp.mouse_id,
                        "cage": m_info["cage"],
                        "ear_punch": m_info["ear_punch"],
                        "roi_id_2010": m_info["roi_id_2010"],
                        "tumour_id": f"{sp.mouse_id}_{'RT' if treatment == 'RT_20Gy' else 'Ctrl'}",
                        "treatment": treatment,
                        "dose_gy": 20 if treatment == "RT_20Gy" else 0,
                        "caliper_side_2010": "LHS" if treatment == "RT_20Gy" else "RHS",
                        "roi_label_2013": roi_obj.name,
                        "scan_date": sp.scan_date,
                        "days_post_rt": sp.days_post_rt,
                        "scan_set": sp.scan_set_label,
                        "bed_quadrant": quad,
                        "pet_points": p_st["points"],
                        "pet_volume_cm3": p_st["volume_cm3"],
                        "pet_min_idg": p_st["min"],
                        "pet_max_idg": p_st["max"],
                        "pet_mean_idg": p_st["mean"],
                        "pet_std_idg": p_st["std"],
                        "pet_p90_idg": p_st["p90"],
                        "pet_p95_idg": p_st["p95"],
                        "pet_sum_idg": p_st["sum"],
                        "muscle_roi_label": muscle_roi.name,
                        "muscle_volume_cm3": pet_musc["volume_cm3"],
                        "muscle_mean_idg": pet_musc["mean"],
                        "muscle_std_idg": pet_musc["std"],
                        "tm_mean": tm_val,
                        "tm_max": round(p_st["max"] / musc_mean, 4) if musc_mean and musc_mean > 0 else np.nan,
                        "tm_p95": round(p_st["p95"] / musc_mean, 4) if musc_mean and musc_mean > 0 else np.nan,
                        "ct_available": ct_crop is not None,
                        "ct_spacing_mm": ct_crop.spacing[0] if ct_crop is not None else np.nan,
                        "ct_points": c_st["points"] if c_st is not None else np.nan,
                        "ct_volume_cm3": c_st["volume_cm3"] if c_st is not None else np.nan,
                        "ct_mean_hu": c_st["mean"] if c_st is not None else np.nan,
                        "ct_std_hu": c_st["std"] if c_st is not None else np.nan,
                        "ct_muscle_mean_hu": ct_musc["mean"] if ct_musc is not None else np.nan,
                        "nearest_caliper_date": cal_date,
                        "nearest_caliper_days_post_rt": cal_day if cal_day is not None else np.nan,
                        "nearest_caliper_mm": cal_mm if cal_mm is not None else np.nan,
                        "cropped_pet_path": pet_out.relative_to(root).as_posix(),
                        "cropped_ct_path": ct_out.relative_to(root).as_posix() if ct_out is not None else "",
                        "cropped_roi_path": roi_out.relative_to(root).as_posix(),
                        "voxel_bundle_path": npz_out.relative_to(root).as_posix(),
                        "qc_image_path": qc_out.relative_to(root).as_posix(),
                    }
                )
            print(
                f"     -> {sp.mouse_id:8s} ({sp.timepoint_slug}, {quad:12s}): "
                f"RT={pet_rt['volume_cm3']:.3f}cc (T/M {tm_rt:.2f}), "
                f"Ctrl={pet_ctrl['volume_cm3']:.3f}cc (T/M {tm_ctrl:.2f}), "
                f"CT={'yes' if ct_crop is not None else 'no'}"
            )

    # 3. Render longitudinal montage per mouse
    print("\n3. Rendering per-mouse longitudinal montages...")
    for m in MOUSE_REGISTRY:
        mid = m["mouse_id"]
        entries = sorted(montage_cache[mid], key=lambda e: e[0].days_post_rt)
        render_longitudinal_montage(montage_root / f"{mid}_longitudinal.png", mid, entries)

    # 4. Assemble and enrich consolidated tables
    mouse_order = {m["mouse_id"]: idx for idx, m in enumerate(MOUSE_REGISTRY)}
    imaging_df = pd.DataFrame(imaging_rows)
    imaging_df["_mo"] = imaging_df["mouse_id"].map(mouse_order)
    imaging_df["_to"] = imaging_df["treatment"].map({"RT_20Gy": 0, "Control_0Gy": 1})
    imaging_df = imaging_df.sort_values(["_mo", "_to", "days_post_rt"]).drop(columns=["_mo", "_to"]).reset_index(drop=True)

    # Enrich caliper_df with pre-RT baseline EF5 T/M & PET volume
    pre_rt_lookup = (
        imaging_df[imaging_df["days_post_rt"] == -1]
        .set_index("tumour_id")[["tm_mean", "pet_mean_idg", "pet_volume_cm3"]]
        .rename(
            columns={
                "tm_mean": "pre_rt_tm_mean",
                "pet_mean_idg": "pre_rt_pet_mean_idg",
                "pet_volume_cm3": "pre_rt_pet_vol_cm3",
            }
        )
    )
    caliper_df = caliper_df.merge(pre_rt_lookup, left_on="tumour_id", right_index=True, how="left")

    # Build tumour_master_wide.csv (12 rows: 1 per tumour)
    wide_rows = []
    for m in MOUSE_REGISTRY:
        mid = m["mouse_id"]
        for treatment in ["RT_20Gy", "Control_0Gy"]:
            tid = f"{mid}_{'RT' if treatment == 'RT_20Gy' else 'Ctrl'}"
            row: Dict[str, object] = {
                "tumour_id": tid,
                "mouse_id": mid,
                "cage": m["cage"],
                "ear_punch": m["ear_punch"],
                "roi_id_2010": m["roi_id_2010"],
                "treatment": treatment,
                "dose_gy": 20 if treatment == "RT_20Gy" else 0,
                "caliper_side_2010": "LHS" if treatment == "RT_20Gy" else "RHS",
                "roi_side_2013": "RHS RT" if treatment == "RT_20Gy" else "LHS Control",
                "survival_status": m["survival_status"],
            }
            for day, tag in [(-1, "d_minus1"), (13, "d13"), (32, "d32")]:
                sub = imaging_df[(imaging_df["tumour_id"] == tid) & (imaging_df["days_post_rt"] == day)]
                if not sub.empty:
                    s = sub.iloc[0]
                    row[f"{tag}_pet_vol_cm3"] = s["pet_volume_cm3"]
                    row[f"{tag}_ct_vol_cm3"] = s["ct_volume_cm3"]
                    row[f"{tag}_pet_mean_idg"] = s["pet_mean_idg"]
                    row[f"{tag}_pet_max_idg"] = s["pet_max_idg"]
                    row[f"{tag}_muscle_mean_idg"] = s["muscle_mean_idg"]
                    row[f"{tag}_tm_mean"] = s["tm_mean"]
                    row[f"{tag}_tm_max"] = s["tm_max"]
                else:
                    for col in ["pet_vol_cm3", "ct_vol_cm3", "pet_mean_idg", "pet_max_idg", "muscle_mean_idg", "tm_mean", "tm_max"]:
                        row[f"{tag}_{col}"] = np.nan

            # Add caliper measurements across all 8 dates
            csub = caliper_df[caliper_df["tumour_id"] == tid].set_index("days_post_rt")
            for cday in [-22, -10, -1, 4, 11, 17, 27, 39]:
                lbl = f"caliper_d{cday:+d}_mm".replace("+", "p").replace("-", "m")
                row[lbl] = float(csub.loc[cday, "diameter_mm"]) if cday in csub.index else np.nan

            wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    mapping_df = pd.DataFrame(
        [{k: v for k, v in m.items() if k != "caliper_key"} for m in MOUSE_REGISTRY]
    )

    # Save tables to both outputs/ and git-tracked results/
    table_map = {
        "mouse_id_mapping.csv": mapping_df,
        "longitudinal_imaging_summary.csv": imaging_df,
        "caliper_growth_curves.csv": caliper_df,
        "tumour_master_wide.csv": wide_df,
        "study_timeline.csv": timeline_df,
    }
    for fname, df in table_map.items():
        df.to_csv(out_tables / fname, index=False)
        df.to_csv(git_results / fname, index=False)

    print("\n4. Summary of saved tables:")
    for fname, df in table_map.items():
        print(f"   - {fname:35s} ({len(df):3d} rows x {len(df.columns):2d} cols)")

    print("\n=== Tumour Master Wide Preview (EF5 T/M & Volumes) ===")
    preview_cols = [
        "tumour_id", "treatment",
        "d_minus1_pet_vol_cm3", "d_minus1_tm_mean",
        "d13_pet_vol_cm3", "d13_tm_mean",
        "d32_pet_vol_cm3", "d32_tm_mean",
        "caliper_dm1_mm", "caliper_dp11_mm", "caliper_dp27_mm", "caliper_dp39_mm",
    ]
    print(wide_df[preview_cols].to_string(index=False))
    print(f"\nDone! Cropped binaries in {cropped_root}; curated tables in {git_results}")
    verify_voxel_bundles(root)


def verify_voxel_bundles(root: Path = Path("/workspace")) -> None:
    """Verifies all 15 exported .npz voxel bundles against longitudinal_imaging_summary.csv."""
    df = pd.read_csv(root / "results/first_ef5_rt_study/longitudinal_imaging_summary.csv")
    bundles = sorted((root / "outputs/first_ef5_rt_study/voxel_bundles").rglob("*.npz"))
    assert len(bundles) == 15, f"Expected 15 bundles, found {len(bundles)}"
    print(f"\n=== Verifying {len(bundles)} .npz Voxel Bundles ===")
    for b in bundles:
        d = np.load(b)
        mid = str(d["mouse_id"])
        day = int(d["days_post_rt"])
        rt_mean = float(d["pet_idg"][d["mask_rt_pet"]].mean())
        rt_tm = float(d["pet_tm"][d["mask_rt_pet"]].mean())
        row = df[(df["mouse_id"] == mid) & (df["days_post_rt"] == day) & (df["treatment"] == "RT_20Gy")].iloc[0]
        assert abs(rt_mean - row["pet_mean_idg"]) < 1e-3, f"PET mean mismatch: {mid} {day}"
        assert abs(rt_tm - row["tm_mean"]) < 1e-3, f"T/M mismatch: {mid} {day}"
        ct_info = "PET-only"
        if bool(d["ct_available"]):
            assert d["ct_on_pet_grid"].shape == d["pet_idg"].shape
            ct_m = float(d["ct_on_pet_grid"][d["mask_rt_pet"]].mean())
            ct_info = f"CT={d['ct_hu'].shape}, ct_on_pet_rt_mean={ct_m:.1f} HU"
        print(f"  OK {b.name:38s} | PET={d['pet_idg'].shape} | RT={rt_mean:.4f} (T/M {rt_tm:.3f}) | {ct_info}")
    print("ALL 15 VOXEL BUNDLES VERIFIED!")


if __name__ == "__main__":
    main()
