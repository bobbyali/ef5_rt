"""
Consolidated processing pipeline for the First EF5 Radiation Therapy (RT) Study (Nov 2010 - Jan 2011).

Processes the 6 A549 subcutaneous tumour mice (c2_M0, c2_M1L, c2_M1R, c2_M1L1R, c2_M2R, c1_M2R)
across all 4 EF5 PET/CT imaging timepoints:
  - Day -15 (2010-11-18, early growth baseline, 20 days post-inoculation; c2_M0, c2_M1L, c2_M1R, c2_M1L1R)
  - Day -1  (2010-12-02, pre-RT, 20 Gy delivered on 2010-12-03; all 6 mice)
  - Day +13 (2010-12-15, post-RT; 5 surviving mice)
  - Day +32 (2011-01-03, post-RT; 4 surviving mice)

Outputs:
  1. Per-mouse, per-timepoint cropped PET DICOM, canonical axis-aligned CT DICOM (when registered CT covers tumours),
     per-mouse binary RT_Image `rois.ROI`, and visual QC overview PNGs under:
       outputs/first_ef5_rt_study/cropped_mice/<mouse_id>/<timepoint>/
  2. Simulator-ready compressed NumPy `.npz` voxel bundles under:
       outputs/first_ef5_rt_study/voxel_bundles/<mouse_id>/<mouse_id>_<timepoint>.npz
  3. Longitudinal montage PNG per mouse under:
       outputs/first_ef5_rt_study/montages/<mouse_id>_longitudinal.png
  4. Curated, publication-ready summary CSV tables (saved to both outputs/first_ef5_rt_study/tables/
     and git-tracked results/first_ef5_rt_study/):
       - mouse_id_mapping.csv
       - longitudinal_imaging_summary.csv
       - caliper_growth_curves.csv
       - tumour_master_wide.csv
       - study_timeline.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
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
        "day_minus15_set": "2010-11-18 Early Growth BR (1) (PET+CT)",
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "None (died before 2011-01-03)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Died between 2010-12-30 and 2011-01-03",
        "notes": "Imaged Day -15 (18 Nov, both tumours), Day -1, and Day +13; last caliper 2010-12-30.",
    },
    {
        "mouse_id": "c2_M1L",
        "cage": "A549 c2",
        "ear_punch": "1L",
        "roi_id_2010": "m2",
        "caliper_key": ("A549 c2", "1L"),
        "day_minus15_set": "2010-11-18 Early Growth TL (2) (PET only, Ctrl contoured)",
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 2 (PET+CT)",
        "day32_set": "2011-01-03 Set 8 (PET only)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "4-timepoint PET series (Day -15 Ctrl only; TL tumours outside 18 Nov CT FOV; Day +32 Set 8 PET-only).",
    },
    {
        "mouse_id": "c2_M1R",
        "cage": "A549 c2",
        "ear_punch": "1R",
        "roi_id_2010": "m3",
        "caliper_key": ("A549 c2", "1R"),
        "day_minus15_set": "2010-11-18 Early Growth TR (3) (PET only, RT contoured)",
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 2 (PET+CT)",
        "day32_set": "2011-01-03 Set 9 (PET+CT)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "4-timepoint PET series (Day -15 RT only; TR tumours outside 18 Nov CT FOV; Dec-Jan all PET+CT). Growth Curves Post-RT 1R swap fixed.",
    },
    {
        "mouse_id": "c2_M1L1R",
        "cage": "A549 c2",
        "ear_punch": "1L1R",
        "roi_id_2010": "m4",
        "caliper_key": ("A549 c2", "1L1R"),
        "day_minus15_set": "2010-11-18 Early Growth BL (4) (PET+CT, partial RT CT)",
        "day_minus1_set": "2010-12-02 Set 1 (PET+CT)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "2011-01-03 Set 8 (PET only)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "4-timepoint PET series (both tumours contoured at all 4 timepoints; Day -15 RT partially at CT edge; Day +32 PET-only). Pre-RT m4 T/M formula bug fixed.",
    },
    {
        "mouse_id": "c2_M2R",
        "cage": "A549 c2",
        "ear_punch": "2R",
        "roi_id_2010": "m5",
        "caliper_key": ("A549 c2", "2R"),
        "day_minus15_set": "None (not imaged on 2010-11-18)",
        "day_minus1_set": "2010-12-02 Set 2 (PET only)",
        "day13_set": "2010-12-15 Set 3 (PET+CT)",
        "day32_set": "2011-01-03 Set 9 (PET+CT)",
        "irradiated_side_2010_caliper": "LHS (image-left, larger X)",
        "irradiated_side_2013_roi": "RHS RT (anatomical right)",
        "control_side_2010_caliper": "RHS (image-right, smaller X)",
        "control_side_2013_roi": "LHS Control (anatomical left)",
        "survival_status": "Survived to study endpoint (sacrificed 2011-01-12)",
        "notes": "Complete 3-timepoint Dec-Jan PET series; not included in 18 Nov 4-mouse bed; Day -1 Set 2 PET-only.",
    },
    {
        "mouse_id": "c1_M2R",
        "cage": "A549 c1",
        "ear_punch": "2R",
        "roi_id_2010": "m6",
        "caliper_key": ("A549 c1", "2R"),
        "day_minus15_set": "None (not imaged on 2010-11-18)",
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
    rt_roi: Optional[str]
    ctrl_roi: Optional[str]
    muscle_roi: str
    pet_filename: str = "image1.dcm"
    ct_filename: Optional[str] = "image2.dcm"
    roi_filename: str = "rois.ROI"
    use_ct: bool = True
    ct_note: str = ""


SCAN_SPECS: List[MouseScanSpec] = [
    # --- Day -15 (2010-11-18, Early Growth Baseline, 20 days post-inoculation) ---
    # Uses calibrated p105421_v2.img.dcm (%ID/g matching stats1.txt; a549_reg_norm0001.dcm was mis-scaled by 1/1290.5).
    # 2010 ROI naming: LHS (larger X) = future RT_20Gy (2013 RHS RT); RHS (smaller X) = future Control_0Gy (2013 LHS Control).
    # CT covers bottom row (BR, BL) only; top row (TL, TR) shoulders at z=[-39.4,-29.2] extend outside CT z_max=-36.1 mm.
    MouseScanSpec(
        "c2_M0", "day-15_2010-11-18", "2010-11-18", -15,
        "2010-11-18_day-15_early_growth", "18 Nov Early Growth",
        "Mouse BR ROI (1)", "BR Mouse LHS Tumor", "BR Mouse RHS Tumor", "BR Mouse BG ROI",
        pet_filename="p105421_v2.img.dcm", ct_filename="a549_reg_norm0002.dcm",
        roi_filename="a549_tumors.roi.ROI", use_ct=True,
    ),
    MouseScanSpec(
        "c2_M1L", "day-15_2010-11-18", "2010-11-18", -15,
        "2010-11-18_day-15_early_growth", "18 Nov Early Growth",
        "Mouse TL ROI (2)", None, "TL Mouse RHS Tumor", "TL Mouse BG ROI",
        pet_filename="p105421_v2.img.dcm", ct_filename="a549_reg_norm0002.dcm",
        roi_filename="a549_tumors.roi.ROI", use_ct=False,
        ct_note="CT excluded: TL tumour (z=[-39.4,-30.7]mm) lies outside CT FOV (z_max=-36.1mm); RT tumour not contoured on 18 Nov.",
    ),
    MouseScanSpec(
        "c2_M1R", "day-15_2010-11-18", "2010-11-18", -15,
        "2010-11-18_day-15_early_growth", "18 Nov Early Growth",
        "Mouse TR ROI (3)", "TR Mouse LHS Tumor", None, "TR Mouse BG ROI",
        pet_filename="p105421_v2.img.dcm", ct_filename="a549_reg_norm0002.dcm",
        roi_filename="a549_tumors.roi.ROI", use_ct=False,
        ct_note="CT excluded: TR tumour (z=[-37.2,-29.2]mm) lies outside CT FOV (z_max=-36.1mm); Control tumour not contoured on 18 Nov.",
    ),
    MouseScanSpec(
        "c2_M1L1R", "day-15_2010-11-18", "2010-11-18", -15,
        "2010-11-18_day-15_early_growth", "18 Nov Early Growth",
        "Mouse BL ROI (4)", "BL Mouse LHS Tumor", "BL Mouse RHS Tumor", "BL Mouse BG ROI",
        pet_filename="p105421_v2.img.dcm", ct_filename="a549_reg_norm0002.dcm",
        roi_filename="a549_tumors.roi.ROI", use_ct=True,
        ct_note="BL LHS (RT) tumour (x=[-32.7,-25.8]mm) sits at CT x_max=-26.3mm edge (~31% CT volume coverage); BL RHS (Ctrl) has 100% CT coverage.",
    ),
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


def _configure_mouse_roi_styles(
    rt_name: Optional[str], ctrl_name: Optional[str], muscle_name: str
) -> None:
    if rt_name is not None:
        ROI_STYLES[rt_name] = {
            "color": "#FF4081",
            "linestyle": "-",
            "linewidth": 2.2,
            "label": f"{rt_name} (Irradiated 20 Gy)",
        }
    if ctrl_name is not None:
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


def _pet_display_clim(spec: MouseScanSpec) -> Tuple[float, float]:
    """Selects PET display window (%ID/g): lower range on Day -15 early-growth scan."""
    return (0.0, 1.2) if spec.days_post_rt <= -10 else (0.0, 2.8)


def _tumour_slice_coords(rt_roi: Optional[RTRoi], ctrl_roi: Optional[RTRoi], muscle_roi: RTRoi) -> Tuple[float, float]:
    tumours = [r for r in (rt_roi, ctrl_roi) if r is not None]
    if not tumours:
        tumours = [muscle_roi]
    y_mid = float(np.mean([r.centroid_mm[1] for r in tumours]))
    z_mid = float(np.mean([r.centroid_mm[2] for r in tumours]))
    return y_mid, z_mid


def render_mouse_qc_overview(
    out_path: Path,
    spec: MouseScanSpec,
    pet_vol: RTImageVolume,
    ct_vol: Optional[RTImageVolume],
    rt_roi: Optional[RTRoi],
    ctrl_roi: Optional[RTRoi],
    muscle_roi: RTRoi,
    tm_rt: float,
    tm_ctrl: float,
) -> None:
    """
    Renders a clean 2-row QC figure for a cropped mouse timepoint:
      Row 1: Mid-tumour Transaxial slice (CT | PET | Overlay, or PET alone if PET-only)
      Row 2: Mid-tumour Coronal slice (CT | PET | Overlay, or PET alone if PET-only)
    """
    sub_rois = [r for r in (rt_roi, ctrl_roi, muscle_roi) if r is not None]
    _configure_mouse_roi_styles(
        rt_roi.name if rt_roi is not None else None,
        ctrl_roi.name if ctrl_roi is not None else None,
        muscle_roi.name,
    )

    ref_vol = ct_vol if ct_vol is not None else pet_vol
    roi_masks = {r.name: build_roi_3d_mask(ref_vol, r) for r in sub_rois}

    y_mid, z_mid = _tumour_slice_coords(rt_roi, ctrl_roi, muscle_roi)

    # Coronal Y zoom around tumours + muscle
    all_pts = np.vstack([r.all_vertices for r in sub_rois])
    y_lo = max(float(ref_vol.y_coords.min()), float(all_pts[:, 1].min()) - 8.0)
    y_hi = min(float(ref_vol.y_coords.max()), float(all_pts[:, 1].max()) + 8.0)

    pet_clim = _pet_display_clim(spec)
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
                f"{plane.capitalize()} EF5 PET (slice {pet_i}, 0-{pet_clim[1]:g} %ID/g)",
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
            ax.set_title(f"{plane.capitalize()} EF5 PET (slice {pet_i}, PET-only, 0-{pet_clim[1]:g} %ID/g)", color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555555")

    m_info = MOUSE_BY_ID[spec.mouse_id]
    tm_rt_str = f"{tm_rt:.2f}" if not np.isnan(tm_rt) else "N/A"
    tm_ctrl_str = f"{tm_ctrl:.2f}" if not np.isnan(tm_ctrl) else "N/A"
    fig.suptitle(
        f"{spec.mouse_id} ({m_info['roi_id_2010']}, {m_info['cage']} ear '{m_info['ear_punch']}') — "
        f"Day {spec.days_post_rt:+d} ({spec.scan_date}, {spec.scan_set_label})\n"
        f"T/M Mean: RT (20 Gy) = {tm_rt_str}  |  Control (0 Gy) = {tm_ctrl_str}",
        color="white",
        fontsize=12.5,
        fontweight="bold",
    )
    handles = []
    if rt_roi is not None:
        handles.append(Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label=f"RT 20 Gy ({rt_roi.name})"))
    if ctrl_roi is not None:
        handles.append(Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label=f"Control 0 Gy ({ctrl_roi.name})"))
    handles.append(Line2D([0], [0], color="#FFD740", linestyle=":", linewidth=2.2, label=f"Muscle ({muscle_roi.name})"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(handles),
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
    entries: List[Tuple[MouseScanSpec, RTImageVolume, Optional[RTImageVolume], Optional[RTRoi], Optional[RTRoi], RTRoi, float, float]],
) -> None:
    """Renders a 1xN longitudinal coronal EF5 PET(/CT) montage across available timepoints for one mouse."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.7), dpi=120, facecolor="#111111")
    if n == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.16, top=0.79, wspace=0.18)

    for ax, (spec, pet_vol, ct_vol, rt_roi, ctrl_roi, muscle_roi, tm_rt, tm_ctrl) in zip(axes, entries):
        sub_rois = [r for r in (rt_roi, ctrl_roi, muscle_roi) if r is not None]
        _configure_mouse_roi_styles(
            rt_roi.name if rt_roi is not None else None,
            ctrl_roi.name if ctrl_roi is not None else None,
            muscle_roi.name,
        )
        ref_vol = ct_vol if ct_vol is not None else pet_vol
        roi_masks = {r.name: build_roi_3d_mask(ref_vol, r) for r in sub_rois}

        _, z_mid = _tumour_slice_coords(rt_roi, ctrl_roi, muscle_roi)
        all_pts = np.vstack([r.all_vertices for r in sub_rois])
        y_lo = max(float(ref_vol.y_coords.min()), float(all_pts[:, 1].min()) - 8.0)
        y_hi = min(float(ref_vol.y_coords.max()), float(all_pts[:, 1].max()) + 8.0)
        pet_clim = _pet_display_clim(spec)

        ax.set_facecolor("black")
        pet_2d, pet_ext, _, _ = get_slice_2d_and_extent(pet_vol, "coronal", z_mid)
        if ct_vol is not None:
            ct_clim = _ct_display_clim(ct_vol)
            ct_2d, ct_ext, _, _ = get_slice_2d_and_extent(ct_vol, "coronal", z_mid)
            ax.imshow(ct_2d, cmap="gray", vmin=ct_clim[0], vmax=ct_clim[1], extent=ct_ext, origin="upper", interpolation="bilinear", aspect="equal")
            ax.imshow(pet_2d, cmap=PET_OVERLAY_CMAP, vmin=pet_clim[0], vmax=pet_clim[1], extent=pet_ext, origin="upper", interpolation="bilinear", aspect="equal")
            ax.set_xlim(ct_ext[0], ct_ext[1])
            mod_tag = "PET/CT"
        else:
            ax.imshow(pet_2d, cmap="inferno", vmin=pet_clim[0], vmax=pet_clim[1], extent=pet_ext, origin="upper", interpolation="nearest", aspect="equal")
            ax.set_xlim(pet_ext[0], pet_ext[1])
            mod_tag = "PET only"

        plot_rois_on_ax(ax, "coronal", z_mid, sub_rois, ref_vol, roi_masks)
        ax.set_ylim(y_lo, y_hi)
        tm_rt_str = f"{tm_rt:.2f}" if not np.isnan(tm_rt) else "N/A"
        tm_ctrl_str = f"{tm_ctrl:.2f}" if not np.isnan(tm_ctrl) else "N/A"
        ax.set_title(
            f"Day {spec.days_post_rt:+d} ({spec.scan_date})\n"
            f"{mod_tag} [0-{pet_clim[1]:g} %ID/g]\n"
            f"T/M: RT={tm_rt_str}, Ctrl={tm_ctrl_str}",
            color="white",
            fontsize=9.5,
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
    rt_roi: Optional[RTRoi],
    ctrl_roi: Optional[RTRoi],
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
    scan_dt = datetime.date.fromisoformat(spec.scan_date)
    days_post_inj = (scan_dt - INJECTION_DATE).days

    pet_idg = np.asarray(pet_crop.data, dtype=np.float32)
    pet_tm = (
        (pet_idg / np.float32(muscle_mean_idg)).astype(np.float32)
        if muscle_mean_idg and muscle_mean_idg > 0
        else np.full_like(pet_idg, np.nan, dtype=np.float32)
    )
    mask_rt_pet = build_roi_3d_mask(pet_crop, rt_roi) if rt_roi is not None else np.zeros(pet_crop.shape, dtype=bool)
    mask_ctrl_pet = build_roi_3d_mask(pet_crop, ctrl_roi) if ctrl_roi is not None else np.zeros(pet_crop.shape, dtype=bool)
    mask_muscle_pet = build_roi_3d_mask(pet_crop, muscle_roi)

    payload: Dict[str, object] = {
        "mouse_id": np.array(spec.mouse_id),
        "roi_id_2010": np.array(m_info["roi_id_2010"]),
        "cage": np.array(m_info["cage"]),
        "ear_punch": np.array(m_info["ear_punch"]),
        "scan_date": np.array(spec.scan_date),
        "days_post_injection": np.int32(days_post_inj),
        "days_post_rt": np.int32(spec.days_post_rt),
        "scan_set": np.array(spec.scan_set_label),
        "has_rt_roi": np.bool_(rt_roi is not None),
        "has_ctrl_roi": np.bool_(ctrl_roi is not None),
        "rt_roi_label": np.array(rt_roi.name if rt_roi is not None else ""),
        "ctrl_roi_label": np.array(ctrl_roi.name if ctrl_roi is not None else ""),
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
        "ct_note": np.array(spec.ct_note),
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
                "mask_rt_ct": build_roi_3d_mask(ct_crop, rt_roi) if rt_roi is not None else np.zeros(ct_crop.shape, dtype=bool),
                "mask_ctrl_ct": build_roi_3d_mask(ct_crop, ctrl_roi) if ctrl_roi is not None else np.zeros(ct_crop.shape, dtype=bool),
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

    # 2. Load each consolidated full-bed session once (avoid reloading large CTs multiple times)
    print(f"\n2. Processing all {len(SCAN_SPECS)} mouse-timepoint scans (cropping PET/CT, ROIs, stats, QC images, .npz bundles)...")
    specs_by_session: Dict[Tuple[str, str, Optional[str], str], List[MouseScanSpec]] = {}
    for sp in SCAN_SPECS:
        key = (sp.session_rel_dir, sp.pet_filename, sp.ct_filename, sp.roi_filename)
        specs_by_session.setdefault(key, []).append(sp)

    imaging_rows: List[Dict[str, object]] = []
    montage_cache: Dict[
        str,
        List[Tuple[MouseScanSpec, RTImageVolume, Optional[RTImageVolume], Optional[RTRoi], Optional[RTRoi], RTRoi, float, float]],
    ] = {}

    for (sess_rel, pet_fn, ct_fn, roi_fn), sess_specs in specs_by_session.items():
        sdir = cons_dir / sess_rel
        print(f"\n   Loading session: {sess_rel} [{pet_fn}] ({len(sess_specs)} study mice)")
        pet_full = load_rt_image_dicom(sdir / pet_fn)
        ct_path = (sdir / ct_fn) if ct_fn is not None else None
        ct_full = load_rt_image_dicom(ct_path) if (ct_path is not None and ct_path.exists()) else None
        roi_ver, rois_all = load_rt_image_roi(sdir / roi_fn)
        roi_map = {r.name: r for r in rois_all}

        if ct_full is not None:
            print(f"     PET={pet_full.shape} {pet_full.spacing} | CT={ct_full.shape} {ct_full.spacing} axis_aligned={ct_full.is_axis_aligned}")
        else:
            print(f"     PET={pet_full.shape} {pet_full.spacing} | CT=None (PET-only)")

        for sp in sess_specs:
            m_info = MOUSE_BY_ID[sp.mouse_id]
            scan_dt = datetime.date.fromisoformat(sp.scan_date)
            days_post_inj = (scan_dt - INJECTION_DATE).days

            body_roi = roi_map[sp.body_roi]
            rt_roi = roi_map[sp.rt_roi] if sp.rt_roi is not None else None
            ctrl_roi = roi_map[sp.ctrl_roi] if sp.ctrl_roi is not None else None
            muscle_roi = roi_map[sp.muscle_roi]
            sub_rois = [r for r in (rt_roi, ctrl_roi, muscle_roi) if r is not None]
            quad = _quadrant(body_roi, pet_full)

            # Crop PET & CT to union of mouse body box + all tumour/muscle ROIs (+ 2 mm pad so no tumour clips)
            pet_crop = crop_volume_to_body_roi(pet_full, body_roi, extra_rois=sub_rois, pad_mm=2.0)
            ct_crop = (
                crop_volume_to_body_roi(
                    ct_full, body_roi, ref_pet_vol=pet_crop, extra_rois=sub_rois, pad_mm=2.0
                )
                if (ct_full is not None and sp.use_ct)
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

            pet_rt = _vol_stats(pet_crop, rt_roi) if rt_roi is not None else None
            pet_ctrl = _vol_stats(pet_crop, ctrl_roi) if ctrl_roi is not None else None
            ct_rt = _vol_stats(ct_crop, rt_roi) if (ct_crop is not None and rt_roi is not None) else None
            ct_ctrl = _vol_stats(ct_crop, ctrl_roi) if (ct_crop is not None and ctrl_roi is not None) else None

            musc_mean = pet_musc["mean"]
            tm_rt = round(pet_rt["mean"] / musc_mean, 4) if (pet_rt is not None and musc_mean and musc_mean > 0) else np.nan
            tm_ctrl = round(pet_ctrl["mean"] / musc_mean, 4) if (pet_ctrl is not None and musc_mean and musc_mean > 0) else np.nan

            # Render QC overview PNG
            qc_out = mdir / f"{sp.mouse_id}_{sp.timepoint_slug}_qc.png"
            render_mouse_qc_overview(qc_out, sp, pet_crop, ct_crop, rt_roi, ctrl_roi, muscle_roi, tm_rt, tm_ctrl)

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
                (sp, pet_crop, ct_crop, rt_roi, ctrl_roi, muscle_roi, tm_rt, tm_ctrl)
            )

            for treatment, roi_obj, p_st, c_st, tm_val in [
                ("RT_20Gy", rt_roi, pet_rt, ct_rt, tm_rt),
                ("Control_0Gy", ctrl_roi, pet_ctrl, ct_ctrl, tm_ctrl),
            ]:
                if roi_obj is None or p_st is None:
                    continue
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
                        "days_post_injection": days_post_inj,
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
                        "ct_note": sp.ct_note,
                        "nearest_caliper_date": cal_date,
                        "nearest_caliper_days_post_rt": cal_day if cal_day is not None else np.nan,
                        "nearest_caliper_mm": cal_mm if cal_mm is not None else np.nan,
                        "cropped_pet_path": pet_out.relative_to(root).as_posix(),
                        "cropped_ct_path": ct_out.relative_to(root).as_posix() if ct_out is not None else "",
                        "cropped_roi_path": roi_out.relative_to(root).as_posix(),
                        "voxel_bundle_path": npz_out.relative_to(root).as_posix(),
                        "nifti_dir": f"outputs/first_ef5_rt_study/nifti/{sp.mouse_id}/{sp.timepoint_slug}",
                        "qc_image_path": qc_out.relative_to(root).as_posix(),
                    }
                )
            rt_msg = f"{pet_rt['volume_cm3']:.3f}cc (T/M {tm_rt:.2f})" if pet_rt is not None else "not contoured"
            ctrl_msg = f"{pet_ctrl['volume_cm3']:.3f}cc (T/M {tm_ctrl:.2f})" if pet_ctrl is not None else "not contoured"
            print(
                f"     -> {sp.mouse_id:8s} ({sp.timepoint_slug:16s}, {quad:12s}): "
                f"RT={rt_msg}, Ctrl={ctrl_msg}, CT={'yes' if ct_crop is not None else 'no'}"
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
            for day, tag in [(-15, "d_minus15"), (-1, "d_minus1"), (13, "d13"), (32, "d32")]:
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

    print("\n=== Tumour Master Wide Preview (EF5 T/M & Volumes, Day -15 to Day +32) ===")
    preview_cols = [
        "tumour_id", "treatment",
        "d_minus15_pet_vol_cm3", "d_minus15_tm_mean",
        "d_minus1_pet_vol_cm3", "d_minus1_tm_mean",
        "d13_pet_vol_cm3", "d13_tm_mean",
        "d32_pet_vol_cm3", "d32_tm_mean",
    ]
    print(wide_df[preview_cols].to_string(index=False))
    print(f"\nDone! Cropped binaries in {cropped_root}; curated tables in {git_results}")
    verify_voxel_bundles(root)
    export_nifti_dataset(root)
    verify_nifti_outputs(root)


def _canonical_array_to_nifti_ras(
    arr_yzx: np.ndarray,
    spacing_dz_dy_dx: np.ndarray,
    origin_lps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a canonical RT_Image array of shape (nz, ny, nx) = (Y, Z, X) in DICOM LPS
    into a standard positive-diagonal NIfTI RAS+ volume of shape (nx, nz, ny) = (X, Y, Z)
    with a 4x4 RAS+ affine matrix (nibabel.aff2axcodes == ('R', 'A', 'S')).
    """
    nz, ny, nx = arr_yzx.shape
    dz, dy, dx = [float(v) for v in spacing_dz_dy_dx]
    ox, oy, oz = [float(v) for v in origin_lps]

    # (Y, Z, X) -> (X, Y, Z), then flip Z so increasing index maps to increasing +Z_ras
    arr_ras = np.ascontiguousarray(np.transpose(arr_yzx, (2, 0, 1))[:, :, ::-1])

    affine_ras = np.array(
        [
            [dx, 0.0, 0.0, -ox],
            [0.0, dz, 0.0, -oy],
            [0.0, 0.0, dy, oz - (ny - 1) * dy],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return arr_ras, affine_ras


def _save_nifti(arr_ras: np.ndarray, affine_ras: np.ndarray, out_path: Path, descrip: str = "") -> Path:
    img = nib.Nifti1Image(arr_ras, affine_ras)
    img.set_qform(affine_ras, code=1)
    img.set_sform(affine_ras, code=1)
    if descrip:
        img.header["descrip"] = descrip.encode("ascii", "ignore")[:79]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))
    return out_path


def _combine_label_map(mask_rt: np.ndarray, mask_ctrl: np.ndarray, mask_muscle: np.ndarray) -> np.ndarray:
    """Combines binary masks into a single uint8 label map: 1=RT_20Gy, 2=Control_0Gy, 3=Muscle_BG."""
    seg = np.zeros(mask_muscle.shape, dtype=np.uint8)
    seg[mask_muscle] = 3
    seg[mask_ctrl] = 2
    seg[mask_rt] = 1
    return seg


def export_nifti_dataset(root: Path = Path("/workspace")) -> None:
    """
    Exports all 19 verified .npz voxel bundles to compressed NIfTI (.nii.gz) volumes
    under outputs/first_ef5_rt_study/nifti/<mouse_id>/<timepoint>/ with standard RAS+ affines
    and BIDS-style JSON sidecar metadata.
    """
    voxel_root = root / "outputs" / "first_ef5_rt_study" / "voxel_bundles"
    nifti_root = root / "outputs" / "first_ef5_rt_study" / "nifti"
    nifti_root.mkdir(parents=True, exist_ok=True)

    bundles = sorted(voxel_root.rglob("*.npz"))
    assert len(bundles) == len(SCAN_SPECS), f"Expected {len(SCAN_SPECS)} .npz bundles, found {len(bundles)}"

    legend = {
        "dataset": "First EF5 RT Study (A549 Subcutaneous Tumours, Nov 2010 - Jan 2011)",
        "coordinate_system": "NIfTI RAS+ (converted from DICOM LPS via diag(-1, -1, 1, 1)); nibabel axcodes = ('R', 'A', 'S')",
        "segmentation_labels": {
            "0": "Background",
            "1": "RT_20Gy (irradiated tumour; 2010 LHS / 2013 RHS RT)",
            "2": "Control_0Gy (contralateral unirradiated tumour; 2010 RHS / 2013 LHS Control)",
            "3": "Muscle_BG (background muscle reference ROI)",
        },
        "files_per_scan": {
            "*_PET_idg.nii.gz": "3D [18F]-EF5 PET uptake in %ID/g (float32)",
            "*_PET_tm.nii.gz": "3D muscle-normalized [18F]-EF5 T/M ratio map (float32)",
            "*_seg_pet.nii.gz": "Multi-label ROI mask on PET grid (uint8)",
            "*_CT.nii.gz": "Native-resolution canonical microCT (float32, when CT available)",
            "*_CT_on_pet.nii.gz": "Canonical microCT resampled onto PET grid (float32, when CT available)",
            "*_seg_ct.nii.gz": "Multi-label ROI mask on native CT grid (uint8, when CT available)",
            "*_metadata.json": "Per-scan sidecar metadata (mouse ID, timing, units, ROI stats/notes)",
        },
    }
    (nifti_root / "label_legend.json").write_text(json.dumps(legend, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Exporting {len(bundles)} Scans to NIfTI (.nii.gz) in {nifti_root} ===")
    for b in bundles:
        d = np.load(b)
        mid = str(d["mouse_id"])
        stem = b.stem  # <mouse_id>_<timepoint_slug>
        tp_slug = stem[len(mid) + 1 :]
        out_dir = nifti_root / mid / tp_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        pet_spacing = d["pet_spacing_mm"]
        pet_origin = d["pet_origin_mm"]

        pet_idg_ras, pet_aff = _canonical_array_to_nifti_ras(d["pet_idg"].astype(np.float32), pet_spacing, pet_origin)
        pet_tm_ras, _ = _canonical_array_to_nifti_ras(d["pet_tm"].astype(np.float32), pet_spacing, pet_origin)
        seg_pet_yzx = _combine_label_map(d["mask_rt_pet"], d["mask_ctrl_pet"], d["mask_muscle_pet"])
        seg_pet_ras, _ = _canonical_array_to_nifti_ras(seg_pet_yzx, pet_spacing, pet_origin)

        files_written: Dict[str, str] = {}
        files_written["pet_idg"] = _save_nifti(
            pet_idg_ras, pet_aff, out_dir / f"{stem}_PET_idg.nii.gz", "EF5 PET (%ID/g)"
        ).name
        files_written["pet_tm"] = _save_nifti(
            pet_tm_ras, pet_aff, out_dir / f"{stem}_PET_tm.nii.gz", "EF5 PET (T/M ratio)"
        ).name
        files_written["seg_pet"] = _save_nifti(
            seg_pet_ras, pet_aff, out_dir / f"{stem}_seg_pet.nii.gz", "ROIs: 1=RT_20Gy, 2=Ctrl_0Gy, 3=Muscle"
        ).name

        ct_avail = bool(d["ct_available"])
        if ct_avail:
            ct_spacing = d["ct_spacing_mm"]
            ct_origin = d["ct_origin_mm"]
            ct_ras, ct_aff = _canonical_array_to_nifti_ras(d["ct_hu"].astype(np.float32), ct_spacing, ct_origin)
            seg_ct_yzx = _combine_label_map(d["mask_rt_ct"], d["mask_ctrl_ct"], d["mask_muscle_ct"])
            seg_ct_ras, _ = _canonical_array_to_nifti_ras(seg_ct_yzx, ct_spacing, ct_origin)
            ct_on_pet_ras, _ = _canonical_array_to_nifti_ras(
                d["ct_on_pet_grid"].astype(np.float32), pet_spacing, pet_origin
            )
            files_written["ct"] = _save_nifti(
                ct_ras, ct_aff, out_dir / f"{stem}_CT.nii.gz", "MicroCT (Gamma Medica HU)"
            ).name
            files_written["ct_on_pet"] = _save_nifti(
                ct_on_pet_ras, pet_aff, out_dir / f"{stem}_CT_on_pet.nii.gz", "MicroCT on PET grid"
            ).name
            files_written["seg_ct"] = _save_nifti(
                seg_ct_ras, ct_aff, out_dir / f"{stem}_seg_ct.nii.gz", "ROIs on CT: 1=RT, 2=Ctrl, 3=Muscle"
            ).name

        meta = {
            "mouse_id": mid,
            "roi_id_2010": str(d["roi_id_2010"]),
            "cage": str(d["cage"]),
            "ear_punch": str(d["ear_punch"]),
            "scan_date": str(d["scan_date"]),
            "days_post_injection": int(d["days_post_injection"]),
            "days_post_rt": int(d["days_post_rt"]),
            "scan_set": str(d["scan_set"]),
            "has_rt_roi": bool(d["has_rt_roi"]),
            "has_ctrl_roi": bool(d["has_ctrl_roi"]),
            "rt_roi_label": str(d["rt_roi_label"]),
            "ctrl_roi_label": str(d["ctrl_roi_label"]),
            "muscle_roi_label": str(d["muscle_roi_label"]),
            "muscle_mean_idg": round(float(d["muscle_mean_idg"]), 6),
            "muscle_std_idg": round(float(d["muscle_std_idg"]), 6),
            "ct_available": ct_avail,
            "ct_note": str(d["ct_note"]),
            "pet_units": "%ID/g",
            "pet_tm_units": "Tumor-to-Muscle ratio (dimensionless)",
            "orientation": "RAS+",
            "segmentation_labels": {"1": "RT_20Gy", "2": "Control_0Gy", "3": "Muscle_BG"},
            "files": files_written,
        }
        (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  Exported {out_dir.relative_to(root).as_posix()} ({len(files_written)} NIfTI files)")


def verify_nifti_outputs(root: Path = Path("/workspace")) -> None:
    """
    Reloads all 19 NIfTI scan sets via nibabel and verifies:
      1. Canonical RAS+ orientation ('R', 'A', 'S') on every volume
      2. Exact %ID/g and T/M intensity means inside label 1 (RT) & label 2 (Ctrl) vs CSV
      3. Co-registration of PET and native CT in physical RAS+ mm (tumour centroid agreement)
      4. Exact shape & affine identity between CT_on_pet and PET_idg
    """
    df = pd.read_csv(root / "results/first_ef5_rt_study/longitudinal_imaging_summary.csv")
    nifti_root = root / "outputs" / "first_ef5_rt_study" / "nifti"
    meta_files = sorted(nifti_root.rglob("*_metadata.json"))
    assert len(meta_files) == len(SCAN_SPECS), f"Expected {len(SCAN_SPECS)} NIfTI scan folders, found {len(meta_files)}"

    print(f"\n=== Verifying {len(meta_files)} NIfTI Scan Sets ===")
    for mf in meta_files:
        meta = json.loads(mf.read_text(encoding="utf-8"))
        out_dir = mf.parent
        mid = meta["mouse_id"]
        day = int(meta["days_post_rt"])

        pet_img = nib.load(str(out_dir / meta["files"]["pet_idg"]))
        tm_img = nib.load(str(out_dir / meta["files"]["pet_tm"]))
        seg_img = nib.load(str(out_dir / meta["files"]["seg_pet"]))

        assert nib.aff2axcodes(pet_img.affine) == ("R", "A", "S"), f"Non-RAS PET affine: {mf}"
        assert np.allclose(pet_img.affine, tm_img.affine) and np.allclose(pet_img.affine, seg_img.affine)

        pet_arr = np.asarray(pet_img.dataobj, dtype=np.float32)
        tm_arr = np.asarray(tm_img.dataobj, dtype=np.float32)
        seg_arr = np.asarray(seg_img.dataobj, dtype=np.uint8)

        parts = []
        for has_flag, lbl_id, trt, tag in [
            (meta["has_rt_roi"], 1, "RT_20Gy", "RT"),
            (meta["has_ctrl_roi"], 2, "Control_0Gy", "Ctrl"),
        ]:
            m = seg_arr == lbl_id
            if has_flag:
                mean_idg = float(pet_arr[m].mean())
                mean_tm = float(tm_arr[m].mean())
                row = df[(df["mouse_id"] == mid) & (df["days_post_rt"] == day) & (df["treatment"] == trt)].iloc[0]
                assert abs(mean_idg - row["pet_mean_idg"]) < 1e-3, f"NIfTI PET mismatch: {mid} {day} {trt}"
                assert abs(mean_tm - row["tm_mean"]) < 1e-3, f"NIfTI T/M mismatch: {mid} {day} {trt}"
                parts.append(f"{tag}={mean_idg:.4f}(T/M {mean_tm:.2f})")
            else:
                assert not m.any(), f"Unexpected label {lbl_id} voxels in {mf}"

        ct_msg = "PET-only"
        if meta["ct_available"]:
            ct_img = nib.load(str(out_dir / meta["files"]["ct"]))
            ct_pet_img = nib.load(str(out_dir / meta["files"]["ct_on_pet"]))
            seg_ct_img = nib.load(str(out_dir / meta["files"]["seg_ct"]))

            assert nib.aff2axcodes(ct_img.affine) == ("R", "A", "S"), f"Non-RAS CT affine: {mf}"
            assert ct_pet_img.shape == pet_img.shape and np.allclose(ct_pet_img.affine, pet_img.affine)
            assert seg_ct_img.shape == ct_img.shape and np.allclose(seg_ct_img.affine, ct_img.affine)

            # Check physical RAS+ centroid co-registration between PET seg and native CT seg
            # Use Ctrl label (2) on c2_M1L1R Day -15 since RT touches the CT FOV edge
            check_lbl = 2 if (mid == "c2_M1L1R" and day == -15) else (1 if meta["has_rt_roi"] else 2)
            pts_pet = np.argwhere(seg_arr == check_lbl).mean(axis=0)
            pts_ct = np.argwhere(np.asarray(seg_ct_img.dataobj, dtype=np.uint8) == check_lbl).mean(axis=0)
            ras_pet = nib.affines.apply_affine(pet_img.affine, pts_pet)
            ras_ct = nib.affines.apply_affine(ct_img.affine, pts_ct)
            dist_mm = float(np.linalg.norm(ras_pet - ras_ct))
            assert dist_mm < 1.0, f"PET/CT RAS centroid mismatch ({dist_mm:.2f} mm) in {mf}"
            ct_msg = f"CT={ct_img.shape}, PET-CT centroid diff={dist_mm:.2f} mm"

        print(f"  OK {out_dir.relative_to(nifti_root).as_posix():30s} | RAS+ | {' | '.join(parts)} | {ct_msg}")
    print(f"ALL {len(meta_files)} NIFTI SCAN SETS VERIFIED!")


def _backfill_nifti_dir_in_tables(root: Path = Path("/workspace")) -> None:
    """Adds nifti_dir column to existing longitudinal_imaging_summary.csv without rerunning full cropping."""
    for p in [
        root / "results/first_ef5_rt_study/longitudinal_imaging_summary.csv",
        root / "outputs/first_ef5_rt_study/tables/longitudinal_imaging_summary.csv",
    ]:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        nifti_dirs = df["voxel_bundle_path"].map(
            lambda s: f"outputs/first_ef5_rt_study/nifti/{Path(s).parent.name}/{Path(s).stem[len(Path(s).parent.name) + 1:]}"
        )
        if "nifti_dir" in df.columns:
            df["nifti_dir"] = nifti_dirs
        else:
            idx = df.columns.get_loc("voxel_bundle_path") + 1
            df.insert(idx, "nifti_dir", nifti_dirs)
        df.to_csv(p, index=False)


def verify_voxel_bundles(root: Path = Path("/workspace")) -> None:
    """Verifies all exported .npz voxel bundles against longitudinal_imaging_summary.csv."""
    df = pd.read_csv(root / "results/first_ef5_rt_study/longitudinal_imaging_summary.csv")
    bundles = sorted((root / "outputs/first_ef5_rt_study/voxel_bundles").rglob("*.npz"))
    expected = len(SCAN_SPECS)
    assert len(bundles) == expected, f"Expected {expected} bundles, found {len(bundles)}"
    print(f"\n=== Verifying {len(bundles)} .npz Voxel Bundles ===")
    for b in bundles:
        d = np.load(b)
        mid = str(d["mouse_id"])
        day = int(d["days_post_rt"])
        has_rt = bool(d["has_rt_roi"])
        has_ctrl = bool(d["has_ctrl_roi"])
        summary_parts = []
        for flag, trt, mkey, lbl in [
            (has_rt, "RT_20Gy", "mask_rt_pet", "RT"),
            (has_ctrl, "Control_0Gy", "mask_ctrl_pet", "Ctrl"),
        ]:
            if flag:
                m = d[mkey]
                mean_idg = float(d["pet_idg"][m].mean())
                mean_tm = float(d["pet_tm"][m].mean())
                row = df[(df["mouse_id"] == mid) & (df["days_post_rt"] == day) & (df["treatment"] == trt)].iloc[0]
                assert abs(mean_idg - row["pet_mean_idg"]) < 1e-3, f"PET mean mismatch: {mid} {day} {trt}"
                assert abs(mean_tm - row["tm_mean"]) < 1e-3, f"T/M mismatch: {mid} {day} {trt}"
                summary_parts.append(f"{lbl}={mean_idg:.4f}(T/M {mean_tm:.2f})")
            else:
                assert not d[mkey].any(), f"Unexpected non-empty mask when {lbl} absent: {mid} {day}"
        ct_info = "PET-only"
        if bool(d["ct_available"]):
            assert d["ct_on_pet_grid"].shape == d["pet_idg"].shape
            ref_mask = d["mask_rt_pet"] if has_rt else d["mask_ctrl_pet"]
            ct_m = float(d["ct_on_pet_grid"][ref_mask].mean())
            ct_info = f"CT={d['ct_hu'].shape}, ct_on_pet_mean={ct_m:.1f} HU"
        print(f"  OK {b.name:38s} | PET={d['pet_idg'].shape} | {' | '.join(summary_parts)} | {ct_info}")
    print(f"ALL {len(bundles)} VOXEL BUNDLES VERIFIED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="First EF5 RT Study (A549) processing & export pipeline.")
    parser.add_argument(
        "--nifti-only",
        action="store_true",
        help="Export and verify NIfTI (.nii.gz) volumes directly from existing verified .npz voxel bundles.",
    )
    args = parser.parse_args()
    if args.nifti_only:
        _backfill_nifti_dir_in_tables()
        export_nifti_dataset()
        verify_nifti_outputs()
    else:
        main()
