"""
Consolidated data curation, verification, and export pipeline for the Second EF5 Radiation Therapy
(RT) Study (April – July 2011; 40-mouse, 80-tumour A549 subcutaneous cohort across 8 single-dose
and fractionated radiotherapy groups, published in Ali et al., PLOS ONE 2015).

Processes all 39 imaged mice across the 10 Siemens Inveon PET/CT 4-mouse scanner beds
(Day 1 Sets 1..5 on 2011-06-01 [i100117], Day 2 Sets 1..5 on 2011-06-02 [i100118]; mouse c4_M1R
was not imaged due to absence of visible tumour uptake) and reconciles all 40 mice / 80 tumours
across all 11 longitudinal caliper measurement timepoints (Day 23 to Day 99 post-injection;
Day -23 to Day +52 post-RT).

Outputs:
  1. 10 full-bed 4-mouse scanner overview PNGs under:
       outputs/second_ef5_rt_study/bed_overviews/<set_slug>_overview.png
  2. Per-mouse cropped PET DICOM, canonical axis-aligned CT DICOM, merged binary RT_Image `rois.ROI`,
     and 2x3 visual QC overview PNGs under:
       outputs/second_ef5_rt_study/cropped_mice/<mouse_id>/
  3. Simulator-ready compressed NumPy `.npz` voxel bundles (compatible with `initialize_tumour_from_npz`
     for both `side="lhs"` and `side="rhs"`) under:
       outputs/second_ef5_rt_study/voxel_bundles/<mouse_id>/<mouse_id>_pre_rt.npz
  4. Standard RAS+ compressed NIfTI (`.nii.gz`) volumes and BIDS-style JSON sidecars under:
       outputs/second_ef5_rt_study/nifti/<mouse_id>/
  5. Curated, publication-ready summary CSV tables and audit documentation (saved to both
     `outputs/second_ef5_rt_study/tables/` and git-tracked `results/second_ef5_rt_study/`):
       - mouse_id_mapping.csv
       - imaging_summary.csv
       - caliper_growth_curves.csv
       - tumour_master_wide.csv
       - study_timeline.csv
       - data_audit_notes.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from src.process_first_ef5_rt_study import _canonical_array_to_nifti_ras, _save_nifti
from src.rt_image_loader import (
    RTRoi,
    RTImageVolume,
    build_roi_3d_mask,
    crop_volume_to_body_roi,
    load_inveon_volume,
    load_rt_image_dicom,
    load_rt_image_roi,
    parse_inveon_hdr,
    save_rt_image_dicom,
    save_rt_image_roi_subset,
)


# ---------------------------------------------------------------------------
# Definitive study dates & treatment regimen definitions
# ---------------------------------------------------------------------------
INJECTION_DATE = datetime.date(2011, 4, 18)
DAY1_SCAN_DATE = datetime.date(2011, 6, 1)   # i100117 (Cages 1-4)
DAY2_SCAN_DATE = datetime.date(2011, 6, 2)   # i100118 (Cages 5-8)
DAY1_RT_DATE = datetime.date(2011, 6, 2)     # Single-fraction RT for Cages 1-3 (Day 45 post-inj)
DAY2_RT_DATE = datetime.date(2011, 6, 3)     # Fraction #1 for Cages 5-8 (Day 46 post-inj)

# Exact mapping of the 11 caliper measurement timepoints (Days Post-Injection -> Calendar Date)
CALIPER_DAYS_TO_DATE: Dict[int, datetime.date] = {
    23: datetime.date(2011, 5, 11),
    30: datetime.date(2011, 5, 18),
    35: datetime.date(2011, 5, 23),
    43: datetime.date(2011, 5, 31),  # Pre-RT baseline (-3d vs June 3 / -2d vs June 2)
    49: datetime.date(2011, 6, 6),   # +3d / +4d post-RT
    56: datetime.date(2011, 6, 13),  # +10d / +11d post-RT
    64: datetime.date(2011, 6, 21),  # +18d / +19d post-RT
    73: datetime.date(2011, 6, 30),  # +27d / +28d post-RT (end of PLOS ONE window)
    79: datetime.date(2011, 7, 6),   # +33d / +34d post-RT (labeled +32d/+33d in v4.xlsb)
    84: datetime.date(2011, 7, 20),  # Khalila measurement (20.07.11; labeled Day 84 / +37d in v4.xlsb)
    99: datetime.date(2011, 7, 26),  # Final endpoint caliper (labeled Day 99 / +52d/+53d in v4.xlsb)
}

# Canonical nominal Days Post-RT labels used in the study reports & spreadsheets
NOMINAL_DAYS_POST_RT: Dict[int, int] = {
    23: -23,
    30: -16,
    35: -11,
    43: -3,
    49: 3,
    56: 10,
    64: 18,
    73: 27,
    79: 33,
    84: 37,
    99: 53,
}

CAGE_REGIMENS: Dict[int, Dict[str, Any]] = {
    1: {
        "group_num": 1,
        "treatment": "1x10Gy",
        "total_dose_gy": 10.0,
        "num_fractions": 1,
        "fraction_dose_gy": 10.0,
        "rt_start_date": "2011-06-02",
        "rt_schedule_note": "Single 10 Gy fraction delivered on 2011-06-02 (1 day after Day 1 PET/CT).",
    },
    2: {
        "group_num": 2,
        "treatment": "1x20Gy",
        "total_dose_gy": 20.0,
        "num_fractions": 1,
        "fraction_dose_gy": 20.0,
        "rt_start_date": "2011-06-02",
        "rt_schedule_note": "Single 20 Gy fraction delivered on 2011-06-02 (1 day after Day 1 PET/CT).",
    },
    3: {
        "group_num": 3,
        "treatment": "1x40Gy",
        "total_dose_gy": 40.0,
        "num_fractions": 1,
        "fraction_dose_gy": 40.0,
        "rt_start_date": "2011-06-02",
        "rt_schedule_note": "Single 40 Gy fraction delivered on 2011-06-02 (1 day after Day 1 PET/CT).",
    },
    4: {
        "group_num": 4,
        "treatment": "Control_0Gy",
        "total_dose_gy": 0.0,
        "num_fractions": 0,
        "fraction_dose_gy": 0.0,
        "rt_start_date": "2011-06-03",
        "rt_schedule_note": "Unirradiated negative control (0 Gy).",
    },
    5: {
        "group_num": 5,
        "treatment": "2x5Gy",
        "total_dose_gy": 10.0,
        "num_fractions": 2,
        "fraction_dose_gy": 5.0,
        "rt_start_date": "2011-06-03",
        "rt_schedule_note": "Fraction 1 (5 Gy) on 2011-06-03; Fraction 2 (5 Gy) on 2011-06-07 (4-day gap due to missing lead shield).",
    },
    6: {
        "group_num": 6,
        "treatment": "2x10Gy",
        "total_dose_gy": 20.0,
        "num_fractions": 2,
        "fraction_dose_gy": 10.0,
        "rt_start_date": "2011-06-03",
        "rt_schedule_note": "Fraction 1 (10 Gy) on 2011-06-03; Fraction 2 (10 Gy) on 2011-06-07 (4-day gap due to missing lead shield).",
    },
    7: {
        "group_num": 7,
        "treatment": "4x5Gy",
        "total_dose_gy": 20.0,
        "num_fractions": 4,
        "fraction_dose_gy": 5.0,
        "rt_start_date": "2011-06-03",
        "rt_schedule_note": "Fraction 1 (5 Gy) on 2011-06-03; Fractions 2-4 (5 Gy daily) on 2011-06-07..09 (4-day gap after Fraction 1).",
    },
    8: {
        "group_num": 8,
        "treatment": "4x10Gy",
        "total_dose_gy": 40.0,
        "num_fractions": 4,
        "fraction_dose_gy": 10.0,
        "rt_start_date": "2011-06-03",
        "rt_schedule_note": "Fraction 1 (10 Gy) on 2011-06-03; Fractions 2-4 (10 Gy daily) on 2011-06-07..09 (4-day gap after Fraction 1).",
    },
}

EAR_PUNCH_ORDER = ["0", "1L", "1R", "1L1R", "2R"]
EAR_TO_MOUSE_NUM = {"0": 1, "1L": 2, "1R": 3, "1L1R": 4, "2R": 5}


@dataclass(frozen=True)
class BedMouseSpec:
    mouse_id: str
    cage_num: int
    ear_punch: str
    mouse_num: int
    scan_set_label: str
    scan_set_slug: str
    scan_date: str
    days_post_injection: int
    days_post_rt: int
    analysed_rel_dir: str
    pet_dcm_rel: str
    ct_img_rel: str
    pet_hdr_rel: str
    body_roi_file: str
    body_roi_name: str
    tumor_roi_file: str
    lhs_roi_name: Optional[str]
    rhs_roi_name: Optional[str]
    bg_roi_file: str
    bg_roi_name: str
    indiv_tumors_rel_dir: Optional[str] = None
    indiv_lhs_pet_roi: Optional[str] = None
    indiv_rhs_pet_roi: Optional[str] = None
    indiv_lhs_ct_roi: Optional[str] = None
    indiv_rhs_ct_roi: Optional[str] = None
    indiv_bg_roi: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Complete manifest of all 39 imaged mice across the 10 scanner beds
# ---------------------------------------------------------------------------
BED_MOUSE_SPECS: List[BedMouseSpec] = [
    # === Day 1 Set 1 (2011-06-01, 3 mice: c1_M0, c1_M1L, c1_M2R) ===
    BedMouseSpec(
        "c1_M0", 1, "0", 1, "Day 1 Set 1", "day1_set1", "2011-06-01", 44, -1,
        "Day 1 Set 1", "ef5_day1_sec1.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v1.pet.img.hdr",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M0_BR",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M0_L group", "c1_M0_R group",
        "ef5_day1_sec1_bg_roi.ROI", "M0 bg ROI",
        indiv_tumors_rel_dir="Set 1/C1 M0",
        indiv_lhs_pet_roi="c1_M0_L PET ROI", indiv_rhs_pet_roi="c1_M0_R PET ROI",
        indiv_lhs_ct_roi="c1_M0_L CT ROI", indiv_rhs_ct_roi="c1_M0_R CT ROI",
        indiv_bg_roi="M0 bg ROI",
    ),
    BedMouseSpec(
        "c1_M1L", 1, "1L", 2, "Day 1 Set 1", "day1_set1", "2011-06-01", 44, -1,
        "Day 1 Set 1", "ef5_day1_sec1.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v1.pet.img.hdr",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M1L_BL",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M1L_L group", "c1_M1L_R group",
        "ef5_day1_sec1_bg_roi.ROI", "M1L bg ROI",
        indiv_tumors_rel_dir="Set 1/C1 M1L",
        indiv_lhs_pet_roi="c1_M1L_L PET ROI", indiv_rhs_pet_roi="c1_M1L_R PET ROI",
        indiv_lhs_ct_roi="c1_M1L_L CT ROI", indiv_rhs_ct_roi="c1_M1L_R CT ROI",
        indiv_bg_roi="M1L bg ROI",
    ),
    BedMouseSpec(
        "c1_M2R", 1, "2R", 5, "Day 1 Set 1", "day1_set1", "2011-06-01", 44, -1,
        "Day 1 Set 1", "ef5_day1_sec1.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v1.pet.img.hdr",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M2R_T",
        "ef5_day1_sec1_roi.roi.ROI", "c1_M2R_L group", "c1_M2R_R group",
        "ef5_day1_sec1_bg_roi.ROI", "M2R bg ROI",
        notes="3-mouse bed (c1_M2R placed alone in top half of bed).",
    ),
    # === Day 1 Set 2 (2011-06-01, 4 mice: c1_M1R, c1_M1L1R, c2_M0, c2_M2R) ===
    BedMouseSpec(
        "c1_M1R", 1, "1R", 3, "Day 1 Set 2", "day1_set2", "2011-06-01", 44, -1,
        "Day 1 Set 2", "ef5_day1_sec20001.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v2.pet.img.hdr",
        "ef5_day1_sec2_roi.roi.ROI", "c1_M1R_BR",
        "ef5_day1_sec2_roi.roi.ROI", "c1_M1R_L group", "c1_M1R_R group",
        "Individual Tumors:Set 1/C1 M1R/rois.ROI", "c1_M1R bg ROI",
        indiv_tumors_rel_dir="Set 1/C1 M1R",
        indiv_lhs_pet_roi="c1_M1R_L PET ROI", indiv_rhs_pet_roi="c1_M1R_R PET ROI",
        indiv_lhs_ct_roi="c1_M1R_L CT ROI", indiv_rhs_ct_roi="c1_M1R_R CT ROI",
        indiv_bg_roi="c1_M1R bg ROI",
    ),
    BedMouseSpec(
        "c1_M1L1R", 1, "1L1R", 4, "Day 1 Set 2", "day1_set2", "2011-06-01", 44, -1,
        "Day 1 Set 2", "ef5_day1_sec20001.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v2.pet.img.hdr",
        "ef5_day1_sec2_roi.roi.ROI", "c1_M1L1R_BL",
        "ef5_day1_sec2_roi.roi.ROI", "c1_M1L1R_L group", "c1_M1L1R_R group",
        "Individual Tumors:Set 1/C1 M1L1R/rois.ROI", "c1_M1L1R bg ROI",
        indiv_tumors_rel_dir="Set 1/C1 M1L1R",
        indiv_lhs_pet_roi="c1_M1L1R_L PET ROI", indiv_rhs_pet_roi="c1_M1L1R_R PET ROI",
        indiv_lhs_ct_roi="c1_M1L1R_L CT ROI", indiv_rhs_ct_roi="c1_M1L1R_R CT ROI",
        indiv_bg_roi="c1_M1L1R bg ROI",
    ),
    BedMouseSpec(
        "c2_M0", 2, "0", 1, "Day 1 Set 2", "day1_set2", "2011-06-01", 44, -1,
        "Day 1 Set 2", "ef5_day1_sec20001.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v2.pet.img.hdr",
        "ef5_day1_sec2_roi.roi.ROI", "c2_M0_TL",
        "ef5_day1_sec2_roi.roi.ROI", "c2_M0_L group", "c2_M0_R group",
        "Individual Tumors:Set 1/C2 M0/rois.ROI", "c2_M0 bg ROI",
        indiv_tumors_rel_dir="Set 1/C2 M0",
        indiv_lhs_pet_roi="c2_M0_L PET ROI", indiv_rhs_pet_roi="c2_M0_R PET ROI",
        indiv_lhs_ct_roi="c2_M0_L CT ROI", indiv_rhs_ct_roi="c2_M0_R CT ROI",
        indiv_bg_roi="c2_M0 bg ROI",
    ),
    BedMouseSpec(
        "c2_M2R", 2, "2R", 5, "Day 1 Set 2", "day1_set2", "2011-06-01", 44, -1,
        "Day 1 Set 2", "ef5_day1_sec20001.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v2.pet.img.hdr",
        "ef5_day1_sec2_roi.roi.ROI", "c2_M2R_TR",
        "ef5_day1_sec2_roi.roi.ROI", "c2_M2R_L group", "c2_M2R_R group",
        "Individual Tumors:Set 1/C2 M2R/rois.ROI", "c2 M2R bg ROI",
        indiv_tumors_rel_dir="Set 1/C2 M2R",
        indiv_lhs_pet_roi="c2_M2R_L PET ROI", indiv_rhs_pet_roi="c2_M2R_R PET ROI",
        indiv_lhs_ct_roi="c2_M2R_L CT ROI", indiv_rhs_ct_roi="c2_M2R_R CT ROI",
        indiv_bg_roi="c2 M2R bg ROI",
    ),
    # === Day 1 Set 3 (2011-06-01, 4 mice: c2_M1L, c2_M1R, c2_M1L1R, c3_M1L; uses 2nd try co-registered PET) ===
    BedMouseSpec(
        "c2_M1L", 2, "1L", 2, "Day 1 Set 3", "day1_set3", "2011-06-01", 44, -1,
        "Day 1 Set 3", "2nd try/ef5_day1_sec3.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v3.pet.img.hdr",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1L_TR",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1L_L group", "c2_M1L_R group",
        "Individual Tumors:Set 1/C2 M1L/rois.ROI", "c2_M1L bg ROI",
        indiv_tumors_rel_dir="Set 1/C2 M1L",
        indiv_lhs_pet_roi="c2_M1L_L PET ROI", indiv_rhs_pet_roi="c2_M1L_R PET ROI",
        indiv_lhs_ct_roi="c2_M1L_L CT ROI", indiv_rhs_ct_roi="c2_M1L_R CT ROI",
        indiv_bg_roi="c2_M1L bg ROI",
    ),
    BedMouseSpec(
        "c2_M1R", 2, "1R", 3, "Day 1 Set 3", "day1_set3", "2011-06-01", 44, -1,
        "Day 1 Set 3", "2nd try/ef5_day1_sec3.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v3.pet.img.hdr",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1R_BR",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1R_L group", "c2_M1R_R group",
        "2nd try/ef5_day1_sec3_bg_roi.ROI", "c2 M1R bg ROI",
        indiv_tumors_rel_dir="Set 1/C2 M1R",
        indiv_lhs_pet_roi="c2_M1R L PET ROI", indiv_rhs_pet_roi="c2_M1R_R PET ROI",
        indiv_lhs_ct_roi="c2_M1R_L CT ROI", indiv_rhs_ct_roi="c2_M1R_R CT ROI",
        indiv_bg_roi="c2 M1R bg ROI",
    ),
    BedMouseSpec(
        "c2_M1L1R", 2, "1L1R", 4, "Day 1 Set 3", "day1_set3", "2011-06-01", 44, -1,
        "Day 1 Set 3", "2nd try/ef5_day1_sec3.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v3.pet.img.hdr",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1L1R_BL",
        "2nd try/ef5_day1_sec3_roi.ROI", "c2_M1L1R_L group", "c2_M1L1R_R group",
        "Individual Tumors:Set 1/C2 M1L1R/rois.ROI", "c2_M1L1R bg ROI",
        indiv_tumors_rel_dir="Set 1/C2 M1L1R",
        indiv_lhs_pet_roi="c2_M1L1R_L PET ROI", indiv_rhs_pet_roi="c2_M1L1R_R PET ROI",
        indiv_lhs_ct_roi=None, indiv_rhs_ct_roi=None,
        indiv_bg_roi="c2_M1L1R bg ROI",
        notes="Twisted mouse posture on bed; PET-only hotspot ROIs used in Individual Tumors (CT ROIs omitted in v4/corrected table).",
    ),
    BedMouseSpec(
        "c3_M1L", 3, "1L", 2, "Day 1 Set 3", "day1_set3", "2011-06-01", 44, -1,
        "Day 1 Set 3", "2nd try/ef5_day1_sec3.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v3.pet.img.hdr",
        "2nd try/ef5_day1_sec3_roi.ROI", "c3_M1L_TL",
        "2nd try/ef5_day1_sec3_roi.ROI", "c3_M1L_L group", "c3_M1L_R group",
        "2nd try/ef5_day1_sec3_bg_roi.ROI", "c3 M1L bg ROI",
        indiv_tumors_rel_dir="Set 1/C3 M1L",
        indiv_lhs_pet_roi="c3_M1L_L PET ROI", indiv_rhs_pet_roi="c3_M1L_R PET ROI",
        indiv_lhs_ct_roi="c3_M1L_L CT ROI", indiv_rhs_ct_roi="c3_M1L_R CT ROI",
        indiv_bg_roi="c3 M1L bg ROI",
    ),
    # === Day 1 Set 4 (2011-06-01, 4 mice: c3_M0, c3_M1R, c3_M1L1R, c3_M2R) ===
    BedMouseSpec(
        "c3_M0", 3, "0", 1, "Day 1 Set 4", "day1_set4", "2011-06-01", 44, -1,
        "Day 1 Set 4", "ef5_day1_sec4.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v4.pet.img.hdr",
        "ef5_day1_sec4_roi.ROI", "c3_M0_TR",
        "ef5_day1_sec4_roi.ROI", "c3_M0_L group", "c3_M0_R group",
        "ef5_day1_sec4_bg_roi.ROI", "c3 M0 bg ROI",
        indiv_tumors_rel_dir="Set 2/C3 M0",
        indiv_lhs_pet_roi="c3_M0_LHS PET ROI", indiv_rhs_pet_roi="c3_M0_RHS PET ROI",
        indiv_lhs_ct_roi="c3_M0_LHS CT ROI", indiv_rhs_ct_roi="c3_M0_RHS CT ROI",
        indiv_bg_roi="c3 M0 bg ROI",
    ),
    BedMouseSpec(
        "c3_M1R", 3, "1R", 3, "Day 1 Set 4", "day1_set4", "2011-06-01", 44, -1,
        "Day 1 Set 4", "ef5_day1_sec4.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v4.pet.img.hdr",
        "ef5_day1_sec4_roi.ROI", "c3_M1R_BL",
        "ef5_day1_sec4_roi.ROI", "c3_M1R_L group", "c3_M1R_R group",
        "ef5_day1_sec4_bg_roi.ROI", "c3 M1R bg ROI",
        indiv_tumors_rel_dir="Set 2/C3 M1R",
        indiv_lhs_pet_roi="c3_M1R_LHS PET ROI", indiv_rhs_pet_roi="c3_M1R_RHS PET ROI",
        indiv_lhs_ct_roi="c3_M1R_LHS CT ROI", indiv_rhs_ct_roi="c3_M1R_RHS CT ROI",
        indiv_bg_roi="c3 M1R bg ROI",
    ),
    BedMouseSpec(
        "c3_M1L1R", 3, "1L1R", 4, "Day 1 Set 4", "day1_set4", "2011-06-01", 44, -1,
        "Day 1 Set 4", "ef5_day1_sec4.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v4.pet.img.hdr",
        "ef5_day1_sec4_roi.ROI", "c3_M1L1R_TL",
        "ef5_day1_sec4_roi.ROI", "c3_M1L1R_L group", "c3_M1L1R_R group",
        "ef5_day1_sec4_bg_roi.ROI", "c3 M1L1R bg ROI",
        indiv_tumors_rel_dir="Set 2/C3 M1L1R",
        indiv_lhs_pet_roi="c3_M1L1R_LHS PET ROI", indiv_rhs_pet_roi="c3_M1L1R_RHS PET ROI",
        indiv_lhs_ct_roi="c3_M1L1R_LHS CT ROI", indiv_rhs_ct_roi="c3_M1L1R_RHS CT ROI",
        indiv_bg_roi="c3 M1L1R bg ROI",
    ),
    BedMouseSpec(
        "c3_M2R", 3, "2R", 5, "Day 1 Set 4", "day1_set4", "2011-06-01", 44, -1,
        "Day 1 Set 4", "ef5_day1_sec4.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v4.pet.img.hdr",
        "ef5_day1_sec4_roi.ROI", "c3_M2R_BR",
        "ef5_day1_sec4_roi.ROI", "c3_M2R_L group", "c3_M2R_R group",
        "ef5_day1_sec4_bg_roi.ROI", "c3 M2R bg ROI",
        indiv_tumors_rel_dir="Set 2/C3 M2R",
        indiv_lhs_pet_roi="c3_M2R_LHS PET ROI", indiv_rhs_pet_roi="c3_M2R_RHS PET ROI",
        indiv_lhs_ct_roi="c3_M2R_LHS CT ROI", indiv_rhs_ct_roi="c3_M2R_RHS CT ROI",
        indiv_bg_roi="c3 M2R bg ROI",
    ),
    # === Day 1 Set 5 (2011-06-01, 4 mice: c4_M0, c4_M1L, c4_M1L1R, c4_M2R; c4_M1R not imaged) ===
    BedMouseSpec(
        "c4_M0", 4, "0", 1, "Day 1 Set 5", "day1_set5", "2011-06-01", 44, -2,
        "Day 1 Set 5", "set5_pet.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c4 M0 body",
        "mice_tumor_rois.ROI", "c4 M0 LHS Tumor", "c4 M0 RHS Tumor",
        "mice_bg_rois.ROI", "c4 M0 bg ROI",
        notes="Low PET signal in both tumours (CT ROIs drawn; PET hotspot ROIs omitted).",
    ),
    BedMouseSpec(
        "c4_M1L", 4, "1L", 2, "Day 1 Set 5", "day1_set5", "2011-06-01", 44, -2,
        "Day 1 Set 5", "set5_pet.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c4 M1L body",
        "mice_tumor_rois.ROI", "c4 M1L LHS Tumor", "c4 M1L RHS Tumor",
        "mice_bg_rois.ROI", "c4 M1L bg ROI",
    ),
    BedMouseSpec(
        "c4_M1L1R", 4, "1L1R", 4, "Day 1 Set 5", "day1_set5", "2011-06-01", 44, -2,
        "Day 1 Set 5", "set5_pet.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c4 M1L1R body",
        "mice_tumor_rois.ROI", "c4 M1L1R LHS Tumor", "c4 M1L1R RHS Tumor",
        "mice_bg_rois.ROI", "c4 M1L1R bg ROI",
        notes="CT ROIs drawn; PET hotspot ROIs omitted due to low/diffuse signal.",
    ),
    BedMouseSpec(
        "c4_M2R", 4, "2R", 5, "Day 1 Set 5", "day1_set5", "2011-06-01", 44, -2,
        "Day 1 Set 5", "set5_pet.dcm", "i100117_v1.ct.img", "i100117_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c4 M2R body",
        "mice_tumor_rois.ROI", "c4 M2R LHS Tumor", "c4 M2R RHS Tumor",
        "mice_bg_rois.ROI", "c4 M2R bg ROI",
    ),
    # === Day 2 Set 1 (2011-06-02, 4 mice: c5_M1L, c5_M1R, c5_M1L1R, c5_M2R) ===
    BedMouseSpec(
        "c5_M1L", 5, "1L", 2, "Day 2 Set 1", "day2_set1", "2011-06-02", 45, -1,
        "Day 2 Set 1", "day2_set1_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v1.pet.img.hdr",
        "mouse_tumor_rois.ROI", "c5 M1L body",
        "mouse_tumor_rois.ROI", "c5 M1L LHS Tumor", "c5 M1L RHS Tumor",
        "mouse_tumor_rois.ROI", "c5 M1L bg ROI",
    ),
    BedMouseSpec(
        "c5_M1R", 5, "1R", 3, "Day 2 Set 1", "day2_set1", "2011-06-02", 45, -1,
        "Day 2 Set 1", "day2_set1_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v1.pet.img.hdr",
        "mouse_tumor_rois.ROI", "c5 M1R body",
        "mouse_tumor_rois.ROI", "c5 M1R LHS Tumor", "c5 M1R RHS Tumor",
        "mouse_tumor_rois.ROI", "c5 M1R bg ROI",
    ),
    BedMouseSpec(
        "c5_M1L1R", 5, "1L1R", 4, "Day 2 Set 1", "day2_set1", "2011-06-02", 45, -1,
        "Day 2 Set 1", "day2_set1_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v1.pet.img.hdr",
        "mouse_tumor_rois.ROI", "c5 M1L1R body",
        "mouse_tumor_rois.ROI", "c5 M1L1R LHS Tumor", "c5 M1L1R RHS Tumor",
        "mouse_tumor_rois.ROI", "c5 M1L1R bg ROI",
    ),
    BedMouseSpec(
        "c5_M2R", 5, "2R", 5, "Day 2 Set 1", "day2_set1", "2011-06-02", 45, -1,
        "Day 2 Set 1", "day2_set1_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v1.pet.img.hdr",
        "mouse_tumor_rois.ROI", "c5 M2R body",
        "mouse_tumor_rois.ROI", "c5 M2R LHS Tumor", "c5 M2R RHS Tumor",
        "mouse_tumor_rois.ROI", "c5 M2R bg ROI",
    ),
    # === Day 2 Set 2 (2011-06-02, 4 mice: c5_M0, c6_M0, c6_M1L, c6_M2R) ===
    BedMouseSpec(
        "c5_M0", 5, "0", 1, "Day 2 Set 2", "day2_set2", "2011-06-02", 45, -1,
        "Day 2 Set 2", "day2_set2_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v2.pet.img.hdr",
        "mice_tumors_bgs_rois.ROI", "c5 M0 body",
        "mice_tumors_bgs_rois.ROI", "c5 M0 LHS Tumor trunc", "c5 M0 RHS Tumor",
        "mice_tumors_bgs_rois.ROI", "c5 M0 bg ROI",
        notes="c5_M0 LHS caliper tumour was 0 mm3 (no palpable LHS growth), though a small shoulder contour ('c5 M0 LHS Tumor trunc') was drawn on PET/CT.",
    ),
    BedMouseSpec(
        "c6_M0", 6, "0", 1, "Day 2 Set 2", "day2_set2", "2011-06-02", 45, -1,
        "Day 2 Set 2", "day2_set2_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v2.pet.img.hdr",
        "mice_tumors_bgs_rois.ROI", "c6 M0 body",
        "mice_tumors_bgs_rois.ROI", "c6 M0 LHS Tumor", "c6 M0 RHS Tumor",
        "mice_tumors_bgs_rois.ROI", "c6 M0 bg ROI",
    ),
    BedMouseSpec(
        "c6_M1L", 6, "1L", 2, "Day 2 Set 2", "day2_set2", "2011-06-02", 45, -1,
        "Day 2 Set 2", "day2_set2_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v2.pet.img.hdr",
        "mice_tumors_bgs_rois.ROI", "c6 M1L body",
        "mice_tumors_bgs_rois.ROI", "c6 M1L LHS Tumor", "c6 M1L RHS Tumor",
        "mice_tumors_bgs_rois.ROI", "c6 M1L bg ROI",
    ),
    BedMouseSpec(
        "c6_M2R", 6, "2R", 5, "Day 2 Set 2", "day2_set2", "2011-06-02", 45, -1,
        "Day 2 Set 2", "day2_set2_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v2.pet.img.hdr",
        "mice_tumors_bgs_rois.ROI", "c6 M2R body",
        "mice_tumors_bgs_rois.ROI", "c6 M2R LHS Tumor", "c6 M2R RHS Tumor",
        "mice_tumors_bgs_rois.ROI", "c6 M2R bg ROI",
        notes="c6_M2R RHS tumour was very small (41 PET voxels) with low signal (no separate PET hotspot ROI).",
    ),
    # === Day 2 Set 3 (2011-06-02, 4 mice: c6_M1R, c6_M1L1R, c7_M0, c7_M1L) ===
    BedMouseSpec(
        "c6_M1R", 6, "1R", 3, "Day 2 Set 3", "day2_set3", "2011-06-02", 45, -1,
        "Day 2 Set 3", "day2_set3_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v3.pet.img.hdr",
        "mice_body_roi.ROI", "c6 M1R body",
        "mice_tumor_rois.ROI", "c6 M1R LHS Tumor", "c6 M1R RHS Tumor",
        "mice_bg_rois.ROI", "c6 M1R bg ROI",
    ),
    BedMouseSpec(
        "c6_M1L1R", 6, "1L1R", 4, "Day 2 Set 3", "day2_set3", "2011-06-02", 45, -1,
        "Day 2 Set 3", "day2_set3_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v3.pet.img.hdr",
        "mice_body_roi.ROI", "c6 M1L1R body",
        "mice_tumor_rois.ROI", "c6 M1L1R LHS Tumor", "c6 M1L1R RHS Tumor",
        "mice_bg_rois.ROI", "c6 M1L1R bg ROI",
        notes="c6_M1L1R LHS had no palpable caliper tumour (0 mm3), though a small contour was drawn; RHS had palpable tumour.",
    ),
    BedMouseSpec(
        "c7_M0", 7, "0", 1, "Day 2 Set 3", "day2_set3", "2011-06-02", 45, -1,
        "Day 2 Set 3", "day2_set3_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v3.pet.img.hdr",
        "mice_body_roi.ROI", "c7 M0 body",
        "mice_tumor_rois.ROI", "c7 M0 LHS Tumor", "c7 M0 RHS Tumor",
        "mice_bg_rois.ROI", "c7 M0 bg ROI",
    ),
    BedMouseSpec(
        "c7_M1L", 7, "1L", 2, "Day 2 Set 3", "day2_set3", "2011-06-02", 45, -1,
        "Day 2 Set 3", "day2_set3_pet.dcm", "i100118_v1.ct.img", "i100118_f18_10min_scan_em_v3.pet.img.hdr",
        "mice_body_roi.ROI", "c7 M1L body",
        "mice_tumor_rois.ROI", "c7 M1L LHS Tumor", "c7 M1L RHS Tumor",
        "mice_bg_rois.ROI", "c7 M1L bg ROI",
    ),
    # === Day 2 Set 4 (2011-06-02, 4 mice: c7_M1R, c7_M1L1R, c7_M2R, c8_M2R) ===
    BedMouseSpec(
        "c7_M1R", 7, "1R", 3, "Day 2 Set 4", "day2_set4", "2011-06-02", 45, -1,
        "Day 2 Set 4", "day2_set4_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v4.pet.img.hdr",
        "mice_tumors_rois.ROI", "c7 M1R body",
        "mice_tumors_rois.ROI", "c7 M1R LHS Tumor", "c7 M1R RHS Tumor",
        "mice_tumors_rois.ROI", "c7 M1R bg ROI",
    ),
    BedMouseSpec(
        "c7_M1L1R", 7, "1L1R", 4, "Day 2 Set 4", "day2_set4", "2011-06-02", 45, -1,
        "Day 2 Set 4", "day2_set4_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v4.pet.img.hdr",
        "mice_tumors_rois.ROI", "c7 M1L1R body",
        "mice_tumors_rois.ROI", "c7 M1L1R LHS Tumor trunc", "c7 M1L1R RHS Tumor trunc",
        "mice_tumors_rois.ROI", "c7 M1L1R bg ROI",
        notes="Truncated at axial FOV edge ('trunc' CT ROIs drawn; separate PET hotspot ROIs omitted).",
    ),
    BedMouseSpec(
        "c7_M2R", 7, "2R", 5, "Day 2 Set 4", "day2_set4", "2011-06-02", 45, -1,
        "Day 2 Set 4", "day2_set4_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v4.pet.img.hdr",
        "mice_tumors_rois.ROI", "c7 M2R body",
        "mice_tumors_rois.ROI", "c7 M2R LHS Tumor", "c7 M2R RHS Tumor",
        "mice_tumors_rois.ROI", "c7 M2R bg ROI",
    ),
    BedMouseSpec(
        "c8_M2R", 8, "2R", 5, "Day 2 Set 4", "day2_set4", "2011-06-02", 45, -1,
        "Day 2 Set 4", "day2_set4_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v4.pet.img.hdr",
        "mice_tumors_rois.ROI", "c8 M2R body",
        "mice_tumors_rois.ROI", "c8 M2R LHS Tumor", "c8 M2R RHS Tumor",
        "mice_tumors_rois.ROI", "c8 M2R bg ROI",
    ),
    # === Day 2 Set 5 (2011-06-02, 4 mice: c8_M0, c8_M1L, c8_M1R, c8_M1L1R) ===
    BedMouseSpec(
        "c8_M0", 8, "0", 1, "Day 2 Set 5", "day2_set5", "2011-06-02", 45, -1,
        "Day 2 Set 5", "day2_set5_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c8 M0 body",
        "mice_tumor_rois.ROI", "c8 M0 LHS Tumor", "c8 M0 RHS Tumor",
        "mice_tumor_rois.ROI", "c8 M0 bg ROI",
    ),
    BedMouseSpec(
        "c8_M1L", 8, "1L", 2, "Day 2 Set 5", "day2_set5", "2011-06-02", 45, -1,
        "Day 2 Set 5", "day2_set5_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c8 M1L body",
        "mice_tumor_rois.ROI", "c8 M1L LHS Tumor", "c8 M1L RHS Tumor",
        "mice_tumor_rois.ROI", "c8 M1L bg ROI",
    ),
    BedMouseSpec(
        "c8_M1R", 8, "1R", 3, "Day 2 Set 5", "day2_set5", "2011-06-02", 45, -1,
        "Day 2 Set 5", "day2_set5_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c8 M1R body",
        "mice_tumor_rois.ROI", "c8 M1R LHS tumor", "c8 M1R RHS Tumor",
        "mice_tumor_rois.ROI", "c8 M1R bg ROI",
    ),
    BedMouseSpec(
        "c8_M1L1R", 8, "1L1R", 4, "Day 2 Set 5", "day2_set5", "2011-06-02", 45, -1,
        "Day 2 Set 5", "day2_set5_pet.dcm", "i100118_v1.ct.img", "i100118_F18_10min_scan_em_v5.pet.img.hdr",
        "mice_tumor_rois.ROI", "c8 M1L1R body",
        "mice_tumor_rois.ROI", "c8 M1L1R LHS Tumor", "c8 M1L1R RHS Tumor",
        "mice_tumor_rois.ROI", "c8 M1L1R bg ROI",
    ),
]

SPEC_BY_MOUSE_ID: Dict[str, BedMouseSpec] = {sp.mouse_id: sp for sp in BED_MOUSE_SPECS}


# ---------------------------------------------------------------------------
# Pure-stdlib Binary Excel (.xlsb) Parser for `Tumor Sizes - v4.xlsb`
# ---------------------------------------------------------------------------
def _parse_xlsb_records(data: bytes):
    pos = 0
    n = len(data)
    while pos < n:
        b1 = data[pos]
        pos += 1
        rec_id = b1
        if (b1 & 0x80) and pos < n:
            b2 = data[pos]
            pos += 1
            rec_id = (b1 & 0x7F) | ((b2 & 0x7F) << 7)
        rec_len = 0
        shift = 0
        while pos < n:
            b = data[pos]
            pos += 1
            rec_len |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        payload = data[pos : pos + rec_len]
        pos += rec_len
        yield rec_id, payload


def _decode_rk(rk_bytes: bytes) -> float:
    rk_u = struct.unpack("<I", rk_bytes)[0]
    rk_s = struct.unpack("<i", rk_bytes)[0]
    is_mult = rk_u & 0x01
    is_int = rk_u & 0x02
    if is_int:
        val = float(rk_s >> 2)
    else:
        raw = (rk_u & ~3) << 32
        val = struct.unpack("<d", struct.pack("<Q", raw))[0]
    if is_mult:
        val /= 100.0
    return val


def read_xlsb_sheets(xlsb_path: Path) -> Dict[str, List[List[str]]]:
    with zipfile.ZipFile(xlsb_path, "r") as zf:
        strings: List[str] = []
        if "xl/sharedStrings.bin" in zf.namelist():
            for rid, pay in _parse_xlsb_records(zf.read("xl/sharedStrings.bin")):
                if rid == 19:  # BrtSSTItem
                    cch = struct.unpack_from("<I", pay, 1)[0]
                    strings.append(pay[5 : 5 + cch * 2].decode("utf-16le", errors="replace"))
        sheets: Dict[str, List[List[str]]] = {}
        for sidx in range(1, 15):
            sname = f"xl/worksheets/sheet{sidx}.bin"
            if sname not in zf.namelist():
                continue
            rows: Dict[int, Dict[int, str]] = {}
            cur_row = -1
            for rid, pay in _parse_xlsb_records(zf.read(sname)):
                if rid == 0:  # BrtRowHdr
                    cur_row = struct.unpack_from("<i", pay, 0)[0]
                    rows.setdefault(cur_row, {})
                elif rid in (2, 13):  # BrtCellRk
                    col = struct.unpack_from("<i", pay, 0)[0]
                    rows.setdefault(cur_row, {})[col] = str(_decode_rk(pay[8:12]))
                elif rid in (5, 16):  # BrtCellReal
                    col = struct.unpack_from("<i", pay, 0)[0]
                    val = struct.unpack_from("<d", pay, 8)[0]
                    rows.setdefault(cur_row, {})[col] = str(val)
                elif rid in (7, 18):  # BrtCellIsst
                    col = struct.unpack_from("<i", pay, 0)[0]
                    isst = struct.unpack_from("<I", pay, 8)[0]
                    rows.setdefault(cur_row, {})[col] = strings[isst] if isst < len(strings) else ""
                elif rid in (6, 17, 8):  # BrtCellSt / BrtFmlaString
                    col = struct.unpack_from("<i", pay, 0)[0]
                    cch = struct.unpack_from("<I", pay, 8)[0]
                    rows.setdefault(cur_row, {})[col] = pay[12 : 12 + cch * 2].decode("utf-16le", errors="replace")
                elif rid == 9:  # BrtFmlaNum
                    col = struct.unpack_from("<i", pay, 0)[0]
                    val = struct.unpack_from("<d", pay, 8)[0]
                    rows.setdefault(cur_row, {})[col] = str(val)
            out_rows: List[List[str]] = []
            for r in sorted(rows.keys()):
                if not rows[r]:
                    continue
                mcol = max(rows[r].keys())
                out_rows.append([rows[r].get(c, "") for c in range(mcol + 1)])
            sheets[f"sheet{sidx}"] = out_rows
        return sheets


# ---------------------------------------------------------------------------
# Spreadsheet metadata & caliper parsers
# ---------------------------------------------------------------------------
def load_reconciled_spreadsheet_metadata(rep_dir: Path) -> Dict[Tuple[int, int, str], Dict[str, Any]]:
    """
    Loads the left-right corrected tumour properties from
    `10 Left-Right Corrected Study Data.xlsx` (`All Properties` tab) and enriches with
    the detailed PET ROI & CT ROI statistics from `10 PET and CT Tumor ROI stats.xlsx`
    (`New EF5 Data` tab) and `Vols w new metric vs PET.xlsx` (`Sheet2` tab).

    Returns a dictionary keyed by `(cage_num, mouse_num, side_LR)` where `side_LR` is `'L'` or `'R'`.
    """
    lr_rows = read_xlsx_sheets(rep_dir / "10 Left-Right Corrected Study Data.xlsx")["All Properties"]
    plos_s2 = read_xlsx_sheets(rep_dir / "Vols w new metric vs PET.xlsx")["Sheet2"]
    stats_rows = read_xlsx_sheets(rep_dir / "10 PET and CT Tumor ROI stats.xlsx")["New EF5 Data"]
    khalila_rows = read_xlsx_sheets(rep_dir / "10 Khalila EF5Tumors.20.07.11.xlsx")["Sheet1"]

    # 1. Parse Khalila 2011-07-20 colour & excision notes per (cage_num, ear_punch)
    khalila_notes: Dict[Tuple[int, str], Dict[str, str]] = {}
    cur_cage = 0
    for r in khalila_rows[1:]:
        if not r or not r[0].strip():
            continue
        c0 = r[0].strip()
        if c0.lower().startswith("cage"):
            cur_cage = int(c0.split()[-1])
            continue
        ear = c0
        if cur_cage > 0 and ear in EAR_TO_MOUSE_NUM:
            l_note = r[4].strip() if len(r) > 4 else ""
            r_note = r[8].strip() if len(r) > 8 else ""
            l_x = r[1].strip() if len(r) > 1 else ""
            if l_x.lower() in ("deceased", "deceased?", "none") and not l_note:
                l_note = l_x
            khalila_notes[(cur_cage, ear)] = {"L": l_note, "R": r_note}

    # 2. Parse `10 PET and CT Tumor ROI stats.xlsx` (`New EF5 Data`)
    # Map each ROI label to its recorded spreadsheet stats
    new_ef5_by_mouse_side: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    new_ef5_bg_by_mouse: Dict[str, Dict[str, float]] = {}
    for r in stats_rows[1:]:
        if len(r) < 10 or not r[1].strip():
            continue
        lbl = r[1].strip()
        low = lbl.lower()
        if "mouse" in low or "body" in low:
            continue
        # Parse cage and ear punch from ROI label
        tokens = lbl.replace("_", " ").split()
        cage_tok = tokens[0].lower()
        if not (cage_tok.startswith("c") and cage_tok[1:].isdigit()):
            if tokens[0] in ("M0", "M1L", "M2R"):
                mid = f"c1_{tokens[0]}"
            else:
                continue
        else:
            cnum = int(cage_tok[1:])
            etok = tokens[1].upper()
            if not etok.startswith("M"):
                etok = "M" + etok
            mid = f"c{cnum}_{etok}"

        pts = float(r[4]) if len(r) > 4 and r[4].strip() else np.nan
        vol = float(r[5]) if len(r) > 5 and r[5].strip() else np.nan
        mn = float(r[6]) if len(r) > 6 and r[6].strip() else np.nan
        mx = float(r[7]) if len(r) > 7 and r[7].strip() else np.nan
        mean = float(r[8]) if len(r) > 8 and r[8].strip() else np.nan
        std = float(r[9]) if len(r) > 9 and r[9].strip() else np.nan
        s_val = float(r[10]) if len(r) > 10 and r[10].strip() else np.nan
        tm_pet = float(r[11]) if len(r) > 11 and r[11].strip() else np.nan
        tm_ct = float(r[12]) if len(r) > 12 and r[12].strip() else np.nan
        hf = np.nan
        if len(r) > 13 and r[13].strip() and r[13].strip().lower() != "n/a":
            try:
                hf = float(r[13])
            except ValueError:
                pass
        ks_d = float(r[15]) if len(r) > 15 and r[15].strip() else np.nan

        if "bg" in low:
            if "new in neck" not in low:
                new_ef5_bg_by_mouse[mid] = {
                    "label": lbl,
                    "points": pts,
                    "volume_cm3": vol,
                    "min_idg": mn,
                    "max_idg": mx,
                    "mean_idg": mean,
                    "std_idg": std,
                    "sum_idg": s_val,
                }
            continue

        # Determine side L or R and kind (PET vs CT)
        side = None
        for t in tokens[2:]:
            tu = t.upper()
            if tu in ("L", "LHS"):
                side = "L"
                break
            if tu in ("R", "RHS"):
                side = "R"
                break
        if side is None:
            continue
        kind = "PET" if "pet" in low else "CT"
        new_ef5_by_mouse_side[(mid, side, kind)] = {
            "label": lbl,
            "points": pts,
            "volume_cm3": vol,
            "min_idg": mn,
            "max_idg": mx,
            "mean_idg": mean,
            "std_idg": std,
            "sum_idg": s_val,
            "tm_pet": tm_pet,
            "tm_ct": tm_ct,
            "hypoxic_fraction": hf,
            "ks_d": ks_d,
        }

    # 3. Combine with `10 Left-Right Corrected Study Data.xlsx` and `Vols w new metric vs PET.xlsx`
    meta_map: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for idx in range(1, 81):
        r_lr = lr_rows[idx]
        r_s2 = plos_s2[idx + 1] + [""] * 15  # rows 2..81 in Sheet2
        cnum = int(float(r_lr[1]))
        mnum = int(float(r_lr[2]))
        ear = EAR_PUNCH_ORDER[mnum - 1]
        mid = f"c{cnum}_M{ear}"
        # In 10 Left-Right Corrected Study Data.xlsx: row 1 is L (1), row 2 is R (0)
        side = "L" if int(float(r_lr[3])) == 1 else "R"

        pet_info = new_ef5_by_mouse_side.get((mid, side, "PET"), {})
        ct_info = new_ef5_by_mouse_side.get((mid, side, "CT"), {})
        bg_info = new_ef5_bg_by_mouse.get(mid, {})
        k_note = khalila_notes.get((cnum, ear), {}).get(side, "")

        def _f_or_nan(val_str: str) -> float:
            try:
                return float(val_str)
            except Exception:
                return np.nan

        pet_tm_lr = _f_or_nan(r_lr[7])
        ct_tm_lr = _f_or_nan(r_lr[8])
        hyp_frac_lr = _f_or_nan(r_lr[9])
        ks_d_lr = _f_or_nan(r_lr[10])
        blue_score = int(float(r_lr[11])) if r_lr[11].strip() else 0
        dose_at_scan_uci = _f_or_nan(r_lr[12])
        time_to_image_hr = _f_or_nan(r_lr[13])
        ct_vol_lr = _f_or_nan(r_lr[14])

        # Sheet2 right-hand table (old/CT combo ROI T/M in col 41, ROIist in col 42, Notes in col 43)
        plos_col_ap_tm = _f_or_nan(r_s2[41])
        roi_author = r_s2[42].strip()
        roi_notes = r_s2[43].strip()

        meta_map[(cnum, mnum, side)] = {
            "mouse_id": mid,
            "cage_num": cnum,
            "mouse_num": mnum,
            "ear_punch": ear,
            "side": side,
            "tumour_id": f"{mid}_{side}",
            "plos_tumour_code": f"c{cnum}_M{mnum}{side}",
            "hi_contrast": int(float(r_lr[6])) if r_lr[6].strip() else 0,
            "blue_score": blue_score,
            "khalila_note_2011_07_20": k_note,
            "dose_at_scan_uci": dose_at_scan_uci if dose_at_scan_uci > 0 else np.nan,
            "time_to_image_hr": time_to_image_hr if time_to_image_hr > 0 else np.nan,
            "report_pet_hotspot_tm": round(float(pet_info.get("tm_pet", pet_tm_lr)), 4)
            if not np.isnan(pet_info.get("tm_pet", np.nan))
            else (round(pet_tm_lr, 4) if pet_tm_lr > 0 else np.nan),
            "report_pet_hotspot_vol_cm3": round(float(pet_info["volume_cm3"]), 4)
            if "volume_cm3" in pet_info and not np.isnan(pet_info["volume_cm3"])
            else np.nan,
            "report_pet_hotspot_mean_idg": round(float(pet_info["mean_idg"]), 4)
            if "mean_idg" in pet_info and not np.isnan(pet_info["mean_idg"])
            else np.nan,
            "report_pet_hotspot_max_idg": round(float(pet_info["max_idg"]), 4)
            if "max_idg" in pet_info and not np.isnan(pet_info["max_idg"])
            else np.nan,
            "report_ct_roi_tm": round(float(ct_info.get("tm_ct", ct_tm_lr)), 4)
            if not np.isnan(ct_info.get("tm_ct", np.nan))
            else (round(ct_tm_lr, 4) if ct_tm_lr > 0 else np.nan),
            "report_ct_roi_vol_cm3": round(float(ct_info["volume_cm3"]), 4)
            if "volume_cm3" in ct_info and not np.isnan(ct_info["volume_cm3"])
            else (round(ct_vol_lr, 4) if ct_vol_lr > 0 else np.nan),
            "report_ct_roi_mean_idg": round(float(ct_info["mean_idg"]), 4)
            if "mean_idg" in ct_info and not np.isnan(ct_info["mean_idg"])
            else np.nan,
            "report_muscle_mean_idg": round(float(bg_info["mean_idg"]), 4)
            if "mean_idg" in bg_info and not np.isnan(bg_info["mean_idg"])
            else np.nan,
            "report_hypoxic_fraction": round(float(pet_info.get("hypoxic_fraction", hyp_frac_lr)), 4)
            if not np.isnan(pet_info.get("hypoxic_fraction", np.nan))
            else (round(hyp_frac_lr, 4) if hyp_frac_lr > 0 else np.nan),
            "report_ks_d": round(float(pet_info.get("ks_d", ks_d_lr)), 4)
            if not np.isnan(pet_info.get("ks_d", np.nan))
            else (round(ks_d_lr, 4) if ks_d_lr > 0 else np.nan),
            "plos_sheet2_combo_tm": round(plos_col_ap_tm, 4) if not np.isnan(plos_col_ap_tm) else np.nan,
            "roi_author": roi_author,
            "spreadsheet_notes": roi_notes,
        }
    return meta_map


def extract_calipers_and_timeline(
    xlsb_path: Path, meta_map: Dict[Tuple[int, int, str], Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts all 11 caliper measurement timepoints (880 tumour-timepoint rows across 40 mice / 80 tumours)
    from `Tumor Sizes - v4.xlsb` (`Caliper Data` / `sheet2`) and builds the reconciled study timeline.
    """
    sheets = read_xlsb_sheets(xlsb_path)
    cal_rows = sheets["sheet2"]  # Caliper Data (881 rows: header + 80 tumours * 11 dates)

    # First pass: collect pre-RT baseline (Day 43 post-injection = 2011-05-31, Days Post-RT = -3) volumes
    pre_rt_baseline: Dict[Tuple[int, int, str], Dict[str, float]] = {}
    parsed_entries: List[Dict[str, Any]] = []

    for r in cal_rows[1:]:
        if len(r) < 10 or not r[0].strip():
            continue
        days_post_inj = int(round(float(r[1])))
        if days_post_inj not in CALIPER_DAYS_TO_DATE:
            continue
        dt = CALIPER_DAYS_TO_DATE[days_post_inj]
        cnum = int(round(float(r[3])))
        mnum = int(round(float(r[4])))
        ear = EAR_PUNCH_ORDER[mnum - 1]
        side = r[6].strip().upper()
        x_mm = float(r[7]) if r[7].strip() else 0.0
        y_mm = float(r[8]) if r[8].strip() else 0.0
        z_mm = float(r[9]) if r[9].strip() else 0.0
        measured_by = r[10].strip() if len(r) > 10 else ""

        # Volume formulas:
        # 1) PLOS ONE / Tech Report metric: V = 0.5 * L * W^2 where L = max(x,y), W = min(x,y)
        #    Note: In `Vols w new metric vs PET.xlsx` Sheet1, L2 = 0.5 * F2 * G2^2 (0.5 * x * y^2).
        #    We provide both exact `0.5 * x * y^2` (matching Sheet1/Sheet2 of Vols w new metric vs PET.xlsx)
        #    and `0.5 * max(x,y) * min(x,y)^2` plus ellipsoid `pi/6 * x * y * z` and v4 `4/3*pi*(x/2)*(y/2)^2`.
        vol_05xy2 = 0.5 * x_mm * (y_mm ** 2)
        L_mm = max(x_mm, y_mm)
        W_mm = min(x_mm, y_mm)
        vol_05LW2 = 0.5 * L_mm * (W_mm ** 2)
        vol_ellipsoid_xyz = (np.pi / 6.0) * x_mm * y_mm * z_mm
        vol_v4_43pi_xy2 = (4.0 / 3.0) * np.pi * (x_mm / 2.0) * ((y_mm / 2.0) ** 2)
        eq_diam_mm = ((6.0 * vol_05xy2 / np.pi) ** (1.0 / 3.0)) if vol_05xy2 > 0 else 0.0

        if days_post_inj == 43:
            pre_rt_baseline[(cnum, mnum, side)] = {
                "vol_05xy2": vol_05xy2,
                "vol_ellipsoid_xyz": vol_ellipsoid_xyz,
                "vol_v4_43pi_xy2": vol_v4_43pi_xy2,
                "x_mm": x_mm,
            }

        parsed_entries.append(
            {
                "cnum": cnum,
                "mnum": mnum,
                "ear": ear,
                "side": side,
                "date": dt,
                "days_post_inj": days_post_inj,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
                "measured_by": measured_by,
                "vol_05xy2": vol_05xy2,
                "vol_05LW2": vol_05LW2,
                "vol_ellipsoid_xyz": vol_ellipsoid_xyz,
                "vol_v4_43pi_xy2": vol_v4_43pi_xy2,
                "eq_diam_mm": eq_diam_mm,
            }
        )

    caliper_records: List[Dict[str, Any]] = []
    for e in parsed_entries:
        key = (e["cnum"], e["mnum"], e["side"])
        reg = CAGE_REGIMENS[e["cnum"]]
        meta = meta_map[key]
        base = pre_rt_baseline.get(key, {})
        b_05xy2 = base.get("vol_05xy2", 0.0)
        b_v4 = base.get("vol_v4_43pi_xy2", 0.0)

        rt_dt = datetime.date.fromisoformat(reg["rt_start_date"])
        exact_days_post_rt = (e["date"] - rt_dt).days
        nom_days_post_rt = NOMINAL_DAYS_POST_RT[e["days_post_inj"]]

        caliper_records.append(
            {
                "mouse_id": meta["mouse_id"],
                "cage": f"A549 c{e['cnum']}",
                "cage_num": e["cnum"],
                "ear_punch": e["ear"],
                "mouse_num": e["mnum"],
                "tumour_id": meta["tumour_id"],
                "plos_tumour_code": meta["plos_tumour_code"],
                "side": e["side"],
                "treatment": reg["treatment"],
                "total_dose_gy": reg["total_dose_gy"],
                "num_fractions": reg["num_fractions"],
                "fraction_dose_gy": reg["fraction_dose_gy"],
                "date": e["date"].isoformat(),
                "days_post_injection": e["days_post_inj"],
                "days_post_rt_nominal": nom_days_post_rt,
                "days_post_rt_exact": exact_days_post_rt,
                "in_plos_one_window": e["days_post_inj"] <= 73,
                "x_mm": round(e["x_mm"], 2),
                "y_mm": round(e["y_mm"], 2),
                "z_mm": round(e["z_mm"], 2),
                "vol_05xy2_mm3": round(e["vol_05xy2"], 4),
                "vol_05LW2_mm3": round(e["vol_05LW2"], 4),
                "vol_ellipsoid_xyz_mm3": round(e["vol_ellipsoid_xyz"], 4),
                "vol_v4_43pi_xy2_mm3": round(e["vol_v4_43pi_xy2"], 4),
                "equiv_diameter_mm": round(e["eq_diam_mm"], 4),
                "norm_vol_05xy2_vs_day_minus3": round(e["vol_05xy2"] / b_05xy2, 4) if b_05xy2 > 0 else np.nan,
                "norm_vol_v4_vs_day_minus3": round(e["vol_v4_43pi_xy2"] / b_v4, 4) if b_v4 > 0 else np.nan,
                "measured_by": e["measured_by"],
                "blue_score": meta["blue_score"],
                "pre_rt_pet_hotspot_tm": meta["report_pet_hotspot_tm"],
                "pre_rt_ct_roi_tm": meta["report_ct_roi_tm"],
                "pre_rt_plos_combo_tm": meta["plos_sheet2_combo_tm"],
            }
        )

    caliper_df = pd.DataFrame(caliper_records)
    caliper_df = caliper_df.sort_values(["cage_num", "mouse_num", "side", "days_post_injection"]).reset_index(drop=True)

    # Build comprehensive study timeline table
    timeline_rows = [
        ("2011-04-18", 0, -46, "Injected 1x10^6 A549 cells s.c. into bilateral shoulder flanks of 40 Balb/c nude mice (8 cages x 5 mice = 80 tumours).", ""),
        ("2011-05-11", 23, -23, "Caliper measurement #1 (orthogonal x, y, z axes in mm).", "Excel serial date offset by 1904/1900 system fixed to 2011-05-11."),
        ("2011-05-18", 30, -16, "Caliper measurement #2 (orthogonal x, y, z axes in mm).", ""),
        ("2011-05-23", 35, -11, "Caliper measurement #3 (orthogonal x, y, z axes in mm; cross-checked by Nicole). Scheduled [18F]-EF5 synthesis failed (May 23-25).", "Imaging postponed 1 week due to cyclotron/HPLC synthesis failures."),
        ("2011-05-30", 42, -4, "Memorial Day holiday (cyclotron not run); May 31 synthesis failed.", ""),
        ("2011-05-31", 43, -3, "Caliper measurement #4 (Pre-treatment baseline, avg largest diameter ~9 mm).", "Used as t=0 pre-treatment volume normalization baseline."),
        ("2011-06-01", 44, -2, "[18F]-EF5 PET/CT Imaging Day 1 (i100117, 5 beds: Day 1 Set 1..5, 19 mice from Cages 1-4; 200 uCi IV tail vein, 10 min static PET + CT).", "Mouse c4_M1R not imaged (no visible signal)."),
        ("2011-06-02", 45, -1, "[18F]-EF5 PET/CT Imaging Day 2 (i100118, 5 beds: Day 2 Set 1..5, 20 mice from Cages 5-8). Single-fraction RT delivered to Cages 1 (1x10 Gy), 2 (1x20 Gy), and 3 (1x40 Gy) via Philips RT-250.", "Exact PET scan timestamps verified from i100118 .pet.img.hdr."),
        ("2011-06-03", 46, 0, "Fraction #1 of fractionated RT delivered to Cages 5 (2x5 Gy), 6 (2x10 Gy), 7 (4x5 Gy), and 8 (4x10 Gy); Cage 4 left unirradiated (0 Gy control).", "Lead shield went missing over the weekend (June 4-6)."),
        ("2011-06-06", 49, 3, "Caliper measurement #5 (3-4 days post-RT start).", ""),
        ("2011-06-07", 50, 4, "Replacement lead shield found; remaining fractions delivered starting June 7 (Fraction #2 for Cages 5-6; Fractions #2-#4 for Cages 7-8).", "Resulted in a 4-day gap between Fraction #1 and Fraction #2."),
        ("2011-06-13", 56, 10, "Caliper measurement #6 (10-11 days post-RT).", ""),
        ("2011-06-21", 64, 18, "Caliper measurement #7 (18-19 days post-RT).", ""),
        ("2011-06-30", 73, 27, "Caliper measurement #8 (27-28 days post-RT; final timepoint in Vols w new metric vs PET.xlsx / PLOS ONE plots).", ""),
        ("2011-07-06", 79, 33, "Caliper measurement #9 (33-34 days post-RT; reported as 32d/33d post-tx in Technical Report & Tumor Sizes - v4.xlsb).", ""),
        ("2011-07-20", 84, 37, "Caliper measurement #10 and tumor excision / blue vs. pink vascular appearance audit by Khalila (10 Khalila EF5Tumors.20.07.11.xlsx).", "Labeled Day 84 (+37d post-RT) in Tumor Sizes - v4.xlsb."),
        ("2011-07-26", 99, 53, "Caliper measurement #11 (final endpoint measurement for surviving mice; labeled Day 99 / +52d/+53d post-RT in Tumor Sizes - v4.xlsb).", ""),
    ]
    timeline_df = pd.DataFrame(
        timeline_rows,
        columns=["date", "days_post_injection", "days_post_rt_nominal", "event", "audit_note"],
    )
    return caliper_df, timeline_df


# ---------------------------------------------------------------------------
# Visualization helpers (Full-bed 4-mouse overviews & per-mouse 2x3 QC plots)
# ---------------------------------------------------------------------------
PET_OVERLAY_CMAP = make_pet_overlay_cmap("inferno", max_alpha=0.78)


def render_bed_overview(
    out_path: Path,
    set_label: str,
    pet_vol: RTImageVolume,
    ct_vol: RTImageVolume,
    bed_specs: List[BedMouseSpec],
    rois_for_bed: List[RTRoi],
) -> None:
    """Renders a 1x3 transaxial mid-tumour overview of a 3- or 4-mouse Inveon scanner bed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body_palette = ["#00E5FF", "#76FF03", "#FF4081", "#FFD740"]
    body_names = [sp.body_roi_name for sp in bed_specs]

    for r in rois_for_bed:
        if r.name in body_names:
            c = body_palette[body_names.index(r.name) % len(body_palette)]
            ROI_STYLES[r.name] = {"color": c, "linestyle": "--", "linewidth": 1.8, "label": r.name}
        elif "bg" in r.name.lower() or "bkgd" in r.name.lower():
            ROI_STYLES[r.name] = {"color": "#FFD740", "linestyle": ":", "linewidth": 2.0, "label": r.name}
        elif any(k in r.name.lower() for k in ("_l ", "_l_", "lhs")):
            ROI_STYLES[r.name] = {"color": "#00E5FF", "linestyle": "-", "linewidth": 2.2, "label": r.name}
        else:
            ROI_STYLES[r.name] = {"color": "#FF4081", "linestyle": "-", "linewidth": 2.2, "label": r.name}

    tumour_rois = [r for r in rois_for_bed if r.name not in body_names and "bg" not in r.name.lower()]
    y_mid = float(np.mean([r.centroid_mm[1] for r in (tumour_rois or rois_for_bed)]))

    roi_masks_ct = {r.name: build_roi_3d_mask(ct_vol, r) for r in rois_for_bed}
    ct_2d, ct_ext, ct_idx, _ = get_slice_2d_and_extent(ct_vol, "transaxial", y_mid)
    pet_2d, pet_ext, pet_idx, _ = get_slice_2d_and_extent(pet_vol, "transaxial", y_mid)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), dpi=120, facecolor="#111111")
    fig.subplots_adjust(left=0.05, right=0.96, bottom=0.17, top=0.83, wspace=0.18)
    titles = [
        f"Full-Bed CT + Body & Tumour ROIs (slice {ct_idx})",
        f"Full-Bed EF5 PET (slice {pet_idx}, 0-2.5 %ID/g)",
        "Full-Bed PET/CT Overlay + ROIs",
    ]
    for idx_ax, ax in enumerate(axes):
        ax.set_facecolor("black")
        if idx_ax == 0:
            ax.imshow(ct_2d, cmap="gray", vmin=-500, vmax=800, extent=ct_ext, origin="upper", aspect="equal")
        elif idx_ax == 1:
            ax.imshow(pet_2d, cmap="inferno", vmin=0, vmax=2.5, extent=pet_ext, origin="upper", aspect="equal")
        else:
            ax.imshow(ct_2d, cmap="gray", vmin=-500, vmax=800, extent=ct_ext, origin="upper", aspect="equal")
            ax.imshow(pet_2d, cmap=PET_OVERLAY_CMAP, vmin=0, vmax=2.5, extent=pet_ext, origin="upper", aspect="equal")
        plot_rois_on_ax(ax, "transaxial", y_mid, rois_for_bed, ct_vol, roi_masks_ct)
        ax.set_xlim(ct_ext[0], ct_ext[1])
        ax.set_ylim(ct_ext[2], ct_ext[3])
        ax.set_title(titles[idx_ax], color="white", fontsize=10.5, fontweight="bold")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#666666")

    mice_str = ", ".join(sp.mouse_id for sp in bed_specs)
    fig.suptitle(
        f"Second EF5 RT Study — {set_label} ({bed_specs[0].scan_date}) [{mice_str}] — Transaxial @ Y={y_mid:.1f} mm",
        color="white",
        fontsize=13,
        fontweight="bold",
    )
    handles = [
        Line2D([0], [0], color=body_palette[i % len(body_palette)], linestyle="--", linewidth=2.0, label=f"{sp.mouse_id} ({sp.body_roi_name})")
        for i, sp in enumerate(bed_specs)
    ] + [
        Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label="LHS Tumour (larger X)"),
        Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label="RHS Tumour (smaller X)"),
        Line2D([0], [0], color="#FFD740", linestyle=":", linewidth=2.2, label="Muscle BG ROI"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(handles), 4),
        frameon=True,
        facecolor="#1E1E1E",
        edgecolor="#555555",
        labelcolor="white",
        fontsize=8.8,
    )
    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_mouse_qc_overview(
    out_path: Path,
    spec: BedMouseSpec,
    pet_vol: RTImageVolume,
    ct_vol: RTImageVolume,
    lhs_roi: Optional[RTRoi],
    rhs_roi: Optional[RTRoi],
    muscle_roi: RTRoi,
    tm_lhs: float,
    tm_rhs: float,
    pet_hotspot_rois: Optional[List[RTRoi]] = None,
) -> None:
    """
    Renders a 2-row x 3-col QC overview (Transaxial & Coronal: CT | EF5 PET | PET/CT Overlay)
    for a single cropped mouse volume.
    """
    sub_rois = [r for r in (lhs_roi, rhs_roi, muscle_roi) if r is not None]
    if lhs_roi is not None:
        ROI_STYLES[lhs_roi.name] = {"color": "#00E5FF", "linestyle": "-", "linewidth": 2.2, "label": f"LHS ({lhs_roi.name})"}
    if rhs_roi is not None:
        ROI_STYLES[rhs_roi.name] = {"color": "#FF4081", "linestyle": "-", "linewidth": 2.2, "label": f"RHS ({rhs_roi.name})"}
    ROI_STYLES[muscle_roi.name] = {"color": "#FFD740", "linestyle": ":", "linewidth": 2.0, "label": f"Muscle ({muscle_roi.name})"}

    if pet_hotspot_rois:
        for pr in pet_hotspot_rois:
            if pr is not None and pr.name not in ROI_STYLES:
                ROI_STYLES[pr.name] = {"color": "#76FF03", "linestyle": "--", "linewidth": 1.6, "label": pr.name}
                sub_rois.append(pr)

    roi_masks = {r.name: build_roi_3d_mask(ct_vol, r) for r in sub_rois}
    tumours = [r for r in (lhs_roi, rhs_roi) if r is not None] or [muscle_roi]
    y_mid = float(np.mean([r.centroid_mm[1] for r in tumours]))
    z_mid = float(np.mean([r.centroid_mm[2] for r in tumours]))

    all_pts = np.vstack([r.all_vertices for r in sub_rois])
    y_lo = max(float(ct_vol.y_coords.min()), float(all_pts[:, 1].min()) - 8.0)
    y_hi = min(float(ct_vol.y_coords.max()), float(all_pts[:, 1].max()) + 8.0)

    pet_clim = (0.0, 2.5)
    ct_clim = (-500.0, 800.0)
    reg = CAGE_REGIMENS[spec.cage_num]

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
            plot_rois_on_ax(ax, plane, coord, sub_rois, ct_vol, roi_masks)
            ax.set_xlim(ct_ext[0], ct_ext[1])
            ax.set_ylim(ylim[0], ylim[1]) if ylim is not None else ax.set_ylim(ct_ext[2], ct_ext[3])
            ax.set_title(col_titles[c_idx], color="white", fontsize=10, fontweight="bold")
            ax.tick_params(colors="white", labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor("#555555")

    tm_l_str = f"{tm_lhs:.2f}" if not np.isnan(tm_lhs) else "N/A"
    tm_r_str = f"{tm_rhs:.2f}" if not np.isnan(tm_rhs) else "N/A"
    fig.suptitle(
        f"{spec.mouse_id} (Cage {spec.cage_num} ear '{spec.ear_punch}', {reg['treatment']}) — "
        f"Pre-RT ({spec.scan_date}, {spec.scan_set_label})\n"
        f"Full-Bed Contour T/M Mean: LHS = {tm_l_str}  |  RHS = {tm_r_str}",
        color="white",
        fontsize=12.5,
        fontweight="bold",
    )
    handles = []
    if lhs_roi is not None:
        handles.append(Line2D([0], [0], color="#00E5FF", linestyle="-", linewidth=2.2, label=f"LHS Tumour ({lhs_roi.name})"))
    if rhs_roi is not None:
        handles.append(Line2D([0], [0], color="#FF4081", linestyle="-", linewidth=2.2, label=f"RHS Tumour ({rhs_roi.name})"))
    if pet_hotspot_rois and any(r is not None for r in pet_hotspot_rois):
        handles.append(Line2D([0], [0], color="#76FF03", linestyle="--", linewidth=1.8, label="PET Hotspot Sub-ROI"))
    handles.append(Line2D([0], [0], color="#FFD740", linestyle=":", linewidth=2.2, label=f"Muscle BG ({muscle_roi.name})"))
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


# ---------------------------------------------------------------------------
# Quantitative ROI stats, .npz Voxel Bundle & NIfTI Exporters
# ---------------------------------------------------------------------------
def _vol_stats(vol: RTImageVolume, roi: Optional[RTRoi]) -> Dict[str, float]:
    if roi is None:
        return {
            "points": 0, "volume_cm3": 0.0, "min": np.nan, "max": np.nan,
            "mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p95": np.nan, "sum": np.nan,
        }
    mask = build_roi_3d_mask(vol, roi)
    vals = np.asarray(vol.data[mask], dtype=np.float64)
    dz, dy, dx = vol.spacing
    vox_cm3 = (dx * dy * dz) / 1000.0
    n = int(vals.size)
    vol_cm3 = n * vox_cm3
    if n == 0:
        return {
            "points": 0, "volume_cm3": 0.0, "min": np.nan, "max": np.nan,
            "mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p95": np.nan, "sum": np.nan,
        }
    return {
        "points": n,
        "volume_cm3": round(vol_cm3, 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std(ddof=1)) if n > 1 else 0.0, 4),
        "p50": round(float(np.percentile(vals, 50)), 4),
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


def save_second_study_voxel_bundle_npz(
    out_path: Path,
    spec: BedMouseSpec,
    pet_crop: RTImageVolume,
    ct_crop: RTImageVolume,
    lhs_roi: Optional[RTRoi],
    rhs_roi: Optional[RTRoi],
    muscle_roi: RTRoi,
    muscle_mean_idg: float,
    muscle_std_idg: float,
    lhs_pet_hotspot_roi: Optional[RTRoi] = None,
    rhs_pet_hotspot_roi: Optional[RTRoi] = None,
) -> Path:
    """
    Exports a simulator-ready compressed NumPy `.npz` voxel bundle for a Second EF5 RT Study mouse.
    Includes both `mask_lhs_*` / `mask_rhs_*` and `mask_rt_*` / `mask_ctrl_*` aliases (where
    `rt` maps to `lhs` and `ctrl` maps to `rhs`) so `initialize_tumour_from_npz` works out-of-the-box.
    """
    reg = CAGE_REGIMENS[spec.cage_num]
    pet_idg = np.asarray(pet_crop.data, dtype=np.float32)
    pet_tm = (
        (pet_idg / np.float32(muscle_mean_idg)).astype(np.float32)
        if muscle_mean_idg and muscle_mean_idg > 0
        else np.full_like(pet_idg, np.nan, dtype=np.float32)
    )

    mask_lhs_pet = build_roi_3d_mask(pet_crop, lhs_roi) if lhs_roi is not None else np.zeros(pet_crop.shape, dtype=bool)
    mask_rhs_pet = build_roi_3d_mask(pet_crop, rhs_roi) if rhs_roi is not None else np.zeros(pet_crop.shape, dtype=bool)
    mask_muscle_pet = build_roi_3d_mask(pet_crop, muscle_roi)

    mask_lhs_hotspot_pet = (
        build_roi_3d_mask(pet_crop, lhs_pet_hotspot_roi)
        if lhs_pet_hotspot_roi is not None
        else np.zeros(pet_crop.shape, dtype=bool)
    )
    mask_rhs_hotspot_pet = (
        build_roi_3d_mask(pet_crop, rhs_pet_hotspot_roi)
        if rhs_pet_hotspot_roi is not None
        else np.zeros(pet_crop.shape, dtype=bool)
    )

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
    ct_on_pet = affine_transform(
        ct_hu,
        matrix=scale,
        offset=offset,
        output_shape=pet_crop.shape,
        order=1,
        mode="constant",
        cval=-1000.0,
    ).astype(np.float32)

    mask_lhs_ct = build_roi_3d_mask(ct_crop, lhs_roi) if lhs_roi is not None else np.zeros(ct_crop.shape, dtype=bool)
    mask_rhs_ct = build_roi_3d_mask(ct_crop, rhs_roi) if rhs_roi is not None else np.zeros(ct_crop.shape, dtype=bool)
    mask_muscle_ct = build_roi_3d_mask(ct_crop, muscle_roi)

    payload: Dict[str, Any] = {
        "mouse_id": np.array(spec.mouse_id),
        "cage": np.array(f"A549 c{spec.cage_num}"),
        "cage_num": np.int32(spec.cage_num),
        "ear_punch": np.array(spec.ear_punch),
        "mouse_num": np.int32(spec.mouse_num),
        "treatment": np.array(reg["treatment"]),
        "total_dose_gy": np.float32(reg["total_dose_gy"]),
        "num_fractions": np.int32(reg["num_fractions"]),
        "fraction_dose_gy": np.float32(reg["fraction_dose_gy"]),
        "scan_date": np.array(spec.scan_date),
        "days_post_injection": np.int32(spec.days_post_injection),
        "days_post_rt": np.int32(spec.days_post_rt),
        "scan_set": np.array(spec.scan_set_label),
        "has_lhs_roi": np.bool_(lhs_roi is not None),
        "has_rhs_roi": np.bool_(rhs_roi is not None),
        "lhs_roi_label": np.array(lhs_roi.name if lhs_roi is not None else ""),
        "rhs_roi_label": np.array(rhs_roi.name if rhs_roi is not None else ""),
        "muscle_roi_label": np.array(muscle_roi.name),
        "has_lhs_pet_hotspot_roi": np.bool_(lhs_pet_hotspot_roi is not None),
        "has_rhs_pet_hotspot_roi": np.bool_(rhs_pet_hotspot_roi is not None),
        # Compatibility keys for voxel_initializer.py (`side="rt"` -> LHS, `side="ctrl"` -> RHS)
        "roi_id_2010": np.array(f"c{spec.cage_num}_M{spec.mouse_num}"),
        "has_rt_roi": np.bool_(lhs_roi is not None),
        "has_ctrl_roi": np.bool_(rhs_roi is not None),
        "rt_roi_label": np.array(lhs_roi.name if lhs_roi is not None else ""),
        "ctrl_roi_label": np.array(rhs_roi.name if rhs_roi is not None else ""),
        "muscle_mean_idg": np.float32(muscle_mean_idg),
        "muscle_std_idg": np.float32(muscle_std_idg),
        "axis_convention": np.array(
            "axis0=Y(origin[1]-k*dz), axis1=Z(origin[2]-j*dy), axis2=X(origin[0]-i*dx)"
        ),
        "pet_idg": pet_idg,
        "pet_tm": pet_tm,
        "mask_lhs_pet": mask_lhs_pet,
        "mask_rhs_pet": mask_rhs_pet,
        "mask_lhs_hotspot_pet": mask_lhs_hotspot_pet,
        "mask_rhs_hotspot_pet": mask_rhs_hotspot_pet,
        "mask_rt_pet": mask_lhs_pet,
        "mask_ctrl_pet": mask_rhs_pet,
        "mask_muscle_pet": mask_muscle_pet,
        "pet_spacing_mm": np.array(pet_crop.spacing, dtype=np.float64),
        "pet_origin_mm": np.array(pet_crop.origin, dtype=np.float64),
        "pet_x_mm": pet_crop.x_coords.astype(np.float64),
        "pet_y_mm": pet_crop.y_coords.astype(np.float64),
        "pet_z_mm": pet_crop.z_coords.astype(np.float64),
        "ct_available": np.bool_(True),
        "ct_note": np.array(spec.notes),
        "ct_on_pet_grid": ct_on_pet,
        "ct_hu": ct_hu,
        "mask_lhs_ct": mask_lhs_ct,
        "mask_rhs_ct": mask_rhs_ct,
        "mask_rt_ct": mask_lhs_ct,
        "mask_ctrl_ct": mask_rhs_ct,
        "mask_muscle_ct": mask_muscle_ct,
        "ct_spacing_mm": np.array(ct_crop.spacing, dtype=np.float64),
        "ct_origin_mm": np.array(ct_crop.origin, dtype=np.float64),
        "ct_x_mm": ct_crop.x_coords.astype(np.float64),
        "ct_y_mm": ct_crop.y_coords.astype(np.float64),
        "ct_z_mm": ct_crop.z_coords.astype(np.float64),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path


def _combine_label_map_second_study(
    mask_lhs: np.ndarray,
    mask_rhs: np.ndarray,
    mask_muscle: np.ndarray,
    mask_lhs_hot: Optional[np.ndarray] = None,
    mask_rhs_hot: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Combines binary masks into a single uint8 label map:
      1 = LHS_Tumour (anatomical left, larger X)
      2 = RHS_Tumour (anatomical right, smaller X)
      3 = Muscle_BG (subscapular muscle background ROI)
      4 = LHS_PET_Hotspot (optional tighter PET-only contour when drawn)
      5 = RHS_PET_Hotspot (optional tighter PET-only contour when drawn)
    """
    seg = np.zeros(mask_muscle.shape, dtype=np.uint8)
    seg[mask_muscle] = 3
    seg[mask_rhs] = 2
    seg[mask_lhs] = 1
    return seg


def export_nifti_dataset_second_study(root: Path = Path("/workspace")) -> None:
    """
    Exports all 39 verified `.npz` voxel bundles to compressed RAS+ NIfTI (`.nii.gz`) volumes
    under `outputs/second_ef5_rt_study/nifti/<mouse_id>/` with BIDS-style JSON sidecar metadata.
    """
    voxel_root = root / "outputs" / "second_ef5_rt_study" / "voxel_bundles"
    nifti_root = root / "outputs" / "second_ef5_rt_study" / "nifti"
    nifti_root.mkdir(parents=True, exist_ok=True)

    bundles = sorted(voxel_root.rglob("*.npz"))
    assert len(bundles) == len(BED_MOUSE_SPECS), f"Expected {len(BED_MOUSE_SPECS)} .npz bundles, found {len(bundles)}"

    legend = {
        "dataset": "Second EF5 RT Study (40-Mouse, 80-Tumour A549 Cohort, Apr-Jul 2011; Ali et al. PLOS ONE 2015)",
        "coordinate_system": "NIfTI RAS+ (converted from Siemens Inveon / RT_Image DICOM LPS); nibabel axcodes = ('R', 'A', 'S')",
        "segmentation_labels": {
            "0": "Background",
            "1": "LHS_Tumour (anatomical left shoulder flank tumour; image-left, larger physical X)",
            "2": "RHS_Tumour (anatomical right shoulder flank tumour; image-right, smaller physical X)",
            "3": "Muscle_BG (subscapular muscle reference ROI)",
        },
        "files_per_mouse": {
            "*_PET_idg.nii.gz": "3D [18F]-EF5 PET uptake in %ID/g (float32)",
            "*_PET_tm.nii.gz": "3D muscle-normalized [18F]-EF5 T/M ratio map (float32)",
            "*_seg_pet.nii.gz": "Multi-label ROI mask on PET grid (uint8: 1=LHS, 2=RHS, 3=Muscle)",
            "*_CT.nii.gz": "Native-resolution canonical Inveon microCT in HU (float32)",
            "*_CT_on_pet.nii.gz": "Inveon microCT resampled onto PET grid (float32)",
            "*_seg_ct.nii.gz": "Multi-label ROI mask on native CT grid (uint8: 1=LHS, 2=RHS, 3=Muscle)",
            "*_metadata.json": "Per-mouse sidecar metadata (regimen, timing, ROI stats, notes)",
        },
    }
    (nifti_root / "label_legend.json").write_text(json.dumps(legend, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== Exporting {len(bundles)} Mice to NIfTI (.nii.gz) in {nifti_root} ===")
    for b in bundles:
        d = np.load(b)
        mid = str(d["mouse_id"])
        stem = b.stem  # <mouse_id>_pre_rt
        out_dir = nifti_root / mid
        out_dir.mkdir(parents=True, exist_ok=True)

        pet_spacing = d["pet_spacing_mm"]
        pet_origin = d["pet_origin_mm"]
        ct_spacing = d["ct_spacing_mm"]
        ct_origin = d["ct_origin_mm"]

        pet_idg_ras, pet_aff = _canonical_array_to_nifti_ras(d["pet_idg"].astype(np.float32), pet_spacing, pet_origin)
        pet_tm_ras, _ = _canonical_array_to_nifti_ras(d["pet_tm"].astype(np.float32), pet_spacing, pet_origin)
        seg_pet_yzx = _combine_label_map_second_study(d["mask_lhs_pet"], d["mask_rhs_pet"], d["mask_muscle_pet"])
        seg_pet_ras, _ = _canonical_array_to_nifti_ras(seg_pet_yzx, pet_spacing, pet_origin)

        ct_ras, ct_aff = _canonical_array_to_nifti_ras(d["ct_hu"].astype(np.float32), ct_spacing, ct_origin)
        seg_ct_yzx = _combine_label_map_second_study(d["mask_lhs_ct"], d["mask_rhs_ct"], d["mask_muscle_ct"])
        seg_ct_ras, _ = _canonical_array_to_nifti_ras(seg_ct_yzx, ct_spacing, ct_origin)
        ct_on_pet_ras, _ = _canonical_array_to_nifti_ras(d["ct_on_pet_grid"].astype(np.float32), pet_spacing, pet_origin)

        files_written = {
            "pet_idg": _save_nifti(pet_idg_ras, pet_aff, out_dir / f"{stem}_PET_idg.nii.gz", "EF5 PET (%ID/g)").name,
            "pet_tm": _save_nifti(pet_tm_ras, pet_aff, out_dir / f"{stem}_PET_tm.nii.gz", "EF5 PET (T/M ratio)").name,
            "seg_pet": _save_nifti(seg_pet_ras, pet_aff, out_dir / f"{stem}_seg_pet.nii.gz", "ROIs: 1=LHS, 2=RHS, 3=Muscle").name,
            "ct": _save_nifti(ct_ras, ct_aff, out_dir / f"{stem}_CT.nii.gz", "Inveon MicroCT (HU)").name,
            "ct_on_pet": _save_nifti(ct_on_pet_ras, pet_aff, out_dir / f"{stem}_CT_on_pet.nii.gz", "Inveon MicroCT on PET grid").name,
            "seg_ct": _save_nifti(seg_ct_ras, ct_aff, out_dir / f"{stem}_seg_ct.nii.gz", "ROIs on CT: 1=LHS, 2=RHS, 3=Muscle").name,
        }

        meta = {
            "mouse_id": mid,
            "cage": str(d["cage"]),
            "cage_num": int(d["cage_num"]),
            "ear_punch": str(d["ear_punch"]),
            "mouse_num": int(d["mouse_num"]),
            "treatment": str(d["treatment"]),
            "total_dose_gy": float(d["total_dose_gy"]),
            "num_fractions": int(d["num_fractions"]),
            "fraction_dose_gy": float(d["fraction_dose_gy"]),
            "scan_date": str(d["scan_date"]),
            "days_post_injection": int(d["days_post_injection"]),
            "days_post_rt": int(d["days_post_rt"]),
            "scan_set": str(d["scan_set"]),
            "has_lhs_roi": bool(d["has_lhs_roi"]),
            "has_rhs_roi": bool(d["has_rhs_roi"]),
            "lhs_roi_label": str(d["lhs_roi_label"]),
            "rhs_roi_label": str(d["rhs_roi_label"]),
            "muscle_roi_label": str(d["muscle_roi_label"]),
            "muscle_mean_idg": round(float(d["muscle_mean_idg"]), 6),
            "muscle_std_idg": round(float(d["muscle_std_idg"]), 6),
            "ct_available": True,
            "notes": str(d["ct_note"]),
            "pet_units": "%ID/g",
            "pet_tm_units": "Tumor-to-Muscle ratio (dimensionless)",
            "ct_units": "Hounsfield Units (HU)",
            "orientation": "RAS+",
            "segmentation_labels": {"1": "LHS_Tumour", "2": "RHS_Tumour", "3": "Muscle_BG"},
            "files": files_written,
        }
        (out_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def verify_second_study_outputs(root: Path = Path("/workspace")) -> None:
    """
    Verifies all 39 `.npz` voxel bundles and all 39 RAS+ NIfTI scan sets against `imaging_summary.csv`.
    """
    df = pd.read_csv(root / "results/second_ef5_rt_study/imaging_summary.csv")
    voxel_root = root / "outputs" / "second_ef5_rt_study" / "voxel_bundles"
    nifti_root = root / "outputs" / "second_ef5_rt_study" / "nifti"

    bundles = sorted(voxel_root.rglob("*.npz"))
    meta_files = sorted(nifti_root.rglob("*_metadata.json"))
    assert len(bundles) == 39, f"Expected 39 .npz bundles, found {len(bundles)}"
    assert len(meta_files) == 39, f"Expected 39 NIfTI folders, found {len(meta_files)}"

    print(f"\n=== Verifying 39 .npz Voxel Bundles & 39 RAS+ NIfTI Sets ===")
    for b, mf in zip(bundles, meta_files):
        d = np.load(b)
        meta = json.loads(mf.read_text(encoding="utf-8"))
        mid = str(d["mouse_id"])
        assert mid == meta["mouse_id"]

        out_dir = mf.parent
        pet_img = nib.load(str(out_dir / meta["files"]["pet_idg"]))
        tm_img = nib.load(str(out_dir / meta["files"]["pet_tm"]))
        seg_img = nib.load(str(out_dir / meta["files"]["seg_pet"]))
        ct_img = nib.load(str(out_dir / meta["files"]["ct"]))
        ct_pet_img = nib.load(str(out_dir / meta["files"]["ct_on_pet"]))
        seg_ct_img = nib.load(str(out_dir / meta["files"]["seg_ct"]))

        assert nib.aff2axcodes(pet_img.affine) == ("R", "A", "S")
        assert nib.aff2axcodes(ct_img.affine) == ("R", "A", "S")
        assert ct_pet_img.shape == pet_img.shape and np.allclose(ct_pet_img.affine, pet_img.affine)

        pet_arr = np.asarray(pet_img.dataobj, dtype=np.float32)
        tm_arr = np.asarray(tm_img.dataobj, dtype=np.float32)
        seg_arr = np.asarray(seg_img.dataobj, dtype=np.uint8)

        parts = []
        for side, lbl_id, mkey in [("L", 1, "mask_lhs_pet"), ("R", 2, "mask_rhs_pet")]:
            m_npz = d[mkey]
            m_nii = seg_arr == lbl_id
            assert int(m_npz.sum()) == int(m_nii.sum()) and int(m_npz.sum()) > 0
            mean_idg = float(d["pet_idg"][m_npz].mean())
            mean_tm = float(d["pet_tm"][m_npz].mean())
            assert abs(mean_idg - float(pet_arr[m_nii].mean())) < 1e-4
            row = df[(df["mouse_id"] == mid) & (df["side"] == side)].iloc[0]
            assert abs(mean_idg - float(row["fullbed_roi_pet_mean_idg"])) < 1e-3
            assert abs(mean_tm - float(row["fullbed_roi_tm_mean"])) < 1e-3
            parts.append(f"{side}={mean_idg:.3f}(T/M {mean_tm:.2f})")

        # Physical RAS+ centroid co-registration check between PET and native CT
        pts_pet = np.argwhere(seg_arr == 1).mean(axis=0)
        pts_ct = np.argwhere(np.asarray(seg_ct_img.dataobj, dtype=np.uint8) == 1).mean(axis=0)
        dist_mm = float(np.linalg.norm(nib.affines.apply_affine(pet_img.affine, pts_pet) - nib.affines.apply_affine(ct_img.affine, pts_ct)))
        assert dist_mm < 1.0, f"PET/CT RAS centroid mismatch ({dist_mm:.2f} mm) for {mid}"

        print(f"  OK {mid:10s} ({meta['treatment']:11s}) | PET={d['pet_idg'].shape} CT={d['ct_hu'].shape} | {' | '.join(parts)} | diff={dist_mm:.2f}mm")
    print("ALL 39 VOXEL BUNDLES & 39 NIFTI SCAN SETS VERIFIED!")


def write_data_audit_notes(out_path: Path, git_path: Path) -> None:
    """Writes a detailed markdown audit report documenting the Second EF5 RT Study curation & reconciliation."""
    content = """# Second EF5 RT Study (May–July 2011, 40-Mouse Cohort) — Data Audit & Reconciliation Notes

## 1. Study Overview & Cohort Design
- **Cohort:** 40 immunocompromised male Balb/c nude mice (8 cages $\\times$ 5 ear-punch mice: `0, 1L, 1R, 1L1R, 2R`), implanted bilaterally on the left (`L` / `LHS`) and right (`R` / `RHS`) shoulder flanks with $10^6$ human A549 lung adenocarcinoma cells on **2011-04-18** ($n = 80$ tumours total).
- **Radiotherapy Regimens (by Cage):** Both tumours on a given mouse received the same treatment regimen using a Philips RT-250 250 kVp orthovoltage X-ray unit with a lead shield exposing the tumours through circular apertures:
  - **Cage 1 (Group 1):** $1 \\times 10\\text{ Gy}$ (10 Gy total, single fraction on 2011-06-02)
  - **Cage 2 (Group 2):** $1 \\times 20\\text{ Gy}$ (20 Gy total, single fraction on 2011-06-02)
  - **Cage 3 (Group 3):** $1 \\times 40\\text{ Gy}$ (40 Gy total, single fraction on 2011-06-02)
  - **Cage 4 (Group 4):** $\\text{Control } 0\\text{ Gy}$ (0 Gy, unirradiated negative control)
  - **Cage 5 (Group 5):** $2 \\times 5\\text{ Gy}$ (10 Gy total, 2 fractions: 2011-06-03 and 2011-06-07)
  - **Cage 6 (Group 6):** $2 \\times 10\\text{ Gy}$ (20 Gy total, 2 fractions: 2011-06-03 and 2011-06-07)
  - **Cage 7 (Group 7):** $4 \\times 5\\text{ Gy}$ (20 Gy total, 4 fractions: 2011-06-03 and 2011-06-07..09)
  - **Cage 8 (Group 8):** $4 \\times 10\\text{ Gy}$ (40 Gy total, 4 fractions: 2011-06-03 and 2011-06-07..09)
- **Publication:** Ali R, Apte S, Vilalta M, et al. (2015) *18F-EF5 PET Is Predictive of Response to Fractionated Radiotherapy in Preclinical Tumor Models*, **PLOS ONE** 10(10): e0139425.

---

## 2. Reconstructed Experimental Timeline & Date Reconciliation
- **Excel 1904 vs. 1900 Date Offset:** Several historical workbooks (`10 Tumor Sizes.xlsx`, `Tumor Sizes - v4.xlsb`, `Vols w new metric vs PET.xlsx`) were saved across Mac Excel (1904 date system) and Windows Excel (1900 date system), shifting serial numbers by $\\pm 1,462$ days (displaying May–July 2011 dates as May–July 2007 or May–July 2013).
- **Exact Anchor Dates:**
  - **Implantation (Day 0):** Monday **2011-04-18** (explicitly stated in *Technical Report v1.2* §4: *"The tumors injected on Mon 18th April"*).
  - **PET/CT Imaging Day 1 (`i100117`, Day 44 post-injection):** Wednesday **2011-06-01**, 21:05–23:37 (`Day 1 Set 1..5`, 19 mice from Cages 1–4; verified from raw Siemens Inveon `.pet.img.hdr` `scan_time` headers). Mouse `c4_M1R` was not imaged.
  - **PET/CT Imaging Day 2 (`i100118`, Day 45 post-injection):** Thursday **2011-06-02**, 21:08–23:31 (`Day 2 Set 1..5`, 20 mice from Cages 5–8; verified from `i100118` `.pet.img.hdr` headers).
  - **Radiotherapy Delivery & Missing Lead Shield Gap:** Single fractions (Cages 1–3) were delivered on Thursday **2011-06-02** (the day after Day 1 imaging). Fraction #1 for fractionated Cages 5–8 was delivered on Friday **2011-06-03** (the day after Day 2 imaging). The lead shield went missing over the weekend (June 4–6); a replacement shield was found on Tuesday **2011-06-07**, resulting in a **4-day gap** between Fraction #1 and Fraction #2.
  - **11 Longitudinal Caliper Timepoints:** All 11 caliper measurement dates (Days 23, 30, 35, 43 [pre-RT baseline], 49, 56, 64, 73, 79, 84, and 99 post-injection, spanning **2011-05-11 to 2011-07-26**) are captured in `caliper_growth_curves.csv` and `tumour_master_wide.csv`. The first 8 timepoints (Days 23–73, i.e., Day -23 to Day +27 post-RT) form the uniform window plotted in `Vols w new metric vs PET.xlsx` (`in_plos_one_window = True`).

---

## 3. Forensic Resolution of Left vs. Right (`LHS` vs. `RHS`) Orientation
- **Physical Scanner Bed Geometry:** In all 10 Siemens Inveon PET/CT beds, mice were positioned prone, head-first in a $2 \\times 2$ arrangement (or 3 mice in `Day 1 Set 1`). In RT_Image physical coordinates (`X` in `[-99, 0] mm`, where transaxial slices are displayed with larger `X` on the left and smaller `X` on the right):
  - Every ROI labeled **`L` / `LHS`** has a larger physical $X$ coordinate (`X ≈ -23 mm` on the right bed column, `X ≈ -54 mm` on the left bed column) — corresponding to **image-left = mouse's anatomical Left (`L`) flank**.
  - Every ROI labeled **`R` / `RHS`** has a smaller physical $X$ coordinate (`X ≈ -42 mm` on the right bed column, `X ≈ -73 mm` on the left bed column) — corresponding to **image-right = mouse's anatomical Right (`R`) flank**.
- **Caliper vs. CT ROI Volume Correlation Proof (*Technical Report* Slide 3):**
  - Pairing caliper `L` (`LHS`) with ROI `L` (`LHS`) and caliper `R` (`RHS`) with ROI `R` (`RHS`) — as done in `10 Left-Right Corrected Study Data.xlsx` and `Vols w new metric vs PET.xlsx` — yields a strong linear correlation between pre-treatment caliper volume ($0.5 x y^2$) and 3D CT ROI volume across the cohort: **Pearson $r = 0.770$**.
  - Swapping `L` and `R` — as occurred in `Tumor Sizes - v4.xlsb` (`Variable Summary` tab, where `PET ROI T/M` and `CT ROI T/M` were inverted between `L` and `R` for every mouse) — destroys the correlation: **Pearson $r = 0.236$**.
  - All curated tables (`imaging_summary.csv`, `caliper_growth_curves.csv`, `tumour_master_wide.csv`) and 3D `.npz` / NIfTI exports use the **verified `L = LHS (larger X), R = RHS (smaller X)` assignment**.

---

## 4. Reconciliation of the Two ROI Contour Sets & Volume Metrics
1. **Full-Bed Anatomical/Combo Contours (`Analysed Data/Day * Set *`) vs. Cropped PET-Hotspot Sub-ROIs (`Individual Tumors/` & `10 PET and CT Tumor ROI stats.xlsx`):**
   - The full-bed `.ROI` files in `Analysed Data/` contain 3D tumour contours for all **39 imaged mice (78 tumour sites)** whose voxel counts and mean `%ID/g` match 100% with the **"CT ROI"** rows of `10 PET and CT Tumor ROI stats.xlsx` (`New EF5 Data`) and with the right-hand table of `Vols w new metric vs PET.xlsx` (`Sheet2` columns `X:AP`, used for the PLOS ONE `T/M > 1.5` / `T/M > 1.75` stratification).
   - For `Day 1 Set 3`, the subfolder `2nd try/` (`ef5_day1_sec3.dcm`, `ef5_day1_sec3_roi.ROI`, `ef5_day1_sec3_bg_roi.ROI`) contains the final co-registered PET volume and contours that match `Individual Tumors/Set 1` and `10 PET and CT Tumor ROI stats.xlsx` to 4 decimal places.
   - Where separate tighter `PET ROI` and `CT ROI` contours were saved on disk in `Individual Tumors/Set 1..2` (14 mice from Day 1 Sets 1–4), we also include their 3D binary masks (`mask_lhs_hotspot_pet`, `mask_rhs_hotspot_pet`) inside the `.npz` voxel bundles and cropped `rois.ROI`.
   - In `imaging_summary.csv` and `tumour_master_wide.csv`, we report **both**:
     - Direct 3D volume-computed full-bed contour metrics (`fullbed_roi_pet_vol_cm3`, `fullbed_roi_pet_mean_idg`, `fullbed_roi_tm_mean`, `fullbed_roi_ct_vol_cm3`, `fullbed_roi_ct_mean_hu`), and
     - Reconciled report/spreadsheet PET-hotspot & CT-ROI metrics (`report_pet_hotspot_tm`, `report_pet_hotspot_vol_cm3`, `report_ct_roi_tm`, `report_hypoxic_fraction`, `plos_sheet2_combo_tm`).
2. **Caliper Volume Formulas:**
   - Raw orthogonal dimensions $(x, y, z\\text{ in mm})$ are preserved for all 880 rows in `caliper_growth_curves.csv`, alongside:
     - `vol_05xy2_mm3`: $V = 0.5 \\, x \\, y^2$ (exact formula in `Vols w new metric vs PET.xlsx` `Sheet1`/`Sheet2` and *Technical Report* §4).
     - `vol_05LW2_mm3`: $V = 0.5 \\, L \\, W^2$ where $L = \\max(x, y), W = \\min(x, y)$.
     - `vol_ellipsoid_xyz_mm3`: $V = \\frac{\\pi}{6} x y z$ (3-axis ellipsoid volume from `10 Tumor Sizes.xlsx`).
     - `vol_v4_43pi_xy2_mm3`: $V = \\frac{4}{3}\\pi (x/2)(y/2)^2$ (formula in `Tumor Sizes - v4.xlsb`).
     - `equiv_diameter_mm`: Spherical-equivalent diameter $d_{\\text{eq}} = (6 V_{0.5xy^2} / \\pi)^{1/3}$.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    git_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    git_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main end-to-end execution
# ---------------------------------------------------------------------------
def main() -> None:
    root = Path("/workspace")
    set5_dir = root / "images" / "set5"
    analysed_dir = set5_dir / "Analysed Data"
    indiv_dir = set5_dir / "Individual Tumors"
    rep_dir = set5_dir / "reports"

    out_root = root / "outputs" / "second_ef5_rt_study"
    bed_ov_root = out_root / "bed_overviews"
    cropped_root = out_root / "cropped_mice"
    voxel_root = out_root / "voxel_bundles"
    out_tables = out_root / "tables"
    git_results = root / "results" / "second_ef5_rt_study"

    for d in [bed_ov_root, cropped_root, voxel_root, out_tables, git_results]:
        d.mkdir(parents=True, exist_ok=True)

    print("1. Loading reconciled spreadsheet metadata & 11-timepoint caliper growth curves...")
    meta_map = load_reconciled_spreadsheet_metadata(rep_dir)
    caliper_df, timeline_df = extract_calipers_and_timeline(
        indiv_dir / "Tumor Sizes - v4.xlsb", meta_map
    )
    print(f"   Loaded metadata for {len(meta_map)} tumours, {len(caliper_df)} caliper rows, and {len(timeline_df)} timeline events.")

    # Group BED_MOUSE_SPECS by scanner bed so each 291 MB CT is loaded only once
    specs_by_bed: Dict[str, List[BedMouseSpec]] = {}
    for sp in BED_MOUSE_SPECS:
        specs_by_bed.setdefault(sp.scan_set_label, []).append(sp)

    def _load_roi_from_ref(sdir: Path, ref_str: str) -> Tuple[str, List[RTRoi]]:
        if ref_str.startswith("Individual Tumors:"):
            rel = ref_str.split(":", 1)[1]
            return load_rt_image_roi(indiv_dir / rel)
        return load_rt_image_roi(sdir / ref_str)

    imaging_by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    mouse_scan_info: Dict[str, Dict[str, Any]] = {}

    print(f"\n2. Processing all 10 scanner beds ({len(BED_MOUSE_SPECS)} imaged mice)...")
    for set_label, bed_specs in specs_by_bed.items():
        sp0 = bed_specs[0]
        sdir = analysed_dir / sp0.analysed_rel_dir
        pet_full = load_rt_image_dicom(sdir / sp0.pet_dcm_rel)
        ct_full = load_inveon_volume(sdir / sp0.ct_img_rel)
        pet_hdr = parse_inveon_hdr(sdir / sp0.pet_hdr_rel)
        scan_timestamp = pet_hdr.get("scan_time", "")

        print(f"\n   === {set_label} ({scan_timestamp}) | PET={pet_full.shape} CT={ct_full.shape} ===")

        # Cache ROI files for this bed
        roi_cache: Dict[str, Tuple[str, Dict[str, RTRoi]]] = {}
        for sp in bed_specs:
            for rf_ref in (sp.body_roi_file, sp.tumor_roi_file, sp.bg_roi_file):
                if rf_ref not in roi_cache:
                    ver, rlist = _load_roi_from_ref(sdir, rf_ref)
                    roi_cache[rf_ref] = (ver, {r.name: r for r in rlist})

        bed_overview_rois: List[RTRoi] = []
        for sp in bed_specs:
            roi_ver, body_map = roi_cache[sp.body_roi_file]
            _, tumor_map = roi_cache[sp.tumor_roi_file]
            _, bg_map = roi_cache[sp.bg_roi_file]

            body_roi = body_map[sp.body_roi_name]
            lhs_roi = tumor_map[sp.lhs_roi_name] if sp.lhs_roi_name else None
            rhs_roi = tumor_map[sp.rhs_roi_name] if sp.rhs_roi_name else None
            muscle_roi = bg_map[sp.bg_roi_name]

            for r in (body_roi, lhs_roi, rhs_roi, muscle_roi):
                if r is not None and r not in bed_overview_rois:
                    bed_overview_rois.append(r)

            # Also load Individual Tumors PET-hotspot ROIs if available on disk
            lhs_hot_roi: Optional[RTRoi] = None
            rhs_hot_roi: Optional[RTRoi] = None
            if sp.indiv_tumors_rel_dir is not None:
                _, ind_rois = load_rt_image_roi(indiv_dir / sp.indiv_tumors_rel_dir / "rois.ROI")
                ind_map = {r.name: r for r in ind_rois}
                if sp.indiv_lhs_pet_roi and sp.indiv_lhs_pet_roi in ind_map:
                    lhs_hot_roi = ind_map[sp.indiv_lhs_pet_roi]
                if sp.indiv_rhs_pet_roi and sp.indiv_rhs_pet_roi in ind_map:
                    rhs_hot_roi = ind_map[sp.indiv_rhs_pet_roi]
                # Special case: c2_M1L1R in Day 1 Set 3 had tiny placeholder contours in 2nd try/ef5_day1_sec3_roi.ROI
                # while its actual evaluated contours are in Individual Tumors/Set 1/C2 M1L1R/rois.ROI
                if sp.mouse_id == "c2_M1L1R" and lhs_hot_roi is not None and rhs_hot_roi is not None:
                    lhs_roi = lhs_hot_roi
                    rhs_roi = rhs_hot_roi

            sub_rois = [r for r in (lhs_roi, rhs_roi, muscle_roi) if r is not None]
            for hr in (lhs_hot_roi, rhs_hot_roi):
                if hr is not None and hr not in sub_rois:
                    sub_rois.append(hr)

            quad = _quadrant(body_roi, pet_full)

            # Crop PET and CT to union of mouse body box + tumour/muscle ROIs (+ 2.0 mm pad)
            pet_crop = crop_volume_to_body_roi(pet_full, body_roi, extra_rois=sub_rois, pad_mm=2.0)
            ct_crop = crop_volume_to_body_roi(ct_full, body_roi, ref_pet_vol=pet_crop, extra_rois=sub_rois, pad_mm=2.0)

            mdir = cropped_root / sp.mouse_id
            pet_out = save_rt_image_dicom(pet_crop, mdir / f"{sp.mouse_id}_PET.dcm")
            ct_out = save_rt_image_dicom(ct_crop, mdir / f"{sp.mouse_id}_CT.dcm")
            roi_out = save_rt_image_roi_subset(roi_ver, sub_rois, mdir / "rois.ROI")

            # Compute 3D ROI statistics directly from cropped volumes
            pet_musc = _vol_stats(pet_crop, muscle_roi)
            ct_musc = _vol_stats(ct_crop, muscle_roi)
            pet_lhs = _vol_stats(pet_crop, lhs_roi)
            pet_rhs = _vol_stats(pet_crop, rhs_roi)
            ct_lhs = _vol_stats(ct_crop, lhs_roi)
            ct_rhs = _vol_stats(ct_crop, rhs_roi)

            musc_mean = pet_musc["mean"]
            tm_lhs = round(pet_lhs["mean"] / musc_mean, 4) if (pet_lhs["points"] > 0 and musc_mean > 0) else np.nan
            tm_rhs = round(pet_rhs["mean"] / musc_mean, 4) if (pet_rhs["points"] > 0 and musc_mean > 0) else np.nan

            # Render per-mouse 2x3 QC overview PNG
            qc_out = mdir / f"{sp.mouse_id}_qc.png"
            render_mouse_qc_overview(
                qc_out, sp, pet_crop, ct_crop, lhs_roi, rhs_roi, muscle_roi, tm_lhs, tm_rhs,
                pet_hotspot_rois=[r for r in (lhs_hot_roi, rhs_hot_roi) if r is not None and r not in (lhs_roi, rhs_roi)],
            )

            # Save simulator-ready .npz voxel bundle
            npz_out = save_second_study_voxel_bundle_npz(
                voxel_root / sp.mouse_id / f"{sp.mouse_id}_pre_rt.npz",
                spec=sp,
                pet_crop=pet_crop,
                ct_crop=ct_crop,
                lhs_roi=lhs_roi,
                rhs_roi=rhs_roi,
                muscle_roi=muscle_roi,
                muscle_mean_idg=float(pet_musc["mean"]),
                muscle_std_idg=float(pet_musc["std"]),
                lhs_pet_hotspot_roi=lhs_hot_roi,
                rhs_pet_hotspot_roi=rhs_hot_roi,
            )

            mouse_scan_info[sp.mouse_id] = {
                "imaged": True,
                "scan_date": sp.scan_date,
                "scan_timestamp": scan_timestamp,
                "scan_set": sp.scan_set_label,
                "bed_quadrant": quad,
                "body_roi_label": body_roi.name,
                "muscle_roi_label": muscle_roi.name,
                "cropped_pet_path": pet_out.relative_to(root).as_posix(),
                "cropped_ct_path": ct_out.relative_to(root).as_posix(),
                "cropped_roi_path": roi_out.relative_to(root).as_posix(),
                "voxel_bundle_path": npz_out.relative_to(root).as_posix(),
                "nifti_dir": f"outputs/second_ef5_rt_study/nifti/{sp.mouse_id}",
                "qc_image_path": qc_out.relative_to(root).as_posix(),
                "notes": sp.notes,
            }

            for side, roi_obj, p_st, c_st, tm_val in [
                ("L", lhs_roi, pet_lhs, ct_lhs, tm_lhs),
                ("R", rhs_roi, pet_rhs, ct_rhs, tm_rhs),
            ]:
                imaging_by_key[(sp.cage_num, sp.mouse_num, side)] = {
                    "imaged": True,
                    "scan_date": sp.scan_date,
                    "scan_timestamp": scan_timestamp,
                    "days_post_injection": sp.days_post_injection,
                    "days_post_rt": sp.days_post_rt,
                    "scan_set": sp.scan_set_label,
                    "bed_quadrant": quad,
                    "fullbed_roi_label": roi_obj.name if roi_obj is not None else "",
                    "fullbed_roi_pet_points": p_st["points"],
                    "fullbed_roi_pet_vol_cm3": p_st["volume_cm3"],
                    "fullbed_roi_pet_min_idg": p_st["min"],
                    "fullbed_roi_pet_max_idg": p_st["max"],
                    "fullbed_roi_pet_mean_idg": p_st["mean"],
                    "fullbed_roi_pet_std_idg": p_st["std"],
                    "fullbed_roi_pet_median_idg": p_st["p50"],
                    "fullbed_roi_pet_p90_idg": p_st["p90"],
                    "fullbed_roi_pet_p95_idg": p_st["p95"],
                    "fullbed_roi_pet_sum_idg": p_st["sum"],
                    "fullbed_muscle_roi_label": muscle_roi.name,
                    "fullbed_muscle_vol_cm3": pet_musc["volume_cm3"],
                    "fullbed_muscle_mean_idg": pet_musc["mean"],
                    "fullbed_muscle_std_idg": pet_musc["std"],
                    "fullbed_roi_tm_mean": tm_val,
                    "fullbed_roi_tm_max": round(p_st["max"] / musc_mean, 4) if (p_st["points"] > 0 and musc_mean > 0) else np.nan,
                    "fullbed_roi_tm_p95": round(p_st["p95"] / musc_mean, 4) if (p_st["points"] > 0 and musc_mean > 0) else np.nan,
                    "ct_available": True,
                    "ct_spacing_mm": round(float(ct_crop.spacing[0]), 4),
                    "fullbed_roi_ct_points": c_st["points"],
                    "fullbed_roi_ct_vol_cm3": c_st["volume_cm3"],
                    "fullbed_roi_ct_mean_hu": c_st["mean"],
                    "fullbed_roi_ct_std_hu": c_st["std"],
                    "fullbed_muscle_ct_mean_hu": ct_musc["mean"],
                    "cropped_pet_path": pet_out.relative_to(root).as_posix(),
                    "cropped_ct_path": ct_out.relative_to(root).as_posix(),
                    "cropped_roi_path": roi_out.relative_to(root).as_posix(),
                    "voxel_bundle_path": npz_out.relative_to(root).as_posix(),
                    "nifti_dir": f"outputs/second_ef5_rt_study/nifti/{sp.mouse_id}",
                    "qc_image_path": qc_out.relative_to(root).as_posix(),
                }

            print(
                f"     -> {sp.mouse_id:8s} ({quad:12s}): "
                f"LHS={pet_lhs['volume_cm3']:.3f}cc (T/M {tm_lhs:.2f}), "
                f"RHS={pet_rhs['volume_cm3']:.3f}cc (T/M {tm_rhs:.2f})"
            )

        # Render full-bed overview PNG
        ov_out = bed_ov_root / f"{sp0.scan_set_slug}_overview.png"
        render_bed_overview(ov_out, set_label, pet_full, ct_full, bed_specs, bed_overview_rois)

    # 3. Assemble complete tables covering all 40 mice / 80 tumours
    print("\n3. Assembling curated summary CSV tables across all 40 mice (80 tumours)...")
    mouse_mapping_rows: List[Dict[str, Any]] = []
    imaging_summary_rows: List[Dict[str, Any]] = []
    wide_rows: List[Dict[str, Any]] = []

    for cnum in range(1, 9):
        reg = CAGE_REGIMENS[cnum]
        for mnum in range(1, 6):
            ear = EAR_PUNCH_ORDER[mnum - 1]
            mid = f"c{cnum}_M{ear}"
            m_scan = mouse_scan_info.get(mid, {"imaged": False})
            meta_L = meta_map[(cnum, mnum, "L")]
            meta_R = meta_map[(cnum, mnum, "R")]

            # Determine survival / endpoint status from caliper curve at Day 79/84/99
            c_sub_m = caliper_df[caliper_df["mouse_id"] == mid]
            d99_max = float(c_sub_m[c_sub_m["days_post_injection"] == 99]["vol_05xy2_mm3"].max())
            d79_max = float(c_sub_m[c_sub_m["days_post_injection"] == 79]["vol_05xy2_mm3"].max())
            d73_max = float(c_sub_m[c_sub_m["days_post_injection"] == 73]["vol_05xy2_mm3"].max())
            if d99_max > 0:
                surv_status = "Survived to Day 99 endpoint (2011-07-26)"
            elif d79_max > 0:
                surv_status = "Measured through Day 79–84 (July 2011); sacrificed/excised before Day 99"
            elif d73_max > 0:
                surv_status = "Measured through Day 73 (2011-06-30); sacrificed/died before Day 79"
            else:
                surv_status = "Complete regression or died before Day 73"

            mouse_mapping_rows.append(
                {
                    "mouse_id": mid,
                    "cage": f"A549 c{cnum}",
                    "cage_num": cnum,
                    "ear_punch": ear,
                    "mouse_num": mnum,
                    "plos_mouse_code": f"c{cnum}_M{mnum}",
                    "treatment": reg["treatment"],
                    "total_dose_gy": reg["total_dose_gy"],
                    "num_fractions": reg["num_fractions"],
                    "fraction_dose_gy": reg["fraction_dose_gy"],
                    "rt_start_date": reg["rt_start_date"],
                    "imaged_ef5_pet_ct": bool(m_scan["imaged"]),
                    "scan_date": m_scan.get("scan_date", ""),
                    "scan_timestamp": m_scan.get("scan_timestamp", ""),
                    "scan_set": m_scan.get("scan_set", "Not imaged"),
                    "bed_quadrant": m_scan.get("bed_quadrant", ""),
                    "dose_at_scan_uci": meta_L["dose_at_scan_uci"],
                    "time_to_image_hr": meta_L["time_to_image_hr"],
                    "roi_author": meta_L["roi_author"],
                    "lhs_orientation": "Image-left (larger X) = anatomical Left flank",
                    "rhs_orientation": "Image-right (smaller X) = anatomical Right flank",
                    "survival_status": surv_status,
                    "khalila_lhs_note": meta_L["khalila_note_2011_07_20"],
                    "khalila_rhs_note": meta_R["khalila_note_2011_07_20"],
                    "notes": m_scan.get("notes", "") or ("Not imaged (no visible EF5 signal)" if not m_scan["imaged"] else ""),
                }
            )

            for side in ("L", "R"):
                key = (cnum, mnum, side)
                meta = meta_map[key]
                img = imaging_by_key.get(key, {})
                c_sub = caliper_df[(caliper_df["mouse_id"] == mid) & (caliper_df["side"] == side)].set_index("days_post_injection")
                pre_cal_vol = float(c_sub.loc[43, "vol_05xy2_mm3"]) if 43 in c_sub.index else np.nan
                pre_cal_eq_d = float(c_sub.loc[43, "equiv_diameter_mm"]) if 43 in c_sub.index else np.nan

                imaging_summary_rows.append(
                    {
                        "tumour_id": meta["tumour_id"],
                        "plos_tumour_code": meta["plos_tumour_code"],
                        "mouse_id": mid,
                        "cage": f"A549 c{cnum}",
                        "cage_num": cnum,
                        "ear_punch": ear,
                        "mouse_num": mnum,
                        "side": side,
                        "treatment": reg["treatment"],
                        "total_dose_gy": reg["total_dose_gy"],
                        "num_fractions": reg["num_fractions"],
                        "fraction_dose_gy": reg["fraction_dose_gy"],
                        "imaged": bool(img.get("imaged", False)),
                        "scan_date": img.get("scan_date", ""),
                        "scan_timestamp": img.get("scan_timestamp", ""),
                        "days_post_injection": img.get("days_post_injection", np.nan),
                        "days_post_rt": img.get("days_post_rt", np.nan),
                        "scan_set": img.get("scan_set", "Not imaged"),
                        "bed_quadrant": img.get("bed_quadrant", ""),
                        "dose_at_scan_uci": meta["dose_at_scan_uci"],
                        "time_to_image_hr": meta["time_to_image_hr"],
                        "hi_contrast": meta["hi_contrast"],
                        "blue_score": meta["blue_score"],
                        "fullbed_roi_label": img.get("fullbed_roi_label", ""),
                        "fullbed_roi_pet_points": img.get("fullbed_roi_pet_points", np.nan),
                        "fullbed_roi_pet_vol_cm3": img.get("fullbed_roi_pet_vol_cm3", np.nan),
                        "fullbed_roi_pet_min_idg": img.get("fullbed_roi_pet_min_idg", np.nan),
                        "fullbed_roi_pet_max_idg": img.get("fullbed_roi_pet_max_idg", np.nan),
                        "fullbed_roi_pet_mean_idg": img.get("fullbed_roi_pet_mean_idg", np.nan),
                        "fullbed_roi_pet_std_idg": img.get("fullbed_roi_pet_std_idg", np.nan),
                        "fullbed_roi_pet_median_idg": img.get("fullbed_roi_pet_median_idg", np.nan),
                        "fullbed_roi_pet_p90_idg": img.get("fullbed_roi_pet_p90_idg", np.nan),
                        "fullbed_roi_pet_p95_idg": img.get("fullbed_roi_pet_p95_idg", np.nan),
                        "fullbed_roi_pet_sum_idg": img.get("fullbed_roi_pet_sum_idg", np.nan),
                        "fullbed_muscle_roi_label": img.get("fullbed_muscle_roi_label", ""),
                        "fullbed_muscle_vol_cm3": img.get("fullbed_muscle_vol_cm3", np.nan),
                        "fullbed_muscle_mean_idg": img.get("fullbed_muscle_mean_idg", np.nan),
                        "fullbed_muscle_std_idg": img.get("fullbed_muscle_std_idg", np.nan),
                        "fullbed_roi_tm_mean": img.get("fullbed_roi_tm_mean", np.nan),
                        "fullbed_roi_tm_max": img.get("fullbed_roi_tm_max", np.nan),
                        "fullbed_roi_tm_p95": img.get("fullbed_roi_tm_p95", np.nan),
                        "report_pet_hotspot_vol_cm3": meta["report_pet_hotspot_vol_cm3"],
                        "report_pet_hotspot_mean_idg": meta["report_pet_hotspot_mean_idg"],
                        "report_pet_hotspot_max_idg": meta["report_pet_hotspot_max_idg"],
                        "report_pet_hotspot_tm": meta["report_pet_hotspot_tm"],
                        "report_ct_roi_vol_cm3": meta["report_ct_roi_vol_cm3"],
                        "report_ct_roi_mean_idg": meta["report_ct_roi_mean_idg"],
                        "report_ct_roi_tm": meta["report_ct_roi_tm"],
                        "report_muscle_mean_idg": meta["report_muscle_mean_idg"],
                        "report_hypoxic_fraction": meta["report_hypoxic_fraction"],
                        "report_ks_d": meta["report_ks_d"],
                        "plos_sheet2_combo_tm": meta["plos_sheet2_combo_tm"],
                        "ct_available": bool(img.get("ct_available", False)),
                        "ct_spacing_mm": img.get("ct_spacing_mm", np.nan),
                        "fullbed_roi_ct_points": img.get("fullbed_roi_ct_points", np.nan),
                        "fullbed_roi_ct_vol_cm3": img.get("fullbed_roi_ct_vol_cm3", np.nan),
                        "fullbed_roi_ct_mean_hu": img.get("fullbed_roi_ct_mean_hu", np.nan),
                        "fullbed_roi_ct_std_hu": img.get("fullbed_roi_ct_std_hu", np.nan),
                        "fullbed_muscle_ct_mean_hu": img.get("fullbed_muscle_ct_mean_hu", np.nan),
                        "pre_rt_caliper_date": "2011-05-31",
                        "pre_rt_caliper_vol_05xy2_mm3": round(pre_cal_vol, 4) if not np.isnan(pre_cal_vol) else np.nan,
                        "pre_rt_caliper_equiv_diam_mm": round(pre_cal_eq_d, 4) if not np.isnan(pre_cal_eq_d) else np.nan,
                        "roi_author": meta["roi_author"],
                        "khalila_note_2011_07_20": meta["khalila_note_2011_07_20"],
                        "notes": meta["spreadsheet_notes"] or m_scan.get("notes", ""),
                        "cropped_pet_path": img.get("cropped_pet_path", ""),
                        "cropped_ct_path": img.get("cropped_ct_path", ""),
                        "cropped_roi_path": img.get("cropped_roi_path", ""),
                        "voxel_bundle_path": img.get("voxel_bundle_path", ""),
                        "nifti_dir": img.get("nifti_dir", ""),
                        "qc_image_path": img.get("qc_image_path", ""),
                    }
                )

                w_row: Dict[str, Any] = {
                    "tumour_id": meta["tumour_id"],
                    "plos_tumour_code": meta["plos_tumour_code"],
                    "mouse_id": mid,
                    "cage": f"A549 c{cnum}",
                    "cage_num": cnum,
                    "ear_punch": ear,
                    "mouse_num": mnum,
                    "side": side,
                    "treatment": reg["treatment"],
                    "total_dose_gy": reg["total_dose_gy"],
                    "num_fractions": reg["num_fractions"],
                    "fraction_dose_gy": reg["fraction_dose_gy"],
                    "imaged": bool(img.get("imaged", False)),
                    "scan_date": img.get("scan_date", ""),
                    "scan_set": img.get("scan_set", "Not imaged"),
                    "dose_at_scan_uci": meta["dose_at_scan_uci"],
                    "time_to_image_hr": meta["time_to_image_hr"],
                    "blue_score": meta["blue_score"],
                    "hi_contrast": meta["hi_contrast"],
                    "fullbed_pet_vol_cm3": img.get("fullbed_roi_pet_vol_cm3", np.nan),
                    "fullbed_ct_vol_cm3": img.get("fullbed_roi_ct_vol_cm3", np.nan),
                    "fullbed_pet_mean_idg": img.get("fullbed_roi_pet_mean_idg", np.nan),
                    "fullbed_pet_max_idg": img.get("fullbed_roi_pet_max_idg", np.nan),
                    "fullbed_muscle_mean_idg": img.get("fullbed_muscle_mean_idg", np.nan),
                    "fullbed_tm_mean": img.get("fullbed_roi_tm_mean", np.nan),
                    "fullbed_tm_p95": img.get("fullbed_roi_tm_p95", np.nan),
                    "report_pet_hotspot_vol_cm3": meta["report_pet_hotspot_vol_cm3"],
                    "report_pet_hotspot_mean_idg": meta["report_pet_hotspot_mean_idg"],
                    "report_pet_hotspot_tm": meta["report_pet_hotspot_tm"],
                    "report_ct_roi_vol_cm3": meta["report_ct_roi_vol_cm3"],
                    "report_ct_roi_tm": meta["report_ct_roi_tm"],
                    "report_hypoxic_fraction": meta["report_hypoxic_fraction"],
                    "plos_sheet2_combo_tm": meta["plos_sheet2_combo_tm"],
                }
                for dpi, nom_d in NOMINAL_DAYS_POST_RT.items():
                    lbl = f"cal_d{nom_d:+d}_vol_05xy2_mm3".replace("+", "p").replace("-", "m")
                    w_row[lbl] = round(float(c_sub.loc[dpi, "vol_05xy2_mm3"]), 2) if dpi in c_sub.index else np.nan
                for dpi, nom_d in NOMINAL_DAYS_POST_RT.items():
                    if nom_d < -3:
                        continue
                    lbl = f"norm_d{nom_d:+d}_vol_05xy2".replace("+", "p").replace("-", "m")
                    val = c_sub.loc[dpi, "norm_vol_05xy2_vs_day_minus3"] if dpi in c_sub.index else np.nan
                    w_row[lbl] = round(float(val), 4) if not pd.isna(val) else np.nan

                w_row["survival_status"] = surv_status
                w_row["khalila_note_2011_07_20"] = meta["khalila_note_2011_07_20"]
                wide_rows.append(w_row)

    mapping_df = pd.DataFrame(mouse_mapping_rows)
    imaging_df = pd.DataFrame(imaging_summary_rows)
    wide_df = pd.DataFrame(wide_rows)

    table_map = {
        "mouse_id_mapping.csv": mapping_df,
        "imaging_summary.csv": imaging_df,
        "caliper_growth_curves.csv": caliper_df,
        "tumour_master_wide.csv": wide_df,
        "study_timeline.csv": timeline_df,
    }
    for fname, df in table_map.items():
        df.to_csv(out_tables / fname, index=False)
        df.to_csv(git_results / fname, index=False)

    write_data_audit_notes(out_tables / "data_audit_notes.md", git_results / "data_audit_notes.md")

    print("\n4. Summary of saved tables:")
    for fname, df in table_map.items():
        print(f"   - {fname:35s} ({len(df):3d} rows x {len(df.columns):2d} cols)")

    export_nifti_dataset_second_study(root)
    verify_second_study_outputs(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Second EF5 RT Study (40-mouse cohort) curation & export pipeline.")
    parser.add_argument("--verify-only", action="store_true", help="Run NIfTI export and verification only.")
    args = parser.parse_args()
    if args.verify_only:
        export_nifti_dataset_second_study()
        verify_second_study_outputs()
    else:
        main()
