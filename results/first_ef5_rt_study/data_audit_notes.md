# First EF5 RT Study (A549 Subcutaneous Tumours, Nov 2010 – Jan 2011) — Data Audit & Reconciliation Notes

## 1. Study Design & Cohort Summary
- **Cell line**: Human lung adenocarcinoma **A549** (`1 × 10^6` cells injected subcutaneously on both left and right shoulders of male Balb/c nude mice on **2010-10-29**; A549+Luc, RKO, Hongbin HNC, and Fred mice imaged alongside were excluded).
- **Cohort**: 6 A549 mice entered the RT study on **2010-12-02** (all 5 mice in Cage 2: `c2_M0`, `c2_M1L`, `c2_M1R`, `c2_M1L1R`, `c2_M2R`, plus `c1_M2R`, the sole Cage 1 survivor after 4 Cage 1 mice died from cage flooding on 2010-11-27).
- **Radiation Therapy (RT)**: Single **20 Gy** dose delivered on **2010-12-03** (Philips RT-250) to one shoulder tumour per mouse (body and contralateral control tumour shielded with lead).
- **EF5 PET/CT Imaging Timepoints** (`~200 µCi [18F]-EF5`, 3 h uptake, 10 min static scan on Siemens/Concorde microPET R4 + Gamma Medica microCT):
  1. **Day -15 (`2010-11-18`, Early Growth Baseline, 20 days post-inoculation)**: 4 Cage 2 mice imaged in a single 2×2 bed (`c2_M0` [`BR (1)`], `c2_M1L` [`TL (2)`], `c2_M1R` [`TR (3)`], `c2_M1L1R` [`BL (4)`]). Bottom row (`c2_M0`, `c2_M1L1R`) has co-registered microCT; top row (`c2_M1L`, `c2_M1R`) tumours lie outside the CT FOV (PET-only).
  2. **Day -1 (`2010-12-02`, Pre-RT)**: All 6 mice (`Set 1`: `c2_M0`, `c2_M1L`, `c2_M1R`, `c2_M1L1R` [PET+CT]; `Set 2`: `c2_M2R`, `c1_M2R` [PET only]).
  3. **Day +13 (`2010-12-15`, Post-RT)**: 5 surviving mice (`c1_M2R` died on 2010-12-12). `Set 2`: `c2_M1L`, `c2_M1R` [PET+CT]; `Set 3`: `c2_M0`, `c2_M1L1R`, `c2_M2R` [PET+CT].
  4. **Day +32 (`2011-01-03`, Post-RT)**: 4 surviving mice (`c2_M0` died between 2010-12-30 and 2011-01-03). `Set 8`: `c2_M1L`, `c2_M1L1R` [PET only]; `Set 9`: `c2_M1R`, `c2_M2R` [PET+CT].
- **Endpoint**: Remaining 4 mice (`c2_M1L`, `c2_M1R`, `c2_M1L1R`, `c2_M2R`) sacrificed on **2011-01-12** (Day +40 post-RT).

---

## 2. Mouse ID Reconciliation (`m1`–`m6` ↔ Ear-Punch Labels ↔ 18 Nov Scan)
Cross-verified via (a) `08 EF5 RT PET and Knottin data.xlsx` (`EF5 RT data` tab), (b) `08 Analysis of EF5 Uptake in A549 RKO Tumors.xlsx` (`Values from Diff Expts` tab: `A549 M1–M4 A` on 18 Nov vs `A549 M1–M6 B` on 02 Dec), (c) `Pre and Post RT EF5 Analysis.xlsx` & `Pre and Post RT EF5 Scans.pptx`, (d) MD5 byte-for-byte cross-matching of `hist_m{2,3,4,5}_*.txt` vs `03 Jan/{008,009}/hist_a549*.txt`, and (e) independent cross-checks on 18 Nov RKO `(2)=1L, (4)=1L1R, (5)=2R` and A549 Cage 2 tumour-side asymmetry (`c2_M0` symmetric; `c2_M1L` RHS/Ctrl dominant; `c2_M1R` LHS/RT dominant, ruling out Cage 1 `1R` which had RHS > LHS; `c2_M1L1R` RHS/Ctrl PET vol > LHS/RT PET vol):

| Canonical ID | 2010 ROI ID | Cage | Ear Punch | Day -15 (18 Nov) | Day -1 (02 Dec) | Day +13 (15 Dec) | Day +32 (03 Jan) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `c2_M0` | `m1` | A549 c2 | `0` | `BR (1)` (PET+CT, RT+Ctrl) | Set 1 (PET+CT) | Set 3 (PET+CT) | *(died before 03 Jan)* |
| `c2_M1L` | `m2` | A549 c2 | `1L` | `TL (2)` (PET only, Ctrl only) | Set 1 (PET+CT) | Set 2 (PET+CT) | Set 8 (PET only) |
| `c2_M1R` | `m3` | A549 c2 | `1R` | `TR (3)` (PET only, RT only) | Set 1 (PET+CT) | Set 2 (PET+CT) | Set 9 (PET+CT) |
| `c2_M1L1R` | `m4` | A549 c2 | `1L1R` | `BL (4)` (PET+CT*, RT+Ctrl) | Set 1 (PET+CT) | Set 3 (PET+CT) | Set 8 (PET only) |
| `c2_M2R` | `m5` | A549 c2 | `2R` | *(not imaged)* | Set 2 (PET only) | Set 3 (PET+CT) | Set 9 (PET+CT) |
| `c1_M2R` | `m6` | A549 c1 | `2R` | *(not imaged)* | Set 2 (PET only) | *(died 12 Dec)* | *(died 12 Dec)* |

*\*On 18 Nov, `c2_M1L1R` (`BL (4)`) RT tumour (`x ∈ [-32.7, -25.8] mm`) sits at the lateral edge of the microCT FOV (`x_max = -26.3 mm`, ~31% CT coverage; PET is 100% intact).*

---

## 3. Critical Legacy Discrepancies Identified & Fixed

1. **2010 vs. 2013 `LHS` / `RHS` Naming Flip (Image-Left vs. Anatomical-Right)**:
   - In the **2010–2011 files** (`mouse_tumor_rois.roi`, `08 EF5 IR Study Growth Curves i.xlsx`, `08 EF5 RT PET and Knottin data.xlsx`, and `08a Project Report.docx` Fig. 13 "The left-side tumors were irradiated"), the **irradiated (20 Gy)** tumour is called **`LHS`** (left side of the transaxial/dorsal image, larger physical `X`) and the **unirradiated control** tumour is called **`RHS`** (right side of the image, smaller physical `X`).
   - In the **2013 re-labelled RT_Image sessions** (`15. EF5 PET/Data/.../rois.ROI` and `Pre and Post RT EF5 Analysis.xlsx`), the exact same physical contours (verified by identical voxel counts, volumes, centroids, and intensity means) were renamed to **`RHS RT`** (anatomical right) for the irradiated tumour and **`LHS Control`** (anatomical left) for the control tumour.
   - **Resolution**: All curated tables in this repository key each tumour unambiguously by `treatment` (`RT_20Gy` vs `Control_0Gy`) and preserve both `caliper_side_2010` (`LHS`/`RHS`) and `roi_label_2013` (`RHS RT`/`LHS Control`).

2. **Formula Error on `c2_M1L1R` (`m4`) Pre-RT `T/M` in `08 EF5 RT PET and Knottin data.xlsx`**:
   - In `EF5 RT data` row 8, `m4 LHS tumor` (mean `0.5227 %IDG`) was accidentally divided by `m3 muscle` (`0.2927`) yielding `T/M = 1.7858`, instead of `m4 muscle` (`0.3212`), which gives the true **`T/M = 1.6273`** (matching `Pre and Post RT EF5 Analysis.xlsx` row 41 and our direct recomputation from the PET volume).

3. **Swapped Post-RT `c2_M1R` Values in `08 EF5 IR Study Growth Curves i.xlsx` (`A549 Analysis` tab)**:
   - Rows 8–9 of `A549 Analysis` put Day +32 `%ID/g = 1.1585 (T/M 1.583)` on `1R LHS` and `1.0927 (T/M 1.493)` on `1R RHS`, swapped relative to the actual `03 Jan Set 9` ROI contours (`A549c2 1R RHS RT` / 2010 `LHS` = `1.0927 %IDG`, `0.534 cc`, `T/M 1.493`; `A549c2 1R LHS Control` / 2010 `RHS` = `1.1585 %IDG`, `0.763 cc`, `T/M 1.583`). Fixed here directly from the 3D ROI masks.

4. **Swapped Pre-RT `c1_M2R` (`m6`) Values in `08 EF5 IR Study Growth Curves i.xlsx` (`A549 Analysis` tab)**:
   - Rows 2–3 of `A549 Analysis` put `0.4656 %IDG (T/M 1.531)` on `c1 2R LHS` and `0.5932 %IDG (T/M 1.950)` on `c1 2R RHS`, swapped relative to the actual `02 Dec Set 2` ROI contours (`c1 M2R RHS RT` / 2010 `m6 LHS` = `0.5932 %IDG`, `T/M 1.950`; `c1 M2R LHS Control` / 2010 `m6 RHS` = `0.4656 %IDG`, `T/M 1.531`). Fixed here directly from the 3D ROI masks.

5. **Rotated Rigidly-Registered MicroCTs on `15 Dec` (`Set 2` and `Set 3`)**:
   - The `512 × 512 × 512` (`0.17 mm` isotropic) microCT volumes in `2010-12-15/set2/image2.dcm` and `set3/image2.dcm` carry non-axis-aligned `ImageOrientationPatient` matrices (in-plane `X–Z` rotations of `~18.0°` and `~9.2°`, respectively).
   - `src/rt_image_loader.py` (`crop_volume_to_body_roi`) was upgraded to read `ImageOrientationPatient` and resample rotated CTs onto the canonical axis-aligned grid (`orientation = [[-1, 0, 0], [0, 0, -1]]`) during single-mouse cropping. Verified against `mouse_set_2a.txt` ground-truth RT_Image CT HU means (`1038.8` vs `1040.7 HU`, `930.7` vs `932.4 HU`, `1031.6` vs `1031.0 HU`).

6. **Timeline Year Typo in `08 EF5 IR Study Growth Curves i.xlsx`**:
   - Rows 20–25 of `Tumor Sizes` entered Dec 15–30 2010 dates with year `2011` (`2011-12-15` to `2011-12-30`). Corrected to `2010-12-15` through `2010-12-30` in `study_timeline.csv` with an `audit_note`.

7. **Note on Higher Muscle `%IDG` on Day +32 (`03 Jan`)**:
   - On Day +32 (`Sets 8 & 9`), A549 muscle background uptake is `0.73–1.17 %IDG` (vs `0.16–0.37 %IDG` on Day -15, `0.29–0.45 %IDG` on Day -1, and `0.49–0.78 %IDG` on Day +13), which depresses Day +32 `T/M` ratios even when raw tumour `%IDG` rises. Both raw `%IDG` (mean, max, p90, p95) and muscle-normalized `T/M` ratios are provided in all tables for modelling flexibility.

8. **18 Nov (`Day -15`) PET Calibration Discrepancy (`p105421_v2.img.dcm` vs `a549_reg_norm0001.dcm`)**:
   - `a549_reg_norm0001.dcm` and `p105421/p105421_v2.img.dcm` have byte-for-byte identical raw `int16` pixel data, origin, spacing, and orientation, but `a549_reg_norm0001.dcm` was inadvertently divided by `1290.500` (`RescaleSlope = 1.212e-6` vs `0.0015644`).
   - **Resolution**: The pipeline loads `p105421_v2.img.dcm`, which reproduces all voxel counts, volumes, means, and maxima in `stats1.txt` and `08 Analysis of EF5 Uptake in A549 RKO Tumors.xlsx` to 4 decimal places.

9. **18 Nov (`Day -15`) CT FOV Coverage & Rotated Orientation (`a549_reg_norm0002.dcm`)**:
   - The `256 × 256 × 256` (`0.20 mm` isotropic) microCT carries a `~3.6°` 3D rotation (resampled to canonical axis-aligned grid during cropping) and spans world `x ∈ [-83.5, -26.3]`, `y ∈ [-50.0, 4.1]`, `z ∈ [-90.4, -36.1] mm`.
   - Bottom-row mice (`c2_M0` [`BR (1)`] and `c2_M1L1R` [`BL (4)`]) lie inside the CT FOV (with `c2_M1L1R` RT tumour touching the `x = -26.3 mm` edge at ~31% CT coverage; PET is 100% intact). Top-row mice (`c2_M1L` [`TL (2)`] and `c2_M1R` [`TR (3)`]) have their shoulders/tumours at `z ∈ [-39.4, -29.2] mm`, outside `z_max = -36.1 mm`, and are therefore exported as PET-only (`ct_available = False`).

10. **18 Nov (`Day -15`) Audit Fixes vs `08 Analysis of EF5 Uptake in A549 RKO Tumors.xlsx`**:
    - In `08 Analysis`, the `Vol` column for `A549 M2L A / M3R A` (`0.1229, 0.1030`) and `M4L A / M4R A` (`0.1671, 0.1662`) was accidentally swapped relative to `stats1.txt` (true volumes: `c2_M1L Ctrl = 0.1671 cc`, `c2_M1R RT = 0.1662 cc`, `c2_M1L1R Ctrl = 0.1229 cc`, `c2_M1L1R RT = 0.1030 cc`). Fixed directly from the 3D ROI masks.
    - Also note that `08 Analysis` used anatomical `L`/`R` for 18 Nov (`A`) rows (opposite of 2010 `LHS`/`RHS` in `stats1.txt`), whereas `02 Dec` (`B`) rows used 2010 `LHS`/`RHS`. Here, all 4 timepoints are mapped consistently by physical coordinate (`larger X = 2010 LHS = 2013 RHS RT = RT_20Gy`; `smaller X = 2010 RHS = 2013 LHS Control = Control_0Gy`), verified from identical `+Y` cranial / `+X` image-left orientation across 18 Nov and 02 Dec.

---

## 4. Simulator-Ready Voxel Bundles (`.npz`)

Each of the **19 mouse-timepoint scans** (4 Day -15 + 6 Day -1 + 5 Day +13 + 4 Day +32) is exported to `outputs/first_ef5_rt_study/voxel_bundles/<mouse_id>/<mouse_id>_<timepoint>.npz` (linked via the `voxel_bundle_path` column in `longitudinal_imaging_summary.csv`):

| Key | Dtype / Shape | Description |
| :--- | :--- | :--- |
| `pet_idg` | `float32 (nz, ny, nx)` | Cropped EF5 PET uptake (`%ID/g`), canonical axis order `(Y, Z, X)` |
| `pet_tm` | `float32 (nz, ny, nx)` | Muscle-normalized EF5 `T/M` ratio map (`pet_idg / muscle_mean_idg`) |
| `has_rt_roi`, `has_ctrl_roi` | `bool` | Whether RT / Control tumour ROI was contoured (`False` for uncontoured contralateral side on Day -15 `c2_M1L` RT and `c2_M1R` Ctrl) |
| `mask_rt_pet`, `mask_ctrl_pet`, `mask_muscle_pet` | `bool (nz, ny, nx)` | Binary 3D masks on the PET voxel grid (all-`False` when `has_*_roi` is `False`) |
| `pet_spacing_mm`, `pet_origin_mm` | `float64 (3,)` | PET voxel spacing `(dz, dy, dx)` and physical origin `(ox, oy, oz)` in mm |
| `pet_x_mm`, `pet_y_mm`, `pet_z_mm` | `float64` | Physical coordinate vectors for each PET axis |
| `ct_available` | `bool` | Whether a co-registered CT covers the tumours (13 of 19 scans) |
| `ct_on_pet_grid` | `float32 (nz, ny, nx)` | *(if CT available)* Canonical CT resampled directly onto the PET voxel grid |
| `ct_hu` | `float32 (nz_ct, ny_ct, nx_ct)` | *(if CT available)* Native-resolution cropped canonical axis-aligned CT |
| `mask_rt_ct`, `mask_ctrl_ct`, `mask_muscle_ct` | `bool (nz_ct, ny_ct, nx_ct)` | *(if CT available)* Binary 3D masks on the native CT grid |
| `ct_spacing_mm`, `ct_origin_mm`, `ct_x_mm`, `ct_y_mm`, `ct_z_mm` | `float64` | *(if CT available)* CT voxel geometry in mm |
| Metadata scalars | `str` / `int32` / `float32` | `mouse_id`, `roi_id_2010`, `cage`, `ear_punch`, `scan_date`, `days_post_injection`, `days_post_rt`, `scan_set`, `muscle_mean_idg`, `muscle_std_idg`, `rt_roi_label`, `ctrl_roi_label`, `muscle_roi_label`, `ct_note` |

```python
import numpy as np

d = np.load("outputs/first_ef5_rt_study/voxel_bundles/c2_M1R/c2_M1R_day13_2010-12-15.npz")
rt_voxels_idg = d["pet_idg"][d["mask_rt_pet"]]
rt_voxels_tm = d["pet_tm"][d["mask_rt_pet"]]
if d["ct_available"]:
    rt_voxels_ct = d["ct_on_pet_grid"][d["mask_rt_pet"]]
```

---

## 5. Publication-Ready NIfTI Outputs (`.nii.gz`)

All 19 mouse-timepoint scans are also exported as compressed NIfTI-1 volumes (`.nii.gz`) under `outputs/first_ef5_rt_study/nifti/<mouse_id>/<timepoint>/` (linked via the `nifti_dir` column in `longitudinal_imaging_summary.csv`), accompanied by `outputs/first_ef5_rt_study/nifti/label_legend.json` and per-scan `*_metadata.json` sidecars (~424 MB total):

- **Coordinate system**: Standard positive-diagonal **NIfTI RAS+** (`nibabel.aff2axcodes == ('R', 'A', 'S')`), converted from DICOM LPS via `diag(-1, -1, 1, 1)` so PET, native CT, `CT_on_pet`, and ROI label maps auto-overlay in **3D Slicer**, **ITK-SNAP**, **FSLeyes**, and Python (`nibabel` / `SimpleITK`) with `≤ 0.11 mm` PET–CT physical centroid agreement.
- **Files per scan**:
  - `*_PET_idg.nii.gz` (`float32`, `%ID/g`)
  - `*_PET_tm.nii.gz` (`float32`, muscle-normalized `T/M`)
  - `*_seg_pet.nii.gz` (`uint8` multi-label mask: `1 = RT_20Gy`, `2 = Control_0Gy`, `3 = Muscle_BG`)
  - *(for the 13 scans with CT)* `*_CT.nii.gz` (`float32`, native resolution), `*_CT_on_pet.nii.gz` (`float32`, resampled onto PET grid), and `*_seg_ct.nii.gz` (`uint8`, native CT grid)
  - `*_metadata.json` (BIDS-style sidecar with mouse ID, scan timing, units, muscle background stats, and audit notes)
- Regenerate NIfTIs alone from existing `.npz` bundles via `python -m src.process_first_ef5_rt_study --nifti-only`.

