# Second EF5 RT Study (May–July 2011, 40-Mouse Cohort) — Data Audit & Reconciliation Notes

## 1. Study Overview & Cohort Design
- **Cohort:** 40 immunocompromised male Balb/c nude mice (8 cages $\times$ 5 ear-punch mice: `0, 1L, 1R, 1L1R, 2R`), implanted bilaterally on the left (`L` / `LHS`) and right (`R` / `RHS`) shoulder flanks with $10^6$ human A549 lung adenocarcinoma cells on **2011-04-18** ($n = 80$ tumours total).
- **Radiotherapy Regimens (by Cage):** Both tumours on a given mouse received the same treatment regimen using a Philips RT-250 250 kVp orthovoltage X-ray unit with a lead shield exposing the tumours through circular apertures:
  - **Cage 1 (Group 1):** $1 \times 10\text{ Gy}$ (10 Gy total, single fraction on 2011-06-02)
  - **Cage 2 (Group 2):** $1 \times 20\text{ Gy}$ (20 Gy total, single fraction on 2011-06-02)
  - **Cage 3 (Group 3):** $1 \times 40\text{ Gy}$ (40 Gy total, single fraction on 2011-06-02)
  - **Cage 4 (Group 4):** $\text{Control } 0\text{ Gy}$ (0 Gy, unirradiated negative control)
  - **Cage 5 (Group 5):** $2 \times 5\text{ Gy}$ (10 Gy total, 2 fractions: 2011-06-03 and 2011-06-07)
  - **Cage 6 (Group 6):** $2 \times 10\text{ Gy}$ (20 Gy total, 2 fractions: 2011-06-03 and 2011-06-07)
  - **Cage 7 (Group 7):** $4 \times 5\text{ Gy}$ (20 Gy total, 4 fractions: 2011-06-03 and 2011-06-07..09)
  - **Cage 8 (Group 8):** $4 \times 10\text{ Gy}$ (40 Gy total, 4 fractions: 2011-06-03 and 2011-06-07..09)
- **Publication:** Ali R, Apte S, Vilalta M, et al. (2015) *18F-EF5 PET Is Predictive of Response to Fractionated Radiotherapy in Preclinical Tumor Models*, **PLOS ONE** 10(10): e0139425.

---

## 2. Reconstructed Experimental Timeline & Date Reconciliation
- **Excel 1904 vs. 1900 Date Offset:** Several historical workbooks (`10 Tumor Sizes.xlsx`, `Tumor Sizes - v4.xlsb`, `Vols w new metric vs PET.xlsx`) were saved across Mac Excel (1904 date system) and Windows Excel (1900 date system), shifting serial numbers by $\pm 1,462$ days (displaying May–July 2011 dates as May–July 2007 or May–July 2013).
- **Exact Anchor Dates:**
  - **Implantation (Day 0):** Monday **2011-04-18** (explicitly stated in *Technical Report v1.2* §4: *"The tumors injected on Mon 18th April"*).
  - **PET/CT Imaging Day 1 (`i100117`, Day 44 post-injection):** Wednesday **2011-06-01**, 21:05–23:37 (`Day 1 Set 1..5`, 19 mice from Cages 1–4; verified from raw Siemens Inveon `.pet.img.hdr` `scan_time` headers). Mouse `c4_M1R` was not imaged.
  - **PET/CT Imaging Day 2 (`i100118`, Day 45 post-injection):** Thursday **2011-06-02**, 21:08–23:31 (`Day 2 Set 1..5`, 20 mice from Cages 5–8; verified from `i100118` `.pet.img.hdr` headers).
  - **Radiotherapy Delivery & Missing Lead Shield Gap:** Single fractions (Cages 1–3) were delivered on Thursday **2011-06-02** (the day after Day 1 imaging). Fraction #1 for fractionated Cages 5–8 was delivered on Friday **2011-06-03** (the day after Day 2 imaging). The lead shield went missing over the weekend (June 4–6); a replacement shield was found on Tuesday **2011-06-07**, resulting in a **4-day gap** between Fraction #1 and Fraction #2.
  - **11 Longitudinal Caliper Timepoints:** All 11 caliper measurement dates (Days 23, 30, 35, 43 [pre-RT baseline], 49, 56, 64, 73, 79, 84, and 99 post-injection, spanning **2011-05-11 to 2011-07-26**) are captured in `caliper_growth_curves.csv` and `tumour_master_wide.csv`. The first 8 timepoints (Days 23–73, i.e., Day -23 to Day +27 post-RT) form the uniform window plotted in `Vols w new metric vs PET.xlsx` (`in_plos_one_window = True`).

---

## 3. Forensic Resolution of Left vs. Right (`LHS` vs. `RHS`) Orientation
- **Physical Scanner Bed Geometry:** In all 10 Siemens Inveon PET/CT beds, mice were positioned prone, head-first in a $2 \times 2$ arrangement (or 3 mice in `Day 1 Set 1`). In RT_Image physical coordinates (`X` in `[-99, 0] mm`, where transaxial slices are displayed with larger `X` on the left and smaller `X` on the right):
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
   - Raw orthogonal dimensions $(x, y, z\text{ in mm})$ are preserved for all 880 rows in `caliper_growth_curves.csv`, alongside:
     - `vol_05xy2_mm3`: $V = 0.5 \, x \, y^2$ (exact formula in `Vols w new metric vs PET.xlsx` `Sheet1`/`Sheet2` and *Technical Report* §4).
     - `vol_05LW2_mm3`: $V = 0.5 \, L \, W^2$ where $L = \max(x, y), W = \min(x, y)$.
     - `vol_ellipsoid_xyz_mm3`: $V = \frac{\pi}{6} x y z$ (3-axis ellipsoid volume from `10 Tumor Sizes.xlsx`).
     - `vol_v4_43pi_xy2_mm3`: $V = \frac{4}{3}\pi (x/2)(y/2)^2$ (formula in `Tumor Sizes - v4.xlsb`).
     - `equiv_diameter_mm`: Spherical-equivalent diameter $d_{\text{eq}} = (6 V_{0.5xy^2} / \pi)^{1/3}$.
