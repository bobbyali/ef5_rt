# Walkthrough & Analysis Report: Set 2 (`day2_set5`) 4-Mouse EF5 PET/CT & Cropped Mice

## 1. Summary of What Was Done
1. **Siemens Inveon Raw `.img`/`.hdr` Loader & Bounding-Box Cropper**:
   - Extended [src/rt_image_loader.py](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/src/rt_image_loader.py) with `load_inveon_volume()`, `load_volume_auto()`, `crop_volume_to_body_roi()`, `save_rt_image_dicom()`, and `save_rt_image_roi_subset()`.
   - Validated the full-bed Set 2 PET (`day2_set5_pet.dcm`) and raw Inveon CT (`i100118_v1.ct.img` + `.hdr`, `632 × 480 × 480`, `291.2 MB`) against RT_Image's exported [day2_set5_stats.txt](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/images/set2/day2_set5_stats.txt), matching all 28 PET and CT ROI rows (`Points`, `Volume`, `Min`, `Max`, `Mean`, `Std Dev`, `Sum`).
2. **Cropped All 4 Mice to Disk**:
   - Used [src/process_set2.py](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/src/process_set2.py) to crop both the PET and CT volumes to each mouse's 3D `body` bounding-box ROI (`c8 M0 body`, `c8 M1R body`, `c8 M1L body`, `c8 M1L1R body`) and saved each mouse's standalone `<mouse>_PET_mouse.dcm`, `<mouse>_CT_mouse.dcm`, and `rois.ROI` (matching Set 1's exact format) under `outputs/set2/cropped_mice/`:
     - `outputs/set2/cropped_mice/c8_M0/` (`PET: 101×40×42`, `CT: 387×151×162`)
     - `outputs/set2/cropped_mice/c8_M1R/` (`PET: 101×40×42`, `CT: 387×148×157`)
     - `outputs/set2/cropped_mice/c8_M1L/` (`PET: 101×36×41`, `CT: 387×134×155`)
     - `outputs/set2/cropped_mice/c8_M1L1R/` (`PET: 101×37×38`, `CT: 387×138×142`)
3. **Generated Scrollable 3-Axis Slices for Each Cropped Mouse**:
   - Reused `run_slice_generator()` from [src/ef5_slice_cli.py](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/src/ef5_slice_cli.py) on all 4 cropped mice, saving scrollable `CT`, `PET`, `Overlay`, and 3-panel `Panel` images across `transaxial`, `coronal`, and `sagittal` planes under `outputs/set2/mice_slices/{c8_M0,c8_M1R,c8_M1L,c8_M1L1R}/`.

---

## 2. Quantitative ROI Statistics Across All 4 Cropped Mice in Set 2
Saved to [outputs/set2/set2_cropped_mice_roi_stats.csv](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set2/set2_cropped_mice_roi_stats.csv) and full-bed [outputs/set2/set2_full_roi_stats.csv](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set2/set2_full_roi_stats.csv):

| Mouse ID | ROI Label | Modality | Units | Points | Volume ($\text{cm}^3$) | Minimum | Maximum | Mean | Std Dev | Sum | $\text{TBR}_{\text{mean}}$ | $\text{TBR}_{\text{max}}$ |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **c8_M0** | `c8 M0 LHS Tumor` | PT | `%IDG` | 141 | 0.0677 | 0.0010 | 1.7936 | **0.4811** | 0.4116 | 0.03255 | **1.612** | **6.009** |
| **c8_M0** | `c8 M0 RHS Tumor` | PT | `%IDG` | 156 | 0.0748 | 0.0000 | 1.4181 | **0.3137** | 0.3257 | 0.02348 | **1.051** | **4.751** |
| **c8_M0** | `c8 M0 bg ROI` | PT | `%IDG` | 325 | 0.1559 | 0.0010 | 1.6751 | **0.2985** | 0.2711 | 0.04654 | 1.000 | 5.612 |
| **c8_M1R** | `c8 M1R LHS tumor` | PT | `%IDG` | 252 | 0.1209 | 0.0000 | 1.1981 | **0.2189** | 0.2712 | 0.02646 | **1.190** | **6.511** |
| **c8_M1R** | `c8 M1R RHS Tumor` | PT | `%IDG` | 498 | 0.2389 | 0.0000 | 1.7244 | **0.2532** | 0.2805 | 0.06050 | **1.376** | **9.372** |
| **c8_M1R** | `c8 M1R bg ROI` | PT | `%IDG` | 283 | 0.1358 | 0.0000 | 1.8196 | **0.1840** | 0.2318 | 0.02498 | 1.000 | 9.889 |
| **c8_M1L** | `c8 M1L LHS Tumor` | PT | `%IDG` | 535 | 0.2567 | 0.0063 | 2.6745 | **0.6389** | 0.5036 | 0.16399 | **3.608** | **15.102** |
| **c8_M1L** | `c8 M1L RHS Tumor` | PT | `%IDG` | 611 | 0.2932 | 0.0000 | 2.3714 | **0.4929** | 0.4874 | 0.14451 | **2.783** | **13.390** |
| **c8_M1L** | `c8 M1L bg ROI` | PT | `%IDG` | 320 | 0.1535 | 0.0013 | 1.0329 | **0.1771** | 0.1952 | 0.02719 | 1.000 | 5.832 |
| **c8_M1L1R** | `c8 M1L1R LHS Tumor` | PT | `%IDG` | 646 | 0.3100 | 0.0000 | 1.9709 | **0.2676** | 0.3495 | 0.08295 | **1.710** | **12.594** |
| **c8_M1L1R** | `c8 M1L1R RHS Tumor` | PT | `%IDG` | 488 | 0.2341 | 0.0000 | 3.0313 | **0.4990** | 0.4938 | 0.11684 | **3.188** | **19.369** |
| **c8_M1L1R** | `c8 M1L1R bg ROI` | PT | `%IDG` | 447 | 0.2145 | 0.0000 | 1.2515 | **0.1565** | 0.2110 | 0.03357 | 1.000 | 7.997 |

---

## 3. Key Output Directories for Set 2
- **Full-Bed 4-Mouse Overviews**: `outputs/set2/full_bed_overview/`
  - [full_bed_transaxial_Y-55.7mm_All_4_Mice_Bed.png](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set2/full_bed_overview/full_bed_transaxial_Y-55.7mm_All_4_Mice_Bed.png)
  - [full_bed_coronal_Z-60.1mm_Bottom_Two_Mice_M0_M1R.png](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set2/full_bed_overview/full_bed_coronal_Z-60.1mm_Bottom_Two_Mice_M0_M1R.png)
  - [full_bed_coronal_Z-31.3mm_Top_Two_Mice_M1L_M1L1R.png](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set2/full_bed_overview/full_bed_coronal_Z-31.3mm_Top_Two_Mice_M1L_M1L1R.png)
- **Cropped Single-Mouse DICOMs & `.ROI` Files**: `outputs/set2/cropped_mice/{c8_M0,c8_M1R,c8_M1L,c8_M1L1R}/`
- **Per-Mouse Scrollable 3-Axis Slices & Panels**: `outputs/set2/mice_slices/{c8_M0,c8_M1R,c8_M1L,c8_M1L1R}/`
