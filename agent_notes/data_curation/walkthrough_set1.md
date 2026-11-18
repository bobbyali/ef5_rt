# Walkthrough & Analysis Report: Set 1 (`c1_M0`) EF5 PET/CT & RT_Image ROIs

## 1. Environment & Git Setup
- **Git Repository**: Initialized on branch `main` in `/home/bobby/science/ef5_data` with [.gitignore](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/.gitignore) excluding `images/`, Python caches, and macOS/AFP resource forks (`*:AFP_Resource`).
- **Docker + JupyterLab**:
  - [Dockerfile](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/Dockerfile) and [docker-compose.yml](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/docker-compose.yml) build `ef5-analysis:latest` (`python:3.12-slim`) with `pydicom`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `pandas`, `SimpleITK`, `nibabel`, `jupyterlab`, and `ipywidgets`.
  - To start **JupyterLab** at any time from WSL/Windows:
    ```bash
    docker compose up -d
    ```
    Then open `http://localhost:8888/lab` in your browser and open [notebooks/explore_set1.ipynb](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/notebooks/explore_set1.ipynb).

---

## 2. RT_Image File Format Details & Coordinate Verification
We implemented a reusable parser in [src/rt_image_loader.py](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/src/rt_image_loader.py) that handles both RT_Image's exported DICOM volumes and its binary `.ROI` format (`version 3.10   07/09/2008`), and verified our voxel rasterization against RT_Image's own exported statistics file ([day2_set5_stats.txt](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/images/set2/day2_set5_stats.txt)), achieving an **exact match** across all metrics (`Points`, `Volume`, `Min`, `Max`, `Mean`, `Std Dev`, `Sum`).

### Set 1 Volume Geometry (`c1_M0`)
| Modality | File | Grid Shape `(Planes, Rows, Cols)` | Voxel Spacing `(dz, dy, dx)` (mm) | Origin `[ox, oy, oz]` (mm) | Units |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PET (`PT`)** | `c1_M0_PET_mouse.dcm` | `(101, 32, 41)` | `(0.7960, 0.7764, 0.7764)` | `[-48.1357, -24.6760, -64.4398]` | `%IDG` |
| **CT (`CT`)** | `c1_M0_CT_mouse.dcm` | `(388, 119, 152)` | `(0.2068, 0.2068, 0.2068)` | `[-47.9667, -24.6036, -64.3002]` | `HU` |

---

## 3. Quantitative ROI Summary Statistics (`set1`)
Full tables are saved in:
- [outputs/set1/set1_roi_stats.csv](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set1/set1_roi_stats.csv) (with Tumour-to-Background Ratios `TBR_mean` and `TBR_max`)
- [outputs/set1/set1_roi_stats.txt](file:////wsl.localhost/Ubuntu/home/bobby/science/ef5_data/outputs/set1/set1_roi_stats.txt) (RT_Image tab-separated format)

| ROI Label | Modality | Units | Points | Volume ($\text{cm}^3$) | Minimum | Maximum | Mean | Std Dev | Sum | $\text{TBR}_{\text{mean}}$ | $\text{TBR}_{\text{max}}$ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **c1_M0_L PET ROI** | PT | `%IDG` | 568 | 0.2725 | 0.0081 | 2.0601 | **0.7366** | 0.3599 | 0.20076 | **2.676** | **7.483** |
| **c1_M0_R PET ROI** | PT | `%IDG` | 200 | 0.0960 | 0.0182 | 1.5825 | **0.6117** | 0.2950 | 0.05870 | **2.222** | **5.748** |
| **M0 bg ROI** | PT | `%IDG` | 250 | 0.1200 | 0.0101 | 1.1818 | **0.2753** | 0.1820 | 0.03303 | 1.000 | 4.293 |
| **c1_M0_R CT ROI** | PT | `%IDG` | 407 | 0.1953 | 0.0000 | 1.5825 | **0.3143** | 0.3163 | 0.06137 | **1.142** | **5.748** |
| **c1_M0_L CT ROI** | PT | `%IDG` | 971 | 0.4659 | 0.0020 | 2.0601 | **0.4734** | 0.3800 | 0.22056 | **1.720** | **7.483** |
| **c1_M0_L PET ROI** | CT | `HU` | 30858 | 0.2727 | -1255.02 | 271.03 | -239.46 | 370.74 | -65.308 | — | — |
| **c1_M0_R PET ROI** | CT | `HU` | 10744 | 0.0950 | -1223.98 | 228.98 | -307.13 | 388.17 | -29.163 | — | — |
| **M0 bg ROI** | CT | `HU` | 13170 | 0.1164 | -84.03 | 1543.96 | 241.00 | 250.38 | 28.051 | — | — |
| **c1_M0_R CT ROI** | CT | `HU` | 22168 | 0.1959 | -1144.03 | 216.04 | -110.54 | 192.28 | -21.657 | — | — |
| **c1_M0_L CT ROI** | CT | `HU` | 52816 | 0.4668 | -1103.00 | 311.04 | -55.55 | 166.69 | -25.929 | — | — |

---

## 4. Generated Visualizations (`outputs/set1/`)

### A. Same-Size Scrollable Images (`1200 × 960 px`, Locked Physical Axes)
Located in `outputs/set1/scrollable/{transaxial,coronal,sagittal}/`.
Every image is rendered at the exact same pixel dimensions and physical coordinate bounding box so you can open any folder in Windows Explorer / Photos and use the Left/Right arrow keys to flip between **1_CT**, **2_PET**, and **3_Overlay** (and across slices) with zero jumping:

- **Transaxial (`outputs/set1/scrollable/transaxial/`)**:
  - `Y = -59.70 mm` (`slice01_Y-59.7mm_Right_Tumour_and_BG_{1_CT,2_PET,3_Overlay}.png`)
  - `Y = -58.11 mm` (`slice02_Y-58.1mm_All_5_ROIs_Core_{1_CT,2_PET,3_Overlay}.png`)
  - `Y = -56.52 mm` (`slice03_Y-56.5mm_Both_Tumours_Mid_{1_CT,2_PET,3_Overlay}.png`)
  - `Y = -54.92 mm` (`slice04_Y-54.9mm_Left_Tumour_Core_{1_CT,2_PET,3_Overlay}.png`)
  - `Y = -52.54 mm` (`slice05_Y-52.5mm_Left_Tumour_Cranial_{1_CT,2_PET,3_Overlay}.png`)
- **Coronal (`outputs/set1/scrollable/coronal/`)**:
  - `Z = -74.96 mm` (`slice01_Z-75.0mm_Right_Tumour_and_BG_Plane_{1_CT,2_PET,3_Overlay}.png`)
  - `Z = -73.40 mm` (`slice02_Z-73.4mm_Both_Left_and_Right_Tumours_{1_CT,2_PET,3_Overlay}.png`)
  - `Z = -71.80 mm` (`slice03_Z-71.8mm_Left_Tumour_Mid_Plane_{1_CT,2_PET,3_Overlay}.png`)
- **Sagittal (`outputs/set1/scrollable/sagittal/`)**:
  - `X = -56.00 mm` (`slice01_X-56.0mm_Left_Tumour_Sagittal_{1_CT,2_PET,3_Overlay}.png`)
  - `X = -62.85 mm` (`slice02_X-62.9mm_Background_ROI_Sagittal_{1_CT,2_PET,3_Overlay}.png`)
  - `X = -73.00 mm` (`slice03_X-73.0mm_Right_Tumour_Sagittal_{1_CT,2_PET,3_Overlay}.png`)

### B. 3-Panel Side-by-Side Comparisons (`outputs/set1/panels/`)
Each file in `outputs/set1/panels/` displays **CT + ROIs | EF5 PET + ROIs | Combined PET/CT Overlay + ROIs** side-by-side for each sample slice across all three planes.
