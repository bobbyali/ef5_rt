# EF5 PET/CT Data Curation & Radiotherapy Hypoxia Modelling (`ef5_rt`)

Python + Docker scientific toolkit for:
1. **Preclinical EF5 PET/CT & RT_Image Data Curation** — loading, auditing, cropping, reconciling, and exporting longitudinal subcutaneous A549 mouse tumour **[¹⁸F]-EF5 PET/CT** imaging datasets, 3D **RT_Image** `.ROI` contours ([NITRC](https://www.nitrc.org/projects/rt_image)), caliper growth curves, `.npz` voxel bundles, and RAS+ NIfTI (`.nii.gz`) volumes.
2. **Mathematical Modelling of Tumour Growth, Hypoxia & Radiotherapy Response** — linked **1D coupled oxygen-PDE spheroid** and **3D CT/EF5-PET image-initialised voxel continuum** models validated against both our longitudinal bilateral A549 cohort (*First EF5 RT Study*) and our published PLOS ONE single-fraction / fractionated radiotherapy study ([Ali et al., PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139425)).

---

## Repository Structure

```text
.
├── Dockerfile                          # Python 3.12-slim scientific imaging + JupyterLab image
├── docker-compose.yml                  # Container service mounting workspace & exposing JupyterLab (:8888)
├── requirements.txt                    # Python dependencies (pydicom, nibabel, numpy, scipy, pandas, matplotlib, etc.)
├── src/
│   ├── rt_image_loader.py              # Unified loader for RT_Image .dcm/.ROI & raw Siemens Inveon .img/.hdr
│   ├── ef5_slice_cli.py                # Configurable CLI & Python API for 3-axis scrollable PET/CT + ROI slices
│   ├── inspect_set1.py                 # Set 1 (c1_M0) inspection, ROI stats, and sample slices
│   ├── process_set2.py                 # Set 2 (4-mouse bed) verification, body-ROI cropping, & per-mouse slicing
│   ├── inspect_set3_rois.py            # Diagnostic inspector for First EF5 RT Study raw .ROI structures
│   ├── inspect_study_records.py        # Audit script reconciling scanner logs, Excel sheets, & caliper tables
│   ├── process_first_ef5_rt_study.py   # First EF5 RT Study curation -> CSVs, .npz voxel bundles, & RAS+ NIfTIs
│   ├── process_second_ef5_rt_study.py  # Second EF5 RT Study (40-mouse PLOS ONE) curation -> CSVs, .npz, & NIfTIs
│   └── modeling/
│       ├── physics.py                  # Shared biophysics: AHF OER, LQ survival, Darcy/stress saturation, EF5 PET
│       ├── legacy_spheroid.py          # 2013 MATLAB ODE audit, Alex Fletcher CodeForBobby port, Byrne Greenspan
│       ├── spheroid_rt_model.py        # Sub-Goal 1: 1D coupled spherical O2-PDE + moving-boundary RT simulator
│       ├── run_spheroid_study.py       # PLOS ONE regimen sweep (0, 1x10, 1x20, 1x40, 2x10, 4x5, 4x10 Gy)
│       ├── voxel_initializer.py        # 3D CT geometry + EF5 PET hypoxia initialiser & vascular field anchor
│       ├── voxel_rt_model.py           # Sub-Goal 2: 3D O2-PDE + Darcy pressure-velocity continuum RT simulator
│       ├── run_voxel_study.py          # 10-tumour bilateral longitudinal study + Day -15 early-growth check
│       ├── generate_visual_report.py   # Generates 1D & 3D visual comparison figures & syncs reports/figures/
│       ├── run_study2_validation.py    # Stages 1 & 2: Blind 80-tumour 1D/3D validation, 4d shield gap, & pre-RT growth
│       ├── sensitivity_identifiability.py # Stage 3: Sobol global sensitivity, Fisher Information, & profile likelihood
│       ├── hierarchical_bayesian.py    # Stage 4: Hierarchical Bayesian MCMC, 8-fold LORO CV, & vascular phenotypes
│       └── generate_study2_report.py   # Generates 7 multi-panel figures & Study 2 validation technical report
├── tests/
│   ├── test_modeling.py                # Automated physics, Byrne analytical, conservation & regression tests (8/8)
│   └── test_study2_validation.py       # Automated tests for Study 2 4-stage validation, 3D bundles, & random effects (4/4)
├── results/
│   ├── first_ef5_rt_study/             # Reconciled CSV tables (calipers, imaging stats, ID map, timeline, audit)
│   ├── second_ef5_rt_study/            # Second EF5 RT Study (40-mouse PLOS ONE) CSV tables & forensic audit
│   ├── ef5_rt_model/                   # Tabular 1D & 3D simulation vs real-data validation summaries
│   └── study2_validation/              # Tabular Stage 1-4 metrics, Sobol indices, FIM, MCMC posteriors, & LORO CV
├── outputs/
│   ├── first_ef5_rt_study/             # Tracked longitudinal montages, per-scan QC PNGs, & CSV tables
│   ├── second_ef5_rt_study/            # Tracked 10 bed overviews, 39 per-mouse QC PNGs, & CSV tables
│   ├── ef5_rt_model/                   # Tracked 1D/3D figures & visual comparison report (NIfTIs excluded)
│   ├── study2_validation/              # Tracked 7 validation figures & 4-stage technical report w/ layperson appendix
│   └── prelim_work/                    # Local Set 1 & Set 2 exploratory slice outputs (gitignored)
├── notebooks/
│   ├── explore_set1.ipynb              # Interactive 3-plane slice explorer for Set 1
│   ├── run_slice_generator.ipynb       # Guide for configuring & running ef5_slice_cli
│   └── explore_ef5_rt_models.ipynb     # Interactive explorer for 1D spheroid & 3D voxel RT models
├── agent_notes/
│   ├── data_curation/                  # Plans & walkthroughs for Set 1, Set 2, and First EF5 RT Study curation
│   └── modelling/                      # Implementation plan & walkthrough for 1D/3D RT modelling
└── .gitignore                          # Excludes raw images/ & large outputs/ while tracking reports/
```

*(Large raw scanner volumes under `images/`, exploratory slice stacks under `outputs/prelim_work/`, and heavy binary outputs such as `.npz` bundles, `.dcm`, and `.nii.gz` NIfTI volumes are excluded via `.gitignore`. Lightweight `.png` figures, `.csv` tables, and markdown reports under `results/`, `outputs/first_ef5_rt_study/`, and `outputs/ef5_rt_model/` are tracked in git.)*

---

## Quick Start (Docker + WSL2)

### 1. Build the Container & Launch JupyterLab
```bash
docker compose build
docker compose up -d
# Open http://localhost:8888/lab for interactive notebooks
```

### 2. Data Curation Pipelines
```bash
# Set 1 & Set 2 inspection / 4-mouse bed cropping
docker compose run --rm ef5 python -m src.inspect_set1
docker compose run --rm ef5 python -m src.process_set2

# First EF5 RT Study: reconcile longitudinal PET/CT + calipers -> CSVs, .npz bundles & RAS+ NIfTIs
docker compose run --rm ef5 python -m src.process_first_ef5_rt_study

# Second EF5 RT Study (40-mouse PLOS ONE cohort): crop 10 Inveon beds -> 39 mice, 11 caliper dates, .npz & RAS+ NIfTIs
docker compose run --rm ef5 python -m src.process_second_ef5_rt_study
```

### 3. Mathematical Modelling, Report Generation & Verification
```bash
# Automated physics, Byrne analytical Greenspan, & 3D conservation/regression tests (8 tests)
docker compose run --rm ef5 python -m unittest discover -s tests -v

# Sub-Goal 1: 1D spheroid validation & PLOS ONE regimen sweep
docker compose run --rm ef5 python -m src.modeling.run_spheroid_study

# Sub-Goal 2: 3D CT/EF5-PET image-initialised study across all 5 bilateral mice + NIfTI export
docker compose run --rm ef5 python -m src.modeling.run_voxel_study

# Regenerate the 1D & 3D visual comparison report figures (synced to outputs/ef5_rt_model/reports/figures/)
docker compose run --rm ef5 python -m src.modeling.generate_visual_report

# Second EF5 RT Study (80-tumour PLOS ONE cohort): run all 4 validation stages & generate technical report
docker compose run --rm ef5 python -m src.modeling.generate_study2_report
```

---

## Part 1: Dataset Curation & Reconciliation

### Unified Raw Scanner & RT_Image Binary Loader (`src/rt_image_loader.py`)
`load_volume_auto(path)` automatically detects and loads both **RT_Image exports** and **raw Siemens Inveon scanner files**:

| File Format | Origin | Loader Function | How It Is Handled |
| :--- | :--- | :--- | :--- |
| **`.dcm`** | **RT_Image Export** | `load_rt_image_dicom()` | Reads headerless Implicit VR Little Endian DICOMs (`force=True`), sanitises float-formatted `NumberOfFrames`, rescales `int16` pixels to `%ID/g` or `HU`, and preserves 3D physical `ImagePositionPatient` origins. |
| **`.img` + `.hdr`** | **Raw Siemens Inveon** | `load_inveon_volume()` | Parses ASCII `.hdr` dimensions, voxel spacing, data type, and scale factor; memory-maps `.img` and aligns axis orientation to RT_Image physical coordinates. |
| **`.ROI`** | **RT_Image Archive (`v3.10`)** | `load_rt_image_roi()` | Parses binary `freehand`, `ellipse`, and `rectangle` 3D contours across arbitrary orthogonal planes (`Y`-normal transaxial tumours, `X`-normal background ellipses, `body` cropping boxes). |

### Curated Cohorts & Datasets
1. **Set 1 (`c1_M0`) & Set 2 (`day2_set5`, 4-mouse bed)**:
   - Verified 100% against reference `day2_set5_stats.txt` (all CT and PET rows).
   - `src/process_set2.py` crops the 4-mouse bed via 80 mm `body` ROIs into standalone per-mouse DICOM + `.ROI` triplets under `outputs/prelim_work/set2/` fed into the shared `src/ef5_slice_cli.py` 3-axis visualiser (`src/inspect_set1.py` writes Set 1 outputs under `outputs/prelim_work/set1/`).
2. **First EF5 RT Study — Longitudinal Bilateral A549 Cohort (`src/process_first_ef5_rt_study.py`)**:
   - Curated and reconciled **5 longitudinal bilateral mice** (`c2_M1R`, `c2_M0`, `c2_M1L`, `c2_M1L1R`, `c2_M2R`; 10 tumours total: one `20 Gy` irradiated + one `0 Gy` contralateral control per mouse) across **4 imaging dates**:
     - `Day -15` (`2010-11-18`): Untreated early-growth baseline scan (integrated for early growth-rate validation).
     - `Day -1` (`2010-12-02`): Pre-RT baseline PET/CT + CT/PET tumour ROIs + muscle background ROIs.
     - `Day +13` (`2010-12-15`) & `Day +32` (`2011-01-03`): Longitudinal post-RT follow-up PET/CT (and PET-only where CT was absent).
   - **Reconciled outputs** tracked in `results/first_ef5_rt_study/`:
     - `mouse_id_mapping.csv`, `study_timeline.csv`, `data_audit_notes.md`
     - `longitudinal_imaging_summary.csv` (volumes, `%ID/g`, muscle-normalised EF5 `T/M` mean/max/p95)
     - `caliper_growth_curves.csv` (Day -15 through Day +39 normalized diameters) & `tumour_master_wide.csv`
   - **Standardised 3D modelling bundles** (local under `outputs/first_ef5_rt_study/`):
     - Compressed `.npz` voxel bundles (`pet_idg`, `pet_tm`, `ct_on_pet_grid`, PET & CT tumour masks, physical coordinate axes)
     - Verified **RAS+ NIfTI (`.nii.gz`)** exports for 3D Slicer / ITK-SNAP inspection.
3. **Second EF5 RT Study — 40-Mouse (80-Tumour) PLOS ONE Cohort (`src/process_second_ef5_rt_study.py`)**:
   - Curated and reconciled **40 bilateral A549 mice** (`c1_M1R` .. `c8_M0`; 80 tumours across 8 regimens: `1x10 Gy`, `1x20 Gy`, `1x40 Gy`, `0 Gy Control`, `2x5 Gy`, `2x10 Gy`, `4x5 Gy`, `4x10 Gy`), published in [Ali et al., PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139425):
     - **Pre-RT EF5 PET/CT (`10 Inveon 4-mouse beds`)**: 39 imaged mice across `Day 1 Set 1..5` (`i100117`, `2011-06-01`) and `Day 2 Set 1..5` (`i100118`, `2011-06-02`) cropped into standalone per-mouse `.dcm`, `.ROI`, `.npz` voxel bundles, and RAS+ `.nii.gz` NIfTI volumes (with both full-bed CT/combination tumour masks and PET-hotspot sub-ROI masks).
     - **11 Longitudinal Caliper Timepoints (`Day 23` to `Day 99`)**: Full reconciliation of 4 pre-RT and 7 post-RT caliper measurement dates (`2011-05-11` through `2011-07-26`), resolving the historical Mac-1904/Windows-1900 Excel date offset, proving correct Left (`LHS`) vs. Right (`RHS`) physical coordinate pairing ($r = 0.770$ vs. swapped $r = 0.236$), and documenting the 4-day fractionation gap (`2011-06-03` to `2011-06-07`) caused by the missing lead shield.
   - **Reconciled outputs** tracked in `results/second_ef5_rt_study/` and `outputs/second_ef5_rt_study/`:
     - `mouse_id_mapping.csv`, `imaging_summary.csv`, `caliper_growth_curves.csv`, `tumour_master_wide.csv`, `study_timeline.csv`, `data_audit_notes.md`, 10 bed overview PNGs, and 39 per-mouse QC PNGs.
4. **Past Postdoc Modelling Archive (`images/set4/past_work/`)**:
   - Curated `rsync` archive (117 files, macOS metadata excluded) of historical 2013 Stanford MATLAB ODEs, Dr. Alex Fletcher's notes & `CodeForBobby.m`, Prof. Helen Byrne's PCMI lecture notes (`pcmi_proofs.pdf`), legacy A549 `7 Gy` data (`rt_data_a549.mat`), and original PLOS ONE comparison figures.

---

## Part 2: Mathematical Modelling & Validation (`src/modeling/`)

Both the 1D and 3D simulators share a single, physically consistent biophysics engine in `src/modeling/physics.py` (units: `mm`, `days`, `mmHg`, `Gy`):

* **Oxygen-Dependent Radiobiology:** Continuous Alper–Howard-Flanders $\text{OER}(p\text{O}_2)$ with corrected Linear-Quadratic survival $S(d, p\text{O}_2) = \exp(-\alpha d_{\text{eff}} - \beta d_{\text{eff}}^2)$ (fixing the legacy 2013 `beta/OER` typo and removing binary `OERh=1` hacks).
* **Mechanical/Darcy Concave-Down Growth Saturation:** Extending Prof. Helen Byrne's PCMI continuum framework, viable proliferation couples oxygen availability (glycolytic mitosis in moderate viable hypoxia, $p_{\text{hyp}} = 0.60\text{ mmHg}$) with internal Darcy pressure and host solid-stress crowding — naturally reproducing rapid early untreated growth (`Day -15 → Day -1`: 3.3–3.6× vs observed 3.4–4.0×) and late concave-down saturation without ad-hoc `Vt` brakes.
* **Emergent Reoxygenation & High-Dose Vascular Response:**
  * Two-stage p53-WT A549 mitotic-catastrophe / senescence clearance ($D_1 \to D_2 \to N$) leaves a slowly clearing necrotic/fibrotic scaffold $N$.
  * Under fractionated doses (`4×5`, `4×10 Gy`), reduced O₂ demand plus Byrne/Jain solid-stress vascular decompression (as post-RT Darcy pressure $P$ drops) drives rapid inter-fraction reoxygenation, collapsing the EF5+ vs EF5− response gap.
  * Thresholded high-dose endothelial injury ($D_{50} = 15.5\text{ Gy}$, Hill $5.0$) spares `5–10 Gy` fractions while capturing microvascular attenuation and acute shrinkage at `20–40 Gy`.
* **Bioreductive EF5 PET Forward/Inverse Mapping:** Koch/Evans binding kinetics ($K_{\text{EF5}} = 5.5\text{ mmHg}$) link local $p\text{O}_2$ and metabolically active cell fractions ($V + w_{D1} D_1$) directly to virtual EF5 `T/M` maps.

### Sub-Goal 1: Coupled 1D Spheroid ODE/PDE Model
* **Analytical & Legacy Benchmarks (`src/modeling/legacy_spheroid.py`):** Ports the 2013 MATLAB ODE (`my_spheroid_07`) and Dr. Alex Fletcher's `CodeForBobby.m`; our spherical finite-volume oxygen PDE matches Helen Byrne's exact analytical Greenspan solution (Eqs. 44–50) to `< 2×10⁻³` relative error.
* **PLOS ONE & A549 Validation (`src/modeling/spheroid_rt_model.py`, `results/ef5_rt_model/spheroid_regimen_summary.csv`):**
  * **Control (`0 Gy`)**: Decelerates smoothly concave-down (`2.07×` Day +13 → `3.32×` Day +32).
  * **`1×10 Gy`**: Both `EF5+` (`2.12×`) and `EF5−` (`1.53×`) grow above baseline with a strong `+0.59×` hypoxia resistance gap.
  * **`4×5 Gy` & `4×10 Gy`**: Inter-fraction reoxygenation narrows the EF5+/− gap (`+0.23×` and `-0.01×`, with `4×10 Gy` staying near baseline).
  * **`1×40 Gy`**: Both phenotypes converge onto the `~0.71–0.73×` residual necrotic/fibrotic floor.
  * **`1×20 Gy` (First Study) & `1×7 Gy` (Legacy)**: Tracks real caliper separation with `EF5+ > EF5−` throughout and smooth virtual EF5 PET decline.

### Sub-Goal 2: 3D Image-Initialised Voxel Model
* **Per-Tumour Image Initialisation (`src/modeling/voxel_initializer.py`):** Resamples each tumour's Day -1 CT mask (geometry) and EF5 PET `T/M` map (hypoxia) onto an isotropic `0.5 mm` 3D grid with a shared physiological host boundary ($p_{\text{bdry}} = 35\text{ mmHg}$) and anchors a tumour-specific 3D vascular supply field $h_v(\mathbf{x})$ (preserving Day -1 spatial PET fidelity $r = 0.84\text{--}0.97$).
* **3D Continuum Mechanics (`src/modeling/voxel_rt_model.py`):** Solves the 3D quasi-steady nonlinear oxygen PDE and 3D Darcy pressure Poisson equation ($-\nabla^2 P = S_{\text{vol}}$, $\mathbf{u} = -\nabla P$) with conservative upwind finite-volume advection and packing projection, exporting both longitudinal metrics (`results/ef5_rt_model/voxel_study_comparison.csv`) and 3D RAS+ NIfTI predictions (`outputs/ef5_rt_model/voxel/nifti/`).
* **Reports & Technical Documentation:**
  * Full technical report (abstract, literature positioning, mathematical formulation, implementation details, and future multi-modal extensions): [outputs/ef5_rt_model/reports/technical_report.md](outputs/ef5_rt_model/reports/technical_report.md)
  * Visual comparison report (side-by-side 1D curves, 5-slice 3D PET/CT vs virtual EF5 & $p\text{O}_2$ ROI stacks, and 5-mouse bilateral growth curves): [outputs/ef5_rt_model/reports/visual_comparison_report.md](outputs/ef5_rt_model/reports/visual_comparison_report.md)

---

## CLI Parameter Reference (`src/ef5_slice_cli.py`)

```bash
docker compose run --rm ef5 python -m src.ef5_slice_cli \
    --pet images/set1/c1_M0_PET_mouse.dcm \
    --ct images/set1/c1_M0_CT_mouse.dcm \
    --roi images/set1/rois.ROI \
    --out-dir outputs/set1_every_pet_slice \
    --step-pet-slices 1 \
    --scope roi
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--pet` | *(required)* | Path to PET volume (`.dcm` or `.img`/`.hdr`) |
| `--ct` | *(required)* | Path to CT volume (`.dcm` or `.img`/`.hdr`) |
| `--roi` | *(required)* | Path to RT_Image binary `.ROI` file |
| `--out-dir` | *(required)* | Output directory for slices, panels, stats, and manifest |
| `--planes` | `transaxial coronal sagittal` | Orthogonal planes to render |
| `--step-pet-slices` | `1` | Step size in integer PET voxels (`1` = every PET slice + closest CT slice) |
| `--step-mm` | `None` | Optional step size in physical millimetres (overrides `--step-pet-slices`) |
| `--scope` | `roi` | `roi` (slices intersecting ROIs ± `--margin-mm`) or `all` (entire volume) |
| `--margin-mm` | `0.0` | Extra physical margin (mm) around ROI bounding box when `--scope roi` |
| `--modes` | `CT PET Overlay Panel` | Visualizations to export per slice |
| `--ct-clim` | `-500 800` | CT grayscale display window (HU) |
| `--pet-clim` | `0.0 2.2` | PET display window (`%ID/g`; use `0.0 2.8` for high-uptake Set 2 tumours) |
| `--overlay-alpha` | `0.78` | Maximum opacity of the smooth-alpha PET layer in fused overlays |
| `--full-long-axis` | `False` | Keep full cranial-caudal Y axis in coronal/sagittal views instead of zooming to ROIs |
| `--dpi` | `120` | Output resolution (`120` DPI = `1200 × 960 px` for single frames) |
