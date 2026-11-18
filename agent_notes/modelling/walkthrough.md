# Walkthrough: EF5 PET/CT-Driven Tumour Growth & Hypoxia Radiotherapy Modelling (`ef5_rt_model`)

## 1. Summary of Changes

All work was executed on the new Git branch **`ef5_rt_model`** (branched off `first_ef5_rt_study`), keeping `images/` and `outputs/` gitignored while tracking modelling code, tests, notebooks, `agent_notes/`, and curated CSV summaries.

### Phase 0: Branching, `agent_notes/` Reorganization & Curated `rsync`
- Created branch `ef5_rt_model` off `first_ef5_rt_study`.
- Reorganized `agent_notes/`:
  - Moved Set 1 & Set 2 notes into `agent_notes/data_curation/`.
  - Tracked the modelling plan (`implementation_plan.md`) and this walkthrough (`walkthrough.md`) under `agent_notes/modelling/`.
- Executed a curated `rsync` from `D:\Research\Stanford\Work\001. SMIS Postdoc\Projects\13. Predicting RT Response` into `images/set4/past_work/` via `images/set4/past_work/rsync_filter.txt`:
  - **117 files (~14.4 MB)** retained (`Code/02`, `04`, `05`, `06`, `07` scripts & `Data/`, `08`, `09`, `Reports/`, and `Data/DCE MRI Subcutaneous Tumor/CT-T2-DCE.pdf`).
  - **0 Apple/Windows metadata files** (`._*`, `.DS_Store`, `Thumbs.db`) verified after transfer.
  - Documented in `images/set4/past_work/ARCHIVE_MANIFEST.md`.

### Phase 1: Shared Biophysics Core & Sub-Goal 1 (Refined Spheroid ODE/PDE RT Model)
- **`src/modeling/physics.py`**: Shared unit-consistent radiobiology (`mm`, `days`, `mmHg`, `Gy`) used identically by both 1D and 3D models — Alper–Howard-Flanders continuous OER($p\text{O}_2$), corrected LQ survival $S(d, p) = \exp(-\alpha\,\text{OMF}(p)\,d - \beta\,[\text{OMF}(p)\,d]^2)$ (fixing the legacy 2013 MATLAB `b / OERh` typo), smooth Hill proliferation/necrosis switches, two-stage mitotic-catastrophe clearance ($D_1 \to D_2 \to \text{cleared} + \eta_{\text{fib}} N$), and bioreductive EF5 PET $\leftrightarrow p\text{O}_2$ forward/inverse mappings.
- **`src/modeling/legacy_spheroid.py`**: Audited Python ports of `my_spheroid_07/08` and Dr. Alex Fletcher's `CodeForBobby.m` (bug-fixed, replacing MATLAB `pdepe` with a fast spherical finite-volume Newton solver verified against Helen Byrne's exact analytical Greenspan Eqs. 44–50 in `pcmi_proofs.pdf`).
- **`src/modeling/spheroid_rt_model.py` & `src/modeling/run_spheroid_study.py`**: Coupled 1D spherical quasi-steady oxygen PDE + moving-boundary Greenspan volume ODE + conservative radial advection of $(V, D_1, D_2, N)$. Eliminates all legacy heuristics (no binary rim width $r_2$, no `Vt` exponential brake, no `OERh = 1` hack).

### Phase 2: Sub-Goal 2 (Tumour-Specific 3D Image-Initialised Voxel Model)
- **`src/modeling/voxel_initializer.py`**: Loads any `.npz` voxel bundle from `outputs/first_ef5_rt_study/voxel_bundles/`, resamples native CT tumour geometry (falling back to PET when PET-only) and pre-RT EF5 PET `T/M` onto an isotropic `0.5 mm` padded 3D grid, converts EF5 $\to p\text{O}_2$, and anchors a tumour-specific 3D vascular supply field $h_v(\mathbf{x})$ so the 3D oxygen PDE reproduces each tumour's measured spatial hypoxia at $t = 0$.
- **`src/modeling/voxel_rt_model.py`**: 3D continuum PDE simulator extending Helen Byrne's PCMI notes (Sec. 5, Eqs. 53–59): warm-started Newton–CG 3D quasi-steady oxygen PDE, voxelwise LQ radiation kill, 3D Darcy pressure-velocity expansion/contraction ($-\nabla^2 P = S_{\text{vol}}$, $\mathbf{u} = -\nabla P$) with conservative upwind finite-volume advection and anti-overshoot packing, volume-attenuated rim vascular remodelling, and virtual EF5 PET forward readout + RAS+ NIfTI export (`outputs/ef5_rt_model/voxel/nifti/`).
- **`src/modeling/run_voxel_study.py`**: Runs both a pre-RT untreated growth check (`Day -15 → Day -1`) and the 10-tumour paired bilateral study (`Day -1 → Day +13 → Day +32`, `RT_20Gy` vs `Control_0Gy`) across all 5 surviving mice (`c2_M1R`, `c2_M0`, `c2_M1L`, `c2_M1L1R`, `c2_M2R`) with one shared parameter set.

### Phase 3: Tests, Interactive Notebook & Documentation
- **`tests/test_modeling.py`**: 7 automated `unittest` checks (7/7 passing in 10.7 s inside Docker).
- **`notebooks/explore_ef5_rt_models.ipynb`**: Interactive `ipywidgets` JupyterLab explorer for both 1D spheroid regimens and 3D mouse tumour slices.
- **`README.md`**: Added documentation and CLI commands for `src/modeling/`.

---

## 2. Verification & Key Quantitative Results

### Automated Tests
```bash
docker compose run --rm ef5 python -m unittest discover -s tests -v
# Ran 7 tests in 10.739s — OK
```

### Sub-Goal 1: Analytical Validation & PLOS ONE Regimens (`results/ef5_rt_model/spheroid_regimen_summary.csv`)
- **Byrne Analytical Greenspan**: Spherical oxygen BVP matches exact analytical solutions (`pcmi_proofs.pdf`, Eqs. 44–50) to `< 1e-10` (non-necrotic) and `< 2e-3` (necrotic core, `R = 850 µm`).
- **PLOS ONE Hallmarks (Day +32 Relative Volume)**:
  - **Control (`0 Gy`)**: `2.71×` (`2.92×` at Day +35), EF5 T/M rises `2.73 → 3.18`.
  - **`1 × 10 Gy`**: `EF5+ = 1.22×` vs `EF5− = 1.09×` (clear hypoxic radioresistance separation).
  - **`1 × 20 Gy`**: `EF5+ = 1.03×` vs `EF5− = 0.85×` (`+0.17` gap).
  - **`1 × 40 Gy`**: `EF5+ = 0.87×` vs `EF5− = 0.85×` (converges cleanly to necrotic/fibrotic core floor, `+0.02` gap).
  - **Iso-total-dose fractionation (`1×20 → 2×10 → 4×5 Gy`)**: Emergent inter-fraction reoxygenation strictly narrows the EF5+ vs EF5− gap (`+0.17 → +0.14 → +0.12`) while preserving higher overall volume at `4×5 Gy` than `1×20 Gy`.
  - **`4 × 10 Gy`**: Gap collapses (`+0.02`).
- Generated figures: `outputs/ef5_rt_model/spheroid/fig1_alex_byrne_validation.png` through `fig4_legacy_vs_refined_and_first_study.png`.

### Sub-Goal 2: 3D Image-Initialised Voxel Study (`results/ef5_rt_model/voxel_study_comparison.csv`)
- **Initial 3D spatial fidelity**: Anchored $h_v(\mathbf{x})$ preserves pre-RT EF5 PET heterogeneity across all 10 tumours (`r = 0.79 – 0.97`, e.g., `c2_M1R_RT: r = 0.968`) and initial CT/PET volume within `~1–5%`.
- **Untreated Controls (`Control_0Gy`)**: Expand smoothly from asymmetric CT contours while developing central hypoxia/necrosis (`c2_M1R_Ctrl`: Sim `0.19 → 0.34 → 0.72 cm³` vs Obs `0.18 → 0.54 → 0.76 cm³`; `c2_M1L_Ctrl`: Sim `0.44 → 0.70 → 1.14 cm³` vs Obs `0.44 → 0.95 → 1.59 cm³`).
- **Irradiated Tumours (`RT_20Gy`)**: Reproduce transient Day +13 expansion followed by Day +32 arrest/contraction (`c2_M2R_RT`: Sim `0.60 → 0.71 → 0.54 cm³` vs Obs `0.60 → 0.81 → 0.42 cm³`), monotonic post-RT reoxygenation (within-mouse `RT / Control` EF5 T/M ratio drops from Day -1 to Day +13 and Day +32), and cross-mouse pre-RT hypoxia resistance ordering (`c2_M1L1R_RT` [T/M 1.63] most sensitive vs `c2_M1L_RT` / `c2_M1R_RT` [T/M 3.04–3.21] most resistant).
- Generated figures & 3D volumes: `outputs/ef5_rt_model/voxel/figures/cohort_summary.png`, `<mouse_id>_observed_vs_simulated.png`, and RAS+ NIfTI outputs under `outputs/ef5_rt_model/voxel/nifti/`.

---

## 3. Ready for Future 40-Mouse PLOS ONE Validation
- Both `initialize_tumour_from_npz()` and `VoxelRTModel.run()` accept arbitrary multi-fraction schedules `dose_schedule=[(day, Gy), ...]` and execute in ~3–10 s per tumour on CPU.
- Once the 40-mouse PLOS ONE imaging + caliper cohort is curated, we can run global parameter fitting on $(\alpha, \beta, k_{d1,\text{prolif}}, \eta_{\text{fibrotic}}, \text{vasc\_damage\_max})$ against the multi-arm caliper trajectories without changing the core PDE architecture.
