"""
Automated unit and regression tests for the Second EF5 RT Study (40-Mouse / 80-Tumour PLOS ONE Cohort)
4-Stage Validation, Global Sensitivity, Identifiability, and Hierarchical Bayesian Inference Suite.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.modeling.hierarchical_bayesian import (
    decompose_hierarchical_growth_random_effects,
    get_bayesian_pop_specs,
)
from src.modeling.physics import RadiobiologyParams
from src.modeling.run_study2_validation import (
    build_rt_schedules,
    compute_clinical_endpoints,
    fit_prert_tumour_kp_max,
    load_study2_cohort,
    simulate_1d_tumour,
)
from src.modeling.sensitivity_identifiability import get_gsa_parameter_specs
from src.modeling.voxel_initializer import initialize_tumour_from_npz


ROOT = Path(__file__).resolve().parents[1]


class TestStudy2ValidationSuite(unittest.TestCase):
    def test_cohort_loader_and_4day_shield_gap_schedules(self) -> None:
        records = load_study2_cohort(ROOT)
        self.assertEqual(len(records), 80)
        imaged_count = sum(1 for r in records if r.imaged)
        self.assertEqual(imaged_count, 78)

        # Verify 4-day missing-lead-shield gap in fractionated cages (e.g. Cage 7: 4x5 Gy)
        act_b, nom_b, act_s, nom_s, off = build_rt_schedules(7, 4, 5.0, "2011-06-02")
        self.assertEqual(off, 2.0)
        self.assertEqual([t for t, _ in act_b], [3.0, 7.0, 8.0, 9.0])
        self.assertEqual([t for t, _ in nom_b], [3.0, 4.0, 5.0, 6.0])
        self.assertEqual([t for t, _ in act_s], [1.0, 5.0, 6.0, 7.0])

    def test_prert_growth_fitting_and_1d_simulation(self) -> None:
        records = load_study2_cohort(ROOT)
        prm = RadiobiologyParams()
        rec_ctrl = next(r for r in records if r.treatment == "Control_0Gy" and r.imaged)
        rec_40gy = next(r for r in records if r.treatment == "1x40Gy" and r.imaged)

        kp_ctrl = fit_prert_tumour_kp_max(rec_ctrl, prm)
        self.assertGreaterEqual(kp_ctrl, 0.065)
        self.assertLessEqual(kp_ctrl, 0.235)

        res_c, pred_c = simulate_1d_tumour(rec_ctrl, prm)
        res_40, pred_40 = simulate_1d_tumour(rec_40gy, prm)
        self.assertEqual(len(pred_c), len(rec_ctrl.post_t_days))
        # Untreated control grows substantially larger than 40 Gy ablative tumour by Day +27 (t=30)
        v_ctrl_d30 = float(np.interp(30.0, res_c.t_days, res_c.rel_volume))
        v_40_d30 = float(np.interp(30.0, res_40.t_days, res_40.rel_volume))
        self.assertGreater(v_ctrl_d30, 2.0)
        self.assertLess(v_40_d30, 1.35)
        self.assertGreater(v_ctrl_d30, v_40_d30 + 1.0)

    def test_3d_voxel_initialisation_on_study2_bundle(self) -> None:
        npz_path = ROOT / "outputs" / "second_ef5_rt_study" / "voxel_bundles" / "c1_M1R" / "c1_M1R_pre_rt.npz"
        if not npz_path.exists():
            self.skipTest("Study 2 c1_M1R_pre_rt.npz bundle not present on disk")
        init_lhs = initialize_tumour_from_npz(npz_path, side="lhs", dx_mm=0.55, pad_mm=6.0)
        init_rhs = initialize_tumour_from_npz(npz_path, side="rhs", dx_mm=0.55, pad_mm=6.0)
        self.assertEqual(init_lhs.tumour_id, "c1_M1R_L")
        self.assertEqual(init_rhs.tumour_id, "c1_M1R_R")
        self.assertGreater(init_lhs.obs_volume_cm3, 0.05)
        self.assertGreater(init_rhs.obs_volume_cm3, 0.05)
        self.assertGreater(init_lhs.obs_tm_mean, 0.8)

    def test_hierarchical_random_effects_and_parameter_specs(self) -> None:
        records = load_study2_cohort(ROOT)
        prm = RadiobiologyParams()
        re_df, re_stats = decompose_hierarchical_growth_random_effects(records, prm)
        self.assertEqual(len(re_df), 80)
        self.assertEqual(re_stats["n_mice"], 40)
        self.assertGreater(re_stats["icc_bilateral_mouse"], 0.0)
        self.assertLess(re_stats["icc_bilateral_mouse"], 1.0)
        self.assertEqual(len(get_gsa_parameter_specs(prm)), 12)
        self.assertEqual(len(get_bayesian_pop_specs(prm)), 7)


if __name__ == "__main__":
    unittest.main()
