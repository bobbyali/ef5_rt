"""
Automated verification & physics conservation tests for src/modeling/ (Sub-goals 1 & 2).

Uses Python's standard library unittest (runs out-of-the-box in the ef5 Docker container):
    docker compose run --rm ef5 python -m unittest discover -s tests -v
"""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.modeling.legacy_spheroid import (
    analytical_byrne_greenspan_oxygen,
    run_alex_fletcher_growth,
    run_legacy_spheroid_2013,
    solve_spherical_oxygen_bvp_step,
)
from src.modeling.physics import (
    RadiobiologyParams,
    ef5_tm_to_po2,
    lq_surviving_fraction,
    oer_alper_howard_flanders,
    omf_oxygen_modifier,
    po2_to_ef5_tm,
)
from src.modeling.spheroid_rt_model import SpheroidRTModel, make_plos_one_scenarios
from src.modeling.voxel_initializer import VoxelTumourInitialState, initialize_tumour_from_npz
from src.modeling.voxel_rt_model import VoxelRTModel, VoxelSimulationConfig


class TestPhysics(unittest.TestCase):
    def test_alper_howard_flanders_oer_and_lq_monotonicity(self) -> None:
        prm = RadiobiologyParams()
        self.assertTrue(np.isclose(oer_alper_howard_flanders(0.0, prm), 1.0, atol=1e-6))
        self.assertTrue(np.isclose(oer_alper_howard_flanders(1e6, prm), prm.oer_max, rtol=1e-4))
        self.assertTrue(np.isclose(omf_oxygen_modifier(0.0, prm), 1.0 / prm.oer_max, atol=1e-6))
        self.assertGreater(omf_oxygen_modifier(40.0, prm), 0.9)

        for d in [2.0, 5.0, 10.0, 20.0]:
            s_hyp = lq_surviving_fraction(d, 0.5, prm)
            s_norm = lq_surviving_fraction(d, 35.0, prm)
            self.assertGreater(s_hyp, s_norm)
            self.assertGreater(s_norm, 0.0)
        self.assertEqual(lq_surviving_fraction(0.0, 10.0, prm), 1.0)

    def test_ef5_forward_inverse_roundtrip_and_necrosis(self) -> None:
        prm = RadiobiologyParams()
        p_vals = np.array([0.6, 2.0, 5.0, 15.0, 35.0])
        tm = po2_to_ef5_tm(p_vals, viable_frac=1.0, params=prm)
        self.assertTrue(np.all(np.diff(tm) < 0.0))
        p_rec = ef5_tm_to_po2(tm, params=prm, viable_frac=1.0)
        self.assertTrue(np.allclose(p_rec, p_vals, rtol=1e-3))
        tm_nec = po2_to_ef5_tm(0.5, viable_frac=0.0, params=prm, d1_frac=0.0)
        self.assertTrue(np.isclose(tm_nec, prm.tm_min, atol=1e-9))


class TestSubGoal1Spheroid(unittest.TestCase):
    def test_spherical_o2_solver_matches_byrne_analytical(self) -> None:
        for R_um in [250.0, 500.0, 850.0]:
            _, c_num = solve_spherical_oxygen_bvp_step(R_um, n_nodes=121)
            ana = analytical_byrne_greenspan_oxygen(R_um, n_nodes=121)
            rel_err = float(np.max(np.abs(c_num - ana["c_mM"])) / 0.21)
            self.assertLess(rel_err, 2e-3, f"R={R_um} um rel_err={rel_err:.2e}")

    def test_alex_fletcher_growth_self_limits_and_legacy_port(self) -> None:
        alex = run_alex_fletcher_growth()
        self.assertGreater(alex["R_um"][-1], alex["R_um"][0])
        self.assertGreater(alex["R_H_um"][-1], alex["R_N_um"][-1])
        self.assertGreater(alex["R_N_um"][-1], 0.0)
        init_slope = alex["R_um"][5] - alex["R_um"][0]
        end_slope = alex["R_um"][-1] - alex["R_um"][-6]
        self.assertLess(end_slope, 0.1 * init_slope)

        leg = run_legacy_spheroid_2013(6.0, 1, hypoxic=True, preserve_legacy_typos=True)
        self.assertGreater(leg.rel_volume.size, 10)
        self.assertTrue(np.all(np.isfinite(leg.rel_volume)))

    def test_refined_spheroid_reproduces_plos_one_hallmarks(self) -> None:
        prm = RadiobiologyParams()
        model = SpheroidRTModel(prm)

        # 1) 1 x 10 Gy: EF5+ is clearly more radioresistant than EF5-
        sc_1x10 = make_plos_one_scenarios([(0.0, 10.0)], "1x10 Gy", t_end_days=35.0)
        r_ctrl = model.run(sc_1x10["Control"])
        r_10_h = model.run(sc_1x10["EF5+"])
        r_10_n = model.run(sc_1x10["EF5-"])

        for res in (r_ctrl, r_10_h, r_10_n):
            tot = res.viable_frac + res.damaged_frac + res.necrotic_frac
            self.assertTrue(np.allclose(tot, 1.0, atol=1e-6))
            self.assertTrue(np.all(res.viable_frac >= 0.0))
            self.assertTrue(np.all(res.damaged_frac >= 0.0))
            self.assertTrue(np.all(res.necrotic_frac >= 0.0))

        self.assertGreater(r_ctrl.rel_volume[-1], 2.0)
        self.assertLess(r_ctrl.rel_volume[-1], 5.5)
        gap_1x10 = float(r_10_h.rel_volume[-1] - r_10_n.rel_volume[-1])
        self.assertGreater(gap_1x10, 0.10, f"Expected EF5+ > EF5- at 1x10 Gy, got gap={gap_1x10:.3f}")

        # 2) 1 x 40 Gy: EF5+ and EF5- converge and shrink below baseline without vanishing
        sc_1x40 = make_plos_one_scenarios([(0.0, 40.0)], "1x40 Gy", t_end_days=35.0)
        r_40_h = model.run(sc_1x40["EF5+"])
        r_40_n = model.run(sc_1x40["EF5-"])
        gap_1x40 = abs(float(r_40_h.rel_volume[-1] - r_40_n.rel_volume[-1]))
        self.assertLess(gap_1x40, 0.06)
        self.assertGreater(r_40_h.rel_volume[-1], 0.50)
        self.assertLess(r_40_h.rel_volume[-1], 0.92)
        self.assertGreater(r_40_n.rel_volume[-1], 0.50)
        self.assertLess(r_40_n.rel_volume[-1], 0.92)

        # 3) Iso-total-dose fractionation (1x20 -> 2x10 -> 4x5 Gy):
        #    emergent inter-fraction reoxygenation strictly reduces the EF5+ vs EF5- gap
        sc_1x20 = make_plos_one_scenarios([(0.0, 20.0)], "1x20 Gy", t_end_days=35.0)
        sc_2x10 = make_plos_one_scenarios([(0.0, 10.0), (1.0, 10.0)], "2x10 Gy", t_end_days=35.0)
        sc_4x5 = make_plos_one_scenarios([(0.0, 5.0), (1.0, 5.0), (2.0, 5.0), (3.0, 5.0)], "4x5 Gy", t_end_days=35.0)
        r_20_h, r_20_n = model.run(sc_1x20["EF5+"]), model.run(sc_1x20["EF5-"])
        r_2x10_h, r_2x10_n = model.run(sc_2x10["EF5+"]), model.run(sc_2x10["EF5-"])
        r_4x5_h, r_4x5_n = model.run(sc_4x5["EF5+"]), model.run(sc_4x5["EF5-"])
        gap_1x20 = float(r_20_h.rel_volume[-1] - r_20_n.rel_volume[-1])
        gap_2x10 = float(r_2x10_h.rel_volume[-1] - r_2x10_n.rel_volume[-1])
        gap_4x5 = float(r_4x5_h.rel_volume[-1] - r_4x5_n.rel_volume[-1])
        self.assertGreater(gap_1x20, gap_2x10)
        self.assertGreater(gap_2x10, gap_4x5)
        self.assertGreater(r_4x5_h.mean_po2_mmHg[20], r_4x5_h.mean_po2_mmHg[0])


class TestSubGoal2Voxel3D(unittest.TestCase):
    def test_3d_voxel_synthetic_sphere_growth_and_rt_contraction(self) -> None:
        prm = RadiobiologyParams()
        model = VoxelRTModel(prm)
        dx = 0.6
        n = 31
        coords = (np.arange(n) - n // 2) * dx
        YY, ZZ, XX = np.meshgrid(coords, coords, coords, indexing="ij")
        R_grid = np.sqrt(YY**2 + ZZ**2 + XX**2)
        phi0 = (R_grid <= 3.6).astype(np.float64)
        st = VoxelTumourInitialState(
            mouse_id="synth",
            tumour_id="synth_sphere",
            treatment="RT_20Gy",
            dose_gy=20.0,
            scan_date="2010-12-02",
            days_post_rt=-1,
            source_npz="synthetic",
            geometry_source="CT",
            dx_mm=dx,
            shape=(n, n, n),
            y_mm=coords,
            z_mm=coords,
            x_mm=coords,
            phi0=phi0,
            V0=phi0.copy(),
            N0=np.zeros_like(phi0),
            pet_tm0=np.where(phi0 > 0.5, 2.0, 1.0),
            pet_idg0=np.where(phi0 > 0.5, 0.8, 0.3),
            po2_0=np.where(phi0 > 0.5, 10.0, 35.0),
            h_vasc0=np.where(phi0 > 0.5, 0.15, 0.0),
            rim_h_vasc=0.15,
            p_bdry_mmHg=35.0,
            obs_volume_cm3=float(phi0.sum() * dx**3 / 1000.0),
            obs_tm_mean=2.0,
            obs_tm_p95=2.0,
        )
        res_ctrl = model.run(st, VoxelSimulationConfig(t_end_days=20.0, dt_days=0.5, dose_schedule=[], snapshot_days=[0.0, 20.0]))
        res_rt = model.run(st, VoxelSimulationConfig(t_end_days=35.0, dt_days=0.5, dose_schedule=[(1.0, 40.0)], snapshot_days=[0.0, 35.0]))

        self.assertTrue(np.all(np.isfinite(res_ctrl.volume_cm3)))
        self.assertTrue(np.all(np.isfinite(res_rt.volume_cm3)))
        self.assertGreater(res_ctrl.rel_volume[-1], 1.6)
        self.assertLess(res_rt.rel_volume[-1], 0.95)
        self.assertGreater(res_rt.mean_po2_mmHg[-1], res_rt.mean_po2_mmHg[0])
        self.assertLessEqual(float(np.max(res_ctrl.snapshots[20.0]["phi"])), 1.05)

    def test_3d_voxel_initialisation_on_real_c2_M1R_bundle(self) -> None:
        npz_path = Path("outputs/first_ef5_rt_study/voxel_bundles/c2_M1R/c2_M1R_day-1_2010-12-02.npz")
        if not npz_path.exists():
            self.skipTest("First EF5 RT Study voxel bundle not present locally.")

        prm = RadiobiologyParams()
        st_rt = initialize_tumour_from_npz(npz_path, "rt", params=prm, dx_mm=0.5)
        self.assertEqual(st_rt.geometry_source, "CT")
        sim_vol0 = float(np.sum(st_rt.phi0) * (st_rt.dx_mm**3) / 1000.0)
        self.assertLess(abs(sim_vol0 - st_rt.obs_volume_cm3) / st_rt.obs_volume_cm3, 0.10)

        model = VoxelRTModel(prm)
        res = model.run(
            st_rt,
            VoxelSimulationConfig(t_end_days=14.0, dt_days=0.5, dose_schedule=[(1.0, 20.0)], snapshot_days=[0.0, 14.0]),
        )
        roi0 = res.snapshots[0.0]["roi_mask"]
        r_init = float(np.corrcoef(st_rt.pet_tm0[roi0], res.snapshots[0.0]["ef5_tm"][roi0])[0, 1])
        self.assertGreater(r_init, 0.70, f"Expected initial 3D EF5 spatial fidelity r > 0.70, got {r_init:.3f}")
        self.assertLess(res.mean_ef5_tm[-1], res.mean_ef5_tm[0])


if __name__ == "__main__":
    unittest.main()
