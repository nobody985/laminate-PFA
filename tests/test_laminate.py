import contextlib
import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from laminate import (
    Laminate, Material, Ply, StiffnessLoss, UnsupportedLoad, analyze, assemble_abd,
    degraded_stiffness, failure_mode, hoffman_coefficients, hoffman_index,
    positive_failure_roots, solve_response, surface_stresses, transformation,
)
from main import main
from reporting import write_outputs


ROOT = Path(__file__).resolve().parents[1]


def material(**changes):
    values = dict(E1=45000., E2=12000., nu12=.25, G12=5000.,
                  Xt=1000., Xc=700., Yt=40., Yc=150., S=60.)
    values.update(changes)
    return Material(**values)


def laminate(angles=(0,), thickness=1., load=(1., 0., 0., 0., 0., 0.), mat=None):
    return Laminate({"m": mat or material()}, tuple(Ply("m", thickness, a) for a in angles), tuple(load))


class ElasticityTests(unittest.TestCase):
    def test_original_compliance_is_preserved(self):
        m = material()
        s = m.compliance()
        self.assertEqual(s[0, 1], -m.nu12/m.E2)
        self.assertNotEqual(s[0, 1], -m.nu12/m.E1)
        delta = 1 - m.nu12**2 * m.E1/m.E2
        expected = [[m.E1/delta, m.nu12*m.E1/delta, 0],
                    [m.nu12*m.E1/delta, m.E2/delta, 0], [0, 0, m.G12]]
        np.testing.assert_allclose(m.stiffness(), expected)

    def test_single_zero_degree_abd(self):
        model = laminate(thickness=2.)
        q = model.materials["m"].stiffness()
        abd, _ = assemble_abd(model)
        np.testing.assert_allclose(abd[:3, :3], q*2)
        np.testing.assert_allclose(abd[:3, 3:], 0, atol=1e-12)
        np.testing.assert_allclose(abd[3:, 3:], q*8/12)

    def test_ninety_degree_rotation(self):
        model = laminate((90,))
        q = model.materials["m"].stiffness()
        _, (qbar,) = assemble_abd(model)
        np.testing.assert_allclose(qbar, q[np.ix_([1, 0, 2], [1, 0, 2])], atol=1e-10)
        np.testing.assert_allclose(transformation(90) @ [11, 22, 3], [22, 11, -3], atol=1e-12)

    def test_forty_five_degree_engineering_shear(self):
        model = laminate((45,))
        q = model.materials["m"].stiffness()
        _, (qbar,) = assemble_abd(model)
        expected = np.array([
            [(q[0, 0]+q[1, 1]+2*q[0, 1]+4*q[2, 2])/4,
             (q[0, 0]+q[1, 1]+2*q[0, 1]-4*q[2, 2])/4, (q[0, 0]-q[1, 1])/4],
            [(q[0, 0]+q[1, 1]+2*q[0, 1]-4*q[2, 2])/4,
             (q[0, 0]+q[1, 1]+2*q[0, 1]+4*q[2, 2])/4, (q[0, 0]-q[1, 1])/4],
            [(q[0, 0]-q[1, 1])/4, (q[0, 0]-q[1, 1])/4,
             (q[0, 0]+q[1, 1]-2*q[0, 1])/4]])
        np.testing.assert_allclose(qbar, expected, atol=1e-10)

    def test_symmetric_layup_has_zero_b(self):
        abd, _ = assemble_abd(laminate((0, 45, -45, 90, 90, -45, 45, 0), thickness=.125))
        np.testing.assert_allclose(abd[:3, 3:], 0, atol=1e-10)

    def test_unsymmetric_layup_couples_extension_and_bending(self):
        model = laminate((0, 90), thickness=.5)
        abd, _ = assemble_abd(model)
        response = solve_response(abd, model.reference_load, model.thickness)
        self.assertGreater(np.linalg.norm(abd[:3, 3:]), 1.)
        self.assertGreater(np.linalg.norm(response[3:]), 1e-7)
        np.testing.assert_allclose(abd @ response, model.reference_load, atol=1e-12)

    def test_pure_bending_surface_stress_and_failure(self):
        model = laminate(thickness=2., load=(0, 0, 0, 1, 0, 0))
        abd, qbars = assemble_abd(model)
        response = solve_response(abd, model.reference_load, 2.)
        stress = surface_stresses(model, qbars, response)
        np.testing.assert_allclose(stress[0], [[-1.5, 0, 0], [1.5, 0, 0]], atol=1e-12)
        result = analyze(model)
        self.assertAlmostEqual(result.first_failure_load_factor, 700*2**2/6)
        self.assertEqual(result.events[0].surface, "top")
        self.assertAlmostEqual(result.events[0].z, -1.)
        self.assertLess(result.events[0].stress[0], 0)

    def test_top_to_bottom_interfaces_with_unequal_thicknesses(self):
        model = Laminate({"m": material()},
                         (Ply("m", .2, 0), Ply("m", .3, 45), Ply("m", .5, 90)),
                         (1, 0, 0, 0, 0, 0))
        np.testing.assert_allclose(model.interfaces, [-.5, -.3, 0, .5], atol=1e-15)

    def test_negative_bending_compresses_bottom_surface(self):
        result = analyze(laminate(thickness=2., load=(0, 0, 0, -1, 0, 0)))
        self.assertEqual(result.events[0].surface, "bottom")
        self.assertAlmostEqual(result.events[0].z, 1.)
        self.assertLess(result.events[0].stress[0], 0)
        self.assertAlmostEqual(result.first_failure_load_factor, 700*2**2/6)

    def test_reversing_stack_and_moments_preserves_stress_field(self):
        model = Laminate({"m": material()},
                         (Ply("m", .2, 0), Ply("m", .3, 35), Ply("m", .5, 90)),
                         (8, -3, 2, .6, -.3, .1))
        reversed_model = replace(model, plies=tuple(reversed(model.plies)),
                                 reference_load=(*model.reference_load[:3],
                                                 *(-v for v in model.reference_load[3:])))
        abd, qbars = assemble_abd(model)
        reversed_abd, reversed_qbars = assemble_abd(reversed_model)
        np.testing.assert_allclose(reversed_abd[:3, :3], abd[:3, :3], atol=1e-10)
        np.testing.assert_allclose(reversed_abd[3:, 3:], abd[3:, 3:], atol=1e-10)
        np.testing.assert_allclose(reversed_abd[:3, 3:], -abd[:3, 3:], atol=1e-10)
        response = solve_response(abd, model.reference_load, model.thickness)
        reversed_response = solve_response(reversed_abd, reversed_model.reference_load, model.thickness)
        np.testing.assert_allclose(reversed_response, response*[1, 1, 1, -1, -1, -1], atol=1e-12)
        stress = surface_stresses(model, qbars, response)
        reversed_stress = surface_stresses(reversed_model, reversed_qbars, reversed_response)
        np.testing.assert_allclose(reversed_stress, stress[::-1, ::-1], atol=1e-10)

    def test_interface_sides_are_distinct(self):
        model = laminate((0, 90), load=(2, 1, .5, .3, -.2, .1))
        abd, qbars = assemble_abd(model)
        response = solve_response(abd, model.reference_load, model.thickness)
        stress = surface_stresses(model, qbars, response)
        self.assertFalse(np.allclose(stress[0, 1], stress[1, 0]))

    def test_integrated_stresses_recover_applied_resultants(self):
        model = laminate((0, 35, -20, 90), thickness=.2, load=(8, -3, 2, .6, -.3, .1))
        states = ("intact", "shear_failed", "transverse_failed", "intact")
        abd, qbars = assemble_abd(model, states)
        response = solve_response(abd, model.reference_load, model.thickness)
        membrane, moment = np.zeros(3), np.zeros(3)
        for i, ply in enumerate(model.plies):
            mid = np.mean(model.interfaces[i:i+2])
            sigma_mid = qbars[i] @ (response[:3] + mid*response[3:])
            slope = qbars[i] @ response[3:]
            membrane += ply.thickness*sigma_mid
            moment += ply.thickness*mid*sigma_mid + ply.thickness**3*slope/12
        np.testing.assert_allclose(np.r_[membrane, moment], model.reference_load, atol=1e-12)

    def test_thickness_scaling_handles_very_thin_laminate(self):
        model = laminate(thickness=1e-6)
        abd, _ = assemble_abd(model)
        response = solve_response(abd, model.reference_load, model.thickness)
        self.assertAlmostEqual(response[0], 1/(45000*1e-6))


class CriterionTests(unittest.TestCase):
    def test_uniaxial_tension_compression_and_shear(self):
        cases = [((1, 0, 0, 0, 0, 0), 2000., "fiber"),
                 ((-1, 0, 0, 0, 0, 0), 1400., "fiber"),
                 ((0, 1, 0, 0, 0, 0), 80., "transverse"),
                 ((0, -1, 0, 0, 0, 0), 300., "transverse"),
                 ((0, 0, 1, 0, 0, 0), 120., "shear"),
                 ((0, 0, -1, 0, 0, 0), 120., "shear")]
        for load, expected, mode in cases:
            with self.subTest(load=load):
                result = analyze(laminate(thickness=2., load=load))
                self.assertAlmostEqual(result.first_failure_load_factor, expected)
                self.assertEqual(result.events[0].mode, mode)
                self.assertAlmostEqual(result.events[0].failure_index, 1.)

    def test_combined_failure_uses_largest_utilization(self):
        for stress, expected in [([650, 33, 0], "transverse"),
                                  ([600, 0, 55], "shear"), ([900, 24, 0], "fiber")]:
            with self.subTest(stress=stress):
                self.assertGreater(hoffman_index(stress, material()), 1.)
                mode, ratios = failure_mode(stress, material(), "intact")
                self.assertTrue(all(r < 1 for r in ratios))
                self.assertEqual(mode, expected)

    def test_exceeded_modes_and_ties_have_defined_priority(self):
        self.assertEqual(failure_mode([1001, 800, 600], material(), "intact")[0], "fiber")
        self.assertEqual(failure_mode([1, 40, 600], material(), "intact")[0], "transverse")
        self.assertEqual(failure_mode([500, 20, 30], material(), "intact")[0], "fiber")
        self.assertEqual(failure_mode([-350, -75, -30], material(), "intact")[0], "fiber")

    def test_quadratic_linear_tangent_and_absent_roots(self):
        self.assertEqual(positive_failure_roots(1., 0.), (1.,))
        self.assertEqual(positive_failure_roots(0., .25), (4.,))
        self.assertEqual(positive_failure_roots(0., 0.), ())
        self.assertEqual(positive_failure_roots(0., -1.), ())
        self.assertEqual(positive_failure_roots(-1., 0.), ())
        self.assertEqual(positive_failure_roots(-1., 2.), (1.,))
        np.testing.assert_allclose(positive_failure_roots(-1., 3.), [(3-math.sqrt(5))/2, (3+math.sqrt(5))/2])

    def test_root_solver_extreme_scaling(self):
        for a, expected in [(1e-300, 1e150), (1e300, 1e-150)]:
            self.assertAlmostEqual(positive_failure_roots(a, 0)[0]/expected, 1.)
        for a, b in [(1e-20, 1.), (1e-20, -1.), (-1e-20, 1.)]:
            for root in positive_failure_roots(a, b):
                # Evaluate relative residual to account for cancellation at a large second root.
                residual = abs((a*root+b)*root - 1.)
                scale = abs(a*root*root) + abs(b*root) + 1
                self.assertLess(residual/scale, 1e-12)

    def test_linear_hoffman_event_end_to_end(self):
        m = material(Xt=1, Xc=2, Yt=2, Yc=4, nu12=0)
        self.assertEqual(hoffman_coefficients([1, 2, 0], m), (0., 1.))
        result = analyze(laminate(load=(1, 2, 0, 0, 0, 0), mat=m))
        self.assertAlmostEqual(result.first_failure_load_factor, 1.)

    def test_no_finite_failure_is_not_reported_as_infinite_strength(self):
        # Positive strengths do not ensure a convex Hoffman quadratic form.
        m = material(Xt=1, Xc=1, Yt=10, Yc=10, nu12=0)
        result = analyze(laminate(load=(1, 2, 0, 0, 0, 0), mat=m))
        self.assertEqual(result.status, "no_finite_failure")
        self.assertIsNone(result.ultimate_load_factor)
        self.assertIsNone(result.first_failure_load_factor)
        self.assertEqual(len(result.history), 1)


class DamageTests(unittest.TestCase):
    def test_classroom_glass_cross_ply_against_independent_membrane_solution(self):
        from verify_cross_ply import verify
        model = Laminate.from_json(ROOT / "examples/glass_cross_ply.json")
        result = analyze(model)
        verify(model, result)
        self.assertEqual(result.final_states, ("fully_failed", "transverse_failed", "fully_failed"))
        self.assertEqual(result.status, "load_unsupported")
        self.assertIsNone(result.ultimate_load_factor)
        self.assertAlmostEqual(result.terminal_load_factor, 1034/6)
        self.assertTrue(any(point.stiffness_rank == 4 for point in result.history))
        self.assertAlmostEqual(result.first_failure_load_factor, 38.361502752587285, places=8)

    def test_three_degradation_matrices(self):
        q = material().stiffness()
        original = q.copy()
        np.testing.assert_array_equal(degraded_stiffness(q, "fully_failed"), np.zeros((3, 3)))
        expected = np.zeros((3, 3))
        expected[0, 0] = q[0, 0]
        np.testing.assert_array_equal(degraded_stiffness(q, "transverse_failed"), expected)
        expected = q.copy()
        expected[2, :] = expected[:, 2] = 0
        np.testing.assert_array_equal(degraded_stiffness(q, "shear_failed"), expected)
        np.testing.assert_array_equal(q, original)

    def test_transverse_damage_retains_compressive_q11(self):
        model = laminate((0, 45))
        states = ("transverse_failed", "intact")
        _, qbars = assemble_abd(model, states)
        stress = surface_stresses(model, qbars, np.array([-.01, 0, 0, 0, 0, 0]), states)
        self.assertAlmostEqual(stress[0, 0, 0], -.01*material().stiffness()[0, 0])
        np.testing.assert_array_equal(stress[0, :, 1:], 0.)

    def test_simultaneous_failures_and_same_load_cascade(self):
        result = analyze(Laminate.from_json(ROOT / "examples/cascade.json"))
        self.assertAlmostEqual(result.first_failure_load_factor, 40.)
        self.assertAlmostEqual(result.ultimate_load_factor, 40.)
        self.assertEqual([e.ply for e in result.events], [1, 4, 2, 3])
        self.assertEqual([e.cascade_step for e in result.events], [0, 0, 1, 1])
        np.testing.assert_allclose([e.load_factor for e in result.events], 40.)
        np.testing.assert_allclose([p.response[0] for p in result.history], [0, .001, .002])
        self.assertEqual(result.final_states, ("fully_failed",)*4)
        # Last saved state precedes final degradation; there is no fictitious post-collapse point.
        self.assertEqual(result.history[-1].states, ("fully_failed", "intact", "intact", "fully_failed"))

    def test_stable_redistribution_then_further_loading(self):
        model = Laminate.from_json(ROOT / "examples/cascade.json")
        materials = dict(model.materials)
        materials["strong"] = replace(materials["strong"], Xt=30.)
        result = analyze(replace(model, materials=materials))
        np.testing.assert_allclose([p.load_factor for p in result.history], [0, 40, 40, 60])
        np.testing.assert_allclose([p.response[0] for p in result.history], [0, .001, .002, .003])
        self.assertEqual(result.history[2].stage, "redistributed")
        self.assertAlmostEqual(result.ultimate_load_factor, 60.)

    def test_proportional_reference_load_scaling_preserves_actual_load(self):
        model = Laminate.from_json(ROOT / "examples/bending.json")
        baseline = analyze(model)
        for scale in (1e-9, 7., 1e9):
            with self.subTest(scale=scale):
                changed = analyze(replace(model, reference_load=tuple(x*scale for x in model.reference_load)))
                self.assertEqual([(e.ply, e.mode, e.cascade_step) for e in baseline.events],
                                 [(e.ply, e.mode, e.cascade_step) for e in changed.events])
                np.testing.assert_allclose([e.load_factor*scale for e in changed.events],
                                           [e.load_factor for e in baseline.events], rtol=1e-9)
                np.testing.assert_allclose([p.response for p in changed.history],
                                           [p.response for p in baseline.history], atol=1e-12)

    def test_examples_have_monotonic_irreversible_finite_history(self):
        order = {"intact": 0, "shear_failed": 1, "transverse_failed": 2, "fully_failed": 3}
        for filename in ("tension", "bending", "cascade", "glass_cross_ply"):
            with self.subTest(example=filename):
                result = analyze(Laminate.from_json(ROOT / f"examples/{filename}.json"))
                expected = "all_plies_failed" if filename in ("tension", "cascade") else "load_unsupported"
                self.assertEqual(result.status, expected)
                self.assertEqual(result.ultimate_load_factor is not None,
                                 all(state == "fully_failed" for state in result.final_states))
                self.assertLessEqual(len(result.events), 3*len(result.model.plies))
                self.assertTrue(np.all(np.diff([p.load_factor for p in result.history]) >= 0))
                for event in result.events:
                    self.assertGreater(order[event.state_after], order[event.state_before])
                    self.assertGreaterEqual(event.failure_index, 1-1e-8)
                for point in result.history:
                    self.assertTrue(np.all(np.isfinite(point.response)))
                    abd, _ = assemble_abd(result.model, point.states)
                    np.testing.assert_allclose(abd @ point.response,
                                               np.asarray(result.model.reference_load)*point.load_factor,
                                               rtol=1e-9, atol=1e-9)

    def test_unsupported_shear_load_is_not_all_plies_failed(self):
        result = analyze(laminate(load=(0, 0, 1, 0, 0, 0)))
        self.assertEqual(result.final_states, ("shear_failed",))
        self.assertEqual(result.status, "load_unsupported")
        self.assertIsNone(result.ultimate_load_factor)
        self.assertAlmostEqual(result.terminal_load_factor, 60.)

    def test_singular_equilibrium_continues_to_all_plies_failed(self):
        model = Laminate.from_json(ROOT / "examples/glass_cross_ply.json")
        result = analyze(replace(model, reference_load=(1/6, 5/6, 0, 0, 0, 0)))
        self.assertEqual(result.status, "all_plies_failed")
        self.assertAlmostEqual(result.ultimate_load_factor, 1034.)
        self.assertEqual(result.final_states, ("fully_failed",)*3)
        self.assertTrue(any(point.stiffness_rank == 4 for point in result.history))

    def test_singular_solver_handles_rotated_zero_stiffness_modes(self):
        model = laminate((45,), load=(.5, .5, .5, 0, 0, 0))
        abd, _ = assemble_abd(model, ("transverse_failed",))
        response = solve_response(abd, model.reference_load, 1., allow_singular=True)
        np.testing.assert_allclose(abd @ response, model.reference_load, atol=1e-12)
        np.testing.assert_allclose(response[:3], np.full(3, 2/(3*material().stiffness()[0, 0])), atol=1e-14)

    def test_singular_solver_rejects_load_in_nullspace(self):
        model = laminate()
        abd, _ = assemble_abd(model, ("transverse_failed",))
        for factor in (1e-9, 1., 1e9):
            with self.assertRaises(UnsupportedLoad):
                solve_response(abd, np.array([1, 1, 0, 0, 0, 0])*factor, 1., allow_singular=True)
            response = solve_response(abd, np.array([1, 0, 0, 0, 0, 0])*factor, 1., allow_singular=True)
            np.testing.assert_allclose(abd @ response, np.array([1, 0, 0, 0, 0, 0])*factor, atol=1e-12)

    def test_indefinite_matrix_not_accepted_by_singular_solver(self):
        with self.assertRaises(StiffnessLoss):
            solve_response(np.diag([1., 1., 1., 1., 1., -1.]), (1, 0, 0, 0, 0, 0),
                           1., allow_singular=True)


class InputAndOutputTests(unittest.TestCase):
    def test_invalid_material_parameters(self):
        for changes in ({"E1": 0}, {"G12": -1}, {"Xt": -1}, {"nu12": float("nan")},
                        {"S": True}, {"E2": float("inf")}, {"E1": 135000, "E2": 10000, "nu12": .3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                material(**changes)

    def test_invalid_model_parameters(self):
        for load in ((0,)*6, (1, 2), (float("nan"), 0, 0, 0, 0, 0), (True, 0, 0, 0, 0, 0)):
            with self.subTest(load=load), self.assertRaises(ValueError):
                laminate(load=load)
        with self.assertRaises(ValueError):
            laminate(angles=())
        with self.assertRaises(ValueError):
            laminate(thickness=-1)
        with self.assertRaises(ValueError):
            Laminate({"m": material()}, (Ply("unknown", 1, 0),), (1, 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            Laminate.from_dict({"materials": {}, "plies": [], "reference_load": [1]*6, "typo": 1})

    def test_material_error_identifies_name(self):
        data = json.loads((ROOT / "examples/tension.json").read_text(encoding="utf-8"))
        data["materials"]["demo"]["nu12"] = 1
        with self.assertRaisesRegex(ValueError, "demo.*非正定"):
            Laminate.from_dict(data)

    def test_singular_initial_stiffness_is_an_input_error(self):
        with self.assertRaises(StiffnessLoss):
            solve_response(np.zeros((6, 6)), (1, 0, 0, 0, 0, 0), 1.)
        with self.assertRaisesRegex(ValueError, "初始层合板"):
            analyze(laminate(mat=material(G12=1e-12)))

    def test_report_csv_and_plot_end_to_end(self):
        from matplotlib.image import imread
        result = analyze(Laminate.from_json(ROOT / "examples/cascade.json"))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            write_outputs(result, target)
            report = (target / "report.txt").read_text(encoding="utf-8")
            self.assertIn("首层失效：λ=40", report)
            self.assertIn("整体失效：λ=40", report)
            self.assertIn("连续失效：是", report)
            self.assertIn("-nu12/E2", report)
            self.assertIn("z 轴向下为正", report)
            self.assertIn("铺层（由上至下）", report)
            with (target / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(result.history))
            self.assertEqual([len(json.loads(row["events"])) for row in rows], [0, 2, 2])
            first_event = json.loads(rows[1]["events"])[0]
            self.assertEqual(first_event["ply"], 1)
            self.assertEqual(first_event["surface"], "top")
            self.assertAlmostEqual(first_event["z"], -2.)
            self.assertAlmostEqual(float(rows[-1]["epsilon_x0"]), .002)
            image = imread(target / "load_strain.png")
            self.assertGreater(image.shape[0], 500)
            self.assertGreater(float(np.std(image)), .05)

    def test_no_root_outputs_are_explicit(self):
        m = material(Xt=1, Xc=1, Yt=10, Yc=10, nu12=0)
        result = analyze(laminate(load=(1, 2, 0, 0, 0, 0), mat=m))
        with tempfile.TemporaryDirectory() as temporary:
            write_outputs(result, temporary)
            report = (Path(temporary) / "report.txt").read_text(encoding="utf-8")
            self.assertIn("未得到有限失效载荷", report)

    def test_cli_rejects_bad_input_without_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary)/"bad.json", Path(temporary)/"output"
            source.write_text('{"materials": {}}', encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                status = main([str(source), "--output", str(target)])
            self.assertEqual(status, 1)
            self.assertIn("计算失败", error.getvalue())
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
