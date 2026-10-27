"""Independent kinematics, signal-ancestry and energy-window checks."""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_energy_closure as study


class EnergyClosureTests(unittest.TestCase):
    def test_exact_three_body_event_closes_for_two_and_three_detections(self):
        parent = study.nominal_parent(1)
        q = study.available_energy(**parent)[0]
        mass = study.response.MASS
        directions = np.array([[[1., 0., 0.], [-.5, np.sqrt(3)/2, 0.], [-.5, -np.sqrt(3)/2, 0.]]])
        momentum = directions*np.sqrt(q/3*(q/3+2*mass))
        beta = parent['parent_momentum']/parent['parent_energy'][:, None]
        energy, lab = study.response.boost(np.full((1, 3), mass+q/3), momentum, beta[:, None])
        unit = study.response.unit(lab)
        for pattern in (np.array([True, True, False]), np.ones(3, bool)):
            cm, _ = study.response.reconstruct(energy-mass, unit, **parent, pattern=pattern)
            np.testing.assert_allclose(study.energy_residual(cm, parent), 0., atol=8e-12)

    def test_momentum_reconstruction_does_not_force_energy_conservation(self):
        parent = study.nominal_parent(1)
        energy = np.array([[1., 1., 0.]])
        direction = np.array([[[0., 0., 1.]]*3])
        cm, _ = study.response.reconstruct(energy, direction, **parent, pattern=np.array([True, True, False]))
        residual = study.energy_residual(cm, parent)
        self.assertGreater(abs(residual[0]), .8)
        self.assertFalse(study.window_masks(residual)[0, 1])

    def test_false_collinear_pair_can_also_satisfy_energy_closure(self):
        # Splitting one charge into two close directions is not always ruled
        # out by a total-energy cut: construct an on-shell counterexample.
        mass, q = study.response.MASS, 8.83
        kinetic = brentq(lambda t: 2*t+np.sqrt(mass*mass+4*t*(t+2*mass))-mass-q, 0., q/2)
        parent = dict(parent_energy=np.array([3*mass+q]), parent_momentum=np.zeros((1, 3)))
        cm, momentum = study.response.reconstruct(np.array([[kinetic, kinetic, 0.]]),
            np.array([[[0., 0., 1.]]*3]), **parent, pattern=np.array([True, True, False]))
        np.testing.assert_allclose(study.energy_residual(cm, parent), 0., atol=5e-12)
        self.assertLess(study.matching.audit.pair_relative_energy(momentum).min(), 1e-12)
        self.assertTrue(study.window_masks(study.energy_residual(cm, parent))[0, 4])

    def test_nominal_parent_uses_only_declared_beam_inputs(self):
        first, second = study.nominal_parent(1), study.nominal_parent(7)
        np.testing.assert_array_equal(second['parent_energy'], np.repeat(first['parent_energy'], 7))
        np.testing.assert_array_equal(second['parent_momentum'], np.repeat(first['parent_momentum'], 7, axis=0))
        with self.assertRaises(ValueError):study.nominal_parent(-1)
        with self.assertRaises(ValueError):study.available_energy(np.ones(1), np.zeros((1, 3)))

    def test_window_nesting_offsets_and_nonfinite_rejection(self):
        residual = np.array([-.49, -.4, -.2, -.1, 0., .1, .2, .4, .49, np.nan, np.inf])
        masks = study.window_masks(residual)
        self.assertTrue(np.all(np.diff(masks[:, :5].astype(int), axis=1) <= 0))
        self.assertTrue(masks[0, 5]); self.assertFalse(masks[0, 6])
        self.assertFalse(masks[8, 5]); self.assertTrue(masks[8, 6])
        self.assertFalse(masks[-2:].any())

    def signals(self, detector, coordinate, light, width):
        tables = SimpleNamespace(materials={'G4_Si': {'inverse_light': lambda value: value}})
        observed = study.matching.form_signals(detector, coordinate, light,
            np.zeros((len(detector), 2, 2, 16)), width, tables)
        return study.matching.match_signals(observed), study.strip_ancestry(detector, coordinate, light, width)

    def test_one_shared_particle_and_two_summed_particles_have_different_ancestry(self):
        detector = np.array([[0, -1, -1], [0, 0, -1]])
        coordinate = np.array([[[4.5, 3.], [0., 0.], [0., 0.]],
                               [[4.5, 2.5], [4.5, 3.5], [0., 0.]]])
        light = np.array([[3., 0., 0.], [1.5, 1.5, 0.]])
        matches, ancestry = self.signals(detector, coordinate, light, .1)
        np.testing.assert_array_equal(study.association_group(matches['sum_first'], ancestry), [1, 0])
        np.testing.assert_array_equal(study.association_group(matches['share_first'], ancestry), [0, 2])

    def test_same_pixel_sum_is_not_called_one_to_one(self):
        matches, ancestry = self.signals(np.array([[0, 0, -1]]),
            np.array([[[4.5, 3.5], [4.5, 3.5], [0., 0.]]]), np.array([[1., 2., 0.]]), 0.)
        self.assertEqual(study.association_group(matches['sum_first'], ancestry)[0], 2)

    def test_wrong_front_back_cross_has_no_common_ancestor(self):
        detector = np.array([[0, 0, -1]])
        coordinate = np.array([[[4.5, 3.5], [8.5, 9.5], [0., 0.]]])
        ancestry = study.strip_ancestry(detector, coordinate, np.array([[1., 2., 0.]]), 0.)
        matched = study.matching.allocate(1)
        matched['multiplicity'][0] = 2
        matched['detector'][0, :2] = 0
        matched['pixel'][0, :2] = [[4., 9.], [8., 3.]]
        self.assertEqual(study.association_group(matched, ancestry)[0], 3)

    def test_ancestry_categories_are_invariant_to_true_particle_order(self):
        detector = np.array([[0, 0, 1]])
        coordinate = np.array([[[4.5, 2.5], [4.5, 3.5], [9.5, 5.5]]])
        light = np.array([[1., 2., 4.]])
        original, ancestry = self.signals(detector, coordinate, light, .1)
        expected = {p: study.association_group(m, ancestry) for p, m in original.items()}
        for order in itertools.permutations(range(3)):
            matches, reordered = self.signals(detector[:, order], coordinate[:, order], light[:, order], .1)
            for p, matched in matches.items():
                np.testing.assert_array_equal(study.association_group(matched, reordered), expected[p])

    def test_histogram_conserves_tails_and_matches_direct_importance_uncertainty(self):
        weights = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        selected = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 0.]])
        classified = np.zeros((4, 2, 2)); classified[0, 0, 0] = classified[1, 1, 1] = classified[2, 0, 0] = 1
        histogram = study.ResidualHistogram(2)
        histogram.add(weights, selected, classified, np.array([-5., .2, 8., 0.]), np.array([0, 1, 2, 3]))
        result = histogram.finish(weights.sum(axis=0), (weights*weights).sum(axis=0))
        np.testing.assert_allclose(result['value'].sum(axis=(2, 3, 4)), weights.T@selected/weights.sum(axis=0)[:, None])
        value = np.array([1., 0., 0., 0.])
        mean = weights.T@value/weights.sum(axis=0)
        error = np.sqrt(np.sum(weights*weights*(value[:, None]-mean)**2, axis=0))/weights.sum(axis=0)
        np.testing.assert_allclose(result['value'][:, 0, 0, 0, 0], mean)
        np.testing.assert_allclose(result['standard_error'][:, 0, 0, 0, 0], error)


if __name__ == '__main__':
    unittest.main()
