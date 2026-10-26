"""Integration checks for channel formation and matching-study statistics."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_matching_study as study
from analysis.efficiency import laursen_162_strip_matching as matching


class MatchingStudyTests(unittest.TestCase):
    def setUp(self):
        # Identity calibration isolates strip addition, noise and threshold logic.
        self.tables = SimpleNamespace(materials={'G4_Si': {'inverse_light': lambda energy: energy}})

    def test_charge_is_summed_before_either_side_threshold(self):
        detector = np.array([[0, 0, -1]])
        coordinates = np.array([[[3.5, 7.5], [3.5, 7.5], [0., 0.]]])
        observed = study.form_signals(detector, coordinates, np.array([[.12, .12, 0.]]),
                                      np.zeros((1, 2, 2, 16)), 0., self.tables)
        self.assertAlmostEqual(observed[0, 0, 0, 3], .24)
        self.assertAlmostEqual(observed[0, 0, 1, 7], .24)
        got = study.match_signals(observed)['sum_first']
        self.assertEqual(got['multiplicity'][0], 1)
        self.assertAlmostEqual(got['energy'][0, 0], .24)

    def test_split_below_threshold_is_not_recovered_with_truth(self):
        detector = np.array([[0, -1, -1]])
        coordinates = np.array([[[4., 7.5], [0., 0.], [0., 0.]]])
        observed = study.form_signals(detector, coordinates, np.array([[.30, 0., 0.]]),
                                      np.zeros((1, 2, 2, 16)), .1, self.tables)
        self.assertEqual(np.count_nonzero(observed[0, 0, 0]), 0)
        self.assertEqual(np.count_nonzero(observed[0, 0, 1]), 1)
        result = study.match_signals(observed)['share_first']
        self.assertEqual(result['multiplicity'][0], 0)
        self.assertTrue(result['unmatched'][0])

    def test_signal_formation_is_invariant_to_particle_storage_order(self):
        detector = np.array([[0, 0, 1]])
        coordinates = np.array([[[3.995, 7.5], [4.005, 9.5], [2.5, 5.0]]])
        light = np.array([[1., 2., 3.]])
        noise = np.random.default_rng(7201).standard_normal((1, 2, 2, 16))
        old = study.form_signals(detector, coordinates, light, noise, .1, self.tables)
        order = [2, 0, 1]
        new = study.form_signals(detector[:, order], coordinates[:, order], light[:, order], noise, .1, self.tables)
        np.testing.assert_allclose(old, new, atol=1e-14)
        for p in matching.POLICIES:
            left, right = study.match_signals(old)[p], study.match_signals(new)[p]
            for key in left:
                np.testing.assert_allclose(left[key], right[key], atol=1e-12)

    def test_side_noise_reproduces_declared_average_resolution(self):
        n = 30000
        detector = np.tile([0, -1, -1], (n, 1))
        coordinates = np.tile([[3.5, 7.5], [0., 0.], [0., 0.]], (n, 1, 1))
        light = np.tile([3., 0., 0.], (n, 1))
        noise = np.random.default_rng(5567).standard_normal((n, 2, 2, 16))
        observed = study.form_signals(detector, coordinates, light, noise, 0., self.tables)
        average = .5*(observed[:, 0, 0, 3]+observed[:, 0, 1, 7])
        self.assertAlmostEqual(average.mean(), 3., delta=.0004)
        self.assertAlmostEqual(average.std(ddof=1), .04/2.354820045, delta=.0004)

    def test_unoccupied_channels_do_not_simulate_noise_only_background(self):
        observed = study.form_signals(np.full((1, 3), -1), np.zeros((1, 3, 2)),
                                      np.zeros((1, 3)), np.full((1, 2, 2, 16), 100.), 0., self.tables)
        np.testing.assert_array_equal(observed, 0.)

    def test_streaming_importance_errors_match_direct_paired_calculation(self):
        rng = np.random.default_rng(665)
        weights = rng.uniform(.1, 4., (37, 3))
        values = rng.integers(-1, 2, (37, 2))
        accumulator = study.Moments(3, (2,))
        for start in range(0, len(weights), 7):
            accumulator.add(weights[start:start+7], values[start:start+7])
        got = accumulator.finish(weights.sum(axis=0), (weights*weights).sum(axis=0))
        mean, error = study.audit.moments(weights, values)
        np.testing.assert_allclose(got['value'], mean, atol=1e-14)
        np.testing.assert_allclose(got['standard_error'], error, atol=1e-14)

    def test_global_assignment_does_not_commit_first_available_pair(self):
        result = matching.match([[1, 1.0], [8, 1.11]], [[2, 1.10], [10, .90]], direct_only=True)['sum_first']
        self.assertEqual(len(result.hits), 2)
        self.assertEqual(result.unmatched, 0)
        self.assertEqual({(f, b) for e, f, b in result.hits}, {(1., 10.), (8., 2.)})

    def test_one_shared_hit_can_mimic_two_ground_channel_alphas(self):
        # One true 3-MeV charge signal split equally between adjacent back
        # strips can be read as two nearly parallel 1.5-MeV particles.
        observed = study.form_signals(np.array([[0, -1, -1]]),
            np.array([[[4.5, 3.0], [0., 0.], [0., 0.]]]), np.array([[3., 0., 0.]]),
            np.zeros((1, 2, 2, 16)), .1, self.tables)
        matched = study.match_signals(observed)
        energy = np.geomspace(.001, 10., 160)
        rows = np.column_stack([energy, np.full(len(energy), 500.), np.full(len(energy), 5.)])
        tables = study.response.RangeTables(dict(materials={name: dict(rows=rows, density_g_cm3=2.3)
            for name in ('G4_B', 'G4_C', 'G4_Si')}))
        prepared = dict(parent_energy=np.array([3*study.response.MASS+8.84]),
                        parent_momentum=np.array([[0., 0., 17.44]]))
        selected, _, classified, _ = study.reconstruct(matched['sum_first'], prepared, tables)
        self.assertEqual(selected[0, 0], 1.)
        self.assertEqual(classified[0, 0, 1, 0], 1.)
        self.assertEqual(matched['share_first']['multiplicity'][0], 1)
        accepted, _, _, _ = study.reconstruct(matched['share_first'], prepared, tables)
        np.testing.assert_array_equal(accepted, 0.)


if __name__ == '__main__':
    unittest.main()
