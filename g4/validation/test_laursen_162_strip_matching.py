"""Observable ambiguities and energy/position invariants for strip matching."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_strip_matching as matcher


class StripMatchingTests(unittest.TestCase):
    def test_isolated_hit_uses_average_and_pixel_intersection(self):
        for result in matcher.match([[4, 2.02]], [[9, 1.98]]).values():
            self.assertEqual(result.hits, ((2., 4., 9.),))
            self.assertFalse(result.ambiguous)
            self.assertEqual(result.unmatched, 0)

    def test_nonadjacent_one_side_sum_recovers_two_particles(self):
        for result in matcher.match([[4, 3.]], [[2, 1.], [7, 2.]]).values():
            self.assertEqual(result.hits, ((1., 4., 2.), (2., 4., 7.)))
            self.assertEqual(result.summed, 1)
            self.assertEqual(result.unmatched, 0)

    def test_adjacent_sum_and_sharing_are_observationally_indistinguishable(self):
        result = matcher.match([[4, 3.]], [[2, 1.], [3, 2.]])
        self.assertEqual(len(result['sum_first'].hits), 2)
        self.assertEqual(len(result['share_first'].hits), 1)
        np.testing.assert_allclose(result['share_first'].hits, [[3., 4., 8/3]])
        self.assertTrue(result['unique'].ambiguous)
        self.assertGreaterEqual(result['unique'].feasible, 2)
        self.assertEqual(result['unique'].hits, ())

    def test_same_pixel_sum_is_not_identifiable_as_two_hits(self):
        result = matcher.match([[4, 4.]], [[7, 4.]])['sum_first']
        # Both physical energy splits (1,3) and (2,2) give these same signals.
        self.assertEqual(result.hits, ((4., 4., 7.),))

    def test_equal_energy_cross_pairings_are_not_resolved_by_strip_order(self):
        for result in matcher.match([[1, 2.], [8, 2.]], [[2, 2.], [10, 2.]]).values():
            self.assertTrue(result.ambiguous)
            self.assertEqual(result.feasible, 2)
            self.assertEqual(result.hits, ())

    def test_near_equal_energies_distinguish_residual_choice_from_unique(self):
        result = matcher.match([[1, 2.], [8, 2.05]], [[2, 2.01], [10, 2.04]])
        self.assertFalse(result['sum_first'].ambiguous)
        self.assertTrue(result['unique'].ambiguous)
        np.testing.assert_allclose(result['sum_first'].hits, [[2.005, 1., 2.], [2.045, 8., 10.]])

    def test_unmatched_signal_does_not_erase_other_measured_particle(self):
        result = matcher.match([[1, 1.], [8, 3.]], [[2, 1.01]])['sum_first']
        np.testing.assert_allclose(result.hits, [[1.005, 1., 2.]])
        self.assertEqual(result.unmatched, 1)

    def test_three_edge_cross_sum_has_multiple_geometric_interpretations(self):
        # True edges could be (F1,B2)=1, (F1,B9)=2, (F8,B9)=4.
        for result in matcher.match([[1, 3.], [8, 4.]], [[2, 1.], [9, 6.]]).values():
            self.assertTrue(result.ambiguous)
            self.assertGreaterEqual(result.feasible, 2)

    def test_three_particles_sharing_a_strip_conserve_total_energy(self):
        result = matcher.match([[4, 6.04]], [[0, 1.], [5, 2.], [10, 3.]])['sum_first']
        self.assertEqual(len(result.hits), 3)
        self.assertAlmostEqual(sum(h[0] for h in result.hits), 6.02)
        self.assertEqual(result.unmatched, 0)

    def test_signal_permutation_front_back_swap_and_reflection(self):
        front, back = np.array([[1, 1.02], [8, 2.01]]), np.array([[2, 1.], [11, 2.]])
        expected = matcher.match(front, back)
        for fi in itertools.permutations(range(2)):
            for bi in itertools.permutations(range(2)):
                self.assertEqual(expected, matcher.match(front[list(fi)], back[list(bi)]))
        transposed = matcher.match(back, front)
        reflected = matcher.match(np.column_stack([15-front[:, 0], front[:, 1]]),
                                  np.column_stack([15-back[:, 0], back[:, 1]]))
        for policy in matcher.POLICIES:
            original = expected[policy].hits
            self.assertEqual(tuple(sorted((e, b, f) for e, f, b in original)), transposed[policy].hits)
            self.assertEqual(tuple(sorted((e, 15-f, 15-b) for e, f, b in original)), reflected[policy].hits)

    def test_ideal_charge_split_conserves_charge_at_boundary_and_edges(self):
        coordinate = np.array([0., .01, .5, .99, 1., 1.01, 15.99])
        first, second, fraction = matcher.split_charge(coordinate, width_mm=.1)
        np.testing.assert_array_equal(first, [0, 0, 0, 0, 0, 0, 15])
        np.testing.assert_array_equal(second, [0, 0, 0, 1, 1, 1, 15])
        self.assertEqual(fraction[4], .5)
        self.assertTrue(np.all((fraction >= 0) & (fraction <= 1)))
        np.testing.assert_allclose((1-fraction)*3.2+fraction*3.2, 3.2)

    def test_invalid_calibration_and_duplicate_strips_are_rejected(self):
        for bad in ([[1, -1]], [[1, 1], [1, 2]], [[16, 1]], [[1.5, 2]], [[2, np.nan]]):
            with self.assertRaises(ValueError):
                matcher.match(bad, [[1, 1]])
        with self.assertRaises(ValueError):
            matcher.match([[1, 1]], [[1, 1]], tolerance=0.)


if __name__ == '__main__':
    unittest.main()
