"""Strip-sum identifiability controls independent of response efficiencies."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_readout_audit as audit


class ReadoutAuditTests(unittest.TestCase):
    def test_one_side_summing_can_recover_both_energies(self):
        # Two particles share a front strip but use distinct back strips.
        detector = np.array([0, 0])
        pixel = np.array([[2, 3], [2, 4]])
        matrix = audit.channel_matrix(detector, pixel)
        energy = np.array([1.5, 2.8])
        np.testing.assert_allclose(np.linalg.lstsq(matrix, matrix @ energy, rcond=None)[0], energy)
        kind = audit.strip_topology([[0, 0, -1]], [[[2, 3], [2, 4], [-1, -1]]], [1, 1, 0])
        self.assertEqual(kind[0], 1)

    def test_same_pixel_leaves_energy_split_undetermined(self):
        matrix = audit.channel_matrix([0, 0], [[2, 3], [2, 3]])
        np.testing.assert_array_equal(matrix @ np.array([1., 3.]), matrix @ np.array([2., 2.]))
        self.assertEqual(np.linalg.matrix_rank(matrix), 1)
        kind = audit.strip_topology([[0, 0, -1]], [[[2, 3], [2, 3], [-1, -1]]], [1, 1, 0])
        self.assertEqual(kind[0], 4)

    def test_three_particle_cross_sums_are_not_a_same_pixel(self):
        pixel = [[0, 0], [0, 1], [1, 1]]
        matrix = audit.channel_matrix([0, 0, 0], pixel)
        self.assertEqual(np.linalg.matrix_rank(matrix), 3)
        kind = audit.strip_topology([[0, 0, 0]], [pixel], [1, 1, 1])
        self.assertEqual(kind[0], 3)

    def test_strip_labels_in_different_detectors_are_independent(self):
        matrix = audit.channel_matrix([0, 1], [[0, 0], [0, 0]])
        self.assertEqual(np.linalg.matrix_rank(matrix), 2)
        kind = audit.strip_topology([[0, 1, -1]], [[[0, 0], [0, 0], [-1, -1]]], [1, 1, 0])
        self.assertEqual(kind[0], 0)

    def test_exhaustive_three_hit_rank_and_permutation_control(self):
        # For at most three known edges of a bipartite front/back graph,
        # a rank deficit requires repeated edges (the same physical pixel).
        pixels = list(itertools.product(range(2), repeat=2))
        for pixel in itertools.product(pixels, repeat=3):
            matrix = audit.channel_matrix([0, 0, 0], pixel)
            expected_rank = len(set(pixel))
            self.assertEqual(np.linalg.matrix_rank(matrix), expected_rank)
            expected = audit.strip_topology([[0, 0, 0]], [pixel], [1, 1, 1])[0]
            self.assertEqual(expected == 4, expected_rank < 3)
            for order in itertools.permutations(range(3)):
                reordered = np.asarray(pixel)[list(order)]
                self.assertEqual(audit.strip_topology([[0, 0, 0]], [reordered], [1, 1, 1])[0], expected)


if __name__ == '__main__':
    unittest.main()
