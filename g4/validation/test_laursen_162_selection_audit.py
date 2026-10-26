"""Physical and regression checks for the thesis selection controls."""
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_selection_audit as audit
from analysis.efficiency import laursen_162_response as response


class SelectionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        energy = np.geomspace(.001, 10., 160)
        rows = np.column_stack([energy, np.full(len(energy), 500.), np.full(len(energy), 5.)])
        cls.tables = response.RangeTables(dict(materials={name: dict(rows=rows, density_g_cm3=2.3)
            for name in ('G4_B', 'G4_C', 'G4_Si')}))
        rng = np.random.default_rng(16290)
        # Closed, non-collinear CM triangle; independent isotropic orientations.
        unit = np.array([[1., 0., 0.], [-.5, np.sqrt(3)/2, 0.], [-.5, -np.sqrt(3)/2, 0.]])
        directions = np.einsum('nij,pj->npi', Rotation.random(1200, random_state=rng).as_matrix(), unit)
        cls.reference = dict(kinetic=np.full((1200, 3), 2.9), directions=directions,
            origins=np.zeros((1200, 3)), depth=rng.random(1200), noise=rng.standard_normal((1200, 3)))

    def test_relative_energy_ground_state_and_common_velocity(self):
        p = np.sqrt(response.MASS * .0918)
        momentum = np.array([[[p, 0., 0.], [-p, 0., 0.], [0., 0., 100.]]])
        relative = audit.pair_relative_energy(momentum)
        self.assertAlmostEqual(relative[0, 0], .0918, places=12)
        np.testing.assert_allclose(relative, audit.pair_relative_energy(momentum + [32., -9., 4.]))
        for order in itertools.permutations(range(3)):
            np.testing.assert_allclose(np.sort(relative), np.sort(audit.pair_relative_energy(momentum[:, order])))

    def test_plane_is_permutation_invariant_including_equal_energies(self):
        p = np.array([[[200., 0., 0.], [-100., 20., 0.], [-90., -20., 8.]],
                      [[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]]])
        expected = audit.plane_deviation(p, 'highest_energy')
        for order in itertools.permutations(range(3)):
            np.testing.assert_allclose(expected, audit.plane_deviation(p[:, order], 'highest_energy'))
        self.assertAlmostEqual(expected[1], np.pi/2)

    def test_plane_rejects_undefined_normal_and_preserves_closed_plane(self):
        p = np.array([[[2., 0., 0.], [-1., 0., 0.], [-1., 0., 0.]],
                      [[2., 0., 0.], [-1., 1., 0.], [-1., -1., 0.]]])
        np.testing.assert_allclose(audit.plane_deviation(p, 'highest_energy'), [np.pi/2, 0.])

    def test_equation_4_1_rays_and_pixel_centres(self):
        # Independent coordinates from the thesis equation, theta_V=0.
        beta = np.deg2rad(105.)
        xprime, yprime = -10.9375, 7.8125
        point = np.array([19.4-xprime*np.cos(beta), -yprime, -4.5-xprime*np.sin(beta)])
        rays = np.broadcast_to(point/np.linalg.norm(point), (1, 3, 3)).copy()
        got = audit.trace(rays, np.zeros((1, 3)), 'equation_4_1')
        np.testing.assert_array_equal(got['detector'], [[0, 0, 0]])
        np.testing.assert_allclose(got['distance'], np.linalg.norm(point))
        np.testing.assert_array_equal(got['pixel'][0, 0], [4, 10])
        np.testing.assert_allclose(got['direction'], rays, atol=1e-14)

    def test_solid_angle_against_independent_surface_quadrature(self):
        normals, horizontal, vertical = audit.basis('equation_4_1')
        nodes, weights = np.polynomial.legendre.leggauss(64)
        x, y = np.meshgrid(nodes*25., nodes*25.)
        quadrature = weights[:, None]*weights[None, :]*25.**2
        for centre, n, h in zip(response.CENTRES, normals, horizontal):
            surface = centre+x[..., None]*h+y[..., None]*vertical
            integrated = np.sum(quadrature*abs(centre@n)/np.linalg.norm(surface, axis=-1)**3)
            self.assertAlmostEqual(integrated, response.rectangle_solid_angle(centre, n, h), places=11)

    def test_full_legacy_replay_matches_frozen_response(self):
        r = self.reference
        old, _ = response.fold(r['kinetic'], r['directions'], r['origins'], r['depth'], r['noise'], response.Response(), self.tables)
        new, _ = audit.fold(r, self.tables, audit.Selection())
        for before, after in zip(old, new):
            np.testing.assert_allclose(before['probability'], after['stages'][:, -1], atol=1e-14)
            np.testing.assert_allclose(before['preselection'], after['stages'][:, 0], atol=1e-14)

    def test_full_selection_and_channel_gate_do_not_depend_on_storage_order(self):
        selection = next(c for c in audit.cases() if c.name == 'combined')
        def counts(r):
            records, _ = audit.fold(r, self.tables, selection)
            result = np.zeros((len(r['kinetic']), 2, 2))
            for record in records:
                k = record['multiplicity']-2
                p = record['stages'][:, -1]
                ground = record['min_pair_energy'] < audit.GATES[1, k]
                result[:, k, 0] += p*ground
                result[:, k, 1] += p*~ground
            return result
        expected = counts(self.reference)
        self.assertTrue(np.all(expected.sum(axis=(1, 2)) <= 1))
        self.assertGreater(expected.sum(), 50)
        for order in itertools.permutations(range(3)):
            r = {key: value[:, order] if key in ('kinetic', 'directions', 'noise') else value
                 for key, value in self.reference.items()}
            np.testing.assert_allclose(expected, counts(r), atol=1e-12)

    def test_paired_difference_uncertainty_vanishes_for_identical_replay(self):
        mean, error = audit.moments(np.array([[1., 2.], [1., 3.]]), np.zeros((2, 2)))
        np.testing.assert_array_equal(mean, 0.)
        np.testing.assert_array_equal(error, 0.)

    def test_component_cut_is_additional_to_magnitude_cut(self):
        p = np.array([[[200., 0., 0.], [-85., 75., 0.], [-84., -75., 0.]]])
        energy = np.sqrt(np.sum(p*p, axis=-1)+response.MASS**2)-response.MASS
        inputs = (energy, p, np.zeros((1, 3), bool), np.ones(3, bool), np.array([[0, 1, 1]]))
        basic = audit.selection_masks(*inputs, audit.Selection(plane='highest_energy'))
        extra = audit.selection_masks(*inputs, audit.Selection(plane='highest_energy', px_limit_MeV=30.))
        self.assertTrue(basic[0, -1])
        self.assertTrue(extra[0, 4])
        self.assertFalse(extra[0, 5])

    def test_one_each_detector_control_is_only_for_double_coincidences(self):
        p = np.array([[[2., 0., 0.], [-1., 1., 0.], [-1., -1., 0.]]]*2)
        energy = np.full((2, 3), 1.)
        detector = np.array([[0, 0, 1], [0, 1, 1]])
        selection = audit.Selection(plane='highest_energy', one_each_detector=True)
        double = audit.selection_masks(energy, p, np.zeros((2, 3), bool), np.array([True, True, False]), detector, selection)
        triple = audit.selection_masks(energy, p, np.zeros((2, 3), bool), np.ones(3, bool), detector, selection)
        np.testing.assert_array_equal(double[:, -1], [False, True])
        np.testing.assert_array_equal(triple[:, -1], [True, True])


if __name__ == '__main__':
    unittest.main()
