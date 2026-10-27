"""Analytic checks for the conditional pixel response and acceptance accounting."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_pixel_posterior as posterior
from analysis.efficiency import laursen_162_pixel_study as study


def linear_tables():
    materials = {}
    e = np.linspace(0., 100., 1001)
    for name, stopping in (('G4_B', 1000.), ('G4_C', 900.), ('G4_Si', 800.)):
        materials[name] = dict(forward=PchipInterpolator(e, e/stopping, extrapolate=False),
            inverse=PchipInterpolator(e/stopping, e, extrapolate=False),
            light=PchipInterpolator(e, .9*e, extrapolate=False),
            inverse_light=PchipInterpolator(.9*e, e, extrapolate=False))
    return SimpleNamespace(materials=materials, table={'materials': {'G4_Si': {'density_g_cm3': 2.33}}},
        remove=lambda kinetic, column, name: materials[name]['inverse'](np.maximum(0., materials[name]['forward'](kinetic)-column)),
        restore=lambda kinetic, column, name: materials[name]['inverse'](materials[name]['forward'](kinetic)+column))


class PixelPosteriorTests(unittest.TestCase):
    def engine(self, **kwargs):
        return posterior.PixelPosterior(linear_tables(), posterior.Quadrature(main_power=7, coarse_power=5, **kwargs))

    def inputs(self):
        matched = study.matching.allocate(1)
        matched['detector'][0] = [0, 1, -1]; matched['pixel'][0] = [[0, 7], [8, 8], [0, 0]]
        matched['multiplicity'][0] = 2; matched['energy'][0] = [2., 3., 0.]
        raw = np.zeros((1, 2, 2, 16))
        raw[0, 0, 0, 0] = raw[0, 0, 1, 7] = 2.
        raw[0, 1, 0, 8] = raw[0, 1, 1, 8] = 3.
        return matched, raw

    def test_kinematic_ceiling_has_exact_two_alpha_remainder(self):
        rng = np.random.default_rng(27)
        direction = study.response.unit(rng.normal(size=(100, 3)))
        kinetic = posterior.kinetic_ceiling(direction); mass = study.response.MASS
        nominal = study.energy.nominal_parent(1)
        outgoing = np.sqrt(kinetic*(kinetic+2*mass))[:, None]*direction
        rest_e = nominal['parent_energy'][0]-(kinetic+mass)
        rest_p = nominal['parent_momentum'][0]-outgoing
        np.testing.assert_allclose(rest_e**2-np.sum(rest_p*rest_p, axis=-1), (2*mass)**2, rtol=0, atol=1e-6)
        self.assertTrue(np.all((kinetic > 5.) & (kinetic < 7.)))
        with self.assertRaises(ValueError):posterior.kinetic_ceiling(np.zeros(3))

    def test_calibration_roundtrip_uses_only_observed_energies(self):
        table = linear_tables(); charge = np.array([[.22, .25], [5.2, 5.3]])
        calibrated = table.materials['G4_Si']['inverse_light'](charge)
        np.testing.assert_allclose(posterior.recover_charge(calibrated, table), charge, atol=1e-12)

    def test_eligibility_excludes_neighbors_reuse_and_fractional_pixels(self):
        matched, raw = self.inputs()
        np.testing.assert_array_equal(posterior.eligible_hits(matched, raw, .1), [[True, False, False]])
        raw[0, 0, 1, 8] = .25
        self.assertFalse(posterior.eligible_hits(matched, raw, .1).any())
        raw[0, 0, 1, 8] = 0.
        matched['pixel'][0, 0, 1] = 7.2
        self.assertFalse(posterior.eligible_hits(matched, raw, .1).any())
        matched['pixel'][0, 0, 1] = 7.
        matched['detector'][0, 1] = 0; matched['pixel'][0, 1, 0] = 0.
        self.assertFalse(posterior.eligible_hits(matched, raw, .1).any())

    def test_truncated_normal_mass_and_first_moment(self):
        sigma = .04; mean = 1.; maximum = np.array([1.])
        nodes, weight = posterior.truncated_charge_nodes(np.array([[mean, mean]]), np.ones((1, 2)), maximum, sigma, 96)
        self.assertAlmostEqual(weight.sum(), .5/np.sqrt(2), places=12)
        expected = mean-sigma/np.sqrt(2)*np.sqrt(2/np.pi)
        self.assertAlmostEqual(float((nodes*weight).sum()/weight.sum()), expected, places=6)
        self.assertTrue(np.all(nodes <= maximum[0]))

    def test_charge_mismatch_and_zero_charge_have_correct_likelihood_penalty(self):
        measured = np.array([[1., 1.], [1.03, .97]])
        _, weight = posterior.truncated_charge_nodes(measured, np.ones((1, 2)), np.array([6.]), .03, 12)
        self.assertAlmostEqual(weight[1].sum()/weight[0].sum(), np.exp(-1.), places=12)
        _, opaque = posterior.truncated_charge_nodes(measured, np.ones((1, 2)), np.array([0.]), .03, 12)
        self.assertEqual(opaque.sum(), 0.)

    def test_inverse_jacobian_includes_all_three_layers(self):
        engine = self.engine(); charge = np.array([[[.5, 1.], [2., 3.]]])
        columns = (np.array([.0001, .0002]), np.array([.0002, .0003]), np.array([.0003, .0004]))
        kinetic, derivative = engine.inverse_energy_jacobian(charge, columns)
        shift = 1000*columns[0]+900*columns[1]+800*columns[2]
        np.testing.assert_allclose(kinetic, charge/.9+shift[None, :, None], atol=1e-12)
        np.testing.assert_allclose(derivative, 1/.9, atol=1e-12)

    def test_subthreshold_neighbor_censoring_penalizes_split_rays(self):
        engine = self.engine(noise_order=48)
        geometry = dict(columns=tuple(np.zeros(2) for _ in range(3)), ceiling=np.ones(2)*6.,
            maximum=np.ones(2)*5.4, fractions=np.array([[1., 1.], [.7, .7]]), geometry_weight=np.ones(2))
        inferred = engine.integrate(np.array([[1., 1.]]), geometry, 0)[1]
        # The .7 rays would create ~0.43 MeV neighbors: absent neighbors make
        # their posterior probability negligible at the D1 0.2 MeV threshold.
        self.assertAlmostEqual(inferred['mean'][0], 1/.9, places=5)
        self.assertLess(inferred['effective_rays'][0], 1.0001)

    def test_zero_evidence_is_reported_without_inventing_energy(self):
        engine = self.engine()
        geometry = dict(columns=tuple(np.zeros(2) for _ in range(3)), ceiling=np.ones(2)*6.,
            maximum=np.zeros(2), fractions=np.ones((2, 2)), geometry_weight=np.ones(2))
        result = engine.integrate(np.array([[1., 1.]]), geometry, 0)
        for row in result:
            self.assertFalse(row['valid'][0]); self.assertTrue(np.isnan(row['mean'][0]))
            self.assertEqual(row['effective_rays'][0], 0.)

    def test_estimate_is_batch_order_invariant_and_prior_sensitive(self):
        engine = self.engine(); values = np.array([[1., 1.01], [2., 1.99], [3., 3.]])
        d = np.zeros(3, int); pixels = np.array([[0, 7], [0, 7], [0, 15]])
        first = engine.estimate(d, pixels, values, .1)
        second = engine.estimate(d[::-1], pixels[::-1], values[::-1], .1)
        self.assertTrue(first['valid'].all())
        for key in first:np.testing.assert_allclose(first[key], second[key][::-1], atol=1e-12)
        self.assertTrue(np.all(first['mean'][:, 2] > first['mean'][:, 1]))
        self.assertTrue(np.all(first['std'] > 0))

    def test_reconstruction_baseline_matches_existing_selection(self):
        matched, _ = self.inputs(); table = linear_tables(); nominal = study.energy.nominal_parent(1)
        direction, columns = study.decomposition.nominal_measurements(matched, table)
        kinetic = study.decomposition.corrected_energy(matched['energy'], columns, table)
        labels, residual = study.evaluate(matched, kinetic, direction, nominal)
        _, _, old_labels, cm = study.matching.reconstruct(matched, nominal, table)
        np.testing.assert_array_equal(labels, old_labels[:, :, 1])
        np.testing.assert_allclose(residual, study.energy.energy_residual(cm, nominal), atol=1e-10)
        self.assertGreater(abs(residual[0]), .001)  # No energy closure is fitted.

    def test_fixed_cohort_retains_losses_and_paired_uncertainty(self):
        w = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        r = np.repeat(np.array([.2, .6, -1.1, .1])[:, None], len(study.CASES), axis=1)
        r[:, 2] = [.5, .1, -.1, .3]
        item = study.FixedCohort(2); item.add(w, np.zeros(4, int), r)
        value = item.finish(w.sum(axis=0)); keep = (abs(r) <= .4).astype(float)
        for model in range(2):
            weight = w[:, model]; change = keep-keep[:, :1]; mean = weight@change/weight.sum()
            error = np.sqrt((weight*weight)@((change-mean)**2))/weight.sum()
            np.testing.assert_allclose(value['rms_MeV'][model, 0]**2, weight@(r*r)/weight.sum())
            np.testing.assert_allclose(value['paired_retention_change'][model, 0], mean)
            np.testing.assert_allclose(value['paired_retention_standard_error'][model, 0], error)
        self.assertFalse(value['valid'][:, 1:].any())

    def test_state_codes_reject_nonexclusive_classification(self):
        values = np.zeros((5, 2, 2)); values[1:] = np.eye(4).reshape(4, 2, 2)
        np.testing.assert_array_equal(study.state_codes(values), np.arange(5))
        values[0] = 1.
        with self.assertRaises(ValueError):study.state_codes(values)


if __name__ == '__main__':
    unittest.main()
