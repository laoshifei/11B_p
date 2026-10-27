"""Mapping, independent kinematic closure, and paired conditional statistics."""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_energy_decomposition as study


class EnergyDecompositionTests(unittest.TestCase):
    def match(self, detector, coordinate, light, width=.1):
        tables = SimpleNamespace(materials={'G4_Si': {'inverse_light': lambda x: x}})
        signals = study.matching.form_signals(detector, coordinate, light,
            np.zeros((len(detector), 2, 2, 16)), width, tables)
        return study.matching.match_signals(signals), study.energy.strip_ancestry(detector, coordinate, light, width)

    def test_mapping_recovers_particle_values_under_all_truth_permutations(self):
        detector = np.array([[0, 0, 1], [0, -1, 1]])
        coordinate = np.array([[[4.5, 2.5], [4.5, 3.5], [9.5, 5.5]],
                               [[4.5, 2.5], [0., 0.], [9.5, 5.5]]])
        light = np.array([[1., 2., 4.], [1., 0., 4.]])
        expected = None
        for order in itertools.permutations(range(3)):
            matched, ancestry = self.match(detector[:, order], coordinate[:, order], light[:, order])
            mapping = study.truth_permutation(matched['sum_first'], ancestry)
            values = np.asarray(order)[mapping]
            if expected is None:
                expected = values
            np.testing.assert_array_equal(values, expected)
            np.testing.assert_array_equal(np.sort(mapping, axis=1), [[0, 1, 2], [0, 1, 2]])

    def test_split_one_source_into_two_hits_is_rejected(self):
        matched, ancestry = self.match(np.array([[0, 1, -1]]),
            np.array([[[4.5, 3.], [9.5, 5.5], [0., 0.]]]), np.array([[3., 4., 0.]]))
        with self.assertRaises(ValueError):
            study.truth_permutation(matched['sum_first'], ancestry)
        mapping = study.truth_permutation(matched['share_first'], ancestry)
        np.testing.assert_array_equal(np.sort(mapping, axis=1), [[0, 1, 2]])

    def test_overlapping_sources_are_rejected_even_with_two_hits(self):
        matched, ancestry = self.match(np.array([[0, 0, 1]]),
            np.array([[[4.5, 3.5], [4.5, 3.5], [9.5, 5.5]]]), np.array([[1., 2., 4.]]))
        with self.assertRaises(ValueError):
            study.truth_permutation(matched['sum_first'], ancestry)

    def exact_event(self, n):
        nominal = study.energy.nominal_parent(n)
        mass = study.response.MASS; q = study.energy.available_energy(**nominal)[0]
        direction = np.array([[[1., 0., 0.], [-.5, np.sqrt(3)/2, 0.], [-.5, -np.sqrt(3)/2, 0.]]])
        momentum = np.broadcast_to(direction*np.sqrt(q/3*(q/3+2*mass)), (n, 3, 3)).copy()
        total, lab = study.response.boost(np.full((n, 3), mass+q/3), momentum,
            (nominal['parent_momentum']/nominal['parent_energy'][:, None])[:, None, :])
        return nominal, total-mass, study.response.unit(lab)

    def test_invalid_observed_energy_is_rejected_but_missing_slot_is_ignored(self):
        nominal, kinetic, direction = self.exact_event(3)
        kinetic[0, 2] = np.nan; kinetic[1, 0] = np.nan; kinetic[2, 2] = np.nan
        result = study.reconstruct_finite(kinetic, direction, np.array([2, 2, 3]), nominal)
        self.assertTrue(np.isfinite(result[0]).all())
        self.assertTrue(np.isnan(result[1:]).all())
        np.testing.assert_allclose(result[0].sum(), study.energy.available_energy(**nominal)[0], atol=1e-10)

    def test_all_true_intervention_closes_without_using_total_energy_constraint(self):
        # A transparent-material control uses an independently constructed,
        # exactly conserving event, with deliberately wrong baseline pixels.
        nominal, kinetic, direction = self.exact_event(2)
        tables = SimpleNamespace(restore=lambda value, column, material: value,
            table={'materials': {'G4_Si': {'density_g_cm3': 2.33}}})
        matched = study.matching.allocate(2)
        matched['multiplicity'][:] = [2, 3]
        matched['detector'][:] = [[0, 1, -1], [0, 0, 1]]
        matched['energy'][:] = kinetic
        matched['pixel'][:] = 7.
        source = dict(kinetic=kinetic, directions=direction)
        residual, components = study.interventions(matched, source, (np.zeros_like(kinetic),)*3,
            kinetic, direction*20., np.tile(np.arange(3), (2, 1)), tables)
        np.testing.assert_allclose(residual[:, 7], 0., atol=1e-10)
        np.testing.assert_allclose(residual[:, 10], components[:, [0, 2]].sum(axis=1), atol=1e-12)
        np.testing.assert_allclose(residual[1, 10], residual[1, 0], atol=1e-12)
        self.assertGreater(abs(residual[0, 0]), .1)

    def test_error_components_use_same_frame_and_keep_parent_offset(self):
        truth = np.array([[1., 2., 3.], [1., 3., 2.]])
        reconstructed = truth+np.array([[.2, -.1, .7], [.3, -.2, .1]])
        component = study.error_components(reconstructed, truth, np.array([2, 3]), np.array([5.9, 5.8]))
        np.testing.assert_allclose(component, [[.1, .7, .1], [.2, 0., .2]], atol=1e-14)
        np.testing.assert_allclose(component.sum(axis=1), reconstructed.sum(axis=1)-[5.9, 5.8])

    def test_covariance_and_paired_errors_agree_with_direct_weighted_calculation(self):
        weights = np.array([[1., 4.], [2., 3.], [3., 2.], [4., 1.]])
        components = np.array([[.1, .3, .01], [.3, .5, .01], [-.2, -.3, .01], [0., -.2, .01]])
        residual = np.repeat(components.sum(axis=1)[:, None], len(study.CASES), axis=1)
        residual[:, 1] = [.1, .9, -.2, .5]
        accumulator = study.ConditionalMoments(2, True)
        accumulator.add(weights, np.zeros(4, int), residual, components)
        result = accumulator.finish(weights.sum(axis=0))
        for model in range(2):
            w = weights[:, model]; d = w.sum()
            mean = w@residual/d
            error = np.sqrt((w*w)@((residual-mean)**2))/d
            keep = (abs(residual) <= .4).astype(float)
            difference = keep-keep[:, :1]; average = w@difference/d
            paired = np.sqrt((w*w)@((difference-average)**2))/d
            centred = components-w@components/d
            covariance = np.einsum('n,ni,nj->ij', w, centred, centred)/d
            np.testing.assert_allclose(result['mean_MeV'][model, 0], mean)
            np.testing.assert_allclose(result['mean_standard_error_MeV'][model, 0], error)
            np.testing.assert_allclose(result['paired_retention_standard_error'][model, 0], paired)
            np.testing.assert_allclose(result['component_covariance_MeV2'][model, 0], covariance, atol=1e-16)
            self.assertGreater(covariance[0, 1], 0.)
            self.assertGreater(covariance.sum(), np.trace(covariance))

    def test_common_finite_cohort_and_histogram_tails_conserve_weight(self):
        weights = np.array([[1.], [2.], [3.]])
        components = np.array([[3., 4., 0.], [-3., -4., 0.], [.1, .2, 0.]])
        residual = np.repeat(components.sum(axis=1)[:, None], len(study.CASES), axis=1)
        residual[2, 4] = np.nan
        accumulator = study.ConditionalMoments(1, True)
        accumulator.add(weights, np.zeros(3, int), residual, components)
        result = accumulator.finish(weights.sum(axis=0))
        self.assertEqual(result['common_per_generated'][0, 0], .5)
        self.assertEqual(result['excluded_per_generated'][0, 0], .5)
        self.assertEqual(result['invalid_per_generated'][0, 0, 4], .5)
        self.assertEqual(result['event_count'][0, 0], 2)
        np.testing.assert_allclose(result['component_histogram'][0, 0].sum(), 1.)
        self.assertEqual(result['component_histogram'][0, 0, 0, 0], 2/3)
        self.assertEqual(result['component_histogram'][0, 0, -1, -1], 1/3)
        self.assertFalse(result['valid'][0, 1:].any())
        self.assertTrue(np.isfinite(result['rms_MeV']).all())


if __name__ == '__main__':
    unittest.main()
