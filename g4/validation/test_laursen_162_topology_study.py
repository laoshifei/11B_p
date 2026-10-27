"""Accounting and identifiability checks independent of the frozen ROOT replay."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_topology_study as study


class LinearCalibration:
    materials = {'G4_Si': {'inverse_light': staticmethod(lambda x: x)}}


def event(hits, multiple=False, ambiguous=False):
    value = study.matching.allocate(1)
    value['multiplicity'][0] = len(hits)
    value['multiple'][0] = multiple
    value['ambiguous'][0] = ambiguous
    for k, (detector, front, back, energy) in enumerate(hits):
        value['detector'][0, k] = detector
        value['pixel'][0, k] = (front, back)
        value['energy'][0, k] = energy
    return value


class TopologyTests(unittest.TestCase):
    def test_identical_strip_law_for_distinct_particle_counts(self):
        result = study.signal_counterexamples(LinearCalibration())
        self.assertEqual(result['noise_draws'], 33)
        self.assertEqual(result['max_absolute_observed_difference_MeV'], 0.)
        np.testing.assert_array_equal(result['policy_multiplicity']['sum_first'], 2)
        np.testing.assert_array_equal(result['policy_multiplicity']['share_first'], 1)
        self.assertTrue(result['policy_ambiguous']['unique'].all())

    def test_same_residual_cost_cannot_choose_sum_or_share(self):
        front = study.matching.matching.signals([(7, 1.2), (8, 1.8)])
        back = study.matching.matching.signals([(9, 3.)])
        candidates = study.matching.matching.primitives(front, back, .150)
        relevant = [p for p in candidates if p.front_mask == 3 and p.back_mask == 1]
        self.assertEqual(len(relevant), 2)
        self.assertEqual({len(p.hits) for p in relevant}, {1, 2})
        self.assertEqual(relevant[0].cost, relevant[1].cost)
        self.assertEqual(relevant[0].complexity, relevant[1].complexity)

    def test_agreement_is_invariant_to_storage_order(self):
        hits = [(0, 2, 7, 1.2), (1, 4, 1, 2.1)]
        group, controls = study.observable_topology(event(hits), event(hits[::-1]))
        np.testing.assert_array_equal(group, [0])
        np.testing.assert_array_equal(controls, [[True, True, True]])

    def test_agreement_does_not_imply_a_single_candidate(self):
        hit = [(0, 3, 2, 1.)]
        group, controls = study.observable_topology(event(hit, True), event(hit, True))
        np.testing.assert_array_equal(group, [1])
        np.testing.assert_array_equal(controls, [[True, True, False]])

    def test_different_topologies_are_deferred_without_truth(self):
        primary = event([(0, 7, 9, 1.2), (0, 8, 9, 1.8)], True)
        alternative = event([(0, 7.6, 9, 3.)], True)
        group, controls = study.observable_topology(primary, alternative)
        np.testing.assert_array_equal(group, [2])
        np.testing.assert_array_equal(controls, [[True, False, False]])

    def test_unresolved_alternative_is_separate(self):
        primary = event([(0, 7, 9, 1.2)], True)
        group, controls = study.observable_topology(primary, event([], True, True))
        np.testing.assert_array_equal(group, [3])
        self.assertFalse(controls[0, 1])

    def test_false_single_candidate_claim_fails(self):
        with self.assertRaises(ArithmeticError):
            study.observable_topology(event([(0, 1, 2, 3.)]), event([(0, 1, 2, 3.1)]))

    def test_weighted_partition_reconstructs_second_moment_and_signed_tails(self):
        weights = np.array([[1., 3.], [2., 1.], [4., 2.], [1., 5.]])
        residual = np.array([[-2., -.5, -.4], [3., 2., 1.5], [.1, .2, .3], [-.2, -.3, -.2]])
        partition = study.ResidualPartition(2)
        partition.add(weights, np.zeros(4, int), np.arange(4), np.array([0, 1, 2, 2]), residual)
        out = partition.finish(weights.sum(axis=0))
        expected = weights.T@(residual*residual)/weights.sum(axis=0)[:, None]
        np.testing.assert_allclose(out['cohort_moments'][:, 0, :, 1], expected)
        np.testing.assert_allclose(out['fraction_of_cohort_second_moment'][:, 0].sum(axis=(1, 2)), 1.)
        np.testing.assert_allclose(out['cohort_moments'][:, 0, 0, 2], weights[0]/weights.sum(axis=0))
        np.testing.assert_allclose(out['cohort_moments'][:, 0, 0, 3], weights[1]/weights.sum(axis=0))
        self.assertTrue(np.isfinite(out['conditional_moments']).all())

    def test_generated_and_conditional_denominators_are_distinct(self):
        partition = study.ResidualPartition(1)
        partition.add(np.ones((3, 1)), np.array([0, 0, -1]), np.array([0, 1, 0]), np.zeros(3, int),
                      np.array([[1., .2, .2], [3., 2., 2.], [100., 100., 100.]]))
        out = partition.finish(np.array([3.]))
        self.assertAlmostEqual(out['cohort_per_generated'][0, 0], 2/3)
        self.assertAlmostEqual(out['cohort_moments'][0, 0, 0, 1], 5.)
        self.assertAlmostEqual(out['cohort_moments'][0, 0, 1, 5], .5)

    def test_importance_scale_does_not_change_partition_results(self):
        outputs = []
        for scale in (1., 7.):
            weights = scale*np.array([[1.], [3.]])
            partition = study.ResidualPartition(1)
            partition.add(weights, np.array([0, 0]), np.array([0, 1]), np.array([0, 2]),
                          np.array([[-2., .1, .1], [4., 2., 2.]]))
            outputs.append(partition.finish(weights.sum(axis=0)))
        for key in ('cohort_moments', 'conditional_standard_error', 'per_generated', 'fraction_of_cohort_second_moment'):
            np.testing.assert_allclose(outputs[0][key], outputs[1][key])

    def test_nonfinite_accepted_residuals_are_reported(self):
        partition = study.ResidualPartition(1)
        partition.add(np.ones((1, 1)), np.array([0]), np.array([0]), np.array([0]), np.array([[1., np.nan, 1.]]))
        out = partition.finish(np.array([1.]))
        self.assertEqual(out['excluded_per_generated'][0, 0], 1.)
        self.assertEqual(out['cohort_per_generated'][0, 0], 0.)

    def test_conditional_error_uses_squared_importance_weights(self):
        partition = study.ResidualPartition(1)
        weights = np.array([[1.], [3.]])
        partition.add(weights, np.zeros(2, int), np.zeros(2, int), np.zeros(2, int),
                      np.array([[1., 1., 1.], [3., 3., 3.]]))
        out = partition.finish(weights.sum(axis=0))
        expected = np.sqrt(1.*(1.-2.5)**2+9.*(3.-2.5)**2)/4.
        self.assertAlmostEqual(out['conditional_standard_error'][0, 0, 0, 0, 0, 0], expected)

    def test_control_accounting_includes_deferred_events(self):
        labels = np.array([[[[1., 0.], [0., 0.]]], [[[0., 1.], [0., 0.]]]])
        keep = np.array([[1, 1, 1], [1, 0, 0]])
        controlled = labels*keep[:, :, None, None]
        deferred = labels-controlled
        np.testing.assert_array_equal(controlled+deferred, np.broadcast_to(labels, controlled.shape))
        self.assertEqual(deferred[:, 1].sum(), 1.)

    def test_refinement_selects_only_low_support_observable_hits(self):
        matched = study.matching.allocate(2)
        matched['detector'][:, 0] = 0
        matched['pixel'][:, 0] = [[0, 2], [0, 3]]
        eligible = np.zeros((2, 3), bool); eligible[:, 0] = True
        observed = np.zeros((2, 2, 2, 16))
        observed[0, 0, 0, 0] = 1.1; observed[0, 0, 1, 2] = 1.2
        post = dict(valid=np.ones((2, 4), bool), mean=np.array([[1.]*4, [2.]*4]),
                    effective_rays=np.array([[4.]*4, [20.]*4]))

        class Engine:
            def estimate(self, detector, pixels, pair, width):
                np.testing.assert_array_equal(detector, [0])
                np.testing.assert_array_equal(pixels, [[0, 2]])
                np.testing.assert_array_equal(pair, [[1.1, 1.2]])
                return dict(valid=np.ones((1, 4), bool), mean=np.array([[1.503, 1.5, 1.6, 1.498]]),
                            effective_rays=np.full((1, 4), 100.))

        refined, records = study.refine_hits(Engine(), matched, observed, eligible, post)
        np.testing.assert_allclose(refined, [1.5, 2.])
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]['within_5keV_controls'])

    def test_well_supported_hits_do_not_trigger_extra_integration(self):
        post = dict(valid=np.ones((0, 4), bool), mean=np.zeros((0, 4)), effective_rays=np.zeros((0, 4)))
        refined, records = study.refine_hits(None, None, None, None, post)
        self.assertEqual(refined.shape, (0,))
        self.assertEqual(records, [])


if __name__ == '__main__':
    unittest.main()
