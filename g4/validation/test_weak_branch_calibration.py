"""Independent analytic and numerical checks of the calibrated-LR workflow."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import xlogy
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import calibrate_162_weak_branch_intervals as study


class CalibrationTests(unittest.TestCase):
    def model(self, counts, efficiencies=None, templates=None, pseudocount=0.):
        return study.boundary._Likelihood(np.asarray(counts, float),
                    np.eye(3)*1e6 if templates is None else templates,
                    np.ones(3) if efficiencies is None else efficiencies, pseudocount)

    def test_profile_matches_binomial_deviance_with_boundary_and_positive_counts(self):
        for counts in (np.array([50., 950., 0.]), np.array([40., 940., 20.])):
            model = self.model(counts)
            n = counts.sum()
            for channel, values in ((1, (.8, .94, .99)), (2, (.001, .01, .03))):
                k = counts[channel]
                for value in values:
                    expected = 2*(xlogy(k, k/(n*value))+xlogy(n-k, (n-k)/(n*(1-value))))
                    self.assertAlmostEqual(study.profile_statistic(model, value, channel), expected, delta=1e-9)

    def test_efficiency_weighted_profile_matches_independent_scalar_optimization(self):
        templates = np.array([[70., 20., 10., 1.], [5., 75., 10., 10.], [5., 10., 50., 35.]])
        efficiency = np.array([.8, .4, .2])
        counts = np.array([76., 630., 154., 140.])
        model = self.model(counts, efficiency, templates, pseudocount=.5)
        shape = (templates+.5)/(templates+.5).sum(axis=1, keepdims=True)
        def log_likelihood(fractions):
            weighted = fractions*efficiency
            return float(counts @ np.log(weighted @ shape/weighted.sum()))
        maximum = log_likelihood(model.mode)
        for channel in (1, 2):
            remaining = [i for i in range(3) if i != channel]
            for value in (.02, .8, .95):
                def objective(x):
                    f = np.zeros(3)
                    f[channel] = value
                    f[remaining[0]], f[remaining[1]] = x, 1-value-x
                    return -log_likelihood(f)
                best = minimize_scalar(objective, bounds=(0, 1-value), method="bounded",
                                       options={"xatol": 1e-14})
                expected = -min(best.fun, objective(0.), objective(1-value))-maximum
                self.assertAlmostEqual(study.profile_log_relative(model, value, channel), expected, delta=2e-7)

    def test_interval_inversion_matches_binomial_roots(self):
        counts = np.array([40., 940., 20.])
        model = self.model(counts)
        n, k, cutoff = counts.sum(), counts[1], 4.2
        def statistic(value):
            return 2*(xlogy(k, k/(n*value))+xlogy(n-k, (n-k)/(n*(1-value))))-cutoff
        expected = [brentq(statistic, 1e-8, k/n), brentq(statistic, k/n, 1-1e-8)]
        np.testing.assert_allclose(study.interval(model, cutoff, 1), expected, atol=1e-11)

    def test_upper_statistic_has_correct_direction_and_zero_limit_covers_boundary(self):
        model = self.model([40., 940., 20.])
        self.assertEqual(study.upper_statistic(model, .01), 0.)
        self.assertGreater(study.upper_statistic(model, .03), 0.)
        null = self.model([50., 950., 0.])
        self.assertEqual(study.upper_statistic(null, 0.), 0.)
        # With no events in the third orthogonal component, q(d)=-2*N*log(1-d).
        expected = 1-np.exp(-3./2000)
        self.assertAlmostEqual(study.upper_limit(null, 3.), expected, delta=1e-11)

    def test_tolerance_order_statistic_satisfies_family_error_allocation(self):
        rank = study.tolerance_rank(2000)
        target = 1-.05/39
        self.assertGreaterEqual(binom.cdf(rank['rank']-1, 2000, .95), target)
        self.assertLess(binom.cdf(rank['rank']-2, 2000, .95), target)
        with self.assertRaisesRegex(ValueError, "Too few"):
            study.tolerance_rank(10)

    def test_cutoff_freeze_uses_separate_null_cases_and_maximum_not_pooled_samples(self):
        rows = {}
        for i, case in enumerate(study.cases((.05,))[:2]):
            rows[case['key']] = dict(case=case, records=dict(
                q_alpha1=np.linspace(0, 1+i, 2000), q_upper=np.linspace(0, 2+i, 2000),
                q0=np.linspace(0, 1 if i == 0 else 1e6, 2000)))
        calibration = dict(configuration=dict(stage='calibration', repetitions=2000), cases=rows)
        frozen = study.freeze_cutoffs(calibration)
        index = frozen['tolerance']['rank']-1
        self.assertEqual(frozen['cutoffs']['q_alpha1'], np.linspace(0, 2, 2000)[index])
        self.assertEqual(frozen['cutoffs']['q_upper'], np.linspace(0, 3, 2000)[index])
        self.assertEqual(frozen['cutoffs']['q0'], np.linspace(0, 1, 2000)[index])
        calibration['configuration']['stage'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'only calibration'):
            study.freeze_cutoffs(calibration)

    def test_seeded_trial_is_independent_of_task_order_and_stage(self):
        pools = dict(templates=np.diag([8000, 4000, 2000]), pure_generated=np.full(3, 10000))
        case = study.cases((.05,))[0]
        population = study.compact_population(pools, case['fractions'])
        first = study.draw(population, 123, case, 7, 10000)
        study.draw(population, 123, case, 2, 10000)
        np.testing.assert_array_equal(first, study.draw(population, 123, case, 7, 10000))
        self.assertFalse(np.array_equal(first, study.draw(population, 124, case, 7, 10000)))

    def test_checkpoint_resume_preserves_trials_and_rejects_configuration_change(self):
        pools = dict(templates=np.diag([80000, 40000, 20000]), pure_generated=np.full(3, 100000))
        options = dict(stage='calibration', repetitions=4, seed=17, budget=2000,
                       workers=1, chunk_size=2, progress_every=2)
        saved = []
        whole = study.run_ensemble(pools, study.cases((.05,))[:2], checkpoint=lambda c: saved.append(copy.deepcopy(c)), **options)
        resumed = study.run_ensemble(pools, study.cases((.05,))[:2], resume=saved[0], **options)
        for key in whole['cases']:
            for field in whole['cases'][key]['records']:
                np.testing.assert_array_equal(whole['cases'][key]['records'][field], resumed['cases'][key]['records'][field])
        options['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            study.run_ensemble(pools, study.cases((.05,))[:2], resume=saved[0], **options)


if __name__ == '__main__':
    unittest.main()
