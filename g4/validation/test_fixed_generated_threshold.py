"""Sampling-law checks for fixed generated and fixed accepted reaction budgets."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import fixed_generated_162_threshold_study as study


class GeneratedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.hist = np.zeros((2, 3, 2, 2), dtype=int)
        for c, (x, y, accepted) in enumerate(((0, 0, 80), (0, 1, 40), (1, 0, 20))):
            self.hist[0, c, x, y] = accepted
            self.hist[1, c, x, y] = accepted // 2
        self.generated = np.array([100, 100, 100])
        self.probability = study.generated_probability(self.hist, self.generated)

    def test_generated_sampling_retains_the_unseen_rejection_probability(self):
        self.assertAlmostEqual(self.probability[-1], .56)
        expected = study.unpack_generated(self.probability, 2, (2, 2))
        np.testing.assert_allclose(expected[0].ravel(), [.16, .24, .04, 0])
        np.testing.assert_allclose(expected[1].ravel(), [.08, .12, .02, 0])

    def test_joint_threshold_counts_follow_binomial_variance_and_covariance(self):
        rng = np.random.default_rng(45)
        outcomes = []
        for _ in range(5000):
            data = study.unpack_generated(rng.multinomial(1000, self.probability), 2, (2, 2))
            self.assertTrue(np.all(data[0] >= data[1]))
            outcomes.append(data.sum(axis=(1, 2)))
        outcomes = np.array(outcomes)
        np.testing.assert_allclose(outcomes.mean(axis=0), [440, 220], atol=.7)
        np.testing.assert_allclose(outcomes.var(axis=0, ddof=1), [246.4, 171.6], rtol=.05)
        self.assertAlmostEqual(np.cov(outcomes.T)[0, 1], 1000*(.22-.44*.22), delta=8)

    def test_top_up_has_multinomial_shapes_and_negative_binomial_cost(self):
        rng = np.random.default_rng(51)
        accepted_probability = np.array([[4/11, 6/11], [1/11, 0]])
        counts, totals = [], []
        for _ in range(4000):
            first = study.unpack_generated(rng.multinomial(100, self.probability), 2, (2, 2))[0]
            filled, total = study.complete_accepted_sample(first, 100, 100, accepted_probability, .44, rng)
            self.assertEqual(filled.sum(), 100)
            self.assertTrue(np.all(filled >= first))
            self.assertGreaterEqual(total, 100)
            counts.append(filled.ravel()); totals.append(total)
        np.testing.assert_allclose(np.mean(counts, axis=0), 100*accepted_probability.ravel(), atol=.25)
        np.testing.assert_allclose(np.var(counts, axis=0, ddof=1),
                                   100*accepted_probability.ravel()*(1-accepted_probability.ravel()),
                                   rtol=.07, atol=.01)
        self.assertAlmostEqual(np.mean(totals), 100/.44, delta=.8)
        self.assertAlmostEqual(np.var(totals, ddof=1), 100*(1-.44)/.44**2, delta=20)

    def test_inconsistent_efficiencies_and_threshold_categories_are_rejected(self):
        bad = self.hist.copy(); bad[1, 0, 1, 1] = 1
        with self.assertRaises(ValueError):
            study.nested_categories(bad, self.generated)
        with self.assertRaises(ValueError):
            study.nested_categories(self.hist, [50, 100, 100])
        with self.assertRaises(ValueError):
            study.generated_probability(self.hist, self.generated, [.2, .6, .3])
        with self.assertRaises(ValueError):
            study.complete_accepted_sample(np.array([[101]]), 100, 100, np.ones((1, 1)), 1, np.random.default_rng(1))

    def test_resuming_preserves_exact_paired_random_streams(self):
        hist = np.array([self.hist[0] - t*self.hist[0]//20 for t in range(7)])
        pools = dict(template_histograms=hist, mixed_histograms=hist,
                     pure_generated=self.generated, mixed_generated=self.generated,
                     template_efficiencies=hist.sum(axis=(2, 3))/self.generated,
                     mixed_efficiencies=hist.sum(axis=(2, 3))/self.generated)
        options = dict(budget=10000, seed=42, local_grid_size=31, progress_every=2)
        full = study.run_trials(pools, repetitions=4, **options)
        checkpoints = []
        study.run_trials(pools, repetitions=2, checkpoint=checkpoints.append, **options)
        resumed = study.run_trials(pools, repetitions=4, resume=checkpoints[-1], **options)
        for name in study.BUDGETS:
            for key in full['records'][name]:
                np.testing.assert_array_equal(full['records'][name][key], resumed['records'][name][key])


if __name__ == "__main__":
    unittest.main()
