"""Checks of the bootstrap sampling law and independent posterior calculation."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import bootstrap_162_response_uncertainty as bootstrap


class BootstrapTests(unittest.TestCase):
    def test_generator_fractions_are_efficiency_weighted_before_sampling(self):
        hist = np.zeros((3, 2, 2), dtype=int)
        hist[0, 0, 0], hist[1, 0, 1], hist[2, 1, 0] = 80, 40, 20
        probability = bootstrap.empirical_probability(hist, [100]*3, [.2, .6, .2])
        np.testing.assert_allclose(probability.ravel(), [4/11, 6/11, 1/11, 0])

    def test_joint_template_bootstrap_preserves_rejections_and_threshold_nesting(self):
        hist = np.zeros((3, 3, 2, 2), dtype=int)
        for threshold, accepted in enumerate((80, 60, 40)):
            hist[threshold, :, 0, 0] = accepted
        categories = bootstrap.template_categories(hist, [100]*3)
        np.testing.assert_array_equal(categories.sum(axis=1), [100]*3)
        np.testing.assert_array_equal(categories[:, -1], [20]*3)
        rng = np.random.default_rng(42)
        counts = []
        for _ in range(1000):
            sampled, generated = bootstrap.resample_templates(categories, (2, 2), rng)
            np.testing.assert_array_equal(generated, [100]*3)
            self.assertTrue(np.all(sampled[0] >= sampled[1]))
            self.assertTrue(np.all(sampled[1] >= sampled[2]))
            counts.append(sampled[:, 0].sum(axis=(1, 2)))
        np.testing.assert_allclose(np.mean(counts, axis=0), [80, 60, 40], atol=.6)
        self.assertGreater(np.corrcoef(np.asarray(counts).T)[0, 2], .2)

    def test_interior_posterior_agrees_with_analytic_dirichlet_marginals(self):
        templates = np.zeros((3, 2, 2))
        templates[0, 0, 0] = templates[1, 0, 1] = templates[2, 1, 0] = 1e9
        data = np.array([[20000, 60000], [20000, 0]])
        result = bootstrap.fit_interior(data, templates, np.ones(3), local_grid_size=201)
        counts = np.array([20000, 60000, 20000])
        for key, quantile in (("lower95", .025), ("median", .5), ("upper95", .975)):
            expected = beta.ppf(quantile, counts+1, data.sum()-counts+2)
            np.testing.assert_allclose(result[key], expected, atol=1.3e-4)

    def test_inconsistent_thresholds_are_rejected(self):
        hist = np.ones((3, 3, 2, 2), dtype=int)
        hist[1, 0, 0, 0] = 2
        with self.assertRaises(ValueError):
            bootstrap.template_categories(hist, [100]*3)

    def test_wilson_interval_at_all_successes_still_has_uncertainty(self):
        lower, upper = bootstrap.wilson_interval(500, 500)
        self.assertLess(lower, 1)
        self.assertAlmostEqual(upper, 1)


if __name__ == "__main__":
    unittest.main()
