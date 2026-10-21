"""Checks of exact-model sampling, support, efficiency, and restart behavior."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import diagnose_162_boundary_intervals as study


class ExactPopulationTests(unittest.TestCase):
    def setUp(self):
        self.pools = dict(templates=np.array([[8, 0, 0, 0], [0, 4, 0, 0], [0, 0, 2, 0]]),
                          pure_generated=np.array([10, 10, 10]))

    def test_smoothed_shapes_preserve_raw_efficiencies_and_structural_zeros(self):
        population = study.exact_population(self.pools)
        expected = .05*.8*np.array([8.5, .5, .5])/9.5 + .95*.4*np.array([.5, 4.5, .5])/5.5
        np.testing.assert_allclose(population["probability"], np.r_[expected, 0., .58])
        np.testing.assert_allclose(population["efficiencies"], [.8, .4, .2])
        np.testing.assert_array_equal(population["active"], [True, True, True, False])
        np.testing.assert_allclose(population["shapes"].sum(axis=1), 1)

    def test_generation_matches_fitted_conditional_distribution(self):
        population = study.exact_population(self.pools, fractions=[.1, .7, .2])
        data = 100000*population["probability"][:-1]
        model = study.boundary._Likelihood(data, population["templates"], population["efficiencies"], .5)
        np.testing.assert_allclose(model.probability(population["fractions"]),
                                   data[data > 0]/data.sum(), rtol=1e-14)
        np.testing.assert_allclose(model.mode, population["fractions"], atol=1e-10)

    def test_multinomial_draw_retains_random_rejection(self):
        population = study.exact_population(self.pools, pseudocount=0)
        np.testing.assert_allclose(population["probability"], [.04, .38, 0, 0, .58])
        draws = np.random.default_rng(271).multinomial(100, population["probability"], size=20000)
        np.testing.assert_array_equal(draws.sum(axis=1), 100)
        accepted = draws[:, :-1].sum(axis=1)
        self.assertAlmostEqual(accepted.mean(), 42, delta=.2)
        self.assertAlmostEqual(accepted.var(), 100*.42*.58, delta=.6)

    def test_invalid_population_inputs_are_rejected(self):
        for kwargs in ({"fractions": [0, 0, 0]}, {"fractions": [-.1, 1.1, 0]},
                       {"pseudocount": -1}):
            with self.assertRaises(ValueError):
                study.exact_population(self.pools, **kwargs)
        broken = copy.deepcopy(self.pools)
        broken["pure_generated"][0] = 1
        with self.assertRaises(ValueError):
            study.exact_population(broken)

    def test_resume_replays_identical_draws_and_rejects_changed_configuration(self):
        pools = dict(templates=np.diag([80000, 40000, 20000]), pure_generated=np.full(3, 100000))
        population = study.exact_population(pools)
        options = dict(budget=2000, grid_size=51, seed=712, progress_every=2)
        whole = study.run_control(population, repetitions=4, **options)
        checkpoints = []
        study.run_control(population, repetitions=2, checkpoint=checkpoints.append, **options)
        resumed = study.run_control(population, repetitions=4, resume=checkpoints[-1], **options)
        for key in whole["records"]:
            np.testing.assert_array_equal(whole["records"][key], resumed["records"][key])
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            study.run_control(study.exact_population(pools, pseudocount=.1),
                              repetitions=4, resume=checkpoints[-1], **options)


if __name__ == "__main__":
    unittest.main()
