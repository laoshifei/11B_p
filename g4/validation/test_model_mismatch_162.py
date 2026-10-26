"""Scientific controls for paired fits, selection accounting, and fixed support."""

from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import model_mismatch_162_study as study
from analysis.bayes.generate_162_model_libraries import macro_text


class ModelMismatchTest(unittest.TestCase):
    def test_notebook_json_key_sorting_cannot_change_rng_namespaces(self):
        original = {name: dict(root=f'{name}.root') for name in study.CASES}
        roundtrip = json.loads(json.dumps(original, sort_keys=True))
        self.assertNotEqual(list(original), list(roundtrip))
        self.assertEqual(study.ordered_library_runs(original), study.ordered_library_runs(roundtrip))
        ids = np.arange(10000)
        for runs in (original, roundtrip):
            index = next(i for i, (name, _) in enumerate(study.ordered_library_runs(runs)) if name == 'coherent')
            subset, _ = study.split_ids(ids, [study.SPLIT_SEED, index])
            expected, _ = study.split_ids(ids, [study.SPLIT_SEED, 4])
            np.testing.assert_array_equal(subset, expected)
        with self.assertRaises(ValueError):
            study.ordered_library_runs({'coherent': {}})

    def test_split_includes_lost_reactions_and_has_no_overlap(self):
        generated = np.arange(101)
        template, data = study.split_ids(generated, [42, 17])
        self.assertEqual(len(np.intersect1d(template, data)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[template, data]), generated)
        # Selection happens after this split; denominators retain all lost IDs.
        selected = generated[generated % 3 == 0]
        h = np.array([[np.isin(template, selected).sum()], [np.isin(data, selected).sum()]])
        probability = study.population(h, [len(template), len(data)], [.4, .6])
        expected = .4*h[0, 0]/len(template)+.6*h[1, 0]/len(data)
        self.assertAlmostEqual(probability[0], expected)
        self.assertAlmostEqual(probability[-1], 1-expected)

    def test_fixed_support_contains_physical_triangle_momenta(self):
        rng = np.random.default_rng(917)
        momenta = rng.normal(size=(20000, 3, 3))
        momenta -= momenta.mean(axis=1, keepdims=True)
        energy = np.sort((momenta*momenta).sum(axis=2), axis=1)[:, ::-1]
        high, middle, low = energy.T
        xy = np.column_stack([np.sqrt(3)*(middle-low)/energy.sum(axis=1),
                               (2*high-middle-low)/energy.sum(axis=1)])
        for bins in (30, 60, 90):
            hist = study.histogram_xy(xy, bins)
            support = study.geometric_support(bins)
            self.assertEqual(hist[~support].sum(), 0)
            self.assertGreater(np.count_nonzero((hist == 0) & support), 0)
            self.assertFalse(support[0, 0])

    def test_matched_asimov_recovers_generated_not_accepted_fractions(self):
        templates = np.array([[91., 5., 4.], [4., 88., 8.], [3., 6., 91.]])
        efficiency = np.array([.4, .9, .7])
        bank = dict(templates=templates, efficiency=efficiency)
        probability = study.exact_population(templates, efficiency)
        fit = study.fit(100000*probability[:-1], bank)
        np.testing.assert_allclose(fit["map"], study.FRACTIONS, atol=1e-10)
        self.assertLess(abs(fit["deviance"]), 1e-7)
        self.assertLess(fit["q0"], 1e-7)

    def test_missing_physics_can_produce_known_false_component(self):
        templates = np.array([[94., 3., 3.], [3., 94., 3.], [3., 3., 94.]])
        bank = dict(templates=templates, efficiency=np.ones(3))
        # True alpha1 has a 10% shape contribution that the wrong model
        # can only express using its third template; its true extra rate is zero.
        actual_alpha1 = .9*templates[1]+.1*templates[2]
        data = 1000*(.05*templates[0]+.95*actual_alpha1)
        result = study.fit(data, bank)
        np.testing.assert_allclose(result["map"], [.05, .855, .095], atol=1e-9)
        self.assertGreater(result["q0"], 100)

    def test_finite_mc_zero_inside_support_stays_in_normalization(self):
        templates = np.array([[9., 1., 0., 0.], [1., 9., 0., 0.], [2., 2., 6., 0.]])
        smooth = templates+.5
        efficiency = np.array([.8, .9, .7])
        probability = study.exact_population(smooth, efficiency)
        self.assertGreater(probability[3], 0)
        bank = dict(templates=smooth, efficiency=efficiency)
        fit = study.fit(100000*probability[:-1], bank)
        np.testing.assert_allclose(fit["map"], study.FRACTIONS, atol=1e-10)

    def test_paired_trials_share_draws_but_stages_do_not(self):
        bank = dict(templates=np.array([[90., 5., 5.], [5., 90., 5.], [5., 5., 90.]]),
                    efficiency=np.array([.7, .9, .8]))
        p = study.exact_population(bank["templates"], bank["efficiency"])
        args = (bank, p, study.EMPIRICAL_SEED, 0, 0, 4, 1000)
        a, b = study._batch(args), study._batch(args)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left["map"], right["map"])
            self.assertEqual(left["accepted"], right["accepted"])
        other = study._batch((bank, p, study.VALIDATION_SEED, 0, 0, 4, 1000))
        self.assertNotEqual([r["accepted"] for r in a], [r["accepted"] for r in other])

    def test_macro_variants_keep_response_and_beam_conditions(self):
        base = (study.ROOT/"g4/macros/model_mismatch_162.mac").read_text()
        variants = {name: macro_text(base, name, 100000) for name in study.MODELS}
        allowed = "/h11b/162AlphaDecayModel "
        controls = {name: [line for line in text.splitlines() if not line.startswith(allowed)]
                    for name, text in variants.items()}
        self.assertEqual(controls["coherent"], controls["incoherent"])
        self.assertEqual(controls["coherent"], controls["legacy"])

    def test_zero_detections_wilson_interval_can_be_plotted(self):
        import matplotlib.pyplot as plt
        state = {}
        for stage in ("exact_validation", "empirical_validation"):
            state[stage] = dict(
                summary={name: dict(false_positive=study.binomial_summary(0, 500)) for name in study.MODELS},
                records={name: dict(map=np.array([[.05, .95, 0.]])) for name in study.MODELS})
        figure = study.plot_ensemble(state)
        figure.canvas.draw()
        self.assertLess(figure.axes[1].get_ylim()[1], 20)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
