"""Regression checks for response matching and efficiency-corrected fractions."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import scan_detector_theta_phi_response_162_channel_fit as scan


class ResponseStudyTests(unittest.TestCase):
    def test_matched_and_calibration_have_distinct_parameter_rules(self):
        args = scan._default_args()
        args.scan = "energy-fwhm"
        args.fixed_theta_deg = 2.0
        args.fixed_phi_deg = 3.0
        args.fixed_energy_fwhm_keV = 100.0
        args.fixed_threshold_keV = 50.0
        args.study = "matched"
        data, template = scan.response_pair(args, 150.0)
        self.assertEqual(data, (2.0, 3.0, 150.0, 50.0))
        self.assertEqual(data, template)
        args.study = "calibration"
        data, template = scan.response_pair(args, 150.0)
        self.assertEqual(data, (2.0, 3.0, 100.0, 50.0))
        self.assertEqual(template, (2.0, 3.0, 150.0, 50.0))

    def test_legacy_template_flags_remain_independent(self):
        args = scan._default_args()
        args.scan = "threshold"
        args.fixed_theta_deg = 2.0
        args.fixed_energy_fwhm_keV = 100.0
        args.apply_threshold_to_benchmark = True
        data, template = scan.response_pair(args, 200.0)
        self.assertEqual(data, (2.0, 0.0, 100.0, 200.0))
        self.assertEqual(template, (0.0, 0.0, 0.0, 200.0))

    def test_response_efficiency_recovers_generator_not_detected_fractions(self):
        # Three perfectly distinguishable channels, with different losses.
        templates = np.zeros((3, 2, 2))
        templates[0, 0, 0] = 80000
        templates[1, 0, 1] = 40000
        templates[2, 1, 0] = 20000
        efficiencies = scan.response_efficiencies(templates, np.full(3, 100000))
        np.testing.assert_allclose(efficiencies, [.8, .4, .2])
        # Generator fractions (.2,.6,.2), followed by the stated efficiencies.
        data = np.array([[4000, 6000], [1000, 0]])
        grid = scan.simplex_grid(41)
        correct = scan.fit_histogram(data, templates, efficiencies, grid, 41, 151, 6, .5)
        wrong = scan.fit_histogram(data, templates, np.ones(3), grid, 41, 151, 6, .5)
        np.testing.assert_allclose(correct["median"], [.2, .6, .2], atol=.002)
        self.assertGreater(np.max(np.abs(wrong["median"]-[.2, .6, .2])), .1)

    def test_zero_efficiency_is_not_silently_fitted(self):
        with self.assertRaises(ValueError):
            scan.response_efficiencies(np.zeros((3, 2, 2)), np.full(3, 100))


if __name__ == "__main__":
    unittest.main()
