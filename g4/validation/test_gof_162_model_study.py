"""Scientific bookkeeping and independence checks for the adequacy gate."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import gof_162_model_study as study


def row(index, deviance=1., q0=0., q_upper=0.):
    return dict(index=index, deviance=deviance, data_deviance=deviance,
                mc_deviance=0., q0=q0, q_upper=q_upper, dhat=.005, shape_gap=0.)


def toy_inputs():
    p = np.arange(1., 146.)
    p /= p.sum()
    return dict(raw_templates=np.full((3, 144), 100), generated=[20000]*3,
                cutoffs=dict(q0=4., q_upper=4.),
                base_references={r: np.tile(p, (3, 1)) for r in study.REFERENCES},
                weighted={r: dict(probability=np.tile(p, (4, 1))) for r in study.REFERENCES})


class AdequacyTests(unittest.TestCase):
    def test_deviance_identity_including_empty_bins(self):
        observed = np.array([0., 5., 25., 70.])
        expected = np.array([4., 8., 28., 60.])
        residual = study.deviance_residuals(observed, expected)
        independent = 2*np.sum(xlogy(observed, observed)-xlogy(observed, expected))
        self.assertAlmostEqual(residual @ residual, independent, places=11)
        self.assertAlmostEqual(residual[0], -np.sqrt(8.))
        self.assertGreater(residual[-1], 0.)
        with self.assertRaises(ValueError):
            study.deviance_residuals([1.], [0.])

    def test_no_reports_is_undefined_coverage(self):
        summary = study.summarize_case([row(0, 10., 8., 8.)], 2., dict(q0=4., q_upper=4.))
        self.assertIsNone(summary['reported_inclusion']['rate'])
        self.assertIsNone(summary['reported_inclusion']['wilson'])
        self.assertEqual(summary['reported_noncontaining']['count'], 0)
        self.assertEqual(summary['inadequacy']['count'], 1)
        self.assertEqual(summary['raw_inclusion']['count'], 0)

    def test_reporting_partition_and_discovery_denominators(self):
        rows = [row(0, 1., 5., 0.), row(1, 1., 6., 9.), row(2, 10., 7., 8.)]
        result = study.summarize_case(rows, 2., dict(q0=4., q_upper=4.))
        self.assertEqual(result['reported_inclusion']['total'], 2)
        self.assertEqual(result['reported_inclusion']['rate'], .5)
        self.assertEqual(result['reported_noncontaining']['rate'], 1/3)
        self.assertEqual(result['reported_discovery']['rate'], 2/3)
        self.assertEqual(result['raw_discovery']['rate'], 1.)
        self.assertEqual(sum(result[k]['count'] for k in
            ['reported_containing', 'reported_noncontaining', 'inadequacy']), 3)

    def test_cutoff_ties_pass(self):
        result = study.summarize_case([row(0, 2., 4., 4.)], 2., dict(q0=4., q_upper=4.))
        self.assertEqual(result['report']['count'], 1)
        self.assertEqual(result['reported_inclusion']['count'], 1)
        self.assertEqual(result['reported_discovery']['count'], 0)

    def test_calibration_rank_guarantee_and_missing_trials(self):
        n = study.PLAN['calibration_repetitions']
        records = {study.case_key(r, d, 'bw'): [row(i, i+shift) for i in range(n)]
                   for shift, (r, d) in enumerate((r, d) for r in study.REFERENCES for d in study.PLAN['direct'])}
        frozen = study.freeze_gof(dict(phase='calibration', records=records))
        rank = frozen['tolerance']['rank']
        self.assertGreaterEqual(binom.cdf(rank-1, n, .95), 1-.05/4)
        self.assertLess(binom.cdf(rank-2, n, .95), 1-.05/4)
        self.assertEqual(frozen['cutoff'], rank-1+3)
        with self.assertRaises(ValueError):
            study.freeze_gof(dict(phase='validation', records=records))
        next(iter(records.values())).pop()
        with self.assertRaises(ValueError):
            study.freeze_gof(dict(phase='calibration', records=records))

    def test_independent_streams_and_rejection_category(self):
        inputs = toy_inputs()
        study.check_inputs(inputs)
        a = study.draw_data(inputs, 'calibration', 'old', 0., 'bw', 5)
        replay = study.draw_data(inputs, 'calibration', 'old', 0., 'bw', 5)
        b = study.draw_data(inputs, 'validation', 'old', 0., 'bw', 5)
        np.testing.assert_array_equal(a, replay)
        self.assertFalse(np.array_equal(a, b))
        self.assertLess(a.sum(), study.PLAN['data_budget'])
        with self.assertRaises(ValueError):
            study.draw_data(inputs, 'calibration', 'old', 0., 'rmatrix', 5)

    def test_weighted_auxiliary_counts_rejected(self):
        inputs = toy_inputs()
        inputs['raw_templates'] = inputs['raw_templates'].astype(float)
        inputs['raw_templates'][0, 0] += .25
        with self.assertRaises(ValueError):
            study.check_inputs(inputs)
        inputs = toy_inputs()
        inputs['weighted']['old']['probability'][0, -1] = 0.
        with self.assertRaises(ValueError):
            study.check_inputs(inputs)

    def test_duplicate_and_invalid_fit_records_rejected(self):
        with self.assertRaises(ValueError):
            study.audit_case([row(0), row(0)], 2)
        with self.assertRaises(ValueError):
            study.audit_case([row(0, float('nan'))], 1)


if __name__ == '__main__':
    unittest.main()
