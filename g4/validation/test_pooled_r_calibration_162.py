"""Guards for pooling, free r injection, paired data, and envelope calibration."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import pooled_r_calibration_162_study as study


def example_parent():
    h = np.zeros((3, 497), dtype=np.int64)
    h[:, :3] = [[120000, 20000, 10000], [100000, 650000, 50000], [20000, 20000, 100000]]
    second = h.copy()
    second[:, :2] += [1000, -1000]
    generated = [200000, 1000000, 200000]
    references = {name: study.banks.reaction_population(raw, generated)
                  for name, raw in (('old', h), ('new', second))}
    parent = study.plain(dict(inputs=dict(templates={name: dict(raw_templates=raw, generated=generated)
        for name, raw in (('old', h), ('new', second))}, references=references,
        reference_generated=dict(old=generated, new=generated),
        summaries=dict(test=dict(disjoint=True)), provenance=dict(test=True)),
        configuration=dict(library_configuration_sha256='example')))
    parent['checkpoint_sha256'] = study.digest(parent)
    parent['completed_at'] = 'example'
    return parent


def calibration_records():
    result = {}
    for index, case in enumerate(study.cases()):
        points = {key: {method: dict(q_upper=np.arange(1000)/1000+index,
                                    q0=np.arange(1000)/1000+2*index)
                        for method in study.finite.METHODS} for key in ('00000', '05000', '20000')}
        result[study.case_id(**case)] = dict(configuration=dict(case=case, phase='calibration',
            seed=study.P['seeds'][case['reference']]['calibration'], repetitions=1000), records=points)
    return result


class PooledRCalibrationTests(unittest.TestCase):
    def test_r_means_alpha0_within_the_nondirect_fraction(self):
        for r in study.P['r_values']:
            f = study.fractions(.02, r)
            self.assertAlmostEqual(f.sum(), 1)
            self.assertAlmostEqual(f[0]/(f[0]+f[1]), r)
            self.assertEqual(f[2], .02)
        np.testing.assert_array_equal(study.fractions(.005, .05), study.finite.fractions(.005))
        with self.assertRaises(ValueError):
            study.fractions(-.01, .05)

    def test_pooling_adds_raw_counts_and_budgets_and_leaves_references_out(self):
        parent = example_parent()
        saved = deepcopy(parent)
        inputs = study.prepare_inputs(parent)
        a, b = (np.asarray(parent['inputs']['templates'][n]['raw_templates']) for n in ('old', 'new'))
        np.testing.assert_array_equal(inputs['raw_templates'], a+b)
        np.testing.assert_array_equal(inputs['generated'], study.P['generated'])
        self.assertEqual(inputs['references'], parent['inputs']['references'])
        self.assertEqual(parent, saved)
        parent['inputs']['references']['old'][0][0] += .01
        with self.assertRaises(ValueError):
            study.prepare_inputs(parent)

    def test_r05_conditional_data_reproduces_parent_and_other_r_changes_injection(self):
        parent = example_parent()
        inputs = study.prepare_inputs(parent)
        for reference in ('old', 'new'):
            for bins in study.P['bins']:
                case = dict(bins=bins, reference=reference, r=.05)
                current = study.draw(inputs, case, .005, 'conditional', 17)
                original = study.prior.draw(parent['inputs'], dict(bins=bins, reference=reference, template='old'),
                    .005, study.P['legacy_r05_conditional_seeds'][reference], 17)
                np.testing.assert_array_equal(current[0], original[0])
                self.assertEqual(current[3], original[3])
        low = study.draw(inputs, dict(bins=60, reference='old', r=.02), .005, 'conditional', 17)[0]
        high = study.draw(inputs, dict(bins=60, reference='old', r=.08), .005, 'conditional', 17)[0]
        self.assertGreater(high[0], low[0]+1000)

    def test_data_and_mc_pair_between_binnings_and_phases_are_separate(self):
        inputs = study.prepare_inputs(example_parent())
        hashes = []
        for phase in ('calibration', 'validation', 'conditional'):
            fine = study.draw(inputs, dict(bins=60, reference='old', r=.02), .005, phase, 17)
            coarse = study.draw(inputs, dict(bins=30, reference='old', r=.02), .005, phase, 17)
            for index in (0, 1):
                np.testing.assert_array_equal(coarse[index], study.binning.coarsen(fine[index], 30))
            self.assertEqual(fine[3], coarse[3])
            hashes.append(fine[3])
            if phase == 'conditional':
                np.testing.assert_array_equal(fine[1], inputs['raw_templates'])
        self.assertEqual(len(set(hashes)), 3)

    def test_envelope_covers_every_declared_reference_r_and_method(self):
        result = study.freeze(calibration_records())
        self.assertEqual(study.P['comparisons'], 2*2*2*3*(3+1))
        self.assertEqual(result['tolerance']['comparisons'], 96)
        for bins in study.P['bins']:
            for method in study.finite.METHODS:
                for statistic in ('q0', 'q_upper'):
                    local = [result['local'][study.case_id(**case)][method][statistic]
                             for case in study.cases() if case['bins'] == bins]
                    self.assertEqual(result['frozen'][str(bins)]['cutoffs'][method][statistic], max(local))

    def test_validation_missing_cases_and_short_calibration_cannot_set_thresholds(self):
        records = calibration_records()
        name = next(iter(records))
        records[name]['configuration']['phase'] = 'validation'
        with self.assertRaises(ValueError):
            study.freeze(records)
        records = calibration_records()
        records.pop(name)
        with self.assertRaises(ValueError):
            study.freeze(records)
        records = calibration_records()
        records[name]['records']['05000']['joint']['q_upper'] = [1.]
        with self.assertRaises(ValueError):
            study.freeze(records)


if __name__ == '__main__':
    unittest.main()
