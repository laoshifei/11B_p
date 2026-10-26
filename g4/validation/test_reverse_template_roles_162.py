"""Check role swapping and paired-data generation without transport reruns."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import reverse_template_roles_162_study as study


class ReversalTests(unittest.TestCase):
    def inputs(self):
        original = dict(raw_templates=[[700, 200], [100, 700], [300, 500]], generated=[1000]*3)
        return dict(inputs=dict(banks={"fresh_large": original}, data_generated=[1000]*3,
            data_histograms=[[680, 210], [110, 690], [280, 510]],
            truth=[[.69, .205, .105], [.105, .695, .20], [.29, .505, .205]]))

    def test_swapped_reference_uses_original_mc_and_preserves_rejection(self):
        base = self.inputs()
        inputs = study.make_inputs(base)
        swapped = inputs['independent_reversed']
        np.testing.assert_array_equal(swapped['banks']['coherent']['raw_templates'], base['inputs']['data_histograms'])
        np.testing.assert_allclose(swapped['truth'], [[.7,.2,.1],[.1,.7,.2],[.3,.5,.2]])
        np.testing.assert_allclose(swapped['truth'].sum(axis=1),1.)
        base['inputs']['data_generated'][1] = 999
        with self.assertRaises(ValueError):
            study.make_inputs(base)

    def test_common_reference_keeps_data_identical_when_only_the_actual_bank_changes(self):
        inputs = study.make_inputs(self.inputs())
        a = study.convergence.finite.draw(inputs['common_original'], .005, study.COMMON_SEED, 17, 10000, fluctuate_mc=False)
        b = study.convergence.finite.draw(inputs['common_reversed'], .005, study.COMMON_SEED, 17, 10000, fluctuate_mc=False)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        self.assertFalse(np.array_equal(a[1], b[1]))


if __name__ == '__main__':
    unittest.main()
