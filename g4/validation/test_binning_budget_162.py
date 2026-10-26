"""Aggregation, pairing and calibration-scope guards for the factorial study."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.bayes import binning_budget_162_study as study


def example_inputs():
    count = study.fine_to_coarse_mapping(60)[1]
    reference = np.zeros((3,count+1))
    reference[:,:3] = [[.7,.1,.1],[.1,.7,.1],[.2,.2,.5]]
    reference[:,-1] = .1
    templates = {}
    for number in range(1,5):
        raw = np.zeros((3,count),np.int64)
        raw[:,:3] = np.array([[70,10,10],[10,70,10],[20,20,50]])*number
        templates[str(number)] = dict(raw_templates=raw,generated=study.banks.GENERATED)
    return study.prepare_inputs(dict(inputs=dict(references=dict(heldout=reference),banks=templates,
        reference_generated=[200000,1000000,200000],reference_ids_sha256=dict(example='fixed'))))


class BinningBudgetTests(unittest.TestCase):
    def test_aggregation_matches_histogram_of_same_points(self):
        rng = np.random.default_rng(43)
        xy = rng.uniform(0,1,(10000,2))
        xy = xy[(xy[:,1]>=xy[:,0]/np.sqrt(3)) & (np.sum(xy**2,axis=1)<=1)]
        histogram = study.convergence.mismatch.histogram_xy
        support = study.convergence.mismatch.geometric_support
        fine = histogram(xy,60)[support(60)]
        coarse = histogram(xy,30)[support(30)]
        np.testing.assert_array_equal(study.coarsen(fine,30),coarse)
        self.assertEqual((len(fine),len(coarse)),(497,144))
        self.assertEqual(int(coarse.sum()),len(xy))
        with self.assertRaises(ValueError):
            study.coarsen(fine,45)

    def test_actual_template_groups_add_counts_and_generated_budgets(self):
        inputs = example_inputs()
        cases = inputs['cases']
        for bins in study.BINS:
            first,second = (cases[study.case_id(bins,(name,))] for name in ('1','2'))
            middle = cases[study.case_id(bins,('1','2'))]
            np.testing.assert_array_equal(middle['raw_templates'],np.array(first['raw_templates'])+second['raw_templates'])
            np.testing.assert_array_equal(middle['generated'],2*np.array(study.banks.GENERATED))
            np.testing.assert_array_equal(cases[study.case_id(bins,('1','2','3','4'))]['generated'],4*np.array(study.banks.GENERATED))

    def test_fine_small_draws_reproduce_parent_and_all_draws_remain_paired(self):
        inputs = example_inputs()
        fine = inputs['cases']['60_x1_1']
        original = dict(truth=inputs['fine_truth'],banks=dict(coherent=fine))
        old = study.finite.draw(original,.005,study.CALIBRATION_SEED,17,100000)
        new = study.paired_draw(inputs,fine,.005,study.CALIBRATION_SEED,17,fluctuate_mc=True)
        for before,after in zip(old,new[:3]):
            np.testing.assert_array_equal(before,after)
        draws = {name:study.paired_draw(inputs,case,.005,study.CALIBRATION_SEED,17,fluctuate_mc=True)
                 for name,case in inputs['cases'].items()}
        self.assertEqual(len({draw[3] for draw in draws.values()}),1)
        for scale in study.GROUPS:
            members = study.GROUPS[scale][0]
            a,b = (draws[study.case_id(bins,members)] for bins in study.BINS)
            np.testing.assert_array_equal(a[0],study.coarsen(b[0],30))
            np.testing.assert_array_equal(a[1],study.coarsen(b[1],30))
        small,middle,large = (draws[name][1] for name in ('60_x1_1','60_x2_12','60_x4_1234'))
        self.assertTrue(np.all(small<=middle) and np.all(middle<=large))

    def test_calibration_is_separate_and_accounts_for_all_six_configurations(self):
        row = {method:dict(q_upper=np.linspace(0,5,1000),q0=np.linspace(0,4,1000)) for method in study.finite.METHODS}
        state = dict(configuration=dict(phase='validation',seed=study.CALIBRATION_SEED,repetitions=1000),
                     records={key:row for key in ('00000','05000','20000')})
        with self.assertRaises(ValueError):
            study.freeze(state)
        state['configuration']['phase']='calibration'
        frozen = study.freeze(state)
        self.assertEqual(frozen['tolerance']['comparisons'],48)
        self.assertEqual(frozen['tolerance']['rank'],971)
        state['configuration']['seed']=study.CONDITIONAL_SEED
        with self.assertRaises(ValueError):
            study.freeze(state)

    def test_pairing_check_rejects_different_pseudo_data(self):
        def record(value):
            return dict(records={'05000':dict(joint=dict(paired_data_sha256=np.array([value])))})
        records=dict(a=record('same'),b=record('same'))
        study.verify_pairing(records,['a','b'])
        records['b']=record('different')
        with self.assertRaises(RuntimeError):
            study.verify_pairing(records,['a','b'])


if __name__=='__main__':
    unittest.main()
