"""Global-library provenance, baseline reproduction and comparison guards."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.bayes import independent_library_162_study as study


def example_inputs():
    count = study.binning.fine_to_coarse_mapping(60)[1]
    old = np.zeros((3,count+1))
    old[:,:3] = [[.7,.1,.1],[.1,.7,.1],[.2,.2,.5]]
    old[:,-1] = .1
    new = old.copy()
    new[:,:3] = [[.6,.2,.1],[.2,.6,.1],[.2,.3,.4]]
    generated = np.asarray(study.P['generated'])
    a = old[:,:-1]*generated[:,None]
    b = new[:,:-1]*generated[:,None]
    return dict(references=dict(old=old,new=new),templates=dict(
        old=dict(raw_templates=a,generated=generated),new=dict(raw_templates=b,generated=generated)))


class IndependentLibraryTests(unittest.TestCase):
    def test_partition_is_before_selection_and_covers_unique_generated_ids(self):
        ids = np.random.default_rng(3).permutation(np.arange(100,500))
        template,reference = study.partition(ids,0)
        self.assertEqual((len(template),len(reference)),(200,200))
        self.assertEqual(len(np.intersect1d(template,reference)),0)
        np.testing.assert_array_equal(np.sort(np.r_[template,reference]),np.sort(ids))
        with self.assertRaises(ValueError):
            study.partition(ids[:-1],0)
        with self.assertRaises(ValueError):
            study.partition([1,1],0)

    def test_conditional_old_draw_exactly_reproduces_full_budget_parent(self):
        inputs = example_inputs()
        for bins in study.P['bins']:
            case = dict(bins=bins,reference='old',template='old')
            current = study.draw(inputs,case,.005,study.P['old_conditional_seed'],17)
            old_case = dict(bins=bins,scale=4,generated=inputs['templates']['old']['generated'],
                            raw_templates=study.binning.coarsen(inputs['templates']['old']['raw_templates'],bins))
            parent = study.binning.paired_draw(dict(fine_truth=inputs['references']['old']),old_case,
                .005,study.P['old_conditional_seed'],17,fluctuate_mc=False)
            for a,b in zip(current[:3],parent[:3]):
                np.testing.assert_array_equal(a,b)
            self.assertEqual(current[3],parent[3])

    def test_data_pairing_within_reference_and_mc_fine_coarse_aggregation(self):
        inputs = example_inputs()
        for reference in ('old','new'):
            seed,_,_,_ = study.phase_settings('conditional',reference)
            a = study.draw(inputs,dict(bins=60,reference=reference,template='old'),.005,seed,17)
            b = study.draw(inputs,dict(bins=60,reference=reference,template='new'),.005,seed,17)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertFalse(np.array_equal(a[1],b[1]))
        fine = study.draw(inputs,dict(bins=60,reference='new',template='new'),.005,study.P['new_calibration_seed'],17,fluctuate_mc=True)
        coarse = study.draw(inputs,dict(bins=30,reference='new',template='new'),.005,study.P['new_calibration_seed'],17,fluctuate_mc=True)
        np.testing.assert_array_equal(coarse[0],study.binning.coarsen(fine[0],30))
        np.testing.assert_array_equal(coarse[1],study.binning.coarsen(fine[1],30))

    def test_new_calibration_is_distinct_from_external_and_validation_samples(self):
        row = {method:dict(q_upper=np.linspace(0,5,1000),q0=np.linspace(0,4,1000)) for method in study.finite.METHODS}
        state = dict(configuration=dict(phase='calibration',seed=study.P['new_calibration_seed'],
            repetitions=1000,case=dict(reference='new')),records={key:row for key in ('00000','05000','20000')})
        self.assertEqual(study.freeze(state)['tolerance']['comparisons'],16)
        self.assertEqual(study.freeze(state)['tolerance']['rank'],969)
        state['configuration']['phase']='validation'
        with self.assertRaises(ValueError):
            study.freeze(state)
        with self.assertRaises(ValueError):
            study.phase_settings('calibration','old')

    def test_paired_contrasts_reject_unrelated_reference_streams(self):
        def ensemble(values,hashes):
            return dict(records={'05000':{method:dict(map=np.column_stack([np.zeros(len(values)),np.zeros(len(values)),values]),
                paired_data_sha256=hashes) for method in study.finite.METHODS}})
        a = ensemble([.1,.2,.3],['a','b','c'])
        b = ensemble([.11,.21,.31],['a','b','c'])
        paired = study.compare_estimates(a,b,paired=True)['05000']['joint']
        self.assertAlmostEqual(paired['right_minus_left'],.01,places=12)
        self.assertLess(paired['mcse'],1e-15)
        c = ensemble([.11,.21,.31],['x','y','z'])
        with self.assertRaises(ValueError):
            study.compare_estimates(a,c,paired=True)
        self.assertGreater(study.compare_estimates(a,c,paired=False)['05000']['joint']['mcse'],0)

    def test_generation_remembers_previous_global_seeds_and_preserves_macro(self):
        metadata = dict(old_libraries=dict(runs=dict(a=dict(validation=dict(seed=11)))),
            other_libraries=dict(runs=dict(b=dict(validation=dict(seed=22)))),
            codex_independent_replication_libraries=dict(runs=dict(c=dict(validation=dict(seed=33)))))
        self.assertEqual(study.generator.previous_seeds(metadata),[11,22])
        base = (study.ROOT/'g4/macros/model_mismatch_162.mac').read_text()
        for name,count in study.generator.INCIDENTS.items():
            macro = study.generator.macro_text(base,name,count)
            self.assertIn(f'/run/beamOn {count}\n',macro)
            self.assertIn('/gun/energy 162 keV\n',macro)
            self.assertIn('/h11b/675ScaleFactor 0\n',macro)
            self.assertIn('/target/thickness 0.12 um\n',macro)


if __name__=='__main__':
    unittest.main()
