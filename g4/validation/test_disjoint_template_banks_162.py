"""Scientific input and calibration guards for the four-bank comparison."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.bayes import disjoint_template_banks_162_study as study


class DisjointBankTests(unittest.TestCase):
    def test_restarts_preserve_original_analytic_likelihood_and_profiles(self):
        mc=np.array([[800.,60.,40.],[20.,850.,30.],[30.,70.,800.]])
        generated=np.array([1000.,1100.,1200.])
        fractions=np.array([.05,.945,.005])
        weights=study.accepted_weights(fractions,mc.sum(axis=1)/generated)
        data=10000*(weights@(mc/mc.sum(axis=1,keepdims=True)))
        a=study.JointMultinomial(data,mc,generated)
        b=study.OriginalJointMultinomial(data,mc,generated)
        np.testing.assert_allclose(a.mode,fractions,atol=1e-7)
        np.testing.assert_allclose(a.mode,b.mode,atol=1e-9)
        self.assertAlmostEqual(a.profile_direct(.01),b.profile_direct(.01),delta=2e-5)

    def test_disjoint_generated_partitions_cover_parent_without_selection(self):
        ids = np.random.default_rng(7).permutation(np.arange(100,500))
        groups = study.split_template(ids)
        self.assertEqual([len(x) for x in groups],[100]*4)
        np.testing.assert_array_equal(np.concatenate(groups),ids)
        self.assertEqual(len(set(np.concatenate(groups))),400)
        for a in range(4):
            for b in range(a+1,4):self.assertEqual(len(np.intersect1d(groups[a],groups[b])),0)
        with self.assertRaises(ValueError):study.split_template(ids[:-1])
        with self.assertRaises(ValueError):study.split_template([1,1,2,3])

    def test_fixed_independent_reference_and_data_are_identical_across_banks(self):
        hist=np.array([[70,20],[10,70],[30,50]])
        reference=study.reaction_population(hist,[100]*3)
        np.testing.assert_allclose(reference,[[.7,.2,.1],[.1,.7,.2],[.3,.5,.2]])
        inputs=dict(references=dict(heldout=reference),banks={str(i):dict(raw_templates=hist+i,generated=[100]*3) for i in range(1,5)})
        draws=[study.convergence.finite.draw(study.trial_inputs(inputs,name,'heldout'),.005,
            study.CONDITIONAL_SEED,17,10000,fluctuate_mc=False) for name in study.BANKS]
        for draw in draws[1:]:np.testing.assert_array_equal(draw[0],draws[0][0])
        self.assertFalse(np.array_equal(draws[0][1],draws[-1][1]))
        with self.assertRaises(ValueError):study.reaction_population(hist,[80]*3)

    def test_only_the_matching_calibration_seed_freezes_cutoffs(self):
        rows={m:dict(q_upper=np.linspace(0,5,1000),q0=np.linspace(0,4,1000)) for m in study.convergence.finite.METHODS}
        state=dict(configuration=dict(seed=study.CONDITIONAL_SEED,repetitions=1000),records={"00000":rows,"05000":rows,"20000":rows})
        with self.assertRaises(ValueError):study.freeze_calibration(state,'heldout')
        state['configuration']['seed']=study.SEEDS['heldout']['calibration']
        frozen=study.freeze_calibration(state,'heldout')
        self.assertEqual(frozen['tolerance']['comparisons'],16)
        with self.assertRaises(ValueError):study.freeze_calibration(state,'pooled')


if __name__=='__main__':unittest.main()
