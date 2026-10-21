"""Independent checks of boundary integration and null-sample generation."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import zero_branch_162_study as study


class BoundaryStudyTests(unittest.TestCase):
    def setUp(self):
        self.templates = np.eye(3)*1e9

    def test_zero_count_limit_agrees_with_dirichlet_and_uses_95th_percentile(self):
        counts = np.array([50.,950.,0.])
        result = study.fit_boundary(counts,self.templates,np.ones(3),
                                    grid_size=801,pseudocount=0.)
        for key,quantile in (("lower95",.025),("median",.5),("upper95",.975)):
            expected=beta.ppf(quantile,counts+1,counts.sum()-counts+2)
            np.testing.assert_allclose(result[key],expected,atol=2e-6)
        expected=beta.ppf(.95,1,counts.sum()+2)
        self.assertAlmostEqual(result["direct_upper95"],expected,delta=2e-6)
        self.assertLess(result["direct_upper95"],result["upper95"][2])
        self.assertEqual(result["map"][2],0.)
        self.assertAlmostEqual(result["q0"],0.,delta=1e-9)

    def test_positive_count_marginals_match_analytic_dirichlet(self):
        counts=np.array([50.,930.,20.])
        result=study.fit_boundary(counts,self.templates,np.ones(3),grid_size=401,pseudocount=0.)
        np.testing.assert_allclose(result["map"],counts/counts.sum(),atol=1e-10)
        for key,q in (("lower95",.025),("median",.5),("upper95",.975)):
            np.testing.assert_allclose(result[key],beta.ppf(q,counts+1,counts.sum()-counts+2),atol=3e-6)

    def test_unequal_efficiency_flat_generator_prior_matches_independent_quadrature(self):
        counts=np.array([2.,7.,0.])
        efficiency=np.array([.8,.4,.2])
        result=study.fit_boundary(counts,self.templates,efficiency,grid_size=401,pseudocount=0.)
        def marginal(d):
            def integrand(f0):
                f=np.array([f0,1-f0-d,d])
                weighted=f*efficiency
                return (weighted[0]/weighted.sum())**2*(weighted[1]/weighted.sum())**7
            return quad(integrand,0,1-d,epsabs=1e-13,epsrel=1e-9)[0]
        norm=quad(marginal,0,1,epsabs=1e-13,epsrel=1e-9)[0]
        expected=brentq(lambda x:quad(marginal,0,x,epsabs=1e-13)[0]/norm-.95,0,1)
        self.assertAlmostEqual(result["direct_upper95"],expected,delta=2e-5)
        mode=counts/efficiency; mode/=mode.sum()
        np.testing.assert_allclose(result["map"],mode,atol=1e-10)

    def test_invalid_inputs_are_rejected(self):
        for counts,eff in (([1,-1,0],[1,1,1]),([0,0,0],[1,1,1]),
                           ([1,2,0],[1,0,1]),([1,2,np.nan],[1,1,1])):
            with self.assertRaises(ValueError):
                study.fit_boundary(counts,self.templates,eff)

    def test_zero_direct_sampling_preserves_rejection_and_generator_weights(self):
        pools=dict(mixed=np.diag([80,40,20]),mixed_generated=np.full(3,100))
        p=study.generated_probability(pools)
        np.testing.assert_allclose(p,[.04,.38,0.,.58])

    def test_resumption_preserves_exact_samples_and_boundary_fits(self):
        templates=np.diag([80000,40000,20000]).reshape(3,1,3)
        pools=dict(templates=templates,mixed=templates,
                   pure_generated=np.full(3,100000),mixed_generated=np.full(3,100000))
        kwargs=dict(budget=2000,grid_size=51,seed=12,progress_every=2)
        whole=study.run_trials(pools,repetitions=4,**kwargs)
        checkpoints=[]
        study.run_trials(pools,repetitions=2,checkpoint=checkpoints.append,**kwargs)
        resumed=study.run_trials(pools,repetitions=4,resume=checkpoints[-1],**kwargs)
        for key in whole["records"]:
            np.testing.assert_array_equal(whole["records"][key],resumed["records"][key])
        self.assertEqual(whole["summary"]["map_boundary_count"],4)


if __name__ == "__main__":
    unittest.main()
