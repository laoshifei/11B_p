"""Independent checks of shared-strip likelihood and joint integration."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_cluster_posterior as cluster


class ClusterTests(unittest.TestCase):
    def test_shared_strip_enters_once_and_creates_negative_charge_covariance(self):
        matrix=np.array([[[1.,1.],[1.,0.],[0.,1.]]])
        data=np.array([3.3,1.0,2.0]);sigma=.03
        mean,covariance,chi2,good=cluster.linear_gaussian(matrix,data,sigma)
        np.testing.assert_allclose(mean[0],[1.1,2.1])
        np.testing.assert_allclose(covariance[0],sigma*sigma/3*np.array([[2.,-1.],[-1.,2.]]))
        self.assertTrue(good[0])
        self.assertAlmostEqual(chi2[0],(.3**2)/(3*sigma*sigma))

    def test_joint_fit_is_invariant_to_strip_row_order(self):
        matrix=np.array([[[1.,1.],[1.,0.],[0.,1.]]]);data=np.array([3.3,1.,2.])
        a=cluster.linear_gaussian(matrix,data,.03)
        b=cluster.linear_gaussian(matrix[:,[2,0,1]],data[[2,0,1]],.03)
        for left,right in zip(a,b):np.testing.assert_allclose(left,right)

    def test_joint_fit_tracks_hit_column_permutation(self):
        matrix=np.array([[[1.,1.],[1.,0.],[0.,1.]]]);data=np.array([3.3,1.,2.])
        a=cluster.linear_gaussian(matrix,data,.03)
        b=cluster.linear_gaussian(matrix[:,:,::-1],data,.03)
        np.testing.assert_allclose(a[0][:,::-1],b[0])
        np.testing.assert_allclose(a[1][:,::-1,::-1],b[1])

    def test_rank_deficiency_is_explicit(self):
        matrix=np.ones((1,2,2))
        _,_,_,good=cluster.linear_gaussian(matrix,np.ones(2),.03)
        self.assertFalse(good[0])

    def test_independent_bounded_normal_probability(self):
        mean=np.zeros((1,2));covariance=np.eye(2)[None];maximum=np.ones((1,2))
        charge,weight=cluster.bounded_gaussian_nodes(mean,covariance,maximum,20)
        expected=(cluster.ndtr(1.)-.5)**2
        self.assertAlmostEqual(weight.sum(),expected,places=13)
        self.assertTrue(np.all((charge>0)&(charge<1)))
        # Truncated standard normal mean on [0,1].
        phi0=1/np.sqrt(2*np.pi);expected_mean=phi0*(1-np.exp(-.5))/(cluster.ndtr(1.)-.5)
        np.testing.assert_allclose((weight[...,None]*charge).sum(axis=1)/weight.sum(axis=1)[:,None],expected_mean,atol=1e-12)

    def test_correlated_normal_quadrant_probability(self):
        rho=-.5
        charge,weight=cluster.bounded_gaussian_nodes(np.zeros((1,2)),np.array([[[1.,rho],[rho,1.]]]),np.full((1,2),9.),64)
        self.assertAlmostEqual(weight.sum(),.25+np.arcsin(rho)/(2*np.pi),places=6)
        self.assertTrue((charge>=0).all())

    def test_strata_cover_pixel_without_overlap(self):
        for strip in (0,1,7,15):
            intervals=cluster.strip_intervals(strip,.1,extra_cut=(strip+.4)*cluster.response.PITCH-25)
            low=max(-25,strip*cluster.response.PITCH-25-.05)
            high=min(25,(strip+1)*cluster.response.PITCH-25+.05)
            self.assertAlmostEqual(sum(b-a for a,b in intervals),high-low)
            for left,right in zip(intervals[:-1],intervals[1:]):self.assertEqual(left[1],right[0])

    def test_zero_width_has_no_sharing_regions(self):
        self.assertEqual(cluster.strip_intervals(7,0.),((-3.125,0.),))

    def test_log_moments_survive_extreme_likelihood_offsets(self):
        result=[]
        for offset in (0.,-10000.,10000.):
            accum=cluster.LogMoments()
            accum.add(np.log(np.array([[1.,3.]]))+offset,np.array([[[1.,2.],[3.,4.]]]))
            result.append(accum.finish())
        for row in result:
            np.testing.assert_allclose(row['mean'],[2.5,3.5],atol=1e-12)
            np.testing.assert_allclose(row['covariance'],np.full((2,2),.75),atol=1e-12)
        self.assertAlmostEqual(result[2]['log_normalization']-result[0]['log_normalization'],10000.)

    def test_log_moments_combine_different_strata(self):
        accum=cluster.LogMoments()
        accum.add(np.log(np.array([[.25]])),np.array([[[1.,2.]]]))
        accum.add(np.log(np.array([[.75]])),np.array([[[3.,4.]]]))
        np.testing.assert_allclose(accum.finish()['mean'],[2.5,3.5])

    def test_zero_evidence_is_reported(self):
        accum=cluster.LogMoments()
        accum.add(np.array([[-np.inf]]),np.full((1,1,2),np.nan))
        self.assertFalse(accum.finish()['valid'])

    def test_observable_eligibility_excludes_unexplained_strip(self):
        m=dict(detector=np.array([0,0,-1]),pixel=np.array([[0,2],[0,5],[0,0.]]),multiplicity=2,ambiguous=False)
        observed=np.zeros((2,2,16));observed[0,0,0]=3.;observed[0,1,[2,5]]=[1.,2.]
        hits,reason=cluster.eligible_cluster(m,observed,True)
        np.testing.assert_array_equal(hits,[0,1]);self.assertEqual(reason,'eligible')
        observed[0,0,7]=.25
        self.assertEqual(cluster.eligible_cluster(m,observed,True)[1],'unexplained_strip')
        self.assertEqual(cluster.eligible_cluster(m,observed,False)[1],'unresolved_topology')

    def test_complete_estimator_canonicalizes_hit_order(self):
        engine=object.__new__(cluster.ClusterPosterior);engine.quadrature=cluster.Quadrature()
        observed=np.zeros((2,16));observed[0,0]=3.;observed[1,[2,5]]=[1.,2.]
        def evaluate(detector,pixels,calibrated,*rule):
            np.testing.assert_array_equal(pixels,[[0,2],[0,5]])
            return dict(mean=np.array([1.,2.]),covariance=np.array([[3.,.2],[.2,4.]]))
        with patch.object(engine,'integrate',side_effect=evaluate):
            result=engine.estimate(0,np.array([[0,5],[0,2]]),observed)
        for value in result.values():
            np.testing.assert_array_equal(value['mean'],[2.,1.])
            np.testing.assert_array_equal(value['covariance'],[[4.,.2],[.2,3.]])


if __name__=='__main__':unittest.main()
