"""Regression checks for weighted truth / unweighted MC separation."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.bayes.rmatrix_162_model import Parameters, boost
from analysis.bayes import rmatrix_162_study as study


class StudyTests(unittest.TestCase):
    def test_rejected_weights_enter_normalization(self):
        blocks=np.array([[[1.,2.,7.]],[[4.,2.,4.]]])
        probability=blocks.sum(axis=0)/blocks.sum()
        base=np.array([[.8,.1,.1],[.1,.8,.1],[.4,.4,.2]])
        inputs=dict(base_references={'old':base},weighted={'old':dict(
            probability=probability,block_histograms=blocks)})
        result=study.mixture(inputs,'old',0,0.)
        np.testing.assert_allclose(result,.05*base[0]+.95*np.array([.25,.2,.55]))
        self.assertAlmostEqual(result.sum(),1.)

    def test_block_bootstrap_is_self_normalized(self):
        blocks=np.array([[[1.,2.,7.]],[[4.,2.,4.]]])
        base=np.array([[.8,.1,.1],[.1,.8,.1],[.4,.4,.2]])
        inputs=dict(base_references={'old':base},weighted={'old':dict(block_histograms=blocks)})
        result=study.mixture(inputs,'old',0,.005,np.array([2,0]))
        expected=.995*(.05*base[0]+.95*np.array([.1,.2,.7]))+.005*base[2]
        np.testing.assert_allclose(result,expected)

    def test_dalitz_categories_match_histogram(self):
        points=np.array([[.05,.1],[.11,.22],[.2,.5],[.03,.8]])
        category=study.category_indices(points)
        hist=study.mismatch.histogram_xy(points,30)[study.mismatch.geometric_support(30)]
        np.testing.assert_array_equal(np.bincount(category,minlength=len(hist)),hist)
        with self.assertRaises(ValueError):study.category_indices(np.array([[-.2,-.4]]))

    def test_reaction_lorentz_round_trip(self):
        mass=Parameters().mass
        p=np.array([[[25.,8.,15.],[-10.,-20.,2.],[-15.,12.,-17.]]])
        e=np.sqrt(mass*mass+np.sum(p*p,axis=-1))
        el,pl=boost(e,p,np.array([[[.001,.002,.003]]]))
        array={}
        for i in range(3):
            array[f'e_alpha{i+1}']=el[:,i]-mass
            array[f'theta_lab_alpha{i+1}']=np.arccos(pl[:,i,2]/np.linalg.norm(pl[:,i],axis=-1))
            array[f'phi_lab_alpha{i+1}']=np.arctan2(pl[:,i,1],pl[:,i,0])
        restored=study.reaction_vectors(array,mass)
        np.testing.assert_allclose(restored[...,:3],p,atol=1e-8,rtol=0)
        np.testing.assert_allclose(restored[...,3],e-mass,atol=1e-9,rtol=0)

    def test_invalid_inputs_fail(self):
        for change in [dict(reduced_width=-1),dict(separation=0),dict(energy_min=4),dict(grid_size=32)]:
            with self.assertRaises(ValueError):Parameters(**change)

    def test_wilson_boundaries(self):
        low,high=study.wilson(0,500)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,0.)
        low,high=study.wilson(500,500)
        self.assertAlmostEqual(high,1.)
        self.assertLess(low,1.)


if __name__=='__main__':unittest.main()
