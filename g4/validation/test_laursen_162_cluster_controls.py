"""Ablations must retain full denominators and reproduce frozen controls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_cluster_controls as controls
from test_laursen_162_cluster_study import fixture, evaluate


def control_fixture(eligible=True):
    group,inferred,baseline = fixture(eligible)
    for estimates in inferred.values():
        for value in estimates.values():
            value.update(geometry_points=1024,strata=4)
    with patch.object(controls.previous.pixel,'evaluate',side_effect=evaluate):
        frozen = controls.cluster.reduce_group(group,inferred,baseline,[0])
    return group,inferred,frozen


class CeilingControlTests(unittest.TestCase):
    def test_clip_is_bounded_idempotent_and_preserves_lower_energies(self):
        direction = np.array([[0.,0.,1.],[0.,0.,-1.]])
        upper = controls.cluster.posterior.single.kinetic_ceiling(direction)
        kinetic = np.array([1.,upper[1]+2.])
        result = controls.ceiling_clip(kinetic,direction)
        np.testing.assert_allclose(result,[1.,upper[1]])
        np.testing.assert_array_equal(controls.ceiling_clip(result,direction),result)
        np.testing.assert_array_equal(kinetic,[1.,upper[1]+2.])

    def test_negative_energy_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            controls.ceiling_clip([-1.],[[0.,0.,1.]])

    def test_scope_precedes_clipping_and_truth_is_not_required(self):
        group,inferred,_ = control_fixture(False)
        row = group['pool'][0];row['baseline_kinetic_MeV'][0] = 100.
        result,hits,_,_ = controls.corrected_cases(row,inferred)
        self.assertEqual(len(hits),0)
        np.testing.assert_array_equal(result[:,0],100.)

    def test_clip_only_changes_eligible_hits_and_does_not_read_truth(self):
        group,inferred,_ = control_fixture()
        row = deepcopy(group['pool'][0]);row['baseline_kinetic_MeV'] = [20.,2.,99.]
        row['truth_kinetic_MeV'] = object()
        result,_,_,_ = controls.corrected_cases(row,inferred)
        self.assertLess(result[1,0],20.)
        self.assertEqual(result[1,1],2.)
        np.testing.assert_array_equal(result[:,2],99.)

    def test_invalid_quadrature_preserves_baseline_without_dropping_event(self):
        group,inferred,_ = control_fixture()
        inferred['17']['main']['valid'] = False
        result,_,_,failed = controls.corrected_cases(group['pool'][0],inferred)
        np.testing.assert_array_equal(result[controls.MAIN],group['pool'][0]['baseline_kinetic_MeV'])
        self.assertEqual(failed[controls.MAIN],1)

    def test_sparse_update_and_direct_contrast_keep_generated_denominator(self):
        group,inferred,frozen = control_fixture()
        with patch.object(controls.previous.pixel,'evaluate',side_effect=evaluate):
            result = controls.reduce_group(group,inferred,frozen,[0])
        np.testing.assert_allclose(result['classification'][0,controls.MAIN,0,0],[.3,.1])
        np.testing.assert_allclose(result['classification'][0,1,0,0],[.4,0.])
        self.assertAlmostEqual(result['main_minus_clip']['value'][0,0,0,0],-.1)
        self.assertAlmostEqual(result['main_minus_clip']['standard_error'][0,0,0,0],np.sqrt(.9)/10)
        self.assertAlmostEqual(result['cohort_moments'][0,0,controls.MAIN,1],.5625)
        self.assertAlmostEqual(result['cohort_moments'][0,0,controls.MAIN,4],.75)

    def test_every_existing_case_is_reproduced_and_cache_tampering_fails(self):
        group,inferred,frozen = control_fixture()
        with patch.object(controls.previous.pixel,'evaluate',side_effect=evaluate):
            result = controls.reduce_group(group,inferred,frozen,[0])
        prior = dict(checksum='frozen',results={ref:dict(summary=frozen) for ref in ('old','new')})
        state = dict(parent_checksum='frozen',cases=controls.CASES,
            results={ref:dict(summary=deepcopy(result)) for ref in ('old','new')})
        controls.validate_cached(state,prior)
        state['results']['old']['summary']['classification'][0,controls.MAIN,0,0,0] += .01
        with self.assertRaises(ArithmeticError):controls.validate_cached(state,prior)

    def test_baseline_changes_are_rejected(self):
        group,inferred,frozen = control_fixture()
        group['pool'][0]['baseline_residual_MeV'] = .1
        with patch.object(controls.previous.pixel,'evaluate',side_effect=evaluate),self.assertRaises(ArithmeticError):
            controls.reduce_group(group,inferred,frozen,[0])


if __name__=='__main__':unittest.main()
