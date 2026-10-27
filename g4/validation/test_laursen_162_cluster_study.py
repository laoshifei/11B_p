"""Sparse updates must retain unchanged reactions and the original cohort denominator."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.efficiency import laursen_162_cluster_study as study


def fixture(eligible=True):
    matched=dict(detector=[0,0,-1],pixel=[[0,2],[0,5],[0,0]],energy=[1.,2.,0.],
                 multiplicity=2,ambiguous=False,multiple=False,unmatched=False,summed=True,shared=False)
    observed=np.zeros((2,2,16));observed[0,0,0]=3.;observed[0,1,[2,5]]=[1.,2.]
    labels=np.zeros((2,2,2));labels[:,0,0]=1.
    row=dict(index=17,weights=[1.],matched=matched,observed=observed,policy_agreement=eligible,
        baseline_kinetic_MeV=[1.,2.,0.],nominal_direction=np.array([[1.,0,0],[1.,0,0],[0,0,0]]),
        baseline_labels=labels,baseline_residual_MeV=0.,ancestry=0)
    group=dict(pool=[row],generated_weight=[10.],squared_generated_weight=[10.])
    classification=np.zeros((1,3,2,2,2));classification[0,1,:,0,0]=.4
    ancestry=np.zeros((1,3,2,2,4,2));ancestry[0,1,:,0,0,0]=.4
    moments=np.zeros((1,4,3,6));moments[0,0,1,4]=1.
    frozen=dict(classification=dict(value=classification),ancestry=dict(value=ancestry),
        residual_partition=dict(cohort_per_generated=np.array([[.4,0,0,0]]),cohort_moments=moments))
    estimates={key:dict(valid=True,mean=[3.,4.],effective_rays=100.,rank_fallback_rays=0) for key in study.posterior.VARIANTS}
    return group,{'17':estimates} if eligible else {},frozen


def evaluate(matched,kinetic,direction,nominal):
    labels=np.zeros((len(kinetic),2,2));changed=kinetic[:,0]>2.
    labels[~changed,0,0]=1.;labels[changed,0,1]=1.
    return labels,1.5*changed


class SparseTests(unittest.TestCase):
    def test_changed_event_keeps_full_generated_denominator(self):
        group,inferred,frozen=fixture()
        with patch.object(study.previous.pixel,'evaluate',side_effect=evaluate):
            result=study.reduce_group(group,inferred,frozen,[0])
        np.testing.assert_allclose(result['classification'][0,1,0,0],[.3,.1])
        np.testing.assert_allclose(result['classification'][0,1,1,0],[.3,0.])
        self.assertAlmostEqual(result['paired_change']['standard_error'][0,1,0,0,0],np.sqrt(.9)/10)

    def test_fixed_cohort_does_not_become_the_selected_pool(self):
        group,inferred,frozen=fixture()
        with patch.object(study.previous.pixel,'evaluate',side_effect=evaluate):
            result=study.reduce_group(group,inferred,frozen,[0])
        value=result['cohort_moments'][0,0,1]
        self.assertAlmostEqual(value[0],.375)
        self.assertAlmostEqual(value[1],.5625)
        self.assertAlmostEqual(value[4],.75)
        self.assertAlmostEqual(value[5],-.25)

    def test_out_of_scope_event_leaves_all_results_unchanged(self):
        group,inferred,frozen=fixture(False)
        with patch.object(study.previous.pixel,'evaluate',side_effect=evaluate):
            result=study.reduce_group(group,inferred,frozen,[0])
        self.assertFalse(np.any(result['paired_change']['value']))
        self.assertFalse(np.any(result['cohort_paired_change_per_generated']['value']))

    def test_failed_integration_is_a_fallback_not_event_removal(self):
        group,inferred,frozen=fixture()
        for value in inferred['17'].values():value['valid']=False
        with patch.object(study.previous.pixel,'evaluate',side_effect=evaluate):
            result=study.reduce_group(group,inferred,frozen,[0])
        np.testing.assert_array_equal(result['diagnostics']['failures'],1)
        self.assertFalse(np.any(result['paired_change']['value']))

    def test_pool_baseline_mismatch_is_detected(self):
        group,inferred,frozen=fixture();group['pool'][0]['baseline_residual_MeV']=.1
        with patch.object(study.previous.pixel,'evaluate',side_effect=evaluate),self.assertRaises(ArithmeticError):
            study.reduce_group(group,inferred,frozen,[0])


class CheckpointTests(unittest.TestCase):
    def test_complete_record_survives_partial_next_write_and_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            first=study.Checkpoint(directory,dict(source='unchanged'))
            first.put(('old','0','17'),dict(mean=np.array([1.,2.])))
            (Path(directory)/'old_0_18.json.pending').write_text('interrupted')
            first.close()
            resumed=study.Checkpoint(directory,dict(source='unchanged'))
            try:
                self.assertEqual(resumed.get(('old','0','17')),dict(mean=[1.,2.]))
                self.assertIsNone(resumed.get(('old','0','18')))
                with self.assertRaises(FileExistsError):resumed.put(('old','0','17'),dict(mean=[3.,4.]))
            finally:resumed.close()

    def test_changed_source_or_plan_cannot_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first=study.Checkpoint(directory,dict(source='old',plan=1));first.close()
            with self.assertRaisesRegex(ValueError,'inputs, plan or source'):
                study.Checkpoint(directory,dict(source='new',plan=1))

    def test_valid_checksum_cannot_hide_a_wrong_event_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            journal=study.Checkpoint(directory,dict(source='same'))
            try:
                journal.put(('old','0','17'),dict(mean=[1.,2.]))
                raw=journal.record_path(('old','0','17')).read_bytes()
                journal.record_path(('new','0','17')).write_bytes(raw)
                with self.assertRaisesRegex(ValueError,'event identity'):
                    journal.get(('new','0','17'))
            finally:journal.close()

    def test_second_writer_is_rejected_until_lock_is_released(self):
        with tempfile.TemporaryDirectory() as directory:
            first=study.Checkpoint(directory,dict(source='same'))
            try:
                with self.assertRaises(BlockingIOError):study.Checkpoint(directory,dict(source='same'))
            finally:first.close()
            resumed=study.Checkpoint(directory,dict(source='same'));resumed.close()


if __name__=='__main__':unittest.main()
