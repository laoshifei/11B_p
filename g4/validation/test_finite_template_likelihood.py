"""Independent analytic and direct-optimization controls for finite MC."""

from pathlib import Path
import errno
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, xlogy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes.finite_template_likelihood import JointMultinomial, accepted_weights, check_stationarity


class FiniteTemplateTests(unittest.TestCase):
    def test_stiff_parameter_score_uses_likelihood_scale_and_checks_bad_optimum(self):
        def stiff(x):
            return float(.5*1e8*x[0]**2), 1e8*np.asarray(x)
        score, decrement = check_stationarity(stiff, np.array([1e-8]), [(-1.,1.)])
        self.assertGreater(score, .1)
        self.assertLess(decrement, 1e-7)
        def unresolved(x):
            return float(.5*x[0]**2), np.asarray(x)
        with self.assertRaises(RuntimeError):
            check_stationarity(unresolved, np.array([.5]), [(-1.,1.)])

    def test_asimov_recovers_generator_fractions_at_unequal_efficiencies(self):
        mc = np.array([[800., 60., 40.], [20., 850., 30.], [30., 70., 800.]])
        generated = np.array([1000., 1100., 1200.])
        fractions = np.array([.05, .945, .005])
        probability = accepted_weights(fractions, mc.sum(axis=1)/generated) @ (mc/mc.sum(axis=1, keepdims=True))
        model = JointMultinomial(10000*probability, mc, generated)
        np.testing.assert_allclose(model.mode, fractions, atol=1e-7)
        self.assertLess(abs(model.loss), 1e-6)
        self.assertLess(model.statistic(.005), 1e-5)
        self.assertLess(model.max_shape_gap, 1.1e-4)

    def test_profile_agrees_with_independent_full_parameter_minimization(self):
        mc = np.array([[190., 30., 20.], [25., 180., 35.], [20., 40., 180.]])
        generated = np.array([300., 320., 340.])
        data = np.array([37., 222., 41.])
        model = JointMultinomial(data, mc, generated)
        reference_shapes = mc/mc.sum(axis=1, keepdims=True)
        reference_efficiency = mc.sum(axis=1)/generated

        def objective(parameters, direct):
            r = 1/(1+np.exp(-parameters[0]))
            shapes = softmax(np.column_stack([parameters[1:7].reshape(3, 2), np.zeros(3)]), axis=1)
            efficiency = 1/(1+np.exp(-parameters[7:]))
            fractions = np.array([r*(1-direct), (1-r)*(1-direct), direct])
            q = accepted_weights(fractions, efficiency) @ shapes
            # All probabilities and counts here are positive, so a plain
            # direct likelihood is independent of EM and sparse-bin code.
            loss = np.sum(xlogy(data, (data/data.sum())/q))
            loss += np.sum(xlogy(mc, reference_shapes/shapes))
            accepted, rejected = mc.sum(axis=1), generated-mc.sum(axis=1)
            loss += np.sum(xlogy(accepted, reference_efficiency/efficiency)
                           +xlogy(rejected, (1-reference_efficiency)/(1-efficiency)))
            return loss

        initial = np.r_[np.log(.12/.88), np.log(reference_shapes[:, :2]/reference_shapes[:, 2:]).ravel(),
                        np.log(reference_efficiency/(1-reference_efficiency))]
        for direct in (.01, .10, .25):
            independent = minimize(objective, initial, args=(direct,), method='BFGS',
                                   options=dict(gtol=1e-6, maxiter=500))
            self.assertAlmostEqual(model.profile_direct(direct), independent.fun, delta=2e-5)

    def test_zero_mc_bin_can_have_positive_fitted_probability(self):
        mc = np.array([[80., 10., 0., 0.], [10., 80., 0., 0.], [5., 5., 80., 0.]])
        model = JointMultinomial([15., 75., 8., 2.], mc, [100., 110., 120.])
        result = model.shape(model.weights)
        self.assertTrue(np.isfinite(model.loss))
        self.assertGreater(result['q'][3], 0)
        np.testing.assert_allclose(result['shapes'].sum(axis=1), 1.)
        self.assertGreater(np.max(result['shapes'][:, 3]), 0)

    def test_nuisance_profiling_cannot_worsen_fixed_shape_fit(self):
        mc = np.array([[90., 7., 3.], [4., 91., 5.], [5., 10., 85.]])
        data = np.array([20., 150., 30.])
        model = JointMultinomial(data, mc, [120., 140., 150.])
        weights = np.array([.1, .8, .1])
        fixed_q = weights @ (mc/mc.sum(axis=1, keepdims=True))
        fixed_loss = np.sum(xlogy(data, (data/data.sum())/fixed_q))
        self.assertLessEqual(model.shape(weights)['loss'], fixed_loss+1e-7)
        self.assertLessEqual(model.profile_direct(.15), model.profile_direct(.15, vary_efficiency=False)+1e-6)

    def test_null_boundary_and_upper_inversion(self):
        mc = np.array([[800., 50., 50.], [50., 800., 50.], [50., 50., 800.]])
        efficiency = mc.sum(axis=1)/np.array([1000., 1100., 1200.])
        weights = accepted_weights([.05, .95, 0.], efficiency)
        data = 1000*(weights @ (mc/mc.sum(axis=1, keepdims=True)))
        model = JointMultinomial(data, mc, [1000., 1100., 1200.])
        self.assertLess(model.mode[2], 1e-7)
        self.assertLess(model.q0, 1e-5)
        upper = model.upper_limit(2.8)
        self.assertGreater(upper, 0.)
        self.assertAlmostEqual(model.statistic(upper), 2.8, delta=2e-4)
        self.assertEqual(model.statistic(.1, upper=True), model.statistic(.1))

    def test_more_mc_approaches_known_shapes(self):
        mc = np.array([[800., 60., 40.], [20., 850., 30.], [30., 70., 800.]])
        generated = np.array([1000., 1100., 1200.])
        fractions = np.array([.05, .94, .01])
        p = accepted_weights(fractions, mc.sum(axis=1)/generated) @ (mc/mc.sum(axis=1, keepdims=True))
        data = 1000*p
        small = JointMultinomial(data, mc, generated)
        large = JointMultinomial(data, mc*10000, generated*10000, tolerance=1e-5)
        self.assertLess(small.statistic(.02), large.statistic(.02))
        np.testing.assert_allclose(large.mode, fractions, atol=2e-7)

    def test_invalid_counts_and_fraction_are_rejected(self):
        for data, mc, generated in (([0., 0.], np.ones((3, 2)), [3., 3., 3.]),
                                    ([1., 1.], np.ones((3, 2)), [1., 3., 3.]),
                                    ([1., -1.], np.ones((3, 2)), [3., 3., 3.])):
            with self.assertRaises(ValueError):
                JointMultinomial(data, mc, generated)

    def test_acceptance_boundary_with_no_mc_rejections(self):
        mc = np.array([[80., 10., 10.], [10., 80., 10.], [10., 10., 80.]])
        model = JointMultinomial([10., 80., 10.], mc, [100., 100., 100.])
        self.assertTrue(np.isfinite(model.statistic(.05)))


class FiniteStudyControls(unittest.TestCase):
    def test_protocol_error_retry_preserves_original_notebook_and_metadata(self):
        from analysis.bayes import finite_template_162_study as study
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'main.ipynb'
            original=dict(cells=[dict(source='keep')],metadata=dict(existing='retained'))
            path.write_text(json.dumps(original))
            write=Path.write_text
            attempts=[]
            def flaky(target,content,*args,**kwargs):
                if target.name.endswith('.pending.ipynb') and not attempts:
                    attempts.append(1)
                    self.assertEqual(json.loads(path.read_text()),original)
                    raise OSError(errno.EPROTO,'simulated shared-folder protocol error')
                return write(target,content,*args,**kwargs)
            with patch.object(Path,'write_text',new=flaky),patch.object(study.time,'sleep'):
                study.save_checkpoint(path,dict(marker=42))
            result=json.loads(path.read_text())
            self.assertEqual(result['cells'],original['cells'])
            self.assertEqual(result['metadata']['existing'],'retained')
            self.assertEqual(result['metadata']['codex_finite_template_study']['marker'],42)
            self.assertFalse(path.with_suffix('.pending.ipynb').exists())

    def test_common_reference_does_not_forbid_observed_mc_bins(self):
        from analysis.bayes import finite_template_162_study as study
        data=np.array([[5,0,2],[0,8,2],[2,2,6]])
        mc=np.array([[0,6,1],[1,0,9],[3,7,0]])
        truth=study.common_reference(data,[10,12,15],mc,[12,13,17])
        np.testing.assert_allclose(truth.sum(axis=1),1)
        self.assertTrue(np.all(truth[:,:-1][(data+mc)>0]>0))
        np.testing.assert_allclose(truth[:,:-1],(data+mc)/np.array([22,25,32])[:,None])

    def test_joint_draw_has_fixed_generated_mc_and_independent_streams(self):
        from analysis.bayes import finite_template_162_study as study
        inputs = dict(truth=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1]]),
                      banks=dict(coherent=dict(generated=[100,120,140],
                          raw_templates=[[70,10,10],[10,100,5],[10,10,100]])))
        first = study.draw(inputs, .005, study.CALIBRATION_SEED, 0, 1000)
        repeated = study.draw(inputs, .005, study.CALIBRATION_SEED, 0, 1000)
        independent = study.draw(inputs, .005, study.VALIDATION_SEED, 0, 1000)
        for a,b in zip(first,repeated):
            np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(first[1].sum(axis=1) <= first[2]))
        np.testing.assert_array_equal(first[2], [100,120,140])
        self.assertFalse(np.array_equal(first[0], independent[0]))
        self.assertFalse(np.array_equal(first[1], independent[1]))
        fixed_a = study.draw(inputs,.005,study.INDEPENDENT_BANK_SEED,0,1000,fluctuate_mc=False)
        fixed_b = study.draw(inputs,.005,study.INDEPENDENT_BANK_SEED,1,1000,fluctuate_mc=False)
        np.testing.assert_array_equal(fixed_a[1],fixed_b[1])
        np.testing.assert_array_equal(fixed_a[1],inputs['banks']['coherent']['raw_templates'])
        self.assertFalse(np.array_equal(fixed_a[0],fixed_b[0]))

    def test_calibration_freezes_only_its_own_grid(self):
        from analysis.bayes import finite_template_162_study as study
        points = {f'{int(d*1e6):05d}': {method: dict(q0=np.linspace(0,3,1000),
                   q_upper=np.linspace(0,4+int(d*1e6),1000)) for method in study.METHODS}
                  for d in study.CALIBRATION_DIRECT}
        state = dict(configuration=dict(repetitions=1000,seed=study.CALIBRATION_SEED),records=points)
        frozen = study.freeze(state)
        for method in study.METHODS:
            self.assertEqual(frozen['cutoffs'][method]['q_upper'],max(frozen['local_upper'][method].values()))
            self.assertGreater(frozen['cutoffs'][method]['q0'],0)
        state['configuration']['seed'] = study.VALIDATION_SEED
        with self.assertRaises(ValueError):
            study.freeze(state)


if __name__ == '__main__':
    unittest.main()
