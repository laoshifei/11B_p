"""Actual-bank numerical and binning controls, saved in the existing notebook."""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.bayes import finite_template_162_study as study
from analysis.bayes.finite_template_likelihood import JointMultinomial, accepted_weights


def validate():
    inputs, selected = study.prepare_inputs()
    numerical = []
    for direct, index in ((0.,75),(.001,177),(.005,0),(.02,75)):
        data, mc, generated = study.draw(inputs, direct, study.CALIBRATION_SEED, index, 100000)
        fixed = study.fixed_model(data,mc,generated)
        start = accepted_weights(fixed.mode,fixed.efficiencies)
        a = JointMultinomial(data,mc,generated,start=start)
        b = JointMultinomial(data,mc,generated,start=start,tolerance=1e-5)
        q = a.statistic(direct)
        q_tight = b.statistic(direct)
        difference = max(float(np.max(abs(a.mode-b.mode))), abs(a.q0-b.q0), abs(q-q_tight))
        if difference > .002:
            raise RuntimeError(f"Numerical refinement failed: {direct}, {index}, {difference}")
        upper = a.upper_limit(3.)
        if abs(a.statistic(upper)-3) > .002 or ((direct<=upper) != (a.statistic(direct,upper=True)<=3)):
            raise RuntimeError("Upper inversion and acceptance test disagree")
        numerical.append(dict(direct=direct, trial=index, mode_difference=np.max(abs(a.mode-b.mode)),
            q0_difference=abs(a.q0-b.q0), profile_difference=abs(q-q_tight),
            inversion_error=abs(a.statistic(upper)-3), generator_upper=upper))
        print('Numerical refinement PASS:',direct,index,flush=True)
    sensitivity = []
    for bins,half in ((30,False),(60,False),(90,False),(60,True)):
        banks,_ = study.mismatch.make_banks(selected,bins=bins,half=half)
        support = study.mismatch.geometric_support(bins)
        hist = np.stack([study.mismatch.histogram_xy(selected[(name,'data')]['xy'],bins)[support]
                         for name in ('alpha0','coherent','phase_space')])
        generated = np.array([selected[(name,'data')]['generated'] for name in ('alpha0','coherent','phase_space')])
        truth = np.column_stack([hist,generated-hist.sum(axis=1)])/generated[:,None]
        data = 100000*(study.fractions(0.)@truth)[:-1]
        bank = banks['coherent']
        result = study.evaluate(data,bank['raw_templates'],bank['generated'],0.)
        sensitivity.append(dict(bins=bins,half=half,template_generated=bank['generated'],result=result))
        print('Binning/statistics control:',bins,half,study.calibration.plain(result),flush=True)
    bank = inputs['banks']['coherent']
    data = 100000*(study.fractions(0.)@np.asarray(inputs['independent_truth']))[:-1]
    starts=[]
    for initial in ([.05,.949,.001],[.10,.80,.10],[.03,.96,.01]):
        model=JointMultinomial(data,bank['raw_templates'],bank['generated'],start=initial)
        starts.append(dict(initial=initial,mode=model.mode,loss=model.loss))
    if np.ptp([row['loss'] for row in starts]) > .002:
        raise RuntimeError('Real-bank multistart likelihoods disagree')
    positive_starts=[]
    positive_data=100000*(study.fractions(.005)@np.asarray(inputs['independent_truth']))[:-1]
    for initial in ([.05,.949,.001],[.10,.80,.10],[.03,.96,.01]):
        model=JointMultinomial(positive_data,bank['raw_templates'],bank['generated'],start=initial)
        positive_starts.append(dict(initial=initial,mode=model.mode,loss=model.loss))
    if (np.ptp([row['loss'] for row in positive_starts]) > .002
            or np.ptp([row['mode'][2] for row in positive_starts]) > 2e-5):
        raise RuntimeError('Positive-injection multistart likelihoods disagree')
    result=dict(completed_at=datetime.now().astimezone().isoformat(),numerical=numerical,
        sensitivity=sensitivity,multistart=starts,positive_multistart=positive_starts,all_passed=True,
        source_sha256={str(path.relative_to(study.ROOT)):study.sha256(path) for path in
            (Path(__file__),study.ROOT/'g4/analysis/bayes/finite_template_likelihood.py',
             study.ROOT/'g4/analysis/bayes/finite_template_162_study.py')},
        note='Numerical/binning controls only; cutoffs are not transferred to other binning or MC budgets')
    study.mismatch.save_key(study.NOTEBOOK,'codex_finite_template_controls',result)
    return result


if __name__=='__main__':
    validate()
