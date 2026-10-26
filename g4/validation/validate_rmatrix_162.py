"""Numerical controls for the opt-in 162-keV analysis amplitude.

These verify implementation, not the adequacy of a nuclear decay model.
The CLI emits results to stdout; the study embeds them in notebook metadata.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import spherical_jn, spherical_yn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.bayes.rmatrix_162_model import AmplitudeModel, Parameters, angular_amplitude
from validate_162_bose import (ReferenceRadial, angular_amplitude as scipy_angular,
                               independent_proposal, reference_weights)


def validate(model=None, seed=1622026102341):
    model = model or AmplitudeModel()
    rng = np.random.default_rng(seed)
    checks = []
    def check(name, value, limit):
        checks.append(dict(name=name, value=float(value), limit=float(limit),
                           passed=bool(np.isfinite(value) and value <= limit)))
    event, phase_space = independent_proposal(2048, rng)
    weight, diagonal, _ = model.weights(event, diagnostics=True)
    check('mass shell MeV^2', np.max(abs((event[...,3]+model.p.mass)**2
        -np.sum(event[...,:3]**2,axis=-1)-model.p.mass**2)), 1e-7)
    check('CM momentum MeV/c', np.max(np.linalg.norm(event[...,:3].sum(axis=1),axis=-1)), 1e-9)
    check('CM energy MeV', np.max(abs(event[...,3].sum(axis=1)-8.8314)), 1e-9)
    vectors = rng.normal(size=(2,128,3))
    check('Cartesian harmonics vs independent SciPy', np.max(abs(angular_amplitude(*vectors)-scipy_angular(*vectors))), 1e-13)
    baseline = reference_weights(event, ReferenceRadial())[:,2]
    check('Existing BW Python reference', np.max(abs(weight[:,0]-baseline)/np.maximum(baseline,1e-25)), 2e-7)
    maximum = 0.
    for perm in itertools.permutations(range(3)):
        alternate = model.weights(event[:128,perm,:])
        maximum = max(maximum, np.max(abs(alternate-weight[:128])/np.maximum(weight[:128],1e-25)))
    check('All six Bose permutations', maximum, 1e-8)
    rotation,_ = np.linalg.qr(rng.normal(size=(3,3)))
    rotation[:,0] *= np.linalg.det(rotation)
    transformed = event[:128].copy()
    transformed[...,:3] = transformed[...,:3]@rotation
    check('Common rotation invariance', np.max(abs(model.weights(transformed)-weight[:128])/np.maximum(weight[:128],1e-25)), 2e-8)
    transformed[...,:3] = -event[:128,...,:3]
    check('Positive parity', np.max(abs(model.weights(transformed)-weight[:128])/np.maximum(weight[:128],1e-25)), 1e-10)
    check('Cauchy bound', max(0., np.max(weight/(3*diagonal))-1), 1e-12)
    neutral = []
    for rho in np.geomspace(.01,30,64):
        f,fp,g,gp=model.coulomb.wave(0,rho)
        expected=np.array([rho*spherical_jn(2,rho), spherical_jn(2,rho)+rho*spherical_jn(2,rho,True),
            -rho*spherical_yn(2,rho), -spherical_yn(2,rho)-rho*spherical_yn(2,rho,True)])
        neutral.append(np.max(abs(np.array([f,fp,g,gp])-expected)/np.maximum(1,abs(expected))))
    check('Neutral-channel closed-form Coulomb limit', max(neutral), 1e-10)
    interpolation=np.zeros(3)
    for channel in (model.pair,model.primary,model.pair_far,model.primary_far):
        energies=np.exp(rng.uniform(np.log(.001),np.log(10),200))
        exact=channel.exact_many(energies)[:,:3]
        approximate=channel.evaluate(energies)
        interpolation[0]=max(interpolation[0], np.max(abs(exact[:,0]-approximate[:,0])))
        interpolation[1]=max(interpolation[1], np.max(abs(exact[:,1]-approximate[:,1])))
        interpolation[2]=max(interpolation[2], np.max(abs(np.exp(1j*exact[:,2])-np.exp(1j*approximate[:,2]))))
    for name,error in zip(('log penetrability','shift','unit external phase'),interpolation):
        check('Interpolation vs outgoing Coulomb ODE: '+name,error,3e-6)
    check('R-matrix shift vanishes at pole', abs(model.pair.evaluate(model.p.pole)[1]-model.shift_at_pole), 1e-14)
    # Explicit scalar equations, direct GSL calls and independent harmonics.
    from validate_162_bose import boost
    scalar=[]
    for row in event[:32]:
        p=row[:,:3]; e=row[:,3]+model.p.mass
        epairs=[]; angles=[]
        for i,j,k in ((0,1,2),(1,2,0),(2,0,1)):
            pp=p[j]+p[k]; en=e[j]+e[k]
            epairs.append(np.sqrt(en*en-pp@pp)-2*model.p.mass)
            angles.append(scipy_angular(p[i:i+1],boost(e[j:j+1],p[j:j+1],-pp[None,:]/en)[1])[0])
        paths=[]
        for i in range(3):
            ej=epairs[i]; ti=row[i,3]; ef=1.5*ti
            lp,shift,phase,_=model.pair.exact(ej)
            lp1,_,phase1,_=model.primary.exact(ef)
            gamma=2*model.p.reduced_width*np.exp(lp)
            den=model.p.pole-ej-model.p.reduced_width*(shift-model.shift_at_pole)-.5j*gamma
            a=np.sqrt((2*np.exp(lp1)/np.sqrt(ti))*(gamma/np.sqrt(ej)))/den
            b=a*np.exp(1j*(phase+phase1))
            j,k=(i+1)%3,(i+2)%3
            correction=np.sqrt(ti/(epairs[j]*epairs[k]))*np.exp(model.pair_far.exact(epairs[j])[0]+model.pair_far.exact(epairs[k])[0]-model.primary_far.exact(ef)[0])
            paths.append(np.array([a,b,b*np.sqrt(correction)])[:,None]*angles[i])
        scalar.append(np.mean(abs(np.sum(paths,axis=0))**2,axis=-1))
    scalar=np.asarray(scalar)
    check('Scalar equations (3)-(5) vs vector weights', np.max(abs(scalar-weight[:32,1:])/np.maximum(scalar,1e-25)), 1e-5)
    finer=AmplitudeModel(Parameters(grid_size=12288))
    alternate=finer.weights(event)
    normal=weight*phase_space[:,None]; normal/=normal.sum(axis=0)
    alt=alternate*phase_space[:,None]; alt/=alt.sum(axis=0)
    check('Doubled Coulomb grid normalized L1',np.max(np.sum(abs(alt-normal),axis=0)),1e-6)
    # A genuinely separate high-precision input: existing mpmath-generated
    # alpha-alpha table, with its stated 2 amu rather than the physical mass.
    from analysis.bayes.rmatrix_162_model import Channel
    original=Channel(model.coulomb,2*931.49410242,4,4.5,Parameters(grid_size=128))
    energies=model.legacy['AlphaAlpha_energy_keV']/1000
    keep=np.flatnonzero((energies>.02)&(energies<10))[::4]
    exact=original.exact_many(energies[keep])[:,0]
    legacy_comparison=dict(max_logP_difference=float(np.max(abs(exact-model.legacy['AlphaAlpha_L2_logP'][keep]))),
        scope='Existing mpmath table: stated 2 amu, 4.5 fm, but complete generation constants unavailable; diagnostic, not a bitwise numerical reference')
    neutral_channel=Channel(model.coulomb,model.p.mass/2,0,4.5,Parameters(grid_size=128))
    energy=np.geomspace(.01,10,64)
    rho=np.sqrt(model.p.mass*energy)*4.5/197.3269804
    f=rho*spherical_jn(2,rho);g=-rho*spherical_yn(2,rho)
    fp=spherical_jn(2,rho)+rho*spherical_jn(2,rho,True)
    gp=-spherical_yn(2,rho)-rho*spherical_yn(2,rho,True)
    values=neutral_channel.exact_many(energy)
    expected=np.column_stack([np.log(rho/(f*f+g*g)),rho*(f*fp+g*gp)/(f*f+g*g)])
    check('Full outgoing ODE vs neutral analytic P/S',np.max(abs(values[:,:2]-expected)),1e-8)
    tighter=model.pair.exact_many(np.array([.001,.1,1.279925288,3.129,8.8]),rtol=2e-12)
    normal=model.pair.exact_many(np.array([.001,.1,1.279925288,3.129,8.8]))
    check('Tenfold tighter Coulomb ODE tolerance',np.max(abs(normal[:,:2]-tighter[:,:2])),2e-7)
    distant=model.pair.exact_many(np.array([.001,.1,1.279925288,3.129,8.8]),boundary_scale=2)
    check('Doubled oscillatory boundary P/S',np.max(abs(normal[:,:2]-distant[:,:2])),2e-7)
    check('Doubled oscillatory boundary complex phase',np.max(abs(np.exp(1j*normal[:,2])-np.exp(1j*distant[:,2]))),2e-7)
    return dict(seed=seed,proposal_events=len(event),checks=checks,
                all_passed=all(c['passed'] for c in checks), model=model.provenance(),legacy_table_diagnostic=legacy_comparison)


if __name__=='__main__':
    result=validate()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['all_passed']:
        raise SystemExit(1)
