"""Data-set-9 literature response benchmark, with explicit missing-input controls.

The CLI alone reads ROOT and calculates. Plots load notebook metadata. Weighted
importance samples remain numerical integration samples, never integer finite-MC
auxiliary observations. No published data fit or new branch limit is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import uproot

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.bayes import rmatrix_162_study as parent
from analysis.efficiency import laursen_162_response as response

ROOT=Path(__file__).resolve().parents[3]
NOTEBOOK=ROOT/'jupyter/codex_0002_162keV_探测器响应研究.ipynb'
KEY='codex_laursen_162_response'
VERSION='2026-10-26-v1-set9-conditional-response'
MODELS=('alpha0','phase_space',*parent.MODELS)
LABELS=('alpha0','Uniform phase space','Coherent BW','R, no external phase','R + external phases','R + phases + Coulomb')
COLORS=('#767676','#999999','#222222','#d98b16','#2475b4','#b13d3d')
PLAN=dict(data_set=9,energy_scope='162-keV resonance; stored reaction kinematics, not 169-keV incident reactions',
          beam_width_mm=2.,beam_profile='uniform square, assumed',depth='uniform boron depth; midpoint correction, assumed',
          response_seed=1622026102621,block_seed=1622026102622,bootstrap_seed=1622026102623,
          blocks=80,bootstrap_repetitions=400,reference_events=200000,
          ground_fraction_example=.051,scenarios=[asdict(r) for r in response.scenarios()],
          scope='Conditional sensitivity study; missing electronics/material distribution/author code not inferred from figures',
          reconstruction='Known parent momentum, pixel-centre direction and mean CSDA correction; exclusive multiplicities 2/3')
ENERGY_EDGES=np.linspace(0,6.5,131)
DALITZ_EDGES=np.linspace(-.36,.36,73)


def digest(value):
    return hashlib.sha256(json.dumps(parent.plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def source_reference(metadata,reference,channel,amplitude):
    library_key='codex_template_convergence_libraries' if reference=='old' else 'codex_independent_replication_libraries'
    library=metadata[library_key];name='alpha0' if channel==0 else 'phase_space'
    row=library['runs'][name];path=ROOT/row['root']
    if sha256(path)!=row['root_sha256']:raise ValueError('ROOT fingerprint changed')
    fields=['event']+[f'{prefix}{i}' for i in (1,2,3) for prefix in ('e_alpha','theta_lab_alpha','phi_lab_alpha','e_3alpha_cm_alpha')]
    with uproot.open(path,handler=uproot.source.file.MemmapSource) as stream:
        arrays=stream['reaction'].arrays(fields,library='np')
    ids=arrays['event']
    partition=parent.convergence.partitions if reference=='old' else parent.pooled.prior.partition
    template,heldout=partition(ids,channel)
    if len(heldout)!=PLAN['reference_events'] or np.intersect1d(template,heldout).size:
        raise ValueError('Unexpected preselection partition')
    selected=np.isin(ids,heldout)
    arrays={k:v[selected] for k,v in arrays.items()}
    kinetic=np.column_stack([arrays[f'e_alpha{i}'] for i in (1,2,3)])
    theta=np.column_stack([arrays[f'theta_lab_alpha{i}'] for i in (1,2,3)])
    phi=np.column_stack([arrays[f'phi_lab_alpha{i}'] for i in (1,2,3)])
    directions=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
    event=parent.reaction_vectors(arrays,response.MASS)
    stored=np.column_stack([arrays[f'e_3alpha_cm_alpha{i}'] for i in (1,2,3)])
    cm_error=float(np.max(abs(event[...,3]-stored)))
    if cm_error>1e-8:raise ArithmeticError('CM reconstruction no longer agrees with ROOT')
    if channel==0:
        weights=np.ones((len(event),1));model_indices=[0]
    else:
        weight_chunks=[]
        for first in range(0,len(event),25000):weight_chunks.append(amplitude.weights(event[first:first+25000]))
        physical=np.concatenate(weight_chunks)
        previous=metadata[parent.KEY]['inputs']['weighted'][reference]['summary']
        if parent.digest(heldout)!=previous['reference_ids_sha256']:
            raise ValueError('Held-out reference identity changed')
        effective=physical.sum(axis=0)**2/(physical*physical).sum(axis=0)
        if not np.allclose(effective,previous['effective_events'],rtol=1e-9,atol=1e-5):
            raise ArithmeticError('Historical importance weights no longer reproduce')
        weights=np.column_stack([np.ones(len(event)),physical]);model_indices=list(range(1,6))
    rng=np.random.default_rng([PLAN['response_seed'],int(reference=='new'),channel])
    xy=rng.uniform(-1.,1.,size=(len(event),2))*PLAN['beam_width_mm']/2
    origins=np.column_stack([xy,-response.FOIL_NORMAL[0]/response.FOIL_NORMAL[2]*xy[:,0]])
    depth=rng.random(len(event));noise=rng.standard_normal((len(event),3))
    order=np.random.default_rng([PLAN['block_seed'],int(reference=='new'),channel]).permutation(len(event))
    blocks=np.empty(len(event),int);blocks[order]=np.arange(len(event))%PLAN['blocks']
    provenance=dict(root=row['root'],root_sha256=row['root_sha256'],global_seed=row['validation']['seed'],
        incident_count=row['validation']['incident_count'],generated=len(heldout),heldout_ids_sha256=parent.digest(heldout),
        selected_ids_sha256=parent.digest(arrays['event']),CM_error_MeV=cm_error,
        q_range_MeV=[float(event[...,3].sum(axis=1).min()),float(event[...,3].sum(axis=1).max())],
        macro_text=row['macro_text'],macro_sha256=row['macro_sha256'],
        generator_source_sha256=library['configuration']['source_sha256'],
        generator_binary_sha256=library['configuration']['binary_sha256'],
        no_virtual_sphere_preselection=True,disjoint_from_auxiliary_MC=True)
    return dict(kinetic=kinetic,directions=directions,cm=event[...,3],weights=weights,
                model_indices=model_indices,origins=origins,depth=depth,noise=noise,blocks=blocks,provenance=provenance)


def integrate(reference,tables):
    weights=reference['weights'];nmodels=weights.shape[1];blocks=reference['blocks']
    sums=weights.sum(axis=0)
    denominator=np.stack([np.bincount(blocks,weights=weights[:,m],minlength=PLAN['blocks']) for m in range(nmodels)],axis=1)
    result=dict(denominator=denominator,model_indices=reference['model_indices'],provenance=reference['provenance'],
                effective_generated=sums*sums/np.sum(weights*weights,axis=0),cases={},generated_spectrum=[],generated_dalitz=[])
    for m in range(nmodels):
        result['generated_spectrum'].append(np.histogram(reference['cm'].ravel(),ENERGY_EDGES,weights=np.repeat(weights[:,m],3))[0])
        result['generated_dalitz'].append(response.symmetric_dalitz(reference['cm'],weights[:,m],DALITZ_EDGES))
    for scenario in response.scenarios():
        records,diagnostics=response.fold(reference['kinetic'],reference['directions'],reference['origins'],
            reference['depth'],reference['noise'],scenario,tables)
        accepted=np.zeros((len(weights),2));before=np.zeros_like(accepted);ambiguous=np.zeros_like(accepted)
        histograms={}
        save_shapes=scenario.name in ('geometry','nominal_assumptions')
        if save_shapes:
            histograms=dict(spectrum=np.zeros((nmodels,2,len(ENERGY_EDGES)-1)),
                            dalitz=np.zeros((nmodels,2,len(DALITZ_EDGES)-1,len(DALITZ_EDGES)-1)))
        for record in records:
            k=record['multiplicity']-2;p=record['probability']
            accepted[:,k]+=p;before[:,k]+=record['preselection']
            ambiguous[:,k]+=p*record['ambiguous']
            if save_shapes:
                # Paper Fig. 4: individual measured particles in DSSSD 1.
                observed=(record['detector']==0)&record['pattern'][None,:]
                for m in range(nmodels):
                    wp=weights[:,m]*p
                    active=wp>0
                    histograms['dalitz'][m,k]+=response.symmetric_dalitz(record['cm_kinetic'][active],wp[active],DALITZ_EDGES)
                    mask=observed & active[:,None]
                    histograms['spectrum'][m,k]+=np.histogram(record['cm_kinetic'][mask],ENERGY_EDGES,
                        weights=np.broadcast_to(wp[:,None],mask.shape)[mask])[0]
        if np.any(accepted.sum(axis=1)>1+1e-12) or np.any(accepted>before+1e-12):
            raise ArithmeticError('Exclusive event probabilities or selection flow broken')
        numerator=np.zeros((PLAN['blocks'],nmodels,2))
        for m in range(nmodels):
            for k in range(2):numerator[:,m,k]=np.bincount(blocks,weights=weights[:,m]*accepted[:,k],minlength=PLAN['blocks'])
        rate=numerator.sum(axis=0)/sums[:,None]
        errors=np.sqrt(np.sum((weights[:,:,None]*(accepted[:,None,:]-rate[None,:,:]))**2,axis=0))/sums[:,None]
        squared=np.sum((weights[:,:,None]*accepted[:,None,:])**2,axis=0)
        effective=np.divide(numerator.sum(axis=0)**2,squared,out=np.zeros_like(squared),where=squared>0)
        entry=dict(efficiency=rate,importance_standard_error=errors,effective_accepted=effective,
            block_numerator=numerator,preselection=weights.T@before/sums[:,None],
            ambiguous_accepted=weights.T@ambiguous/sums[:,None],invalid_corrections=diagnostics['invalid_corrections'])
        if save_shapes:
            entry.update(histograms)
            entry['dalitz_outside_display_weight']=numerator.sum(axis=0)-histograms['dalitz'].sum(axis=(2,3))
            if np.any(entry['dalitz_outside_display_weight'] < -1e-6):raise ArithmeticError('Dalitz double-counted events')
        result['cases'][scenario.name]=entry
        print(reference['provenance']['root'].split('/')[-1],scenario.name,'done',flush=True)
    return result


def combine(groups,reference):
    cases=[r.name for r in response.scenarios()]
    denominator=np.concatenate([g['denominator'] for g in groups],axis=1)
    rng=np.random.default_rng([PLAN['bootstrap_seed'],int(reference=='new')])
    # Ground-state and phase-space libraries are independent; paired model
    # alternatives share the same phase-space resampling in each replicate.
    draws=[rng.multinomial(PLAN['blocks'],np.full(PLAN['blocks'],1/PLAN['blocks']),size=PLAN['bootstrap_repetitions']) for g in groups]
    answer={}
    for case in cases:
        point=np.concatenate([g['cases'][case]['efficiency'] for g in groups],axis=0)
        boot=np.concatenate([np.einsum('rb,bmk->rmk',draw,g['cases'][case]['block_numerator'])/
                             (draw@g['denominator'])[:,:,None] for g,draw in zip(groups,draws)],axis=1)
        fraction=PLAN['ground_fraction_example']
        seen=fraction*point[0]/(fraction*point[0]+(1-fraction)*point[-1])
        seen_boot=fraction*boot[:,0]/(fraction*boot[:,0]+(1-fraction)*boot[:,-1])
        answer[case]=dict(efficiency=point,bootstrap_interval=np.quantile(boot,[.025,.975],axis=0),
            paired_Coulomb_minus_BW=point[-1]-point[2],
            paired_Coulomb_minus_BW_interval=np.quantile(boot[:,-1]-boot[:,2],[.025,.975],axis=0),
            observed_alpha0_fraction_example=seen,
            observed_alpha0_interval=np.quantile(seen_boot,[.025,.975],axis=0))
    return answer


def numerical_checks(tables):
    dense=response.RangeTables(tables);coarse=response.RangeTables(tables,stride=2)
    energies=np.geomspace(.010,6.5,700)
    worst=0.
    for material in ('G4_B','G4_C','G4_Si'):
        for column in (5e-6,15e-6,50e-6,200e-6,1000e-6):
            worst=max(worst,float(np.max(abs(dense.remove(energies,column,material)-coarse.remove(energies,column,material)))))
        remaining=dense.remove(energies,5e-6,material)
        mask=remaining>0
        if np.max(abs(dense.restore(remaining[mask],5e-6,material)-energies[mask]))>1e-6:
            raise ArithmeticError('CSDA forward/inverse closure failed')
    if worst>.001:raise ArithmeticError('Stopping-grid effect exceeds 1 keV')
    silicon_range_um=float(dense.materials['G4_Si']['forward'](5.9)/tables['materials']['G4_Si']['density_g_cm3']*1e4)
    if silicon_range_um>=60:raise ArithmeticError('Active silicon does not stop 5.9-MeV alpha')
    return dict(grid_halving_max_energy_difference_MeV=worst,range_Si_at_5p9MeV_um=silicon_range_um,
        solid_angle_fraction=[response.rectangle_solid_angle(c,n,h)/(4*np.pi) for c,n,h in zip(response.CENTRES,response.NORMALS,response.HORIZONTAL)])


def run(stopping_path,output,parent_notebook=parent.NOTEBOOK):
    output=Path(output)
    if output.exists():raise ValueError('Preserve the completed result; use its cache')
    metadata=json.loads(Path(parent_notebook).read_text())['metadata']
    previous=metadata[parent.KEY]
    if parent.digest({k:v for k,v in previous.items() if k!='checkpoint_sha256'})!=previous['checkpoint_sha256']:
        raise ValueError('Parent study checksum changed')
    table=json.loads(Path(stopping_path).read_text())
    checks=numerical_checks(table)
    sources=[Path(__file__),Path(response.__file__),ROOT/'g4/analysis/bayes/rmatrix_162_model.py',
             ROOT/'g4/validation/export_162_stopping_power.cc']
    configuration=dict(version=VERSION,plan=PLAN,parent_checkpoint=previous['checkpoint_sha256'],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        interpreter=sys.executable,packages={k:importlib.metadata.version(k) for k in ('numpy','scipy','matplotlib','uproot')},
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},stopping_sha256=sha256(stopping_path))
    protocol=output.with_name('response_protocol.json')
    if protocol.exists() and json.loads(protocol.read_text())!=parent.plain(configuration):raise ValueError('Protocol changed')
    protocol.write_text(json.dumps(configuration,ensure_ascii=False,indent=2))
    started=datetime.now().astimezone().isoformat();begin=time.monotonic()
    print('Building declared R-matrix amplitude tables',flush=True)
    amplitude=parent.AmplitudeModel();ranges=response.RangeTables(table)
    references={};summary={}
    for reference in ('old','new'):
        groups=[]
        for channel in (0,2):
            print('Loading preselection generation data',reference,channel,flush=True)
            loaded=source_reference(metadata,reference,channel,amplitude)
            groups.append(integrate(loaded,ranges))
        references[reference]=groups;summary[reference]=combine(groups,reference)
    result=parent.plain(dict(configuration=configuration,stopping=table,checks=checks,references=references,
        summary=summary,energy_edges=ENERGY_EDGES,dalitz_edges=DALITZ_EDGES,
        source_snapshots={str(p.relative_to(ROOT)):p.read_text() for p in sources},
        started_at=started,completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-begin))
    result['checksum']=digest(result)
    output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
    print('Literature response study complete',result['completed_at'],flush=True)
    return result


def load_cached(notebook=NOTEBOOK):
    state=json.loads(Path(notebook).read_text())['metadata'][KEY]
    if digest({k:v for k,v in state.items() if k!='checksum'})!=state['checksum']:
        raise ValueError('Literature-response cache changed')
    return state


def plot_geometry(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for d,(centre,horizontal,color) in enumerate(zip(response.CENTRES,response.HORIZONTAL,('#2475b4','#b13d3d'))):
        edge=centre+np.array([-25,25])[:,None]*horizontal
        axes[0].plot(edge[:,2],edge[:,0],lw=5,color=color,label=f'DSSSD {d+1}, {state["checks"]["solid_angle_fraction"][d]*100:.2f}% of 4pi')
    tangent=np.array([response.FOIL_NORMAL[2],0.,-response.FOIL_NORMAL[0]])
    foil=np.array([-8,8])[:,None]*tangent
    axes[0].plot(foil[:,2],foil[:,0],color='black',label='Foil plane; beta=126 deg')
    axes[0].annotate('Beam +z',xy=(0,0),xytext=(-35,0),arrowprops=dict(arrowstyle='->'))
    axes[0].scatter([0],[0],color='black',s=12)
    axes[0].set(xlabel='z (mm)',ylabel='x (mm)',aspect='equal',title='Figure 9(a): data-set-9 geometry')
    axes[0].legend(fontsize=8,loc='upper right')
    energy=np.linspace(0,.8,300)
    for d,color in enumerate(('#2475b4','#b13d3d')):
        for label,mid,style in [('low',(.1,.2),'--'),('nominal',(.2,.35),'-'),('high',(.3,.5),':')]:
            axes[1].plot(energy*1000,response.trigger_efficiency(energy,mid[d],.3),style,color=color,label=f'DSSSD {d+1}: {label}')
    axes[1].set(xlabel='Visible energy (keV)',ylabel='Assumed single-particle trigger efficiency',
                title='Figure 9(b): sensitivity assumptions, not measured curves')
    axes[1].legend(fontsize=8)
    return fig


def plot_efficiency(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for k,ax in enumerate(axes):
        for case,offset,color in [('geometry',-.20,'#999999'),('nominal_assumptions',.20,'#2475b4')]:
            for reference,shift,marker in [('old',-.035,'o'),('new',.035,'s')]:
                entry=state['summary'][reference][case]
                y=100*np.array(entry['efficiency'])[:,k]
                ci=100*np.array(entry['bootstrap_interval'])[:,:,k]
                ax.errorbar(np.arange(6)+offset+shift,y,yerr=np.maximum(0,np.vstack([y-ci[0],ci[1]-y])),
                            fmt=marker,color=color,capsize=2,label=f'{case.replace("_assumptions","")}, {reference}')
        ax.set_xticks(np.arange(6),['alpha0','Phase space','BW','R, no phase','R + phases','R + Coulomb'],rotation=25)
        ax.set(ylabel='Accepted events / generated events (%)',title=f'Figure 10({"ab"[k]}): exclusive multiplicity {k+2}')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed literature geometry; vertical bars show reference-MC uncertainty only')
    return fig


def plot_dalitz(state,reference='old'):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig,axes=plt.subplots(2,4,figsize=(12.5,6.8),layout='constrained',sharex=True,sharey=True)
    group=state['references'][reference][1];names=['Uniform phase space','Coherent BW','R + phases','R + phases + Coulomb'];indices=[0,1,3,4]
    hist=np.asarray(group['cases']['nominal_assumptions']['dalitz'])
    for k in range(2):
        for col,(index,name) in enumerate(zip(indices,names)):
            values=hist[index,k]/hist[index,k].sum()
            im=axes[k,col].imshow(values.T,origin='lower',extent=[-.36,.36,-.36,.36],
                                 norm=LogNorm(vmin=1e-5,vmax=.02),cmap='viridis')
            theta=np.linspace(0,2*np.pi,300)
            axes[k,col].plot(np.cos(theta)/3,np.sin(theta)/3,'w:',lw=.6)
            axes[k,col].set_title(f'Figure 11({"abcdefgh"[4*k+col]}): m={k+2}\n{name}',fontsize=9)
            if col==0:axes[k,col].set_ylabel('Dalitz psi')
            if k==1:axes[k,col].set_xlabel('Dalitz chi')
    fig.colorbar(im,ax=axes,label='Conditional probability per bin (log scale)',shrink=.8)
    fig.suptitle('Reconstructed response control; old reference; each panel normalized separately')
    return fig


def plot_spectrum(state,reference='old'):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    edges=np.asarray(state['energy_edges']);width=np.diff(edges)
    group=state['references'][reference][1]
    for k,ax in enumerate(axes):
        for index,label,color in zip(range(5),LABELS[1:],COLORS[1:]):
            counts=np.asarray(group['cases']['nominal_assumptions']['spectrum'])[index,k]
            ax.stairs(counts/(counts.sum()*width),edges,label=label,color=color)
        ax.set(xlabel='Reconstructed CM alpha energy (MeV)',ylabel='Conditional measured-particle density (1/MeV)',
               xlim=(0,6),title=f'Figure 12({"ab"[k]}): DSSSD 1; exclusive m={k+2}')
    axes[0].legend(fontsize=8)
    fig.suptitle('alpha1 model controls; detected particles only; no digitized experimental data')
    return fig


def plot_sensitivity(state):
    import matplotlib.pyplot as plt
    cases=['nominal_assumptions','low_threshold','high_threshold','carbon_5','carbon_15','no_adsorption',
           'backing_reversed','global_trigger','strip_ambiguity_veto','detectors_out_2mm','detectors_in_2mm']
    labels=['Nominal assumptions','Lower thresholds','Higher thresholds','Carbon 5 ug/cm2','Carbon 15 ug/cm2',
            'No adsorption','Backing reversed','Global OR trigger','Reject shared strips','Outward 2 mm','Inward 2 mm']
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained',sharey=True)
    for k,ax in enumerate(axes):
        for reference,shift,marker,color in [('old',-.08,'o','#2475b4'),('new',.08,'s','#b13d3d')]:
            values=[100*state['summary'][reference][case]['efficiency'][-1][k] for case in cases]
            ax.plot(values,np.arange(len(cases))+shift,marker,ls='none',color=color,label=reference)
        ax.set_yticks(np.arange(len(cases)),labels)
        ax.set(xlabel='R + phases + Coulomb acceptance (%)',title=f'Figure 13({"ab"[k]}): exclusive m={k+2}')
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Selected response scenarios; their spread is not an experimental confidence interval')
    return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stopping',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-notebook',type=Path,default=parent.NOTEBOOK)
    args=parser.parse_args()
    run(args.stopping,args.output,args.parent_notebook)
