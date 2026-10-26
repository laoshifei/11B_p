"""162-keV R-matrix/Coulomb stress test of the frozen BW inference.

Only an explicit CLI reads ROOT and runs fits. Use disjoint held-out uniform
phase-space reactions, common response streams and importance weights.
The actual unweighted pooled BW MC counts remain the fit auxiliary sample.
Weighted reference uncertainty is assessed by paired event-block bootstrap;
weighted counts never enter the finite-MC multinomial auxiliary likelihood.
New physics laws do not inherit the BW procedure's coverage guarantee.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.stats import norm
import uproot

if __package__ in (None, ''):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.bayes import pooled_r_calibration_162_study as pooled
from analysis.bayes.rmatrix_162_model import AmplitudeModel, MODELS, LABELS, boost

ROOT,NOTEBOOK=pooled.ROOT,pooled.NOTEBOOK
binning,banks,finite,convergence=pooled.binning,pooled.banks,pooled.finite,pooled.convergence
mismatch,scan=convergence.mismatch,convergence.mismatch.scan
digest,plain=pooled.digest,pooled.plain
KEY='codex_rmatrix_162_study'
VERSION='2026-10-23-v1-weighted-reference-frozen-bw'
PLAN=dict(bins=30,r=.05,direct=[0.,.005],data_budget=100000,repetitions=500,
          bootstrap_repetitions=100,blocks=80,response='Historical replay: 2 deg, 100 keV FWHM, 100 keV/alpha',
          data_seed=1622026102342,block_seed=1622026102343,bootstrap_seed=1622026102344,
          primary='Frozen pooled BW joint likelihood and its 30-bin cutoffs; no retuning after model change')


def reaction_vectors(arrays,mass):
    kinetic=np.column_stack([arrays[f'e_alpha{i}'] for i in (1,2,3)])
    theta=np.column_stack([arrays[f'theta_lab_alpha{i}'] for i in (1,2,3)])
    phi=np.column_stack([arrays[f'phi_lab_alpha{i}'] for i in (1,2,3)])
    direction=np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],axis=-1)
    momentum=np.sqrt(kinetic*(kinetic+2*mass))[...,None]*direction
    energy=kinetic+mass
    beta=-momentum.sum(axis=1)/energy.sum(axis=1)[:,None]
    e,p=boost(energy,momentum,beta[:,None,:])
    return np.concatenate([p,(e-mass)[...,None]],axis=-1)


def category_indices(xy,bins=30):
    support=mismatch.geometric_support(bins)
    positions=np.full(support.shape,-1,np.int64)
    positions[support]=np.arange(support.sum())
    xy=np.asarray(xy)
    coord=np.floor((xy-mismatch.LIMITS[0])/(mismatch.LIMITS[1]-mismatch.LIMITS[0])*bins).astype(int)
    if np.any(coord<0) or np.any(coord>=bins):
        raise ValueError('Dalitz point outside declared range')
    category=positions[coord[:,0],coord[:,1]]
    if np.any(category<0):
        raise ValueError('Dalitz point outside kinematic support')
    return category


def reduce_reference(reference,library,parent,model,scratch):
    row=library['runs']['phase_space']
    path=ROOT/row['root']
    if convergence.sha256(path)!=row['root_sha256']:
        raise ValueError('Phase-space ROOT changed')
    ids=scan.load_reaction_event_ids(path)
    partition=convergence.partitions if reference=='old' else pooled.prior.partition
    template,heldout=partition(ids,2)
    if np.intersect1d(template,heldout).size or len(heldout)!=200000:
        raise ValueError('Expected disjoint 200000-event held-out phase-space bank')
    size=int(ids.max())+1
    role=np.zeros(size,bool);role[heldout]=True
    category=np.full(size,int(mismatch.geometric_support(30).sum()),np.int32)
    weights=np.zeros((size,len(MODELS)))
    blocks=np.full(size,-1,np.int32)
    reference_id=int(reference=='new')
    order=np.random.default_rng([PLAN['block_seed'],reference_id]).permutation(heldout)
    blocks[order]=np.arange(len(order))%PLAN['blocks']
    moments=np.zeros((len(MODELS),3));spectra=np.zeros((len(MODELS),120))
    dalitz=np.zeros((len(MODELS),60,60))
    energy_edges=np.linspace(0,6,121)
    conservation=0.;cm_recovery=0.;q_range=[np.inf,-np.inf]
    fields=['event']+[f'{prefix}{i}' for i in (1,2,3) for prefix in
        ('e_alpha','theta_lab_alpha','phi_lab_alpha','e_3alpha_cm_alpha')]
    with uproot.open(path,handler=uproot.source.file.MemmapSource) as stream:
        for chunk in stream['reaction'].iterate(fields,step_size=25000,library='np'):
            keep=role[chunk['event']]
            selected={k:v[keep] for k,v in chunk.items()}
            if not np.any(keep):continue
            event=reaction_vectors(selected,model.p.mass)
            stored=np.column_stack([selected[f'e_3alpha_cm_alpha{i}'] for i in (1,2,3)])
            cm_recovery=max(cm_recovery,float(np.max(abs(event[...,3]-stored))))
            conservation=max(conservation,float(np.max(np.linalg.norm(event[...,:3].sum(axis=1),axis=-1))))
            q=event[...,3].sum(axis=1);q_range=[min(q_range[0],float(q.min())),max(q_range[1],float(q.max()))]
            w=model.weights(event)
            weights[selected['event']]=w
            fraction=np.sort(event[...,3],axis=1)/q[:,None]
            observable=np.column_stack([fraction[:,0],fraction[:,2],np.sum(fraction*fraction,axis=1)])
            moments+=w.T@observable
            for m in range(len(MODELS)):
                spectra[m]+=np.histogram(event[...,3].ravel(),bins=energy_edges,weights=np.repeat(w[:,m],3))[0]
                for permutation in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)):
                    a,b,c=event[:,permutation,3].T
                    x,y=np.sqrt(3)*(b-c)/q,(2*a-b-c)/q
                    dalitz[m]+=np.histogram2d(x,y,bins=60,range=[mismatch.LIMITS,mismatch.LIMITS],weights=w[:,m]/6)[0]
    if cm_recovery>1e-8 or conservation>1e-8 or np.any(weights[heldout].sum(axis=0)<=0):
        raise RuntimeError('CM reconstruction failed')
    response_seed=convergence.RESPONSE_SEED if reference=='old' else pooled.prior.P['response_seed']
    rng=np.random.default_rng(np.random.SeedSequence([response_seed,2,1]))
    complete_count=quality_count=0
    with tempfile.TemporaryDirectory(prefix=f'rmatrix-{reference}-',dir=scratch) as folder:
        values=np.memmap(Path(folder)/'sphere.dat',mode='w+',dtype=float,shape=(size,3,7))
        counts=np.zeros((size,3),np.int32)
        sphere_fields=['event_id','pdg','h11b_particle_source','generator_particle_index',
            'kinetic_energy_MeV','px_MeV_c','py_MeV_c','pz_MeV_c','x_mm','y_mm','z_mm']
        with uproot.open(path,handler=uproot.source.file.MemmapSource) as stream:
            for chunk in stream['virtual_sphere'].iterate(sphere_fields,step_size=100000,library='np'):
                convergence.accumulate_sphere(values,counts,chunk)
        complete=np.all(counts==1,axis=1)
        for first in range(0,size,100000):
            last=min(size,first+100000)
            event_ids=np.arange(first,last)[complete[first:last]&role[first:last]]
            rows=np.asarray(values[event_ids]).copy()
            events=dict(event_id=event_ids,kinetic=rows[:,:,0],momentum=rows[:,:,1:4],hit_position=rows[:,:,4:7])
            complete_count+=len(event_ids)
            events=scan.apply_mask(events,scan.quality_mask(events['kinetic'],events['momentum'],1e-4))
            quality_count+=len(events['event_id'])
            rec,_=scan.reconstruct_with_granularity(events,2.,2.,0.,0.,'sphere-center',np.array([0.,0.,170.]))
            measured,_,_=scan.apply_closure_energy_response(rec,100.,100.,rng.standard_normal(rec['kinetic'].shape))
            xy=np.column_stack(scan.ordered_dalitz_xy(measured))
            category[measured['event_id']]=category_indices(xy)
        del values
    base=np.asarray(parent['inputs']['references'][reference])
    expected=binning.coarsen(base[2,:-1]*len(heldout),30)
    raw=np.bincount(category[heldout],minlength=len(expected)+1)
    if not np.allclose(raw[:-1],expected,atol=1e-8,rtol=0):
        raise RuntimeError('Held-out phase-space response stream no longer replays exactly')
    w=weights[heldout];cat=category[heldout];block=blocks[heldout]
    weighted=np.zeros((PLAN['blocks'],len(MODELS),len(raw)))
    squared=np.zeros_like(weighted)
    for m in range(len(MODELS)):
        np.add.at(weighted[:,m,:],(block,cat),w[:,m])
        np.add.at(squared[:,m,:],(block,cat),w[:,m]**2)
    sums=w.sum(axis=0)
    physical=dict(energy_edges=energy_edges,spectrum_density=spectra/(3*sums[:,None]*np.diff(energy_edges)),
        symmetric_dalitz=dalitz/sums[:,None,None],moments=moments/sums[:,None])
    if not np.allclose(np.sum(physical['symmetric_dalitz'],axis=(1,2)),1,atol=1e-12):
        raise RuntimeError('Physical Dalitz lost weight')
    return plain(dict(block_histograms=weighted,block_sumw2=squared,
        probability=weighted.sum(axis=0)/sums[:,None],physical=physical,
        summary=dict(generated=len(heldout),complete=complete_count,quality=quality_count,accepted=int(raw[:-1].sum()),
            effective_events=sums*sums/(w*w).sum(axis=0),maximum_weight_fraction=w.max(axis=0)/sums,
            efficiency=weighted[:,:,:-1].sum(axis=(0,2))/sums,
            q_range_MeV=q_range,CM_recovery_max_MeV=cm_recovery,CM_momentum_max_MeV=conservation,
            reference_ids_sha256=digest(heldout),no_overlap_with_pooled_MC=True,
            root=row['root'],root_sha256=row['root_sha256'],global_seed=row['validation']['seed'],
            response_seed=response_seed,macro_text=row['macro_text'],macro_sha256=row['macro_sha256'],
            original_source_sha256=library['configuration']['source_sha256'],
            original_binary_sha256=library['configuration']['binary_sha256'])))


def mixture(inputs,reference,model_index,direct,block_counts=None):
    base=np.asarray(inputs['base_references'][reference]).copy()
    reference_state=inputs['weighted'][reference]
    if block_counts is None:
        alpha1=np.asarray(reference_state['probability'])[model_index]
    else:
        total=np.asarray(block_counts)@np.asarray(reference_state['block_histograms'])[:,model_index,:]
        alpha1=total/total.sum()
    base[1]=alpha1
    probability=pooled.fractions(direct,PLAN['r'])@base
    if np.any(probability<0) or not np.isclose(probability.sum(),1,atol=1e-12):
        raise ValueError('Invalid weighted reference probability')
    return probability


def evaluate(data,inputs,direct,*,upper=False):
    mc=np.asarray(inputs['raw_templates']);generated=np.asarray(inputs['generated'])
    fixed=finite.fixed_model(data,mc,generated)
    joint=banks.JointMultinomial(data,mc,generated,start=finite.accepted_weights(fixed.mode,fixed.efficiencies))
    out=dict(map=joint.mode,q0=joint.q0,q_upper=joint.statistic(direct,upper=True),
        deviance=2*joint.loss,shape_gap=joint.max_shape_gap,restarts=len(joint.shape_restart_history))
    if upper:out['upper']=joint.upper_limit(inputs['cutoffs']['q_upper'])
    return plain(out)


_INPUTS=None
def _initialize(inputs):
    global _INPUTS
    _INPUTS=inputs


def _chunk(task):
    phase,reference,direct,first,last=task
    records=[]
    for index in range(first,last):
        blocks=None
        if phase=='bootstrap':
            rng=np.random.default_rng([PLAN['bootstrap_seed'],int(reference=='new'),index])
            blocks=rng.multinomial(PLAN['blocks'],np.full(PLAN['blocks'],1/PLAN['blocks']))
        for m in range(len(MODELS)):
            p=mixture(_INPUTS,reference,m,direct,blocks)
            if phase=='data':
                rng=np.random.default_rng([PLAN['data_seed'],int(reference=='new'),m,int(direct*1e6),index])
                data=rng.multinomial(PLAN['data_budget'],p)[:-1]
            else:data=PLAN['data_budget']*p[:-1]
            try:
                result=evaluate(data,_INPUTS,direct)
            except Exception as exc:
                raise RuntimeError(f'No failed fit omitted: {task}, index={index}, model={MODELS[m]}') from exc
            records.append(dict(index=index,model=MODELS[m],**result))
    return records


def wilson(count,n):
    z=norm.ppf(.975);p=count/n;den=1+z*z/n
    centre=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(centre-half),float(centre+half)]


def summarize(records,inputs):
    output={}
    for reference in ('old','new'):
        output[reference]={}
        for direct in PLAN['direct']:
            key=f'{direct:.3f}';output[reference][key]={}
            data=[r for k,rows in records.items() if k.startswith(f'data:{reference}:{key}:') for r in rows]
            boot=[r for k,rows in records.items() if k.startswith(f'bootstrap:{reference}:{key}:') for r in rows]
            baseboot=np.array([r['map'][2] for r in boot if r['model']=='bw'])
            for model in MODELS:
                selected=[r for r in data if r['model']==model]
                samples=[r for r in boot if r['model']==model]
                if len(selected)!=PLAN['repetitions'] or len(samples)!=PLAN['bootstrap_repetitions']:
                    raise ValueError('Incomplete ensemble')
                rejected=sum(r['q0']>inputs['cutoffs']['q0'] for r in selected)
                included=sum(r['q_upper']<=inputs['cutoffs']['q_upper'] for r in selected)
                d=np.array([r['map'][2] for r in selected])
                b=np.array([r['map'][2] for r in samples]);delta=b-baseboot
                output[reference][key][model]=dict(mean=float(d.mean()),median=float(np.median(d)),
                    rejection_count=rejected,rejection_rate=rejected/len(d),rejection_wilson=wilson(rejected,len(d)),
                    inclusion_count=included,inclusion_rate=included/len(d),inclusion_wilson=wilson(included,len(d)),
                    mean_deviance=float(np.mean([r['deviance'] for r in selected])),
                    bootstrap_mean=float(b.mean()),bootstrap_sd=float(b.std(ddof=1)),
                    paired_shift_mean=float(delta.mean()),paired_shift_sd=float(delta.std(ddof=1)),
                    paired_shift_percentiles=np.percentile(delta,[2.5,97.5]).tolist())
    return output


def run(scratch,*,workers=8):
    if not 1<=workers<=8:raise ValueError('Use 1..8 workers')
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    document=json.loads(NOTEBOOK.read_text());parent=document['metadata'][pooled.KEY]
    if not parent.get('completed_at'):raise ValueError('Pooled calibration is incomplete')
    if digest({k:v for k,v in parent.items() if k not in ('completed_at','checkpoint_sha256')})!=parent['checkpoint_sha256']:
        raise ValueError('Pooled calibration checksum changed')
    if KEY in document['metadata']:raise ValueError('Preserve existing completed R-matrix study')
    validation=json.loads((scratch/'validation.json').read_text())
    if not validation['all_passed']:raise ValueError('Numerical physics validation failed')
    fingerprints={str(p.relative_to(ROOT)):convergence.sha256(p) for p in
        [Path(__file__),Path(__file__).with_name('rmatrix_162_model.py'),ROOT/'g4/validation/validate_rmatrix_162.py']}
    configuration=dict(version=VERSION,plan=PLAN,source_sha256=fingerprints,
        parent_checkpoint=parent['checkpoint_sha256'],validation_sha256=digest(validation),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    cache=scratch/'reduced.json'
    if cache.exists():
        cached=json.loads(cache.read_text())
        if cached['configuration']!=configuration:raise ValueError('Reduction checkpoint changed')
        inputs=cached['inputs']
    else:
        model=AmplitudeModel()
        weighted={}
        for reference,library_key in [('old','codex_template_convergence_libraries'),('new','codex_independent_replication_libraries')]:
            print(f'Reducing {reference} held-out phase-space reactions',flush=True)
            weighted[reference]=reduce_reference(reference,document['metadata'][library_key],parent,model,scratch)
            print(weighted[reference]['summary'],flush=True)
        references={name:np.column_stack([binning.coarsen(np.asarray(p)[:,:-1],30),np.asarray(p)[:,-1]])
                    for name,p in parent['inputs']['references'].items()}
        # This is an intentional stress test: freeze the current BW thresholds.
        cutoffs=parent['calibration']['frozen']['30']['cutoffs']['joint'] if 'calibration' in parent else parent['frozen']['30']['cutoffs']['joint']
        inputs=plain(dict(weighted=weighted,base_references=references,
            raw_templates=binning.coarsen(parent['inputs']['raw_templates'],30),
            generated=parent['inputs']['generated'],cutoffs=cutoffs,model=model.provenance()))
        cache.write_text(json.dumps(dict(configuration=configuration,inputs=inputs),ensure_ascii=False,allow_nan=False))
    raw=np.asarray(inputs['raw_templates']);generated=np.asarray(inputs['generated'])
    if np.any(raw!=np.floor(raw)) or np.any(generated!=np.floor(generated)) or np.any(raw.sum(axis=1)>generated):
        raise ValueError('The auxiliary MC must be actual integer counts, never importance weights')
    start=time.monotonic();asymptotic={};matched={}
    for reference in ('old','new'):
        asymptotic[reference]={};matched[reference]={}
        for direct in PLAN['direct']:
            key=f'{direct:.3f}';asymptotic[reference][key]={};matched[reference][key]={}
            for m,model in enumerate(MODELS):
                probability=mixture(inputs,reference,m,direct)
                data=PLAN['data_budget']*probability[:-1]
                result=evaluate(data,inputs,direct,upper=True)
                asymptotic[reference][key][model]=result
                # Arithmetic closure only: exact matched weighted law, no MC claim.
                channels=np.asarray(inputs['base_references'][reference]).copy()
                channels[1]=np.asarray(inputs['weighted'][reference]['probability'])[m]
                efficiency=channels[:,:-1].sum(axis=1)
                oracle=mismatch.boundary._Likelihood(data,channels[:,:-1],efficiency,0.)
                matched[reference][key][model]=oracle.mode.tolist()
                if np.max(abs(oracle.mode-pooled.fractions(direct,PLAN['r'])))>1e-7:
                    raise RuntimeError('Exact-law arithmetic closure failed')
                print('Asimov',reference,key,model,result,flush=True)
    checkpoint=scratch/'ensembles.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(input_sha256=digest(inputs),records={})
    if state['input_sha256']!=digest(inputs):raise ValueError('Fit checkpoint inputs changed')
    tasks={}
    for phase,total in [('data',PLAN['repetitions']),('bootstrap',PLAN['bootstrap_repetitions'])]:
        for reference in ('old','new'):
            for direct in PLAN['direct']:
                for first in range(0,total,25):
                    tasks[f'{phase}:{reference}:{direct:.3f}:{first:04d}']=(phase,reference,direct,first,min(total,first+25))
    pending={k:v for k,v in tasks.items() if k not in state['records']}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize,initargs=(inputs,)) as pool:
        futures={pool.submit(_chunk,t):k for k,t in pending.items()}
        last_notice=time.monotonic()
        for future in as_completed(futures):
            state['records'][futures[future]]=future.result()
            if len(state['records'])%4==0 or len(state['records'])==len(tasks):
                temp=checkpoint.with_suffix('.pending');temp.write_text(json.dumps(state,allow_nan=False));temp.replace(checkpoint)
            if time.monotonic()-last_notice>20:
                print(f"Completed {len(state['records'])}/{len(tasks)} fit chunks",flush=True);last_notice=time.monotonic()
    records={key:state['records'][key] for key in sorted(tasks)}
    summary=summarize(records,inputs)
    result=plain(dict(configuration=configuration,inputs=inputs,validation=validation,
        asimov=asymptotic,matched_exact_closure=matched,summary=summary,records=records,
        completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-start))
    result['checkpoint_sha256']=digest(result)
    pooled.prior.generator.save_metadata(NOTEBOOK,KEY,result)
    (scratch/'completed.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))
    print('R-matrix sensitivity COMPLETE',flush=True)
    return result


def load_cached(notebook=NOTEBOOK):
    state=json.loads(Path(notebook).read_text())['metadata'][KEY]
    if digest({k:v for k,v in state.items() if k!='checkpoint_sha256'})!=state['checkpoint_sha256']:
        raise ValueError('R-matrix cached result checksum changed')
    return state


def plot_physics(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    colors=['black','#d98b16','#2475b4','#b13d3d']
    physical=state['inputs']['weighted']['new']['physical']
    edges=np.asarray(physical['energy_edges']);centre=(edges[:-1]+edges[1:])/2
    for m,(label,color) in enumerate(zip(LABELS,colors)):
        axes[0].plot(centre,physical['spectrum_density'][m],color=color,label=label)
        for reference,style in [('old','--'),('new','-')]:
            p=np.asarray(state['inputs']['weighted'][reference]['probability'])[m,:-1]
            # Fixed ordered Dalitz x projection after the common response.
            support=mismatch.geometric_support(30);plane=np.zeros((30,30));plane[support]=p/p.sum()
            x=np.linspace(*mismatch.LIMITS,31);xx=(x[:-1]+x[1:])/2
            axes[1].plot(xx,plane.sum(axis=1),style,color=color,alpha=.85)
    axes[0].set(xlabel='Single alpha CM kinetic energy (MeV)',ylabel='Probability density (1/MeV)',title='Figure 15(a): before response')
    axes[1].set(xlabel='Ordered Dalitz x',ylabel='Probability per x bin',xlim=(0,.9),title='Figure 15(b): matched response; dashed/solid = old/new')
    axes[0].legend(fontsize=8);fig.suptitle('162 keV: independent R-matrix / Coulomb controls')
    return fig


def plot_fit(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    labels=['BW','R, no phase','R + phase','R + Coulomb'];x=np.arange(4)
    for reference,offset,color in [('old',-.07,'#2475b4'),('new',.07,'#b13d3d')]:
        for direct,marker,style in [(0.,'o','-'),(.005,'s','--')]:
            key=f'{direct:.3f}'
            rows=[state['asimov'][reference][key][m] for m in MODELS]
            error=[state['summary'][reference][key][m]['bootstrap_sd']*100 for m in MODELS]
            axes[0].errorbar(x+offset,[r['map'][2]*100 for r in rows],yerr=error,fmt=marker+style,color=color,label=f'{reference}, injected {direct*100:g}%')
            axes[1].plot(x+offset,[r['upper']*100 for r in rows],marker+style,color=color)
    for ax in axes:ax.set_xticks(x,labels,rotation=15)
    axes[0].set(ylabel='Fitted phase-space fraction (%)',title='Figure 28(a): expected counts; bars = reference MC bootstrap SD')
    axes[1].set(ylabel='Upper limit (%)',title='Figure 28(b): frozen BW cutoff, model stress test')
    axes[0].legend(fontsize=8);fig.suptitle('162 keV: actual pooled BW templates, identical detector response')
    return fig


def plot_dalitz(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(15,4.2),layout='constrained',sharex=True,sharey=True)
    values=np.asarray(state['inputs']['weighted']['new']['physical']['symmetric_dalitz'])
    maximum=float(values.max())
    for i,ax in enumerate(axes):
        im=ax.imshow(values[i].T,origin='lower',extent=[*mismatch.LIMITS,*mismatch.LIMITS],
                     cmap='viridis',vmin=0,vmax=maximum,aspect='equal')
        ax.set(title=f'Figure 16({"abcd"[i]})\n{LABELS[i]}',xlabel='Symmetric Dalitz x')
    axes[0].set_ylabel('Symmetric Dalitz y')
    fig.colorbar(im,ax=axes,label='Probability per bin (one common scale)',shrink=.7)
    fig.suptitle('162 keV, before response: six permutations share each event weight')
    return fig


def plot_rates(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for reference,offset,color in [('old',-.06,'#2475b4'),('new',.06,'#b13d3d')]:
        for ax,direct,quantity,interval in [(axes[0],0.,'rejection_rate','rejection_wilson'),(axes[1],.005,'inclusion_rate','inclusion_wilson')]:
            rows=[state['summary'][reference][f'{direct:.3f}'][m] for m in MODELS]
            value=np.array([r[quantity] for r in rows])*100
            bounds=np.array([r[interval] for r in rows]).T*100
            ax.errorbar(np.arange(4)+offset,value,yerr=np.maximum(0,np.stack([value-bounds[0],bounds[1]-value])),fmt='o',color=color,label=reference)
    for ax in axes:ax.set_xticks(np.arange(4),['BW','R, no phase','R + phase','R + Coulomb'],rotation=15);ax.set_ylim(-3,103);ax.legend()
    axes[0].set(ylabel='Null rejection (%)',title='Figure 29(a): zero phase-space component')
    axes[1].set(ylabel='Upper-limit inclusion (%)',title='Figure 29(b): injected phase-space component = 0.5%')
    axes[0].axhline(5,color='gray',ls=':');axes[1].axhline(95,color='gray',ls=':')
    fig.suptitle('Frozen BW inference on weighted empirical laws; 500 data trials per point')
    return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run(args.scratch,workers=args.workers)
