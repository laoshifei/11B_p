"""Independent global-library replication with frozen and recalibrated tests.

Importing this module never generates events or starts ensembles. The CLI
reduces verified new ROOT samples and compares old/new reference-template
crossings. Notebook plots consume only cached results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import tempfile
import time

import numpy as np
import uproot

if __package__ in (None, ""):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.bayes import binning_budget_162_study as binning
from analysis.bayes import generate_162_replication_libraries as generator

banks, finite, convergence = binning.banks, binning.finite, binning.convergence
ROOT, NOTEBOOK = binning.ROOT, binning.NOTEBOOK
digest, plain = binning.digest, binning.plain
KEY = "codex_independent_library_162_study"
VERSION = "2026-10-23-v1-independent-global-library-crossing"
P = generator.PROTOCOL


def partition(ids, channel):
    ids = np.asarray(ids,np.int64)
    if ids.ndim!=1 or len(ids)<2 or len(np.unique(ids))!=len(ids) or len(ids)%2 or ids.min()<0:
        raise ValueError('Need unique, nonnegative reaction IDs and equal preselection halves')
    order = np.random.default_rng(np.random.SeedSequence([P['split_seed'],channel])).permutation(ids)
    return order[:len(order)//2],order[len(order)//2:]


def reduce_channel(path, channel, scratch):
    """Allocate before cuts, stream all generator indices, respond once."""
    scan,mismatch = convergence.mismatch.scan,convergence.mismatch
    ids = scan.load_reaction_event_ids(path)
    allocated = partition(ids,channel)
    if any(len(group)!=P['generated'][channel] for group in allocated):
        raise ValueError('New full template/reference generated budgets differ from the frozen plan')
    size = int(np.max(ids))+1
    role = np.full(size,-1,np.int8)
    for index,group in enumerate(allocated):
        role[group] = index
    support = mismatch.geometric_support(60)
    histograms = np.zeros((2,int(support.sum())),np.int64)
    summary = dict(generated=[len(g) for g in allocated],ids_sha256=[digest(g) for g in allocated],
        disjoint=not np.intersect1d(*allocated).size,complete=[0,0],quality=[0,0],accepted=[0,0])
    with tempfile.TemporaryDirectory(prefix=f'replication-sphere-{channel}-',dir=scratch) as folder:
        values = np.memmap(Path(folder)/'values.dat',mode='w+',dtype=np.float64,shape=(size,3,7))
        counts = np.zeros((size,3),np.int32)
        fields = ['event_id','pdg','h11b_particle_source','generator_particle_index',
                  'kinetic_energy_MeV','px_MeV_c','py_MeV_c','pz_MeV_c','x_mm','y_mm','z_mm']
        with uproot.open(path,handler=uproot.source.file.MemmapSource) as stream:
            for chunk in stream['virtual_sphere'].iterate(fields,step_size=100000,library='np'):
                convergence.accumulate_sphere(values,counts,chunk)
        complete = np.all(counts==1,axis=1)
        if np.any(complete & (role<0)):
            raise RuntimeError('Complete sphere triplet without a generated reaction')
        rngs = [np.random.default_rng(np.random.SeedSequence([P['response_seed'],channel,r])) for r in range(2)]
        for first in range(0,size,100000):
            last = min(size,first+100000)
            for subset in range(2):
                event = np.arange(first,last)[complete[first:last] & (role[first:last]==subset)]
                rows = np.asarray(values[event]).copy()
                events = dict(event_id=event,kinetic=rows[:,:,0],momentum=rows[:,:,1:4],hit_position=rows[:,:,4:7])
                summary['complete'][subset] += len(event)
                events = scan.apply_mask(events,scan.quality_mask(events['kinetic'],events['momentum'],1e-4))
                summary['quality'][subset] += len(events['event_id'])
                if not len(events['event_id']):
                    continue
                reconstructed,_ = scan.reconstruct_with_granularity(events,2.,2.,0.,0.,'sphere-center',np.array([0.,0.,170.]))
                measured,_,_ = scan.apply_closure_energy_response(reconstructed,100.,100.,rngs[subset].standard_normal(reconstructed['kinetic'].shape))
                xy = np.column_stack(scan.ordered_dalitz_xy(measured))
                histograms[subset] += mismatch.histogram_xy(xy,60)[support].astype(np.int64)
                summary['accepted'][subset] += len(xy)
        del values
    if not np.array_equal(histograms.sum(axis=1),summary['accepted']):
        raise RuntimeError('Reduction lost accepted events')
    return histograms,summary


def prepare_inputs(parent, libraries, scratch):
    if not libraries.get('completed_at'):
        raise ValueError('Independent simulation libraries are incomplete')
    configuration = libraries['configuration']
    if configuration['protocol']!=P or configuration['parent_checkpoint_sha256']!=parent['checkpoint_sha256']:
        raise ValueError('Generation and analysis protocols differ')
    expected = {str(b):parent['frozen'][f'{b}_x4'] for b in P['bins']}
    if configuration['frozen_old_cutoffs']!=expected:
        raise ValueError('Frozen external-test cutoffs changed')
    fingerprint = dict(parent=parent['checkpoint_sha256'],library_configuration=digest(configuration),
        roots={name:libraries['runs'][name]['root_sha256'] for name in convergence.CHANNELS},
        source_sha256=convergence.sha256(__file__))
    for name in convergence.CHANNELS:
        row = libraries['runs'][name]
        if convergence.sha256(ROOT/row['root'])!=row['root_sha256']:
            raise ValueError(f'Changed independent ROOT: {name}')
    cache = Path(scratch)/'reduced_inputs.json'
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached['fingerprint']!=fingerprint:
            raise ValueError('Independent reduction cache changed')
        return cached['inputs']
    template,data,summaries = [],[],{}
    for channel,name in enumerate(convergence.CHANNELS):
        h,summary = reduce_channel(ROOT/libraries['runs'][name]['root'],channel,scratch)
        template.append(h[0])
        data.append(h[1])
        summaries[name] = summary
        print(f"Reduced NEW {name}: generated={summary['generated']}, accepted={summary['accepted']}",flush=True)
    old = parent['inputs']['cases']['60_x4_1234']
    generated = np.asarray(P['generated'])
    references = dict(old=parent['inputs']['fine_truth'],new=banks.reaction_population(data,generated))
    templates = dict(old=dict(raw_templates=old['raw_templates'],generated=old['generated']),
                     new=dict(raw_templates=np.stack(template),generated=generated))
    if not np.array_equal(old['generated'],generated):
        raise ValueError('Old/new full generated budgets differ')
    zeros = {f'{reference}_{template}':[int(np.asarray(bank['raw_templates'])[c][np.asarray(references[reference])[c,:-1]==0].sum()) for c in range(3)]
             for reference in references for template,bank in templates.items()}
    result = plain(dict(references=references,templates=templates,new_reference_histograms=data,
        reference_generated=dict(old=parent['inputs']['reference_generated'],new=generated),
        summaries=summaries,zero_reference_mc_events=zeros,old_frozen=expected,
        provenance=dict(roots={name:libraries['runs'][name]['root'] for name in convergence.CHANNELS},
            root_sha256=fingerprint['roots'],global_seeds={name:libraries['runs'][name]['validation']['seed'] for name in convergence.CHANNELS},
            split_seed=P['split_seed'],response_seed=P['response_seed'],support_bins=dict(fine=497,coarse=144),
            scope='Two independent global simulation libraries, disjoint within-library template/reference halves; finite empirical references')))
    cache.write_text(json.dumps(dict(fingerprint=fingerprint,inputs=result),ensure_ascii=False,allow_nan=False))
    return result


def case_id(bins, reference, template):
    return f'{bins}_{reference}_{template}'


def draw(inputs, case, direct, seed, index, *, fluctuate_mc=False):
    truth = np.asarray(inputs['references'][case['reference']])
    data = binning.data_draw(truth,direct,seed,index)
    data_sha = digest(data)
    bank = inputs['templates'][case['template']]
    generated = np.asarray(bank['generated'])
    if fluctuate_mc:
        identity = int(round(direct*1e6))
        mc = np.stack([np.random.default_rng(np.random.SeedSequence([seed,identity,index,1,c])).multinomial(int(g),p)[:-1]
                       for c,(g,p) in enumerate(zip(generated,truth))])
    else:
        mc = np.asarray(bank['raw_templates'])
    return binning.coarsen(data,case['bins']),binning.coarsen(mc,case['bins']),generated,data_sha


_INPUTS, _CASE = None, None


def _initialize(inputs, case):
    global _INPUTS, _CASE
    _INPUTS, _CASE = inputs,case


def _chunk(task):
    direct,seed,first,last,fluctuate_mc = task
    output = {method:[] for method in finite.METHODS}
    for index in range(first,last):
        try:
            data,mc,generated,data_sha = draw(_INPUTS,_CASE,direct,seed,index,fluctuate_mc=fluctuate_mc)
            evaluated = banks.evaluate(data,mc,generated,direct)
            for method in output:
                evaluated[method]['paired_data_sha256'] = data_sha
                output[method].append(evaluated[method])
        except Exception as error:
            raise RuntimeError(f'Trial failed without omission: case={_CASE}, seed={seed}, d={direct}, index={index}') from error
    return plain(output)


def phase_settings(phase, reference):
    if reference not in ('old','new'):
        raise ValueError('Unknown reference')
    if phase=='calibration':
        if reference!='new':
            raise ValueError('Only the new empirical reference is recalibrated')
        return P['new_calibration_seed'],P['calibration_repetitions'],P['calibration_direct'],True
    if phase=='validation':
        if reference!='new':
            raise ValueError('Only the new matched population is revalidated')
        return P['new_validation_seed'],P['validation_repetitions'],P['direct'],True
    if phase=='conditional':
        seed = P['old_conditional_seed'] if reference=='old' else P['new_conditional_seed']
        return seed,P['validation_repetitions'],P['direct'],False
    raise ValueError('Unknown ensemble phase')


def ensemble(inputs, case, path, phase, *, workers=8, reused=None):
    seed,repetitions,points,fluctuate_mc = phase_settings(phase,case['reference'])
    config = plain(dict(version=VERSION,case=case,phase=phase,seed=seed,repetitions=repetitions,
        direct_values=points,budget=P['data_budget'],fluctuate_mc=fluctuate_mc,
        population_sha256=digest(inputs['references'][case['reference']]),input_sha256=digest(inputs),chunk_size=25))
    if reused is not None:
        prior = reused['configuration']
        if (phase!='conditional' or case['reference']!='old' or case['template']!='old'
                or prior['seed']!=seed or prior['repetitions']!=repetitions or prior['fluctuate_mc']
                or prior['budget']!=P['data_budget'] or prior['direct_values']!=points
                or prior['fine_population_sha256']!=config['population_sha256']
                or prior['case']['bins']!=case['bins'] or prior['case']['scale']!=4):
            raise ValueError('Baseline raw records do not match the frozen old/old experiment')
        config['reused_statistics_sha256'] = digest(reused['records'])
        return dict(configuration=config,records=binning.array_records(reused['records']),reused=True)
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else dict(configuration=config,chunks={})
    if state['configuration']!=config:
        raise ValueError('Replication ensemble checkpoint configuration changed')
    tasks = {f'{int(round(d*1e6)):05d}:{first}':(d,seed,first,min(first+25,repetitions),fluctuate_mc)
             for d in points for first in range(0,repetitions,25)}
    for key,chunk in state['chunks'].items():
        if key not in tasks or set(chunk)!=set(finite.METHODS) or any(len(rows)!=tasks[key][3]-tasks[key][2] for rows in chunk.values()):
            raise ValueError('Invalid complete-chunk checkpoint')
    def save():
        temporary = path.with_suffix('.pending')
        temporary.write_text(json.dumps(plain(state),allow_nan=False))
        temporary.replace(path)
    start,last_notice = time.monotonic(),time.monotonic()
    pending = {key:task for key,task in tasks.items() if key not in state['chunks']}
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize,initargs=(inputs,case)) as pool:
            futures = {pool.submit(_chunk,task):key for key,task in pending.items()}
            for future in as_completed(futures):
                try:
                    state['chunks'][futures[future]] = future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
                count = sum(len(c['joint']) for c in state['chunks'].values())
                if count%500==0 or len(state['chunks'])==len(tasks):
                    save()
                if time.monotonic()-last_notice>20:
                    print(f'{phase} {case}: {count}/{len(tasks)*25} completed trials; {time.monotonic()-start:.1f}s',flush=True)
                    last_notice=time.monotonic()
    output = {}
    for direct in points:
        key = f'{int(round(direct*1e6)):05d}'
        output[key] = {}
        for method in finite.METHODS:
            rows = [row for first in range(0,repetitions,25) for row in state['chunks'][f'{key}:{first}'][method]]
            output[key][method] = {name:np.asarray([row[name] for row in rows]) for name in rows[0]}
    save()
    return dict(configuration=config,records=output,reused=False)


def freeze(state):
    configuration = state['configuration']
    if configuration['phase']!='calibration' or configuration['seed']!=P['new_calibration_seed'] or configuration['case']['reference']!='new':
        raise ValueError('Only the predeclared new-reference calibration may set new cutoffs')
    rank = finite.calibration.tolerance_rank(configuration['repetitions'],comparisons=P['new_calibration_comparisons'])
    local,cutoffs = {},{}
    for method in finite.METHODS:
        local[method] = {key:float(np.sort(point[method]['q_upper'])[rank['rank']-1]) for key,point in state['records'].items()}
        cutoffs[method] = dict(q_upper=max(local[method].values()),
            q0=float(np.sort(state['records']['00000'][method]['q0'])[rank['rank']-1]))
    return dict(cutoffs=cutoffs,local_upper=local,tolerance=rank,
        scope='New held-out empirical reference; two binnings, full MC budget, two methods, d=0/.005/.02, r=.05')


def controls(inputs):
    output = []
    for bins in P['bins']:
        for reference in ('old','new'):
            for template in ('old','new'):
                case = dict(bins=bins,reference=reference,template=template)
                seed,_,_,_ = phase_settings('conditional',reference)
                data,mc,generated,_ = draw(inputs,case,.005,seed,17)
                fixed = finite.fixed_model(data,mc,generated)
                a = banks.JointMultinomial(data,mc,generated,start=banks.accepted_weights(fixed.mode,fixed.efficiencies))
                b = banks.JointMultinomial(data,mc,generated,start=a.weights,tolerance=1e-5)
                c = banks.JointMultinomial(data,mc,generated,start=[.1,.8,.1])
                loss = np.ptp([a.loss,b.loss,c.loss])
                direct = np.ptp([a.mode[2],b.mode[2],c.mode[2]])
                profile = abs(a.statistic(.005)-b.statistic(.005))
                if loss>.002 or direct>2e-5 or profile>.002:
                    raise RuntimeError(f'Precision/multistart control failed: {case}')
                output.append(dict(case=case,seed=seed,trial=17,loss_range=loss,direct_range=direct,
                                   profile_difference=profile,all_passed=True))
    return output


def compare_estimates(left, right, *, paired):
    """Right minus left; pairing applies only to the same data reference."""
    output = {}
    for key in left['records']:
        output[key] = {}
        for method in finite.METHODS:
            a,b = (np.asarray(state['records'][key][method]['map'])[:,2] for state in (left,right))
            if paired:
                if not np.array_equal(left['records'][key][method]['paired_data_sha256'],right['records'][key][method]['paired_data_sha256']):
                    raise ValueError('Do not pair different pseudo-data populations or trials')
                difference = b-a
                mean,se = difference.mean(),difference.std(ddof=1)/np.sqrt(len(difference))
            else:
                mean = b.mean()-a.mean()
                se = np.sqrt(b.var(ddof=1)/len(b)+a.var(ddof=1)/len(a))
            output[key][method] = dict(right_minus_left=mean,mcse=se,paired=paired)
    return output


def verify_all_pairing(records):
    for reference in ('old','new'):
        names = [f'conditional_{case_id(b,reference,t)}' for b in P['bins'] for t in ('old','new')]
        binning.verify_pairing(records,names)
    for phase in ('calibration','validation'):
        binning.verify_pairing(records,[f'{phase}_{b}' for b in P['bins']])
    a = records['conditional_30_old_old']['records']['05000']['joint']['paired_data_sha256']
    b = records['conditional_30_new_old']['records']['05000']['joint']['paired_data_sha256']
    if np.array_equal(a,b):
        raise ValueError('Old and new references unexpectedly share an identical data stream')


def run(*,notebook=NOTEBOOK,scratch,workers=8):
    start = time.monotonic()
    work = Path(scratch)
    work.mkdir(parents=True,exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    if KEY in document['metadata']:
        raise ValueError('Preserve the existing independent-replication checkpoint')
    parent = document['metadata'][binning.KEY]
    libraries = document['metadata'][generator.KEY]
    if digest({k:v for k,v in parent.items() if k not in ('completed_at','checkpoint_sha256')})!=parent['checkpoint_sha256']:
        raise ValueError('Verified binning/budget parent changed')
    inputs = prepare_inputs(parent,libraries,work)
    paths = [Path(__file__),Path(generator.__file__),Path(binning.__file__),Path(banks.__file__),
        Path(finite.__file__),Path(convergence.__file__),
        ROOT/'g4/analysis/bayes/finite_template_likelihood.py',
        ROOT/'g4/analysis/bayes/model_mismatch_162_study.py',
        ROOT/'g4/analysis/bayes/calibrate_162_weak_branch_intervals.py']
    configuration = plain(dict(version=VERSION,started_at=datetime.now().astimezone().isoformat(),protocol=P,
        parent_checkpoint_sha256=parent['checkpoint_sha256'],library_configuration_sha256=digest(libraries['configuration']),
        source_sha256={str(p.relative_to(ROOT)):convergence.sha256(p) for p in paths},analysis_workers=workers))
    plan = work/'plan.json'
    if plan.exists():
        previous = json.loads(plan.read_text())
        configuration['started_at'] = previous['started_at']
        if configuration!=previous:
            raise ValueError('Do not silently change the independent-library analysis protocol')
    plan.write_text(json.dumps(configuration,ensure_ascii=False,indent=1))
    snapshots = {str(p.relative_to(ROOT)):p.read_text() for p in paths}
    numerical = controls(inputs)
    print('Eight noisy-data precision/multistart controls PASS',flush=True)
    records,frozen,matched = {},{},dict(old_cutoffs={},new_cutoffs={})
    for bins in P['bins']:
        case = dict(bins=bins,reference='new',template='new')
        state = ensemble(inputs,case,work/f'calibration_{bins}.json','calibration',workers=workers)
        records[f'calibration_{bins}'] = state
        frozen[str(bins)] = freeze(state)
        print(f'Frozen NEW-reference {bins}: {frozen[str(bins)]["cutoffs"]}',flush=True)
    for bins in P['bins']:
        case = dict(bins=bins,reference='new',template='new')
        state = ensemble(inputs,case,work/f'validation_{bins}.json','validation',workers=workers)
        records[f'validation_{bins}'] = state
        matched['old_cutoffs'][str(bins)] = finite.summarize(state,inputs['old_frozen'][str(bins)])
        matched['new_cutoffs'][str(bins)] = finite.summarize(state,frozen[str(bins)])
        print(f'NEW-reference matched validation {bins} COMPLETE',flush=True)
    old_records = convergence.unpack(parent['raw_records'])
    primary,recalibrated,expected = {},{},{}
    for bins in P['bins']:
        for reference in ('old','new'):
            for template in ('old','new'):
                name = case_id(bins,reference,template)
                case = dict(bins=bins,reference=reference,template=template)
                reused = old_records[f'conditional_{bins}_x4_1234'] if (reference,template)==('old','old') else None
                state = ensemble(inputs,case,work/f'conditional_{name}.json','conditional',workers=workers,reused=reused)
                records[f'conditional_{name}'] = state
                primary[name] = finite.summarize(state,inputs['old_frozen'][str(bins)])
                if reference=='new':
                    recalibrated[name] = finite.summarize(state,frozen[str(bins)])
                expected[name] = {}
                bank = inputs['templates'][template]
                for direct in P['direct']:
                    data = binning.coarsen(P['data_budget']*(finite.fractions(direct)@np.asarray(inputs['references'][reference]))[:-1],bins)
                    expected[name][f'{int(round(direct*1e6)):05d}'] = banks.evaluate(data,np.asarray(binning.coarsen(bank['raw_templates'],bins)),np.asarray(bank['generated']),direct)
                print(f'External actual library {name} COMPLETE',flush=True)
    verify_all_pairing(records)
    contrasts = dict(template={},reference={},binning={})
    for bins in P['bins']:
        for reference in ('old','new'):
            contrasts['template'][f'{bins}_{reference}'] = compare_estimates(
                records[f'conditional_{case_id(bins,reference,"old")}'],records[f'conditional_{case_id(bins,reference,"new")}'],paired=True)
        for template in ('old','new'):
            contrasts['reference'][f'{bins}_{template}'] = compare_estimates(
                records[f'conditional_{case_id(bins,"old",template)}'],records[f'conditional_{case_id(bins,"new",template)}'],paired=False)
    for reference in ('old','new'):
        for template in ('old','new'):
            contrasts['binning'][f'{reference}_{template}'] = compare_estimates(
                records[f'conditional_{case_id(60,reference,template)}'],records[f'conditional_{case_id(30,reference,template)}'],paired=True)
    for relative,checksum in configuration['source_sha256'].items():
        if convergence.sha256(ROOT/relative)!=checksum:
            raise ValueError(f'Source changed during replication: {relative}')
    total = sum(len(p['joint']['map']) for state in records.values() for p in state['records'].values())
    reused_count = sum(len(p['joint']['map']) for state in records.values() if state['reused'] for p in state['records'].values())
    if (total,reused_count)!=(26000,4000):
        raise RuntimeError('Incomplete independent-replication record counts')
    state = plain(dict(configuration=configuration,inputs=inputs,new_frozen=frozen,matched_summary=matched,
        primary_summary=primary,recalibrated_summary=recalibrated,expected=expected,contrasts=contrasts,
        numerical_controls=numerical,pairing_checks=dict(within_reference=True,between_binnings=True,new_calibration_validation=True),
        raw_records=convergence.pack(records),source_snapshots=snapshots,elapsed_seconds=time.monotonic()-start,
        counts=dict(fit_trial_cases=total,reused_fit_trial_cases=reused_count,new_fit_trial_cases=total-reused_count,
            calibration=6000,matched_validation=4000,conditional=16000,
            distinct_calibration_data=3000,distinct_validation_data=2000,distinct_conditional_data=4000),
        interpretation='Primary external evaluation retains old frozen thresholds; new-reference recalibration and closure are separate. Only two global simulation realizations, not a population-wide conditional guarantee.'))
    state['checkpoint_sha256'] = digest(state)
    state['completed_at'] = datetime.now().astimezone().isoformat()
    generator.save_metadata(notebook,KEY,state)
    print(f'Independent global-library replication COMPLETE: {total} cases, {reused_count} reused; {state["elapsed_seconds"]:.1f}s',flush=True)
    return state


def plots(state):
    """Cached-only point-estimate, external-inclusion and closure figures."""
    import matplotlib.pyplot as plt
    pairs = [('old','old'),('old','new'),('new','old'),('new','new')]
    labels = ['Rold/Mold','Rold/Mnew','Rnew/Mold','Rnew/Mnew']
    x = np.arange(4)
    figures = []
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax,bins in zip(axes,P['bins']):
        names = [case_id(bins,r,t) for r,t in pairs]
        rows = [state['primary_summary'][name]['05000']['joint'] for name in names]
        ax.errorbar(x,[100*r['mean'] for r in rows],yerr=[100*r['mcse'] for r in rows],fmt='o',capsize=3,label='Mean over pseudo-data')
        ax.plot(x,[100*state['expected'][name]['05000']['joint']['map'][2] for name in names],'s',label='Expected-count fit')
        ax.axhline(.5,color='k',ls=':')
        ax.set(title=f'{bins} x {bins}; independent global libraries',xticks=x,xticklabels=labels,
            ylabel='Estimated fraction at true d=0.5% (%)',xlabel='Data reference / actual MC template')
        ax.legend(fontsize=8)
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax,bins in zip(axes,P['bins']):
        lower = 100.
        for offset,key,label in ((-.15,'02000','d=0.2%'),(0.,'05000','d=0.5%'),(.15,'20000','d=2%')):
            rows = [state['primary_summary'][case_id(bins,r,t)][key]['joint']['coverage'] for r,t in pairs]
            p = np.array([row['rate'] for row in rows])*100
            interval = np.array([row['wilson95'] for row in rows])*100
            lower = min(lower,float(interval.min()))
            ax.errorbar(x+offset,p,yerr=np.maximum(np.vstack([p-interval[:,0],interval[:,1]-p]),0),fmt='o',capsize=3,label=label)
        ax.axhline(95,color='k',ls=':')
        ax.set(title=f'{bins} x {bins}; OLD frozen cutoffs',xticks=x,xticklabels=labels,
            ylabel='Conditional upper-limit inclusion (%)',xlabel='Data reference / actual MC template',
            ylim=(max(0.,min(90.,lower-2)),101))
        ax.legend(fontsize=8)
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    lower = [100.,100.]
    for bins,color in ((30,'C0'),(60,'C1')):
        for cutoff,style in (('old_cutoffs','--'),('new_cutoffs','-')):
            label = f'{bins} x {bins}; '+('old cutoff' if cutoff=='old_cutoffs' else 'new calibration')
            matched = state['matched_summary'][cutoff][str(bins)]
            conditional = (state['primary_summary'] if cutoff=='old_cutoffs' else state['recalibrated_summary'])[case_id(bins,'new','new')]
            for index,points in enumerate((matched,conditional)):
                keys = sorted(points)
                rows = [points[key]['joint']['coverage'] for key in keys]
                p = np.array([row['rate'] for row in rows])*100
                interval = np.array([row['wilson95'] for row in rows])*100
                lower[index] = min(lower[index],float(interval.min()))
                axes[index].errorbar([int(key)/1e4 for key in keys],p,
                    yerr=np.maximum(np.vstack([p-interval[:,0],interval[:,1]-p]),0),marker='o',
                    ls=style,color=color,capsize=2,label=label)
    for index,ax in enumerate(axes):
        ax.axhline(95,color='k',ls=':')
        ax.set(xlabel='True uniform phase-space fraction (%)',ylabel='Upper-limit inclusion (%)',
            ylim=(max(0.,min(90.,lower[index]-2)),101))
        ax.legend(fontsize=8)
    axes[0].set_title('NEW reference; data and MC regenerated')
    axes[1].set_title('NEW reference; fixed NEW actual MC')
    figures.append(fig)
    return figures


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8,choices=range(1,9))
    args = parser.parse_args()
    run(scratch=args.scratch,workers=args.workers)
