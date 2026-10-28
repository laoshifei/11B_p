"""Ceiling-only ablation and propagated quadrature controls on frozen events.

Hard clipping is an estimator control, not a posterior or an independently
validated detector correction. No event is selected by its truth or residual.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_cluster_study as cluster

parent, previous, matching, energy = cluster.parent, cluster.previous, cluster.matching, cluster.energy
KEY = 'codex_laursen_162_cluster_controls'
VERSION = '2026-10-27-v1-ceiling-ablation'
CASES = ('previous_flat', 'ceiling_clip', *cluster.posterior.VARIANTS)
MAIN = CASES.index('main')
PLAN = dict(cases=CASES, scope=cluster.PLAN['eligibility'],
    ceiling_control='For the same observable eligible hits only, min(previous emitted kinetic energy, single-alpha ceiling at nominal direction)',
    interpretation='Hard clipping is a deterministic estimator ablation, not a normalized posterior or a unique causal decomposition',
    fixed_cohort='Same original Base category and generated denominator as the frozen cluster study; no residual-dependent selection',
    paired_contrast='Integrated main minus ceiling-only, evaluated on the same weighted events',
    numerical_scale_MeV=cluster.PLAN['numerical_scale_MeV'],
    numerical_scope='Reuse all four frozen quadratures; propagate their paired event changes, without claiming new convergence',
    limitations=['The ceiling shares the nominal parent-energy input with the closure diagnostic',
        'Quadrature spreads and effective-ray counts are not confidence intervals or posterior widths',
        'No new geometry quadrature, reaction simulation, experimental calibration, or branch fit',
        'Ancestry and original residuals are used only in post-estimation audits'])


def ceiling_clip(kinetic, direction):
    kinetic = np.asarray(kinetic, float)
    ceiling = cluster.posterior.single.kinetic_ceiling(direction)
    if kinetic.shape != ceiling.shape or not np.isfinite(kinetic).all() or np.any(kinetic < 0):
        raise ValueError('Expected finite nonnegative energies and matching directions')
    return np.minimum(kinetic, ceiling)


def corrected_cases(row, inferred):
    """Use observable matching only; preserve every out-of-scope hit and event."""
    baseline = np.asarray(row['baseline_kinetic_MeV'], float)
    kinetic = np.broadcast_to(baseline, (len(CASES), len(baseline))).copy()
    hits, reason = cluster.scope(row)
    failures = np.zeros(len(CASES), int)
    if len(hits):
        kinetic[1, hits] = ceiling_clip(baseline[hits], np.asarray(row['nominal_direction'])[hits])
        for case, name in enumerate(CASES[2:], 2):
            estimate = inferred[str(row['index'])][name]
            if estimate['valid']:
                value = np.asarray(estimate['mean'], float)
                if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('Invalid frozen posterior means')
                kinetic[case, hits] = value
            else:
                failures[case] += 1
    return kinetic, hits, reason, failures


def reduce_group(group, inferred, frozen, models):
    pool = group['pool']; n = len(pool)
    if not n:
        raise ValueError('Expected the nonempty frozen observable pool')
    weights = np.asarray([row['weights'] for row in pool])
    sums, squared = np.asarray(group['generated_weight']), np.asarray(group['squared_generated_weight'])
    matched = {key:np.asarray([row['matched'][key] for row in pool]) for key in pool[0]['matched']}
    direction = np.asarray([row['nominal_direction'] for row in pool])
    old_labels = np.asarray([row['baseline_labels'] for row in pool])
    old_residual = np.asarray([row['baseline_residual_MeV'] for row in pool])
    kinetic = []; records = []; reasons = Counter(); failures = np.zeros(len(CASES), int)
    for row in pool:
        values, hits, reason, failed = corrected_cases(row, inferred)
        kinetic.append(values); reasons[reason] += 1; failures += failed
        if not len(hits):
            continue
        estimates = inferred[str(row['index'])]
        valid = all(value['valid'] for value in estimates.values())
        differences = {name:float(np.max(abs(np.asarray(estimates[name]['mean'])-estimates['main']['mean'])))
                       for name in ('coarse', 'alternate', 'high_charge_order')} if valid else None
        removed = values[0,hits]-values[1,hits]
        pixels = np.asarray(row['matched']['pixel'])[hits]
        records.append(dict(index=row['index'], capped_hits=int(np.sum(removed > 0)),
            cap_removed_MeV=removed, pixels=pixels, shared_side=int(pixels[0,1] == pixels[1,1]),
            all_quadratures_valid=valid, energy_control_difference_MeV=differences,
            main_effective_rays=estimates['main']['effective_rays'],
            main_geometry_points=estimates['main']['geometry_points'], main_strata=estimates['main']['strata'],
            ancestry=row['ancestry'], original_residual_MeV=row['baseline_residual_MeV']))
    kinetic = np.asarray(kinetic)
    classification = np.zeros((n,len(CASES),2,2,2)); residual = np.zeros((n,len(CASES)))
    nominal = energy.nominal_parent(n)
    for case in range(len(CASES)):
        labels, r = previous.pixel.evaluate(matched,kinetic[:,case],direction,nominal)
        classification[:,case,0] = labels
        classification[:,case,1] = labels*(abs(r) <= .4)[:,None,None]
        residual[:,case] = r
    if (not np.array_equal(classification[:,0],old_labels)
            or not np.allclose(residual[:,0],old_residual,atol=1e-10,rtol=0)):
        raise ArithmeticError('Frozen baseline does not reproduce')
    baseline_full = np.asarray(frozen['classification'])[models,0]
    def accumulate(values):
        moments = matching.Moments(len(models),values.shape[1:])
        moments.add(weights,values)
        return moments.finish(sums,squared)
    difference = classification-classification[:,:1]
    paired = accumulate(difference)
    full = baseline_full[:,None]+paired['value']
    ancestry = np.eye(4)[[row['ancestry'] for row in pool]]
    ancestry_change = accumulate(difference[...,None,:]*ancestry[:,None,None,None,:,None])
    ancestry_full = np.asarray(frozen['ancestry'])[models,:1]+ancestry_change['value']
    category = previous.pixel.state_codes(old_labels[:,0])-1
    values = np.zeros((n,4,len(CASES),len(previous.METRICS)))
    for bucket in range(4):
        use = category == bucket; r = residual[use]
        if not np.isfinite(r).all():
            raise ArithmeticError('A fixed-cohort residual became nonfinite')
        keep = abs(r) <= .4
        statistic = np.stack([r,r*r,r < -1.,r > 1.,keep,keep.astype(float)-keep[:,:1]],axis=-1)
        values[use,bucket] = statistic-statistic[:,:1]
    moment_change = accumulate(values)
    cohort_weight = np.asarray(frozen['cohort_per_generated'])[models]
    old_moments = np.asarray(frozen['cohort_moments'])[models,:,0]
    conditional = old_moments[:,:,None]+np.divide(moment_change['value'],cohort_weight[...,None,None],
        out=np.zeros_like(moment_change['value']),where=cohort_weight[...,None,None] > 0)
    if np.min(full) < -1e-12 or np.max(full) > 1+1e-12 or np.min(conditional[...,1]) < -1e-10:
        raise ArithmeticError('Updated probabilities or second moments are invalid')
    if not np.allclose(ancestry_full.sum(axis=-2),full,atol=1e-12,rtol=0):
        raise ArithmeticError('Ancestry accounting does not close')
    compare = accumulate(classification[:,MAIN]-classification[:,1])
    compare_moments = accumulate(values[:,:,MAIN]-values[:,:,1])
    for record in records:
        i = next(i for i,row in enumerate(pool) if row['index'] == record['index'])
        record['fixed_category'] = int(category[i])
        record['residual_MeV'] = residual[i]
    return dict(classification=full,paired_change=paired,ancestry=ancestry_full,
        cohort_per_generated=cohort_weight,cohort_moments=conditional,
        cohort_paired_change_per_generated=moment_change,
        main_minus_clip=compare,main_minus_clip_moments_per_generated=compare_moments,
        diagnostics=dict(pool_events=n,eligible_events=len(records),reasons=dict(reasons),failures=failures,
            capped_events=sum(row['capped_hits'] > 0 for row in records),
            capped_hits=sum(row['capped_hits'] for row in records)),records=records)


def validate_cached(state, frozen):
    if state['parent_checksum'] != frozen['checksum'] or tuple(state['cases']) != CASES:
        raise ValueError('Frozen cluster parent or case definitions changed')
    selection = [CASES.index(name) for name in cluster.CASES]
    for ref in ('old','new'):
        new,old = state['results'][ref]['summary'],frozen['results'][ref]['summary']
        for name in ('classification','ancestry'):
            if not np.allclose(np.asarray(new[name])[:,selection],old[name],atol=1e-12,rtol=0):
                raise ArithmeticError('Frozen '+name+' changed')
        if not np.allclose(np.asarray(new['cohort_moments'])[:,:,selection],old['cohort_moments'],atol=1e-10,rtol=0):
            raise ArithmeticError('Frozen cohort moments changed')
        full = np.asarray(new['classification'])
        if not np.allclose(full[:,MAIN]-full[:,1],new['main_minus_clip']['value'],atol=1e-12,rtol=0):
            raise ArithmeticError('Direct paired contrast does not close')
        if not np.allclose(np.asarray(new['ancestry']).sum(axis=-2),full,atol=1e-12,rtol=0):
            raise ArithmeticError('Ancestry total changed')


def run(cache_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    frozen = cluster.checked_state(cache_path)
    pool = frozen['input_pool']
    if cluster.parent.digest({k:v for k,v in pool.items() if k!='checksum'}) != frozen['input_pool_checksum']:
        raise ValueError('Frozen input pool checksum changed')
    sources = {**frozen['source_sha256'],**frozen['frozen_source_sha256']}
    source = str(Path(__file__).relative_to(parent.ROOT))
    sources[source] = parent.sha256(Path(__file__))
    for name,sha in sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    results = {}
    for reference in ('old','new'):
        groups = [reduce_group(pool['results'][reference][channel],frozen['results'][reference]['inferred'][channel],
            frozen['results'][reference]['summary'],[0] if channel=='0' else list(range(1,6))) for channel in ('0','2')]
        fields = [name for name in groups[0] if name not in ('records','diagnostics')]
        summary = {name:previous.combine([group[name] for group in groups]) for name in fields}
        results[reference] = dict(summary=summary,
            diagnostics={channel:group['diagnostics'] for channel,group in zip(('0','2'),groups)},
            records={channel:group['records'] for channel,group in zip(('0','2'),groups)})
        print(reference,'ceiling-control events:',[group['diagnostics']['capped_events'] for group in groups],flush=True)
    state = parent.parent.plain(dict(version=VERSION,plan=PLAN,cases=CASES,parent_checksum=frozen['checksum'],
        input_pool_checksum=frozen['input_pool_checksum'],source_sha256={source:sources[source]},
        frozen_source_sha256={k:v for k,v in sources.items() if k!=source},source_snapshot=Path(__file__).read_text(),
        source_configuration=frozen['source_configuration'],inherited_evidence=frozen['evidence'],
        results=results,interpreter=sys.executable,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent.ROOT,text=True).strip(),
        started_at=started,completed_at=datetime.now().astimezone().isoformat(),elapsed_seconds=time.monotonic()-begin))
    validate_cached(state,frozen)
    for name,sha in sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Source changed during controls: '+name)
    state = cluster.write_checked(output,state)
    print('Completed controls:',state['checksum'],flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k:v for k,v in state.items() if k!='checksum'}) != state['checksum']:
        raise ValueError('Ceiling-control cache checksum changed')
    return state


def plot_ablation(state):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x = np.arange(len(CASES))
    for row,(model,bucket) in enumerate(((0,0),(5,1))):
        for ref,style in (('old','-'),('new','--')):
            values = np.asarray(state['results'][ref]['summary']['cohort_moments'])[model,bucket]
            axes[row,0].plot(x,1000*np.sqrt(np.maximum(0,values[:,1])),style,marker='o',label=ref)
            axes[row,1].plot(x,100*values[:,4],style,marker='o',label=ref)
        for col in (0,1):
            axes[row,col].set_xticks(x,['Previous','Ceiling only','Coarse','Main','Alt. seed','Charge 8'],rotation=25)
            axes[row,col].set_title(f'29({chr(97+2*row+col)}) alpha{row}, original m=2 own-label cohort')
            axes[row,col].grid(alpha=.2)
        axes[row,0].set_ylabel('Residual RMS / keV');axes[row,1].set_ylabel('Retention within +/-400 keV / %')
    axes[0,0].legend()
    fig.suptitle('Ceiling-only estimator control; same frozen events, labels and denominators')
    return fig


def plot_numerical(state):
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for row,ref in enumerate(('old','new')):
        for col,variant in enumerate(('coarse','alternate')):
            ax = axes[row,col]
            for channel,marker in (('0','o'),('2','^')):
                records = [r for r in state['results'][ref]['records'][channel] if r['all_quadratures_valid']]
                ax.scatter([r['main_effective_rays'] for r in records],
                    [1000*r['energy_control_difference_MeV'][variant] for r in records],
                    s=20,alpha=.65,marker=marker,label='alpha0' if channel=='0' else 'phase proposal')
            ax.axhline(20,color='firebrick',linestyle='--',linewidth=1)
            ax.set_title(f'30({chr(97+2*row+col)}) {ref}: main vs '+('coarse grid' if col==0 else 'alternate seed'))
            ax.set_xlabel('Effective geometry rays (diagnostic only)');ax.set_ylabel('Max. hit energy difference / keV')
            ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle('Frozen quadrature spread; dashed line is the declared 20 keV control scale')
    return fig


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cluster-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.cluster_cache,args.output)
