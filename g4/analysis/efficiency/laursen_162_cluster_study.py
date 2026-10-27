"""Sparse, exactly accounted response update for observable shared-strip clusters.

The input pool contains all preselection non-isolated crossing events from
the frozen references, not just accepted events or large residuals. Every
unchanged event retains the previous Flat response. Only explicit CLI calls
extract a pool or run the conditional two-hit quadrature.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_cluster_posterior as posterior
from analysis.efficiency import laursen_162_topology_study as previous

parent, matching, energy, material = previous.parent, previous.matching, previous.energy, previous.material
KEY = 'codex_laursen_162_cluster_study'
VERSION = '2026-10-27-v2-checkpointed-shared-strip-clusters'
CASES = ('previous_flat', *posterior.VARIANTS)
QUADRATURE = posterior.Quadrature()
PLAN = dict(cases=CASES, quadrature=asdict(QUADRATURE), windows=['none', 'w400'],
    pool='Resolved m2/m3, a foil-crossing observed support, and no previously eligible isolated hit; before kinematic or energy cuts',
    eligibility='Two integer D1 hits sharing a strip, complete D1 signal support, sum/share policy agreement',
    observations='Every occupied physical strip is used once in charge space; other strips are censored at the frozen threshold',
    latent='Two pixel positions, shared 2 mm square beam origin and common uniform boron depth; seven geometry variables',
    prior='Product of Flat T and isotropic solid-angle priors, each with the previous single-alpha ceiling; not a full joint three-alpha amplitude',
    estimator='Both source kinetic-energy posterior means; keep nominal momentum directions and fixed observed matching',
    scope='Conditional fixed two-particle interpretation; candidate agreement does not establish truth or model posterior probability',
    charge_integral='Correlated two-dimensional Gaussian, sequential truncated-normal transform and tensor Gauss-Legendre rule',
    strata='Exact strip-sharing breakpoints and a foil-plane split; preserve each rectangle volume',
    fallback='Any zero-evidence variant preserves both baseline energies for that cluster; no event is removed for numerical failure',
    fixed_cohort='Original Base accepted category, as in the previous pixel/topology study; update Flat moments by paired event differences',
    numerical_scale_MeV=.020,
    limitations=['Fixed ideal sharing and electronic response, without measured calibration',
        'Product single-particle supports omit full three-alpha energy and angular correlations',
        'Nominal directions retained; no fit to the event energy sum',
        'Unresolved matching, neighbors-only patterns and larger D1 clusters remain separately accounted',
        'No new Geant4 reaction, weak-component fit, or experimental-response validation'])


def checked_state(path):
    state = json.loads(Path(path).read_text())
    if parent.digest({k:v for k,v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Input state checksum changed: '+str(path))
    return state


def write_checked(path, value):
    """Publish only a complete, checksum-verified artifact; preserve finished files."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    state = parent.parent.plain(value)
    state['checksum'] = parent.digest({k:v for k,v in state.items() if k != 'checksum'})
    raw = json.dumps(state,ensure_ascii=False,allow_nan=False).encode('utf-8')
    pending = path.with_name(path.name+'.pending')
    with pending.open('wb',buffering=0) as stream:
        offset = 0
        while offset < len(raw):
            block = raw[offset:offset+65536]
            written = stream.write(block)
            if written is None or not 0 < written <= len(block):
                raise OSError('Invalid checkpoint write length')
            offset += written
    if checked_state(pending)['checksum'] != state['checksum']:
        raise OSError('Checkpoint write did not reproduce')
    if path.exists():
        raise FileExistsError(path)
    pending.replace(path)
    return state


class Checkpoint:
    """One writer, immutable completed records, and exact input/source bindings."""
    def __init__(self, path, bindings):
        self.path = Path(path); self.path.mkdir(parents=True,exist_ok=True)
        self.lock = (self.path/'.lock').open('a')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        path = self.path/'manifest.json'
        plain = parent.parent.plain(bindings)
        if path.exists():
            self.manifest = checked_state(path)
            if self.manifest['bindings'] != plain:
                self.close()
                raise ValueError('Checkpoint inputs, plan or source changed')
        else:
            self.manifest = write_checked(path,dict(bindings=plain,
                created_at=datetime.now().astimezone().isoformat()))

    def record_path(self, identity):
        return self.path/('_'.join(identity)+'.json')

    def get(self, identity):
        path = self.record_path(identity)
        if not path.exists():
            return None
        record = checked_state(path)
        if record['identity'] != list(identity) or record['manifest_checksum'] != self.manifest['checksum']:
            raise ValueError('Checkpoint event identity or source binding changed')
        return record['value']

    def put(self, identity, value):
        write_checked(self.record_path(identity),dict(identity=list(identity),value=value,
            manifest_checksum=self.manifest['checksum'],completed_at=datetime.now().astimezone().isoformat()))

    def elapsed(self):
        return (datetime.now().astimezone()-datetime.fromisoformat(self.manifest['created_at'])).total_seconds()

    def close(self):
        self.lock.close()


def scope(row):
    matched = {key:np.asarray(value) for key,value in row['matched'].items()}
    return posterior.eligible_cluster(matched, np.asarray(row['observed']), row['policy_agreement'])


def extract_pool(output):
    """Freeze the observable preselection pool; audit the full baseline first."""
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve an existing observable pool')
    metadata2 = json.loads(parent.NOTEBOOK.read_text())['metadata']
    frozen = metadata2[previous.KEY]
    if parent.digest({k:v for k,v in frozen.items() if k!='checksum'}) != frozen['checksum']:
        raise ValueError('Previous response checksum mismatch')
    sources = {**frozen['source_sha256'], **frozen['frozen_source_sha256']}
    for name,sha in sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    journal = Checkpoint(str(output)+'.progress',dict(mode='pool',parent_checksum=frozen['checksum'],
        source_sha256=sources,extraction_source_sha256=parent.sha256(Path(__file__))))
    tables = previous.response.RangeTables(metadata2[parent.KEY]['stopping'])
    metadata3 = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    amplitude = parent.parent.AmplitudeModel()
    begin = time.monotonic(); started = datetime.now().astimezone().isoformat(); results = {}
    for reference in ('old','new'):
        results[reference] = {}
        for channel in (0,2):
            identity = ('pool',reference,str(channel))
            saved = journal.get(identity)
            if saved is not None:
                results[reference][str(channel)] = saved
                print('Resumed complete pool group:',reference,channel,flush=True)
                continue
            source = parent.source_reference(metadata3,reference,channel,amplitude)
            prepared = matching.prepare(source,tables); weights = source['weights']; models = weights.shape[1]
            rng = np.random.default_rng([matching.PLAN['noise_seed'],int(reference=='new'),channel])
            pool = []; patterns = Counter(); baseline = matching.Moments(models,(2,2,2))
            for first in range(0,len(weights),matching.PLAN['chunk_size']):
                last = min(first+matching.PLAN['chunk_size'],len(weights)); size = last-first
                detector = prepared['geometry']['detector'][first:last]
                coordinate,light = prepared['coordinate'][first:last],prepared['light'][first:last]
                raw = matching.form_signals(detector,coordinate,light,rng.standard_normal((size,2,2,16)),.1,tables)
                matches = matching.match_signals(raw); matched = matches['sum_first']
                topology,keep = previous.observable_topology(matched,matches['share_first'])
                eligible = posterior.single.eligible_hits(matched,raw,.1)
                crossing = material.pixel_foil_bounds(matched,.1)['crosses']
                direction,columns = previous.decomposition.nominal_measurements(matched,tables)
                kinetic = previous.decomposition.corrected_energy(matched['energy'],columns,tables)
                labels,residual = previous.pixel.evaluate(matched,kinetic,direction,energy.nominal_parent(size))
                classified = np.stack([labels,labels*(abs(residual)<=.4)[:,None,None]],axis=1)
                baseline.add(weights[first:last],classified)
                ancestry = energy.association_group(matched,energy.strip_ancestry(detector,coordinate,light,.1))
                use = np.isin(matched['multiplicity'],[2,3]) & ~matched['ambiguous'] & crossing.any(axis=1) & ~eligible.any(axis=1)
                for row in np.flatnonzero(use):
                    hits = np.flatnonzero(matched['detector'][row]==0); pixels = matched['pixel'][row,hits]
                    integral = bool(np.all(abs(pixels-np.rint(pixels))<1e-10))
                    complete = integral and all(set(np.flatnonzero(raw[row,0,side]))==set(np.rint(pixels[:,side]).astype(int)) for side in (0,1))
                    repeated = tuple(len(set(pixels[:,side]))<len(hits) for side in (0,1))
                    patterns[str((len(hits),integral,complete,bool(keep[row,1]),*repeated))] += 1
                    pool.append(dict(index=first+int(row),weights=weights[first+row],observed=raw[row],
                        matched={k:v[row] for k,v in matched.items()},baseline_labels=classified[row],
                        baseline_residual_MeV=residual[row],baseline_kinetic_MeV=kinetic[row],nominal_direction=direction[row],
                        topology=int(topology[row]),policy_agreement=bool(keep[row,1]),d1_hits=hits,integer_pixels=integral,
                        complete_d1_support=complete,ancestry=int(ancestry[row]),
                        truth_kinetic_MeV=source['kinetic'][first+row],truth_direction=source['directions'][first+row],
                        truth_origins_mm=source['origins'][first+row],truth_depth=source['depth'][first+row]))
                if last % 50000 == 0:
                    print(reference,channel,last,'observable pool',len(pool),flush=True)
            sums,squared = weights.sum(axis=0),(weights*weights).sum(axis=0)
            expected = np.asarray(frozen['results'][reference]['summary']['classification']['value'])
            selection = [0] if channel==0 else list(range(1,6))
            if not np.allclose(baseline.finish(sums,squared)['value'],expected[selection,0],atol=1e-12,rtol=0):
                raise ArithmeticError('Complete baseline does not reproduce before extracting the pool')
            results[reference][str(channel)] = dict(pool=pool,patterns=dict(patterns),generated_weight=sums,
                squared_generated_weight=squared,provenance=source['provenance'])
            journal.put(identity,results[reference][str(channel)])
    state = parent.parent.plain(dict(parent_checksum=frozen['checksum'],results=results,source_sha256=sources,
        extraction_source=Path(__file__).read_text(),extraction_source_sha256=parent.sha256(Path(__file__)),
        stopping=metadata2[parent.KEY]['stopping'],source_configuration=frozen['source_configuration'],evidence=frozen['evidence'],
        started_at=journal.manifest['created_at'],completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=journal.elapsed(),attempt_started_at=started,attempt_elapsed_seconds=time.monotonic()-begin,
        checkpoint_manifest=journal.manifest))
    state = write_checked(output,state)
    journal.close()
    print('Observable pool complete:',state['checksum'],flush=True)


def reduce_group(group, inferred, frozen, models):
    pool = group['pool']; n = len(pool); weights = np.asarray([row['weights'] for row in pool])
    sums = np.asarray(group['generated_weight']); squared = np.asarray(group['squared_generated_weight'])
    matched = {key:np.asarray([row['matched'][key] for row in pool]) for key in pool[0]['matched']}
    baseline = np.asarray([row['baseline_kinetic_MeV'] for row in pool])
    direction = np.asarray([row['nominal_direction'] for row in pool])
    old_labels = np.asarray([row['baseline_labels'] for row in pool])
    old_residual = np.asarray([row['baseline_residual_MeV'] for row in pool])
    nominal = energy.nominal_parent(n)
    eligible = np.zeros(n, bool); reasons = Counter()
    classification = np.zeros((n, len(CASES), 2, 2, 2)); residual = np.zeros((n, len(CASES)))
    failures = np.zeros(len(CASES)-1, int)
    for case in range(len(CASES)):
        kinetic = baseline.copy()
        for index, row in enumerate(pool):
            hits, reason = scope(row)
            if case == 0:
                reasons[reason] += 1; eligible[index] = len(hits) > 0
            if case and len(hits):
                result = inferred[str(row['index'])][posterior.VARIANTS[case-1]]
                if result['valid']:
                    kinetic[index, hits] = result['mean']
                else:
                    failures[case-1] += 1
        labels, r = previous.pixel.evaluate(matched, kinetic, direction, nominal)
        classification[:, case, 0] = labels
        classification[:, case, 1] = labels*(abs(r) <= .4)[:, None, None]
        residual[:, case] = r
    if (not np.array_equal(classification[:, 0], old_labels)
            or not np.allclose(residual[:, 0], old_residual, atol=1e-10, rtol=0)):
        raise ArithmeticError('Frozen pool baseline no longer reproduces')
    if not np.array_equal(classification[~eligible], np.broadcast_to(old_labels[~eligible, None], classification[~eligible].shape)):
        raise ArithmeticError('An out-of-scope event changed')
    change = matching.Moments(len(models), (len(CASES), 2, 2, 2))
    change.add(weights, classification-classification[:, :1])
    paired = change.finish(sums, squared)
    baseline_full = np.asarray(frozen['classification']['value'])[models, 1]
    # Cases remain the second dimension in all published arrays.
    full = baseline_full[:, None]+np.asarray(paired['value'])
    if np.min(full) < -1e-12 or np.max(full) > 1+1e-12:
        raise ArithmeticError('Updated acceptance outside [0,1]')
    full = np.clip(full, 0., 1.)
    ancestry = np.eye(4)[[row['ancestry'] for row in pool]]
    audit = matching.Moments(len(models), (len(CASES), 2, 2, 4, 2))
    audit.add(weights, (classification-classification[:, :1])[..., None, :]*ancestry[:, None, None, None, :, None])
    ancestry_change = audit.finish(sums, squared)
    ancestry_full = np.asarray(frozen['ancestry']['value'])[models, 1:2]+ancestry_change['value']
    category = previous.pixel.state_codes(old_labels[:, 0])-1
    changes = matching.Moments(len(models), (4, len(CASES), len(previous.METRICS)))
    values = np.zeros((n, 4, len(CASES), len(previous.METRICS)))
    for bucket in range(4):
        use = category == bucket
        if not np.isfinite(residual[use]).all():
            raise ArithmeticError('A fixed-cohort event has a nonfinite residual')
        r = residual[use]; keep = abs(r) <= .4
        statistic = np.stack([r, r*r, r < -1., r > 1., keep,
                              keep.astype(float)-keep[:, :1]], axis=-1)
        values[use, bucket] = statistic-statistic[:, :1]
    changes.add(weights, values)
    moment_change = changes.finish(sums, squared)
    cohort = frozen['residual_partition']
    cohort_weight = np.asarray(cohort['cohort_per_generated'])[models]
    old_moments = np.asarray(cohort['cohort_moments'])[models, :, 1]
    conditional = old_moments[:, :, None]+np.divide(moment_change['value'], cohort_weight[..., None, None],
        out=np.zeros_like(moment_change['value']), where=cohort_weight[..., None, None] > 0)
    if np.min(conditional[..., 1]) < -1e-10:
        raise ArithmeticError('Updated second moment is negative')
    diagnostics = dict(eligible_events=int(eligible.sum()), pool_events=n, reasons=dict(reasons),
        failures=failures, maximum_energy_control_difference_MeV=0., above_20keV=0,
        rank_fallback_rays=0, minimum_main_effective_rays=None)
    support = []
    for estimates in inferred.values():
        if all(value['valid'] for value in estimates.values()):
            main = np.asarray(estimates['main']['mean'])
            delta = max(float(np.max(abs(np.asarray(estimates[key]['mean'])-main)))
                        for key in ('coarse', 'alternate', 'high_charge_order'))
            diagnostics['maximum_energy_control_difference_MeV'] = max(diagnostics['maximum_energy_control_difference_MeV'], delta)
            diagnostics['above_20keV'] += delta > PLAN['numerical_scale_MeV']
        support.append(estimates['main']['effective_rays'])
        diagnostics['rank_fallback_rays'] += sum(v['rank_fallback_rays'] for v in estimates.values())
    if support:
        diagnostics['minimum_main_effective_rays'] = min(support)
    if not np.allclose(ancestry_full.sum(axis=-2), full, atol=1e-12, rtol=0):
        raise ArithmeticError('Updated ancestry does not reproduce acceptance')
    if not np.allclose(full[:, 0], baseline_full, atol=1e-12, rtol=0):
        raise ArithmeticError('The previous full Flat baseline changed')
    return dict(classification=full, paired_change=paired, ancestry=ancestry_full,
        cohort_per_generated=cohort_weight, cohort_moments=conditional,
        cohort_paired_change_per_generated=moment_change, diagnostics=diagnostics)


_ENGINE = None


def initialize_worker(stopping):
    global _ENGINE
    _ENGINE = posterior.ClusterPosterior(previous.response.RangeTables(stopping), QUADRATURE)


def infer_observables(pixels, calibrated):
    return _ENGINE.estimate(0, np.asarray(pixels), np.asarray(calibrated))


def run(pool_path, output, workers=4, max_clusters=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    if not 1 <= workers <= 4:
        raise ValueError('Use one to four workers with one math thread each')
    if max_clusters is not None and max_clusters < 1:
        raise ValueError('The batch limit must be positive')
    if any(os.environ.get(key) != '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')):
        raise ValueError('Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before running')
    pool = checked_state(pool_path); frozen = previous.load_cached()
    if pool['parent_checksum'] != frozen['checksum']:
        raise ValueError('Pool and previous response do not match')
    sources = {**frozen['source_sha256'], **frozen['frozen_source_sha256']}
    for name,sha in sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen source changed: '+name)
    paths = [Path(__file__), Path(posterior.__file__)]
    source_sha = {str(path.relative_to(parent.ROOT)):parent.sha256(path) for path in paths}
    journal = Checkpoint(str(output)+'.progress',dict(mode='inference',version=VERSION,plan=PLAN,
        pool_checksum=pool['checksum'],parent_checksum=frozen['checksum'],source_sha256=source_sha,
        frozen_source_sha256=sources))
    tasks = []
    for reference in ('old','new'):
        for channel in ('0','2'):
            for row in pool['results'][reference][channel]['pool']:
                hits, _ = scope(row)
                if len(hits):
                    tasks.append(((reference,channel,str(row['index'])),
                        np.asarray(row['matched']['pixel'])[hits], np.asarray(row['observed'])[0]))
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    inferred = {ref:{channel:{} for channel in ('0','2')} for ref in ('old','new')}
    remaining = []
    for task in tasks:
        identity = task[0]; saved = journal.get(identity)
        if saved is None:
            remaining.append(task)
        else:
            ref,channel,index = identity
            inferred[ref][channel][index] = saved
    resumed = len(tasks)-len(remaining)
    batch = remaining if max_clusters is None else remaining[:max_clusters]
    print('Observable eligible clusters:',len(tasks),'resumed:',resumed,'this batch:',len(batch),'workers:',workers,flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(pool['stopping'],)) as executor:
        pending = {executor.submit(infer_observables,pixels,observed):identity for identity,pixels,observed in batch}
        for count,future in enumerate(as_completed(pending),1):
            ref,channel,index = pending[future]
            estimate = future.result()
            journal.put((ref,channel,index),estimate)
            inferred[ref][channel][index] = estimate
            if count % 20 == 0 or count == len(batch):
                print(resumed+count,'/',len(tasks),'conditional clusters checkpointed',flush=True)
    for name,sha in {**sources,**source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Source changed during integration')
    if len(batch) < len(remaining):
        print('Batch saved; rerun the same command to resume.',flush=True)
        journal.close()
        return
    results = {}
    for reference in ('old','new'):
        groups = [reduce_group(pool['results'][reference][channel], inferred[reference][channel],
            frozen['results'][reference]['summary'], [0] if channel == '0' else list(range(1,6))) for channel in ('0','2')]
        fields = [name for name in groups[0] if name != 'diagnostics']
        summary = {name:previous.combine([group[name] for group in groups]) for name in fields}
        results[reference] = dict(summary=summary, diagnostics={channel:group['diagnostics'] for channel,group in zip(('0','2'),groups)},
            inferred={channel:{index:rows[index] for index in sorted(rows,key=int)} for channel,rows in inferred[reference].items()})
    for name,sha in {**sources,**source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Source changed during integration')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, cases=CASES, parent_checksum=frozen['checksum'],
        input_pool=pool, input_pool_checksum=pool['checksum'], input_pool_results_sha256=parent.digest(pool['results']),
        source_sha256=source_sha, source_snapshots={str(p.relative_to(parent.ROOT)):p.read_text() for p in paths},
        frozen_source_sha256=sources, source_configuration=frozen['source_configuration'], evidence=frozen['evidence'],
        results=results, interpreter=sys.executable, workers=workers,
        checkpoint_manifest=journal.manifest,resumed_clusters=resumed,newly_computed_clusters=len(batch),
        runtime_environment={key:os.environ.get(key,'') for key in ('LD_PRELOAD','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent.ROOT,text=True).strip(),
        started_at=journal.manifest['created_at'], completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=journal.elapsed(),attempt_started_at=started,attempt_elapsed_seconds=time.monotonic()-begin))
    state = write_checked(output,state)
    journal.close()
    print('Cluster study complete:',state['completed_at'],flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    return checked_notebook(notebook)[KEY]


def checked_notebook(notebook):
    metadata = json.loads(Path(notebook).read_text())['metadata']
    state = metadata[KEY]
    if parent.digest({k:v for k,v in state.items() if k!='checksum'}) != state['checksum']:
        raise ValueError('Cluster cache checksum changed')
    return metadata


def validate_cached(state, frozen):
    if state['parent_checksum'] != frozen['checksum']:
        raise ValueError('Cluster parent changed')
    for reference in ('old','new'):
        full = np.asarray(state['results'][reference]['summary']['classification'])
        base = np.asarray(frozen['results'][reference]['summary']['classification']['value'])[:,1]
        change = np.asarray(state['results'][reference]['summary']['paired_change']['value'])
        if not np.allclose(full,base[:,None]+change,atol=1e-12,rtol=0):
            raise ArithmeticError('Sparse acceptance update does not close')
        ancestry = np.asarray(state['results'][reference]['summary']['ancestry'])
        if not np.allclose(ancestry.sum(axis=-2),full,atol=1e-12,rtol=0):
            raise ArithmeticError('Sparse ancestry update does not close')
        part = state['results'][reference]['summary']
        gamma = np.asarray(part['cohort_per_generated'])
        old = np.asarray(frozen['results'][reference]['summary']['residual_partition']['cohort_moments'])[:,:,1]
        delta = np.asarray(part['cohort_paired_change_per_generated']['value'])
        if not np.allclose(np.asarray(part['cohort_moments'])*gamma[...,None,None],
                           old[:,:,None]*gamma[...,None,None]+delta,atol=1e-10,rtol=0):
            raise ArithmeticError('Sparse cohort moments do not close')


def plot_acceptance(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(len(CASES))
    for row,model in enumerate((0,5)):
        own=0 if model==0 else 1
        for reference,style in (('old','-'),('new','--')):
            values=np.asarray(state['results'][reference]['summary']['classification'])[model]
            for mult,color in ((0,'#246a9a'),(1,'#b74b3a')):
                for col,label in ((0,own),(1,1-own)):
                    axes[row,col].plot(x,100*values[:,1,mult,label],style,color=color,marker='o',label=f'm={mult+2}, {reference}')
        for col in (0,1):
            ax=axes[row,col];ax.set_xticks(x,['Previous Flat','Coarse','Main','Alt. geometry','Charge order 8'],rotation=18)
            ax.set_ylabel('Acceptance per generated channel / %');ax.grid(alpha=.2)
            ax.set_title(f'27({chr(97+2*row+col)}) alpha{row}, '+('own label' if col==0 else 'wrong label'))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Conditional shared-strip response; full re-selection with +/-400 keV control window')
    return fig


def plot_cohort(state):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    x=np.arange(len(CASES))
    for row,(model,bucket) in enumerate(((0,0),(5,1))):
        for reference,style in (('old','-'),('new','--')):
            value=np.asarray(state['results'][reference]['summary']['cohort_moments'])[model,bucket]
            axes[row,0].plot(x,1000*np.sqrt(np.maximum(0.,value[:,1])),style,marker='o',label=reference)
            axes[row,1].plot(x,100*value[:,4],style,marker='o',label=reference)
        for col in (0,1):
            ax=axes[row,col];ax.set_xticks(x,['Previous Flat','Coarse','Main','Alt. geometry','Charge order 8'],rotation=18)
            ax.set_title(f'28({chr(97+2*row+col)}) alpha{row}, fixed original m=2 own-label cohort');ax.grid(alpha=.2)
        axes[row,0].set_ylabel('Energy residual RMS / keV')
        axes[row,1].set_ylabel('Retention within +/-400 keV / %')
    axes[0,0].legend()
    fig.suptitle('Fixed original events and labels; only observable eligible shared-strip clusters change')
    return fig


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--pool',type=Path)
    action.add_argument('--extract-pool',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-clusters',type=int,help='Checkpoint at most this many new clusters, then stop; rerun to resume')
    args=parser.parse_args()
    if args.extract_pool:
        extract_pool(args.extract_pool)
    else:
        if args.output is None:parser.error('--pool requires --output')
        run(args.pool,args.output,args.workers,args.max_clusters)
