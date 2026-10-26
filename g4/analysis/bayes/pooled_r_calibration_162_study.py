"""Pooled independent MC templates and a prespecified alpha0-fraction grid.

The CLI uses verified cached histograms; it never runs Geant4. Import and
plotting are cache-only. Calibration, matched validation, and fixed-template
conditional trials are separate. The fit leaves all mixture fractions free.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import independent_library_162_study as prior

ROOT, NOTEBOOK = prior.ROOT, prior.NOTEBOOK
binning, banks, finite, convergence = prior.binning, prior.banks, prior.finite, prior.convergence
digest, plain = prior.digest, prior.plain
KEY = "codex_pooled_r_calibration_162_study"
VERSION = "2026-10-23-v1-pooled-mc-r-grid"
P = dict(bins=[30, 60], references=['old', 'new'], r_values=[.02, .05, .08],
         direct=[0., .002, .005, .02], calibration_direct=[0., .005, .02],
         data_budget=100000, generated=[400000, 2000000, 400000],
         calibration_repetitions=1000, validation_repetitions=500,
         seeds=dict(old=dict(calibration=1622026102311, validation=1622026102312,
                             conditional=1622026102313),
                    new=dict(calibration=1622026102314, validation=1622026102315,
                             conditional=1622026102316)),
         legacy_r05_conditional_seeds=dict(old=1622026102293, new=1622026102303),
         comparisons=96, primary='Per-binning maximum over both references and all declared r/d calibration points')


def fractions(direct, r):
    if not (0 <= direct <= 1 and 0 <= r <= 1):
        raise ValueError('Mixture fractions must lie in [0, 1]')
    return np.array([r*(1-direct), (1-r)*(1-direct), direct])


def case_id(bins, reference, r):
    return f'{bins}_{reference}_r{int(round(r*100)):02d}'


def cases():
    return [dict(bins=b, reference=reference, r=r)
            for b in P['bins'] for reference in P['references'] for r in P['r_values']]


def prepare_inputs(parent):
    if not parent.get('completed_at'):
        raise ValueError('Independent-library parent is incomplete')
    if digest({k: v for k, v in parent.items() if k not in ('completed_at', 'checkpoint_sha256')}) != parent['checkpoint_sha256']:
        raise ValueError('Independent-library parent checksum changed')
    source = parent['inputs']
    raw = [np.asarray(source['templates'][name]['raw_templates']) for name in P['references']]
    generated = [np.asarray(source['templates'][name]['generated']) for name in P['references']]
    for h, g in zip(raw, generated):
        if (h.shape != (3, 497) or g.shape != (3,) or np.any(h != np.floor(h))
                or np.any(g != np.floor(g)) or np.any(g <= 0)
                or np.any(h < 0) or np.any(h.sum(axis=1) > g)):
            raise ValueError('Pooled MC must use raw generated-event histograms')
    pooled = np.sum(raw, axis=0).astype(np.int64)
    total = np.sum(generated, axis=0).astype(np.int64)
    if not np.array_equal(total, P['generated']):
        raise ValueError('Pooled generated budgets differ from the declared plan')
    references = source['references']
    if any(np.asarray(p).shape != (3, 498) or np.any(np.asarray(p) < 0)
           or not np.allclose(np.asarray(p).sum(axis=1), 1, atol=1e-12, rtol=0)
           for p in references.values()):
        raise ValueError('Invalid held-out reference with rejection category')
    if not all(row['disjoint'] for row in source['summaries'].values()):
        raise ValueError('New reference overlaps its template')
    return plain(dict(raw_templates=pooled, generated=total, references=references,
        reference_generated=source['reference_generated'],
        member_sha256={name: digest(source['templates'][name]) for name in P['references']},
        reference_sha256={name: digest(p) for name, p in references.items()},
        accepted=pooled.sum(axis=1), rejected=total-pooled.sum(axis=1),
        provenance=dict(parent_checkpoint_sha256=parent['checkpoint_sha256'],
            parent_library_configuration_sha256=parent['configuration']['library_configuration_sha256'],
            new_library=source['provenance'],
            rule='Sum old/new template raw counts and generated budgets; neither held-out reference enters MC')))


def phase_settings(phase, reference):
    if reference not in P['references'] or phase not in ('calibration', 'validation', 'conditional'):
        raise ValueError('Unknown phase/reference')
    calibration = phase == 'calibration'
    return (P['seeds'][reference][phase],
            P['calibration_repetitions'] if calibration else P['validation_repetitions'],
            P['calibration_direct'] if calibration else P['direct'], phase != 'conditional')


def draw(inputs, case, direct, phase, index):
    seed, _, _, fluctuate = phase_settings(phase, case['reference'])
    truth = np.asarray(inputs['references'][case['reference']])
    r = case['r']
    identity = [int(round(r*1e6)), int(round(direct*1e6)), index]
    if phase == 'conditional' and r == .05:
        # Reuse the exact old/new r=.05 data, so the pooling contrast is paired.
        data = binning.data_draw(truth, direct, P['legacy_r05_conditional_seeds'][case['reference']], index)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, *identity, 0]))
        data = rng.multinomial(P['data_budget'], fractions(direct, r) @ truth)[:-1]
    generated = np.asarray(inputs['generated'])
    if fluctuate:
        mc = np.stack([np.random.default_rng(np.random.SeedSequence([seed, *identity, 1, c]))
                       .multinomial(int(g), p)[:-1]
                       for c, (g, p) in enumerate(zip(generated, truth))])
    else:
        mc = np.asarray(inputs['raw_templates'])
    return binning.coarsen(data, case['bins']), binning.coarsen(mc, case['bins']), generated, digest(data)


_INPUTS, _CASE, _PHASE = None, None, None


def _initialize(inputs, case, phase):
    global _INPUTS, _CASE, _PHASE
    _INPUTS, _CASE, _PHASE = inputs, case, phase


def _chunk(task):
    direct, first, last = task
    output = {method: [] for method in finite.METHODS}
    for index in range(first, last):
        try:
            data, mc, generated, data_sha = draw(_INPUTS, _CASE, direct, _PHASE, index)
            evaluated = banks.evaluate(data, mc, generated, direct)
            for method in output:
                evaluated[method]['paired_data_sha256'] = data_sha
                output[method].append(evaluated[method])
        except Exception as error:
            raise RuntimeError(f'Trial failed without omission: {_PHASE}, {_CASE}, d={direct}, index={index}') from error
    return plain(output)


def ensemble(inputs, case, path, phase, *, workers=8):
    seed, repetitions, points, fluctuate = phase_settings(phase, case['reference'])
    config = plain(dict(version=VERSION, case=case, phase=phase, seed=seed,
        repetitions=repetitions, direct_values=points, budget=P['data_budget'],
        fluctuate_mc=fluctuate, input_sha256=digest(inputs), chunk_size=25,
        legacy_conditional_seed=P['legacy_r05_conditional_seeds'][case['reference']]
            if phase == 'conditional' and case['r'] == .05 else None))
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else dict(configuration=config, chunks={})
    if state['configuration'] != config:
        raise ValueError('Pooled r-grid checkpoint configuration changed')
    tasks = {f'{int(round(d*1e6)):05d}:{first}': (d, first, min(first+25, repetitions))
             for d in points for first in range(0, repetitions, 25)}
    for key, chunk in state['chunks'].items():
        if (key not in tasks or set(chunk) != set(finite.METHODS)
                or any(len(rows) != tasks[key][2]-tasks[key][1] for rows in chunk.values())):
            raise ValueError('Invalid complete trial-chunk checkpoint')
    def save():
        temporary = path.with_suffix('.pending')
        temporary.write_text(json.dumps(plain(state), allow_nan=False))
        temporary.replace(path)
    start, last_notice = time.monotonic(), time.monotonic()
    pending = {key: task for key, task in tasks.items() if key not in state['chunks']}
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize, initargs=(inputs, case, phase)) as pool:
            futures = {pool.submit(_chunk, task): key for key, task in pending.items()}
            for future in as_completed(futures):
                try:
                    state['chunks'][futures[future]] = future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
                count = sum(len(chunk['joint']) for chunk in state['chunks'].values())
                if count % 500 == 0 or len(state['chunks']) == len(tasks):
                    save()
                if time.monotonic()-last_notice > 20:
                    print(f'{phase} {case}: {count}/{len(points)*repetitions} completed; {time.monotonic()-start:.1f}s', flush=True)
                    last_notice = time.monotonic()
    records = {}
    for d in points:
        key = f'{int(round(d*1e6)):05d}'
        records[key] = {}
        for method in finite.METHODS:
            rows = [row for first in range(0, repetitions, 25) for row in state['chunks'][f'{key}:{first}'][method]]
            records[key][method] = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    save()
    return dict(configuration=config, records=records)


def freeze(calibrations):
    """One cutoff per method/binning, enveloping both references and r grid."""
    expected = {case_id(**case) for case in cases()}
    if set(calibrations) != expected:
        raise ValueError('All declared references, r values, and binnings must calibrate')
    rank = finite.calibration.tolerance_rank(P['calibration_repetitions'], comparisons=P['comparisons'])
    local = {}
    for name, state in calibrations.items():
        config = state['configuration']
        case = config['case']
        if (config['phase'] != 'calibration' or case_id(**case) != name
                or config['seed'] != P['seeds'][case['reference']]['calibration']
                or config['repetitions'] != P['calibration_repetitions']
                or set(state['records']) != {'00000', '05000', '20000'}):
            raise ValueError('Only the complete prespecified calibration samples may set thresholds')
        local[name] = {}
        for method in finite.METHODS:
            upper = {}
            for key, point in state['records'].items():
                values = np.asarray(point[method]['q_upper'])
                if values.shape != (P['calibration_repetitions'],) or not np.isfinite(values).all():
                    raise ValueError('Incomplete/nonfinite calibration statistics')
                upper[key] = float(np.sort(values)[rank['rank']-1])
            q0 = np.asarray(state['records']['00000'][method]['q0'])
            if q0.shape != (P['calibration_repetitions'],) or not np.isfinite(q0).all():
                raise ValueError('Incomplete/nonfinite null calibration')
            local[name][method] = dict(q0=float(np.sort(q0)[rank['rank']-1]),
                                        q_upper=max(upper.values()), local_upper=upper)
    frozen = {}
    for bins in P['bins']:
        members = [case_id(**case) for case in cases() if case['bins'] == bins]
        cutoffs, active = {}, {}
        for method in finite.METHODS:
            cutoffs[method], active[method] = {}, {}
            for statistic in ('q0', 'q_upper'):
                member = max(members, key=lambda name: local[name][method][statistic])
                cutoffs[method][statistic] = local[member][method][statistic]
                active[method][statistic] = member
        frozen[str(bins)] = dict(cutoffs=cutoffs, active=active, tolerance=rank,
            scope='Both finite empirical references and r=.02/.05/.08; d=0/.005/.02; matched regenerated MC/data only')
    return dict(frozen=frozen, local=local, tolerance=rank)


def controls(inputs):
    output = []
    for case in cases():
        data, mc, generated, _ = draw(inputs, case, .005, 'conditional', 17)
        fixed = finite.fixed_model(data, mc, generated)
        a = banks.JointMultinomial(data, mc, generated, start=banks.accepted_weights(fixed.mode, fixed.efficiencies))
        b = banks.JointMultinomial(data, mc, generated, start=a.weights, tolerance=1e-5)
        c = banks.JointMultinomial(data, mc, generated, start=[.1, .8, .1])
        loss = np.ptp([a.loss, b.loss, c.loss])
        direct = np.ptp([a.mode[2], b.mode[2], c.mode[2]])
        profile = abs(a.statistic(.005)-b.statistic(.005))
        if loss > .002 or direct > 2e-5 or profile > .002:
            raise RuntimeError(f'Precision/multistart control failed: {case}')
        output.append(dict(case=case, trial=17, loss_range=loss, direct_range=direct,
                           profile_difference=profile, all_passed=True))
    return output


def verify_pairing(records):
    for phase in ('calibration', 'validation', 'conditional'):
        for reference in P['references']:
            for r in P['r_values']:
                binning.verify_pairing(records, [f'{phase}_{case_id(b, reference, r)}' for b in P['bins']])
    for case in cases():
        name = case_id(**case)
        hashes = [records[f'{phase}_{name}']['records']['05000']['joint']['paired_data_sha256']
                  for phase in ('calibration', 'validation', 'conditional')]
        if any(set(hashes[i]).intersection(hashes[j]) for i in range(3) for j in range(i)):
            raise ValueError('Calibration, validation, and conditional data overlap')


def run(*, notebook=NOTEBOOK, scratch, workers=8):
    start = time.monotonic()
    work = Path(scratch)
    work.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    if KEY in document['metadata']:
        raise ValueError('Preserve the existing completed pooled r-grid study')
    parent = document['metadata'][prior.KEY]
    inputs = prepare_inputs(parent)
    del document
    for relative, checksum in parent['configuration']['source_sha256'].items():
        if convergence.sha256(ROOT/relative) != checksum:
            raise ValueError(f'Parent implementation changed: {relative}')
    paths = [ROOT/relative for relative in parent['configuration']['source_sha256']]
    paths += [Path(__file__)]
    configuration = plain(dict(version=VERSION, protocol=P, analysis_workers=workers,
        started_at=datetime.now().astimezone().isoformat(),
        parent_checkpoint_sha256=parent['checkpoint_sha256'], input_sha256=digest(inputs),
        source_sha256={str(p.relative_to(ROOT)): convergence.sha256(p) for p in paths}))
    plan = work/'plan.json'
    if plan.exists():
        previous = json.loads(plan.read_text())
        configuration['started_at'] = previous['started_at']
        if previous != configuration:
            raise ValueError('Do not change the prespecified pooled r-grid analysis')
    plan.write_text(json.dumps(configuration, ensure_ascii=False, indent=1))
    snapshots = {str(p.relative_to(ROOT)): p.read_text() for p in paths}
    numerical = controls(inputs)
    print('Twelve noisy-data precision/multistart controls PASS', flush=True)
    records, calibrations = {}, {}
    for case in cases():
        name = case_id(**case)
        state = ensemble(inputs, case, work/f'calibration_{name}.json', 'calibration', workers=workers)
        records[f'calibration_{name}'] = calibrations[name] = state
        print(f'Calibration {name} COMPLETE', flush=True)
    calibrated = freeze(calibrations)
    # Freeze before either validation ensemble is examined.
    (work/'frozen.json').write_text(json.dumps(plain(calibrated), indent=1))
    print(f'Envelope thresholds FROZEN: {calibrated["frozen"]}', flush=True)
    matched, conditional, expected = {}, {}, {}
    for phase, summary in (('validation', matched), ('conditional', conditional)):
        for case in cases():
            name = case_id(**case)
            state = ensemble(inputs, case, work/f'{phase}_{name}.json', phase, workers=workers)
            records[f'{phase}_{name}'] = state
            summary[name] = finite.summarize(state, calibrated['frozen'][str(case['bins'])])
            print(f'{phase} {name} COMPLETE', flush=True)
    for case in cases():
        name = case_id(**case)
        expected[name] = {}
        for d in P['direct']:
            data = P['data_budget']*(fractions(d, case['r']) @ np.asarray(inputs['references'][case['reference']]))[:-1]
            expected[name][f'{int(round(d*1e6)):05d}'] = banks.evaluate(
                binning.coarsen(data, case['bins']), binning.coarsen(inputs['raw_templates'], case['bins']),
                np.asarray(inputs['generated']), d)
    verify_pairing(records)
    parent_records = convergence.unpack(parent['raw_records'])
    contrasts = dict(pooling_r05={}, binning={}, reference={})
    for bins in P['bins']:
        for reference in P['references']:
            merged = records[f'conditional_{case_id(bins, reference, .05)}']
            for template in P['references']:
                original = parent_records[f'conditional_{prior.case_id(bins, reference, template)}']
                contrasts['pooling_r05'][f'{bins}_{reference}_minus_{template}'] = prior.compare_estimates(original, merged, paired=True)
        for r in P['r_values']:
            contrasts['reference'][f'{bins}_r{int(round(r*100)):02d}'] = prior.compare_estimates(
                records[f'conditional_{case_id(bins, "old", r)}'], records[f'conditional_{case_id(bins, "new", r)}'], paired=False)
    for reference in P['references']:
        for r in P['r_values']:
            contrasts['binning'][f'{reference}_r{int(round(r*100)):02d}'] = prior.compare_estimates(
                records[f'conditional_{case_id(60, reference, r)}'], records[f'conditional_{case_id(30, reference, r)}'], paired=True)
    total = sum(len(point['joint']['map']) for state in records.values() for point in state['records'].values())
    if total != 84000:
        raise RuntimeError('Incomplete r-grid trial count')
    numerical_summary = dict(max_shape_gap=max(float(np.max(p['joint']['shape_gap'])) for s in records.values() for p in s['records'].values()),
        max_mode_decrement=max(float(np.max(p['joint']['mode_decrement'])) for s in records.values() for p in s['records'].values()),
        shape_restarts=sum(int(np.sum(p['joint']['shape_restart_count'])) for s in records.values() for p in s['records'].values()))
    if numerical_summary['max_shape_gap'] > 1.01e-4:
        raise RuntimeError('A fit missed its declared shape tolerance')
    for relative, checksum in configuration['source_sha256'].items():
        if convergence.sha256(ROOT/relative) != checksum:
            raise ValueError(f'Source changed during r-grid calibration: {relative}')
    state = plain(dict(configuration=configuration, inputs=inputs, calibration=calibrated,
        matched_summary=matched, conditional_summary=conditional, expected=expected, contrasts=contrasts,
        numerical_controls=numerical, numerical_summary=numerical_summary,
        pairing_checks=dict(binning=True, independent_phases=True, legacy_r05_data=True),
        counts=dict(fit_trial_cases=84000, calibration=36000, matched_validation=24000,
                    conditional=24000, distinct_data_draws=42000, legacy_data_draws=4000),
        raw_records=convergence.pack(records), source_snapshots=snapshots,
        elapsed_seconds=time.monotonic()-start,
        interpretation='Common envelope across finite empirical references and r grid; conditional fixed-MC validation is separate; no continuous-r or physical-branching guarantee'))
    state['checkpoint_sha256'] = digest(state)
    state['completed_at'] = datetime.now().astimezone().isoformat()
    prior.generator.save_metadata(notebook, KEY, state)
    print(f'Pooled r-grid study COMPLETE: {total} fit-trial cases; {state["elapsed_seconds"]:.1f}s', flush=True)
    return state


def plots(state, parent):
    """Cached-only figures: pooled means, inclusion, and weak-component power."""
    import matplotlib.pyplot as plt
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, bins in zip(axes, P['bins']):
        for offset, reference, color in ((-.07, 'old', 'C0'), (.07, 'new', 'C1')):
            rows = [parent['primary_summary'][prior.case_id(bins, reference, t)]['05000']['joint'] for t in P['references']]
            rows += [state['conditional_summary'][case_id(bins, reference, .05)]['05000']['joint']]
            ax.errorbar(np.arange(3)+offset, [100*r['mean'] for r in rows],
                yerr=[100*r['mcse'] for r in rows], fmt='o-', capsize=3, color=color, label=f'R{reference}; paired data')
        ax.axhline(.5, color='k', ls=':')
        ax.set(title=f'{bins} x {bins}; r=5%, true d=0.5%', xticks=[0, 1, 2],
               xticklabels=['Mold', 'Mnew', 'Mold + Mnew'], ylabel='Mean estimated fraction (%)', xlabel='Actual MC template')
        ax.legend(fontsize=8)
    figures.append(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), constrained_layout=True)
    labels = [f'R{ref}\nr={int(round(r*100))}%' for ref in P['references'] for r in P['r_values']]
    for row, (kind, title) in enumerate((('matched_summary', 'Regenerated data + MC'), ('conditional_summary', 'Fixed pooled MC'))):
        for column, bins in enumerate(P['bins']):
            ax = axes[row, column]
            lower = 88.
            for offset, key, label in ((-.17, '02000', 'd=0.2%'), (0, '05000', 'd=0.5%'), (.17, '20000', 'd=2%')):
                points = [state[kind][case_id(bins, ref, r)][key]['joint']['coverage'] for ref in P['references'] for r in P['r_values']]
                p = 100*np.array([point['rate'] for point in points])
                bounds = 100*np.array([point['wilson95'] for point in points])
                lower = min(lower, float(bounds.min())-2)
                ax.errorbar(np.arange(6)+offset, p, yerr=np.maximum(np.vstack([p-bounds[:, 0], bounds[:, 1]-p]), 0),
                            fmt='o', capsize=2, label=label)
            ax.axhline(95, color='k', ls=':')
            ax.set(title=f'{bins} x {bins}; {title}', xticks=np.arange(6), xticklabels=labels,
                   ylabel='Upper-limit inclusion (%)', ylim=(max(0, lower), 101))
            ax.tick_params(axis='x', labelsize=8)
            ax.legend(fontsize=8)
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, reference in zip(axes, P['references']):
        for bins, color in ((30, 'C0'), (60, 'C1')):
            for kind, style, label in (('matched_summary', '--', 'regenerated MC'), ('conditional_summary', '-', 'fixed pooled MC')):
                points = [state[kind][case_id(bins, reference, r)]['05000']['joint']['rejection'] for r in P['r_values']]
                p = 100*np.array([point['rate'] for point in points])
                bounds = 100*np.array([point['wilson95'] for point in points])
                ax.errorbar(100*np.array(P['r_values']), p, yerr=np.maximum(np.vstack([p-bounds[:, 0], bounds[:, 1]-p]), 0),
                    marker='o', ls=style, color=color, capsize=2, label=f'{bins} x {bins}; {label}')
        ax.set(title=f'R{reference}; true d=0.5%', xlabel='Sequential alpha0 fraction r (%)',
               ylabel='Rejection of d=0 (%)', xticks=[2, 5, 8])
        ax.legend(fontsize=8)
    figures.append(fig)
    return figures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    run(scratch=args.scratch, workers=args.workers)
