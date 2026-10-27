"""Audit residual ancestry and observable sum/share identifiability at 162 keV.

The explicit CLI replays frozen reactions and noise. Imports and plots only
read completed notebook caches. Truth supports partition the results after
reconstruction; they never decide the estimator or the ambiguity controls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_pixel_study as pixel

posterior, material = pixel.posterior, pixel.material
decomposition, energy, matching, parent, response = pixel.decomposition, pixel.energy, pixel.matching, pixel.parent, pixel.response
KEY = 'codex_laursen_162_topology_study'
VERSION = '2026-10-27-v1-ancestry-and-observable-ambiguity'
PROFILE = 'split_sum_first'
CASES = ('baseline', 'flat_2048', 'flat_refined_low_support')
CONTROLS = ('retain_all', 'policy_agreement', 'single_candidate')
ISOLATION = ('no_foil_crossing_hit', 'crossing_without_eligible_hit', 'eligible_isolated_hit')
TOPOLOGY = ('single_candidate', 'multiple_policy_agreement', 'policy_disagreement', 'alternative_unresolved')
METRICS = ('mean_MeV', 'second_MeV2', 'negative_tail1', 'positive_tail1', 'retention400', 'paired_retention400')
REFINEMENT = replace(pixel.QUADRATURE, main_power=16, coarse_power=14, noise_order=24, batch_hits=1)
PLAN = dict(profile=PROFILE, sharing_width_mm=.100, noise_seed=pixel.PLAN['noise_seed'],
    chunk_size=pixel.PLAN['chunk_size'], windows=['none', 'w400'], cases=CASES,
    fixed_cohort='Baseline accepted category, before energy window; same events and labels in all three energy cases',
    residual_groups=list(energy.GROUPS), isolation_groups=ISOLATION,
    topology_groups=TOPOLOGY, controls=CONTROLS,
    agreement='Same unordered detector/pixel/energy hit tuples for sum-first and share-first; both resolved, at most three hits',
    single_candidate='No detector has multiple retained minimum-complexity interpretations in the frozen candidate family',
    deferred='Control-withheld accepted events retain their original labels in a separate deferred array; never disappear from accounting',
    refinement=dict(trigger='Main 2048-point evidence absent or effective geometry rays below 8, observable hits only',
        configuration=asdict(REFINEMENT), comparison_tolerance_MeV=.005,
        estimator='65536-point Flat T mean with 24-point noise quadrature; 16384 and independent 65536 controls',
        unresolved='Report any >5 keV difference, do not remove the hit or declare precision calibration'),
    counterexample='One particle sharing on one face and two particles summing on the other can give the same occupied-strip mean charges and noise law',
    limitations=['Ideal linear charge sharing, occupied-strip Gaussian noise and frozen hard thresholds only',
        'Candidate agreement is a diagnostic control, not a measured probability of correct matching',
        'Signal-level counterexamples do not assert equal three-alpha kinematic priors or equal integrated evidence',
        'No pulse-shape or resolved timing information is modeled',
        'No new reactions, hardware calibration, absolute branching ratio or weak-component limit'])


def canonical_hits(matched):
    """Order observable hits, so storage-slot permutations cannot change agreement."""
    detector = np.asarray(matched['detector'])
    value = np.concatenate([detector[..., None], matched['pixel'], matched['energy'][..., None]], axis=-1)
    value = np.where((detector >= 0)[..., None], value, -1.)
    order = np.lexsort(tuple(value[..., k] for k in (3, 2, 1, 0)), axis=1)
    return np.take_along_axis(value, order[..., None], axis=1)


def observable_topology(primary, alternative):
    """Only matched observables enter; no ancestry, true count, or energy sum."""
    resolved = (~primary['ambiguous'] & ~alternative['ambiguous']
                & (primary['multiplicity'] <= 3) & (alternative['multiplicity'] <= 3))
    agree = resolved & (primary['multiplicity'] == alternative['multiplicity'])
    agree &= np.all(np.isclose(canonical_hits(primary), canonical_hits(alternative), atol=1e-10, rtol=0), axis=(1, 2))
    unique = ~primary['multiple'] & ~primary['ambiguous'] & (primary['multiplicity'] <= 3)
    if np.any(unique & ~agree):
        raise ArithmeticError('A single retained interpretation changed with policy')
    group = np.full(len(agree), 3, int)
    group[resolved & ~agree] = 2
    group[agree] = 1
    group[unique] = 0
    return group, np.column_stack([np.ones(len(agree), bool), agree, unique])


def isolation_group(matched, eligible):
    crossing = material.pixel_foil_bounds(matched, PLAN['sharing_width_mm'])['crosses'].any(axis=1)
    return np.where(eligible.any(axis=1), 2, np.where(crossing, 1, 0))


class ResidualPartition:
    """Weighted raw moments, with exact reconstruction of the unpartitioned cohort."""
    def __init__(self, models):
        self.weight = np.zeros((models, 4, 4, 3))
        self.weight2 = np.zeros_like(self.weight)
        self.count = np.zeros((4, 4, 3), int)
        shape = (*self.weight.shape, len(CASES), len(METRICS))
        self.first, self.w2first, self.w2second = (np.zeros(shape) for _ in range(3))
        self.excluded = np.zeros((models, 4))

    def add(self, weights, category, association, isolation, residuals):
        weights, residuals = np.asarray(weights), np.asarray(residuals)
        if residuals.shape != (len(weights), len(CASES)):
            raise ValueError('Expected three residual cases per event')
        common = np.isfinite(residuals).all(axis=1)
        for bucket in range(4):
            self.excluded[:, bucket] += weights[(category == bucket) & ~common].sum(axis=0)
            for ancestry in range(4):
                for observable in range(3):
                    use = (category == bucket) & (association == ancestry) & (isolation == observable) & common
                    w, r = weights[use], residuals[use]
                    keep = abs(r) <= .4
                    value = np.stack([r, r*r, r < -1., r > 1., keep,
                                      keep.astype(float)-keep[:, :1]], axis=-1).reshape(len(r), len(CASES)*len(METRICS))
                    key = (slice(None), bucket, ancestry, observable)
                    self.weight[key] += w.sum(axis=0)
                    self.weight2[key] += (w*w).sum(axis=0)
                    self.count[bucket, ancestry, observable] += len(r)
                    shape = (weights.shape[1], len(CASES), len(METRICS))
                    self.first[key] += (w.T@value).reshape(shape)
                    self.w2first[key] += ((w*w).T@value).reshape(shape)
                    self.w2second[key] += ((w*w).T@(value*value)).reshape(shape)

    def finish(self, generated):
        denominator = np.where(self.weight > 0, self.weight, 1.)[..., None, None]
        mean = self.first/denominator
        variance = self.w2second-2*mean*self.w2first+mean*mean*self.weight2[..., None, None]
        scale = abs(self.w2second)+abs(2*mean*self.w2first)+mean*mean*self.weight2[..., None, None]
        tolerance = 16*np.finfo(float).eps*scale
        if np.any(variance < -tolerance):
            raise ArithmeticError('Residual sampling variance is negative beyond roundoff')
        variance = np.where(abs(variance) <= tolerance, 0., variance)
        error = np.sqrt(np.maximum(0., variance))/denominator
        cohort_weight = self.weight.sum(axis=(2, 3))
        cohort_first = self.first.sum(axis=(2, 3))
        safe = np.where(cohort_weight > 0, cohort_weight, 1.)
        total_second = cohort_first[..., 1]
        contribution = np.divide(self.first[..., 1], total_second[:, :, None, None, :],
                                 out=np.zeros_like(self.first[..., 1]), where=total_second[:, :, None, None, :] > 0)
        return dict(valid=self.weight > 0, event_count=np.broadcast_to(self.count, self.weight.shape),
            per_generated=self.weight/generated[:, None, None, None],
            excluded_per_generated=self.excluded/generated[:, None],
            conditional_moments=mean, conditional_standard_error=error,
            raw_moments_per_generated=self.first/generated[:, None, None, None, None, None],
            fraction_of_cohort_second_moment=contribution,
            cohort_per_generated=cohort_weight/generated[:, None],
            cohort_moments=cohort_first/safe[..., None, None])


def signal_counterexamples(tables):
    """Detector-level observational equivalence, without a reaction-truth fit."""
    pairs = np.array([(charge, fraction) for charge in (1.5, 3., 5.) for fraction in (.2, .4, .7)])
    size = len(pairs)
    one_d = np.full((size, 3), -1, int); two_d = one_d.copy()
    one_d[:, 0] = 0; two_d[:, :2] = 0
    one_xy = np.zeros((size, 3, 2)); two_xy = one_xy.copy()
    half = PLAN['sharing_width_mm']/(2*response.PITCH)
    one_xy[:, 0, 0] = 8-half+2*half*pairs[:, 1]
    one_xy[:, 0, 1] = 9.5
    two_xy[:, :2, 0] = [7.5, 8.5]; two_xy[:, :2, 1] = 9.5
    one_q = np.zeros((size, 3)); two_q = one_q.copy()
    one_q[:, 0] = pairs[:, 0]
    # Use the forward function's floating-point fraction for exact charge bookkeeping.
    _, _, fraction = matching.matching.split_charge(one_xy[:, 0, 0], response.PITCH, PLAN['sharing_width_mm'])
    two_q[:, 0] = one_q[:, 0]*(1-fraction); two_q[:, 1] = one_q[:, 0]*fraction
    rng = np.random.default_rng(1622026102703)
    maximum_difference = 0.; zero_noise = None
    for draw in range(33):
        noise = np.zeros((size, 2, 2, 16)) if draw == 0 else rng.standard_normal((size, 2, 2, 16))
        one = matching.form_signals(one_d, one_xy, one_q, noise, PLAN['sharing_width_mm'], tables)
        two = matching.form_signals(two_d, two_xy, two_q, noise, PLAN['sharing_width_mm'], tables)
        maximum_difference = max(maximum_difference, float(abs(one-two).max()))
        if not np.array_equal(one, two):
            raise ArithmeticError('The signal-level counterexample is not identical')
        if draw == 0:
            zero_noise = one
    matches = matching.match_signals(zero_noise)
    return dict(charge_and_fraction=pairs, noise_draws=33, noise_seed=1622026102703,
        max_absolute_observed_difference_MeV=maximum_difference,
        observed_front7_front8_back9_MeV=zero_noise[:, 0, [0, 0, 1], [7, 8, 9]],
        policy_multiplicity={p: m['multiplicity'] for p, m in matches.items()},
        policy_ambiguous={p: m['ambiguous'] for p, m in matches.items()},
        physical_scope='Charges after material transport; no claim that both hypotheses have the same reaction prior or full-event evidence')


def refine_hits(engine, matched, observed, eligible, post):
    """A predeclared numerical trigger using only observable-hit inference."""
    indices = np.flatnonzero(~post['valid'][:, 1] | (post['effective_rays'][:, 1] < 8))
    refined = post['mean'][:, 1].copy()
    records = []
    if not len(indices):
        return refined, records
    event, hit = np.where(eligible)
    rows, hits = event[indices], hit[indices]
    detector = matched['detector'][rows, hits]
    pixels = np.rint(matched['pixel'][rows, hits]).astype(int)
    measured = observed[rows[:, None], detector[:, None], np.arange(2), pixels]
    high = engine.estimate(detector, pixels, measured, PLAN['sharing_width_mm'])
    if not high['valid'].all():
        raise ArithmeticError('Refined numerical controls have zero evidence')
    refined[indices] = high['mean'][:, 1]
    for k, index in enumerate(indices):
        differences = abs(high['mean'][k, [0, 3]]-high['mean'][k, 1])
        records.append(dict(chunk_event=int(rows[k]), hit=int(hits[k]), detector=int(detector[k]),
            pixel=pixels[k], calibrated_pair_MeV=measured[k],
            original_mean_MeV=post['mean'][index], original_effective_rays=post['effective_rays'][index],
            refined_mean_MeV=high['mean'][k], refined_effective_rays=high['effective_rays'][k],
            control_differences_MeV=differences,
            within_5keV_controls=bool(np.all(differences <= PLAN['refinement']['comparison_tolerance_MeV']))))
    return refined, records


def reduce_reference(source, tables, reference, channel, engine, fine_engine):
    prepared = matching.prepare(source, tables)
    detector = prepared['geometry']['detector']; weights = source['weights']; models = weights.shape[1]
    shapes = dict(classification=(len(CASES), 2, 2, 2),
        ancestry=(len(CASES), 2, 2, 4, 2),
        topology=(len(CASES), 2, 2, 4, 2),
        controlled=(len(CONTROLS), 2, 2, 2), deferred=(len(CONTROLS), 2, 2, 2),
        controlled_ancestry=(len(CONTROLS), 2, 2, 4, 2),
        paired_refinement_change=(2, 2, 2))
    moments = {name: matching.Moments(models, shape) for name, shape in shapes.items()}
    partition = ResidualPartition(models)
    records = []; raw_counts = dict(eligible_hits=0, refined_hits=0)
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first; w = weights[first:last]
        noise = rng.standard_normal((size, 2, 2, 16))
        raw = matching.form_signals(detector[first:last], prepared['coordinate'][first:last],
            prepared['light'][first:last], noise, PLAN['sharing_width_mm'], tables)
        matches = matching.match_signals(raw); matched = matches['sum_first']
        topology, keep = observable_topology(matched, matches['share_first'])
        inferred = pixel.infer_profiles({'primary': matched}, raw, PLAN['sharing_width_mm'], engine)['primary']
        eligible, post = inferred['eligible'], inferred['posterior']
        refined, local_records = refine_hits(fine_engine, matched, raw, eligible, post)
        for record in local_records:
            record['reference_event_index'] = first+record.pop('chunk_event')
        records.extend(local_records)
        raw_counts['eligible_hits'] += int(eligible.sum()); raw_counts['refined_hits'] += len(local_records)
        direction, columns = decomposition.nominal_measurements(matched, tables)
        baseline = decomposition.corrected_energy(matched['energy'], columns, tables)
        nominal = energy.nominal_parent(size)
        classification = np.zeros((size, len(CASES), 2, 2, 2))
        residuals = np.zeros((size, len(CASES)))
        for case in range(len(CASES)):
            kinetic = baseline.copy()
            if case:
                estimates = post['mean'][:, 1] if case == 1 else refined
                kinetic[eligible] = np.where(np.isfinite(estimates), estimates, baseline[eligible])
            labels, residual = pixel.evaluate(matched, kinetic, direction, nominal)
            classification[:, case, 0] = labels
            classification[:, case, 1] = labels*(abs(residual) <= .4)[:, None, None]
            residuals[:, case] = residual
        # Truth is introduced only after all estimators and controls have been fixed.
        ancestry = energy.strip_ancestry(detector[first:last], prepared['coordinate'][first:last],
            prepared['light'][first:last], PLAN['sharing_width_mm'])
        association = energy.association_group(matched, ancestry)
        association_mask = np.eye(4)[association]; topology_mask = np.eye(4)[topology]
        controlled = classification[:, 1:2]*keep[:, :, None, None, None]
        fields = dict(classification=classification,
            ancestry=classification[..., None, :]*association_mask[:, None, None, None, :, None],
            topology=classification[..., None, :]*topology_mask[:, None, None, None, :, None],
            controlled=controlled, deferred=classification[:, 1:2]-controlled,
            controlled_ancestry=controlled[..., None, :]*association_mask[:, None, None, None, :, None],
            paired_refinement_change=classification[:, 2]-classification[:, 1])
        for name, value in fields.items():
            moments[name].add(w, value)
        partition.add(w, pixel.state_codes(classification[:, 0, 0])-1,
                      association, isolation_group(matched, eligible), residuals)
        if last % 25000 == 0 or last == len(weights):
            print(reference, channel, last, '/', len(weights), 'topology events; refined hits', len(records), flush=True)
    sums, squared = weights.sum(axis=0), (weights*weights).sum(axis=0)
    summary = {name: moment.finish(sums, squared) for name, moment in moments.items()}
    summary['residual_partition'] = partition.finish(sums)
    summary['raw_counts'] = {key: np.full(models, value, int) for key, value in raw_counts.items()}
    return dict(summary=summary, refinement_records=records, provenance=source['provenance'])


def validate_summary(summary, previous):
    classified = np.asarray(summary['classification']['value'])
    frozen = np.asarray(previous['classification']['value'])[:, [0, 2]]
    if not np.allclose(classified[:, :2], frozen, atol=1e-12, rtol=0):
        raise ArithmeticError('Baseline and Flat acceptance no longer reproduce the frozen pixel cache')
    for key in ('ancestry', 'topology'):
        if not np.allclose(np.asarray(summary[key]['value']).sum(axis=-2), classified, atol=1e-12, rtol=0):
            raise ArithmeticError(key+' partition lost classified weight')
    controlled, deferred = (np.asarray(summary[key]['value']) for key in ('controlled', 'deferred'))
    if not np.allclose(controlled+deferred, classified[:, 1:2], atol=1e-12, rtol=0):
        raise ArithmeticError('Deferred events disappeared from control accounting')
    if not np.allclose(np.asarray(summary['controlled_ancestry']['value']).sum(axis=-2), controlled, atol=1e-12, rtol=0):
        raise ArithmeticError('Controlled ancestry lost weight')
    if np.any(np.diff(controlled, axis=1) > 1e-12):
        raise ArithmeticError('Observable controls are not nested')
    partition = summary['residual_partition']; old = previous['fixed_cohort']
    weight = np.asarray(partition['per_generated'])
    raw = np.asarray(partition['raw_moments_per_generated'])
    aggregate = np.asarray(partition['cohort_moments'])
    if not np.allclose(weight.sum(axis=(2, 3)), old['per_generated'], atol=1e-12, rtol=0):
        raise ArithmeticError('Residual cohort weights changed')
    if np.any(partition['excluded_per_generated']):
        raise ArithmeticError('Nonfinite residual excluded an accepted event')
    for target, metric, transform in [('mean_MeV', 0, lambda x:x), ('rms_MeV', 1, np.square),
                                      ('retention', 4, lambda x:x)]:
        expected = transform(np.asarray(old[target])[..., [0, 2]])
        if not np.allclose(aggregate[..., :2, metric], expected, atol=1e-10, rtol=1e-11):
            raise ArithmeticError('Partition does not reproduce fixed-cohort '+target)
    reconstructed = (np.asarray(partition['conditional_moments'])*weight[..., None, None]).sum(axis=(2, 3))
    if not np.allclose(reconstructed, raw.sum(axis=(2, 3)), atol=1e-12, rtol=0):
        raise ArithmeticError('Conditional residual moments do not close')


def combine(values):
    if isinstance(values[0], dict):
        return {key: combine([v[key] for v in values]) for key in values[0]}
    return np.concatenate(values, axis=0)


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    metadata2 = json.loads(parent.NOTEBOOK.read_text())['metadata']
    previous, frozen = metadata2[pixel.KEY], metadata2[parent.KEY]
    for state in (previous, frozen):
        if parent.digest({k:v for k,v in state.items() if k != 'checksum'}) != state['checksum']:
            raise ValueError('Parent cache checksum mismatch')
    frozen_sources = {**previous['frozen_source_sha256'], **previous['source_sha256']}
    source_sha = {str(Path(__file__).relative_to(parent.ROOT)): parent.sha256(Path(__file__))}
    for name, sha in frozen_sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed: '+name)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    tables = response.RangeTables(frozen['stopping'])
    counterexamples = signal_counterexamples(tables)
    metadata3 = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    amplitude = parent.parent.AmplitudeModel()
    engine = posterior.PixelPosterior(tables, pixel.QUADRATURE)
    fine_engine = posterior.PixelPosterior(tables, REFINEMENT)
    results = {}
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata3, reference, channel, amplitude),
                    tables, reference, channel, engine, fine_engine) for channel in (0, 2)]
        summary = combine([group['summary'] for group in groups])
        validate_summary(summary, previous['results'][reference]['summary'][PROFILE])
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups],
            refinement_records={str(channel): group['refinement_records'] for channel, group in zip((0, 2), groups)})
    for name, sha in {**frozen_sources, **source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Numerical source changed during replay')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, parent_checksum=previous['checksum'],
        cases=CASES, controls=CONTROLS, metrics=METRICS,
        source_sha256=source_sha, source_snapshots={str(Path(__file__).relative_to(parent.ROOT)): Path(__file__).read_text()},
        frozen_source_sha256=frozen_sources, source_configuration=previous['source_configuration'], evidence=previous['evidence'],
        counterexamples=counterexamples, results=results, interpreter=sys.executable,
        runtime_environment={key: os.environ.get(key, '') for key in ('LD_PRELOAD', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=parent.ROOT, text=True).strip(),
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin))
    state['checksum'] = parent.digest(state)
    with output.open('x') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
    print('Topology study complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k:v for k,v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Topology cache checksum changed')
    return state


def plot_residual_sources(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ['#427aa1', '#e09f3e', '#b14a46', '#6b5b95']
    names = ['One-to-one support', 'One-to-many', 'Overlap / many-to-one', 'Cross / mixed']
    for row, (model, bucket) in enumerate(((0, 0), (5, 1))):
        for col, reference in enumerate(('old', 'new')):
            ax = axes[row, col]
            data = state['results'][reference]['summary']['residual_partition']
            # Contributions sum to the cohort E[r^2], not to an average of group RMS.
            weight = np.asarray(data['cohort_per_generated'])[model, bucket]
            raw = np.asarray(data['raw_moments_per_generated'])[model, bucket].sum(axis=1)
            values = raw[:, :, 1]/weight
            bottom = np.zeros(len(CASES))
            for group, (color, name) in enumerate(zip(colors, names)):
                ax.bar(np.arange(len(CASES)), values[group], bottom=bottom, color=color, label=name)
                bottom += values[group]
            ax.set_xticks(np.arange(len(CASES)), ['Baseline', 'Flat 2048', 'Low-support refined'])
            ax.set_ylabel('Contribution to mean squared residual / MeV$^2$')
            ax.set_title(f'25({chr(97+row*2+col)})  alpha{row}, m=2 own-label cohort, {reference}')
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed baseline events and labels; ancestry is a diagnostic after reconstruction')
    return fig


def plot_ambiguity_controls(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(CONTROLS))
    for row, model in enumerate((0, 5)):
        own = 0 if model == 0 else 1
        for reference, style in (('old', '-'), ('new', '--')):
            data = np.asarray(state['results'][reference]['summary']['controlled']['value'])[model]
            for mult, color in ((0, '#246a9a'), (1, '#b74b3a')):
                for col, label in ((0, own), (1, 1-own)):
                    axes[row, col].plot(x, 100*data[:, 1, mult, label], style, color=color, marker='o',
                                       label=f'm={mult+2}, {reference}')
        for col in (0, 1):
            ax = axes[row, col]
            ax.set_xticks(x, ['Retain all', 'Policy agreement', 'Single candidate'])
            ax.set_ylabel('Retained acceptance per generated channel / %')
            ax.set_title(f'26({chr(97+row*2+col)})  alpha{row}, '+('own label' if col == 0 else 'wrong label'))
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Flat 2048 energy, +/-400 keV control window; withheld events remain a deferred category')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
