"""Replay an observable local pixel posterior, including acceptance migrations.

The explicit CLI reuses the frozen ROOT references. Importing and plotting only
read a completed cache. Truth ancestry is an audit after reconstruction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_pixel_posterior as posterior

material = posterior.material
decomposition, energy, matching, parent, response = material.decomposition, material.energy, material.matching, material.parent, material.response
KEY = 'codex_laursen_162_pixel_study'
VERSION = '2026-10-27-v1-local-observable-posterior'
PROFILES = material.PROFILES
CASES = ('baseline', *posterior.VARIANTS)
QUADRATURE = posterior.Quadrature()
PLAN = dict(noise_seed=matching.PLAN['noise_seed'], chunk_size=matching.PLAN['chunk_size'],
    quadrature=asdict(QUADRATURE), windows=['none', 'w400'], window_half_width_MeV=.4,
    eligibility='Nonambiguous m2/m3, integer pixels crossing foil, both pulses present, adjacent strips absent, neither strip reused',
    forward='Frozen CSDA B,C,Si and light calibration; independent occupied-strip Gaussian noise; absent shared neighbors are censored below threshold',
    latent='Full observed pixel rectangle extended by half sharing width; uniform 2 mm square beam in foil plane; uniform reaction depth',
    angular_prior='Isotropic emission, hence detector area Jacobian abs(n_D dot (r-o))/abs(r-o)^3',
    energy_priors=['Joint density constant in kinetic energy and solid angle',
                   'Joint density p(T)*(T+m) in kinetic energy and solid angle: single-particle momentum-volume control, not three-body phase space'],
    prior_domain='Direction-dependent single-alpha ceiling from fixed 162 keV parent and remaining invariant mass >= 2 alpha masses',
    estimator='Posterior mean emitted LAB kinetic energy; nominal momentum directions and missing-particle reconstruction retained',
    fallback='Zero numerical evidence keeps baseline energy; report every fallback, without rejecting that event for this reason',
    diagnostic_fields=['valid_hits', 'fallback_hits', 'sum_posterior_std_MeV', 'sum_effective_rays',
                       'sum_delta_from_main_MeV', 'sum_squared_delta_from_main_MeV2', 'hits_delta_from_main_above_20keV'],
    migration_states=['rejected', 'm2_gs', 'm2_exc', 'm3_gs', 'm3_exc'],
    fixed_cohort='All baseline accepted events, fixed baseline multiplicity and central gate label, before energy window; no truth ancestry restriction',
    limitations=['Single-particle priors are declared controls, not calibrated channel spectra',
                 'No event energy-sum fit, but single-particle physical support means closure is conditional on a kinematic prior',
                 'Nominal directions, isolated-hit approximation, ideal sharing and unrecognized overlaps remain',
                 'No new Geant4 reactions, hardware calibration or external validation'])


def evaluate(matched, kinetic, direction, nominal):
    """Repeat the frozen selection for changed energies, without truth inputs."""
    n = len(kinetic)
    labels = np.zeros((n, 2, 2))
    residual = np.full(n, np.nan)
    invalid = (matched['detector'] >= 0) & (~np.isfinite(kinetic) | (kinetic < 0))
    safe = np.where(np.isfinite(kinetic) & (kinetic >= 0), kinetic, 0.)
    config = matching.audit.Selection(plane='highest_energy', geometry=matching.PLAN['geometry'], hard_threshold=True)
    for multiplicity in (2, 3):
        rows = np.flatnonzero((matched['multiplicity'] == multiplicity) & ~matched['ambiguous'])
        if not len(rows):
            continue
        pattern = np.arange(3) < multiplicity
        cm, momentum = response.reconstruct(safe[rows], direction[rows], nominal['parent_energy'][rows], nominal['parent_momentum'][rows], pattern)
        selected = matching.audit.selection_masks(cm, momentum, invalid[rows], pattern, matched['detector'][rows], config)[:, -1]
        k = multiplicity-2
        ground = matching.audit.pair_relative_energy(momentum).min(axis=1) < matching.audit.GATES[1, k]
        labels[rows, k, 0] = selected & ground
        labels[rows, k, 1] = selected & ~ground
        valid = ~invalid[rows, :multiplicity].any(axis=1)
        residual[rows[valid]] = cm[valid].sum(axis=1)-energy.available_energy(**nominal)[rows[valid]]
    return labels, residual


def state_codes(labels):
    """Zero is rejection; the four exclusive accepted categories follow."""
    flat = np.asarray(labels).reshape(len(labels), 4)
    if np.any((flat != 0) & (flat != 1)) or np.any(flat.sum(axis=1) > 1):
        raise ValueError('Expected exclusive binary classification')
    return (flat*np.arange(1, 5)).sum(axis=1).astype(int)


class FixedCohort:
    def __init__(self, models):
        self.weight = np.zeros((models, 4)); self.weight2 = np.zeros_like(self.weight)
        self.count = np.zeros(4, int)
        shape = (models, 4, len(CASES), 5)
        self.first, self.w2first, self.w2second = [np.zeros(shape) for _ in range(3)]
        self.excluded = np.zeros((models, 4))

    def add(self, weights, category, residual):
        common = np.isfinite(residual).all(axis=1)
        for bucket in range(4):
            self.excluded[:, bucket] += weights[(category == bucket) & ~common].sum(axis=0)
            rows = np.flatnonzero((category == bucket) & common)
            w, r = weights[rows], residual[rows]; w2 = w*w
            keep = (abs(r) <= .4).astype(float)
            value = np.stack([r, r*r, keep, keep-keep[:, :1], abs(r) > 1.], axis=-1).reshape(len(rows), len(CASES)*5)
            self.weight[:, bucket] += w.sum(axis=0); self.weight2[:, bucket] += w2.sum(axis=0); self.count[bucket] += len(rows)
            shape = (w.shape[1], len(CASES), 5)
            self.first[:, bucket] += (w.T@value).reshape(shape)
            self.w2first[:, bucket] += (w2.T@value).reshape(shape)
            self.w2second[:, bucket] += (w2.T@(value*value)).reshape(shape)

    def finish(self, generated):
        denominator = np.where(self.weight > 0, self.weight, 1.)[..., None, None]
        value = self.first/denominator
        se = np.sqrt(np.maximum(0., self.w2second-2*value*self.w2first+value*value*self.weight2[..., None, None]))/denominator
        return dict(valid=self.weight > 0, event_count=np.broadcast_to(self.count, self.weight.shape),
            per_generated=self.weight/generated[:, None], excluded_per_generated=self.excluded/generated[:, None],
            mean_MeV=value[..., 0], mean_standard_error_MeV=se[..., 0], rms_MeV=np.sqrt(np.maximum(0., value[..., 1])),
            std_MeV=np.sqrt(np.maximum(0., value[..., 1]-value[..., 0]**2)),
            retention=value[..., 2], retention_standard_error=se[..., 2],
            paired_retention_change=value[..., 3], paired_retention_standard_error=se[..., 3],
            tail1_probability=value[..., 4], tail1_standard_error=se[..., 4])


def infer_profiles(matches, observed, width, engine):
    """Compute identical observable hits once when policies share them."""
    keys, eligibility = {}, {}
    for p, matched in matches.items():
        eligible = posterior.eligible_hits(matched, observed, width)
        rows, hits = np.where(eligible)
        keys[p] = np.column_stack([rows, matched['detector'][rows, hits], np.rint(matched['pixel'][rows, hits]).astype(int)])
        eligibility[p] = eligible
    all_keys = np.concatenate(list(keys.values()), axis=0)
    unique, inverse = np.unique(all_keys, axis=0, return_inverse=True)
    r, d, f, b = unique.T
    pair = np.column_stack([observed[r, d, 0, f], observed[r, d, 1, b]])
    inferred = engine.estimate(d, unique[:, 2:], pair, width)
    result = {}; offset = 0
    for p, eligible in eligibility.items():
        selection = inverse[offset:offset+len(keys[p])]; offset += len(keys[p])
        result[p] = dict(eligible=eligible, posterior={k: v[selection] for k, v in inferred.items()})
    return result


def reduce_reference(source, tables, reference, channel, engine):
    prepared = matching.prepare(source, tables)
    detector = prepared['geometry']['detector']; weights = source['weights']; models = weights.shape[1]
    shapes = dict(classification=(len(CASES), 2, 2, 2), paired_change=(len(CASES), 2, 2, 2),
                  migration=(len(CASES), 2, 5, 5), ancestry=(len(CASES), 2, 2, 4, 2),
                  eligibility=(2,), hit_diagnostics=(4, 7))
    moments = {p: {name: matching.Moments(models, shape) for name, shape in shapes.items()} for p in PROFILES}
    cohorts = {p: FixedCohort(models) for p in PROFILES}
    counts = {p: dict(eligible_hits=0, fallbacks=np.zeros(4, int), low_effective_rays=np.zeros(4, int)) for p in PROFILES}
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first; w = weights[first:last]
        noise = rng.standard_normal((size, 2, 2, 16)); matches, inference, ancestry = {}, {}, {}
        for width in (0., .100):
            raw = matching.form_signals(detector[first:last], prepared['coordinate'][first:last], prepared['light'][first:last], noise, width, tables)
            selected = {('split_' if width else '')+k: v for k, v in matching.match_signals(raw).items() if ('split_' if width else '')+k in PROFILES}
            matches.update(selected); inference.update(infer_profiles(selected, raw, width, engine))
            ancestry[width] = energy.strip_ancestry(detector[first:last], prepared['coordinate'][first:last], prepared['light'][first:last], width)
        nominal = energy.nominal_parent(size)
        for p in PROFILES:
            matched = matches[p]; inferred = inference[p]; eligible = inferred['eligible']; post = inferred['posterior']
            direction, columns = decomposition.nominal_measurements(matched, tables)
            baseline = decomposition.corrected_energy(matched['energy'], columns, tables)
            classification = np.zeros((size, len(CASES), 2, 2, 2)); residuals = np.zeros((size, len(CASES)))
            for case in range(len(CASES)):
                kinetic = baseline.copy()
                if case:
                    kinetic[eligible] = np.where(post['valid'][:, case-1], post['mean'][:, case-1], baseline[eligible])
                labels, residual = evaluate(matched, kinetic, direction, nominal)
                classification[:, case, 0] = labels
                classification[:, case, 1] = labels*(abs(residual) <= .4)[:, None, None]
                residuals[:, case] = residual
            # Frozen reference must agree before changing selection or labels.
            selected, _, old_labels, old_cm = matching.reconstruct(matched, nominal, tables)
            if not np.array_equal(old_labels[:, :, 1], classification[:, 0, 0]):
                raise ArithmeticError('Baseline classification differs from frozen matching')
            rows = np.flatnonzero(selected.any(axis=1))
            if not np.allclose(residuals[rows, 0], energy.energy_residual(old_cm, nominal)[rows], atol=1e-10, rtol=0):
                raise ArithmeticError('Baseline energy residual differs from frozen matching')
            moments[p]['classification'].add(w, classification)
            moments[p]['paired_change'].add(w, classification-classification[:, :1])
            migration = np.zeros((size, len(CASES), 2, 5, 5))
            for window in (0, 1):
                before = state_codes(classification[:, 0, window])
                for case in range(len(CASES)):
                    after = state_codes(classification[:, case, window])
                    migration[np.arange(size), case, window, before, after] = 1.
            moments[p]['migration'].add(w, migration)
            # Only this audit, never inference/selection, sees truth ancestry.
            association = energy.association_group(matched, ancestry[.100 if p.startswith('split_') else 0.])
            one_hot = (association[:, None] == np.arange(4)).astype(float)
            moments[p]['ancestry'].add(w, classification[..., None, :]*one_hot[:, None, None, None, :, None])
            cohorts[p].add(w[rows], state_codes(classification[rows, 0, 0])-1, residuals[rows])
            moments[p]['eligibility'].add(w, np.column_stack([eligible.any(axis=1), eligible.sum(axis=1)]))
            diag = np.zeros((size, 4, 7)); event, _ = np.where(eligible)
            both = post['valid'] & post['valid'][:, 1:2]
            delta = np.where(both, post['mean']-post['mean'][:, 1:2], 0.)
            values = np.stack([post['valid'], ~post['valid'], np.nan_to_num(post['std']), post['effective_rays'], delta, delta*delta, abs(delta) > .020], axis=-1)
            np.add.at(diag, event, values)
            moments[p]['hit_diagnostics'].add(w, diag)
            counts[p]['eligible_hits'] += len(event)
            counts[p]['fallbacks'] += (~post['valid']).sum(axis=0)
            counts[p]['low_effective_rays'] += (post['valid'] & (post['effective_rays'] < 8)).sum(axis=0)
        if last % 25000 == 0 or last == len(weights):
            print(reference, channel, last, '/', len(weights), 'pixel events complete', flush=True)
    sums, squared = weights.sum(axis=0), (weights*weights).sum(axis=0)
    summary = {p: {name: moment.finish(sums, squared) for name, moment in fields.items()} for p, fields in moments.items()}
    for p in PROFILES:
        summary[p]['fixed_cohort'] = cohorts[p].finish(sums)
        summary[p]['raw_counts'] = {k: np.broadcast_to(v, (models, *np.shape(v))) for k, v in counts[p].items()}
    return dict(summary=summary, provenance=source['provenance'])


def validate_summary(summary, previous):
    for p in PROFILES:
        row = summary[p]; value = np.asarray(row['classification']['value'])
        frozen = np.asarray(previous[p]['nominal_beam']['classification']['value'])[:, [0, 2]]
        if not np.allclose(value[:, 0], frozen, atol=1e-12, rtol=0):
            raise ArithmeticError('Full baseline counts no longer reproduce energy-closure cache')
        migration = np.asarray(row['migration']['value'])
        destination = migration.sum(axis=-2)[..., 1:].reshape(value.shape)
        source = migration.sum(axis=-1)[..., 1:].reshape(value.shape)
        if not np.allclose(destination, value, atol=1e-12, rtol=0) or not np.allclose(source, value[:, :1], atol=1e-12, rtol=0):
            raise ArithmeticError('Acceptance migration accounting failed')
        if not np.allclose(migration.sum(axis=(-1, -2)), 1., atol=1e-12, rtol=0):
            raise ArithmeticError('Migration lost generated weight')
        if not np.allclose(np.asarray(row['ancestry']['value']).sum(axis=-2), value, atol=1e-12, rtol=0):
            raise ArithmeticError('Ancestry audit lost classified weight')
        cohort = row['fixed_cohort']
        total = np.asarray(cohort['per_generated'])+np.asarray(cohort['excluded_per_generated'])
        if not np.allclose(total, value[:, 0, 0].reshape(-1, 4), atol=1e-12, rtol=0):
            raise ArithmeticError('Fixed-cohort accounting failed')
        if not np.any(cohort['excluded_per_generated']):
            kept = np.asarray(cohort['per_generated'])*np.asarray(cohort['retention'])[..., 0]
            if not np.allclose(kept, value[:, 0, 1].reshape(-1, 4), atol=1e-12, rtol=0):
                raise ArithmeticError('Fixed baseline window no longer closes')


def numerical_probe(tables):
    """Predeclared observable grid, independent of generated reaction truth."""
    records = np.array([(back, charge, offset, width) for back in (0, 7, 15)
                        for charge in (.25, .75, 1.5, 3., 5.5) for offset in (0., .06) for width in (0., .1)])
    result = {}
    for name, config in (('main', QUADRATURE), ('higher_geometry', replace(QUADRATURE, main_power=12)),
                         ('higher_noise', replace(QUADRATURE, noise_order=24))):
        engine = posterior.PixelPosterior(tables, config)
        estimates = np.empty((len(records), 4))
        for width in (0., .1):
            rows = np.flatnonzero(records[:, 3] == width)
            q = np.column_stack([records[rows, 1]+records[rows, 2]/2, records[rows, 1]-records[rows, 2]/2])
            calibrated = tables.materials['G4_Si']['inverse_light'](q)
            values = engine.estimate(np.zeros(len(rows), int), np.column_stack([np.zeros(len(rows), int), records[rows, 0].astype(int)]), calibrated, width)
            if not values['valid'].all():
                raise ArithmeticError('Predeclared numerical probe has unsupported observations')
            estimates[rows] = values['mean']
        result[name] = estimates
    return dict(columns=['back_strip', 'mean_visible_MeV', 'front_minus_back_visible_MeV', 'sharing_width_mm'],
                records=records, means_MeV=result)


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    metadata2 = json.loads(parent.NOTEBOOK.read_text())['metadata']
    previous, baseline, frozen = metadata2[material.KEY], metadata2[energy.KEY], metadata2[parent.KEY]
    for state in (previous, baseline, frozen):
        if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
            raise ValueError('Parent cache checksum mismatch')
    frozen_sources = {**previous['frozen_source_sha256'], **previous['source_sha256']}
    source_paths = [Path(__file__), Path(posterior.__file__)]
    source_sha = {str(path.relative_to(parent.ROOT)): parent.sha256(path) for path in source_paths}
    for name, sha in frozen_sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed: '+name)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    tables = response.RangeTables(frozen['stopping'])
    print('Checking predeclared observable-only numerical grid', flush=True)
    probe = numerical_probe(tables)
    print('Numerical grid complete; loading frozen reaction references', flush=True)
    metadata3 = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    amplitude = parent.parent.AmplitudeModel(); engine = posterior.PixelPosterior(tables, QUADRATURE)
    def combine(values):
        if isinstance(values[0], dict):
            return {key: combine([v[key] for v in values]) for key in values[0]}
        return np.concatenate(values, axis=0)
    results = {}
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata3, reference, channel, amplitude), tables, reference, channel, engine) for channel in (0, 2)]
        summary = combine([group['summary'] for group in groups])
        validate_summary(summary, baseline['results'][reference]['summary'])
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups])
    for name, sha in {**frozen_sources, **source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Numerical source changed during replay')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, cases=CASES, profiles=PROFILES,
        parent_checksum=previous['checksum'], baseline_checksum=baseline['checksum'],
        source_sha256=source_sha, source_snapshots={str(p.relative_to(parent.ROOT)): p.read_text() for p in source_paths},
        frozen_source_sha256=frozen_sources, source_configuration=previous['source_configuration'], evidence=previous['evidence'],
        numerical_probe=probe, results=results, interpreter=sys.executable,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent.ROOT, text=True).strip(),
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin))
    state['checksum'] = parent.digest(state)
    with output.open('x') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
    print('Pixel posterior study complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Pixel posterior cache checksum changed')
    return state


def plot_acceptance(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(3); cases = [0, 2, 3]
    for row, model in enumerate((0, 5)):
        own = 0 if model == 0 else 1
        for reference, style in (('old', '-'), ('new', '--')):
            data = np.asarray(state['results'][reference]['summary']['split_sum_first']['classification']['value'])[model]
            for window, color, name in ((0, '#246a9a', 'No E window'), (1, '#b74b3a', '+/-400 keV')):
                axes[row, 0].plot(x, 100*data[cases, window, 0, own], style, color=color, marker='o', label=name+', '+reference)
                axes[row, 1].plot(x, 100*data[cases, window, 0, 1-own], style, color=color, marker='o', label=name+', '+reference)
        for col in (0, 1):
            ax = axes[row, col]; ax.set_xticks(x, ['Baseline', 'Flat T', 'Momentum volume'])
            ax.set_title(f'23({chr(97+row*2+col)})  alpha{row}, m=2, '+('own channel' if col == 0 else 'wrong channel'))
            ax.set_ylabel('Acceptance per generated channel / %'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Full re-selection and channel assignment; ideal sharing, sum-first matching')
    return fig


def plot_fixed_cohort(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = [0, 1, 2, 4, 3]; x = np.arange(len(cases))
    for row, (model, bucket) in enumerate(((0, 0), (5, 1))):
        for reference, style in (('old', '-'), ('new', '--')):
            data = state['results'][reference]['summary']['split_sum_first']['fixed_cohort']
            axes[row, 0].plot(x, 1000*np.asarray(data['rms_MeV'])[model, bucket, cases], style, marker='o', label=reference)
            axes[row, 1].plot(x, 100*np.asarray(data['retention'])[model, bucket, cases], style, marker='o', label=reference)
        for col in (0, 1):
            ax = axes[row, col]; ax.set_xticks(x, ['Baseline', 'Flat 512', 'Flat 2048', 'Alt. 2048', 'Volume 2048'], rotation=15)
            ax.set_title(f'24({chr(97+row*2+col)})  alpha{row}, baseline m=2 own-channel cohort'); ax.grid(alpha=.2)
        axes[row, 0].set_ylabel('Energy residual RMS / keV')
        axes[row, 1].set_ylabel('Fixed-cohort retention within +/-400 keV / %')
    axes[0, 0].legend()
    fig.suptitle('Fixed baseline cohorts: energy change before any re-selection')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
