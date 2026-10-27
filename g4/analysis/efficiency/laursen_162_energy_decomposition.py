"""Truth interventions on a frozen, one-to-one strip-support cohort.

Only the explicit CLI replays reactions. Interventions never reselect or
reclassify events and are diagnostics, not an achievable detector response.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_energy_closure as energy

matching, parent, response = energy.matching, energy.parent, energy.response
KEY = 'codex_laursen_162_energy_decomposition'
VERSION = '2026-10-27-v1-frozen-cohort-decomposition'
PROFILES = ('sum_first', 'split_sum_first', 'split_share_first')
CASES = ('baseline', 'D', 'L', 'DL', 'E', 'DE', 'LE', 'DLE',
         'impact_direction', 'true_angle_mid_depth', 'true_missing')
COMPONENTS = ('observed', 'missing', 'parent_reference')
OBSERVED_EDGES = np.linspace(-.8, .8, 81)
MISSING_EDGES = np.linspace(-2., 2., 81)
PLAN = dict(noise_seed=matching.PLAN['noise_seed'], chunk_size=matching.PLAN['chunk_size'],
    cohort='Baseline nominal-beam selection, central channel labels, one-to-one strip support, before energy window',
    comparison='Same baseline labels and same events for every intervention; no re-selection or channel reclassification',
    D='True emitted LAB direction in momentum reconstruction only; correction columns are a separate switch',
    L='True B/C/Si columns including event depth and angle in inverse CSDA correction',
    E='True active-Si entrance energy replaces the matched strip energy; combined readout/matching diagnostic, not electronic noise alone',
    factorial='All eight combinations of D/L/E; conditional interventions, not independent additive variance components',
    impact_direction='True impact position viewed from nominal origin; baseline energy and correction columns',
    true_angle_mid_depth='True momentum direction and true-angle material columns, with depth fixed to one half',
    true_missing='Replace only the missing CM kinetic energy by truth after baseline reconstruction; diagnostic does not conserve reconstructed total momentum',
    frame='All reconstructed and true four-vectors use the same fixed nominal-beam CM boost',
    decomposition='DeltaE=observed error + missing error + true-sum minus nominal available energy; retain full covariance',
    window_half_width_MeV=.4, histogram_profile='split_sum_first',
    finite_policy='Conditional moments use the intersection of finite measured-energy inversions across all controls; report excluded weight',
    limitations=['Truth is unavailable in experiment', 'Frozen approximate geometry, electronics and reaction weights',
                 'Cohort conditioned on baseline selection and channel assignment',
                 'No external efficiency validation, response fit, optimal window or weak-branch limit'])


def truth_permutation(matched, ancestry):
    """Map accepted hit slots to distinct true particles; complete the m=2 slot."""
    n = len(matched['multiplicity'])
    multiplicity = matched['multiplicity']
    active = np.arange(3)[None, :] < multiplicity[:, None]
    if (np.any(~np.isin(multiplicity, (2, 3)))
            or not np.array_equal(active, matched['detector'] >= 0)
            or np.any(energy.association_group(matched, ancestry) != 0)):
        raise ValueError('Expected selected, contiguous, one-to-one m=2/3 hits')
    mapped = np.full((n, 3), -1, int)
    for hit in range(3):
        position = matched['pixel'][:, hit]
        rounded = np.rint(position)
        position = np.where(abs(position-rounded) < 1e-10, rounded, position)
        low = np.clip(np.floor(position).astype(int), 0, 15)
        high = np.clip(np.ceil(position).astype(int), 0, 15)
        d = np.maximum(matched['detector'][:, hit], 0)
        f = ancestry[np.arange(n), d, 0, low[:, 0]] | ancestry[np.arange(n), d, 0, high[:, 0]]
        b = ancestry[np.arange(n), d, 1, low[:, 1]] | ancestry[np.arange(n), d, 1, high[:, 1]]
        common = f & b
        for particle in range(3):
            mapped[active[:, hit] & (common == 1 << particle), hit] = particle
    double = multiplicity == 2
    mapped[double, 2] = 3-mapped[double, 0]-mapped[double, 1]
    if not np.all(np.sort(mapped, axis=1) == np.arange(3)):
        raise ArithmeticError('True-particle mapping is not a permutation')
    return mapped


def nominal_measurements(matched, tables):
    detector = matched['detector']; d = np.maximum(detector, 0)
    _, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
    position = response.CENTRES[d]+((matched['pixel'][:, :, 0]+.5)*response.PITCH-25.)[:, :, None]*horizontal[d]
    position += ((matched['pixel'][:, :, 1]+.5)*response.PITCH-25.)[:, :, None]*vertical
    direction = response.unit(position)
    columns = matching.audit.columns(direction, detector, np.full(len(detector), .5),
                                     response.Response(), tables, matching.PLAN['geometry'])
    return direction, columns


def corrected_energy(entering, columns, tables):
    boron, carbon, silicon = columns
    return tables.restore(tables.restore(tables.restore(entering, silicon, 'G4_Si'), carbon, 'G4_C'), boron, 'G4_B')


def reconstruct_finite(kinetic, direction, multiplicity, nominal):
    """Never turn invalid measured inversions into accepted finite energies."""
    active = np.arange(3)[None, :] < multiplicity[:, None]
    valid = np.all(~active | (np.isfinite(kinetic) & (kinetic >= 0)), axis=1)
    result = np.full_like(kinetic, np.nan)
    for m in (2, 3):
        rows = np.flatnonzero(valid & (multiplicity == m))
        if len(rows):
            safe = np.where(active[rows], kinetic[rows], 0.)
            result[rows] = response.reconstruct(safe, direction[rows],
                nominal['parent_energy'][rows], nominal['parent_momentum'][rows], np.arange(3) < m)[0]
    return result


def error_components(reconstructed_cm, true_cm, multiplicity, available):
    active = np.arange(3)[None, :] < multiplicity[:, None]
    difference = reconstructed_cm-true_cm
    return np.column_stack([np.sum(np.where(active, difference, 0.), axis=1),
                            np.sum(np.where(~active, difference, 0.), axis=1),
                            true_cm.sum(axis=1)-available])


def interventions(matched, source, truth_columns, entering, impact, mapping, tables):
    n = len(mapping); rows = np.arange(n)[:, None]
    true_direction = source['directions'][rows, mapping]
    columns = tuple(value[rows, mapping] for value in truth_columns)
    true_entering = entering[rows, mapping]
    direction, nominal_columns = nominal_measurements(matched, tables)
    nominal = energy.nominal_parent(n)
    available = energy.available_energy(**nominal)
    multiplicity = matched['multiplicity']
    residual = np.empty((n, len(CASES)))
    baseline = None
    for case in range(8):
        corrected = corrected_energy(true_entering if case & 4 else matched['energy'],
                                     columns if case & 2 else nominal_columns, tables)
        cm = reconstruct_finite(corrected, true_direction if case & 1 else direction, multiplicity, nominal)
        residual[:, case] = cm.sum(axis=1)-available
        if case == 0:
            baseline = cm
    measured = corrected_energy(matched['energy'], nominal_columns, tables)
    cm = reconstruct_finite(measured, response.unit(impact[rows, mapping]), multiplicity, nominal)
    residual[:, 8] = cm.sum(axis=1)-available
    mid_columns = matching.audit.columns(true_direction, matched['detector'], np.full(n, .5),
        response.Response(), tables, matching.PLAN['geometry'])
    cm = reconstruct_finite(corrected_energy(matched['energy'], mid_columns, tables), true_direction, multiplicity, nominal)
    residual[:, 9] = cm.sum(axis=1)-available
    kinetic = source['kinetic'][rows, mapping]
    true_momentum = np.sqrt(kinetic*(kinetic+2*response.MASS))[..., None]*true_direction
    true_total, _ = response.boost(kinetic+response.MASS, true_momentum,
                                  (-nominal['parent_momentum']/nominal['parent_energy'][:, None])[:, None, :])
    true_cm = true_total-response.MASS
    components = error_components(baseline, true_cm, multiplicity, available)
    residual[:, 10] = components[:, 0]+components[:, 2]
    if not np.allclose(components.sum(axis=1), residual[:, 0], atol=1e-10, rtol=0):
        raise ArithmeticError('Observed/missing/parent energy errors do not close')
    return residual, components


class ConditionalMoments:
    """Self-normalized importance estimates on fixed, common-finite cohorts."""
    def __init__(self, models, histogram=False):
        shape = (models, 4, len(CASES))
        self.weight = np.zeros((models, 4)); self.weight2 = np.zeros_like(self.weight)
        self.count = np.zeros(4, int)
        self.invalid = np.zeros(shape); self.excluded = np.zeros_like(self.weight)
        self.first, self.second, self.w2first, self.w2second = [np.zeros(shape) for _ in range(4)]
        self.keep, self.w2keep, self.w2difference, self.w2difference2 = [np.zeros(shape) for _ in range(4)]
        self.component = np.zeros((models, 4, 3)); self.cross = np.zeros((models, 4, 3, 3))
        self.histogram = np.zeros((models, 2, len(OBSERVED_EDGES)+1, len(MISSING_EDGES)+1)) if histogram else None

    def add(self, weights, category, residual, components):
        finite = np.isfinite(residual)
        common = finite.all(axis=1)
        for bucket in range(4):
            mask = category == bucket
            self.invalid[:, bucket] += weights[mask].T @ (~finite[mask])
            self.excluded[:, bucket] += weights[mask & ~common].sum(axis=0)
            rows = np.flatnonzero(mask & common)
            w, r, c = weights[rows], residual[rows], components[rows]
            w2 = w*w
            self.weight[:, bucket] += w.sum(axis=0)
            self.weight2[:, bucket] += w2.sum(axis=0)
            self.count[bucket] += len(rows)
            for dest, value in ((self.first, r), (self.second, r*r)):
                dest[:, bucket] += w.T @ value
            self.w2first[:, bucket] += w2.T @ r
            self.w2second[:, bucket] += w2.T @ (r*r)
            keep = (abs(r) <= PLAN['window_half_width_MeV']).astype(float)
            difference = keep-keep[:, :1]
            self.keep[:, bucket] += w.T @ keep
            self.w2keep[:, bucket] += w2.T @ keep
            self.w2difference[:, bucket] += w2.T @ difference
            self.w2difference2[:, bucket] += w2.T @ (difference*difference)
            self.component[:, bucket] += w.T @ c
            self.cross[:, bucket] += np.einsum('nm,ni,nj->mij', w, c, c)
            if self.histogram is not None and bucket < 2:
                x = np.searchsorted(OBSERVED_EDGES, c[:, 0], side='right')
                y = np.searchsorted(MISSING_EDGES, c[:, 1], side='right')
                index = x*(len(MISSING_EDGES)+1)+y
                for model in range(w.shape[1]):
                    self.histogram[model, bucket] += np.bincount(index, weights=w[:, model],
                        minlength=self.histogram.shape[-2]*self.histogram.shape[-1]).reshape(self.histogram.shape[-2:])

    def finish(self, generated_weight):
        denominator = np.where(self.weight > 0, self.weight, 1.)
        d = denominator[:, :, None]
        mean = self.first/d; mean2 = self.second/d
        keep = self.keep/d; difference = keep-keep[:, :, :1]
        w2 = self.weight2[:, :, None]
        error = np.sqrt(np.maximum(0., self.w2second-2*mean*self.w2first+mean*mean*w2))/d
        keep_error = np.sqrt(np.maximum(0., (1-2*keep)*self.w2keep+keep*keep*w2))/d
        difference_error = np.sqrt(np.maximum(0., self.w2difference2-2*difference*self.w2difference+difference*difference*w2))/d
        component = self.component/d
        covariance = self.cross/d[:, :, :, None]-component[:, :, :, None]*component[:, :, None, :]
        variance = np.maximum(0., mean2-mean*mean)
        if (not np.allclose(component.sum(axis=-1), mean[:, :, 0], atol=1e-10, rtol=0)
                or not np.allclose(covariance.sum(axis=(-1, -2)), variance[:, :, 0], atol=1e-10, rtol=0)):
            raise ArithmeticError('Component covariance does not reconstruct total variance')
        result = dict(valid=self.weight > 0, event_count=np.broadcast_to(self.count, self.weight.shape),
            common_per_generated=self.weight/generated_weight[:, None],
            excluded_per_generated=self.excluded/generated_weight[:, None],
            invalid_per_generated=self.invalid/generated_weight[:, None, None],
            effective_sample_size=np.divide(self.weight*self.weight, self.weight2, out=np.zeros_like(self.weight), where=self.weight2 > 0),
            mean_MeV=mean, mean_standard_error_MeV=error, std_MeV=np.sqrt(variance), rms_MeV=np.sqrt(np.maximum(0., mean2)),
            retention=keep, retention_standard_error=keep_error,
            paired_retention_change=difference, paired_retention_standard_error=difference_error,
            component_mean_MeV=component, component_covariance_MeV2=covariance)
        if self.histogram is not None:
            histogram = self.histogram/denominator[:, :2, None, None]
            if not np.allclose(histogram.sum(axis=(-1, -2)), result['valid'][:, :2], atol=1e-12):
                raise ArithmeticError('Component histogram lost weight, including tails')
            result['component_histogram'] = histogram
        return result


def reduce_reference(source, tables, reference_name, channel):
    prepared = matching.prepare(source, tables)
    detector = prepared['geometry']['detector']
    columns = matching.audit.columns(source['directions'], detector, source['depth'],
        response.Response(), tables, matching.PLAN['geometry'])
    b, c, si = columns
    entering = tables.remove(tables.remove(tables.remove(source['kinetic'], b, 'G4_B'), c, 'G4_C'), si, 'G4_Si')
    distance = np.where(detector >= 0, prepared['geometry']['distance'], 0.)
    impact = source['origins'][:, None, :]+distance[:, :, None]*source['directions']
    weights = source['weights']; models = weights.shape[1]
    rates = {p: {k: matching.Moments(models, (2, 2)) for k in ('baseline', 'w400')} for p in PROFILES}
    diagnostics = {p: ConditionalMoments(models, p == PLAN['histogram_profile']) for p in PROFILES}
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference_name == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first
        noise = rng.standard_normal((size, 2, 2, 16))
        matches = {}; ancestry = {}
        for width in (0., .100):
            observed = matching.form_signals(detector[first:last], prepared['coordinate'][first:last],
                prepared['light'][first:last], noise, width, tables)
            matches.update({('split_' if width else '')+k: v for k, v in matching.match_signals(observed).items()})
            ancestry[width] = energy.strip_ancestry(detector[first:last], prepared['coordinate'][first:last], prepared['light'][first:last], width)
        nominal = energy.nominal_parent(size)
        for p in PROFILES:
            matched = matches[p]; bits = ancestry[.100 if p.startswith('split_') else 0.]
            selected, _, labels, cm = matching.reconstruct(matched, nominal, tables)
            one_to_one = energy.association_group(matched, bits) == 0
            assigned = labels[:, :, 1]*one_to_one[:, None, None]
            base_residual = energy.energy_residual(cm, nominal)
            rates[p]['baseline'].add(weights[first:last], assigned)
            rates[p]['w400'].add(weights[first:last], assigned*(abs(base_residual) <= .4)[:, None, None])
            rows = np.flatnonzero(one_to_one & selected.any(axis=1))
            if not len(rows):
                continue
            subset = {k: v[rows] for k, v in matched.items()}
            mapping = truth_permutation(subset, bits[rows])
            absolute = first+rows
            source_part = {k: source[k][absolute] for k in ('kinetic', 'directions')}
            residual, components = interventions(subset, source_part, tuple(v[absolute] for v in columns),
                entering[absolute], impact[absolute], mapping, tables)
            if not np.allclose(residual[:, 0], base_residual[rows], atol=1e-10, rtol=0):
                raise ArithmeticError('Intervention baseline changed the frozen reconstruction')
            category = (subset['multiplicity']-2)*2+assigned[rows, :, 1].sum(axis=1).astype(int)
            diagnostics[p].add(weights[absolute], category, residual, components)
        if last % 50000 == 0 or last == len(weights):
            print(reference_name, channel, last, '/', len(weights), 'decomposition events complete', flush=True)
    sums, squared_sums = weights.sum(axis=0), (weights*weights).sum(axis=0)
    summary = {p: diagnostics[p].finish(sums) for p in PROFILES}
    for p in PROFILES:
        for name, moment in rates[p].items():
            summary[p][name+'_per_generated'] = moment.finish(sums, squared_sums)
    return dict(summary=summary, provenance=source['provenance'])


def validate_summary(summary, baseline):
    for p in PROFILES:
        row = summary[p]
        old = np.asarray(baseline[p]['nominal_beam']['association_classification']['value'])
        for name, window in (('baseline', 0), ('w400', 2)):
            if not np.allclose(row[name+'_per_generated']['value'], old[:, window, :, 0, :], atol=1e-12, rtol=0):
                raise ArithmeticError('Frozen cohort or window count changed')
        cohort = np.asarray(row['baseline_per_generated']['value']).reshape(6, 4)
        if not np.allclose(np.asarray(row['common_per_generated'])+np.asarray(row['excluded_per_generated']), cohort, atol=1e-12, rtol=0):
            raise ArithmeticError('Common-finite cohort does not conserve baseline weight')


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    baseline = energy.load_cached(); frozen = parent.load_cached()
    frozen_sources = {**baseline['frozen_source_sha256'], **baseline['source_sha256']}
    source_name = str(Path(__file__).relative_to(parent.ROOT)); source_sha = parent.sha256(__file__)
    for name, sha in frozen_sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed: '+name)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    print('Loading verified references and frozen amplitude model', flush=True)
    amplitude = parent.parent.AmplitudeModel(); tables = response.RangeTables(frozen['stopping'])
    results = {}
    def combine(values):
        if isinstance(values[0], dict):
            return {k: combine([v[k] for v in values]) for k in values[0]}
        return np.concatenate(values, axis=0)
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata, reference, channel, amplitude), tables, reference, channel)
                  for channel in (0, 2)]
        summary = combine([g['summary'] for g in groups])
        validate_summary(summary, baseline['results'][reference]['summary'])
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups])
    for name, sha in {**frozen_sources, source_name: source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Numerical sources changed during calculation')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, profiles=PROFILES, cases=CASES,
        components=COMPONENTS, category_axes=['m2_gs', 'm2_exc', 'm3_gs', 'm3_exc'],
        observed_edges=OBSERVED_EDGES, missing_edges=MISSING_EDGES,
        parent_checksum=baseline['checksum'], source_sha256={source_name: source_sha},
        source_snapshots={source_name: Path(__file__).read_text()}, frozen_source_sha256=frozen_sources,
        evidence=baseline['evidence'], source_configuration=baseline['source_configuration'],
        results=results, interpreter=sys.executable,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent.ROOT, text=True).strip(),
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin))
    state['checksum'] = parent.digest(state)
    with output.open('x') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
    print('Energy decomposition complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Energy-decomposition cache checksum changed')
    return state


def plot_controls(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for panel, (ax, (model, bucket)) in enumerate(zip(axes.flat, ((0, 0), (0, 2), (5, 1), (5, 3)))):
        for reference, style, marker in (('old', '-', 'o'), ('new', '--', 's')):
            row = state['results'][reference]['summary']['split_sum_first']
            ax.plot(np.arange(8), 1000*np.asarray(row['rms_MeV'])[model, bucket, :8],
                    linestyle=style, marker=marker, label=reference)
        ax.set_xticks(np.arange(8), ['Base', 'D', 'L', 'DL', 'E', 'DE', 'LE', 'DLE'])
        ax.set_yscale('log'); ax.grid(alpha=.2, which='both')
        ax.set_ylabel('Energy residual RMS / keV')
        ax.set_title(f'19({chr(97+panel)})  alpha{0 if model == 0 else 1}, m={bucket//2+2}, frozen {"gs" if model == 0 else "exc"} label')
    axes[0, 0].legend()
    fig.suptitle('Fixed one-to-one cohort; true direction D, material columns L, active-Si energy E')
    return fig


def plot_components(state):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    arrays = []
    for model, channel in ((0, 0), (5, 1)):
        for reference in ('old', 'new'):
            arrays.append(100*np.asarray(state['results'][reference]['summary']['split_sum_first']['component_histogram'])[model, channel])
    norm = LogNorm(vmin=.001, vmax=max(value.max() for value in arrays))
    for panel, (ax, data) in enumerate(zip(axes.flat, arrays)):
        mesh = ax.pcolormesh(state['observed_edges'], state['missing_edges'], np.ma.masked_less_equal(data[1:-1, 1:-1].T, 0.), norm=norm, cmap='viridis')
        x = np.asarray(state['observed_edges'])
        for offset, style in ((0., '-'), (-.4, '--'), (.4, '--')):
            ax.plot(x, -x+offset, color='tomato', linestyle=style, linewidth=1)
        ax.set(xlabel='Observed-alpha energy error / MeV', ylabel='Missing-alpha energy error / MeV',
            xlim=(x[0], x[-1]), ylim=(MISSING_EDGES[0], MISSING_EDGES[-1]),
            title=f'20({chr(97+panel)})  alpha{panel//2}, {("old", "new")[panel%2]}; tails {data.sum()-data[1:-1, 1:-1].sum():.3f}%')
    fig.colorbar(mesh, ax=axes, label='Conditional cohort % per 20 x 50 keV bin')
    fig.suptitle('Baseline m=2: observed and missing errors in the same nominal CM frame')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
