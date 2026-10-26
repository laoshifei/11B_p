"""Replay strip-level signal formation and observable-only matching controls.

The explicit CLI calculates; notebook imports only read completed caches.
Known truth is used to generate signals and to audit them after matching, never
as a matcher input. This is a declared model family, not the historical code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_selection_audit as audit
from analysis.efficiency import laursen_162_strip_matching as matching

parent, response = audit.parent, audit.response
KEY = 'codex_laursen_162_matching_study'
VERSION = '2026-10-26-v1-observable-strip-matching'
PROFILES = ('direct_only', 'sum_first', 'share_first', 'unique',
            'split_sum_first', 'split_share_first', 'split_unique')
DIAGNOSTICS = ('ambiguity_rejected', 'multiple_feasible_interpretations', 'unmatched_signal',
               'used_summing', 'used_sharing', 'wrong_pixel_in_selected_geometric_triple')
PLAN = dict(geometry='equation_4_1', plane='highest_energy', threshold_MeV=[.200, .350],
            matching_tolerance_MeV=.150, sharing_widths_mm=[0., .100],
            sharing_profile='Ideal linear charge-conserving split on either face; internal boundaries only',
            side_fwhm_MeV=float(.040*np.sqrt(2)),
            noise_seed=1622026102631, chunk_size=5000,
            energy='Sum visible charge, add independent per-channel Gaussian noise, hard-cut each side, then calibrate to deposited energy',
            priority='Maximum consumed signals, minimum non-direct grouping complexity, then explicit sum/share priority and energy residual',
            overflow='More than three reconstructed particles or any unresolved detector ambiguity rejects the event',
            wrong_pixel_diagnostic='Evaluated only without injected charge splitting; split-profile zeros are placeholders, not accuracy claims',
            omissions=['Noise-only triggers, background and TDC jitter', 'Front/back sharing asymmetry and extra interstrip loss',
                       'Arbitrary overlapping sharing clouds', 'Dead strips and per-strip calibration', 'Historical geometry and matcher'])


def prepare(source, tables):
    geometry = audit.trace(source['directions'], source['origins'], PLAN['geometry'])
    detector = geometry['detector']
    config = response.Response()
    b, c, si = audit.columns(source['directions'], detector, source['depth'], config, tables, PLAN['geometry'])
    entering = tables.remove(tables.remove(tables.remove(source['kinetic'], b, 'G4_B'), c, 'G4_C'), si, 'G4_Si')
    light = np.where(detector >= 0, tables.materials['G4_Si']['light'](entering), 0.)
    distance = np.where(detector >= 0, geometry['distance'], 0.)
    position = source['origins'][:, None, :] + distance[:, :, None]*source['directions']
    _, h, v = audit.basis(PLAN['geometry'])
    relative = position-response.CENTRES[np.maximum(detector, 0)]
    coordinate = np.stack([np.sum(relative*h[np.maximum(detector, 0)], axis=-1), relative@v], axis=-1)
    coordinate = (coordinate+response.HALF_SIZE)/response.PITCH
    coordinate[detector < 0] = 0.
    if np.any((coordinate < -1e-10) | (coordinate > 16+1e-10)):
        raise ArithmeticError('Trace intersection outside active rectangle')
    coordinate = np.clip(coordinate, 0., np.nextafter(16., 0.))
    parent_momentum = (np.sqrt(source['kinetic']*(source['kinetic']+2*response.MASS))[..., None]*source['directions']).sum(axis=1)
    return dict(geometry=geometry, coordinate=coordinate, light=light,
                parent_momentum=parent_momentum, parent_energy=(source['kinetic']+response.MASS).sum(axis=1))


def form_signals(detector, coordinate, light, noise, width, tables):
    """One noise draw per occupied physical strip; threshold before matching."""
    n = len(detector)
    signal = np.zeros((n, 2, 2, 16))
    for particle in range(3):
        rows = np.flatnonzero((detector[:, particle] >= 0) & (light[:, particle] > 0))
        d = detector[rows, particle]
        for side in (0, 1):
            first, second, fraction = matching.split_charge(coordinate[rows, particle, side], response.PITCH, width)
            charge = light[rows, particle]
            np.add.at(signal, (rows, d, side, first), charge*(1-fraction))
            np.add.at(signal, (rows, d, side, second), charge*fraction)
    if not np.allclose(signal.sum(axis=(1, 3)), light.sum(axis=1)[:, None], atol=1e-12):
        raise ArithmeticError('Charge was not conserved on both faces')
    measured = np.maximum(0., signal+PLAN['side_fwhm_MeV']/2.354820045*noise)
    threshold = np.maximum(PLAN['threshold_MeV'], response.Response().adc_MeV)
    keep = (signal > 0) & (measured >= threshold[None, :, None, None])
    calibrated = tables.materials['G4_Si']['inverse_light'](measured)
    if not np.isfinite(calibrated[keep]).all():
        raise ArithmeticError('Channel calibration outside stopping table')
    return np.where(keep, calibrated, 0.)


def allocate(size):
    return dict(energy=np.zeros((size, 3)), pixel=np.zeros((size, 3, 2)), detector=np.full((size, 3), -1, int),
                multiplicity=np.zeros(size, int), ambiguous=np.zeros(size, bool),
                multiple=np.zeros(size, bool), unmatched=np.zeros(size, bool),
                summed=np.zeros(size, bool), shared=np.zeros(size, bool))


def match_signals(observed, direct=False):
    """This boundary accepts no particle truth or reconstructed kinematics."""
    profiles = ('direct_only',) if direct else matching.POLICIES
    result = {p: allocate(len(observed)) for p in profiles}
    empty = {p: matching.Match((), False, 1, 0) for p in matching.POLICIES}
    for event, detectors in enumerate(observed):
        matches = []
        for detector in detectors:
            indices = [np.flatnonzero(face > 0) for face in detector]
            if not len(indices[0]) and not len(indices[1]):
                matches.append(empty)
            else:
                inputs = [np.column_stack([index, face[index]]) for index, face in zip(indices, detector)]
                matches.append(matching.match(*inputs, PLAN['matching_tolerance_MeV'], direct_only=direct))
        for p in profiles:
            chosen = [m['sum_first' if direct else p] for m in matches]
            row = result[p]
            row['ambiguous'][event] = any(m.ambiguous for m in chosen)
            row['multiple'][event] = any(m.feasible > 1 for m in chosen)
            row['unmatched'][event] = any(m.unmatched for m in chosen)
            row['summed'][event] = any(m.summed for m in chosen)
            row['shared'][event] = any(m.shared for m in chosen)
            hits = [(d, *h) for d, m in enumerate(chosen) for h in m.hits]
            row['multiplicity'][event] = len(hits)
            if len(hits) <= 3 and not row['ambiguous'][event]:
                for k, (d, energy, f, b) in enumerate(hits):
                    row['energy'][event, k] = energy
                    row['detector'][event, k] = d
                    row['pixel'][event, k] = (f, b)
    return result


def reconstruct(matched, prepared, tables):
    n = len(matched['energy'])
    selected = np.zeros((n, 2))
    flow = np.zeros((n, 2, len(audit.STAGES)))
    classified = np.zeros((n, 2, 3, 2))
    cm_output = np.zeros((n, 3))
    detector = matched['detector']
    _, h, v = audit.basis(PLAN['geometry'])
    d = np.maximum(detector, 0)
    positions = response.CENTRES[d]+((matched['pixel'][:, :, 0]+.5)*response.PITCH-25.)[:, :, None]*h[d]
    positions += ((matched['pixel'][:, :, 1]+.5)*response.PITCH-25.)[:, :, None]*v
    direction = response.unit(positions)
    b, c, si = audit.columns(direction, detector, np.full(n, .5), response.Response(), tables, PLAN['geometry'])
    corrected = tables.restore(tables.restore(tables.restore(matched['energy'], si, 'G4_Si'), c, 'G4_C'), b, 'G4_B')
    invalid = (detector >= 0) & ~np.isfinite(corrected)
    corrected = np.nan_to_num(corrected, nan=0.)
    config = audit.Selection(plane='highest_energy', geometry=PLAN['geometry'], hard_threshold=True)
    for multiplicity in (2, 3):
        rows = np.flatnonzero((matched['multiplicity'] == multiplicity) & ~matched['ambiguous'])
        if not len(rows):
            continue
        pattern = np.arange(3) < multiplicity
        cm, momentum = response.reconstruct(corrected[rows], direction[rows], prepared['parent_energy'][rows],
                                             prepared['parent_momentum'][rows], pattern)
        stages = audit.selection_masks(cm, momentum, invalid[rows], pattern, detector[rows], config)
        k = multiplicity-2
        flow[rows, k] = stages
        selected[rows, k] = stages[:, -1]
        ground = audit.pair_relative_energy(momentum).min(axis=1)[:, None] < audit.GATES[:, k]
        classified[rows, k, :, 0] = stages[:, -1, None]*ground
        classified[rows, k, :, 1] = stages[:, -1, None]*~ground
        cm_output[rows] = cm
    if not np.allclose(classified.sum(axis=-1), selected[:, :, None]):
        raise ArithmeticError('Channel labels do not conserve accepted events')
    return selected, flow, classified, cm_output


class Moments:
    """Streaming normalized-importance moments, including paired differences."""
    def __init__(self, models, shape):
        self.shape = (models, *shape)
        self.num = np.zeros((models, int(np.prod(shape))))
        self.w2x = np.zeros_like(self.num)
        self.w2x2 = np.zeros_like(self.num)

    def add(self, weights, value):
        value = np.asarray(value, float).reshape((len(weights), -1))
        self.num += weights.T @ value
        squared = weights*weights
        self.w2x += squared.T @ value
        self.w2x2 += squared.T @ (value*value)

    def finish(self, sums, squared_sums):
        mean = self.num/sums[:, None]
        variance = self.w2x2-2*mean*self.w2x+mean*mean*squared_sums[:, None]
        error = np.sqrt(np.maximum(0., variance))/sums[:, None]
        return dict(value=mean.reshape(self.shape), standard_error=error.reshape(self.shape))


def reduce_reference(source, tables, reference_name, channel):
    prepared = prepare(source, tables)
    weights = source['weights']; models = weights.shape[1]
    shapes = dict(efficiency=(2,), cutflow=(2, len(audit.STAGES)), classification=(2, 3, 2),
                  geometric_to_matched=(4, 6), diagnostics=(len(DIAGNOSTICS),), paired_minus_sum_first=(2,))
    accum = {p: {name: Moments(models, shape) for name, shape in shapes.items()} for p in PROFILES}
    histograms = {p: np.zeros((models, 2, len(parent.ENERGY_EDGES)-1)) for p in PROFILES}
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference_name == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first
        data = {key: value[first:last] for key, value in prepared.items() if key != 'geometry'}
        geometry = {key: value[first:last] for key, value in prepared['geometry'].items()}
        w = weights[first:last]
        noise = rng.standard_normal((size, 2, 2, 16))
        raw = form_signals(geometry['detector'], data['coordinate'], data['light'], noise, 0., tables)
        matched = match_signals(raw, direct=True)
        matched.update(match_signals(raw))
        split = form_signals(geometry['detector'], data['coordinate'], data['light'], noise, .100, tables)
        matched.update({'split_'+k: v for k, v in match_signals(split).items()})
        geometric = (geometry['detector'] >= 0).sum(axis=1)
        evaluated = {p: reconstruct(m, data, tables) for p, m in matched.items()}
        for p, m in matched.items():
            selected, flow, classified, cm = evaluated[p]
            bucket = np.where(m['ambiguous'], 5, np.minimum(m['multiplicity'], 4))
            migration = np.zeros((size, 4, 6)); migration[np.arange(size), geometric, bucket] = 1.
            wrong = np.zeros(size, bool)
            if not p.startswith('split_'):
                for i in np.flatnonzero((geometric == 3) & (selected[:, 1] > 0)):
                    truth = Counter((int(d), float(x), float(y)) for d, (x, y) in zip(geometry['detector'][i], geometry['pixel'][i]))
                    observed = Counter((int(d), float(x), float(y)) for d, (x, y) in zip(m['detector'][i], m['pixel'][i]))
                    wrong[i] = truth != observed
            diagnostic = np.column_stack([m['ambiguous'], m['multiple'], m['unmatched'], m['summed'], m['shared'], wrong])
            fields = dict(efficiency=selected, cutflow=flow, classification=classified,
                          geometric_to_matched=migration, diagnostics=diagnostic,
                          paired_minus_sum_first=selected-evaluated['sum_first'][0])
            for name, value in fields.items():
                accum[p][name].add(w, value)
            for k in (0, 1):
                observed = (m['detector'] == 0) & (selected[:, k, None] > 0)
                for model in range(models):
                    histograms[p][model, k] += np.histogram(cm[observed], parent.ENERGY_EDGES,
                        weights=np.broadcast_to(w[:, model, None], observed.shape)[observed])[0]
        if last % 50000 == 0 or last == len(weights):
            print(reference_name, channel, last, '/', len(weights), 'strip events complete', flush=True)
    sums, squared_sums = weights.sum(axis=0), (weights*weights).sum(axis=0)
    cases = {p: {name: moment.finish(sums, squared_sums) for name, moment in fields.items()} for p, fields in accum.items()}
    for p, row in cases.items():
        row['spectrum_per_generated'] = histograms[p]/sums[:, None, None]
        if (not np.allclose(row['geometric_to_matched']['value'].sum(axis=(1, 2)), 1.)
                or np.any(np.diff(row['cutflow']['value'], axis=-1) > 1e-12)
                or not np.allclose(row['classification']['value'].sum(axis=-1), row['efficiency']['value'][:, :, None])):
            raise ArithmeticError('Matching flow does not conserve event weight')
    return dict(cases=cases, provenance=source['provenance'], model_indices=source['model_indices'])


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    frozen = parent.load_cached(); selection = audit.load_cached()
    if selection['parent_checksum'] != frozen['checksum'] or parent.sha256(audit.__file__) != selection['source_sha256']:
        raise ValueError('Selection baseline changed')
    for name, sha in frozen['configuration']['source_sha256'].items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed')
    paths = [Path(__file__), Path(matching.__file__)]
    sources = {str(p.relative_to(parent.ROOT)): parent.sha256(p) for p in paths}
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    print('Loading verified references and fixed amplitude model', flush=True)
    amplitude = parent.parent.AmplitudeModel(); tables = response.RangeTables(frozen['stopping'])
    results = {}
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata, reference, channel, amplitude), tables, reference, channel)
                  for channel in (0, 2)]
        summary = {}
        for p in PROFILES:
            summary[p] = {name: ({field: np.concatenate([g['cases'][p][name][field] for g in groups], axis=0)
                                 for field in ('value', 'standard_error')} if name != 'spectrum_per_generated' else
                                np.concatenate([g['cases'][p][name] for g in groups], axis=0))
                          for name in groups[0]['cases'][p]}
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups],
            previous_combined=selection['results'][reference]['summary']['combined'])
    if any(parent.sha256(parent.ROOT/name) != sha for name, sha in sources.items()):
        raise ValueError('Matching sources changed during calculation')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, profiles=PROFILES, diagnostics=DIAGNOSTICS,
        migration_axes=['model', 'geometric_multiplicity_0_1_2_3', 'matched_0_1_2_3_overflow_ambiguous'],
        source_sha256=sources, source_snapshots={str(p.relative_to(parent.ROOT)): p.read_text() for p in paths},
        parent_checksum=selection['checksum'], response_checksum=frozen['checksum'],
        source_configuration=frozen['configuration'], evidence=selection['thesis'],
        interpreter=sys.executable, energy_edges=parent.ENERGY_EDGES, results=results,
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin,
        scope='Observable-only matching under declared hypotheses; not the historical matcher, experiment fit, or a branch limit'))
    state['checksum'] = parent.digest(state)
    output.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False))
    print('Matching study complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Matching cache checksum changed')
    return state


def plot_efficiency(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = ('direct_only', 'sum_first', 'share_first', 'unique', 'split_sum_first', 'split_share_first', 'split_unique')
    labels = ('Direct only', 'Sum first', 'Share first', 'Unique', 'Split + sum', 'Split + share', 'Split + unique')
    for row, model in enumerate((0, 5)):
        for k in (0, 1):
            ax = axes[row, k]
            centre = 100*state['evidence']['set9_efficiency'][row][k]
            error = 100*state['evidence']['set9_total_error'][row][k]
            ax.axhspan(centre-error, centre+error, color='#c2b69b', alpha=.35, label='Thesis set 9, total error')
            for ref, dx, color, marker in (('old', -.08, '#246a9a', 'o'), ('new', .08, '#bd453e', 's')):
                summary = state['results'][ref]['summary']
                y = [100*summary[p]['classification']['value'][model][k][1][row] for p in names]
                err = [100*summary[p]['classification']['standard_error'][model][k][1][row] for p in names]
                ax.errorbar(np.arange(len(names))+dx, y, yerr=err, fmt=marker, color=color, ms=4, capsize=2, label=ref+' reference')
            ax.set_xticks(range(len(names)), labels, rotation=35, ha='right')
            ax.set_ylabel('Own channel / generated (%)')
            ax.set_title(f'15({"abcd"[2*row+k]}) '+('alpha0' if row==0 else 'R + phases + Coulomb')+f', multiplicity {k+2}')
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    return fig


def plot_spectrum(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for j, ref in enumerate(('old', 'new')):
        summary = state['results'][ref]['summary']
        for name, label, color in (('sum_first', 'No split, sum first', '#246a9a'),
                                   ('share_first', 'No split, share first', '#d98b16'),
                                   ('split_sum_first', 'Split, sum first', '#444444'),
                                   ('split_share_first', 'Split, share first', '#ad3452')):
            values = 100*np.asarray(summary[name]['spectrum_per_generated'])[0, 0]
            axes[j].stairs(values, state['energy_edges'], color=color, label=label)
        axes[j].set(xlim=(0, 6.5), xlabel='Measured alpha CM energy in D1 (MeV)',
                    ylabel='Measured alphas / generated decays (% per 50 keV)',
                    title=f'16({"ab"[j]}) alpha0, multiplicity 2, {ref} reference')
        axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
