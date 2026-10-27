"""Independent energy-closure controls after frozen observable strip matching.

Only the explicit CLI replays ROOT events. Imports and plots read completed
notebook caches. Window widths and offsets are declared sensitivity controls,
not recovered experimental cuts. Signal ancestry is a post-selection audit.
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
from analysis.efficiency import laursen_162_matching_study as matching

parent, response = matching.parent, matching.response
KEY = 'codex_laursen_162_energy_closure'
VERSION = '2026-10-26-v1-independent-energy-closure'
FRAMES = ('nominal_beam', 'known_parent_control')
GROUPS = ('one_to_one_support', 'one_to_many', 'many_to_one_or_overlap', 'cross_or_mixed')
WINDOWS = (('none', 0., None), ('w800', 0., .800), ('w400', 0., .400),
           ('w200', 0., .200), ('w100', 0., .100),
           ('w400_minus90', -.090, .400), ('w400_plus90', .090, .400))
RESIDUAL_EDGES = np.linspace(-4., 4., 161)
PLAN = dict(beam_kinetic_MeV=.162, reaction_Q_MeV=8.682,
    proton_mass_MeV=938.272013, alpha_mass_MeV=response.MASS,
    three_alpha_threshold_MeV=7.275,
    nominal_parent='E=3*m_alpha+Q+T_beam, p=(0,0,sqrt(T_beam*(T_beam+2*m_proton)))',
    nominal_scope='Fixed incident beam and stationary target; no event truth in reconstruction or energy selection',
    known_parent_scope='Frozen event-level parent four-vector; diagnostic only, not an observable experimental input',
    missing_alpha='Momentum conservation followed by the alpha mass shell; no total-energy constraint',
    residual='sum(T_alpha_CM_reconstructed) - (sqrt(E_parent^2-p_parent^2)-3*m_alpha)',
    excitation='7.275 MeV + sum(T_alpha_CM_reconstructed); same residual, no extra independent cut',
    windows=[dict(name=n, centre_MeV=c, half_width_MeV=w) for n, c, w in WINDOWS],
    window_status='Predeclared sensitivity family; no fixed energy window established in checked method sections',
    offset_scope='Energy-reference offset only; not a simulation of material growth, threshold drift or calibration uncertainty',
    ancestry='Post-match strip contribution sets only; support association is not proof of accurate energy reconstruction',
    noise_seed=matching.PLAN['noise_seed'], chunk_size=matching.PLAN['chunk_size'],
    evidence=dict(thesis_pdf_pages=[90, 91, 92, 93, 94, 99, 100],
        thesis_visual_pages=[91], journal_pdf_pages=[2, 6, 7], journal_visual_pages=[6],
        journal_doi='10.1140/epja/i2016-16271-2',
        journal_sha256='14cb3ff898e9fc69f5980b75389b941f5d9847f7daffa94d0f7208f5ffa38754',
        review_status='Targeted primary-source check; library note remains an AI draft, no new human adoption',
        constants='Q and 3-alpha threshold from cited paper/thesis; proton mass from installed Geant4 CLHEP PhysicalConstants.h'),
    limitations=['No measured energy-window calibration', 'Frozen approximate geometry and electronics',
                 'No background, TDC jitter, asymmetric sharing or extra interstrip losses',
                 'No new reaction simulation, experimental fit, or weak-branch limit'])


def nominal_parent(size):
    """A declared beam input, independent of generated particles and channels."""
    if not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError('Expected a nonnegative event count')
    kinetic = PLAN['beam_kinetic_MeV']
    energy = np.full(size, 3*response.MASS+PLAN['reaction_Q_MeV']+kinetic)
    momentum = np.zeros((size, 3))
    momentum[:, 2] = np.sqrt(kinetic*(kinetic+2*PLAN['proton_mass_MeV']))
    return dict(parent_energy=energy, parent_momentum=momentum)


def available_energy(parent_energy, parent_momentum):
    energy, momentum = np.asarray(parent_energy, float), np.asarray(parent_momentum, float)
    if (energy.ndim != 1 or momentum.shape != (len(energy), 3)
            or not np.isfinite(energy).all() or not np.isfinite(momentum).all()
            or np.any(energy <= 0)):
        raise ValueError('Expected finite parent four-vectors')
    square = energy*energy-np.sum(momentum*momentum, axis=1)
    if np.any(square < (3*response.MASS)**2):
        raise ValueError('Parent below the three-alpha threshold')
    return np.sqrt(square)-3*response.MASS


def energy_residual(kinetic_cm, parent):
    kinetic = np.asarray(kinetic_cm, float)
    if kinetic.shape != (len(parent['parent_energy']), 3) or not np.isfinite(kinetic).all():
        raise ValueError('Expected three finite reconstructed CM kinetic energies')
    return kinetic.sum(axis=1)-available_energy(parent['parent_energy'], parent['parent_momentum'])


def window_masks(residual):
    residual = np.asarray(residual, float)
    if residual.ndim != 1:
        raise ValueError('Expected one energy residual per event')
    return np.stack([np.isfinite(residual) & (True if width is None else abs(residual-centre) <= width)
                     for _, centre, width in WINDOWS], axis=1)


def strip_ancestry(detector, coordinate, light, width):
    """Forward contribution bit sets, never supplied to the matcher or cuts."""
    ancestry = np.zeros((len(detector), 2, 2, 16), np.uint8)
    for particle in range(3):
        rows = np.flatnonzero((detector[:, particle] >= 0) & (light[:, particle] > 0))
        d = detector[rows, particle]
        for side in (0, 1):
            first, second, fraction = matching.matching.split_charge(
                coordinate[rows, particle, side], response.PITCH, width)
            for strips, share in ((first, 1-fraction), (second, fraction)):
                active = share > 0
                np.bitwise_or.at(ancestry, (rows[active], d[active], side, strips[active]), 1 << particle)
    return ancestry


def association_group(matched, ancestry):
    """Exclusive support categories; overlapping contributions stay explicit.

    Fractional matched coordinates describe adjacent sharing groups. Each
    reconstructed hit is checked against the union AND intersection of the
    contributing truth bits on its front/back support. Same-pixel overlaps
    are never silently called a correct one-to-one reconstruction.
    """
    n = len(matched['multiplicity'])
    active = matched['detector'] >= 0
    common = np.zeros((n, 3), np.uint8)
    compatible = np.zeros((n, 3), bool)
    for hit in range(3):
        d = np.maximum(matched['detector'][:, hit], 0)
        position = matched['pixel'][:, hit]
        rounded = np.rint(position)
        position = np.where(abs(position-rounded) < 1e-10, rounded, position)
        low = np.clip(np.floor(position).astype(int), 0, 15)
        high = np.clip(np.ceil(position).astype(int), 0, 15)
        f0, f1 = (ancestry[np.arange(n), d, 0, index[:, 0]] for index in (low, high))
        b0, b1 = (ancestry[np.arange(n), d, 1, index[:, 1]] for index in (low, high))
        common[:, hit] = (f0 | f1) & (b0 | b1)
        compatible[:, hit] = (f0 & f1 & b0 & b1) == common[:, hit]
    single = (common > 0) & ((common & (common-1)) == 0)
    repeated = np.zeros(n, bool)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        repeated |= active[:, i] & active[:, j] & (common[:, i] == common[:, j])
    valid = (matched['multiplicity'] > 0) & (matched['multiplicity'] <= 3) & ~matched['ambiguous']
    result = np.full(n, 3, int)
    resolved = valid & np.all(~active | (single & compatible), axis=1)
    result[resolved & ~repeated] = 0
    result[resolved & repeated] = 1
    multiple = valid & np.all(~active | (common > 0), axis=1) & np.any(active & ~single, axis=1)
    result[multiple] = 2
    return result


class ResidualHistogram:
    """Generated-normalized event bins including underflow and overflow."""
    def __init__(self, models):
        self.shape = (models, 2, len(GROUPS), 2, len(RESIDUAL_EDGES)+1)
        self.first = np.zeros((models, int(np.prod(self.shape[1:]))))
        self.second = np.zeros_like(self.first)

    def add(self, weights, selected, classified, residual, groups):
        bins = np.searchsorted(RESIDUAL_EDGES, residual, side='right')
        for k in (0, 1):
            rows = np.flatnonzero(selected[:, k])
            channel = (classified[rows, k, 1] > 0).astype(int)
            index = ((k*len(GROUPS)+groups[rows])*2+channel)*(len(RESIDUAL_EDGES)+1)+bins[rows]
            for model in range(weights.shape[1]):
                w = weights[rows, model]
                self.first[model] += np.bincount(index, weights=w, minlength=self.first.shape[1])
                self.second[model] += np.bincount(index, weights=w*w, minlength=self.first.shape[1])

    def finish(self, sums, squared_sums):
        mean = self.first/sums[:, None]
        variance = (1-2*mean)*self.second+mean*mean*squared_sums[:, None]
        return dict(value=mean.reshape(self.shape),
                    standard_error=(np.sqrt(np.maximum(0., variance))/sums[:, None]).reshape(self.shape))


def reduce_reference(source, tables, reference_name, channel):
    prepared = matching.prepare(source, tables)
    weights = source['weights']; models = weights.shape[1]
    shapes = dict(efficiency=(len(WINDOWS), 2), classification=(len(WINDOWS), 2, 2),
        association_classification=(len(WINDOWS), 2, len(GROUPS), 2),
        paired_loss=(len(WINDOWS), 2), paired_channel_loss=(len(WINDOWS), 2, 2))
    accum = {p: {frame: {name: matching.Moments(models, shape) for name, shape in shapes.items()}
                 for frame in FRAMES} for p in matching.PROFILES}
    differences = {p: matching.Moments(models, (len(WINDOWS), 2, 2)) for p in matching.PROFILES}
    histograms = {p: {frame: ResidualHistogram(models) for frame in FRAMES} for p in matching.PROFILES}
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference_name == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first
        data = {key: value[first:last] for key, value in prepared.items() if key != 'geometry'}
        detector = prepared['geometry']['detector'][first:last]
        w = weights[first:last]
        noise = rng.standard_normal((size, 2, 2, 16))
        raw = matching.form_signals(detector, data['coordinate'], data['light'], noise, 0., tables)
        matched = matching.match_signals(raw, direct=True)
        matched.update(matching.match_signals(raw))
        split = matching.form_signals(detector, data['coordinate'], data['light'], noise, .100, tables)
        matched.update({'split_'+k: v for k, v in matching.match_signals(split).items()})
        ancestry = {width: strip_ancestry(detector, data['coordinate'], data['light'], width) for width in (0., .100)}
        inputs = dict(nominal_beam=nominal_parent(size), known_parent_control=data)
        for p, m in matched.items():
            groups = association_group(m, ancestry[.100 if p.startswith('split_') else 0.])
            group_mask = np.eye(len(GROUPS))[groups]
            evaluated = {}
            for frame in FRAMES:
                selected, _, labels, kinetic = matching.reconstruct(m, inputs[frame], tables)
                classified = labels[:, :, 1, :]
                residual = energy_residual(kinetic, inputs[frame])
                masks = window_masks(residual)
                efficiency = selected[:, None, :]*masks[:, :, None]
                assigned = classified[:, None, :, :]*masks[:, :, None, None]
                fields = dict(efficiency=efficiency, classification=assigned,
                    association_classification=assigned[:, :, :, None, :]*group_mask[:, None, None, :, None],
                    paired_loss=selected[:, None, :]-efficiency,
                    paired_channel_loss=classified[:, None, :, :]-assigned)
                for name, value in fields.items():
                    accum[p][frame][name].add(w, value)
                histograms[p][frame].add(w, selected, classified, residual, groups)
                evaluated[frame] = assigned
            differences[p].add(w, evaluated['nominal_beam']-evaluated['known_parent_control'])
        if last % 50000 == 0 or last == len(weights):
            print(reference_name, channel, last, '/', len(weights), 'energy events complete', flush=True)
    sums, squared_sums = weights.sum(axis=0), (weights*weights).sum(axis=0)
    summary = {}
    for p, frames in accum.items():
        summary[p] = {frame: {name: m.finish(sums, squared_sums) for name, m in fields.items()}
                      for frame, fields in frames.items()}
        for frame in FRAMES:
            summary[p][frame]['residual_histogram'] = histograms[p][frame].finish(sums, squared_sums)
        summary[p]['paired_nominal_minus_known'] = differences[p].finish(sums, squared_sums)
    return dict(summary=summary, provenance=source['provenance'])


def validate_summary(summary, baseline=None):
    for p, frames in summary.items():
        for frame in FRAMES:
            row = frames[frame]
            efficiency = np.asarray(row['efficiency']['value'])
            classified = np.asarray(row['classification']['value'])
            groups = np.asarray(row['association_classification']['value'])
            histogram = np.asarray(row['residual_histogram']['value'])
            if (not np.allclose(classified.sum(axis=-1), efficiency, atol=1e-12)
                    or not np.allclose(groups.sum(axis=-2), classified, atol=1e-12)
                    or not np.allclose(histogram.sum(axis=(2, 4)), classified[:, 0], atol=1e-12)
                    or np.any(np.diff(efficiency[:, :5], axis=1) > 1e-12)
                    or not np.allclose(row['paired_loss']['value'], efficiency[:, :1]-efficiency, atol=1e-12)):
                raise ArithmeticError('Energy selection, ancestry or histogram weights do not close')
        if baseline is not None:
            row = frames['known_parent_control']
            if (not np.allclose(np.asarray(row['efficiency']['value'])[:, 0], baseline[p]['efficiency']['value'], atol=1e-12, rtol=0)
                    or not np.allclose(np.asarray(row['classification']['value'])[:, 0], np.asarray(baseline[p]['classification']['value'])[:, :, 1], atol=1e-12, rtol=0)):
                raise ArithmeticError('Frozen matching baseline was not reproduced')


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    baseline = matching.load_cached(); frozen = parent.load_cached()
    if baseline['response_checksum'] != frozen['checksum']:
        raise ValueError('Response baseline mismatch')
    frozen_sources = dict(frozen['configuration']['source_sha256'])
    frozen_sources.update(baseline['source_sha256'])
    selection = matching.audit.load_cached()
    if selection['checksum'] != baseline['parent_checksum']:
        raise ValueError('Selection baseline mismatch')
    frozen_sources[str(Path(matching.audit.__file__).relative_to(parent.ROOT))] = selection['source_sha256']
    for name, sha in frozen_sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed: '+name)
    source_name = str(Path(__file__).relative_to(parent.ROOT))
    source_sha = parent.sha256(__file__)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    print('Loading verified references and fixed amplitude model', flush=True)
    amplitude = parent.parent.AmplitudeModel()
    tables = response.RangeTables(frozen['stopping'])
    results = {}
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata, reference, channel, amplitude), tables, reference, channel)
                  for channel in (0, 2)]
        summary = {}
        for p in matching.PROFILES:
            summary[p] = {frame: {name: {field: np.concatenate([g['summary'][p][frame][name][field] for g in groups], axis=0)
                for field in ('value', 'standard_error')} for name in groups[0]['summary'][p][frame]} for frame in FRAMES}
            summary[p]['paired_nominal_minus_known'] = {field: np.concatenate([g['summary'][p]['paired_nominal_minus_known'][field] for g in groups], axis=0)
                                                       for field in ('value', 'standard_error')}
        validate_summary(summary, baseline['results'][reference]['summary'])
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups])
    for name, sha in {**frozen_sources, source_name: source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Numerical sources changed during calculation')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, groups=GROUPS, frames=FRAMES,
        profiles=matching.PROFILES, windows=[n for n, _, _ in WINDOWS], residual_edges=RESIDUAL_EDGES,
        histogram_axes=['model', 'multiplicity_2_3', 'association_group', 'assigned_gs_exc', 'underflow_bins_overflow'],
        parent_checksum=baseline['checksum'], source_sha256={source_name: source_sha},
        source_snapshots={source_name: Path(__file__).read_text()}, frozen_source_sha256=frozen_sources,
        evidence=baseline['evidence'], source_configuration=baseline['source_configuration'],
        results=results, interpreter=sys.executable,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent.ROOT, text=True).strip(),
        nominal_available_MeV=float(available_energy(**nominal_parent(1))[0]),
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin))
    state['checksum'] = parent.digest(state)
    with output.open('x') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
    print('Energy-closure study complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Energy-closure cache checksum changed')
    return state


def plot_residuals(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ('One-to-one support', 'One-to-many', 'Many-to-one / overlap', 'Cross / mixed')
    colors = ('#246a9a', '#bd453e', '#d98b16', '#675392')
    cases = ((0, 0, 'split_share_first'), (5, 0, 'split_sum_first'), (0, 1, 'sum_first'), (5, 1, 'sum_first'))
    for panel, (ax, (model, k, profile)) in enumerate(zip(axes.flat, cases)):
        for ref, style in (('old', '-'), ('new', '--')):
            data = np.asarray(state['results'][ref]['summary'][profile]['nominal_beam']['residual_histogram']['value'])
            counts = 100*data[model, k].sum(axis=1)
            for group, label in enumerate(labels):
                ax.stairs(counts[group, 1:-1], state['residual_edges'], color=colors[group], linestyle=style,
                          label=label if ref == 'old' else None)
        ax.axvspan(-.4, .4, color='grey', alpha=.12)
        ax.set(xlim=(-3, 3), yscale='log', ylim=(1e-5, 30), xlabel='Reconstructed CM energy residual (MeV)',
               ylabel='Events / generated (%) per 50 keV',
               title=f'17({"abcd"[panel]}) '+('alpha0' if model == 0 else 'R + phases + Coulomb')+f', m={k+2}\n{profile.replace("_", " ")}')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Post-match ancestry audit; solid: old reference, dashed: new reference')
    return fig


def plot_windows(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, model in enumerate((0, 5)):
        for k in (0, 1):
            ax = axes[row, k]
            for profile, color in (('sum_first', '#246a9a'), ('split_sum_first', '#bd453e'), ('split_share_first', '#d98b16')):
                for ref, style, marker in (('old', '-', 'o'), ('new', '--', 's')):
                    data = state['results'][ref]['summary'][profile]['nominal_beam']['classification']
                    values = 100*np.asarray(data['value'])[model, :5, k, row]
                    error = 100*np.asarray(data['standard_error'])[model, :5, k, row]
                    ax.errorbar(np.arange(5), values, yerr=error, color=color, linestyle=style, marker=marker,
                                ms=3, capsize=2, label=profile.replace('_', ' ') if ref == 'old' else None)
            ax.set_xticks(np.arange(5), ('None', '800', '400', '200', '100'))
            ax.set(xlabel='Symmetric half-width (keV)', ylabel='Own channel / generated (%)',
                   title=f'18({"abcd"[2*row+k]}) '+('alpha0' if model == 0 else 'R + phases + Coulomb')+f', m={k+2}')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Declared energy-window controls; solid: old reference, dashed: new reference')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
