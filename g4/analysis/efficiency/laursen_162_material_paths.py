"""Layer and angular diagnostics on frozen one-to-one strip cohorts.

Truth paths are post-selection controls. Pixel/foil bounds use observed strip
support and the declared geometry only; they do not change the event selection.
Only the explicit CLI replays ROOT samples. Imports and plots read caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_energy_decomposition as decomposition

energy, matching, parent, response = decomposition.energy, decomposition.matching, decomposition.parent, decomposition.response
KEY = 'codex_laursen_162_material_paths'
VERSION = '2026-10-27-v1-material-angle-support'
PROFILES = decomposition.PROFILES
CASES = ('baseline', 'B', 'C', 'BC', 'S', 'BS', 'CS', 'BCS',
         'B_true_angle_mid_depth', 'C_true_angle_nominal_side', 'C_nominal_angle_true_side')
ANGLE_EDGES = np.array([0., .5, 1., 2., 5., 10., 20., 40., 90.])
PARTITIONS = ('overall', 'nominal_angle', 'true_angle', 'foil_support')
PLAN = dict(noise_seed=matching.PLAN['noise_seed'], chunk_size=matching.PLAN['chunk_size'],
    cohort=decomposition.PLAN['cohort'], comparison=decomposition.PLAN['comparison'],
    reconstruction='All layer controls retain baseline matched energy, momentum directions and nominal parent',
    layers=['B: boron exit column, including depth', 'C: exit adsorption plus side-dependent backing', 'S: dead silicon'],
    combinations='Replace each of three columns by its true value, independently; inverse order stays S,C,B',
    auxiliary=['Only B uses true angle with depth one half',
               'Only C uses true absolute foil cosine, retaining nominal exit-side backing',
               'Only C uses true exit-side backing, retaining nominal absolute foil cosine'],
    angle='Minimum angle above foil plane among measured hits: asin(abs(direction dot foil_normal)), degrees',
    angle_edges_deg=ANGLE_EDGES.tolist(), window_half_width_MeV=.4, tail_threshold_MeV=1.,
    support='Full observed strip rectangles, extended by half the declared sharing width and clipped to active area',
    beam='Frozen uniform square in x/y, width 2 mm; origins lie in nominal foil plane',
    crossing='Support includes both signs of (hit-origin) dot foil_normal; computed without event truth',
    cosine_bounds='Conservative bounds using exact affine projection extrema and vertex maximum distance; no uniform angular average',
    foil_support_labels=['no crossing', 'D1 crossing only', 'D2 crossing only', 'both detectors crossing'],
    finite_policy='Compare common finite inversions; retain baseline counts and report excluded/invalid generated weight',
    limitations=['Geometry and slab material model are still conditional approximations',
                 'Pixel risk flag is an audit, not a cut or a calibrated response',
                 'No new Geant4 events, no author hardware calibration, no external efficiency validation'])


def pixel_foil_bounds(matched, sharing_width_mm=0., beam_width_mm=2.):
    """Conservative direction bounds from observed support, without truth.

    Norm is convex, so its maximum over the pixel/source box is attained at
    a vertex. The detector-normal projection supplies a lower distance bound.
    Foil projections are affine; their sign range is exact because all allowed
    origins lie in the foil plane. This does not assign a distribution inside
    the support or regularize its tangent directions.
    """
    if not 0 <= sharing_width_mm < response.PITCH or beam_width_mm < 0:
        raise ValueError('Invalid sharing or beam width')
    detector = np.asarray(matched['detector']); active = detector >= 0
    pixel = np.asarray(matched['pixel'], float)
    if pixel.shape != (*detector.shape, 2) or np.any(active & (detector > 1)) or not np.isfinite(pixel).all():
        raise ValueError('Invalid observed support')
    rounded = np.rint(pixel)
    pixel = np.where(abs(pixel-rounded) < 1e-10, rounded, pixel)
    if np.any(active[..., None] & ((pixel < 0) | (pixel > 15))):
        raise ValueError('Strip coordinates outside active detector')
    low = np.maximum(-response.HALF_SIZE, np.floor(pixel)*response.PITCH-response.HALF_SIZE-sharing_width_mm/2)
    high = np.minimum(response.HALF_SIZE, (np.ceil(pixel)+1)*response.PITCH-response.HALF_SIZE+sharing_width_mm/2)
    d = np.maximum(detector, 0)
    normals, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
    vertices = np.stack([response.CENTRES[d]+np.where(i, high[..., 0], low[..., 0])[..., None]*horizontal[d]
                         +np.where(j, high[..., 1], low[..., 1])[..., None]*vertical
                         for i, j in itertools.product((0, 1), repeat=2)], axis=-2)
    xy = np.array(list(itertools.product((-beam_width_mm/2, beam_width_mm/2), repeat=2)))
    origins = np.column_stack([xy, -response.FOIL_NORMAL[0]/response.FOIL_NORMAL[2]*xy[:, 0]])
    vectors = vertices[..., :, None, :]-origins
    distance_max = np.linalg.norm(vectors, axis=-1).max(axis=(-1, -2))
    projection = vertices@response.FOIL_NORMAL
    signed_low, signed_high = projection.min(axis=-1), projection.max(axis=-1)
    crosses = active & (signed_low <= 0) & (signed_high >= 0)
    absolute_low = np.where(crosses, 0., np.minimum(abs(signed_low), abs(signed_high)))
    absolute_high = np.maximum(abs(signed_low), abs(signed_high))
    detector_projection = np.sum((response.CENTRES[d][..., None, :]-origins)*normals[d][..., None, :], axis=-1)
    minimum_projection, maximum_projection = detector_projection.min(axis=-1), detector_projection.max(axis=-1)
    distance_min = np.where((minimum_projection <= 0) & (maximum_projection >= 0), 0.,
                            np.minimum(abs(minimum_projection), abs(maximum_projection)))
    cosine_low = absolute_low/np.maximum(distance_max, 1e-14)
    cosine_high = np.minimum(1., absolute_high/np.maximum(distance_min, 1e-14))
    return dict(low_mm=low, high_mm=high, crosses=crosses,
                cosine_low=np.where(active, cosine_low, 0.), cosine_high=np.where(active, cosine_high, 1.))


def geometry_inventory():
    matched = matching.allocate(2*16*16)
    records = np.array(list(itertools.product(range(2), range(16), range(16))))
    matched['detector'][:, 0] = records[:, 0]
    matched['pixel'][:, 0] = records[:, 1:]
    result = pixel_foil_bounds(matched)
    normal = response.FOIL_NORMAL
    _, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
    crossing_x = -(response.CENTRES@normal)/(horizontal@normal)
    position = response.CENTRES[records[:, 0]]+((records[:, 1]+.5)*response.PITCH-response.HALF_SIZE)[:, None]*horizontal[records[:, 0]]
    position += ((records[:, 2]+.5)*response.PITCH-response.HALF_SIZE)[:, None]*vertical
    direction = response.unit(position)
    return dict(direct_pixels_crossing=records[result['crosses'][:, 0]],
                foil_intersection_xprime_mm=crossing_x,
                nominal_angle_deg=np.degrees(np.arcsin(np.clip(abs(direction@normal), 0., 1.))).reshape(2, 16, 16))


def material_controls(matched, true_direction, true_columns, tables):
    direction, nominal_columns = decomposition.nominal_measurements(matched, tables)
    nominal = energy.nominal_parent(len(direction)); available = energy.available_energy(**nominal)
    residual = np.empty((len(direction), len(CASES)))
    for case in range(8):
        columns = tuple(true_columns[layer] if case & (1 << layer) else nominal_columns[layer] for layer in range(3))
        cm = decomposition.reconstruct_finite(decomposition.corrected_energy(matched['energy'], columns, tables),
            direction, matched['multiplicity'], nominal)
        residual[:, case] = cm.sum(axis=1)-available
    config = response.Response()
    mid = matching.audit.columns(true_direction, matched['detector'], np.full(len(direction), .5),
                                 config, tables, matching.PLAN['geometry'])
    nominal_dot = direction@response.FOIL_NORMAL; true_dot = true_direction@response.FOIL_NORMAL
    carbon_angle = (config.adsorption_exit_ug_cm2+config.backing_ug_cm2*(config.backing_side*nominal_dot > 0))*1e-6/np.maximum(abs(true_dot), 1e-14)
    carbon_side = (config.adsorption_exit_ug_cm2+config.backing_ug_cm2*(config.backing_side*true_dot > 0))*1e-6/np.maximum(abs(nominal_dot), 1e-14)
    carbon_angle = np.where(matched['detector'] >= 0, carbon_angle, 0.)
    carbon_side = np.where(matched['detector'] >= 0, carbon_side, 0.)
    auxiliary = ((mid[0], nominal_columns[1], nominal_columns[2]),
                 (nominal_columns[0], carbon_angle, nominal_columns[2]),
                 (nominal_columns[0], carbon_side, nominal_columns[2]))
    for case, columns in enumerate(auxiliary, 8):
        cm = decomposition.reconstruct_finite(decomposition.corrected_energy(matched['energy'], columns, tables),
            direction, matched['multiplicity'], nominal)
        residual[:, case] = cm.sum(axis=1)-available
    return residual, direction


def angle_bucket(direction, active):
    angles = np.degrees(np.arcsin(np.clip(abs(direction@response.FOIL_NORMAL), 0., 1.)))
    minimum = np.where(active, angles, 90.).min(axis=1)
    return np.searchsorted(ANGLE_EDGES[1:-1], minimum, side='right')


class PartitionMoments:
    """Weighted conditional moments and paired errors in a fixed partition."""
    def __init__(self, models, partitions):
        self.partitions = partitions
        self.weight = np.zeros((models, 4, partitions)); self.weight2 = np.zeros_like(self.weight)
        self.count = np.zeros((4, partitions), int)
        shape = (*self.weight.shape, len(CASES), 6)
        self.first, self.w2first, self.w2second = [np.zeros(shape) for _ in range(3)]

    def add(self, weights, category, partition, residual):
        if not np.isfinite(residual).all():
            raise ValueError('Partition moments require a common finite cohort')
        keep = (abs(residual) <= PLAN['window_half_width_MeV']).astype(float)
        values = np.stack([residual, residual*residual, keep, keep-keep[:, :1],
                           (abs(residual) > PLAN['tail_threshold_MeV']).astype(float),
                           (residual > PLAN['tail_threshold_MeV']).astype(float)], axis=-1)
        for bucket in range(4):
            for part in range(self.partitions):
                rows = np.flatnonzero((category == bucket) & (partition == part))
                if not len(rows):
                    continue
                w = weights[rows]; w2 = w*w; value = values[rows].reshape(len(rows), -1)
                self.weight[:, bucket, part] += w.sum(axis=0)
                self.weight2[:, bucket, part] += w2.sum(axis=0)
                self.count[bucket, part] += len(rows)
                shape = (w.shape[1], len(CASES), 6)
                self.first[:, bucket, part] += (w.T@value).reshape(shape)
                self.w2first[:, bucket, part] += (w2.T@value).reshape(shape)
                self.w2second[:, bucket, part] += (w2.T@(value*value)).reshape(shape)

    def finish(self, generated):
        safe = np.where(self.weight > 0, self.weight, 1.)
        d = safe[..., None, None]; w2 = self.weight2[..., None, None]
        average = self.first/d
        error = np.sqrt(np.maximum(0., self.w2second-2*average*self.w2first+average*average*w2))/d
        cohort = self.weight.sum(axis=-1, keepdims=True)
        fraction = self.weight/np.where(cohort > 0, cohort, 1.)
        fraction_error = np.sqrt(np.maximum(0., (1-2*fraction)*self.weight2+fraction*fraction*self.weight2.sum(axis=-1, keepdims=True)))/np.where(cohort > 0, cohort, 1.)
        return dict(valid=self.weight > 0, event_count=np.broadcast_to(self.count, self.weight.shape),
            per_generated=self.weight/generated[:, None, None], cohort_fraction=fraction,
            cohort_fraction_standard_error=fraction_error,
            effective_sample_size=np.divide(self.weight*self.weight, self.weight2, out=np.zeros_like(self.weight), where=self.weight2 > 0),
            mean_MeV=average[..., 0], mean_standard_error_MeV=error[..., 0],
            rms_MeV=np.sqrt(np.maximum(0., average[..., 1])),
            std_MeV=np.sqrt(np.maximum(0., average[..., 1]-average[..., 0]**2)),
            retention=average[..., 2], retention_standard_error=error[..., 2],
            paired_retention_change=average[..., 3], paired_retention_standard_error=error[..., 3],
            tail1_probability=average[..., 4], tail1_standard_error=error[..., 4], upper_tail1_probability=average[..., 5])


def reduce_reference(source, tables, reference, channel):
    prepared = matching.prepare(source, tables)
    detector = prepared['geometry']['detector']
    true_columns = matching.audit.columns(source['directions'], detector, source['depth'],
        response.Response(), tables, matching.PLAN['geometry'])
    weights = source['weights']; models = weights.shape[1]
    specs = dict(overall=1, nominal_angle=len(ANGLE_EDGES)-1, true_angle=len(ANGLE_EDGES)-1, foil_support=4)
    moments = {p: {name: PartitionMoments(models, count) for name, count in specs.items()} for p in PROFILES}
    shapes = dict(baseline=(2, 2), w400=(2, 2), invalid=(4, len(CASES)), excluded=(4,), side_mismatch=(4,))
    rates = {p: {name: matching.Moments(models, shape) for name, shape in shapes.items()} for p in PROFILES}
    rng = np.random.default_rng([PLAN['noise_seed'], int(reference == 'new'), channel])
    for first in range(0, len(weights), PLAN['chunk_size']):
        last = min(first+PLAN['chunk_size'], len(weights)); size = last-first
        noise = rng.standard_normal((size, 2, 2, 16)); matches = {}; ancestry = {}
        for width in (0., .100):
            observed = matching.form_signals(detector[first:last], prepared['coordinate'][first:last], prepared['light'][first:last], noise, width, tables)
            matches.update({('split_' if width else '')+k: v for k, v in matching.match_signals(observed).items()})
            ancestry[width] = energy.strip_ancestry(detector[first:last], prepared['coordinate'][first:last], prepared['light'][first:last], width)
        nominal = energy.nominal_parent(size)
        for p in PROFILES:
            matched = matches[p]; width = .100 if p.startswith('split_') else 0.; bits = ancestry[width]
            selected, _, labels, cm = matching.reconstruct(matched, nominal, tables)
            associated = energy.association_group(matched, bits) == 0
            assigned = labels[:, :, 1]*associated[:, None, None]
            base = energy.energy_residual(cm, nominal)
            rates[p]['baseline'].add(weights[first:last], assigned)
            rates[p]['w400'].add(weights[first:last], assigned*(abs(base) <= .4)[:, None, None])
            rows = np.flatnonzero(associated & selected.any(axis=1))
            if not len(rows):
                continue
            subset = {k: v[rows] for k, v in matched.items()}
            mapping = decomposition.truth_permutation(subset, bits[rows]); absolute = first+rows
            truth = source['directions'][absolute[:, None], mapping]
            columns = tuple(value[absolute[:, None], mapping] for value in true_columns)
            residual, direction = material_controls(subset, truth, columns, tables)
            if not np.allclose(residual[:, 0], base[rows], atol=1e-10, rtol=0):
                raise ArithmeticError('Material baseline changed the frozen reconstruction')
            category = (subset['multiplicity']-2)*2+assigned[rows, :, 1].sum(axis=1).astype(int)
            finite = np.isfinite(residual); common = finite.all(axis=1)
            indicators = np.eye(4)[category]
            rates[p]['invalid'].add(weights[absolute], indicators[:, :, None]*(~finite)[:, None, :])
            rates[p]['excluded'].add(weights[absolute], indicators*(~common)[:, None])
            support = pixel_foil_bounds(subset, width, parent.PLAN['beam_width_mm'])
            active = subset['detector'] >= 0
            cosine = abs(truth@response.FOIL_NORMAL)
            if np.any(active & ((cosine < support['cosine_low']-1e-10) | (cosine > support['cosine_high']+1e-10))):
                raise ArithmeticError('True direction outside conservative observable bounds')
            mismatch = active & ((direction@response.FOIL_NORMAL > 0) != (truth@response.FOIL_NORMAL > 0))
            if np.any(mismatch & ~support['crosses']):
                raise ArithmeticError('Exit-side ambiguity outside crossing support')
            rates[p]['side_mismatch'].add(weights[absolute], indicators*mismatch.any(axis=1)[:, None])
            support_group = np.zeros(len(rows), int)
            for d in (0, 1):
                support_group += (1 << d)*np.any(support['crosses'] & (subset['detector'] == d), axis=1)
            partitions = dict(overall=np.zeros(len(rows), int), nominal_angle=angle_bucket(direction, active),
                              true_angle=angle_bucket(truth, active), foil_support=support_group)
            for name, partition in partitions.items():
                moments[p][name].add(weights[absolute][common], category[common], partition[common], residual[common])
        if last % 50000 == 0 or last == len(weights):
            print(reference, channel, last, '/', len(weights), 'material events complete', flush=True)
    sums, squared = weights.sum(axis=0), (weights*weights).sum(axis=0)
    summary = {p: {name: item.finish(sums) for name, item in fields.items()} for p, fields in moments.items()}
    for p in PROFILES:
        summary[p]['rates'] = {name: moment.finish(sums, squared) for name, moment in rates[p].items()}
    return dict(summary=summary, provenance=source['provenance'])


def validate_summary(summary, previous):
    for p in PROFILES:
        row = summary[p]; old = previous[p]; overall = row['overall']
        for name in ('baseline', 'w400'):
            if not np.allclose(row['rates'][name]['value'], old[name+'_per_generated']['value'], atol=1e-12, rtol=0):
                raise ArithmeticError('Frozen one-to-one counts changed')
        cohort = np.asarray(row['rates']['baseline']['value']).reshape(-1, 4)
        common = np.asarray(overall['per_generated'])[..., 0]
        if not np.allclose(common+row['rates']['excluded']['value'], cohort, atol=1e-12, rtol=0):
            raise ArithmeticError('Finite-cohort accounting failed')
        if not np.any(row['rates']['excluded']['value']):
            for name in ('mean_MeV', 'rms_MeV', 'retention'):
                if not np.allclose(np.asarray(overall[name])[:, :, 0, [0, 7]], np.asarray(old[name])[:, :, [0, 2]], atol=1e-10, rtol=0):
                    raise ArithmeticError('Baseline or all-material control differs from decomposition')
        for name in PARTITIONS[1:]:
            part = row[name]; weight = np.asarray(part['cohort_fraction'])
            if not np.allclose(np.asarray(part['per_generated']).sum(axis=-1), common, atol=1e-12, rtol=0):
                raise ArithmeticError('Partition lost common-finite weight')
            for field in ('mean_MeV', 'retention', 'tail1_probability'):
                if not np.allclose(np.sum(weight[..., None]*part[field], axis=2), np.asarray(overall[field])[:, :, 0], atol=1e-10, rtol=0):
                    raise ArithmeticError('Partition does not reproduce overall moments')
            if not np.allclose(np.sum(weight[..., None]*np.asarray(part['rms_MeV'])**2, axis=2), np.asarray(overall['rms_MeV'])[:, :, 0]**2, atol=1e-10, rtol=0):
                raise ArithmeticError('Partition lost squared-residual weight')


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed results; use the notebook cache')
    previous = decomposition.load_cached(); frozen = parent.load_cached()
    frozen_sources = {**previous['frozen_source_sha256'], **previous['source_sha256']}
    source_name = str(Path(__file__).relative_to(parent.ROOT)); source_sha = parent.sha256(__file__)
    for name, sha in frozen_sources.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Frozen numerical source changed: '+name)
    started = datetime.now().astimezone().isoformat(); begin = time.monotonic()
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    print('Loading frozen references, amplitude weights and geometry', flush=True)
    amplitude = parent.parent.AmplitudeModel(); tables = response.RangeTables(frozen['stopping'])
    def combine(values):
        if isinstance(values[0], dict):
            return {key: combine([v[key] for v in values]) for key in values[0]}
        return np.concatenate(values, axis=0)
    results = {}
    for reference in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata, reference, channel, amplitude), tables, reference, channel) for channel in (0, 2)]
        summary = combine([g['summary'] for g in groups])
        validate_summary(summary, previous['results'][reference]['summary'])
        results[reference] = dict(summary=summary, provenance=[g['provenance'] for g in groups])
    for name, sha in {**frozen_sources, source_name: source_sha}.items():
        if parent.sha256(parent.ROOT/name) != sha:
            raise ValueError('Numerical sources changed during replay')
    state = parent.parent.plain(dict(version=VERSION, plan=PLAN, profiles=PROFILES, cases=CASES,
        partitions=PARTITIONS, category_axes=previous['category_axes'], geometry_inventory=geometry_inventory(),
        parent_checksum=previous['checksum'], source_sha256={source_name: source_sha},
        source_snapshots={source_name: Path(__file__).read_text()}, frozen_source_sha256=frozen_sources,
        evidence=previous['evidence'], source_configuration=previous['source_configuration'],
        results=results, interpreter=sys.executable,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=parent.ROOT, text=True).strip(),
        started_at=started, completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-begin))
    state['checksum'] = parent.digest(state)
    with output.open('x') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
    print('Material-path study complete:', state['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Material-path cache checksum changed')
    return state


def plot_layers(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for panel, (ax, (model, bucket)) in enumerate(zip(axes.flat, ((0, 0), (0, 2), (5, 1), (5, 3)))):
        for reference, style, marker in (('old', '-', 'o'), ('new', '--', 's')):
            row = state['results'][reference]['summary']['split_sum_first']['overall']
            ax.plot(np.arange(8), 1000*np.asarray(row['rms_MeV'])[model, bucket, 0, :8], linestyle=style, marker=marker, label=reference)
        ax.set_xticks(np.arange(8), ['Base', 'B', 'C', 'BC', 'S', 'BS', 'CS', 'BCS'])
        ax.set_yscale('log'); ax.grid(alpha=.2, which='both'); ax.set_ylabel('Energy residual RMS / keV')
        ax.set_title(f'21({chr(97+panel)})  alpha{0 if model == 0 else 1}, m={bucket//2+2}')
    axes[0, 0].legend()
    fig.suptitle('True material columns; fixed baseline momentum directions and measured energies')
    return fig


def plot_angles(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(ANGLE_EDGES)-1)
    labels = [f'{lo:g}-{hi:g}' for lo, hi in zip(ANGLE_EDGES[:-1], ANGLE_EDGES[1:])]
    for row_index, (model, bucket) in enumerate(((0, 0), (5, 1))):
        for reference, style in (('old', '-'), ('new', '--')):
            data = state['results'][reference]['summary']['split_sum_first']['nominal_angle']
            mass = np.asarray(data['cohort_fraction'])[model, bucket]
            tail = np.asarray(data['tail1_probability'])[model, bucket, :, 0]
            axes[row_index, 0].plot(x, 100*mass, style, color='#246a9a', marker='o', label='All cohort, '+reference)
            axes[row_index, 0].plot(x, 100*mass*tail, style, color='#ba463b', marker='s', label='|residual| > 1 MeV, '+reference)
            valid = np.asarray(data['valid'])[model, bucket]
            for case, color, name in ((0, '#246a9a', 'Base'), (7, '#ba463b', 'BCS')):
                y = 1000*np.asarray(data['rms_MeV'])[model, bucket, :, case]
                axes[row_index, 1].plot(x, np.where(valid, y, np.nan), style, color=color, marker='o', label=name+', '+reference)
        for col in (0, 1):
            ax = axes[row_index, col]; panel = row_index*2+col
            ax.set_xticks(x, labels, rotation=25); ax.set_xlabel('Minimum nominal angle above foil plane / deg')
            ax.set_title(f'22({chr(97+panel)})  alpha{row_index}, m=2, frozen channel label')
            ax.grid(alpha=.2)
        axes[row_index, 0].set_ylabel('Fraction of fixed cohort / %')
        axes[row_index, 1].set_ylabel('Conditional RMS / keV'); axes[row_index, 1].set_yscale('log')
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    fig.suptitle('Angular localization of tails; ideal sharing with sum-first matching')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
