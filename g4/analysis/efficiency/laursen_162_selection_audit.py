"""Versioned selection controls supported by Laursen's PhD dissertation.

The frozen 2016-paper response remains unchanged. This module measures the
effects of the explicit detector-plane equation, a permutation-invariant
leading-energy plane convention, hard thresholds, and relative-energy gates.
It does not reproduce strip matching, the authors' full geometry, or a fit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.efficiency import laursen_162_study as parent
from analysis.efficiency import laursen_162_response as response

KEY = 'codex_laursen_162_selection_audit'
VERSION = '2026-10-26-v1-thesis-selection-controls'
THESIS_SHA256 = '4af7788388896ce395b68304fcc827e691f6be4d99a4396693ffad961639107d'
# Approximate readings of Fig. 5.9 (PDF p95, printed p78), not exact settings.
# Rows: narrow, central, wide; columns: multiplicity two, three. MeV.
GATES = np.array([[.350, .200], [.400, .220], [.450, .240]])
STAGES = ('readout', 'finite_correction', 'total_momentum', 'angle_sum',
          'plane', 'px_control', 'detector_control')
PAIRS = ((0, 1), (1, 2), (2, 0))


@dataclass(frozen=True)
class Selection:
    name: str = 'legacy_replay'
    geometry: str = 'legacy'
    plane: str = 'legacy'
    hard_threshold: bool = False
    threshold_MeV: tuple = (.200, .350)
    px_limit_MeV: float | None = None
    one_each_detector: bool = False

    def __post_init__(self):
        if self.geometry not in ('legacy', 'equation_4_1'):
            raise ValueError('Unknown geometry convention')
        if self.plane not in ('legacy', 'highest_energy'):
            raise ValueError('Unknown plane convention')
        if len(self.threshold_MeV) != 2 or min(self.threshold_MeV) < 0:
            raise ValueError('Invalid thresholds')
        if self.px_limit_MeV is not None and self.px_limit_MeV <= 0:
            raise ValueError('Invalid component cut')


def cases():
    base = Selection()
    plane = replace(base, name='leading_plane', plane='highest_energy')
    combined = replace(plane, name='combined', geometry='equation_4_1', hard_threshold=True)
    return [base, plane, replace(plane, name='hard_threshold', hard_threshold=True),
            replace(base, name='geometry_equation', geometry='equation_4_1'), combined,
            replace(combined, name='px_control', px_limit_MeV=30.),
            replace(combined, name='one_each_detector', one_each_detector=True),
            replace(combined, name='hard_low', threshold_MeV=(.100, .200)),
            replace(combined, name='hard_high', threshold_MeV=(.300, .500))]


def basis(convention):
    if convention == 'legacy':
        return response.NORMALS, response.HORIZONTAL, response.VERTICAL
    if convention != 'equation_4_1':
        raise ValueError('Unknown geometry convention')
    # Thesis Eq. (4.1), theta_V=0: dx/dx'=-cos(theta_D),
    # dz/dx'=-sin(theta_D), dy/dy'=-1. Retain the paper's 50-mm
    # square and 16 equal pixels; this is an orientation-only control.
    beta = response.BETAS
    normal = np.column_stack([np.sin(beta), np.zeros(2), -np.cos(beta)])
    horizontal = np.column_stack([-np.cos(beta), np.zeros(2), -np.sin(beta)])
    return normal, horizontal, -response.VERTICAL


def trace(directions, origins, convention):
    if convention == 'legacy':
        return response.trace(directions, origins)
    directions, origins = np.asarray(directions, float), np.asarray(origins, float)
    if (directions.ndim != 3 or directions.shape[1:] != (3, 3)
            or origins.shape != (len(directions), 3)
            or not np.isfinite(origins).all() or not np.isfinite(directions).all()
            or not np.allclose(np.linalg.norm(directions, axis=-1), 1)):
        raise ValueError('Expected three finite unit directions and one origin per event')
    normals, horizontal, vertical = basis(convention)
    shape = directions.shape[:2]
    distance, detector = np.full(shape, np.inf), np.full(shape, -1, int)
    pixel = np.full((*shape, 2), -1, int)
    reconstructed = directions.copy()
    for d, (centre, normal, h) in enumerate(zip(response.CENTRES, normals, horizontal)):
        denominator = directions @ normal
        numerator = (centre - origins) @ normal
        t = np.divide(numerator[:, None], denominator, out=np.full(shape, np.inf),
                      where=abs(denominator) > 1e-14)
        relative = origins[:, None, :] + np.where(np.isfinite(t), t, 0.)[:, :, None] * directions - centre
        x, y = relative @ h, relative @ vertical
        half, pitch = response.HALF_SIZE, response.PITCH
        hit = (t > 0) & (t < distance) & (x >= -half) & (x < half) & (y >= -half) & (y < half)
        ix, iy = np.floor((x + half) / pitch).astype(int), np.floor((y + half) / pitch).astype(int)
        position = centre + ((ix + .5) * pitch - half)[:, :, None] * h
        position += ((iy + .5) * pitch - half)[:, :, None] * vertical
        distance[hit], detector[hit] = t[hit], d
        pixel[hit] = np.stack([ix, iy], axis=-1)[hit]
        reconstructed[hit] = response.unit(position)[hit]
    return dict(detector=detector, pixel=pixel, direction=reconstructed, distance=distance)


def columns(directions, detector, depth, config, tables, convention):
    boron, carbon, _ = response.columns(directions, detector, depth, config, tables)
    normals, _, _ = basis(convention)
    d = np.maximum(detector, 0)
    incidence = np.maximum(abs(np.sum(directions * normals[d], axis=-1)), 1e-14)
    silicon = np.asarray(config.dead_um)[d] * 1e-4 * tables.table['materials']['G4_Si']['density_g_cm3'] / incidence
    return boron, carbon, np.where(detector >= 0, silicon, 0.)


def pair_relative_energy(momentum):
    """Thesis Eq. (5.6)/paper Eq. (6), evaluated for all three CM pairs."""
    momentum = np.asarray(momentum, float)
    if momentum.ndim != 3 or momentum.shape[1:] != (3, 3) or not np.isfinite(momentum).all():
        raise ValueError('Expected finite N x 3 x 3 momenta')
    return np.stack([np.sum((momentum[:, i] - momentum[:, j])**2, axis=1) / (4 * response.MASS)
                     for i, j in PAIRS], axis=1)


def plane_deviation(momentum, convention):
    """Absolute angle out of the plane in radians.

    Highest reconstructed CM energy is an explicit label convention, not a
    claim to know the physical primary alpha in a Bose-symmetrized event.
    Equal leading energies use the most restrictive tied angle, making this
    selection invariant to arbitrary storage permutations. Undefined planes
    are rejected; the legacy branch preserves its original zero-normal rule.
    """
    direction = response.unit(momentum)
    if convention == 'legacy':
        normal = response.unit(np.cross(direction[:, 0], direction[:, 1]))
        return np.arcsin(np.clip(abs(np.sum(normal * direction[:, 2], axis=1)), 0, 1))
    if convention != 'highest_energy':
        raise ValueError('Unknown plane convention')
    magnitudes = np.linalg.norm(momentum, axis=-1)
    candidates = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        normal = np.cross(direction[:, j], direction[:, k])
        length = np.linalg.norm(normal, axis=1)
        sine = np.divide(abs(np.sum(normal * direction[:, i], axis=1)), length,
                         out=np.ones(len(length)), where=length > 1e-14)
        angle = np.arcsin(np.clip(sine, 0, 1))
        angle[magnitudes[:, i] <= 0] = np.pi / 2
        candidates.append(angle)
    leading = np.isclose(magnitudes, magnitudes.max(axis=1)[:, None], rtol=1e-12, atol=1e-12)
    return np.max(np.where(leading, np.stack(candidates, axis=1), 0.), axis=1)


def selection_masks(kinetic, momentum, invalid, pattern, detector, selection):
    """Cumulative cuts, keeping m3 rejections separate from m2 detections."""
    valid = ~np.any(invalid[:, pattern], axis=1) & (kinetic.sum(axis=1) > 0)
    masks = [np.ones(len(kinetic), bool), valid.copy()]
    triple = pattern.sum() == 3
    total = momentum.sum(axis=1)
    if triple:
        valid &= np.linalg.norm(total, axis=1) < 50.
    masks.append(valid.copy())
    if triple:
        unit = response.unit(momentum)
        angles = sum(np.arccos(np.clip(np.sum(unit[:, i] * unit[:, j], axis=1), -1, 1)) for i, j in PAIRS)
        valid &= abs(angles - 2 * np.pi) <= np.deg2rad(10.)
    masks.append(valid.copy())
    if triple:
        valid &= plane_deviation(momentum, selection.plane) <= np.deg2rad(10.)
    masks.append(valid.copy())
    if triple and selection.px_limit_MeV is not None:
        valid &= abs(total[:, 0]) < selection.px_limit_MeV
    masks.append(valid.copy())
    if not triple and selection.one_each_detector:
        measured = detector[:, pattern]
        valid &= (measured[:, 0] >= 0) & (measured[:, 1] >= 0) & (measured[:, 0] != measured[:, 1])
    masks.append(valid.copy())
    return np.stack(masks, axis=1)


def fold(reference, tables, selection):
    """Freeze all unspecified response inputs at the prior study's values."""
    config = response.Response()
    kinetic, directions = reference['kinetic'], reference['directions']
    geometry = trace(directions, reference['origins'], selection.geometry)
    detector = geometry['detector']
    hit = detector >= 0
    parent_momentum = (np.sqrt(kinetic * (kinetic + 2 * response.MASS))[..., None] * directions).sum(axis=1)
    parent_energy = (kinetic + response.MASS).sum(axis=1)
    b, c, si = columns(directions, detector, reference['depth'], config, tables, selection.geometry)
    entering = tables.remove(tables.remove(tables.remove(kinetic, b, 'G4_B'), c, 'G4_C'), si, 'G4_Si')
    light = tables.materials['G4_Si']['light'](entering)
    measured = np.maximum(0., light + config.fwhm_MeV / 2.354820045 * reference['noise'])
    adc = hit & (entering > 0) & (measured >= np.asarray(config.adc_MeV)[np.maximum(detector, 0)])
    midpoint = np.asarray(selection.threshold_MeV)[np.maximum(detector, 0)]
    trigger = (measured >= midpoint).astype(float) if selection.hard_threshold else response.trigger_efficiency(measured, midpoint, config.trigger_width_MeV)
    trigger *= hit * (entering > 0)
    probabilities = response.hit_pattern_probabilities(adc, trigger, 'per_particle')
    rec_direction = geometry['direction']
    b, c, si = columns(rec_direction, detector, np.full(len(kinetic), .5), config, tables, selection.geometry)
    recovered = tables.materials['G4_Si']['inverse_light'](measured)
    corrected = tables.restore(tables.restore(tables.restore(recovered, si, 'G4_Si'), c, 'G4_C'), b, 'G4_B')
    invalid = hit & ~np.isfinite(corrected)
    corrected = np.nan_to_num(corrected, nan=0.)
    records = []
    for bit in (3, 5, 6, 7):
        pattern = np.array([(bit >> i) & 1 for i in range(3)], bool)
        cm, pcm = response.reconstruct(corrected, rec_direction, parent_energy, parent_momentum, pattern)
        masks = selection_masks(cm, pcm, invalid, pattern, detector, selection)
        records.append(dict(multiplicity=int(pattern.sum()), pattern=pattern,
            stages=probabilities[:, bit, None] * masks, min_pair_energy=pair_relative_energy(pcm).min(axis=1)))
    return records, geometry


def moments(weights, values):
    sums = weights.sum(axis=0)
    flat = values.reshape((len(values), -1))
    means = weights.T @ flat / sums[:, None]
    errors = np.stack([np.sqrt(np.sum((weights[:, m, None] * (flat - means[m]))**2, axis=0)) / sums[m]
                       for m in range(weights.shape[1])])
    shape = (weights.shape[1], *values.shape[1:])
    return means.reshape(shape), errors.reshape(shape)


def reduce_reference(reference, tables):
    weights = reference['weights']
    results, baseline = {}, None
    for selection in cases():
        records, geometry = fold(reference, tables, selection)
        flow = np.zeros((len(weights), 2, len(STAGES)))
        assigned = np.zeros((len(weights), 2, len(GATES), 2))
        for record in records:
            k = record['multiplicity'] - 2
            flow[:, k] += record['stages']
            p = record['stages'][:, -1]
            is_ground = record['min_pair_energy'][:, None] < GATES[:, k]
            assigned[:, k, :, 0] += p[:, None] * is_ground
            assigned[:, k, :, 1] += p[:, None] * ~is_ground
        selected = flow[:, :, -1]
        if (np.any(np.diff(flow, axis=2) > 1e-12) or np.any(selected.sum(axis=1) > 1 + 1e-12)
                or not np.allclose(assigned.sum(axis=-1), selected[:, :, None])):
            raise ArithmeticError('Selection/gate event budget does not close')
        if baseline is None:
            baseline = selected.copy()
        efficiency, standard_error = moments(weights, selected)
        difference, difference_error = moments(weights, selected - baseline)
        classified, classified_error = moments(weights, assigned)
        geometric = np.stack([(geometry['detector'] >= 0).sum(axis=1) == k for k in range(4)], axis=1)
        results[selection.name] = dict(efficiency=efficiency, importance_standard_error=standard_error,
            paired_minus_legacy=difference, paired_standard_error=difference_error,
            cutflow=moments(weights, flow)[0], classification=classified,
            classification_standard_error=classified_error, geometric_multiplicity=moments(weights, geometric)[0])
        print(reference['provenance']['root'].split('/')[-1], selection.name, 'complete', flush=True)
    return dict(provenance=reference['provenance'], model_indices=reference['model_indices'], cases=results)


def run(output, thesis_pdf):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve the completed result; read its cache')
    if parent.sha256(thesis_pdf) != THESIS_SHA256:
        raise ValueError('Thesis PDF fingerprint differs from audited edition')
    frozen = parent.load_cached()
    for name, value in frozen['configuration']['source_sha256'].items():
        if parent.sha256(parent.ROOT / name) != value:
            raise ValueError('Frozen numerical source changed; explicit reconciliation required')
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    started = datetime.now().astimezone().isoformat()
    begin = time.monotonic()
    amplitude = parent.parent.AmplitudeModel()
    tables = response.RangeTables(frozen['stopping'])
    results = {}
    for name in ('old', 'new'):
        groups = [reduce_reference(parent.source_reference(metadata, name, channel, amplitude), tables) for channel in (0, 2)]
        summary = {case.name: {key: np.concatenate([g['cases'][case.name][key] for g in groups], axis=0)
                              for key in groups[0]['cases'][case.name]} for case in cases()}
        if not np.allclose(summary['legacy_replay']['efficiency'], frozen['summary'][name]['nominal_assumptions']['efficiency'], atol=1e-12, rtol=1e-10):
            raise ArithmeticError('Legacy replay does not reproduce frozen efficiencies')
        results[name] = dict(summary=summary, provenance=[g['provenance'] for g in groups])
    normals, horizontal, _ = basis('equation_4_1')
    result = parent.parent.plain(dict(version=VERSION, parent_checksum=frozen['checksum'],
        source_sha256=parent.sha256(__file__), source_snapshot=Path(__file__).read_text(),
        source_configuration=frozen['configuration'], interpreter=sys.executable,
        cases=[asdict(c) for c in cases()], stages=STAGES, gates_MeV=GATES,
        gate_scope='Approximate figure readings; low/high are sensitivity settings, not confidence intervals',
        classification_axes=['model', 'multiplicity_2_3', 'narrow_central_wide_gate', 'assigned_ground_excited'],
        thesis=dict(author='Kasper Lind Laursen', title='Broad resonances in 12C and 20Ne',
            institution='Aarhus University', cover_date='December 2014', catalogue_year=2015, doi=None,
            url='https://phys.au.dk/fileadmin/user_upload/Phd_thesis/ThesisKasperLindLaursen.pdf',
            sha256=THESIS_SHA256, pdf_pages=[69, 71, 84, 85, 86, 87, 88, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 107, 108, 109],
            visually_checked_pdf_pages=[69, 91, 92, 95, 108], review_status='Targeted primary-source audit, no human adoption recorded',
            set9_efficiency=[[.298, .143], [.288, .0062]], set9_total_error=[[.013, .009], [.012, .0008]],
            efficiency_axes=['ground_excited', 'multiplicity_2_3'],
            version_scope='Thesis Table 6.2, not a replacement for the 2016 journal results'),
        equation_geometry=dict(normals=normals, horizontal=horizontal,
            solid_angle_fraction=[response.rectangle_solid_angle(c, n, h) / (4*np.pi)
                                  for c, n, h in zip(response.CENTRES, normals, horizontal)]),
        results=results, started_at=started, completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=time.monotonic()-begin,
        scope='Conditional analysis controls; no strip matching, experimental-data fit or physical-branch constraint'))
    result['checksum'] = parent.digest(result)
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    print('Selection audit complete:', result['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Selection audit checksum changed')
    return state


def plot_efficiency(state):
    """Figure 14: independent replay and thesis-version comparison only."""
    import matplotlib.pyplot as plt
    names = ('legacy_replay', 'leading_plane', 'hard_threshold', 'geometry_equation', 'combined')
    labels = ('Legacy', 'Leading plane', 'Hard threshold', 'Plane equation', 'Combined')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, model in enumerate((0, 5)):
        for k in (0, 1):
            ax = axes[row, k]
            centre = 100 * state['thesis']['set9_efficiency'][row][k]
            error = 100 * state['thesis']['set9_total_error'][row][k]
            ax.axhspan(centre-error, centre+error, color='#c2b69b', alpha=.35, label='Thesis set 9, total error')
            for reference, dx, marker, color in (('old', -.08, 'o', '#246a9a'), ('new', .08, 's', '#bd453e')):
                summary = state['results'][reference]['summary']
                y = [100 * summary[n]['classification'][model][k][1][row] for n in names]
                err = [100 * summary[n]['classification_standard_error'][model][k][1][row] for n in names]
                ax.errorbar(np.arange(len(names))+dx, y, yerr=err, fmt=marker, color=color, ms=4, capsize=2, label=reference+' reference')
            ax.set_xticks(range(len(names)), labels, rotation=25, ha='right')
            ax.set_ylabel('Selected in own channel / generated (%)')
            ax.set_title(f'14({"abcd"[2*row+k]})  '+('alpha0' if model==0 else 'R + phases + Coulomb')+f', multiplicity {k+2}')
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=8)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--thesis-pdf', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.thesis_pdf)
