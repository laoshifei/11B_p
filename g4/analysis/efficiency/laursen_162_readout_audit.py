"""Audit exclusive multiplicity budgets and strip topology in the frozen response.

This diagnoses which hits share readout channels. It does not implement an
experimental front/back matching algorithm or infer missing calibration data.
"""
from __future__ import annotations

import argparse
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

KEY = 'codex_laursen_162_readout_audit'
TOPOLOGY = ('no_shared_strip', 'front_only', 'back_only',
            'both_sides_different_pairs', 'same_pixel')
VERSION = '2026-10-26-v1-multiplicity-budget-and-strip-topology'


def strip_topology(detector, pixel, pattern):
    """Disjoint true-hit categories, not a reconstruction success decision."""
    detector, pixel, pattern = np.asarray(detector), np.asarray(pixel), np.asarray(pattern, bool)
    if detector.ndim != 2 or detector.shape[1] != 3 or pixel.shape != (*detector.shape, 2):
        raise ValueError('Expected three detector/pixel labels per event')
    if pattern.shape != (3,) or pattern.sum() not in (2, 3):
        raise ValueError('Expected a reconstructable particle pattern')
    front = np.zeros(len(detector), bool)
    back = np.zeros_like(front)
    same_pixel = np.zeros_like(front)
    for i, j in itertools.combinations(np.flatnonzero(pattern), 2):
        same_detector = (detector[:, i] >= 0) & (detector[:, i] == detector[:, j])
        f = same_detector & (pixel[:, i, 0] == pixel[:, j, 0])
        b = same_detector & (pixel[:, i, 1] == pixel[:, j, 1])
        front |= f
        back |= b
        same_pixel |= f & b
    category = front.astype(int) + 2 * back.astype(int)
    category[same_pixel] = 4
    return category


def channel_matrix(detector, pixel):
    """Known-association noiseless strip sums A @ particle_energies.

    This matrix is only an identifiability control; the experimental algorithm
    must infer the association, account for thresholds and allow charge sharing.
    """
    detector, pixel = np.asarray(detector), np.asarray(pixel)
    if detector.ndim != 1 or pixel.shape != (len(detector), 2) or not 1 <= len(detector) <= 3:
        raise ValueError('Expected one to three measured particles')
    if np.any(detector < 0) or np.any(pixel < 0):
        raise ValueError('Only detected particles have strip sums')
    labels = sorted({(int(d), side, int(p[side])) for d, p in zip(detector, pixel) for side in (0, 1)})
    return np.array([[float((int(d), side, int(p[side])) == label)
                      for d, p in zip(detector, pixel)]
                     for label in labels for side in [label[1]]])


def reduce_reference(reference, tables):
    records, diagnostics = response.fold(reference['kinetic'], reference['directions'],
        reference['origins'], reference['depth'], reference['noise'], response.Response(), tables)
    geometry = diagnostics['geometry']
    multiplicity = (geometry['detector'] >= 0).sum(axis=1)
    weights = reference['weights']
    sums = weights.sum(axis=0)
    selected = np.zeros((len(weights), 2))
    preselection = np.zeros_like(selected)
    topology = np.zeros((weights.shape[1], 2, len(TOPOLOGY)))
    topology_pre = np.zeros_like(topology)
    for record in records:
        k = record['multiplicity'] - 2
        p, before = record['probability'], record['preselection']
        category = strip_topology(geometry['detector'], geometry['pixel'], record['pattern'])
        if not np.array_equal(category > 0, record['ambiguous']):
            raise ArithmeticError('Shared-strip definition differs from frozen response')
        selected[:, k] += p
        preselection[:, k] += before
        for c in range(len(TOPOLOGY)):
            topology[:, k, c] += weights.T @ (p * (category == c)) / sums
            topology_pre[:, k, c] += weights.T @ (before * (category == c)) / sums
    geometric = np.stack([weights.T @ (multiplicity == k) / sums for k in range(4)], axis=1)
    flow = np.zeros((weights.shape[1], 4, 3))
    for k in range(4):
        flow[:, k, :2] = weights.T @ (selected * (multiplicity == k)[:, None]) / sums[:, None]
    flow[:, :, 2] = geometric - flow[:, :, :2].sum(axis=2)
    if np.any(flow < -1e-12) or not np.allclose(flow.sum(axis=(1, 2)), 1):
        raise ArithmeticError('Event-budget flow is not conserved')
    if not np.allclose(topology.sum(axis=2), weights.T @ selected / sums[:, None]):
        raise ArithmeticError('Topology categories do not partition accepted probability')
    return dict(model_indices=reference['model_indices'], provenance=reference['provenance'],
        generated=geometric, flow=flow, selected=weights.T @ selected / sums[:, None],
        preselection=weights.T @ preselection / sums[:, None],
        selected_topology=topology, preselection_topology=topology_pre,
        invalid_corrections=diagnostics['invalid_corrections'])


def run(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve completed audit; use its cache')
    frozen = parent.load_cached()
    for name, value in frozen['configuration']['source_sha256'].items():
        if parent.sha256(parent.ROOT / name) != value:
            raise ValueError('Frozen numerical source changed; do not silently replay')
    metadata = json.loads(parent.parent.NOTEBOOK.read_text())['metadata']
    tables = response.RangeTables(frozen['stopping'])
    started = datetime.now().astimezone().isoformat()
    begin = time.monotonic()
    print('Replaying the same four preselection references and response streams.', flush=True)
    amplitude = parent.parent.AmplitudeModel()
    results = {}
    for name in ('old', 'new'):
        groups = []
        for channel in (0, 2):
            source = parent.source_reference(metadata, name, channel, amplitude)
            groups.append(reduce_reference(source, tables))
            print(name, channel, 'topology and multiplicity flow complete', flush=True)
        summary = {key: np.concatenate([g[key] for g in groups], axis=0) for key in
                   ('generated', 'flow', 'selected', 'preselection', 'selected_topology', 'preselection_topology')}
        if not np.allclose(summary['selected'], frozen['summary'][name]['nominal_assumptions']['efficiency'], atol=1e-12, rtol=1e-10):
            raise ArithmeticError('Nominal response did not reproduce')
        geometry_expected = frozen['summary'][name]['geometry']['efficiency']
        if not np.allclose(summary['generated'][:, 2:], geometry_expected, atol=1e-12, rtol=1e-10):
            raise ArithmeticError('Geometric multiplicity did not reproduce')
        prior = np.concatenate([np.asarray(g['cases']['nominal_assumptions']['ambiguous_accepted'])
                                for g in frozen['references'][name]], axis=0)
        if not np.allclose(summary['selected_topology'][:, :, 1:].sum(axis=2), prior, atol=1e-12, rtol=1e-10):
            raise ArithmeticError('Shared-strip accepted mass did not reproduce')
        results[name] = dict(summary=summary, groups=groups)
    result = parent.parent.plain(dict(version=VERSION, parent_checksum=frozen['checksum'],
        source_sha256=parent.sha256(__file__), source_snapshot=Path(__file__).read_text(),
        interpreter=sys.executable, packages=frozen['configuration']['packages'],
        source_configuration=frozen['configuration'], topology_labels=TOPOLOGY,
        flow_axes=['model', 'geometric_multiplicity_0_1_2_3', 'selected_m2_m3_or_rejected'],
        results=results, started_at=started, completed_at=datetime.now().astimezone().isoformat(),
        elapsed_seconds=time.monotonic()-begin,
        scope='Known true-hit topology and accepted-event budgets; not electronics, matching efficiency, or a new branch fit'))
    result['checksum'] = parent.digest(result)
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    print('Readout audit complete:', result['completed_at'], flush=True)


def load_cached(notebook=parent.NOTEBOOK):
    state = json.loads(Path(notebook).read_text())['metadata'][KEY]
    if parent.digest({k: v for k, v in state.items() if k != 'checksum'}) != state['checksum']:
        raise ValueError('Readout audit checksum changed')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
