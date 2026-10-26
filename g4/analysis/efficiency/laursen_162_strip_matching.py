"""Observable-only front/back matching controls, not the historical AUSA code.

Input signals contain only strip numbers and calibrated deposited energies.
All interpretations use disjoint groups of signals. The supported families are
direct pairs, one-to-many sums (up to three hits), adjacent charge sharing,
and the three-edge 2-by-2 sum topology. Overlapping charge-sharing clouds are
not modeled by this family. Priority and ambiguity rules are explicit controls.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools

import numpy as np

POLICIES = ('sum_first', 'share_first', 'unique')


@dataclass(frozen=True)
class Primitive:
    front_mask: int
    back_mask: int
    hits: tuple  # (deposited energy MeV, front strip coordinate, back coordinate)
    cost: float
    complexity: int
    summed: int = 0
    shared: int = 0


@dataclass(frozen=True)
class Match:
    hits: tuple
    ambiguous: bool
    feasible: int
    unmatched: int
    summed: int = 0
    shared: int = 0


def signals(values):
    values = np.asarray(values, float).reshape((-1, 2))
    if (len(values) > 6 or not np.isfinite(values).all()
            or np.any(values[:, 1] <= 0) or np.any(values[:, 0] != np.floor(values[:, 0]))
            or np.any((values[:, 0] < 0) | (values[:, 0] >= 16))
            or len(set(values[:, 0])) != len(values)):
        raise ValueError('Expected at most six unique strips 0..15 with positive finite energies')
    return values[np.argsort(values[:, 0])]


def mask(indices):
    return sum(1 << i for i in indices)


@lru_cache(maxsize=4)
def cross_system(missing):
    edges = tuple(e for e in itertools.product(range(2), repeat=2) if e != missing)
    matrix = np.array([[float(i == row) for i, j in edges] for row in range(2)]
                      + [[float(j == row) for i, j in edges] for row in range(2)])
    return edges, matrix, np.linalg.pinv(matrix)


def primitives(front, back, tolerance, direct_only=False):
    out = []
    for i, (f, ef) in enumerate(front):
        for j, (b, eb) in enumerate(back):
            if abs(ef-eb) <= tolerance:
                out.append(Primitive(1 << i, 1 << j, ((float((ef+eb)/2), float(f), float(b)),),
                                     float(((ef-eb)/tolerance)**2), 0))
    if direct_only:
        return out
    for transpose, single, many in ((False, front, back), (True, back, front)):
        for i, (strip, energy) in enumerate(single):
            for size in (2, 3):
                for indices in itertools.combinations(range(len(many)), size):
                    selected = many[list(indices)]
                    total = selected[:, 1].sum()
                    if abs(energy-total) > tolerance:
                        continue
                    corrected_total = (energy+total)/2
                    split = selected[:, 1]*corrected_total/total
                    hits = tuple((float(e), float(p) if transpose else float(strip),
                                  float(strip) if transpose else float(p)) for e, (p, _) in zip(split, selected))
                    fm, bm = (mask(indices), 1 << i) if transpose else (1 << i, mask(indices))
                    cost = float(((energy-total)/tolerance)**2)
                    out.append(Primitive(fm, bm, hits, cost, size-1, summed=1))
                    if size == 2 and selected[1, 0]-selected[0, 0] == 1:
                        centre = float(selected[:, 0] @ selected[:, 1]/total)
                        hit = (float(corrected_total), centre if transpose else float(strip),
                               float(strip) if transpose else centre)
                        out.append(Primitive(fm, bm, (hit,), cost, 1, shared=1))
    for fi in itertools.combinations(range(len(front)), 2):
        for bi in itertools.combinations(range(len(back)), 2):
            f, b = front[list(fi)], back[list(bi)]
            ef, eb = f[:, 1].sum(), b[:, 1].sum()
            if abs(ef-eb) > tolerance:
                continue
            if f[1, 0]-f[0, 0] == 1 and b[1, 0]-b[0, 0] == 1:
                hit = (float((ef+eb)/2), float(f[:, 0] @ f[:, 1]/ef), float(b[:, 0] @ b[:, 1]/eb))
                out.append(Primitive(mask(fi), mask(bi), (hit,), float(((ef-eb)/tolerance)**2), 2, shared=1))
            observed = np.r_[f[:, 1], b[:, 1]]
            for missing in itertools.product(range(2), repeat=2):
                edges, matrix, inverse = cross_system(missing)
                energies = inverse @ observed
                if np.any(energies <= 1e-12):
                    continue
                residual = matrix @ energies-observed
                if np.max(abs(residual)) > tolerance:
                    continue
                hits = tuple((float(e), float(f[i, 0]), float(b[j, 0])) for e, (i, j) in zip(energies, edges))
                out.append(Primitive(mask(fi), mask(bi), hits, float(np.sum((residual/tolerance)**2)), 2, summed=1))
    return out


def match(front, back, tolerance=.150, direct_only=False):
    """Return all three policy results from the same observed signals.

    First maximize the number of consumed signals, then minimize non-direct
    grouping complexity. Sum/share controls choose their structural priority,
    followed by minimum energy residual. Residual ties remain ambiguous.
    Unique requires a single physical interpretation at the tolerance, without
    using energy residuals to resolve multiple feasible interpretations.
    No truth, expected particle count, reaction energy or kinematic cut enters.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Matching tolerance must be finite and positive')
    front, back = signals(front), signals(back)
    nf, nb = len(front), len(back)
    if not nf or not nb:
        result = Match((), False, 1, nf+nb)
        return dict.fromkeys(POLICIES, result)
    if nf == nb == 1:
        f, ef = front[0]; b, eb = back[0]
        good = abs(ef-eb) <= tolerance
        result = Match(((float((ef+eb)/2), float(f), float(b)),) if good else (), False, 1, 0 if good else 2)
        return dict.fromkeys(POLICIES, result)
    elements = primitives(front, back, tolerance, direct_only)
    by_front = [[p for p in elements if p.front_mask & (1 << i)] for i in range(nf)]
    best_rank = None
    candidates = {}

    def visit(remaining, used_front, used_back, hits, cost, complexity, summed, shared):
        nonlocal best_rank, candidates
        if not remaining:
            rank = (-used_front.bit_count()-used_back.bit_count(), complexity)
            if best_rank is not None and rank > best_rank:
                return
            if best_rank is None or rank < best_rank:
                best_rank, candidates = rank, {}
            ordered = tuple(sorted(hits))
            signature = tuple(tuple(round(v, 12) for v in h) for h in ordered)
            entry = (ordered, cost, summed, shared)
            if signature not in candidates or cost < candidates[signature][1]:
                candidates[signature] = entry
            return
        first = (remaining & -remaining).bit_length()-1
        visit(remaining & ~(1 << first), used_front, used_back, hits, cost, complexity, summed, shared)
        for p in by_front[first]:
            if p.front_mask & remaining != p.front_mask or p.back_mask & used_back:
                continue
            visit(remaining ^ p.front_mask, used_front | p.front_mask, used_back | p.back_mask,
                  hits+p.hits, cost+p.cost, complexity+p.complexity, summed+p.summed, shared+p.shared)

    visit((1 << nf)-1, 0, 0, (), 0., 0, 0, 0)
    feasible = list(candidates.values())
    unmatched = nf+nb+best_rank[0]
    result = {}
    for policy in POLICIES:
        allowed = feasible
        if policy != 'unique':
            priorities = [x[3] if policy == 'sum_first' else -x[3] for x in allowed]
            minimum = min(priorities)
            allowed = [x for x, priority in zip(allowed, priorities) if priority == minimum]
            cost = min(x[1] for x in allowed)
            allowed = [x for x in allowed if np.isclose(x[1], cost, rtol=1e-10, atol=1e-12)]
        ambiguous = len(allowed) != 1
        hits, _, summed, shared = allowed[0] if not ambiguous else ((), 0, 0, 0)
        result[policy] = Match(hits, ambiguous, len(feasible), unmatched, summed, shared)
    return result


def split_charge(coordinate, pitch_mm=3.125, width_mm=0.):
    """Ideal charge-conserving linear split at internal strip boundaries.

    coordinate is measured in strip pitches, zero at the active-area edge.
    width is a declared response control, not a measured collection profile.
    """
    coordinate = np.asarray(coordinate, float)
    if (not np.isfinite(coordinate).all() or np.any((coordinate < 0) | (coordinate >= 16))
            or not 0 <= width_mm < pitch_mm):
        raise ValueError('Invalid active-area coordinate or sharing width')
    first = np.floor(coordinate).astype(int)
    second, fraction = first.copy(), np.zeros_like(coordinate)
    if width_mm:
        boundary = np.rint(coordinate).astype(int)
        half = width_mm/(2*pitch_mm)
        inside = (boundary > 0) & (boundary < 16) & (abs(coordinate-boundary) < half)
        first = np.where(inside, boundary-1, first)
        second = np.where(inside, boundary, second)
        fraction = np.where(inside, (coordinate-boundary+half)/(2*half), fraction)
    return first, second, fraction
