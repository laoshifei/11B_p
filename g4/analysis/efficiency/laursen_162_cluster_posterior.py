"""Conditional two-hit material response from each observed strip exactly once.

This is a fixed-topology control. The source-energy prior is the product of
the declared single-alpha Flat T supports, not a full three-alpha amplitude.
No generated channel, true positions, or event energy sum is an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools

import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr
from scipy.stats import qmc

from . import laursen_162_pixel_posterior as single

material, matching, response = single.material, single.matching, single.response


@dataclass(frozen=True)
class Quadrature:
    main_power: int = 8
    coarse_power: int = 6
    charge_order: int = 4
    high_charge_order: int = 8
    seed: int = 1622026102711
    alternate_seed: int = 1622026102712
    sharing_width_mm: float = .100


VARIANTS = ('coarse', 'main', 'alternate', 'high_charge_order')


def eligible_cluster(matched, observed, policy_agreement):
    """D1 two-hit shared-strip cluster with complete support and policy agreement."""
    if (not policy_agreement or matched['ambiguous'] or matched['multiplicity'] not in (2, 3)):
        return np.empty(0, int), 'unresolved_topology'
    hits = np.flatnonzero(np.asarray(matched['detector']) == 0)
    if len(hits) != 2:
        return np.empty(0, int), 'not_two_d1_hits'
    pixel = np.asarray(matched['pixel'])[hits]
    if not np.all(abs(pixel-np.rint(pixel)) < 1e-10):
        return np.empty(0, int), 'fractional_pixel'
    pixel = np.rint(pixel).astype(int)
    if len(set(map(tuple, pixel))) != 2:
        return np.empty(0, int), 'duplicate_pixel'
    for side in (0, 1):
        if set(np.flatnonzero(observed[0, side] > 0)) != set(pixel[:, side]):
            return np.empty(0, int), 'unexplained_strip'
    if all(len(set(pixel[:, side])) == 2 for side in (0, 1)):
        return np.empty(0, int), 'neighbors_only'
    return hits, 'eligible'


def strip_intervals(strip, width, extra_cut=None):
    """Partition exactly at the linear sharing breakpoints and an optional foil cut."""
    if not isinstance(strip, (int, np.integer)) or not 0 <= strip < 16 or not 0 <= width < response.PITCH:
        raise ValueError('Invalid strip or sharing width')
    low = max(-response.HALF_SIZE, strip*response.PITCH-response.HALF_SIZE-width/2)
    high = min(response.HALF_SIZE, (strip+1)*response.PITCH-response.HALF_SIZE+width/2)
    cuts = [low, high]
    for edge in (strip, strip+1):
        if 0 < edge < 16:
            centre = edge*response.PITCH-response.HALF_SIZE
            cuts.extend(point for point in (centre-width/2, centre+width/2) if low < point < high)
    if extra_cut is not None and low < extra_cut < high:
        cuts.append(float(extra_cut))
    cuts = np.unique(cuts)
    return tuple(zip(cuts[:-1], cuts[1:]))


@lru_cache(maxsize=16)
def unit_points(power, seed):
    return qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(power)


@lru_cache(maxsize=8)
def tensor_rule(order):
    if order < 2:
        raise ValueError('Need at least two charge nodes per dimension')
    points, weights = np.polynomial.legendre.leggauss(order)
    points, weights = (points+1)/2, weights/2
    grid = np.array(list(itertools.product(range(order), repeat=2)))
    return points[grid], np.prod(weights[grid], axis=1)


def linear_gaussian(design, measured, sigma):
    """One row per physical observed strip, including a summed strip only once."""
    design, measured = np.asarray(design, float), np.asarray(measured, float)
    if (design.ndim != 3 or design.shape[2] != 2 or measured.shape != (design.shape[1],)
            or not np.isfinite(design).all() or not np.isfinite(measured).all() or sigma <= 0):
        raise ValueError('Expected a finite two-charge design and observed strip vector')
    precision = np.einsum('gsi,gsj->gij', design, design)
    determinant = np.linalg.det(precision)
    good = (determinant > 1e-12) & np.all(design.sum(axis=2) > 0, axis=1)
    safe = np.where(good[:, None, None], precision, np.eye(2))
    inverse = np.linalg.inv(safe)
    mean = np.einsum('gij,gsj,s->gi', inverse, design, measured)
    covariance = sigma*sigma*inverse
    residual = np.einsum('gsi,gi->gs', design, mean)-measured
    chi2 = np.sum(residual*residual, axis=1)/(sigma*sigma)
    return mean, covariance, chi2, good


def bounded_gaussian_nodes(mean, covariance, maximum, order):
    """Sequential conditional-normal transform, including both truncation probabilities."""
    points, weights = tensor_rule(order)
    factor = np.linalg.cholesky(covariance)
    n, count = len(mean), len(points)
    z = np.zeros((n, count, 2)); probability = np.ones((n, count))
    for axis in (0, 1):
        conditional = mean[:, axis, None]+np.einsum('gi,gni->gn', factor[:, axis, :axis], z[:, :, :axis])
        scale = factor[:, axis, axis, None]
        low, high = ndtr(-conditional/scale), ndtr((maximum[:, axis, None]-conditional)/scale)
        span = np.maximum(0., high-low)
        quantile = low+span*points[None, :, axis]
        z[:, :, axis] = ndtri(np.clip(quantile, np.nextafter(0., 1.), np.nextafter(1., 0.)))
        probability *= span
    charge = mean[:, None, :]+np.einsum('gij,gnj->gni', factor, z)
    return charge, probability*weights


class LogMoments:
    def __init__(self):
        self.offset = -np.inf
        self.total = 0.; self.ray_square = 0.
        self.first = np.zeros(2); self.second = np.zeros((2, 2))

    def add(self, log_weight, kinetic):
        finite = np.isfinite(log_weight)
        if not finite.any():
            return
        current = float(log_weight[finite].max())
        offset = max(current, self.offset)
        rescale = np.exp(self.offset-offset)
        self.total *= rescale; self.first *= rescale; self.second *= rescale
        self.ray_square *= rescale*rescale
        weight = np.exp(log_weight-offset)
        value = np.where(finite[..., None], kinetic, 0.)
        self.total += weight.sum()
        self.first += np.einsum('gn,gni->i', weight, value)
        self.second += np.einsum('gn,gni,gnj->ij', weight, value, value)
        self.ray_square += np.sum(weight.sum(axis=1)**2)
        self.offset = offset

    def finish(self):
        if not self.total > 0:
            return dict(valid=False, mean=np.zeros(2), covariance=np.zeros((2, 2)), effective_rays=0., log_normalization=None)
        mean = self.first/self.total
        covariance = self.second/self.total-np.outer(mean, mean)
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-8:
            raise ArithmeticError('Posterior energy covariance is not positive semidefinite')
        return dict(valid=True, mean=mean, covariance=covariance,
            effective_rays=self.total*self.total/self.ray_square,
            log_normalization=self.offset+np.log(self.total))


class ClusterPosterior:
    def __init__(self, tables, quadrature=Quadrature()):
        self.tables, self.quadrature = tables, quadrature
        self.single = single.PixelPosterior(tables)
        self.sigma = self.single.sigma

    def rectangles(self, detector, pixels):
        _, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
        normal = response.FOIL_NORMAL
        horizontal_projection = horizontal[detector]@normal
        cuts = []
        for front, back in pixels:
            ycentre = (back+.5)*response.PITCH-response.HALF_SIZE
            crossing = -(response.CENTRES[detector]@normal+ycentre*(vertical@normal))/horizontal_projection
            cuts.extend([strip_intervals(int(front), self.quadrature.sharing_width_mm, crossing),
                         strip_intervals(int(back), self.quadrature.sharing_width_mm)])
        return tuple(itertools.product(*cuts))

    def geometry(self, detector, rectangle, points):
        n = len(points)
        low, high = np.asarray(rectangle).T
        local = (low+points[:, :4]*(high-low)).reshape(n, 2, 2)
        normal, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
        position = response.CENTRES[detector]+local[..., :1]*horizontal[detector]+local[..., 1:]*vertical
        xy = (points[:, 4:6]-.5)*material.parent.PLAN['beam_width_mm']
        origin = np.column_stack([xy, -response.FOIL_NORMAL[0]/response.FOIL_NORMAL[2]*xy[:, 0]])
        vector = position-origin[:, None, :]; distance = np.linalg.norm(vector, axis=-1)
        direction = vector/distance[..., None]
        jacobian_area = np.prod(abs(vector@normal[detector])/distance**3, axis=1)
        columns = matching.audit.columns(direction, np.full((n, 2), detector), points[:, 6],
                                        response.Response(), self.tables, matching.PLAN['geometry'])
        ceiling = single.kinetic_ceiling(direction)
        boron, carbon, silicon = columns
        active = self.tables.remove(self.tables.remove(self.tables.remove(ceiling, boron, 'G4_B'), carbon, 'G4_C'), silicon, 'G4_Si')
        maximum = self.tables.materials['G4_Si']['light'](active)
        coordinate = np.clip((local+response.HALF_SIZE)/response.PITCH, 0., np.nextafter(16., 0.))
        design = np.zeros((n, 32, 2))
        for hit in (0, 1):
            for side in (0, 1):
                first, second, fraction = matching.matching.split_charge(coordinate[:, hit, side], response.PITCH, self.quadrature.sharing_width_mm)
                np.add.at(design, (np.arange(n), side*16+first, hit), 1-fraction)
                np.add.at(design, (np.arange(n), side*16+second, hit), fraction)
        return dict(columns=columns, ceiling=ceiling, maximum=maximum, design=design,
                    measure=jacobian_area*np.prod(high-low)/n)

    def integrate(self, detector, pixels, calibrated, power, seed, order):
        observed = np.asarray(calibrated, float).reshape(32)
        occupied = observed > 0
        measured = single.recover_charge(observed[occupied], self.tables)
        accumulator = LogMoments(); fallback_rays = 0
        points = unit_points(power, seed)
        for rectangle in self.rectangles(detector, pixels):
            geometry = self.geometry(detector, rectangle, points)
            design = geometry['design']; matrix = design[:, occupied]
            mean, covariance, chi2, good = linear_gaussian(matrix, measured, self.sigma)
            charge, qweight = bounded_gaussian_nodes(mean, covariance, geometry['maximum'], order)
            log_prefactor = np.log(2*np.pi)+.5*np.linalg.slogdet(covariance)[1]-.5*chi2
            # Rank-poor geometries use the same finite charge box, not a discarded prior region.
            if np.any(~good):
                grid, weights = tensor_rule(order)
                charge[~good] = geometry['maximum'][~good, None, :]*grid[None, :, :]
                qweight[~good] = weights[None, :]
                log_prefactor[~good] = np.log(np.maximum(geometry['maximum'][~good], np.finfo(float).tiny)).sum(axis=1)
                fallback_rays += int((~good).sum())
            prediction = np.einsum('gsi,gni->gns', design, charge)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_weight = np.log(qweight)+log_prefactor[:, None]+np.log(geometry['measure'])[:, None]
                if np.any(~good):
                    difference = prediction[~good][:, :, occupied]-measured
                    log_weight[~good] -= .5*np.sum(difference*difference, axis=2)/(self.sigma*self.sigma)
                log_weight -= occupied.sum()*np.log(self.sigma*np.sqrt(2*np.pi))
                absent = prediction[:, :, ~occupied]
                censoring = np.where(absent > 0, log_ndtr((self.single.threshold[detector]-absent)/self.sigma), 0.)
                log_weight += censoring.sum(axis=2)
                kinetic = np.zeros_like(charge)
                valid = np.all(charge > 0, axis=2)
                valid &= np.all(matrix.sum(axis=2) > 0, axis=1)[:, None]
                for hit in (0, 1):
                    columns = tuple(layer[:, hit] for layer in geometry['columns'])
                    restored, jacobian = self.single.inverse_energy_jacobian(charge[None, :, :, hit], columns)
                    kinetic[:, :, hit] = restored[0]
                    valid &= np.isfinite(restored[0]) & np.isfinite(jacobian[0]) & (jacobian[0] > 0)
                    valid &= (restored[0] <= geometry['ceiling'][:, hit, None]+.0001)
                    log_weight += np.log(jacobian[0])
                valid &= np.all(charge <= geometry['maximum'][:, None, :]+1e-10, axis=2)
                log_weight = np.where(valid & np.isfinite(log_weight), log_weight, -np.inf)
            accumulator.add(log_weight, kinetic)
        result = accumulator.finish()
        result.update(strata=len(self.rectangles(detector, pixels)), geometry_points=len(points)*len(self.rectangles(detector, pixels)),
                      rank_fallback_rays=fallback_rays, charge_nodes=order*order)
        return result

    def estimate(self, detector, pixels, calibrated):
        pixels = np.asarray(pixels)
        calibrated = np.asarray(calibrated, float)
        if (detector not in (0, 1) or pixels.shape != (2, 2) or not np.isfinite(pixels).all()
                or np.any(pixels != np.floor(pixels)) or np.any((pixels < 0) | (pixels > 15))
                or len(set(map(tuple, pixels))) != 2 or calibrated.shape != (2, 16)
                or not np.isfinite(calibrated).all() or np.any(calibrated < 0)):
            raise ValueError('Expected two distinct integer pixels and two physical strip faces')
        for side in (0, 1):
            if set(np.flatnonzero(calibrated[side])) != set(pixels[:, side]):
                raise ValueError('The observed support must be explained by the fixed two-hit topology')
        q = self.quadrature
        cases = [(q.coarse_power, q.seed, q.charge_order), (q.main_power, q.seed, q.charge_order),
                 (q.main_power, q.alternate_seed, q.charge_order), (q.main_power, q.seed, q.high_charge_order)]
        order = np.lexsort((pixels[:, 1], pixels[:, 0])); inverse = np.argsort(order)
        result = {name:self.integrate(detector, pixels[order].astype(int), calibrated, *case) for name,case in zip(VARIANTS,cases)}
        for value in result.values():
            value['mean'] = value['mean'][inverse]
            value['covariance'] = value['covariance'][inverse][:, inverse]
        return result
