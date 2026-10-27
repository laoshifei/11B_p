"""Observable-only local material posterior for isolated edge-strip hits.

This conditional response prototype uses declared geometry/energy priors. It
does not receive generated particles, channel labels, or an event energy sum.
Nominal momentum directions are retained by the calling reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from . import laursen_162_material_paths as material

matching, response, energy = material.matching, material.response, material.energy
VARIANTS = ('flat_coarse', 'flat', 'phase_volume', 'flat_alternate')


@dataclass(frozen=True)
class Quadrature:
    main_power: int = 11
    coarse_power: int = 9
    noise_order: int = 12
    seed: int = 1622026102701
    alternate_seed: int = 1622026102702
    batch_hits: int = 32
    interpolation_tolerance_MeV: float = .0001


def kinetic_ceiling(direction):
    """Single-alpha ceiling from a remainder invariant mass of at least 2m."""
    direction = np.asarray(direction, float)
    if direction.shape[-1] != 3 or not np.isfinite(direction).all() or not np.allclose(np.linalg.norm(direction, axis=-1), 1.):
        raise ValueError('Expected unit emission directions')
    nominal = energy.nominal_parent(1)
    total, momentum = nominal['parent_energy'][0], nominal['parent_momentum'][0]
    mass = response.MASS
    invariant = total*total-momentum@momentum
    constant = (invariant-3*mass*mass)/2
    projected = direction@momentum
    coefficient = total*total-projected*projected
    pmax = (constant*projected+total*np.sqrt(constant*constant-coefficient*mass*mass))/coefficient
    return pmax*pmax/(np.sqrt(mass*mass+pmax*pmax)+mass)


def recover_charge(calibrated, tables):
    """Invert the exact frozen calibration spline, using observed values only."""
    calibrated = np.asarray(calibrated, float)
    row = tables.materials['G4_Si']
    charge = row['light'](calibrated)
    derivative = row['inverse_light'].derivative()
    for _ in range(4):
        charge -= (row['inverse_light'](charge)-calibrated)/derivative(charge)
    if (not np.isfinite(charge).all() or np.any(charge < 0)
            or not np.allclose(row['inverse_light'](charge), calibrated, atol=1e-10, rtol=0)):
        raise ArithmeticError('Observed charge calibration could not be inverted')
    return charge


def eligible_hits(matched, observed, sharing_width_mm):
    """Integer, isolated, non-reused strips with geometry crossing the foil."""
    detector = matched['detector']; pixel = matched['pixel']; n = len(detector)
    active = detector >= 0
    rounded = np.rint(pixel).astype(int)
    integral = np.all(abs(pixel-rounded) < 1e-10, axis=-1)
    valid_event = np.isin(matched['multiplicity'], (2, 3)) & ~matched['ambiguous']
    eligible = active & integral & valid_event[:, None]
    eligible &= material.pixel_foil_bounds(matched, sharing_width_mm)['crosses']
    for hit in range(3):
        d = np.maximum(detector[:, hit], 0)
        for side in (0, 1):
            strip = np.clip(rounded[:, hit, side], 0, 15)
            eligible[:, hit] &= observed[np.arange(n), d, side, strip] > 0
            for offset in (-1, 1):
                neighbor = strip+offset
                occupied = (neighbor >= 0) & (neighbor < 16) & (observed[np.arange(n), d, side, np.clip(neighbor, 0, 15)] > 0)
                eligible[:, hit] &= ~occupied
            for other in range(3):
                if other != hit:
                    reused = active[:, other] & (detector[:, other] == detector[:, hit]) & (abs(pixel[:, other, side]-pixel[:, hit, side]) < 1e-10)
                    eligible[:, hit] &= ~reused
    return eligible


def truncated_charge_nodes(measured, fractions, maximum, sigma, order):
    """Integrate the product of two observed Gaussian pulses within 0..qmax.

    The prior is on emitted energy, not on already-triggered hits: no division
    by a single-strip trigger probability is applied to the observed pulses.
    """
    measured, fractions = np.asarray(measured), np.asarray(fractions)
    norm = np.sum(fractions*fractions, axis=1)
    safe = np.maximum(norm, 1e-100)
    mean = (measured[:, 0, None]*fractions[None, :, 0]+measured[:, 1, None]*fractions[None, :, 1])/safe
    width = sigma/np.sqrt(safe)
    mismatch = np.sum((measured[:, None, :]-mean[..., None]*fractions[None, :, :])**2, axis=-1)/(sigma*sigma)
    low, high = ndtr(-mean/width), ndtr((maximum[None, :]-mean)/width)
    probability = np.maximum(0., high-low)
    u, weights = np.polynomial.legendre.leggauss(order)
    u, weights = (u+1)/2, weights/2
    quantile = low[..., None]+probability[..., None]*u
    quantile = np.clip(quantile, np.nextafter(0., 1.), np.nextafter(1., 0.))
    nodes = mean[..., None]+width[None, :, None]*ndtri(quantile)
    nodes = np.where(probability[..., None] > 0, nodes, .1)
    scale = probability*np.exp(-mismatch/2)/np.sqrt(safe)
    scale *= np.all(fractions > 0, axis=-1)
    return nodes, scale[..., None]*weights


class PixelPosterior:
    def __init__(self, tables, quadrature=Quadrature()):
        if not 0 <= quadrature.coarse_power <= quadrature.main_power or quadrature.noise_order < 2 or quadrature.batch_hits < 1:
            raise ValueError('Invalid quadrature configuration')
        self.tables, self.quadrature = tables, quadrature
        self.derivatives = {name: {key: row[key].derivative() for key in ('forward', 'inverse', 'inverse_light')}
                            for name, row in tables.materials.items()}
        self.sigma = matching.PLAN['side_fwhm_MeV']/2.354820045
        self.threshold = np.maximum(matching.PLAN['threshold_MeV'], response.Response().adc_MeV)

    @lru_cache(maxsize=128)
    def geometry(self, detector, front, back, width, alternate=False):
        config = self.quadrature
        seed = config.alternate_seed if alternate else config.seed
        points = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(config.main_power)
        matched = matching.allocate(1); matched['detector'][0, 0] = detector; matched['pixel'][0, 0] = [front, back]
        bounds = material.pixel_foil_bounds(matched, width)
        low, high = bounds['low_mm'][0, 0], bounds['high_mm'][0, 0]
        local = low+points[:, :2]*(high-low)
        normals, horizontal, vertical = matching.audit.basis(matching.PLAN['geometry'])
        position = response.CENTRES[detector]+local[:, :1]*horizontal[detector]+local[:, 1:2]*vertical
        xy = (points[:, 2:4]-.5)*material.parent.PLAN['beam_width_mm']
        origin = np.column_stack([xy, -response.FOIL_NORMAL[0]/response.FOIL_NORMAL[2]*xy[:, 0]])
        vector = position-origin; distance = np.linalg.norm(vector, axis=1)
        direction = vector/distance[:, None]
        jacobian_area = abs(vector@normals[detector])/distance**3
        columns = matching.audit.columns(direction[:, None, :], np.full((len(points), 1), detector),
            points[:, 4], response.Response(), self.tables, matching.PLAN['geometry'])
        columns = tuple(column[:, 0] for column in columns)
        ceiling = kinetic_ceiling(direction)
        b, c, si = columns
        active = self.tables.remove(self.tables.remove(self.tables.remove(ceiling, b, 'G4_B'), c, 'G4_C'), si, 'G4_Si')
        maximum = self.tables.materials['G4_Si']['light'](active)
        coordinate = np.clip((local+response.HALF_SIZE)/response.PITCH, 0., np.nextafter(16., 0.))
        fractions = []
        for side, strip in enumerate((front, back)):
            first, second, fraction = matching.matching.split_charge(coordinate[:, side], response.PITCH, width)
            fractions.append(np.where(first == strip, 1-fraction, 0.)+np.where(second == strip, fraction, 0.))
        return dict(columns=columns, ceiling=ceiling, maximum=maximum,
                    fractions=np.stack(fractions, axis=1), geometry_weight=jacobian_area)

    def inverse_energy_jacobian(self, charge, columns):
        row = self.tables.materials['G4_Si']
        kinetic = row['inverse_light'](charge)
        jacobian = self.derivatives['G4_Si']['inverse_light'](charge)
        for material_name, column in zip(('G4_Si', 'G4_C', 'G4_B'), columns[::-1]):
            row = self.tables.materials[material_name]; derivative = self.derivatives[material_name]
            target = row['forward'](kinetic)+column[None, :, None]
            jacobian *= derivative['forward'](kinetic)*derivative['inverse'](target)
            kinetic = row['inverse'](target)
        return kinetic, jacobian

    def integrate(self, measured, geometry, detector, *, alternate=False):
        q = self.quadrature
        charge, gaussian_weight = truncated_charge_nodes(measured, geometry['fractions'], geometry['maximum'], self.sigma, q.noise_order)
        kinetic, jacobian = self.inverse_energy_jacobian(charge, geometry['columns'])
        valid = np.isfinite(kinetic) & np.isfinite(jacobian) & (jacobian > 0) & (charge > 0)
        valid &= kinetic <= geometry['ceiling'][None, :, None]+q.interpolation_tolerance_MeV
        missing = 1-geometry['fractions']
        censoring = np.ones_like(charge)
        for side in (0, 1):
            absent = ndtr((self.threshold[detector]-charge*missing[None, :, side, None])/self.sigma)
            censoring *= np.where(missing[None, :, side, None] > 0, absent, 1.)
        safe_energy = np.where(valid, kinetic, 0.)
        weight = np.where(valid, jacobian, 0.)*gaussian_weight*censoring*geometry['geometry_weight'][None, :, None]
        if not np.isfinite(weight).all():
            raise ArithmeticError('Nonfinite posterior quadrature weight')
        volume = np.sqrt(safe_energy*(safe_energy+2*response.MASS))*(safe_energy+response.MASS)
        rules = [(len(geometry['ceiling']), False)] if alternate else [(2**q.coarse_power, False), (2**q.main_power, False), (2**q.main_power, True)]
        results = []
        for count, phase_volume in rules:
            w = weight[:, :count]*(volume[:, :count] if phase_volume else 1.)
            t = safe_energy[:, :count]
            total = w.sum(axis=(1, 2)); supported = total > 0
            mean = np.divide((w*t).sum(axis=(1, 2)), total, out=np.full(len(measured), np.nan), where=supported)
            second = np.divide((w*t*t).sum(axis=(1, 2)), total, out=np.full(len(measured), np.nan), where=supported)
            ray_weight = w.sum(axis=-1)
            effective = np.divide(total*total, (ray_weight*ray_weight).sum(axis=1), out=np.zeros(len(measured)), where=supported)
            results.append(dict(mean=mean, std=np.sqrt(np.maximum(0., second-mean*mean)), valid=supported, effective_rays=effective))
        return results

    def estimate(self, detector, pixels, calibrated_pair, width):
        """Return four posterior point-estimate controls for observed hit pairs."""
        detector, pixels = np.asarray(detector, int), np.asarray(pixels, int)
        measured = recover_charge(calibrated_pair, self.tables)
        n = len(detector)
        if pixels.shape != (n, 2) or measured.shape != (n, 2) or np.any((pixels < 0) | (pixels > 15)) or np.any((detector < 0) | (detector > 1)):
            raise ValueError('Invalid observable hit arrays')
        output = {field: np.full((n, 4), np.nan) for field in ('mean', 'std')}
        output.update(valid=np.zeros((n, 4), bool), effective_rays=np.zeros((n, 4)))
        keys = np.column_stack([detector, pixels])
        for key in np.unique(keys, axis=0):
            rows = np.flatnonzero(np.all(keys == key, axis=1))
            for start in range(0, len(rows), self.quadrature.batch_hits):
                group = rows[start:start+self.quadrature.batch_hits]
                evaluated = self.integrate(measured[group], self.geometry(*map(int, key), float(width)), int(key[0]))
                evaluated += self.integrate(measured[group], self.geometry(*map(int, key), float(width), True), int(key[0]), alternate=True)
                for variant, result in enumerate(evaluated):
                    for field, value in result.items():
                        output[field][group, variant] = value
        return output
