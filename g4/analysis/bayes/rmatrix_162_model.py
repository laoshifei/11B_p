"""Opt-in 162-keV L2 amplitude controls for importance integration.

Laursen et al., EPJA 52, 271 (2016), DOI 10.1140/epja/i2016-16271-2,
PDF pp. 2-3, equations (3)-(5). These are analysis weights, not a new
Geant4 production sampler or a claim to reproduce the measured M4 curves.
All three cyclic paths are summed before squaring. The initial J=2 is
unpolarized. Energies/masses are MeV, radii fm. No import-time computation.
"""
from __future__ import annotations

import ctypes as ct
import ctypes.util
from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp

from validation.audit_162_angular_reference import clebsch_gordan

MASS = 3727.3794066
HBARC = 197.3269804
ALPHA = 1 / 137.035999084
MODELS = ('bw', 'rmatrix_no_phase', 'rmatrix', 'rmatrix_coulomb')
LABELS = ('Current coherent BW', 'R matrix, no external phase',
          'R matrix + external phases', 'R matrix + phases + Coulomb')
VERSION = '2026-10-23-v1-rmatrix-l2-importance'


@dataclass(frozen=True)
class Parameters:
    pole: float = 3.129
    reduced_width: float = 1.075  # gamma'^2, NOT Gamma or gamma'
    pair_radius: float = 4.5
    primary_radius: float = 1.42 * (4**(1/3) + 8**(1/3))
    separation: float = 10.
    mass: float = MASS
    grid_size: int = 6144
    energy_min: float = .001
    energy_max: float = 10.

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Parameters must be finite and positive')
        if self.grid_size < 128 or self.energy_min >= self.pole or self.energy_max <= self.pole:
            raise ValueError('Invalid interpolation domain')


class _Result(ct.Structure):
    _fields_ = [('val', ct.c_double), ('err', ct.c_double)]


class CoulombFunctions:
    """GSL F,G and derivatives, with no unreported overflow/clipping.

    The GSL error handler is disabled only around a synchronous call and
    restored immediately. Tables are built serially before worker processes.
    Scaled GSL overflow results are not silently treated as ordinary values.
    """
    def __init__(self):
        library = ctypes.util.find_library('gsl')
        if not library:
            raise RuntimeError('libgsl is required for Coulomb wave functions')
        self.lib = ct.CDLL(library)
        self.lib.gsl_set_error_handler_off.restype = ct.c_void_p
        self.lib.gsl_set_error_handler.argtypes = [ct.c_void_p]
        self.function = self.lib.gsl_sf_coulomb_wave_FG_e
        self.function.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_int,
            *([ct.POINTER(_Result)] * 4), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        self.function.restype = ct.c_int
        self.version = ct.c_char_p.in_dll(self.lib, 'gsl_version').value.decode()

    def wave(self, eta, rho, orbital_l=2):
        if not (np.isfinite(eta) and np.isfinite(rho) and eta >= 0 and rho > 0):
            raise ValueError('Invalid Coulomb arguments')
        f, fp, g, gp = (_Result() for _ in range(4))
        ef, eg = ct.c_double(), ct.c_double()
        old = self.lib.gsl_set_error_handler_off()
        try:
            status = self.function(eta, rho, orbital_l, 0, ct.byref(f), ct.byref(fp),
                                  ct.byref(g), ct.byref(gp), ct.byref(ef), ct.byref(eg))
        finally:
            self.lib.gsl_set_error_handler(old)
        if status or ef.value or eg.value:
            raise RuntimeError(f'Unscaled GSL Coulomb call failed: status={status}, eta={eta}, rho={rho}')
        values = np.array([f.val, fp.val, g.val, gp.val])
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite Coulomb functions')
        return values


class Channel:
    """P=rho/(F^2+G^2); S=rho*(FF'+GG')/(F^2+G^2).

    omega_l=sum_{n=1}^l atan(eta/n) is the relative Coulomb phase
    (sigma_l-sigma_0), phi=arg(G+iF). The declared phase convention
    is retained in provenance; it is not an independently recovered author
    implementation. Interpolation never silently clamps outside its domain.
    """
    def __init__(self, coulomb, reduced_mass, charge_product, radius, p):
        self.coulomb, self.mu, self.zz, self.radius = coulomb, reduced_mass, charge_product, radius
        self.grid = np.geomspace(p.energy_min, p.energy_max, p.grid_size)
        values = self.exact_many(self.grid)
        self.max_wronskian_error = float(np.max(abs(values[:, 3]-1)))
        if self.max_wronskian_error > 2e-7:
            raise RuntimeError('Coulomb Wronskian check failed')
        self.interpolators = [PchipInterpolator(np.log(self.grid), v, extrapolate=False)
                              for v in (values[:, 0], values[:, 1], np.unwrap(values[:, 2]))]

    def exact_many(self, energy, *, rtol=2e-11, boundary_scale=1.):
        """Propagate outgoing Coulomb logarithmic derivative from outside barrier.

        Direct GSL calls close to its algorithm boundaries can return WKB
        approximations with percent-level jumps and a correct Wronskian.
        Use GSL only at rho_start=max(30,2*eta+15), then solve the exact
        Coulomb ODE inward with DOP853. For L=rho*O'/O=S+iP:
        dS/dlog(rho)=S+l(l+1)+2*eta*rho-rho^2-S^2+P^2;
        dlog(P)/dlog(rho)=1-2*S; dphi/dlog(rho)=P.
        Log P preserves positivity deep below the barrier. The overall 2pi
        phase ambiguity cancels in exp(i*phase).
        """
        energy = np.atleast_1d(energy).astype(float)
        if not np.isfinite(energy).all() or np.any(energy <= 0):
            raise ValueError('Channel energies must be positive')
        rho = np.sqrt(2*self.mu*energy)*self.radius/HBARC
        eta = self.zz*ALPHA*np.sqrt(self.mu/(2*energy))
        if boundary_scale < 1:
            raise ValueError('Boundary must remain in the oscillatory region')
        start = boundary_scale*np.maximum(30.,2*eta+15.)
        f,fp,g,gp = np.array([self.coulomb.wave(e,r) for e,r in zip(eta,start)]).T
        norm = f*f+g*g
        initial = np.stack([start*(f*fp+g*gp)/norm,np.log(start/norm),np.arctan2(f,g)])
        delta = np.log(rho/start)
        def ode(t, value):
            shift,logp,phase = value.reshape(3,-1)
            r = start*np.exp(delta*t)
            p = np.exp(logp)
            return (np.stack([shift+6+2*eta*r-r*r-shift*shift+p*p,1-2*shift,p])*delta).ravel()
        result = solve_ivp(ode,(0.,1.),initial.ravel(),method='DOP853',rtol=rtol,
                           atol=rtol/50,t_eval=[1.])
        if not result.success or not np.isfinite(result.y).all():
            raise RuntimeError('Outgoing Coulomb ODE did not converge')
        shift,logp,phi = result.y[:,-1].reshape(3,-1)
        omega = np.arctan(eta)+np.arctan(eta/2)
        return np.column_stack([logp,shift,omega-phi,fp*g-f*gp])

    def exact(self, energy):
        return self.exact_many([energy])[0]

    def evaluate(self, energy):
        e = np.asarray(energy, float)
        if not np.isfinite(e).all() or np.any(e <= 0):
            raise ValueError('Channel energies must be finite and positive')
        result = np.stack([f(np.log(e)) for f in self.interpolators], axis=-1)
        outside = (e < self.grid[0]) | (e > self.grid[-1])
        if np.any(outside):
            # Rare off-grid points are evaluated, never dropped or clamped.
            flat = result.reshape(-1, 3)
            for index in np.flatnonzero(outside):
                flat[index] = self.exact(e.ravel()[index])[:3]
        return result


def boost(energy, momentum, beta):
    b2 = np.sum(beta*beta, axis=-1)
    if np.any(b2 >= 1):
        raise ValueError('Superluminal boost')
    gamma = 1/np.sqrt(1-b2)
    bp = np.sum(beta*momentum, axis=-1)
    factor = np.divide(gamma-1, b2, out=np.zeros_like(b2), where=b2 > 0)*bp+gamma*energy
    return gamma*(energy+bp), momentum+factor[..., None]*beta


def angular_amplitude(primary, secondary):
    harmonics = []
    for vector in (primary, secondary):
        u = vector/np.linalg.norm(vector, axis=-1, keepdims=True)
        xy = u[..., 0]+1j*u[..., 1]
        y1 = -np.sqrt(15/(8*np.pi))*u[..., 2]*xy
        y2 = np.sqrt(15/(32*np.pi))*xy*xy
        harmonics.append(np.stack([y2.conj(), -y1.conj(), np.sqrt(5/(16*np.pi))*(3*u[..., 2]**2-1), y1, y2], axis=-1))
    out = np.zeros(primary.shape[:-1]+(5,), complex)
    for m in range(-2, 3):
        for ms in range(-2, 3):
            mp = m-ms
            if abs(mp) <= 2:
                out[..., m+2] += clebsch_gordan(2, mp, 2, ms, 2, m)*harmonics[0][..., mp+2]*harmonics[1][..., ms+2]
    return out


class AmplitudeModel:
    def __init__(self, parameters=Parameters()):
        self.p = parameters
        self.coulomb = CoulombFunctions()
        self.pair = Channel(self.coulomb, parameters.mass/2, 4, parameters.pair_radius, parameters)
        self.primary = Channel(self.coulomb, 2*parameters.mass/3, 8, parameters.primary_radius, parameters)
        self.pair_far = Channel(self.coulomb, parameters.mass/2, 4, parameters.separation, parameters)
        self.primary_far = Channel(self.coulomb, 2*parameters.mass/3, 8, parameters.separation, parameters)
        self.shift_at_pole = self.pair.evaluate(parameters.pole)[1]
        text = (Path(__file__).resolve().parents[2]/'src/CoulombPenetrabilityData.cc').read_text()
        self.legacy = {}
        for prefix in ('AlphaAlpha', 'Alpha8Be'):
            for suffix in ('energy_keV', 'L2_logP'):
                name = prefix+'_'+suffix
                match = re.search(r'const double '+name+r'\[\]\s*=\s*\{(.*?)\};', text, re.S)
                if not match:
                    raise ValueError(f'Missing existing BW input {name}')
                self.legacy[name] = np.fromstring(match[1].rstrip().rstrip(','), sep=',')

    def legacy_ratio(self, prefix, energy, reference):
        grid, values = self.legacy[prefix+'_energy_keV'], self.legacy[prefix+'_L2_logP']
        logp = lambda e: np.interp(np.log(np.maximum(np.asarray(e)*1000, grid[0])), np.log(grid), values)
        out = np.exp(logp(energy)-logp(reference))
        out = np.where(np.asarray(energy) > 0, out, 0.)
        if prefix == 'Alpha8Be':
            out = np.where(np.asarray(energy)*1000 >= grid[0], out, 0.)
        return out

    def bw_density(self, eaa, q):
        def raw(e):
            width = 1.513*self.legacy_ratio('AlphaAlpha', e, 3.12184)
            first = self.legacy_ratio('Alpha8Be', q-e, q-3.12184)
            return np.where((e > 0) & (e < q), first*width/((3.12184-e)**2+width**2/4), 0.)
        location = eaa/q*1024
        lower = np.minimum(np.floor(location), 1023)
        value = raw(lower*q/1024)*(1-(location-lower))+raw((lower+1)*q/1024)*(location-lower)
        width = 1.513*self.legacy_ratio('AlphaAlpha', eaa, 3.12184)
        return value, 3.12184-eaa-.5j*width

    def weights(self, event, *, diagnostics=False):
        """Weights relative to uniform relativistic three-body phase space.

        event: (...,3,4), px,py,pz,T in the parent CM. The published
        nonrelativistic dynamics uses E_first=3*T_i/2, exact invariant
        E_jk for the pair and exact rotation-free pair boosts for angles.
        This declared hybrid convention is an approximation, not a refit.
        """
        event = np.asarray(event)
        if event.ndim != 3 or event.shape[1:] != (3, 4) or not np.isfinite(event).all() or np.any(event[..., 3] <= 0):
            raise ValueError('Expected positive finite CM four-vectors')
        momentum, energy = event[..., :3], event[..., 3]+self.p.mass
        q = event[..., 3].sum(axis=1)
        pairs, secondaries = [], []
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            ep, pp = energy[:, j]+energy[:, k], momentum[:, j]+momentum[:, k]
            pairs.append(np.sqrt(ep**2-np.sum(pp*pp, axis=-1))-2*self.p.mass)
            secondaries.append(boost(energy[:, j], momentum[:, j], -pp/ep[:, None])[1])
        pairs = np.stack(pairs, axis=1)
        pair_values = self.pair.evaluate(pairs)
        far_pair_values = self.pair_far.evaluate(pairs)
        amplitudes = []
        for i in range(3):
            eaa, ti = pairs[:, i], event[:, i, 3]
            first = 1.5*ti
            a = angular_amplitude(momentum[:, i], secondaries[i])
            density, denominator = self.bw_density(eaa, q)
            pp = np.linalg.norm(momentum[:, i], axis=-1)
            pr = np.sqrt(eaa*(4*self.p.mass+eaa))/2
            bw = np.sqrt(density/(pp*pr))*abs(denominator)/denominator
            lp2, shift, phase2 = pair_values[:, i].T
            lp1, _, phase1 = self.primary.evaluate(first).T
            width2 = 2*self.p.reduced_width*np.exp(lp2)
            den = self.p.pole-eaa-self.p.reduced_width*(shift-self.shift_at_pole)-.5j*width2
            magnitude = np.exp(.5*(np.log(2)+lp1+np.log(2*self.p.reduced_width)+lp2-.5*np.log(ti*eaa)))
            radial = magnitude/den
            phased = radial*np.exp(1j*(phase1+phase2))
            j, k = (i+1)%3, (i+2)%3
            log_correction = (.5*np.log(ti/(pairs[:, j]*pairs[:, k]))
                +far_pair_values[:, j, 0]+far_pair_values[:, k, 0]-self.primary_far.evaluate(first)[:, 0])
            corrected = phased*np.exp(.5*log_correction)
            factors = np.stack([bw, radial, phased, corrected], axis=1)
            amplitudes.append(factors[..., None]*a[:, None, :])
        paths = np.stack(amplitudes, axis=2)
        coherent = np.mean(abs(paths.sum(axis=2))**2, axis=-1)
        diagonal = np.mean(abs(paths)**2, axis=-1).sum(axis=2)
        if not np.isfinite(coherent).all() or np.any(coherent < 0) or np.any(coherent > 3*diagonal*(1+2e-12)):
            raise RuntimeError('Invalid amplitude or violated Bose Cauchy bound')
        return (coherent, diagonal, pairs) if diagnostics else coherent

    def provenance(self):
        return dict(version=VERSION, parameters=asdict(self.p), gsl=self.coulomb.version,
            models=list(MODELS), labels=list(LABELS),
            phase='omega_l=sum atan(eta/n), phi=arg(G+iF), exp(i*(omega-phi)); l=2',
            dynamics='E_first=1.5*T_i; pair invariant E_jk; exact pair boost; NR channel reduced masses',
            wronskian_max=max(c.max_wronskian_error for c in (self.pair,self.primary,self.pair_far,self.primary_far)),
            coulomb_solver='GSL oscillatory boundary + exact outgoing-logarithmic-derivative DOP853; rtol=2e-11',
            scope='Analysis importance weights; unpolarized pure L2; no experimental M4 reproduction claimed')
