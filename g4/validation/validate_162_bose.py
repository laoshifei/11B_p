#!/usr/bin/env python3
"""Validate the opt-in 162-keV L2 BW Bose amplitude and production sampler.

Independent SciPy spherical harmonics, Python Lorentz kinematics, and an
isotropic phase-space importance integral check the compiled C++ core.
No experimental agreement or complete Laursen M4 reproduction is claimed.
Imports do not change plotting backends; results are returned, not saved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.polynomial import Legendre
from scipy.special import sph_harm_y
from scipy.stats import kstest

from audit_162_angular_reference import clebsch_gordan, single_path_l2_pdf

PROJECT = Path(__file__).resolve().parents[1]
DRIVER = PROJECT / "build/validate_162_bose"
MASS, Q, POLE, WIDTH = 3727.3794066, 8.8314, 3.12184, 1.513
MODE_NAMES = {1: "Single path L2 BW", 2: "Incoherent 3 paths", 3: "Coherent 3 paths"}


def driver_call(args, rows=None, driver=DRIVER):
    stream = None
    if rows is not None:
        stream = io.StringIO()
        np.savetxt(stream, np.atleast_2d(rows), fmt="%.17g")
    process = subprocess.run([str(driver), *map(str, args)],
                             input=None if stream is None else stream.getvalue(),
                             capture_output=True, text=True, check=True)
    return np.loadtxt(io.StringIO(process.stdout), ndmin=2)


def boost(energy, momentum, beta):
    """Vectorized active Lorentz boost; all arguments have a leading event axis."""
    b2 = np.sum(beta**2, axis=-1)
    gamma = 1 / np.sqrt(1-b2)
    bp = np.sum(beta*momentum, axis=-1)
    factor = np.divide(gamma-1, b2, out=np.zeros_like(b2), where=b2 > 0)*bp + gamma*energy
    return gamma*(energy+bp), momentum+factor[..., None]*beta


def angular_amplitude(primary, secondary):
    angles = []
    for direction in (primary, secondary):
        u = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        theta, phi = np.arccos(np.clip(u[..., 2], -1, 1)), np.arctan2(u[..., 1], u[..., 0])
        angles.append({m: sph_harm_y(2, m, theta, phi) for m in range(-2, 3)})
    values = []
    for total_m in range(-2, 3):
        value = np.zeros(primary.shape[:-1], dtype=complex)
        for secondary_m in range(-2, 3):
            primary_m = total_m-secondary_m
            if abs(primary_m) <= 2:
                value += (clebsch_gordan(2, primary_m, 2, secondary_m, 2, total_m)
                          * angles[0][primary_m]*angles[1][secondary_m])
        values.append(value)
    return np.stack(values, axis=-1)


class ReferenceRadial:
    """Python implementation of the documented BW with the same input tables.

    The Coulomb tables are shared physical inputs, not independently measured
    data. Interpolation, kinematics, and complex amplitudes are separate code.
    """
    def __init__(self, q=Q, bins=1024, mass=MASS):
        self.q, self.bins, self.mass = q, bins, mass
        text = (PROJECT/"src/CoulombPenetrabilityData.cc").read_text()
        self.tables = {}
        for prefix in ("AlphaAlpha", "Alpha8Be"):
            for suffix in ("energy_keV", "L2_logP"):
                name = prefix+"_"+suffix
                match = re.search(r"const double " + name + r"\[\]\s*=\s*\{(.*?)\};", text, re.S)
                if match is None:
                    raise ValueError(f"Missing physical input table {name}")
                self.tables[name] = np.fromstring(match[1].rstrip().rstrip(","), sep=",")
        self.grid = np.linspace(0, q, bins+1)
        self.values = self.raw(self.grid)
        self.cdf = np.r_[0, np.cumsum(np.diff(self.grid)*(self.values[:-1]+self.values[1:])/2)]

    def ratio(self, prefix, energy, reference):
        energy = np.asarray(energy)
        grid = self.tables[prefix+"_energy_keV"]
        logp = self.tables[prefix+"_L2_logP"]
        out = np.exp(np.interp(np.log(np.maximum(energy*1000, grid[0])), np.log(grid), logp)
                     - np.interp(np.log(reference*1000), np.log(grid), logp))
        out = np.where(energy > 0, out, 0)
        if prefix == "Alpha8Be":
            out = np.where(energy*1000 >= grid[0], out, 0)
        return out

    def width(self, energy):
        return WIDTH*self.ratio("AlphaAlpha", energy, POLE)

    def raw(self, energy):
        energy = np.asarray(energy)
        gamma = self.width(energy)
        value = (self.ratio("Alpha8Be", self.q-energy, self.q-POLE)
                 * gamma/((POLE-energy)**2 + gamma**2/4))
        return np.where((energy > 0) & (energy < self.q), value, 0)

    def density(self, energy):
        return np.interp(energy, self.grid, self.values, left=0, right=0)

    def distribution(self, energy):
        energy = np.clip(energy, 0, self.q)
        i = np.minimum((energy/self.q*self.bins).astype(int), self.bins-1)
        dx = energy-self.grid[i]
        slope = np.diff(self.values)[i]/(self.q/self.bins)
        return (self.cdf[i]+self.values[i]*dx+slope*dx**2/2)/self.cdf[-1]


def momenta(q, pair_energy, mass=MASS):
    total, pair = 3*mass+q, 2*mass+pair_energy
    p = np.sqrt(np.maximum(0, (total-mass-pair)*(total+mass+pair)
                           *(total-mass+pair)*(total+mass-pair)))/(2*total)
    secondary = np.sqrt(np.maximum(0, pair_energy*(4*mass+pair_energy)))/2
    return p, secondary


def reference_weights(event, radial):
    """Event has shape (N,3,4), columns px,py,pz,kinetic in the parent CM."""
    mass = radial.mass
    p, e = event[..., :3], event[..., 3]+mass
    paths = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        ep, pp = e[:, j]+e[:, k], p[:, j]+p[:, k]
        eaa = np.sqrt(ep**2-np.sum(pp**2, axis=-1))-2*mass
        primary, secondary = momenta(radial.q, eaa, mass)
        _, rest = boost(e[:, j], p[:, j], -pp/ep[:, None])
        angular = angular_amplitude(p[:, i], rest)
        denominator = POLE-eaa-0.5j*radial.width(eaa)
        # Unit-modulus inverse-BW phase, independently evaluated by division.
        phase = abs(denominator)/denominator
        magnitude = np.sqrt(np.divide(radial.density(eaa), primary*secondary,
                            out=np.zeros_like(eaa), where=primary*secondary > 0))
        paths.append(angular*(magnitude*phase)[:, None])
    amplitudes = np.stack(paths, axis=1)
    diagonal = np.mean(abs(amplitudes)**2, axis=-1)
    coherent = np.mean(abs(amplitudes.sum(axis=1))**2, axis=-1)
    incoherent = diagonal.sum(axis=1)
    return np.column_stack([diagonal[:, 0], incoherent, coherent, coherent/(3*incoherent)])


def independent_proposal(n, rng, q=Q):
    """Uniform pair energy and two independent isotropic directions.

    This is intentionally different from the C++ BW/angle proposal.
    Multiply by p*q*I for relativistic phase-space integration in dE_pair.
    """
    pair_energy = rng.uniform(0, q, n)
    p, secondary = momenta(q, pair_energy)
    directions = rng.normal(size=(2, n, 3))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    p1, pr = p[:, None]*directions[0], secondary[:, None]*directions[1]
    e1 = np.sqrt(MASS**2+p**2)
    pair_mass = 2*MASS+pair_energy
    ep = np.sqrt(pair_mass**2+p**2)
    beta = -p1/ep[:, None]
    e2, p2 = boost(pair_mass/2, pr, beta)
    e3, p3 = boost(pair_mass/2, -pr, beta)
    event = np.stack([np.column_stack([p1, e1-MASS]), np.column_stack([p2, e2-MASS]),
                      np.column_stack([p3, e3-MASS])], axis=1)
    return event, p*secondary


def observables(event, q=Q):
    """Symmetric energy moments and distribution bins, with event-level errors."""
    fractions = np.sort(event[..., 3], axis=1)/q
    values = [fractions[:, 0], fractions[:, 2], np.sum(fractions**2, axis=1)]
    names = ["min_T/Q", "max_T/Q", "sum_(T/Q)^2"]
    for lo, hi in zip(np.linspace(0, 2/3, 13)[:-1], np.linspace(0, 2/3, 13)[1:]):
        values.append(np.mean((fractions >= lo) & (fractions < hi), axis=1))
        names.append(f"single_alpha_bin_{lo:.3f}_{hi:.3f}")
    return names, np.column_stack(values)


def estimate(values, weights=None):
    if weights is None:
        return values.mean(axis=0), values.std(axis=0, ddof=1)/np.sqrt(len(values))
    weights = np.asarray(weights)
    mu = np.sum(values*weights[:, None], axis=0)/weights.sum()
    se = np.sqrt(np.sum((weights[:, None]*(values-mu))**2, axis=0))/weights.sum()
    return mu, se


def run_validation(n=50000, n_reference=120000, seed=16220261019, driver=DRIVER):
    if min(n, n_reference) < 2000:
        raise ValueError("Distribution validation requires at least 2000 events per sample")
    if not Path(driver).is_file():
        raise FileNotFoundError("Build first: cmake --build g4/build --target validate_162_bose -j8")
    rng = np.random.default_rng(seed)
    radial = ReferenceRadial()
    checks = []

    def check(name, value, bound):
        passed = bool(np.isfinite(value) and value <= bound)
        checks.append(dict(name=name, value=float(value), upper_bound=float(bound), passed=passed))

    p, s = rng.normal(size=(2, 256, 3))
    cpp = driver_call(["angular"], np.column_stack([p, s]), driver).reshape(-1, 5, 2)
    check("complex angular amplitude vs SciPy", np.max(abs(cpp[..., 0]+1j*cpp[..., 1]-angular_amplitude(p, s))), 1e-13)
    energy_grid = np.linspace(0, Q, 20001)
    cpp_radial = driver_call(["radial", Q, 1024], energy_grid[:, None], driver)
    reference_radial = np.column_stack([radial.density(energy_grid), radial.raw(energy_grid), radial.width(energy_grid)])
    check("BW density and width vs Python", np.max(abs(cpp_radial-reference_radial)/np.maximum(1, abs(reference_radial))), 1e-11)

    samples = {mode: driver_call(["sample", mode, n, seed+mode, Q, 1024], driver=driver)
               for mode in MODE_NAMES}
    events = {mode: rows[:, :12].reshape(-1, 3, 4) for mode, rows in samples.items()}
    all_events = np.concatenate(list(events.values()))
    check("CM energy conservation MeV", np.max(abs(all_events[..., 3].sum(axis=1)-Q)), 1e-9)
    check("CM momentum conservation MeV/c", np.max(np.linalg.norm(all_events[..., :3].sum(axis=1), axis=-1)), 1e-9)
    check("mass shell MeV^2", np.max(abs((all_events[..., 3]+MASS)**2-np.sum(all_events[..., :3]**2, axis=-1)-MASS**2)), 1e-7)
    check("finite positive kinetic energies", float(not (np.isfinite(all_events).all() and np.all(all_events[..., 3] > 0))), 0)

    event = events[2][:1024]
    weights = samples[2][:1024, 12:16]
    reference = reference_weights(event, radial)
    check("full complex three-path weights vs Python", np.max(abs(weights-reference)/np.maximum(abs(reference), 1e-15)), 1e-8)
    permutation_error = 0
    for permutation in itertools.permutations(range(3)):
        permuted = driver_call(["weights", Q, 1024], event[:, permutation, :].reshape(-1, 12), driver)
        permutation_error = max(permutation_error, np.max(abs(permuted[:, 1:]-weights[:, 1:])/weights[:, 1:]))
    check("all six particle permutations", permutation_error, 1e-8)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    rotated = event.copy()
    rotated[..., :3] = event[..., :3] @ rotation
    rotw = driver_call(["weights", Q, 1024], rotated.reshape(-1, 12), driver)
    check("rotation invariance", np.max(abs(rotw-weights)/np.maximum(abs(weights), 1e-15)), 1e-8)
    rotated[..., :3] = -event[..., :3]
    parity = driver_call(["weights", Q, 1024], rotated.reshape(-1, 12), driver)
    check("parity invariance", np.max(abs(parity-weights)/np.maximum(abs(weights), 1e-15)), 1e-8)
    acceptance = np.concatenate([rows[:, 15] for rows in samples.values()])
    check("analytic Cauchy bound violation", max(0., -acceptance.min(), acceptance.max()-1), 0)
    ratios = samples[2][:, 14]/samples[2][:, 13]
    check("both signs of interference present", float(not (ratios.min() < 1 and ratios.max() > 1)), 0)
    exhausted = subprocess.run([str(driver), "exhaustion-check"], capture_output=True, text=True, check=True)
    check("rejection limit raises without fallback", float("PASS" not in exhausted.stdout), 0)
    invalid = subprocess.run([str(driver), "sample", "3", "1", "1", str(Q), "0"], capture_output=True, text=True)
    check("invalid radial grid rejected", float(invalid.returncode == 0), 0)

    single = events[1]
    energy, momentum = single[..., 3]+MASS, single[..., :3]
    ep, pp = energy[:, 1]+energy[:, 2], momentum[:, 1]+momentum[:, 2]
    eaa = np.sqrt(ep**2-np.sum(pp**2, axis=1))-2*MASS
    _, rest = boost(energy[:, 1], momentum[:, 1], -pp/ep[:, None])
    x = np.sum(rest*momentum[:, 0], axis=1)/(np.linalg.norm(rest, axis=1)*np.linalg.norm(momentum[:, 0], axis=1))
    coefficients, coefficients_se = estimate(np.column_stack([5*Legendre.basis(2)(x), 9*Legendre.basis(4)(x)]))
    coefficient_z = (coefficients-np.array([-15/49, 36/49]))/coefficients_se
    check("single-path L2 angular coefficients z", np.max(abs(coefficient_z)), 6)
    ks = kstest(eaa, radial.distribution)
    check("single-path BW Kolmogorov distance", ks.statistic*np.sqrt(n), 2.5)
    isotropy_z = []
    for mode, current in events.items():
        directions = current[:, 0, :3]/np.linalg.norm(current[:, 0, :3], axis=1, keepdims=True)
        features = np.column_stack([directions, directions**2-1/3])
        mu, se = estimate(features)
        isotropy_z.extend((mu/se).tolist())
    check("all model primary-slot isotropy z", np.max(np.abs(isotropy_z)), 6)
    names, v1 = observables(events[1])
    _, v2 = observables(events[2])
    m1, s1 = estimate(v1)
    m2, s2 = estimate(v2)
    check("single/incoherent unlabelled distributions z", np.max(abs(m1-m2)/np.sqrt(s1**2+s2**2)), 6)

    phase, jacobian = independent_proposal(n_reference, rng)
    phase_weights = reference_weights(phase, radial)
    _, phase_values = observables(phase)
    importance = phase_weights[:, 2]*jacobian
    reference_mean, reference_se = estimate(phase_values, importance)
    _, v3 = observables(events[3])
    m3, s3 = estimate(v3)
    z = (m3-reference_mean)/np.sqrt(s3**2+reference_se**2)
    check("coherent vs independent phase-space integral z", np.max(abs(z)), 6)
    reweighted_mean, reweighted_se = estimate(v2, samples[2][:, 15])
    check("coherent vs reweighted diagonal mixture z", np.max(abs(m3-reweighted_mean)/np.sqrt(s3**2+reweighted_se**2)), 6)
    fine = ReferenceRadial(bins=4096)
    f0, f1 = radial.density(energy_grid), fine.density(energy_grid)
    radial_tv = np.trapezoid(abs(f0/np.trapezoid(f0, energy_grid)-f1/np.trapezoid(f1, energy_grid)), energy_grid)/2
    check("1024/4096 radial-grid shape TV", radial_tv, 2e-4)
    fine_weights = reference_weights(phase, fine)[:, 2]*jacobian
    fine_mean, _ = estimate(phase_values, fine_weights)
    grid_change = np.max(abs(fine_mean-reference_mean))
    check("1024/4096 normalized coherent moment change", grid_change, 2e-4)

    hashes = {}
    for rel in ("include/H11B162BoseModel.hh", "src/H11B162BoseModel.cc", "src/CoulombPenetrability.cc",
                "src/CoulombPenetrabilityData.cc", "src/H11BReaction.cc", "include/Constants.hh",
                "include/DataStructure.hh", "include/RootIO.hh", "src/RootIO.cc", "src/H11BConfig.cc",
                "include/H11BParticleLabel.hh", "src/H11BTrackInformation.cc",
                "validation/validate_162_bose_driver.cc", "validation/validate_162_bose.py",
                "validation/audit_162_angular_reference.py", "macros/validation_162_bose_l2.mac"):
        hashes[rel] = hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()
    result = dict(computed_at=datetime.now().astimezone().isoformat(), n_per_model=n,
                  n_reference=n_reference, seed=seed, q_MeV=Q, alpha_mass_MeV=MASS,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                  source_sha256=hashes, driver_sha256=hashlib.sha256(Path(driver).read_bytes()).hexdigest(),
                  checks=checks, all_passed=all(row["passed"] for row in checks),
                  min_acceptance=float(acceptance.min()), max_acceptance=float(acceptance.max()),
                  interference_ratio_range=[float(ratios.min()), float(ratios.max())],
                  coherent_mean_attempts=float(samples[3][:, 16].mean()), coherent_max_attempts=int(samples[3][:, 16].max()),
                  angular_coefficients=coefficients.tolist(), angular_coefficient_se=coefficients_se.tolist(),
                  bw_ks_pvalue=float(ks.pvalue), reference_effective_n=float(importance.sum()**2/np.sum(importance**2)),
                  observable_names=names, coherent_means=m3.tolist(), coherent_se=s3.tolist(),
                  reference_means=reference_mean.tolist(), reference_se=reference_se.tolist(),
                  reference_z=z.tolist(), radial_grid_tv=float(radial_tv), coherent_grid_change=float(grid_change))
    return result, dict(samples=samples, events=events, single_cosine=x, single_pair_energy=eaa,
                        reference_event=phase, reference_weights=importance)


def plot_validation(data):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    events = data["events"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    edges = np.linspace(-1, 1, 41)
    axes[0].hist(data["single_cosine"], edges, density=True, histtype="step", label="C++ single path")
    x = np.linspace(-1, 1, 501)
    axes[0].plot(x, single_path_l2_pdf(x), "k--", label="Independent L2 angular law")
    axes[0].set(xlabel=r"$\cos\chi$ in pair rest frame", ylabel="Density in dx", title="Fig. 1(a)  Single-path limit")
    axes[0].legend(fontsize=8)
    ratios = data["samples"][2][:, 14]/data["samples"][2][:, 13]
    axes[1].hist(ratios, np.linspace(0, 3, 61), density=True, histtype="step")
    axes[1].axvline(1, color="black", ls="--")
    axes[1].set(xlabel=r"$I_{coh}/I_{inc}$", ylabel="Density in the diagonal proposal",
                title="Fig. 1(b)  Destructive (<1) / constructive (>1)")
    figures = [fig]
    # Fill all six permutations equally for EVERY model, removing arbitrary
    # Monte Carlo slot choices without introducing coherence by plotting.
    histograms = []
    edges = np.linspace(-1, 1, 71)
    for event in events.values():
        f = event[..., 3]/event[..., 3].sum(axis=1, keepdims=True)
        hist = np.zeros((70, 70))
        for perm in itertools.permutations(range(3)):
            z = f[:, perm]
            xx, yy = np.sqrt(3)*(z[:, 1]-z[:, 0]), 3*z[:, 2]-1
            hist += np.histogram2d(xx, yy, bins=(edges, edges))[0]/(6*len(event))
        histograms.append(hist)
    vmax = max(h.max() for h in histograms)
    fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.5), layout="constrained")
    norm = LogNorm(vmin=1e-5, vmax=vmax)
    for ax, (mode, hist) in zip(axes, zip(MODE_NAMES, histograms)):
        handle = ax.pcolormesh(edges, edges, np.ma.masked_where(hist.T <= 0, hist.T), norm=norm, cmap="viridis")
        ax.set(xlabel=r"$\sqrt{3}(T_2-T_1)/Q$", ylabel=r"$3T_3/Q-1$",
               title=f"Fig. 2({chr(96+mode)})  {MODE_NAMES[mode]}", aspect="equal")
    fig.colorbar(handle, ax=axes, label="Probability per bin; all 6 permutations")
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    bins = np.linspace(0, 6, 61)
    colors = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}
    for mode, event in events.items():
        hist = np.histogram(event[..., 3], bins)[0]
        axes[0].stairs(hist/(3*len(event)*np.diff(bins)), bins, color=colors[mode], label=MODE_NAMES[mode])
    reference = data["reference_event"][..., 3]
    w = np.repeat(data["reference_weights"], 3)
    h = np.histogram(reference.ravel(), bins, weights=w)[0]
    axes[0].stairs(h/(w.sum()*np.diff(bins)), bins, color="black", ls="--", lw=1, label="Independent phase-space integral")
    axes[0].set(xlabel="Single-alpha CM kinetic energy (MeV)", ylabel="Density per MeV", title="Fig. 3(a)  Bose effect with fixed BW inputs")
    axes[0].legend(fontsize=8)
    difference = histograms[2]-histograms[1]
    limit = np.max(abs(difference))
    handle = axes[1].pcolormesh(edges, edges, difference.T, vmin=-limit, vmax=limit, cmap="coolwarm")
    axes[1].set(xlabel=r"$\sqrt{3}(T_2-T_1)/Q$", ylabel=r"$3T_3/Q-1$",
               title="Fig. 3(b)  Coherent minus incoherent", aspect="equal")
    fig.colorbar(handle, ax=axes[1], label="Probability difference per bin")
    figures.append(fig)
    return figures


def validate_root(path, expected_mode, expected_entries, *, mixed=False, resonance=162,
                  alpha_mass=3727.379, reaction_q=8.682885616791):
    """Check an actual Geant4 run, including CM/LAB conversion and saved flags."""
    import uproot

    path = Path(path)
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as file:
        runinfo = {k: v.tolist() for k, v in file["RunInfo"].arrays(library="np").items()}
        keys = ["event", "resonance_id", "h11b_reaction_channel", "projectile_kinetic_lab",
                "h11b162_alpha_decay_model", "h11b162_decay_model_used", "h11b162_labels_are_bookkeeping",
                "h11b162_bose_radial_bins", "h11b162_bose_sampling_attempts", "h11b162_bose_incoherent_weight",
                "h11b162_bose_coherent_weight", "h11b162_bose_acceptance", "secondary_angular_model"]
        keys += [f"{key}_alpha{i}" for i in (1, 2, 3) for key in ("e", "theta_lab", "phi_lab")]
        keys += [f"alpha{i}_role" for i in (1, 2, 3)] + [f"projectile_p{axis}_lab" for axis in "xyz"]
        a = file["reaction"].arrays(keys, library="np")
        sphere = file["virtual_sphere"].arrays(library="np") if "virtual_sphere" in file else None
    n = len(a["event"])
    assert n == expected_entries and np.array_equal(np.sort(a["event"]), np.arange(n))
    assert runinfo["output_schema_version"] == [9]
    assert runinfo["configured_162_alpha_decay_model"] == [expected_mode]
    assert runinfo["configured_162_bose_radial_bins"] == [1024]
    assert runinfo["configured_162_bose_max_attempts"] == [100000]
    assert np.all(a["resonance_id"] == resonance)
    assert np.all(a["h11b162_alpha_decay_model"] == expected_mode)
    kinetic = np.column_stack([a[f"e_alpha{i}"] for i in (1, 2, 3)])
    theta = np.column_stack([a[f"theta_lab_alpha{i}"] for i in (1, 2, 3)])
    phi = np.column_stack([a[f"phi_lab_alpha{i}"] for i in (1, 2, 3)])
    assert np.isfinite(kinetic).all() and (kinetic > 0).all()
    # Defaults verified against G4Alpha/G4Proton/G4NucleiProperties in 11.4.2.
    # The standalone fixed-Q benchmark intentionally supplies its own mass.
    magnitude = np.sqrt(kinetic*(kinetic+2*alpha_mass))
    momentum = magnitude[..., None]*np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
    incoming = np.column_stack([a[f"projectile_p{axis}_lab"] for axis in "xyz"])
    momentum_residual = float(np.max(np.linalg.norm(momentum.sum(axis=1)-incoming, axis=1)))
    q_reaction = kinetic.sum(axis=1)-a["projectile_kinetic_lab"]
    energy_residual = float(np.max(abs(q_reaction-reaction_q)))
    assert energy_residual < 1e-7 and momentum_residual < 1e-7
    selected = (a["h11b_reaction_channel"] == 2) & (resonance == 162)
    used = expected_mode if resonance == 162 else 0
    assert np.all(a["h11b162_decay_model_used"][selected] == used)
    assert np.all(a["h11b162_decay_model_used"][~selected] == 0)
    if mixed:
        assert 0 < selected.sum() < n and np.all(np.isin(a["h11b_reaction_channel"], [1, 2]))
    elif resonance == 162:
        assert selected.all()
    roles = np.column_stack([a[f"alpha{i}_role"] for i in (1, 2, 3)])
    bookkeeping = selected & (expected_mode in (2, 3))
    assert np.array_equal(a["h11b162_labels_are_bookkeeping"].astype(bool), bookkeeping)
    assert np.all(roles[bookkeeping] == 6)
    assert np.all(roles[~bookkeeping] == [1, 2, 2])
    amplitude_selected = selected & (expected_mode != 0)
    max_weight_error, max_attempts, mean_attempts = 0., 0, 0.
    if amplitude_selected.any():
        assert np.all(a["secondary_angular_model"][amplitude_selected] == 3)
        inc = a["h11b162_bose_incoherent_weight"][amplitude_selected]
        coh = a["h11b162_bose_coherent_weight"][amplitude_selected]
        accept = a["h11b162_bose_acceptance"][amplitude_selected]
        assert (inc > 0).all() and np.isfinite(coh).all() and ((accept >= 0) & (accept <= 1)).all()
        assert np.allclose(accept, coh/(3*inc), rtol=1e-12, atol=1e-14)
        attempts = a["h11b162_bose_sampling_attempts"][amplitude_selected]
        assert (attempts >= 1).all() and (attempts < 100000).all()
        max_attempts, mean_attempts = int(attempts.max()), float(attempts.mean())
        if expected_mode != 3:
            assert (attempts == 1).all()
        for i in np.flatnonzero(amplitude_selected)[:12]:
            energies = kinetic[i]+alpha_mass
            beta = momentum[i].sum(axis=0)/energies.sum()
            ec, pc = boost(energies, momentum[i], np.broadcast_to(-beta, (3, 3)))
            event = np.column_stack([pc, ec-alpha_mass])[None]
            radial = ReferenceRadial(q=float(ec.sum()-3*alpha_mass), mass=alpha_mass)
            ref = reference_weights(event, radial)[0, 1:]
            recorded = [a["h11b162_bose_incoherent_weight"][i], a["h11b162_bose_coherent_weight"][i],
                        a["h11b162_bose_acceptance"][i]]
            max_weight_error = max(max_weight_error, float(np.max(abs(recorded-ref)/np.maximum(abs(ref), 1e-15))))
        assert max_weight_error < 1e-6
    else:
        assert (a["h11b162_bose_sampling_attempts"] == 0).all()
    sphere_count = 0
    if sphere is not None and len(sphere["event_id"]):
        truth = (sphere["h11b_reaction_channel"] == 2) & (sphere["h11b_particle_source"] == 1)
        sphere_count = int(truth.sum())
        assert sphere_count > 0 and (sphere["h11b_particle_role"][truth] == 6).all()
        assert np.isin(sphere["generator_particle_index"][truth], [0, 1, 2]).all()
    return dict(root=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), runinfo=runinfo,
                alpha_mass_MeV=alpha_mass, reaction_q_MeV=reaction_q,
                entries=n, alpha_count=kinetic.size, alpha0_count=int(np.sum(a["h11b_reaction_channel"] == 1)),
                alpha1_count=int(selected.sum()), energy_residual_MeV=energy_residual,
                momentum_residual_MeV=momentum_residual, max_reconstructed_weight_relative_error=max_weight_error,
                mean_attempts=mean_attempts, max_attempts=max_attempts, sphere_tagged_alpha_count=sphere_count,
                all_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--reference-n", type=int, default=120000)
    parser.add_argument("--driver", type=Path, default=DRIVER)
    args = parser.parse_args()
    result, _ = run_validation(args.n, args.reference_n, driver=args.driver)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
