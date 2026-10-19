#!/usr/bin/env python3
"""Read-only, at-generation validation of the recorded 162-keV sample.

Run from a terminal or import ``validate_generator`` from Note.ipynb.  No
simulation, fit, detector correction, or permanent output is performed by
default.  The Laursen Eq. (2) reweighting is an angular sensitivity control,
not an implementation of that paper's full, Bose-symmetrized M4 model.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import subprocess
from pathlib import Path

import numpy as np
import uproot
from numpy.polynomial import Legendre
from scipy.stats import chi2


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "data/spectrum_162_lab_05_95_00.root"
ALPHA_MASS_MEV = 3727.3794066
# Laursen 2016, PDF p. 2: rounded reaction Q, used with 1-keV tolerance.
LITERATURE_Q_MEV = 8.682


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parameters(project=PROJECT):
    """Read current source literals; do not silently use stale plot constants."""
    text = (project / "include/Constants.hh").read_text()
    names = {
        "er_keV": "H11B162ResonanceEnergy",
        "width_keV": "H11B162TotalWidth",
        "gp_keV": "H11B162ProtonWidth",
        "ga0_keV": "H11B162Alpha0Width",
        "ga1_keV": "H11B162Alpha1Width",
    }
    parameters = {}
    for key, name in names.items():
        match = re.search(rf"constexpr G4double {name}\s*=\s*([\d.eE+-]+)\s*\*\s*keV", text)
        if not match:
            raise ValueError(f"Inspect changed source definition: {name}")
        parameters[key] = float(match[1])
    if not re.search(r"H11BIncludeGammaChannelInCrossSection\s*=\s*false", text):
        raise ValueError("This validation expects a three-alpha-only cross section.")
    table_text = (project / "src/CoulombPenetrabilityData.cc").read_text()
    for key, name in [("table_e", "P11B_energy_keV"), ("table_logp", "P11B_L1_logP")]:
        match = re.search(rf"const double {name}\[\] = \{{(.*?)\}};", table_text, re.S)
        if not match:
            raise ValueError(f"Missing Coulomb table {name}")
        # C++ initializer lists have a trailing comma; do not let fromstring
        # interpret the empty final field as a spurious numeric entry.
        parameters[key] = np.array([float(token) for token in re.findall(
            r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", match[1])])
    if len(parameters["table_e"]) != len(parameters["table_logp"]):
        raise ValueError("Coulomb table length mismatch")
    if np.any(parameters["table_e"] <= 0) or np.any(np.diff(parameters["table_e"]) <= 0):
        raise ValueError("Coulomb table energies must be positive and ordered")
    return parameters


def sigma162_barn(ecm_keV, parameters):
    """Independent Python evaluation of the documented C++ BW prescription."""
    p = parameters
    e = np.asarray(ecm_keV, dtype=float)
    if np.any(e <= 0):
        raise ValueError("Cross-section energies must be positive")
    logp = np.interp(np.log(e), np.log(p["table_e"]), p["table_logp"])
    logpr = np.interp(np.log(p["er_keV"]), np.log(p["table_e"]), p["table_logp"])
    gp = p["gp_keV"] * np.exp(logp - logpr)
    ga = p["ga0_keV"] + p["ga1_keV"]
    missing = max(0.0, p["width_keV"] - p["gp_keV"] - ga)
    prefactor = np.pi * 197326.9804**2 / (2 * (11 / 12) * 931494.10242 * e)
    return prefactor / 100 * (5 / 8) * gp * ga / ((e-p["er_keV"])**2 + (gp+ga+missing)**2/4)


def angular_pdf(x, a2=-0.489, a4=0.647):
    return (1.0+a2*Legendre.basis(2)(x)+a4*Legendre.basis(4)(x))/2


def laursen_eq2_pdf(x):
    """Published Eq. (2), normalized in cos(theta_2), BEFORE symmetrization."""
    normalization = 2 * 1.12 + 2 * 3.2 / 3 - 2 * 3.2 / 5
    return (1.12 + 3.2*x*x - 3.2*x**4) / normalization


def rest_frame(energy, momentum, beta):
    """Boost one four-vector into the frame moving with beta (c=1)."""
    b2 = np.sum(beta**2, axis=-1)
    gamma = 1 / np.sqrt(1-b2)
    bp = np.sum(beta*momentum, axis=-1)
    coefficient = np.divide(gamma-1, b2, out=np.zeros_like(b2), where=b2>0)*bp-gamma*energy
    return gamma*(energy-bp), momentum + coefficient[:, None]*beta


def validate_generator(root_path=DEFAULT_ROOT, *, chunk_size=50000):
    """Return metadata, checks, histograms and external consistency comparisons.

    All reaction entries are read in bounded chunks. No detector cuts or narrow
    proton-energy gates are made. These biased samples only validate conditional
    generator shapes, not absolute yields or physical reaction-depth profiles.
    """
    path = Path(root_path).resolve()
    parameters = load_parameters()
    angle_edges = np.linspace(-1, 1, 61)
    energy_edges = np.linspace(0, 6.1, 123)
    dalitz_edges = np.linspace(-1/3, 1/3, 101)
    angular = {k: {"n": 0, "hist": np.zeros(60), "sum": np.zeros(5), "sum2": np.zeros(5)}
               for k in ("alpha0_primary", "alpha1_primary", "alpha1_secondary")}
    energy_hist = np.zeros((2, 122))
    weighted_energy_hist = np.zeros(122)
    dalitz = np.zeros((2, 100, 100))
    scalar = dict(n=0, invalid=0, q_min=np.inf, q_max=-np.inf, q_sum=0.0,
                  max_momentum_residual=0.0, max_cm_residual=0.0,
                  max_sigma_relative_residual=0.0, proton_min=np.inf,
                  proton_max=-np.inf, proton_sum=0.0, weight_sum=0.0,
                  weight_square_sum=0.0, max_angle_frame_difference=0.0)
    event_ids = []
    channels = {}
    config_values = {}
    config_branches = ["resonance_id", "cross_section_bias_factor", "scale_factor_162",
                       "scale_factor_675", "configured_162_alpha0_branching_fraction",
                       "direct_decay_fraction_162", "primary_a1_used", "primary_a2_used",
                       "secondary_a2_used", "secondary_a4_used", "secondary_angular_model",
                       "sigma_gamma_total_barn"]
    branches = ["event", "h11b_reaction_channel", "projectile_kinetic_lab", "sigma_162_model_barn",
                "projectile_px_lab", "projectile_py_lab", "projectile_pz_lab",
                "cos_theta_primary_cm", "cos_theta_secondary_correlation"] + config_branches
    for i in range(1, 4):
        branches += [f"e_alpha{i}", f"theta_lab_alpha{i}", f"phi_lab_alpha{i}", f"e_3alpha_cm_alpha{i}"]

    def record_angle(key, values):
        a = angular[key]
        a["n"] += len(values)
        a["hist"] += np.histogram(values, angle_edges)[0]
        for order in range(1, 5):
            v = Legendre.basis(order)(values)
            a["sum"][order] += v.sum()
            a["sum2"][order] += np.square(v).sum()

    with uproot.open(path, handler=uproot.source.file.MemmapSource) as file:
        tree = file["reaction"]
        missing = set(branches)-set(tree.keys())
        if missing:
            raise ValueError(f"Required branches missing: {sorted(missing)}")
        runinfo = file["RunInfo"].arrays(library="np") if "RunInfo" in file else {}
        runinfo = {k: v.tolist() for k, v in runinfo.items()}
        for a in tree.iterate(branches, step_size=chunk_size, library="np"):
            n = len(a["event"])
            scalar["n"] += n
            event_ids.append(a["event"])
            ch = a["h11b_reaction_channel"]
            for channel, expected_a2 in ((1, 0.6), (2, 0.0)):
                selected = ch == channel
                if (np.any(a["primary_a1_used"][selected] != 0.0)
                        or np.any(a["primary_a2_used"][selected] != expected_a2)):
                    raise ValueError("Primary angular settings differ from this baseline")
            if (np.any(a["secondary_a2_used"][ch == 2] != -0.489)
                    or np.any(a["secondary_a4_used"][ch == 2] != 0.647)
                    or np.any(a["secondary_angular_model"][ch == 2] != 1)):
                raise ValueError("Secondary angular settings differ from this baseline")
            for c, count in zip(*np.unique(ch, return_counts=True)):
                channels[int(c)] = channels.get(int(c), 0)+int(count)
            for k in config_branches:
                config_values.setdefault(k, set()).update(np.unique(a[k]).tolist())
            kinetic = np.column_stack([a[f"e_alpha{i}"] for i in range(1, 4)])
            theta = np.column_stack([a[f"theta_lab_alpha{i}"] for i in range(1, 4)])
            phi = np.column_stack([a[f"phi_lab_alpha{i}"] for i in range(1, 4)])
            cm_saved = np.column_stack([a[f"e_3alpha_cm_alpha{i}"] for i in range(1, 4)])
            bad = (~np.isfinite(kinetic).all(axis=1) | (kinetic <= 0).any(axis=1)
                   | ~np.isfinite(theta).all(axis=1) | ~np.isfinite(phi).all(axis=1)
                   | ~np.isfinite(cm_saved).all(axis=1)
                   | ~np.isfinite(a["projectile_kinetic_lab"])
                   | ~np.isfinite(a["cos_theta_primary_cm"])
                   | ((ch == 2) & ~np.isfinite(a["cos_theta_secondary_correlation"])))
            scalar["invalid"] += int(bad.sum())
            if np.any(bad):
                raise ValueError("Nonfinite or nonpositive generated alpha energies/directions")
            p_mag = np.sqrt(kinetic*(kinetic+2*ALPHA_MASS_MEV))
            momentum = p_mag[:, :, None] * np.stack(
                [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=2)
            energy = kinetic+ALPHA_MASS_MEV
            total_p = momentum.sum(axis=1)
            incoming_p = np.column_stack([a[f"projectile_p{j}_lab"] for j in "xyz"])
            if not np.isfinite(incoming_p).all():
                raise ValueError("Nonfinite incoming momentum")
            scalar["max_momentum_residual"] = max(scalar["max_momentum_residual"],
                float(np.linalg.norm(total_p-incoming_p, axis=1).max()))
            cm_beta = total_p/energy.sum(axis=1)[:, None]
            cm = np.column_stack([rest_frame(energy[:, i], momentum[:, i], cm_beta)[0]-ALPHA_MASS_MEV
                                  for i in range(3)])
            scalar["max_cm_residual"] = max(scalar["max_cm_residual"], float(np.abs(cm-cm_saved).max()))
            q = kinetic.sum(axis=1)-a["projectile_kinetic_lab"]
            scalar["q_sum"] += q.sum()
            scalar["q_min"] = min(scalar["q_min"], float(q.min()))
            scalar["q_max"] = max(scalar["q_max"], float(q.max()))
            ep = a["projectile_kinetic_lab"]*1000
            scalar["proton_min"] = min(scalar["proton_min"], float(ep.min()))
            scalar["proton_max"] = max(scalar["proton_max"], float(ep.max()))
            scalar["proton_sum"] += ep.sum()
            # Cross-section C++ uses A_target/(A_projectile+A_target), not exact invariant e_cm_p11B.
            expected_sigma = sigma162_barn(ep*11/12, parameters)
            residual = np.abs(a["sigma_162_model_barn"]/expected_sigma-1)
            scalar["max_sigma_relative_residual"] = max(scalar["max_sigma_relative_residual"], float(residual.max()))
            for channel, key in [(1, "alpha0_primary"), (2, "alpha1_primary")]:
                record_angle(key, a["cos_theta_primary_cm"][ch == channel])
            exc = ch == 2
            record_angle("alpha1_secondary", a["cos_theta_secondary_correlation"][exc])
            # Independent pair-rest-frame reconstruction, to document the angle convention.
            pair_beta = (momentum[:, 1]+momentum[:, 2])/(energy[:, 1]+energy[:, 2])[:, None]
            p1pair = rest_frame(energy[:, 0], momentum[:, 0], pair_beta)[1]
            p2pair = rest_frame(energy[:, 1], momentum[:, 1], pair_beta)[1]
            cosine = np.sum(p1pair*p2pair, axis=1)/(np.linalg.norm(p1pair, axis=1)*np.linalg.norm(p2pair, axis=1))
            scalar["max_angle_frame_difference"] = max(scalar["max_angle_frame_difference"],
                float(np.abs(cosine[exc]-a["cos_theta_secondary_correlation"][exc]).max(initial=0)))
            for channel in (1, 2):
                energy_hist[channel-1] += np.histogram(cm[ch == channel].ravel(), energy_edges)[0]
            # Only change the secondary-angle marginal; hold the current line shape fixed.
            x = a["cos_theta_secondary_correlation"][exc]
            baseline = angular_pdf(x, a["secondary_a2_used"][exc], a["secondary_a4_used"][exc])
            if np.any(baseline <= 0):
                raise ValueError("Angular control requires positive proposal density")
            weights = laursen_eq2_pdf(x)/baseline
            scalar["weight_sum"] += weights.sum()
            scalar["weight_square_sum"] += np.square(weights).sum()
            weighted_energy_hist += np.histogram(cm[exc].ravel(), energy_edges, weights=np.repeat(weights, 3))[0]
            eps = cm[exc]/cm[exc].sum(axis=1)[:, None]
            # Coordinate relabelling for display ONLY: does not add amplitude interference.
            for perm in itertools.permutations(range(3)):
                e = eps[:, perm]
                xx = (e[:, 0]+2*e[:, 1]-1)/np.sqrt(3)
                yy = e[:, 0]-1/3
                dalitz[0] += np.histogram2d(xx, yy, bins=(dalitz_edges, dalitz_edges))[0]/6
                dalitz[1] += np.histogram2d(xx, yy, bins=(dalitz_edges, dalitz_edges), weights=weights)[0]/6

    n = scalar["n"]
    if n == 0:
        raise ValueError("Empty reaction tree")
    if set(channels) != {1, 2} or config_values["resonance_id"] != {162}:
        raise ValueError("This baseline requires only the two 162-keV sequential channels")
    if config_values["direct_decay_fraction_162"] != {0.0} or config_values["sigma_gamma_total_barn"] != {0.0}:
        raise ValueError("This baseline requires gamma and direct components disabled")
    if config_values["scale_factor_675"] != {0.0}:
        raise ValueError("675-keV component must be disabled")
    ids = np.concatenate(event_ids)
    scalar["unique_events"] = len(np.unique(ids))
    scalar["event_min"], scalar["event_max"] = int(ids.min()), int(ids.max())
    scalar["q_mean"] = scalar["q_sum"]/n
    scalar["proton_mean"] = scalar["proton_sum"]/n
    scalar["reweight_neff"] = scalar["weight_sum"]**2/scalar["weight_square_sum"]

    specs = {"alpha0_primary": [1, 0, 0.6, 0, 0],
             "alpha1_primary": [1, 0, 0, 0, 0],
             "alpha1_secondary": [1, 0, -0.489, 0, 0.647]}
    angle_rows = []
    for key, a in angular.items():
        cdf = (Legendre(specs[key])/2).integ()
        expected = a["n"]*np.diff(cdf(angle_edges))
        statistic = float(np.sum((a["hist"]-expected)**2/expected))
        a["expected"] = expected
        a["chi2"], a["dof"], a["p"] = statistic, len(expected)-1, float(chi2.sf(statistic, len(expected)-1))
        for order in (1, 2, 4):
            mean = a["sum"][order]/a["n"]
            variance = max(0, a["sum2"][order]/a["n"]-mean**2)
            estimate = (2*order+1)*mean
            se = (2*order+1)*np.sqrt(variance/(a["n"]-1))
            angle_rows.append(dict(sample=key, order=order, estimate=estimate, se=se,
                                   configured=specs[key][order], z=(estimate-specs[key][order])/se))
    grid = np.linspace(-1, 1, 20001)
    scalar["angular_tv_eq2"] = float(np.trapezoid(np.abs(angular_pdf(grid)-laursen_eq2_pdf(grid)), grid)/2)
    checks = [
        ("162-keV sequential channels only", True, channels),
        ("Unique incident event IDs", scalar["unique_events"] == n, scalar["unique_events"]),
        ("Positive, finite generated alpha kinematics", scalar["invalid"] == 0, scalar["invalid"]),
        ("Energy conservation vs rounded Q=8.682 MeV (1 keV)", max(abs(scalar["q_min"]-LITERATURE_Q_MEV), abs(scalar["q_max"]-LITERATURE_Q_MEV)) < 0.001, scalar["q_mean"]),
        ("Momentum conservation (5e-5 MeV/c)", scalar["max_momentum_residual"] < 5e-5, scalar["max_momentum_residual"]),
        ("Independent CM reconstruction (5e-6 MeV)", scalar["max_cm_residual"] < 5e-6, scalar["max_cm_residual"]),
        ("Stored sigma vs current BW prescription (relative 1e-8)", scalar["max_sigma_relative_residual"] < 1e-8, scalar["max_sigma_relative_residual"]),
        ("Full three-alpha spectrum count = 3N", int(energy_hist.sum()) == 3*n, int(energy_hist.sum())),
    ]
    model_sigma_mb = float(sigma162_barn(parameters["er_keV"], parameters)*1000)
    f0_values = config_values["configured_162_alpha0_branching_fraction"]
    if len(f0_values) != 1:
        raise ValueError("Mixed branching configurations in input")
    f0 = next(iter(f0_values))
    scalar["branch_fraction_observed"] = channels[1]/n
    scalar["branch_fraction_configured"] = f0
    scalar["branch_fraction_z"] = (channels[1]-n*f0)/np.sqrt(n*f0*(1-f0))
    checks.append(("Configured branching sampling (absolute binomial z < 5)",
                   abs(scalar["branch_fraction_z"]) < 5, scalar["branch_fraction_z"]))
    for key, values in angular.items():
        checks.append((f"{key}: configured-density diagnostic (p > 1e-6)",
                       values["p"] > 1e-6, values["p"]))
    external = [
        dict(quantity="sigma_alpha0 at fitted resonance [mb]", model=f0*model_sigma_mb, paper=2.0, uncertainty=0.2, source="Munch 2018 p5 Eq10, own measurement"),
        dict(quantity="sigma_alpha1 at fitted resonance [mb]", model=(1-f0)*model_sigma_mb, paper=38.0, uncertainty=5.0, source="Munch 2018 p5 Eq12, own measurement"),
        dict(quantity="sigma_total at fitted resonance [mb]", model=model_sigma_mb, paper=41.0, uncertainty=3.0, source="Munch 2018 p6 Eq15, combined recommendation"),
        dict(quantity="resonance parameter E_lab [keV]", model=parameters["er_keV"]*12/11, paper=162.6, uncertainty=0.5, source="Munch 2018 p4, thick-target fit; fixed width"),
        dict(quantity="configured alpha0 fraction [%]", model=100*f0, paper=5.1, uncertainty=0.5, source="Laursen 2016 p10, excludes gs ghost; INPUT compatibility only"),
    ]
    for row in external:
        row["difference_in_paper_uncertainty"] = (row["model"]-row["paper"])/row["uncertainty"]
    tracked = ["src/H11BReaction.cc", "src/H11BCrossSection.cc", "src/H11BAngularDistribution.cc",
               "src/CoulombPenetrabilityData.cc", "include/Constants.hh", "macros/spectrum_162_lab_05_95_00.mac"]
    provenance = dict(root=str(path), root_sha256=sha256(path), root_bytes=path.stat().st_size,
                      source_sha256={p: sha256(PROJECT/p) for p in tracked}, runinfo=runinfo,
                      config_values={k: sorted(v) for k, v in config_values.items()},
                      analysis_script_sha256=sha256(__file__),
                      git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                      source_note="Current source checkout; original ROOT generation commit is not inferred from current HEAD.")
    return dict(provenance=provenance, scalar=scalar, parameters=parameters, channels=channels,
                checks=checks, angular=angular, angle_rows=angle_rows, external=external,
                angle_edges=angle_edges, energy_edges=energy_edges, energy_hist=energy_hist,
                weighted_energy_hist=weighted_energy_hist, dalitz_edges=dalitz_edges, dalitz=dalitz)


def plot_validation(result):
    """Return editable figures; do not change the caller's Matplotlib backend."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    r = result
    grid = np.linspace(-1, 1, 501)
    mid = (r["angle_edges"][:-1]+r["angle_edges"][1:])/2
    bw = np.diff(r["angle_edges"])
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), layout="constrained")
    for ax, key, title in zip(axes, r["angular"], [r"$\alpha_0$: primary angle", r"$\alpha_1$: primary angle", r"$\alpha_1$: secondary angle"]):
        a = r["angular"][key]
        ax.errorbar(mid, a["hist"]/a["n"]/bw, np.sqrt(a["hist"])/a["n"]/bw,
                    fmt=".", ms=3, color="black", label="Recorded generator")
        ax.stairs(a["expected"]/a["n"]/bw, r["angle_edges"], color="#0072B2", label="Configured density")
        if key == "alpha1_secondary":
            ax.plot(grid, laursen_eq2_pdf(grid), color="#D55E00", label="Laursen Eq. (2) only")
        ax.set(xlabel=r"$\cos\theta$", ylabel="Probability density", title=title, xlim=(-1, 1))
        ax.text(.03, .04, f"N={a['n']:,}\nconfigured: chi2/ndf={a['chi2']:.1f}/{a['dof']}, p={a['p']:.3g}",
                transform=ax.transAxes, fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
    figures.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), layout="constrained")
    densities = [h/h.sum()/np.diff(r["dalitz_edges"])[0]**2 for h in r["dalitz"]]
    norm = Normalize(0, max(np.quantile(h[h>0], .995) for h in densities))
    for ax, h, title in zip(axes[:2], densities, ["Current generator", "Eq. (2) angular control only"]):
        im = ax.pcolormesh(r["dalitz_edges"], r["dalitz_edges"], h.T, cmap="viridis", norm=norm, shading="auto")
        ax.set(xlabel=r"$\chi$", ylabel=r"$\psi$", title=title, aspect="equal")
    fig.colorbar(im, ax=axes[:2], label="Normalized density", shrink=.8)
    e = r["energy_edges"]
    for h, label, color in [(r["energy_hist"][1], "Current generator", "#0072B2"),
                            (r["weighted_energy_hist"], "Eq. (2) angular control", "#D55E00")]:
        axes[2].stairs(h/h.sum()/np.diff(e), e, label=label, color=color)
    axes[2].set(xlabel=r"$T_\alpha$ in three-alpha CM [MeV]", ylabel="Probability density [1/MeV]",
                title="Alpha1-channel single-alpha spectrum")
    axes[2].legend(fontsize=8)
    fig.suptitle("Generator-level sensitivity; no detector response, no Bose-interference correction", fontsize=11)
    figures.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    ep = np.linspace(148, 180, 801)
    axes[0].plot(ep, 1000*sigma162_barn(ep*11/12, r["parameters"]), color="#0072B2", label="Current physical BW cross section")
    axes[0].errorbar([162.6], [41.0], xerr=[.5], yerr=[3.0], fmt="o", color="black",
                     label="Munch combined peak normalization")
    axes[0].set(xlabel="Proton LAB energy [keV]", ylabel="Three-alpha reaction cross section [mb]", title="External normalization / energy check")
    axes[0].legend(fontsize=8)
    # The measured angular fit has different beam/target conditions. No chi-square is computed.
    current = Legendre([1, 0, .6])(grid)
    published = Legendre([1, -.358, .249, -.106, -.009])(grid)
    axes[1].plot(grid, current/Legendre([1, 0, .6])(0), color="#0072B2", label="Current alpha0: monoenergetic input")
    axes[1].plot(grid, published/Legendre([1, -.358, .249, -.106, -.009])(0), color="#D55E00",
                 label="Munch Eq. (5): Ep=175 keV, thick target")
    axes[1].set(xlabel=r"$\cos\theta_{CM}$", ylabel=r"$W(\theta)/W(90^\circ)$", title="Different conditions: reference shapes only")
    axes[1].legend(fontsize=8)
    figures.append(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--save-dir", type=Path, help="Optional figure directory; otherwise write no files")
    args = parser.parse_args()
    result = validate_generator(args.root)
    for label, passed, value in result["checks"]:
        print(f"{'PASS' if passed else 'FAIL'} | {label} | {value}")
    for row in result["external"]:
        print(row)
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for i, figure in enumerate(plot_validation(result), 1):
            figure.savefig(args.save_dir/f"generator_162_validation_{i}.png", dpi=160)
    print("This is not a detector-folded experimental validation or a measurement of branching fractions.")
    return 0 if all(c[1] for c in result["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
