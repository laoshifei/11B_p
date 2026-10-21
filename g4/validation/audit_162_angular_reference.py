#!/usr/bin/env python3
"""Audit angular references for 162-keV p+11B without changing the generator.

The angular part of a single sequential J=2 -> j=2 -> 0 path is evaluated
independently of the printed Laursen Eq. (2). No Bose symmetrization, radial
R-matrix amplitude, detector simulation, or experimental fit is performed.
The accompanying local notebook records the literature and interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.polynomial import Legendre
from scipy.integrate import quad
from scipy.special import sph_harm_y
from scipy.stats import chi2


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "data/spectrum_162_lab_05_95_00.root"
TREADO_A2, TREADO_A4 = -0.489, 0.647
ALPHA_MASS_MEV = 3727.3794066


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legendre_pdf(x, a2=TREADO_A2, a4=TREADO_A4):
    """Probability density with respect to x=cos(chi), not chi."""
    x2 = np.asarray(x)**2
    return (1 + a2 * (3*x2-1)/2 + a4 * (35*x2*x2-30*x2+3)/8) / 2


def published_eq2_pdf(x):
    """Literal published Eq. (2), normalized in dx; its derivation is audited."""
    x = np.asarray(x)
    return (1.12 + 3.2 * x**2 - 3.2 * x**4) / (2 * 1.12 + 2 * 3.2 / 3 - 2 * 3.2 / 5)


def single_path_l2_pdf(x):
    """Unpolarized single-path L=j=J=2 angular coupling, normalized in dx."""
    x = np.asarray(x)
    return 5 / 28 * (9 * x**4 - 9 * x**2 + 4)


def clebsch_gordan(j1, m1, j2, m2, total_j, total_m):
    """Integer-spin Racah factorial sum, Condon-Shortley convention.

    The independent total-J-squared eigensolver below checks squared
    coefficients without using this factorial formula.
    """
    if any(not isinstance(v, (int, np.integer)) for v in (j1, m1, j2, m2, total_j, total_m)):
        raise ValueError("Only integer angular momenta are supported")
    if (min(j1, j2, total_j) < 0 or m1 + m2 != total_m
            or abs(m1) > j1 or abs(m2) > j2 or abs(total_m) > total_j
            or not abs(j1 - j2) <= total_j <= j1 + j2):
        return 0.0
    f = math.factorial
    numerator = ((2 * total_j + 1) * f(j1 + j2 - total_j)
                 * f(j1 - j2 + total_j) * f(-j1 + j2 + total_j)
                 * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2)
                 * f(total_j + total_m) * f(total_j - total_m))
    prefactor = math.sqrt(numerator / f(j1 + j2 + total_j + 1))
    series = 0.0
    for k in range(j1 + j2 + total_j + 1):
        args = (k, j1 + j2 - total_j - k, j1 - m1 - k, j2 + m2 - k,
                total_j - j2 + m1 + k, total_j - j1 - m2 + k)
        if min(args) >= 0:
            series += (-1)**k / math.prod(f(v) for v in args)
    return prefactor * series


def cg_squared_from_j2(primary_l, total_m, total_j=2, secondary_j=2):
    """Obtain |<L,0;j,M|J,M>|^2 by diagonalizing (L+j)^2."""
    basis = [(ml, total_m - ml) for ml in range(-primary_l, primary_l + 1)
             if abs(total_m - ml) <= secondary_j]
    matrix = np.zeros((len(basis), len(basis)))
    ll, jj = primary_l * (primary_l + 1), secondary_j * (secondary_j + 1)
    for i, (ml, mj) in enumerate(basis):
        matrix[i, i] = ll + jj + 2 * ml * mj
        raised = (ml + 1, mj - 1)
        if raised in basis:
            k = basis.index(raised)
            matrix[i, k] = matrix[k, i] = math.sqrt((ll - ml * (ml + 1)) * (jj - mj * (mj - 1)))
    values, vectors = np.linalg.eigh(matrix)
    index = int(np.argmin(abs(values - total_j * (total_j + 1))))
    if abs(values[index] - total_j * (total_j + 1)) > 1e-10:
        raise ValueError("Requested total angular momentum not present")
    return float(vectors[basis.index((0, total_m)), index]**2)


def pdf_from_coupling(x, primary_l=2, *, method="racah"):
    """Fix the primary direction to z and sum over an unpolarized parent."""
    theta = np.arccos(np.clip(x, -1, 1))
    weights = np.array([
        clebsch_gordan(primary_l, 0, 2, m, 2, m)**2 if method == "racah"
        else cg_squared_from_j2(primary_l, m) for m in range(-2, 3)])
    if method not in ("racah", "eigensolver"):
        raise ValueError("Unknown coupling method")
    density = sum(w * np.abs(sph_harm_y(2, m, theta, 0))**2
                  for m, w in zip(range(-2, 3), weights))
    return 2 * np.pi * density / weights.sum()


def joint_angular_density(primary, secondary, primary_l=2):
    """Angular part of Eq. (3) for one path, averaged over M=-2,...,2.

    Both directions use common coordinate axes; radial factors are held
    constant. This is a density in dOmega_primary dOmega_secondary.
    """
    p = np.asarray(primary, dtype=float)
    s = np.asarray(secondary, dtype=float)
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    s = s / np.linalg.norm(s, axis=-1, keepdims=True)
    tp, pp = np.arccos(np.clip(p[..., 2], -1, 1)), np.arctan2(p[..., 1], p[..., 0])
    ts, ps = np.arccos(np.clip(s[..., 2], -1, 1)), np.arctan2(s[..., 1], s[..., 0])
    density = np.zeros(np.broadcast_shapes(tp.shape, ts.shape))
    for total_m in range(-2, 3):
        amplitude = np.zeros_like(density, dtype=complex)
        for secondary_m in range(-2, 3):
            primary_m = total_m - secondary_m
            if abs(primary_m) <= primary_l:
                cg = clebsch_gordan(primary_l, primary_m, 2, secondary_m, 2, total_m)
                amplitude += (cg * sph_harm_y(primary_l, primary_m, tp, pp)
                              * sph_harm_y(2, secondary_m, ts, ps))
        density += np.abs(amplitude)**2 / 5
    return density


def planar_weight(theta, shift_deg=0.0, a2=TREADO_A2, a4=TREADO_A4):
    """Treado's in-plane intensity; theta is radians, NOT a polar-angle PDF."""
    x = np.cos(np.asarray(theta) - np.deg2rad(shift_deg))
    return 2 * legendre_pdf(x, a2, a4)


def total_variation(f, g, lo=-1.0, hi=1.0):
    """Shape distance between normalized densities; not a p-value."""
    return quad(lambda x: abs(float(f(x) - g(x))) / 2, lo, hi,
                epsabs=2e-11, limit=300)[0]


def reference_audit(seed=16220261019):
    grid = np.linspace(-1, 1, 1001)
    cg_racah = [clebsch_gordan(2, 0, 2, m, 2, m)**2 for m in range(-2, 3)]
    cg_eigen = [cg_squared_from_j2(2, m) for m in range(-2, 3)]
    rng = np.random.default_rng(seed)
    primary, secondary = rng.normal(size=(2, 256, 3))
    primary /= np.linalg.norm(primary, axis=1)[:, None]
    secondary /= np.linalg.norm(secondary, axis=1)[:, None]
    x = np.sum(primary * secondary, axis=1)
    density = joint_angular_density(primary, secondary)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    rotated = joint_angular_density(primary @ rotation, secondary @ rotation)
    models = {"current": legendre_pdf, "published_eq2": published_eq2_pdf,
              "single_path_l2": single_path_l2_pdf}
    coefficients = {name: {
        "normalization": quad(lambda z: float(pdf(z)), -1, 1)[0],
        **{f"a{k}": (2 * k + 1) * quad(lambda z: float(pdf(z) * Legendre.basis(k)(z)), -1, 1)[0]
           for k in (2, 4)}} for name, pdf in models.items()}
    # Normalize in-plane intensities with dtheta, not dx. The angular interval
    # has period pi. This is a shape-only control, not an experimental fit.
    normalizations = {name: quad(lambda t: float(2 * pdf(np.cos(t))), 0, np.pi)[0]
                      for name, pdf in models.items()}
    plane_eq2 = lambda t: 2 * published_eq2_pdf(np.cos(t)) / normalizations["published_eq2"]
    plane_l2 = lambda t: 2 * single_path_l2_pdf(np.cos(t)) / normalizations["single_path_l2"]
    shift_rows = []
    for shift in (-22.0, -12.0, -2.0, 0.0):
        plane = lambda t: planar_weight(t, shift) / normalizations["current"]
        shift_rows.append(dict(shift_deg=shift,
                              tv_vs_eq2=total_variation(plane, plane_eq2, 0, np.pi),
                              tv_vs_l2=total_variation(plane, plane_l2, 0, np.pi)))
    return dict(seed=seed, cg_squared_racah=cg_racah, cg_squared_eigensolver=cg_eigen,
                max_cg_squared_difference=float(np.max(np.abs(np.array(cg_racah)-cg_eigen))),
                max_coupling_closed_form_difference=float(np.max(abs(pdf_from_coupling(grid)-single_path_l2_pdf(grid)))),
                max_eigensolver_closed_form_difference=float(np.max(abs(pdf_from_coupling(grid, method="eigensolver")-single_path_l2_pdf(grid)))),
                max_rotation_difference=float(np.max(abs(density-rotated))),
                max_general_amplitude_difference=float(np.max(abs(density*8*np.pi**2-single_path_l2_pdf(x)))),
                max_axis_reversal_difference=float(np.max(abs(single_path_l2_pdf(grid)-single_path_l2_pdf(-grid)))),
                coefficients=coefficients,
                total_variation_dx=dict(current_vs_eq2=total_variation(legendre_pdf, published_eq2_pdf),
                                        current_vs_l2=total_variation(legendre_pdf, single_path_l2_pdf),
                                        eq2_vs_l2=total_variation(published_eq2_pdf, single_path_l2_pdf)),
                plane_shift_controls=shift_rows)


def _rest_frame(energy, momentum, beta):
    b2 = np.sum(beta * beta, axis=1)
    gamma = 1 / np.sqrt(1 - b2)
    bp = np.sum(beta * momentum, axis=1)
    factor = np.divide(gamma - 1, b2, out=np.zeros_like(b2), where=b2 > 0) * bp - gamma * energy
    return gamma * (energy - bp), momentum + factor[:, None] * beta


def _new_summary(bins):
    return dict(n=0, hist=np.zeros(len(bins)-1, dtype=int), sums=np.zeros(2), sums2=np.zeros(2))


def _accumulate(summary, x, bins):
    summary["n"] += len(x)
    summary["hist"] += np.histogram(x, bins)[0]
    for i, order in enumerate((2, 4)):
        v = (2 * order + 1) * Legendre.basis(order)(x)
        summary["sums"][i] += v.sum()
        summary["sums2"][i] += np.square(v).sum()


def audit_recorded_sample(path=DEFAULT_ROOT, chunk_size=50000, primary_half_width_deg=5.0,
                          azimuth_half_width_deg=10.0):
    """Read existing ROOT in chunks and apply explicitly idealized gates.

    The secondary azimuth gate is in the 8Be rest frame around the
    beam-primary plane, with constant azimuthal width. It is NOT Treado's
    detector aperture, LAB theta2 scan, peak fit, or experimental acceptance.
    """
    import uproot

    path = Path(path).resolve()
    if not 0 < primary_half_width_deg < 90 or not 0 < azimuth_half_width_deg < 90:
        raise ValueError("Gate half-widths must lie between 0 and 90 degrees")
    bins = np.linspace(-1, 1, 41)
    summaries = {key: _new_summary(bins) for key in ("all_alpha1", "primary_lab_90", "primary_90_and_plane")}
    azimuth_bins = np.linspace(-np.pi, np.pi, 25)
    azimuth_hist = np.zeros(24, dtype=int)
    maximum_angle_difference = 0.0
    count, energy_min, energy_max = 0, np.inf, -np.inf
    columns = ["h11b_reaction_channel", "resonance_id", "projectile_kinetic_lab",
               "primary_a1_used", "primary_a2_used", "secondary_a2_used", "secondary_a4_used",
               "secondary_angular_model", "cos_theta_secondary_correlation"]
    columns += [f"projectile_p{axis}_lab" for axis in "xyz"]
    columns += [f"{key}_alpha{i}" for i in (1, 2, 3) for key in ("e", "theta_lab", "phi_lab")]
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as file:
        tree = file["reaction"]
        missing = set(columns)-set(tree.keys())
        if missing:
            raise ValueError(f"Missing branches: {sorted(missing)}")
        runinfo = {k: v.tolist() for k, v in file["RunInfo"].arrays(library="np").items()} if "RunInfo" in file else {}
        for arrays in tree.iterate(columns, step_size=chunk_size, library="np"):
            count += len(arrays["h11b_reaction_channel"])
            chosen = (arrays["h11b_reaction_channel"] == 2) & (arrays["resonance_id"] == 162)
            a = {k: v[chosen] for k, v in arrays.items()}
            if not len(a["resonance_id"]):
                continue
            expected = {"primary_a1_used": 0.0, "primary_a2_used": 0.0,
                        "secondary_a2_used": TREADO_A2, "secondary_a4_used": TREADO_A4,
                        "secondary_angular_model": 1}
            for key, value in expected.items():
                if not np.all(a[key] == value):
                    raise ValueError(f"Unexpected recorded setting: {key}")
            kinetic = np.column_stack([a[f"e_alpha{i}"] for i in (1, 2, 3)])
            theta = np.column_stack([a[f"theta_lab_alpha{i}"] for i in (1, 2, 3)])
            phi = np.column_stack([a[f"phi_lab_alpha{i}"] for i in (1, 2, 3)])
            magnitude = np.sqrt(kinetic*(kinetic+2*ALPHA_MASS_MEV))
            momentum = magnitude[:, :, None] * np.stack(
                [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=-1)
            energy = kinetic + ALPHA_MASS_MEV
            pair_beta = (momentum[:, 1]+momentum[:, 2]) / (energy[:, 1]+energy[:, 2])[:, None]
            _, primary_rest = _rest_frame(energy[:, 0], momentum[:, 0], pair_beta)
            _, secondary_rest = _rest_frame(energy[:, 1], momentum[:, 1], pair_beta)
            primary_unit = primary_rest / np.linalg.norm(primary_rest, axis=1)[:, None]
            secondary_unit = secondary_rest / np.linalg.norm(secondary_rest, axis=1)[:, None]
            x = np.clip(np.sum(primary_unit*secondary_unit, axis=1), -1, 1)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite reconstructed angle")
            maximum_angle_difference = max(maximum_angle_difference,
                float(np.max(abs(x-a["cos_theta_secondary_correlation"]))))
            incoming = np.column_stack([a[f"projectile_p{axis}_lab"] for axis in "xyz"])
            primary_lab_cos = np.sum(incoming*momentum[:, 0], axis=1) / (
                np.linalg.norm(incoming, axis=1)*magnitude[:, 0])
            gate_primary = abs(primary_lab_cos) <= np.sin(np.deg2rad(primary_half_width_deg))
            # Both boosts lie in the beam-primary plane, so its normal is
            # unchanged. At beam-collinear endpoints the plane is undefined;
            # those events are outside the 90-degree gate.
            normal = np.cross(incoming, momentum[:, 0])
            norm = np.linalg.norm(normal, axis=1)
            normal = np.divide(normal, norm[:, None], out=np.zeros_like(normal), where=norm[:, None] > 0)
            in_plane = np.cross(normal, primary_unit)
            azimuth = np.arctan2(np.sum(secondary_unit*normal, axis=1),
                                 np.sum(secondary_unit*in_plane, axis=1))
            gate_plane = gate_primary & (abs(np.sin(azimuth)) <= np.sin(np.deg2rad(azimuth_half_width_deg)))
            for key, mask in (("all_alpha1", np.ones(len(x), dtype=bool)),
                              ("primary_lab_90", gate_primary), ("primary_90_and_plane", gate_plane)):
                _accumulate(summaries[key], x[mask], bins)
            azimuth_hist += np.histogram(azimuth[gate_primary], azimuth_bins)[0]
            energy_min = min(energy_min, float(a["projectile_kinetic_lab"].min()))
            energy_max = max(energy_max, float(a["projectile_kinetic_lab"].max()))
    primitive = Legendre([0.5, 0, TREADO_A2/2, 0, TREADO_A4/2]).integ()
    probabilities = np.diff(primitive(bins))
    for summary in summaries.values():
        n = summary["n"]
        if n < 2:
            raise ValueError("Too few selected events")
        summary["coefficients"] = (summary["sums"]/n).tolist()
        summary["coefficient_se"] = np.sqrt(np.maximum(0,
            summary["sums2"]-summary["sums"]**2/n)/(n*(n-1))).tolist()
        expected = n*probabilities
        summary["chi2"] = float(np.sum((summary["hist"]-expected)**2/expected))
        summary["ndf"] = len(expected)-1
        summary["p_value"] = float(chi2.sf(summary["chi2"], summary["ndf"]))
        for key in ("hist", "sums", "sums2"):
            summary[key] = summary[key].tolist()
    uniform = azimuth_hist.sum()/len(azimuth_hist)
    phi_chi2 = float(np.sum((azimuth_hist-uniform)**2/uniform))
    return dict(root=str(path), root_sha256=sha256(path), runinfo=runinfo,
                reaction_entries=count, summaries=summaries, cos_edges=bins.tolist(),
                proton_energy_range_MeV=[energy_min, energy_max],
                max_reconstructed_angle_difference=maximum_angle_difference,
                primary_lab_gate_deg=[90-primary_half_width_deg, 90+primary_half_width_deg],
                secondary_azimuth_half_width_deg=azimuth_half_width_deg,
                azimuth_edges_rad=azimuth_bins.tolist(), azimuth_hist=azimuth_hist.tolist(),
                azimuth_chi2=phi_chi2, azimuth_ndf=len(azimuth_hist)-1,
                azimuth_p_value=float(chi2.sf(phi_chi2, len(azimuth_hist)-1)))


def run_audit(root_path=DEFAULT_ROOT, *, include_root=True, chunk_size=50000):
    reference = reference_audit()
    sample = audit_recorded_sample(root_path, chunk_size=chunk_size) if include_root else None
    source_paths = [Path(__file__), PROJECT/"src/H11BAngularDistribution.cc",
                    PROJECT/"src/H11BReaction.cc", PROJECT/"include/Constants.hh",
                    PROJECT/"macros/spectrum_162_lab_05_95_00.mac"]
    provenance = dict(computed_at=datetime.now().astimezone().isoformat(),
                      current_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
                      source_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in source_paths},
                      note="Current checkout hashes; no claim that current HEAD generated the historical ROOT file.")
    return dict(provenance=provenance, reference=reference, sample=sample)


def plot_audit(result):
    """Return figures without changing the caller's Matplotlib backend."""
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 1001)
    fig, ax = plt.subplots(figsize=(9.4, 5.6), layout="constrained")
    for fn, color, label in ((legendre_pdf, "#0072B2", "Current empirical coefficients"),
                             (published_eq2_pdf, "#D55E00", "Published Eq. (2), literal"),
                             (single_path_l2_pdf, "#009E73", "Unpolarized single-path L=2 coupling")):
        ax.plot(x, fn(x), color=color, label=label, lw=2)
    ax.set(xlabel=r"$x=\cos\chi$ in the $^8$Be rest frame", ylabel=r"Density in $dx$", xlim=(-1, 1))
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.16))
    ax.set_title("Angular references: no Bose sum or detector response")
    figures = [fig]
    theta = np.linspace(0, np.pi, 721)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    for shift, linestyle in ((0, "-"), (-12, "--")):
        w = planar_weight(theta, shift)
        w /= np.trapezoid(w, theta)/np.pi
        axes[0].plot(np.rad2deg(theta), w, linestyle, color="#0072B2", label=f"Treado coefficients, shift {shift} deg")
    for fn, color, label in ((published_eq2_pdf, "#D55E00", "Published Eq. (2)"),
                             (single_path_l2_pdf, "#009E73", "Single-path L=2")):
        w = fn(np.cos(theta))
        w /= np.trapezoid(w, theta)/np.pi
        axes[0].plot(np.rad2deg(theta), w, color=color, label=label)
    axes[0].set(xlabel="In-plane angle (deg)", ylabel="Intensity / in-plane mean", xlim=(0, 180),
                title="Plane comparison (not a polar-angle PDF)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
    rows = result["reference"]["plane_shift_controls"]
    for key, color, label in (("tv_vs_eq2", "#D55E00", "vs published Eq. (2)"),
                              ("tv_vs_l2", "#009E73", "vs single-path L=2")):
        axes[1].plot([r["shift_deg"] for r in rows], [r[key] for r in rows], "o-", color=color, label=label)
    axes[1].set(xlabel="Treado symmetry-axis shift (deg)", ylabel="In-plane shape distance (TV)",
                title="Shift sensitivity; not a fit to experimental points", ylim=(0, None))
    axes[1].legend(fontsize=8)
    figures.append(fig)
    sample = result.get("sample")
    if sample:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        edges = np.asarray(sample["cos_edges"])
        labels = {"all_alpha1": "All generated alpha1 events", "primary_lab_90": "Primary LAB 90 +/- 5 deg",
                  "primary_90_and_plane": "Plus rest-frame azimuth gate"}
        for key, row in sample["summaries"].items():
            n, hist = row["n"], np.asarray(row["hist"])
            if key == "primary_90_and_plane":
                axes[0].errorbar((edges[1:]+edges[:-1])/2, hist/n/np.diff(edges),
                                 np.sqrt(hist)/n/np.diff(edges), fmt=".", color="#CC79A7", label=labels[key])
            else:
                axes[0].stairs(hist/n/np.diff(edges), edges, label=labels[key], alpha=.75)
        axes[0].plot(x, legendre_pdf(x), "k--", lw=1, label="Configured density")
        axes[0].set(xlabel=r"$\cos\chi$", ylabel="Density in dx", xlim=(-1, 1), title="Idealized selection control")
        axes[0].legend(fontsize=8)
        edges = np.rad2deg(sample["azimuth_edges_rad"])
        hist = np.asarray(sample["azimuth_hist"])
        axes[1].stairs(hist/hist.sum()/np.diff(edges), edges)
        axes[1].axhline(1/360, color="black", ls="--", label="Uniform azimuth")
        axes[1].set(xlabel="Secondary azimuth around primary axis (deg)", ylabel="Density per degree",
                    title="Within primary LAB 90 +/- 5 deg", xlim=(-180, 180), ylim=(0, None))
        axes[1].legend(fontsize=8)
        figures.append(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--skip-root", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=50000)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.root, include_root=not args.skip_root, chunk_size=args.chunk_size),
                     ensure_ascii=False, indent=2))
