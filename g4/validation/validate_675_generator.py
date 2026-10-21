#!/usr/bin/env python3
"""Read-only checks of a pure 675-keV coherent L1/L3 Geant4 sample.

The report concerns at-generation kinematics and recorded sampling diagnostics.
It does not independently validate the complex decay amplitude or experimental
agreement. Reading is bounded in chunks; importing does not change Matplotlib's
backend. Default execution prints a compact report and writes no files.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import uproot
from numpy.polynomial import Legendre


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "data/spectrum_675_lab_coherent_L1L3_1M_20261020.root"
# Actual installed Geant4 masses, independently queried for the preceding runs.
ALPHA_MASS_MEV = 3727.379
PROTON_MASS_MEV = 938.272013
BORON11_MASS_MEV = 10252.547872616791
REACTION_Q_MEV = PROTON_MASS_MEV + BORON11_MASS_MEV - 3 * ALPHA_MASS_MEV


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def boost_to_rest(energy, momentum, beta):
    """Boost each row into a frame moving with beta, using c=1."""
    beta2 = np.sum(beta * beta, axis=-1)
    gamma = 1 / np.sqrt(1 - beta2)
    dot = np.sum(beta * momentum, axis=-1)
    coefficient = np.divide(gamma - 1, beta2, out=np.zeros_like(beta2), where=beta2 > 0) * dot - gamma * energy
    return gamma * (energy - dot), momentum + coefficient[:, None] * beta


def validate_675_sample(root_path=DEFAULT_ROOT, *, incident_count=1_000_000,
                        expected_threads=8, chunk_size=50_000):
    """Return checks, provenance and small histograms, without saving files.

    Incident count is the prescribed beamOn budget. The observed reaction count
    is reported separately and never silently increased to meet that budget.
    Orientation diagnostics use the per-event mean of the three alpha moments,
    accounting for correlations of particles belonging to the same event.
    """
    path = Path(root_path).resolve()
    energy_edges = np.linspace(0, 7, 141)  # 50 keV, each alpha counted once.
    dalitz_edges = np.linspace(-1 / 3, 1 / 3, 101)
    ratio_edges = np.linspace(0, 1, 101)
    energy_hist = np.zeros(140, dtype=np.int64)
    dalitz_hist = np.zeros((100, 100))
    ratio_hist = np.zeros(100, dtype=np.int64)
    attempts_hist = np.zeros(10001, dtype=np.int64)
    residuals = dict(energy_MeV=0., momentum_MeV_c=0., cm_saved_MeV=0.,
                     cm_energy_MeV=0., cm_momentum_MeV_c=0.)
    moments_sum, moments_sum2 = np.zeros(4), np.zeros(4)
    state = dict(reactions=0, invalid=0, bad_configuration=0, envelope_violations=0,
                 attempt_limit_hits=0, max_weight_envelope_ratio=0., max_attempts=0,
                 attempt_sum=0, proton_min_keV=np.inf, proton_max_keV=-np.inf,
                 q3_min_MeV=np.inf, q3_max_MeV=-np.inf)
    all_ids = []
    expected = {
        "resonance_id": 675, "h11b_reaction_channel": 4,
        "scale_factor_162": 0, "scale_factor_675": 1,
        "direct_decay_fraction_675": 0, "sigma_gamma_total_barn": 0,
        "cross_section_bias_factor": 1e11,
        "h11b675_alpha_decay_model": 1, "h11b675_decay_model_used": 2,
        "h11b675_strict_coherent_l13": 1,
        "h11b675_strict_permutation_symmetrized": 1,
        "h11b675_strict_l1_fraction": .76,
        "h11b675_strict_l13_phase": .67 * 2 * np.pi,
        "h11b675_strict_8be_lambda_energy_keV": 3037.,
        "h11b675_strict_8be_reduced_width_squared_keV": 1075.,
        "h11b162_decay_model_used": 0, "h11b162_bose_sampling_attempts": 0,
    }
    fields = list(expected) + ["event", "projectile_kinetic_lab",
                              "h11b675_strict_weight", "h11b675_strict_weight_max",
                              "h11b675_strict_sampling_attempts"]
    fields += [f"{name}_alpha{i}" for i in (1, 2, 3)
               for name in ("e", "theta_lab", "phi_lab", "e_3alpha_cm")]
    fields += [f"projectile_p{axis}_lab" for axis in "xyz"]
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as stream:
        tree = stream["reaction"]
        missing = set(fields) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing validation fields: {sorted(missing)}")
        runinfo = {key: value.tolist() for key, value in stream["RunInfo"].arrays(library="np").items()}
        for a in tree.iterate(fields, step_size=chunk_size, library="np"):
            n = len(a["event"])
            state["reactions"] += n
            all_ids.append(a["event"])
            bad_config = np.zeros(n, dtype=bool)
            for name, value in expected.items():
                bad_config |= ~np.isclose(a[name], value, atol=1e-12, rtol=1e-12)
            state["bad_configuration"] += int(bad_config.sum())
            if bad_config.any():
                raise ValueError("Sample is not the prescribed pure 675 coherent L1/L3 configuration")
            kinetic = np.column_stack([a[f"e_alpha{i}"] for i in (1, 2, 3)])
            theta = np.column_stack([a[f"theta_lab_alpha{i}"] for i in (1, 2, 3)])
            phi = np.column_stack([a[f"phi_lab_alpha{i}"] for i in (1, 2, 3)])
            saved_cm = np.column_stack([a[f"e_3alpha_cm_alpha{i}"] for i in (1, 2, 3)])
            tp = a["projectile_kinetic_lab"]
            incoming = np.column_stack([a[f"projectile_p{axis}_lab"] for axis in "xyz"])
            invalid = (~np.isfinite(kinetic).all(axis=1) | (kinetic <= 0).any(axis=1)
                       | ~np.isfinite(theta).all(axis=1) | ~np.isfinite(phi).all(axis=1)
                       | ~np.isfinite(saved_cm).all(axis=1) | (saved_cm <= 0).any(axis=1)
                       | ~np.isfinite(incoming).all(axis=1) | ~np.isfinite(tp) | (tp <= 0))
            state["invalid"] += int(invalid.sum())
            if invalid.any():
                raise ValueError("Invalid generated energies, directions or incoming momentum")
            energy = kinetic + ALPHA_MASS_MEV
            magnitude = np.sqrt(kinetic * (kinetic + 2 * ALPHA_MASS_MEV))
            momentum = magnitude[..., None] * np.stack(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
            beta = momentum.sum(axis=1) / energy.sum(axis=1)[:, None]
            rest = [boost_to_rest(energy[:, i], momentum[:, i], beta) for i in range(3)]
            cm = np.column_stack([row[0] - ALPHA_MASS_MEV for row in rest])
            pcm = np.stack([row[1] for row in rest], axis=1)
            q3 = np.sqrt((PROTON_MASS_MEV + BORON11_MASS_MEV)**2
                         + 2 * BORON11_MASS_MEV * tp) - 3 * ALPHA_MASS_MEV
            new_residuals = {
                "energy_MeV": np.max(abs(kinetic.sum(axis=1) - tp - REACTION_Q_MEV)),
                "momentum_MeV_c": np.max(np.linalg.norm(momentum.sum(axis=1) - incoming, axis=1)),
                "cm_saved_MeV": np.max(abs(cm - saved_cm)),
                "cm_energy_MeV": np.max(abs(cm.sum(axis=1) - q3)),
                "cm_momentum_MeV_c": np.max(np.linalg.norm(pcm.sum(axis=1), axis=1)),
            }
            for key,value in new_residuals.items():
                residuals[key] = max(residuals[key], float(value))
            weight = a["h11b675_strict_weight"]
            envelope = a["h11b675_strict_weight_max"]
            attempts = a["h11b675_strict_sampling_attempts"]
            if not (np.isfinite(weight).all() and np.isfinite(envelope).all()
                    and (weight > 0).all() and (envelope > 0).all()):
                raise ValueError("Invalid recorded sampling weights")
            ratio = weight / envelope
            state["envelope_violations"] += int((ratio > 1).sum())
            state["attempt_limit_hits"] += int((attempts >= 10000).sum())
            if (attempts < 1).any():
                raise ValueError("Nonpositive sampling attempt count")
            state["max_weight_envelope_ratio"] = max(state["max_weight_envelope_ratio"], float(ratio.max()))
            state["max_attempts"] = max(state["max_attempts"], int(attempts.max()))
            state["attempt_sum"] += int(attempts.sum())
            state["proton_min_keV"] = min(state["proton_min_keV"], float(tp.min() * 1000))
            state["proton_max_keV"] = max(state["proton_max_keV"], float(tp.max() * 1000))
            state["q3_min_MeV"] = min(state["q3_min_MeV"], float(q3.min()))
            state["q3_max_MeV"] = max(state["q3_max_MeV"], float(q3.max()))
            energy_hist += np.histogram(cm.ravel(), energy_edges)[0]
            ratio_hist += np.histogram(ratio, ratio_edges)[0]
            attempts_hist += np.bincount(np.minimum(attempts, 10000), minlength=10001)
            eps = cm / cm.sum(axis=1)[:, None]
            # Six coordinate relabelings, total display weight ONE per event.
            for permutation in itertools.permutations(range(3)):
                ep = eps[:, permutation]
                x = (ep[:, 1] - ep[:, 2]) / np.sqrt(3)
                y = ep[:, 0] - 1 / 3
                dalitz_hist += np.histogram2d(x, y, bins=(dalitz_edges, dalitz_edges))[0] / 6
            cosine = pcm[..., 2] / np.linalg.norm(pcm, axis=-1)
            event_moments = np.column_stack([Legendre.basis(order)(cosine).mean(axis=1)
                                            for order in range(1, 5)])
            moments_sum += event_moments.sum(axis=0)
            moments_sum2 += (event_moments**2).sum(axis=0)
    n = state["reactions"]
    if n == 0:
        raise ValueError("Empty reaction tree")
    ids = np.concatenate(all_ids)
    state["unique_events"] = len(np.unique(ids))
    state["event_min"], state["event_max"] = int(ids.min()), int(ids.max())
    state["mean_attempts"] = state["attempt_sum"] / n
    means = moments_sum / n
    errors = np.sqrt(np.maximum(moments_sum2 / n - means**2, 0) / (n - 1))
    z = np.divide(means, errors, out=np.zeros_like(means), where=errors > 0)
    seed_values = sorted(set(runinfo.get("random_seed", [])))
    checks = [
        dict(name="pure 675 model configuration", passed=state["bad_configuration"] == 0, value=state["bad_configuration"]),
        dict(name="valid finite positive kinematics", passed=state["invalid"] == 0, value=state["invalid"]),
        dict(name="unique incident IDs within beamOn budget", passed=(state["unique_events"] == n
             and 0 <= state["event_min"] <= state["event_max"] < incident_count and n <= incident_count), value=n),
        # RootIO fills RunInfo in worker 0 only; thread completion is in the log.
        dict(name="RunInfo single global seed record", passed=(len(runinfo.get("random_seed", [])) == 1 and len(seed_values) == 1 and seed_values[0] > 0), value=seed_values),
        dict(name="LAB energy conservation [MeV]", passed=residuals["energy_MeV"] < 1e-7, value=residuals["energy_MeV"]),
        dict(name="LAB momentum conservation [MeV/c]", passed=residuals["momentum_MeV_c"] < 1e-7, value=residuals["momentum_MeV_c"]),
        dict(name="independent CM saved-energy reconstruction [MeV]", passed=residuals["cm_saved_MeV"] < 5e-6, value=residuals["cm_saved_MeV"]),
        dict(name="CM energy budget [MeV]", passed=residuals["cm_energy_MeV"] < 1e-7, value=residuals["cm_energy_MeV"]),
        dict(name="CM momentum closure [MeV/c]", passed=residuals["cm_momentum_MeV_c"] < 1e-7, value=residuals["cm_momentum_MeV_c"]),
        dict(name="three-alpha spectrum count = 3N", passed=int(energy_hist.sum()) == 3*n, value=int(energy_hist.sum())),
        dict(name="Dalitz permutation display weight = N", passed=bool(abs(dalitz_hist.sum()-n) < 1e-5), value=float(dalitz_hist.sum())),
        dict(name="stored weight/envelope <= 1", passed=state["envelope_violations"] == 0, value=state["max_weight_envelope_ratio"]),
        dict(name="sampling attempts below 10000", passed=state["attempt_limit_hits"] == 0, value=state["max_attempts"]),
        dict(name="event-correlated CM orientation moment diagnostic |z| < 6", passed=bool((abs(z) < 6).all()), value=z.tolist()),
    ]
    return dict(
        computed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        scope="Pure 675 component at generation; kinematics and saved sampler diagnostics. No independent complex-amplitude or experimental validation.",
        incident_count=incident_count, reaction_count=n, alpha_count=3*n,
        masses_MeV=dict(alpha=ALPHA_MASS_MEV, proton=PROTON_MASS_MEV, boron11=BORON11_MASS_MEV),
        reaction_q_MeV=REACTION_Q_MEV, scalar=state, residuals=residuals,
        orientation=dict(orders=[1,2,3,4], means=means.tolist(), standard_errors=errors.tolist(), z=z.tolist(),
                         unit="event mean of three alpha Legendre moments"),
        checks=checks, all_passed=all(row["passed"] for row in checks),
        provenance=dict(root=str(path), root_bytes=path.stat().st_size, root_sha256=sha256(path),
                        runinfo=runinfo, requested_threads=expected_threads,
                        thread_verification="Master and worker completion counts must be checked in the Geant4 log; RunInfo is filled only by worker 0.",
                        analysis_script_sha256=sha256(__file__)),
        histograms=dict(energy_edges_MeV=energy_edges.tolist(), energy_counts=energy_hist.tolist(),
                        dalitz_edges=dalitz_edges.tolist(), dalitz_event_weights=dalitz_hist.tolist(),
                        weight_ratio_edges=ratio_edges.tolist(), weight_ratio_counts=ratio_hist.tolist(),
                        attempts_counts=attempts_hist.tolist()),
    )


def plot_validation(result):
    """Return an editable Figure from small cached histograms, no ROOT reread."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    h = result["histograms"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    axes[0, 0].stairs(h["energy_counts"], h["energy_edges_MeV"], color="black", lw=1.1)
    axes[0, 0].set(xlabel="Alpha CM kinetic energy [MeV]", ylabel="Counts (50 keV bins)",
                   title="All three alpha particles, each counted once", xlim=(0,7))
    dalitz = np.asarray(h["dalitz_event_weights"], float)
    edges = np.asarray(h["dalitz_edges"])
    density = dalitz/(result["reaction_count"]*np.diff(edges)[0]**2)
    positive = density[density > 0]
    image = axes[0, 1].pcolormesh(edges, edges, np.ma.masked_less_equal(density.T, 0),
                                 shading="auto", cmap="viridis",
                                 norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    fig.colorbar(image, ax=axes[0, 1], label="Normalized density (log scale)")
    axes[0, 1].set(xlabel=r"$\chi_D=(\epsilon_2-\epsilon_3)/\sqrt{3}$",
                   ylabel=r"$\psi_D=\epsilon_1-1/3$", title="Six relabelings with 1/6 weight each",
                   xlim=(-1/3,1/3), ylim=(-1/3,1/3), aspect="equal")
    axes[1, 0].stairs(h["weight_ratio_counts"], h["weight_ratio_edges"], color="#0072B2")
    axes[1, 0].set(xlabel="Stored weight / stored envelope", ylabel="Accepted events",
                   title="Recorded sampling weights", xlim=(0,1))
    axes[1, 0].text(.97,.95,f'Max ratio: {result["scalar"]["max_weight_envelope_ratio"]:.6f}',
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=9)
    counts = np.asarray(h["attempts_counts"])
    last = result["scalar"]["max_attempts"]
    indices = np.arange(1,last+1)
    axes[1, 1].step(indices, counts[1:last+1], where="mid", color="#D55E00")
    axes[1, 1].set(xlabel="Attempts per accepted event", ylabel="Accepted events (log scale)",
                   title="Sampling-attempt distribution", yscale="log", xlim=(.5,last+1))
    axes[1, 1].text(.97,.95,f'Mean: {result["scalar"]["mean_attempts"]:.3f}\nMax: {last}; limit: 10000',
                    transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle(f'675 keV coherent L1/L3: {result["reaction_count"]:,} generated reactions', fontsize=12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--incident-count", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    result = validate_675_sample(args.root, incident_count=args.incident_count, expected_threads=args.threads)
    compact = {key:value for key,value in result.items() if key not in ("histograms", "provenance")}
    compact["root_sha256"] = result["provenance"]["root_sha256"]
    print(json.dumps(compact,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
