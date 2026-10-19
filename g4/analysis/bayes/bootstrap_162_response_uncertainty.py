"""Paired data/template bootstrap for the interior 162-keV mixture benchmark.

The independent mixed ROOT file defines an empirical, response-folded sampling
population. Channel labels construct that population at a prescribed generator
composition; they are never passed to the fit. Multinomial sampling is exactly
event bootstrap after reduction to the sufficient Dalitz histogram.

Template bootstrap includes rejected reactions and preserves the coupling of
the three nested thresholds. All intervals remain conditional on the template
used in that fit. Bootstrap spread is reported separately, not relabelled as a
posterior with template uncertainty integrated out.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from . import scan_detector_theta_phi_response_162_channel_fit as scan

VERSION = "2026-10-16-v1-paired-empirical-bootstrap"
THRESHOLDS = np.array([50.0, 100.0, 150.0])
TRUE_FRACTIONS = np.array([0.2, 0.6, 0.2])
ENSEMBLES = ("data_only", "template_only", "joint")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empirical_probability(channel_histograms, generated_counts, fractions):
    """Accepted-event law for known *generator* fractions, including losses."""
    hist = np.asarray(channel_histograms, dtype=float)
    generated = np.asarray(generated_counts, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if (hist.shape[0] != 3 or generated.shape != (3,) or fractions.shape != (3,)
            or np.any(generated <= 0) or np.any(hist < 0)
            or not np.isclose(fractions.sum(), 1) or np.any(fractions < 0)):
        raise ValueError("Invalid population counts or generator fractions")
    counts = hist.reshape(3, -1)
    if np.any(counts.sum(axis=1) > generated):
        raise ValueError("Accepted events exceed generated events")
    probability = fractions @ (counts / generated[:, None])
    if probability.sum() <= 0:
        raise ValueError("No accepted sampling population")
    return (probability / probability.sum()).reshape(hist.shape[1:])


def template_categories(histograms, generated_counts):
    """Joint outcome of each generated event: threshold tier/bin or rejection."""
    hist = np.asarray(histograms)
    generated = np.asarray(generated_counts)
    if hist.ndim != 4 or hist.shape[:2] != (3, 3):
        raise ValueError("Expected [threshold, channel, x, y] histograms")
    if np.any(hist < 0) or np.any(hist != np.rint(hist)):
        raise ValueError("Bootstrap needs integer event counts")
    layers = np.stack((hist[0] - hist[1], hist[1] - hist[2], hist[2]), axis=1)
    if np.any(layers < 0):
        raise ValueError("Thresholds must use nested events and identical Dalitz bins")
    rejected = generated - hist[0].sum(axis=(1, 2))
    if np.any(rejected < 0):
        raise ValueError("Accepted events exceed generated events")
    return np.column_stack((layers.reshape(3, -1), rejected)).astype(np.int64)


def resample_templates(categories, histogram_shape, rng):
    """Fixed generated count per channel; shared bootstrap for all thresholds."""
    totals = categories.sum(axis=1)
    draws = np.array([rng.multinomial(int(n), row / n)
                      for n, row in zip(totals, categories)])
    layers = draws[:, :-1].reshape(3, 3, *histogram_shape)
    hist = np.stack((layers.sum(axis=1), layers[:, 1:].sum(axis=1), layers[:, 2]))
    return hist, totals


def build_pools(input_paths, n_fit=100000, bins=60, seed=162, energy_seed=1620):
    """Read ROOT once and reduce independent event pools to sufficient counts."""
    inputs = scan.load_response_inputs(*input_paths)
    centre = np.array([0.0, 0.0, 170.0])
    limits = (-1.05, 1.05)

    def responded(events, normal, threshold):
        rec, _ = scan.reconstruct_with_granularity(
            events, 2.0, 2.0, 0.0, 0.0, "sphere-center", centre)
        return scan.apply_closure_energy_response(rec, 100.0, threshold, normal)[0]

    template_histograms = np.zeros((3, 3, bins, bins), dtype=np.int64)
    rng = np.random.default_rng(energy_seed + 1)
    for channel, events in enumerate(inputs["pure_events"]):
        measured = responded(events, rng.normal(size=events["kinetic"].shape), THRESHOLDS[0])
        for index, threshold in enumerate(THRESHOLDS):
            keep = np.all(measured["kinetic"] >= threshold / 1000.0, axis=1)
            template_histograms[index, channel] = scan.dalitz_histogram(
                scan.apply_mask(measured, keep), bins, limits)
        print(f"Prepared nested templates: {scan.CHANNEL_KEYS[channel]}", flush=True)

    events = inputs["mixed_events"]
    measured = responded(events, np.random.default_rng(energy_seed).normal(
        size=events["kinetic"].shape), 100.0)
    ids, channels = scan.prepare_reaction_channel_lookup(
        inputs["reaction_event_ids"], inputs["reaction_channels"])
    positions = np.searchsorted(ids, measured["event_id"])
    if np.any(positions >= ids.size) or not np.array_equal(ids[positions], measured["event_id"]):
        raise RuntimeError("Accepted events absent from reaction truth")
    accepted_channels = channels[positions]
    mixed_generated = np.array([np.count_nonzero(channels == c) for c in scan.CHANNEL_IDS])
    mixed_histograms = np.array([
        scan.dalitz_histogram(scan.apply_mask(measured, accepted_channels == c), bins, limits)
        for c in scan.CHANNEL_IDS], dtype=np.int64)
    probability = empirical_probability(mixed_histograms, mixed_generated, TRUE_FRACTIONS)
    shuffled = np.random.default_rng(seed).permutation(inputs["reaction_event_ids"])
    fixed_events, prefix_count = scan.select_exact_nfit(measured, shuffled, n_fit)
    fixed_histogram = scan.dalitz_histogram(fixed_events, bins, limits).astype(np.int64)
    _, prefix_truth = scan.generated_prefix_truth(shuffled, prefix_count, ids, channels)
    categories = template_categories(template_histograms, inputs["pure_generated_counts"])
    return {
        "template_histograms": template_histograms, "categories": categories,
        "pure_generated": inputs["pure_generated_counts"],
        "mixed_histograms": mixed_histograms, "mixed_generated": mixed_generated,
        "probability": probability, "fixed_histogram": fixed_histogram,
        "prefix_count": prefix_count, "prefix_truth": prefix_truth,
        "fixed_histogram_sha256": hashlib.sha256(fixed_histogram.astype(float).tobytes()).hexdigest(),
        "quality_counts": [len(e["event_id"]) for e in inputs["pure_events"]] + [len(events["event_id"])],
        "runinfo": inputs["runinfo"], "input_paths": [str(p) for p in inputs["paths"]],
        "settings": {"n_fit": n_fit, "bins": bins, "seed": seed, "energy_seed": energy_seed,
                     "data_response": [2.0, 2.0, 100.0, 100.0],
                     "template_thresholds_keV": THRESHOLDS.tolist(),
                     "true_fractions": TRUE_FRACTIONS.tolist()},
    }


def fit_interior(data, templates, efficiencies, local_grid_size=101):
    """Same flat-generator-simplex posterior, with a convex continuous MAP.

    Maximization in accepted weights replaces the expensive global/refinement
    grids. Posterior integration remains in generator fractions. This helper
    deliberately rejects boundary cases, which require a separate study.
    """
    active = np.any(templates > 0, axis=0) | (data > 0)
    counts = np.asarray(data[active], dtype=float)
    probabilities = np.asarray(templates[:, active], dtype=float) + 0.5
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    efficiencies = np.asarray(efficiencies, dtype=float)
    if counts.sum() <= 0 or np.any(efficiencies <= 0):
        raise ValueError("Nonpositive data count or efficiency")
    difference = probabilities[:2] - probabilities[2]
    relative = counts / counts.sum()

    def objective(w):
        q = probabilities[2] + w @ difference
        if np.any(q <= 0):
            return np.inf, np.zeros(2)
        return -relative @ np.log(q), -difference @ (relative / q)

    start = np.array([0.25, 0.5, 0.25]) * efficiencies
    start /= start.sum()
    optimum = minimize(objective, start[:2], jac=True, method="SLSQP",
                       bounds=[(1e-9, 1 - 1e-9)] * 2,
                       constraints={"type": "ineq", "fun": lambda w: 1 - 1e-9 - w.sum(),
                                    "jac": lambda w: -np.ones(2)},
                       options={"ftol": 1e-13, "maxiter": 100})
    if not optimum.success:
        raise RuntimeError(f"Mixture optimizer failed: {optimum.message}")
    w = optimum.x
    # Polish the small interior stationary point with its exact Hessian.
    for _ in range(8):
        q = probabilities[2] + w @ difference
        gradient = -difference @ (relative / q)
        hessian = (difference * (relative / q**2)) @ difference.T
        step = np.linalg.solve(hessian, gradient)
        candidate = w - step
        if min(*candidate, 1 - candidate.sum()) <= 1e-8:
            raise ValueError("Boundary mixture: use a boundary-aware posterior study")
        w = candidate
        if np.max(np.abs(step)) < 1e-11:
            break
    if np.max(np.abs(objective(w)[1])) > 1e-7:
        raise RuntimeError("Mixture mode failed the stationarity check")
    accepted = np.array([w[0], w[1], 1 - w.sum()])
    mode = accepted / efficiencies
    mode /= mode.sum()
    weighted = probabilities * efficiencies[:, None]
    g = mode @ weighted
    total = mode @ efficiencies
    derivative = weighted[:2] - weighted[2]
    de = efficiencies[:2] - efficiencies[2]
    curvature = ((derivative * (counts / g**2)) @ derivative.T
                 - counts.sum() * np.outer(de, de) / total**2)
    if np.min(np.linalg.eigvalsh(curvature)) <= 0:
        raise RuntimeError("Nonpositive posterior curvature")
    covariance = np.linalg.inv(curvature)
    sigma = np.sqrt(np.r_[np.diag(covariance), covariance.sum()])
    if np.any(mode < 7 * sigma):
        raise ValueError("Posterior reaches a boundary; interior benchmark is not applicable")
    fractions = scan.adaptive_local_grid(mode, covariance, .01, 6.0, local_grid_size)
    posterior, _ = scan.evaluate_posterior(fractions, probabilities, efficiencies, counts)
    quantiles = np.array([scan.weighted_quantiles(fractions[:, c], posterior, [.025, .5, .975])
                          for c in range(3)])
    return {"median": quantiles[:, 1], "lower95": quantiles[:, 0],
            "upper95": quantiles[:, 2], "map": mode}


def validate_fitter(pools, seed=162500, tolerance=1e-4):
    """Compare all three templates on fixed and fluctuated data to the old fit."""
    rng = np.random.default_rng(seed)
    fluctuated = rng.multinomial(pools["settings"]["n_fit"], pools["probability"].ravel())
    fluctuated = fluctuated.reshape(pools["probability"].shape)
    sampled, totals = resample_templates(pools["categories"], fluctuated.shape, rng)
    differences = {101: [], 201: []}
    convergence = []
    cases = []
    reference = []
    for label, data, histograms in (
        ("fixed", pools["fixed_histogram"], pools["template_histograms"]),
        ("joint", fluctuated, sampled),
    ):
        for index, hist in enumerate(histograms):
            eff = scan.response_efficiencies(hist, pools["pure_generated"])
            old = scan.fit_histogram(data, hist, eff, scan.simplex_grid(101), 101, 201, 6, .5)
            new = fit_interior(data, hist, eff, 101)
            fine = fit_interior(data, hist, eff, 201)
            for key in ("median", "lower95", "upper95"):
                differences[101].extend(np.abs(new[key] - old[key]))
                differences[201].extend(np.abs(fine[key] - old[key]))
                convergence.extend(np.abs(new[key] - fine[key]))
            cases.append((data, hist, eff, fine))
            if label == "fixed":
                reference.append(old)
            print(f"Numerical comparison: {label}, threshold={THRESHOLDS[index]:g} keV", flush=True)
    selected_grid = 101
    if max(differences[101]) > tolerance or max(convergence) > tolerance:
        selected_grid = 201
        convergence = []
        for data, hist, eff, fine in cases:
            finer = fit_interior(data, hist, eff, 401)
            for key in ("median", "lower95", "upper95"):
                convergence.extend(np.abs(fine[key] - finer[key]))
    worst = float(max(differences[selected_grid]))
    if worst > tolerance or max(convergence) > tolerance:
        raise RuntimeError(f"Posterior numerical difference {worst:g} exceeds {tolerance:g}")
    return {"max_reference_difference": worst, "max_grid_difference": float(max(convergence)),
            "tolerance": tolerance, "cases": 6, "reference": reference,
            "selected_grid": selected_grid,
            "coarse_reference_difference": float(max(differences[101]))}


def wilson_interval(successes, count, z=1.959963984540054):
    p = np.asarray(successes, dtype=float) / count
    centre = (p + z*z / (2*count)) / (1 + z*z/count)
    half = z * np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return np.stack((centre - half, centre + half), axis=-1)


def run_bootstrap(pools, repetitions=500, seed=16220261016, progress_every=25,
                  local_grid_size=101):
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")
    n_fit = int(pools["settings"]["n_fit"])
    shape = pools["probability"].shape
    rng_data, rng_template = [np.random.default_rng(s)
                              for s in np.random.SeedSequence(seed).spawn(2)]
    base_templates = pools["template_histograms"]
    base_efficiencies = np.array([scan.response_efficiencies(h, pools["pure_generated"])
                                 for h in base_templates])
    baseline = [fit_interior(pools["fixed_histogram"], h, e, local_grid_size)
                for h, e in zip(base_templates, base_efficiencies)]
    records = {name: {key: np.empty((repetitions, 3, 3))
                      for key in ("median", "lower95", "upper95")}
               for name in ENSEMBLES}
    started = time.monotonic()
    for repetition in range(repetitions):
        data = rng_data.multinomial(n_fit, pools["probability"].ravel()).reshape(shape)
        if data.sum() != n_fit:
            raise RuntimeError("Data bootstrap lost accepted events")
        histograms, totals = resample_templates(pools["categories"], shape, rng_template)
        if not np.array_equal(totals, pools["pure_generated"]):
            raise RuntimeError("Template bootstrap changed generated totals")
        for index in range(3):
            efficiency = scan.response_efficiencies(histograms[index], totals)
            jobs = (("data_only", data, base_templates[index], base_efficiencies[index]),
                    ("template_only", pools["fixed_histogram"], histograms[index], efficiency),
                    ("joint", data, histograms[index], efficiency))
            for name, counts, hist, eff in jobs:
                result = fit_interior(counts, hist, eff, local_grid_size)
                for key in records[name]:
                    records[name][key][repetition, index] = result[key]
        if (repetition + 1) % progress_every == 0 or repetition + 1 == repetitions:
            print(f"Bootstrap {repetition+1}/{repetitions}: "
                  f"{time.monotonic()-started:.1f} s, {9*(repetition+1)} fits", flush=True)
    summaries = {}
    for name, result in records.items():
        median, lo, hi = (result[k] for k in ("median", "lower95", "upper95"))
        paired = median - median[:, 1:2, :]
        result["paired_shift"] = paired
        summary = {
            "mean": median.mean(axis=0), "sd": median.std(axis=0, ddof=1),
            "mean_width": (hi-lo).mean(axis=0),
            "paired_mean": paired.mean(axis=0), "paired_sd": paired.std(axis=0, ddof=1),
            "paired_mcse": paired.std(axis=0, ddof=1) / np.sqrt(repetitions),
            "paired_q95": np.quantile(paired, [.025, .975], axis=0),
        }
        if name != "template_only":
            hit = (lo <= TRUE_FRACTIONS) & (TRUE_FRACTIONS <= hi)
            summary.update(bias=median.mean(axis=0)-TRUE_FRACTIONS,
                           bias_mcse=median.std(axis=0, ddof=1)/np.sqrt(repetitions),
                           coverage=hit.mean(axis=0), coverage_hits=hit.sum(axis=0),
                           coverage_wilson95=wilson_interval(hit.sum(axis=0), repetitions))
        else:
            summary["mean_offset_from_fixed"] = median.mean(axis=0)-np.array([r["median"] for r in baseline])
        summaries[name] = summary
    return {"records": records, "summaries": summaries, "baseline": baseline,
            "repetitions": repetitions, "seed": seed, "fits": repetitions*9,
            "seconds": time.monotonic()-started, "thresholds_keV": THRESHOLDS.copy(),
            "true_fractions": TRUE_FRACTIONS.copy(), "local_grid_size": local_grid_size,
            "interval_scope": "Conditional on each fitted template; not template-marginalized"}


def plot_bootstrap_summary(results):
    """Three figures: matched statistics, calibration shifts, and coverage."""
    colors = ("#0072B2", "#D55E00", "#009E73")
    channels = (r"$\alpha_0$", r"$\alpha_1$", "Direct template")
    summaries = results["summaries"]
    figures = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    labels = ("Data only", "Templates only", "Data + templates")
    for c, ax in enumerate(axes):
        values = [summaries[name]["sd"][1, c]*100 for name in ENSEMBLES]
        ax.bar(np.arange(3), values, color=colors, alpha=.85)
        ax.set_xticks(np.arange(3), labels, rotation=20, ha="right")
        ax.set_ylabel("Empirical SD [pp]")
        ax.set_title(channels[c]); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Matched 100 keV threshold: separate sources of sampling variation")
    figures.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    x = results["thresholds_keV"] - 100
    for ax, name, title in zip(axes, ENSEMBLES, labels):
        summary = summaries[name]
        for c, (color, channel) in enumerate(zip(colors, channels)):
            ax.plot(x, summary["paired_mean"][:, c]*100, "o-", color=color, label=channel)
            ax.fill_between(x, summary["paired_q95"][0, :, c]*100,
                            summary["paired_q95"][1, :, c]*100, color=color, alpha=.15)
        ax.axhline(0, color="black", lw=.7); ax.axvline(0, color="gray", ls=":")
        ax.set_title(title); ax.set_xlabel("Template threshold minus true [keV]")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Paired shift from matched fit [pp]"); axes[0].legend(fontsize=9)
    fig.suptitle("Calibration effect: mean and central 95% bootstrap spread")
    figures.append(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, name, title in zip(axes, ("data_only", "joint"), ("Data only", "Data + templates")):
        summary = summaries[name]
        for c, (color, channel) in enumerate(zip(colors, channels)):
            y = summary["coverage"][:, c]*100
            intervals = summary["coverage_wilson95"][:, c]*100
            ax.errorbar(x + (c-1)*3, y,
                        yerr=np.maximum(np.stack((y-intervals[:, 0], intervals[:, 1]-y)), 0),
                        fmt="o-", color=color, label=channel, capsize=3)
        ax.axhline(95, color="black", ls="--", lw=1, label="95% reference")
        ax.set_ylim(0, 102); ax.set_xlabel("Template threshold minus true [keV]")
        ax.set_title(title); ax.grid(alpha=.2)
    axes[0].set_ylabel("Intervals containing prescribed truth [%]"); axes[0].legend(fontsize=8)
    fig.suptitle(f"Conditional interval inclusion: 95% Wilson error bars for {results['repetitions']} trials")
    figures.append(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs=4)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    pools = build_pools(args.inputs)
    validation = validate_fitter(pools)
    print("Numerical validation:", validation["max_reference_difference"])
    results = run_bootstrap(pools, args.repetitions, local_grid_size=validation["selected_grid"])
    print("Finished", results["fits"], "fits; results are in memory only.")


if __name__ == "__main__":
    main()
