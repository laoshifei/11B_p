"""Matched-response threshold study at fixed generated or accepted counts.

Independent mixed events define the empirical pseudoexperiment population;
pure-channel events define fixed fit templates. Rejected reactions enter the
sampling budget, but never the conditional Dalitz likelihood. All fractions
refer to the prescribed generator mixture, not the accepted mixture.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import bootstrap_162_response_uncertainty as bootstrap

scan = bootstrap.scan
VERSION = "2026-10-16-v2-fixed-generated-threshold-checkpoints"
THRESHOLDS = np.array([0., 50., 100., 150., 200., 300., 500.])
TRIAL_THRESHOLDS = np.array([0., 100., 500.])
TRUE_FRACTIONS = np.array([.2, .6, .2])
BUDGETS = ("fixed_generated", "fixed_accepted")


def nested_categories(histograms, generated_counts):
    """Per-channel counts in exclusive threshold tiers/bins and rejection."""
    hist = np.asarray(histograms)
    generated = np.asarray(generated_counts)
    if (hist.ndim != 4 or hist.shape[0] < 1 or hist.shape[1] != 3
            or generated.shape != (3,) or np.any(generated <= 0)
            or not np.all(np.isfinite(hist)) or np.any(hist < 0)
            or np.any(hist != np.rint(hist))
            or np.any(generated != np.rint(generated))):
        raise ValueError("Expected integer [threshold, channel, x, y] counts")
    layers = np.concatenate((hist[:-1] - hist[1:], hist[-1:]), axis=0)
    rejected = generated - hist[0].sum(axis=(1, 2))
    if np.any(layers < 0) or np.any(rejected < 0):
        raise ValueError("Threshold counts must be nested within generated totals")
    return np.column_stack((layers.transpose(1, 0, 2, 3).reshape(3, -1), rejected)).astype(np.int64)


def generated_probability(histograms, generated_counts, fractions=TRUE_FRACTIONS):
    """Unconditional law per generated reaction, including rejected outcomes."""
    categories = nested_categories(histograms, generated_counts)
    fractions = np.asarray(fractions, dtype=float)
    if (fractions.shape != (3,) or np.any(fractions < 0)
            or not np.all(np.isfinite(fractions)) or not np.isclose(fractions.sum(), 1)):
        raise ValueError("Invalid generator fractions")
    probability = fractions @ (categories / np.asarray(generated_counts)[:, None])
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-12)
    return probability / probability.sum()


def unpack_generated(draw, thresholds, histogram_shape):
    """One coupled reaction sample becomes nested accepted histograms."""
    layers = np.asarray(draw)[:-1].reshape(thresholds, *histogram_shape)
    return np.cumsum(layers[::-1], axis=0)[::-1]


def complete_accepted_sample(histogram, n_generated, target, probability, efficiency, rng):
    """Continue after a generated sample until exactly target events pass.

    Accepted-bin counts and waiting failures are independent conditional on
    the number of missing successes. A negative-binomial draw reproduces the
    stopping-time reaction count without introducing count data into the fit.
    """
    missing = int(target - np.sum(histogram))
    if missing < 0 or not 0 < efficiency <= 1:
        raise ValueError("Invalid top-up target or efficiency")
    if missing == 0:
        return np.asarray(histogram).copy(), int(n_generated)
    added = rng.multinomial(missing, np.asarray(probability).ravel()).reshape(histogram.shape)
    failures = rng.negative_binomial(missing, efficiency)
    return histogram + added, int(n_generated + missing + failures)


def build_pools(input_paths, bins=60, energy_seed=1620):
    """Fold each event once; reuse identical measured energies across cuts."""
    inputs = scan.load_response_inputs(*input_paths)
    histograms = np.zeros((len(THRESHOLDS), 3, bins, bins), dtype=np.int64)
    mixed_histograms = np.zeros_like(histograms)
    centre = np.array([0., 0., 170.])
    limits = (-1.05, 1.05)

    def measured(events, rng):
        reconstructed, _ = scan.reconstruct_with_granularity(
            events, 2., 2., 0., 0., "sphere-center", centre)
        normal = rng.normal(size=events["kinetic"].shape)
        return scan.apply_closure_energy_response(reconstructed, 100., 0., normal)[0]

    def histogram(x, y, keep):
        result = np.histogram2d(x[keep], y[keep], bins=bins, range=[limits, limits])[0]
        if result.sum() != np.count_nonzero(keep):
            raise RuntimeError("Accepted events fell outside Dalitz bins")
        return result.astype(np.int64)

    rng = np.random.default_rng(energy_seed + 1)
    for channel, events in enumerate(inputs["pure_events"]):
        response = measured(events, rng)
        x, y = scan.ordered_dalitz_xy(response)
        minimum = response["kinetic"].min(axis=1)
        for t, threshold in enumerate(THRESHOLDS):
            histograms[t, channel] = histogram(x, y, minimum >= threshold / 1000.)
        print(f"Prepared seven matched thresholds: {scan.CHANNEL_KEYS[channel]}", flush=True)

    response = measured(inputs["mixed_events"], np.random.default_rng(energy_seed))
    ids, channels = scan.prepare_reaction_channel_lookup(
        inputs["reaction_event_ids"], inputs["reaction_channels"])
    positions = np.searchsorted(ids, response["event_id"])
    if np.any(positions >= ids.size) or not np.array_equal(ids[positions], response["event_id"]):
        raise RuntimeError("Accepted events absent from reaction truth")
    labels = channels[positions]
    generated = np.array([np.count_nonzero(channels == c) for c in scan.CHANNEL_IDS])
    x, y = scan.ordered_dalitz_xy(response)
    minimum = response["kinetic"].min(axis=1)
    for t, threshold in enumerate(THRESHOLDS):
        keep = minimum >= threshold / 1000.
        for c, channel in enumerate(scan.CHANNEL_IDS):
            mixed_histograms[t, c] = histogram(x, y, keep & (labels == channel))
    nested_categories(histograms, inputs["pure_generated_counts"])
    nested_categories(mixed_histograms, generated)
    print("Prepared independent mixed population, including rejected reactions.", flush=True)
    return {
        "template_histograms": histograms, "mixed_histograms": mixed_histograms,
        "pure_generated": inputs["pure_generated_counts"], "mixed_generated": generated,
        "template_efficiencies": histograms.sum(axis=(2, 3)) / inputs["pure_generated_counts"],
        "mixed_efficiencies": mixed_histograms.sum(axis=(2, 3)) / generated,
        "quality_counts": [len(e["event_id"]) for e in inputs["pure_events"] + [inputs["mixed_events"]]],
        "runinfo": inputs["runinfo"], "input_paths": [str(p) for p in inputs["paths"]],
        "settings": {"thresholds_keV": THRESHOLDS.copy(), "bins": bins,
                     "response": [2., 2., 100.], "energy_seed": energy_seed,
                     "true_fractions": TRUE_FRACTIONS.copy()},
    }


def validate_numerics(pools, budget=100000, seed=162701, tolerance=1e-4):
    """Check real integer data against the old fit and a denser posterior grid."""
    probability = generated_probability(pools["mixed_histograms"], pools["mixed_generated"])
    rng = np.random.default_rng(seed)
    shape = pools["mixed_histograms"].shape[-2:]
    samples = unpack_generated(rng.multinomial(budget, probability), len(THRESHOLDS), shape)
    cases, reference_differences, convergence = [], [], []
    for index, threshold in enumerate(THRESHOLDS):
        for name in BUDGETS:
            data = samples[index]
            if name == "fixed_accepted":
                q = bootstrap.empirical_probability(
                    pools["mixed_histograms"][index], pools["mixed_generated"], TRUE_FRACTIONS)
                efficiency = pools["mixed_efficiencies"][index] @ TRUE_FRACTIONS
                data, _ = complete_accepted_sample(data, budget, budget, q, efficiency, rng)
            hist = pools["template_histograms"][index]
            eff = pools["template_efficiencies"][index]
            old = scan.fit_histogram(data, hist, eff, scan.simplex_grid(101), 101, 201, 6., .5)
            new = bootstrap.fit_interior(data, hist, eff, 201)
            fine = bootstrap.fit_interior(data, hist, eff, 401)
            for key in ("median", "lower95", "upper95"):
                reference_differences.extend(np.abs(old[key] - new[key]))
                convergence.extend(np.abs(new[key] - fine[key]))
            cases.append((data, hist, eff, fine))
        print(f"Numerical cross-check: both budgets, threshold={threshold:g} keV", flush=True)
    selected_grid = 201
    if max(reference_differences) > tolerance:
        raise RuntimeError("Continuous fit does not reproduce the reference fit")
    if max(convergence) > tolerance:
        selected_grid = 401
        convergence = []
        for data, hist, eff, fine in cases:
            finer = bootstrap.fit_interior(data, hist, eff, 801)
            for key in ("median", "lower95", "upper95"):
                convergence.extend(np.abs(fine[key] - finer[key]))
    if max(convergence) > tolerance:
        raise RuntimeError("Posterior integration has not converged to requested tolerance")
    return {"selected_grid": selected_grid, "max_reference_difference": float(max(reference_differences)),
            "max_grid_difference": float(max(convergence)), "tolerance": tolerance,
            "cases": len(cases), "seed": seed}


def expected_scan(pools, budget=100000, local_grid_size=201):
    """Asimov histograms: separate event loss from information per accepted event."""
    efficiency = pools["mixed_efficiencies"] @ TRUE_FRACTIONS
    results = {name: {key: [] for key in ("median", "lower95", "upper95")}
               for name in BUDGETS}
    for t, hist in enumerate(pools["template_histograms"]):
        q = bootstrap.empirical_probability(pools["mixed_histograms"][t],
                                            pools["mixed_generated"], TRUE_FRACTIONS)
        for name, total in (("fixed_generated", budget * efficiency[t]), ("fixed_accepted", budget)):
            fit = bootstrap.fit_interior(q * total, hist, pools["template_efficiencies"][t], local_grid_size)
            for key in results[name]:
                results[name][key].append(fit[key])
    for name, result in results.items():
        result.update({key: np.array(value) for key, value in result.items()})
        result["width"] = result["upper95"] - result["lower95"]
        result["residual"] = result["median"] - TRUE_FRACTIONS
    return {"fits": results, "efficiency": efficiency, "expected_accepted": budget * efficiency,
            "expected_generated_for_accepted": budget / efficiency, "budget": budget,
            "thresholds_keV": THRESHOLDS.copy(), "local_grid_size": local_grid_size}


def run_trials(pools, repetitions=500, budget=100000, seed=16220261017,
               local_grid_size=201, progress_every=25, checkpoint=None, resume=None):
    """Paired data-only trials; templates fixed and matched at every threshold."""
    if repetitions < 2 or budget < 1 or int(budget) != budget:
        raise ValueError("Need at least two trials and a positive integer reaction budget")
    indices = np.array([np.flatnonzero(THRESHOLDS == t)[0] for t in TRIAL_THRESHOLDS])
    population = pools["mixed_histograms"][indices]
    probability = generated_probability(population, pools["mixed_generated"])
    efficiencies = pools["mixed_efficiencies"][indices] @ TRUE_FRACTIONS
    accepted_probability = [bootstrap.empirical_probability(h, pools["mixed_generated"], TRUE_FRACTIONS)
                            for h in population]
    shape = population.shape[-2:]
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    records = {name: {key: np.empty((repetitions, len(indices), 3))
                      for key in ("median", "lower95", "upper95")}
               for name in BUDGETS}
    for result in records.values():
        result["accepted"] = np.empty((repetitions, len(indices)), dtype=np.int64)
        result["generated"] = np.empty((repetitions, len(indices)), dtype=np.int64)
    digest = hashlib.sha256()
    for key in ("template_histograms", "mixed_histograms", "pure_generated", "mixed_generated"):
        digest.update(np.asarray(pools[key], dtype=np.int64).tobytes())
    configuration = dict(budget=budget, seed=seed, local_grid_size=local_grid_size,
                         population_sha256=digest.hexdigest(), histogram_shape=list(shape),
                         fractions=TRUE_FRACTIONS.tolist(), thresholds=TRIAL_THRESHOLDS.tolist())
    completed_trials, previous_seconds = 0, 0.
    if resume is not None:
        if resume["configuration"] != configuration:
            raise ValueError("Checkpoint uses different populations or fit settings")
        completed_trials = int(resume["completed"])
        previous_seconds = float(resume["seconds"])
        if not 0 <= completed_trials <= repetitions:
            raise ValueError("Invalid checkpoint length")
        for name, record in records.items():
            for key, values in record.items():
                saved = np.asarray(resume["records"][name][key])
                if saved.shape != values[:completed_trials].shape or not np.all(np.isfinite(saved)):
                    raise ValueError("Incomplete checkpoint records")
                values[:completed_trials] = saved
    started = time.monotonic()
    for repetition in range(repetitions):
        generated_draw = streams[0].multinomial(budget, probability)
        histograms = unpack_generated(generated_draw, len(indices), shape)
        for j, index in enumerate(indices):
            data = histograms[j]
            completed, total = complete_accepted_sample(
                data, budget, budget, accepted_probability[j], efficiencies[j], streams[j+1])
            for name, histogram, n_generated in (
                ("fixed_generated", data, budget), ("fixed_accepted", completed, total),
            ):
                if repetition < completed_trials:
                    # Replay only the cheap random draws, preserving the exact
                    # original streams without repeating completed fits.
                    if (records[name]["accepted"][repetition, j] != histogram.sum()
                            or records[name]["generated"][repetition, j] != n_generated):
                        raise ValueError("Checkpoint does not reproduce its event budgets")
                    continue
                fit = bootstrap.fit_interior(histogram, pools["template_histograms"][index],
                                             pools["template_efficiencies"][index], local_grid_size)
                for key in ("median", "lower95", "upper95"):
                    records[name][key][repetition, j] = fit[key]
                records[name]["accepted"][repetition, j] = histogram.sum()
                records[name]["generated"][repetition, j] = n_generated
        if repetition+1 > completed_trials and ((repetition+1) % progress_every == 0 or repetition+1 == repetitions):
            if checkpoint is not None:
                checkpoint(dict(configuration=configuration, completed=repetition+1,
                                seconds=previous_seconds + time.monotonic()-started,
                                records={name: {key: value[:repetition+1].copy()
                                                for key, value in record.items()}
                                         for name, record in records.items()}))
            print(f"Matched trials {repetition+1}/{repetitions}: "
                  f"{time.monotonic()-started:.1f} s, {6*(repetition+1)} fits", flush=True)
    summaries = {}
    for name, record in records.items():
        median, lo, hi = (record[k] for k in ("median", "lower95", "upper95"))
        hit = (lo <= TRUE_FRACTIONS) & (TRUE_FRACTIONS <= hi)
        width = hi - lo
        relative_width = width / width[:, :1]
        summaries[name] = {
            "mean": median.mean(axis=0), "bias": median.mean(axis=0)-TRUE_FRACTIONS,
            "sd": median.std(axis=0, ddof=1), "bias_mcse": median.std(axis=0, ddof=1)/np.sqrt(repetitions),
            "mean_width": width.mean(axis=0), "width_mcse": width.std(axis=0, ddof=1)/np.sqrt(repetitions),
            "mean_width_ratio_to_zero": relative_width.mean(axis=0),
            "width_ratio_mcse": relative_width.std(axis=0, ddof=1)/np.sqrt(repetitions),
            "coverage": hit.mean(axis=0), "coverage_hits": hit.sum(axis=0),
            "coverage_wilson95": bootstrap.wilson_interval(hit.sum(axis=0), repetitions),
            "accepted_mean": record["accepted"].mean(axis=0),
            "accepted_sd": record["accepted"].std(axis=0, ddof=1),
            "generated_mean": record["generated"].mean(axis=0),
            "generated_sd": record["generated"].std(axis=0, ddof=1),
        }
    paired_width_ratio = ((records["fixed_generated"]["upper95"]-records["fixed_generated"]["lower95"])
                          / (records["fixed_accepted"]["upper95"]-records["fixed_accepted"]["lower95"]))
    return {"records": records, "summaries": summaries, "thresholds_keV": TRIAL_THRESHOLDS.copy(),
            "budget": budget, "repetitions": repetitions, "seed": seed, "fits": 6*repetitions,
            "seconds": previous_seconds + time.monotonic()-started, "local_grid_size": local_grid_size,
            "paired_width_ratio_mean": paired_width_ratio.mean(axis=0),
            "paired_width_ratio_mcse": paired_width_ratio.std(axis=0, ddof=1)/np.sqrt(repetitions)}


def plot_summary(pools, expected, trials):
    """Return three editable figures without writing result files."""
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$\alpha_0$", r"$\alpha_1$", "Direct template")
    figures = []
    x = THRESHOLDS
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    for c, (color, label) in enumerate(zip(colors, labels)):
        axes[0].plot(x, pools["mixed_efficiencies"][:, c]*100, "o-", color=color, label=label)
        axes[0].plot(x, pools["template_efficiencies"][:, c]*100, ":", color=color, alpha=.6)
    axes[0].set_ylabel("Accepted reactions [%]")
    axes[0].set_title("Solid: mixed population; dotted: templates")
    axes[0].legend(fontsize=9)
    axes[1].plot(x, expected["expected_accepted"], "o-", label="Accepted at 100,000 generated")
    axes[1].plot(x, expected["expected_generated_for_accepted"], "s-", label="Generated for 100,000 accepted")
    axes[1].axhline(expected["budget"], color="gray", ls=":")
    axes[1].set_ylabel("Expected reaction / event count")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.set_xlabel("Matched per-alpha threshold [keV]"); ax.grid(alpha=.2)
    fig.suptitle("Same response in data and templates: event-budget cost")
    figures.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for c, ax in enumerate(axes):
        for name, marker, label in (("fixed_generated", "o", "100,000 generated"),
                                    ("fixed_accepted", "s", "100,000 accepted")):
            widths = expected["fits"][name]["width"][:, c]
            ax.plot(x, 100*(widths/widths[0]-1), marker+"-", label=label)
        ax.axhline(0, color="gray", ls=":")
        ax.set_xlabel("Matched per-alpha threshold [keV]")
        ax.set_title(labels[c]); ax.grid(alpha=.2)
    axes[0].set_ylabel("95% interval width change from 0 keV [%]")
    axes[0].legend(fontsize=9)
    fig.suptitle("Expected-count (Asimov) scan: loss versus information per accepted event")
    figures.append(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    x = TRIAL_THRESHOLDS
    for c, ax in enumerate(axes):
        for name, offset, color, label in (("fixed_generated", -7, "#0072B2", "100,000 generated"),
                                          ("fixed_accepted", 7, "#D55E00", "100,000 accepted")):
            summary = trials["summaries"][name]
            ax.errorbar(x+offset, summary["bias"][:, c]*100,
                        yerr=1.96*summary["bias_mcse"][:, c]*100,
                        fmt="o-", color=color, capsize=3, label=label)
        ax.axhline(0, color="gray", ls=":")
        ax.set_xlabel("Matched per-alpha threshold [keV]")
        ax.set_title(labels[c]); ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean estimator bias [pp]")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"{trials['repetitions']} data-only trials: mean bias with 1.96 MCSE bars")
    figures.append(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs=4)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    pools = build_pools(args.inputs)
    validation = validate_numerics(pools)
    expected_scan(pools, local_grid_size=validation["selected_grid"])
    results = run_trials(pools, repetitions=args.repetitions, local_grid_size=validation["selected_grid"])
    print("Finished", results["fits"], "fits; results are in memory only.")


if __name__ == "__main__":
    main()
