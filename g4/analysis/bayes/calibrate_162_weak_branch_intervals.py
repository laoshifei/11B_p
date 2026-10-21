"""Finite-grid calibration and held-out validation of weak-component inference.

Use the existing conditional multinomial likelihood and Bayesian integrator.
Sampling includes rejected reactions; fitting conditions on accepted counts.
Calibration takes a conservative maximum over a declared finite truth grid,
not a claim of uniform coverage over the continuous parameter space. The
third component is the existing uniform phase-space template.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import diagnose_162_boundary_intervals as diagnostic

boundary = diagnostic.boundary
VERSION = "2026-10-19-v1-finite-grid-weak-branch-calibration"
DIRECT_GRID = (0., .001, .002, .005, .01, .02)
CALIBRATION_RATIOS = (.045, .05, .055)
VALIDATION_RATIOS = (.045, .0475, .05, .0525, .055)
CALIBRATION_SEED = 16220261023
VALIDATION_SEED = 16220261024
NUMERICAL_SEED = 16220261025


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def cases(ratios):
    return [dict(key=f"d{int(round(d*1e6)):05d}_r{int(round(r*1e6)):05d}",
                 rng_id=di*100000+int(round(r*1e6)), direct=d, ratio=r,
                 fractions=[r*(1-d), (1-r)*(1-d), d],
                 calibration_ratio=r in CALIBRATION_RATIOS)
            for di, d in enumerate(DIRECT_GRID) for r in ratios]


def compact_population(pools, fractions):
    original = diagnostic.exact_population(pools, fractions=fractions)
    active = original["active"]
    return dict(templates=original["templates"][:, active],
                efficiencies=original["efficiencies"],
                probability=np.r_[original["probability"][:-1].reshape(active.shape)[active],
                                  original["probability"][-1]],
                fractions=np.asarray(fractions), pseudocount=original["pseudocount"],
                population_sha256=original["population_sha256"])


def profile_log_relative(model, value, channel):
    """Profile the two remaining generator fractions at fixed f[channel].

    The endpoints have different total efficiencies. Optimize their mixture
    in accepted-event weights, then transform back to generator weights.
    """
    if channel not in (0, 1, 2) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Need a channel index and a fraction in [0, 1]")
    remaining = [c for c in range(3) if c != channel]
    endpoints = np.zeros((2, 3))
    endpoints[:, channel] = value
    endpoints[0, remaining[0]] = 1-value
    endpoints[1, remaining[1]] = 1-value
    efficiency = endpoints @ model.efficiencies
    shapes = model.probability(endpoints)
    weight = boundary._mix_mode(shapes[0], shapes[1], model.relative)
    generated = (weight/efficiency[1]) / ((1-weight)/efficiency[0]+weight/efficiency[1])
    fractions = (1-generated)*endpoints[0]+generated*endpoints[1]
    value = float(model.log_relative(fractions))
    if value > 1e-6:
        raise RuntimeError("Profile likelihood exceeds the verified global maximum")
    return min(0., value)


def profile_statistic(model, value, channel):
    return -2*profile_log_relative(model, value, channel)


def upper_statistic(model, value):
    """Upper-limit ordering: downward fluctuations can reject a large d."""
    return 0. if model.mode[2] > value else profile_statistic(model, value, 2)


def interval(model, cutoff, channel):
    """Invert a constant, frozen profile-LR cutoff on the physical simplex."""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Interval cutoff must be finite and positive")
    centre = float(model.mode[channel])
    def shifted(value):
        return profile_statistic(model, value, channel)-cutoff
    low = (0. if centre == 0 or shifted(0.) <= 0 else
           brentq(shifted, 0., centre, xtol=1e-12))
    high = (1. if centre == 1 or shifted(1.) <= 0 else
            brentq(shifted, centre, 1., xtol=1e-12))
    return np.array([low, high])


def upper_limit(model, cutoff):
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Upper-limit cutoff must be finite and positive")
    centre = float(model.mode[2])
    def shifted(value):
        return profile_statistic(model, value, 2)-cutoff
    return float(1. if centre == 1 or shifted(1.) <= 0 else
                 brentq(shifted, centre, 1., xtol=1e-12))


def tolerance_rank(n, *, coverage=.95, family_confidence=.95, comparisons=39):
    """One-sided nonparametric tolerance bound with a Bonferroni allocation.

    For order statistic X_(k), P[F(X_(k)) >= coverage] is at least
    P[Binomial(n, coverage) <= k-1], also conservatively for ties. The
    family probability concerns Monte Carlo calibration, not data coverage.
    """
    if (int(n) != n or n < 1 or int(comparisons) != comparisons or comparisons < 1
            or not 0 < coverage < 1 or not 0 < family_confidence < 1):
        raise ValueError("Invalid calibration probability or sample size")
    failure = (1-family_confidence)/comparisons
    rank = int(binom.ppf(1-failure, n, coverage))+1
    if rank > n:
        raise ValueError("Too few calibration trials for the requested tolerance guarantee")
    return dict(rank=rank, repetitions=n, coverage=coverage,
                family_confidence=family_confidence, comparisons=comparisons,
                point_failure_budget=failure,
                point_confidence=float(binom.cdf(rank-1, n, coverage)))


def freeze_cutoffs(calibration, *, coverage=.95, family_confidence=.95):
    if calibration["configuration"]["stage"] != "calibration":
        raise ValueError("Cutoffs must use only calibration samples")
    rows = calibration["cases"]
    comparisons = 2*len(rows)+sum(row["case"]["direct"] == 0 for row in rows.values())
    rank = tolerance_rank(calibration["configuration"]["repetitions"], coverage=coverage,
                          family_confidence=family_confidence, comparisons=comparisons)
    local = {}
    for key, row in rows.items():
        statistics = ["q_alpha1", "q_upper"]
        if row["case"]["direct"] == 0:
            statistics.append("q0")
        local[key] = {}
        for name in statistics:
            values = np.asarray(row["records"][name])
            if (values.shape != (rank["repetitions"],) or np.any(values < 0)
                    or not np.all(np.isfinite(values))):
                raise ValueError("Calibration records are incomplete or invalid")
            local[key][name] = float(np.sort(values)[rank["rank"]-1])
    cutoffs = {name: max(row[name] for row in local.values() if name in row)
               for name in ("q_alpha1", "q_upper", "q0")}
    if any(value <= 0 for value in cutoffs.values()):
        raise ValueError("Degenerate calibration grid")
    return dict(cutoffs=cutoffs, local=local, tolerance=rank,
                calibration_sha256=digest(calibration),
                calibration_seed=calibration["configuration"].get("seed"),
                scope="Finite calibration grid only; off-grid behavior requires validation")


def draw(population, seed, case, index, budget):
    rng = np.random.default_rng(np.random.SeedSequence([seed, case["rng_id"], index]))
    return rng.multinomial(budget, population["probability"])[:-1]


def evaluate(data, population, *, cutoffs=None, bayesian=False, grid_size=201):
    model = boundary._Likelihood(data, population["templates"],
                                 population["efficiencies"], population["pseudocount"])
    truth = population["fractions"]
    record = dict(accepted=float(data.sum()), map=model.mode,
                  q_alpha1=profile_statistic(model, truth[1], 1),
                  q_upper=upper_statistic(model, truth[2]),
                  q0=profile_statistic(model, 0., 2))
    if cutoffs is not None:
        record.update(alpha1_interval=interval(model, cutoffs["q_alpha1"], 1),
                      direct_upper=upper_limit(model, cutoffs["q_upper"]))
        a = record["alpha1_interval"]
        if ((a[0] <= truth[1] <= a[1]) != (record["q_alpha1"] <= cutoffs["q_alpha1"])
                or (truth[2] <= record["direct_upper"]) != (record["q_upper"] <= cutoffs["q_upper"])):
            raise RuntimeError("Interval inversion disagrees with its acceptance test")
    if bayesian:
        fitted = boundary.fit_boundary(data, population["templates"], population["efficiencies"],
                                       grid_size=grid_size, pseudocount=population["pseudocount"])
        np.testing.assert_allclose(fitted["map"], model.mode, rtol=0, atol=1e-10)
        np.testing.assert_allclose(fitted["q0"], record["q0"], rtol=0, atol=1e-6)
        record.update(bayes_median=fitted["median"], bayes_lower=fitted["lower95"],
                      bayes_upper=fitted["upper95"], bayes_direct_upper=fitted["direct_upper95"])
    return record


_WORKER = None


def _initialize(pools, configuration, cutoffs):
    global _WORKER
    _WORKER = pools, configuration, cutoffs


def _chunk(task):
    pools, configuration, cutoffs = _WORKER
    case, start, stop = task
    population = compact_population(pools, case["fractions"])
    bayesian = configuration["stage"] == "validation" and case["ratio"] == .05
    records = {}
    for index in range(start, stop):
        data = draw(population, configuration["seed"], case, index, configuration["budget"])
        result = evaluate(data, population, cutoffs=cutoffs, bayesian=bayesian,
                          grid_size=configuration["grid_size"])
        for key, value in result.items():
            records.setdefault(key, []).append(plain(value))
    return dict(case=case, start=start, stop=stop, records=records)


def run_ensemble(pools, case_list, *, stage, repetitions, seed, budget=100000,
                 frozen=None, workers=4, chunk_size=100, grid_size=201,
                 resume=None, checkpoint=None, progress_every=1000):
    """Deterministic per-trial streams, bounded workers, and checked resumption."""
    if (stage not in ("calibration", "validation") or int(repetitions) != repetitions
            or repetitions < 2 or int(budget) != budget or budget < 1
            or int(workers) != workers or not 1 <= workers <= 8
            or int(chunk_size) != chunk_size or chunk_size < 1 or progress_every < 1
            or not case_list or len({c["key"] for c in case_list}) != len(case_list)):
        raise ValueError("Invalid ensemble configuration")
    if (stage == "validation") != (frozen is not None):
        raise ValueError("Validation needs frozen cutoffs; calibration must not use them")
    if frozen is not None and frozen.get("calibration_seed") == seed:
        raise ValueError("Calibration and validation must use independent random streams")
    source_hash = {name: diagnostic.sha256(module.__file__) for name, module in
                   (("boundary", boundary), ("diagnostic", diagnostic))}
    source_hash["study"] = diagnostic.sha256(__file__)
    population_hash = digest({k: pools[k] for k in ("templates", "pure_generated")})
    config = dict(version=VERSION, source_sha256=source_hash, population_sha256=population_hash,
                  cases=case_list, stage=stage, repetitions=repetitions, seed=int(seed),
                  budget=budget, grid_size=grid_size, chunk_size=chunk_size,
                  frozen_sha256=None if frozen is None else digest(frozen))
    tasks = {(c["key"]+f":{start}"): (c, start, min(start+chunk_size, repetitions))
             for c in case_list for start in range(0, repetitions, chunk_size)}
    chunks, previous_seconds = {}, 0.
    if resume is not None:
        if resume["configuration"] != config:
            raise ValueError("Checkpoint configuration changed")
        previous_seconds = float(resume["seconds"])
        chunks = dict(resume["chunks"])
        for key, item in chunks.items():
            if key not in tasks or (item["case"], item["start"], item["stop"]) != tasks[key]:
                raise ValueError("Invalid checkpoint task")
            expected = item["stop"]-item["start"]
            fields = {"accepted", "map", "q_alpha1", "q_upper", "q0"}
            if stage == "validation":
                fields.update(("alpha1_interval", "direct_upper"))
                if item["case"]["ratio"] == .05:
                    fields.update(("bayes_median", "bayes_lower", "bayes_upper", "bayes_direct_upper"))
            if (set(item["records"]) != fields or any(len(v) != expected or not np.all(np.isfinite(v))
                                           for v in item["records"].values())):
                raise ValueError("Incomplete checkpoint records")
    cutoffs = None if frozen is None else frozen["cutoffs"]
    pending = {k: v for k, v in tasks.items() if k not in chunks}
    completed = sum(c["stop"]-c["start"] for c in chunks.values())
    last_saved = completed
    started = time.monotonic()

    def receive(key, result):
        nonlocal completed, last_saved
        chunks[key] = result
        completed += result["stop"]-result["start"]
        if completed-last_saved >= progress_every or len(chunks) == len(tasks):
            seconds = previous_seconds+time.monotonic()-started
            print(f"{stage}: {completed}/{len(case_list)*repetitions} trials; {seconds:.1f} s", flush=True)
            if checkpoint:
                checkpoint(dict(configuration=config, chunks=chunks.copy(), seconds=seconds))
            last_saved = completed

    if workers == 1:
        _initialize(pools, config, cutoffs)
        for key, task in pending.items():
            receive(key, _chunk(task))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(pools, config, cutoffs)) as executor:
            futures = {executor.submit(_chunk, task): key for key, task in pending.items()}
            for future in as_completed(futures):
                receive(futures[future], future.result())
    assembled = {}
    for case in case_list:
        parts = [chunks[case["key"]+f":{start}"] for start in range(0, repetitions, chunk_size)]
        keys = parts[0]["records"]
        records = {k: np.concatenate([np.asarray(part["records"][k]) for part in parts]) for k in keys}
        assembled[case["key"]] = dict(case=case, records=records)
    return dict(configuration=config, cases=assembled,
                seconds=previous_seconds+time.monotonic()-started,
                worker_limit=workers, total_trials=completed)


def summarize(validation, frozen):
    if validation["configuration"]["frozen_sha256"] != digest(frozen):
        raise ValueError("Validation was performed with different cutoffs")
    cutoff = frozen["cutoffs"]
    summaries = {}
    for key, row in validation["cases"].items():
        r = {k: np.asarray(v) for k, v in row["records"].items()}
        truth = np.asarray(row["case"]["fractions"])
        n = len(r["q0"])
        def rate(hits):
            count = int(hits.sum())
            return dict(count=count, total=n, rate=count/n,
                        wilson95=boundary.bootstrap.wilson_interval(count, n))
        interval_ = r["alpha1_interval"]
        result = dict(case=row["case"],
                      alpha1_coverage=rate((interval_[:, 0] <= truth[1]) & (truth[1] <= interval_[:, 1])),
                      upper_coverage=rate(truth[2] <= r["direct_upper"]),
                      rejection=rate(r["q0"] > cutoff["q0"]),
                      alpha1_mean_width=float(np.mean(interval_[:, 1]-interval_[:, 0])),
                      upper_quantiles=np.quantile(r["direct_upper"], [.16, .5, .84]),
                      mle_bias=r["map"].mean(axis=0)-truth,
                      accepted_mean=float(r["accepted"].mean()),
                      at_boundary_count=int(np.sum(r["map"][:, 2] == 0)))
        if "bayes_median" in r:
            low, high = r["bayes_lower"][:, 1], r["bayes_upper"][:, 1]
            result.update(bayes_alpha1_inclusion=rate((low <= truth[1]) & (truth[1] <= high)),
                          bayes_upper_coverage=rate(truth[2] <= r["bayes_direct_upper"]),
                          bayes_alpha1_bias=float(r["bayes_median"][:, 1].mean()-truth[1]),
                          bayes_alpha1_bias_mcse=float(r["bayes_median"][:, 1].std(ddof=1)/np.sqrt(n)),
                          bayes_alpha1_mean_width=float(np.mean(high-low)),
                          bayes_upper_quantiles=np.quantile(r["bayes_direct_upper"], [.16, .5, .84]))
        summaries[key] = result
    return summaries


def validate_numerics(pools, *, budget=100000, tolerance=1e-5):
    """Check actual positive-injection quadrature and profile-root precision."""
    rows = []
    for case in cases((.05,)):
        population = compact_population(pools, case["fractions"])
        samples = [("asimov", budget*population["probability"][:-1]),
                   ("sample", draw(population, NUMERICAL_SEED, case, 0, budget))]
        for label, data in samples:
            model = boundary._Likelihood(data, population["templates"], population["efficiencies"], .5)
            if label == "asimov":
                np.testing.assert_allclose(model.mode, case["fractions"], atol=1e-9, rtol=0)
            a = boundary.fit_boundary(data, population["templates"], population["efficiencies"], grid_size=201)
            b = boundary.fit_boundary(data, population["templates"], population["efficiencies"],
                                      grid_size=401, tail_drop=40.)
            difference = max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k]))))
                             for k in ("median", "lower95", "upper95", "direct_upper95"))
            endpoints = interval(model, 4., 1)
            residual = max(abs(profile_statistic(model, float(x), 1)-4.)
                           for x in endpoints if 0 < x < 1)
            upper = upper_limit(model, 3.)
            residual = max(residual, abs(profile_statistic(model, upper, 2)-3.))
            rows.append(dict(case=case["key"], data=label, posterior_difference=difference,
                             root_statistic_residual=residual))
            print("Numerics:", case["key"], label, flush=True)
    maximum = max(r["posterior_difference"] for r in rows)
    if maximum > tolerance or max(r["root_statistic_residual"] for r in rows) > 1e-6:
        raise RuntimeError("Numerical validation failed")
    return dict(cases=rows, max_posterior_difference=maximum, tolerance=tolerance,
                numerical_seed=NUMERICAL_SEED)


def refine_validation(pools, validation, *, tolerance=1e-5):
    """Refine real production extremes at every nominal injection strength."""
    config = validation["configuration"]
    rows = []
    for row in validation["cases"].values():
        if row["case"]["ratio"] != .05:
            continue
        r = {k: np.asarray(v) for k, v in row["records"].items()}
        indices = {int(np.argmin(r["bayes_direct_upper"])), int(np.argmax(r["bayes_direct_upper"])),
                   int(np.argmin(r["bayes_median"][:, 1])), int(np.argmax(r["q0"]))}
        pop = compact_population(pools, row["case"]["fractions"])
        for index in sorted(indices):
            data = draw(pop, config["seed"], row["case"], index, config["budget"])
            fitted = boundary.fit_boundary(data, pop["templates"], pop["efficiencies"],
                                           grid_size=401, tail_drop=40.)
            pairs = (("bayes_median", "median"), ("bayes_lower", "lower95"),
                     ("bayes_upper", "upper95"), ("bayes_direct_upper", "direct_upper95"))
            difference = max(float(np.max(np.abs(r[a][index]-fitted[b]))) for a, b in pairs)
            rows.append(dict(case=row["case"]["key"], trial=index, difference=difference))
        print("Refined production:", row["case"]["key"], flush=True)
    maximum = max(r["difference"] for r in rows)
    if maximum > tolerance:
        raise RuntimeError("Production posterior integration needs refinement")
    return dict(cases=rows, max_difference=maximum, tolerance=tolerance)


def plot_summary(summaries):
    """Editable figures; no backend selection, file output, or table generation."""
    import matplotlib.pyplot as plt
    nominal = [r for r in summaries.values() if r["case"]["ratio"] == .05]
    nominal.sort(key=lambda r: r["case"]["direct"])
    x = np.array([r["case"]["direct"] for r in nominal])*100
    positions = np.arange(len(x))
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ax, keys, title in zip(axes, (("bayes_alpha1_inclusion", "alpha1_coverage"),
                                     ("bayes_upper_coverage", "upper_coverage")),
                               (r"$\alpha_1$ interval inclusion", "One-sided upper-limit coverage")):
        for offset, key, color, label in zip((-.09, .09), keys, ("#0072B2", "#D55E00"),
                                             ("Bayesian", "Calibrated profile LR")):
            values = np.array([r[key]["rate"] for r in nominal])*100
            bounds = np.array([r[key]["wilson95"] for r in nominal])*100
            ax.errorbar(positions+offset, values, yerr=np.maximum(0., np.array([values-bounds[:, 0], bounds[:, 1]-values])),
                        fmt="o", capsize=3, color=color, label=label)
        ax.axhline(95, color="black", ls="--", lw=1)
        ax.set(xticks=positions, xticklabels=[f"{v:g}" for v in x], ylim=(65, 101),
               xlabel="Injected phase-space fraction [%]", ylabel="Trials containing truth [%]", title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Independent validation: 500 trials per point; r = 0.05; Wilson 95% bars")
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for ratio in VALIDATION_RATIOS:
        selected = sorted((r for r in summaries.values() if r["case"]["ratio"] == ratio),
                          key=lambda r: r["case"]["direct"])
        values = np.array([r["rejection"]["rate"] for r in selected])*100
        axes[0].plot(x, values, "o--" if ratio not in CALIBRATION_RATIOS else "o-",
                     label=f"r = {ratio:g}"+(" (held out)" if ratio not in CALIBRATION_RATIOS else ""))
    axes[0].axhline(5, color="black", ls=":", lw=1)
    axes[0].set(xlabel="Injected phase-space fraction [%]", ylabel="Reject d = 0 [%]",
                title="Detection probability at the calibrated test threshold", ylim=(-2, 102))
    axes[0].legend(fontsize=8)
    for key, color, label in (("bayes_upper_quantiles", "#0072B2", "Bayesian"),
                              ("upper_quantiles", "#D55E00", "Calibrated profile LR")):
        quantiles = np.array([r[key] for r in nominal])*100
        axes[1].plot(x, quantiles[:, 1], "o-", color=color, label=label)
        axes[1].fill_between(x, quantiles[:, 0], quantiles[:, 2], color=color, alpha=.16)
    axes[1].plot(x, x, "k:", label="Injected truth")
    axes[1].set(xlabel="Injected phase-space fraction [%]", ylabel="Upper limit [%]",
                title="Upper-limit median and central 68% trial spread")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
    figures.append(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[3]/"jupyter"/diagnostic.REFERENCE_NAME)
    parser.add_argument("--calibration-trials", type=int, default=2000)
    parser.add_argument("--validation-trials", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pools = diagnostic.load_reference(args.reference)["pools"]
    calibration = run_ensemble(pools, cases(CALIBRATION_RATIOS), stage="calibration",
                               repetitions=args.calibration_trials, seed=CALIBRATION_SEED, workers=args.workers)
    frozen = freeze_cutoffs(calibration)
    print("Frozen cutoffs:", frozen["cutoffs"], flush=True)
    validation = run_ensemble(pools, cases(VALIDATION_RATIOS), stage="validation", frozen=frozen,
                              repetitions=args.validation_trials, seed=VALIDATION_SEED, workers=args.workers)
    for key, row in summarize(validation, frozen).items():
        print(key, "alpha1 coverage:", row["alpha1_coverage"]["rate"],
              "upper coverage:", row["upper_coverage"]["rate"], "rejection:", row["rejection"]["rate"])


if __name__ == "__main__":
    main()
