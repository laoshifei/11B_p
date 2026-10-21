"""Compare a model-exact null population with the independent 162-keV pool.

The fitted shapes, efficiencies, flat generator-simplex prior, and boundary
integrator are unchanged. Only the data-generating distribution changes.
This diagnoses pointwise inclusion; it does not calibrate confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import zero_branch_162_study as boundary

VERSION = "2026-10-19-v1-model-exact-boundary-control"
DEFAULT_SEED = 16220261021
# Chapter A retains the original zero-branch checkpoint and provenance schema.
REFERENCE_NAME = "codex_0003_162keV_弱分支检验与区间校准.ipynb"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(path):
    """Read the completed notebook and verify its saved pools and source code."""
    path = Path(path).resolve()
    notebook = json.loads(path.read_text())
    metadata = notebook["metadata"]
    if (metadata.get("codex_run_status") != "completed"
            or metadata.get("codex_report_status") != "complete"):
        raise ValueError("The independent-pool reference is incomplete")
    saved = metadata["codex_checkpoint"]
    configuration = saved["trials"]["configuration"]
    if (configuration["version"] != boundary.VERSION
            or configuration["budget"] != 100000
            or saved["trials"]["repetitions"] != 500
            or configuration["fractions"] != boundary.TRUE_FRACTIONS.tolist()
            or configuration["grid_size"] != 201):
        raise ValueError("Reference conditions differ from the planned control")
    # Verify against this project's code even when the notebook is archived.
    root = Path(__file__).resolve().parents[3]
    for name, expected in saved["provenance"]["script_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Reference analysis source changed: {name}")
    pools = {k: np.asarray(v) for k, v in saved["pools"].items()}
    digest = hashlib.sha256()
    for key in ("templates", "mixed", "pure_generated", "mixed_generated"):
        digest.update(np.asarray(pools[key], np.int64).tobytes())
    if digest.hexdigest() != configuration["population_sha256"]:
        raise ValueError("Reference histogram cache changed")
    records = {k: np.asarray(v) for k, v in saved["trials"]["records"].items()}
    if any(len(v) != 500 or not np.all(np.isfinite(v)) for v in records.values()):
        raise ValueError("Reference trial records are incomplete")
    return dict(pools=pools, trials=saved["trials"], asimov=saved["asimov"],
                provenance=saved["provenance"], inputs=saved["inputs"],
                notebook_sha256=sha256(path))


def exact_population(pools, *, fractions=boundary.TRUE_FRACTIONS, pseudocount=.5):
    """Build unconditional probabilities from exactly the fitted shapes.

    Smooth only the fixed union of occupied template bins. Efficiencies use
    raw accepted/generated counts, without pseudocounts. Include a rejected
    outcome so each trial has a fixed generated budget, not accepted budget.
    """
    templates = np.asarray(pools["templates"], float)
    generated = np.asarray(pools["pure_generated"], float)
    fractions = np.asarray(fractions, float)
    if (templates.ndim < 2 or templates.shape[0] != 3 or generated.shape != (3,)
            or fractions.shape != (3,) or not np.all(np.isfinite(templates))
            or np.any(templates < 0) or not np.all(np.isfinite(generated))
            or np.any(generated <= 0) or not np.all(np.isfinite(fractions))
            or np.any(fractions < 0) or abs(fractions.sum()-1) > 1e-12
            or not np.isfinite(pseudocount) or pseudocount < 0):
        raise ValueError("Invalid template counts, generated totals, fractions, or smoothing")
    raw_accepted = templates.reshape(3, -1).sum(axis=1)
    efficiency = raw_accepted / generated
    if np.any(efficiency <= 0) or np.any(efficiency > 1):
        raise ValueError("Each channel needs an efficiency in (0, 1]")
    active = np.any(templates > 0, axis=0)
    shapes = np.zeros_like(templates)
    smoothed = templates[:, active] + pseudocount
    shapes[:, active] = smoothed / smoothed.sum(axis=1, keepdims=True)
    accepted = (fractions*efficiency) @ shapes.reshape(3, -1)
    rejected = 1-float(fractions @ efficiency)
    probability = np.r_[accepted, max(0., rejected)]
    if not np.isclose(probability.sum(), 1., rtol=0, atol=2e-14):
        raise RuntimeError("Generated probabilities failed normalization")
    probability /= probability.sum()
    digest = hashlib.sha256()
    for name, array in (("templates", templates), ("generated", generated),
                        ("fractions", fractions), ("probability", probability)):
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    return dict(templates=templates, efficiencies=efficiency, shapes=shapes,
                probability=probability, active=active, fractions=fractions.copy(),
                pseudocount=float(pseudocount), population_sha256=digest.hexdigest())


def _fit(data, population, **options):
    fit = boundary.fit_boundary(data, population["templates"], population["efficiencies"],
                                pseudocount=population["pseudocount"], **options)
    if fit["active_bins"] != int(population["active"].sum()):
        raise RuntimeError("Data changed the supposedly frozen fitted support")
    return fit


def summarize(records, fractions=boundary.TRUE_FRACTIONS):
    """Use the same estimators as the reference and add directional misses."""
    r = {k: np.asarray(v) for k, v in records.items()}
    fractions = np.asarray(fractions)
    n = len(r["median"])
    median, low, high = (r[k] for k in ("median", "lower95", "upper95"))
    hits = (low[:, :2] <= fractions[:2]) & (fractions[:2] <= high[:, :2])
    upper = r["direct_upper95"]
    at_boundary = r["map"][:, 2] == 0
    return dict(mean=median.mean(axis=0), bias=median.mean(axis=0)-fractions,
                sd=median.std(axis=0, ddof=1),
                bias_mcse=median.std(axis=0, ddof=1)/np.sqrt(n),
                mean_width=(high-low).mean(axis=0),
                sequential_hits=hits.sum(axis=0), sequential_inclusion=hits.mean(axis=0),
                sequential_wilson95=boundary.bootstrap.wilson_interval(hits.sum(axis=0), n),
                truth_below_interval=(fractions[:2] < low[:, :2]).sum(axis=0),
                truth_above_interval=(fractions[:2] > high[:, :2]).sum(axis=0),
                upper95_quantiles=np.quantile(upper, [.025, .16, .5, .84, .975]),
                upper95_mean=float(upper.mean()), upper95_sd=float(upper.std(ddof=1)),
                upper95_mean_mcse=float(upper.std(ddof=1)/np.sqrt(n)),
                map_boundary_count=int(at_boundary.sum()),
                map_boundary_fraction=float(at_boundary.mean()),
                map_boundary_wilson95=boundary.bootstrap.wilson_interval(at_boundary.sum(), n),
                accepted_mean=float(r["accepted"].mean()),
                accepted_sd=float(r["accepted"].std(ddof=1)),
                q0_quantiles=np.quantile(r["q0"], [.5, .9, .95, .99]))


def run_control(population, *, repetitions=500, budget=100000, seed=DEFAULT_SEED,
                grid_size=201, progress_every=25, checkpoint=None, resume=None):
    """Independent draws with deterministic checkpoint/restart support."""
    if (int(repetitions) != repetitions or repetitions < 2
            or int(budget) != budget or budget < 1 or progress_every < 1):
        raise ValueError("Need integer repetitions >= 2, positive budget and progress interval")
    configuration = dict(version=VERSION, fitter_version=boundary.VERSION,
                         fitter_sha256=sha256(boundary.__file__),
                         study_sha256=sha256(__file__),
                         population_sha256=population["population_sha256"],
                         fractions=population["fractions"].tolist(),
                         pseudocount=population["pseudocount"], budget=budget,
                         seed=seed, grid_size=grid_size, tail_drop=32.)
    records = {k: np.empty((repetitions, 3)) for k in ("median", "lower95", "upper95", "map")}
    records.update({k: np.empty(repetitions) for k in ("direct_upper95", "q0", "accepted")})
    completed, previous_seconds = 0, 0.
    if resume is not None:
        if resume["configuration"] != configuration:
            raise ValueError("Checkpoint configuration changed")
        completed, previous_seconds = int(resume["completed"]), float(resume["seconds"])
        if not 0 <= completed <= repetitions:
            raise ValueError("Invalid checkpoint length")
        for key, values in records.items():
            saved = np.asarray(resume["records"][key])
            if saved.shape != values[:completed].shape or not np.all(np.isfinite(saved)):
                raise ValueError("Incomplete checkpoint records")
            values[:completed] = saved
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    shape = population["templates"].shape[1:]
    for i in range(repetitions):
        draw = rng.multinomial(budget, population["probability"])
        data = draw[:-1].reshape(shape)
        if i < completed:
            if data.sum() != records["accepted"][i]:
                raise ValueError("Checkpoint random sequence changed")
            continue
        result = _fit(data, population, grid_size=grid_size)
        for key in records:
            records[key][i] = data.sum() if key == "accepted" else result[key]
        if (i+1) % progress_every == 0 or i+1 == repetitions:
            seconds = previous_seconds + time.monotonic()-started
            print(f"Model-exact trials {i+1}/{repetitions}: {seconds:.1f} s", flush=True)
            if checkpoint is not None:
                checkpoint(dict(configuration=configuration, completed=i+1, seconds=seconds,
                                records={k: v[:i+1].copy() for k, v in records.items()}))
    summary = summarize(records, population["fractions"])
    summary["expected_accepted"] = float(budget*(1-population["probability"][-1]))
    return dict(configuration=configuration, records=records, summary=summary,
                repetitions=repetitions, seconds=previous_seconds+time.monotonic()-started,
                scope="Model-exact fixed-template posterior; no interval calibration or discovery claim")


def _differences(a, b):
    return {key: float(np.max(np.abs(np.asarray(a[key])-np.asarray(b[key]))))
            for key in ("median", "lower95", "upper95", "direct_upper95")}


def validate_population(population, *, budget=100000, seed=16220261022, tolerance=1e-5):
    """Check the actual fitted probability, Asimov truth, grid, and tails."""
    probability, active = population["probability"], population["active"]
    probe = boundary._Likelihood(active.astype(float), population["templates"],
                                 population["efficiencies"], population["pseudocount"])
    fitted = probe.probability(population["fractions"])
    generated = probability[:-1].reshape(active.shape)[active] / (1-probability[-1])
    equality_error = float(np.max(np.abs(fitted-generated)))
    if equality_error > 2e-14:
        raise RuntimeError("Generating and fitted conditional probabilities differ")
    shape = active.shape
    samples = [("asimov", budget*probability[:-1].reshape(shape))]
    rng = np.random.default_rng(seed)
    for i in range(3):
        samples.append((f"null_trial_{i+1}", rng.multinomial(budget, probability)[:-1].reshape(shape)))
    rows = []
    for name, data in samples:
        a = _fit(data, population, grid_size=201, return_density=name == "asimov")
        b = _fit(data, population, grid_size=401)
        c = _fit(data, population, grid_size=401, tail_drop=40.)
        rows.append(dict(case=name, grid=_differences(a, b), tail=_differences(b, c)))
        if name == "asimov":
            asimov = a
            if np.max(np.abs(a["map"]-population["fractions"])) > 1e-9 or a["q0"] > 1e-6:
                raise RuntimeError("Exact expected counts did not recover the prescribed truth")
        print(f"Exact-population integration check: {name}", flush=True)
    maximum = max(max(row[k].values()) for row in rows for k in ("grid", "tail"))
    if maximum > tolerance:
        raise RuntimeError(f"Boundary integration needs refinement: {maximum:g}")
    return dict(asimov=asimov, cases=rows, probability_max_error=equality_error,
                max_difference=maximum, tolerance=tolerance, seed=seed)


def refine_trials(population, trials, tolerance=1e-5):
    r = {k: np.asarray(v) for k, v in trials["records"].items()}
    order = np.argsort(r["direct_upper95"])
    selected = {int(order[0]), int(order[-1]), int(order[len(order)//2]),
                int(np.argmin(r["median"][:, 1])), int(np.argmax(r["median"][:, 1])),
                int(np.argmax(r["q0"]))}
    on_boundary = np.flatnonzero(r["map"][:, 2] == 0)
    if on_boundary.size:
        selected.add(int(on_boundary[0]))
    cfg = trials["configuration"]
    rng, rows = np.random.default_rng(cfg["seed"]), []
    for i in range(trials["repetitions"]):
        data = rng.multinomial(cfg["budget"], population["probability"])[:-1]
        if i not in selected:
            continue
        fitted = _fit(data.reshape(population["active"].shape), population,
                      grid_size=401, tail_drop=40.)
        rows.append(dict(trial=i+1, differences=_differences({k: v[i] for k, v in r.items()}, fitted)))
    maximum = max(max(row["differences"].values()) for row in rows)
    if maximum > tolerance:
        raise RuntimeError(f"Production refinement failed: {maximum:g}")
    return dict(cases=rows, max_difference=maximum, tolerance=tolerance,
                refined_grid=401, refined_tail_drop=40.)


def compare(control, reference):
    """Independent-ensemble differences and Monte Carlo standard errors."""
    a, b = summarize(control["records"]), summarize(reference["records"])
    na, nb = control["repetitions"], reference["repetitions"]
    pa, pb = a["sequential_inclusion"], b["sequential_inclusion"]
    return dict(control=a, reference=b,
                bias_difference=a["bias"]-b["bias"],
                bias_difference_mcse=np.sqrt(a["bias_mcse"]**2+b["bias_mcse"]**2),
                inclusion_difference=pa-pb,
                inclusion_difference_mcse=np.sqrt(pa*(1-pa)/na+pb*(1-pb)/nb),
                upper_mean_difference=a["upper95_mean"]-b["upper95_mean"],
                upper_mean_difference_mcse=float(np.hypot(a["upper95_mean_mcse"], b["upper95_mean_mcse"])))


def plot_comparison(control, reference, exact_asimov, reference_asimov):
    """Return editable figures; save nothing and leave the backend unchanged."""
    import matplotlib.pyplot as plt
    comparison = compare(control, reference)
    colors, labels = ("#0072B2", "#D55E00"), ("A: model-exact", "B: independent pool")
    fig1, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for offset, key, color, label in zip((-.06, .06), ("control", "reference"), colors, labels):
        s = comparison[key]
        y, interval = 100*s["sequential_inclusion"], 100*s["sequential_wilson95"]
        axes[0].errorbar(np.arange(2)+offset, y,
                        yerr=np.maximum(0, np.array([y-interval[:, 0], interval[:, 1]-y])),
                        fmt="o", capsize=4, color=color, label=label)
    axes[0].axhline(95, color="black", ls="--", label="95% reference")
    axes[0].set(xticks=[0, 1], xticklabels=[r"$\alpha_0$", r"$\alpha_1$"],
                ylabel="Truth inclusion [%]", title="Central 95% credible intervals", ylim=(65, 100))
    axes[0].legend(fontsize=8)
    residuals = [100*(np.asarray(t["records"]["median"])[:, 1]-.95)
                 for t in (control, reference)]
    low, high = min(v.min() for v in residuals), max(v.max() for v in residuals)
    margin = max(.01, .03*(high-low))
    bins = np.linspace(low-margin, high+margin, 35)
    for trials, color, label in zip((control, reference), colors, labels):
        medians = np.asarray(trials["records"]["median"])
        axes[1].hist(100*(medians[:, 1]-.95), bins=bins, histtype="step", lw=1.5,
                     color=color, label=label)
    axes[1].axvline(0, color="black", ls="--")
    axes[1].set(xlabel=r"$\alpha_1$ posterior median minus 95% [pp]", ylabel="Trials",
                title="Residual distribution")
    axes[1].legend(fontsize=8)
    fig1.suptitle("5/95/0 truth; 500 independent trials per population")

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for trials, asimov, color, label in zip((control, reference),
            (exact_asimov, reference_asimov), colors, labels):
        values = 100*np.sort(trials["records"]["direct_upper95"])
        axes[0].step(values, np.arange(1, len(values)+1)/len(values), where="post", color=color, label=label)
        axes[1].plot(100*np.asarray(asimov["direct_grid"]),
                     np.asarray(asimov["direct_density"])/100, color=color, label=label)
    axes[0].set(xlabel="Phase-space fraction: one-sided 95% upper limit [%]",
                ylabel="Fraction of trials", title="Upper-limit distribution", ylim=(0, 1))
    axes[1].set(xlabel="Phase-space generator fraction [%]", ylabel="Posterior density per %",
                title="Expected-count posterior")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig2.suptitle("100,000 generated reactions; matched response; fixed templates")
    return [fig1, fig2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[3]/"jupyter"/REFERENCE_NAME)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    reference = load_reference(args.reference)
    population = exact_population(reference["pools"])
    validation = validate_population(population)
    trials = run_control(population, repetitions=args.repetitions)
    refine_trials(population, trials)
    print("Inclusion:", trials["summary"]["sequential_inclusion"])
    print("Bias [pp]:", 100*trials["summary"]["bias"])
    print("Direct upper-limit median [%]:", 100*trials["summary"]["upper95_quantiles"][2])
    print("Asimov mode:", validation["asimov"]["map"])


if __name__ == "__main__":
    main()
