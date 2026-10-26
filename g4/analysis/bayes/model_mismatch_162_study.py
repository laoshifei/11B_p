"""Paired 162-keV fits with matched response and deliberately different physics.

The third shape is a uniform phase-space control, not a measured direct-decay
branch. All inference is conditional on the finite, fixed Monte Carlo banks.
Simulation and this study run only through their explicit command entry points.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from itertools import permutations
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import xlogy
from scipy.stats import norm
import uproot

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import calibrate_162_weak_branch_intervals as calibration
from analysis.bayes import scan_detector_theta_phi_response_162_channel_fit as scan
from analysis.bayes import zero_branch_162_study as boundary
from analysis.bayes.generate_162_model_libraries import CASES, NOTEBOOK, ROOT, sha256

VERSION = "2026-10-22-v1-paired-model-mismatch"
MODELS = ("coherent", "incoherent", "legacy")
FRACTIONS = np.array([.05, .95, 0.])
LIMITS = (-1.05, 1.05)
SPLIT_SEED = 1622026102201
RESPONSE_SEED = 1622026102202
CALIBRATION_SEED = 1622026102203
VALIDATION_SEED = 1622026102204
EMPIRICAL_SEED = 1622026102205


def save_key(notebook, key, value):
    """Read the latest document so other notebook metadata are preserved."""
    notebook = Path(notebook)
    document = json.loads(notebook.read_text())
    document["metadata"][key] = calibration.plain(value)
    temporary = notebook.with_suffix(".pending.ipynb")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False)+"\n")
    temporary.replace(notebook)


def split_ids(ids, seed):
    """Partition generated reactions before selection, including lost events."""
    ids = np.asarray(ids, np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("Need at least two unique generated event IDs")
    shuffled = np.random.default_rng(seed).permutation(ids)
    middle = len(ids)//2
    return np.sort(shuffled[:middle]), np.sort(shuffled[middle:])


def ordered_library_runs(runs):
    """RNG namespaces must not depend on JSON/notebook dictionary ordering."""
    if set(runs) != set(CASES):
        raise ValueError("Incomplete or unexpected model-library set")
    return [(name, runs[name]) for name in CASES]


def _clip(polygon, normal):
    if not polygon:
        return []
    result = []
    previous = np.asarray(polygon[-1])
    previous_value = previous @ normal
    for current in polygon:
        current = np.asarray(current)
        value = current @ normal
        if (value >= 0) != (previous_value >= 0):
            result.append(previous+(current-previous)*(-previous_value)/(value-previous_value))
        if value >= 0:
            result.append(current)
        previous, previous_value = current, value
    return result


def geometric_support(bins=60, limits=LIMITS):
    """Bins intersecting the ordered equal-mass Dalitz wedge and unit disk.

    This conservative kinematic support is specified before observing Monte
    Carlo. A joint Monte Carlo zero inside it remains a sampling zero.
    """
    edges = np.linspace(*limits, bins+1)
    mask = np.zeros((bins, bins), bool)
    for i, x in enumerate(edges[:-1]):
        for j, y in enumerate(edges[:-1]):
            polygon = [np.array(p) for p in ((x, y), (edges[i+1], y),
                       (edges[i+1], edges[j+1]), (x, edges[j+1]))]
            for normal in (np.array([1., 0.]), np.array([-1/np.sqrt(3), 1.])):
                polygon = _clip(polygon, normal)
            if not polygon:
                continue
            nearest = min(float(p @ p) for p in polygon)
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                delta = b-a
                if delta @ delta > 0:
                    point = a+np.clip(-(a @ delta)/(delta @ delta), 0, 1)*delta
                    nearest = min(nearest, float(point @ point))
            mask[i, j] = nearest <= 1+1e-12
    return mask


def histogram_xy(xy, bins):
    histogram = np.histogram2d(*np.asarray(xy).T, bins=bins, range=[LIMITS, LIMITS])[0]
    if histogram.sum() != len(xy):
        raise RuntimeError("Dalitz points outside the declared histogram")
    support = geometric_support(bins)
    if np.any(histogram[~support]):
        raise RuntimeError("Observed points outside the geometric support")
    return histogram


def population(histograms, generated, fractions=FRACTIONS):
    """Unconditional draw law: accepted bins plus one rejection category."""
    h, generated, fractions = map(np.asarray, (histograms, generated, fractions))
    if (h.ndim != 2 or generated.shape != (len(h),) or fractions.shape != (len(h),)
            or np.any(generated <= 0) or np.any(h < 0)
            or np.any(h.sum(axis=1) > generated+1e-8)
            or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1)):
        raise ValueError("Invalid generated population")
    accepted = (fractions/generated) @ h
    probability = np.r_[accepted, max(0., 1-accepted.sum())]
    return probability/probability.sum()


def exact_population(templates, efficiency):
    probability = (FRACTIONS*efficiency) @ (templates/templates.sum(axis=1, keepdims=True))
    return np.r_[probability, 1-probability.sum()]


def fit(data, bank, bayesian=False):
    model = boundary._Likelihood(data, bank["templates"], bank["efficiency"], 0.)
    result = dict(map=model.mode, q0=calibration.profile_statistic(model, 0., 2),
                  accepted=float(np.sum(data)),
                  deviance=float(2*np.sum(xlogy(model.counts,
                         model.relative/model.qmode))))
    if bayesian:
        result["bayesian"] = boundary.fit_boundary(data, bank["templates"], bank["efficiency"],
                                  grid_size=201, pseudocount=0., return_density=True)
    return result


def prepare_libraries(notebook=NOTEBOOK):
    libraries = json.loads(Path(notebook).read_text())["metadata"]["codex_model_mismatch_libraries"]
    if not libraries.get("completed_at"):
        raise ValueError("Wait until all matched simulation libraries are complete")
    selected, summaries, physical = {}, {}, {}
    for number, (name, run) in enumerate(ordered_library_runs(libraries["runs"])):
        path = ROOT/run["root"]
        if sha256(path) != run["root_sha256"]:
            raise RuntimeError(f"Input ROOT changed: {path}")
        ids = scan.load_reaction_event_ids(path)
        complete = scan.load_complete_three_alpha(path, load_hit_position=True)
        keep = scan.quality_mask(complete["kinetic"], complete["momentum"], 1e-4)
        events = scan.apply_mask(complete, keep)
        if not np.all(np.isin(events["event_id"], ids)):
            raise RuntimeError("Sphere event IDs missing from generated reactions")
        subsets = dict(template=ids)
        if name in ("alpha0", "coherent"):
            subsets = dict(zip(("template", "data"), split_ids(ids, [SPLIT_SEED, number])))
        summaries[name] = dict(generated=len(ids), complete=len(complete["event_id"]),
                               quality=len(events["event_id"]), subsets={})
        for subset_number, (subset, allocated) in enumerate(subsets.items()):
            sample = scan.apply_mask(events, np.isin(events["event_id"], allocated))
            reconstructed, angle = scan.reconstruct_with_granularity(
                sample, 2., 2., 0., 0., "sphere-center", np.array([0., 0., 170.]))
            rng = np.random.default_rng(np.random.SeedSequence([RESPONSE_SEED, number, subset_number]))
            measured, _, response = scan.apply_closure_energy_response(
                reconstructed, 100., 100., rng.standard_normal(reconstructed["kinetic"].shape))
            xy = np.column_stack(scan.ordered_dalitz_xy(measured))
            first_half, _ = split_ids(allocated, [SPLIT_SEED, number, subset_number, 17])
            selected[(name, subset)] = dict(generated=len(allocated), xy=xy,
                half_xy=xy[np.isin(measured["event_id"], first_half)], half_generated=len(first_half))
            summaries[name]["subsets"][subset] = dict(generated=len(allocated),
                complete_quality=len(sample["event_id"]), accepted=len(measured["event_id"]),
                efficiency=len(measured["event_id"])/len(allocated),
                ids_sha256=calibration.digest(allocated), angle=angle, energy=response)
        if name in MODELS:
            with uproot.open(path, handler=uproot.source.file.MemmapSource) as stream:
                fields = [f"e_3alpha_cm_alpha{i}" for i in (1, 2, 3)]+["event"]
                arrays = stream["reaction"].arrays(fields, library="np")
            rows = np.isin(arrays["event"], subsets["template"])
            energies = np.column_stack([arrays[f"e_3alpha_cm_alpha{i}"][rows] for i in (1, 2, 3)])
            spectrum, energy_edges = np.histogram(energies, bins=120, range=(0., 6.))
            dalitz = np.zeros((90, 90))
            for permutation in permutations(range(3)):
                a, b, c = energies[:, permutation].T
                x, y = np.sqrt(3)*(b-c)/(a+b+c), (2*a-b-c)/(a+b+c)
                dalitz += np.histogram2d(x, y, bins=90, range=[LIMITS, LIMITS])[0]/6
            if not np.isclose(dalitz.sum(), len(energies)) or spectrum.sum() != 3*len(energies):
                raise RuntimeError("Physical plot lost generated particles/events")
            physical[name] = dict(events=len(energies), energy_edges=energy_edges,
                energy_density=spectrum/(3*len(energies)*np.diff(energy_edges)),
                symmetric_dalitz_probability=dalitz/len(energies))
        print(f"Loaded {name}: {summaries[name]}", flush=True)
    provenance = dict(library_configuration=libraries["configuration"],
        roots={name: {key: row[key] for key in ("root", "root_sha256", "macro_sha256")}
               for name, row in ordered_library_runs(libraries["runs"])}, split_seed=SPLIT_SEED,
        library_rng_ids={name: index for index, name in enumerate(CASES)},
        response_seed=RESPONSE_SEED, reconstruction_mass_MeV=scan.ALPHA_MASS_MEV,
        response=dict(theta_bin_deg=2., phi_bin_deg=2., energy_FWHM_keV=100.,
                      each_alpha_threshold_keV=100., sphere_center_mm=[0., 0., 170.]),
        limits=LIMITS, fraction_definition="generated reaction fractions; third=uniform phase space")
    return selected, summaries, physical, provenance


def make_banks(selected, bins=60, pseudocount=.5, half=False):
    support = geometric_support(bins)
    histograms, generated = {}, {}
    for name in ("alpha0", "phase_space", *MODELS):
        row = selected[(name, "template")]
        histograms[name] = histogram_xy(row["half_xy"] if half else row["xy"], bins)[support]
        generated[name] = row["half_generated"] if half else row["generated"]
    banks = {}
    for name in MODELS:
        hist = np.stack([histograms["alpha0"], histograms[name], histograms["phase_space"]])
        gen = np.array([generated["alpha0"], generated[name], generated["phase_space"]])
        banks[name] = dict(templates=hist+pseudocount, raw_templates=hist,
            efficiency=hist.sum(axis=1)/gen, generated=gen, support=support,
            bins=bins, pseudocount=pseudocount)
    # This null experiment never draws the third component. Its unused row
    # merely keeps the shared population API's three-channel ordering.
    data_hist = np.stack([histogram_xy(selected[(name, "data")]["xy"], bins)[support]
                          for name in ("alpha0", "coherent")])
    data_generated = np.array([selected[(name, "data")]["generated"] for name in ("alpha0", "coherent")])
    probability = population(np.vstack([data_hist, np.zeros(support.sum())]),
                              np.r_[data_generated, 1])
    return banks, probability


def _batch(args):
    bank, probability, seed, identity, first, last, budget = args
    records = []
    for index in range(first, last):
        rng = np.random.default_rng(np.random.SeedSequence([seed, identity, index]))
        data = rng.multinomial(budget, probability)[:-1]
        records.append(fit(data, bank))
    return records


def ensemble(banks, probabilities, seed, repetitions, budget, workers, *, paired=False):
    jobs, labels = [], []
    for number, name in enumerate(MODELS):
        for first in range(0, repetitions, 100):
            labels.append(name)
            jobs.append((banks[name], probabilities[name], seed, 0 if paired else number,
                         first, min(first+100, repetitions), budget))
    grouped = {name: [] for name in MODELS}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name, records in zip(labels, pool.map(_batch, jobs)):
            grouped[name].extend(records)
            print(f"seed {seed}, {name}: {len(grouped[name])}/{repetitions} fits", flush=True)
    return {name: {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            for name, rows in grouped.items()}


def binomial_summary(successes, repetitions):
    z = norm.ppf(.975)
    p = successes/repetitions
    centre = (p+z*z/(2*repetitions))/(1+z*z/repetitions)
    half = z*np.sqrt(p*(1-p)/repetitions+z*z/(4*repetitions**2))/(1+z*z/repetitions)
    return dict(successes=int(successes), repetitions=int(repetitions), rate=p,
                wilson95=[centre-half, centre+half])


def summarize(records, cutoffs):
    return {name: dict(false_positive=binomial_summary(np.count_nonzero(row["q0"]>cutoffs[name]), len(row["q0"])),
        extra_map_quantiles=np.quantile(row["map"][:, 2], [.025, .5, .975]),
        mean_map=row["map"].mean(axis=0), mean_bias=row["map"].mean(axis=0)-FRACTIONS,
        mean_MCSE=row["map"].std(axis=0, ddof=1)/np.sqrt(len(row["q0"])),
        q0_quantiles=np.quantile(row["q0"], [.025, .5, .975]),
        deviance_quantiles=np.quantile(row["deviance"], [.025, .5, .975]))
        for name, row in records.items()}


LABELS = {"coherent": "Coherent L2", "incoherent": "Incoherent L2", "legacy": "Legacy A2/A4"}
COLORS = {"coherent": "C0", "incoherent": "C1", "legacy": "C2"}


def plot_energy(state):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for name in MODELS:
        row = state["physical"][name]
        ax.stairs(row["energy_density"], row["energy_edges"], label=LABELS[name], color=COLORS[name])
    ax.set(xlabel=r"Single-alpha CM kinetic energy (MeV)", ylabel=r"Probability density (MeV$^{-1}$)",
           xlim=(0, 6), title="162 keV: generated alpha1 channel, before response")
    ax.legend()
    return fig


def plot_physical_dalitz(state):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True, constrained_layout=True)
    maximum = max(np.asarray(state["physical"][name]["symmetric_dalitz_probability"]).max() for name in MODELS)
    for ax, name in zip(axes, MODELS):
        h = np.asarray(state["physical"][name]["symmetric_dalitz_probability"])
        im = ax.imshow(np.ma.masked_equal(h.T, 0), origin="lower", extent=(*LIMITS, *LIMITS),
                       norm=LogNorm(vmin=1e-6, vmax=maximum), cmap="viridis")
        ax.set(title=LABELS[name], xlabel=r"$x=\sqrt{3}(T_2-T_3)/\sum T$", xlim=(-1.05, 1.05))
    axes[0].set_ylabel(r"$y=(2T_1-T_2-T_3)/\sum T$")
    fig.colorbar(im, ax=axes, label="Probability per bin (six permutations, weight 1/6)", shrink=.82)
    return fig


def plot_asimov(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for name in MODELS:
        row = state["asimov"][name]
        posterior = row["bayesian"]
        axes[0].plot(100*np.asarray(posterior["direct_grid"]),
                     np.asarray(posterior["direct_density"])/100, color=COLORS[name], label=LABELS[name])
        axes[0].axvline(100*posterior["direct_upper95"], color=COLORS[name], ls="--", alpha=.7)
    axes[0].set(xlabel="Extra phase-space fraction d (%)", ylabel="Posterior density per percentage point",
                title="Coherent-data Asimov fits; dashed: Bayesian U95")
    axes[0].legend()
    coordinates = np.arange(3)
    axes[1].bar(coordinates, [state["asimov"][name]["deviance"] for name in MODELS],
                color=[COLORS[name] for name in MODELS])
    axes[1].set(xticks=coordinates, xticklabels=[LABELS[name] for name in MODELS],
                ylabel="Conditional multinomial deviance", title="Residual lack of fit (diagnostic, no p-value)")
    return fig


def plot_ensemble(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for name in MODELS:
        values = 100*np.sort(np.asarray(state["empirical_validation"]["records"][name]["map"])[:, 2])
        axes[0].step(values, np.arange(1, len(values)+1)/len(values), where="post", color=COLORS[name], label=LABELS[name])
    axes[0].set(xlabel="Fitted extra phase-space fraction d (%)", ylabel="Empirical CDF",
                title="Paired coherent-data trials; true d = 0")
    axes[0].legend()
    maximum_interval = 0.
    for stage, offset, marker in (("exact_validation", -.09, "o"), ("empirical_validation", .09, "s")):
        rows = [state[stage]["summary"][name]["false_positive"] for name in MODELS]
        rate = 100*np.array([row["rate"] for row in rows])
        bounds = 100*np.array([row["wilson95"] for row in rows])
        errors = np.vstack((rate-bounds[:, 0], bounds[:, 1]-rate))
        if errors.min() < -1e-10:
            raise ValueError("Wilson interval does not contain the observed rate")
        # At 0/n, roundoff can put the lower Wilson endpoint ~1e-19 above 0.
        maximum_interval = max(maximum_interval, float(bounds[:, 1].max()))
        axes[1].errorbar(np.arange(3)+offset, rate, yerr=np.maximum(errors, 0.),
                         fmt=marker, capsize=3, label="Exact template law" if stage == "exact_validation" else "Coherent data law")
    axes[1].axhline(5, color=".5", ls="--", lw=1)
    axes[1].set(xticks=np.arange(3), xticklabels=[LABELS[name] for name in MODELS],
                ylabel="False positive rate (%)", title="Frozen local-null cutoffs; Wilson 95% intervals",
                ylim=(-.5, min(103., max(10., 1.15*maximum_interval))))
    axes[1].legend()
    return fig


def plot_residuals(state):
    import matplotlib.pyplot as plt
    budget = state["configuration"]["budget"]
    data = budget*np.asarray(state["empirical_probability"][:-1])
    residuals = []
    support = np.asarray(state["banks"]["coherent"]["support"], bool)
    for name in MODELS:
        bank = state["banks"][name]
        templates, efficiency = np.asarray(bank["templates"]), np.asarray(bank["efficiency"])
        weights = np.asarray(state["asimov"][name]["map"])*efficiency
        expected = data.sum()*(weights @ (templates/templates.sum(axis=1, keepdims=True)))/weights.sum()
        image = np.full(support.shape, np.nan)
        image[support] = (data-expected)/np.sqrt(expected)
        residuals.append(image)
    maximum = max(float(np.nanmax(abs(h))) for h in residuals)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, name, image in zip(axes, MODELS, residuals):
        im = ax.imshow(image.T, origin="lower", extent=(*LIMITS, *LIMITS),
                       cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        ax.set(title=LABELS[name], xlabel="Ordered Dalitz x", xlim=(0, 1.05), ylim=(0, 1.05))
    axes[0].set_ylabel("Ordered Dalitz y")
    fig.colorbar(im, ax=axes, label=r"Asimov residual $(n_b-\mu_b)/\sqrt{\mu_b}$", shrink=.85)
    return fig


def run(notebook=NOTEBOOK, workers=4, repetitions=500, calibration_repetitions=2000, budget=100000):
    if not 1 <= workers <= 8:
        raise ValueError("Use at most 8 workers")
    selected, summaries, physical, provenance = prepare_libraries(notebook)
    banks, empirical = make_banks(selected)
    configuration = dict(version=VERSION, provenance=provenance, budget=budget, bins=60,
        pseudocount=.5, calibration_repetitions=calibration_repetitions, repetitions=repetitions,
        calibration_seed=CALIBRATION_SEED, validation_seed=VALIDATION_SEED, empirical_seed=EMPIRICAL_SEED,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            (Path(__file__), ROOT/"g4/analysis/bayes/zero_branch_162_study.py",
             ROOT/"g4/analysis/bayes/calibrate_162_weak_branch_intervals.py",
             ROOT/"g4/analysis/bayes/scan_detector_theta_phi_response_162_channel_fit.py")},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    previous = json.loads(Path(notebook).read_text())["metadata"].get("codex_model_mismatch_study")
    if previous and previous["configuration"] != calibration.plain(configuration):
        raise ValueError("Existing study has a different configuration")
    state = previous or dict(configuration=configuration, started_at=datetime.now().astimezone().isoformat(),
        input_summary=summaries, physical=physical, banks=banks, empirical_probability=empirical)
    checkpoint = lambda: save_key(notebook, "codex_model_mismatch_study", state)
    if "asimov" not in state:
        exact = exact_population(banks["coherent"]["templates"], banks["coherent"]["efficiency"])
        analytic = fit(budget*exact[:-1], banks["coherent"])
        if np.max(abs(analytic["map"]-FRACTIONS)) > 1e-8 or abs(analytic["deviance"]) > 1e-6:
            raise RuntimeError("Exact model Asimov closure failed")
        state["analytic_closure"] = analytic
        state["asimov"] = {name: fit(budget*empirical[:-1], bank, bayesian=True)
                            for name, bank in banks.items()}
        state["sensitivity"] = []
        for bins, pseudo, half in [(30,.5,False), (90,.5,False), (60,.1,False), (60,1.,False), (60,.5,True)]:
            varied, data = make_banks(selected, bins, pseudo, half)
            state["sensitivity"].append(dict(bins=bins, pseudocount=pseudo, half_templates=half,
                fits={name: fit(budget*data[:-1], bank) for name, bank in varied.items()}))
        state["quadrature_checks"] = {}
        for name, bank in banks.items():
            fine = boundary.fit_boundary(budget*empirical[:-1], bank["templates"], bank["efficiency"],
                                          grid_size=401, tail_drop=40., pseudocount=0.)
            original = state["asimov"][name]["bayesian"]
            difference = {key: float(np.max(abs(np.asarray(original[key])-fine[key])))
                          for key in ("median", "lower95", "upper95", "direct_upper95")}
            state["quadrature_checks"][name] = difference
            if max(difference.values()) > 1e-5:
                raise RuntimeError(f"Posterior needs finer integration: {name}: {difference}")
        checkpoint()
        print("Asimov and numerical checks complete", flush=True)
    exact = {name: exact_population(bank["templates"], bank["efficiency"]) for name, bank in banks.items()}
    if "calibration" not in state:
        state["calibration"] = ensemble(banks, exact, CALIBRATION_SEED, calibration_repetitions, budget, workers)
        rank = calibration.tolerance_rank(calibration_repetitions, comparisons=3)
        state["null_calibration"] = dict(tolerance=rank,
            cutoffs={name: float(np.sort(row["q0"])[rank["rank"]-1]) for name, row in state["calibration"].items()},
            scope="Pointwise at f=(0.05,0.95,0), N=100000, conditional on each fixed template bank")
        checkpoint()
    cutoffs = state["null_calibration"]["cutoffs"]
    if "exact_validation" not in state:
        records = ensemble(banks, exact, VALIDATION_SEED, repetitions, budget, workers)
        state["exact_validation"] = dict(records=records, summary=summarize(records, cutoffs))
        checkpoint()
    if "empirical_validation" not in state:
        records = ensemble(banks, {name: empirical for name in MODELS}, EMPIRICAL_SEED,
                           repetitions, budget, workers, paired=True)
        state["empirical_validation"] = dict(records=records, summary=summarize(records, cutoffs))
        coherent = records["coherent"]["q0"]>cutoffs["coherent"]
        paired = {}
        for name in MODELS[1:]:
            difference = (records[name]["q0"]>cutoffs[name]).astype(float)-coherent
            paired[name] = dict(rate_difference=float(difference.mean()),
                paired_MCSE=float(difference.std(ddof=1)/np.sqrt(repetitions)),
                mismatch_only=int(np.count_nonzero(difference == 1)),
                coherent_only=int(np.count_nonzero(difference == -1)))
        state["empirical_validation"]["paired_with_coherent"] = paired
        checkpoint()
    state["completed_at"] = datetime.now().astimezone().isoformat()
    state["checkpoint_sha256"] = calibration.digest({key: value for key, value in state.items()
        if key not in ("checkpoint_sha256", "completed_at")})
    checkpoint()
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.notebook, args.workers)
