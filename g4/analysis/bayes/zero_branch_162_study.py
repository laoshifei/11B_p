"""Boundary-aware, matched-response study of the 162-keV 5/95/0 mixture.

The likelihood conditions on accepted Dalitz counts. A flat prior in the
generator-fraction simplex is integrated numerically, including direct=0.
The reported direct limit is the marginal 95th percentile, not the 97.5th
percentile of a central interval. Templates remain fixed and conditional.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import brentq

from . import bootstrap_162_response_uncertainty as bootstrap
from . import fixed_generated_162_threshold_study as budget_study

VERSION = "2026-10-19-v1-zero-branch-boundary-integration"
TRUE_FRACTIONS = np.array([.05, .95, 0.])
RESPONSE = (2., 2., 100., 100.)


def _mix_mode(p0, p1, counts):
    """Exact maximum of a two-component multinomial mixture on [0, 1]."""
    difference = p1 - p0

    def score(weight):
        q = (1 - weight) * p0 + weight * p1
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.divide(difference, q, out=np.zeros_like(q), where=q > 0)
        if np.any((q == 0) & (difference > 0)):
            return np.inf
        if np.any((q == 0) & (difference < 0)):
            return -np.inf
        return float(counts @ terms)

    if score(0.) <= 0:
        return 0.
    if score(1.) >= 0:
        return 1.
    return brentq(score, 0., 1., xtol=2e-14, rtol=1e-13)


class _Likelihood:
    def __init__(self, data, templates, efficiencies, pseudocount):
        data, templates = np.asarray(data, float), np.asarray(templates, float)
        self.efficiencies = np.asarray(efficiencies, float)
        if (templates.shape != (3,) + data.shape or self.efficiencies.shape != (3,)
                or not np.all(np.isfinite(data)) or np.any(data < 0) or data.sum() <= 0
                or not np.all(np.isfinite(templates)) or np.any(templates < 0)
                or not np.all(np.isfinite(self.efficiencies))
                or np.any(self.efficiencies <= 0) or np.any(self.efficiencies > 1)
                or not np.isfinite(pseudocount) or pseudocount < 0):
            raise ValueError("Invalid data, templates, efficiencies, or pseudocount")
        active = np.any(templates > 0, axis=0) | (data > 0)
        probabilities = templates[:, active] + pseudocount
        if np.any(probabilities.sum(axis=1) <= 0):
            raise ValueError("Each channel needs a nonempty template")
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        positive = data[active] > 0
        self.p = probabilities[:, positive]
        self.counts = data[active][positive]
        self.relative = self.counts / self.counts.sum()
        if np.any(self.p.sum(axis=0) == 0):
            raise ValueError("Observed bins have zero probability under every channel")
        self.active_bins = int(active.sum())
        self.mode, self.qmode = self._mode()

    def probability(self, fractions):
        weighted = np.asarray(fractions) * self.efficiencies
        return (weighted @ self.p) / weighted.sum(axis=-1, keepdims=True)

    def _mode(self):
        # In accepted weights the log likelihood is concave. Profile its
        # direct weight and find the zero of the monotone envelope score.
        def profiled(v):
            p0 = (1-v)*self.p[1] + v*self.p[2]
            p1 = (1-v)*self.p[0] + v*self.p[2]
            x = _mix_mode(p0, p1, self.relative)
            sequential = x*self.p[0] + (1-x)*self.p[1]
            q = (1-v)*sequential + v*self.p[2]
            score = (np.inf if np.any(q == 0) else
                     float(self.relative @ ((self.p[2]-sequential)/q)))
            return x, q, score

        if profiled(0.)[2] <= 0:
            v = 0.
        else:
            if profiled(1-1e-12)[2] >= 0:
                raise ValueError("Pure-direct vertex is outside this sequential-mixture study")
            v = brentq(lambda d: profiled(d)[2], 0., 1-1e-12,
                       xtol=2e-14, rtol=1e-13)
        x, q, _ = profiled(v)
        accepted = np.array([(1-v)*x, (1-v)*(1-x), v])
        fractions = accepted / self.efficiencies
        fractions /= fractions.sum()
        gradient = self.p @ (self.relative / q)
        gap = float(gradient.max() - accepted @ gradient)
        if gap > 1e-8:
            raise RuntimeError(f"Constrained mode failed its optimality check: {gap:g}")
        return fractions, q

    def log_relative(self, fractions):
        with np.errstate(divide="ignore"):
            return np.log(self.probability(fractions)/self.qmode) @ self.counts

    def profile(self, value, axis):
        # At fixed d or fixed r=f0/(1-d), the remaining parameter is another
        # two-endpoint generator mixture, with an efficiency-weighted shape.
        if axis == "d":
            endpoints = np.array([[0., 1-value, value], [1-value, 0., value]])
        else:
            endpoints = np.array([[value, 1-value, 0.], [0., 0., 1.]])
        efficiencies = endpoints @ self.efficiencies
        shapes = self.probability(endpoints)
        accepted = _mix_mode(shapes[0], shapes[1], self.relative)
        generated = (accepted/efficiencies[1]) / (
            (1-accepted)/efficiencies[0] + accepted/efficiencies[1])
        f = (1-generated)*endpoints[0] + generated*endpoints[1]
        return float(self.log_relative(f))

    def bounds(self, axis, centre, drop):
        def shifted(x):
            return self.profile(x, axis) + drop
        low = (0. if centre == 0 or shifted(0.) >= 0 else
               brentq(shifted, 0., centre, xtol=1e-14))
        high = (1. if centre == 1 or shifted(1.) >= 0 else
                brentq(shifted, centre, 1., xtol=1e-14))
        return low, high


def _integral_at(grid, density, cumulative, values):
    """Integrate the piecewise-linear density to one location per column."""
    values = np.broadcast_to(values, density.shape[1:])
    indices = np.clip(np.searchsorted(grid, values, side="right")-1, 0, len(grid)-2)
    columns = np.arange(density.shape[1])
    dx = np.clip(values-grid[indices], 0., np.diff(grid)[indices])
    first, last = density[indices, columns], density[indices+1, columns]
    result = cumulative[indices, columns] + dx*(first + .5*dx*(last-first)/np.diff(grid)[indices])
    return np.where(values <= grid[0], 0., np.where(values >= grid[-1], cumulative[-1], result))


def fit_boundary(data, templates, efficiencies, *, grid_size=201, tail_drop=32.,
                 pseudocount=.5, return_density=False):
    """Integrate the flat-generator-simplex posterior through the d=0 edge.

    Integrate in r=f0/(1-d), d=f_direct, retaining the Jacobian (1-d).
    Trapezoidal endpoint weights give the physical boundary its correct
    integration weight. Profile bounds and grid refinement are checked
    separately; the posterior is not replaced by a truncated Gaussian.
    """
    if int(grid_size) != grid_size or grid_size < 31 or tail_drop < 20:
        raise ValueError("Need grid_size >= 31 and tail_drop >= 20")
    model = _Likelihood(data, templates, efficiencies, pseudocount)
    dmode = model.mode[2]
    rmode = model.mode[0] / (1-dmode)
    r = np.linspace(*model.bounds("r", rmode, tail_drop), grid_size)
    d = np.linspace(*model.bounds("d", dmode, tail_drop), grid_size)
    rr, dd = np.meshgrid(r, d, indexing="ij")
    fractions = np.stack((rr*(1-dd), (1-rr)*(1-dd), dd), axis=-1).reshape(-1, 3)
    log_density = np.empty(len(fractions))
    for start in range(0, len(fractions), 512):
        log_density[start:start+512] = model.log_relative(fractions[start:start+512])
    density = np.exp(log_density.reshape(grid_size, grid_size)) * (1-dd)
    mass = trapezoid(trapezoid(density, r, axis=0), d)
    if not np.isfinite(mass) or mass <= 0:
        raise RuntimeError("Invalid integrated posterior mass")
    density /= mass
    cumulative_r = cumulative_trapezoid(density, r, axis=0, initial=0.)
    marginal_d = cumulative_r[-1]
    cumulative_d = cumulative_trapezoid(marginal_d, d, initial=0.)

    def cdf(value, channel):
        if channel == 2:
            return float(_integral_at(d, marginal_d[:, None], cumulative_d[:, None], value)[0])
        limit = value / np.maximum(1-d, 1e-300)
        if channel == 0:
            integral = _integral_at(r, density, cumulative_r, limit)
        else:
            integral = cumulative_r[-1] - _integral_at(r, density, cumulative_r, 1-limit)
        return float(trapezoid(integral, d))

    quantiles = np.array([[brentq(lambda x: cdf(x,c)-q, 0., 1., xtol=2e-13)
                           for q in (.025, .5, .975)] for c in range(3)])
    upper = brentq(lambda x: cdf(x,2)-.95, 0., 1., xtol=2e-13)
    null_log_relative = model.profile(0., "d")
    result = dict(median=quantiles[:,1], lower95=quantiles[:,0], upper95=quantiles[:,2],
                  direct_upper95=float(upper), map=model.mode,
                  q0=max(0., -2*null_log_relative),
                  at_direct_boundary=bool(model.mode[2] == 0.),
                  grid_size=grid_size, tail_drop=tail_drop,
                  bounds=np.array([[r[0],r[-1]], [d[0],d[-1]]]),
                  active_bins=model.active_bins)
    if return_density:
        result.update(r_grid=r, direct_grid=d, direct_density=marginal_d)
    return result


def nominal_pools(pools):
    """Extract the common 100-keV response from the completed threshold study."""
    settings = pools["settings"]
    if (not np.array_equal(settings["response"], RESPONSE[:3])
            or settings["energy_seed"] != 1620 or settings["bins"] != 60):
        raise ValueError("Pools have different response, seeds, or binning")
    thresholds = np.asarray(settings["thresholds_keV"])
    matches = np.flatnonzero(thresholds == RESPONSE[3])
    if len(matches) != 1:
        raise ValueError("The nominal threshold is absent or duplicated")
    index = matches[0]
    return dict(templates=np.asarray(pools["template_histograms"])[index],
                mixed=np.asarray(pools["mixed_histograms"])[index],
                pure_generated=np.asarray(pools["pure_generated"]),
                mixed_generated=np.asarray(pools["mixed_generated"]))


def generated_probability(pools, fractions=TRUE_FRACTIONS):
    mixed = np.asarray(pools["mixed"])
    return budget_study.generated_probability(
        mixed.reshape(1, 3, 1, -1), pools["mixed_generated"], fractions)


def validate_numerics(pools, *, budget=100000, seed=16220261020, tolerance=1e-5):
    """Check real null samples, small-injection Asimov cases, and truncation.

    Injection cases here only validate quadrature; they are not a detection
    sensitivity study. The production ensemble remains exactly 5/95/0.
    """
    probability = generated_probability(pools)
    shape = pools["mixed"].shape[1:]
    rng = np.random.default_rng(seed)
    samples = [("null_asimov", budget*probability[:-1].reshape(shape))]
    for i in range(6):
        samples.append((f"null_trial_{i+1}", rng.multinomial(budget, probability)[:-1].reshape(shape)))
    for direct in (.005, .02):
        fractions = np.array([.05*(1-direct), .95*(1-direct), direct])
        p = generated_probability(pools, fractions)
        samples.append((f"injected_asimov_{direct:g}", budget*p[:-1].reshape(shape)))
    efficiency = bootstrap.scan.response_efficiencies(pools["templates"], pools["pure_generated"])
    comparisons = []
    for label, data in samples:
        a = fit_boundary(data, pools["templates"], efficiency, grid_size=201)
        b = fit_boundary(data, pools["templates"], efficiency, grid_size=401)
        c = fit_boundary(data, pools["templates"], efficiency, grid_size=401, tail_drop=40.)
        def differences(left, right):
            return {key: float(np.max(np.abs(left[key]-right[key])))
                    for key in ("median", "lower95", "upper95", "direct_upper95")}
        comparisons.append(dict(case=label, grid=differences(a,b), tail=differences(b,c),
                                direct_map=a["map"][2], direct_upper95=b["direct_upper95"]))
        print(f"Boundary integration check: {label}", flush=True)
    grid_error = max(max(row["grid"].values()) for row in comparisons)
    tail_error = max(max(row["tail"].values()) for row in comparisons)
    if max(grid_error, tail_error) > tolerance:
        raise RuntimeError(f"Boundary integration needs refinement: grid={grid_error:g}, tail={tail_error:g}")
    return dict(cases=comparisons, max_grid_difference=grid_error,
                max_tail_difference=tail_error, tolerance=tolerance,
                selected_grid=201, tail_drop=32., seed=seed)


def run_trials(pools, *, repetitions=500, budget=100000, seed=16220261019,
               grid_size=201, progress_every=25, checkpoint=None, resume=None):
    """Fixed-generated, data-only null trials with deterministic resumption."""
    if repetitions < 2 or budget < 1 or int(budget) != budget:
        raise ValueError("Need at least two trials and an integer positive budget")
    probability = generated_probability(pools)
    shape = pools["mixed"].shape[1:]
    efficiency = bootstrap.scan.response_efficiencies(pools["templates"], pools["pure_generated"])
    digest = hashlib.sha256()
    for key in ("templates", "mixed", "pure_generated", "mixed_generated"):
        digest.update(np.asarray(pools[key], np.int64).tobytes())
    configuration = dict(version=VERSION, budget=budget, seed=seed, grid_size=grid_size,
                         population_sha256=digest.hexdigest(), fractions=TRUE_FRACTIONS.tolist())
    records = {key: np.empty((repetitions,3)) for key in ("median","lower95","upper95","map")}
    records.update({key: np.empty(repetitions) for key in ("direct_upper95","q0","accepted")})
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
    for i in range(repetitions):
        data = rng.multinomial(budget, probability)[:-1].reshape(shape)
        if i < completed:
            if data.sum() != records["accepted"][i]:
                raise ValueError("Checkpoint random sequence changed")
            continue
        result = fit_boundary(data, pools["templates"], efficiency, grid_size=grid_size)
        for key in records:
            records[key][i] = data.sum() if key == "accepted" else result[key]
        if (i+1) % progress_every == 0 or i+1 == repetitions:
            seconds = previous_seconds + time.monotonic()-started
            print(f"Zero-direct trials {i+1}/{repetitions}: {seconds:.1f} s", flush=True)
            if checkpoint is not None:
                checkpoint(dict(configuration=configuration, completed=i+1, seconds=seconds,
                                records={k:v[:i+1].copy() for k,v in records.items()}))
    median, low, high = (records[k] for k in ("median","lower95","upper95"))
    hits = (low[:,:2] <= TRUE_FRACTIONS[:2]) & (TRUE_FRACTIONS[:2] <= high[:,:2])
    upper = records["direct_upper95"]
    boundary = records["map"][:,2] == 0
    summary = dict(mean=median.mean(axis=0), bias=median.mean(axis=0)-TRUE_FRACTIONS,
                   sd=median.std(axis=0,ddof=1), bias_mcse=median.std(axis=0,ddof=1)/np.sqrt(repetitions),
                   mean_width=(high-low).mean(axis=0), sequential_inclusion=hits.mean(axis=0),
                   sequential_hits=hits.sum(axis=0),
                   sequential_wilson95=bootstrap.wilson_interval(hits.sum(axis=0),repetitions),
                   upper95_mean=float(upper.mean()), upper95_sd=float(upper.std(ddof=1)),
                   upper95_mean_mcse=float(upper.std(ddof=1)/np.sqrt(repetitions)),
                   upper95_quantiles=np.quantile(upper,[.025,.16,.5,.84,.975]),
                   map_boundary_count=int(boundary.sum()), map_boundary_fraction=float(boundary.mean()),
                   map_boundary_wilson95=bootstrap.wilson_interval(boundary.sum(),repetitions),
                   accepted_mean=float(records["accepted"].mean()),
                   accepted_sd=float(records["accepted"].std(ddof=1)),
                   expected_accepted=float(budget*(1-probability[-1])),
                   q0_quantiles=np.quantile(records["q0"],[.5,.9,.95,.99]))
    return dict(configuration=configuration, records=records, summary=summary,
                repetitions=repetitions, seconds=previous_seconds+time.monotonic()-started,
                interval_scope="Flat generator-simplex prior; conditional on fixed finite templates; no discovery claim")


def plot_summary(trials, asimov):
    """Return editable figures without saving files or selecting a backend."""
    import matplotlib.pyplot as plt
    r, s = trials["records"], trials["summary"]
    figures = []
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].hist((r["median"][:,0]-.05)*100,bins=25,color="#0072B2",alpha=.8)
    axes[0].axvline(0,color="black",ls="--",label="Prescribed truth")
    axes[0].axvline(s["bias"][0]*100,color="#D55E00",label="Mean residual")
    axes[0].set(xlabel=r"$\alpha_0$ posterior median minus 5% [pp]",ylabel="Trials",
                title="Weak sequential branch")
    axes[0].legend(fontsize=9)
    axes[1].hist(r["direct_upper95"]*100,bins=25,color="#009E73",alpha=.8)
    axes[1].axvline(s["upper95_quantiles"][2]*100,color="black",ls="--",label="Median upper limit")
    axes[1].set(xlabel="Direct one-sided 95% upper limit [%]",ylabel="Trials",
                title="True direct fraction = 0")
    axes[1].legend(fontsize=9)
    fig.suptitle("Matched response; 100,000 generated reactions; fixed templates")
    figures.append(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    axes[0].plot(asimov["direct_grid"]*100,asimov["direct_density"]/100,color="#009E73")
    axes[0].axvline(asimov["direct_upper95"]*100,color="#D55E00",ls="--",label="One-sided 95% limit")
    axes[0].set(xlabel="Direct generator fraction [%]",ylabel="Posterior density per %",
                title="Null expected-count posterior")
    axes[0].legend(fontsize=9)
    values=np.sort(r["direct_upper95"]*100)
    axes[1].step(values,np.arange(1,len(values)+1)/len(values),where="post",color="#0072B2")
    axes[1].set(xlabel="Direct one-sided 95% upper limit [%]",ylabel="Fraction of trials",
                title="Upper-limit distribution",ylim=(0,1))
    for ax in axes: ax.grid(alpha=.2)
    figures.append(fig)
    return figures


def validate_trial_refinement(pools, trials, tolerance=1e-5):
    """Refine observed extremes and representative trials after production."""
    record = trials["records"]
    upper = np.asarray(record["direct_upper95"])
    order = np.argsort(upper)
    indices = {int(order[0]),int(order[-1]),int(order[len(order)//2]),
               int(np.argmin(record["median"][:,0])),int(np.argmax(record["median"][:,0])),
               int(np.argmax(record["q0"]))}
    boundary = np.flatnonzero(record["map"][:,2] == 0)
    if boundary.size: indices.add(int(boundary[0]))
    configuration = trials["configuration"]
    probability = generated_probability(pools)
    efficiency = bootstrap.scan.response_efficiencies(pools["templates"],pools["pure_generated"])
    rng = np.random.default_rng(configuration["seed"])
    rows=[]
    for i in range(trials["repetitions"]):
        data=rng.multinomial(configuration["budget"],probability)[:-1].reshape(pools["mixed"].shape[1:])
        if i not in indices: continue
        refined=fit_boundary(data,pools["templates"],efficiency,grid_size=401,tail_drop=40.)
        differences={key:float(np.max(np.abs(record[key][i]-refined[key])))
                     for key in ("median","lower95","upper95","direct_upper95")}
        rows.append(dict(trial=i+1,differences=differences))
    maximum=max(max(row["differences"].values()) for row in rows)
    if maximum > tolerance:
        raise RuntimeError(f"Production-trial refinement failed: {maximum:g}")
    return dict(cases=rows,max_difference=maximum,tolerance=tolerance,
                refined_grid=401,refined_tail_drop=40.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("seq_be_gs","seq_be2plus","direct","mixed"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--repetitions",type=int,default=500)
    args=parser.parse_args()
    pools=nominal_pools(budget_study.build_pools([getattr(args,n) for n in
                       ("seq_be_gs","seq_be2plus","direct","mixed")]))
    validation=validate_numerics(pools)
    result=run_trials(pools,repetitions=args.repetitions,grid_size=validation["selected_grid"])
    print("Direct 95% limit quantiles:",result["summary"]["upper95_quantiles"])


if __name__ == "__main__":
    main()
