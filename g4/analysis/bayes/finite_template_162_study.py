"""Joint finite-template calibration and weak positive injections at 162 keV.

Explicit CLI only. Reuses the five checked ROOT libraries, splitting the
phase-space library into disjoint template/data halves. All output numbers
are local metadata in the existing main notebook, with resumable chunks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime
import errno
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import calibrate_162_weak_branch_intervals as calibration
from analysis.bayes import model_mismatch_162_study as mismatch
from analysis.bayes.finite_template_likelihood import JointMultinomial, accepted_weights
from analysis.bayes.generate_162_model_libraries import NOTEBOOK, ROOT, sha256

VERSION = "2026-10-22-v2-common-reference-finite-template"
CALIBRATION_DIRECT = (0., .001, .005, .01, .02)
VALIDATION_DIRECT = (0., .001, .002, .005, .01, .02)
CALIBRATION_SEED = 1622026102271
VALIDATION_SEED = 1622026102272
INDEPENDENT_BANK_SEED = 1622026102275
PHASE_SPLIT_SEED = 1622026102263
PHASE_RESPONSE_SEED = 1622026102264
METHODS = ("fixed", "joint")
REFERENCE_CUTOFF = 2.705543454095404


def fractions(direct):
    return np.array([.05*(1-direct), .95*(1-direct), direct])


def save_checkpoint(notebook, state):
    """Keep the valid original until a complete temporary notebook verifies.

    VirtualBox shared-folder writes occasionally return EPROTO/EBUSY. Retrying
    an atomic checkpoint changes no sampling streams or numerical results.
    """
    notebook=Path(notebook)
    temporary=notebook.with_suffix('.pending.ipynb')
    for attempt in range(3):
        try:
            document=json.loads(notebook.read_text())
            document['metadata']['codex_finite_template_study']=calibration.plain(state)
            content=json.dumps(document,ensure_ascii=False,indent=1,allow_nan=False)+'\n'
            temporary.write_text(content)
            if temporary.stat().st_size != len(content.encode()):
                raise OSError(errno.EIO,'Incomplete temporary notebook')
            checked=json.loads(temporary.read_text())
            if len(checked['cells']) != len(document['cells']):
                raise OSError(errno.EIO,'Temporary notebook cell mismatch')
            temporary.replace(notebook)
            return
        except OSError as error:
            if error.errno not in (errno.EPROTO,errno.EIO,errno.EBUSY,errno.EAGAIN,errno.ESTALE) or attempt==2:
                raise
            time.sleep(.5*(attempt+1))
        except json.JSONDecodeError:
            if attempt==2:
                raise
            time.sleep(.5*(attempt+1))


def common_reference(data_histograms, data_generated, mc_histograms, mc_generated):
    """Pooled fixed-G MLE from disjoint, same-physics samples, with rejection.

    Used as a declared empirical generating population, not as a statement
    that unobserved bins are physically forbidden.
    """
    data, mc = np.asarray(data_histograms), np.asarray(mc_histograms)
    dg, mg = np.asarray(data_generated), np.asarray(mc_generated)
    if (data.shape != mc.shape or data.ndim != 2 or dg.shape != (len(data),)
            or mg.shape != dg.shape or np.any(data < 0) or np.any(mc < 0)
            or np.any(data.sum(axis=1) > dg) or np.any(mc.sum(axis=1) > mg)):
        raise ValueError("Invalid independent reference counts")
    total = dg+mg
    pooled = data+mc
    return np.column_stack([pooled,total-pooled.sum(axis=1)])/total[:,None]


def prepare_inputs(notebook=NOTEBOOK):
    # Original study remains unchanged. It verifies each input ROOT hash.
    with redirect_stdout(io.StringIO()):
        selected, summaries, _, provenance = mismatch.prepare_libraries(notebook)
    libraries = json.loads(Path(notebook).read_text())["metadata"]["codex_model_mismatch_libraries"]
    path = ROOT/libraries["runs"]["phase_space"]["root"]
    ids = mismatch.scan.load_reaction_event_ids(path)
    split = dict(zip(("template", "data"), mismatch.split_ids(ids, PHASE_SPLIT_SEED)))
    events = mismatch.scan.load_complete_three_alpha(path, load_hit_position=True)
    events = mismatch.scan.apply_mask(events, mismatch.scan.quality_mask(events["kinetic"], events["momentum"], 1e-4))
    phase_summary = {}
    for index, (subset, allocated) in enumerate(split.items()):
        sample = mismatch.scan.apply_mask(events, np.isin(events["event_id"], allocated))
        rec, _ = mismatch.scan.reconstruct_with_granularity(sample, 2., 2., 0., 0., "sphere-center", np.array([0., 0., 170.]))
        rng = np.random.default_rng(np.random.SeedSequence([PHASE_RESPONSE_SEED, index]))
        measured, _, _ = mismatch.scan.apply_closure_energy_response(rec, 100., 100., rng.standard_normal(rec["kinetic"].shape))
        xy = np.column_stack(mismatch.scan.ordered_dalitz_xy(measured))
        half, _ = mismatch.split_ids(allocated, [PHASE_SPLIT_SEED, index, 17])
        selected[("phase_space", subset)] = dict(generated=len(allocated), xy=xy,
            half_xy=xy[np.isin(measured["event_id"], half)], half_generated=len(half))
        phase_summary[subset] = dict(generated=len(allocated), accepted=len(xy),
            ids_sha256=calibration.digest(allocated), efficiency=len(xy)/len(allocated))
    support = mismatch.geometric_support(60)
    hist = np.stack([mismatch.histogram_xy(selected[(name, "data")]["xy"], 60)[support]
                     for name in ("alpha0", "coherent", "phase_space")])
    generated = np.array([selected[(name, "data")]["generated"] for name in ("alpha0", "coherent", "phase_space")])
    independent_truth = np.column_stack([hist, generated-hist.sum(axis=1)])/generated[:, None]
    banks, _ = mismatch.make_banks(selected)
    truth = common_reference(hist, generated, banks['coherent']['raw_templates'], banks['coherent']['generated'])
    provenance.update(phase_split_seed=PHASE_SPLIT_SEED, phase_response_seed=PHASE_RESPONSE_SEED,
        phase_summary=phase_summary, truth_population="Common pooled same-physics reference; fresh calibration/validation streams",
        reference_generated=(generated+banks['coherent']['generated']).tolist())
    return dict(banks=banks, truth=truth, independent_truth=independent_truth, data_histograms=hist, data_generated=generated,
                support=support, provenance=provenance, original_input_summary=summaries), selected


def fixed_model(data, mc, generated):
    efficiency = np.sum(mc, axis=1)/generated
    # Explicit full-support smoothing, as in the previous fixed-bank study.
    return mismatch.boundary._Likelihood(data, np.asarray(mc)+.5, efficiency, 0.)


def evaluate(data, mc, generated, direct, *, upper_cutoffs=None):
    fixed = fixed_model(data, mc, generated)
    joint = JointMultinomial(data, mc, generated,
                             start=accepted_weights(fixed.mode, fixed.efficiencies))
    fixed_q = calibration.profile_statistic(fixed, direct, 2)
    output = dict(fixed=dict(map=fixed.mode, q0=calibration.profile_statistic(fixed, 0., 2),
                            q_upper=0. if fixed.mode[2] > direct else fixed_q),
                  joint=dict(map=joint.mode, q0=joint.q0, q_upper=joint.statistic(direct, upper=True),
                             shape_gap=joint.max_shape_gap, mode_gap=joint.mode_gap,
                             mode_score=joint.mode_score, mode_decrement=joint.mode_decrement,
                             shape_iterations=joint.max_shape_iterations))
    output["joint"]["total_deviance"] = 2*joint.loss
    if upper_cutoffs is not None:
        output["fixed"]["upper"] = calibration.upper_limit(fixed, upper_cutoffs["fixed"])
        output["joint"]["upper"] = joint.upper_limit(upper_cutoffs["joint"])
    return output


def draw(inputs, direct, seed, index, budget, *, fluctuate_mc=True):
    identity = int(round(direct*1e6))
    truth = np.asarray(inputs["truth"])
    data_rng = np.random.default_rng(np.random.SeedSequence([seed, identity, index, 0]))
    data = data_rng.multinomial(budget, fractions(direct) @ truth)[:-1]
    generated = np.asarray(inputs["banks"]["coherent"]["generated"], np.int64)
    mc = (np.array([np.random.default_rng(np.random.SeedSequence([seed, identity, index, 1, c])).multinomial(int(g), p)[:-1]
                   for c, (g, p) in enumerate(zip(generated, truth))]) if fluctuate_mc
          else np.asarray(inputs["banks"]["coherent"]["raw_templates"]))
    return data, mc, generated


_WORKER_INPUTS = None


def _initialize(inputs):
    global _WORKER_INPUTS
    _WORKER_INPUTS = inputs


def _chunk(task):
    direct, seed, first, last, budget, fluctuate_mc = task
    records = {name: [] for name in METHODS}
    for index in range(first, last):
        try:
            data, mc, generated = draw(_WORKER_INPUTS, direct, seed, index, budget, fluctuate_mc=fluctuate_mc)
            result = evaluate(data, mc, generated, direct)
        except Exception as error:
            raise RuntimeError(f"Trial failed: seed={seed}, d={direct}, index={index}") from error
        for name, row in result.items():
            records[name].append(row)
    return calibration.plain(records)


def ensemble(inputs, direct_values, seed, repetitions, budget, workers, *, resume=None, checkpoint=None, fluctuate_mc=True):
    config = dict(direct_values=direct_values, seed=seed, repetitions=repetitions, budget=budget,
                  population_sha256=calibration.digest(inputs["truth"]), chunk_size=25, fluctuate_mc=fluctuate_mc)
    state = resume or dict(configuration=calibration.plain(config), chunks={})
    if state["configuration"] != calibration.plain(config):
        raise ValueError("Ensemble checkpoint configuration changed")
    tasks = {f"{int(round(d*1e6)):05d}:{first}": (d, seed, first, min(first+25, repetitions), budget, fluctuate_mc)
             for d in direct_values for first in range(0, repetitions, 25)}
    for key, item in state["chunks"].items():
        if key not in tasks or set(item) != set(METHODS):
            raise ValueError("Unknown checkpoint chunk")
        expected_count = tasks[key][3]-tasks[key][2]
        if any(len(rows) != expected_count for rows in item.values()):
            raise ValueError("Incomplete checkpoint chunk")
    pending = {key: value for key, value in tasks.items() if key not in state["chunks"]}
    start = time.monotonic()
    last_notice = start
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(inputs,)) as pool:
            jobs = {pool.submit(_chunk, task): key for key, task in pending.items()}
            for future in as_completed(jobs):
                key = jobs[future]
                try:
                    state["chunks"][key] = future.result()
                except Exception:
                    for other in jobs:
                        other.cancel()
                    raise
                completed = sum(len(item["joint"]) for item in state["chunks"].values())
                if checkpoint and (completed % 500 == 0 or len(state["chunks"]) == len(tasks)):
                    checkpoint(state)
                if time.monotonic()-last_notice > 20:
                    print(f"seed {seed}: {completed}/{len(direct_values)*repetitions} joint-MC trials; {time.monotonic()-start:.1f} s this pass", flush=True)
                    last_notice = time.monotonic()
    state["records"] = {}
    for direct in direct_values:
        key = f"{int(round(direct*1e6)):05d}"
        state["records"][key] = {}
        for method in METHODS:
            rows = [row for first in range(0, repetitions, 25)
                    for row in state["chunks"][key+f":{first}"][method]]
            state["records"][key][method] = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    return state


def freeze(calibration_state):
    if calibration_state["configuration"]["seed"] != CALIBRATION_SEED:
        raise ValueError("Only independent calibration samples may set cutoffs")
    repetitions = calibration_state["configuration"]["repetitions"]
    points = calibration_state["records"]
    rank = calibration.tolerance_rank(repetitions, comparisons=2*(len(points)+1))
    cutoffs, local = {}, {}
    for method in METHODS:
        local[method] = {key: float(np.sort(row[method]["q_upper"])[rank["rank"]-1]) for key, row in points.items()}
        cutoffs[method] = dict(q_upper=max(local[method].values()),
            q0=float(np.sort(points["00000"][method]["q0"])[rank["rank"]-1]))
    return dict(cutoffs=cutoffs, local_upper=local, tolerance=rank,
                scope="Finite d grid at r=0.05; includes independently fluctuated fixed-G MC and data")


def summarize(ensemble_state, frozen):
    summaries = {}
    for key, point in ensemble_state["records"].items():
        direct = int(key)/1e6
        summaries[key] = {}
        for method, records in point.items():
            d = np.asarray(records["map"])[:, 2]
            repetitions = len(d)
            summaries[key][method] = dict(
                direct=direct, mean=float(d.mean()), bias=float(d.mean()-direct),
                std=float(d.std(ddof=1)), mcse=float(d.std(ddof=1)/np.sqrt(repetitions)),
                map_quantiles=np.quantile(d, [.025, .5, .975]),
                rejection=mismatch.binomial_summary(np.count_nonzero(records["q0"]>frozen["cutoffs"][method]["q0"]), repetitions),
                reference_rejection=mismatch.binomial_summary(np.count_nonzero(records["q0"]>REFERENCE_CUTOFF), repetitions),
                coverage=mismatch.binomial_summary(np.count_nonzero(records["q_upper"]<=frozen["cutoffs"][method]["q_upper"]), repetitions),
                reference_coverage=mismatch.binomial_summary(np.count_nonzero(records["q_upper"]<=REFERENCE_CUTOFF), repetitions))
    return summaries


def expected_comparisons(inputs, budget):
    output = {}
    for direct in VALIDATION_DIRECT:
        data = budget*(fractions(direct) @ np.asarray(inputs["independent_truth"]))[:-1]
        output[f"{int(round(direct*1e6)):05d}"] = {}
        for name, bank in inputs["banks"].items():
            result = evaluate(data, np.asarray(bank["raw_templates"]), np.asarray(bank["generated"]), direct)
            output[f"{int(round(direct*1e6)):05d}"][name] = result
        print(f"Expected-count cross-fits: d={direct:g}", flush=True)
    return output


def plots(state):
    """Cached-number figures only. No fitting, simulation, or tables."""
    import matplotlib.pyplot as plt
    colors = dict(fixed="C1", joint="C0")
    labels = dict(fixed="Fixed smoothed templates", joint="Joint multinomial MC")
    figure1, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for method in METHODS:
        rows = state["summary"]
        x = np.array([int(key)/1e4 for key in sorted(rows)])
        for ax, item, title in ((axes[0], "coverage", "One-sided upper coverage"),
                                (axes[1], "rejection", "Rejection of d = 0")):
            values = [rows[key][method][item] for key in sorted(rows)]
            p = np.array([row["rate"] for row in values])*100
            bounds = np.array([row["wilson95"] for row in values])*100
            errors = np.maximum(np.vstack([p-bounds[:,0], bounds[:,1]-p]),0.)
            ax.errorbar(x, p, yerr=errors, marker="o", color=colors[method], label=labels[method], capsize=3)
            ax.set(xlabel="Injected generator d (%)", ylabel="Rate (%)", title=title, ylim=(-2,103))
    axes[0].axhline(95, color=".5", ls="--")
    axes[1].axhline(5, color=".5", ls="--")
    axes[0].legend(); axes[1].legend()
    lowest = min(row[method]["coverage"]["wilson95"][0] for row in state["summary"].values() for method in METHODS)*100
    axes[0].set_ylim(max(0., min(90., lowest-3)), 101)
    figure2, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for method in METHODS:
        rows = state["summary"]
        x = np.array([int(key)/1e4 for key in sorted(rows)])
        reference = 100*np.array([state["independent_bank_summary"][key][method]["coverage"]["rate"] for key in sorted(rows)])
        calibrated = 100*np.array([rows[key][method]["coverage"]["rate"] for key in sorted(rows)])
        axes[0].plot(x, reference, "o--", color=colors[method], label=labels[method]+"; actual fixed MC bank")
        axes[0].plot(x, calibrated, "o-", color=colors[method], label=labels[method]+"; regenerated MC")
        records = state["validation"]["records"]["00000"][method]
        values = 100*np.sort(np.asarray(records["map"])[:,2])
        axes[1].step(values, np.arange(1,len(values)+1)/len(values), where="post", color=colors[method], label=labels[method])
    axes[0].set(xlabel="Injected generator d (%)", ylabel="Upper coverage (%)", ylim=(0,101), title="Same frozen cutoffs; different MC ensembles")
    axes[0].axhline(95,color=".5",ls="--"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Fitted d at true zero (%)", ylabel="Empirical CDF", title="Data and pure MC regenerated independently")
    axes[1].legend()
    figure3, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    xs = 100*np.array(VALIDATION_DIRECT)
    for name in mismatch.MODELS:
        for method, style in (("joint", "-"), ("fixed", "--")):
            ys = [100*state["expected"][f"{int(round(d*1e6)):05d}"][name][method]["map"][2] for d in VALIDATION_DIRECT]
            axes[0].plot(xs, ys, marker="o", ls=style, color=mismatch.COLORS[name], label=mismatch.LABELS[name]+("; joint MC" if method=="joint" else "; fixed"))
    axes[0].plot(xs,xs,"k:",label="Truth")
    axes[0].set(xlabel="Injected generator d (%)", ylabel="Expected-count fitted d (%)", title="Same coherent data, different physical templates")
    axes[0].legend(fontsize=8)
    curves = state["zero_profiles"]
    for method in METHODS:
        axes[1].plot(100*np.asarray(curves["direct"]), curves[method], color=colors[method], label=labels[method])
        axes[1].axhline(state["frozen"]["cutoffs"][method]["q_upper"], color=colors[method], ls="--", alpha=.7)
    axes[1].set(xlabel="Tested generator d (%)", ylabel="Profile likelihood ratio", title="Coherent independent-bank zero-data profile", ylim=(0,12))
    axes[1].legend()
    return [figure1, figure2, figure3]


def run(notebook=NOTEBOOK, *, workers=4, calibration_repetitions=1000, validation_repetitions=500):
    if not 1 <= workers <= 8:
        raise ValueError("Use 1..8 workers")
    notebook = Path(notebook)
    previous = json.loads(notebook.read_text())["metadata"].get("codex_finite_template_study")
    sources = [Path(__file__), ROOT/"g4/analysis/bayes/finite_template_likelihood.py",
               ROOT/"g4/analysis/bayes/model_mismatch_162_study.py", ROOT/"g4/analysis/bayes/zero_branch_162_study.py",
               ROOT/"g4/analysis/bayes/calibrate_162_weak_branch_intervals.py"]
    configuration = dict(version=VERSION, source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sources},
        calibration_repetitions=calibration_repetitions, validation_repetitions=validation_repetitions,
        budget=100000, bins=60, shape_gap_tolerance=1e-4,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    if previous and previous["configuration"] != configuration:
        raise ValueError("Study configuration changed; preserve the old checkpoint before rerunning")
    if previous:
        state = previous
        inputs = state["inputs"]
    else:
        inputs, selected = prepare_inputs(notebook)
        state = dict(configuration=configuration, inputs=inputs, started_at=datetime.now().astimezone().isoformat())
    def checkpoint():
        save_checkpoint(notebook, state)
    if "expected" not in state:
        state["expected"] = expected_comparisons(inputs, configuration["budget"])
        checkpoint()
    if "calibration" not in state or "records" not in state["calibration"]:
        def save_cal(item):
            state["calibration"] = item
            checkpoint()
        state["calibration"] = ensemble(inputs, CALIBRATION_DIRECT, CALIBRATION_SEED,
            calibration_repetitions, configuration["budget"], workers, resume=state.get("calibration"), checkpoint=save_cal)
        state["frozen"] = freeze(state["calibration"])
        checkpoint()
    if "validation" not in state or "records" not in state["validation"]:
        def save_val(item):
            state["validation"] = item
            checkpoint()
        state["validation"] = ensemble(inputs, VALIDATION_DIRECT, VALIDATION_SEED,
            validation_repetitions, configuration["budget"], workers, resume=state.get("validation"), checkpoint=save_val)
        checkpoint()
    state["summary"] = summarize(state["validation"], state["frozen"])
    if "independent_bank_validation" not in state or "records" not in state["independent_bank_validation"]:
        def save_independent(item):
            state["independent_bank_validation"] = item
            checkpoint()
        state["independent_bank_validation"] = ensemble(inputs, VALIDATION_DIRECT, INDEPENDENT_BANK_SEED,
            validation_repetitions, configuration["budget"], workers,
            resume=state.get("independent_bank_validation"), checkpoint=save_independent, fluctuate_mc=False)
        checkpoint()
    state["independent_bank_summary"] = summarize(state["independent_bank_validation"], state["frozen"])
    if "zero_profiles" not in state:
        bank = inputs["banks"]["coherent"]
        data = configuration["budget"]*(fractions(0.) @ np.asarray(inputs["independent_truth"]))[:-1]
        fixed = fixed_model(data, np.asarray(bank["raw_templates"]), np.asarray(bank["generated"]))
        joint = JointMultinomial(data, bank["raw_templates"], bank["generated"],
                                 start=accepted_weights(fixed.mode, fixed.efficiencies))
        direct = np.linspace(0., .01, 41)
        state["zero_profiles"] = dict(direct=direct,
            fixed=[calibration.profile_statistic(fixed,d,2) for d in direct],
            joint=[joint.statistic(d) for d in direct],
            fixed_upper=calibration.upper_limit(fixed,state["frozen"]["cutoffs"]["fixed"]["q_upper"]),
            joint_upper=joint.upper_limit(state["frozen"]["cutoffs"]["joint"]["q_upper"]))
    state["completed_at"] = datetime.now().astimezone().isoformat()
    state["checkpoint_sha256"] = calibration.digest({k:v for k,v in state.items() if k not in ("completed_at","checkpoint_sha256")})
    checkpoint()
    print("Finite-template study complete", state["completed_at"], flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.notebook, workers=args.workers)
