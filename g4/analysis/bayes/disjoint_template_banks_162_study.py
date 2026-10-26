"""Four disjoint fixed-size MC banks against one held-out 162-keV reference.

Explicit CLI only. Replays verified response streams; no Geant4 simulation.
Fresh calibration for the smaller MC budget is separate from validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import tempfile
import time

import numpy as np
import uproot

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import template_convergence_162_study as convergence
from analysis.bayes.finite_template_likelihood import JointMultinomial as OriginalJointMultinomial, accepted_weights
from analysis.bayes.generate_162_convergence_libraries import save_metadata

ROOT, NOTEBOOK = convergence.ROOT, convergence.NOTEBOOK
KEY = "codex_disjoint_template_banks_study"
VERSION = "2026-10-22-v2-four-banks-certified-shape-restarts"
BANKS = ("1", "2", "3", "4")
GENERATED = (50000, 250000, 50000)
CALIBRATION_DIRECT = (0., .005, .02)
VALIDATION_DIRECT = (0., .002, .005, .02)
SEEDS = {"heldout": dict(calibration=1622026102291, validation=1622026102292),
         "pooled": dict(calibration=1622026102294, validation=1622026102295)}
CONDITIONAL_SEED = 1622026102293


class JointMultinomial(OriginalJointMultinomial):
    """Same likelihood and gap tolerance; restart a stalled shape iteration.

    Initial probabilities are numerical starting values, never extra MC
    observations. Every accepted restart satisfies the original concavity
    gap bound. Other numerical failures still stop the study.
    """

    def __init__(self, *args, **kwargs):
        self.shape_restart_history = []
        super().__init__(*args, **kwargs)

    def shape(self, weights):
        try:
            return super().shape(weights)
        except RuntimeError as error:
            if not str(error).startswith("Shape profiling did not converge"):
                raise
            row = dict(weights=np.asarray(weights).tolist(), original_failure=str(error), attempts=[])
            for blend in (0., .001, .01, .1, .5):
                self._warm = (1-blend)*self.reference_shape+blend*self.relative[None, :]
                try:
                    result = super().shape(weights)
                except RuntimeError as attempt:
                    if not str(attempt).startswith("Shape profiling did not converge"):
                        raise
                    row["attempts"].append(dict(blend=blend,failure=str(attempt)))
                    continue
                row["attempts"].append(dict(blend=blend,converged_gap=result["gap"],iterations=result["iterations"]))
                self.shape_restart_history.append(row)
                return result
            raise RuntimeError(f"All certified shape restarts failed: {row}") from error


def evaluate(data, mc, generated, direct):
    finite = convergence.finite
    fixed = finite.fixed_model(data, mc, generated)
    joint = JointMultinomial(data, mc, generated,start=accepted_weights(fixed.mode,fixed.efficiencies))
    q_fixed = finite.calibration.profile_statistic(fixed,direct,2)
    q_joint = joint.statistic(direct,upper=True)
    return dict(fixed=dict(map=fixed.mode,q0=finite.calibration.profile_statistic(fixed,0.,2),
        q_upper=0. if fixed.mode[2]>direct else q_fixed),
        joint=dict(map=joint.mode,q0=joint.q0,q_upper=q_joint,shape_gap=joint.max_shape_gap,
            mode_gap=joint.mode_gap,mode_score=joint.mode_score,mode_decrement=joint.mode_decrement,
            shape_iterations=joint.max_shape_iterations,total_deviance=2*joint.loss,
            shape_restart_count=len(joint.shape_restart_history),
            shape_restart_details=json.dumps(joint.shape_restart_history) if joint.shape_restart_history else ""))


_WORKER_INPUTS = None


def _initialize(inputs):
    global _WORKER_INPUTS
    _WORKER_INPUTS = inputs


def _chunk(task):
    direct,seed,first,last,fluctuate_mc = task
    output = {method:[] for method in convergence.finite.METHODS}
    for index in range(first,last):
        try:
            data,mc,generated = convergence.finite.draw(_WORKER_INPUTS,direct,seed,index,100000,fluctuate_mc=fluctuate_mc)
            result = evaluate(data,mc,generated,direct)
        except Exception as error:
            raise RuntimeError(f"Trial failed without omission: seed={seed}, d={direct}, index={index}") from error
        for method in output:
            output[method].append(result[method])
    return convergence.finite.calibration.plain(output)


def ensemble_checkpoint(inputs,path,direct_values,seed,repetitions,*,workers=4,fluctuate_mc=True):
    """Deterministic original draws, with the certified-restart evaluator."""
    plain = convergence.finite.calibration.plain
    path = Path(path)
    config = plain(dict(input_sha256=convergence.finite.calibration.digest(inputs),version=VERSION,
        direct_values=direct_values,seed=seed,repetitions=repetitions,budget=100000,
        fluctuate_mc=fluctuate_mc,chunk_size=25,
        population_sha256=convergence.finite.calibration.digest(inputs["truth"])))
    state = json.loads(path.read_text()) if path.exists() else dict(configuration=config,chunks={})
    if state["configuration"] != config:
        raise ValueError("Certified-restart ensemble checkpoint changed")
    tasks = {f"{int(round(d*1e6)):05d}:{first}":(d,seed,first,min(first+25,repetitions),fluctuate_mc)
             for d in direct_values for first in range(0,repetitions,25)}
    for key,chunk in state["chunks"].items():
        if key not in tasks or any(len(chunk[m])!=tasks[key][3]-tasks[key][2] for m in convergence.finite.METHODS):
            raise ValueError("Invalid trial chunk checkpoint")
    def save():
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(plain(state),allow_nan=False))
        temporary.replace(path)
    start,last_notice = time.monotonic(),time.monotonic()
    pending = {key:value for key,value in tasks.items() if key not in state["chunks"]}
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize,initargs=(inputs,)) as pool:
            futures = {pool.submit(_chunk,value):key for key,value in pending.items()}
            for future in as_completed(futures):
                try:
                    state["chunks"][futures[future]] = future.result()
                except Exception:
                    for other in futures:other.cancel()
                    raise
                count = sum(len(c["joint"]) for c in state["chunks"].values())
                if count%500==0 or len(state["chunks"])==len(tasks):save()
                if time.monotonic()-last_notice>20:
                    print(f"seed {seed}: {count}/{len(direct_values)*repetitions} certified trials; {time.monotonic()-start:.1f}s",flush=True)
                    last_notice=time.monotonic()
    state["records"] = {}
    for direct in direct_values:
        key = f"{int(round(direct*1e6)):05d}"
        state["records"][key] = {}
        for method in convergence.finite.METHODS:
            rows=[row for first in range(0,repetitions,25) for row in state["chunks"][f"{key}:{first}"][method]]
            state["records"][key][method]={name:np.asarray([r[name] for r in rows]) for name in rows[0]}
    save()
    return state


def split_template(ids):
    """Split the existing random order before any selection; no resampling."""
    ids = np.asarray(ids, np.int64)
    if ids.ndim != 1 or not len(ids) or len(ids) % 4 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need unique generated IDs divisible into four equal banks")
    return np.split(ids, 4)


def reaction_population(histograms, generated):
    histograms, generated = np.asarray(histograms), np.asarray(generated)
    if (histograms.ndim != 2 or generated.shape != (len(histograms),)
            or np.any(generated <= 0) or np.any(histograms < 0)
            or np.any(histograms.sum(axis=1) > generated)):
        raise ValueError("Invalid reaction budget or histogram")
    return np.column_stack([histograms, generated-histograms.sum(axis=1)])/generated[:, None]


def reduce_channel(path, channel, base, scratch):
    """Replay only the original template-half response; partition by reaction."""
    scan, mismatch = convergence.mismatch.scan, convergence.mismatch
    ids = scan.load_reaction_event_ids(path)
    template, heldout = convergence.partitions(ids, channel)
    previous = base["inputs"]["summaries"][convergence.CHANNELS[channel]]
    digest = convergence.finite.calibration.digest
    if digest(template) != previous["template_ids_sha256"] or digest(heldout) != previous["data_ids_sha256"]:
        raise ValueError("Original independent partition changed")
    groups = split_template(template)
    if len(groups[0]) != GENERATED[channel] or np.intersect1d(template, heldout).size:
        raise ValueError("Incorrect generated bank size or reference overlap")
    size = int(np.max(ids))+1
    bank_id = np.full(size, -1, np.int8)
    for number, group in enumerate(groups):
        bank_id[group] = number
    support = mismatch.geometric_support(60)
    hist = np.zeros((4, int(support.sum())), np.int64)
    summary = dict(generated_per_bank=len(groups[0]),
        bank_ids_sha256=[digest(group) for group in groups], heldout_ids_sha256=digest(heldout),
        template_ids_sha256=digest(template), complete=[0]*4, quality=[0]*4, accepted=[0]*4,
        overlap_with_reference=0)
    with tempfile.TemporaryDirectory(prefix=f"bank-sphere-{channel}-", dir=scratch) as directory:
        values = np.memmap(Path(directory)/"values.dat", mode="w+", dtype=np.float64, shape=(size, 3, 7))
        counts = np.zeros((size, 3), np.int32)
        fields = ["event_id", "pdg", "h11b_particle_source", "generator_particle_index",
                  "kinetic_energy_MeV", "px_MeV_c", "py_MeV_c", "pz_MeV_c", "x_mm", "y_mm", "z_mm"]
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as stream:
            for chunk in stream["virtual_sphere"].iterate(fields, step_size=100000, library="np"):
                convergence.accumulate_sphere(values, counts, chunk)
        complete = np.all(counts == 1, axis=1)
        rng = np.random.default_rng(np.random.SeedSequence([convergence.RESPONSE_SEED, channel, 0]))
        for first in range(0, size, 100000):
            last = min(first+100000, size)
            event = np.arange(first, last)[complete[first:last] & (bank_id[first:last] >= 0)]
            rows = np.asarray(values[event]).copy()
            events = dict(event_id=event, kinetic=rows[:, :, 0], momentum=rows[:, :, 1:4], hit_position=rows[:, :, 4:7])
            for bank in range(4):
                summary["complete"][bank] += int(np.count_nonzero(bank_id[event] == bank))
            events = scan.apply_mask(events, scan.quality_mask(events["kinetic"], events["momentum"], 1e-4))
            for bank in range(4):
                summary["quality"][bank] += int(np.count_nonzero(bank_id[events["event_id"]] == bank))
            if not len(events["event_id"]):
                continue
            rec, _ = scan.reconstruct_with_granularity(events, 2., 2., 0., 0., "sphere-center", np.array([0., 0., 170.]))
            measured, _, _ = scan.apply_closure_energy_response(rec, 100., 100., rng.standard_normal(rec["kinetic"].shape))
            xy = np.column_stack(scan.ordered_dalitz_xy(measured))
            for bank in range(4):
                selected = bank_id[measured["event_id"]] == bank
                hist[bank] += mismatch.histogram_xy(xy[selected], 60)[support].astype(np.int64)
                summary["accepted"][bank] += int(np.count_nonzero(selected))
        del values
    expected = np.asarray(base["inputs"]["banks"]["fresh_large"]["raw_templates"])[channel]
    if not np.array_equal(hist.sum(axis=0), expected):
        raise RuntimeError("Disjoint histograms do not exactly reconstruct the original full template")
    summary["reconstructs_original_histogram"] = True
    return hist, summary


def prepare_inputs(base, libraries, scratch):
    identity = dict(parent_checkpoint_sha256=base["checkpoint_sha256"],
        source_sha256=convergence.sha256(__file__),
        roots={name: libraries["runs"][name]["root_sha256"] for name in convergence.CHANNELS})
    cache = Path(scratch)/"reduced_inputs.json"
    for name in convergence.CHANNELS:
        row = libraries["runs"][name]
        if convergence.sha256(ROOT/row["root"]) != row["root_sha256"]:
            raise ValueError(f"Changed ROOT: {name}")
    if cache.exists():
        previous = json.loads(cache.read_text())
        if previous["identity"] != identity:
            raise ValueError("Bank reduction checkpoint changed")
        return previous["inputs"]
    counts, summaries = [], {}
    for channel, name in enumerate(convergence.CHANNELS):
        hist, summary = reduce_channel(ROOT/libraries["runs"][name]["root"], channel, base, scratch)
        counts.append(hist)
        summaries[name] = summary
        print(f"Reduced {name}: four banks G={GENERATED[channel]}, accepted={summary['accepted']}; full histogram EXACT", flush=True)
    banks = {name: dict(raw_templates=np.stack([hist[number] for hist in counts]), generated=np.array(GENERATED))
             for number, name in enumerate(BANKS)}
    reference = reaction_population(base["inputs"]["data_histograms"], base["inputs"]["data_generated"])
    if not np.array_equal(reference, base["inputs"]["independent_truth"]):
        raise RuntimeError("Held-out reference probabilities changed")
    result = convergence.finite.calibration.plain(dict(banks=banks,
        references=dict(heldout=reference, pooled=base["inputs"]["truth"]),
        reference_generated=base["inputs"]["data_generated"],
        reference_ids_sha256={name: summaries[name]["heldout_ids_sha256"] for name in convergence.CHANNELS},
        summaries=summaries,
        zero_reference_mc_events={name:[int(np.asarray(bank["raw_templates"])[c][reference[c,:-1] == 0].sum())
                                       for c in range(3)] for name,bank in banks.items()},
        scope="Four disjoint event partitions of one simulation per channel; not four new global seeds"))
    cache.write_text(json.dumps(dict(identity=identity, inputs=result), ensure_ascii=False, allow_nan=False))
    return result


def trial_inputs(inputs, bank, reference):
    return dict(truth=inputs["references"][reference], banks={"coherent": inputs["banks"][bank]})


def freeze_calibration(state, reference):
    if reference not in SEEDS or state["configuration"]["seed"] != SEEDS[reference]["calibration"]:
        raise ValueError("Validation or conditional trials cannot freeze thresholds")
    points = state["records"]
    rank = convergence.finite.calibration.tolerance_rank(state["configuration"]["repetitions"], comparisons=16)
    local, cutoffs = {}, {}
    for method in convergence.finite.METHODS:
        local[method] = {key:float(np.sort(row[method]["q_upper"])[rank["rank"]-1]) for key,row in points.items()}
        cutoffs[method] = dict(q_upper=max(local[method].values()),
            q0=float(np.sort(points["00000"][method]["q0"])[rank["rank"]-1]))
    return dict(cutoffs=cutoffs, local_upper=local, tolerance=rank,
        scope=f"{reference} empirical population; G=50000/250000/50000, d=0/0.005/0.02, r=0.05; two reference choices and two methods")


def bank_spread(records):
    output = {}
    for key in records[BANKS[0]]["records"]:
        output[key] = {}
        for method in convergence.finite.METHODS:
            estimates = np.stack([records[name]["records"][key][method]["map"] for name in BANKS])[:, :, 2]
            means = estimates.mean(axis=1)
            pairs = {}
            for left in range(4):
                for right in range(left+1, 4):
                    difference = estimates[right]-estimates[left]
                    pairs[f"{left+1}-{right+1}"] = dict(right_minus_left=difference.mean(),
                        mcse=difference.std(ddof=1)/np.sqrt(len(difference)))
            output[key][method] = dict(bank_means=means, mean_of_bank_means=means.mean(),
                sample_sd_of_bank_means=means.std(ddof=1), range_of_bank_means=np.ptp(means), pairs=pairs,
                scope="Descriptive spread of four disjoint partitions conditional on the fixed parent library/reference")
    return output


def numerical_controls(inputs):
    output = []
    for name in BANKS:
        current = trial_inputs(inputs, name, "heldout")
        data, mc, generated = convergence.finite.draw(current, .005, CONDITIONAL_SEED, 17, 100000, fluctuate_mc=False)
        fixed = convergence.finite.fixed_model(data, mc, generated)
        a = JointMultinomial(data, mc, generated, start=accepted_weights(fixed.mode, fixed.efficiencies))
        b = JointMultinomial(data, mc, generated, start=a.weights, tolerance=1e-5)
        other = JointMultinomial(data, mc, generated, start=[.10,.80,.10])
        loss_range = np.ptp([a.loss,b.loss,other.loss])
        direct_range = np.ptp([a.mode[2],b.mode[2],other.mode[2]])
        profile_difference = abs(a.statistic(.005)-b.statistic(.005))
        if loss_range > .002 or direct_range > 2e-5 or profile_difference > .002:
            raise RuntimeError(f"Actual noisy-data numerical control failed: bank {name}")
        output.append(dict(bank=name,trial=17,seed=CONDITIONAL_SEED,loss_range=loss_range,
            direct_range=direct_range,profile_difference=profile_difference,all_passed=True))
    return output


def run(*, notebook=NOTEBOOK, scratch, workers=4):
    start = time.monotonic()
    started_at = datetime.now().astimezone().isoformat()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    base = document["metadata"][convergence.KEY]
    libraries = document["metadata"]["codex_template_convergence_libraries"]
    del document
    digest, plain = convergence.finite.calibration.digest, convergence.finite.calibration.plain
    if digest({k:v for k,v in base.items() if k not in ("completed_at","checkpoint_sha256")}) != base["checkpoint_sha256"]:
        raise ValueError("Original verified histogram checkpoint changed")
    paths = [Path(__file__),Path(convergence.__file__),Path(convergence.finite.__file__),
             ROOT/"g4/analysis/bayes/finite_template_likelihood.py"]
    configuration = plain(dict(version=VERSION,parent_checkpoint_sha256=base["checkpoint_sha256"],
        banks=BANKS,generated=GENERATED,data_budget=100000,calibration_direct=CALIBRATION_DIRECT,
        validation_direct=VALIDATION_DIRECT,calibration_repetitions=1000,validation_repetitions=500,
        seeds=SEEDS,conditional_seed=CONDITIONAL_SEED,
        source_sha256={str(p.relative_to(ROOT)):convergence.sha256(p) for p in paths}))
    plan = scratch/"plan.json"
    if plan.exists() and json.loads(plan.read_text()) != configuration:
        raise ValueError("Do not silently change the four-bank study protocol")
    plan.write_text(json.dumps(configuration,ensure_ascii=False,indent=1))
    snapshots = {str(p.relative_to(ROOT)):p.read_text() for p in paths}
    inputs = prepare_inputs(base,libraries,scratch)
    controls = numerical_controls(inputs)
    print("Four-bank noisy-data precision/multistart controls PASS",flush=True)
    frozen, matched, records = {}, {}, {}
    for reference, seeds in SEEDS.items():
        current = trial_inputs(inputs,"1",reference)
        print(f"Calibrating {reference} population at the new MC budget",flush=True)
        calibration = ensemble_checkpoint(current,scratch/f"calibration_{reference}.json",
            CALIBRATION_DIRECT,seeds["calibration"],1000,workers=workers)
        frozen[reference] = freeze_calibration(calibration,reference)
        validation = ensemble_checkpoint(current,scratch/f"validation_{reference}.json",
            VALIDATION_DIRECT,seeds["validation"],500,workers=workers)
        matched[reference] = convergence.finite.summarize(validation,frozen[reference])
        records[f"calibration_{reference}"],records[f"validation_{reference}"] = calibration,validation
    conditional, summaries, expected = {}, {key:{} for key in SEEDS}, {}
    for name in BANKS:
        current = trial_inputs(inputs,name,"heldout")
        print(f"Fixed bank {name}: the same independent reference and paired data",flush=True)
        result = ensemble_checkpoint(current,scratch/f"conditional_bank_{name}.json",
            VALIDATION_DIRECT,CONDITIONAL_SEED,500,workers=workers,fluctuate_mc=False)
        conditional[name],records[f"conditional_{name}"] = result,result
        for reference in SEEDS:
            summaries[reference][name] = convergence.finite.summarize(result,frozen[reference])
        expected[name] = {}
        bank = inputs["banks"][name]
        for direct in VALIDATION_DIRECT:
            data = 100000*(convergence.finite.fractions(direct)@np.asarray(inputs["references"]["heldout"]))[:-1]
            expected[name][f"{int(round(direct*1e6)):05d}"] = evaluate(data,
                np.asarray(bank["raw_templates"]),np.asarray(bank["generated"]),direct)
    for relative, checksum in configuration["source_sha256"].items():
        if convergence.sha256(ROOT/relative) != checksum:
            raise ValueError(f"Source changed during the study: {relative}")
    recovery = json.loads((scratch/"recovery.json").read_text()) if (scratch/"recovery.json").exists() else None
    state = plain(dict(configuration=configuration,started_at=started_at,inputs=inputs,recovery_history=recovery,
        frozen=frozen,matched_summary=matched,conditional_summary=summaries,expected=expected,
        bank_spread=bank_spread(conditional),numerical_controls=controls,raw_records=convergence.pack(records),
        source_snapshots=snapshots,elapsed_seconds=time.monotonic()-start,
        interpretation="All four actual banks see the same held-out empirical data population. Cutoff-reference choices are reported separately, not selected after validation."))
    state["checkpoint_sha256"] = digest(state)
    state["completed_at"] = datetime.now().astimezone().isoformat()
    save_metadata(notebook,KEY,state)
    print(f"Four-bank study complete: 18000 trials, {state['elapsed_seconds']:.1f}s",flush=True)
    return state


def plots(state):
    """Cached-only plots for the main notebook's Fig. 17 and Fig. 18."""
    import matplotlib.pyplot as plt
    figures = []
    fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for name in BANKS:
        points = state["conditional_summary"]["heldout"][name]
        keys = sorted(points)
        rows = [points[key]["joint"] for key in keys]
        axes[0].errorbar([int(key)/1e4 for key in keys],[100*r["bias"] for r in rows],
            yerr=[100*r["mcse"] for r in rows],fmt="o-",capsize=3,label=f"Bank {name}")
    axes[0].axhline(0,color="k",ls=":")
    axes[0].set(xlabel="True uniform phase-space fraction (%)",ylabel="Mean bias (percentage points)")
    axes[0].legend()
    key = "05000"
    expected = [100*state["expected"][name][key]["joint"]["map"][2] for name in BANKS]
    rows = [state["conditional_summary"]["heldout"][name][key]["joint"] for name in BANKS]
    axes[1].plot(range(1,5),expected,"s--",label="Expected-count fit")
    axes[1].errorbar(range(1,5),[100*r["mean"] for r in rows],yerr=[100*r["mcse"] for r in rows],
        fmt="o-",capsize=3,label="Mean over pseudo-data")
    axes[1].axhline(.5,color="k",ls=":")
    axes[1].set(xticks=range(1,5),xlabel="Disjoint template bank",ylabel="Estimated fraction at true d=0.5% (%)")
    axes[1].legend()
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,reference,title in zip(axes,("heldout","pooled"),("Held-out-reference calibration","Pooled-reference calibration")):
        for name in BANKS:
            points = state["conditional_summary"][reference][name]
            keys = sorted(points)
            rows = [points[key]["joint"]["coverage"] for key in keys]
            rates = np.array([r["rate"] for r in rows])
            interval = np.array([r["wilson95"] for r in rows])
            errors = 100*np.maximum(np.vstack([rates-interval[:,0],interval[:,1]-rates]),0)
            ax.errorbar([int(key)/1e4 for key in keys],100*rates,yerr=errors,fmt="o-",capsize=3,label=f"Bank {name}")
        ax.axhline(95,color="k",ls=":")
        ax.set(title=title,xlabel="True uniform phase-space fraction (%)",
            ylabel="Conditional upper-limit inclusion (%)",ylim=(-1,102))
        ax.legend(loc="lower right")
    figures.append(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    run(scratch=args.scratch,workers=args.workers)
