"""Paired binning and nested-MC-budget study of the 162-keV weak component.

Explicit CLI only. Uses verified cached counts, never starts Geant4. All
coarse counts are exact sums of the same fine counts, including pseudo-data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import disjoint_template_banks_162_study as banks

ROOT, NOTEBOOK = banks.ROOT, banks.NOTEBOOK
finite, convergence = banks.convergence.finite, banks.convergence
digest, plain = finite.calibration.digest, finite.calibration.plain
KEY = "codex_binning_budget_162_study"
VERSION = "2026-10-22-v1-paired-binning-nested-budget"
BINS = (30, 60)
GROUPS = {1: (("1",), ("2",), ("3",), ("4",)),
          2: (("1", "2"), ("3", "4")), 4: (("1", "2", "3", "4"),)}
CALIBRATION_SEED = banks.SEEDS["heldout"]["calibration"]
VALIDATION_SEED = banks.SEEDS["heldout"]["validation"]
CONDITIONAL_SEED = banks.CONDITIONAL_SEED
COMPARISONS = len(BINS)*len(GROUPS)*len(finite.METHODS)*(len(banks.CALIBRATION_DIRECT)+1)
_MAPPINGS = {}


def fine_to_coarse_mapping(bins):
    """Map every fine geometric-support bin into a target support bin."""
    if bins not in BINS:
        raise ValueError("Only the declared 30 and 60 grids are supported")
    if bins not in _MAPPINGS:
        source = convergence.mismatch.geometric_support(60)
        target = convergence.mismatch.geometric_support(bins)
        positions = np.full(target.shape, -1, np.int64)
        positions[target] = np.arange(target.sum())
        x, y = np.nonzero(source)
        factor = 60//bins
        mapping = positions[x//factor, y//factor]
        if np.any(mapping < 0):
            raise RuntimeError("A fine support bin is outside the coarse support")
        _MAPPINGS[bins] = mapping, int(target.sum())
    return _MAPPINGS[bins]


def coarsen(values, bins):
    """Sum the last, accepted-bin axis without changing acceptance counts."""
    values = np.asarray(values)
    mapping, size = fine_to_coarse_mapping(bins)
    if values.shape[-1] != len(mapping):
        raise ValueError("Input must use the complete fine geometric support")
    if bins == 60:
        return values.copy()
    output = np.zeros((*values.shape[:-1], size), dtype=values.dtype)
    for row, target in zip(values.reshape(-1, len(mapping)), output.reshape(-1, size)):
        np.add.at(target, mapping, row)
    if not np.allclose(output.sum(axis=-1), values.sum(axis=-1), rtol=0, atol=1e-9):
        raise RuntimeError("Coarsening changed accepted counts")
    return output


def config_id(bins, scale):
    return f"{bins}_x{scale}"


def case_id(bins, members):
    return f"{config_id(bins,len(members))}_{''.join(members)}"


def prepare_inputs(parent):
    truth = np.asarray(parent["inputs"]["references"]["heldout"])
    source = parent["inputs"]["banks"]
    cases = {}
    for bins in BINS:
        for scale, groups in GROUPS.items():
            for members in groups:
                raw = np.stack([source[name]["raw_templates"] for name in members]).sum(axis=0)
                generated = np.stack([source[name]["generated"] for name in members]).sum(axis=0)
                cases[case_id(bins,members)] = dict(bins=bins,scale=scale,members=members,
                    raw_templates=coarsen(raw,bins),generated=generated,configuration=config_id(bins,scale))
    return plain(dict(fine_truth=truth,cases=cases,
        reference_generated=parent["inputs"]["reference_generated"],
        reference_ids_sha256=parent["inputs"]["reference_ids_sha256"],
        truth={str(b):np.column_stack([coarsen(truth[:,:-1],b),truth[:,-1]]) for b in BINS},
        support_bins={str(b):fine_to_coarse_mapping(b)[1] for b in BINS},
        mapping={str(b):fine_to_coarse_mapping(b)[0] for b in BINS},
        scope="Fixed held-out empirical reference; nested groups of the same finite parent templates"))


def data_draw(truth, direct, seed, index):
    identity = int(round(direct*1e6))
    rng = np.random.default_rng(np.random.SeedSequence([seed,identity,index,0]))
    return rng.multinomial(100000,finite.fractions(direct)@np.asarray(truth))[:-1]


def paired_draw(inputs, case, direct, seed, index, *, fluctuate_mc):
    """Draw at fine resolution, then aggregate; nest independent MC quarters.

    Quarter 0 is exactly the previous four-bank study's MC draw. Additional
    quarters use separate SeedSequence namespaces and are added, not redrawn.
    """
    truth = np.asarray(inputs["fine_truth"])
    data = data_draw(truth,direct,seed,index)
    data_sha256 = digest(data)
    if fluctuate_mc:
        mc = np.zeros((3,len(data)),np.int64)
        identity = int(round(direct*1e6))
        for quarter in range(case["scale"]):
            for channel, (generated,p) in enumerate(zip(banks.GENERATED,truth)):
                namespace = [seed,identity,index,1,channel]
                if quarter:
                    namespace.append(quarter)
                rng = np.random.default_rng(np.random.SeedSequence(namespace))
                mc[channel] += rng.multinomial(generated,p)[:-1]
        mc = coarsen(mc,case["bins"])
    else:
        mc = np.asarray(case["raw_templates"])
    return coarsen(data,case["bins"]),mc,np.asarray(case["generated"]),data_sha256


_INPUTS, _CASE = None, None


def _initialize(inputs, case):
    global _INPUTS, _CASE
    _INPUTS, _CASE = inputs, case


def _chunk(task):
    direct,seed,first,last,fluctuate_mc = task
    output = {method:[] for method in finite.METHODS}
    for index in range(first,last):
        try:
            data,mc,generated,data_sha = paired_draw(_INPUTS,_CASE,direct,seed,index,fluctuate_mc=fluctuate_mc)
            evaluated = banks.evaluate(data,mc,generated,direct)
            for method in output:
                evaluated[method]["paired_data_sha256"] = data_sha
                output[method].append(evaluated[method])
        except Exception as error:
            raise RuntimeError(f"Trial failed without omission: case={_CASE['configuration']}, seed={seed}, d={direct}, index={index}") from error
    return plain(output)


def array_records(records):
    return {key:{method:{name:np.asarray(value) for name,value in row.items()} for method,row in point.items()}
            for key,point in records.items()}


def ensemble(inputs, case, path, phase, *, workers, reused=None):
    calibration = phase == "calibration"
    seed = CALIBRATION_SEED if calibration else VALIDATION_SEED if phase == "validation" else CONDITIONAL_SEED
    if phase not in ("calibration","validation","conditional"):
        raise ValueError("Unknown study phase")
    direct_values = banks.CALIBRATION_DIRECT if calibration else banks.VALIDATION_DIRECT
    repetitions = 1000 if calibration else 500
    config = plain(dict(version=VERSION,case=case,phase=phase,seed=seed,repetitions=repetitions,
        direct_values=direct_values,budget=100000,fluctuate_mc=phase!="conditional",
        fine_population_sha256=digest(inputs["fine_truth"]),chunk_size=25,
        generated_quarter=banks.GENERATED,input_sha256=digest(inputs)))
    if reused is not None:
        prior = reused["configuration"]
        if (case["bins"] != 60 or case["scale"] != 1 or prior["seed"] != seed
                or prior["repetitions"] != repetitions or prior["direct_values"] != list(direct_values)
                or prior["fluctuate_mc"] != config["fluctuate_mc"]
                or prior["population_sha256"] != config["fine_population_sha256"]):
            raise ValueError("Reused raw trials do not match this paired experiment")
        records = array_records(reused["records"])
        for key,point in records.items():
            hashes = [digest(data_draw(inputs["fine_truth"],int(key)/1e6,seed,index)) for index in range(repetitions)]
            for row in point.values():
                row["paired_data_sha256"] = np.asarray(hashes)
        config["reused_raw_statistics_sha256"] = digest(reused["records"])
        return dict(configuration=config,records=records,reused=True)
    path = Path(path)
    state = json.loads(path.read_text()) if path.exists() else dict(configuration=config,chunks={})
    if state["configuration"] != config:
        raise ValueError("Binning/budget checkpoint configuration changed")
    tasks = {f"{int(round(d*1e6)):05d}:{first}":(d,seed,first,min(first+25,repetitions),phase!="conditional")
             for d in direct_values for first in range(0,repetitions,25)}
    for key,chunk in state["chunks"].items():
        if key not in tasks or set(chunk)!=set(finite.METHODS) or any(len(rows)!=tasks[key][3]-tasks[key][2] for rows in chunk.values()):
            raise ValueError("Invalid complete-chunk checkpoint")
    def save():
        temporary = path.with_suffix(".pending")
        temporary.write_text(json.dumps(plain(state),allow_nan=False))
        temporary.replace(path)
    start,last_notice = time.monotonic(),time.monotonic()
    pending = {key:task for key,task in tasks.items() if key not in state["chunks"]}
    if pending:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize,initargs=(inputs,case)) as pool:
            futures = {pool.submit(_chunk,task):key for key,task in pending.items()}
            for future in as_completed(futures):
                try:
                    state["chunks"][futures[future]] = future.result()
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
                count = sum(len(c["joint"]) for c in state["chunks"].values())
                if count%500==0 or len(state["chunks"])==len(tasks):
                    save()
                if time.monotonic()-last_notice>20:
                    print(f"{case['configuration']} {phase}: {count}/{len(tasks)*25} certified trials; {time.monotonic()-start:.1f}s",flush=True)
                    last_notice=time.monotonic()
    records = {}
    for direct in direct_values:
        key = f"{int(round(direct*1e6)):05d}"
        records[key] = {}
        for method in finite.METHODS:
            rows = [row for first in range(0,repetitions,25) for row in state["chunks"][f"{key}:{first}"][method]]
            records[key][method] = {name:np.asarray([row[name] for row in rows]) for name in rows[0]}
    save()
    return dict(configuration=config,records=records,reused=False)


def freeze(state):
    configuration = state["configuration"]
    if configuration["phase"]!="calibration" or configuration["seed"]!=CALIBRATION_SEED:
        raise ValueError("Only prespecified calibration trials can freeze thresholds")
    rank = finite.calibration.tolerance_rank(configuration["repetitions"],comparisons=COMPARISONS)
    local,cutoffs = {},{}
    for method in finite.METHODS:
        local[method] = {key:float(np.sort(point[method]["q_upper"])[rank["rank"]-1]) for key,point in state["records"].items()}
        cutoffs[method] = dict(q_upper=max(local[method].values()),
            q0=float(np.sort(state["records"]["00000"][method]["q0"])[rank["rank"]-1]))
    return dict(cutoffs=cutoffs,local_upper=local,tolerance=rank,
        scope="Six declared binning/budget configurations, two methods, d=0/.005/.02, r=.05; held-out empirical population")


def verify_pairing(records, names):
    first = records[names[0]]["records"]
    for name in names[1:]:
        for key,point in first.items():
            if not np.array_equal(point["joint"]["paired_data_sha256"],records[name]["records"][key]["joint"]["paired_data_sha256"]):
                raise RuntimeError(f"Pseudo-data pairing changed: {name}, {key}")


def controls(inputs):
    output = []
    for bins in BINS:
        for scale,groups in GROUPS.items():
            name = case_id(bins,groups[0])
            data,mc,generated,_ = paired_draw(inputs,inputs["cases"][name],.005,CONDITIONAL_SEED,17,fluctuate_mc=False)
            fixed = finite.fixed_model(data,mc,generated)
            a = banks.JointMultinomial(data,mc,generated,start=banks.accepted_weights(fixed.mode,fixed.efficiencies))
            b = banks.JointMultinomial(data,mc,generated,start=a.weights,tolerance=1e-5)
            c = banks.JointMultinomial(data,mc,generated,start=[.1,.8,.1])
            loss = np.ptp([a.loss,b.loss,c.loss])
            direct = np.ptp([a.mode[2],b.mode[2],c.mode[2]])
            profile = abs(a.statistic(.005)-b.statistic(.005))
            if loss>.002 or direct>2e-5 or profile>.002:
                raise RuntimeError(f"Noisy-data precision/multistart control failed: {name}")
            output.append(dict(case=name,loss_range=loss,direct_range=direct,profile_difference=profile,all_passed=True))
    return output


def binning_contrasts(records):
    output = {}
    for groups in GROUPS.values():
        for members in groups:
            left,right = (records[f"conditional_{case_id(b,members)}"]["records"] for b in BINS)
            output[''.join(members)] = {}
            for key in left:
                output[''.join(members)][key] = {}
                for method in finite.METHODS:
                    difference = left[key][method]["map"][:,2]-right[key][method]["map"][:,2]
                    output[''.join(members)][key][method] = dict(coarse_minus_fine=difference.mean(),
                        paired_mcse=difference.std(ddof=1)/np.sqrt(len(difference)))
    return output


def run(*, notebook=NOTEBOOK, scratch, workers=8):
    start = time.monotonic()
    scratch = Path(scratch)
    scratch.mkdir(parents=True,exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    if KEY in document["metadata"]:
        raise ValueError("Preserve the existing study; do not overwrite its checkpoint")
    parent = document["metadata"][banks.KEY]
    if digest({k:v for k,v in parent.items() if k not in ("checkpoint_sha256","completed_at")})!=parent["checkpoint_sha256"]:
        raise ValueError("Verified parent checkpoint changed")
    inputs = prepare_inputs(parent)
    parent_records = convergence.unpack(parent["raw_records"])
    paths = [Path(__file__),Path(banks.__file__),Path(finite.__file__),Path(convergence.__file__),
             ROOT/"g4/analysis/bayes/finite_template_likelihood.py",
             ROOT/"g4/analysis/bayes/model_mismatch_162_study.py",
             ROOT/"g4/analysis/bayes/calibrate_162_weak_branch_intervals.py"]
    configuration = plain(dict(version=VERSION,started_at=datetime.now().astimezone().isoformat(),
        parent_checkpoint_sha256=parent["checkpoint_sha256"],bins=BINS,groups=GROUPS,
        calibration_seed=CALIBRATION_SEED,validation_seed=VALIDATION_SEED,conditional_seed=CONDITIONAL_SEED,
        calibration_repetitions=1000,validation_repetitions=500,comparisons=COMPARISONS,
        data_budget=100000,calibration_direct=banks.CALIBRATION_DIRECT,validation_direct=banks.VALIDATION_DIRECT,
        source_sha256={str(p.relative_to(ROOT)):convergence.sha256(p) for p in paths}))
    plan = scratch/"plan.json"
    if plan.exists():
        existing = json.loads(plan.read_text())
        configuration["started_at"] = existing["started_at"]
        if existing!=configuration:
            raise ValueError("Do not silently change the prespecified binning/budget protocol")
    plan.write_text(json.dumps(configuration,ensure_ascii=False,indent=1))
    snapshots = {str(p.relative_to(ROOT)):p.read_text() for p in paths}
    numerical = controls(inputs)
    print("Six noisy-data precision/multistart controls PASS; support bins 30=144, 60=497",flush=True)
    records,frozen,matched,conditional,expected = {},{},{},{},{}
    for bins in BINS:
        for scale,groups in GROUPS.items():
            name,cfg = case_id(bins,groups[0]),config_id(bins,scale)
            reused = parent_records["calibration_heldout"] if (bins,scale)==(60,1) else None
            state = ensemble(inputs,inputs["cases"][name],scratch/f"calibration_{cfg}.json","calibration",workers=workers,reused=reused)
            records[f"calibration_{cfg}"] = state
            frozen[cfg] = freeze(state)
            print(f"Frozen {cfg}: {frozen[cfg]['cutoffs']}",flush=True)
    verify_pairing(records,[f"calibration_{config_id(b,s)}" for b in BINS for s in GROUPS])
    for bins in BINS:
        for scale,groups in GROUPS.items():
            name,cfg = case_id(bins,groups[0]),config_id(bins,scale)
            reused = parent_records["validation_heldout"] if (bins,scale)==(60,1) else None
            state = ensemble(inputs,inputs["cases"][name],scratch/f"validation_{cfg}.json","validation",workers=workers,reused=reused)
            records[f"validation_{cfg}"] = state
            matched[cfg] = finite.summarize(state,frozen[cfg])
            print(f"Matched validation {cfg} COMPLETE",flush=True)
    verify_pairing(records,[f"validation_{config_id(b,s)}" for b in BINS for s in GROUPS])
    for name,case in inputs["cases"].items():
        reused = parent_records[f"conditional_{case['members'][0]}"] if (case["bins"],case["scale"])==(60,1) else None
        state = ensemble(inputs,case,scratch/f"conditional_{name}.json","conditional",workers=workers,reused=reused)
        records[f"conditional_{name}"] = state
        conditional[name] = finite.summarize(state,frozen[case["configuration"]])
        expected[name] = {}
        for direct in banks.VALIDATION_DIRECT:
            data = coarsen(100000*(finite.fractions(direct)@np.asarray(inputs["fine_truth"]))[:-1],case["bins"])
            expected[name][f"{int(round(direct*1e6)):05d}"] = banks.evaluate(data,np.asarray(case["raw_templates"]),np.asarray(case["generated"]),direct)
        print(f"Actual fixed template {name} COMPLETE",flush=True)
    verify_pairing(records,[f"conditional_{name}" for name in inputs["cases"]])
    for relative,checksum in configuration["source_sha256"].items():
        if convergence.sha256(ROOT/relative)!=checksum:
            raise ValueError(f"Study source changed during computation: {relative}")
    contrasts = binning_contrasts(records)
    total = sum(len(p["joint"]["map"]) for s in records.values() for p in s["records"].values())
    reused_count = sum(len(p["joint"]["map"]) for s in records.values() if s["reused"] for p in s["records"].values())
    if (total,reused_count)!=(58000,13000):
        raise RuntimeError("Incomplete study record counts")
    state = plain(dict(configuration=configuration,inputs=inputs,frozen=frozen,matched_summary=matched,
        conditional_summary=conditional,expected=expected,binning_contrasts=contrasts,numerical_controls=numerical,
        raw_records=convergence.pack(records),source_snapshots=snapshots,
        counts=dict(fit_trial_cases=total,reused_raw_fit_trial_cases=reused_count,new_fit_trial_cases=total-reused_count,
            distinct_calibration_data=3000,distinct_validation_data=2000,distinct_conditional_data=2000),
        pairing_checks=dict(calibration=True,validation=True,conditional=True),elapsed_seconds=time.monotonic()-start,
        interpretation="Same held-out empirical data reference, paired data, nested MC counts; conditional coverage is distinct from matched data-and-MC regeneration."))
    state["checkpoint_sha256"] = digest(state)
    state["completed_at"] = datetime.now().astimezone().isoformat()
    banks.save_metadata(notebook,KEY,state)
    print(f"Binning/budget study COMPLETE: {total} cases, {reused_count} reused; {state['elapsed_seconds']:.1f}s",flush=True)
    return state


def plots(state):
    """Three cached-only figures; callers assign the notebook figure numbers."""
    import matplotlib.pyplot as plt
    names = [members for groups in GROUPS.values() for members in groups]
    labels = [''.join(m) for m in names]
    x = np.arange(len(names))
    figures = []
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,bins in zip(axes,BINS):
        cases = [case_id(bins,m) for m in names]
        rows = [state["conditional_summary"][name]["05000"]["joint"] for name in cases]
        ax.errorbar(x,[100*r["mean"] for r in rows],yerr=[100*r["mcse"] for r in rows],fmt="o",capsize=3,label="Mean over paired pseudo-data")
        ax.plot(x,[100*state["expected"][name]["05000"]["joint"]["map"][2] for name in cases],"s",label="Expected-count fit")
        ax.axhline(.5,color="k",ls=":")
        ax.set(title=f"{bins} x {bins} Dalitz bins",xticks=x,xticklabels=labels,
            xlabel="Template members: four / two / one bank(s)",ylabel="Estimated fraction at true d=0.5% (%)")
        ax.legend(fontsize=8)
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,bins in zip(axes,BINS):
        lower = 100.
        for offset,key,label in ((-.16,"02000","d=0.2%"),(0.,"05000","d=0.5%"),(.16,"20000","d=2%")):
            rows = [state["conditional_summary"][case_id(bins,m)][key]["joint"]["coverage"] for m in names]
            p = np.array([r["rate"] for r in rows])*100
            interval = np.array([r["wilson95"] for r in rows])*100
            lower = min(lower,float(interval.min()))
            ax.errorbar(x+offset,p,yerr=np.maximum(np.vstack([p-interval[:,0],interval[:,1]-p]),0),fmt="o",capsize=3,label=label)
        ax.axhline(95,color="k",ls=":")
        ax.set(title=f"{bins} x {bins} Dalitz bins",xticks=x,xticklabels=labels,ylim=(max(0.,min(90.,lower-2)),101),
            xlabel="Template members: four / two / one bank(s)",ylabel="Conditional upper-limit inclusion (%)")
        ax.legend(fontsize=8)
    figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    lower = 100.
    for bins in BINS:
        for scale in GROUPS:
            rows = state["matched_summary"][config_id(bins,scale)]
            keys = sorted(rows)
            truth = np.array([int(k)/1e4 for k in keys])
            label = f"{bins} x {bins}, MC x{scale}"
            axes[0].errorbar(truth,[100*rows[k]["joint"]["bias"] for k in keys],
                yerr=[100*rows[k]["joint"]["mcse"] for k in keys],marker="o",capsize=2,label=label)
            p = np.array([100*rows[k]["joint"]["coverage"]["rate"] for k in keys])
            bounds = np.array([rows[k]["joint"]["coverage"]["wilson95"] for k in keys])*100
            lower = min(lower,float(bounds.min()))
            axes[1].errorbar(truth,p,yerr=np.maximum(np.vstack([p-bounds[:,0],bounds[:,1]-p]),0),marker="o",capsize=2,label=label)
    axes[0].axhline(0,color="k",ls=":")
    axes[1].axhline(95,color="k",ls=":")
    axes[0].set(xlabel="True uniform phase-space fraction (%)",ylabel="Mean bias (percentage points)",title="Data and MC both regenerated")
    axes[1].set(xlabel="True uniform phase-space fraction (%)",ylabel="Upper-limit inclusion (%)",title="Independent matched validation",ylim=(max(0.,min(90.,lower-2)),101))
    for ax in axes:
        ax.legend(fontsize=8)
    figures.append(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=8,choices=range(1,9))
    arguments = parser.parse_args()
    run(scratch=arguments.scratch,workers=arguments.workers)
