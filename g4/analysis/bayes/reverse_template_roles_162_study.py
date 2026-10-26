"""Swap the independent 162-keV template/validation halves without new MC.

Explicit CLI only. Reuses verified histogram caches, the original large-MC
cutoffs and disjoint event-ID partitions; never recalibrates on these results.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import template_convergence_162_study as convergence
from analysis.bayes.finite_template_likelihood import JointMultinomial, accepted_weights
from analysis.bayes.generate_162_convergence_libraries import save_metadata

ROOT, NOTEBOOK = convergence.ROOT, convergence.NOTEBOOK
KEY = "codex_template_role_reversal"
VERSION = "2026-10-22-v1-independent-role-reversal"
COMMON_SEED = 1622026102288
INDEPENDENT_SEED = 1622026102287
DIRECT = convergence.VALIDATION_DIRECT


def make_inputs(base):
    inputs = base["inputs"]
    original = inputs["banks"]["fresh_large"]
    reversed_bank = dict(raw_templates=np.asarray(inputs["data_histograms"]),
                         generated=np.asarray(inputs["data_generated"]))
    if not np.array_equal(original["generated"], reversed_bank["generated"]):
        raise ValueError("Role reversal requires equal generated budgets for the frozen cutoff")
    mc, generated = np.asarray(original["raw_templates"]), np.asarray(original["generated"])
    swapped_reference = np.column_stack([mc, generated-mc.sum(axis=1)])/generated[:, None]
    return {
        "common_original": dict(truth=inputs["truth"], banks={"coherent": original}),
        "common_reversed": dict(truth=inputs["truth"], banks={"coherent": reversed_bank}),
        "independent_reversed": dict(truth=swapped_reference, banks={"coherent": reversed_bank}),
    }


def numerical_controls(base, inputs):
    """Check noisy trials too: Asimov recovery need not equal the mean MLE."""
    original = dict(truth=base["inputs"]["independent_truth"],
                    banks={"coherent": base["inputs"]["banks"]["fresh_large"]})
    checks = []
    for name, current, seed in (("original", original, convergence.INDEPENDENT_REFERENCE_SEED),
                                 ("reversed", inputs["independent_reversed"], INDEPENDENT_SEED)):
        for index in (0, 17, 161):
            data, mc, generated = convergence.finite.draw(current, .005, seed, index, 100000, fluctuate_mc=False)
            fixed = convergence.finite.fixed_model(data, mc, generated)
            model = JointMultinomial(data, mc, generated,
                start=accepted_weights(fixed.mode, fixed.efficiencies))
            tight = JointMultinomial(data, mc, generated, start=model.weights, tolerance=1e-5)
            starts = []
            for initial in ([.05, .949, .001], [.1, .8, .1], [.03, .96, .01]):
                other = JointMultinomial(data, mc, generated, start=initial)
                starts.append(dict(initial=initial, loss=other.loss, mode=other.mode))
            losses = [model.loss, tight.loss, *[row["loss"] for row in starts]]
            direct = [model.mode[2], tight.mode[2], *[row["mode"][2] for row in starts]]
            if np.ptp(losses) > .002 or np.ptp(direct) > 2e-5:
                raise RuntimeError(f"Noisy-data multistart/refinement failed: {name}, {index}")
            checks.append(dict(bank=name, seed=seed, trial=index, mode=model.mode,
                loss_range=np.ptp(losses), direct_range=np.ptp(direct), starts=starts,
                refined_profile_difference=abs(model.statistic(.005)-tight.statistic(.005))))
    return checks


def paired_summary(a, b, frozen):
    output = {}
    for key in a["records"]:
        output[key] = {}
        for method in convergence.finite.METHODS:
            left, right = a["records"][key][method], b["records"][key][method]
            cutoff = frozen["cutoffs"][method]["q_upper"]
            difference = (np.asarray(right["q_upper"]) <= cutoff).astype(float)-(np.asarray(left["q_upper"]) <= cutoff)
            delta = np.asarray(right["map"])[:, 2]-np.asarray(left["map"])[:, 2]
            output[key][method] = dict(coverage_difference=difference.mean(),
                coverage_mcse=difference.std(ddof=1)/np.sqrt(len(difference)),
                estimate_difference=delta.mean(), estimate_mcse=delta.std(ddof=1)/np.sqrt(len(delta)))
    return output


def run(*, notebook=NOTEBOOK, scratch, workers=4):
    start = time.monotonic()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(notebook).read_text())["metadata"][convergence.KEY]
    if convergence.finite.calibration.digest({k:v for k,v in base.items()
            if k not in ("completed_at", "checkpoint_sha256")}) != base["checkpoint_sha256"]:
        raise ValueError("Original convergence checkpoint changed")
    paths = [Path(__file__), Path(convergence.__file__), Path(convergence.finite.__file__),
             ROOT/"g4/analysis/bayes/finite_template_likelihood.py"]
    configuration = dict(version=VERSION, source_checkpoint_sha256=base["checkpoint_sha256"],
        direct=DIRECT, repetitions=500, data_budget=100000,
        common_seed=COMMON_SEED, independent_seed=INDEPENDENT_SEED,
        source_sha256={str(p.relative_to(ROOT)): convergence.sha256(p) for p in paths},
        frozen_cutoff_policy="Reuse large-budget cutoffs without recalibration", frozen=base["frozen"]["large"])
    plan = scratch/"plan.json"
    plain = convergence.finite.calibration.plain
    if plan.exists() and json.loads(plan.read_text()) != plain(configuration):
        raise ValueError("Role-reversal protocol changed")
    plan.write_text(json.dumps(plain(configuration), indent=1))
    inputs = make_inputs(base)
    checks = numerical_controls(base, inputs)
    print("Noisy-data numerical controls PASS", flush=True)
    records, summaries = {}, {}
    for name, current in inputs.items():
        seed = INDEPENDENT_SEED if name == "independent_reversed" else COMMON_SEED
        print(f"Starting {name}: fixed template, frozen cutoff, seed {seed}", flush=True)
        record = convergence.ensemble_checkpoint(current, scratch/f"{name}.json", DIRECT,
            seed, 500, workers=workers, fluctuate_mc=False)
        records[name] = record
        summaries[name] = convergence.finite.summarize(record, configuration["frozen"])
    expected = {}
    for direct in DIRECT:
        data = 100000*(convergence.finite.fractions(direct)@inputs["independent_reversed"]["truth"])[:-1]
        bank = inputs["independent_reversed"]["banks"]["coherent"]
        expected[f"{int(round(direct*1e6)):05d}"] = convergence.finite.evaluate(data,
            bank["raw_templates"], bank["generated"], direct)
    result = plain(dict(configuration=configuration, inputs=inputs, numerical_controls=checks,
        summaries=summaries, independent_expected=expected,
        paired_common=paired_summary(records["common_original"], records["common_reversed"], configuration["frozen"]),
        original_independent_summary=base["independent_summary"]["fresh_large"],
        original_independent_expected={key: value["fresh_large"] for key,value in base["independent_expected"].items()},
        raw_records=convergence.pack(records), source_snapshots={str(p.relative_to(ROOT)):p.read_text() for p in paths},
        partitions=base["inputs"]["summaries"], elapsed_seconds=time.monotonic()-start,
        interpretation="Two disjoint halves exchange roles. Common-reference data are paired; held-out references change on reversal."))
    result["checkpoint_sha256"] = convergence.finite.calibration.digest(result)
    result["completed_at"] = datetime.now().astimezone().isoformat()
    save_metadata(notebook, KEY, result)
    print(f"Role reversal complete, 6000 trials: {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def plot(state):
    """Cached mean-bias comparison; numbered Fig. 16 by the notebook."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    groups = [("Common reference", state["summaries"]["common_original"], state["summaries"]["common_reversed"]),
              ("Disjoint empirical references", state["original_independent_summary"], state["summaries"]["independent_reversed"])]
    for ax, (title, original, swapped) in zip(axes, groups):
        for values, label in ((original, "Original roles"), (swapped, "Swapped roles")):
            keys = sorted(values)
            rows = [values[key]["joint"] for key in keys]
            ax.errorbar([int(key)/1e4 for key in keys], [100*r["bias"] for r in rows],
                yerr=[100*r["mcse"] for r in rows], fmt="o-", capsize=3, label=label)
        ax.axhline(0, color="k", ls=":")
        ax.set(title=title, xlabel="True uniform phase-space fraction (%)",
               ylabel="Mean bias (percentage points)")
        ax.legend()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(scratch=args.scratch, workers=args.workers)
