"""Independent template-size and replacement controls at 162 keV.

Only the CLI starts fits. Notebook plotting reads cached numbers. Large ROOT
input is reduced in bounded blocks, retaining generated (preselection) budgets.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zlib

import numpy as np
import uproot

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import finite_template_162_study as finite
from analysis.bayes import model_mismatch_162_study as mismatch
from analysis.bayes.generate_162_convergence_libraries import KEY as LIBRARY_KEY, save_metadata
from analysis.bayes.generate_162_model_libraries import ROOT, NOTEBOOK, sha256

VERSION = "2026-10-22-v1-independent-template-convergence"
KEY = "codex_template_convergence_study"
CHANNELS = ("alpha0", "coherent", "phase_space")
SPLIT_SEED = 1622026102280
RESPONSE_SEED = 1622026102284
CALIBRATION_SEED = 1622026102281
VALIDATION_SEED = 1622026102282
CONDITIONAL_SEED = 1622026102283
INDEPENDENT_REFERENCE_SEED = 1622026102285
CALIBRATION_DIRECT = (0., .005, .02)
VALIDATION_DIRECT = (0., .002, .005, .02)
BUDGETS = {
    "fresh_small": (100000, 100000, 50000),
    "fresh_medium": (200000, 250000, 100000),
    "coherent_only": (100000, 1000000, 50000),
    "fresh_large": (200000, 1000000, 200000),
}


def partitions(ids, channel):
    """Random, disjoint generated-ID halves; prefix ranks precede all cuts."""
    ids = np.asarray(ids, np.int64)
    if ids.ndim != 1 or len(ids) < 2 or len(np.unique(ids)) != len(ids) or ids.min() < 0:
        raise ValueError("Need unique, nonnegative generated IDs")
    ordered = np.random.default_rng(np.random.SeedSequence([SPLIT_SEED, channel])).permutation(ids)
    return ordered[:len(ids)//2], ordered[len(ids)//2:]


def allocated_budgets(template_count, channel):
    """Cap nominal budgets at actual generated reactions, never at accepted MC.

    A rare incident proton without a reaction is not a rejected three-alpha
    reaction. Keep its absence out of the reaction-fraction denominator.
    """
    targets = {name: budget[channel] for name, budget in BUDGETS.items()}
    if template_count < .99*max(targets.values()):
        raise ValueError("Substantially fewer generated reactions than planned")
    return {name: min(target, template_count) for name, target in targets.items()}


def accumulate_sphere(values, counts, chunk):
    """Index original generator alphas by event, including boundary-split rows.

    No sequential-role cut: coherent labels are bookkeeping. Duplicated alpha
    indices invalidate a triplet rather than silently replacing an observation.
    """
    scan = mismatch.scan
    keep = ((chunk["pdg"] == scan.ALPHA_PDG)
            & (chunk["h11b_particle_source"] == scan.SOURCE_H11B_REACTION_PRODUCT)
            & (chunk["generator_particle_index"] >= 0)
            & (chunk["generator_particle_index"] <= 2))
    event = np.asarray(chunk["event_id"][keep], np.int64)
    index = np.asarray(chunk["generator_particle_index"][keep], np.int64)
    if len(event) and (event.min() < 0 or event.max() >= len(counts)):
        raise ValueError("Sphere event outside generated ID range")
    np.add.at(counts, (event, index), 1)
    fields = ("kinetic_energy_MeV", "px_MeV_c", "py_MeV_c", "pz_MeV_c", "x_mm", "y_mm", "z_mm")
    for number, field in enumerate(fields):
        values[event, index, number] = chunk[field][keep]


def reduce_library(path, ids, channel, scratch):
    """Stream ROOT into a temporary mmap, then respond in 100k-event blocks."""
    template, validation = partitions(ids, channel)
    size = int(np.max(ids)) + 1
    role = np.full(size, -1, np.int8)
    rank = np.full(size, -1, np.int32)
    role[template], role[validation] = 0, 1
    rank[template] = np.arange(len(template))
    support = mismatch.geometric_support(60)
    limits = allocated_budgets(len(template), channel)
    histograms = {name: np.zeros(support.sum(), np.int64) for name in limits}
    data_histogram = np.zeros(support.sum(), np.int64)
    summary = dict(template_generated=len(template), data_generated=len(validation),
        template_ids_sha256=finite.calibration.digest(template),
        data_ids_sha256=finite.calibration.digest(validation),
        template_prefix_generated=limits,
        template_prefix_ids_sha256={name: finite.calibration.digest(template[:limit]) for name, limit in limits.items()},
        complete=[0, 0], quality=[0, 0], accepted=[0, 0])
    with tempfile.TemporaryDirectory(prefix=f"sphere-{channel}-", dir=scratch) as directory:
        values = np.memmap(Path(directory)/"values.dat", mode="w+", dtype=np.float64, shape=(size, 3, 7))
        counts = np.zeros((size, 3), np.int32)
        fields = ["event_id", "pdg", "h11b_particle_source", "generator_particle_index",
                  "kinetic_energy_MeV", "px_MeV_c", "py_MeV_c", "pz_MeV_c", "x_mm", "y_mm", "z_mm"]
        with uproot.open(path, handler=uproot.source.file.MemmapSource) as stream:
            for chunk in stream["virtual_sphere"].iterate(fields, step_size=100000, library="np"):
                accumulate_sphere(values, counts, chunk)
        complete = np.all(counts == 1, axis=1)
        if np.any(complete & (role < 0)):
            raise RuntimeError("Sphere event has no generated reaction")
        rngs = [np.random.default_rng(np.random.SeedSequence([RESPONSE_SEED, channel, subset]))
                for subset in range(2)]
        for first in range(0, size, 100000):
            last = min(first+100000, size)
            for subset in range(2):
                event_ids = np.arange(first, last)[complete[first:last] & (role[first:last] == subset)]
                rows = np.asarray(values[event_ids]).copy()
                events = dict(event_id=event_ids, kinetic=rows[:, :, 0],
                              momentum=rows[:, :, 1:4], hit_position=rows[:, :, 4:7])
                summary["complete"][subset] += len(event_ids)
                events = mismatch.scan.apply_mask(events, mismatch.scan.quality_mask(events["kinetic"], events["momentum"], 1e-4))
                summary["quality"][subset] += len(events["event_id"])
                if not len(events["event_id"]):
                    continue
                rec, _ = mismatch.scan.reconstruct_with_granularity(events, 2., 2., 0., 0.,
                    "sphere-center", np.array([0., 0., 170.]))
                measured, _, _ = mismatch.scan.apply_closure_energy_response(rec, 100., 100.,
                    rngs[subset].standard_normal(rec["kinetic"].shape))
                xy = np.column_stack(mismatch.scan.ordered_dalitz_xy(measured))
                summary["accepted"][subset] += len(xy)
                if subset == 1:
                    data_histogram += mismatch.histogram_xy(xy, 60)[support].astype(np.int64)
                else:
                    selected_rank = rank[measured["event_id"]]
                    for name, limit in limits.items():
                        histograms[name] += mismatch.histogram_xy(xy[selected_rank < limit], 60)[support].astype(np.int64)
        del values
    if data_histogram.sum() != summary["accepted"][1]:
        raise RuntimeError("Reduction lost validation events")
    if histograms["fresh_large"].sum() != summary["accepted"][0]:
        raise RuntimeError("Reduction lost template events")
    return histograms, data_histogram, summary


def prepare_inputs(notebook=NOTEBOOK, scratch=None):
    document = json.loads(Path(notebook).read_text())
    libraries = document["metadata"][LIBRARY_KEY]
    if not libraries.get("completed_at"):
        raise ValueError("Independent simulation libraries are not finished")
    originals = document["metadata"]["codex_model_mismatch_libraries"]["runs"]
    fingerprint = dict(roots={name: libraries["runs"][name]["root_sha256"] for name in CHANNELS},
        original_roots={name: originals[name]["root_sha256"] for name in CHANNELS},
        split_seed=SPLIT_SEED, response_seed=RESPONSE_SEED, budgets=BUDGETS,
        reduction_source_sha256=sha256(__file__))
    for source in (libraries["runs"], originals):
        for name in CHANNELS:
            if sha256(ROOT/source[name]["root"]) != source[name]["root_sha256"]:
                raise RuntimeError(f"Changed ROOT: {source[name]['root']}")
    cache = Path(scratch)/"reduced_inputs.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["fingerprint"] != finite.calibration.plain(fingerprint):
            raise ValueError("Reduced input checkpoint changed")
        return cached["inputs"]
    del document
    old, _ = finite.prepare_inputs(notebook)
    new_hist = {name: [] for name in BUDGETS}
    data_hist, data_generated, summaries = [], [], {}
    for channel, name in enumerate(CHANNELS):
        row = libraries["runs"][name]
        path = ROOT / row["root"]
        if sha256(path) != row["root_sha256"]:
            raise RuntimeError(f"Changed input ROOT: {path}")
        ids = mismatch.scan.load_reaction_event_ids(path)
        hist, dh, summary = reduce_library(path, ids, channel, scratch)
        summaries[name] = summary
        for bank in BUDGETS:
            new_hist[bank].append(hist[bank])
        data_hist.append(dh)
        data_generated.append(summary["data_generated"])
        print(f"Reduced {name}: template/data G={summary['template_generated']}/{summary['data_generated']}; "
              f"accepted={summary['accepted']}", flush=True)
    banks = {"old": old["banks"]["coherent"]}
    for name in BUDGETS:
        generated = [summaries[channel]["template_prefix_generated"][name] for channel in CHANNELS]
        banks[name] = dict(raw_templates=np.array(new_hist[name]), generated=np.array(generated))
    old_bank = banks["old"]
    if not np.array_equal(old_bank["generated"], BUDGETS["fresh_small"]):
        raise ValueError("Old and replacement-small MC budgets must agree")
    # Calibration closure population only. Actual held-out data below stay separate.
    pooled = (np.array(new_hist["fresh_large"])+np.array(data_hist)
              +np.array(old_bank["raw_templates"])+np.array(old["data_histograms"]))
    pooled_g = (np.array(banks["fresh_large"]["generated"])+np.array(data_generated)
                +np.array(old_bank["generated"])+np.array(old["data_generated"]))
    truth = np.column_stack([pooled, pooled_g-pooled.sum(axis=1)])/pooled_g[:, None]
    independent_truth = np.column_stack([data_hist, np.array(data_generated)-np.sum(data_hist, axis=1)])/np.array(data_generated)[:, None]
    result = dict(banks=banks, truth=truth, independent_truth=independent_truth,
        data_histograms=data_hist, data_generated=data_generated, reference_generated=pooled_g,
        zero_reference_mc_events={name: [int(np.asarray(bank['raw_templates'])[c][np.asarray(data_hist)[c] == 0].sum())
                                         for c in range(3)] for name, bank in banks.items()},
        summaries=summaries, provenance=dict(roots={name: libraries["runs"][name]["root"] for name in CHANNELS},
            root_sha256=fingerprint["roots"], split_seed=SPLIT_SEED, response_seed=RESPONSE_SEED,
            response=old["provenance"]["response"], bins=60, support_bins=int(old["support"].sum()),
            truth_population="Declared pooled same-physics closure reference, including old/new MC and held-out samples",
            independent_population="New validation halves only, disjoint from every actual template bank",
            reference_scope="Both are finite empirical populations, not physical truth; no guaranteed physical conditional coverage"))
    result = finite.calibration.plain(result)
    cache.write_text(json.dumps(dict(fingerprint=fingerprint, inputs=result), ensure_ascii=False, allow_nan=False))
    return result


def bank_inputs(inputs, name):
    return dict(truth=inputs["truth"], banks={"coherent": inputs["banks"][name]})


def freeze_calibration(state, comparisons=16):
    if state["configuration"]["seed"] != CALIBRATION_SEED:
        raise ValueError("Only predeclared calibration trials can freeze cutoffs")
    points = state["records"]
    rank = finite.calibration.tolerance_rank(state["configuration"]["repetitions"], comparisons=comparisons)
    cutoffs, local = {}, {}
    for method in finite.METHODS:
        local[method] = {key: float(np.sort(row[method]["q_upper"])[rank["rank"]-1]) for key, row in points.items()}
        cutoffs[method] = dict(q_upper=max(local[method].values()),
            q0=float(np.sort(points["00000"][method]["q0"])[rank["rank"]-1]))
    return dict(cutoffs=cutoffs, local_upper=local, tolerance=rank,
        scope="Calibration at d=0,0.005,0.02 and r=0.05; multiplicity across two budgets and two methods")


def ensemble_checkpoint(inputs, path, direct_values, seed, repetitions, *, workers=4, fluctuate_mc=True):
    identity = finite.calibration.digest(inputs)
    path = Path(path)
    resume = None
    if path.exists():
        cached = json.loads(path.read_text())
        if cached["input_sha256"] != identity:
            raise ValueError("Ensemble input bank changed")
        resume = cached["state"]
    def save(value):
        pending = path.with_suffix(".pending")
        pending.write_text(json.dumps(finite.calibration.plain(dict(input_sha256=identity, state=value)), allow_nan=False))
        pending.replace(path)
    state = finite.ensemble(inputs, direct_values, seed, repetitions, 100000, workers,
        resume=resume, checkpoint=save, fluctuate_mc=fluctuate_mc)
    save(state)
    return state


def pack(value):
    payload = json.dumps(finite.calibration.plain(value), separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
    return dict(payload_zlib_base64=base64.b64encode(zlib.compress(payload, 9)).decode(),
        payload_sha256=hashlib.sha256(payload).hexdigest(), uncompressed_bytes=len(payload))


def unpack(envelope):
    payload = zlib.decompress(base64.b64decode(envelope["payload_zlib_base64"]))
    if hashlib.sha256(payload).hexdigest() != envelope["payload_sha256"]:
        raise ValueError("Compressed study records changed")
    return json.loads(payload)


def run(notebook=NOTEBOOK, *, scratch, workers=4, calibration_repetitions=1000, validation_repetitions=500):
    start = time.monotonic()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    configuration = dict(version=VERSION, calibration_direct=CALIBRATION_DIRECT,
        validation_direct=VALIDATION_DIRECT, calibration_repetitions=calibration_repetitions,
        validation_repetitions=validation_repetitions, data_budget=100000, budgets=BUDGETS,
        seeds=dict(split=SPLIT_SEED, response=RESPONSE_SEED, calibration=CALIBRATION_SEED,
                   validation=VALIDATION_SEED, conditional=CONDITIONAL_SEED,
                   independent_reference=INDEPENDENT_REFERENCE_SEED),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT/"g4/analysis/bayes/finite_template_likelihood.py",
            ROOT/"g4/analysis/bayes/finite_template_162_study.py",
            ROOT/"g4/analysis/bayes/generate_162_convergence_libraries.py",
            ROOT/"g4/analysis/bayes/generate_162_model_libraries.py",
            Path(mismatch.__file__), Path(mismatch.scan.__file__),
            Path(finite.calibration.__file__), Path(mismatch.boundary.__file__))})
    plan = scratch / "analysis_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != finite.calibration.plain(configuration):
        raise ValueError("Do not mix analysis protocols in one scratch directory")
    plan.write_text(json.dumps(finite.calibration.plain(configuration), indent=1))
    inputs = prepare_inputs(notebook, scratch)
    state = dict(configuration=configuration, inputs=inputs, started_at=datetime.now().astimezone().isoformat(),
                 frozen={}, matched_summary={}, conditional_summary={}, independent_summary={})
    state['source_snapshots'] = {name: (ROOT/name).read_text() for name in configuration['source_sha256']}
    records = {}
    for group, name in (("small", "fresh_small"), ("large", "fresh_large")):
        current = bank_inputs(inputs, name)
        cal = ensemble_checkpoint(current, scratch/f"calibration_{group}.json", CALIBRATION_DIRECT,
            CALIBRATION_SEED, calibration_repetitions, workers=workers)
        frozen = freeze_calibration(cal)
        state["frozen"][group] = frozen
        validation = ensemble_checkpoint(current, scratch/f"validation_{group}.json", VALIDATION_DIRECT,
            VALIDATION_SEED, validation_repetitions, workers=workers)
        state["matched_summary"][group] = finite.summarize(validation, frozen)
        records[f"calibration_{group}"], records[f"validation_{group}"] = cal, validation
    for name in ("old", "fresh_small", "fresh_large"):
        current = bank_inputs(inputs, name)
        conditional = ensemble_checkpoint(current, scratch/f"conditional_{name}.json", VALIDATION_DIRECT,
            CONDITIONAL_SEED, validation_repetitions, workers=workers, fluctuate_mc=False)
        group = "large" if name == "fresh_large" else "small"
        state["conditional_summary"][name] = finite.summarize(conditional, state["frozen"][group])
        records[f"conditional_{name}"] = conditional
        # Independent held-out empirical reference: transport the frozen cut,
        # never retune it with this stress test. This is not physical coverage.
        current = dict(current, truth=inputs["independent_truth"])
        independent = ensemble_checkpoint(current, scratch/f"independent_{name}.json", VALIDATION_DIRECT,
            INDEPENDENT_REFERENCE_SEED, validation_repetitions, workers=workers, fluctuate_mc=False)
        state["independent_summary"][name] = finite.summarize(independent, state["frozen"][group])
        records[f"independent_{name}"] = independent
    state["independent_expected"] = {}
    for direct in VALIDATION_DIRECT:
        key = f"{int(round(direct*1e6)):05d}"
        data = 100000*(finite.fractions(direct)@np.asarray(inputs["independent_truth"]))[:-1]
        state["independent_expected"][key] = {}
        for name, bank in inputs["banks"].items():
            state["independent_expected"][key][name] = finite.evaluate(data,
                np.asarray(bank["raw_templates"]), np.asarray(bank["generated"]), direct)
        print(f"Independent expected-count comparison: d={direct:g}", flush=True)
    state["raw_records"] = pack(records)
    state["elapsed_seconds"] = time.monotonic()-start
    state = finite.calibration.plain(state)
    state["checkpoint_sha256"] = finite.calibration.digest(state)
    state["completed_at"] = datetime.now().astimezone().isoformat()
    save_metadata(notebook, KEY, state)
    print(f"Convergence study complete: {state['elapsed_seconds']:.1f} s", flush=True)
    return state


def plots(state):
    """Three cached-only figures, numbered 13..15 by the main notebook."""
    import matplotlib.pyplot as plt
    figures = []
    names, labels = ("old", "fresh_small", "fresh_medium", "coherent_only", "fresh_large"), (
        "Old", "Fresh, same G", "Fresh, medium", "Fresh, alpha1 only", "Fresh, large")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for method, marker in (("fixed", "s"), ("joint", "o")):
        for ax, direct in zip(axes, (0., .005)):
            point = state["independent_expected"][f"{int(direct*1e6):05d}"]
            ax.plot(np.arange(len(names)), [100*point[name][method]["map"][2] for name in names],
                    marker=marker, label=method.capitalize())
            ax.axhline(100*direct, color="k", ls=":")
            ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right")
            ax.set(ylabel="Estimated uniform phase-space fraction (%)", title=f"Independent validation reference, true d={100*direct:g}%")
            ax.legend()
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, label in (("old", "Old bank"), ("fresh_small", "Fresh bank, same G"), ("fresh_large", "Fresh bank, large G")):
        keys = sorted(state["independent_summary"][name])
        x = np.array([int(k)/1e4 for k in keys])
        rows = [state["independent_summary"][name][k]["joint"] for k in keys]
        coverage = [100*r["coverage"]["rate"] for r in rows]
        lo = [100*(r["coverage"]["rate"]-r["coverage"]["wilson95"][0]) for r in rows]
        hi = [100*(r["coverage"]["wilson95"][1]-r["coverage"]["rate"]) for r in rows]
        axes[0].errorbar(x, coverage, yerr=np.maximum([lo, hi], 0), marker="o", capsize=3, label=label)
        axes[1].errorbar(x, [100*r["bias"] for r in rows], yerr=[100*r["mcse"] for r in rows], marker="o", capsize=3, label=label)
    axes[0].axhline(95, color="k", ls=":")
    axes[1].axhline(0, color="k", ls=":")
    axes[0].set(ylabel="Held-out-reference upper-limit inclusion (%)", ylim=(-1, 102))
    axes[1].set(ylabel="Held-out-reference mean bias (percentage points)")
    for ax in axes:
        ax.set_xlabel("True uniform phase-space fraction (%)")
        ax.legend()
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for group, label in (("small", "Small MC budget"), ("large", "Large MC budget")):
        keys = sorted(state["matched_summary"][group])
        x = [int(k)/1e4 for k in keys]
        rows = [state["matched_summary"][group][k]["joint"] for k in keys]
        for ax, statistic in zip(axes, ("coverage", "rejection")):
            rates = np.array([r[statistic]["rate"] for r in rows])
            interval = np.array([r[statistic]["wilson95"] for r in rows])
            errors = 100*np.maximum(np.vstack([rates-interval[:, 0], interval[:, 1]-rates]), 0)
            ax.errorbar(x, 100*rates, yerr=errors, fmt="o-", capsize=3, label=label)
    axes[0].axhline(95, color="k", ls=":")
    axes[0].set(ylabel="Matched unconditional upper-limit inclusion (%)", ylim=(-1, 102))
    axes[1].set(ylabel="Matched detection probability (%)", ylim=(-1, 101))
    for ax in axes:
        ax.set_xlabel("True uniform phase-space fraction (%)")
        ax.legend()
    figures.append(fig)
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(scratch=args.scratch, workers=args.workers)
