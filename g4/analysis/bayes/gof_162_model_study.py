"""Calibrated model-adequacy gate for the existing 162-keV BW inference.

The auxiliary MC stays fixed at its actual integer counts. Independent BW
data calibrate the joint deviance; new data validate all four empirical laws.
An inadequate fit reports no constraint. Rejection is never counted as upper
limit coverage. Importing or plotting this module does not run fits or ROOT.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.special import xlogy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes import rmatrix_162_study as parent

ROOT, NOTEBOOK = parent.ROOT, parent.NOTEBOOK
digest, plain = parent.digest, parent.plain
KEY = "codex_gof_162_model_study"
VERSION = "2026-10-26-v1-conditional-mc-joint-deviance"
REFERENCES = ("old", "new")
MODELS = parent.MODELS
PLAN = dict(
    bins=30, r=.05, direct=[0., .005], data_budget=100000,
    calibration_repetitions=1000, validation_repetitions=500,
    seeds=dict(calibration=1622026102611, validation=1622026102612),
    coverage=.95, family_confidence=.95, comparisons=4,
    mc="Actual fixed pooled integer MC; conditional data trials only",
    statistic="Joint data-plus-MC deviance at the unrestricted mixture optimum",
    primary="Maximum of four prespecified BW order-statistic cutoffs; reject only above it",
    scope="Two empirical references, r=5%, d=0/0.5%, existing response and MC only",
)


def case_key(reference, direct, model):
    return f"{reference}:{direct:.3f}:{model}"


def check_inputs(inputs):
    mc = np.asarray(inputs["raw_templates"], float)
    generated = np.asarray(inputs["generated"], float)
    if (mc.shape != (3, 144) or generated.shape != (3,)
            or not np.isfinite(mc).all() or not np.isfinite(generated).all()
            or np.any(mc < 0) or np.any(mc != np.floor(mc))
            or np.any(generated != np.floor(generated)) or np.any(generated <= 0)
            or np.any(mc.sum(axis=1) <= 0) or np.any(mc.sum(axis=1) > generated)):
        raise ValueError("Expected actual integer auxiliary MC on the 144-bin support")
    for reference in REFERENCES:
        base = np.asarray(inputs["base_references"][reference], float)
        weighted = np.asarray(inputs["weighted"][reference]["probability"], float)
        for p, shape in [(base, (3, 145)), (weighted, (4, 145))]:
            if (p.shape != shape or not np.isfinite(p).all() or np.any(p < 0)
                    or not np.allclose(p.sum(axis=1), 1., atol=1e-11, rtol=0)):
                raise ValueError("Reference probabilities must include the rejection category")
    if any(not np.isfinite(inputs["cutoffs"][k]) or inputs["cutoffs"][k] <= 0
           for k in ("q0", "q_upper")):
        raise ValueError("Invalid frozen weak-component cutoffs")


def compact_inputs(state):
    source = state["inputs"]
    result = {k: source[k] for k in ("raw_templates", "generated", "cutoffs", "base_references")}
    result["weighted"] = {r: dict(probability=source["weighted"][r]["probability"])
                          for r in REFERENCES}
    check_inputs(result)
    return result


def draw_data(inputs, phase, reference, direct, model, index):
    if phase not in PLAN["seeds"] or reference not in REFERENCES or model not in MODELS:
        raise ValueError("Unknown trial identity")
    if direct not in PLAN["direct"] or index < 0 or int(index) != index:
        raise ValueError("Unknown injection or trial index")
    if phase == "calibration" and model != "bw":
        raise ValueError("Only BW trials may calibrate the adequacy gate")
    rng = np.random.default_rng([PLAN["seeds"][phase], REFERENCES.index(reference),
                                 MODELS.index(model), int(round(direct*1e6)), index])
    p = parent.mixture(inputs, reference, MODELS.index(model), direct)
    # The generated count is fixed. The accepted count fluctuates, and the
    # likelihood conditions on it; no physical beam exposure is inferred.
    return rng.multinomial(PLAN["data_budget"], p)[:-1]


def deviance_residuals(observed, expected):
    n, mu = np.asarray(observed, float), np.asarray(expected, float)
    if (n.shape != mu.shape or not np.isfinite(n).all() or not np.isfinite(mu).all()
            or np.any(n < 0) or np.any(mu < 0) or np.any((n > 0) & (mu == 0))):
        raise ValueError("Invalid observed/expected counts")
    pieces = 2*(xlogy(n, n)-xlogy(n, mu)-n+mu)
    if np.min(pieces) < -1e-7:
        raise ArithmeticError("Negative deviance contribution")
    return np.sign(n-mu)*np.sqrt(np.maximum(0., pieces))


def fit_counts(data, inputs, direct, *, diagnostic=False, tolerance=1e-4, start=None):
    mc, generated = np.asarray(inputs["raw_templates"]), np.asarray(inputs["generated"])
    if start is None:
        fixed = parent.finite.fixed_model(data, mc, generated)
        start = parent.finite.accepted_weights(fixed.mode, fixed.efficiencies)
    joint = parent.banks.JointMultinomial(data, mc, generated, start=start, tolerance=tolerance)
    optimum = joint.shape(joint.weights)
    expected = np.sum(data)*optimum["q"]
    residual = deviance_residuals(data, expected)
    mc_expected = mc.sum(axis=1)[:, None]*optimum["shapes"]
    mc_deviance = float(2*np.sum(xlogy(mc, mc)-xlogy(mc, mc_expected)))
    data_deviance = float(residual @ residual)
    total = float(2*joint.loss)
    if not np.isclose(data_deviance+mc_deviance, total, atol=2e-6, rtol=1e-9):
        raise ArithmeticError("Joint deviance does not equal data plus auxiliary-MC terms")
    q_upper = joint.statistic(direct, upper=True)
    result = dict(dhat=float(joint.mode[2]), accepted=int(np.sum(data)) if not diagnostic else float(np.sum(data)),
                  deviance=total, data_deviance=data_deviance, mc_deviance=mc_deviance,
                  q0=float(joint.q0), q_upper=float(q_upper),
                  shape_gap=joint.max_shape_gap, restarts=len(joint.shape_restart_history))
    if diagnostic:
        result.update(map=joint.mode, expected=expected, observed=data, residual=residual,
                      upper=joint.upper_limit(inputs["cutoffs"]["q_upper"]))
    return plain(result)


_INPUTS = None


def _initialize(inputs):
    global _INPUTS
    _INPUTS = inputs


def _chunk(task):
    phase, reference, direct, model, first, last = task
    output = []
    for index in range(first, last):
        try:
            data = draw_data(_INPUTS, phase, reference, direct, model, index)
            result = fit_counts(data, _INPUTS, direct)
        except Exception as error:
            raise RuntimeError(f"Failed trial retained as failure: {task}, index={index}") from error
        output.append(dict(index=index, **result))
    return output


def audit_case(rows, repetitions):
    indices = [r["index"] for r in rows]
    if sorted(indices) != list(range(repetitions)):
        raise ValueError("Incomplete or duplicated trial indices")
    for row in rows:
        for key in ("deviance", "data_deviance", "mc_deviance", "q0", "q_upper", "dhat", "shape_gap"):
            if not np.isfinite(row[key]) or row[key] < -1e-6:
                raise ValueError(f"Invalid {key} in trial {row['index']}")
        if row["dhat"] > 1 or row["shape_gap"] > 1.0001e-4:
            raise ValueError("Fit outside the fraction or numerical tolerance bounds")


def ensemble(inputs, phase, scratch, configuration_sha256, workers):
    repetitions = PLAN[f"{phase}_repetitions"]
    models = ("bw",) if phase == "calibration" else MODELS
    tasks = {}
    for reference in REFERENCES:
        for direct in PLAN["direct"]:
            for model in models:
                for first in range(0, repetitions, 25):
                    key = f"{case_key(reference, direct, model)}:{first:04d}"
                    tasks[key] = (phase, reference, direct, model, first, min(first+25, repetitions))
    path = Path(scratch)/f"{phase}.json"
    config = dict(phase=phase, configuration_sha256=configuration_sha256)
    state = json.loads(path.read_text()) if path.exists() else dict(configuration=config, chunks={})
    if state["configuration"] != config:
        raise ValueError("Checkpoint protocol or source changed")
    for key, rows in state["chunks"].items():
        if key not in tasks or [r["index"] for r in rows] != list(range(tasks[key][-2], tasks[key][-1])):
            raise ValueError("Corrupt resumable chunk")
    pending = {k: v for k, v in tasks.items() if k not in state["chunks"]}
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize, initargs=(inputs,)) as pool:
            futures = {pool.submit(_chunk, task): key for key, task in pending.items()}
            notice = time.monotonic()
            for future in as_completed(futures):
                state["chunks"][futures[future]] = future.result()
                if len(state["chunks"]) % 8 == 0 or len(state["chunks"]) == len(tasks):
                    temporary = path.with_suffix(".pending")
                    temporary.write_text(json.dumps(state, allow_nan=False))
                    temporary.replace(path)
                if time.monotonic()-notice > 20:
                    print(f"{phase}: {len(state['chunks'])}/{len(tasks)} chunks", flush=True)
                    notice = time.monotonic()
    records = {}
    for reference in REFERENCES:
        for direct in PLAN["direct"]:
            for model in models:
                key = case_key(reference, direct, model)
                rows = [row for task in sorted(tasks) if task.startswith(key+":") for row in state["chunks"][task]]
                audit_case(rows, repetitions)
                records[key] = rows
    return dict(phase=phase, records=records)


def freeze_gof(calibration):
    if calibration["phase"] != "calibration":
        raise ValueError("Validation data cannot set a cutoff")
    expected = {case_key(r, d, "bw") for r in REFERENCES for d in PLAN["direct"]}
    if set(calibration["records"]) != expected:
        raise ValueError("Missing or extra calibration population")
    rank = parent.finite.calibration.tolerance_rank(
        PLAN["calibration_repetitions"], coverage=PLAN["coverage"],
        family_confidence=PLAN["family_confidence"], comparisons=PLAN["comparisons"])
    local = {}
    for key, rows in calibration["records"].items():
        audit_case(rows, PLAN["calibration_repetitions"])
        local[key] = float(np.sort([r["deviance"] for r in rows])[rank["rank"]-1])
    return dict(cutoff=max(local.values()), local=local, tolerance=rank,
                calibration_sha256=digest(calibration), rule="reject if D > cutoff; ties pass")


def rate(count, total):
    if not 0 <= count <= total:
        raise ValueError("Invalid rate denominator")
    return dict(count=int(count), total=int(total), rate=count/total if total else None,
                wilson=parent.wilson(count, total) if total else None)


def summarize_case(rows, cutoff, weak_cutoffs):
    accepted = np.array([r["deviance"] <= cutoff for r in rows])
    included = np.array([r["q_upper"] <= weak_cutoffs["q_upper"] for r in rows])
    discovery = np.array([r["q0"] > weak_cutoffs["q0"] for r in rows])
    n, passed = len(rows), int(accepted.sum())
    good, bad = int(np.sum(accepted & included)), int(np.sum(accepted & ~included))
    if good+bad+n-passed != n:
        raise ArithmeticError("Reporting outcomes do not partition all trials")
    return dict(
        inadequacy=rate(n-passed, n), report=rate(passed, n),
        raw_inclusion=rate(int(included.sum()), n),
        reported_inclusion=rate(good, passed),
        reported_containing=rate(good, n), reported_noncontaining=rate(bad, n),
        raw_discovery=rate(int(discovery.sum()), n),
        reported_discovery=rate(int(np.sum(accepted & discovery)), n),
        discovery_among_reports=rate(int(np.sum(accepted & discovery)), passed),
        deviance_quantiles=np.quantile([r["deviance"] for r in rows], [.025, .5, .975]).tolist(),
        mean_dhat=float(np.mean([r["dhat"] for r in rows])),
    )


def numerical_controls(inputs):
    checks = []
    for reference in REFERENCES:
        for model, direct in [("bw", 0.), ("bw", .005), ("rmatrix_no_phase", 0.),
                              ("rmatrix", .005), ("rmatrix_coulomb", .005)]:
            data = draw_data(inputs, "validation", reference, direct, model, 17)
            baseline = fit_counts(data, inputs, direct)
            for start in ([.02, .96, .02], [.08, .915, .005]):
                refined = fit_counts(data, inputs, direct, start=start, tolerance=1e-6)
                differences = {k: abs(refined[k]-baseline[k])
                               for k in ("dhat", "deviance", "q0", "q_upper")}
                if max(differences.values()) > 2e-4:
                    raise ArithmeticError(f"Numerical refinement failed: {differences}")
                checks.append(dict(reference=reference, model=model, direct=direct, index=17,
                                   start=start, differences=differences))
    return checks


def run(scratch, *, workers=8, notebook=NOTEBOOK):
    if not 1 <= workers <= 8:
        raise ValueError("Use 1..8 workers")
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    if KEY in document["metadata"]:
        raise ValueError("Preserve the completed study; load its cache instead")
    source = document["metadata"][parent.KEY]
    if not source.get("completed_at") or digest({k: v for k, v in source.items() if k != "checkpoint_sha256"}) != source["checkpoint_sha256"]:
        raise ValueError("Incomplete or modified parent study")
    inputs = compact_inputs(source)
    modules = [Path(__file__), Path(__file__).with_name("rmatrix_162_study.py"),
               Path(__file__).with_name("rmatrix_162_model.py"),
               Path(__file__).with_name("model_mismatch_162_study.py"),
               Path(__file__).with_name("finite_template_likelihood.py"),
               Path(__file__).with_name("disjoint_template_banks_162_study.py"),
               Path(__file__).with_name("finite_template_162_study.py"),
               Path(__file__).with_name("calibrate_162_weak_branch_intervals.py"),
               Path(__file__).with_name("diagnose_162_boundary_intervals.py"),
               Path(__file__).with_name("pooled_r_calibration_162_study.py")]
    snapshots = {str(p.relative_to(ROOT)): p.read_text() for p in modules}
    configuration = plain(dict(version=VERSION, protocol=PLAN, workers=workers,
        parent_checkpoint=source["checkpoint_sha256"], input_sha256=digest(inputs),
        source_sha256={k: parent.convergence.sha256(ROOT/k) for k in snapshots},
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        interpreter=sys.executable, python=sys.version,
        packages={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "matplotlib", "uproot")}))
    config_digest = digest(configuration)
    protocol_file = scratch/"protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != configuration:
        raise ValueError("Prespecified protocol changed")
    protocol_file.write_text(json.dumps(configuration, ensure_ascii=False, indent=2))
    started = datetime.now().astimezone().isoformat()
    start_time = time.monotonic()
    calibration = ensemble(inputs, "calibration", scratch, config_digest, workers)
    frozen = freeze_gof(calibration)
    frozen_path = scratch/"frozen.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError("Frozen adequacy cutoff changed")
    frozen_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    print("FROZEN before validation:", frozen, flush=True)
    validation = ensemble(inputs, "validation", scratch, config_digest, workers)
    summary = {k: summarize_case(rows, frozen["cutoff"], inputs["cutoffs"])
               for k, rows in validation["records"].items()}
    expected = {}
    for reference in REFERENCES:
        for model in MODELS:
            p = parent.mixture(inputs, reference, MODELS.index(model), .005)
            row = fit_counts(PLAN["data_budget"]*p[:-1], inputs, .005, diagnostic=True)
            previous = source["asimov"][reference]["0.005"][model]
            if abs(row["deviance"]-previous["deviance"]) > 2e-4 or abs(row["upper"]-previous["upper"]) > 1e-6:
                raise ArithmeticError("Historical expected-count fit no longer reproduces")
            expected[case_key(reference, .005, model)] = row
    controls = numerical_controls(inputs)
    result = plain(dict(configuration=configuration, inputs=inputs, source_snapshots=snapshots,
        calibration=calibration, frozen=frozen, validation=validation, summary=summary,
        expected=expected, numerical_controls=controls, started_at=started,
        completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.monotonic()-start_time))
    result["checkpoint_sha256"] = digest(result)
    # Re-read at persistence time so other notebook content is preserved.
    current = json.loads(Path(notebook).read_text())
    if KEY in current["metadata"] or current["metadata"][parent.KEY]["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise ValueError("Notebook parent or destination changed during calculation")
    parent.banks.save_metadata(notebook, KEY, result)
    (scratch/"completed.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    print("Model adequacy study COMPLETE", result["completed_at"], flush=True)
    return result


def load_cached(notebook=NOTEBOOK):
    document = json.loads(Path(notebook).read_text())
    state = document["metadata"][KEY]
    if digest({k: v for k, v in state.items() if k != "checkpoint_sha256"}) != state["checkpoint_sha256"]:
        raise ValueError("Study checksum changed")
    if document["metadata"][parent.KEY]["checkpoint_sha256"] != state["configuration"]["parent_checkpoint"]:
        raise ValueError("Parent provenance changed")
    return state


def plot_deviance(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained", sharey=True)
    labels = ["BW", "R, no external phase", "R + external phases", "R + phases + Coulomb"]
    colors = ["#202020", "#d98b16", "#2475b4", "#b13d3d"]
    for i, (reference, ax) in enumerate(zip(REFERENCES, axes)):
        for model, label, color in zip(MODELS, labels, colors):
            rows = state["validation"]["records"][case_key(reference, .005, model)]
            x = np.sort([r["deviance"] for r in rows])
            ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", color=color, label=label)
        ax.axvline(state["frozen"]["cutoff"], color="#777777", ls="--", label="Frozen BW cutoff")
        ax.set(xlabel="Joint deviance D (log scale)", xscale="log", ylim=(0, 1.02),
               title=f"Figure 30({'ab'[i]}): {reference} reference")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("162 keV; 0.5% injection; 500 independent validation trials per curve")
    return fig


def plot_reporting(state):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    labels = ["BW", "R, no phase", "R + phases", "R + Coulomb"]
    x = np.arange(4)
    for reference, offset, color in [("old", -.09, "#2475b4"), ("new", .09, "#b13d3d")]:
        for direct, shift, marker in [(0., -.025, "o"), (.005, .025, "s")]:
            rows = [state["summary"][case_key(reference, direct, m)]["inadequacy"] for m in MODELS]
            y = 100*np.array([r["rate"] for r in rows])
            bounds = 100*np.array([r["wilson"] for r in rows]).T
            axes[0].errorbar(x+offset+shift, y,
                yerr=np.maximum(0, np.stack([y-bounds[0], bounds[1]-y])), fmt=marker,
                color=color, label=f"{reference}, d={direct*100:g}%", capsize=2)
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set(ylabel="Model rejected (%)", ylim=(-3, 103), title="Figure 31(a): adequacy rejection")
    axes[0].axhline(5, color="gray", ls=":")
    axes[0].legend(fontsize=8)
    positions = np.arange(8)
    rows = [state["summary"][case_key(r, .005, m)] for m in MODELS for r in REFERENCES]
    bottom = np.zeros(8)
    for field, label, color in [("reported_containing", "Report; upper limit contains injection", "#248777"),
                                 ("reported_noncontaining", "Report; upper limit misses injection", "#b13d3d"),
                                 ("inadequacy", "No constraint: model inadequate", "#c9c9c9")]:
        y = 100*np.array([r[field]["rate"] for r in rows])
        axes[1].bar(positions, y, bottom=bottom, color=color, width=.72, label=label)
        bottom += y
    axes[1].set_xticks(positions, [f"{m}\n{r}" for m in labels for r in REFERENCES], rotation=20)
    axes[1].set(ylabel="Fraction of all trials (%)", ylim=(0, 103),
                title="Figure 31(b): complete reporting outcomes, d=0.5%")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("A rejected model supplies no upper-limit constraint; denominators are kept explicit")
    return fig


def plot_residuals(state):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    support = parent.mismatch.geometric_support(PLAN["bins"])
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.6), layout="constrained", sharex=True, sharey=True)
    maximum = max(float(np.max(np.abs(r["residual"]))) for r in state["expected"].values())
    normalization = TwoSlopeNorm(vcenter=0, vmin=-maximum, vmax=maximum)
    labels = ["BW", "R, no external phase", "R + external phases", "R + phases + Coulomb"]
    for row, reference in enumerate(REFERENCES):
        for col, (model, label) in enumerate(zip(MODELS, labels)):
            result = state["expected"][case_key(reference, .005, model)]
            plane = np.full(support.shape, np.nan)
            plane[support] = result["residual"]
            ax = axes[row, col]
            im = ax.imshow(plane.T, origin="lower", extent=[*parent.mismatch.LIMITS]*2,
                           cmap="RdBu_r", norm=normalization)
            ax.set_title(f"Figure 32({'abcdefgh'[4*row+col]}): {reference}\n{label}", fontsize=10)
            ax.set(xlim=(0, 1.05), ylim=(0, 1.05))
            if row == 1:
                ax.set_xlabel("Ordered Dalitz x")
            if col == 0:
                ax.set_ylabel("Ordered Dalitz y")
    fig.colorbar(im, ax=axes, label="Signed data deviance residual (descriptive, not a z score)", shrink=.8)
    fig.suptitle("Expected counts, d=0.5%; response-folded reference minus profiled BW fit")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.scratch, workers=args.workers)
