"""Generate explicitly matched 162-keV libraries, only when run as a command.

Simulation metadata are checkpointed in the existing local main notebook.
ROOT files stay local. Different runs execute sequentially with <=8 threads.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "jupyter/codex_0003_162keV_弱分支检验与区间校准.ipynb"
VERSION = "2026-10-22-v1-matched-geant4-libraries"
CASES = {
    "alpha0": (0, 1, 2),
    "phase_space": (0, 3, 1),
    "legacy": (0, 2, 1),
    "incoherent": (2, 2, 1),
    "coherent": (3, 2, 2),
}
MODE_NAMES = {0: "legacyLegendreA2A4", 2: "incoherentL2BW", 3: "boseSymmetricL2BW"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checkpoint(notebook, state):
    """Preserve every existing notebook cell and output; replace one metadata key."""
    path = Path(notebook)
    document = json.loads(path.read_text())
    document["metadata"]["codex_model_mismatch_libraries"] = state
    temporary = path.with_suffix(".pending.ipynb")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def macro_text(base, name, incidents):
    mode, channel, _ = CASES[name]
    replacements = {
        "/run/beamOn 100000": f"/run/beamOn {incidents}",
        "/h11b/162AlphaDecayModel boseSymmetricL2BW": f"/h11b/162AlphaDecayModel {MODE_NAMES[mode]}",
        "/h11b/162Alpha0BranchingFraction 0": f"/h11b/162Alpha0BranchingFraction {int(channel == 1)}",
    }
    if channel == 3:
        replacements.update({"/h11b/enableDirectDecay false": "/h11b/enableDirectDecay true",
                             "/h11b/162SequentialDecayFraction 1": "/h11b/162SequentialDecayFraction 0"})
    for original, replacement in replacements.items():
        if base.count(original + "\n") != 1:
            raise ValueError(f"Missing or ambiguous macro command: {original}")
        base = base.replace(original + "\n", replacement + "\n")
    return base


def validate_library(path, name, incidents):
    mode, channel, _ = CASES[name]
    ids = []
    residual = 0.
    maximum_attempts = 0
    maximum_acceptance = 0.
    minimum_acceptance = 1.
    fields = ["event", "resonance_id", "h11b_reaction_channel", "projectile_kinetic_lab",
              "h11b162_alpha_decay_model", "h11b162_decay_model_used",
              "h11b162_labels_are_bookkeeping", "h11b162_bose_sampling_attempts",
              "h11b162_bose_incoherent_weight", "h11b162_bose_coherent_weight",
              "h11b162_bose_acceptance", "e_alpha1", "e_alpha2", "e_alpha3"]
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as stream:
        tree = stream["reaction"]
        if tree.num_entries == 0:
            raise RuntimeError("Empty reaction library")
        for data in tree.iterate(fields, step_size=50000, library="np"):
            ids.append(data["event"])
            for field, value in {"resonance_id": 162, "h11b_reaction_channel": channel,
                                 "h11b162_alpha_decay_model": mode,
                                 "h11b162_decay_model_used": mode if channel == 2 else 0,
                                 "h11b162_labels_are_bookkeeping": int(mode in (2, 3) and channel == 2)}.items():
                if not np.all(data[field] == value):
                    raise RuntimeError(f"Wrong {field} in {name}")
            energies = np.column_stack([data[f"e_alpha{i}"] for i in (1, 2, 3)])
            if not np.isfinite(energies).all() or not (energies > 0).all():
                raise RuntimeError("Invalid kinetic energies")
            residual = max(residual, float(np.max(abs(energies.sum(axis=1)-data["projectile_kinetic_lab"]-8.682885616791))))
            if mode in (2, 3):
                acceptance = data["h11b162_bose_acceptance"]
                if not (np.isfinite(acceptance).all() and np.all((acceptance >= 0) & (acceptance <= 1))):
                    raise RuntimeError("Bose acceptance exceeds its analytic bound")
                if not np.allclose(acceptance, data["h11b162_bose_coherent_weight"] /
                                   (3*data["h11b162_bose_incoherent_weight"]), rtol=1e-12, atol=1e-14):
                    raise RuntimeError("Wrong saved Bose weights")
                attempts = data["h11b162_bose_sampling_attempts"]
                if not np.all((attempts >= 1) & (attempts < 100000)):
                    raise RuntimeError("Sampling-attempt bound reached")
                maximum_attempts = max(maximum_attempts, int(attempts.max()))
                maximum_acceptance = max(maximum_acceptance, float(acceptance.max()))
                minimum_acceptance = min(minimum_acceptance, float(acceptance.min()))
        ids = np.concatenate(ids)
        if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= incidents:
            raise RuntimeError("Invalid global event IDs")
        if residual >= 1e-7:
            raise RuntimeError("Energy conservation failed")
        runinfo = {key: value.tolist() for key, value in stream["RunInfo"].arrays(library="np").items()}
        for field, value in {"output_schema_version": 9, "virtual_sphere_enabled": 1,
                             "configured_162_alpha_decay_model": mode,
                             "virtual_sphere_radius_mm": 25.}.items():
            if not np.all(np.asarray(runinfo[field]) == value):
                raise RuntimeError(f"Unexpected RunInfo {field}")
        seeds = sorted(set(map(int, runinfo["random_seed"])))
        if len(seeds) != 1:
            raise RuntimeError("Multiple global random seeds")
        sphere_entries = int(stream["virtual_sphere"].num_entries)
        if sphere_entries <= 0:
            raise RuntimeError("No virtual-sphere crossings")
    return dict(reactions=len(ids), incident_count=incidents, alpha_count=3*len(ids),
                event_range=[int(ids.min()), int(ids.max())], runinfo=runinfo, seed=seeds[0],
                sphere_entries=sphere_entries, max_energy_residual_MeV=residual,
                max_attempts=maximum_attempts, acceptance_range=[minimum_acceptance, maximum_acceptance]
                if mode in (2, 3) else None, all_passed=True)


def generate(events=100000, threads=8, *, notebook=NOTEBOOK, scratch=None, prefix="model_mismatch_162_20261022"):
    if int(events) != events or events < 2000 or not 1 <= threads <= 8:
        raise ValueError("Need at least 2000 base incidents and 1..8 threads")
    notebook = Path(notebook)
    base_path = ROOT / "g4/macros/model_mismatch_162.mac"
    base = base_path.read_text()
    binary = ROOT / "g4/build/HB"
    sources = ["g4/macros/model_mismatch_162.mac", "g4/CMakeLists.txt"]
    sources += [str(p.relative_to(ROOT)) for folder in ("include", "src")
                for p in sorted((ROOT/"g4"/folder).glob("*")) if p.suffix in (".hh", ".cc")]
    configuration = dict(version=VERSION, base_incidents=int(events), threads=threads, prefix=prefix,
                         git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                         source_sha256={name: sha256(ROOT/name) for name in sources}, binary_sha256=sha256(binary))
    previous = json.loads(notebook.read_text())["metadata"].get("codex_model_mismatch_libraries")
    if previous:
        if previous["configuration"] != configuration:
            raise ValueError("Library configuration changed; do not silently reuse existing results")
        state = previous
    else:
        state = dict(configuration=configuration, started_at=datetime.now().astimezone().isoformat(), runs={})
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="11bp-model-libraries-"))
    work.mkdir(parents=True, exist_ok=True)
    for name, (_, _, multiplier) in CASES.items():
        if name in state["runs"]:
            row = state["runs"][name]
            if sha256(ROOT/row["root"]) != row["root_sha256"]:
                raise RuntimeError(f"Existing {name} ROOT changed")
            print(f"Verified existing {name}", flush=True)
            continue
        incidents = int(events*multiplier)
        text = macro_text(base, name, incidents)
        macro = work / f"{name}.mac"
        macro.write_text(text)
        output = work / name
        if output.exists() and list(output.glob("*.root")):
            raise RuntimeError(f"Uncheckpointed output exists: {output}")
        final = ROOT / "g4/data" / f"{prefix}_{name}.root"
        if final.exists():
            raise FileExistsError(final)
        env = os.environ.copy()
        env["HB_DATA_DIR"] = str(output)
        env["OMP_NUM_THREADS"] = "1"
        start = time.monotonic()
        log_path = work / f"{name}.log"
        print(f"Starting {name}: {incidents} incident protons, {threads} threads", flush=True)
        with log_path.open("w") as log:
            process = subprocess.Popen([str(binary), str(macro), str(threads)], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=25)
                    break
                except subprocess.TimeoutExpired:
                    print(f"{name}: running for {time.monotonic()-start:.0f} s", flush=True)
        log = log_path.read_text()
        if code or re.search(r"COMMAND NOT FOUND|G4Exception-START|Fatal Exception|Illegal parameter", log):
            raise RuntimeError(f"Geant4 failed; inspect {log_path}")
        if not re.search(r"^\s*Number of events processed\s*:\s*"+str(incidents)+r"\s*$", log, re.M):
            raise RuntimeError(f"Missing completed global event count in {log_path}")
        pieces = sorted(output.glob("*.root"))
        if len(pieces) != threads:
            raise RuntimeError(f"Expected {threads} thread ROOT files, found {len(pieces)}")
        hadd = shutil.which("hadd") or "/opt/root/6.34.10/bin/hadd"
        with (work/f"{name}_hadd.log").open("w") as merge_log:
            merged = work / f"{name}_merged.root"
            subprocess.run([hadd, str(merged), *map(str, pieces)], stdout=merge_log, stderr=subprocess.STDOUT, check=True)
        checked = validate_library(merged, name, incidents)
        shutil.copy2(merged, final)
        digest = sha256(final)
        if digest != sha256(merged):
            raise RuntimeError("Promoted ROOT differs from its validated source")
        state["runs"][name] = dict(root=str(final.relative_to(ROOT)), root_sha256=digest,
            macro_text=text, macro_sha256=hashlib.sha256(text.encode()).hexdigest(),
            log=str(log_path), log_sha256=sha256(log_path), original_thread_files=[str(p) for p in pieces],
            elapsed_seconds=time.monotonic()-start, checked_at=datetime.now().astimezone().isoformat(),
            validation=checked)
        write_checkpoint(notebook, state)
        print(f"Completed {name}: {checked['reactions']} reactions, {checked['sphere_entries']} crossings, "
              f"{time.monotonic()-start:.1f} s; seed {checked['seed']}", flush=True)
    state["completed_at"] = datetime.now().astimezone().isoformat()
    write_checkpoint(notebook, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    generate(args.events, args.threads, scratch=args.scratch)
