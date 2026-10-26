"""Independent high-statistics 162-keV samples for template convergence.

Explicit CLI only; sequential Geant4 runs, at most eight workers.  Physics,
transport and response geometry match the earlier model-mismatch libraries.
Checkpoints are local metadata in the existing 0003 notebook.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.bayes.generate_162_model_libraries import (
    NOTEBOOK, ROOT, macro_text, sha256, validate_library,
)

VERSION = "2026-10-22-v1-independent-convergence-libraries"
KEY = "codex_template_convergence_libraries"
INCIDENTS = {"coherent": 2000000, "alpha0": 400000, "phase_space": 400000}
PREFIX = "template_convergence_162_20261022"


def save_metadata(notebook, key, state):
    """Replace one key atomically; tolerate shared-folder protocol failures."""
    notebook = Path(notebook)
    temporary = notebook.with_suffix(".pending.ipynb")
    for attempt in range(3):
        try:
            document = json.loads(notebook.read_text())
            document["metadata"][key] = state
            content = json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
            temporary.write_text(content)
            if temporary.stat().st_size != len(content.encode()):
                raise OSError(errno.EIO, "Incomplete notebook checkpoint")
            checked = json.loads(temporary.read_text())
            if len(checked["cells"]) != len(document["cells"]):
                raise OSError(errno.EIO, "Notebook cell count changed")
            temporary.replace(notebook)
            return
        except OSError as error:
            if error.errno not in (errno.EPROTO, errno.EIO, errno.EBUSY, errno.EAGAIN, errno.ESTALE) or attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


def generate(*, notebook=NOTEBOOK, scratch=None, threads=8):
    if not 1 <= threads <= 8:
        raise ValueError("Use 1..8 simulation threads")
    document = json.loads(Path(notebook).read_text())
    original = document["metadata"]["codex_model_mismatch_libraries"]
    binary = ROOT / "g4/build/HB"
    sources = dict(original["configuration"]["source_sha256"])
    for name, checksum in sources.items():
        if sha256(ROOT / name) != checksum:
            raise RuntimeError(f"Physics/geometry changed relative to the original libraries: {name}")
    if sha256(binary) != original["configuration"]["binary_sha256"]:
        raise RuntimeError("Executable differs from the original matched simulation")
    sources["g4/HB.cc"] = sha256(ROOT / "g4/HB.cc")
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    configuration = dict(version=VERSION, incidents=INCIDENTS, threads=threads, prefix=PREFIX,
        source_sha256=sources, binary_sha256=sha256(binary),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        original_library_seeds={name: row["validation"]["seed"] for name, row in original["runs"].items()},
        partition="Disjoint template/validation halves allocated before detector selection")
    state = document["metadata"].get(KEY)
    if state is None:
        state = dict(configuration=configuration, started_at=datetime.now().astimezone().isoformat(), runs={})
    elif state["configuration"] != configuration:
        raise ValueError("Convergence generation checkpoint configuration changed")
    del document
    work = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix="11bp-template-generation-"))
    work.mkdir(parents=True, exist_ok=True)
    save_metadata(notebook, KEY, state)
    base = (ROOT / "g4/macros/model_mismatch_162.mac").read_text()
    for name, incidents in INCIDENTS.items():
        if name in state["runs"]:
            row = state["runs"][name]
            if sha256(ROOT / row["root"]) != row["root_sha256"]:
                raise RuntimeError(f"Changed convergence ROOT: {name}")
            print(f"Verified existing {name}", flush=True)
            continue
        output = work / name
        final = ROOT / "g4/data" / f"{PREFIX}_{name}.root"
        merged = final.with_suffix(".partial.root")
        if final.exists() or merged.exists() or (output.exists() and list(output.glob("*.root"))):
            raise FileExistsError(f"Uncheckpointed simulation output for {name}; preserve and inspect")
        macro = work / f"{name}.mac"
        text = macro_text(base, name, incidents)
        macro.write_text(text)
        env = os.environ.copy()
        env.update(HB_DATA_DIR=str(output), OMP_NUM_THREADS="1")
        start = time.monotonic()
        log_path = work / f"{name}.log"
        print(f"Starting {name}: {incidents} incidents, {threads} threads; {log_path}", flush=True)
        with log_path.open("w") as stream:
            process = subprocess.Popen([str(binary), str(macro), str(threads)], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=25)
                    break
                except subprocess.TimeoutExpired:
                    print(f"{name}: running for {time.monotonic()-start:.0f} s", flush=True)
        log = log_path.read_text()
        if code or re.search(r"COMMAND NOT FOUND|G4Exception-START|Fatal Exception|Illegal parameter", log):
            raise RuntimeError(f"Geant4 failed: {log_path}")
        if not re.search(r"^\s*Number of events processed\s*:\s*"+str(incidents)+r"\s*$", log, re.M):
            raise RuntimeError(f"Missing global completion count: {log_path}")
        pieces = sorted(output.glob("*.root"))
        if len(pieces) != threads:
            raise RuntimeError(f"Expected {threads} thread outputs, found {len(pieces)}")
        hadd = shutil.which("hadd") or "/opt/root/6.34.10/bin/hadd"
        with (work / f"{name}_hadd.log").open("w") as stream:
            subprocess.run([hadd, str(merged), *map(str, pieces)], stdout=stream,
                stderr=subprocess.STDOUT, check=True)
        checked = validate_library(merged, name, incidents)
        forbidden = set(configuration["original_library_seeds"].values())
        forbidden.update(row["validation"]["seed"] for row in state["runs"].values())
        if checked["seed"] in forbidden:
            raise RuntimeError("A new simulation reused a previous global seed")
        checksum = sha256(merged)
        merged.replace(final)
        state["runs"][name] = dict(root=str(final.relative_to(ROOT)), root_sha256=checksum,
            macro_text=text, macro_sha256=hashlib.sha256(text.encode()).hexdigest(),
            log=str(log_path), log_sha256=sha256(log_path),
            original_thread_files=[dict(path=str(p), sha256=sha256(p)) for p in pieces],
            elapsed_seconds=time.monotonic()-start, checked_at=datetime.now().astimezone().isoformat(),
            validation=checked)
        save_metadata(notebook, KEY, state)
        print(f"Completed {name}: {checked['reactions']} reactions, seed {checked['seed']}; "
              f"{time.monotonic()-start:.1f} s", flush=True)
    state["completed_at"] = datetime.now().astimezone().isoformat()
    save_metadata(notebook, KEY, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    generate(scratch=args.scratch, threads=args.threads)
