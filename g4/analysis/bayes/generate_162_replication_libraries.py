"""Independent global-seed replication of the matched 162-keV pure channels.

Explicit CLI only. Sequential simulation, <=8 threads, local ROOT output.
The existing physics/binary and the external-test protocol are fingerprinted
before the first new outcome. No existing research checkpoint is replaced.
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
import time

if __package__ in (None, ""):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from analysis.bayes.generate_162_convergence_libraries import save_metadata
from analysis.bayes.generate_162_model_libraries import ROOT, NOTEBOOK, macro_text, sha256, validate_library

KEY = "codex_independent_replication_libraries"
VERSION = "2026-10-23-v1-independent-global-seed-libraries"
PREFIX = "independent_replication_162_20261023"
INCIDENTS = {"coherent":2000000,"alpha0":400000,"phase_space":400000}
PROTOCOL = dict(bins=[30,60],generated=[200000,1000000,200000],data_budget=100000,
    direct=[0.,.002,.005,.02],calibration_direct=[0.,.005,.02],
    validation_repetitions=500,calibration_repetitions=1000,
    old_conditional_seed=1622026102293,new_conditional_seed=1622026102303,
    new_calibration_seed=1622026102301,new_validation_seed=1622026102302,
    split_seed=1622026102300,response_seed=1622026102304,
    new_calibration_comparisons=16,
    primary_cutoffs="Frozen old-reference full-budget cutoffs from binning/budget study",
    reference_template_crossing="old/old, old/new, new/old, new/new; paired within each reference",
    recalibration="New-reference matched closure reported separately; never selected after external results")


def previous_seeds(metadata):
    values = set()
    for key,state in metadata.items():
        if key==KEY or not isinstance(state,dict) or "libraries" not in key:
            continue
        for row in state.get("runs",{}).values():
            if "validation" in row and "seed" in row["validation"]:
                values.add(int(row["validation"]["seed"]))
    return sorted(values)


def generate(*,notebook=NOTEBOOK,scratch,threads=8):
    if not 1<=threads<=8:
        raise ValueError("Use 1..8 simulation threads")
    work = Path(scratch)
    work.mkdir(parents=True,exist_ok=True)
    document = json.loads(Path(notebook).read_text())
    metadata = document["metadata"]
    original = metadata["codex_template_convergence_libraries"]
    parent = metadata["codex_binning_budget_162_study"]
    binary = ROOT/'g4/build/HB'
    sources = dict(original['configuration']['source_sha256'])
    for name,checksum in sources.items():
        if sha256(ROOT/name)!=checksum:
            raise RuntimeError(f"Physics, geometry or original library source changed: {name}")
    if sha256(binary)!=original['configuration']['binary_sha256']:
        raise RuntimeError("Executable differs from the matched original simulation")
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    configuration = dict(version=VERSION,prefix=PREFIX,incidents=INCIDENTS,threads=threads,
        protocol=PROTOCOL,parent_checkpoint_sha256=parent['checkpoint_sha256'],
        frozen_old_cutoffs={str(b):parent['frozen'][f'{b}_x4'] for b in PROTOCOL['bins']},
        source_sha256=sources,binary_sha256=sha256(binary),forbidden_previous_seeds=previous_seeds(metadata),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    state = metadata.get(KEY)
    if state is None:
        state = dict(configuration=configuration,started_at=datetime.now().astimezone().isoformat(),runs={})
    elif state['configuration']!=configuration:
        raise ValueError("Do not change the frozen independent-replication generation protocol")
    del document,metadata
    save_metadata(notebook,KEY,state)
    base = (ROOT/'g4/macros/model_mismatch_162.mac').read_text()
    for name,incidents in INCIDENTS.items():
        if name in state['runs']:
            row = state['runs'][name]
            if sha256(ROOT/row['root'])!=row['root_sha256']:
                raise RuntimeError(f"Checkpointed replication ROOT changed: {name}")
            print(f'Verified completed {name}',flush=True)
            continue
        output = work/name
        final = ROOT/'g4/data'/f'{PREFIX}_{name}.root'
        merged = final.with_suffix('.partial.root')
        if final.exists() or merged.exists() or (output.exists() and list(output.glob('*.root'))):
            raise FileExistsError(f'Preserve and inspect uncheckpointed output: {name}')
        text = macro_text(base,name,incidents)
        macro = work/f'{name}.mac'
        macro.write_text(text)
        log_path = work/f'{name}.log'
        env = os.environ.copy()
        env.update(HB_DATA_DIR=str(output),OMP_NUM_THREADS='1')
        start = time.monotonic()
        print(f'Starting {name}: {incidents} incident protons; {threads} threads; {log_path}',flush=True)
        with log_path.open('w') as stream:
            process = subprocess.Popen([str(binary),str(macro),str(threads)],cwd=ROOT,env=env,
                                       stdout=stream,stderr=subprocess.STDOUT)
            while True:
                try:
                    code = process.wait(timeout=25)
                    break
                except subprocess.TimeoutExpired:
                    print(f'{name}: simulation running {time.monotonic()-start:.0f}s',flush=True)
        log = log_path.read_text()
        if code or re.search(r'COMMAND NOT FOUND|G4Exception-START|Fatal Exception|Illegal parameter',log):
            raise RuntimeError(f'Geant4 replication failed: {log_path}')
        if not re.search(r'^\s*Number of events processed\s*:\s*'+str(incidents)+r'\s*$',log,re.M):
            raise RuntimeError(f'Missing completed global event count: {log_path}')
        pieces = sorted(output.glob('*.root'))
        if len(pieces)!=threads:
            raise RuntimeError(f'Expected {threads} thread files, found {len(pieces)}')
        merge_log = work/f'{name}_hadd.log'
        hadd = shutil.which('hadd') or '/opt/root/6.34.10/bin/hadd'
        with merge_log.open('w') as stream:
            subprocess.run([hadd,str(merged),*map(str,pieces)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        checked = validate_library(merged,name,incidents)
        forbidden = set(configuration['forbidden_previous_seeds'])
        forbidden.update(row['validation']['seed'] for row in state['runs'].values())
        if checked['seed'] in forbidden:
            raise RuntimeError('New run reused a previous global seed; preserve the output')
        if checked['reactions']!=incidents:
            raise RuntimeError('Replication has a different generated-reaction budget; preserve and review')
        checksum = sha256(merged)
        merged.replace(final)
        state['runs'][name] = dict(root=str(final.relative_to(ROOT)),root_sha256=checksum,
            macro_text=text,macro_sha256=hashlib.sha256(text.encode()).hexdigest(),
            log=str(log_path),log_sha256=sha256(log_path),log_text=log,merge_log_text=merge_log.read_text(),
            thread_files=[dict(path=str(p),sha256=sha256(p)) for p in pieces],
            validation=checked,elapsed_seconds=time.monotonic()-start,checked_at=datetime.now().astimezone().isoformat())
        save_metadata(notebook,KEY,state)
        print(f"Completed {name}: {checked['reactions']} reactions, NEW seed {checked['seed']}; {time.monotonic()-start:.1f}s",flush=True)
    state['completed_at'] = datetime.now().astimezone().isoformat()
    save_metadata(notebook,KEY,state)
    print('Independent global-seed generation COMPLETE',flush=True)
    return state


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8,choices=range(1,9))
    args = parser.parse_args()
    generate(scratch=args.scratch,threads=args.threads)
