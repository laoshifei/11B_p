"""Validate newly added literature-response cells using the actual Jupyter kernel.

Stage locally before applying outputs to the large project notebooks. Original
cells and metadata are checked against the saved baseline; only declared
historical source edits and new-cell outputs are permitted.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
INTERPRETER='/home/xc/software/jupyter/.venv/bin/python'
SCRATCH=Path('/tmp/11bp-literature-response-162')
REVISION='codex_laursen_notebook_revision'
READOUT_REVISION='codex_laursen_readout_revision'
THESIS_REVISION='codex_laursen_thesis_revision'
MATCHING_REVISION='codex_laursen_matching_revision'
ENERGY_REVISION='codex_laursen_energy_revision'
DECOMPOSITION_REVISION='codex_laursen_decomposition_revision'
MATERIAL_REVISION='codex_laursen_material_revision'
PIXEL_REVISION='codex_laursen_pixel_revision'
TOPOLOGY_REVISION='codex_laursen_topology_revision'
CLUSTER_REVISION='codex_laursen_cluster_revision'
CLUSTER_CONTROLS_REVISION='codex_laursen_cluster_controls_revision'
FIGURES={'laursen162-geometry':9,'laursen162-efficiency':10,'laursen162-dalitz':11,
         'laursen162-spectrum':12,'laursen162-sensitivity':13,'laursen162-thesis-figure':14,
         'laursen162-matching-efficiency':15,'laursen162-matching-spectrum':16,
         'laursen162-energy-residuals':17,'laursen162-energy-windows':18,
         'laursen162-decomposition-controls':19,'laursen162-decomposition-components':20,
         'laursen162-material-layers':21,'laursen162-material-angles':22,
         'laursen162-pixel-acceptance':23,'laursen162-pixel-cohort':24,
         'laursen162-topology-residuals':25,'laursen162-topology-controls':26,
         'laursen162-cluster-acceptance':27,'laursen162-cluster-cohort':28,
         'laursen162-cluster-controls-ablation':29,'laursen162-cluster-controls-numerical':30}


def phase_spec(phase,number):
    if phase=='response':return REVISION,'',6 if number==2 else 1,5 if number==2 else 0
    if phase=='readout':
        if number!=2:raise ValueError('The readout audit adds code only to notebook 0002')
        return READOUT_REVISION,'readout_',1,0
    if phase=='thesis':return THESIS_REVISION,'thesis_',2 if number==2 else 1,1 if number==2 else 0
    if phase=='matching':return MATCHING_REVISION,'matching_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='energy':return ENERGY_REVISION,'energy_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='decomposition':return DECOMPOSITION_REVISION,'decomposition_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='material':return MATERIAL_REVISION,'material_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='pixel':return PIXEL_REVISION,'pixel_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='topology':return TOPOLOGY_REVISION,'topology_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='cluster':return CLUSTER_REVISION,'cluster_',3 if number==2 else 1,2 if number==2 else 0
    if phase=='cluster_controls':return CLUSTER_CONTROLS_REVISION,'cluster_controls_',3 if number==2 else 1,2 if number==2 else 0
    raise ValueError('Unknown validation phase')


def notebook_path(number):
    paths=list((ROOT/'jupyter').glob(f'codex_{number:04d}_*.ipynb'))
    if len(paths)!=1:raise ValueError('Expected exactly one main notebook')
    return paths[0]


def check_history(document,baseline,revision=REVISION):
    previous=json.loads(Path(baseline).read_text())
    by_id={cell['id']:cell for cell in document['cells']}
    if len(by_id)!=len(document['cells']):raise ValueError('Duplicate cell IDs')
    edits=document['metadata'][revision]['historical_source_edits']
    for old in previous['cells']:
        cell=by_id[old['id']]
        if old.get('outputs',[])!=cell.get('outputs',[]):raise ValueError('Historical output changed')
        if old['id'] not in edits and old!=cell:raise ValueError('Unexpected historical cell edit')
        if old['id'] in edits and edits[old['id']]!=old['source']:raise ValueError('Historical source snapshot changed')
        if old['id'] in edits and {k:v for k,v in old.items() if k!='source'}!={k:v for k,v in cell.items() if k!='source'}:
            raise ValueError('Historical fields other than declared source changed')
    for key,value in previous['metadata'].items():
        if document['metadata'][key]!=value:raise ValueError(f'Historical metadata changed: {key}')
    return len(previous['cells'])


def write_verified(path,raw,expected_current_sha=None):
    """Avoid oversized shared-filesystem writes; verify before atomic replace."""
    path=Path(path);pending=path.with_suffix('.pending.ipynb')
    expected=hashlib.sha256(raw).hexdigest()
    view=memoryview(raw)
    with pending.open('wb',buffering=0) as stream:
        offset=0
        while offset<len(view):
            # The shared-folder driver can misreport writes at 1 MiB. Keep
            # transfers within 64 KiB and still verify the complete byte stream.
            chunk=view[offset:offset+65536]
            count=stream.write(chunk)
            if count is None or not 0<count<=len(chunk):
                raise OSError('Invalid notebook write length; original file preserved')
            offset+=count
    with pending.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=expected:
            raise OSError('Notebook write checksum mismatch; original file preserved')
    if expected_current_sha is not None:
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_current_sha:
                raise ValueError('Notebook changed during output writing; reconcile before applying')
    pending.replace(path)


def stage(number,scratch=SCRATCH,phase='response'):
    if sys.executable!=INTERPRETER:raise RuntimeError('Use the specified Jupyter interpreter')
    revision,prefix,expected_cells,expected_figures=phase_spec(phase,number)
    path=notebook_path(number);scratch=Path(scratch)
    raw=path.read_bytes();input_sha=hashlib.sha256(raw).hexdigest()
    document=json.loads(raw);del raw
    added_ids=document['metadata'][revision]['added_cell_ids']
    selected=[nbformat.from_dict(json.loads(json.dumps(c))) for c in document['cells']
              if c['cell_type']=='code' and c['id'] in added_ids]
    if len(selected)!=expected_cells:raise ValueError('Unexpected new code-cell count')
    for cell in selected:
        if isinstance(cell.source,list):cell.source=''.join(cell.source)
    temporary=nbformat.v4.new_notebook(cells=selected)
    temporary.metadata.kernelspec=dict(name='python3',display_name='Python 3 (Jupyter)',language='python')
    print(f'Executing notebook {number:04d}: {len(selected)} new cells.',flush=True)
    NotebookClient(temporary,kernel_name='python3',timeout=1800,
        resources={'metadata':{'path':str(path.parent)}}).execute()
    text='';figures=0
    by_id={c['id']:c for c in document['cells']}
    for cell in temporary.cells:
        for output in cell.get('outputs',[]):
            if output.output_type=='error':raise RuntimeError(f'{output.ename}: {output.evalue}')
            if output.output_type=='stream':text+=output.text
            if 'image/png' in output.get('data',{}):
                (scratch/f'kernel_figure_{FIGURES[cell.id]}.png').write_bytes(base64.b64decode(output.data['image/png']))
                figures+=1
        target=by_id[cell.id]
        target['outputs']=json.loads(json.dumps(cell.outputs))
        target['execution_count']=cell.execution_count
        target['metadata']['execution']=dict(cell.metadata.get('execution',{}))
    if INTERPRETER not in text or figures!=expected_figures:raise RuntimeError('Kernel or figure count mismatch')
    history=check_history(document,scratch/f'original_{prefix}{number:04d}.ipynb',revision)
    audit=dict(completed_at=datetime.now().astimezone().isoformat(),interpreter=INTERPRETER,
        input_notebook_sha256=input_sha,number=number,phase=phase,executed_cell_ids=[c.id for c in selected],
        figures=figures,errors=0,historical_cells_preserved=history,historical_metadata_preserved=True,
        stdout=text,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    document['metadata'][revision]['kernel_validation']=audit
    nbformat.validate(document)
    staged=scratch/f'validated_{prefix}{number:04d}.ipynb'
    write_verified(staged,(json.dumps(document,ensure_ascii=False,indent=1,allow_nan=False)+'\n').encode('utf-8'))
    print(text,flush=True)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    return staged


def apply(staged,phase='response'):
    raw=Path(staged).read_bytes();document=json.loads(raw)
    revision=phase_spec(phase,2)[0]
    audit=document['metadata'][revision]['kernel_validation']
    number=audit['number'];path=notebook_path(number)
    expected_figures=phase_spec(phase,number)[3]
    if (audit['interpreter']!=INTERPRETER or audit['errors'] or
            audit['figures']!=expected_figures or audit.get('phase','response')!=phase):
        raise ValueError('Invalid kernel validation')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=audit['input_notebook_sha256']:
        raise ValueError('Notebook changed since validation; reconcile before applying')
    nbformat.validate(document)
    write_verified(path,raw,audit['input_notebook_sha256'])
    print(f'Applied validated outputs to {path.name}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    task=parser.add_mutually_exclusive_group(required=True)
    task.add_argument('--number',type=int,choices=[1,2,3])
    task.add_argument('--apply-staged',type=Path)
    parser.add_argument('--scratch',type=Path,default=SCRATCH)
    parser.add_argument('--phase',choices=['response','readout','thesis','matching','energy','decomposition','material','pixel','topology','cluster','cluster_controls'],default='response')
    args=parser.parse_args()
    if args.apply_staged:apply(args.apply_staged,args.phase)
    else:stage(args.number,args.scratch,args.phase)
