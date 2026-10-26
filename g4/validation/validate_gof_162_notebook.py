"""Execute only the new cache-reading cells in the actual Jupyter kernel.

Preserve every historical output. This does not rerun simulation or fitting.
Local kernel sockets require permission when invoked in a restricted sandbox.
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

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT/'jupyter/codex_0003_162keV_弱分支检验与区间校准.ipynb'
IDS = ('gof162-load', 'gof162-deviance', 'gof162-reporting', 'gof162-residuals')
INTERPRETER = '/home/xc/software/jupyter/.venv/bin/python'


def persist_staged(notebook, staged):
    """Apply an already validated notebook after checking its input has not changed."""
    path, staged = Path(notebook), Path(staged)
    serialized = staged.read_bytes()
    document = json.loads(serialized)
    audit = document['metadata']['codex_gof_notebook_revision']['kernel_validation']
    if (audit['executed_cell_ids'] != list(IDS) or audit['png_figures'] != 3
            or audit['errors'] != 0 or audit['interpreter'] != INTERPRETER):
        raise RuntimeError('Invalid staged kernel validation')
    if hashlib.sha256(path.read_bytes()).hexdigest() != audit['input_notebook_sha256']:
        raise RuntimeError('Notebook changed since validation; reconcile before applying outputs')
    nbformat.validate(document)
    pending = path.with_suffix('.pending.ipynb')
    print('Applying validated outputs; the original remains intact until replacement.', flush=True)
    pending.write_bytes(serialized)
    pending.replace(path)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def validate(notebook=NOTEBOOK, *, output_dir=Path('/tmp/11bp-gof-162'), baseline=None,
             stage_only=False):
    if sys.executable != INTERPRETER:
        raise RuntimeError('Use the project-specified Jupyter interpreter')
    path, output_dir = Path(notebook), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_bytes = path.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).digest()
    document = json.loads(original_bytes)
    del original_bytes
    print('Notebook loaded; executing four cache-only cells in the actual kernel.', flush=True)
    by_id = {c['id']: c for c in document['cells']}
    if any(name not in by_id or by_id[name]['cell_type'] != 'code' for name in IDS):
        raise ValueError('Missing new cache-only code cells')
    selected = [nbformat.from_dict(json.loads(json.dumps(by_id[name]))) for name in IDS]
    for cell in selected:
        if isinstance(cell.source, list):
            cell.source = ''.join(cell.source)
    temporary = nbformat.v4.new_notebook(cells=selected)
    temporary.metadata.kernelspec = dict(name='python3', display_name='Python 3 (Jupyter)', language='python')
    client = NotebookClient(temporary, kernel_name='python3', timeout=1800,
                            resources={'metadata': {'path': str(path.parent)}})
    client.execute()
    print('Actual kernel execution finished; checking figures and historical content.', flush=True)
    png_count = 0
    output_text = ''
    for cell in temporary.cells:
        for output in cell.get('outputs', []):
            if output.output_type == 'error':
                raise RuntimeError(f"Kernel error: {output.ename}: {output.evalue}")
            if output.output_type == 'stream':
                output_text += output.text
            if 'image/png' in output.get('data', {}):
                number = {'gof162-deviance': 30, 'gof162-reporting': 31, 'gof162-residuals': 32}[cell.id]
                (output_dir/f'kernel_figure_{number}.png').write_bytes(base64.b64decode(output.data['image/png']))
                png_count += 1
    if INTERPRETER not in output_text or png_count != 3:
        raise RuntimeError('Kernel identity or embedded figure validation failed')
    if hashlib.sha256(path.read_bytes()).digest() != original_sha256:
        raise RuntimeError('Notebook changed during kernel execution; reconcile before saving')
    for cell in temporary.cells:
        target = by_id[cell.id]
        target['outputs'] = json.loads(json.dumps(cell.outputs))
        target['execution_count'] = cell.execution_count
        target['metadata']['execution'] = dict(cell.metadata.get('execution', {}))
    audit = dict(completed_at=datetime.now().astimezone().isoformat(),
                 kernel='python3', interpreter=INTERPRETER,
                 executed_cell_ids=list(IDS), code_cells=4, png_figures=png_count, errors=0,
                 language_info=dict(temporary.metadata.get('language_info', {})),
                 stdout=output_text,
                 input_notebook_sha256=original_sha256.hex(),
                 validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if baseline is not None:
        old = json.loads(Path(baseline).read_text())
        edits = document['metadata']['codex_gof_notebook_revision']['historical_source_edits']
        for cell in old['cells']:
            current = by_id[cell['id']]
            if current.get('outputs', []) != cell.get('outputs', []):
                raise RuntimeError('Historical output changed')
            if cell['id'] not in edits and current != cell:
                raise RuntimeError('Unexpected historical cell edit')
            if cell['id'] in edits and edits[cell['id']] != cell['source']:
                raise RuntimeError('Historical source snapshot changed')
        for key, value in old['metadata'].items():
            if document['metadata'][key] != value:
                raise RuntimeError(f'Historical metadata changed: {key}')
        audit['historical_cells_checked'] = len(old['cells'])
        audit['historical_metadata_preserved'] = True
        del old
    document['metadata']['codex_gof_notebook_revision']['kernel_validation'] = audit
    nbformat.validate(document)
    staged = output_dir/'validated.ipynb'
    if staged.resolve() == path.resolve():
        raise ValueError('Staging must not overwrite the input notebook')
    staged.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False)+'\n')
    print('Historical content and structure checked; validated outputs staged at', staged, flush=True)
    print(output_text, flush=True)
    if not stage_only:
        persist_staged(path, staged)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=NOTEBOOK)
    parser.add_argument('--output-dir', type=Path, default=Path('/tmp/11bp-gof-162'))
    parser.add_argument('--baseline', type=Path)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument('--stage-only', action='store_true',
                           help='Execute and validate, then stage locally without rewriting the large notebook')
    operation.add_argument('--apply-staged', type=Path,
                           help='Apply a previously validated staged notebook without starting a kernel')
    args = parser.parse_args()
    if args.apply_staged is not None:
        persist_staged(args.notebook, args.apply_staged)
    else:
        validate(args.notebook, output_dir=args.output_dir, baseline=args.baseline,
                 stage_only=args.stage_only)
