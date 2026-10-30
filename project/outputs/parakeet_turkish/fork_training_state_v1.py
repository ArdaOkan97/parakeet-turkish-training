"""Preserve a completed run, then fork its full training state under a new ID.

Only the top-level ``run_id`` changes in the two serialized checkpoints. No
training, model allocation, inference, optimizer reset, or CUDA call occurs.
The immutable manifest is committed last; an incomplete fork must never run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUTPUT))
from atomic_io_retry_v1 import commit_writer, read_bytes_shared, write_json
import torch


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(read_bytes_shared(path).decode('utf-8'))


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


class Checks:
    def __init__(self):
        self.count = 0
        self.tensor_count = 0

    def require(self, condition, message):
        self.count += 1
        if not condition:
            raise ValueError(message)


def exact_equal(left, right, check: Checks, path='state'):
    """Compare every leaf, preserving container type, tensor dtype and values."""
    check.require(type(left) is type(right), f'{path}: type differs')
    if isinstance(left, torch.Tensor):
        check.tensor_count += 1
        check.require(left.device.type == right.device.type == 'cpu', f'{path}: non-CPU tensor')
        check.require(left.dtype == right.dtype and left.shape == right.shape,
                      f'{path}: tensor dtype/shape differs')
        check.require(torch.equal(left, right), f'{path}: tensor values differ')
    elif isinstance(left, dict):
        check.require(list(left.keys()) == list(right.keys()), f'{path}: ordered keys differ')
        for key in left:
            exact_equal(left[key], right[key], check, f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        check.require(len(left) == len(right), f'{path}: sequence length differs')
        for index, (a, b) in enumerate(zip(left, right)):
            exact_equal(a, b, check, f'{path}[{index}]')
    else:
        check.require(left == right, f'{path}: value differs')


def validate_fork(source, fork, source_id, target_id, check: Checks):
    check.require(source_id != target_id, 'Source and target IDs must differ')
    check.require(source['run_id'] == source_id and fork['run_id'] == target_id,
                  'Fork/source run identity differs')
    check.require(list(source.keys()) == list(fork.keys()), 'Top-level state keys differ')
    for key in source:
        if key != 'run_id':
            exact_equal(source[key], fork[key], check, key)


def checked_id(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', value):
        raise argparse.ArgumentTypeError('Run IDs must contain letters, digits, underscores or hyphens')
    return value


def inside(path: Path, parent: Path):
    return path.resolve().is_relative_to(parent.resolve())


def copy_verified(source: Path, destination: Path, records: dict, check: Checks):
    check.require(source.is_file() and not source.is_symlink(), f'Snapshot source is not a regular file: {source}')
    check.require(not destination.exists(), f'Snapshot destination exists: {destination}')
    before = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The complete immutable snapshot is referenced only by the last manifest.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    check.require(before == digest(destination) == digest(source), f'Snapshot bytes/source changed: {source}')
    records[str(source.resolve())] = {'sha256': before, 'size_bytes': source.stat().st_size,
                                    'snapshot_path': str(destination.resolve())}


def copy_tree(source: Path, destination: Path, records: dict, check: Checks):
    check.require(source.is_dir() and not source.is_symlink(), f'Snapshot source directory missing: {source}')
    for p in sorted(source.rglob('*')):
        check.require(not p.is_symlink(), f'Snapshot symlink refused: {p}')
        if p.is_file():
            copy_verified(p, destination / p.relative_to(source), records, check)


def run(args):
    torch.set_num_threads(2)
    check = Checks()
    source_id, target_id = args.source_run_id, args.target_run_id
    check.require(source_id != target_id, 'Refusing to overwrite an existing run')
    source_run = OUTPUT / 'runs' / source_id
    target_run = OUTPUT / 'runs' / target_id
    target_latest = ROOT / 'work' / (target_id + '_latest.pt')
    target_best = ROOT / 'work' / (target_id + '_best.pt')
    snapshot = args.output_dir.resolve()
    check.require(inside(snapshot, OUTPUT / 'runs'), 'Snapshot must be inside the workspace runs directory')
    check.require(not inside(snapshot, source_run), 'Snapshot cannot be inside the source run')
    check.require(not target_run.exists(), 'Target run directory already exists')
    check.require(not target_latest.exists() and not target_best.exists(), 'Target checkpoint already exists')
    check.require(not snapshot.exists(), 'Snapshot directory already exists; incomplete outputs are retained for diagnosis')
    inherited_steps = sorted({args.expected_best_step, args.expected_latest_step})
    for step in inherited_steps:
        check.require(not (OUTPUT/'results'/(target_id+'_step_'+str(step))).exists(),
                      f'Target inherited DEV folder already exists at step{step}')
    verification = read_json(args.source_verification)
    check.require(verification.get('passed') is True and verification.get('status') == 'complete',
                  'Source final verification must be complete and passed')
    check.require(verification.get('run_id') == source_id and verification.get('selected_step') == args.expected_best_step,
                  'Source final verification identity/selected step differs')
    source_loop = OUTPUT / 'runs' / verification['loop_id']
    loop_state = read_json(source_loop / 'loop_status.json')
    check.require(loop_state.get('status') == 'complete', 'Source controller must have completed before preservation')
    check.require(loop_state['final_selection']['run_id'] == source_id, 'Source controller final winner differs')
    receipt = read_json(args.predecessor_exit_receipt)
    check.require(receipt['kind'] == 'root_owned_predecessor_exit_receipt_v1'
                  and receipt['root_observed_owned_session_completed'] is True
                  and receipt['root_observed_exit_code'] == 0, 'Owned predecessor exit receipt is invalid')
    check.require(receipt['predecessor_loop_id'] == verification['loop_id']
                  and receipt['predecessor_status'] == 'complete'
                  and receipt['predecessor_audit_passed'] is True, 'Predecessor receipt identity/status differs')
    check.require(receipt['predecessor_audit_sha256'] == digest(args.source_verification)
                  and receipt['predecessor_status_sha256'] == digest(source_loop/'loop_status.json'),
                  'Predecessor receipt audit/status bytes differ')
    selection = read_json(source_run / 'training_selection.json')
    config = read_json(source_run / 'run_config.json')
    check.require(selection['run_id'] == config['run_id'] == source_id, 'Source run identity differs')
    check.require(selection['completed_steps'] == args.expected_latest_step
                  and selection['selected_step'] == args.expected_best_step, 'Source selection step differs')
    check.require(selection['accepted'] is True, 'Source selected DEV model was rejected')
    check.require(read_json(OUTPUT/'checkpoint/finalization_provenance.json')['run_id'] == source_id,
                  'Published checkpoint no longer identifies the source run')
    candidate = Path(receipt['immutable_prior_candidate']).resolve()
    check.require(inside(candidate, ROOT/'work/models'), 'Preserved export candidate must be inside work/models')
    hf_files = receipt['prior_hf_files_sha256']
    check.require(set(hf_files) == {'config.json','finalization_provenance.json','generation_config.json',
                                   'model.safetensors','processor_config.json','tokenizer.json','tokenizer_config.json'},
                  'Prior export file set differs')
    for name, expected in hf_files.items():
        check.require(digest(candidate/name) == digest(OUTPUT/'checkpoint'/name) == expected,
                      f'Preserved/published export differs: {name}')
    for name, expected in config['code_sha256'].items():
        check.require(digest(OUTPUT / name) == expected, f'Frozen source changed: {name}')
    source_audit_sha = digest(args.source_verification)

    sources = {}
    states = {}
    for kind, step in [('latest', args.expected_latest_step), ('best', args.expected_best_step)]:
        p = ROOT / 'work' / (source_id + '_' + kind + '.pt')
        data = read_bytes_shared(p)
        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        check.require(state['schema'] == 2 and state['run_id'] == source_id and state['step'] == step,
                      f'{kind} checkpoint identity/schema differs')
        check.require(state['selected_step'] == args.expected_best_step, f'{kind} selected step differs')
        check.require(state['config_identity'] == config['config_identity'], f'{kind} config identity differs')
        check.require(len(state['weights']) == len(state['optimizer']['state']) == 207,
                      f'{kind} full weights/optimizer missing')
        check.require([x['step'] for x in state['history']] == list(range(1, step + 1)),
                      f'{kind} history is not contiguous')
        sources[kind] = {'path': str(p.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
                         'size_bytes': len(data), 'step': step}
        states[kind] = state
        del data
    check.require(states['latest']['best_score'] == states['best']['best_score'], 'Latest/best DEV score differs')
    check.require(states['latest']['history'] == read_json(source_run / 'training_history.json'),
                  'Latest checkpoint history differs from durable source history')
    snapshot.mkdir(parents=True, exist_ok=False)
    records = {}
    copy_tree(source_run, snapshot/'source_run', records, check)
    copy_tree(source_loop, snapshot/'source_loop', records, check)
    # The root receipt pins the existing immutable, byte-identical 2.5GB
    # candidate. Retain that model rather than making a redundant model copy.
    for name in hf_files:
        if name != 'model.safetensors':
            copy_verified(OUTPUT/'checkpoint'/name, snapshot/'published_checkpoint_metadata'/name, records, check)
    copy_verified(args.predecessor_exit_receipt, snapshot/'predecessor_exit_receipt.json', records, check)
    for folder in sorted((OUTPUT/'results').glob(source_id+'*')):
        if folder.is_dir():
            copy_tree(folder, snapshot/'results'/folder.name, records, check)
    strong_folder = OUTPUT/'results/strong_whisper_large_v3_greedy_test_v2'
    check.require(strong_folder.exists(), 'Verified strong TEST result folder is missing')
    copy_tree(strong_folder, snapshot/'results'/strong_folder.name, records, check)
    # These shared pointers/reports may legitimately change in a later run.
    for p in sorted(OUTPUT.iterdir()):
        if p.is_file() and p.suffix in ('.json','.txt'):
            copy_verified(p, snapshot/'top_level'/p.name, records, check)
    for kind in ['latest','best']:
        p = Path(sources[kind]['path'])
        copy_verified(p, snapshot/'source_checkpoints'/p.name, records, check)

    forks = {}
    for kind, destination in [('latest',target_latest), ('best',target_best)]:
        original = states[kind]
        fork = dict(original)
        fork['run_id'] = target_id
        before_checks, before_tensors = check.count, check.tensor_count
        validate_fork(original, fork, source_id, target_id, check)
        commit_writer(destination, lambda temporary: torch.save(fork, temporary))
        reloaded = torch.load(io.BytesIO(read_bytes_shared(destination)), map_location='cpu', weights_only=True)
        validate_fork(original, reloaded, source_id, target_id, check)
        forks[kind] = {'path':str(destination.resolve()), 'sha256':digest(destination),
                       'size_bytes':destination.stat().st_size, 'step':reloaded['step'],
                       'selected_step':reloaded['selected_step'], 'source_sha256':sources[kind]['sha256'],
                       'exact_comparison_checks':check.count-before_checks,
                       'tensor_leaves_compared':check.tensor_count-before_tensors}
        del reloaded
    dev_reuse = {}
    for step in inherited_steps:
        source_dev = OUTPUT/'results'/(source_id+'_step_'+str(step))
        target_dev = OUTPUT/'results'/(target_id+'_step_'+str(step))
        check.require(set(p.name for p in source_dev.iterdir() if p.is_file())
                      == {'metadata.json','summary.json',*(f'validation_{lang}.jsonl' for lang in ['tr','en','es','it','fr','de'])},
                      f'Inherited step{step} must contain DEV only')
        reused_files = {}
        copy_tree(source_dev, target_dev, reused_files, check)
        metadata = read_json(source_dev/'metadata.json')
        dev_reuse[str(step)] = {'source_path':str(source_dev.resolve()), 'target_path':str(target_dev.resolve()),
                                'files':reused_files,'evaluation_reused_without_new_inference':True,
                                'model_fingerprint':metadata['model_fingerprint'],
                                'checkpoint_role':'best' if step == args.expected_best_step else 'latest',
                                'identity_basis':'Fork changes only checkpoint top-level run_id; every model tensor is identical. Metadata and DEV predictions are copied byte-for-byte.'}
    for kind, source in sources.items():
        check.require(digest(Path(source['path'])) == source['sha256'], f'Original {kind} checkpoint changed')
    check.require(digest(args.source_verification) == source_audit_sha, 'Source verification changed during fork')
    check.require(torch.cuda.is_initialized() is False, 'Fork initialized CUDA')
    manifest = {'schema':1, 'kind':'exact_training_state_run_id_fork_v1', 'status':'complete', 'passed':True,
                'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'source_run_id':source_id, 'target_run_id':target_id,
                'source_loop_id':verification['loop_id'], 'source_verification_path':str(args.source_verification.resolve()),
                'source_verification_sha256':source_audit_sha, 'source_checkpoints':sources,
                'predecessor_exit_receipt_path':str(args.predecessor_exit_receipt.resolve()),
                'predecessor_exit_receipt_sha256':digest(args.predecessor_exit_receipt),
                'preserved_published_export':{'kind':'existing_immutable_byte_identical_candidate',
                                            'candidate_path':str(candidate),'files_sha256':hf_files,
                                            'model_copy_omitted_by_root_authorization':True},
                'fork_checkpoints':forks, 'source_completed_step':args.expected_latest_step,
                'inherited_selected_step':args.expected_best_step, 'only_changed_path':'run_id',
                'history_and_dev_namespace_policy':'Inherited rows are unchanged. DEV801/907 belong to source_run_id; new evaluations use target_run_id.',
                'inherited_development_reuse':dev_reuse,
                'config_identity_sha256':json_digest(config['config_identity']),
                'inherited_config_identity':config['config_identity'],
                'snapshot_files':records, 'snapshot_file_count':len(records),
                'snapshot_bytes':sum(x['size_bytes'] for x in records.values()),
                'fork_helper_path':str(Path(__file__).resolve()), 'fork_helper_sha256':digest(Path(__file__)),
                'io_helper_sha256':digest(OUTPUT/'atomic_io_retry_v1.py'),
                'checks':check.count,'tensor_leaves_compared':check.tensor_count,
                'cuda_initialized':False, 'source_artifacts_mutated':False,
                'purpose':'Logical continuation with identical full training state under a fresh run ID; no optimizer restart or TEST selection.'}
    path = snapshot/'fork_manifest.json'
    write_json(path, manifest)
    print(json.dumps({'passed':True,'status':'complete','source_run_id':source_id,'target_run_id':target_id,
                      'manifest_path':str(path),'manifest_sha256':digest(path),'checks':check.count,
                      'fork_checkpoints':forks,'snapshot_file_count':len(records),'snapshot_bytes':manifest['snapshot_bytes'],
                      'cuda_initialized':False}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id',type=checked_id,required=True)
    parser.add_argument('--target-run-id',type=checked_id,required=True)
    parser.add_argument('--source-verification',type=Path,required=True)
    parser.add_argument('--predecessor-exit-receipt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--expected-latest-step',type=int,default=907)
    parser.add_argument('--expected-best-step',type=int,default=801)
    args=parser.parse_args()
    if not 0 <= args.expected_best_step <= args.expected_latest_step:
        parser.error('Expected checkpoint steps must satisfy 0 <= best <= latest')
    try:
        run(args)
    except Exception as error:
        print(json.dumps({'passed':False,'status':'failed','error':str(error)}),file=sys.stderr)
        raise


if __name__=='__main__':
    main()
