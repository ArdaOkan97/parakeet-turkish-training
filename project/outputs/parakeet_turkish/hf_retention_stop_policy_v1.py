"""INERT stop-policy preparation. Only root may convert a CLOSED CPU state.

The only permitted serialized change is max_retention_failures: 2 -> 0.
Strict DEV qualification, loss, source/data identity, optimizer and sampler stay.
No Torch/NeMo import occurs during module import or stdlib preflight.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import run_hf_quality_loop_v1 as quality
from atomic_io_retry_v1 import read_json

HERE, ROOT = quality.HERE, quality.ROOT
KIND = 'HF_declared_retention_stop_policy_transition_v1'
CHANGE_PATH = 'config_identity.training_arguments.max_retention_failures'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def scoped(path): return quality.runtime.previous.scoped(path)
def record(path): return {'path': str(Path(path).resolve()), 'sha256': sha(path)}
def now(): return datetime.now(timezone.utc).isoformat()


def inspect_source(source_loop, run_id, *, require_closed=False,
                   source_closed_confirmed=False, exit_receipt=None, allow_policy0=False):
    """Metadata-only preparation; live checkpoints are never opened here."""
    source_loop = scoped(source_loop)
    state_path = source_loop / 'loop_status.json'
    state = read_json(state_path)
    arm = next((a for a in state.get('trials', []) if a.get('run_id') == run_id), None)
    if arm is None and state.get('run_id') == run_id and state.get('worker_arguments'):
        arm = state
    require(arm is not None, 'Declared owned source arm required')
    run_dir = HERE / 'runs' / run_id
    config_path = run_dir / 'run_config.json'
    config = read_json(config_path)
    require(config.get('run_id') == run_id and config.get('batch_size') == 8
        and config.get('accumulation') == 2 and config.get('eval_every_updates') == 250,
        'Only the existing B8x2/cadence250 arm is eligible')
    old_policy = config.get('max_retention_failures')
    require(old_policy == 2 or (allow_policy0 and old_policy == 0), 'Stop policy must be explicitly2 or previously declared0')
    require(config['config_identity']['training_arguments']['max_retention_failures'] == old_policy,
        'Published stop policy and checked identity disagree')
    require(config['tr_clips_per_update'] == 8 and config['replay_clips_per_update'] == 8,
        'Preserve TR8/replay8 effective16')
    for name, digest in quality.SOURCE_PINS.items():
        require(sha(HERE / name) == digest, 'Frozen runtime changed: ' + name)
    info = {'source_loop': str(source_loop), 'run_id': run_id, 'arm': arm,
        'config': config, 'config_path': str(config_path), 'state_path': str(state_path),
        'first_pass_target_step': (config['turkish_clips_used'] + 7) // 8,
        'policy_before': old_policy, 'source_closure_confirmed': require_closed,
        'latest_path': str(ROOT / 'work' / (run_id + '_latest.pt')),
        'best_path': str(ROOT / 'work' / (run_id + '_best.pt'))}
    if not require_closed:
        return info
    require(source_closed_confirmed and exit_receipt is not None,
        'Root-observed parent+worker exit receipt and confirmation are required')
    require(state.get('status') in ('quality_window_ended', 'train_window_ended', 'first_uniform_pass_ended')
        and arm.get('child_exit_code') == 0 and not arm.get('resource_review_required'),
        'Successfully saved closed child and terminal non-resource parent required')
    require(arm.get('cooperative_stop_reason') in (None, 'window_budget_checkpoint_boundary', 'first_uniform_pass_boundary'),
        'Resource/progress stops need a separate fix; policy transition cannot bypass them')
    receipt_path = scoped(exit_receipt)
    exit_value = read_json(receipt_path)
    require(exit_value.get('kind') == 'HF_owned_parent_and_worker_exit_receipt_v1'
        and exit_value.get('passed') is True and exit_value.get('run_id') == run_id
        and scoped(exit_value['source_loop']) == source_loop
        and exit_value.get('parent_exited') is True and exit_value.get('worker_exited') is True
        and exit_value.get('parent_observed_exit_code') == 0 and exit_value.get('child_exit_code') == 0
        and bool(exit_value.get('parent_owned_session_id')) and bool(exit_value.get('observed_at_utc'))
        and exit_value.get('source_status_sha256') == sha(state_path),
        'Root exit receipt does not bind the actual terminal parent/status')
    paths = arm['paths']
    identity = read_json(paths['identity'])
    require(exit_value.get('worker_pid') == identity.get('pid') and identity.get('expected_run_id') == run_id,
        'Root worker exit must refer to the recorded actual owned worker PID')
    for role in ('io', 'cache', 'memory'):
        require(read_json(paths[role]).get('status') == 'complete', 'Incomplete closed runtime evidence: ' + role)
    native_path = Path(paths.get('fresh_runtime', paths.get('native', '')))
    require(native_path.is_file() and read_json(native_path.with_suffix('.completion.json')).get('status') == 'complete'
        and read_json(native_path.with_suffix('.completion.json')).get('sources_unchanged') is True,
        'Actual native runtime completion/source proof required')
    history_path, selection_path = run_dir / 'training_history.json', run_dir / 'selection.json'
    history, selection = read_json(history_path), read_json(selection_path)
    require(selection.get('run_id') == run_id and selection.get('accepted') is True
        and selection['completed_steps'] == len(history) > 0
        and [r['step'] for r in history] == list(range(1, len(history) + 1)),
        'Contiguous durable history and a strictly qualified BEST are required')
    require(selection['completed_steps'] < info['first_pass_target_step'], 'First uniform pass is already complete')
    selected_gate = (read_json(run_dir / 'initial_development.json') if selection['selected_step'] == 0
        else history[selection['selected_step'] - 1]['validation'])
    require(selected_gate['retention_gate_passed'] is True
        and selected_gate['turkish_wer'] == selection['turkish_validation_wer'], 'BEST lacks actual strict all5 DEV qualification')
    expected_stop = 'consecutive_retention_gate_failures' if old_policy == 2 else ('controller_stop_file', 'soft_max_wall_seconds')
    require(selection['stop_early_reason'] == expected_stop if isinstance(expected_stop, str)
        else selection['stop_early_reason'] in expected_stop, 'Only retention-stop2 or later declared0 budget closure is eligible')
    require(Path(selection['latest_weights']).resolve() == Path(info['latest_path'])
        and Path(selection['weights']).resolve() == Path(info['best_path']), 'Actual latest/BEST paths differ')
    info.update(history=history, selection=selection, step=len(history),
        evidence={key: record(path) for key, path in {
            'controller_status': state_path, 'run_config': config_path, 'history': history_path,
            'selection': selection_path, 'exit_receipt': receipt_path, 'identity': paths['identity'],
            'native_runtime': native_path, 'native_completion': native_path.with_suffix('.completion.json'),
            'io': paths['io'], 'cache': paths['cache'], 'memory': paths['memory'],
            'latest': info['latest_path'], 'best': info['best_path']}.items()})
    return info


def tree_fingerprint(value, torch_module):
    """Exact tensor bytes and recursively typed values; compact counts only."""
    digest = hashlib.sha256(); counts = {'tensor_leaves': 0, 'tensor_bytes': 0, 'scalar_leaves': 0}
    def walk(item):
        if isinstance(item, torch_module.Tensor):
            require(item.device.type == 'cpu', 'CPU-only checkpoint tensors required')
            raw = item.detach().contiguous().reshape(-1).view(torch_module.uint8).numpy()
            digest.update(b'T' + str(item.dtype).encode() + repr(tuple(item.shape)).encode())
            digest.update(memoryview(raw)); counts['tensor_leaves'] += 1; counts['tensor_bytes'] += raw.nbytes
        elif isinstance(item, dict):
            digest.update(b'D' + str(len(item)).encode())
            for key, child in item.items(): walk(key); walk(child)
        elif isinstance(item, (list, tuple)):
            digest.update(('L' if isinstance(item, list) else 'U').encode() + str(len(item)).encode())
            for child in item: walk(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b':' + json.dumps(item, ensure_ascii=False, allow_nan=False).encode())
            counts['scalar_leaves'] += 1
        else:
            raise TypeError('Unsupported checkpoint value: ' + type(item).__name__)
    walk(value)
    return {'sha256': digest.hexdigest(), **counts}


def verify_receipt(path, *, check_files=True, check_mutable_sources=False):
    value = read_json(scoped(path))
    require(value.get('kind') == KIND and value.get('passed') is True and value.get('status') == 'CPU_converted'
        and value.get('only_changed_state_path') == CHANGE_PATH
        and value.get('policy_transition') == {'max_retention_failures': {'before': 2, 'after': 0}}
        and value.get('scientific_stop_policy_changed') is True
        and value.get('strict_all5_BEST_gate_changed') is False
        and value.get('all_other_state_exact_after_serialization') is True
        and value.get('CUDA_initialized') is False,
        'Complete explicit2->0 transition receipt required')
    require(value.get('converter_sha256') == sha(__file__), 'Frozen conversion implementation changed')
    require(value['converted_config_identity']['training_arguments']['max_retention_failures'] == 0,
        'Converted identity does not declare stop0')
    expected = json.loads(json.dumps(value['source_config_identity']))
    expected['training_arguments']['max_retention_failures'] = 0
    require(expected == value['converted_config_identity'], 'Another scientific/data/runtime identity changed')
    require(set(value['source_evidence']) == set(value['snapshots'])
        and all(value['source_evidence'][role]['sha256'] == item['sha256'] for role, item in value['snapshots'].items()),
        'Immutable snapshots do not bind every original source')
    if check_files:
        items = [value['converted_checkpoint'], *value['snapshots'].values()]
        if check_mutable_sources: items.extend(value['source_evidence'].values())
        for item in items:
            require(sha(item['path']) == item['sha256'], 'Transition evidence changed: ' + item['path'])
    return value


def main(args):
    info = inspect_source(args.source_loop, args.run_id, require_closed=not args.prepare_only,
        source_closed_confirmed=args.source_closed_confirmed, exit_receipt=args.exit_receipt)
    folder = scoped(args.output_dir)
    require(folder != scoped(args.source_loop) and not folder.is_relative_to(HERE / 'runs' / args.run_id),
        'NEW evidence directory outside original run required')
    if args.prepare_only or args.validate_only:
        print(json.dumps({'passed': True, 'status': 'inert_preparation' if args.prepare_only else 'closed_CPU_preflight',
            'source_closed': info['source_closure_confirmed'], 'run_id': args.run_id,
            'first_pass_target_step': info['first_pass_target_step'], 'checkpoint_copied': False,
            'Torch_imported': 'torch' in sys.modules, 'GPU_launch': False}))
        return
    require(args.root_cpu_conversion_authorized and not folder.exists(), 'Root CPU authorization and NEW conversion directory required')
    import torch # ONLY root's actual closed-state conversion, never agent preparation.
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'Never initialize CUDA for conversion')
    folder.mkdir(parents=True)
    snapshots = {}
    for role, item in info['evidence'].items():
        target = folder / ('source_' + role + Path(item['path']).suffix)
        shutil.copyfile(item['path'], target)
        require(sha(target) == item['sha256'], 'Closed byte snapshot differs: ' + role)
        snapshots[role] = record(target)
    state = torch.load(snapshots['latest']['path'], map_location='cpu', weights_only=True)
    require(state['run_id'] == args.run_id and state['step'] == info['step']
        and state['config_identity'] == info['config']['config_identity'] and state['history'] == info['history']
        and state['stop_reason'] == 'consecutive_retention_gate_failures'
        and state['retention_failures'] >= 2 and state['selected_step'] == info['selection']['selected_step']
        and len(state['weights']) == len(state['optimizer']['state']) == 207,
        'Closed full state differs from durable source/config/counters')
    original = tree_fingerprint(state, torch)
    before = json.loads(json.dumps(state['config_identity']))
    state['config_identity']['training_arguments']['max_retention_failures'] = 0
    after = json.loads(json.dumps(state['config_identity']))
    converted_tree = tree_fingerprint(state, torch)
    target = folder / 'immutable_full_state_stop_policy0.pt'
    with target.open('xb') as stream: torch.save(state, stream)
    del state
    restored = torch.load(target, map_location='cpu', weights_only=True)
    require(tree_fingerprint(restored, torch) == converted_tree, 'Serialized target state differs')
    restored['config_identity']['training_arguments']['max_retention_failures'] = 2
    require(tree_fingerprint(restored, torch) == original, 'A field other than the one stop policy changed')
    preserved_failures = restored['retention_failures']; del restored
    for item in info['evidence'].values(): require(sha(item['path']) == item['sha256'], 'Closed source changed during conversion')
    receipt = {'schema': 1, 'kind': KIND, 'status': 'CPU_converted', 'passed': True,
        'run_id': args.run_id, 'source_step': info['step'], 'inherited_selected_step': info['selection']['selected_step'],
        'first_pass_target_step': info['first_pass_target_step'], 'only_changed_state_path': CHANGE_PATH,
        'policy_transition': {'max_retention_failures': {'before': 2, 'after': 0}},
        'scientific_stop_policy_changed': True, 'same_science_continuation_claimed': False,
        'strict_all5_BEST_gate_changed': False, 'resource_progress_guards_changed': False,
        'weights_Adam_sampler_RNG_history_selection_counters_reset': False,
        'retention_failure_counter_carried': preserved_failures,
        'all_other_state_exact_after_serialization': True, 'source_state_tree': original,
        'converted_state_tree': converted_tree, 'source_config_identity': before,
        'source_config_identity_sha256': canonical(before), 'converted_config_identity': after,
        'converted_config_identity_sha256': canonical(after), 'source_evidence': info['evidence'],
        'snapshots': snapshots, 'converted_checkpoint': record(target), 'converter_sha256': sha(__file__),
        'created_at_utc': now(), 'CUDA_initialized': torch.cuda.is_initialized(),
        'scope': 'Declared internal stopping-policy transition only; rejected latest remains exploratory and BEST selection stays strict.'}
    with (folder / 'policy_transition_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'passed': True, 'receipt': str(folder / 'policy_transition_receipt.json'),
        'source_step': info['step'], 'tensor_leaves_verified': original['tensor_leaves'], 'CUDA_initialized': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-loop', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exit-receipt', type=Path)
    parser.add_argument('--source-closed-confirmed', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--root-cpu-conversion-authorized', action='store_true')
    main(parser.parse_args())
