"""Small stdlib fixtures only; never load live checkpoints or import Torch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/parakeet_turkish'))
import hf_retention_stop_policy_v1 as policy
import launch_hf_stop_policy_resume_v1 as launcher
import resume_hf_stop_policy_v1 as controller

checks = []
def check(ok, name):
    if not ok: raise AssertionError(name)
    checks.append(name)
def rejects(call, name):
    try: call()
    except (ValueError, RuntimeError): checks.append(name); return
    raise AssertionError('Unexpected acceptance: ' + name)
def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')

check('torch' not in sys.modules and 'nemo' not in sys.modules, 'all_modules_import_without_Torch_or_NeMo')
scratch = ROOT / 'work/research' / ('stop_policy_fixtures_' + uuid.uuid4().hex)
scratch.mkdir()
check(scratch.resolve().is_relative_to(ROOT), 'fixtures_confined_to_workspace')
run_id = 'fixture_closed_policy'
run = scratch / 'outputs/runs' / run_id; run.mkdir(parents=True)
loop = scratch / 'source_loop'; loop.mkdir()
paths = {role: str(loop / (role + '.json')) for role in ('io', 'cache', 'memory', 'identity', 'native')}
for role in ('io', 'cache', 'memory'): dump(Path(paths[role]), {'status': 'complete'})
dump(Path(paths['identity']), {'pid': 987654, 'expected_run_id': run_id})
dump(Path(paths['native']), {'status': 'ready'})
dump(Path(paths['native']).with_suffix('.completion.json'), {'status': 'complete', 'sources_unchanged': True})
config = {'run_id': run_id, 'batch_size': 8, 'accumulation': 2, 'eval_every_updates': 250,
    'max_retention_failures': 2, 'tr_clips_per_update': 8, 'replay_clips_per_update': 8,
    'turkish_clips_used': 100, 'config_identity': {'training_arguments': {'max_retention_failures': 2}}}
history = [{'step': i, 'validation': {'retention_gate_passed': i == 1, 'turkish_wer': .6 if i == 1 else .7}} for i in range(1, 5)]
latest = scratch / 'work' / (run_id + '_latest.pt'); latest.parent.mkdir()
latest.write_bytes(b'fixture-only-no-Torch-checkpoint'); best = latest.with_name(run_id + '_best.pt'); best.write_bytes(b'fixture-best')
selection = {'run_id': run_id, 'accepted': True, 'completed_steps': 4, 'selected_step': 1,
    'turkish_validation_wer': .6, 'stop_early_reason': 'consecutive_retention_gate_failures',
    'latest_weights': str(latest), 'weights': str(best)}
state = {'status': 'quality_window_ended', 'trials': [{'run_id': run_id, 'child_exit_code': 0,
    'resource_review_required': False, 'worker_arguments': [], 'paths': paths}]}
for name, value in [('run_config', config), ('training_history', history), ('selection', selection)]: dump(run / (name + '.json'), value)
dump(loop / 'loop_status.json', state)
exit_path = loop / 'root_exit_receipt.json'
exit_value = {'kind': 'HF_owned_parent_and_worker_exit_receipt_v1', 'passed': True, 'run_id': run_id,
    'source_loop': str(loop), 'parent_exited': True, 'worker_exited': True, 'parent_observed_exit_code': 0,
    'child_exit_code': 0, 'parent_owned_session_id': 'CPU_fixture_only', 'worker_pid': 987654,
    'observed_at_utc': '2026-10-11T12:00:00+00:00', 'source_status_sha256': policy.sha(loop / 'loop_status.json')}
dump(exit_path, exit_value)
saved_here, saved_root, saved_pins = policy.HERE, policy.ROOT, policy.quality.SOURCE_PINS
try:
    policy.HERE = scratch / 'outputs'; policy.ROOT = scratch; policy.quality.SOURCE_PINS = {}
    invoke = lambda: policy.inspect_source(loop, run_id, require_closed=True, source_closed_confirmed=True, exit_receipt=exit_path)
    valid = invoke()
    check(valid['step'] == 4 and valid['first_pass_target_step'] == 13, 'closed_retention2_fixture_accepts_and_computes_first_pass')
    rejects(lambda: policy.inspect_source(loop, run_id, require_closed=True, source_closed_confirmed=False, exit_receipt=exit_path), 'root_closed_confirmation_required')
    rejects(lambda: policy.inspect_source(loop, run_id, require_closed=True, source_closed_confirmed=True), 'observed_parent_and_worker_exit_receipt_required')
    for key, changed, name in [('worker_exited', False, 'live_worker_receipt_rejected'), ('worker_pid', 123, 'wrong_owned_worker_PID_rejected'), ('source_status_sha256', 'wrong', 'stale_exit_status_hash_rejected')]:
        bad = {**exit_value, key: changed}; dump(exit_path, bad); rejects(invoke, name)
    dump(exit_path, exit_value)
    selection['stop_early_reason'] = 'controller_stop_file'; dump(run / 'selection.json', selection)
    rejects(invoke, 'resource_or_generic_stop_cannot_convert_policy2')
    selection['stop_early_reason'] = 'consecutive_retention_gate_failures'; dump(run / 'selection.json', selection)
    state['trials'][0]['resource_review_required'] = True; dump(loop / 'loop_status.json', state)
    rejects(invoke, 'resource_review_required_never_bypassed')
    state['trials'][0]['resource_review_required'] = False; dump(loop / 'loop_status.json', state)
    history[0]['validation']['retention_gate_passed'] = False; dump(run / 'training_history.json', history)
    rejects(invoke, 'unqualified_BEST_rejected')
    history[0]['validation']['retention_gate_passed'] = True; dump(run / 'training_history.json', history)
    dump(Path(paths['memory']), {'status': 'failed'}); rejects(invoke, 'incomplete_actual_runtime_proof_rejected')
    dump(Path(paths['memory']), {'status': 'complete'})
    config['max_retention_failures'] = 0; config['config_identity']['training_arguments']['max_retention_failures'] = 0; dump(run / 'run_config.json', config)
    rejects(invoke, 'first_converter_never_reinterprets_stop0_as_2to0')
finally:
    policy.HERE, policy.ROOT, policy.quality.SOURCE_PINS = saved_here, saved_root, saved_pins

# Exact recursive byte comparison tested without importing tensor libraries.
class Buffer(bytearray):
    @property
    def nbytes(self): return len(self)
class Tensor:
    device = types.SimpleNamespace(type='cpu'); dtype = 'fixture-float32'; shape = (2,)
    def __init__(self, data): self.data = Buffer(data)
    def detach(self): return self
    def contiguous(self): return self
    def reshape(self, *args): return self
    def view(self, *args): return self
    def numpy(self): return self.data
fake_torch = types.SimpleNamespace(Tensor=Tensor, uint8='fixture-uint8')
tree = {'config_identity': {'training_arguments': {'max_retention_failures': 2}}, 'weights': {'x': Tensor(b'12345678')},
    'optimizer': {'step': Tensor(b'1234')}, 'sampler': {'order': ['a', 'b'], 'position': 1},
    'RNG': (3, [123, 456], None), 'retention_failures': 2, 'history': [{'step': 1}], 'selected_step': 0}
old = policy.tree_fingerprint(tree, fake_torch)
tree['config_identity']['training_arguments']['max_retention_failures'] = 0
check(policy.tree_fingerprint(tree, fake_torch) != old, 'one_stop_policy_field_changes_tree_digest')
tree['config_identity']['training_arguments']['max_retention_failures'] = 2
check(policy.tree_fingerprint(tree, fake_torch) == old, 'restoring_only_policy_recovers_full_tree_digest')
for field, value in [('retention_failures', 0), ('selected_step', 1), ('sampler', {'order': ['b', 'a'], 'position': 1})]:
    saved = tree[field]; tree[field] = value
    check(policy.tree_fingerprint(tree, fake_torch) != old, 'tree_detects_' + field + '_mutation')
    tree[field] = saved
tree['weights']['x'].data[0] = 99
check(policy.tree_fingerprint(tree, fake_torch) != old, 'tree_detects_single_tensor_byte_mutation')

for hours in (0, .05, 6.01):
    rejects(lambda: controller.prepare(argparse.Namespace(window_hours=hours, poll_seconds=5)), 'bounded_window_rejects_' + str(hours))

# Invoke the real thin main() with only DLL preload/nested worker stubbed.  It
# must publish this process's NEW proof before entering the unchanged chain.
original = scratch / 'immutable_original_ready.json'
original.write_text('{}', encoding='utf-8')
receipt_path = scratch / 'fixture_policy_receipt.json'
receipt_path.write_text('{}', encoding='utf-8')
saved_validate, saved_nested = launcher.validate, launcher.runpy.run_path
saved_preload = sys.modules.get('preload_nemo_linker_v1')
events = []
def stub_preload():
    check('torch' not in sys.modules, 'preload_stub_entered_before_Torch')
    events.append('preload')
    return {'kind': 'coherent_native_cuda12_9_preload_v1',
        'module_sha256': launcher.base.PINS['preload_nemo_linker_v1.py'],
        'actual_binding_linker': {'version': [12, 9]}, 'cuda_initialized': False}
try:
    sys.modules['preload_nemo_linker_v1'] = types.SimpleNamespace(preload=stub_preload)
    for fail in (False, True):
        proof = scratch / ('fresh_failure.json' if fail else 'fresh_success.json')
        completion = proof.with_suffix('.completion.json')
        worker = ['--run-id', run_id, '--max-retention-failures', '0']
        launcher.validate = lambda args: ([], worker, original,
            {'run_id': run_id, 'pid': -1}, proof, completion,
            {'policy_transition': {'max_retention_failures': {'before': 2, 'after': 0}}})
        def nested(path, **kwargs):
            check(events[-1] == 'preload', 'actual_launcher_orders_preload_before_nested_chain_' + str(fail))
            evidence = json.loads(proof.read_text(encoding='utf-8'))
            check(evidence['status'] == 'ready' and evidence['pid'] > 0
                and evidence['original_first_start_ready']['original_first_start_pid'] == -1,
                'fresh_actual_PID_never_relabels_old_identity_' + str(fail))
            events.append('nested')
            if fail: raise RuntimeError('fixture nested worker error')
        launcher.runpy.run_path = nested
        args = argparse.Namespace(validate_only=False, policy_receipt=receipt_path,
            original_native_provenance_sha256=policy.sha(original))
        previous_argv = list(sys.argv)
        if fail: rejects(lambda: launcher.main(args), 'nested_worker_error_preserved')
        else: launcher.main(args)
        result = json.loads(completion.read_text(encoding='utf-8'))
        check(result['status'] == ('failed' if fail else 'complete') and result['sources_unchanged'] is True,
            'truthful_completion_and_source_checks_' + str(fail))
        check(sys.argv == previous_argv, 'argv_restored_after_nested_chain_' + str(fail))
finally:
    launcher.validate, launcher.runpy.run_path = saved_validate, saved_nested
    if saved_preload is None: sys.modules.pop('preload_nemo_linker_v1', None)
    else: sys.modules['preload_nemo_linker_v1'] = saved_preload
check('torch' not in sys.modules and 'nemo' not in sys.modules, 'fixtures_finish_without_Torch_NeMo_or_CUDA')
result = {'schema': 1, 'kind': 'HF_stop_policy_contingency_stdlib_preparation_v1', 'passed': True,
    'check_count': len(checks), 'checks': checks, 'source_sha256': {
        Path(module.__file__).name: policy.sha(module.__file__) for module in (policy, launcher, controller)},
    'scratch': str(scratch), 'actual_root_TorchCPU_conversion_executed': False,
    'actual_fresh_preload_or_GPU_resume_executed': False, 'live_checkpoint_copied_or_deserialized': False,
    'scope': 'Pure typed-tree and closed-source eligibility fixtures; real830+tensor conversion and hardware restore remain root-only pending.'}
target = ROOT / 'outputs/parakeet_turkish/HF_stop_policy_contingency_CPU_verification_v1.json'
target.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'passed': True, 'check_count': len(checks), 'report': str(target), 'Torch_imported': False}))
