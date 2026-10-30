"""Small independent source/identity review; no tensor/framework imports or writes to live evidence."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
LOOP = HERE / 'runs/adaptive_20261011_hf_return_window_01'
REPORT = HERE / 'HF_862_runtime_independent_source_review_v1.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    checks = []
    def check(ok, label):
        if not ok:
            raise AssertionError(label)
        checks.append(label)

    old_path, new_path = HERE/'train_lora_batched_data_v10.py', HERE/'train_lora_batched_data_v11.py'
    old, new = ast.parse(old_path.read_text()), ast.parse(new_path.read_text())
    old_funcs = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    new_funcs = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
    check(set(old_funcs) == set(new_funcs), 'same_training_function_set')
    normalized_run = copy.deepcopy(new_funcs['run'])
    original_source_names = next(node.value for node in old_funcs['run'].body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='source_names' for t in node.targets))
    for node in normalized_run.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='source_names' for t in node.targets):
            node.value = copy.deepcopy(original_source_names)
    check(ast.dump(normalized_run) == ast.dump(old_funcs['run']), 'run_identical_except_explicit_source_names')
    for name in old_funcs.keys()-{'run','exact_resume_identity_matches'}:
        check(ast.dump(old_funcs[name]) == ast.dump(new_funcs[name]), 'identical_function_'+name)
    # The entire normalized run equality covers optimizer/sampler/RNG restoration,
    # mean-U/KD, row bounds, native cache cleanup, evaluation and retention counters.
    check(new_path.read_text().splitlines()[543].strip() == 'torch.cuda.synchronize()', 'existing_optimizer_sync_line544')

    fork_path = LOOP/'fork_snapshot/initial_state_862/fork_manifest.json'
    fork, receipt = read(fork_path), read(LOOP/'runtime_identity_continuation_v1.json')
    check(sha(fork_path)=='2aeeed4836a3e536893d527b1d3fa01afcfdc34444ca0a835ffd59d0f45265f1', 'root_actual_fork_manifest_bound')
    check(fork['passed'] is True and fork['source_step']==862 and fork['inherited_selected_step']==0
          and fork['only_changed_state_path']=='run_id', 'root_fork_only_run_id_change_recorded')
    config = read(receipt['source_run_config']['path'])
    identity = config['config_identity']
    expected = copy.deepcopy(identity)
    for previous, item in receipt['approved_source_changes'].items():
        expected['source_sha256'].pop(previous)
        expected['source_sha256'][item['new_filename']] = item['new_sha256']
    addition = receipt['added_identity_validator']
    expected['source_sha256'][addition['filename']] = addition['sha256']
    expected['input_sha256'].pop(receipt['replaced_previous_migration']['path'])
    backend = receipt['native_backend_transition']
    expected['input_sha256'].pop(str(Path(backend['old_path']).resolve()))
    ready_path = Path(backend['new_path']).resolve()
    ready = read(ready_path)
    expected['input_sha256'][str(ready_path)] = sha(ready_path)
    receipt_path = (LOOP/'runtime_identity_continuation_v1.json').resolve()
    expected['input_sha256'][str(receipt_path)] = sha(receipt_path)
    expected['native_loss']['native_backend_provenance_sha256'] = sha(ready_path)
    check(ready['declared_allocator_policy']=={'cap_gb':8,'end_cache_gb':4,'native_inactive_threshold_bytes':750000000}, 'actual_ready_cap8_end4_native750MB')

    spec = importlib.util.spec_from_file_location('independent_identity_reader', HERE/'hf_exact_runtime_continuation_v1.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_sha, cache = helper.sha, {}
    def cached_sha(path):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = original_sha(path)
        return cache[key]
    helper.sha = cached_sha
    args = SimpleNamespace(run_id=receipt['target_run_id'], resume_exact=Path(receipt['fork_latest']['path']),
                           resume_runtime_migration=receipt_path, native_backend_provenance=ready_path)
    saved = {'step':862,'selected_step':0,'config_identity':identity}
    callback_calls = []
    def native_identity_callback(value):
        check(ready['run_id']==value.run_id and ready['trainer_sha256']==sha(new_path)
              and ready['adapter_sha256']==sha(HERE/'native_tdt_checkpointed_v3.py'), 'actual_ready_new_worker_same_native3')
        callback_calls.append(True)
    check(helper.exact_identity_matches(saved,args,expected,HERE,native_identity_callback) is True,
          'actual_receipt_ready_sources_accept_identity_only_continuation')
    mutations = {
        'head_lr': lambda item: item['training_arguments'].__setitem__('head_lr', 1e-3),
        'replay_kl': lambda item: item['training_arguments'].__setitem__('replay_kl', 0.),
        'replay_fraction': lambda item: item['training_arguments'].__setitem__('replay_fraction', .25),
        'eval_cadence': lambda item: item['training_arguments'].__setitem__('eval_every_updates', 1000),
        'tokenizer': lambda item: item.__setitem__('extended_tokenizer_sha256', '0'*64),
        'native_sigma': lambda item: item['native_loss'].__setitem__('sigma', .02),
        'unapproved_source': lambda item: item['source_sha256'].__setitem__('extra.py', '0'*64),
    }
    for label, mutation in mutations.items():
        changed = copy.deepcopy(expected)
        mutation(changed)
        try:
            helper.exact_identity_matches(saved,args,changed,HERE,lambda _: None)
        except ValueError:
            checks.append('rejected_scientific_change_'+label)
        else:
            raise AssertionError('accepted '+label)
    check(len(callback_calls)==1, 'native_identity_called_once_only_after_exact_comparison')
    check('torch' not in sys.modules and 'nemo' not in sys.modules, 'no_framework_or_GPU_import')
    report = {'schema':1,'passed':True,'check_count':len(checks),'checks':checks,
        'source_sha256':{name:sha(HERE/name) for name in ('train_lora_batched_data_v10.py',
            'train_lora_batched_data_v11.py','hf_exact_runtime_continuation_v1.py',
            'launch_native_training_v5.py','resume_hf_training_862_v1.py','native_tdt_checkpointed_v3.py')},
        'fork_manifest_sha256':sha(fork_path),'runtime_receipt_sha256':sha(receipt_path),
        'native_ready_sha256':sha(ready_path),'checkpoint_fork_sha256_checked_without_loading':cached_sha(args.resume_exact),
        'scientific_update_and_restore_AST_identical':True,'latest_ES_retention_failure_reset':False,
        'root_measured_full_state_tree_identity_reused':True,'current_review_loaded_checkpoint':False,
        'scope':'Independent source/actual identity receipt and ready evidence checks with seven specific forbidden-change fixtures. Full207 Adam/sampler/RNG/history/retention content equality is the root-executed fork tree proof; this review imports no frameworks and performs no hardware test.',
        'controller_review':'New worker uses immutable862 fork input/BEST0, copies only DEV, keeps scientific argv unchanged, owned-worker memory/stop/keepawake scope. No launch blocker found; operational runtime readiness is checked before optimizer steps.'}
    with REPORT.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({'passed':True,'checks':len(checks),'report':str(REPORT),'Torch_imported':False}))


if __name__=='__main__':
    main()
