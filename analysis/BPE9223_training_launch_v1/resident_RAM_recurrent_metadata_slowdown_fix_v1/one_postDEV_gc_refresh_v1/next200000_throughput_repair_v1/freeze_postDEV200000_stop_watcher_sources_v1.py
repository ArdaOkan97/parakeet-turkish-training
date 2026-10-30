"""Source-only creation ledger/AST/handoff; never imports a candidate."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import psutil

P = Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
W = Path(__file__).parent
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from resource_watchdog_v2 import sample_memory


def pin(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(name, value):
    path = W / name
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    with path.open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    return pin(path)


def identity(pid):
    proc = psutil.Process(pid)
    return {'pid': proc.pid, 'create_time': proc.create_time(), 'ppid': proc.ppid()}


began = time.monotonic()
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
initial = sample_memory(os.getpid(), 'watcher_source_only_preparation')
initial_pin = write('watcher_source_only_prep_initial_resource_sample_v1.json', initial)
assert initial['paired_sampling_within_one_second'] is True
assert initial['worker_private_memory']['private_bytes'] < 512 * 1024**2
assert initial['system_memory']['commit_available_bytes'] >= 5055306368
assert initial['system_memory']['physical_available_bytes'] >= 2305306368
assert shutil.disk_usage('E:/').free >= 100000000000
names = ['postDEV200000_stop_watcher_v1.py',
         'verify_postDEV200000_stop_watcher_CPU_v1.py',
         'run_postDEV200000_stop_watcher_CPU_external_v1.py']
sources = [pin(W / name) for name in names]
ledger = []
for item in sources:
    raw = Path(item['path']).read_bytes()
    text = raw.decode('utf-8')
    tree = ast.parse(text, filename=item['path'])
    # Fresh sources have no historical base. This is an honest complete creation
    # inverse, not a claim of equivalence to a prior accepted watcher.
    base = b''
    forward = base + text.encode('utf-8')
    inverse = forward[:-len(raw)]
    assert forward == raw and inverse == base
    ledger.append({'source': item, 'base': {'existed_before_creation': False, 'bytes': 0,
        'sha256': hashlib.sha256(base).hexdigest()},
        'ordered_RAW_steps': [{'operation': 'create_complete_new_UTF8_source',
                               'old': '', 'new': text, 'count': 1}],
        'whole_forward_exact': True, 'whole_inverse_to_empty_exact': True,
        'AST_parse_only_no_import_or_compile': True,
        'AST_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'top_level_imports': [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))],
        'functions': [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]})
ledger_pin = write('postDEV200000_watcher_fresh_complete_source_RAW_AST_ledger_v1.json',
    {'kind': 'ASR18_fresh_watcher_complete_creation_RAW_AST_ledger_v1',
     'source_freshness_not_old_source_equivalence_claimed': True, 'sources': ledger})
scope = W.parent / 'ROOT_next200000_eval_throughput_repair_INACTIVE_sources_and_CPU_fixture_preparation_scope_v1.json'
inputs = sources + [pin(scope), pin(W / 'freeze_postDEV200000_stop_watcher_sources_v1.py'),
    pin(S / 'resource_watchdog_v2.py'), pin(P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
    pin(S / 'fastmss_resident_feature_stream_v14.py'),
    pin(Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000026\configuration.json')),
    pin(Path(r'E:\gpu_training_scratchpad\storage_config.json')),
    pin(Path(r'E:\gpu_training_scratchpad\use_storage.ps1')), ledger_pin]
protected = [
    {'pid': 19320, 'create_time': 1792803517.7236164, 'ppid': 39856},
    {'pid': 39856, 'create_time': 1792803517.7165897, 'ppid': 21928},
    {'pid': 21928, 'create_time': 1792803516.8916032, 'ppid': 41376},
    {'pid': 13472, 'create_time': 1792822495.866724, 'ppid': 39752},
    {'pid': 39752, 'create_time': 1792822495.8054123, 'ppid': 19320},
    {'pid': 26892, 'create_time': 1792652904.3359847, 'ppid': 30128},
    {'pid': 30128, 'create_time': 1792652904.283631, 'ppid': 10644}]
cpu = Path(r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe')
proof_review = W / 'ROOT_postDEV200000_stop_watcher_finite_CPU_exact_ONCE_execution_scope_v1.json'
watch_review = W / 'ROOT_postDEV200000_stop_watcher_exact_ONCE_execution_scope_v1.json'
handoff = {'kind': 'ASR18_postDEV200000_supported_stop_watcher_INACTIVE_source_and_CPU_handoff_v1',
    'preparation_ROOT': pin(scope), 'execution_admitted': False,
    'watcher_source': sources[0], 'fixture_source': sources[1], 'external_source': sources[2],
    'full_RAW_AST_ledger': ledger_pin, 'input_pins': inputs,
    'fixture_output': str(W / 'watcher_finite_CPU_actual_01'),
    'external_output': str(W / 'watcher_finite_CPU_external_actual_01'),
    'proposed_CPU_execution_review_path': str(proof_review),
    'proposed_CPU_review_kind': 'ROOT_postDEV200000_stop_watcher_finite_CPU_exact_ONCE_execution_scope_v1',
    'proposed_CPU_prefix': [str(cpu), '-X', 'utf8', '-B'],
    'proposed_CPU_arguments': [sources[2]['path'], '--ROOT-review', str(proof_review)],
    'protected_live_identities': protected,
    'proposed_live_watcher_review_kind': 'ROOT_postDEV200000_stop_watcher_exact_ONCE_execution_scope_v1',
    'proposed_live_watcher_review_path': str(watch_review),
    'proposed_live_watcher_arguments': [sources[0]['path'], '--ROOT-review', str(watch_review),
        '--output', str(W / 'postDEV200000_live_watcher_actual_01')],
    'proposed_live_output': str(W / 'postDEV200000_live_watcher_actual_01'),
    'ROOT_live_review_must_include': ['kind', 'accepted', 'reviewer', 'watcher_source',
                                    'expected_current', 'output', 'input_pins'],
    'live_expected_current': {
        'source': inputs[6], 'configuration': inputs[8], 'CPU_source': inputs[7],
        'native': protected[0], 'native_launcher': protected[1], 'wrapper': protected[2],
        'old_CPU': protected[3], 'old_CPU_launcher': protected[4],
        'reader': protected[5], 'reader_launcher': protected[6], 'cohort_start': 196000},
    'resources': {'CPU_proof_external_wall_seconds': 120, 'watcher_wall_seconds': 3600,
        'watcher_poll_seconds': .25, 'private_or_proof_sampled_full_family_bytes': 536870912,
        'full_prospective_commit_bytes': 5055306368, 'full_prospective_physical_bytes': 2305306368,
        'runtime_commit_bytes': 2268435456, 'runtime_physical_bytes': 1768435456,
        'E_free_bytes': 100000000000, 'numeric_initial_and_live_final_persisted_before_predicate': True,
        'proof_descendant_identities_maximum': 32, 'each_CPU_proof_log_bytes': 4194304,
        'individual_source_JSON_receipt_bytes': 33554432, 'stop_payload_bytes': 65536,
        'CPU_hidden': '-1', 'four_thread_caps': 1, 'priority': 'BelowNormal'},
    'deliberate_source_refinement': 'Serialize fully; owned E temp flush+fsync; recheck ALL decisive predicates then Windows SAME-volume exclusive os.rename to supported stop.json. Never os.replace/overwrite; preserve failed owned temp. No partial target on prepublish serialize/fsync error.',
    'truthful_progress_contract': 'Progress actual200000 emits before checkpoint and last_saved may199800; latest+full persistence200000 and returned recorded DEV are authoritative. No stale progress.last_saved demand.',
    'startup_contract': 'New exact200k folder and CPU_stderr.log must exist, actual_CPU_ready.json absent; all predicates immediately rechecked; race-free pause or zero tail NOT promised.',
    'predicate_proof': {'actual_helpers': ['collect_stage', 'decisive_recheck', 'publish_marker'],
        'real_owned_CPU_PID_ctime_and_ancestry': True, 'real_E_atomic_file_actor': True,
        'scientific_metadata': 'METHOD_STUB generated854/207/215/full350/q/credits, no actual scientific proof',
        'strict_negatives': ['beforeDEV', 'partial350', 'badpreservation', 'checkpointmismatch',
            'checkpointpending', 'missedfrontier', 'beforefrontier', 'badwait', 'badcredits',
            'wrongsource', 'alreadyready', 'missingstartuplog', 'wrongctime', 'wrongconfigpin',
            'foreignstop', 'serializationfailure', 'fsyncfailure', 'readyrace', 'foreigntargetrace'],
        'unread_missing_planbody_and_prediction_files': True,
        'no_actual_training_stop_or_state_or_GPU': True},
    'actual_stop_followup_NOT_admitted': 'Owner alone consumes26493 after actual terminal closure; preserve original cooperative exception/masking errors, Monitor/scalar/CPU closes, exact deaths/unknown OS exits, latest full200000, actual published/optimizer gap0or1 and all historical tail. Full saved-state restore+genuine redo before separately admitted successor.',
    'no_actual_proof_arm_import_or_control_executed': True}
handoff_pin = write('postDEV200000_stop_watcher_INACTIVE_source_CPU_handoff_v1.json', handoff)
assert time.monotonic() - began < 60
result = {'kind': 'ASR18_postDEV200000_watcher_source_only_preparation_result_v1',
    'preparer': identity(os.getpid()), 'preparer_launcher': identity(os.getppid()),
    'source_pins': sources, 'ledger': ledger_pin, 'handoff': handoff_pin,
    'input_pins_after': [pin(item['path']) for item in inputs],
    'initial_resource_sample': initial_pin, 'AST_only_no_candidate_import_compile_or_execution': True,
    'wall_seconds': time.monotonic() - began}
assert result['input_pins_after'] == inputs
result_pin = write('postDEV200000_watcher_source_only_preparation_result_v1.json', result)
print(json.dumps({'result': result_pin, 'handoff': handoff_pin, 'ledger': ledger_pin,
                  'sources': sources, 'preparer': result['preparer'],
                  'preparer_launcher': result['preparer_launcher']}), flush=True)
