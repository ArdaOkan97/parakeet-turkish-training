"""Source-only freeze: no candidate import/execution or scientific owner call."""
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
S = P / 'outputs/parakeet_turkish'
W = Path(__file__).parent
sys.path.insert(0, str(S))
from resource_watchdog_v2 import sample_memory


def pin(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b'\n'
    with Path(path).open('xb') as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    return pin(path)


def who(pid):
    p = psutil.Process(pid)
    return {'pid': p.pid, 'create_time': p.create_time(), 'ppid': p.ppid()}


began = time.monotonic()
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
initial = sample_memory(os.getpid(), 'binary_attribution_source_only_freeze')
initial_pin = write(W / 'binary_attribution_source_freeze_initial_numeric_v1.json', initial)
assert initial['paired_sampling_within_one_second']
assert initial['worker_private_memory']['private_bytes'] < 512 * 1024**2
assert initial['system_memory']['commit_available_bytes'] >= 5122415232
assert initial['system_memory']['physical_available_bytes'] >= 2372415232
assert shutil.disk_usage('E:/').free >= 100000000000
base_path = W / 'run_postDEV200000_stop_watcher_CPU_external_v1.py'
base = base_path.read_bytes()
text = base.decode()
edits = [
    ('ROOT_postDEV200000_stop_watcher_finite_CPU_exact_ONCE_execution_scope_v1',
     'ROOT_plan_binary_and_main_path_finite_CPU_exact_ONCE_execution_scope_v1'),
    ('ASR18_postDEV200000_watcher_finite_CPU_external_v1', 'ASR18_plan_binary_main_path_finite_CPU_external_v1'),
    ('{19320, 39856, 21928, 13472, 39752, 26892, 30128}', '{19320, 39856, 21928, 42068, 40536, 26892, 30128}'),
    ('5055306368', '5122415232'), ('2305306368', '2372415232'),
    ('2268435456', '2335544320'), ('1768435456', '1835544320')]
steps = []
for old, new in edits:
    count = text.count(old)
    assert count > 0
    text = text.replace(old, new)
    steps.append({'old': old, 'new': new, 'count': count})
external = W / 'run_plan_binary_main_path_CPU_external_v1.py'
ast.parse(text)
with external.open('xb') as file:
    file.write(text.encode())
    file.flush()
    os.fsync(file.fileno())
inverse = text
for step in reversed(steps):
    assert inverse.count(step['new']) == step['count']
    inverse = inverse.replace(step['new'], step['old'])
assert inverse.encode() == base
external_ledger = write(W / 'binary_attribution_external_full_RAW_inverse_v1.json',
    {'base': pin(base_path), 'successor': pin(external), 'ordered_RAW_steps': steps,
     'whole_forward_inverse_exact': True, 'resources_more_conservative_by_distinct64MiB': True,
     'all_owned_cleanup_family_wall_logic_otherwise_exact': True})
names = ['plan_binary_validation_v4.py', 'main_path_scalar_attribution_v1.py',
         'synthetic_original_full20_plan_fixture_v1.py', 'verify_plan_binary_and_main_path_CPU_v3.py',
         'run_plan_binary_main_path_CPU_external_v1.py']
sources = [pin(W / name) for name in names]
fresh = []
for source in sources:
    raw = Path(source['path']).read_bytes()
    text = raw.decode()
    tree = ast.parse(text)
    fresh.append({'source': source, 'base_bytes': 0, 'base_existed_before_creation': False,
        'complete_new_UTF8_source': text, 'whole_creation_forward_inverse_to_empty_exact': True,
        'AST_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'AST_only_no_import_compile_or_execution': True})
fresh_pin = write(W / 'binary_attribution_full_fresh_source_RAW_AST_v1.json', {'fresh_sources': fresh})
scope = W.parent / 'ROOT_next204000_eval_recurrent_throughput_repair_INACTIVE_preparation_scope_v1.json'
inputs = sources + [pin(scope), pin(Path(__file__)), external_ledger, fresh_pin,
    pin(S / 'resource_watchdog_v2.py'), pin(S / 'bpe_broad_production_state_v1.py'),
    pin(S / 'bpe_production_window_coordinator_v4.py'), pin(S / 'bpe_training_schedule_v1.py'),
    pin(S / 'asr18_BPE9223_source_context_v2.py'), pin(S / 'fastmss_resident_feature_stream_v14.py'),
    pin(P / 'work/research/run_ASR18_BPE9223_resident_RAM_training_v37.py'),
    pin(Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000026\configuration.json')),
    pin(Path(r'E:\gpu_training_scratchpad\storage_config.json')),
    pin(Path(r'E:\gpu_training_scratchpad\use_storage.ps1'))]
proof_scope = W / 'ROOT_plan_binary_and_main_path_finite_CPU_exact_ONCE_execution_scope_v1.json'
protected = [
    {'pid': 19320, 'create_time': 1792803517.7236164, 'ppid': 39856},
    {'pid': 39856, 'create_time': 1792803517.7165897, 'ppid': 21928},
    {'pid': 21928, 'create_time': 1792803516.8916032, 'ppid': 41376},
    {'pid': 42068, 'create_time': 1792830275.0700862, 'ppid': 40536},
    {'pid': 40536, 'create_time': 1792830275.0545292, 'ppid': 19320},
    {'pid': 26892, 'create_time': 1792652904.3359847, 'ppid': 30128},
    {'pid': 30128, 'create_time': 1792652904.283631, 'ppid': 10644}]
handoff = {'kind': 'ASR18_plan_binary_main_path_INACTIVE_sources_CPU_handoff_v1',
    'execution_admitted': False, 'preparation_ROOT': pin(scope), 'sources': sources,
    'binary_helper_source': sources[0], 'attribution_helper_source': sources[1],
    'synthetic_plan_builder_source': sources[2], 'fixture_source': sources[3], 'external_source': sources[4],
    'input_pins': inputs, 'full_fresh_source_RAW_AST': fresh_pin, 'external_exact_RAW_inverse': external_ledger,
    'proposed_ROOT_CPU_scope': str(proof_scope),
    'proposed_CPU_prefix': [r'E:\gpu_training_scratchpad\scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe', '-X', 'utf8', '-B'],
    'proposed_CPU_arguments': [str(external), '--ROOT-review', str(proof_scope)],
    'fixture_output': str(W / 'binary_attribution_CPU_actual_01'),
    'external_output': str(W / 'binary_attribution_CPU_external_actual_01'),
    'protected_live_identities': protected,
    'resource_proposal': {'full_family_private_bytes': 536870912, 'external_wall_seconds': 120,
        'full_prospective_commit_bytes': 5122415232, 'full_prospective_physical_bytes': 2372415232,
        'runtime_commit_bytes': 2335544320, 'runtime_physical_bytes': 1835544320,
        'distinct_extra_bookkeeping_bytes': 67108864, 'no_freed_credit': True,
        'all_original225_FULL_packet_queue_prefetch_prior16MiB_reserves_preserved': True,
        'E_free_bytes': 100000000000, 'CPU_hidden': '-1', 'thread_caps': 1, 'priority': 'BelowNormal'},
    'binary_limits': {'wire_bytes': 2097152, 'full_visits_and_memo_insertions': 262144,
        'container_items': 65536, 'Unicode_codepoints': 131072, 'integer_bits': 4096,
        'unchanged_original_CPython_recursion_limit_maximum': 1000,
        'interpreter_eligibility': 'CPython3.11.13 64bit ONLY; all others original canonical fallback',
        'conservative_memo_table_allowance_bytes': 16777216,
        'memo_writer_pickler_scientific_refs_destroyed_each_call': True,
        'actual_CPU_allocation_cap_proof_pending': True},
    'runtime_versions_already_observed_metadata_only': {
        'Eproof': '3.11.13 Anaconda Jun5 2025 MSCv1929 AMD64 64bit',
        'production_nemo_native': '3.11.13 Anaconda Jun5 2025 MSCv1929 AMD64 64bit',
        'no_candidate_import_or_environment_change': True},
    'primary_source': 'https://raw.githubusercontent.com/python/cpython/v3.11.13/Modules/_pickle.c',
    'safety_source_rationale': 'persistent_id(None) runs before builtin save paths; strict exact-type and scalar/visit bounds precede encoding/memo growth. reducer_override rejects all custom reduction. No unpickle. Original firstcanonical may execute subclasses, so a separately timed originalconfirmation after strict binary must match before caching; failure/mismatch disables cache and returns first result. Actual MemoryError propagates.',
    'canonical_calls_honest': 'Original baseline once AND additional confirmation once for an eligible plan, separately timed/counted. Every subsequent original boundary fresh complete strictbinary/equality; mismatch/unsupported expectedcapacity -> originalcanonical exactly once, no refresh. No first-call failure swallowed.',
    'benchmark': {'source': 'REAL original make_plan; METHOD_STUB scientific schedule/rows only',
        'frozen_shapes': ['metadata_heavy', 'text_heavy'], 'fixed_order_each': ['original', 'candidate', 'candidate', 'original'],
        'original_updates_each_trial': 20, 'original_digest_boundaries_each_trial': 60,
        'concurrent_scalar_rows_each_trial': 4000, 'all8_trials_required': True,
        'performance_eligibility': '>=2%better main-loop wall AND main-thread CPU in BOTH frozen shapes',
        'no_production_throughput_or_recurring_cause_claim': True},
    'attribution': {'fixed_main_phase_labels': 11, 'no_payload_callables_frames_or_history': True,
        'wall_threadCPU_counts_sums_maxima_raised_clockerrors': True,
        'actual_authoritative_propose_separate': True,
        'packet_preparation_existing_scalar_excludes_acquire_and_send': True,
        'combined_new_original20row_nested_summary_max_bytes': 16384,
        'no_exclusive_or_addable_nested_wall_claim': True},
    'pending_next_stages': ['ROOT whole source/CPU exact scope', 'ONE actual CPU proof and independent acceptance',
        'source38 typed/config integration', 'ROOT actual204k stop and closure', 'final ONE production binding',
        '204020..209020 allpauses actual long validation >70 and <=5% drift with phase evidence']}
handoff_pin = write(W / 'binary_attribution_INACTIVE_sources_CPU_handoff_v1.json', handoff)
assert time.monotonic() - began < 60
assert [pin(item['path']) for item in inputs] == inputs
result = write(W / 'binary_attribution_source_only_preparation_result_v1.json',
    {'handoff': handoff_pin, 'sources': sources, 'full_RAW_AST': fresh_pin,
     'input_pins_unchanged': True, 'no_candidate_import_or_execution': True,
     'preparer': who(os.getpid()), 'preparer_launcher': who(os.getppid()),
     'initial_numeric_sample': initial_pin, 'wall_seconds': time.monotonic() - began})
print(json.dumps({'result': result, 'handoff': handoff_pin, 'sources': sources,
                  'preparer': who(os.getpid()), 'preparer_launcher': who(os.getppid())}), flush=True)
