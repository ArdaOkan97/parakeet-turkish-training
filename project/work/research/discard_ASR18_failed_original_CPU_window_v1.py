"""Reviewed discard-only retirement for the specific caller5 exit-race failure.

The old frontend01 normal completion was never sealed. This adapter does not
invent it, reuse features for training, or reinterpret the enclosing TOOL1.
It validates the actual failure/retirement chain, archives metadata, then uses
only the frozen exact-owned-file engine. ROOT must review this new boundary.
"""
import argparse
from copy import deepcopy
import datetime
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[2]
S = P / 'outputs/parakeet_turkish'
sys.path.insert(0, str(S))
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
import fastmss_phase_aware_feature_window_v2 as sink
from fastmss_owned_process_closure_v1 import verify as verify_process
from fastmss_row_serial_cpu_v2 import environment, resource_guard
from asr18_owned_family_observation_v2 import same_process

A = Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
RUN = A / 'full_DATA_two_window_native_01'
CPU = RUN / 'CPU_window_0'
PAYLOAD = CPU / 'payload'
FAILURE = RUN / 'actual_owned_CLOSED1_v1.json'
FAILURE_SHA = '98ff0218e5e82c169f160b43a1581e87f212221884259463a5a032f14957df9a'
SINK_SHA = 'cae521b4e578302a1864e2ba3eac23f5de8642b0842e9cdd667bcd4549f52450'


def exact(left, right, message):
    require(left == right and digest(left) == digest(right), message)


def clean_guard(path, producer, code):
    value = read_json(path)
    require(type(value['actual_exit_code']) is int and value['actual_exit_code'] == code
            and value['pid'] == producer['pid'] and value['monitor_clean'] is True
            and value['monitor_joined'] is True and value['observed']['monitor_error'] is None
            and type(value['observed']['samples']) is int and value['observed']['samples'] >= 3,
            'Exact actual native guard completion required')
    return value


def verify_failure(failure, owner_pin, guard):
    import psutil
    require(failure['kind'] == 'ASR18_finite_full_DATA_multiwindow_actual_failure_v1'
            and failure['owner'] == '/root/memory_diagnosis'
            and type(failure['actual_exec_tool_exit_code']) is int
            and failure['actual_exec_tool_exit_code'] == 1
            and failure['actual_tool_chunk_id'] == 'd8079e'
            and failure['owned_session'] == 87069
            and failure['actual_GPU_child_launched'] is False
            and type(failure['actual_optimizer_updates']) is int and failure['actual_optimizer_updates'] == 0
            and type(failure['actual_authoritative_draws_consumed']) is int
            and failure['actual_authoritative_draws_consumed'] == 0
            and failure['original_checkpoint_or_journal_written'] is False,
            'Only the source-bound never-native actual failed trial may be discarded')
    guard()
    require(failure['captured_identities'] and all(not same_process(v) for v in failure['captured_identities'])
            and not psutil.pid_exists(failure['parent_native_PID']),
            'All exact captured identities and actual parent must be retired now')
    lease = read_json(P / 'work/gpu_training_owner.json')
    require(lease['owner'] == '/root/memory_diagnosis' and lease['actual_GPU_running'] is False
            and lease.get('actual_owned_worker_pid') is None,
            'No active native child can share this discard boundary')
    require(not list(RUN.glob('GPU_process_*')) and read_json(RUN / 'failure.json')['actual_native_progress'] is None,
            'No GPU/bootstrap/native progress is allowed in this specific boundary')
    exact(owner_pin, failure['partial_feature_owner'], 'Frozen failed owner differs')


def main(review_path, folder):
    environment()
    guard = resource_guard
    guard()
    require(folder.resolve().parent == RUN.resolve() and not folder.exists(),
            'New E-only evidence namespace outside transient cache')
    own = record(__file__)
    failure_pin = record(FAILURE)
    require(failure_pin['sha256'] == FAILURE_SHA and record(S / 'fastmss_phase_aware_feature_window_v2.py')['sha256'] == SINK_SHA,
            'Preserve the exact failed trial and frozen owned-file engine')
    require(review_path.resolve().is_relative_to(A.resolve()), 'ROOT review must be E metadata')
    review_pin = record(review_path)
    review = read_json(review_path)
    require(review['reviewer'] == '/root' and review['accepted'] is True
            and review['scope'] == 'discard_failed_original_CPU_window_after_supervision_race_v1'
            and review['production_approval_claimed'] is False,
            'Independent exact ROOT discard-source review required')
    exact(review['source'], own, 'ROOT reviewed another discard implementation')
    exact(review['failed_trial'], failure_pin, 'ROOT reviewed another actual failure')
    failure = read_json(FAILURE)
    owner_pin = failure['partial_feature_owner']
    exact(review['partial_owner'], owner_pin, 'ROOT reviewed another partial cache')
    verify_failure(failure, owner_pin, guard)
    owner, cache_folder, plan = sink.owner_folder(owner_pin, guard)
    require(owner['updates'] == 20 and owner['window_index'] == 0
            and owner['prepared_is_committed'] is False
            and owner['sampler_journal_frontier_advanced'] is False
            and not (cache_folder / 'manifest.json').exists(),
            'The incomplete twenty-update window is provisional only')
    inventory = [record(path) for path in sorted(cache_folder.iterdir())]
    exact(inventory, failure['partial_feature_inventory'], 'Failed owned inventory changed')
    require(len(inventory) == 69 and sum(v['bytes'] for v in inventory) == 11_964_790,
            'Only the sealed exact failed inventory is in scope')
    active_names = sorted(v.name for v in cache_folder.glob('active_*.json'))
    require(active_names == ['active_00.json', 'active_01.json'], 'Unknown outstanding producer; preserve all')
    for relative in ('reference_child_00/actual_owned_process0.json',
                     'reference_child_01/actual_owned_process0.json',
                     'frontend_child_00/actual_owned_process0.json'):
        verify_process(read_json(PAYLOAD / relative), kind='ASR18_owned_CPU_process_completion_v1',
                       owner='/root/memory_diagnosis', guard=guard)
    normal00 = sink.verify_feature_closure(record(PAYLOAD / 'frontend_closed_00.json'), owner_pin,
                                          success=True, guard=guard)
    require(normal00['update_index'] == 0 and not (PAYLOAD / 'frontend_closed_01.json').exists(),
            'Preserve the missing frontend01 normal closure; never manufacture one')
    retirement_pin = record(PAYLOAD / 'frontend_child_01/failed_owned_family_actual_retirement.json')
    retirement = read_json(retirement_pin['path'])
    require(retirement['passed'] is False and retirement['normal_leaf_exit0_claimed'] is False
            and type(retirement['actual_launcher_exit_code']) is int
            and retirement['actual_launcher_exit_code'] == 0
            and retirement['retirement']['all_captured_owned_identities_observed_dead'] is True
            and retirement['sampler_or_draw_advanced_by_cleanup'] is False,
            'Truthful failed-controller observed retirement, not normal success')
    captured = retirement['retirement']['captured_identities']
    for index in (0, 1):
        active = read_json(cache_folder / f'active_{index:02}.json')
        producer = active['producer']
        report = read_json(PAYLOAD / f'frontend_{index:02}.json')
        exact(active['owner'], owner_pin, 'Active file owner mismatch')
        exact(report['producer'], producer, 'Active/report producer mismatch')
        exact(report['owner'], owner_pin, 'Frontend report owner mismatch')
        require(report['passed'] is True and report['update_index'] == index
                and report['actual_optimizer_updates'] == 0
                and report['sampler_journal_frontier_advanced'] is False
                and not same_process(producer), 'Only closed provisional producers are in scope')
        clean_guard(PAYLOAD / f'frontend_child_{index:02}/guard/closure.json', producer, 0)
        update_path = cache_folder / f'update_{index:02}.json'
        exact(record(update_path), report['update_record'], 'Frontend emitted update binding changed')
        item = read_json(update_path)
        exact(item['producer'], producer, 'Emitted update producer changed')
        require(item['prepared_is_committed'] is False and item['sampler_journal_frontier_advanced'] is False,
                'Preparation never advances a scientific frontier')
        for row in item['rows']:
            for key in ('input_features', 'attention_mask'):
                exact(record(row[key]['file']['path']), row[key]['file'], 'Owned feature/mask content changed')
        if index == 1:
            require(any(v == producer and digest(v) == digest(producer) for v in captured),
                    'Failed retirement must include the exact active frontend01 identity')
            require(producer['ppid'] == 31140 and any(v['pid'] == 31140 for v in captured),
                    'Exact observed failed Popen lineage required')
    outer = read_json(CPU / 'guard/closure.json')
    require(type(outer['actual_exit_code']) is int and outer['actual_exit_code'] == 1
            and outer['pid'] == 31180 and outer['monitor_joined'] is True
            and outer['monitor_clean'] is True and outer['observed']['monitor_error'] is None,
            'The enclosing controller remains actual1')
    folder.mkdir()
    boundary = dict(kind='ASR18_failed_CPU_window_discard_only_boundary_v1',
        owner=owner_pin, session=owner['session'], phase=owner['phase'], failed_trial=failure_pin,
        ROOT_review=review_pin, actual_exec_tool_exit_code=1, actual_tool_chunk_id='d8079e',
        normal_frontend01_closure_missing_preserved=True, native_work_started=False,
        actual_optimizer_updates=0, all_captured_exact_identities_dead=True,
        scientific_success_or_feature_reuse_for_native_claimed=False, production_enabled=False)
    write_json(folder / 'boundary.json', boundary)
    verify_failure(failure, owner_pin, guard)
    store = sink._Store(cache_folder, owner, guard, sink.capacity_lock)
    archive = dict(kind='ASR18_discarded_failed_CPU_window_durable_metadata_v1',
        boundary=deepcopy(boundary), failure=failure_pin,
        normal00_feature_closure=record(PAYLOAD / 'frontend_closed_00.json'),
        failed01_retirement=retirement_pin, missing_normal01_closure_preserved=True,
        source=own, original_provisional_inventory=inventory)
    removed, archived = sink._retire_files(store, archive=archive,
        evidence_path=folder / 'durable_metadata.json')
    exact(record(FAILURE), failure_pin, 'Original failure evidence changed')
    require(not any(name in sys.modules for name in ('torch', 'numpy', 'scipy', 'av', 'soundfile', 'pyarrow', 'transformers')),
            'Discard is metadata/hash-only; no numerical/audio imports')
    result = dict(kind='ASR18_failed_original_CPU_feature_window_actual_discard_v1', passed=True,
        producer=sink.process_identity(), source=own, ROOT_review=review_pin,
        failed_trial=failure_pin, boundary=record(folder / 'boundary.json'),
        preserved_metadata=archived, removed_exact_owned_files=removed,
        previous_normal_frontend01_completion_invented=False,
        enclosing_trial_actual_exit_code=1, original_audio_checkpoint_journal_touched=False,
        actual_native_updates=0, shared_feature_cache_empty=not list(sink.CACHE.iterdir()),
        sampler_journal_frontier_advanced=False, production_enabled=False)
    write_json(folder / 'report.json', result)
    print(json.dumps({'passed': True, 'owned_files_removed': len(removed),
        'archive': archived, 'shared_feature_cache_empty': result['shared_feature_cache_empty']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--folder', required=True, type=Path)
    args = parser.parse_args()
    main(args.review, args.folder)
