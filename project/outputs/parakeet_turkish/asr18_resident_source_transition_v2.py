"""Truthful cooperative source transition; never converts a failed exit to normal0."""
import argparse, ctypes, hashlib, json, os, time
from ctypes import wintypes
from pathlib import Path
from atomic_io_retry_v1 import read_json, write_json
from bpe_broad_audio_v1 import record, require

ROOT = Path(r'E:\gpu_training_scratchpad')


def check_saved_relation(saved, whole, actual_updates, saved_updates):
    require(whole['path'] == saved['path'] and whole['bytes'] >= saved['offset'], 'Same retained journal, complete saved prefix')
    require(actual_updates >= saved_updates, 'Actual update counter cannot precede saved frontier')
    if whole['bytes'] == saved['offset']:
        require(whole['sha256'] == saved['sha256'] and actual_updates == saved_updates,
                'No-tail transition requires every actual live update saved')
    return whole['bytes'] - saved['offset']


def checked_source_transition_resume(pin, *, base, folder, pointer_pin):
    from fastmss_owned_process_closure_v2 import dead
    require(record(pin['path']) == pin, 'Exact actual transition closure')
    value = read_json(pin['path'])
    intent_pin = base['resident_source_transition_intent']
    require(value['intent'] == intent_pin and record(intent_pin['path']) == intent_pin, 'Reviewed explicit transition intent')
    intent = read_json(intent_pin['path'])
    require(intent['kind'] == 'ASR18_authorized_private_prefetch_source_transition_intent_v1'
            and intent['branch'] == value['branch'] == str(folder)
            and value['kind'] == 'ASR18_actual_cooperative_prefetch_source_transition_TOOL1_v1'
            and value['intentional_source_transition'] is True and value['normal_native0_claimed'] is False
            and value['actual_native_exit_code'] is None and value['native_numeric_OS_exit_available'] is False
            and value['actual_native_nonzero_wait_observed_by_old_wrapper'] is True
            and value['actual_exec_tool_exit_code'] == 1 and value['actual_exec_tool_chunk'] == 'cfcee9'
            and value['observer_actual_exec_tool_exit_code'] == 1 and value['observer_actual_tool_chunk'] == '3f3f34',
            'Actual intentional abnormal exit, never resource failure or normal0 relabel')
    require(value['checkpoint_pointer'] == pointer_pin and value['saved_optimizer_updates'] >= intent['minimum_saved_updates']
            and all(dead(w) for w in intent['captured_identities']), 'Exact saved frontier and every old identity dead')
    require(len(intent['captured_identities']) == 7 and value['captured_identities'] == intent['captured_identities']
            and value['all_old_processes_observed_dead'] is True and value['saved_optimizer_updates'] == 24600,
            'Only the exact seven closed old identities and actual24600 transition')
    for key in ('old_driver_source', 'old_wrapper_source', 'old_configuration', 'old_invocation_configuration', 'old_boot'):
        require(record(intent[key]['path']) == intent[key], 'Preserve old source/configuration/owner evidence')
    for key in ('stop_request', 'original_failure', 'original_native_log', 'observer_failure', 'old_session_actual_TOOL1',
                'checkpoint_pointer', 'checkpoint_persistence', 'whole_journal_including_preserved_tail'):
        require(record(value[key]['path']) == value[key], 'Preserved actual stop/failure/state/journal evidence')
    failure = read_json(value['original_failure']['path'])
    require(failure['error'] == "ValueError('Actual owner cooperative stop')" and failure['producer'] == intent['old_native']
            and failure['actual_optimizer_updates'] == value['actual_optimizer_updates_before_exit']
            and failure['last_checkpoint'] == value['saved_optimizer_updates'],
            'This entry only accepts the exact requested cooperative guard exit')
    require(value['original_monitor'] is None and value['original_monitor_absent'] is True
            and value['clean_monitor_report_claimed'] is False
            and not (Path(intent['old_invocation'])/'monitor_completion.json').exists(),
            'Guarded teardown did not finish; never fabricate joined/clean monitor proof')
    require('queued=submit_plan(private)' in failure['traceback'] and 'Actual owner cooperative stop' in failure['traceback'],
            'Actual stop was in private future planning, before new positive native work')
    with Path(value['original_native_log']['path']).open('rb') as stream:
        stream.seek(max(0,stream.seek(0,2)-16384)); teardown=stream.read().decode('utf-8')
    require('scopes.close()' in teardown and 'resident_gc_scope' in teardown and 'live_state_fingerprint' in teardown
            and 'ValueError: Actual owner cooperative stop' in teardown,
            'Exact preserved stop guard re-entry during source4 GC-scope teardown')
    pointer = read_json(pointer_pin['path'])
    require(pointer['committed_updates'] == value['saved_optimizer_updates']
            and pointer['journal_prefix'] == value['saved_journal_prefix'], 'Exact coherent saved journal authority')
    stop = read_json(value['stop_request']['path']); proof = read_json(value['checkpoint_persistence']['path'])
    require(stop['intent'] == intent_pin and stop['observed_verified_checkpoint']['pointer'] == pointer_pin
            and proof['named_tensors_verified'] == 854 and proof['Adam_states_verified'] == 215,
            'Requested verified checkpoint is the actual all-state restore frontier')
    tail = check_saved_relation(value['saved_journal_prefix'], value['whole_journal_including_preserved_tail'],
                                value['actual_optimizer_updates_before_exit'], value['saved_optimizer_updates'])
    require(tail == value['unsaved_tail_bytes'] == 0 and value['actual_optimizer_updates_before_exit'] == 24600,
            'Exact actual zero-tail saved24600; no broadened failure adoption')
    from bpe_bounded_production_journal_v4 import prefix
    require(prefix(Path(value['saved_journal_prefix']['path']), value['saved_journal_prefix']['offset'], lambda: None).hexdigest()
            == value['saved_journal_prefix']['sha256'], 'Exact saved prefix, never replay later tail')
    if tail:
        require(record(value['preserved_unsaved_tail']['path']) == value['preserved_unsaved_tail']
                and value['preserved_unsaved_tail']['bytes'] == tail, 'Actual tail kept separately and in original journal')
    return value

