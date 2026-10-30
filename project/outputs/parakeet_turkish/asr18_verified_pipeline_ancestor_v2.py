"""Closed actual22 ancestry for a NEW explicitly diagnostic cleaner hook."""
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v5 import read_bound
from fastmss_owned_process_closure_v2 import verify
from asr18_deferred_GPU_recovery_v2 import verified_requests


def same(a,b):return a==b and digest(a)==digest(b)


def verify_ancestor(value,*,guard):
    require(value['kind']=='ASR18_verified_actual_pipeline22_diagnostic_ancestor_v1'
            and type(value['global_optimizer_updates']) is int and value['global_optimizer_updates']==22
            and value['diagnostic_only'] is True and value['production_enabled'] is False,
            'Only actual22 closed diagnostic ancestry, never a production source')
    for key in ('actual_TOOL0','report','configuration','session','checkpoint_pointer','native_parent_source',
                'native_engine_source','deferred_recovery_requests','source_manifest'):
        require(same(record(value[key]['path']),value[key]),'Immutable actual22 ancestor changed: '+key)
    actual=read_bound(value['actual_TOOL0'],guard);report=read_bound(value['report'],guard)
    require(actual['kind']=='ASR18_continuation21_22_operator_observed_TOOL_CLOSED0_v1'
            and actual['owner']=='/root/memory_diagnosis' and type(actual['actual_tool_exit_code']) is int
            and actual['actual_tool_exit_code']==0 and actual['actual_tool_completion_observed'] is True
            and type(actual['actual_tool_chunk_id']) is str and bool(actual['actual_tool_chunk_id'])
            and same(actual['report'],value['report']) and same(actual['configuration'],value['configuration'])
            and same(actual['source_manifest'],value['source_manifest'])
            and same(actual['parent_source'],value['native_parent_source'])
            and same(actual['native_engine_source'],value['native_engine_source'])
            and actual['initial_inherited_global_updates']==20 and actual['actual_new_optimizer_updates']==2
            and actual['actual_final_global_updates']==22 and report['passed'] is True
            and report['kind']=='ASR18_actual_checkpoint20_continuation21_22_multiwindow_diagnostic_v1'
            and report['initial_inherited_global_updates']==20 and report['actual_native_updates']==2
            and report['actual_final_global_updates']==22 and report['production_enabled'] is False
            and report['qualified_checkpoint'] is False and report['all_deferred_recovery_obligations_discharged'] is True,
            'Actual operator TOOL0 plus complete real21/22 parent result')
    requests=verified_requests(value['deferred_recovery_requests'],session=value['session'],
        parent_source=value['native_parent_source'],guard=guard)
    captures=[part for window in report['windows'] for part in window['checkpoints']]
    require(same(captures,requests['checkpoints']) and same(captures[-1]['pointer'],value['checkpoint_pointer'])
            and same(actual['current_checkpoint_pointer'],value['checkpoint_pointer']),
            'Every NEW21/22 capture and final actual22 pointer are mandatory')
    recovery_pin=report['all_captured_checkpoints_actual_fresh_GPU_recovery']
    require(same(actual['actual_recovery_process0'],report['actual_recovery_process0'])
            and same(actual['all_captured_checkpoint_recovery'],recovery_pin),'Actual TOOL0 recovery binding')
    process=read_bound(report['actual_recovery_process0'],guard)
    verify(process,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    worker=read_bound(process['report'],guard);recovery=read_bound(recovery_pin,guard)
    require(same(process['worker_source'],value['native_parent_source'])
            and same(process['parent_source'],value['native_parent_source'])
            and worker['passed'] is True and worker['worker_mode']=='restore_only'
            and same(worker['recovery_report'],recovery_pin) and worker['all_captured_checkpoints_actually_recovered'] is True
            and recovery['passed'] is True and recovery['all_captured_checkpoints_recovered'] is True
            and same(recovery['request'],value['deferred_recovery_requests'])
            and same(recovery['native_process_completions'],requests['native_process_completions'])
            and recovery['actual_global_native_updates']==22 and recovery['actual_native_updates_in_recovery_process']==0
            and recovery['actual_draws_consumed_in_recovery']==0 and len(recovery['recovered_checkpoints'])==2,
            'Actual separate complete21/22 fresh recovery0, not persistence or metadata alone')
    for capture,pin in zip(captures,recovery['recovered_checkpoints']):
        restored=read_bound(pin,guard)
        require(restored['passed'] is True and same(restored['pointer'],capture['pointer'])
                and same(restored['checkpoint'],capture['checkpoint'])
                and same(restored['pre_save_fingerprint'],capture['pre_save_fingerprint'])
                and same(restored['next_draw_packet'],capture['next_draw_packet'])
                and restored['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
                and restored['both_samplers_journal_prefix_and_next_draw_exact'] is True
                and restored['native_updates_performed']==restored['draws_consumed']==0,
                'Actual all207/threeAdam/allRNG/source frontiers exact at both NEW captures')
    pointer=read_bound(value['checkpoint_pointer'],guard)
    require(pointer['diagnostic_only'] is True and pointer['committed_updates']==22
            and pointer['all_window_updates_committed'] is True
            and same(pointer['identity']['caller_identity']['session'],value['session'])
            and same(pointer['identity']['caller_identity']['engine'],value['native_engine_source']),
            'Original actual22 source/scientific state and session retained')
    return pointer
