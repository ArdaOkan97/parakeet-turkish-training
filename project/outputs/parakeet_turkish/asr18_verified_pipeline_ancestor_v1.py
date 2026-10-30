"""Pure metadata validation of the real20 ancestor; never a production seed."""
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v5 import read_bound
from fastmss_owned_process_closure_v2 import verify
from asr18_deferred_GPU_recovery_v1 import verified_requests


def same(a,b):
    return a==b and digest(a)==digest(b)


def verify_ancestor(value,*,guard):
    require(value['kind']=='ASR18_verified_actual_fullpopulation20_diagnostic_ancestor_v1'
            and type(value['global_optimizer_updates']) is int and value['global_optimizer_updates']==20
            and value['diagnostic_only'] is True and value['production_enabled'] is False,'Exact actual20 diagnostic ancestor')
    for name in ('actual_TOOL0','report','session','checkpoint_pointer','native_parent_source',
                 'native_engine_source','deferred_recovery_requests','source_manifest'):
        pin=value[name];require(same(record(pin['path']),pin),'Immutable ancestor pin changed: '+name)
    actual=read_bound(value['actual_TOOL0'],guard);report=read_bound(value['report'],guard)
    require(actual['kind']=='ASR18_fullpopulation_firstwindow20_conv_operator_observed_TOOL_CLOSED0_v1'
            and actual['owner']=='/root/memory_diagnosis' and type(actual['actual_tool_exit_code']) is int
            and actual['actual_tool_exit_code']==0 and actual['actual_tool_completion_observed'] is True
            and type(actual['actual_tool_chunk_id']) is str and bool(actual['actual_tool_chunk_id'])
            and same(actual['report'],value['report']) and same(actual['source_manifest'],value['source_manifest'])
            and report['passed'] is True and report['production_enabled'] is False and report['qualified_checkpoint'] is False
            and type(report['actual_native_updates']) is int and report['actual_native_updates']==20
            and report['all_deferred_recovery_obligations_discharged'] is True,'Actual owner TOOL0 and complete pipeline report')
    requests=verified_requests(value['deferred_recovery_requests'],session=value['session'],
        parent_source=value['native_parent_source'],guard=guard)
    captures=[item for window in report['windows'] for item in window['checkpoints']]
    require(same(captures,requests['checkpoints']) and same(captures[-1]['pointer'],value['checkpoint_pointer'])
            and requests['expected_global_new_updates']==20,'Complete original capture chain through20')
    recovery_pin=report['all_captured_checkpoints_actual_fresh_GPU_recovery']
    require(same(actual['actual_recovery_process0'],report['actual_recovery_process0'])
            and same(actual['all_captured_checkpoint_recovery'],recovery_pin),'Actual TOOL0 binds exact recovery')
    process=read_bound(report['actual_recovery_process0'],guard)
    verify(process,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=guard)
    require(same(process['worker_source'],value['native_parent_source'])
            and same(process['parent_source'],value['native_parent_source']),'Actual frozen recovery entrypoint')
    worker=read_bound(process['report'],guard);recovery=read_bound(recovery_pin,guard)
    require(worker['passed'] is True and worker['worker_mode']=='restore_only'
            and same(worker['recovery_report'],recovery_pin) and worker['all_captured_checkpoints_actually_recovered'] is True
            and recovery['passed'] is True and recovery['all_captured_checkpoints_recovered'] is True
            and same(recovery['request'],value['deferred_recovery_requests'])
            and same(recovery['native_process_completions'],requests['native_process_completions'])
            and type(recovery['actual_global_native_updates']) is int and recovery['actual_global_native_updates']==20
            and type(recovery['actual_native_updates_in_recovery_process']) is int and recovery['actual_native_updates_in_recovery_process']==0
            and type(recovery['actual_draws_consumed_in_recovery']) is int and recovery['actual_draws_consumed_in_recovery']==0
            and len(recovery['recovered_checkpoints'])==len(captures),'Actual separate every-capture zero-draw recovery0')
    for item,pin in zip(captures,recovery['recovered_checkpoints']):
        recovered=read_bound(pin,guard)
        require(recovered['passed'] is True and same(recovered['pointer'],item['pointer'])
                and same(recovered['checkpoint'],item['checkpoint'])
                and same(recovered['pre_save_fingerprint'],item['pre_save_fingerprint'])
                and same(recovered['next_draw_packet'],item['next_draw_packet'])
                and recovered['all207_threeAdam_all4RNG_match_actual_pre_save'] is True
                and recovered['both_samplers_journal_prefix_and_next_draw_exact'] is True
                and type(recovered['native_updates_performed']) is int and recovered['native_updates_performed']==0
                and type(recovered['draws_consumed']) is int and recovered['draws_consumed']==0
                and recovered['separate_fresh_GPU_recovery'] is True,'Actual all207/threeAdam/allRNG/sampler/prefix/nextdraw exact')
    pointer=read_bound(value['checkpoint_pointer'],guard)
    require(pointer['diagnostic_only'] is True and type(pointer['committed_updates']) is int
            and pointer['committed_updates']==20 and pointer['all_window_updates_committed'] is True
            and same(pointer['identity']['caller_identity']['session'],value['session'])
            and same(pointer['identity']['caller_identity']['engine'],value['native_engine_source']),
            'Original saved checkpoint20 scientific/session identity retained')
    return pointer
