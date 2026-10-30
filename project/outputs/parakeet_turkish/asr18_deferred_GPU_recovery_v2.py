"""Inactive fresh-worker exact recovery of every persistence-checked save.

The caller must observe all training GPU workers actually CLOSED0 before this
module runs in a distinct sole-owned GPU process. There are no native updates,
frontend/augmentation calls or model constructors here. The350MB phase wrapper
must cover restore, live fingerprint, samplers, prefix and next-draw checks.
"""
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_streaming_training_state_v1 import restore
from asr18_checkpoint_next_draw_v1 import next_draw_packet
from asr18_live_state_fingerprint_v1 import live_state_fingerprint, require_exact_live_restore
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
from fastmss_phase_aware_plan_bridge_v5 import read_bound
from fastmss_owned_process_closure_v2 import verify as verify_process


def verified_requests(binding, *, session, parent_source, guard):
    value = read_bound(binding, guard)
    require(value['kind'] == 'ASR18_continuation21_22_all_new_checkpoint_GPU_recovery_requests_v2'
            and value['owner'] == '/root/memory_diagnosis'
            and value['production_enabled'] is False and value['session'] == session
            and value['parent_source'] == parent_source
            and type(value['expected_global_new_updates']) is int
            and value['expected_global_new_updates']==22
            and type(value['initial_global_updates']) is int and value['initial_global_updates']==20
            and type(value['expected_new_updates']) is int and value['expected_new_updates']==2
            and type(value['native_process_completions']) is list
            and 1 <= len(value['native_process_completions']) <= 2
            and type(value['native_window_completions']) is list
            and 1 <= len(value['native_window_completions']) <= 2,
            'Exact finite source/session and actually closed training chain required')
    from asr18_verified_pipeline_ancestor_v1 import verify_ancestor
    verify_ancestor(value['ancestor'],guard=guard)
    declared_session=read_bound(session,guard)
    require(declared_session['initial_optimizer_updates']==20 and declared_session['ancestor']==value['ancestor'],
            'Exact inherited global20/session prefix; no scientific state reset')
    reported = []; actual_updates = 0
    for pin in value['native_process_completions']:
        process = read_bound(pin, guard)
        verify_process(process, kind='ASR18_owned_GPU_process_completion_v1',
                       owner='/root/memory_diagnosis', guard=guard)
        require(process['worker_source'] == parent_source and process['parent_source'] == parent_source,
                'Actual source-bound training parent/worker, not a fabricated exit')
        report = read_bound(process['report'], guard)
        require(report['passed'] is True and report['production_enabled'] is False
                and report['qualified_checkpoint'] is False
                and type(report['actual_native_updates_this_process']) is int
                and report['actual_native_updates_this_process'] > 0
                and report['all_exact_GPU_checkpoint_restores_deferred'] is True,
                'Only actual positive training workers with truthful deferred recovery')
        actual_updates += report['actual_native_updates_this_process']
        reported.extend(report['windows'])
    require(digest(reported) == digest(value['native_window_completions'])
            and actual_updates == value['expected_new_updates']
            and actual_updates+value['initial_global_updates']==value['expected_global_new_updates'],
            'Complete actual native window chain/update total differs')
    checkpoints = []; total = 0
    for pin in reported:
        window = read_bound(pin, guard)
        require(window['passed'] is True and window['exact_GPU_restores_deferred'] is True
                and window['all_checkpoint_persistence_verified'] is True
                and window['overall_diagnostic_pass_before_separate_restore'] is False,
                'Training persistence alone must not claim complete diagnostic recovery')
        total += window['actual_new_updates']; checkpoints.extend(window['checkpoints'])
    require(total == actual_updates and len(checkpoints)==2
            and digest(checkpoints) == digest(value['checkpoints']),
            'Every actual1/10/final checkpoint is mandatory; no selected subset')
    indices = []
    for item in checkpoints:
        pointer = read_bound(item['pointer'], guard)
        proof = read_bound(item['checkpoint_persistence_proof'], guard)
        caller = pointer['identity']['caller_identity']
        update = pointer['committed_updates']; indices.append(update)
        require(type(update) is int and value['initial_global_updates'] < update <= value['expected_global_new_updates']
                and caller['ancestor_checkpoint']==value['ancestor']['checkpoint_pointer']
                and caller['ancestor_actual_TOOL0']==value['ancestor']['actual_TOOL0']
                and caller['session'] == session and caller['engine'] == value['native_engine_source']
                and caller['exact_GPU_restore_deferred_until_actual_native_exit'] is True
                and item['checkpoint'] == pointer['checkpoint']
                and proof['checkpoint_pointer'] == item['pointer']
                and proof['checkpoint'] == pointer['checkpoint'] and proof['payload'] == pointer['payload']
                and proof['pre_save_fingerprint'] == caller['pre_save_fingerprint'] == item['pre_save_fingerprint']
                and proof['next_draw_packet'] == caller['next_draw_packet'] == item['next_draw_packet']
                and proof['worker_source'] == parent_source and proof['engine_source'] == value['native_engine_source']
                and proof['kind'] == 'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
                and proof['source'] == record(Path(__file__).with_name('asr18_streamed_persistence_fingerprint_v1.py'))
                and proof['passed'] is True and proof['actual_GPU_restore_performed'] is False
                and proof['live_unchanged_after_persistence'] is True
                and proof['deferred_restore_required_before_overall_diagnostic_pass'] is True
                and proof['live_after_state_sha256'] == proof['independent_pre_save_state_sha256']
                    == proof['saved_state_sha256']
                and proof['sampler_state_sha256'] == pointer['sampler_state_sha256']
                and proof['journal_prefix'] == pointer['journal_prefix'],
                'Each immutable pointer must bind actual PRE-save/persistence/draw evidence')
        require(record(value['native_engine_source']['path']) == value['native_engine_source'],
                'Actual training engine source changed')
    require(indices == [21,22],
            'All captured checkpoints must be uniquely ordered through actual final native update')
    return value


def recover_all(binding, *, model, optimizer, context, augmentation, session,
                parent_source, torch, folder, run_phase, guard):
    requests = verified_requests(binding, session=session, parent_source=parent_source, guard=guard)
    folder = Path(folder).resolve()
    require(folder.is_relative_to(Path(r'E:\gpu_training_scratchpad').resolve())
            and not folder.exists() and callable(run_phase), 'Fresh separate recovery namespace/phase guard')
    folder.mkdir(); recovered = []
    for index,item in enumerate(requests['checkpoints']):
        guard(); pointer = read_bound(item['pointer'], guard)
        expected = normalize_fingerprint(read_bound(item['pre_save_fingerprint'], guard))
        expected_next = read_bound(item['next_draw_packet'], guard)
        def operation():
            state, journal, branch = restore(pointer['checkpoint'], torch=torch, model=model,
                optimizer=optimizer, identity=pointer['identity'], augmentation=augmentation,
                new_journal_path=folder/('checkpoint_'+str(pointer['committed_updates'])+'_restored_prefix.jsonl'),
                diagnostic_only=True, guard=guard)
            schedule = context.new_schedule(); schedule.load_state_dict(normalize_schedule_state(state))
            actual = live_state_fingerprint(optimizer, torch=torch, guard=guard)
            require_exact_live_restore(expected, actual)
            require(optimizer.updates == schedule.committed_updates() == pointer['committed_updates']
                    and digest(state) == pointer['sampler_state_sha256']
                    and all(journal.binding()[key] == pointer['journal_prefix'][key] for key in ('offset','sha256')),
                    'Fresh GPU recovery all-source clocks/samplers/authoritative prefix differ')
            actual_next = next_draw_packet(schedule, context, optimizer_updates=optimizer.updates, guard=guard)
            require(digest(actual_next) == digest(expected_next),
                    'Fresh recovered actual next draw/fullsource labels/contexts/replay RNG differ')
            require_exact_live_restore(expected, live_state_fingerprint(optimizer,torch=torch,guard=guard))
            require(digest(schedule.state_dict()) == pointer['sampler_state_sha256'],
                    'Recovery next-draw comparison must not consume a draw')
            return {'passed':True,'pointer':item['pointer'],'checkpoint':pointer['checkpoint'],
                'pre_save_fingerprint':item['pre_save_fingerprint'],'next_draw_packet':item['next_draw_packet'],
                'actual_restored_state_sha256':actual['state_sha256'],'updates':optimizer.updates,
                'all207_threeAdam_all4RNG_match_actual_pre_save':True,
                'both_samplers_journal_prefix_and_next_draw_exact':True,'branch':branch,
                'native_updates_performed':0,'draws_consumed':0,'separate_fresh_GPU_recovery':True}
        proof,memory = run_phase('deferred_checkpoint_'+str(pointer['committed_updates']),operation)
        proof['recovery_memory'] = memory
        path = folder/('actual_recovery_'+str(pointer['committed_updates'])+'.json');write_json(path,proof)
        recovered.append(record(path))
    require(len(recovered) == len(requests['checkpoints']) and optimizer.updates == requests['expected_global_new_updates'],
            'Every checkpoint and final all-source frontier must actually recover before passing')
    result={'kind':'ASR18_actual_continuation21_22_all_new_captures_fresh_GPU_recovery_v2','passed':True,
        'request':binding,'session':session,'native_process_completions':requests['native_process_completions'],
        'recovered_checkpoints':recovered,'all_captured_checkpoints_recovered':True,
        'actual_global_native_updates':requests['expected_global_new_updates'],
        'initial_inherited_global_updates':20,'actual_new_training_updates_in_source_chain':2,
        'ancestor':requests['ancestor'],'all_old_ancestor_captures_recovered_in_prior_actual0_only':True,
        'actual_native_updates_in_recovery_process':0,'actual_draws_consumed_in_recovery':0,
        'qualified_checkpoint':False,'production_enabled':False,'source':record(__file__)}
    path=folder/'all_recoveries.json';write_json(path,result);return record(path)
