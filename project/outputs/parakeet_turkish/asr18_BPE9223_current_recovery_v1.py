"""Inactive fresh recovery of ONE current full-run checkpoint.

The actual native worker must have closed normally before this component is
called in a different sole-owned GPU worker. It constructs no model, launches
no process and performs no optimization, audio processing or authoritative draw.
Historical checkpoints are covered by persistence receipts, never fresh restores.
"""
from pathlib import Path
import os
import time

from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record, require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree, select, verify_saved_metadata
from fastmss_phase_aware_plan_bridge_v7 import read_bound
from fastmss_owned_process_closure_v2 import verify as verify_process, identity as process_identity
from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
from asr18_live_state_fingerprint_v1 import live_state_fingerprint, require_exact_live_restore
from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
from resource_watchdog_v2 import sample_memory


ROOT = Path(r'E:\gpu_training_scratchpad')
RESTORE_INCREMENT = 350000000
STARTUP_BOUND = 3000000000
WHOLE_WORKER_BOUND = STARTUP_BOUND + RESTORE_INCREMENT


def _bound(pin, guard, maximum=16 * 1024**2):
    require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}
            and type(pin['bytes']) is int and 0 < pin['bytes'] <= maximum,
            'Exact bounded actual metadata binding required')
    require(Path(pin['path']).resolve().is_relative_to(ROOT.resolve()), 'E metadata only')
    return read_bound(pin, guard)


def checked_request(request_pin, *, guard):
    """Light parent preflight; repeats inside actual fresh recovery worker."""
    request = _bound(request_pin, guard)
    require(request['kind'] == 'ASR18_full_run_CURRENT_checkpoint_fresh_recovery_request_v1'
            and type(request['identity']) is dict and request['identity']
            and type(request['planning_arguments']) is dict,
            'One immutable original full-run identity and planning inputs')
    for key in ('native_worker_source', 'native_parent_source',
                'recovery_worker_source', 'recovery_parent_source'):
        pin = request[key]
        require(type(pin) is dict and record(pin['path']) == pin, 'Exact process source: ' + key)
    native = _bound(request['current_native_process0'], guard)
    verify_process(native, kind='ASR18_owned_GPU_process_completion_v1',
                   owner='/root/memory_diagnosis', guard=guard)
    require(native['worker_source'] == request['native_worker_source']
            and native['parent_source'] == request['native_parent_source'],
            'Actual normally closed native process/source identity')
    report = _bound(native['report'], guard)
    capture = request['current_checkpoint_capture']
    require(report['passed'] is True and type(report['diagnostic_only']) is bool and report['diagnostic_only']==request['identity']['owner_validation_only']
            and report['run_identity_sha256'] == digest(request['identity'])
            and type(report['actual_native_updates_this_process']) is int
            and report['actual_native_updates_this_process'] > 0
            and report['all_exact_GPU_checkpoint_restores_deferred'] is True
            and digest(report['current_checkpoint']) == digest(capture),
            'Actual current complete native report, never a selected historical save')
    pointer = _bound(capture['pointer'], guard, 2 * 1024**2)
    update = pointer['committed_updates']
    require(type(update) is int and 1 <= update <= 401072
            and type(report['final_optimizer_updates']) is int
            and report['final_optimizer_updates'] == update
            and pointer['kind'] == 'ASR18_production_window_completed_checkpoint_pointer_v2'
            and type(pointer['diagnostic_only']) is bool and pointer['diagnostic_only']==request['identity']['owner_validation_only']
            and pointer['all_window_updates_committed'] is True
            and pointer['writer_returned_and_payload_verified'] is True,
            'Current completed-window exact source frontier')
    caller = pointer['identity']['caller_identity']
    require(caller['full_run_lifecycle_identity'] == request['identity']
            and caller['run_identity_sha256'] == digest(request['identity'])
            and capture['checkpoint'] == pointer['checkpoint']
            and capture['pre_save_fingerprint'] == caller['pre_save_fingerprint']
            and capture['next_draw_packet'] == caller['next_draw_packet']
            and capture['publication_role'] == 'completed_window_end'
            and capture['checkpoint_persistence_verified'] is True
            and capture['exact_GPU_restore_performed'] is False
            and capture['current_latest_recovery_required_before_phase_DEV_winner_advance'] is True,
            'Current actual PRE/save/nextdraw identity and honest deferred recovery')
    metadata = _bound(pointer['checkpoint'], guard, 2 * 1024**2)
    state = decode_scalar_tree(select(metadata['tree'], 'sampler'))
    semantic = verify_saved_metadata(metadata, identity=pointer['identity'], optimizer_updates=update,
        sampler_state=state, journal_prefix=pointer['journal_prefix'], diagnostic_only=request['identity']['owner_validation_only'])
    require(digest(state) == pointer['sampler_state_sha256']
            and metadata['payload'] == pointer['payload']
            and semantic == pointer['saved_semantic_state_verified'], 'Actual saved scalar/payload identity')
    pre = normalize_fingerprint(_bound(capture['pre_save_fingerprint'], guard))
    require(type(pre['state']['updates']) is int and pre['state']['updates'] == update
            and len(pre['state']['weights']) == 207 and len(pre['state']['Adam_partitions']) == 3
            and pre['state']['clocks'] == [64250 + update, 2000 + update, 62250 + update],
            'Actual independent PRE-save207/threeAdam/mature clocks')
    proof = _bound(capture['checkpoint_persistence_proof'], guard)
    require(proof['kind'] == 'ASR18_actual_streamed_saved_state_matches_independent_pre_save_v1'
            and proof['passed'] is True
            and proof['source'] == record(Path(__file__).with_name('asr18_streamed_persistence_fingerprint_v1.py'))
            and proof['worker_source'] == request['native_worker_source']
            and proof['checkpoint_pointer'] == capture['pointer']
            and proof['checkpoint'] == pointer['checkpoint'] and proof['payload'] == pointer['payload']
            and proof['pre_save_fingerprint'] == capture['pre_save_fingerprint']
            and proof['next_draw_packet'] == capture['next_draw_packet']
            and proof['independent_pre_save_state_sha256'] == proof['saved_state_sha256']
                == proof['live_after_state_sha256'] == pre['state_sha256']
            and proof['live_unchanged_after_persistence'] is True
            and proof['actual_GPU_restore_performed'] is False
            and proof['saved_semantic_state_verified'] == semantic
            and proof['sampler_state_sha256'] == pointer['sampler_state_sha256']
            and proof['journal_prefix'] == pointer['journal_prefix'],
            'Actual native-source PRE/saved/live persistence receipt')
    return request, pointer


def recover_current(request_pin, *, model, optimizer, bootstrap_snapshot, torch, stage,
                    admit, guard, journal_factory, journal_factory_source,
                    worker_source, parent_source, folder):
    request, pointer = checked_request(request_pin, guard=guard)
    require(request['recovery_worker_source'] == worker_source
            and request['recovery_parent_source'] == parent_source
            and record(worker_source['path']) == worker_source
            and record(parent_source['path']) == parent_source,
            'Exact source-bound fresh recovery service')
    native = _bound(request['current_native_process0'], guard)
    require(native['producer']['pid'] != os.getpid() and optimizer.updates == 0
            and not optimizer.poisoned and len(optimizer.parameters) == 207,
            'Different actual fresh worker with unadvanced original bootstrap')
    folder = Path(folder).resolve()
    require(folder.is_relative_to(ROOT.resolve()) and folder != ROOT.resolve()
            and not folder.exists() and folder.parent.is_dir(), 'Fresh E recovery result directory')
    folder.mkdir()
    guard()
    before = sample_memory(os.getpid(), 'full_run_current_recovery_before')
    require(before['paired_sampling_within_one_second']
            and before['worker_private_memory']['available'] and before['system_memory']['available'],
            'Fresh paired recovery observation')
    current = before['worker_private_memory']['private_bytes']
    require(current <= STARTUP_BOUND, 'Unchanged3GB current startup bound; no margin reduction')
    admit('full_run_current_checkpoint_restore', RESTORE_INCREMENT)
    monitor = PhasePrivateMonitor(); monitor.start(); began = time.monotonic()
    returned = False; result = None
    try:
        from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223, full_run_next_draw_packet
        from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
        from asr18_BPE9223_training_state_v1 import restore
        from asr18_BPE9223_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
        context = FullRunSourceContext9223(request['planning_arguments'], guard=guard)
        try:
            require(digest(bootstrap_snapshot) == digest(context.snapshot),
                    'Original BPE64250/parkedLoRA62250 snapshot, never diagnostic seed')
            capture = request['current_checkpoint_capture']
            expected = normalize_fingerprint(_bound(capture['pre_save_fingerprint'], guard))
            expected_next = _bound(capture['next_draw_packet'], guard)
            augmentation = MetadataCheckpointBinding(request['augmentation_identity'], guard=guard)
            state, journal, branch = restore(pointer['checkpoint'], torch=torch, model=model,
                optimizer=optimizer, identity=pointer['identity'], augmentation=augmentation,
                journal_factory=journal_factory, journal_factory_source=journal_factory_source,
                diagnostic_only=request['identity']['owner_validation_only'], guard=guard)
            schedule = context.new_schedule(); schedule.load_state_dict(normalize_schedule_state(state))
            actual = live_state_fingerprint(optimizer, torch=torch, guard=guard)
            require_exact_live_restore(expected, actual)
            require(optimizer.updates == schedule.committed_updates() == pointer['committed_updates']
                    and digest(state) == pointer['sampler_state_sha256']
                    and all(journal.binding()[key] == pointer['journal_prefix'][key]
                            for key in ('offset', 'sha256')), 'Actual complete sampler/replay/RNG/prefix frontier')
            next_packet = full_run_next_draw_packet(schedule, context, optimizer_updates=optimizer.updates, guard=guard)
            require(digest(next_packet) == digest(expected_next), 'Full-source provisional next draw or terminal packet exact')
            require_exact_live_restore(expected, live_state_fingerprint(optimizer, torch=torch, guard=guard))
            require(digest(schedule.state_dict()) == pointer['sampler_state_sha256'],
                    'Recovery verification consumed no authoritative draw')
            result = {'passed': True, 'pointer': capture['pointer'], 'checkpoint': pointer['checkpoint'],
                'pre_save_fingerprint': capture['pre_save_fingerprint'], 'next_draw_packet': capture['next_draw_packet'],
                'actual_restored_state_sha256': actual['state_sha256'], 'updates': optimizer.updates,
                'all207_threeAdam_all4RNG_match_actual_pre_save': True,
                'both_samplers_journal_prefix_and_next_draw_exact': True, 'branch': branch,
                'native_updates_performed': 0, 'draws_consumed': 0, 'separate_fresh_GPU_recovery': True}
        finally:
            context.close()
        returned = True
    finally:
        memory = monitor.finish(); memory['wall_seconds'] = time.monotonic() - began
        covered = memory['private_peak_bytes'] <= current + RESTORE_INCREMENT
        write_json(folder/'restore_phase_memory.json', {'before_private_bytes': current,
            'increment_bytes': RESTORE_INCREMENT, 'whole_worker_bound_bytes': WHOLE_WORKER_BOUND,
            'actual': memory, 'covered': covered, 'operation_returned': returned,
            'margin_bytes': current + RESTORE_INCREMENT - memory['private_peak_bytes']})
    require(returned and covered and memory['private_peak_bytes'] <= WHOLE_WORKER_BOUND,
            'Full restore/context/fingerprint/nextdraw must fit prospective350MB and whole3.35GB')
    result['recovery_memory'] = memory
    item_path = folder/'actual_current_recovery.json'; write_json(item_path, result)
    report = {'kind': 'ASR18_full_run_actual_CURRENT_checkpoint_recovered_v2', 'passed': True,
        'diagnostic_only': request['identity']['owner_validation_only'], 'request': request_pin, 'source': record(__file__), 'producer': process_identity(),
        'worker_source': worker_source, 'parent_source': parent_source,
        'current_latest_checkpoint_recovered': True, 'historical_fresh_GPU_restores_claimed': False,
        'current_native_process0': request['current_native_process0'],
        'run_identity_sha256': digest(request['identity']), 'recovered_checkpoints': [record(item_path)],
        'actual_native_updates_in_recovery_process': 0, 'actual_draws_consumed_in_recovery': 0,
        'qualified_checkpoint': False, 'production_enabled': False}
    report_path = folder/'current_recovery_report.json'; write_json(report_path, report)
    stage('full_run_current_checkpoint_recovery_verified', updates=optimizer.updates,
          report=record(report_path), memory=memory)
    return record(report_path)
