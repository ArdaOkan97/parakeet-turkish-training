"""One bounded stdlib metadata publication; never reads tensors or history arrays."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
RUN_ID = 'adaptive_20261013_TR_uniform_headlr3e5_cosine_01'
RUN = OUT / 'runs' / RUN_ID
LOOP = OUT / 'runs' / (RUN_ID + '_loop_01')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.metadata.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
status = read(LOOP / 'loop_status.json')
assert status['status'] == 'running' and status['run_id'] == RUN_ID
proof_path = RUN / 'majority_initial_source_equivalence.json'
proof = read(proof_path)
assert proof['passed'] and proof['source_selected_step'] == 25109
assert proof['compared_fixed_DEV_utterances'] == 350
assert proof['all_raw_transcripts_and_emitted_paths_exact'] and proof['all_seven_ASR_counts_exact']
assert proof['RNG_restored_after_added_evaluation'] and not proof['sampler_drawn_during_check']
config = read(RUN / 'run_config.json')
identity = read(status['paths']['identity'])
log = Path(status['paths']['log'])
with log.open('rb') as stream:
    stream.seek(max(0, log.stat().st_size - 2_000_000))
    lines = stream.read().decode('utf-8', errors='replace').splitlines()
rows = []
for line in lines:
    if line.startswith('{"step":'):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row['step'] > 25109:
            rows.append(row)
assert rows, 'Actual positive optimizer stdout required'
last = rows[-1]
assert last['head_lr'] == 3e-5 and last['encoder_lr_schedule'] == 'cosine'
step = int(last['step'])
cycle = last['Turkish_domain_coverage']['matched_cycle']
policy_record = record(OUT / 'completed_encoder_schedule_policy_selection_v1.json')
constant_record = record(OUT / 'matched_encoder_constant_actual_CLOSED43238_fullcycle_preservation_v1.json')
cosine_record = record(OUT / 'matched_encoder_cosine_actual_CLOSED43238_fullcycle_preservation_v1.json')
dispatch = {
    'schema': 1, 'kind': 'TSC_uniform_head_LR_actual_source350_and_positive_updates_v1',
    'passed': True, 'observed_at_utc': now, 'run_id': RUN_ID,
    'actual_controller_session_id': 64368, 'actual_launch_tool_chunk': 'a4f720',
    'actual_native_launcher_PID': status['owned_launcher_pid'], 'actual_worker_PID': identity['pid'],
    'source_step': 25109, 'current_optimizer_step': step, 'positive_updates': step - 25109,
    'source_receipt': record(OUT / 'TSC_uniform_headlr3e5_cosine25109_source_v1.json'),
    'source350_proof': record(proof_path), 'source350_exact': True,
    'source350_all_six_retention_gates_passed': proof['all_six_retention_gates_passed'],
    'source350_failed_gates_not_relabelled': True,
    'actual_head_lr': last['head_lr'], 'actual_encoder_lr': last['lora_lr'],
    'encoder_lr_schedule': last['encoder_lr_schedule'],
    'matched_cycle': cycle, 'retention_stop_policy': last['retention_stop_policy'],
    'fresh_strict_memory_admission': status['fresh_memory_admission'],
    'latest_owned_memory': status['system_and_owned_private_memory'],
    'owned_identity': record(status['paths']['identity']),
    'native_backend_ready': record(status['paths']['native']),
    'controller': record(OUT / 'run_TSC_uniform_head_lr_v1.py'),
    'worker': record(OUT / 'train_lora_batched_tsc_uniform_head_lr_v1.py'),
    'journal_adapter': record(OUT / 'run_TSC_uniform_head_lr_journal_v1.py'),
    'encoder_policy_choice': policy_record,
    'matched_control_reuse': cosine_record,
    'all207_restored_and_initial_save_precede_positive_update': True,
    'evidence_scope': 'Actual source350 proof and native optimizer stdout, frozen save-before-loop ordering; no live checkpoint hashing or tensor inspection.',
    'standalone_monitor_env_parser_import_passed': False,
    'standalone_parser_limit': 'Monitor-env -S import stopped at missing top-level torch before ML imported; real native parser/validation was exercised by the positive worker.',
    'quality_improvement_or_gradient_equivalence_claimed': False,
    'reserved_TEST_used': False, 'pagefile_or_guards_changed': False,
    'current_frontier_is_durable_checkpoint_claimed': False,
}
dispatch_path = OUT / 'matched_headlr3e5_cosine_actual_source350_positive_updates_v1.json'
write(dispatch_path, dispatch)
dispatch_record = record(dispatch_path)
lease_path = ROOT / 'work/gpu_training_owner.json'
prior_lease = ROOT / 'work/gpu_training_owner_pre_headlr_launch_v1.json'
assert not prior_lease.exists(), 'One-time prior lease preservation must not be overwritten'
prior_lease.write_bytes(lease_path.read_bytes())
lease = {
    'schema': 1, 'kind': 'single_GPU_training_ownership_v1', 'owner': '/root/benchmark_design',
    'authorized_by': 'Direct user WER iteration authorization and ROOT sole-GPU delegation',
    'status': 'actual_head_LR_control_source350_passed_positive_updates', 'updated_at_utc': now,
    'target_run_id': RUN_ID, 'actual_GPU_running': True, 'actual_GPU_launched': True,
    'running': True, 'reserved': True, 'GPU_reserved': True,
    'ROOT_will_not_launch_concurrent_GPU': True,
    'actual_controller_session_id': 64368, 'actual_launch_tool_chunk': 'a4f720',
    'actual_owned_native_launcher_pid': status['owned_launcher_pid'], 'actual_owned_worker_pid': identity['pid'],
    'actual_controller_loop': str(LOOP.resolve()), 'actual_completed_step': step,
    'source_frontier': 25109, 'actual_encoder_schedule_updates': step - 25109,
    'current_matched_cycle_unique': cycle['cycle_unique'], 'current_matched_cycle_complete': False,
    'actual_head_lr': last['head_lr'], 'actual_applied_encoder_lr': last['lora_lr'],
    'actual_source350_passed': True, 'source350_and_first_optimizer_pending': False,
    'actual_dispatch_receipt': dispatch_record, 'actual_source_receipt': dispatch['source_receipt'],
    'prior_ownership_metadata': record(prior_lease),
    'prior_constant_complete': constant_record, 'prior_cosine_complete': cosine_record,
    'measured_encoder_policy_choice': policy_record,
    'qualified_champion_step': 42500, 'qualified_champion_TR_WER': 279 / 825,
    'qualified_champion_scope': 'Archived completed cosine control; independent of current head branch',
    'actual_fresh_strict_memory_admission': status['fresh_memory_admission'],
    'latest_owned_worker_self_system_commit_observation': status['system_and_owned_private_memory']['system_memory'],
    'retention_early_stop_active': False, 'normal_budget_exact_chaining_active': True,
    'owner_task': 'Supervise complete18129-update matched head-LR control and complete endDEV; exact resume actual normal budget closures.',
    'permanent_resource_policy': {'pagefile_change_authorized': False, 'pagefile_request_prohibited': True, 'guards_unchanged': True, 'user_apps_untouched': True},
    'objective_complete': False,
}
write(lease_path, lease)
plan_path = OUT / 'experiment_plan.json'
plan = read(plan_path)
champion = {'run_id': 'adaptive_20261012_TR_uniform_encoder_cosine_01', 'step': 42500, 'TR_word_errors': 279, 'TR_reference_words': 825, 'TR_WER': 279 / 825, 'all_six_passed': True, 'archive_preservation': cosine_record}
current = {
    'run_id': RUN_ID, 'status': lease['status'], 'actual_controller_session_id': 64368,
    'actual_native_PID': status['owned_launcher_pid'], 'actual_worker_PID': identity['pid'],
    'actual_loop': str(LOOP.resolve()), 'actual_GPU_running': True,
    'source_step': 25109, 'current_frontier': step, 'local_encoder_schedule_updates': step - 25109,
    'matched_updates': 18129, 'configured_end_step': 43238, 'mode': 'cosine',
    'declared_head_lr': 3e-5, 'applied_encoder_lr': last['lora_lr'],
    'source350_passed': True, 'source350_and_first_optimizer_pending': False,
    'cycle2_unique_including_carried': cycle['cycle_unique'], 'retention_stop_policy': 'monitor_only',
    'actual_dispatch_receipt': dispatch_record, 'global_qualified_champion': champion,
    'last_complete_DEV_step': 25109,
    'last_DEV_counts': {key: value['asr'] for key, value in proof['comparisons'].items()},
    'last_complete_DEV_TR_WER': proof['comparisons']['tr']['asr']['wer'],
    'last_DEV_all_six_passed': False, 'last_DEV_scope': 'Actual reproduced source25109; no new head trial periodic DEV claimed yet',
    'source_retention_counter': 23, 'actual_history_transport': 'append_only_journal_with_DEV_and_closure_snapshots',
}
plan.update(updated_at_utc=now, actual_GPU_running=True, current_training=current, actual_current_training=current,
    completed_constant_control=constant_record, measured_encoder_policy_choice=policy_record)
plan['status_authority'] = f'Matched encoder controls both CLOSED0 at43,238/full cycle253,798. Cosine qualified BEST279/825 versus constant286/825 selected cosine from one measured pair. Head-LR3e-5 challenger actually running at{step:,} (+{step-25109:,}), owned64368/native{status["owned_launcher_pid"]}/worker{identity["pid"]}; source350 exact and original207 Adam restored before live head LR override. Same original25,109 state/examples, full18,129 updates, strict winner gates and monitor-only retention; TEST frozen.'
plan['automatic_execution_policy']['GPU_ownership'] = 'benchmark_design owns sole head-LR trial session64368; ROOT will not launch concurrent GPU.'
plan['current_post_epoch_trial'] = {'status': lease['status'], 'run_id': RUN_ID, 'session_id': 64368, 'source_step': 25109, 'matched_next_cycle_updates': 18129, 'single_scientific_factor': 'Head LR1e-5 to3e-5; selected encoder cosine held fixed', 'measured_policy_choice': policy_record, 'matched_control_reuse': cosine_record, 'actual_dispatch': dispatch_record}
write(plan_path, plan)
protocol_path = OUT / 'post_epoch_experiment_protocol.json'
protocol = read(protocol_path)
protocol.update(status='encoder_pair_complete_cosine_selected_head_LR3e5_actually_training', updated_at_utc=now)
protocol['encoder_schedule_actual_status']['policy_selection_completed'] = True
protocol['encoder_schedule_actual_status']['policy_choice'] = policy_record
protocol['encoder_schedule_actual_status']['constant'] = {'status': 'actual_CLOSED0_complete_fullcycle', 'run_id': 'adaptive_20261012_TR_uniform_encoder_constant_01', 'completed_step': 43238, 'qualifying_BEST_step': 38500, 'qualifying_BEST_TR_WER': 286/825, 'preservation': constant_record}
for item in protocol['ordered_next_work']:
    if item['order'] == 3:
        item.update(status='both_full_cycles_completed_cosine_selected_from_actual_curve', actual_policy_choice=policy_record)
    if item['order'] == 4:
        item.update(status='actual_head_LR3e5_source350_passed_positive_updates', actual_policy_choice_or_source_admission=True, actual_dispatch=dispatch_record, matched_control_reuse=cosine_record)
write(protocol_path, protocol)
print(json.dumps({'passed': True, 'step': step, 'positive_updates': step - 25109, 'cycle_unique': cycle['cycle_unique'], 'dispatch': dispatch_record, 'lease_updated': True, 'plan_updated': True, 'protocol_updated': True}))
