"""Bounded saved JSON/log status maintenance; never opens checkpoint bytes."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
RUN = 'adaptive_20261012_TR_uniform_full_epoch_01'
DIR = OUT / 'runs' / RUN
LOOP = OUT / 'runs' / (RUN + '_loop_01')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def record(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def tail(path, cap):
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - cap))
        return handle.read().decode('utf-8', 'replace')

def atomic(path, value):
    tmp = path.with_suffix('.status.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)

status = read(LOOP / 'loop_status.json')
assert status['status'] == 'running' and status['owned_launcher_pid'] == 5516
history_tail = tail(DIR / 'training_history.json', 8_000_000)
decoder = json.JSONDecoder()
validation = {}
for match in re.finditer(r'(?m)^  \{', history_tail):
    try:
        row, _ = decoder.raw_decode(history_tail, match.start() + 2)
    except ValueError:
        continue
    if row.get('step') in (12750, 13000) and 'validation' in row:
        validation[row['step']] = row
winner = validation[12750]
last_dev = validation[13000]
assert winner['validation']['retention_gate_passed'] is True
assert winner['selection_diagnostics']['improved_qualifying_best'] is True
assert winner['selection_diagnostics']['consecutive_retention_failures'] == 0
assert last_dev['selection_diagnostics']['improved_qualifying_best'] is False

log_tail = tail(Path(status['paths']['log']), 2_200_000)
best_lines = [line for line in log_tail.splitlines()
              if line.startswith('dev step') and 'best12750' in line]
assert any(line.startswith('dev step12750 ') for line in best_lines)
assert any(line.startswith('dev step13000 ') for line in best_lines)
progress = []
for line in log_tail.splitlines():
    if not line.startswith('{'):
        continue
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if 'step' in row and 'loss' in row:
        progress.append(row)
current = progress[-1]
uniform = current['Turkish_domain_coverage']['uniform_pass']
assert current['step'] >= 13000
assert current['head_lr'] == 1e-5 and current['lora_lr'] == 1e-4
assert current['retention_stop_policy'] == 'monitor_only'
assert uniform['draws'] == uniform['unique_clips_seen'] == 28274 + 14 * (current['step'] - 9000)
assert uniform['repeat_visits'] == 0 and uniform['full_pass_complete'] is False
summary_path = OUT / 'results' / (RUN + '_step_12750') / 'summary.json'
summary = read(summary_path)
counts = {lang: summary['validation_' + lang]['asr']
          for lang in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')}
limits = {'en': 69, 'es': 52, 'it': 30, 'fr': 74, 'de': 70, 'pt': 66}
assert counts['tr']['word_errors'] == 314 and counts['tr']['reference_words'] == 825
assert all(counts[lang]['word_errors'] <= limit for lang, limit in limits.items())
best = ROOT / 'work' / (RUN + '_best.pt')
best_stat = best.stat()  # Stat only. No live checkpoint hash or tensor load.
now = datetime.now(timezone.utc).isoformat()
receipt = {
    'kind': 'USER_full_epoch12750_actual_qualified_BEST_milestone_v1',
    'passed': True, 'observed_at_utc': now, 'run_id': RUN,
    'actual_controller_session_id': 67197, 'status': 'actual_sole_GPU_running',
    'qualifying_BEST_step': 12750, 'TR_word_errors': 314, 'TR_reference_words': 825,
    'TR_WER': 314 / 825, 'all_six_retention_passed': True,
    'counts': {lang: {key: score[key] for key in ('word_errors', 'reference_words', 'utterances')}
               for lang, score in counts.items()},
    'retention_max_word_errors': limits, 'summary': record(summary_path),
    'saved_selection_metadata': winner['selection_diagnostics'],
    'saved_BEST_evidence': {
        'checkpoint_path': str(best.resolve()), 'bytes': best_stat.st_size,
        'last_write_utc': datetime.fromtimestamp(best_stat.st_mtime, timezone.utc).isoformat(),
        'completed_save_then_log_order': True, 'bounded_log_confirmation': best_lines,
        'live_checkpoint_bytes_read_or_hashed': False,
        'periodic_selection_json_not_required_for_in_memory_selection': True,
    },
    'latest_complete_DEV_step': 13000,
    'latest_complete_DEV_all_six_passed': last_dev['validation']['retention_gate_passed'],
    'latest_complete_DEV_counts': last_dev['validation']['retention_word_errors'],
    'latest_monitored_failure_counter': last_dev['selection_diagnostics']['consecutive_retention_failures'],
    'current_published_step': current['step'],
    'full_epoch_continuation_updates': current['step'] - 9000,
    'uniform_unique_TR_clips': uniform['unique_clips_seen'],
    'uniform_pool_clips': 253798, 'uniform_fraction': uniform['fraction_unique_seen'],
    'uniform_repeat_visits': 0, 'full_pass_complete': False,
    'configured_end_step': 25109, 'remaining_updates': 25109 - current['step'],
    'retention_early_stop_active': False, 'training_recipe_or_source_changed': False,
    'scope': 'Saved small DEV summary, bounded 8 MB history tail, bounded log tail, and live BEST file stat only.',
    'progress_is_not_a_durable_checkpoint': True, 'objective_complete': False,
}
receipt_path = OUT / 'USER_full_epoch12750_actual_qualified_BEST_milestone_v1.json'
with receipt_path.open('x', encoding='utf-8') as handle:
    json.dump(receipt, handle, indent=2, ensure_ascii=False)
receipt_record = record(receipt_path)

lease_path = ROOT / 'work/gpu_training_owner.json'
lease = read(lease_path)
assert lease['owner'] == '/root/benchmark_design' and lease['actual_controller_session_id'] == 67197
lease.update(status='actual_USER_full_epoch_running_with_qualified_BEST12750', updated_at_utc=now,
    owner_task='Supervise the sole full-epoch run through all 253798 unique Turkish clips and complete end DEV. Retention is monitoring and champion selection only; exact normal budget windows continue automatically.',
    actual_completed_step=current['step'], actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],
    actual_full_epoch_continuation_updates=current['step']-9000,
    actual_PT_focus_updates=current['PT_focus_updates'], qualified_champion_step=12750,
    qualified_champion_TR_WER=314/825, qualified_champion_receipt=receipt_record,
    actual_last_complete_DEV_step=13000, actual_retention_counter=receipt['latest_monitored_failure_counter'],
    actual_DEV_receipt=receipt_record, current_complete_DEV_all_six_passed=False,
    running=True, reserved=True, actual_GPU_running=True, GPU_reserved=True,
    full_pass_complete=False, progress_is_not_durable_checkpoint=True)
# Older source closure fields describe previous attempts, never the live window.
if 'actual_closure_receipt' in lease:
    lease['prior_closed_source_receipt'] = lease.pop('actual_closure_receipt')
atomic(lease_path, lease)

plan_path = OUT / 'experiment_plan.json'
plan = read(plan_path)
plan['updated_at_utc'] = now
plan['objective_complete'] = False
plan['status_authority'] = (
    f'Actual sole full-epoch session 67197 remains running at published step {current["step"]}, '
    f'with {uniform["unique_clips_seen"]}/253798 new-pass unique Turkish clips. '
    'Saved all-six-qualified BEST12750 has Turkish 314/825 (38.0606%); latest complete DEV13000 '
    'fails PT68/66, so BEST12750 is preserved and training continues unchanged. '
    'Retention monitors/selects champions only; actual full pass and end DEV remain pending.')
policy = plan['automatic_execution_policy']
policy.update(
    healthy_training='Leave the sole GPU child running through the full pass. Do not prepare scalar screens.',
    poor_latest='Preserve the strict all-six-qualified champion while continuing the coherent full pass. Retention failures do not stop training.',
    full_data_priority='Complete the current uniform Turkish pass before changing the recipe. Exact normal budget windows continue despite monitored retention failures; resource, nonfinite, and crash protections remain active.',
    fullpass_stop='No retention, plateau, patience, or target-WER stop before actual 253798 new-pass unique clips. Require complete end DEV; memory/nonfinite/crash protections remain active.',
    Portuguese='PT is included in replay and every strict all-six champion gate. No renewed grace or arbitrary counter reset. Failed DEV gates are monitoring and selection only during the full pass.',
    Turkish_workload='Actual 14TR + 1PT + 1uniform-old-five per B8x2 update. PT preservation40/2/20 and old-five66.6667/3.3333/33.3333; current tested LR and full-state frontier continue unchanged.')
actual = plan['actual_current_training']
actual.update(observed_at_utc=now, completed_steps=current['step'],
    full_epoch_continuation_updates=current['step']-9000, uniform_unique_TR_clips=uniform['unique_clips_seen'],
    uniform_fraction=uniform['fraction_unique_seen'], remaining_updates=25109-current['step'],
    latest_complete_DEV_step=13000, latest_complete_DEV_all_six_passed=False,
    latest_monitored_failure_counter=receipt['latest_monitored_failure_counter'],
    qualifying_BEST_step=12750, qualifying_BEST_TR_WER=314/825, qualifying_BEST_receipt=receipt_record,
    status='actual_sole_GPU_running', full_pass_complete=False, objective_complete=False,
    progress_is_not_durable_checkpoint=True)
item = next(item for item in plan['ordered_experiments'] if item['id'] == '04')
item.update(
    recipe='Coherent latest9000 all207 weights/Adam/sampler/RNG/history; no coverage rewind. PT token200 reverted to40 and retention stop became monitor-only. Feature20/duration2/head1e-5/LoRA1e-4/14TR1PT1old-five/B8x2/native4/cap8 unchanged. Strict all-six champion gates and resource/nonfinite protections preserved.',
    fullpass_early_stop_policy=policy['fullpass_stop'], selection_policy='Select minimum Turkish DEV WER only among strict all-six-qualified checkpoints. Preserve previous champions and all failed-branch history; continue the full epoch through monitored retention failures.',
    pending=['Continue the unchanged full run through actual253798 unique Turkish clips and complete end DEV.'],
    frozen_worker_sha256=lease['worker_sha256'], frozen_controller_sha256=lease['controller_sha256'],
    actual_controller_session_id=67197, actual_completed_steps=current['step'],
    actual_full_epoch_continuation_updates=current['step']-9000,
    actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'], actual_new_uniform_unique_clips=uniform['unique_clips_seen'],
    current_remaining_uniform_updates=25109-current['step'], current_configured_end_step=25109,
    actual_full_pass_completed=False, actual_GPU_training_running=True,
    current_all6_selected_BEST_step=12750, current_all6_selected_BEST_TR_WER=314/825,
    current_failure_counter=receipt['latest_monitored_failure_counter'],
    latest_actual_DEV_step=13000, latest_actual_DEV_counts={lang: {key: score[key] for key in ('word_errors','reference_words','utterances')} for lang,score in {lang: read(OUT/'results'/(RUN+'_step_13000')/'summary.json')['validation_'+lang]['asr'] for lang in counts}.items()},
    latest_actual_TR_word_errors=317, latest_all_six_passed=False,
    latest_original_five_passed=True, latest_Portuguese_passed=False,
    latest_Portuguese_passed_by_exact_word_count=False, latest_failed_retention_languages=['pt'],
    current_PT_word_errors=68, current_TR_word_errors=317,
    current_lora_lr_fork_updates=current['lora_lr_fork_updates'], actual_PT_focus_updates=current['PT_focus_updates'],
    actual_majority_updates=current['majority_updates'], actual_new_uniform_updates=current['uniform_full_updates'],
    current_source_step=9000, current_frontier_source_step=9000,
    current_source_checkpoint=lease['actual_source_receipt']['path'],
    active_coherent_coverage_source='LATEST9000 28274 unique; current continuation has no rewind.',
    latest_user_steering='Complete the full Turkish pass; retention failures monitor/select champions only, never early-stop training.',
    actual_full_epoch_DEV_receipt=receipt_path.name, latest_DEV_review=receipt_path.name,
    all_six_champion_yet=True, retention_early_stop_active=False)
for stale in ('actual_ROOT_closed_exit_code','actual_ROOT_closed_exit_chunk','actual_ROOT_controller_session','actual_ROOT_launch_tool_chunk'):
    if stale in item:
        item.setdefault('historical_attempt_metadata', {})[stale] = item.pop(stale)
atomic(plan_path, plan)
note = (f'Full epoch milestone: actual sole GPU session 67197 remains running at published step {current["step"]}. '
    'Saved DEV12750 is all-six-qualified: TR314/825 (38.0606%); EN57<=69, ES52<=52, IT23<=30, FR64<=74, DE68<=70, PT66<=66. '
    'History marks improved qualifying BEST, BEST file stat and post-save logs confirm BEST12750; no checkpoint bytes were read or hashed. '
    'DEV13000 TR317/825 and PT68>66 leaves BEST12750 unchanged. '
    f'Current new-pass coverage is {uniform["unique_clips_seen"]}/253798 unique clips ({100*uniform["fraction_unique_seen"]:.2f}%), with no new-pass repeats. '
    'Training continues unchanged under the user full-epoch policy: retention monitors/selects champions only, resource/nonfinite safeguards remain active, and normal budget windows chain exact full state. '
    'Full pass/end DEV remain pending. Evidence: ' + receipt_path.name)
for task in ('ASR-15', 'ASR-15.1'):
    subprocess.run([str(OUT/'tools/backlog.exe'), 'task', 'edit', task, '--status', 'In Progress', '--append-notes', note],
                   cwd=OUT, check=True, capture_output=True, encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'), str(OUT/'refresh_backlog.py')],
               cwd=ROOT, check=True, capture_output=True, encoding='utf-8')
print(json.dumps({'receipt': receipt_record, 'BEST':12750, 'step':current['step'],
                  'unique_TR':uniform['unique_clips_seen'], 'lease_plan_Backlog_updated':True}))
