"""Actual owned closure bookkeeping, small JSON plus one bounded history tail."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import ctypes, hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'outputs/parakeet_turkish'
RUN = 'adaptive_20261012_TR_uniform_majority_PT_focus_02'
DIR = HERE / 'runs' / RUN
LOOP = HERE / 'runs' / (RUN + '_loop_01')

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def record(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}

def atomic(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

controller = load(LOOP / 'loop_status.json')
completion_path = LOOP / (RUN + '_native_backend.completion.json')
completion = load(completion_path)
selection_path = DIR / 'selection.json'
selection = load(selection_path)
summary_path = HERE / 'results' / (RUN + '_step_8250') / 'summary.json'
summary = load(summary_path)
source_proof_path = DIR / 'majority_initial_source_equivalence.json'
source_proof = load(source_proof_path)
assert controller['status'] == 'train_window_ended' and controller['child_exit_code'] == 0
assert controller['stop_category'] == 'retention' and controller['completed_steps'] == 8250
assert completion['status'] == 'complete' and completion['sources_unchanged'] is True
assert selection['completed_steps'] == 8250 and selection['consecutive_retention_failures'] == 5
assert selection['accepted'] is False and selection['selected_step'] is None
assert source_proof['passed'] and source_proof['actual_GPU_decode']
assert source_proof['compared_fixed_DEV_utterances'] == 350
assert source_proof['source_selected_step'] == 8000

# Check only the three owned, previously recorded PIDs. No process enumeration.
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
kernel.OpenProcess.restype = ctypes.c_void_p
kernel.CloseHandle.argtypes = [ctypes.c_void_p]
absences = []
for pid in (39204, 33432, 5728):
    ctypes.set_last_error(0)
    handle = kernel.OpenProcess(0x1000, 0, pid)
    error = ctypes.get_last_error()
    if handle:
        kernel.CloseHandle(handle)
        raise RuntimeError(f'Owned PID {pid} still exists; do not release GPU lease')
    assert error == 87, (pid, error)
    absences.append({'pid': pid, 'OpenProcess_handle': None, 'winerror': error, 'absent': True})

history_path = DIR / 'training_history.json'
with history_path.open('rb') as stream:
    stream.seek(max(0, history_path.stat().st_size - 350000))
    tail = stream.read().decode('utf-8', errors='ignore')
position = tail.rfind('\n  {\n    "step":')
assert position >= 0
raw = tail[position + 3:].strip()
assert raw.endswith(']')
row = json.loads(raw[:-1].rstrip())
assert row['step'] == 8250 and row['PT_focus_updates'] == 250
uniform = row['Turkish_domain_coverage']['uniform_pass']
assert uniform['draws'] == uniform['unique_clips_seen'] == 17774
assert uniform['repeat_visits'] == 0 and not uniform['full_pass_complete']
focus = row['Turkish_domain_coverage']['PT_focus_transition']
assert focus['Portuguese_draws_after_transition'] == focus['old_five_draws_after_transition'] == 250

plan_path = HERE / 'experiment_plan.json'
plan = load(plan_path)
limits = plan['retention_max_fixed_DEV_word_errors']
counts = {language: summary['validation_' + language]['asr'] for language in ('tr', 'en', 'es', 'it', 'fr', 'de', 'pt')}
assert all(score['utterances'] == 50 for score in counts.values())
count_pass = {language: counts[language]['word_errors'] <= limits[language] for language in ('en', 'es', 'it', 'fr', 'de', 'pt')}
assert all(count_pass.values())
baseline_proof = load(HERE / 'Portuguese_retention_baselines_v1.json')
baseline = baseline_proof['baseline_overlay']['validation_pt']['asr']
actual_pt = counts['pt']
assert baseline['word_errors'] == 60 and actual_pt['word_errors'] == 66
assert actual_pt['reference_words'] == baseline['reference_words'] == 1095
float_limit = max(0.005, 0.1 * baseline['wer'])
float_threshold = baseline['wer'] + float_limit
exact_base = Fraction(60, 1095)
exact_threshold = exact_base + max(Fraction(1, 200), exact_base / 10)
assert Fraction(66, 1095) == exact_threshold
assert actual_pt['wer'] > float_threshold
assert row['validation']['retention_gate_passed'] is False

now = datetime.now(timezone.utc).isoformat()
receipt = {
    'schema': 1, 'kind': 'PT_focus02_owned_actual_CLOSED_numeric_retention_boundary_v1',
    'observed_at_utc': now, 'status': 'actual_CLOSED0_retention_numeric_boundary_review_required',
    'run_id': RUN, 'actual_owned_exit': {'session_id': 51956, 'exit_code': 0, 'tool_chunk': '7eb92f'},
    'actual_owned_PID_absence': absences, 'native_child_exit_code': 0,
    'completed_steps': 8250, 'PT_focus_updates': 250, 'uniform_origin_step': 6500,
    'quota_boundary_step': 7621, 'lower_LoRA_boundary_step': 7750, 'PT_focus_boundary_step': 8000,
    'new_uniform_unique_TR_clips': 17774, 'uniform_TR_pool': 253798,
    'new_uniform_fraction': uniform['fraction_unique_seen'], 'repeat_visits': 0,
    'configured_end_step': 25109, 'remaining_global_updates': 25109 - 8250,
    'full_new_uniform_pass_complete': False, 'source350_actual_passed': True,
    'all207_initial_serialization_and_native_positive_updates_completed': True,
    'head_lr': row['head_lr'], 'lora_lr': row['lora_lr'],
    'actual_language_weights': row['preservation_coefficients_by_language'],
    'source8000_word_errors': {'tr': 329, 'en': 59, 'es': 49, 'it': 23, 'fr': 65, 'de': 70, 'pt': 72},
    'actual_DEV8250_ASR': counts, 'fixed_retention_max_word_errors': limits,
    'all_six_pass_by_exact_word_counts': True, 'retention_language_count_passed': count_pass,
    'actual_recorded_gate_passed': False, 'actual_stop_category': 'retention',
    'actual_failure_counter': 5, 'actual_selection_accepted': False, 'actual_selected_step': None,
    'numerical_boundary': {'language': 'pt', 'base_word_errors': 60, 'candidate_word_errors': 66,
        'reference_words': 1095, 'candidate_wer_float': actual_pt['wer'],
        'base_wer_float': baseline['wer'], 'limit_float': float_limit,
        'summed_threshold_float': float_threshold, 'float_excess': actual_pt['wer'] - float_threshold,
        'exact_candidate_equals_exact_allowed_threshold': True,
        'source': 'train_lora_batched_tsc_uniform_majority_ptfocus_v2.py:204',
        'interpretation': 'Binary floating addition rejects a mathematically permitted boundary; no retention threshold relaxation requested'},
    'source350': record(source_proof_path), 'summary': record(summary_path),
    'controller_status': record(LOOP / 'loop_status.json'), 'selection': record(selection_path),
    'runtime_completion': record(completion_path), 'sources_unchanged': True,
    'runtime_evidence': {}, 'observed_peak_owned_private_bytes': controller['peak_observed_owned_private_bytes'],
    'last_actual_history_row_canonical_sha256': hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
    'latest_checkpoint': {'path': selection['latest_weights'], 'bytes': Path(selection['latest_weights']).stat().st_size,
                          'checkpoint_bytes_read_or_hashed_by_this_check': False},
    'no_model_audio_checkpoint_or_GPU_reads': True, 'goal_complete': False,
    'continuation_requires_separate_numerical_gate_correction_and_source_review': True,
    'actual_false_gate_history_or_selection_rewritten': False,
}
for role in ('memory', 'cache', 'io'):
    path = LOOP / (RUN + '_' + role + '.json')
    evidence = load(path)
    assert evidence['status'] == 'complete' and evidence['sources_unchanged'] is True
    receipt['runtime_evidence'][role] = record(path)
receipt_path = HERE / 'PT_focus02_actual_CLOSED8250_numeric_boundary_review_v1.json'
with receipt_path.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, ensure_ascii=False, indent=2)
receipt_record = record(receipt_path)

lease_path = ROOT / 'work/gpu_training_owner.json'
lease = load(lease_path)
assert lease['actual_controller_session_id'] == 51956 and lease['owner'] == '/root/benchmark_design'
lease.update(status='actual_CLOSED0_GPU_released_to_ROOT_for_numeric_gate_review', owner='/root',
    previous_owner='/root/benchmark_design', updated_at_utc=now, actual_exit_code=0,
    actual_closed_tool_chunk='7eb92f', actual_GPU_running=False, GPU_reserved=False,
    ROOT_will_not_launch_concurrent_GPU=False, actual_completed_step=8250,
    actual_PT_focus_updates=250, actual_new_uniform_unique_TR_clips=17774,
    actual_source_receipt=record(HERE / 'TSC_majority8000_PT_focus_ROOT_source_v2.json'),
    actual_closure_receipt=receipt_record, continuation_requires_numeric_gate_review=True)
atomic(lease_path, lease)
plan['status_authority'] = 'Owned PT_focus02 session51956 actuallyCLOSED0/7eb92f at8250 after250 updates; PT72→66 exactlyallowed66, TR329 unchanged, alloldfivecountspass. Frozenworkerfloatthreshold rejected66 by6.94e-18; actualfailure5/selectionNone retained. GPUreleasedtoROOT; numericalgatecorrection/source-review needed beforecontinuation. Fullunique17774/253798,no repeats; goal incomplete.'
plan['actual_current_training'] = {**receipt, 'actual_GPU_running': False,
    'worker': 'train_lora_batched_tsc_uniform_majority_ptfocus_v2.py',
    'controller': 'run_TSC_uniform_majority_ptfocus_v2.py', 'logger': 'track_continuation_v14.py',
    'same_Trackio_project': 'parakeet-turkish', 'closure_receipt': receipt_record}
item = next(item for item in plan['ordered_experiments'] if item['id'] == '04')
item.update(status='actual_PT_focused250_CLOSED_numeric_gate_repair_pending', run_id=RUN,
    actual_current_run=RUN, source_receipt='TSC_majority8000_PT_focus_ROOT_source_v2.json',
    actual_completed_steps=8250, actual_PT_focus_updates=250, actual_GPU_training_running=False,
    actual_new_uniform_unique_TR_clips=17774, actual_new_uniform_unique_clips=17774,
    current_frontier_source_step=8000, current_remaining_uniform_updates=16859,
    current_configured_end_step=25109, active_configured_end_step=25109,
    actual_majority_updates=629, current_lora_lr_fork_updates=500,
    actual_closed_session=51956, actual_closed_exit_code=0, actual_closed_tool_chunk='7eb92f',
    actual_closure_review=receipt_path.name, current_failure_counter=5,
    latest_original_five_passed=True, latest_Portuguese_passed_by_exact_word_count=True,
    latest_actual_float_gate_passed=False, all_six_champion_yet=False,
    current_PT_word_errors=66, current_TR_word_errors=329,
    initial_all_six_champion_claimed=False, actual_full_pass_completed=False,
    execution_owner='ROOT after benchmark_design actual owned closure and explicit GPU lease release')
item['pending'] = ['Correct numerical comparison at exact fixed retention boundary in separate sibling; preserve recorded history',
    'ROOT review of actual latest8250 full-state continuation without coverage/moment/grace reset',
    'Complete existing253798 unique Turkish pass subject to unchanged retention/resource gates',
    'Frozen heldout/export/confidence objective verification']
item.setdefault('actual_DEV_points', []).append({'run_id': RUN, 'step': 8250, 'PT_focus_updates': 250,
    'TR_WER': counts['tr']['wer'], 'actual_counts': {k:{a:v[a] for a in ('word_errors','reference_words','utterances')} for k,v in counts.items()},
    'all_six_passed_by_exact_counts': True, 'actual_recorded_gate_passed': False,
    'strict_all_six_champion_selected': False, 'numerical_boundary_bug': True,
    'summary': record(summary_path), 'closure_receipt': receipt_record})
plan['automatic_execution_policy']['PT_focused_replay'] = 'PTfocused250 actualCLOSED0 at8250: PT72→66 exactlyfixedlimit, oldfivepass andTR329 unchanged; binaryfloatthreshold falselystopped. Preserveactualfailure5/history/selectionNone. No unchangedretry; new numericalgatecorrection+ROOTsource-review maycontinue SAMEpass/fullstate. GPUreleasedtoROOT, V14solemirror retainsallmetrics.'
plan['automatic_execution_policy']['GPU_ownership'] = 'Read work/gpu_training_owner.json: benchmark_design actualsession51956 CLOSED0/7eb92f, knownPIDs absent, lease releasedtoROOT. No GPU is running under benchmark_design. Nextlaunch requires source/numericalgate review, actualmemoryadmission, and soleownership.'
plan['updated_at_utc'] = now
plan['objective_complete'] = False
atomic(plan_path, plan)
notes = 'Actual PT_focus02 owned51956 CLOSED0/7eb92f at8250 after250 positiveupdates; nativechild0/complete,sourcesunchanged,allownedPIDsabsent. Source350exact and207initialsave passed. TR329/82539.8788 unchanged; EN60≤69 ES50≤52 IT24≤30 FR63≤74 DE70≤70 PT66/1095≤66 (sourcePT72). Exactall6wordcountspass, but frozenWERfloatgate compares .06027397260273973 > .06027397260273972, falselyrejectingallowedboundary by6.94e-18; recordedfailure5/acceptedFalse/selectionNone preserved. Uniformunique17774/253798,no repeat,configuredend25109; coverage/moments/grace neverreset. GPUlease releasedtoROOT; next requires minimal separate numericalgate correction/source-review, not unchangedretry. SameprojectV14solelogger continues. Evidence PT_focus02_actual_CLOSED8250_numeric_boundary_review_v1.json; no checkpointbytes/model/audio/GPUread in closurebookkeeping. Goal incomplete.'
for task in ('ASR-15', 'ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'), 'task', 'edit', task, '--status', 'In Progress', '--append-notes', notes],
        cwd=HERE, check=True, capture_output=True, text=True, encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'), str(HERE/'refresh_backlog.py')],
    cwd=ROOT, check=True, capture_output=True, text=True, encoding='utf-8')
print(json.dumps({'receipt': receipt_record, 'closed_session': 51956, 'exit': 0, 'tool_chunk': '7eb92f',
    'PT_word_errors': 66, 'TR_word_errors': 329, 'all_six_count_pass': True,
    'GPU_lease_released_to_ROOT': True, 'plan_Backlog_refresh_updated': True}, ensure_ascii=False))
