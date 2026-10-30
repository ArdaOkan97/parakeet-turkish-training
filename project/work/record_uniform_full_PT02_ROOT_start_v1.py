"""Record actual ROOT-owned launch evidence, without touching frozen training sources."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def rec(path):
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
old = OUT / 'runs/adaptive_20261012_TR_uniform_full_PT_01_loop_01'
new = OUT / 'runs/adaptive_20261012_TR_uniform_full_PT_02_loop_01'
completion = read(old / 'adaptive_20261012_TR_uniform_full_PT_01_native_backend.completion.json')
plan = read(new / 'plan.json')
state = read(new / 'loop_status.json')
identity = read(new / 'adaptive_20261012_TR_uniform_full_PT_02_identity.json')
ready = read(new / 'adaptive_20261012_TR_uniform_full_PT_02_native_backend.json')
assert completion['status'] == 'failed' and completion['sources_unchanged'] is False
assert state['status'] == 'running' and state['owned_launcher_pid'] == 34620
assert identity['pid'] == ready['pid'] == 23024 and identity['ppid'] == 34620
assert plan['controller_sha256'] == ready['launcher_sha256'] == rec(OUT/'run_TSC_uniform_full_v1.py')['sha256']
assert ready['trainer_sha256'] == rec(OUT/'train_lora_batched_tsc_uniform_full_v1.py')['sha256']
assert plan['worker_arguments_sha256'] == ready['worker_arguments_sha256'] == identity['worker_arguments_sha256']
receipt = {
    'kind': 'ROOT_actual_uniform_PT02_after_source_race_repair_v1',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'failed_attempt_preserved': rec(old/'loop_status.json'),
    'failed_native_completion': rec(old/'adaptive_20261012_TR_uniform_full_PT_01_native_backend.completion.json'),
    'actual_ROOT_failed_session': 22754, 'actual_ROOT_failed_exit_code': 1,
    'actual_ROOT_failed_exit_tool_chunk': '7081a7',
    'failed_attempt_updates': 0,
    'owned_closure_evidence': 'ROOT run_window finally completed child.wait cleanup before actual controller CLOSED1; native finally completed; independent Get-Process both PIDs ObjectNotFound.',
    'cause': 'Tracing edit changed controller after preflight/dispatch. Actual dispatcher/source integrity guards rejected mixed versions before optimization.',
    'repair': 'Freeze current tracing-only controller d876, new scientific namespace and source receipt, fresh matching preflight/plan/native/identity, fresh admitted index.',
    'actual_ROOT_started_session': 36363, 'actual_ROOT_launch_tool_chunk': 'b0a6fd',
    'run_id': plan['run_id'], 'owned_launcher_pid': 34620, 'actual_worker_pid': 23024,
    'source_receipt': rec(OUT/'TSC_uniform_full_PT02_recovery_ROOT_source_v1.json'),
    'preflight': rec(OUT/'TSC_uniform_full_PT02_lazy_ROOT_preflight_v1.json'),
    'plan': rec(new/'plan.json'), 'native_ready': rec(new/'adaptive_20261012_TR_uniform_full_PT_02_native_backend.json'),
    'source_hashes_consistent': True, 'first_training_update_observed': False,
    'whole_training_peak_fit_claimed': False,
    'prior_streamed_training_admission': rec(ROOT/'work/indexed-TRAIN/adaptive_20261012_TR_uniform_full_PT_01/index_provenance.json'),
    'prior_admission_reused': False,
    'fresh_TR_index': str(ROOT/'work/indexed-TRAIN/adaptive_20261012_TR_uniform_full_PT_02'),
    'PT_initial_errors': 96, 'PT_strict_limit': 66, 'PT_recovery_grace_new_updates': 1000,
    'original_five_guards_active': True, 'all_six_champion_required': True,
    'TEST_used': False, 'goal_complete': False,
}
dest = OUT/'TSC_uniform_full_PT02_actual_ROOT_start_v1.json'
assert not dest.exists()
dest.write_text(json.dumps(receipt, indent=2)+'\n',encoding='utf-8')
plan_path = OUT/'experiment_plan.json'
science = read(plan_path)
item = next(x for x in science['ordered_experiments'] if x['id']=='04')
item.update(status='full_uniform_actual_startup', run_id=plan['run_id'],
    source_receipt='TSC_uniform_full_PT02_recovery_ROOT_source_v1.json',
    actual_GPU_child_started=True, actual_GPU_training_started=False,
    actual_new_uniform_updates=0, actual_full_pass_completed=False,
    actual_ROOT_controller_session=36363, actual_ROOT_launch_tool_chunk='b0a6fd',
    actual_start_receipt=dest.name, source_version_mismatch_repaired=True,
    previous_full_attempt='adaptive_20261012_TR_uniform_full_PT_01',
    previous_full_attempt_actual_CLOSED_exit_code=1, previous_full_attempt_updates=0,
    streamed_training_admission_completed_prior_attempt=True,
    frozen_controller_sha256=plan['controller_sha256'],
    pending=['Fresh streamed admission, source350 and all207-state gates',
             'Actual first optimizer updates and whole-runtime peak memory measurement',
             'Complete253798-clip new uniform pass with fixed DEV250'])
science['objective'] = 'Competitive Turkish Parakeet v3 FLEURS WER with EN/ES/IT/FR/DE/PT retention, final held-out comparison, usable export and confidence evaluation'
science['status_authority'] = 'Actual controller/native closure and complete fixed DEV determine status. Historical trials retain original six-set scope; new full run uses seven sets and explicit Portuguese recovery grace. PT02 sole GPU child is in startup, no optimizer updates yet at this receipt; always read current actual history/completion.'
science['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
plan_path.write_text(json.dumps(science,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'recorded': rec(dest), 'run_id': plan['run_id'], 'actual_optimizer_updates': 0}))
