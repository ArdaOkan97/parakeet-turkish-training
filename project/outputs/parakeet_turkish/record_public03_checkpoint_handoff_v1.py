"""Root receipt for the cooperative checkpoint handoff to the NeMo loss arm."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from resource_watchdog_v2 import sample_memory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP_ID = 'adaptive_20261011_public_data_03'
LOOP = OUT / 'runs' / LOOP_ID
RUN_ID = LOOP_ID + '_public_rank8_control'
RUN = OUT / 'runs' / RUN_ID

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def record(path):
    return {'path': str(path), 'sha256': digest(path)}

state = json.loads((LOOP / 'loop_status.json').read_text(encoding='utf-8'))
assert state['status'] == 'training_window_complete_objective_incomplete' and state['active_run_id'] is None
assert state['resource_review_required'] is False and len(state['trials']) == 1
trial = state['trials'][0]
assert trial['exit_code'] == 0 and trial['completed_steps'] == 18 and trial['selected_step'] == 0
assert trial['stop_early_reason'] == 'controller_stop_file' and trial['accepted'] is True
selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))
assert selection['completed_steps'] == 18 and selection['selected_step'] == 0 and selection['accepted']
assert selection['heldout_tests_evaluated'] is False
assert abs(selection['turkish_validation_wer'] - 0.6496969696969697) < 1e-12
identity = json.loads(Path(trial['worker_identity_path']).read_text(encoding='utf-8'))
worker_pid = identity.get('pid', identity.get('worker_pid'))
assert worker_pid is not None, identity
memory = sample_memory(pid=worker_pid, phase='root_confirm_cooperative_handoff_worker_absent')
worker = memory['worker_private_memory']
assert worker.get('available') is False and worker.get('stage') == 'OpenProcess' and worker.get('error') == 87, worker
original = ROOT / 'work/adaptive_20261010_continuation_03_resume_B8_best.pt'
assert digest(original) == '4790b0a53c0e99ba6ea1a2b8bd9bbdd1d9787ca80b9a38ea6107701c098b98ed'
receipt = {
    'kind': 'root_owned_cooperative_native_loss_handoff_receipt_v1',
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'predecessor_loop_id': LOOP_ID,
    'predecessor_run_id': RUN_ID,
    'owned_exec_session_id': 40682,
    'root_observed_owned_session_completed': True,
    'root_observed_exit_code': 0,
    'actual_child_exit_code': trial['exit_code'],
    'actual_worker_pid': worker_pid,
    'worker_absent_confirmed': True,
    'worker_absence_observation': worker,
    'completed_optimizer_updates': 18,
    'selected_step': 0,
    'selected_turkish_DEV_wer': selection['turkish_validation_wer'],
    'selected_checkpoint_is_original_warm_start_quality': True,
    'reason': 'User prefers maintained NeMo code; root requested a cooperative saved-checkpoint handoff to its verified official TDT loss. Initial status observation was step0;18 updates completed before the stop was consumed.',
    'loop_status': record(LOOP / 'loop_status.json'),
    'controller': record(OUT / 'run_experiment_loop_continuation_v6.py'),
    'worker_identity': record(Path(trial['worker_identity_path'])),
    'selection': record(RUN / 'selection.json'),
    'run_config': record(RUN / 'run_config.json'),
    'history': record(RUN / 'training_history.json'),
    'selected_checkpoint': record(Path(selection['weights'])),
    'durable_latest_checkpoint': record(ROOT / 'work' / (RUN_ID + '_latest.pt')),
    'original_source_checkpoint': record(original),
    'original_source_step': 1871,
    'goal_complete': False,
    'test_evaluated': False,
}
target = OUT / 'public_data_03_native_handoff_receipt_v1.json'
assert not target.exists(), 'Do not overwrite an immutable receipt.'
target.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'path': str(target), 'sha256': digest(target), 'completed_optimizer_updates': 18,
                  'selected_step': 0, 'root_exit_code': 0, 'actual_worker_pid': worker_pid}))
