"""Record the intentional, fully exited B1-to-B8 controller transition."""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/parakeet_turkish'
sys.path.insert(0, str(OUT))
from atomic_io_retry_v1 import write_json


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


prior_id = 'adaptive_20261010_recovery_01'
prior_dir = OUT / 'runs' / prior_id
status_path = prior_dir / 'loop_status.json'
status = load(status_path)
assert status['status'] == 'failed' and not status.get('active_run_id')
assert status['failure_message'] == 'Trial ID/stop marker already exists.'
assert status['awake_restoration']['enabled'] is False
original = load(OUT / 'runs/adaptive_20261010_01/loop_status.json')
assert status['deadline_at_utc'] == original['deadline_at_utc']
source_sha = sha(OUT / 'run_experiment_loop_recovery.py')
assert source_sha == '5712954946ee3f68ebfc7e25075ecec2e718432f2393bf327c51d6c4905e406c'
run_id = prior_id + '_screen_C'
run_dir = OUT / 'runs' / run_id
selection = load(run_dir / 'selection.json')
assert selection['run_id'] == run_id and selection['accepted'] is True
assert selection['stop_early_reason'] == 'controller_stop_file'
entry = [row for row in status['trials'] if row['run_id'] == run_id]
assert len(entry) == 1 and entry[0]['exit_code'] == 0 and entry[0]['status'] == 'complete'
snapshot = prior_dir / 'b8_transition_prior_status.json'
manifest = prior_dir / 'b8_transition.json'
assert not snapshot.exists() and not manifest.exists(), 'Preserve an existing transition record'
shutil.copy2(status_path, snapshot)
completed = {'run_id': run_id, 'selected_step': selection['selected_step'],
             'completed_steps': selection['completed_steps'], 'accepted': selection['accepted']}
files = {'run_config': run_dir / 'run_config.json',
         'training_history': run_dir / 'training_history.json',
         'selection': run_dir / 'selection.json',
         'best_checkpoint': Path(selection['weights']),
         'latest_checkpoint': ROOT / 'work' / (run_id + '_latest.pt')}
for key, path in files.items():
    path = path.resolve()
    assert path.is_relative_to(ROOT.resolve()) and path.is_file()
    completed[key + '_path'] = str(path)
    completed[key + '_sha256'] = sha(path)
guards = []
for suffix in ('screen_C', 'screen_B2', 'extend_1'):
    path = ROOT / 'work' / (prior_id + '_' + suffix + '.stop')
    assert path.is_file()
    guards.append({'path': str(path), 'sha256': sha(path)})
record = {'schema': 1, 'kind': 'user_prioritized_B8_transition',
          'user_authorized_throughput_priority': True, 'intentional_transition': True,
          'prior_controller_exited': True, 'prior_controller_observed_exit_code': 1,
          'transition_reason': 'User prioritized measured batch-8 throughput. C stopped cooperatively and saved final DEV/checkpoint; explicit owned guard markers prevented further B1/B2 GPU children.',
          'created_at_utc': datetime.now(timezone.utc).isoformat(),
          'original_loop_id': 'adaptive_20261010_01',
          'prior_recovery_loop_id': prior_id,
          'target_recovery_loop_id': 'adaptive_20261010_recovery_b8_01',
          'original_deadline_at_utc': original['deadline_at_utc'],
          'prior_controller_sha256': source_sha,
          'prior_status_snapshot_path': str(snapshot),
          'prior_status_snapshot_sha256': sha(snapshot),
          'completed_C': completed, 'owned_guard_markers': guards,
          'target_configuration': {'batch_size': 8, 'accumulation': 2,
                                   'effective_batch_size': 16, 'warmup_steps': 25},
          'selection_policy': 'Actual DEV quality and all five retention gates; strict BF16 numerical equivalence remains failed.',
          'gpu_concurrency_policy': 'Prior GPU child exited before any successor training child; no concurrent GPU experiments during the transition.'}
write_json(manifest, record)
print(json.dumps({'manifest': str(manifest), 'sha256': sha(manifest),
                  'C_completed_steps': selection['completed_steps'],
                  'original_deadline_at_utc': original['deadline_at_utc']}))
