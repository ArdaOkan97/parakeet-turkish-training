"""Record the root-observed interruption without changing predecessor artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from resource_watchdog_v2 import sample_memory

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP_ID = 'adaptive_20261011_public_data_02'
LOOP = OUT / 'runs' / LOOP_ID
STATUS = LOOP / 'loop_status.json'
SOURCE = ROOT / 'work/adaptive_20261010_continuation_03_resume_B8_best.pt'
RUN_ID = LOOP_ID + '_public_rank8_control'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

status = json.loads(STATUS.read_text(encoding='utf-8'))
memory = sample_memory(pid=5212, phase='root_confirm_predecessor_absent')
source_sha = digest(SOURCE)
assert source_sha == '4790b0a53c0e99ba6ea1a2b8bd9bbdd1d9787ca80b9a38ea6107701c098b98ed'
assert status['current_phase']['optimizer_updates_started'] is False
assert status['current_phase']['latest_published_step'] == 0
candidate_files = [OUT / 'runs' / RUN_ID / name for name in ('run_config.json', 'history.jsonl')]
candidate_files += [ROOT / 'work' / (RUN_ID + suffix) for suffix in ('_latest.pt', '_best.pt')]
assert not any(path.exists() for path in candidate_files)
worker = memory['worker_private_memory']
assert worker.get('available') is False and worker.get('stage') == 'OpenProcess' and worker.get('error') == 87, worker
receipt = {
    'kind': 'root_owned_pre_optimizer_preparation_exit_receipt_v1',
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'predecessor_loop_id': LOOP_ID,
    'root_observed_owned_session_completed': True,
    'root_observed_exit_code': 1,
    'owned_exec_session_id': 95658,
    'actual_worker_pid': 5212,
    'worker_absent_confirmed': True,
    'worker_absence_observation': worker,
    'published_optimizer_steps': 0,
    'termination': 'Root sent Ctrl-C to its owned preparation session; the tool returned EXIT1. No cooperative Python finally or child EXIT0 is claimed.',
    'reason': 'Pause unused Turkish teacher encoding before optimizer work; run the matching throughput pilot and resume with verified frame metadata.',
    'loop_status_path': str(STATUS),
    'loop_status_sha256': digest(STATUS),
    'preserved_predecessor_status': status['status'],
    'preserved_predecessor_updated_at_utc': status['updated_at_utc'],
    'stale_status_is_not_a_live_process_claim': True,
    'source_checkpoint_path': str(SOURCE),
    'source_checkpoint_sha256': source_sha,
    'source_step': 1871,
    'source_checkpoint_unchanged': True,
    'unpublished_training_artifacts': [{'path': str(path), 'exists': False} for path in candidate_files],
    'training_log_path': str(LOOP / (RUN_ID + '_training.log')),
    'training_log_sha256': digest(LOOP / (RUN_ID + '_training.log')),
    'goal_complete': False,
    'test_evaluated': False,
}
target = OUT / 'public_data_02_preparation_exit_receipt_v1.json'
assert not target.exists(), 'Do not overwrite an immutable receipt.'
target.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'path': str(target), 'sha256': digest(target), 'published_optimizer_steps': 0, 'worker_absent_confirmed': True}))
