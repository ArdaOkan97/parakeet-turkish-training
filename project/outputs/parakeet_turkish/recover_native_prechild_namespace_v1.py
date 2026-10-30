"""Archive a root-observed pre-child failure without changing training state."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LOOP = OUT / 'runs/adaptive_20261011_native_memory_fix_01'
ARCHIVE = OUT / 'native_memory_fix_01_prechild_failure_v1'
CONTROLLER_SHA = 'c65e9f37eb0db4641552d35ccda52b5604223223deb88d7d9ebfcfe61f0f051d'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def scoped(path):
    path = path.resolve()
    assert path.is_relative_to(ROOT.resolve()), 'Move must remain inside the explicit workspace'
    return path


def main():
    assert sha(OUT / 'run_experiment_loop_continuation_v8.py') == CONTROLLER_SHA
    state_path = LOOP / 'loop_status.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    assert state['status'] == 'failed_review_required' and state['active_run_id'] is None
    assert state['trials'] == [] and state['failure_message'] == 'Stage namespace must be fresh'
    manifest = json.loads((LOOP / 'fork_snapshot/fork_manifest.json').read_text(encoding='utf-8'))
    latest = manifest['fork_checkpoint']['latest']
    best = manifest['fork_checkpoint']['best']
    preseed = scoped(Path(latest['mutable_work_path']))
    immutable = scoped(Path(latest['path']))
    work_best = scoped(Path(best['mutable_work_path']))
    assert sha(preseed) == sha(immutable) == latest['sha256']
    assert sha(work_best) == best['sha256']
    assert not (OUT / 'runs' / manifest['target_run_id']).exists()
    assert not (LOOP / (manifest['target_run_id'] + '_native_backend.json')).exists()
    assert not ARCHIVE.exists()
    records = {'failed_loop_status_sha256': sha(state_path),
               'preseed_latest_sha256': sha(preseed), 'immutable_latest_sha256': sha(immutable),
               'work_best_sha256': sha(work_best)}
    scoped(ARCHIVE).mkdir()
    moves = ((preseed, ARCHIVE / 'preseeded_latest63.pt'),
             (state_path, ARCHIVE / 'loop_status.json'),
             (LOOP / 'trial_inputs', ARCHIVE / 'trial_inputs'))
    for source, destination in moves:
        scoped(source).rename(scoped(destination))
    assert not preseed.exists() and sha(immutable) == latest['sha256']
    assert sha(work_best) == best['sha256']
    receipt = {'kind': 'root_observed_native_prechild_failure_recovery_v1', 'passed': True,
               'root_observed_exec_chunk': 'd0b356', 'root_observed_exit_code': 1,
               'owned_child_launched': False, 'optimizer_updates_in_failed_attempt': 0,
               'source_controller_sha256': CONTROLLER_SHA, 'training_sources_changed': False,
               'immutable_latest63_and_work_best0_preserved': True,
               'moves': [{'from': str(s), 'to': str(d)} for s, d in moves], **records,
               'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (ARCHIVE / 'recovery_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'prechild_failure_archived': True, 'exact_resume_input_unchanged': True}))


if __name__ == '__main__':
    main()
