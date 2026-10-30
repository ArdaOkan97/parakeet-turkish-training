"""Record observed process exits and saved artifacts; no model or GPU imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def main():
    source_id = 'adaptive_20261011_native_split_resume_01'
    source = OUT / 'runs' / source_id
    selection = read(source / 'selection.json')
    history_path = source / 'training_history.json'
    history = read(history_path)
    assert selection['completed_steps'] == history[-1]['step'] == 862
    assert selection['stop_early_reason'] == 'controller_stop_file'
    artifacts = {}
    for name in ('weights', 'latest_weights'):
        path = Path(selection[name])
        assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file()
        artifacts[name] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    save(OUT / 'full_nemo_gpu_handoff_receipt_v1.json', {
        'kind': 'root_observed_cooperative_GPU_handoff',
        'source_run_id': source_id, 'controller_session': 14515,
        'observed_controller_exit_code': 0, 'observed_exit_tool_chunk': '93d28b',
        'source_completed_updates': 862, 'saved_checkpoint_artifacts': artifacts,
        'source_training_history_sha256': sha(history_path),
        'reason': 'Root requested a saved handoff to actual native NeMo Lightning training',
        'quality_target_achieved': False, 'source_best_qualifying_TR_DEV_WER': selection['turkish_validation_wer'],
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    failed_id = 'full_nemo_turkish_20261011_pilot_01'
    failed = OUT / 'runs' / failed_id
    log = OUT / f'{failed_id}.log'
    assert 'PermissionError' in log.read_text(encoding='utf-8', errors='replace')
    save(failed / 'trial_outcome.json', {
        'run_id': failed_id, 'status': 'failed', 'actual_completed_optimizer_updates': 0,
        'error': 'NeMo archive extraction PermissionError in sandbox default temporary directory',
        'stop_reason': 'startup_archive_extraction_failed_before_model_initialization',
        'selected_step': None, 'selected_checkpoint': None,
        'observed_session': 97894, 'observed_exit_code': 1, 'observed_exit_tool_chunk': 'ac915f',
        'log_path': str(log), 'log_sha256': sha(log),
        'model_initialized': False, 'Lightning_fit_called': False,
        'TEST_evaluated': False, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({'saved_handoff_step': 862, 'native_startup_failure_recorded': True}))


if __name__ == '__main__':
    main()
