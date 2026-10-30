"""Record root-observed predecessor exit and preserve shared dashboard pointers."""
from pathlib import Path
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs/parakeet_turkish'
LOOP = OUT / 'runs/adaptive_20261010_recovery_b8_01'
DEST = OUT / 'continuation_prerequisites_v1'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()

def main():
    if DEST.exists():
        raise FileExistsError('Immutable receipt already exists')
    state = json.loads((LOOP/'loop_status.json').read_text(encoding='utf-8'))
    assert state['status'] == 'complete'
    assert state['finalization_exit_code'] == state['strong_test_exit_code'] == state['final_audit_exit_code'] == 0
    assert state['recovery']['controller_sha256'] == 'c16d0c61e0c4aa48d4bb93ef5ef8b8192032581d264a3be8fee369f4077a1213'
    assert sha(OUT/'run_experiment_loop_recovery_v2.py') == state['recovery']['controller_sha256']
    audit_path = LOOP/'adaptive_results_verification.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert audit['passed'] is True
    DEST.mkdir()
    preserved = {}
    for name in ['run_config.json', 'training_history.json', 'selection.json',
                 'vocabulary_audit.json', 'benchmark_comparison.json',
                 'confidence_calibration.json', 'adapted_results_path.json']:
        source = OUT/name
        if source.is_file():
            target = DEST/name
            shutil.copy2(source, target)
            assert sha(source) == sha(target)
            preserved[name] = {'path': str(target), 'sha256': sha(target)}
    chosen = state['final_selection']
    candidate = Path(chosen['candidate'])
    provenance = candidate/'finalization_provenance.json'
    assert provenance.is_file()
    hf_manifest = {str(path.relative_to(candidate)): sha(path)
                   for path in sorted(candidate.rglob('*')) if path.is_file()}
    checkpoint = OUT/'checkpoint'
    checkpoint_manifest = {str(path.relative_to(checkpoint)): sha(path)
                           for path in sorted(checkpoint.rglob('*')) if path.is_file()}
    assert hf_manifest == checkpoint_manifest
    receipt = {
        'schema': 1, 'kind': 'root_owned_predecessor_exit_receipt_v1',
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'owned_exec_session_id': 13816, 'root_observed_exit_code': 0,
        'root_observed_owned_session_completed': True,
        'predecessor_loop_id': state['loop_id'], 'predecessor_status': state['status'],
        'predecessor_status_path': str(LOOP/'loop_status.json'),
        'predecessor_status_sha256': sha(LOOP/'loop_status.json'),
        'predecessor_audit_path': str(audit_path), 'predecessor_audit_sha256': sha(audit_path),
        'predecessor_audit_passed': True,
        'predecessor_deadline_unchanged': state['deadline_at_utc'],
        'immutable_prior_candidate': str(candidate), 'prior_hf_files_sha256': hf_manifest,
        'preserved_shared_pointers': preserved,
        'continuation_authorization': 'User explicitly requested continued training and the previously discussed experiment loop after verification, plus diagnosing and preventing the slowdown.',
        'next_window_policy': 'A new bounded three-hour loop; the predecessor deadline and artifacts remain unchanged.'
    }
    target = DEST/'predecessor_exit_receipt.json'
    target.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'receipt': str(target), 'sha256': sha(target),
                      'prior_export_files': len(hf_manifest), 'audit_checks': audit.get('passed_check_count')}))

if __name__ == '__main__':
    main()
