"""Read-only closed-run review and root-observed exit receipt for run02.

No CUDA, checkpoint deserialization, TEST evaluation, or source mutation.
The CPU fork helper independently checks the full serialized states.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LOOP = 'adaptive_20261010_continuation_02'
RUN = LOOP + '_resume_B8'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    status_path = OUT / 'runs' / LOOP / 'loop_status.json'
    s = read(status_path)
    assert s['status'] == 'resource_review_required'
    assert s['resource_stop_reason'] == 'system_commit_at_or_above95percent'
    t = s['trials'][-1]
    assert t['run_id'] == RUN and t['exit_code'] == 0
    assert t['durable_completed_updates'] == t['completed_steps'] == 1821
    rd = OUT / 'runs' / RUN
    cfg_path, hist_path, selection_path = [rd / n for n in ('run_config.json', 'training_history.json', 'selection.json')]
    cfg, history, selection = map(read, (cfg_path, hist_path, selection_path))
    assert cfg['run_id'] == selection['run_id'] == RUN
    assert len(history) == 1821 and [r['step'] for r in history] == list(range(1, 1822))
    for row in history:
        for k in ('loss', 'gradient_norm', 'train_wall_seconds', 'audio_seen_seconds'):
            assert math.isfinite(row[k]) and row[k] >= 0
    v = history[-1]['validation']
    assert v['retention_gate_passed']
    assert set(v['retention_wer_deltas']) == {'en', 'es', 'it', 'fr', 'de'}
    assert all(v['retention_wer_deltas'][k] <= v['retention_wer_limits'][k] for k in v['retention_wer_deltas'])
    assert selection['selected_step'] == selection['completed_steps'] == 1821
    assert selection['turkish_validation_wer'] == v['turkish_wer']
    assert not selection['heldout_tests_evaluated']
    summary_path = OUT / 'results' / (RUN + '_step_1821') / 'summary.json'
    summary = read(summary_path)
    assert summary['validation_tr']['asr']['wer'] == v['turkish_wer']
    assert all(summary['validation_' + lang]['asr']['utterances'] == 50 for lang in ('tr', 'en', 'es', 'it', 'fr', 'de'))
    latest, best = [ROOT / 'work' / (RUN + '_' + n + '.pt') for n in ('latest', 'best')]
    parent_path = OUT / 'continuation_prerequisites_v2' / 'predecessor_exit_receipt.json'
    parent = read(parent_path)
    obs = t['watchdog_observations']
    peak = max(range(len(obs)), key=lambda i: obs[i]['system_memory']['commit_fraction'])
    review = {
        'schema': 1, 'kind': 'root_closed_continuation_resource_review_v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_status_path': str(status_path),
        'source_status_sha256': sha(status_path), 'source_loop_id': LOOP, 'source_run_id': RUN,
        'critical_guard_correctly_executed': True, 'trigger': obs[peak],
        'following_observation': obs[peak + 1],
        'allocator_cap_bytes': t['cache_evidence']['allocator_cap_bytes'],
        'cache_evidence': t['cache_evidence'], 'actual_allocation_owner_resolved': False,
        'telemetry_limit': 'Popen PID is the Windows venv redirector; its private bytes do not represent the Torch worker.',
        'next_runtime_hypothesis': 'Bound transient allocator reserve with a new9GB cap and4GB post-update cleanup. Validate on actual GPU before adaptive restart.',
        'critical_policy_retained': 'Immediate95% system commit or<2GB available commit; sustained90% three fresh polls.',
        'training_math_change_authorized': False, 'quality_gates_waived': False,
        'durable_step': 1821, 'TR_DEV_wer': v['turkish_wer'], 'all_five_DEV_retention_gates_passed': True,
        'fresh_TEST_evaluated': False, 'fresh_export_published': False,
    }
    review_path = OUT / 'continuation_resource_review_v2.json'
    receipt_path = OUT / 'continuation_prerequisites_v3' / 'predecessor_exit_receipt.json'
    for p in (review_path, receipt_path):
        if p.exists():
            raise FileExistsError(p)
    review_path.write_text(json.dumps(review, indent=2) + '\n', encoding='utf8')
    receipt = {
        'schema': 1, 'kind': 'root_owned_resource_review_exit_receipt_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'root_observed_owned_session_completed': True, 'root_owned_exec_session': 3634,
        'root_observed_exit_code': 1,
        'root_exit_observation_source': 'Root write_stdin(session3634) returned completed exit_code1 on2026-10-11T05:47UTC; worker exit0 recorded in closed controller report.',
        'predecessor_loop_id': LOOP, 'predecessor_run_id': RUN,
        'predecessor_status': s['status'], 'predecessor_status_path': str(status_path),
        'predecessor_status_sha256': sha(status_path),
        'predecessor_controller_path': str(OUT / 'run_experiment_loop_continuation_v2.py'),
        'predecessor_controller_sha256': sha(OUT / 'run_experiment_loop_continuation_v2.py'),
        'predecessor_deadline_at_utc': s['deadline_at_utc'], 'predecessor_finished_at_utc': s['finished_at_utc'],
        'child_exit_code': 0, 'child_durable_step': 1821, 'child_selected_step': 1821,
        'predecessor_audit_available': False, 'predecessor_training_checkpoint_review_passed': True,
        'review_scope': 'Actual closed durable history, DEV selection, finite counters, runtime hashes; full serialized state identity remains independently verified by CPU fork helper.',
        'latest_dev_wer': v['turkish_wer'], 'latest_dev_retention_gate_passed': True,
        'resource_stop_reason': s['resource_stop_reason'], 'resource_review_path': str(review_path),
        'resource_review_sha256': sha(review_path), 'review_conclusion': review['next_runtime_hypothesis'],
        'gpu_relaunch_authorization': 'User asks repair/resume adaptive experiments until competitive Turkish WER. Root-only actual GPU probe before changed runtime restart.',
        'prior_audited_receipt_path': parent['prior_audited_receipt_path'],
        'prior_audited_receipt_sha256': parent['prior_audited_receipt_sha256'],
        'immutable_prior_candidate': parent['immutable_prior_candidate'],
        'prior_hf_files_sha256': parent['prior_hf_files_sha256'],
        'source_config_identity': cfg['config_identity'],
        'runtime_provenance': {},
    }
    for key, p in (('child_run_config', cfg_path), ('child_training_history', hist_path), ('child_selection', selection_path), ('source_latest', latest), ('source_best', best)):
        receipt[key + '_path'], receipt[key + '_sha256'] = str(p), sha(p)
    for key in ('cache_provenance_path', 'io_provenance_path', 'memory_guard_provenance_path'):
        p = Path(t[key])
        receipt['runtime_provenance'][key] = {'path': str(p), 'sha256': sha(p)}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'passed': True, 'receipt_path': str(receipt_path), 'receipt_sha256': sha(receipt_path), 'durable_step': 1821, 'TR_DEV_wer': v['turkish_wer'], 'GPU_used': False}))


if __name__ == '__main__':
    main()
