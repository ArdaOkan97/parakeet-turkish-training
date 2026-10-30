"""Root-observed closed03 receipt for a new-data fresh-optimizer warm start."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
LOOP = 'adaptive_20261010_continuation_03'
RUN = LOOP + '_resume_B8'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    status_path = OUT/'runs'/LOOP/'loop_status.json'
    status = read(status_path)
    trial = status['trials'][-1]
    assert status['status'] == 'resource_review_required'
    assert trial['run_id'] == RUN and trial['exit_code'] == 0
    assert trial['durable_completed_updates'] == trial['completed_steps'] == 1871
    directory = OUT/'runs'/RUN
    config_path, history_path, selection_path = [directory/name for name in ('run_config.json','training_history.json','selection.json')]
    config, history, selection = map(read, (config_path,history_path,selection_path))
    assert config['run_id'] == selection['run_id'] == RUN
    assert [row['step'] for row in history] == list(range(1,1872))
    assert all(math.isfinite(row[key]) and row[key]>=0 for row in history for key in ('loss','gradient_norm','train_wall_seconds','audio_seen_seconds'))
    validation = history[-1]['validation']
    assert validation['retention_gate_passed'] and set(validation['retention_wer_deltas']) == {'en','es','it','fr','de'}
    assert all(validation['retention_wer_deltas'][lang] <= validation['retention_wer_limits'][lang] for lang in validation['retention_wer_deltas'])
    assert selection['selected_step'] == selection['completed_steps'] == 1871
    assert selection['turkish_validation_wer'] == validation['turkish_wer'] == 0.6496969696969697
    assert not selection['heldout_tests_evaluated']
    summary_path = OUT/'results'/(RUN+'_step_1871')/'summary.json'
    summary = read(summary_path)
    assert summary['validation_tr']['asr']['wer'] == validation['turkish_wer']
    assert all(summary['validation_'+lang]['asr']['utterances'] == 50 for lang in ('tr','en','es','it','fr','de'))
    for name,digest in config['code_sha256'].items():
        assert sha(OUT/name) == digest
    parent = read(OUT/'continuation_prerequisites_v3/predecessor_exit_receipt.json')
    receipt = {
        'schema':1,'kind':'root_owned_resource_review_exit_receipt_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'root_observed_owned_session_completed':True,'root_owned_exec_session':26491,'root_observed_exit_code':1,
        'root_exit_observation_source':'Root write_stdin(session26491) returned completed exit_code1 in this turn; closed controller records child exit0.',
        'predecessor_loop_id':LOOP,'predecessor_run_id':RUN,'predecessor_status':status['status'],
        'predecessor_status_path':str(status_path),'predecessor_status_sha256':sha(status_path),
        'predecessor_controller_path':str(OUT/'run_experiment_loop_continuation_v3.py'),
        'predecessor_controller_sha256':sha(OUT/'run_experiment_loop_continuation_v3.py'),
        'predecessor_finished_at_utc':status['finished_at_utc'],'predecessor_deadline_at_utc':status['deadline_at_utc'],
        'child_exit_code':0,'child_durable_step':1871,'child_selected_step':1871,
        'predecessor_audit_available':False,'predecessor_training_checkpoint_review_passed':True,
        'review_scope':'Closed durable finite history, exact DEV source counts, selection, frozen source hashes, root-owned exit; fresh optimizer warmstart only, no exact-resume state claim.',
        'latest_dev_wer':validation['turkish_wer'],'latest_dev_retention_gate_passed':True,
        'resource_stop_reason':status['resource_stop_reason'],
        'review_conclusion':'Legacy Win32 process-available telemetry reached96.24%; historical authoritative system GetPerformanceInfo absent. Current APIs agree. Future accurate global/self guards preserve95% and2GB thresholds. Expanded data uses a new optimizer/sampler.',
        'gpu_relaunch_authorization':'User-authorized local GPU adaptive experiments until competitive Turkish DEV WER; root-only GPU ownership.',
        'source_config_identity':config['config_identity'],'runtime_provenance':{},
        'fresh_TEST_evaluated':False,'fresh_export_published':False,
        'prior_audited_receipt_path':parent['prior_audited_receipt_path'],
        'prior_audited_receipt_sha256':parent['prior_audited_receipt_sha256'],
        'immutable_prior_candidate':parent['immutable_prior_candidate'],'prior_hf_files_sha256':parent['prior_hf_files_sha256'],
    }
    for key,path in (('child_run_config',config_path),('child_training_history',history_path),('child_selection',selection_path),
                     ('source_latest',ROOT/'work'/(RUN+'_latest.pt')),('source_best',ROOT/'work'/(RUN+'_best.pt'))):
        receipt[key+'_path'],receipt[key+'_sha256'] = str(path),sha(path)
    for key in ('cache_provenance_path','io_provenance_path','memory_guard_provenance_path','worker_identity_path'):
        path=Path(trial[key]); receipt['runtime_provenance'][key]={'path':str(path),'sha256':sha(path)}
    output=OUT/'continuation_prerequisites_v4/predecessor_exit_receipt.json'
    assert not output.exists()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'receipt':str(output),'sha256':sha(output),'step':1871,'CUDA_used':False}))


if __name__ == '__main__':
    main()
