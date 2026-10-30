"""CPU-only completion worker guards; no inference or Backlog mutations."""
import copy
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT/'outputs/parakeet_turkish/finish_experiment.py'
spec = importlib.util.spec_from_file_location('finish_experiment', source)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
scratch = ROOT/'work/research'/f'completion_guards_{uuid.uuid4().hex[:10]}'
scratch.mkdir()
worker.OUTPUT = scratch
run_id = 'cpu_guard_run'
run_dir = scratch/'runs'/run_id
run_dir.mkdir(parents=True)
config = {'run_id':run_id,'steps':2,'turkish_clips_used':2166}
worker.write_json(run_dir/'run_config.json',config)
history = [{**{key:1.0 for key in worker.LOSS_FIELDS},'step':step,
    'validation':{'retention_gate_passed':True,'turkish_wer':.7},'turkish_unique_clips_seen':2*step} for step in (1,2)]
worker.write_json(run_dir/'training_history.json',history)
weights = scratch/'fake_selected.pt'; weights.write_bytes(b'guard fixture; not a model')
selection = {'run_id':run_id,'accepted':True,'selected_step':2,
    'turkish_validation_wer':.7,'weights':str(weights),'heldout_tests_evaluated':False}
worker.write_json(run_dir/'selection.json',selection)
config_sha = worker.sha256_file(run_dir/'run_config.json')
actual_history, actual_selection = worker.wait_for_training(run_dir,run_id,config,config_sha,time.monotonic()+1,1)
assert actual_history==history and actual_selection==selection
assert worker.validate_history(history,2)==2

def rejects(call, exception=RuntimeError):
    try:
        call()
    except exception:
        return True
    raise AssertionError('Invalid completion evidence accepted')

bad = copy.deepcopy(history);bad[1]['step']=1
assert rejects(lambda:worker.validate_history(bad,2))
bad = copy.deepcopy(history);bad[1]['tdt_loss']=float('nan')
assert rejects(lambda:worker.validate_history(bad,2))
bad_selection = {**selection,'run_id':'other'}
assert rejects(lambda:worker.validate_selection(bad_selection,run_id,history,2))
bad_selection = {**selection,'turkish_validation_wer':.8}
assert rejects(lambda:worker.validate_selection(bad_selection,run_id,history,2))
assert rejects(lambda:worker.wait_for_training(run_dir,run_id,config,'wrong',time.monotonic()+1,1))
worker.write_json(run_dir/'training_history.json',history[:1])
assert rejects(lambda:worker.wait_for_training(run_dir,run_id,config,config_sha,time.monotonic()+.03,1),TimeoutError)
worker.write_json(run_dir/'training_history.json',history)

# Rejected merged exports preserve the original trainer selection; restarting
# the finite worker reconstructs that accepted training selection correctly.
worker.write_json(run_dir/'training_selection.json',selection)
worker.write_json(run_dir/'selection.json',{**selection,'accepted':False,'merged':True})
_,original = worker.wait_for_training(run_dir,run_id,config,config_sha,time.monotonic()+1,1)
assert original==selection

baseline={};adapted={};comparison={};calibration={}
for language in worker.LANGUAGES:
    before={'wer':1.0 if language=='tr' else .05,'cer':.5 if language=='tr' else .02,'utterances':100}
    after={'wer':.9 if language=='tr' else .051,'cer':.4 if language=='tr' else .021,'utterances':100}
    baseline[f'test_{language}']={'asr':before};adapted[f'test_{language}']={'asr':after}
    comparison[language]={'wer':{'ci_percentage_points':[-11,-9] if language=='tr' else [-.2,.4]}}
    calibration[language]={'temperature':1.0}
worker.write_json(scratch/'results/base_v3/summary.json',baseline)
summary=scratch/'results/selected/summary.json'
worker.write_json(summary,adapted)
worker.write_json(run_dir/'adapted_results_path.json',{'tag':'selected','summary':str(summary)})
worker.write_json(run_dir/'benchmark_comparison.json',comparison)
worker.write_json(run_dir/'confidence_calibration.json',calibration)
final_selection={**selection,'heldout_tests_evaluated':True,'checkpoint':str(scratch/'checkpoint')}
report=worker.build_report(run_id,config,history,final_selection,'verification_pending')
assert report['pilot_improvement_and_retention_gates_passed']
assert 'highly inaccurate' in report['turkish_accuracy_assessment']
assert 'quality_goal_met_on_pilot_test' not in report
worker.save_report(report)
assert 'highly inaccurate' in (run_dir/'final_report.txt').read_text(encoding='utf-8')
adapted['test_it']['asr']['wer']=.1
worker.write_json(summary,adapted)
failed_gate=worker.build_report(run_id,config,history,final_selection,'verification_pending')
assert not failed_gate['pilot_improvement_and_retention_gates_passed']
assert not failed_gate['per_language_test']['it']['heldout_retention_gate_passed']

# Exercise only a trivial Python child, proving nonzero exit logs are retained.
assert rejects(lambda:worker.run_child(run_dir,'cpu_child_failure',
    [sys.executable,'-c','raise SystemExit(7)'],time.monotonic()+2))
assert (run_dir/'completion_logs/cpu_child_failure.stderr.log').exists()
result={'test_device':'CPU only','gpu_execution':False,'backlog_mutation':False,
    'actual_completed_training_ready_guard':True,'invalid_history_selection_config_rejected':True,
    'incomplete_training_wait_bounded':True,'preserved_training_selection_restart':True,
    'measured_report_retention_gate_pass_and_fail':True,'poor_turkish_accuracy_explicit':True,
    'overall_quality_goal_label_absent':True,'child_failure_logs_retained':True}
(ROOT/'outputs/parakeet_turkish/completion_worker_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
