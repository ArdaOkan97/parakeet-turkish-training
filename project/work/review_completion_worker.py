"""CPU-only guards and report wording regression for the finite worker."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parakeet_turkish'
source=OUTPUT/'finish_experiment.py'
spec=importlib.util.spec_from_file_location('finish_experiment_review',source)
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
checks=[]

def passes(name,condition):
    assert condition,name
    checks.append(name)

def rejects(name,function,kind=RuntimeError):
    try:function()
    except kind:checks.append(name)
    else:raise AssertionError(name)

with tempfile.TemporaryDirectory(dir=ROOT/'work',prefix='completion_review_') as directory:
    root=Path(directory);run=root/'runs/fixture';run.mkdir(parents=True)
    configuration={'run_id':'fixture','steps':2,'turkish_clips_used':20}
    worker.write_json(run/'run_config.json',configuration)
    history=[dict(step=step,**{field:1. for field in worker.LOSS_FIELDS}) for step in (1,2)]
    passes('Every configured contiguous finite optimizer step accepted',worker.validate_history(history,2)==2)
    rejects('Skipped optimizer step rejected',lambda:worker.validate_history([history[1]],2))
    rejects('Extra optimizer step rejected',lambda:worker.validate_history(history,1))
    malformed=[dict(history[0],tdt_loss=float('nan'))]
    rejects('Nonfinite training loss rejected',lambda:worker.validate_history(malformed,2))
    weights=root/'selected.pt';weights.write_bytes(b'fixture only')
    history[0]['validation']={'retention_gate_passed':True,'turkish_wer':.8}
    accepted={'run_id':'fixture','accepted':True,'selected_step':1,'turkish_validation_wer':.8,'weights':str(weights)}
    worker.validate_selection(accepted,'fixture',history,2)
    checks.append('Accepted selection matches recorded development gate and weights')
    rejects('Mismatch between selected and history development score rejected',lambda:worker.validate_selection(dict(accepted,turkish_validation_wer=.7),'fixture',history,2))
    rejects('Run identity mismatch rejected',lambda:worker.validate_selection(accepted,'different',history,2))
    worker.write_json(run/'training_history.json',history)
    worker.write_json(run/'selection.json',accepted)
    config_sha=worker.sha256_file(run/'run_config.json')
    got=worker.wait_for_training(run,'fixture',configuration,config_sha,time.monotonic()+1,1)
    passes('Wait releases only from completed history plus valid final selection',got==(history,accepted))
    rejects('Configuration mutation rejected before finalization',lambda:worker.wait_for_training(run,'fixture',configuration,'wrong',time.monotonic()+1,1))
    worker.write_json(run/'training_history.json',history[:1])
    rejects('Partial history cannot release an existing selection',lambda:worker.wait_for_training(run,'fixture',configuration,config_sha,time.monotonic()-1,1),TimeoutError)
    worker.OUTPUT=root
    adapted=root/'adapted_summary.json'
    baseline={};candidate={};comparison={};calibration={}
    for language in worker.LANGUAGES:
        baseline['test_'+language]={'asr':{'utterances':100,'wer':.9 if language=='tr' else .05,'cer':.5 if language=='tr' else .025}}
        candidate['test_'+language]={'asr':{'utterances':100,'wer':.8 if language=='tr' else .05,'cer':.4 if language=='tr' else .025}}
        comparison[language]={'wer':{'ci_percentage_points':[-11.,-9.] if language=='tr' else [-1.,1.]}}
        calibration[language]={'temperature':1.}
    worker.write_json(root/'results/base_v3/summary.json',baseline)
    worker.write_json(adapted,candidate)
    worker.write_json(run/'adapted_results_path.json',{'tag':'fixture_adapted','summary':str(adapted)})
    worker.write_json(run/'benchmark_comparison.json',comparison)
    worker.write_json(run/'confidence_calibration.json',calibration)
    selection=dict(accepted,heldout_tests_evaluated=True,checkpoint=str(root/'checkpoint'))
    report=worker.build_report('fixture',configuration,history,selection,'verification_pending')
    passes('Relative Turkish improvement can pass the explicit pilot gates',report['pilot_improvement_and_retention_gates_passed'] is True)
    passes('80 percent WER remains explicitly highly inaccurate research ASR','highly inaccurate' in report['turkish_accuracy_assessment'] and '80.000%' in report['turkish_accuracy_assessment'])
    passes('Useful Turkish accuracy is never implied by pilot acceptance',any('does not guarantee' in x and 'useful Turkish accuracy' in x for x in report['limitations']))

record={'passed':True,'gpu_used':False,'worker_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'auditor_sha256':hashlib.sha256((OUTPUT/'verify_confidence_robustness.py').read_bytes()).hexdigest(),
    'checks':checks,'read_only_review':[
        'Probe CLI model aliases and external-reference directories match the reusable auditor.',
        'Only proven canonical path/config differences may be normalized; published-file hashes checked.',
        'Child failures/timeouts retain logs and are not reported as success.',
        'Final completed report receives a second CPU audit before backlog completion; report is not rewritten afterward.'
    ],'limitations':[
        'Tests use synthetic stdlib fixtures and do not run training, model inference, finalization or Backlog mutation.',
        'Actual selected checkpoint canonical-fingerprint comparison remains to be checked after training.',
        'Worker waits for its finite deadline if training stops without publishing a selection.'
    ]}
target=OUTPUT/'completion_worker_independent_review.json'
target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'checks':len(checks),'output':str(target)},indent=2))
