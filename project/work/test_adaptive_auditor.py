"""CPU-only adaptive-auditor regression; no final adaptive result is fabricated."""
import copy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1];OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_adaptive_results as worker
import run_experiment_loop as controller
protected=['confidence.py','runtime.py','benchmark.py','metrics.py','efficient_tdt.py','efficient_tdt_v2.py','train_lora.py','encoder_lora.py','batched_lora.py','train_lora_batched.py','final_report.txt']
protected_hashes={name:worker.file_sha(OUTPUT/name) for name in protected}

fixture=ROOT/'work/adaptive_auditor_real_pilot_fixture';fixture.mkdir(exist_ok=True)
baseline,baseline_provenance=controller.validate_baseline()
incoming,provenance=controller.validate_selection('lora_20261010_01')
frozen={'run_id':incoming['run_id'],'wer':incoming['turkish_validation_wer'],
        'weights':incoming['weights'],'selected_step':incoming['selected_step'],'trial':None,
        'provenance':provenance,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'selection_split':'FLEURS validation only','competitive_threshold_wer':baseline['metrics']['wer']+.02,
        'baseline_provenance':baseline_provenance}
actual_selection=worker.json_load(OUTPUT/'runs/lora_20261010_01/selection.json')
profile=worker.json_load(OUTPUT/'batching_speed_comparison.json')
state={'loop_id':fixture.name,'status':'synthetic CPU regression fixture',
       'selection_split':'FLEURS validation only','best_development_candidate':{key:frozen[key] for key in ('run_id','wer','weights','selected_step','trial','provenance')},
       'controller_sha256':worker.file_sha(OUTPUT/'run_experiment_loop.py'),'trials':[],
       'final_selection':actual_selection,'merged_turkish_development_wer':actual_selection['merged_development_gate']['turkish_wer'],
       'competitive_on_development':False,'profile_sha256':worker.file_sha(OUTPUT/'batching_speed_comparison.json'),
       'selected_configuration':profile['selected_configuration']}
for name,value in [('frozen_final_selection.json',frozen),('loop_status.json',state)]:
    (fixture/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
args=SimpleNamespace(frozen_selection=fixture/'frozen_final_selection.json',baseline_results=OUTPUT/'results/base_v3',strong_test_comparison=None)
checks=worker.Checks();result=worker.audit(args,checks)
result['passed']=True;report=worker.text_report(result)
tests=[{'name':'completed original pilot900 predictions/export/scoring/calibration reused as CPU integration fixture','passed':True,'checks':len(checks.items)},
       {'name':'high Turkish WER report explicitly rejects useful accuracy claim','passed':'not useful Turkish transcription accuracy' in report},
       {'name':'no missing strong TEST inferred as competitive TEST quality','passed':'competitive TEST quality is unverified' in report}]

# Independent small fixtures exercise valid early-stop and initial-model winners.
base=worker.json_load(OUTPUT/'results/base_v3/summary.json')
small=ROOT/'work/adaptive_history_guard_fixture';(small/'runs/fixture').mkdir(parents=True,exist_ok=True)
worker.OUTPUT=small
def gate(wer):
    return {'turkish_wer':wer,'retention_wer_deltas':{lang:0. for lang in worker.LANGUAGES[:-1]},'retention_gate_passed':True}
initial=gate(.8);(small/'runs/fixture/initial_development.json').write_text(json.dumps(initial))
config={'steps':5,'selection_split':'FLEURS validation only'}
def row(step,wer):
    return {'step':step,'tdt_loss':1.,'replay_kl':.01,'replay_duration_kl':.01,'replay_feature_distillation':.001,'gradient_norm':.3,'validation':gate(wer)}
history=[row(1,.7),row(2,.65)]
selection={'accepted':True,'selection_split':'FLEURS validation only','heldout_tests_evaluated':False,
           'selected_step':2,'completed_steps':2,'configured_steps':5,'turkish_validation_wer':.65}
worker.selected_history('fixture',selection,config,history,base,worker.Checks())
tests.append({'name':'valid selected step2 of completed2/configured5 early stop','passed':True})
zero={**selection,'selected_step':0,'completed_steps':0,'turkish_validation_wer':.8}
worker.selected_history('fixture',zero,config,[],base,worker.Checks())
tests.append({'name':'valid step0 with no completed optimizer updates','passed':True})
worse={**selection,'selected_step':0,'turkish_validation_wer':.8}
worker.selected_history('fixture',worse,config,[row(1,.9),row(2,.85)],base,worker.Checks())
tests.append({'name':'valid step0 retained over worse later DEV checkpoints','passed':True})
boundary=copy.deepcopy(base);boundary['validation_en']['asr']['wer']=.020
edge=gate(.65);edge['retention_wer_deltas']['en']=.025-.020
worker.check_gate(edge,boundary,worker.Checks(),'exact0.005 retention boundary')
tests.append({'name':'direct WER retention accepts exact0.005 boundary despite subtraction roundoff','passed':True})
corruptions=[('history step gap',lambda s,c,h:h[1].__setitem__('step',3)),
             ('completed-count mismatch',lambda s,c,h:s.__setitem__('completed_steps',4)),
             ('configured count exceeded',lambda s,c,h:c.__setitem__('steps',1)),
             ('TEST selection forbidden',lambda s,c,h:s.__setitem__('heldout_tests_evaluated',True)),
             ('wrong best checkpoint',lambda s,c,h:s.update(selected_step=1,turkish_validation_wer=.7)),
             ('wrong DEV score',lambda s,c,h:s.__setitem__('turkish_validation_wer',.6)),
             ('nonfinite gradient',lambda s,c,h:h[0].__setitem__('gradient_norm',float('nan'))),
             ('wrong retention arithmetic',lambda s,c,h:h[1]['validation']['retention_wer_deltas'].__setitem__('en',.1)),
             ('boolean selected step',lambda s,c,h:s.__setitem__('selected_step',True))]
for name,mutate in corruptions:
    s,c,h=copy.deepcopy(selection),copy.deepcopy(config),copy.deepcopy(history);mutate(s,c,h)
    try:worker.selected_history('fixture',s,c,h,base,worker.Checks())
    except Exception as error:tests.append({'name':name,'passed':True,'failure':str(error)})
    else:tests.append({'name':name,'passed':False})
worker.OUTPUT=OUTPUT
tests.append({'name':'all protected code/pilot report bytes unchanged','passed':protected_hashes=={name:worker.file_sha(OUTPUT/name) for name in protected}})
payload={'passed':all(t['passed'] for t in tests),'gpu_used':False,
         'scope':'Real original-pilot900 predictions/FP32 export/calibration CPU integration fixture; synthetic early-stop/step0 guard cases. No final adaptive or strong TEST inference/result is claimed.',
         'auditor_sha256':worker.file_sha(OUTPUT/'verify_adaptive_results.py'),'positive_checks':len(checks.items),
         'protected_sha256':protected_hashes,'tests':tests,
         'real_pilot_fixture_quality':{k:result[k] for k in ('run_id','selected_step','winner_history','competitive_on_development','heldout_old_language_gates_passed')},
         'english_shared_correct_fixture':result['english_confidence']}
target=OUTPUT/'adaptive_auditor_cpu_verification.json';target.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':payload['passed'],'positive_checks':len(checks.items),'tests':len(tests),'output':str(target)},indent=2))
raise SystemExit(0 if payload['passed'] else 1)
