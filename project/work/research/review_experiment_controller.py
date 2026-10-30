"""CPU-only controller fixtures; every child is a mock, never GPU or TEST.

Copies DEV-only evidence into isolated fixture outputs. All subprocess launch,
CLI, finalization and clock operations are mocked. Captures code SHA so failed
guards are findings about that exact controller version, not a launch approval.
"""
import argparse
import ast
import hashlib
import json
import shutil
import sys
import types
import uuid
from pathlib import Path

ROOT=Path.cwd();OUTPUT=ROOT/'outputs/parakeet_turkish'
source_path=OUTPUT/'run_experiment_loop.py'
source=source_path.read_text(encoding='utf-8')
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
fixture_root=ROOT/'work/research/controller_guard_fixtures'/source_sha[:12]/uuid.uuid4().hex[:8]
fixture_root.mkdir(parents=True,exist_ok=True)
snapshot=fixture_root/'reviewed_controller.py';snapshot.write_text(source,encoding='utf-8')
trainer_source=(OUTPUT/'train_lora_batched.py').read_text(encoding='utf-8')
parser_node=next(node for node in ast.parse(trainer_source).body if isinstance(node,ast.FunctionDef) and node.name=='parser')
parser_ns={'argparse':argparse,'Path':Path,'DATA':ROOT/'work/data','__doc__':'fixture parser'}
exec(compile(ast.Module(body=[parser_node],type_ignores=[]),'<trainer parser>','exec'),parser_ns)
results=[]

def jsonwrite(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')

class Clock:
 def __init__(self):self.value=0.
 def monotonic(self):return self.value
 def sleep(self,seconds):
  self.value+=seconds
  if self.value>12000:raise RuntimeError('Fixture hard guard: mock exceeded12000 seconds')

def exercise(name,*,profile_pass=True,profile_improved=True,wrong_baseline=False,mode='no_trials',hours=.01,checkpointing=True):
 loop_id='cpu_guard_'+name+'_'+uuid.uuid4().hex[:6]
 area=fixture_root/name;fake_out=area/'outputs';fake_out.mkdir(parents=True,exist_ok=True)
 profile={'passed':profile_pass,'throughput_improved':profile_improved,'selected_configuration':{'batch_size':2,'accumulation':2,'tdt_implementation':'v2','chunk_size':16,'encoder_checkpointing':checkpointing}}
 jsonwrite(fake_out/'batching_speed_comparison.json',profile)
 baseline_path=fake_out/'results/strong_whisper_large_v3_greedy_dev_verified';baseline_path.mkdir(parents=True,exist_ok=True)
 for filename in ('summary.json','validation_tr.jsonl','status.json'):
  shutil.copy2(OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified'/filename,baseline_path/filename)
 shutil.copy2(OUTPUT/'strong_baseline_dev_verification.json',fake_out/'strong_baseline_dev_verification.json')
 shutil.copy2(OUTPUT/'metrics.py',fake_out/'metrics.py')
 for file in OUTPUT.glob('*.py'):
  shutil.copy2(file,fake_out/file.name)
 if not (fake_out/'benchmark_strong_tr_test.py').is_file():
  (fake_out/'benchmark_strong_tr_test.py').write_text('# CPU fixture subprocess is mocked\n')
 if not (fake_out/'verify_adaptive_results.py').is_file():
  (fake_out/'verify_adaptive_results.py').write_text('# CPU fixture subprocess is mocked\n')
 original_baseline=fake_out/'results/base_v3';original_baseline.mkdir(parents=True,exist_ok=True)
 shutil.copy2(OUTPUT/'results/base_v3/validation_tr.jsonl',original_baseline/'validation_tr.jsonl')
 shutil.copy2(OUTPUT/'expanded_turkish_preparation.json',fake_out/'expanded_turkish_preparation.json')
 shutil.copy2(OUTPUT/'expanded_turkish_verification.json',fake_out/'expanded_turkish_verification.json')
 if wrong_baseline:
  value=json.loads((baseline_path/'summary.json').read_text());value['split']='test'
  jsonwrite(baseline_path/'summary.json',value)
 incoming=fake_out/'runs/lora_20261010_01';incoming.mkdir(parents=True,exist_ok=True)
 for filename in ('training_selection.json','selection.json','run_config.json','training_history.json','vocabulary_audit.json'):
  p=OUTPUT/'runs/lora_20261010_01'/filename
  if p.is_file():shutil.copy2(p,incoming/filename)
 m=types.ModuleType('mock_controller_'+name);m.__file__=str(snapshot)
 exec(compile(source,str(snapshot),'exec'),m.__dict__)
 m.OUTPUT=fake_out;m.ROOT=ROOT
 clock=Clock();m.time=clock
 m.keep_awake=lambda enable:{'supported':True,'enabled':enable,'mock':True}
 commands=[];processes=[];notes=[];finalizer_commands=[]
 class Child:
  def __init__(self,command,**kwargs):
   self.command=command;self.pid=987;self.returncode=None;self.birth=clock.value;self.finished_at=None;self.terminated=False
   commands.append(command);processes.append(self)
   rid=command[command.index('--run-id')+1];self.rid=rid
   self.stop=Path(command[command.index('--stop-file')+1])
   self.run=fake_out/'runs'/rid;self.run.mkdir(parents=True,exist_ok=True)
   history=[{'step':1,'turkish_epoch':.25,'validation':{'turkish_wer':.80,'retention_gate_passed':True}}]
   if mode=='history_error':history=[{}]
   jsonwrite(self.run/'training_history.json',history)
   weights=area/(rid+'_weights.pt');weights.write_bytes(b'fixture-only-no-model')
   accepted=mode in ('wrong_run_selection','accepted_new','finalizer_timeout','wrong_final_identity','strong_test_timeout','merged_target_missed','final_audit_failed','final_audit_timeout')
   selection={'accepted':accepted,'run_id':'unrelated_run' if mode=='wrong_run_selection' else rid,'selected_step':1,'turkish_validation_wer':.01 if mode=='wrong_run_selection' else .80,'weights':str(weights),'selection_split':'FLEURS validation only','heldout_tests_evaluated':False,'reason':'CPU fixture selection','completed_steps':1,'turkish_epochs_completed':.25}
   if accepted and mode!='wrong_run_selection':
    import torch
    config=json.loads((incoming/'run_config.json').read_text());config.update(run_id=rid,steps=1)
    weights=ROOT/'work'/('fixture_'+rid+'.pt')
    torch.save({'run_id':rid,'step':1,'lora_metadata':config['lora_layout']},weights)
    selection['weights']=str(weights);jsonwrite(self.run/'run_config.json',config)
   jsonwrite(self.run/'selection.json',selection)
  def poll(self):
   if self.returncode is not None:return self.returncode
   if mode=='stage_overrun':
    if clock.value-self.birth>=1200:self.returncode=0
   elif mode=='global_deadline':
    if self.stop.is_file():self.returncode=0
   elif mode=='history_error':pass
   elif mode=='backlog_failure':
    if clock.value-self.birth>=120:self.returncode=0
   else:self.returncode=0
   if self.returncode is not None:self.finished_at=clock.value
   return self.returncode
  def terminate(self):self.terminated=True;self.returncode=-15;self.finished_at=clock.value
  def kill(self):self.terminated=True;self.returncode=-9;self.finished_at=clock.value
  def wait(self,timeout=None):return self.returncode
 class Subprocess:
  CREATE_NO_WINDOW=0;DEVNULL=-3;PIPE=-1;STDOUT=-2
  TimeoutExpired=__import__('subprocess').TimeoutExpired
  Popen=Child
  def run(self,command,**kwargs):
   if any(str(x).endswith('finalize_lora.py') for x in command):
    finalizer_commands.append({'command':command,'timeout':kwargs.get('timeout')})
    if mode=='finalizer_timeout':
     clock.value+=kwargs['timeout'];raise self.TimeoutExpired(command,kwargs['timeout'])
    clock.value+=60
    rid=command[command.index('--run-id')+1];p=fake_out/'runs'/rid/'selection.json';s=json.loads(p.read_text());s.update(heldout_tests_evaluated=True,merged=True,merged_development_gate={'passed':True,'turkish_wer':.12 if mode=='merged_target_missed' else .10})
    if mode=='wrong_final_identity':s['run_id']='unrelated_final_run'
    jsonwrite(p,s)
   if any(str(x).endswith('benchmark_strong_tr_test.py') for x in command):
    finalizer_commands.append({'kind':'strong_test','command':command,'timeout':kwargs.get('timeout')})
    if mode=='strong_test_timeout':
     clock.value+=kwargs['timeout'];raise self.TimeoutExpired(command,kwargs['timeout'])
    clock.value+=30
    frozen=Path(command[command.index('--frozen-selection')+1]);rid=command[command.index('--selected-run-id')+1]
    jsonwrite(fake_out/'adaptive_strong_baseline_test_comparison.json',{'status':'complete','passed':True,'run_id':rid,'frozen_selection_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()})
   if any(str(x).endswith('verify_adaptive_results.py') for x in command):
    finalizer_commands.append({'kind':'final_audit','command':command,'timeout':kwargs.get('timeout')})
    if mode=='final_audit_timeout':
     clock.value+=kwargs['timeout'];raise self.TimeoutExpired(command,kwargs['timeout'])
    clock.value+=15
    frozen=Path(command[command.index('--frozen-selection')+1]);selected=json.loads(frozen.read_text());out=Path(command[command.index('--output-dir')+1])
    jsonwrite(out/'adaptive_results_verification.json',{'status':'complete','passed':mode!='final_audit_failed','run_id':selected['run_id'],'loop_id':loop_id,'selected_step':selected['selected_step'],'frozen_selection_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()})
   return types.SimpleNamespace(returncode=0,stdout='',stderr='')
 m.subprocess=Subprocess()
 def note(text):
  notes.append(text)
  if mode=='backlog_failure' and len(notes)>1:raise RuntimeError('Mock Backlog CLI failure')
 m.backlog_note=note
 error=None
 try:m.main(argparse.Namespace(loop_id=loop_id,hours=hours))
 except BaseException as exception:error={'type':type(exception).__name__,'message':str(exception)}
 status=m.read(fake_out/'runs'/loop_id/'loop_status.json',{})
 cli=[]
 for command in commands:
  try:
   parser_ns['parser']().parse_args(command[5:]);cli.append({'passed':True})
  except SystemExit as exception:cli.append({'passed':False,'exit_code':exception.code})
 return {'name':name,'error':error,'status':status.get('status'),'best_candidate':status.get('best_development_candidate'),'competitive_on_development':status.get('competitive_on_development'),'awake_restored':status.get('awake_restoration',{}).get('enabled') is False,'total_wall_hours':status.get('total_wall_hours'),'children_launched':len(processes),'cli_compatible':all(x['passed'] for x in cli),'commands':commands,
  'child_cleanup_after_error':all(p.returncode is not None for p in processes),'clock_seconds':clock.value,'child_elapsed_seconds':[(p.finished_at if p.finished_at is not None else clock.value)-p.birth for p in processes],
  'stage_requested_seconds':[float(p.command[p.command.index('--max-wall-seconds')+1]) for p in processes],'stop_requested':[p.stop.exists() for p in processes],'finalizer_calls':finalizer_commands}

for name,kwargs in [
 ('profile_rejected',{'profile_pass':False}),
 ('unimproved_profile_rejected',{'profile_improved':False}),
 ('valid_no_trial_short_window',{}),
 ('wrong_baseline_split',{'wrong_baseline':True}),
 ('trainer_cli',{'mode':'rejected_selection','hours':.5}),
 ('wrong_run_selection',{'mode':'wrong_run_selection','hours':.5}),
 ('global_deadline',{'mode':'global_deadline','hours':.5}),
 ('stage_overrun',{'mode':'stage_overrun','hours':1}),
 ('backlog_failure',{'mode':'backlog_failure','hours':.5}),
 ('history_error',{'mode':'history_error','hours':.5}),
 ('checkpoint_disabled_cli',{'mode':'rejected_selection','hours':.5,'checkpointing':False}),
 ('accepted_new',{'mode':'accepted_new','hours':1}),
 ('finalizer_timeout',{'mode':'finalizer_timeout','hours':1}),
 ('wrong_final_identity',{'mode':'wrong_final_identity','hours':1}),
 ('strong_test_timeout',{'mode':'strong_test_timeout','hours':1}),
 ('merged_target_missed',{'mode':'merged_target_missed','hours':1}),
 ('final_audit_failed',{'mode':'final_audit_failed','hours':1}),
 ('final_audit_timeout',{'mode':'final_audit_timeout','hours':1}),
]:
 results.append(exercise(name,**kwargs))
 by={x['name']:x for x in results}
checks=[
 ('requires_passed_profile',by['profile_rejected']['error'] is not None and by['profile_rejected']['children_launched']==0),
 ('requires_measured_speed_improvement',by['unimproved_profile_rejected']['error'] is not None and by['unimproved_profile_rejected']['children_launched']==0),
 ('short_window_has_no_gpu_trial',by['valid_no_trial_short_window']['error'] is None and by['valid_no_trial_short_window']['children_launched']==0),
 ('rejects_ineligible_or_tampered_baseline',by['wrong_baseline_split']['error'] is not None),
 ('all_controller_trainer_argv_parse',by['trainer_cli']['cli_compatible']),
 ('rejects_wrong_run_child_selection',by['wrong_run_selection']['best_candidate'].get('run_id')=='lora_20261010_01'),
 ('global_deadline_requests_cooperative_stop',any(by['global_deadline']['stop_requested']) and by['global_deadline']['clock_seconds']<=2400),
 ('stage_quota_independently_enforced',all(t<=q+200 and (t<=q or stopped) for t,q,stopped in zip(by['stage_overrun']['child_elapsed_seconds'],by['stage_overrun']['stage_requested_seconds'],by['stage_overrun']['stop_requested']))),
 ('backlog_failure_never_orphans_child',by['backlog_failure']['child_cleanup_after_error'] and by['backlog_failure']['status'] in ('failed','complete_without_better_candidate')),
 ('exception_stops_owned_child_and_records_failure',by['history_error']['error'] is not None and by['history_error']['child_cleanup_after_error'] and by['history_error']['status']=='failed'),
 ('awake_request_restored_on_success_and_exception',all(by[n]['awake_restored'] for n in ('history_error','trainer_cli','accepted_new','finalizer_timeout'))),
 ('checkpoint_disabled_argv_supported',by['checkpoint_disabled_cli']['cli_compatible'] and all('--no-encoder-checkpoint' in c for c in by['checkpoint_disabled_cli']['commands'])),
 ('measured_target_passed_to_every_trainer',all(float(c[c.index('--target-turkish-wer')+1])==.1193939393939394 for row in results for c in row['commands'])),
 ('valid_new_dev_candidate_finalizes_once',by['accepted_new']['status']=='complete' and len([c for c in by['accepted_new']['finalizer_calls'] if c.get('kind') is None])==1),
 ('finalizer_has_explicit_cap_and_remaining_budget',all(0<call['timeout']<=1800 for row in results for call in row['finalizer_calls'])),
 ('finalizer_timeout_retained_as_failure',by['finalizer_timeout']['status']=='finalization_timed_out' and by['finalizer_timeout']['total_wall_hours']<=1),
 ('wrong_final_run_cannot_report_success',by['wrong_final_identity']['status']=='finalization_failed'),
 ('strong_test_timeout_retained_with_shared_cap',by['strong_test_timeout']['status']=='strong_test_timed_out' and by['strong_test_timeout']['total_wall_hours']<=1),
 ('competitive_flag_uses_reloaded_merged_dev',by['merged_target_missed']['status']=='complete' and by['merged_target_missed']['competitive_on_development'] is False),
 ('complete_requires_independent_final_audit',by['final_audit_failed']['status']=='final_audit_failed'),
 ('final_audit_timeout_retained_with_shared_cap',by['final_audit_timeout']['status']=='final_audit_timed_out' and by['final_audit_timeout']['total_wall_hours']<=1),
]
tree=ast.parse(source)
final_runs=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='run' and isinstance(node.func.value,ast.Name) and node.func.value.id=='subprocess']
checks.append(('all_blocking_subprocess_runs_bounded',all(any(k.arg=='timeout' for k in node.keywords) for node in final_runs)))
report={'scope':'CPU-only fixtures; all subprocesses/clocks/CLI/finalization mocked; no GPU or held-out test predictions','reviewed_controller_sha256':source_sha,'reviewed_trainer_sha256':hashlib.sha256((OUTPUT/'train_lora_batched.py').read_bytes()).hexdigest(),'reviewed_baseline_summary_sha256':hashlib.sha256((OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified/summary.json').read_bytes()).hexdigest(),
 'review_complete':True,'all_tested_guards_passed':all(v for _,v in checks),'checks':[{'name':n,'passed':v} for n,v in checks],'fixtures':results,'fixture_root':str(fixture_root),'note':'Failed checks are launch findings about the exact reviewed version. This report does not approve a GPU experiment.'}
(OUTPUT/'experiment_controller_cpu_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('review_complete','all_tested_guards_passed','reviewed_controller_sha256','checks')},indent=2))
