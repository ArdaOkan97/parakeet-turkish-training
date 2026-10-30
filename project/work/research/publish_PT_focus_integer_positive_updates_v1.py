"""One actual positive-update receipt; bounded JSON tail and small metadata only."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_majority_PT_focus_integer_01';DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def record(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
state=read(LOOP/'loop_status.json');assert state['status']=='running' and state['owned_launcher_pid']==17172
with (DIR/'training_history.json').open('rb') as stream:
    stream.seek(max(0,os.fstat(stream.fileno()).st_size-350000));tail=stream.read().decode('utf-8','ignore')
position=tail.rfind('\n  {\n    "step":');assert position>=0
text=tail[position+3:].strip();assert text.endswith(']');row=json.loads(text[:-1].rstrip())
assert row['step']>8250 and row['PT_focus_updates']==row['step']-8000
uniform=row['Turkish_domain_coverage']['uniform_pass'];focus=row['Turkish_domain_coverage']['PT_focus_transition']
assert uniform['draws']==uniform['unique_clips_seen']==17774+14*(row['step']-8250)
assert uniform['repeat_visits']==0 and focus['source_step']==8000
assert focus['Portuguese_draws_after_transition']==focus['old_five_draws_after_transition']==row['PT_focus_updates']
assert row['head_lr']==1e-5 and row['lora_lr']==1e-4
proof_path=DIR/'majority_initial_source_equivalence.json';proof=read(proof_path)
assert proof['passed'] and proof['actual_GPU_decode'] and proof['compared_fixed_DEV_utterances']==350
assert proof['source_selected_step']==8250 and proof['all_six_retention_gates_passed'] is True
config=read(DIR/'run_config.json');assert config['retention_numeric_correction']['source_failure_counter']==5
assert config['retention_numeric_correction']['corrected_failure_counter']==0
assert config['retention_numeric_correction']['qualified_initial_BEST']==8250
best=ROOT/'work'/(RUN+'_best.pt');latest=ROOT/'work'/(RUN+'_latest.pt');assert best.is_file() and latest.is_file()
identity=read(LOOP/(RUN+'_identity.json'));assert identity['pid']==26792 and identity['ppid']==17172
now=datetime.now(timezone.utc).isoformat()
actual={'schema':1,'kind':'PT_focus_integer_actual_source350_positive_updates_v1','passed':True,'observed_at_utc':now,
    'status':'actual_sole_GPU_running','run_id':RUN,'actual_controller_session_id':43870,'actual_launch_tool_chunk':'6016e6',
    'actual_native_launcher_pid':17172,'actual_worker_pid':26792,'completed_steps':row['step'],
    'new_updates_after_integer_correction':row['step']-8250,'PT_focus_updates':row['PT_focus_updates'],
    'uniform_full_updates':row['step']-6500,'majority_updates':row['step']-7621,'lora_lr_fork_updates':row['step']-7750,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_lr_fork_origin_step':7750,'PT_focus_origin_step':8000,
    'new_uniform_unique_TR_clips':uniform['unique_clips_seen'],'uniform_TR_pool':253798,'repeat_visits':0,
    'configured_end_step':25109,'remaining_updates':25109-row['step'],'full_pass_complete':False,
    'actual_source350':record(proof_path),'source_all_six_count_qualified':True,'source_PT_word_errors':66,
    'source_TR_word_errors':329,'source_false_gate_history_preserved':True,
    'declared_corrected_source_failure_counter':0,'declared_corrected_initial_BEST_step':8250,
    'all207_initial_serialization_and_actual_native_optimizer_updates_complete':True,
    'initial_BEST_checkpoint':{'path':str(best),'bytes':best.stat().st_size,'checkpoint_bytes_read':False},
    'latest_checkpoint':{'path':str(latest),'bytes':latest.stat().st_size,'checkpoint_bytes_read':False},
    'head_lr':row['head_lr'],'lora_lr':row['lora_lr'],'preservation_coefficients_by_language':row['preservation_coefficients_by_language'],
    'run_config':record(DIR/'run_config.json'),'identity':record(LOOP/(RUN+'_identity.json')),
    'observed_peak_owned_private_bytes':state['peak_observed_owned_private_bytes'],
    'new_complete_DEV_pending':True,'source_equals_source_is_not_new_quality_gain':True,
    'cold_updates_not_steady_throughput_claimed':True,'goal_complete':False,'GPU_or_model_read_by_this_receipt':False}
out=HERE/'PT_focus_integer_actual_source350_positive_updates_v1.json'
with out.open('x',encoding='utf-8') as stream:json.dump(actual,stream,indent=2)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath);assert lease['owner']=='/root/benchmark_design' and lease['actual_controller_session_id']==43870
lease.update(status='actual_sole_GPU_integer_gate_positive_updates',updated_at_utc=now,actual_owned_worker_pid=26792,
    actual_completed_step=row['step'],actual_PT_focus_updates=row['PT_focus_updates'],actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],
    source350_and_first_optimizer_pending=False,actual_source350_passed=True,actual_positive_update_receipt=record(out))
atomic(leasepath,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan['updated_at_utc']=now;plan['objective_complete']=False
plan['status_authority']=f'Actualsole43870 LIVE {row["step"]},newintegercontinuation{row["step"]-8250}/PTtotal{row["PT_focus_updates"]}; actualsource350all6 passed/207initialBEST8250+latest saved/nativepositiveupdates. Uniformunique{uniform["unique_clips_seen"]}/253798,no repeats; fixedintegergates unchanged,historicalfalse preserved. NewDEVpending.'
plan['actual_current_training']=actual
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_integer_gate_corrected_samepass_running',run_id=RUN,actual_current_run=RUN,
    actual_GPU_training_running=True,current_frontier_source_step=8250,actual_completed_steps=row['step'],
    actual_PT_focus_updates=row['PT_focus_updates'],actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],
    actual_new_uniform_unique_clips=uniform['unique_clips_seen'],current_remaining_uniform_updates=25109-row['step'],
    current_configured_end_step=25109,current_run_source350_passed=True,current_run_first_update_pending=False,
    current_failure_counter=0,current_failure_counter_after_qualified_accounting_pending=False,
    current_actual_initial_BEST_step=8250,all_six_champion_yet=True,latest_actual_float_gate_passed=None,
    retention_comparison='exact_integer_original_limits',actual_positive_update_receipt=out.name,
    actual_majority_updates=row['step']-7621,current_lora_lr_fork_updates=row['step']-7750)
item['pending']=['NextcompleteDEV8500/strictall6selection','Continueactualexisting253798-unique fullpass withunchangedmemoryguards','Frozenheldout/competitivegoal verification']
plan['automatic_execution_policy']['PT_focused_replay']='Healthysole43870 actualsource350all6passed+full207initialBEST8250saved andpositiveupdates. Exactintegeroriginalgates; historicalfalse records preserved. Same14TR1PT1oldfive/LRs/frontier; no renewedgrace/moments reset. NextcompleteDEV authoritative; budgetonlyautoexact, retention/resource/manualreview.'
atomic(planpath,plan)
notes=f'Actualintegergate continuation sole43870 LIVEglobal{row["step"]}/new{row["step"]-8250},PTfocustotal{row["PT_focus_updates"]},native17172/worker26792. Source3508250exactall7raw/emittedpaths/counts passed (PT66/TR329 all6countsgates); full207initialBEST8250/latest serialized andnativepositiveupdates. Sourcefalsehistory5/None preserved; onlynewnamespace count-qualified0/BEST8250 accounting. Uniformunique{uniform["unique_clips_seen"]}/253798,0repeat,end25109; LR1e-4/head1e-5/mix14TR1PT1oldfive unchanged. NewDEV8500pending; initialsourceequality isnotqualitygain/fullpassclaim. Plan/lease consistent; sameprojectlogger compatibility coordinated. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'positive_updates':row['step']-8250,'step':row['step'],'unique_TR':uniform['unique_clips_seen'],
    'receipt':record(out),'lease_plan_Backlog_updated':True}))
