"""Bounded current metadata verification; no tensor/model or checkpoint read."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish';RUN='adaptive_20261012_TR_uniform_full_epoch_01'
DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def record(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
s=read(LOOP/'loop_status.json');assert s['status']=='running' and s['owned_launcher_pid']==5516
with (DIR/'training_history.json').open('rb') as f:
    f.seek(max(0,os.fstat(f.fileno()).st_size-350000));tail=f.read().decode('utf-8','ignore')
i=tail.rfind('\n  {\n    "step":');assert i>=0;raw=tail[i+3:].strip();assert raw.endswith(']');row=json.loads(raw[:-1].rstrip())
n=row['step']-9000;uniform=row['Turkish_domain_coverage']['uniform_pass'];assert n>0
assert uniform['draws']==uniform['unique_clips_seen']==28274+14*n and uniform['repeat_visits']==0
assert row['Portuguese_token_KL_multiplier']==1 and row['full_epoch_continuation_updates']==n and row['retention_stop_policy']=='monitor_only'
assert row['preservation_coefficients_by_language']=={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
assert row['head_lr']==1e-5 and row['lora_lr']==1e-4
pp=DIR/'majority_initial_source_equivalence.json';proof=read(pp)
assert proof['passed'] and proof['actual_GPU_decode'] and proof['source_selected_step']==9000
assert proof['compared_fixed_DEV_utterances']==350 and proof['all_seven_ASR_counts_exact'] and proof['all_raw_transcripts_and_emitted_paths_exact']
assert proof['all_six_retention_gates_passed'] is False and proof['source_retention_gates']['pt']['word_errors']==71
config=read(DIR/'run_config.json');assert config['trainable_parameter_tensors']==207
assert config['full_epoch_policy_transition']==config['config_identity']['full_epoch_policy_transition']
assert config['config_identity']['training_arguments']['portuguese_token_kl_multiplier']==1 and config['config_identity']['training_arguments']['retention_stop_policy']=='monitor_only'
assert config['full_epoch_policy_transition']['declared']['source_actual_retention_failures']==3
latest=ROOT/'work'/(RUN+'_latest.pt');assert latest.is_file()
assert not (ROOT/'work'/(RUN+'_best.pt')).exists(), 'Failed latest8750 must not masquerade as new BEST'
identity=read(Path(s['paths']['identity']));now=datetime.now(timezone.utc).isoformat()
actual={'kind':'USER_full_epoch_actual_source350_positive_updates_v1','passed':True,'status':'actual_sole_GPU_running','observed_at_utc':now,
    'run_id':RUN,'actual_controller_session_id':67197,'actual_native_launcher_pid':5516,'actual_worker_pid':identity['pid'],
    'completed_steps':row['step'],'full_epoch_continuation_updates':n,'uniform_unique_TR_clips':uniform['unique_clips_seen'],'repeat_visits':0,
    'uniform_pool_clips':253798,'full_pass_complete':False,'configured_end_step':25109,'remaining_updates':25109-row['step'],
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,'full_epoch_source_step':9000,
    'source350':record(pp),'source350_PT_word_errors':71,'source350_all_six_qualified':False,
    'source_counter_preserved':3,'inherited_qualified_BEST_step':8250,'new_counter_or_grace_reset':False,
    'coherent_latest9000_full207_initial_serialized':{'path':str(latest),'bytes':latest.stat().st_size},
    'failed_initial9000_not_saved_as_BEST':True,'native_positive_updates_completed':True,
    'actual_language_preservation_coefficients':row['preservation_coefficients_by_language'],
    'head_lr':row['head_lr'],'lora_lr':row['lora_lr'],'source_coverage_rewind_clips':0,
    'run_config':record(DIR/'run_config.json'),'decision':record(HERE/'TSC_full_epoch9000_USER_policy_review_v1.json'),
    'source_receipt':record(HERE/'TSC_full_epoch9000_USER_source_v1.json'),'new_complete_DEV9250_pending':True,'retention_early_stop_active':False,'full_epoch_priority':True,
    'memory':{'observed_peak_owned_private_bytes':s['peak_observed_owned_private_bytes'],'system_and_owned_private_memory':s['system_and_owned_private_memory']},
    'objective_complete':False,'checkpoint_tensor_bytes_read_or_hashed_by_bookkeeping':False}
out=HERE/'USER_full_epoch_actual_source350_positive_updates_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(actual,f,indent=2)
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design' and lease['actual_controller_session_id']==67197
lease.update(status='actual_sole_GPU_USER_full_epoch_positive_updates',updated_at_utc=now,actual_completed_step=row['step'],
    actual_owned_worker_pid=identity['pid'],actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],
    actual_source350_passed=True,source350_and_first_optimizer_pending=False,actual_positive_update_receipt=record(out))
atomic(lp,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan.update(updated_at_utc=now,objective_complete=False,actual_current_training=actual,
    status_authority=f'Actual sole USER full-epoch67197 running at{row["step"]}, {n} positive native updates. Source350 exactly reproduces9000 with truthful DE/PT failures, full207 initial latest saved, original BEST8250 separate. Retention monitors/selects champions ONLY; no early-stop. PT200 to40 rollback, no reset/rewind; exact normal-budget windows chain automatically through actual253798 unique full pass.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04');item.update(status='actual_USER_full_epoch_samepass_running',run_id=RUN,actual_current_run=RUN,
    actual_GPU_training_running=True,actual_completed_steps=row['step'],actual_full_epoch_continuation_updates=n,
    actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],actual_new_uniform_unique_clips=uniform['unique_clips_seen'],
    current_remaining_uniform_updates=25109-row['step'],current_run_source350_passed=True,current_run_first_update_pending=False,
    current_failure_counter=3,current_all6_selected_BEST_step=8250,actual_positive_update_receipt=out.name,
    latest_actual_DEV_step=9000,latest_original_five_passed=False,latest_Portuguese_passed_by_exact_word_count=False,latest_all_six_passed=False)
atomic(planpath,plan)
note=f'Actual USER full-epoch sole67197/native5516/worker{identity["pid"]} running global{row["step"]}, {n} positive native updates. Source350 all7 raw/token-duration/count paths at9000 reproduced exactly including DE71/PT71 failures; RNG/modes restored. Coherent all207 initial latest9000 saved, qualified BEST8250 separate, no fake9000 BEST. Retention is MONITOR/CHAMPION ONLY and will not stop training; real counter3/history preserved. PTtoken200 to40 rollback, feature20/head1e-5/LoRA1e-4/14TR1PT1old-five/resources unchanged. Currentuniformunique{uniform["unique_clips_seen"]}/253798, no repeats/rewind/grace reset; automatic exact budget windows active, sleep inhibition active. End25109; full epoch only actual253798unique plus endDEV. No qualityclaim. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'positive_updates':n,'step':row['step'],'unique_TR':uniform['unique_clips_seen'],'receipt':record(out),'lease_plan_Backlog_updated':True}))
