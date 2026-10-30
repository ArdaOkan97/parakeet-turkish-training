"""Actual source350 and positive updates, bounded metadata-only bookkeeping."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_featureKD5x_02';DIR=HERE/'runs'/RUN;LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def record(path):
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}
def atomic(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(path)
state=read(LOOP/'loop_status.json');assert state['status']=='running' and state['owned_launcher_pid']==23796
with (DIR/'training_history.json').open('rb') as f:
    f.seek(max(0,os.fstat(f.fileno()).st_size-350000));tail=f.read().decode('utf-8','ignore')
i=tail.rfind('\n  {\n    "step":');assert i>=0;raw=tail[i+3:].strip();assert raw.endswith(']');row=json.loads(raw[:-1].rstrip())
n=row['step']-8250;assert n>0 and row['PT_focus_updates']==row['step']-8000
uniform=row['Turkish_domain_coverage']['uniform_pass'];assert uniform['draws']==uniform['unique_clips_seen']==17774+14*n and uniform['repeat_visits']==0
assert row['feature_distillation_coefficient']==20 and row['replay_kl_coefficient']==40 and row['duration_kl_coefficient']==2
assert row['head_lr']==1e-5 and row['lora_lr']==1e-4
assert row['preservation_coefficients_by_language']=={'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]}
proof_path=DIR/'majority_initial_source_equivalence.json';proof=read(proof_path)
assert proof['passed'] and proof['actual_GPU_decode'] and proof['all_six_retention_gates_passed']
assert proof['source_selected_step']==8250 and proof['compared_fixed_DEV_utterances']==350
config=read(DIR/'run_config.json');constraint=config['retention_constraint_transition'];assert constraint==config['config_identity']['retention_constraint_transition']
assert constraint['declared']['rewound_TR_clips']==7000 and config['config_identity']['training_arguments']['feature_distillation']==20
best=ROOT/'work'/(RUN+'_best.pt');latest=ROOT/'work'/(RUN+'_latest.pt');assert best.is_file() and latest.is_file()
now=datetime.now(timezone.utc).isoformat()
actual={'kind':'feature5x_actual_source350_positive_updates_v1','passed':True,'status':'actual_sole_GPU_running',
    'observed_at_utc':now,'run_id':RUN,'actual_session_id':98758,'actual_launch_tool_chunk':'017543',
    'actual_native_launcher_pid':23796,'actual_worker_pid':18932,'completed_steps':row['step'],'feature_constraint_updates':n,
    'PT_focus_updates':row['PT_focus_updates'],'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'uniform_unique_TR_clips':uniform['unique_clips_seen'],'uniform_TR_pool':253798,'repeat_visits':0,'full_pass_complete':False,
    'configured_end_step':25109,'remaining_updates':25109-row['step'],'coherent_source_role':'qualified_BEST8250',
    'source_uniform_unique_clips':17774,'failed8750_branch_uniform_unique_clips':24774,'explicit_source_rewind_clips':7000,
    'failed_branch_history_preserved':True,'new_counter_or_grace_reset':False,'initial_source_retention_failures_preserved':0,
    'all207_initial_BEST_latest_serialization_and_native_positive_updates':True,'source350':record(proof_path),
    'actual_language_preservation_coefficients':row['preservation_coefficients_by_language'],'head_lr':row['head_lr'],'lora_lr':row['lora_lr'],
    'only_changed_constraint':'feature_distillation4_to20','run_config':record(DIR/'run_config.json'),
    'diagnostic':record(HERE/'retention_components_8750_actual_GPU_v3.json'),'decision':record(HERE/'TSC_feature5x_measured_retention_repair_review_v1.json'),
    'initial_BEST_checkpoint':{'path':str(best),'bytes':best.stat().st_size},'latest_checkpoint':{'path':str(latest),'bytes':latest.stat().st_size},
    'source_equality_is_not_quality_gain':True,'new_complete_DEV_pending':True,'checkpoint_bytes_read_or_hashed_by_bookkeeping':False,
    'observed_peak_owned_private_bytes':state['peak_observed_owned_private_bytes'],'goal_complete':False}
out=HERE/'feature5x_actual_source350_positive_updates_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(actual,f,indent=2)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath);assert lease['owner']=='/root/benchmark_design' and lease['actual_controller_session_id']==98758
lease.update(status='actual_sole_GPU_feature5x_positive_updates',updated_at_utc=now,actual_owned_worker_pid=18932,actual_completed_step=row['step'],
    actual_PT_focus_updates=row['PT_focus_updates'],actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],
    source350_and_first_optimizer_pending=False,actual_source350_passed=True,actual_positive_update_receipt=record(out))
atomic(leasepath,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan['updated_at_utc']=now;plan['objective_complete']=False;plan['actual_current_training']=actual
plan['status_authority']=f'Actualsolefeature5x98758 LIVE{row["step"]}/new{n}; source350all7exact+all6countqualified andinitialfull207BEST8250+latest saved/nativepositiveupdates. CoherentBEST8250rewind17774source, currentunique{uniform["unique_clips_seen"]}/253798,no repeats. Onlyfeature4→20; newDEVpending.'
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_measured_feature5x_samepass_running',run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=True,
    actual_completed_steps=row['step'],actual_PT_focus_updates=row['PT_focus_updates'],actual_feature_constraint_updates=n,
    actual_new_uniform_unique_TR_clips=uniform['unique_clips_seen'],actual_new_uniform_unique_clips=uniform['unique_clips_seen'],
    current_remaining_uniform_updates=25109-row['step'],current_run_source350_passed=True,current_run_first_update_pending=False,
    current_failure_counter=0,current_actual_initial_BEST_step=8250,current_all6_selected_BEST_step=8250,
    actual_positive_update_receipt=out.name,latest_actual_DEV_step=8250,latest_original_five_passed=True,
    latest_Portuguese_passed_by_exact_word_count=True,latest_all_six_passed=True)
item['pending']=['NextactualDEV8500/strictall6 gates','Continuecoherentexisting253798-unique fullpass subjecttoguards','Frozenheldout/competitivegoal verification']
plan['automatic_execution_policy']['PT_focused_replay']='Measuredfeature5xactual93732firstattemptfailedbeforemodel duecopiedCLIfeature4 predicate; preserved0updates. Correctedv2actual98758 live source350all6/207initialsave/positiveupdates. SameLR/mix/integergates/Adam/RNG/horizon; explicitBEST8250rewindsource17774, no fakefailed24774carry. NextDEVauthoritative; budgetonlymayautoexact, retention/resource/manualreview.'
atomic(planpath,plan)
notes=f'Actualfeature5x02 sole98758 LIVEglobal{row["step"]}/new{n},native23796/worker18932. Source3508250all7raw/emitted/count equality+all6integergates passed; coherentfull207BEST8250/latest initializedandnativepositiveupdates. OnlyFeature4→20,PT20/oldfive100/3; KL40/duration2/LR1e-4/head1e-5/mix14TR1PT1oldfive unchanged. Currentuniformunique{uniform["unique_clips_seen"]}/253798,0repeat, from17774BESTsource (7000cliprewindvsfailed8750/24774explicit); failedbranchpreserved, no counter/grace/momentreset/end25109. NewDEV8500pending,noqualitygain/fullpassclaim. V16sole82097/PID14588 all46oldmaps preserved/sameproject. Priorfeature01CLOSED1/ab1b6d0updates mechanicalpredicate preserved/repairedinnewv2. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'step':row['step'],'positive_updates':n,'unique_TR':uniform['unique_clips_seen'],'receipt':record(out),'plan_Backlog_lease_updated':True}))
