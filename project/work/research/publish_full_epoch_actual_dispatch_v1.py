"""Publish factual owned launch; source350/initialsave/update remain pending."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_full_epoch_01';LOOP=HERE/'runs'/(RUN+'_loop_01')
p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True);p.add_argument('--chunk',required=True);a=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def record(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
s=read(LOOP/'loop_status.json');assert s['status']=='running' and s['run_id']==RUN
now=datetime.now(timezone.utc).isoformat()
actual={'kind':'USER_full_epoch_actual_sole_GPU_dispatch_v1','status':'actual_sole_GPU_startup','observed_at_utc':now,
    'run_id':RUN,'actual_session_id':a.session,'actual_launch_tool_chunk':a.chunk,'actual_owned_native_launcher_pid':s['owned_launcher_pid'],
    'source_step':9000,'completed_steps':9000,'new_updates':0,'source_checkpoint_role':'coherent_latest',
    'uniform_unique_TR_clips':28274,'source_retention_failures_preserved':3,'inherited_qualified_BEST_step':8250,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16109,'coverage_rewind_clips':0,'new_grace_or_counter_reset':False,
    'only_changed_coefficient':{'Portuguese_token_KL_before':200,'Portuguese_token_KL_after':40,'multiplier':1,'retention_stop_policy':'monitor_only'},
    'actual_language_preservation_coefficients':{'pt':[40,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
    'head_lr':1e-5,'lora_lr':1e-4,'source350_initial_save_and_native_update_pending':True,
    'source_receipt':record(HERE/'TSC_full_epoch9000_USER_source_v1.json'),'preflight':record(HERE/'full_epoch_actual_input_preflight_v1.json'),
    'actual_parser_validation':record(HERE/'full_epoch_actual_worker_parser_validation_v1.json'),
    'retention_early_stop_active':False,'strict_integer_champion_gates_unchanged':True,
    'decision':record(HERE/'TSC_full_epoch9000_USER_policy_review_v1.json'),
    'worker':record(HERE/'train_lora_batched_tsc_uniform_full_epoch_v1.py'),'controller':record(HERE/'run_TSC_uniform_full_epoch_v1.py'),
    'goal_complete':False,'quality_recovery_claimed':False,'checkpoint_bytes_read_by_bookkeeping':False}
out=HERE/'USER_full_epoch_actual_sole_GPU_dispatch_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(actual,f,indent=2)
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design'
for key in ('actual_exit_code','actual_closed_tool_chunk','actual_owned_worker_pid'):lease.pop(key,None)
lease.update(status='actual_sole_GPU_USER_full_epoch_source9000_startup',updated_at_utc=now,actual_controller_session_id=a.session,
    actual_launch_tool_chunk=a.chunk,actual_owned_native_launcher_pid=s['owned_launcher_pid'],actual_GPU_running=True,running=True,
    reserved=True,GPU_reserved=True,ROOT_will_not_launch_concurrent_GPU=True,actual_GPU_launched=True,target_run_id=RUN,
    source_frontier=9000,actual_completed_step=9000,actual_new_uniform_unique_TR_clips=28274,source_failure_counter_preserved=3,
    actual_source350_passed=False,source350_and_first_optimizer_pending=True,actual_controller_loop=str(LOOP),
    worker_sha256=actual['worker']['sha256'],controller_sha256=actual['controller']['sha256'],actual_source_receipt=actual['source_receipt'],actual_dispatch_receipt=record(out))
atomic(lp,lease)
pp=HERE/'experiment_plan.json';plan=read(pp);plan.update(updated_at_utc=now,objective_complete=False,actual_current_training=actual,
    status_authority='Actual sole USER full-epoch continuation launched from coherent latest9000 after real source/native/parser preflight. Retention monitor-only and PT200 to40 rollback; counter3/BEST8250/28274unique retained without rewind. Sleep inhibition and automatic exact normal-budget chaining active. Source350 and native positive updates pending.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_USER_full_epoch_samepass_startup',run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=True,
    actual_completed_steps=9000,current_frontier_source_step=9000,actual_new_uniform_unique_TR_clips=28274,
    actual_new_uniform_unique_clips=28274,current_remaining_uniform_updates=16109,current_configured_end_step=25109,
    current_failure_counter=3,current_all6_selected_BEST_step=8250,current_run_source350_passed=False,current_run_first_update_pending=True,
    source_receipt='TSC_full_epoch9000_USER_source_v1.json',prepared_worker='train_lora_batched_tsc_uniform_full_epoch_v1.py',
    prepared_boundary='tsc_uniform_full_epoch_transition_v1.py',current_run_actual_dispatch_receipt=out.name,
    current_Portuguese_token_KL=40,current_old_five_token_KL=40*5/3,current_feature_distillation=20,
    explicit_failed_branch_TR_clip_rewind=0,execution_owner=f'/root/benchmark_design sole GPU session{a.session}')
for key in ('actual_closed_session','actual_closed_exit_code','actual_closed_tool_chunk'):item.pop(key,None)
item.update(retention_early_stop_active=False,champion_integer_gates_unchanged=True,current_actual_initial_BEST_step=None)
item['pending']=['Actual same8750 source350 and all207 initial latest save','Actual native positive updates','Continue through actual253798 unique pass/end DEV; retention monitors/selects champions only']
plan['automatic_execution_policy']['GPU_ownership']=f'benchmark_design owns sole session{a.session}; ROOT will not launch concurrent GPU. Actual lease/runtime progress authoritative.'
plan['automatic_execution_policy']['PT_focused_replay']='USER priority: retention failures update history/counters and strict integer champion eligibility only; never stop the full pass. Coherent latest9000/no rewind, PT200 to40 rollback/feature20/LRs/mix unchanged. Actual complete budget windows autoexact regardless of failed gates; resource/nonfinite/crash protections unchanged.'
atomic(pp,plan)
note=f'Actual sole USER full-epoch continuation session{a.session}/{a.chunk} launched from coherent latest9000 after real archive/source/native/parser preflight. Preserve full207 weights/Adam/RNG/history/sampler28274unique, counter3/BEST8250, no rewind or new grace. PTtoken200 to40 rollback; feature20/head1e-5/LoRA1e-4/14TR1PT1old-five/native resources unchanged. RETENTION MONITOR ONLY: every failure is logged and strict all-six integer gates still select champions, but no retention early-stop before full epoch. Normal budget windows automatically exact-resume even if retention fails. Sleep inhibition active; source350/native positive updates pending. End25109/16109updates left; fullpass only actual253798unique and complete end DEV. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'dispatch':record(out),'session':a.session,'native_pid':s['owned_launcher_pid'],'lease_plan_Backlog_updated':True}))
