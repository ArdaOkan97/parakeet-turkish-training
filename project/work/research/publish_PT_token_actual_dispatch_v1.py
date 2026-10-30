"""Publish factual owned launch; source350/initialsave/update remain pending."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_PTtokenKD5x_01';LOOP=HERE/'runs'/(RUN+'_loop_01')
p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True);p.add_argument('--chunk',required=True);a=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def record(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
def atomic(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
s=read(LOOP/'loop_status.json');assert s['status']=='running' and s['run_id']==RUN
now=datetime.now(timezone.utc).isoformat()
actual={'kind':'PT_token5x_actual_sole_GPU_dispatch_v1','status':'actual_sole_GPU_startup','observed_at_utc':now,
    'run_id':RUN,'actual_session_id':a.session,'actual_launch_tool_chunk':a.chunk,'actual_owned_native_launcher_pid':s['owned_launcher_pid'],
    'source_step':8750,'completed_steps':8750,'new_updates':0,'source_checkpoint_role':'coherent_latest',
    'uniform_unique_TR_clips':24774,'source_retention_failures_preserved':2,'inherited_qualified_BEST_step':8250,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_updates':16359,'coverage_rewind_clips':0,'new_grace_or_counter_reset':False,
    'only_changed_coefficient':{'Portuguese_token_KL_before':40,'Portuguese_token_KL_after':200,'multiplier':5},
    'actual_language_preservation_coefficients':{'pt':[200,2,20],'old_five':[40*5/3,2*5/3,20*5/3]},
    'head_lr':1e-5,'lora_lr':1e-4,'source350_initial_save_and_native_update_pending':True,
    'source_receipt':record(HERE/'TSC_PT_token8750_ROOT_source_v1.json'),'preflight':record(HERE/'PT_token5x_actual_input_preflight_v1.json'),
    'actual_parser_validation':record(HERE/'PT_token5x_actual_worker_parser_validation_v1.json'),
    'actual_CPU_per_clip_binding':record(HERE/'PT_token5x_actual_CPU_per_clip_binding_v1.json'),
    'decision':record(HERE/'TSC_PT_token5x_measured_retention_repair_review_v1.json'),
    'worker':record(HERE/'train_lora_batched_tsc_uniform_pttoken_v1.py'),'controller':record(HERE/'run_TSC_uniform_pttoken_v1.py'),
    'goal_complete':False,'quality_recovery_claimed':False,'checkpoint_bytes_read_by_bookkeeping':False}
out=HERE/'PT_token5x_actual_sole_GPU_dispatch_v1.json'
with out.open('x',encoding='utf-8') as f:json.dump(actual,f,indent=2)
lp=ROOT/'work/gpu_training_owner.json';lease=read(lp);assert lease['owner']=='/root/benchmark_design'
for key in ('actual_exit_code','actual_closed_tool_chunk','actual_owned_worker_pid'):lease.pop(key,None)
lease.update(status='actual_sole_GPU_PT_token5x_source8750_startup',updated_at_utc=now,actual_controller_session_id=a.session,
    actual_launch_tool_chunk=a.chunk,actual_owned_native_launcher_pid=s['owned_launcher_pid'],actual_GPU_running=True,running=True,
    reserved=True,GPU_reserved=True,ROOT_will_not_launch_concurrent_GPU=True,actual_GPU_launched=True,target_run_id=RUN,
    source_frontier=8750,actual_completed_step=8750,actual_new_uniform_unique_TR_clips=24774,source_failure_counter_preserved=2,
    actual_source350_passed=False,source350_and_first_optimizer_pending=True,actual_controller_loop=str(LOOP),
    worker_sha256=actual['worker']['sha256'],controller_sha256=actual['controller']['sha256'],actual_source_receipt=actual['source_receipt'],actual_dispatch_receipt=record(out))
atomic(lp,lease)
pp=HERE/'experiment_plan.json';plan=read(pp);plan.update(updated_at_utc=now,objective_complete=False,actual_current_training=actual,
    status_authority='Actual sole PT-token5x continuation launched from coherent latest8750 after archived-source and real parser/native preflight. PT-only token40 to200; counter2 and BEST8250 retained. Source350, initial full207 serialization and positive native updates pending.')
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_PT_token5x_samepass_startup',run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=True,
    actual_completed_steps=8750,current_frontier_source_step=8750,actual_new_uniform_unique_TR_clips=24774,
    actual_new_uniform_unique_clips=24774,current_remaining_uniform_updates=16359,current_configured_end_step=25109,
    current_failure_counter=2,current_all6_selected_BEST_step=8250,current_run_source350_passed=False,current_run_first_update_pending=True,
    source_receipt='TSC_PT_token8750_ROOT_source_v1.json',prepared_worker='train_lora_batched_tsc_uniform_pttoken_v1.py',
    prepared_boundary='tsc_uniform_pttoken_constraint_transition_v1.py',current_run_actual_dispatch_receipt=out.name,
    current_Portuguese_token_KL=200,current_old_five_token_KL=40*5/3,current_feature_distillation=20,
    explicit_failed_branch_TR_clip_rewind=0,execution_owner=f'/root/benchmark_design sole GPU session{a.session}')
for key in ('actual_closed_session','actual_closed_exit_code','actual_closed_tool_chunk'):item.pop(key,None)
item['pending']=['Actual same8750 source350 and all207 initial latest save','Actual native positive updates','Next complete DEV9000: pass all six or stop at genuine counter3']
plan['automatic_execution_policy']['GPU_ownership']=f'benchmark_design owns sole session{a.session}; ROOT will not launch concurrent GPU. Actual lease/runtime progress authoritative.'
plan['automatic_execution_policy']['PT_focused_replay']='Only PT tokenKL40 to200 from coherent latest8750; old-five coefficients, feature20, duration2, mix, LRs and all207 state unchanged. Counter2/BEST8250 and coverage24774 retained without rewind or renewed grace. Budget-only actual closed0 may exact-continue; retention/resource/manual stops require evidence review.'
atomic(pp,plan)
note=f'Actual sole GPU PT-token5x session{a.session}/{a.chunk} launched from coherent latest8750 after real archive/preflight/parser validation. Same full207 weights/Adam/RNG/history/sampler24774 unique, no rewind; genuine counter2/BEST8250 retained. Only Portuguese token KL40 to200; feature20/duration2 and all old-five coefficients, LR1e-4/head1e-5,14TR1PT1old-five/native memory guards unchanged. Source350, initial latest8750 full207 save and native positive updates pending. Next complete DEV9000 must pass all six or stop counter3. End25109/16359 remaining; no quality or full-pass claim. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',note],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'dispatch':record(out),'session':a.session,'native_pid':s['owned_launcher_pid'],'lease_plan_Backlog_updated':True}))
