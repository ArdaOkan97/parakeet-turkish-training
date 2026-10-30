"""Factual measured repair launch; no tensor/model/GPU reads."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_featureKD5x_02';LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def record(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)
state=read(LOOP/'loop_status.json');assert state['status']=='running' and state['owned_launcher_pid']==23796
now=datetime.now(timezone.utc).isoformat();actual={'kind':'feature5x_measured_retention_repair_actual_dispatch_v1','run_id':RUN,
    'status':'actual_sole_GPU_startup','actual_GPU_running':True,'actual_session_id':98758,'actual_launch_tool_chunk':'017543',
    'actual_owned_native_launcher_pid':23796,'observed_at_utc':now,'source_step':8250,'source_checkpoint_role':'qualified_BEST',
    'completed_steps':8250,'new_updates':0,'uniform_unique_TR_clips':17774,'failed_latest_branch_unique_clips':24774,
    'explicit_TR_clip_rewind':7000,'source_failed_branch_history_preserved':True,
    'uniform_origin_step':6500,'quota_boundary_step':7621,'lora_boundary_step':7750,'PT_focus_boundary_step':8000,
    'configured_end_step':25109,'remaining_update_budget':16859,'source350_initial207_positive_updates_pending':True,
    'declared_only_change':{'feature_distillation':{'source':4,'active':20},'token_KL':40,'duration_KL':2},
    'head_lr':1e-5,'lora_lr':1e-4,'TR_replay_mix':'14TR+1PT+1uniformoldfive/B8x2',
    'source_failure_counter_preserved':0,'source_selected_BEST_preserved':8250,'new_grace_or_counter_migration':False,
    'diagnostic':record(HERE/'retention_components_8750_actual_GPU_v3.json'),
    'measured_decision':record(HERE/'TSC_feature5x_measured_retention_repair_review_v1.json'),
    'source_receipt':record(HERE/'TSC_feature5x_BEST8250_ROOT_source_v2.json'),
    'preflight':record(HERE/'feature5x_actual_input_preflight_v2.json'),
    'worker':record(HERE/'train_lora_batched_tsc_uniform_constraint_v2.py'),
    'controller':record(HERE/'run_TSC_uniform_constraint_v2.py'),
    'objective_complete':False,'actual_initial_equality_is_not_quality_gain':True}
out=HERE/'feature5x_actual_sole_GPU_dispatch_v2.json'
with out.open('x',encoding='utf-8') as stream:json.dump(actual,stream,indent=2)
leasepath=ROOT/'work/gpu_training_owner.json';lease=read(leasepath);assert lease['owner']=='/root/benchmark_design'
for key in ('actual_exit_code','actual_closed_tool_chunk','actual_owned_worker_pid'):lease.pop(key,None)
lease.update(status='actual_sole_GPU_measured_feature5x_BEST8250_startup',updated_at_utc=now,actual_controller_session_id=98758,
    actual_launch_tool_chunk='017543',actual_owned_native_launcher_pid=23796,actual_GPU_running=True,running=True,reserved=True,GPU_reserved=True,
    ROOT_will_not_launch_concurrent_GPU=True,target_run_id=RUN,source_frontier=8250,actual_completed_step=8250,
    actual_new_uniform_unique_TR_clips=17774,actual_source350_passed=False,source350_and_first_optimizer_pending=True,
    actual_controller_loop=str(LOOP),worker_sha256=actual['worker']['sha256'],controller_sha256=actual['controller']['sha256'],
    actual_source_receipt=actual['source_receipt'],actual_dispatch_receipt=record(out))
atomic(leasepath,lease)
planpath=HERE/'experiment_plan.json';plan=read(planpath);plan['updated_at_utc']=now;plan['objective_complete']=False
plan['status_authority']='Actualsole98758/native23796 LIVE featureKD4→20 startup aftermeasured55.59s one-modelhybridDE/PTdiagnostic andrealpreflight. CoherentqualifiedBEST8250/all207state; honest7000cliprewind17774vsfailed24774; source350/initialsave/nativeupdatespending. Noqualityclaim.'
plan['actual_current_training']=actual
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_measured_feature5x_samepass_startup',run_id=RUN,actual_current_run=RUN,actual_GPU_training_running=True,
    source_receipt='TSC_feature5x_BEST8250_ROOT_source_v2.json',current_frontier_source_step=8250,actual_completed_steps=8250,
    actual_new_uniform_unique_TR_clips=17774,actual_new_uniform_unique_clips=17774,current_remaining_uniform_updates=16859,
    current_configured_end_step=25109,prepared_worker='train_lora_batched_tsc_uniform_constraint_v2.py',
    prepared_boundary='tsc_uniform_retention_constraint_transition_v1.py',current_run_source350_passed=False,
    current_run_first_update_pending=True,current_run_actual_dispatch_receipt=out.name,current_failure_counter=0,
    current_actual_initial_BEST_step=8250,current_all6_selected_BEST_step=8250,actual_feature_distillation_coefficient=20,
    source_feature_distillation_coefficient=4,explicit_failed_branch_TR_clip_rewind=7000,
    active_coherent_coverage_source='BEST8250_17774_unique_NOT_failedLATEST8750_24774',
    failed_8750_branch_closure='PT_focus_integer_actual_CLOSED8750_retention_review_v1.json',
    execution_owner='/root/benchmark_design soleGPU98758, ROOT will notlaunchconcurrent')
for key in ('actual_closed_session','actual_closed_exit_code','actual_closed_tool_chunk'):item.pop(key,None)
item['recipe']='Coherent all207 BEST8250 weights+Adam/sampler/RNG/history, same14TR1PT1oldfive/LR1e-4+head1e-5/native4/cap8; ONLYfeatureKD4→20 afteractualencoder-dominantconditionalDE/PTdiagnostic. Integergates/no newgrace/counterreset; end25109. Honest7000cliprewind fromfailed8750 branch, preserveitsfullhistory.'
item['pending']=['Actualsource350same8250/full207initialBEST+latest/positiveupdates','Continuecoherentexisting253798-unique fullpass subjecttounchangedretention/resourceguards','Frozenheldout/competitivequalitygoalverification']
plan['automatic_execution_policy']['GPU_ownership']='benchmark_design owns sole98758 feature5x continuation; no concurrentROOTGPU. Actuallease/runtimeprogress authoritative.'
plan['automatic_execution_policy']['PT_focused_replay']='Actualmeasuredfeature4→20 repair fromcoherentqualifiedBEST8250, allotherloss/LR/mix/stateunchanged. Failed8750branchpreserved; explicit17774coverage rewind, nofake24774carry. Source350/firstupdatepending. Budgetonlymayautoexactcontinue; retention/resource/manualneedsreview.'
atomic(planpath,plan)
notes='Actualmeasuredretentiondiagnostic80543CLOSED0/f4a424: oneGPUmodel200fixedDE/PT,55.594s,C latestencoder+bestheads72/70,D bestencoder+latestheads70/67 vsbest70/66/latest71/71. Encoderconditionaldriftdominateshere;DEinteraction−1 meansnotuniversalcapacitycause. Soleactualfeature4→20 continuation98758/017543/native23796 launchedafterrealpreflightd17971. CoherentqualifiedBEST8250 ALL207weights+Adam/RNG/history/sampler17774unique; explicit7000cliprewind vsfailed8750/24774, failedbranchpreserved. KL40/duration2/LR1e-4/head1e-5/14TR1PT1oldfive/nativeguards/integergatesunchanged;counter0carried/no newgrace. Source350/initialsave/positiveupdatespending; end25109/16859remaining/fullpassincomplete. V15actualCLOSED0/e8bb1a, soleV16actual82097/PID14588 preserves46refs/sameproject. Evidence '+out.name
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'dispatch':record(out),'plan_Backlog_lease_updated':True}))
