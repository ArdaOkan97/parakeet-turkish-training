"""Publish actual owned dispatch, preserving separate prior closure facts."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_majority_PT_focus_integer_01';LOOP=HERE/'runs'/(RUN+'_loop_01')
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def record(path):
    raw=path.read_bytes();return {'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
state=read(LOOP/'loop_status.json');assert state['status']=='running' and state['owned_launcher_pid']==17172
now=datetime.now(timezone.utc).isoformat()
actual={'kind':'actual_PT_focus_integer_gate_samepass_dispatch_v1','run_id':RUN,
    'status':'actual_owned_running_source350_pending','actual_controller_session_id':43870,
    'actual_launch_tool_chunk':'6016e6','actual_owned_native_launcher_pid':17172,'actual_GPU_running':True,
    'source_step':8250,'completed_steps':8250,'configured_end_step':25109,'remaining_updates_at_start':16859,
    'new_uniform_unique_TR_clips_at_start':17774,'source350_and_first_optimizer_pending':True,
    'head_lr':1e-5,'lora_lr':1e-4,'PT_focus_origin_step':8000,'PT_focus_updates':250,
    'quota_boundary_step':7621,'uniform_origin_step':6500,'lower_LoRA_boundary_step':7750,
    'source_receipt':record(HERE/'TSC_PT_focus8250_integer_ROOT_source_v1.json'),
    'preflight':record(HERE/'PT_focus8250_integer_actual_input_preflight_v1.json'),
    'worker':record(HERE/'train_lora_batched_tsc_uniform_majority_ptfocus_v3.py'),
    'controller':record(HERE/'run_TSC_uniform_majority_ptfocus_v3.py'),
    'objective_complete':False,'observed_at_utc':now}
out=HERE/'PT_focus8250_integer_actual_owned_dispatch_v1.json'
with out.open('x',encoding='utf-8') as stream:json.dump(actual,stream,indent=2)
path=HERE/'experiment_plan.json';plan=read(path);plan['updated_at_utc']=now;plan['objective_complete']=False
plan['status_authority']='Actualownedintegergate samepass43870 LIVE source350/native17172 afterrealinputpreflight. Closed8250source exactall6qualified; historicalfalsegate records preserved. Full207initialsave/positiveupdates pending.'
plan['actual_current_training']=actual
item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
item.update(status='actual_integer_gate_corrected_samepass_startup',run_id=RUN,actual_current_run=RUN,
    source_receipt='TSC_PT_focus8250_integer_ROOT_source_v1.json',current_frontier_source_step=8250,
    actual_completed_steps=8250,actual_GPU_training_running=True,actual_PT_focus_updates=250,
    actual_new_uniform_unique_TR_clips=17774,current_remaining_uniform_updates=16859,
    current_configured_end_step=25109,prepared_worker='train_lora_batched_tsc_uniform_majority_ptfocus_v3.py',
    prepared_boundary='tsc_uniform_majority_ptfocus_integer_transition_v1.py',current_run_source350_passed=False,
    current_run_first_update_pending=True,current_run_actual_dispatch_receipt=out.name,
    current_failure_counter_before_accounting=5,current_failure_counter_after_qualified_accounting_pending=0,
    source_all6_qualified_by_exact_counts=True,execution_owner='/root/benchmark_design soleGPUactual43870')
for key in ('actual_closed_session','actual_closed_exit_code','actual_closed_tool_chunk'):item.pop(key,None)
item['pending']=['Actualsource350same8250/full207initialBEST+latest/firstnativeupdate',
    'Continueexisting253798unique fullpass without reshuffle/moments/grace reset',
    'Strictall6DEV250 and unchangedresourceguards']
plan['automatic_execution_policy']['GPU_ownership']='benchmark_design owns43870 soleGPU integergatecontinuation. ROOT mustnotlaunchconcurrent; actuallease/source/runtimefacts authoritative.'
plan['automatic_execution_policy']['PT_focused_replay']='SamePTmix/state from8250 with exactintegeroriginalthresholds, noepsilon/relaxation. Historicalfalsegate5/None immutable; newinitial0/BEST8250 onlyafteractual350. Source350/initialsave/positiveupdates pending. Budgetonly mayautoexactcontinue; retention/resource/manualneedreview.'
temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)
notes='ActualsoleGPUintegergate samepass launched43870/6016e6 native17172 afterrealinputpreflight866a40. Newrun '+RUN+' fromCLOSED8250/all207 Adam/RNG/history/sampler17774unique,end25109/16859remaining. Exactintegerlimits69/52/30/74/70/66 unchanged; source8250all6countspass. Historicalfalsegate5/None immutable; new0/BEST8250 projection requiresactual350. Same14TR1PT1oldfive/LRs1e-4+1e-5/native4/cap8,no renewedgrace/reset. Source350/firstupdate pending; noquality/fullcoverageclaim. Leaseexclusivebenchmark_design; V14sameproject preserved.'
for task in ('ASR-15','ASR-15.1'):
    subprocess.run([str(HERE/'tools/backlog.exe'),'task','edit',task,'--status','In Progress','--append-notes',notes],
        cwd=HERE,check=True,capture_output=True,encoding='utf-8')
subprocess.run([str(ROOT/'work/training-env/Scripts/python.exe'),str(HERE/'refresh_backlog.py')],
    cwd=ROOT,check=True,capture_output=True,encoding='utf-8')
print(json.dumps({'dispatch':record(out),'plan_Backlog_refresh_updated':True}))
