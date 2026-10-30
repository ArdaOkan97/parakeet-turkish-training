"""Record ROOT-owned dispatch/closure and publish honest current plan status."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/parakeet_turkish'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rec(p):
    with p.open('rb') as s:h=hashlib.file_digest(s,'sha256').hexdigest()
    return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h}
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--owned-session',type=int,required=True);p.add_argument('--launch-chunk',required=True)
p.add_argument('--previous-run-id',required=True);p.add_argument('--previous-owned-session',type=int,required=True);p.add_argument('--previous-closed-chunk',required=True)
p.add_argument('--receipt-output',type=Path,required=True);a=p.parse_args()
old=OUT/'runs'/(a.previous_run_id+'_loop_01');new=OUT/'runs'/(a.run_id+'_loop_01')
closed=read(old/'loop_status.json');completion_path=old/(a.previous_run_id+'_native_backend.completion.json'); done=read(completion_path)
assert closed['status']=='failed' and closed['child_exit_code']==1 and done['status']=='failed'
state=read(new/'loop_status.json');ready=read(new/(a.run_id+'_native_backend.json'));plan=read(new/'plan.json');identity=read(new/(a.run_id+'_identity.json'))
assert state['status']=='running' and ready['pid']==identity['pid'] and identity['ppid']==state['owned_launcher_pid']
assert plan['controller_sha256']==ready['launcher_sha256']==rec(OUT/'run_TSC_uniform_full_v1.py')['sha256']
assert ready['trainer_sha256']==rec(OUT/'train_lora_batched_tsc_uniform_full_v1.py')['sha256']
assert plan['worker_arguments_sha256']==ready['worker_arguments_sha256']==identity['worker_arguments_sha256']
updates=max(0,state.get('published_step',6500)-6500)
receipt={'kind':'ROOT_owned_full_uniform_current_dispatch_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'previous_failed_controller':rec(old/'loop_status.json'),'previous_failed_native':rec(completion_path),
    'previous_actual_ROOT_session':a.previous_owned_session,'previous_actual_ROOT_exit_code':1,'previous_actual_ROOT_exit_tool_chunk':a.previous_closed_chunk,
    'previous_failure':done['error'],'previous_updates':max(0,closed.get('published_step',6500)-6500),
    'actual_ROOT_session':a.owned_session,'actual_launch_tool_chunk':a.launch_chunk,'run_id':a.run_id,
    'owned_launcher_pid':state['owned_launcher_pid'],'actual_native_pid':identity['pid'],
    'plan':rec(new/'plan.json'),'native_ready':rec(new/(a.run_id+'_native_backend.json')),
    'source_hashes_consistent':True,'actual_new_optimizer_updates_published':updates,
    'actual_first_update_observed':updates>0,'complete_uniform_pass_claimed':False,'whole_peak_fit_claimed':False,
    'same_Trackio_project':'parakeet-turkish','sole_mirror_session':37748,'sole_mirror_pid':30716,
    'TEST_used':False,'objective_complete':False}
assert not a.receipt_output.exists();a.receipt_output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
path=OUT/'experiment_plan.json';science=read(path);item=next(x for x in science['ordered_experiments'] if x['id']=='04')
item.update(status='running' if updates else 'full_uniform_actual_startup',run_id=a.run_id,
    source_receipt='TSC_uniform_full_PT04_recovery_ROOT_source_v1.json',actual_ROOT_controller_session=a.owned_session,
    actual_ROOT_launch_tool_chunk=a.launch_chunk,actual_start_receipt=a.receipt_output.name,
    actual_GPU_child_started=True,actual_GPU_training_started=updates>0,actual_new_uniform_updates=updates,
    frozen_controller_sha256=plan['controller_sha256'],frozen_worker_sha256=ready['trainer_sha256'],
    previous_full_attempt=a.previous_run_id,previous_full_attempt_actual_CLOSED_exit_code=1,
    previous_full_attempt_updates=receipt['previous_updates'],startup_tracing_keyword_collision_repaired=True,
    source_initial350_independent_evidence='TSC_uniform_full_PT02_initial350_independent_saved_evidence_v1.json')
science['status_authority']='Actual controller/native closure and complete DEV determine status. Current '+a.run_id+' has '+str(updates)+' published new optimizer updates at this receipt; read fresh actual history/completion. Earlier attempts closed before optimization after source350 passed. Full pass and final quality remain incomplete.'
science['updated_at_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(science,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'receipt':rec(a.receipt_output),'actual_new_updates':updates,'native_pid':identity['pid']}))
