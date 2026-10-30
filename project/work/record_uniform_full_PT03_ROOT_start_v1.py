from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/parakeet_turkish'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rec(p):
    with p.open('rb') as s: h=hashlib.file_digest(s,'sha256').hexdigest()
    return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h}
old=OUT/'runs/adaptive_20261012_TR_uniform_full_PT_02_loop_01'; new=OUT/'runs/adaptive_20261012_TR_uniform_full_PT_03_loop_01'
closed=read(old/'loop_status.json'); done=read(old/'adaptive_20261012_TR_uniform_full_PT_02_native_backend.completion.json')
state=read(new/'loop_status.json'); ready=read(new/'adaptive_20261012_TR_uniform_full_PT_03_native_backend.json'); plan=read(new/'plan.json')
assert closed['child_exit_code']==1 and done['status']=='failed' and done['sources_unchanged']
assert state['status']=='running' and state['owned_launcher_pid']==31716 and ready['pid']==5324
assert plan['controller_sha256']==ready['launcher_sha256']==rec(OUT/'run_TSC_uniform_full_v1.py')['sha256']
assert ready['trainer_sha256']==rec(OUT/'train_lora_batched_tsc_uniform_full_v1.py')['sha256']
eq=OUT/'runs/adaptive_20261012_TR_uniform_full_PT_02/uniform_full_initial_source_equivalence.json'
assert read(eq)['passed'] and read(eq)['all_raw_transcripts_and_emitted_paths_exact'] and read(eq)['Portuguese50_raw_paths_and_ASR_exact']
receipt={'kind':'ROOT_actual_PT03_after_startup_name_resolution_repair_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'failed_attempt_status':rec(old/'loop_status.json'),'failed_native_completion':rec(old/'adaptive_20261012_TR_uniform_full_PT_02_native_backend.completion.json'),
    'failed_actual_ROOT_session':36363,'failed_actual_ROOT_exit_code':1,'failed_actual_ROOT_exit_tool_chunk':'06883f','failed_new_optimizer_updates':0,
    'actual_initial350_equivalence':rec(eq),'actual_all207_Adam_restore_reached':True,'actual_initial_state_serialization_completed':False,
    'cause':'Local record training-history variable shadows imported file-record function before configuration publication.',
    'repair_proof':rec(OUT/'TSC_uniform_full_record_shadow_ROOT_verification_v1.json'),
    'prior_sources_archived':str(ROOT/'work/closed-sources/uniform_full_PT02_dispatch_v1'),
    'sole_new_ROOT_session':92196,'actual_launch_tool_chunk':'953911','owned_launcher_pid':31716,'actual_native_pid':5324,
    'new_plan':rec(new/'plan.json'),'new_native_ready':rec(new/'adaptive_20261012_TR_uniform_full_PT_03_native_backend.json'),
    'startup_status_only':True,'training_memory_peak_fit_claimed':False,'objective_complete':False,'TEST_used':False,
    'same_Trackio_project':'parakeet-turkish','sole_Trackio_mirror_session':37748,'sole_Trackio_mirror_pid':30716}
dest=OUT/'TSC_uniform_full_PT03_actual_ROOT_start_v1.json'; assert not dest.exists(); dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
path=OUT/'experiment_plan.json'; p=read(path); item=next(x for x in p['ordered_experiments'] if x['id']=='04')
item.update(status='full_uniform_actual_startup',run_id=plan['run_id'],source_receipt='TSC_uniform_full_PT03_recovery_ROOT_source_v1.json',
    actual_ROOT_controller_session=92196,actual_ROOT_launch_tool_chunk='953911',actual_start_receipt=dest.name,
    actual_GPU_child_started=True,actual_GPU_training_started=False,actual_new_uniform_updates=0,
    frozen_controller_sha256=plan['controller_sha256'],frozen_worker_sha256=ready['trainer_sha256'],
    previous_full_attempt='adaptive_20261012_TR_uniform_full_PT_02',previous_full_attempt_actual_CLOSED_exit_code=1,
    previous_full_attempt_updates=0,previous_initial350_equivalence_passed=True,
    startup_name_resolution_bug_repaired=True,actual_initial_state_roundtrip_claimed=False)
p['status_authority']='Actual controller/native closure and complete DEV are authoritative. Current PT03 sole GPU child is in startup at receipt time; original350 equality has actually passed in PT02. Full updates and fit remain pending. Always inspect fresh current history/completion.'
p['updated_at_utc']=datetime.now(timezone.utc).isoformat(); path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'receipt':rec(dest),'run_id':plan['run_id'],'actual_optimization_updates':0}))
