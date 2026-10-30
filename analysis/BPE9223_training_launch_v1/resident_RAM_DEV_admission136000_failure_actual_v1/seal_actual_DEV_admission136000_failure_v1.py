"""Seal one actual DEV-admission failure; no retry, tensor payload or model work."""
import datetime, hashlib, json, os, shutil, time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'); B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01'); A=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_training_actual_13'); I=B/'invocations/owner_000012'; D=Path(__file__).parent
started=time.monotonic(); psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(ok,message):
    if not ok: raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
    p=Path(p); raw=p.read_bytes(); return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,value):
    require(not p.exists(),'Preserve existing evidence '+str(p)); p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def snapshot(p,name):
    out=D/name; require(not out.exists(),'One snapshot'); shutil.copyfile(p,out); return pin(out)
require(shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original E reserve')
failure=read(I/'actual_failure.json'); progress=read(B/'progress/latest.json'); latest=read(B/'latest_checkpoint.json'); pointer_path=Path(latest['pointer']['path']); pointer=read(pointer_path); persistence_path=Path(latest['persistence']['path']); persistence=read(persistence_path)
expected="ValueError('Actual allocation increment plus full extra prefetch reservation and unchanged commit headroom')"
require(failure['error']==expected and failure['actual_optimizer_updates']==failure['last_checkpoint']==136000,'Exact new prospective DEV failure at coherent136000')
require('evaluate_coherent_DEV' in failure['traceback'] and "admit('resident_optimizer_idle_serial350_DEV',2100000000)" in failure['traceback'],'Exact DEV allocation gate, not EOF')
require(progress['actual_optimizer_updates']==pointer['committed_updates']==latest['committed_updates']==136000 and latest['full_state_verified'] is True,'All committed live updates saved')
require(pointer['journal_prefix']==progress['journal_prefix']==persistence['journal_prefix'],'Same exact current saved journal prefix')
require(persistence['passed'] and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854 and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'] and persistence['saved_semantic_state_verified']['passed'],'Original full persistence and semantic proof metadata')
journal=Path(pointer['journal_prefix']['path']); expected_bytes=pointer['journal_prefix']['offset']; require(journal.stat().st_size==expected_bytes,'No unsaved journal suffix')
h=hashlib.sha256(); count=0
with journal.open('rb') as source:
    while True:
        block=source.read(1024*1024)
        if not block:break
        h.update(block); count+=len(block)
        require(time.monotonic()-started<180 and psutil.Process().memory_info().private<512*1024**2,'Bounded metadata observation')
require(count==expected_bytes and h.hexdigest()==pointer['journal_prefix']['sha256'],'One actual full journal pass equals saved prefix; zero tail')
identities=[('native',27232,1792655703.93878),('native_interpreter_launcher',25192,1792655703.9323223),('wrapper',35268,None),('CPU',548,1792655894.7857172),('CPU_interpreter_launcher',35232,None)]
deaths=[]
for role,pid,ctime in identities:
    try:
        process=psutil.Process(pid); current=process.create_time(); absent=False
    except psutil.NoSuchProcess:current=None;absent=True
    exact_dead=absent or (ctime is not None and abs(current-ctime)>.001)
    require(exact_dead,'Recorded identity must be dead '+role)
    deaths.append({'role':role,'pid':pid,'recorded_create_time':ctime,'pid_actually_absent':absent,'current_create_time_if_reused':current,'recorded_identity_dead':exact_dead,'historical_create_time_recorded':ctime is not None})
lease=read(P/'work/gpu_training_owner.json'); require(lease['status']=='production_resident_RAM_process_closed' and lease['actual_GPU_running'] is False and lease['committed_updates']==136000,'Actual closed GPU lease')
monitor=read(I/'monitor_completion.json'); require(monitor['joined'] and monitor['clean'] and monitor['error'] is None,'Actual joined clean monitor')
periodic=read(B/'periodic_DEV.json'); require(all(e['optimizer_updates']<136000 for e in periodic['completed_evaluations']),'No claimed136000 evaluation')
inputs={'actual_failure':pin(I/'actual_failure.json'),'current_progress_snapshot':snapshot(B/'progress/latest.json','actual_last_progress.json'),'latest_checkpoint_snapshot':snapshot(B/'latest_checkpoint.json','actual_latest_checkpoint.json'),'pointer':pin(pointer_path),'persistence':pin(persistence_path),'checkpoint_metadata':pin(pointer['checkpoint']['path']),'CPU_ready':pin(I/'CPU/actual_CPU_ready.json'),'CPU_stderr':pin(I/'CPU/CPU_stderr.log'),'native_log':pin(A/'native_000012.log'),'current_child':pin(A/'current_child.json'),'monitor_completion':pin(I/'monitor_completion.json'),'private_metadata_worker_completion':pin(I/'private_metadata_worker_completion.json'),'lease_snapshot':snapshot(P/'work/gpu_training_owner.json','actual_closed_lease.json'),'periodic_state_snapshot':snapshot(B/'periodic_DEV.json','actual_pending136000_periodic_state.json'),'configuration':pin(I/'configuration.json'),'driver':pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v16.py'),'CPU_source':pin(P/'outputs/parakeet_turkish/fastmss_resident_feature_stream_v7.py')}
result={'kind':'ASR18_actual_saved136000_resident_DEV_prospective_admission_failure_closure_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owned_session':64681,'actual_outer_TOOL_exit_code':1,'actual_TOOL_chunk':'88d7ec','native_numeric_OS_exit_code':None,'native_wait_known_nonzero':True,'wrapper_numeric_OS_exit_code_separately_recorded':False,'wrapper_uncaught_ValueError_observed':True,'CPU_numeric_OS_exit_code':None,'native_and_CPU_numeric_unknown_not_inferred':True,'actual_failure':failure,'actual_committed_and_fully_saved_updates':136000,'unsaved_positive_tail_updates':0,'whole_journal_verified_once':{'path':str(journal),'bytes':count,'sha256':h.hexdigest()},'saved_prefix_equals_whole_journal':True,'all_recorded_identities_dead':deaths,'monitor_completion':monitor,'failure_phase':'Prospective 2100000000-byte resident optimizer-idle serial350 DEV admission before evaluator/model calls','required_commit_available_before_imminent_allocation':2100000000+268435456+2000000000,'failing_historical_commit_available_bytes':None,'failing_sample_not_written_before_require':True,'no_inferred_freed_memory_or_OOM_claim':True,'pending136000_DEV_not_completed':True,'pending136000_DEV_must_precede136001':True,'actual_full_state_verification_metadata':{'named_tensors':854,'weights':207,'Adam_states':215,'PRE_SAVED_equal':True,'saved_semantic_passed':True},'inputs':inputs,'no_tensor_payload_reread':True,'no_retry_or_GPU_probe':True,'planned_target':401072,'administrative_update_stop':None}
out=D/'actual_failure_closure.json';write(out,result);print(json.dumps(pin(out)),flush=True)
