"""Compact actual helper-identity failure closure; no retry or tensor payload read."""
import datetime,hashlib,json,shutil,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000013';A=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_training_actual_14');D=Path(__file__).parent;start=time.monotonic()
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def req(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def snapshot(p,name):
 q=D/name;req(not q.exists(),'Preserve '+name);shutil.copyfile(p,q);return pin(q)
req(shutil.disk_usage('E:/gpu_training_scratchpad').free>=100000000000,'Original E reserve')
failure=read(I/'actual_failure.json');progress=read(B/'progress/latest.json');latest=read(B/'latest_checkpoint.json');pointer_path=Path(latest['pointer']['path']);pointer=read(pointer_path);pvpath=Path(latest['persistence']['path']);pv=read(pvpath)
req(failure['error']=="ValueError('Keep the exact previously submitted private plan')" and 'close_private_preparation_for_DEV' in failure['traceback'] and failure['actual_optimizer_updates']==failure['last_checkpoint']==140000,'Exact new failed helper identity check')
req(progress['actual_optimizer_updates']==latest['committed_updates']==pointer['committed_updates']==140000 and latest['full_state_verified'],'All committed updates saved140000')
req(progress['journal_prefix']==pointer['journal_prefix']==pv['journal_prefix'],'Exact current saved journal prefix')
req(pv['passed'] and pv['named_tensors_verified']==854 and pv['weights_verified']==207 and pv['Adam_states_verified']==215 and pv['independent_pre_save_state_sha256']==pv['saved_state_sha256'] and pv['saved_semantic_state_verified']['passed'],'Full854/207/215/PRE-SAVED/semantic metadata')
journal=Path(pointer['journal_prefix']['path']);req(journal.stat().st_size==pointer['journal_prefix']['offset'],'Zero unsaved journal suffix');h=hashlib.sha256();count=0
with journal.open('rb')as source:
 while True:
  block=source.read(1024*1024)
  if not block:break
  h.update(block);count+=len(block);req(time.monotonic()-start<180 and psutil.Process().memory_info().private<512*1024**2,'Finite metadata observation')
req(count==284584907 and h.hexdigest()==pointer['journal_prefix']['sha256']=='caeb1f6fe40c867028bff40b0aa7f838a7a9d185872855213adc4d2a75d697af','One real journal pass equals whole saved prefix')
identities=[('native',41200,1792660196.3122573),('native_interpreter_launcher',38908,1792660196.3051417),('wrapper',31740,None),('CPU',26908,1792660472.4757633),('CPU_interpreter_launcher',33564,None)];deaths=[]
for role,pid,ctime in identities:
 try:current=psutil.Process(pid).create_time();absent=False
 except psutil.NoSuchProcess:current=None;absent=True
 dead=absent or(ctime is not None and abs(current-ctime)>.001);req(dead,'Actual old identity dead '+role)
 deaths.append({'role':role,'pid':pid,'recorded_create_time':ctime,'pid_actually_absent':absent,'current_create_time_if_reused':current,'recorded_identity_dead':True,'historical_create_time_recorded':ctime is not None})
monitor=read(I/'monitor_completion.json');req(monitor['joined'] and monitor['clean'] and monitor['error'] is None,'Actual joined clean monitor')
periodic=read(B/'periodic_DEV.json');req(all(x['optimizer_updates']<140000 for x in periodic['completed_evaluations']),'No140k DEV claim')
current=I/'plans/plan_000200.json';future=I/'plans/plan_000201.json';cp=read(current);fp=read(future);req(cp['initial_committed_updates']==139980 and fp['initial_committed_updates']==140000 and len(cp['updates'])==len(fp['updates'])==20,'Actual ordered current and queued future plan files')
lease=read(P/'work/gpu_training_owner.json');req(lease['status']=='production_resident_RAM_process_closed' and lease['actual_GPU_running'] is False,'Actual idle closed lease')
inputs={'actual_failure':pin(I/'actual_failure.json'),'last_progress_snapshot':snapshot(B/'progress/latest.json','actual_last_progress.json'),'latest_checkpoint_snapshot':snapshot(B/'latest_checkpoint.json','actual_latest_checkpoint.json'),'pointer':pin(pointer_path),'persistence':pin(pvpath),'checkpoint_metadata':pin(pointer['checkpoint']['path']),'monitor_completion':pin(I/'monitor_completion.json'),'native_log':pin(A/'native_000013.log'),'CPU_stderr':pin(I/'CPU/CPU_stderr.log'),'CPU_ready':pin(I/'CPU/actual_CPU_ready.json'),'current_child':pin(A/'current_child.json'),'CPU_helper_source':pin(P/'outputs/parakeet_turkish/asr18_resident_periodic_CPU_quiescence_v1.py'),'driver_source':pin(P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v17.py'),'effective_configuration':pin(I/'configuration.json'),'private_metadata_worker_completion':pin(I/'private_metadata_worker_completion.json'),'lease_snapshot':snapshot(P/'work/gpu_training_owner.json','actual_closed_lease.json'),'periodic_state_snapshot':snapshot(B/'periodic_DEV.json','actual_pending140000_periodic_state.json'),'last_consumed_plan':pin(current),'next_unconsumed_plan':pin(future)}
result={'kind':'ASR18_actual_saved140000_periodic_CPU_quiescence_plan_identity_failure_closure_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owned_session':86052,'actual_outer_TOOL_exit_code':1,'actual_TOOL_chunk':'d34ac7','native_numeric_OS_exit_code':None,'native_wait_known_nonzero':True,'wrapper_numeric_OS_exit_code_separately_recorded':False,'wrapper_uncaught_ValueError_observed':True,'CPU_numeric_OS_exit_code':None,'actual_failure':failure,'all_committed_and_fully_saved_updates':140000,'unsaved_positive_tail_updates':0,'whole_journal_verified_once':{'path':str(journal),'bytes':count,'sha256':h.hexdigest()},'saved_prefix_equals_whole_journal':True,'all_recorded_identities_dead':deaths,'monitor_completion':monitor,'pending140000_DEV_not_completed':True,'pending140000_DEV_must_precede140001':True,'failed_before_private_packet_drain_and_CPU_close':True,'real_periodic_CPU_quiescence_wait0_not_claimed':True,'failing_compound_assertion':'record(plan_pin[path]) == plan_pin and read_json(plan_pin[path]) == queued_plan','which_operand_failed_not_recorded':True,'unavailable_live_queued_Python_object_not_invented':True,'no_resource_admission_denial_record':not(I/'actual_prospective_admission_denial.json').exists(),'inputs':inputs,'no_tensor_payload_reread':True,'no_retry_or_GPU_probe':True,'full_target':401072,'administrative_update_stop':None}
out=D/'actual_failure_closure.json';req(not out.exists(),'One actual closure');out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(pin(out)),flush=True)
