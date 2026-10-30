"""Seal the consumed diagnostic failure; no counter/profile analysis or retry."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent;B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000025'
SCOPE=F/'ROOT_source36_post_refresh_readiness_fixed60_exact_ONCE_execution_scope_v1.json'
EXTERNAL=F/'existing_source36_thread_CPU_external_result_v1.json';OUT=F/'existing_source36_thread_CPU_actual_01'
TOOL=F/'existing_source36_thread_CPU_actual01_TOOL_result_v1.json';COMPLETE=F/'existing_source36_thread_CPU_actual01_failed_completion_v1.json'
ME=psutil.Process();ME.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);BEGIN=time.monotonic()
def require(v,m):
 if not v:raise ValueError(m)
def pin(p,cap=8*1024**2):
 p=Path(p).resolve();require(p.stat().st_size<=cap,'Bounded admitted artifact/source only');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):
 require(Path(p).stat().st_size<=8*1024**2,'Compact JSON only');return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def exact(v):
 try:return abs(psutil.Process(v['pid']).create_time()-v['create_time'])<.001
 except psutil.NoSuchProcess:return False
def write(p,v):
 with p.open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def guard(admission=False):
 samples=[sample_memory(p.pid,'source36_failed_diagnostic_metadata_closure')for p in (ME,ME.parent())];m=samples[-1]['system_memory']
 require(all(s['paired_sampling_within_one_second']for s in samples) and sum(s['worker_private_memory']['private_bytes']for s in samples)<=536870912 and time.monotonic()-BEGIN<120 and m['commit_available_bytes']>=(5055306368 if admission else 2268435456) and m['physical_available_bytes']>=(2305306368 if admission else 1768435456) and shutil.disk_usage('E:/').free>=100000000000,'Original512MiB/fullprospective512/full225/FULLprefetch/floors/E100GB/120s metadata guard')
 return samples
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')) and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'E process-local CPU-only metadata environment')
initial=guard(True);review=read(SCOPE);r=read(EXTERNAL)
require(pin(SCOPE)['sha256']=='f27aaba811925ad1ce5a2668357eb77c916859645747392c39cecc0ed7415bbb' and pin(EXTERNAL)['sha256']=='796927ad9b794a2cd6fc0ee2ca3cd753597d384af10650751a584e73a23948f4','Exact consumed scope and failed external result')
require(r['passed'] is False and r['Popen_returncode'] is None and r['inputs_before_after_exact'] is True,'Honest actual failed diagnostic; child numeric return UNRECORDED')
owned={}
for v in [r['external_identity'],r['launcher_identity'],*r['recorded_identities'],*r['required_external_cleanup']]:owned[(v['pid'],v['create_time'])]=v
deaths=[]
for v in owned.values():
 require(not exact(v),'Wait only for actual known owned metadata identity death; never kill');deaths.append({**v,'exact_dead':True})
require(len(deaths)==6,'All six actual job identities including external interpreter launcher')
artifacts={}
caps={'thread_CPU_counters.jsonl':33554432,'counter_poll_brackets.jsonl':1048576,'counter_poll_preconditions.jsonl':1048576,'profiler.stdout.log':8388608,'profiler.stderr.log':4194304,'profile.speedscope.json':67108864,'report.json':8388608,'trigger_existing_last5_timing_rows.jsonl':65536}
for name,cap in caps.items():
 p=OUT/name
 if p.exists():
  st=p.stat();require(st.st_size<=cap,'Original bounded partial-artifact cap')
  artifacts[name]={'path':str(p.resolve()),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':'UNRECORDED: STAT only; no new counter/profile/log/timing content read by closure','STAT_only':True}
 else:artifacts[name]={'path':str(p.resolve()),'absent':True}
 guard()
protected=[]
for v in review['protected_production_identities']:
 require(exact(v),'Same exact protected production/reader cohort must be alive');protected.append({**v,'exact_alive':True})
progress=read(B/'progress/latest.json')
require(progress['producer']==review['target_identity'] and progress['source']['sha256']=='d8b9797195bf5dd1981e3a4043f86fba369fd4e7706d9535038e7f7b788e1d5a' and not progress['optimizer_poisoned'] and not progress['schedule_poisoned'] and progress['planned_native_updates']==401072 and progress['administrative_update_stop'] is None and not(I/'actual_failure.json').exists(),'Diagnostic failure is not production failure; same42997/source36 healthy')
write(TOOL,{'kind':'ASR18_source36_thread_CPU_actual01_failed_TOOL_result_v1','tool':'functions.exec/tools.exec_command/write_stdin','diagnostic_session':61885,'initial_chunk':'4225d7','final_chunk':'60dfb9','actual_TOOL_exit_code':1,'initial_external_scope':pin(SCOPE),'external_result':pin(EXTERNAL),'actual_external_Popen_returncode':None,'separate_observer_and_profiler_numeric_OS_exits':'UNRECORDED','actual_guard_error':r['external_guard_error'],'actual_required_owned_cleanup':r['required_external_cleanup'],'production_session42997_consumed':False,'no_retry':True})
result={'kind':'ASR18_source36_thread_CPU_actual01_failed_diagnostic_completion_v1','failure_evidence_sealed':True,'diagnostic_passed':False,'complete_attribution_available':False,'scope':pin(SCOPE),'actual_TOOL_result':pin(TOOL),'external_result':pin(EXTERNAL),'actual_external_wall_seconds_after_cleanup':r['external_wall_seconds'],'actual_sampled_family_private_peak_bytes':r['sampled_family_private_peak_bytes'],'actual_guard_error':r['external_guard_error'],'guard_elapsed_at_raise_separate_scalar':'UNRECORDED','source_guard_condition':'family_total<=536870912 AND elapsed<120; saved peak is47132672 and after-cleanup wall120.453; no reason for profiler noncompletion is inferred','actual_Popen_returncode':None,'required_owned_cleanup':r['required_external_cleanup'],'cleanup_error':r.get('cleanup_error'),'all_six_recorded_job_identities_exact_dead':deaths,'partial_artifact_STAT_only_without_counter_profile_log_timing_content_read':artifacts,'actual_input_count':len(r['input_pins_before']),'actual_input_pins_before_after_exact':r['inputs_before_after_exact'],'all_owned_query_handles_closed':'UNKNOWN: inner final report absent; never inferred from process death','retained_query_lifetime_final_sample':'UNRECORDED: no completed inner report','observer_or_profiler_final_numeric_OS_exit_codes':'UNRECORDED','profile_role_mapping':'UNAVAILABLE: no completed profile/report','protected_seven_exact_alive':protected,'actual_compact_production_progress':progress,'metadata_identity':who(ME),'metadata_interpreter_launcher_identity':who(ME.parent()),'metadata_pair_alive_at_write_require_independent_post_exit_exact_death_check':True,'metadata_source':pin(__file__),'initial_metadata_guard':initial,'final_metadata_guard':guard(),'no_helper_profiler_counter_import_or_execution_GC_GPU_model_audio_state_source_guard_LR_reader_or_production_operation':True,'no_diagnostic_Backlog_or_automatic_retry':True,'human_speed_fix_open':True}
write(COMPLETE,result);print(json.dumps({'completion':pin(COMPLETE),'actual_TOOL':pin(TOOL),'known_job_deaths':deaths,'metadata_identity':result['metadata_identity'],'metadata_launcher_identity':result['metadata_interpreter_launcher_identity'],'actual_progress':progress['actual_optimizer_updates'],'saved':progress['last_saved_updates']}),flush=True)
