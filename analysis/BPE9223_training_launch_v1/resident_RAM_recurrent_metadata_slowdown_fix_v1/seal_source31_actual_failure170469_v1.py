"""One bounded compact failure seal. Saved-prefix STAT + ONLY new suffix; no old/payload/audio reads."""
import hashlib,json,math,os,shutil,sys,time
from pathlib import Path
from datetime import datetime,timezone
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).parent;N=F.parent;OUT=N/'resident_RAM_original_interval170469_failure_actual_v1'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000023';A=N/'resident_RAM_training_actual_24';C=I/'CPU_after_periodic_DEV_000168000'
def require(v,m):
 if not v:raise ValueError(m)
def read(p,cap=4*1024**2):
 p=Path(p);require(p.stat().st_size<=cap,'Bounded metadata');return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p,cap=4*1024**2):
 p=Path(p).resolve();require(p.stat().st_size<=cap,'Only compact/source pin');b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:json.dump(v,f,indent=2);f.write('\n')
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def dead(v):
 try:
  p=psutil.Process(v['pid']);ct=p.create_time();return {'identity':v,'exact_dead':abs(ct-v['create_time'])>=.001,'observed_current_create_time':ct,'separate_numeric_OS_exit':'UNKNOWN'}
 except psutil.NoSuchProcess:return {'identity':v,'exact_dead':True,'observation':'NoSuchProcess','separate_numeric_OS_exit':'UNKNOWN'}
def guard():
 s=sample_memory(os.getpid(),'source31_failure170469_compact_suffix_only_metadata');m=s['system_memory']
 require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<536870912 and m['commit_available_bytes']>=5055306368 and m['physical_available_bytes']>=2305306368 and shutil.disk_usage('E:/').free>=100000000000,'Original metadata CPU512/fullprospective/full225+prefetch/E100GB')
 return s
def main():
 psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);initial=guard();require(not OUT.exists(),'ONE fresh failure seal, no re-audit')
 failure=read(I/'actual_failure.json');cpu=read(C/'actual_CPU_failure.json');monitor=read(I/'monitor_completion.json');worker=read(I/'private_metadata_worker_completion.json');latest=read(B/'latest_checkpoint.json');pointer=read(latest['pointer']['path']);persist=read(latest['persistence']['path'])
 require(failure['actual_optimizer_updates']==170469 and failure['last_checkpoint']==170400 and cpu['completed_updates']==2469,'Actual recorded frontiers, no native+1 assumption')
 require(latest['committed_updates']==pointer['committed_updates']==persist['optimizer_updates']==170400 and latest['full_state_verified']is True and persist['passed']is True,'Actual latest standard170400')
 require(persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215 and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256']and persist['saved_semantic_state_verified']['passed']is True and persist['journal_prefix']==pointer['journal_prefix'],'Full854207215/PRE-SAVED/semantic metadata')
 require(monitor['joined']is True and monitor['clean']is True and monitor['error']is None,'Actual originalMonitor clean, separate from CPUdecoder failure')
 child=read(A/'current_child.json');boot=read(I/'boot.json');ready=read(C/'actual_CPU_ready.json');cpu_identity=ready['producer_identity']
 identities=[boot['producer'],child['launcher'],{'pid':34044,'create_time':1792752130.7049348,'ppid':30216},cpu_identity,{'pid':24620,'create_time':1792753467.9153557,'ppid':2456}]
 deaths=[dead(v)for v in identities];require(all(v['exact_dead']for v in deaths),'All five exact production family identities dead')
 tool=F/'actual_source31_failure170469_owned2901_TOOL_v1.json';require(read(tool)['exit_code']==1,'Actual closedowned2901 TOOL1, never normal0')
 j=Path(pointer['journal_prefix']['path']);st=j.stat();offset=pointer['journal_prefix']['offset'];length=st.st_size-offset
 require(0<length<=1048576,'ONLY bounded actual unsaved suffix, no prefix/whole read')
 with j.open('rb')as f:f.seek(offset);tail=f.read(length)
 require(len(tail)==length and tail.endswith(b'\n')and j.stat().st_size==st.st_size,'Stable exact completed suffix')
 events=[json.loads(line)for line in tail.splitlines()];require([e['optimizer_updates']for e in events]==list(range(170401,170470)),'Actual69positive saved+1..native frontier suffix')
 for e in events:
  n=e['native'];require(n['update']==e['optimizer_updates']and math.isfinite(n['mean_objective'])and n['mean_objective']>0 and math.isfinite(n['global_grad_norm'])and n['global_grad_norm']>0,'Every actual suffixnativepositivefinite objective/gradient')
  require(len(n['source_ids'])==len(set(n['source_ids']))==16 and n['languages'].count('tr')==14 and e['broad_draws']==14*e['optimizer_updates']and e['replay_updates']==e['optimizer_updates']and e['cleaner_draws']==0,'Exact actual population/cursors retained')
 OUT.mkdir();copy=OUT/'preserved69_unsaved_positive_journal_suffix.jsonl'
 with copy.open('xb')as f:f.write(tail)
 stderr=C/'CPU_stderr.log';require(stderr.stat().st_size<=65536,'Bounded actualCPUstderr');stderr_copy=OUT/'actual_CPU_stderr_snapshot.log'
 with stderr_copy.open('xb')as f:f.write(stderr.read_bytes())
 require(b'Original interval beyond source unresolved'in stderr_copy.read_bytes(),'Actual original decoder2 coverage failure trace')
 paths={'actual_failure':I/'actual_failure.json','actual_CPU_failure':C/'actual_CPU_failure.json','originalMonitor_completion':I/'monitor_completion.json','private_metadata_worker_completion':I/'private_metadata_worker_completion.json','actual_CPU_stderr':stderr,'CPU_ready':C/'actual_CPU_ready.json','boot':I/'boot.json','runtime':I/'configuration.json','current_child':A/'current_child.json','latest_checkpoint':B/'latest_checkpoint.json','saved_pointer':Path(latest['pointer']['path']),'saved_persistence':Path(latest['persistence']['path']),'saved_pre_save':B/'checkpoints/update_000170400/pre_save.json','owned2901_actual_TOOL':tool}
 payload=Path(pointer['payload']['path']);pst=payload.stat();require(pst.st_size==pointer['payload']['bytes'],'Saved payload STAT only, no tensor read/hash')
 result={'kind':'ASR18_source31_original_interval170469_saved170400_actual_failure_closure_v1','observed_UTC':datetime.now(timezone.utc).isoformat(),'failure_kind':'Actual CPU original decoder interval coverage error; native release rejected erroredtransport','failure':failure,'CPU_failure':cpu,'originalMonitor':monitor,'private_metadata_worker':worker,'actual_outer_owned_session':2901,'actual_outer_TOOL_exit':1,'closed_session_never_consume_again':True,'known_exact_production_deaths':deaths,'separate_native_CPU_wrapper_numeric_OS_exits':'UNKNOWN; not inferred from outerTOOL1 or absence','saved_full_checkpoint':{'update':170400,'named_tensors':854,'weights':207,'Adam_states':215,'independent_PRE_SAVED_equality':True,'semantic_passed':True,'pointer':latest['pointer'],'persistence':latest['persistence'],'payload_STAT_only':{'path':str(payload.resolve()),'bytes':pst.st_size,'mtime_ns':pst.st_mtime_ns},'payload_read':False},'journal':{'original_STAT_only':{'path':str(j.resolve()),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns},'saved_prefix':pointer['journal_prefix'],'prefix_bytes_rehashed_or_reread':False,'new_suffix_start':offset,'new_suffix_end':st.st_size,'new_suffix_bytes':length,'new_suffix_sha256':hashlib.sha256(tail).hexdigest(),'preserved_suffix':pin(copy),'ordered_actual_updates':[170401,170469],'actual_count':len(events),'all_positivefinite_native':True,'all69_require_genuine_redo_after170400restore':True,'source_of_last_consumed_packet':events[-1]['prepared_evidence']},'evidence':{k:pin(v)for k,v in paths.items()},'CPU_stderr_snapshot':pin(stderr_copy),'CPU_exact_failed_plan_index_row':'NOT YET LOCALIZED; completed2469 alone is not a row proof; no native+1 inference','no_decoder_audio_frontend_model_GPU_or_new_sampler_operation':True,'no_retry_or_guard_source_configuration_change':True,'initial_guard':initial,'final_guard':guard(),'sealer_identity':who(psutil.Process()),'sealer_launcher_identity':who(psutil.Process().parent())}
 out=OUT/'actual_failure_closure.json';write(out,result)
 print(json.dumps({'closure':pin(out),'preserved_suffix':pin(copy),'CPU_stderr_snapshot':pin(stderr_copy),'native_actual':170469,'saved':170400,'suffix_count':len(events),'suffix_offset':offset,'suffix_end':st.st_size,'all5_exact_dead':True}),flush=True)
if __name__=='__main__':main()
