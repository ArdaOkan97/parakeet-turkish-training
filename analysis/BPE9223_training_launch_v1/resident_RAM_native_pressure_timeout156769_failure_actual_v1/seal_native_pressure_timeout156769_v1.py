"""Routine compact failure seal: only the169 unsaved NEW journal18 suffix, no prefix/whole/payload/audio."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');F=Path(__file__).parent
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01');I=B/'invocations/owner_000018';A=N/'resident_RAM_training_actual_19'
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def guard():
 s=sample_memory(os.getpid(),'source24_timeout156769_routine_metadata_seal');m=s['system_memory']
 require(s['paired_sampling_within_one_second']and s['worker_private_memory']['private_bytes']<512*1024**2
  and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456
  and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original finite metadata512MiB/180s/full256MiB/headroom/E100GB')
 return s
initial=guard();m=initial['system_memory']
require(m['commit_available_bytes']>=2000000000+268435456+512*1024**2 and m['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB metadata allowance')
me=psutil.Process();parent=me.parent()
identity={'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()};parent_identity={'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}
write(F/'actual_metadata_entry.json',{'observer':identity,'launcher':parent_identity,'guard_initial':initial})
failure=read(I/'actual_failure.json');denial=failure['native_pressure_denial'];memory=denial['last_denied_sample']['system_memory'];monitor=read(I/'monitor_completion.json')
require(failure['producer']=={'pid':42304,'create_time':1792716760.1698885,'ppid':38476}and failure['actual_optimizer_updates']==156769 and failure['last_checkpoint']==156600,'Exact final156769/saved156600')
require(failure['error']=="TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')"and 'line 910, in main' in failure['traceback'],'Actual FIRST pre-take pressure gate, not post-readiness gate')
require(denial['observed_wait_seconds']==300 and denial['maximum_wait_seconds']==300 and denial['observations']==600
 and denial['required_commit_available_bytes']==4768435456 and denial['required_physical_available_bytes']==1768435456
 and memory['commit_available_bytes']==4669493248 and memory['physical_available_bytes']==11246845952
 and denial['no_packet_propose_native_or_authoritative_operation_admitted']is True,'Exact last denial only; no inferred historical bytes')
require(monitor['joined']and monitor['clean']and monitor['error']is None and monitor['samples']==5817 and monitor['maximum_sampled_private_bytes']==7032066048,'Original Monitor clean, distinct scheduling buffer timeout')
require((I/'CPU/CPU_stderr.log').stat().st_size==0,'CPU stderr actually empty, not CPU-specific failure claim')
family_path=A/'actual_source24_live_runtime_family_after156600.json';family=read(family_path);deaths=[]
for who in family:
 try:
  proc=psutil.Process(who['pid']);now=proc.create_time();same=abs(now-who['create_time'])<.001
  observed={'pid_present':True,'current_create_time':now,'original_exact_identity_alive':same}
 except psutil.NoSuchProcess:observed={'pid_present':False,'original_exact_identity_alive':False}
 require(not observed['original_exact_identity_alive'],'Old exact runtime identity must be dead: '+who['role'])
 deaths.append({**who,'death_observation':observed,'exact_original_identity_dead':True})
CK=B/'checkpoints/update_000156600';pointer=read(CK/'pointer.json');persistence=read(CK/'persistence_verified.json');saved=pointer['journal_prefix']
require(pointer['committed_updates']==156600 and persistence['passed']and persistence['optimizer_updates']==156600
 and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215
 and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256']
 and persistence['saved_semantic_state_verified']['passed']and persistence['saved_semantic_state_verified']['saved_identity_frontier_sampler_and_journal_verified_before_publication'],'Full saved156600854/207/215 PRE-SAVED and semantic binding')
require(saved=={'path':str(B/'journals/commits_000018.jsonl'),'offset':317194933,'sha256':'eb6b816bff1f40706a4595a045c2c3b9cafc5eaedccff6dcc010fab0d0e2a199'},'Published exact saved156600 journal18 prefix')
journal=Path(saved['path']);before=journal.stat();require(saved['offset']<before.st_size<saved['offset']+1024**2,'Bounded169-tail read only')
with journal.open('rb')as f:f.seek(saved['offset']);raw=f.read(1024**2+1)
require(raw.endswith(b'\n')and len(raw)==before.st_size-saved['offset'],'Complete exact tail bytes')
events=[json.loads(line)for line in raw.splitlines()]
require(len(events)==169 and all(e['optimizer_updates']==e['native']['update']==e['prepared_evidence']['actual_positive_native_update']==u for u,e in zip(range(156601,156770),events)),'169 exact ordered positive unsaved actual native events')
tail_path=F/'preserved_unsaved_positive_tail156601_156769.jsonl'
with tail_path.open('xb')as f:f.write(raw)
after=journal.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Original journal unchanged during suffix read')
guard()
progress_path=F/'last_emitted_positive_progress.json'
with progress_path.open('xb')as f:f.write((B/'progress/latest.json').read_bytes())
progress=read(progress_path);require(progress['producer']==failure['producer']and progress['actual_optimizer_updates']==156760 and progress['last_saved_updates']==156600,'Progress20 snapshot is156760, not final156769')
latest_path=F/'last_published_checkpoint.json'
with latest_path.open('xb')as f:f.write((B/'latest_checkpoint.json').read_bytes())
latest=read(latest_path);require(latest['committed_updates']==156600 and latest['full_state_verified'],'Current full checkpoint remains156600')
lease_path=F/'closed_GPU_owner_lease.json'
with lease_path.open('xb')as f:f.write((P/'work/gpu_training_owner.json').read_bytes())
lease=read(lease_path);require(lease['status']=='production_resident_RAM_process_closed'and lease['actual_GPU_running']is False,'Closed owner lease')
outer=read(F/'owned_session96057_actual_exit_observation.json');require(outer['exit_code']==1 and outer['chunk_id']=='102679','Actual owned outer TOOL1')
worker=read(I/'private_metadata_worker_completion.json');require(worker['joined']and worker['worker_connections_closed_on_worker']and worker['actual_sampler_replay_or_native_advance']is False,'Private planner thread/context closed')
paths={'actual_failure':I/'actual_failure.json','original_monitor_completion':I/'monitor_completion.json','CPU_stderr':I/'CPU/CPU_stderr.log','CPU_ready':I/'CPU/actual_CPU_ready.json','native_log':A/'native_000018.log','current_child':A/'current_child.json','boot':I/'boot.json','effective_configuration':I/'configuration.json','pointer':CK/'pointer.json','persistence':CK/'persistence_verified.json','last_published_checkpoint_snapshot':latest_path,'last_emitted_progress_snapshot':progress_path,'outer_TOOL_exit':F/'owned_session96057_actual_exit_observation.json','closed_owner_lease_snapshot':lease_path,'private_metadata_worker_completion':I/'private_metadata_worker_completion.json','recorded_exact_runtime_family':family_path,'original_source24_execution_review':N/'ROOT_owned_packet_credit_saved156400_execution_review_v1.json','prior_actual_recovery_review':N/'ROOT_owned_packet_credit_saved156400_recovered156600_actual_verification_v1.json','sealer_source':Path(__file__),'metadata_entry':F/'actual_metadata_entry.json'}
inputs={k:pin(p)for k,p in paths.items()}
value={'kind':'ASR18_actual_source24_saved156600_native_preupdate_pressure_timeout156769_closure_v1','owned_session':96057,'actual_outer_TOOL_exit_code':1,'actual_outer_TOOL_chunk_id':'102679','native_numeric_OS_exit_code':None,'CPU_numeric_OS_exit_code':None,'wrapper_numeric_OS_exit_code_separately_recorded':None,'no_unknown_numeric_OS_exit_invented':True,'actual_native_optimizer_updates':156769,'last_emitted_positive_progress_updates':156760,'last_fully_saved_optimizer_updates':156600,'unsaved_positive_tail_count':169,'all_unsaved_live_updates_require_coherent_reload':True,'actual_failure':failure,'original_monitor_completion':monitor,'all_five_recorded_runtime_identities_dead':deaths,'inputs':inputs,'saved_journal_prefix_as_verified_by_published_persistence':saved,'saved_prefix_rehashed_in_this_observer':False,'whole_journal_rehashed_in_this_observer':False,'closed_journal_stat':{'path':str(journal),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'read_scope':'ONLY bytes from saved156600 prefix offset to EOF; no original prefix/whole read'},'preserved_unsaved_positive_tail':pin(tail_path),'tail_offsets':{'start_inclusive':saved['offset'],'end_exclusive':after.st_size},'ordered_positive_tail_updates':list(range(156601,156770)),'original_journal_stat_unchanged':True,'original_checkpoints_sources_and_environment_untouched':True,'no_journal_branch_copy_restore_or_training_launch':True,'no_packet_propose_native_operation_admitted_by_timeout':True,'timeout_location':'FIRST full pre-take pressure gate at driver24 line910, before wait_update_ready','new_timeout_gate_distinct_from_original_monitor_floor_failure':True,'actual_last_commit_deficit_to_added_gate_bytes':denial['required_commit_available_bytes']-memory['commit_available_bytes'],'actual_last_commit_headroom_above_original_floor_bytes':memory['commit_available_bytes']-2000000000,'no_credit_or_transport_failure_reported':True,'CPU10_normal0_or_kernel_handle_close_receipt_available':False,'cause_limits':'The unchanged conservative added buffer was not met at the last persisted sample after300s. Original Monitor closed clean. This proves neither required native allocation, measured saving, external process cause, OOM, original-floor crossing nor identical bytes across all600 observations. Separate historical native/CPU/wrapper numeric OS exits are unavailable. The169 unsaved positive attempts are historical and require genuine redo from156600, not tail replay. The published saved-prefix digest is retained without recomputing prefix/whole. No unchanged retry or buffer reduction is authorized.','metadata_observer':identity,'metadata_launcher':parent_identity,'guard_initial':initial,'guard_final':guard()}
write(F/'actual_failure_closure.json',value)
print(json.dumps({'closure':pin(F/'actual_failure_closure.json'),'tail':pin(tail_path),'actual_final':156769,'saved':156600,'all5_exact_dead':True,'observer':identity,'launcher':parent_identity}),flush=True)

