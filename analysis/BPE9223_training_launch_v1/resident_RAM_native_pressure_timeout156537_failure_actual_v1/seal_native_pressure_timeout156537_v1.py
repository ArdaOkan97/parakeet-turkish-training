"""Bounded actual failure seal: published saved prefix plus ONLY the closed137-event suffix."""
import datetime,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')
I=B/'invocations/owner_000017';A=N/'resident_RAM_training_actual_18';F=Path(__file__).parent
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,value):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
def guard():
 s=sample_memory(os.getpid(),'source22_saved156400_actual_gate_timeout_metadata');m=s['system_memory']
 require(s['paired_sampling_within_one_second'] and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original finite metadata/headroom/full256MiB/E guards')
 return s
initial=guard()
require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective metadata allowance')
me=psutil.Process();parent=me.parent()
identity={'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()}
parent_identity=None if parent is None else {'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}
write(F/'actual_metadata_entry.json',{'observer':identity,'launcher':parent_identity,'storage_environment':{k:os.environ.get(k) for k in ['GPU_TRAINING_STORAGE_ROOT','TORCH_EXTENSIONS_DIR','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}})
require(not (F/'actual_failure_closure.json').exists(),'One immutable closure')
failure=read(I/'actual_failure.json');monitor=read(I/'monitor_completion.json');denial=failure['native_pressure_denial']
require(failure['producer']=={'pid':12328,'create_time':1792711453.1490774,'ppid':38036} and failure['actual_optimizer_updates']==156537 and failure['last_checkpoint']==156400,'Exact failure producer and actual saved frontier')
require(failure['error']=="TimeoutError('Bounded native pre-update commit-pressure wait exhausted; preserve state, no unchanged retry')" and denial['no_packet_propose_native_or_authoritative_operation_admitted'] is True,'Typed new scheduling timeout before packet/propose/native')
sample=denial['last_denied_sample'];memory=sample['system_memory']
require(denial['observations']==600 and denial['observed_wait_seconds']==300.0 and denial['maximum_wait_seconds']==300,'Actual bounded timeout observations')
require(memory['commit_available_bytes']==4676124672 and denial['required_commit_available_bytes']==4768435456 and memory['physical_available_bytes']==11188850688 and denial['required_physical_available_bytes']==1768435456 and sample['worker_private_memory']['private_bytes']==5490585600,'Exact persisted pre-update denied sample')
require(monitor['joined'] is True and monitor['clean'] is True and monitor['error'] is None,'Original actual monitor clean; distinct from original floor failure')
require(not (I/'actual_failing_headroom_sample.json').exists(),'No falsely asserted original-floor failure receipt')
family_path=A/'actual_source22_live_runtime_family_after156400.json';family=read(family_path)
require([(x['role'],x['pid'],x['create_time'])for x in family]==[('native',12328,1792711453.1490774),('native_interpreter_launcher',38036,1792711453.1400745),('wrapper',5628,1792711452.6819165),('CPU',28556,1792711652.8914447),('CPU_interpreter_launcher',37316,1792711652.8715985)],'Exact actual recorded family')
deaths=[]
for old in family:
 try:current=psutil.Process(old['pid']).create_time();absent=False;dead=abs(current-old['create_time'])>=.001
 except psutil.NoSuchProcess:current=None;absent=True;dead=True
 require(dead,'Exact old identity dead '+old['role'])
 deaths.append({'role':old['role'],'pid':old['pid'],'recorded_create_time':old['create_time'],'current_PID_absent':absent,'current_create_time_if_reused':current,'recorded_identity_dead':dead})
CK=B/'checkpoints/update_000156400';pointer=read(CK/'pointer.json');persistence=read(CK/'persistence_verified.json');latest=read(B/'latest_checkpoint.json');saved=pointer['journal_prefix']
require(latest['committed_updates']==156400 and latest['full_state_verified'] is True and latest['pointer']==pin(CK/'pointer.json') and latest['persistence']==pin(CK/'persistence_verified.json'),'Actual latest full saved156400')
require(pointer['committed_updates']==156400 and persistence['passed'] is True and persistence['optimizer_updates']==156400 and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'] and persistence['saved_semantic_state_verified']['passed'] is True and persistence['saved_semantic_state_verified']['saved_identity_frontier_sampler_and_journal_verified_before_publication'] is True,'Full published854/207/215 PRE-SAVED and semantic equality')
require(saved=={'path':str(B/'journals/commits_000017.jsonl'),'offset':316802360,'sha256':'a5db7e40e8372c32c73b09b15ce87af7d3d9fed12bd8b1fede30f5fc3b461887'},'Exact published saved-prefix binding')
journal=Path(saved['path']);before=journal.stat();require(saved['offset']<before.st_size<saved['offset']+1024**2,'Bounded exact closed suffix only')
with journal.open('rb')as f:f.seek(saved['offset']);raw=f.read(1024**2+1)
require(len(raw)==before.st_size-saved['offset'] and raw.endswith(b'\n'),'Complete bounded suffix ending at closed EOF')
tail_path=F/'preserved_unsaved_positive_tail156401_156537.jsonl'
with tail_path.open('xb')as f:f.write(raw)
events=[json.loads(line)for line in raw.splitlines()]
require(len(events)==137 and all(e['optimizer_updates']==e['native']['update']==e['prepared_evidence']['actual_positive_native_update']==u for u,e in zip(range(156401,156538),events)),'137 actual ordered positive attempts in closed suffix')
after=journal.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Original closed journal stat unchanged throughout bounded read')
progress_path=F/'last_emitted_positive_progress.json'
with progress_path.open('xb')as f:f.write((B/'progress/latest.json').read_bytes())
progress=read(progress_path)
require(progress['actual_optimizer_updates']==156520 and progress['last_saved_updates']==156400 and progress['producer']==failure['producer'] and progress['journal_prefix']['offset']<after.st_size,'Emitted progress is stale at156520; not falsely final156537')
lease_path=P/'work/gpu_training_owner.json';lease=read(lease_path)
require(lease['status']=='production_resident_RAM_process_closed' and lease['actual_GPU_running'] is False,'Closed lease corroborates exact process deaths')
paths={'actual_failure':I/'actual_failure.json','monitor_completion':I/'monitor_completion.json','CPU_stderr':I/'CPU/CPU_stderr.log','native_log':A/'native_000017.log','current_child':A/'current_child.json','boot':I/'boot.json','effective_configuration':I/'configuration.json','pointer':CK/'pointer.json','persistence':CK/'persistence_verified.json','latest_checkpoint':B/'latest_checkpoint.json','last_emitted_progress':progress_path,'outer_TOOL_exit':F/'owned_session35083_actual_exit_observation.json','closed_GPU_owner_lease':lease_path,'private_metadata_worker_completion':I/'private_metadata_worker_completion.json','prior_recorded_runtime_family':family_path,'driver':P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v22.py','wrapper':P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v22.py','CPU_protocol':P/'outputs/parakeet_turkish/fastmss_resident_feature_stream_v9.py','gate_helper':N/'resident_RAM_system_headroom156300_failure_actual_v1/asr18_native_update_commit_pressure_gate_v2.py','source_configuration':N/'resident_RAM_system_headroom156300_failure_actual_v1/production_configuration_v1/configuration.json','sealer_source':Path(__file__),'metadata_entry':F/'actual_metadata_entry.json'}
inputs={k:pin(p)for k,p in paths.items()};require(inputs['CPU_stderr']['bytes']==0,'Actual CPU stderr empty')
tail=pin(tail_path)
value={'kind':'ASR18_actual_saved156400_source22_native_preupdate_pressure_timeout_closure_v1','owned_session':35083,'actual_outer_TOOL_exit_code':1,'actual_outer_TOOL_chunk_id':'dd5bb7','native_numeric_OS_exit_code':None,'CPU_numeric_OS_exit_code':None,'wrapper_numeric_OS_exit_code_separately_recorded':None,'no_unknown_numeric_OS_exit_invented':True,'actual_native_optimizer_updates':156537,'last_emitted_positive_progress_updates':156520,'last_fully_saved_optimizer_updates':156400,'unsaved_positive_tail_count':137,'all_unsaved_live_updates_require_coherent_reload':True,'actual_failure':failure,'monitor_completion':monitor,'all_recorded_identities_dead':deaths,'inputs':inputs,'saved_journal_prefix_as_verified_by_published_persistence':saved,'saved_prefix_rehashed_in_this_observer':False,'whole_journal_rehashed_in_this_observer':False,'closed_journal_stat':{'path':str(journal),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'read_scope':'ONLY bytes from saved prefix offset to EOF; prior prefix/whole not read'},'preserved_unsaved_positive_tail':tail,'tail_offsets':{'start_inclusive':saved['offset'],'end_exclusive':after.st_size},'ordered_positive_tail_updates':list(range(156401,156538)),'original_journal_stat_unchanged':True,'original_checkpoints_and_sources_untouched':True,'no_journal_branch_copy_restore_or_training_launch':True,'no_packet_propose_native_operation_admitted_by_timeout':True,'new_timeout_gate_is_distinct_from_original_monitor_floor_failure':True,'actual_commit_deficit_to_added_gate_bytes':denial['required_commit_available_bytes']-memory['commit_available_bytes'],'actual_commit_headroom_above_original_monitor_floor_bytes':memory['commit_available_bytes']-2000000000,'cause_limits':'The new conservative admission buffer was not met at its last persisted sample after300s. This does not prove a required native allocation, measured saving, external-process cause, original2GB-floor crossing, OOM, or that every denied observation had identical bytes. The original monitor closed clean. Numeric native/CPU/wrapper OS exit codes remain unknown. Saved-prefix digest is the published854/207/215 persistence binding; this observer intentionally read only137unsaved events and did not recompute old prefix or whole journal.','metadata_observer':identity,'metadata_launcher':parent_identity,'guard_initial':initial,'guard_final':guard()}
write(F/'actual_failure_closure.json',value)
print(json.dumps({'closure':pin(F/'actual_failure_closure.json'),'tail':tail,'range':value['tail_offsets'],'actual':156537,'saved':156400,'identities':deaths,'observer':identity,'launcher':parent_identity}),flush=True)
