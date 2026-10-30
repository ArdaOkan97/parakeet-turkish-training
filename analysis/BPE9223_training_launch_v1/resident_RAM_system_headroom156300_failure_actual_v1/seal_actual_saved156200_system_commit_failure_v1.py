"""Actual source20 system-commit failure closure; no recovery or changed policy."""
import datetime,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000016'
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
A=N/'resident_RAM_training_actual_17';F=Path(__file__).parent
began=time.monotonic();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):
 p=Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def guard():
 s=sample_memory(os.getpid(),'source20_saved156200_actual_failure_metadata');m=s['system_memory']
 require(s['paired_sampling_within_one_second'] and s['worker_private_memory']['private_bytes']<512*1024**2 and m['commit_available_bytes']>=2000000000+268435456 and m['physical_available_bytes']>=1500000000+268435456 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Unchanged finite metadata/headroom/full256MiB/E guards')
 return s
initial=guard()
require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2 and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective metadata allowance')
require(not (F/'actual_failure_closure.json').exists(),'One immutable closure')
observer=psutil.Process();parent=observer.parent();observer_identity={'pid':observer.pid,'create_time':observer.create_time(),'ppid':observer.ppid()}
parent_identity=None if parent is None else {'pid':parent.pid,'create_time':parent.create_time(),'ppid':parent.ppid()}
failure=read(I/'actual_failure.json');monitor=read(I/'monitor_completion.json');sample=read(I/'actual_failing_headroom_sample.json')
require(failure['producer']=={'pid':24824,'create_time':1792682123.6010237,'ppid':24240} and failure['actual_optimizer_updates']==156300 and failure['last_checkpoint']==156200 and 'Actual resident memory monitor' in failure['error'] and 'Sensible real system headroom' in failure['error'],'Exact actual source20 failure')
require(monitor['joined'] is True and monitor['clean'] is False and 'Sensible real system headroom' in monitor['error'],'Actual joined monitor failure')
require(sample['paired_sampling_within_one_second'] and sample['system_memory']['commit_available_bytes']==1943629824 and sample['system_memory']['physical_available_bytes']==10373808128 and sample['worker_private_memory']['private_bytes']==7415681024,'Exact persisted failing sample, not postexit baseline')
expected=[('native',24824,1792682123.6010237),('native_interpreter_launcher',24240,1792682123.5941548),('wrapper',41344,None),('CPU',32980,1792708037.735279),('CPU_interpreter_launcher',24936,None)]
deaths=[]
for role,pid,ctime in expected:
 try:current=psutil.Process(pid).create_time();absent=False;dead=ctime is not None and abs(current-ctime)>=.001
 except psutil.NoSuchProcess:current=None;absent=True;dead=True
 require(dead,'Recorded old identity dead '+role)
 deaths.append({'role':role,'pid':pid,'recorded_create_time':ctime,'current_PID_absent':absent,'current_create_time_if_reused':current,'recorded_identity_dead':dead,'historical_create_time_unknown':ctime is None})
latest=read(B/'latest_checkpoint.json');CK=B/'checkpoints/update_000156200'
require(latest['committed_updates']==156200 and latest['full_state_verified'] is True and latest['pointer']==pin(CK/'pointer.json') and latest['persistence']==pin(CK/'persistence_verified.json'),'Exact latest fully saved156200')
pointer=read(CK/'pointer.json');persistence=read(CK/'persistence_verified.json');saved=pointer['journal_prefix']
require(pointer['committed_updates']==156200 and persistence['passed'] and persistence['optimizer_updates']==156200 and persistence['named_tensors_verified']==854 and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215 and persistence['journal_prefix']==saved and persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Saved full854/207/215 and PRE-SAVED equality')
journal=Path(saved['path']);require(journal==B/'journals/commits_000016.jsonl','Exact old active lineage')
stat=journal.stat();require(stat.st_size>saved['offset'],'Actual positive historical suffix')
tail_path=F/'preserved_unsaved_positive_tail156201_156300.jsonl';require(not tail_path.exists(),'One retained original suffix')
whole_h=hashlib.sha256();prefix_h=hashlib.sha256();tail_h=hashlib.sha256();offset=0
with journal.open('rb') as source,tail_path.open('xb') as target:
 while True:
  block=source.read(4*1024**2)
  if not block:break
  whole_h.update(block);n=min(len(block),max(0,saved['offset']-offset))
  if n:prefix_h.update(block[:n])
  if n<len(block):target.write(block[n:]);tail_h.update(block[n:])
  offset+=len(block);guard()
require(journal.stat().st_size==stat.st_size==offset and prefix_h.hexdigest()==saved['sha256'],'One streamed exact prefix/whole observation; original unchanged')
tail=pin(tail_path);require(tail['sha256']==tail_h.hexdigest() and tail['bytes']==offset-saved['offset'],'Preserved exact suffix bytes')
events=[json.loads(line)for line in tail_path.read_bytes().splitlines()]
require(len(events)==100 and all(x['optimizer_updates']==x['native']['update']==x['prepared_evidence']['actual_positive_native_update']==u for u,x in zip(range(156201,156301),events)),'100 actual ordered positive attempts; no assumed native+1 diagnosis')
progress_path=F/'last_emitted_positive_progress.json';require(not progress_path.exists(),'One final emitted observation');progress_path.write_bytes((B/'progress/latest.json').read_bytes())
progress=read(progress_path);require(progress['actual_optimizer_updates']==156300 and progress['producer']==failure['producer'] and progress['journal_prefix']=={'path':str(journal),'offset':offset,'sha256':whole_h.hexdigest()},'Final emitted156300 matches whole closed journal')
actual_tool={'kind':'ASR18_owned_session_actual_TOOL_exit_observation_v1','owned_session':47103,'actual_TOOL_exit_code':1,'actual_TOOL_chunk_id':'3f03f5','observed_output_final_line':'ValueError: Actual native process failed; preserve all state/logs, no unchanged retry','full_TOOL_output_preserved_in_actual_tool_transcript':True,'separate_native_numeric_OS_exit_code':None,'separate_CPU_numeric_OS_exit_code':None,'separate_wrapper_numeric_OS_exit_code':None,'no_unknown_numeric_code_inferred':True}
tool_path=F/'owned_session47103_actual_exit_observation.json';tool_path.write_text(json.dumps(actual_tool,indent=2)+'\n',encoding='utf-8')
inputs={k:pin(p)for k,p in {'actual_failure':I/'actual_failure.json','monitor_completion':I/'monitor_completion.json','failing_headroom_sample':I/'actual_failing_headroom_sample.json','CPU_stderr':I/'CPU_after_periodic_DEV_000156000/CPU_stderr.log','CPU_ready':I/'CPU_after_periodic_DEV_000156000/actual_CPU_ready.json','native_log':A/'native_000016.log','current_child':A/'current_child.json','boot':I/'boot.json','effective_configuration':I/'configuration.json','pointer':CK/'pointer.json','persistence':CK/'persistence_verified.json','last_emitted_progress':progress_path,'outer_TOOL_exit':tool_path,'closed_GPU_owner_lease':P/'work/gpu_training_owner.json','private_metadata_worker_completion':I/'private_metadata_worker_completion.json'}.items()}
lease=read(inputs['closed_GPU_owner_lease']['path']);require(lease['status']=='production_resident_RAM_process_closed' and lease['actual_GPU_running'] is False,'Actual idle closed lease; historical session label not active owner')
value={'kind':'ASR18_actual_saved156200_source20_system_commit_failure_closure_v1','owned_session':47103,'actual_outer_TOOL_exit_code':1,'actual_outer_TOOL_chunk_id':'3f03f5','native_numeric_OS_exit_code':None,'CPU_numeric_OS_exit_code':None,'wrapper_numeric_OS_exit_code_separately_recorded':None,'no_unknown_numeric_OS_exit_invented':True,'actual_native_optimizer_updates':156300,'last_fully_saved_optimizer_updates':156200,'unsaved_positive_tail_count':100,'all_unsaved_live_updates_require_coherent_reload':True,'actual_failure':failure,'monitor_completion':monitor,'actual_failing_headroom_sample':sample,'all_recorded_identities_dead':deaths,'inputs':inputs,'saved_prefix_verified_once':saved,'whole_journal_verified_once':{'path':str(journal),'bytes':offset,'sha256':whole_h.hexdigest()},'preserved_unsaved_positive_tail':tail,'ordered_positive_tail_updates':list(range(156201,156301)),'actual_preserved_tail_must_not_be_replayed_as_persisted_state':True,'old_journal_unchanged':True,'no_journal_branch_copy_restore_or_training_launch':True,'source20_CPU9_numerical_resource_science_policy_unchanged':True,'latest156000_quality_remains_historical_and_already_accepted':True,'cause_limitations':'The actual guard crossing is measured system-wide commit headroom; physical memory remained above10GB. No individual external-process attribution, actual CPU private sample at that instant, or claimed measured workload saving is inferred.','current_owner_session_field_is_historical_diagnostic':lease.get('current_owned_session'),'metadata_observer':observer_identity,'metadata_parent':parent_identity,'guard_initial':initial,'guard_final':guard()}
path=F/'actual_failure_closure.json';path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'closure':pin(path),'saved_prefix':saved,'whole':value['whole_journal_verified_once'],'tail':tail,'known_dead':deaths,'metadata_observer':observer_identity,'metadata_parent':parent_identity}),flush=True)
