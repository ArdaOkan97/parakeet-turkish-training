"""Seal only the actual188k metadata TOOL result and exact recorded process closure."""
import hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from resource_watchdog_v2 import sample_memory
F=Path(__file__).resolve().parent
A=F.parent.parent/'resident_RAM_training_actual_27'
B=Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000026'
T=Path('E:/gpu_training_scratchpad/analysis/BPE9223_Trackio_live_v1')
began=time.monotonic()
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
def require(v,m):
 if not v:raise ValueError(m)
def pin(p):
 p=Path(p);raw=p.read_bytes();return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n')as f:
  json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def bounded(p,limit=65536):
 p=Path(p);require(p.stat().st_size<=limit,'Bounded compact metadata only');return json.loads(p.read_text(encoding='utf-8-sig'))
def identity(pid):
 process=psutil.Process(pid);return {'pid':pid,'create_time':process.create_time(),'ppid':process.ppid()}
def alive(who):
 try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])<.001
 except psutil.NoSuchProcess:return False
initial=sample_memory(os.getpid(),'periodic188000_metadata_completion_initial')
initial_path=A/'actual_periodic188000_completion_initial_numeric_before_predicate_v1.json'
write(initial_path,{'sample':initial,'required_commit':5055306368,'required_physical':2305306368,
 'captured_before_predicate_and_metadata_reads':True,'completion_identity':identity(os.getpid()),
 'completion_launcher':identity(os.getppid())})
require(initial['paired_sampling_within_one_second']and initial['worker_private_memory']['private_bytes']<536870912
 and initial['system_memory']['commit_available_bytes']>=5055306368
 and initial['system_memory']['physical_available_bytes']>=2305306368
 and shutil.disk_usage('E:/').free>=100000000000,'Original full prospective metadata resource admission')
scope_path=F/'ROOT_periodic188000_source37_CPU14_metadata_only_exact_execution_scope_v1.json'
require(pin(scope_path)['sha256']=='5609f9fdb7cc66df45b96bbfbc46509c89cf71b573a8a5f20b329269742adcd3','Exact consumed scope')
status_path=A/'actual_periodic188000_and_resumed_training_status.json'
require(pin(status_path)['sha256']=='f8adfa4b1a04d0f6505e871b34d12b6dd997c7cce31ee5b3cf1d90f8b3ea5194','Exact actual observer output')
status=bounded(status_path)
observer_identities=[status['sealer_identity'],status['sealer_launcher']]
dead=[dict(who,observed_exact_dead=not alive(who))for who in observer_identities]
require(all(v['observed_exact_dead']for v in dead),'Actual original metadata observer and launcher exactdead')
tool_path=A/'actual_periodic188000_source37_metadata_TOOL_result_v1.json'
write(tool_path,{'kind':'ASR18_periodic188000_actual_owned_metadata_TOOL_result_v1','actual_TOOL_exit_code':0,
 'actual_TOOL_chunk':'f7cc19','direct_result_no_yielded_session':True,'actual_Popen_exit_code':'UNRECORDED_NOT_A_POPEN_FIXTURE',
 'individual_observer_and_launcher_numeric_OS_exit_codes':'UNRECORDED_UNKNOWN_NOT_INFERRED',
 'ROOT_scope':pin(scope_path),'source':status['metadata_sealer_source'],'actual_status':pin(status_path),
 'recorded_original_observer_identity':status['sealer_identity'],'recorded_original_launcher_identity':status['sealer_launcher'],
 'outer_manual_PowerShell_ctime':'UNRECORDED_UNKNOWN','outer_manual_PowerShell_pid_from_source_launcher_ppid':status['sealer_launcher']['ppid'],
 'outer_shell_exact_death_not_claimed':True,'no_retry_or_rerun':True})
progress=bounded(B/'progress/latest.json')
require(progress['actual_optimizer_updates']>=188220 and progress['last_saved_updates']>=188200
 and not progress['optimizer_poisoned']and not progress['schedule_poisoned']
 and progress['planned_native_updates']==401072 and progress['administrative_update_stop']is None
 and not(I/'actual_failure.json').exists(),'Current healthy full-run frontier')
protected=[{'pid':19320,'create_time':1792803517.7236164},{'pid':39856,'create_time':1792803517.7165897},
 {'pid':21928,'create_time':1792803516.8916032},{'pid':36416,'create_time':1792806903.9606233},
 {'pid':4092,'create_time':1792806903.9441268},{'pid':26892,'create_time':1792652904.3359847},
 {'pid':30128,'create_time':1792652904.283631}]
require(all(alive(who)for who in protected),'Same exact production5 and reader2 alive')
reader=bounded(T/'current_status.json')
last=sample_memory(os.getpid(),'periodic188000_metadata_completion_final')
require(last['paired_sampling_within_one_second']and last['worker_private_memory']['private_bytes']<536870912
 and last['system_memory']['commit_available_bytes']>=2268435456
 and last['system_memory']['physical_available_bytes']>=1768435456
 and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-began<180,'Original runtime/self/wall/E guards')
completion_path=A/'actual_periodic188000_source37_metadata_completion_v1.json'
write(completion_path,{'kind':'ASR18_periodic188000_source37_metadata_completion_v1','passed':True,
 'actual_TOOL':pin(tool_path),'actual_status':pin(status_path),'source':status['metadata_sealer_source'],
 'source_initial_numeric_sample':status['initial_metadata_full_resource_sample'],
 'manual_prelaunch_gate':pin(A/'actual_periodic188000_manual_prelaunch_gate_v1.json'),
 'initial_completion_numeric_sample':pin(initial_path),'recorded_original_observer_pair':dead,
 'recorded_original_observer_pair_exact_dead':True,'completion_identity':identity(os.getpid()),
 'completion_launcher':identity(os.getppid()),'completion_pair_alive_at_write_ROOT_can_check_after_TOOL_return':True,
 'outer_manual_PowerShell_historical_ctime':'UNRECORDED_UNKNOWN','no_outer_shell_exactdeath_or_numeric_OS_exit_invention':True,
 'protected7_exact_alive':[identity(who['pid'])for who in protected],
 'actual_optimizer_updates':progress['actual_optimizer_updates'],'last_saved_updates':progress['last_saved_updates'],
 'native_updates_this_process':progress['native_updates_this_process'],'both_poisons_false':True,
 'full401072_adminNone':True,'reader_current_compact':reader,'last_resource_sample':last,
 'no_journal_plan_body_archive_tensor_audio_report_prediction_reaudit':True,
 'no_training_reader_environment_source_LR_GC_guard_or_policy_change':True})
print(json.dumps({'TOOL_receipt':pin(tool_path),'completion':pin(completion_path),
 'observer_pair':dead,'completion_identity':identity(os.getpid()),'completion_launcher':identity(os.getppid()),
 'actual_optimizer_updates':progress['actual_optimizer_updates'],'last_saved_updates':progress['last_saved_updates']}),flush=True)
