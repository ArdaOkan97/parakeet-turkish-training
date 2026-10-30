"""Read-only closure of the actual resident headroom stop; no training launch."""
import hashlib,json,os,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000010';O=N/'resident_RAM_training_actual_11'
F=N/'resident_RAM_headroom133137_failure_actual_v1'
started=time.monotonic();peak=0;guard_samples=0
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def guard():
    global peak,guard_samples
    s=sample_memory(os.getpid(),'readonly_headroom_failure_closure');guard_samples+=1
    require(s['paired_sampling_within_one_second'],'Fresh paired actual memory observation')
    peak=max(peak,s['worker_private_memory']['private_bytes'])
    require(peak<=512*1024**2 and time.monotonic()-started<=180,'Bounded closure observer')
    m=s['system_memory'];require(m['commit_available_bytes']>=2000000000 and m['physical_available_bytes']>=1500000000,'Unchanged real system headroom')
    require(shutil.disk_usage('E:/').free>=100000000000,'E reserve preserved')
    return s
def pin(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb')as f:
        while True:
            block=f.read(4*1024**2)
            if not block:break
            h.update(block);guard()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def write_new(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    with Path(path).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def death(who,label):
    pid=who['pid'];ctime=who.get('create_time');current=None
    try:current=psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:pass
    original_dead=current is None or ctime is not None and abs(current-ctime)>=.001
    require(original_dead,'Original owned process must be proved dead: '+label)
    return {'label':label,'original_identity':who,'observed_pid_absent':current is None,
        'observed_current_create_time':current,'original_identity_dead':original_dead,
        'historical_create_time_unavailable':ctime is None}
initial_memory=guard();require(not F.exists(),'Fresh immutable failure namespace')
failure=read(I/'actual_failure.json');headroom=read(I/'actual_failing_headroom_sample.json')
current=read(O/'current_child.json');ready=read(I/'CPU/actual_CPU_ready.json')
latest=read(B/'latest_checkpoint.json');pointer=read(latest['pointer']['path'])
persistence=read(latest['persistence']['path']);monitor=read(I/'monitor_completion.json')
require(failure['actual_optimizer_updates']==133137 and failure['last_checkpoint']==133000,'Exact actual failed/saved frontier')
require(current['invocation']==10 and failure['producer']['pid']==35820,'Exact original owner invocation')
require(latest['committed_updates']==pointer['committed_updates']==133000 and latest['full_state_verified'] is True,'Actual coherent saved133000')
require(persistence['passed'] is True and persistence['weights_verified']==207 and persistence['Adam_states_verified']==215 and persistence['named_tensors_verified']==854,'Existing verified full saved state')
require(persistence['independent_pre_save_state_sha256']==persistence['saved_state_sha256'],'Independent PRE/SAVED equality retained')
require(pointer['journal_prefix']==persistence['journal_prefix'],'Same saved whole-journal prefix authority')
require('Sensible real system headroom' in failure['error'] and monitor['joined'] is True and monitor['clean'] is False,'Truthful abnormal monitor closure')
require(headroom['system_memory']['commit_available_bytes']==1639993344 and headroom['system_memory']['physical_available_bytes']==12542930944,'Exact captured failing resource sample')
deaths=[death(failure['producer'],'native'),death(current['launcher'],'native interpreter launcher'),
    death(ready['producer_identity'],'resident CPU'),death({'pid':ready['producer_identity']['ppid'],'create_time':None},'CPU interpreter launcher'),
    death({'pid':current['launcher']['ppid'],'create_time':None},'outer wrapper')]
require(not (I/'normal_completion.json').exists(),'No fabricated normal completion')
F.mkdir();write_new(F/'actual_owned_process_deaths.json',{'observed_at_utc':datetime.now(timezone.utc).isoformat(),'identities':deaths,'known_identity_scope_only':True})
journal=Path(pointer['journal_prefix']['path']);before=journal.stat();prefix_end=pointer['journal_prefix']['offset']
whole=hashlib.sha256();prefix=hashlib.sha256();tail_hash=hashlib.sha256();position=0;tail_bytes=0
tail=F/'preserved_positive_unsaved_tail_133001_133137.jsonl'
with journal.open('rb')as src,tail.open('xb')as dst:
    while True:
        block=src.read(4*1024**2)
        if not block:break
        whole.update(block);n=max(0,min(len(block),prefix_end-position));prefix.update(block[:n])
        suffix=block[n:]
        if suffix:dst.write(suffix);tail_hash.update(suffix);tail_bytes+=len(suffix)
        position+=len(block);guard()
    dst.flush();os.fsync(dst.fileno())
after=journal.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Dead-owner journal unchanged during one-pass observation')
require(position>=prefix_end and prefix.hexdigest()==pointer['journal_prefix']['sha256'],'Exact saved journal prefix verified once')
require(0<tail_bytes<=2*1024**2,'Bounded immutable unsaved tail copy')
events=[json.loads(line)for line in tail.read_bytes().splitlines()]
require(len(events)==137,'Exactly137 historical positive tail events')
for u,event in zip(range(133001,133138),events):
    require(event['kind']=='ASR18_committed_training_update_v1' and event['optimizer_updates']==u and event['native']['update']==u
        and event['prepared_evidence']['actual_positive_native_update']==u,'Exact ordered real positive native/prepared/journal event')
evidence_paths={'actual_failure':I/'actual_failure.json','failing_headroom_sample':I/'actual_failing_headroom_sample.json',
    'monitor_completion':I/'monitor_completion.json','private_metadata_worker_completion':I/'private_metadata_worker_completion.json',
    'CPU_ready':I/'CPU/actual_CPU_ready.json','CPU_stderr':I/'CPU/CPU_stderr.log','native_log':O/'native_000010.log',
    'current_child':O/'current_child.json','effective_configuration':I/'configuration.json','progress_at_last_emission':B/'progress/latest.json',
    'latest_checkpoint_publication':B/'latest_checkpoint.json','saved_pointer':Path(latest['pointer']['path']),
    'saved_persistence':Path(latest['persistence']['path']),'saved_checkpoint_metadata':Path(pointer['checkpoint']['path']),
    'lease_at_closed_owner':P/'work/gpu_training_owner.json','driver_source':P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v12.py',
    'wrapper_source':P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v12.py'}
pins={k:pin(v)for k,v in evidence_paths.items()}
require(pins['saved_pointer']==latest['pointer'] and pins['saved_persistence']==latest['persistence'],'Genuine exact current checkpoint pins')
final_memory=guard()
report={'kind':'ASR18_actual_resident_RAM_headroom133137_saved133000_abnormal_closure_v1',
    'actual_owned_session':80857,'actual_TOOL_exit_code':1,'actual_TOOL_chunk_id':'d0a49f',
    'actual_TOOL_error':'Actual native process failed; preserve all state/logs, no unchanged retry',
    'native_OS_exit_code':None,'native_OS_exit_code_not_observed':True,'actual_wrapper_wait_code_was_nonzero':True,
    'wrapper_OS_exit_code':None,'wrapper_numeric_OS_exit_not_separately_observed':True,'CPU_OS_exit_code':None,
    'actual_failure':pins['actual_failure'],'actual_failed_optimizer_updates':133137,'saved_optimizer_updates':133000,
    'original_failure_was_resource_guard_not_normal0':True,'actual_failure_was_not_an_EOF_or_source_interval_failure':True,
    'actual_failing_headroom_sample':headroom,'monitor_completion':monitor,'owned_process_deaths':pin(F/'actual_owned_process_deaths.json'),
    'all_known_owned_identities_dead':True,'unrecorded_historical_create_times_not_fabricated':True,
    'source_evidence':pins,'saved_state_verified':{'weights':207,'Adam_states':215,'named_tensors':854,'PRE_equals_SAVED':True,'full_state_verified':True},
    'saved_journal_prefix':{'path':str(journal),'offset':prefix_end,'sha256':prefix.hexdigest(),'actual_streamed_hash_match':True},
    'whole_old_journal':{'path':str(journal),'bytes':position,'sha256':whole.hexdigest()},
    'preserved_unsaved_positive_tail':{'path':str(tail),'bytes':tail_bytes,'sha256':tail_hash.hexdigest(),
        'actual_ordered_event_count':137,'first_optimizer_updates':133001,'last_optimizer_updates':133137,
        'all_native_and_prepared_positive_update_counters_match':True,'historical_attempts_not_active_restored_lineage':True},
    'tail_is_exact_whole_journal_suffix_after_saved_prefix':True,'whole_journal_hash_passes':1,
    'original_journal_checkpoint_logs_sources_or_environment_mutated':False,'no_truncation_reset_tail_replay_or_unchanged_retry':True,
    'normal_completion_claimed':False,'GPU_or_training_job_launched':False,'new_source_or_resource_remedy_admitted':False,
    'observer':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'guard_samples':guard_samples,
        'private_peak_bytes':peak,'wall_seconds':time.monotonic()-started,'initial_memory':initial_memory,'final_memory':final_memory,
        'CUDA_masked':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'}}
write_new(F/'actual_failure_closure.json',report)
print(json.dumps({'receipt':pin(F/'actual_failure_closure.json'),'failed_updates':133137,'saved_updates':133000,
    'tail_events':len(events),'journal_bytes':position,'all_known_identities_dead':True}),flush=True)
