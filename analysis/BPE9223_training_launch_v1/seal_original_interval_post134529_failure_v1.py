"""One read-only new failure/journal closure pass; no training or audio job."""
import hashlib,json,os,shutil,sys,time
from datetime import datetime,timezone
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
N=Path(__file__).parent;B=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01')
I=B/'invocations/owner_000011';A=N/'resident_RAM_training_actual_12';F=N/'resident_RAM_original_interval_post134529_failure_actual_v1'
began=time.monotonic();peak=0;samples=0
def require(ok,msg):
    if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def guard():
    global peak,samples
    s=sample_memory(os.getpid(),'readonly_original_interval134529_failure_closure');samples+=1
    require(s['paired_sampling_within_one_second'],'Fresh actual paired resource sample');peak=max(peak,s['worker_private_memory']['private_bytes'])
    require(peak<=512*1024**2 and time.monotonic()-began<=180,'Finite512MiB180s metadata closure')
    require(s['system_memory']['commit_available_bytes']>=2000000000 and s['system_memory']['physical_available_bytes']>=1500000000,'Unchanged actual headroom guards')
    require(shutil.disk_usage('E:/').free>=100000000000,'E100GB reserve');return s
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);guard()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def write(p,v):
    with Path(p).open('xb')as f:f.write((json.dumps(v,indent=2,ensure_ascii=False)+'\n').encode('utf-8'));f.flush();os.fsync(f.fileno())
def death(who,label):
    now=None
    try:now=psutil.Process(who['pid']).create_time()
    except psutil.NoSuchProcess:pass
    c=who.get('create_time');gone=now is None or c is not None and abs(now-c)>=.001
    require(gone,'Actually dead original identity: '+label)
    return {'label':label,'original_identity':who,'observed_current_create_time':now,'original_identity_dead':True,
        'observed_pid_absent':now is None,'historical_create_time_unavailable':c is None}
initial=guard();require(not F.exists(),'Fresh immutable failure output')
failure=read(I/'actual_failure.json');monitor=read(I/'monitor_completion.json');CPU=read(I/'CPU/actual_CPU_failure.json')
current=read(A/'current_child.json');ready=read(I/'CPU/actual_CPU_ready.json');latest=read(B/'latest_checkpoint.json')
pointer=read(latest['pointer']['path']);persist=read(latest['persistence']['path'])
require(failure['actual_optimizer_updates']==134529 and failure['last_checkpoint']==134400
    and 'Original interval beyond source unresolved'in failure['error']and monitor['joined']and monitor['clean']and monitor['error']is None,
    'Actual CPU original-source coverage failure, not resource/normal0')
require(CPU['completed_updates']==1529 and CPU['message']=="ValueError('Original interval beyond source unresolved')"
    and current['invocation']==11 and failure['producer']['pid']==39588,'Actual current CPU/native counters and invocation')
require(latest['committed_updates']==pointer['committed_updates']==134400 and latest['full_state_verified']
    and persist['passed']and persist['named_tensors_verified']==854 and persist['weights_verified']==207 and persist['Adam_states_verified']==215
    and persist['independent_pre_save_state_sha256']==persist['saved_state_sha256']and pointer['journal_prefix']==persist['journal_prefix'],
    'Actual full coherent134400 saved frontier and independent PRE/SAVED equality')
deaths=[death(failure['producer'],'native'),death(current['launcher'],'native interpreter launcher'),
    death(ready['producer_identity'],'resident CPU'),death({'pid':ready['producer_identity']['ppid'],'create_time':None},'CPU interpreter launcher'),
    death({'pid':current['launcher']['ppid'],'create_time':None},'outer wrapper')]
require(not(I/'normal_completion.json').exists(),'No fabricated normal0')
F.mkdir();write(F/'actual_owned_process_deaths.json',{'identities':deaths,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'exact_recorded_scope_only':True})
# CPU sent1529 complete updates; native committed those same1529. The next plan update is identified by the physical ordered plan, not native+1 alone.
plan_path=I/'plans/plan_000077.json';next_plan_path=I/'plans/plan_000078.json';plan=read(plan_path);next_plan=read(next_plan_path)
require(plan['initial_committed_updates']==134520 and plan['updates'][8]['global_update']==134529
    and plan['updates'][9]['global_update']==134530 and len(plan['updates'])==20
    and next_plan['initial_committed_updates']==134540,'Actual adjacent ordered current/future plans')
remaining={'source_plan':pin(plan_path),'initial_updates':134520,'first_unconsumed_update_index':9,
    'candidate_global_updates':134530,'updates':plan['updates'][9:],'not_assumed_native_plus_one':True,
    'CPU_actual_completed_updates':1529,'native_actual_completed_updates_since133000':1529,
    'later_private_plan_not_authoritative':pin(next_plan_path)}
write(F/'actual_ordered_remaining_plan_metadata.json',remaining)
write(F/'actual_candidate134530_rows.json',{'source_plan':pin(plan_path),'update_index':9,'global_update':134530,
    'rows':plan['updates'][9]['rows'],'contexts':plan['updates'][9]['contexts'],'actual_failed_row_not_yet_identified':True})
journal=Path(pointer['journal_prefix']['path']);before=journal.stat();offset=pointer['journal_prefix']['offset']
require(journal==B/'journals/commits_000011.jsonl'and offset==273582680,'Exact new failed branch saved offset')
whole=hashlib.sha256();prefix=hashlib.sha256();tailhash=hashlib.sha256();pos=0;tailbytes=0;tail=F/'preserved_positive_unsaved_tail_134401_134529.jsonl'
with journal.open('rb')as src,tail.open('xb')as dst:
    for block in iter(lambda:src.read(4*1024**2),b''):
        whole.update(block);n=max(0,min(len(block),offset-pos));prefix.update(block[:n]);suffix=block[n:]
        if suffix:dst.write(suffix);tailhash.update(suffix);tailbytes+=len(suffix)
        pos+=len(block);guard()
    dst.flush();os.fsync(dst.fileno())
after=journal.stat();require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    and prefix.hexdigest()==pointer['journal_prefix']['sha256']and 0<tailbytes<=2*1024**2,'One exact saved-prefix/whole/suffix pass, original unchanged')
events=[json.loads(x)for x in tail.read_bytes().splitlines()];require(len(events)==129,'Exactly129 actual positive historical events')
for u,event in zip(range(134401,134530),events):
    require(event['optimizer_updates']==event['native']['update']==event['prepared_evidence']['actual_positive_native_update']==u,'Ordered positive native/preparation/journal counters')
paths={'actual_failure':I/'actual_failure.json','monitor_completion':I/'monitor_completion.json',
    'private_metadata_worker_completion':I/'private_metadata_worker_completion.json','CPU_failure_frame':I/'CPU/actual_CPU_failure.json',
    'CPU_ready':I/'CPU/actual_CPU_ready.json','CPU_stderr':I/'CPU/CPU_stderr.log','native_log':A/'native_000011.log',
    'current_child':A/'current_child.json','effective_configuration':I/'configuration.json','progress_at_last_emission':B/'progress/latest.json',
    'latest_checkpoint_publication':B/'latest_checkpoint.json','saved_pointer':Path(latest['pointer']['path']),
    'saved_persistence':Path(latest['persistence']['path']),'saved_checkpoint_metadata':Path(pointer['checkpoint']['path']),
    'lease_at_closed_owner':P/'work/gpu_training_owner.json','driver_source':P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v15.py',
    'wrapper_source':P/'work/research/run_ASR18_BPE9223_resident_RAM_autocontinue_v15.py','CPU_source':S/'fastmss_resident_feature_stream_v6.py'}
pins={k:pin(v)for k,v in paths.items()};final=guard()
report={'kind':'ASR18_actual_resident_RAM_original_interval_post134529_saved134400_abnormal_closure_v1',
    'actual_owned_session':60255,'actual_TOOL_exit_code':1,'actual_TOOL_chunk_id':'4c0093',
    'native_OS_exit_code':None,'native_numeric_OS_exit_not_observed':True,'actual_wrapper_wait_code_was_nonzero':True,
    'wrapper_OS_exit_code':None,'wrapper_numeric_OS_exit_not_separately_observed':True,'CPU_OS_exit_code':None,
    'actual_failed_optimizer_updates':134529,'saved_optimizer_updates':134400,'CPU_completed_updates_this_owner':1529,
    'actual_failure':pins['actual_failure'],'monitor_completion':monitor,'source_evidence':pins,
    'original_failure_was_source_coverage_not_resource_guard_or_normal0':True,'owned_process_deaths':pin(F/'actual_owned_process_deaths.json'),
    'all_known_owned_identities_dead':True,'unrecorded_historical_create_times_not_fabricated':True,
    'saved_state_verified':{'weights':207,'Adam_states':215,'named_tensors':854,'PRE_equals_SAVED':True,'full_state_verified':True},
    'saved_journal_prefix':{'path':str(journal),'offset':offset,'sha256':prefix.hexdigest(),'actual_streamed_hash_match':True},
    'whole_old_journal':{'path':str(journal),'bytes':pos,'sha256':whole.hexdigest()},
    'preserved_unsaved_positive_tail':{'path':str(tail),'bytes':tailbytes,'sha256':tailhash.hexdigest(),
        'actual_ordered_event_count':129,'first_optimizer_updates':134401,'last_optimizer_updates':134529,
        'historical_attempts_not_active_restored_lineage':True},
    'ordered_current_plan_remaining_metadata':pin(F/'actual_ordered_remaining_plan_metadata.json'),
    'candidate134530_rows':pin(F/'actual_candidate134530_rows.json'),'actual_failed_row_not_yet_identified':True,
    'tail_exact_whole_journal_suffix':True,'whole_journal_hash_passes':1,'normal_completion_claimed':False,
    'original_journal_checkpoint_logs_sources_or_environment_mutated':False,'no_truncation_reset_tail_replay_or_unchanged_retry':True,
    'GPU_training_audio_or_new_remedy_launched':False,
    'observer':{'pid':os.getpid(),'create_time':psutil.Process().create_time(),'guard_samples':samples,'private_peak_bytes':peak,
        'wall_seconds':time.monotonic()-began,'initial_memory':initial,'final_memory':final,'CUDA_masked':os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'}}
write(F/'actual_failure_closure.json',report);print(json.dumps({'receipt':pin(F/'actual_failure_closure.json'),
    'saved':134400,'actual':134529,'tail_count':129,'candidate_update':134530,'candidate_YODAS_rows':3,'all_recorded_identities_dead':True}),flush=True)
