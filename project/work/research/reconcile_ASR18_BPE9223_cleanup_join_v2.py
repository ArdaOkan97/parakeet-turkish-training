"""One readonly postmortem reconciliation; never performs feature cleanup.

No old handle/process0 is reconstructed and no original report is rewritten.
The real caller must observe this NEW child exit0 separately. Old physical
cleanup remains attributed to its true worker9 and actual observer6.
"""
import argparse, ctypes, hashlib, json, os, re, shutil, stat, sys, threading, time
from ctypes import wintypes as w
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
AREA=ROOT/'analysis/BPE9223_training_launch_v1/cleanup_join_reconciliation_v2'
P2=ROOT/'runs/BPE9223_weak2_clean1_20261018_02/checkpoints/transient/p000000002'
FAILED={'path':str(ROOT/'analysis/BPE9223_training_launch_v1/full_training_operational_resource_actual_04/actual_cleanup_join_failure_TOOL1_v1.json'),
    'bytes':12799,'sha256':'b949198d94edd1790ae8acb40d29d9fa72d5498e5da851b466f618fd9cf80303'}
PARENT={'path':str(PROJECT/'work/research/run_ASR18_BPE9223_full_run_owner_v6.py'),
    'bytes':45163,'sha256':'3e171acca950dc5c3cf452890c81e63c13cba7c15c68be7b87e5d700e2699760'}
NORMALIZER={'path':str(PROJECT/'outputs/parakeet_turkish/asr18_streamed_persistence_fingerprint_v1.py'),
    'bytes':10561,'sha256':'a7a58cd99e2805cf4997734d75b7f7ca9996177dcb2a4a5a7f7fd9892dfcd3e3'}
KIND='ASR18_readonly_postmortem_completed_feature_cleanup_reconciliation_v1'
MAXIMUM=16*1024**2;MAX_IO=16*1024**3;LIMIT=512_000_000
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
psapi=ctypes.WinDLL('psapi',use_last_error=True)
class Counters(ctypes.Structure):
    _fields_=[('cb',w.DWORD),('faults',w.DWORD),*[(x,ctypes.c_size_t) for x in
        ('peak_ws','ws','peak_page','page','peak_nonpage','nonpage','pagefile','peak_pagefile','private')]]
class Perf(ctypes.Structure):
    _fields_=[('cb',w.DWORD),*[(x,ctypes.c_size_t) for x in
        ('commit','limit','peak','physical','available','system_cache','kernel_total','kernel_paged','kernel_nonpaged','page_size')],
        ('handles',w.DWORD),('processes',w.DWORD),('threads',w.DWORD)]
class IO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):
    _fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),
        ('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):
    _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
        ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
kernel.GetCurrentProcess.restype=w.HANDLE
kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
kernel.GetProcessTimes.argtypes=[w.HANDLE,*[ctypes.POINTER(w.FILETIME)]*4];kernel.GetProcessTimes.restype=w.BOOL
kernel.GetProcessIoCounters.argtypes=[w.HANDLE,ctypes.POINTER(IO)];kernel.GetProcessIoCounters.restype=w.BOOL
kernel.SetPriorityClass.argtypes=[w.HANDLE,w.DWORD];kernel.SetPriorityClass.restype=w.BOOL
kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];kernel.CreateJobObjectW.restype=w.HANDLE
kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];kernel.SetInformationJobObject.restype=w.BOOL
kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];kernel.AssignProcessToJobObject.restype=w.BOOL
psapi.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(Counters),w.DWORD];psapi.GetProcessMemoryInfo.restype=w.BOOL
psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(Perf),w.DWORD];psapi.GetPerformanceInfo.restype=w.BOOL
HANDLE=kernel.GetCurrentProcess();began=time.monotonic();checks=[];samples=0;peak=0;high_count=0
def require(value,message):
    if not value:raise ValueError(message)
def creation(handle):
    values=[w.FILETIME() for _ in range(4)]
    require(kernel.GetProcessTimes(handle,*[ctypes.byref(x) for x in values]),'Exact process creation time available')
    return ((values[0].dwHighDateTime<<32)+values[0].dwLowDateTime)/10_000_000-11644473600
PRODUCER={'pid':os.getpid(),'ppid':os.getppid(),'create_time':creation(HANDLE)}
def dead(value):
    require(type(value) is dict and set(value)=={'pid','ppid','create_time'}
        and type(value['pid']) is int and value['pid']>0 and type(value['ppid']) is int and value['ppid']>0
        and type(value['create_time']) in (int,float) and value['create_time']>0,'Exact captured identity')
    handle=kernel.OpenProcess(0x1000,False,value['pid'])
    if not handle:
        require(ctypes.get_last_error()==87,'Unavailable process is not presumed dead on access failure')
        return True
    try:return abs(creation(handle)-value['create_time'])>.001
    finally:kernel.CloseHandle(handle)
def io_bytes():
    value=IO();require(kernel.GetProcessIoCounters(HANDLE,ctypes.byref(value)),'Actual IO counters available')
    return value.rb+value.wb
INITIAL_IO=io_bytes()
def guard():
    global samples,peak,high_count
    private=Counters();private.cb=ctypes.sizeof(private)
    system=Perf();system.cb=ctypes.sizeof(system)
    require(psapi.GetProcessMemoryInfo(HANDLE,ctypes.byref(private),ctypes.sizeof(private)),'Actual private bytes available')
    require(psapi.GetPerformanceInfo(ctypes.byref(system),ctypes.sizeof(system)),'Actual system commit available')
    ratio=system.commit/system.limit;headroom=(system.limit-system.commit)*system.page_size
    high_count=high_count+1 if ratio>.90 else 0
    require(private.private<=LIMIT and time.monotonic()-began<180 and io_bytes()-INITIAL_IO<=MAX_IO,'Fixed512MB/180s/16GiB scope')
    require(ratio<.95 and high_count<3 and headroom>=2_000_000_000,'Unchanged system90x3/95/2GB headroom')
    require(shutil.disk_usage(ROOT).free>=100_000_000_000,'E100GB reserve')
    samples+=1;peak=max(peak,private.private)
    return {'private_bytes':private.private,'system_commit_fraction':ratio,'system_headroom_bytes':headroom}
def scoped(value,source=False):
    require(type(value) is str,'Absolute path string required');path=Path(value)
    require(path.is_absolute() and str(path.resolve())==value
        and (path.is_relative_to(ROOT) or source and path.is_relative_to(PROJECT)),'Exact E/existing-project scope')
    for part in (path,*path.parents):
        if part.exists():require(not part.is_symlink() and not getattr(part.lstat(),'st_file_attributes',0)&1024,'No reparse aliases')
        if part in (ROOT,PROJECT):break
    return path
def record(path,maximum=None):
    path=scoped(str(Path(path).resolve()),source=True);before=path.stat()
    require(stat.S_ISREG(before.st_mode) and (maximum is None or before.st_size<=maximum),'Bounded regular input')
    hashed=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();hashed.update(block)
    require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'Input signature changed')
    return {'path':str(path),'bytes':before.st_size,'sha256':hashed.hexdigest()}
def pin(value):
    require(type(value) is dict and set(value)=={'path','bytes','sha256'} and type(value['bytes']) is int and value['bytes']>0
        and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),'Exact record3')
    scoped(value['path'],source=True);return value
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def read(value):
    pin(value);require(value['bytes']<=MAXIMUM and record(value['path'],MAXIMUM)==value,'Bounded immutable JSON pin')
    def unique(items):
        result={}
        for key,item in items:require(key not in result,'Duplicate JSON key');result[key]=item
        return result
    def invalid(value):raise ValueError('Nonfinite JSON '+value)
    return json.loads(Path(value['path']).read_bytes(),object_pairs_hook=unique,parse_constant=invalid)
def check(name,value):guard();require(value,name);checks.append(name)
def typed_zero(value):return type(value) is int and value==0
def main(request_path,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','CPU-only CUDA mask required')
    require(all(os.environ.get(key)=='1' for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Four unchanged thread caps1')
    require(kernel.SetPriorityClass(HANDLE,0x4000),'BelowNormal priority required')
    job=kernel.CreateJobObjectW(None,None);require(job,'New owned metadata512MB job')
    limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=LIMIT;limits.job_memory=LIMIT
    require(kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits))
        and kernel.AssignProcessToJobObject(job,HANDLE),'Owned metadata job512MB admission')
    timer=threading.Timer(180,lambda:os._exit(75));timer.daemon=True;timer.start();guard()
    request_pin=record(request_path,MAXIMUM);request=read(request_pin)
    require(set(request)=={'kind','ROOT_review','failure','parent_source','output_directory'}
        and request['kind']=='ASR18_ROOT_bound_cleanup_join_reconciliation_request_v1','Exact reconciliation request schema')
    output=scoped(str(Path(output).resolve()));require((output.is_relative_to(AREA)
        or output.parent==P2 and re.fullmatch(r'cleanup_join_reconciliation_[0-9]{6}',output.name)) and output.is_dir()
        and str(output)==request['output_directory'] and not (output/'report.json').exists(),'Fresh metadata output only in approved reconciliation area')
    source=record(__file__,256*1024);review=read(request['ROOT_review'])
    check('genuine_ROOT_exact_source_failure_observer_output_scope',review['kind']=='ROOT_cleanup_join_reconciliation_readonly_execution_review_v1'
        and review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='exact_readonly_cleanup_join_reconciliation_no_physical_cleanup'
        and review['source']==source and review['failure']==request['failure']==FAILED
        and review['parent_source']==request['parent_source']==PARENT and review['output_directory']==str(output)
        and review['maximum_private_bytes']==LIMIT and review['maximum_seconds']==180 and review['maximum_IO_bytes']==MAX_IO)
    require(record(PARENT['path'],256*1024)==PARENT,'Actual frozen code6 observer source')
    check('original_fingerprint_transport_normalizer_source',review['normalizer_source']==NORMALIZER
        and record(NORMALIZER['path'],256*1024)==NORMALIZER)
    sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
    import asr18_streamed_persistence_fingerprint_v1 as fingerprint_transport
    require(record(fingerprint_transport.__file__,256*1024)==NORMALIZER,'Actual original transport normalizer import')
    failed=read(FAILED)
    check('original_owner_remains_TOOL1_no_old_process0',failed['kind']=='ASR18_production40_cleanup_report_observer_join_failure_TOOL1_v1'
        and failed['actual_tool_exit_code']==failed['actual_exec_tool_exit_code']==1
        and failed['cleanup_process0_proof_written'] is False and failed['actual_native_normal_exit0_claimed'] is False
        and type(failed['actual_committed_optimizer_updates']) is int and failed['actual_committed_optimizer_updates']==40
        and failed['actual_new_optimizer_updates']==20 and failed['no_source_optimizer_RNG_sampler_or_history_reset'] is True)
    check('all_old_captured_identities_dead',failed['all_captured_identities_dead'] is True and all(dead(x) for x in failed['all_captured_identities']))
    original=read(failed['actual_cleanup_report']);drain=read(failed['actual_cleanup_Popen_wait0_and_family_drain']);old_guard=read(failed['actual_cleanup_guard'])
    old_folder=Path(failed['actual_cleanup_report']['path']).parent;workspace=old_folder.parent
    empty_log=record(old_folder/'worker.log',MAXIMUM)
    check('ROOT_bound_original_closed_empty_worker_log',review['original_cleanup_empty_worker_log']==empty_log
        and empty_log['bytes']==0 and empty_log['sha256']==hashlib.sha256(b'').hexdigest() and workspace==P2)
    check('no_old_process0_inferred_or_written',not (old_folder/'actual_owned_process0.json').exists())
    check('original_physical_report_source_mismatch_preserved',original['passed'] is True
        and original['kind']=='FastMSS_aggregate_CPU_window_owned_retirement_v1'
        and original['worker_source']==failed['parent_source'] and original['parent_source']==failed['original_report_parent_source']==failed['parent_source']
        and failed['actual_Popen_observer_source']==PARENT and original['parent_source']!=PARENT)
    for value in (failed['parent_source'],failed['controller_source'],failed['runtime_source']):require(record(value['path'],256*1024)==value,'Historical executing source changed')
    check('historical_actual_wait0_guard_and_ancestry',drain['kind']=='ASR18_outer_CPU_actual_wait0_captured_family_drain_v1'
        and typed_zero(drain['actual_Popen_wait_code']) and drain['all_captured_owned_identities_observed_dead'] is True
        and dead(drain['launcher']) and original['producer']['ppid']==drain['launcher']['pid'] and dead(original['producer'])
        and any(item==original['producer'] for item in failed['all_captured_identities'])
        and old_guard['pid']==original['producer']['pid'] and typed_zero(old_guard['actual_exit_code'])
        and old_guard['monitor_clean'] is True and old_guard['monitor_joined'] is True
        and old_guard['observed']['monitor_error'] is None and type(old_guard['observed']['samples']) is int and old_guard['observed']['samples']>=3)
    boot=read(record(old_folder/'guard/boot.json',MAXIMUM))
    check('historical_cleanup_guard_boot_matches_actual_producer',boot=={'pid':original['producer']['pid'],'ppid':original['producer']['ppid']})
    request_old=read(record(workspace/'cleanup_request.json',MAXIMUM));archive=read(failed['retained_feature_metadata'])
    archived_owner=archive['metadata']['owner.json'];archived_manifest=archive['metadata']['manifest.json']
    check('archive_original_owner_and_manifest_bytes_exact',archive['kind']=='FastMSS_aggregate_CPU_window_durable_metadata_v1'
        and archive['owner_binding']==original['owner']==request_old['owner']
        and archive['manifest_binding']==original['manifest']==request_old['manifest']
        and hashlib.sha256(canonical(archived_owner)).hexdigest()==original['owner']['sha256']
        and len(canonical(archived_owner))==original['owner']['bytes']
        and hashlib.sha256(canonical(archived_manifest)).hexdigest()==original['manifest']['sha256']
        and len(canonical(archived_manifest))==original['manifest']['bytes']
        and original['preserved_metadata']==failed['retained_feature_metadata']
        and request_old['archive_path']==failed['retained_feature_metadata']['path']
        and archive['boundary']==request_old['boundary'] and digest(archive['boundary'])==original['boundary_sha256'])
    names=archived_owner['owned_names'];window=scoped(str(Path(original['owner']['path']).parent))
    check('exact_owned_window_scope_and_682_names',window==ROOT/'temp/BPE9223_recorded_room_CPU_windows_v1/p000000002'
        and archived_owner['window_id']=='p000000002' and archived_owner['window_index']==1
        and type(names) is list and len(names)==len(set(names))==682
        and all(type(name) is str and Path(name).name==name and '/' not in name and '\\' not in name for name in names))
    expected={str(window/name) for name in names};removed=original['removed_exact_owned_files']
    check('all_and_only_original_682_owned_files_absent',type(removed) is list and len(removed)==len(set(removed))==682
        and set(removed)==expected and all(not scoped(path).exists() for path in removed) and not window.exists())
    check('archived_small_metadata_complete',set(archive['metadata'])=={name for name in names if name.endswith('.json')}
        and original['original_shared_oracle_untouched'] is True and original['sampler_journal_frontier_advanced'] is False
        and original['feature_files_only_deleted'] is True and original['original_audio_source_checkpoint_journal_untouched'] is True)
    cache=window.parent;check('original_feature_cache_empty',cache.is_dir() and not any(cache.iterdir()))
    pointer=read(failed['checkpoint_pointer']);metadata=read(failed['checkpoint_metadata']);pre=fingerprint_transport.normalize_fingerprint(read(failed['PRE_save_fingerprint']))
    completion=read(failed['actual_completed_native_window']);nextdraw=read(failed['next_draw_packet'])
    check('coherent_saved40_scope_and_unchanged_full_labels',type(pointer['committed_updates']) is int and pointer['committed_updates']==40
        and type(pointer['committed_window_updates']) is int and pointer['committed_window_updates']==20
        and pointer['all_window_updates_committed'] is True and pointer['writer_returned_and_payload_verified'] is True
        and pointer['checkpoint']==failed['checkpoint_metadata']==archive['boundary']['checkpoint']
        and pointer['payload']==failed['current_payload_binding_not_rehashed']==metadata['payload']
        and archive['boundary']['coherent_checkpoint_pointer']==failed['checkpoint_pointer']
        and completion['global_new_updates']==40 and len(completion['checkpoints'])==1
        and completion['checkpoints'][0]['pointer']==failed['checkpoint_pointer']
        and completion['checkpoints'][0]['pre_save_fingerprint']==failed['PRE_save_fingerprint']
        and completion['checkpoints'][0]['next_draw_packet']==failed['next_draw_packet'])
    require(scoped(pointer['payload']['path']).stat().st_size==pointer['payload']['bytes'],'Payload size only, no integrity/restore inference')
    check('actual_PRE40_mature_clocks_and_semantic_frontier',pre['state']['updates']==40
        and pre['state']['clocks']==failed['optimizer_clocks']==[64290,2040,62290]
        and pre['state_sha256']==digest(pre['state']) and len(pre['state']['weights'])==207 and len(pre['state']['Adam_partitions'])==3
        and pointer['sampler_state_sha256']==failed['sampler_state_sha256']
        and pointer['journal_prefix']==failed['journal_prefix'] and pointer['saved_semantic_state_verified']['passed'] is True)
    immutable=[FAILED,request['ROOT_review'],request_pin,failed['configuration'],failed['resume_request'],failed['execution_review'],
        failed['actual_cleanup_report'],failed['actual_cleanup_Popen_wait0_and_family_drain'],failed['actual_cleanup_guard'],
        failed['retained_feature_metadata'],failed['checkpoint_pointer'],failed['checkpoint_metadata'],failed['PRE_save_fingerprint'],
        failed['next_draw_packet'],failed['current_registered_journal'],failed['storage_state'],failed['durable_latest'],failed['pool_registry'],failed['p2_plan']]
    immutable += [record(workspace/'cleanup_request.json',MAXIMUM),record(old_folder/'guard/boot.json',MAXIMUM)]
    immutable += [empty_log,PARENT,source,NORMALIZER,failed['parent_source'],failed['controller_source'],failed['runtime_source']]
    before=[record(value['path']) for value in immutable]
    check('exact_original_immutable_input_signatures',before==immutable)
    storage=read(failed['storage_state'])
    check('unpublished40_not_relabelled_as_published_or_current_recovered',storage['latest']=='p000000001'
        and storage['pending_publication']==failed['pending_publication'] and storage['pending_publication']['id']=='p000000002'
        and storage['pending_retirement'] is None and failed['p2_archive_and_durable_publication_pass_claimed'] is False
        and failed['next_p3_CPU_or_positive_above40_claimed'] is False)
    check('journal_full40_prefix_unchanged_no_suffix',failed['current_registered_journal']['bytes']==pointer['journal_prefix']['offset']
        and failed['current_registered_journal']['sha256']==pointer['journal_prefix']['sha256'])
    after=[record(value['path']) for value in immutable];check('all_authoritative_inputs_unchanged',before==after)
    check('no_new_physical_cleanup_or_original_report_rewrite',not window.exists()
        and not (old_folder/'actual_owned_process0.json').exists() and record(failed['actual_cleanup_report']['path'])==failed['actual_cleanup_report'])
    report={'kind':KIND,'passed':True,'producer':PRODUCER,'worker_source':source,'source':source,'parent_source':PARENT,
        'request':request_pin,'ROOT_review':request['ROOT_review'],'fingerprint_transport_normalizer_source':NORMALIZER,'original_failed_owner':FAILED,'current_cleanup_join_failure':FAILED,
        'physical_cleanup_performed_in_this_invocation':False,'historical_cleanup_process0_inferred_or_written':False,
        'original_cleanup_report':failed['actual_cleanup_report'],'original_cleanup_worker_source':original['worker_source'],
        'original_report_parent_source':original['parent_source'],'actual_original_Popen_observer_source':PARENT,
        'original_actual_Popen_wait0_and_family_drain':failed['actual_cleanup_Popen_wait0_and_family_drain'],
        'original_clean_joined_guard':failed['actual_cleanup_guard'],'original_physical_cleanup_producer':original['producer'],
        'original_cleanup_empty_worker_log':empty_log,'original_physical_cleanup':failed['actual_cleanup_report'],
        'owner':original['owner'],'manifest':original['manifest'],'preserved_metadata':failed['retained_feature_metadata'],
        'historical_removed_exact_owned_files':removed,'historical_removed_owned_file_count':682,'all_original_owned_files_currently_absent':True,
        'removed_exact_owned_files':removed,'original_shared_oracle_untouched':True,
        'all_original_owned_files_absent':True,'verified_removed_exact_owned_files_count':682,
        'original_feature_files_only_deleted':True,'original_audio_source_checkpoint_journal_untouched':True,
        'feature_files_only_deleted':True,'physical_cleanup_scope':'historical_original_report_only',
        'current_committed_optimizer_updates':40,'current_checkpoint_pointer':failed['checkpoint_pointer'],
        'checkpoint_pointer':failed['checkpoint_pointer'],
        'current_checkpoint_metadata':failed['checkpoint_metadata'],'PRE_save_fingerprint':failed['PRE_save_fingerprint'],
        'next_draw_packet':failed['next_draw_packet'],'sampler_state_sha256':failed['sampler_state_sha256'],
        'journal_prefix':failed['journal_prefix'],'all_original_input_signatures':before,'checks':checks,
        'sampler_journal_frontier_advanced':False,'tensor_payload_bytes_read_or_parsed':0,'model_GPU_audio_or_delete_operations':0,
        'current40_fresh_restore_or_publication_claimed':False,'actual_new_helper_process0_still_required':True,
        'resource_observation':guard(),'guard_samples':samples,'peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began,
        'actual_process_IO_bytes':io_bytes()-INITIAL_IO,'at_utc':datetime.now(timezone.utc).isoformat()}
    raw=canonical(report);require(len(raw)<=1024**2,'Bounded final metadata report1MiB')
    with (output/'report.json').open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    timer.cancel();kernel.CloseHandle(job)
    print(json.dumps({'passed':True,'report':record(output/'report.json'),'checks':len(checks),'physical_cleanup_performed':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--folder',required=True)
    arguments=parser.parse_args();main(arguments.request,arguments.folder)
