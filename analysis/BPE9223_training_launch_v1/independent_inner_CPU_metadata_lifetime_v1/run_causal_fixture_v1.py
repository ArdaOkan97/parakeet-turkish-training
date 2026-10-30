"""Fresh-process metadata-only A/B of original CPU7 retained plan lifetimes.

Both cases execute unchanged real public7 and boundary APIs, with invalid ID !
stopping before any cache lock/owner creation. Existing reference actual0 is
verified, never recreated or claimed as a new child. No audio/GPU/feature job.
"""
import argparse, ctypes, gc, hashlib, importlib, json, os, subprocess, sys, threading, time
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad');P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_inner_CPU_metadata_lifetime_v1'
RUN=ROOT/'runs/BPE9223_weak2_clean1_20261018_02';WORK=RUN/'checkpoints/transient/p000000002'
PROPOSAL=WORK/'operational_CPU_window_configuration_000002.json'
REVIEW=ROOT/'analysis/BPE9223_training_launch_v1/operational_CPU_authorities_v2/ROOT_static_CPU_execution_resource_review.json'
REFERENCE0=WORK/'CPU/payload/reference_process/actual_owned_process0.json'
REFERENCE=WORK/'CPU/payload/reference/report.json'
args=argparse.ArgumentParser();args.add_argument('--case',choices=('original','release'));a=args.parse_args()
CASE=AREA/(a.case or 'supervisor');CASE.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research'),str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m=importlib.import_module('audit_and_fit_append_only_v5');m.AREA=CASE;g=m.Guard()
from ctypes import wintypes as w
class IO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):_fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):_fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True);api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
began=time.monotonic();timer=threading.Timer(60,lambda:os._exit(75));timer.daemon=True;timer.start()
def outer_guard():
    g.check();io=g.process.io_counters()
    assert time.monotonic()-began<60 and io.read_bytes+io.write_bytes-g.initial_IO<=8*1024**3
    assert g.process.memory_info().private<512_000_000
def record(path):
    path=Path(path).resolve();hashed=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):outer_guard();hashed.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashed.hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    path=Path(path);assert not path.exists();path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def authoritative():
    paths=[PROPOSAL,REVIEW,REFERENCE0,REFERENCE,RUN/'checkpoints/state.json',RUN/'checkpoints/catalog.sqlite',
        RUN/'control/worker_control_pool/registry.json',RUN/'durable/latest.json',RUN/'control/recovery_000002/acknowledgement.json',
        RUN/'control/p000000002.plan.json',WORK/'operational_bridge_000002.json']
    # Paths are fixed exact known artifacts, never a recursive source scan.
    return [record(path) for path in paths]
def cache_names():
    path=ROOT/'temp/BPE9223_recorded_room_CPU_windows_v1'
    return {'exists':path.exists(),'names':sorted(p.name for p in path.iterdir()) if path.exists() else []}

if a.case is None:
    before=authoritative();cache_before=cache_names();proposal=read(PROPOSAL)
    python=proposal['python'];assert record(python['path'])==python
    results=[];family_peak=0;completed_child_IO=0
    import psutil
    for name in ('original','release'):
        log=CASE/(name+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen([python['path'],'-X','utf8','-B',str(Path(__file__).resolve()),'--case',name],
                stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ))
            identity={'PID':child.pid,'create_time':psutil.Process(child.pid).create_time()};highest_IO=0
            try:
                while child.poll() is None:
                    outer_guard()
                    try:
                        process=psutil.Process(child.pid);private=process.memory_info().private
                        io=process.io_counters();highest_IO=max(highest_IO,io.read_bytes+io.write_bytes)
                    except psutil.NoSuchProcess:private=0
                    total=private+g.process.memory_info().private;family_peak=max(family_peak,total)
                    io=g.process.io_counters()
                    assert completed_child_IO+highest_IO+io.read_bytes+io.write_bytes-g.initial_IO<=8*1024**3
                    assert total<512_000_000 and log.stat().st_size<1024**2
                    time.sleep(.05)
                assert child.wait()==0, name+' metadata case failed unexpectedly; preserve evidence'
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
        item=read(AREA/name/'case_report.json')
        assert item['case']==name and item['observation_complete'] is True
        completed_child_IO+=max(highest_IO,item['actual_process_IO_bytes'])
        item={'case':name,'actual_wait_code':child.returncode,'launcher_identity':identity,
            'report':record(AREA/name/'case_report.json'),'observation':item}
        results.append(item)
    assert authoritative()==before and cache_names()==cache_before
    report={'kind':'ASR18_independent_inner_CPU_metadata_lifetime_two_fresh_process_cases_v1','passed':True,
        'producer':g.main_identity,'source':record(__file__),'conditions':results,'authoritative_pins_unchanged':before,
        'cache_top_level_unchanged':cache_before,'actual_case_processes':2,'new_reference_audio_children':0,
        'GPU_model_audio_or_cache_owner_operations':False,'inner_limit_unchanged_bytes':64*1024**2,
        'entire_fixture_family_peak_private_bytes':family_peak,'hard_entire_family_bytes':512_000_000,
        'elapsed_seconds':time.monotonic()-began,'actual_supervisor_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO,
        'actual_case_process_IO_bytes_conservatively_summed':completed_child_IO,
        'all_cases_safe_name_reached':all(x['observation']['safe_name_rejection_reached'] for x in results),
        'original_failed_release_reached_safe_name':not results[0]['observation']['safe_name_rejection_reached'] and results[1]['observation']['safe_name_rejection_reached'],
        'production_controller_success_or_absolute_memory_fit_claimed':False}
    write(AREA/'causal_fixture_report_v1.json',report)
    g.publish(phase='two_fresh_metadata_cases_COMPLETE',GPU_used=False)
    timer.cancel();print(json.dumps({'passed':True,'report':record(AREA/'causal_fixture_report_v1.json'),
        'observations':[{'case':x['case'],'safe_name_reached':x['observation']['safe_name_rejection_reached'],
            'first_failure':x['observation']['first_failure'],'stages':x['observation']['stages']} for x in results]}))
else:
    controller=importlib.import_module('run_ASR18_BPE9223_CPU_window_v7')
    core=controller.core;public=controller.public
    core.environment()
    from asr18_BPE9223_source_context_v4 import verify_CPU_configuration,verify_CPU_prior_boundary
    stages={};phase='startup';first_failure=None
    def observe():
        outer_guard();value=core.resource_guard();private=value['worker_private_memory']['private_bytes']
        stat=stages.setdefault(phase,{'samples':0,'minimum_private_bytes':private,'peak_private_bytes':private,'last_private_bytes':private})
        stat['samples']+=1;stat['minimum_private_bytes']=min(stat['minimum_private_bytes'],private)
        stat['peak_private_bytes']=max(stat['peak_private_bytes'],private);stat['last_private_bytes']=private
        return value
    def guard():
        value=observe()
        core.require(value['worker_private_memory']['private_bytes']<=64*1024**2,'Unchanged64MiB parent reserve')
        return value
    reached=False;error=None;summary=None
    proposal_pin=core.record(PROPOSAL);proposal=core.read_bound(proposal_pin)
    review_pin=core.record(REVIEW);review=core.read_bound(review_pin,core.resource_guard)
    try:
        phase='genuine_startup_configuration'
        reviewed=verify_CPU_configuration(proposal_pin,review_pin,guard=core.resource_guard)
        assert core.same(reviewed,review['configuration'])
        assert core.same(proposal['sources']['parent'],core.record(controller.__file__))
        assert core.same(proposal['python'],core.record(sys.executable))
        phase='verify_source_contract';source,plan=public.verify_source_contract(proposal['source_contract'],guard=guard)
        numerical=core.read_bound(source['numerical_CPU_contract'],guard)
        phase='original_initial_full_rebuild'
        rebuilt=core.bridge.bridge_plan(plan['coordinator_plan'],DATA_evidence=plan['DATA_evidence'],
            session_binding=plan['session'],window_index=plan['window_index'],prior_window_boundary=plan['prior_window_boundary'],
            verify_prior_boundary=lambda boundary:verify_CPU_prior_boundary(boundary,guard=guard),guard=guard)
        assert core.same(rebuilt,plan)
        # Stream selected row/target identities, without another full JSON copy.
        hashed=hashlib.sha256()
        for update in plan['updates']:
            for row in update['rows']:hashed.update(core.canonical(row))
        summary={'bridge':source['bridge'],'coordinator_plan':plan['coordinator_plan'],
            'session':plan['session'],'phase':plan['phase'],'updates':len(plan['updates']),
            'rows':sum(len(u['rows']) for u in plan['updates']),'ordered_full_row_target_sha256':hashed.hexdigest(),
            'initial_sampler_sha256':plan['initial_sampler_sha256'],'final_sampler_sha256':plan['final_sampler_sha256']}
        phase='existing_real_reference0_verification'
        public.verify_reference_window(core.record(REFERENCE,guard),core.record(REFERENCE0,guard),
            contract_pin=proposal['source_contract'],guard=guard)
        phase='retained_metadata_before_create';observe()
        if a.case=='release':
            del rebuilt,plan;gc.collect();phase='unused_metadata_released_before_create';observe()
        phase='real_public7_create_window_independent_rebuild'
        public.create_window(proposal['source_contract'],window_id='!',frontend=numerical['frontend'],guard=guard,
            verify_prior_boundary=lambda boundary:verify_CPU_prior_boundary(boundary,guard=guard))
        raise AssertionError('Unsafe ID unexpectedly created a window')
    except ValueError as exc:
        error=str(exc);first_failure={'phase':phase,'error':error,'private_bytes':g.process.memory_info().private}
        if error=='Fresh exact safe window ID':reached=True
        elif error!='Unchanged64MiB parent reserve':raise
    assert error is not None
    result={'kind':'ASR18_inner_CPU7_metadata_lifetime_causal_case_v1','case':a.case,'observation_complete':True,
        'producer':g.main_identity,'source':record(__file__),'CPU_controller_source':core.record(controller.__file__),
        'actual_interpreter':core.record(sys.executable),'proposal':proposal_pin,'ROOT_review':review_pin,
        'existing_reference_report':core.record(REFERENCE),'existing_reference_actual0':core.record(REFERENCE0),
        'safe_name_rejection_reached':reached,'first_failure':first_failure,'stages':stages,'full_plan_identity':summary,
        'inner64MiB_guard_relaxed':False,'tracemalloc_used':False,'cache_owner_or_feature_window_created':False,
        'new_reference_audio_child_or_GPU_job':False,'elapsed_seconds':time.monotonic()-began,
        'peak_private_bytes':g.peak,'actual_process_IO_bytes':g.process.io_counters().read_bytes+g.process.io_counters().write_bytes-g.initial_IO,
        'scope_limits':['Fixture guard/import and small measurement aggregates add common observer overhead.',
            'The real existing reference0 is reverified; no vanished run_leaf local handle or admission dictionary is invented.',
            'Reaching unsafe-ID rejection proves the real metadata rebuild completed, not frontend or production admission.']}
    write(CASE/'case_report.json',result);g.publish(phase='metadata_case_COMPLETE',case=a.case,GPU_used=False)
    timer.cancel();print(json.dumps({'case':a.case,'safe_name_reached':reached,'first_failure':first_failure}))
