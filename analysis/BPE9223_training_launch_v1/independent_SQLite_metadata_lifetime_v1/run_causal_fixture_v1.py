"""Read-only actual-parent metadata A/B/A cache experiment; no owner mutation.

Original context2 constructors, source checks, plan and next-draw APIs execute.
Only the fixture-owned live SQLite connections receive shrink_memory. No cache
limit, reader, scientific state or on-disk DB is changed. Instrumented private
usage is a differential result, not a training admission under the 64MB gate.
"""
import copy, ctypes, gc, hashlib, importlib, json, os, sys, threading, time, tracemalloc
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad');PROJECT=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20')
AREA=ROOT/'analysis/BPE9223_training_launch_v1/independent_SQLite_metadata_lifetime_v1'
sys.path[:0]=[str(PROJECT/'outputs/parakeet_turkish'),str(PROJECT/'work/research'),str(ROOT/'analysis/Turkish_labels_BPE_circumflex_v1')]
m=importlib.import_module('audit_and_fit_append_only_v5');m.AREA=AREA
g=m.Guard();g.max_seconds=180;g.maximum_bytes=8*1024**3
from ctypes import wintypes as w
class JobIO(ctypes.Structure):_fields_=[(x,ctypes.c_ulonglong) for x in ('r','w','o','rb','wb','ob')]
class Basic(ctypes.Structure):_fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',w.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('active',w.DWORD),('affinity',ctypes.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
class Limits(ctypes.Structure):_fields_=[('basic',Basic),('io',JobIO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
limits=Limits();limits.basic.flags=0x100|0x200;limits.process_memory=512_000_000;limits.job_memory=512_000_000
api=ctypes.WinDLL('kernel32',use_last_error=True);api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
if not api.SetInformationJobObject(g.job,9,ctypes.byref(limits),ctypes.sizeof(limits)):raise ctypes.WinError(ctypes.get_last_error())
began=time.monotonic();timer=threading.Timer(180,lambda:os._exit(75));timer.daemon=True;timer.start()
def guard():
    g.check();assert time.monotonic()-began<180 and g.process.memory_info().private<512_000_000
    counters=g.process.io_counters();assert counters.read_bytes+counters.write_bytes-g.initial_IO<=8*1024**3
def record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(1024**2):guard();h.update(block)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def read(pin):
    assert record(pin['path'])==pin
    return json.loads(Path(pin['path']).read_text(encoding='utf-8-sig'))
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def private():return int(g.process.memory_info().private)
measurements=[];checks=[]
def check(name,value):guard();assert value,name;checks.append({'name':name,'passed':True})
def sample(name):
    guard();counters=g.process.io_counters();current=private()
    traced,peak=tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (None,None)
    value={'stage':name,'seconds':time.monotonic()-began,'private_bytes':current,
        'traced_live_bytes':traced,'traced_peak_bytes':peak,
        'tracemalloc_internal_bytes':tracemalloc.get_tracemalloc_memory() if tracemalloc.is_tracing() else 0,
        'allocated_Python_blocks':sys.getallocatedblocks(),'gc_counts':list(gc.get_count()),
        'actual_process_IO_bytes':counters.read_bytes+counters.write_bytes-g.initial_IO}
    measurements.append(value);return value

sample('fixture_guard_imports_before_tracemalloc')
tracemalloc.start(1)
# Same module import families as the actual parent, without constructing an owner.
parent=importlib.import_module('run_ASR18_BPE9223_full_run_owner_v6')
controller=importlib.import_module('asr18_BPE9223_full_run_controller_v6')
source=importlib.import_module('asr18_BPE9223_source_context_v2')
bridge=importlib.import_module('fastmss_BPE9223_plan_bridge_v7')
release=importlib.import_module('asr18_BPE9223_parent_SQLite_cache_release_v1')
from bpe_streaming_checkpoint_semantics_v2 import select,decode_scalar_tree
from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
from bpe_declared_geometry_index_v1 import signature
check('exact_actual_parent_context2_helper1',source.FullRunSourceContext9223 is release.FullRunSourceContext9223 and Path(source.__file__).name=='asr18_BPE9223_source_context_v2.py')
check('exact_frozen_helper1',record(release.__file__)['sha256']=='09a0bb269e8d8406dc502c22abb39170de4ff39bdffdc2ceb7e28e88e8e6e2ad')
check('no_model_or_numerical_runtime_import',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
sample('actual_parent_module_imports')
closure_pin={'path':str(ROOT/'analysis/BPE9223_training_launch_v1/full_training_operational_resource_actual_02/actual_owned_resource_stop_TOOL1_v2.json'),
    'bytes':12045,'sha256':'f2de61fab27fe24935a76702fa0e426e9c7121a1d00436c4770ff26d1faacc91'}
closure=read(closure_pin);config=read(closure['configuration']);request=read(closure['resume_request'])
def saved_sampler():
    pointer=read(closure['checkpoint_pointer']);metadata=read(pointer['checkpoint'])
    state=normalize_schedule_state(decode_scalar_tree(select(metadata['tree'],'sampler')))
    assert digest(state)==pointer['sampler_state_sha256']
    return state
saved=saved_sampler();expected_state=digest(saved)
expected_next=request['owner_state']['next_draw_packet_sha256']
del request;gc.collect();sample('actual_config_and_normalized_saved_sampler20')
ctx=None;helper_results=[];manual=[];rounds=[];connection_states=[];fifth=None
try:
    ctx=source.FullRunSourceContext9223(config['planning_arguments'],guard=guard)
    schedule=ctx.new_schedule();schedule.load_state_dict(saved)
    check('actual_context_is_original_parent_class',type(ctx) is source.FullRunSourceContext9223)
    check('saved_sampler20_not_source0',schedule.committed_updates()==20 and digest(schedule.state_dict())==expected_state)
    targets=release._targets(ctx,guard)
    for name,connection,binding,settings,before in targets:
        check('original_quiescent_reader_'+name,settings==(1,2,-4096) and not connection.in_transaction and connection.total_changes==0)
        connection_states.append({'name':name,'path':str(Path(binding['path']).resolve()),'signature':before,'settings':list(settings),'total_changes':0})
    fifth_connection=ctx.cleaner_rows.source._db
    fifth={'settings':[fifth_connection.execute('PRAGMA '+key).fetchone()[0] for key in ('query_only','temp_store','cache_size')],
        'signature':signature(ctx.cleaner_rows.source.directory/'catalog.sqlite'),'cache_release_exercised':False}
    sample('genuine_context_constructed_four4MiB_readers_plus_source_index')

    def expected_plan_digests():
        expected=read(closure['p2_plan'])
        original_bridge_pin=next(pin for pin in closure['p2_metadata_files_preserved'] if Path(pin['path']).name=='operational_bridge_000002.json')
        old_bridge=read(original_bridge_pin)
        return digest(expected),digest(old_bridge['updates'])
    expected_plan=None;expected_normalized=None
    def plan_and_normalize(label):
        before=digest(schedule.state_dict())
        plan=ctx.plan(schedule,completed_boundaries=0)
        sample(label+'_plan_retained')
        normalized=bridge.normalize_updates(plan,population=2631115,guard=guard,mode=bridge.FULL_MODE)
        result={'plan_digest':digest(plan),'normalized_bridge_digest':digest(normalized),
            'selected_full_rows_digest':digest([u['rows'] for u in plan['updates']]),
            'updates':len(plan['updates']),'TR_draws':sum(len(u['TR_draws']) for u in plan['updates']),
            'replay_draws':sum(len(u['replay_rows']) for u in plan['updates'])}
        sample(label+'_plan_and_normalized_bridge_retained')
        assert before==digest(schedule.state_dict())==expected_state
        del plan,normalized;gc.collect();sample(label+'_all_local_metadata_released_fullGC')
        return result
    first=plan_and_normalize('A1_original_p2_planning')
    expected_plan,expected_normalized=expected_plan_digests();gc.collect()
    check('p2_plan_exact_against_actual_preserved_plan',first['plan_digest']==expected_plan)
    check('full_normalized_bridge_exact_against_actual_preserved_bridge',first['normalized_bridge_digest']==expected_normalized)
    check('same_full20_280TR40replay_provisional',first['updates']==20 and first['TR_draws']==280 and first['replay_draws']==40)
    next_before=source.full_run_next_draw_packet(schedule,ctx,optimizer_updates=20,guard=guard)
    check('exact_actual_saved_nextdraw_before_release',digest(next_before)==expected_next and digest(schedule.state_dict())==expected_state)
    del next_before;gc.collect();sample('A1_before_perDB_manual_release')
    for name,connection,binding,settings,before in release._targets(ctx,guard):
        start=sample('B1_'+name+'_before_shrink')
        assert list(connection.execute('PRAGMA shrink_memory'))==[]
        assert tuple(connection.execute('PRAGMA '+key).fetchone()[0] for key in ('query_only','temp_store','cache_size'))==settings
        assert signature(binding['path'])==before and not connection.in_transaction and connection.total_changes==0
        gc.collect();end=sample('B1_'+name+'_after_shrink_fullGC')
        manual.append({'name':name,'private_delta_bytes':end['private_bytes']-start['private_bytes'],
            'traced_live_delta_bytes':end['traced_live_bytes']-start['traced_live_bytes'],
            'signature_and_settings_unchanged':True})
    second=plan_and_normalize('A2_exact_requery_after_manual_release')
    check('A2_same_full_plan_rows_targets_and_bridge',second==first)
    sample('B2_before_actual_frozen_helper1')
    helper_results.append(release.release_owned_readonly_caches(ctx,guard=guard,measure=private))
    gc.collect();sample('B2_after_actual_frozen_helper1_fullGC')
    third=plan_and_normalize('A3_exact_requery_after_actual_helper')
    check('A3_same_full_plan_rows_targets_and_bridge',third==first)
    sample('B3_before_actual_helper_repeat')
    helper_results.append(release.release_owned_readonly_caches(ctx,guard=guard,measure=private))
    gc.collect();sample('B3_after_actual_helper_repeat_fullGC')
    packet=source.full_run_next_draw_packet(schedule,ctx,optimizer_updates=20,guard=guard)
    check('exact_nextdraw_and_sampler_history_after_all_releases',digest(packet)==expected_next and digest(schedule.state_dict())==expected_state)
    del packet;gc.collect()
    for name,connection,binding,settings,before in release._targets(ctx,guard):
        check('same_locked_connection_signature_policy_'+name,signature(binding['path'])==before and connection.total_changes==0)
    check('fifth_cleaner_source_index_policy_unchanged',fifth['settings']==[fifth_connection.execute('PRAGMA '+key).fetchone()[0] for key in ('query_only','temp_store','cache_size')] and signature(ctx.cleaner_rows.source.directory/'catalog.sqlite')==fifth['signature'])
    rounds=[first,second,third]
    live_context_counts={'broad_row_cache':len(ctx.resolver.cache),'cleaner_source_row_cache':len(ctx.cleaner_rows.source._cache),
        'replay_evidence_rows':len(ctx.replay_evidence),'owned_connections_tested':4,'extra_source_index_connections_not_released':1}
    sample('verified_final_live_context_before_close')
    # One final allocation summary only, after the A/B/A private measurements.
    snapshot=tracemalloc.take_snapshot()
    top=[{'traceback':str(stat.traceback),'size_bytes':stat.size,'count':stat.count} for stat in snapshot.statistics('lineno')[:20]]
    del snapshot;gc.collect();sample('tracemalloc_summary_observer_overhead_after_experiment')
finally:
    if ctx is not None:ctx.close()
sample('all_fixture_owned_context_readers_closed')
tracemalloc.stop();gc.collect();sample('tracemalloc_stopped_observer_overhead_released')
guard();check('no_model_audio_or_numerical_runtime_at_finish',not any(x in sys.modules for x in ('torch','numpy','tokenizers','soundfile')))
report={'kind':'ASR18_independent_actual_parent_SQLite_cache_causal_A_B_A_fixture_v1','owner':'/root/label_bpe_audit','passed':True,
    'producer':g.main_identity,'source':record(__file__),'actual_failed_boundary':closure_pin,'configuration':closure['configuration'],
    'actual_parent_source':record(parent.__file__),'actual_context_source':record(source.__file__),'proposed_release_helper':record(release.__file__),
    'checks':checks,'check_count':len(checks),'measurements':measurements,'manual_per_connection_release':manual,
    'actual_helper_replays':helper_results,'exact_round_digests':rounds,'live_context_object_counts':live_context_counts,
    'fifth_source_index_observation':fifth,'top_traced_live_allocations_after_experiment':top,
    'initial_connection_states':connection_states,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':g.peak,
    'hard_leaf_limit_bytes':512_000_000,'maximum_seconds':180,'maximum_actual_process_IO_bytes':8*1024**3,
    'actual_model_tensor_payload_audio_GPU_or_authoritative_draw_operations':False,'actual_owner_ACK_protect_publication_or_namespace_mutation':False,
    'connections_closed_only_at_fixture_end':True,'cache_limits_query_only_or_resource_guards_changed':False,
    'production_admission_or_absolute64MB_fit_proven':False,
    'scope_limits':['Tracemalloc and fixture report objects add observer overhead; differential releases are the causal result.',
        'Original context constructor performs its existing immutable metadata hashes and original replay opaque binding/label reads; no new corpus audit is implemented.',
        'SQLite cache release is tested on fixture-owned read-only connections only; actual stopped owner and namespace are unchanged.',
        'XZ/native allocator contribution is not inferred from this SQLite experiment.']}
out=AREA/'causal_SQLite_fixture_report_v1.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g.publish(phase='SQLite_causal_fixture_COMPLETE',check_count=len(checks),tensor_payload_bytes_read=0);timer.cancel()
print(json.dumps({'passed':True,'report':record(out),'check_count':len(checks),'elapsed_seconds':report['elapsed_seconds'],'peak_private_bytes':g.peak,'manual_deltas':manual}))
