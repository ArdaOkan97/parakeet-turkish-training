"""One review-gated metadata-only proof; no training admission or live mutation.

Original context/row/declaration methods and real readonly DB handles are used.
The FullRunSourceContext.__new__ constructor scaffold is explicitly a METHOD
STUB for integration authority, not repeated full population/hash admission.
"""
import argparse,ast,ctypes,gc,hashlib,json,os,shutil,sys,threading,time,traceback,types,weakref
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
import asr18_BPE9223_source_context_v2 as context
import asr18_BPE9223_private_scalar_prefetch_v1 as prefetch
import bpe_declared_geometry_index_v1 as geometry
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from bpe_broad_production_state_v1 import canonical,digest,CompletePassCursor,require
from asr18_fresh_single_open_metadata_signature_v2 import FreshSingleOpenSignature,UnsupportedFastSignature,SignatureHandleCloseFailure,INVALID_HANDLE_VALUE,ORIGINAL_SIGNATURE
from asr18_fresh_metadata_signature_scope_v2 import FreshMetadataSignatureScope
from asr18_observe_original_gc_collect_v2 import CollectScalarObservation,ObserveOriginalCollectScope,ORIGINAL_COLLECT
import asr18_observe_original_gc_collect_v2 as collect_module
F=Path(__file__).parent;OUT=F/'fresh_signature_collect_observation_actual_CPU_01'
INPUT=F/'exact_already_planned280_scalar_metadata_input_v1.json'
RUNTIME=F.parent/'verified_population_v1/final_runtime_bindings_9223_v1.json'
CONFIG=Path(r'E:\gpu_training_scratchpad\runs\BPE9223_resident_RAM_20261018_01\invocations\owner_000022\configuration.json')
BEGIN=time.monotonic();CASES=[]

def pin(path):
    p=Path(path).resolve();b=p.read_bytes();require(len(b)<=8*1024**2,'Compact/source input only')
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def load(path):
    p=Path(path);require(p.stat().st_size<=8*1024**2,'Bounded metadata JSON');return json.loads(p.read_text(encoding='utf-8'))
def check():require(time.monotonic()-BEGIN<180,'Finite180s metadata-only fixture')
def case(name,details=None):CASES.append({'name':name,'passed':True,'details':details})
def error_shape(call):
    try:return {'returned':call()}
    except OSError as e:return {'type':type(e).__name__,'errno':e.errno,'winerror':getattr(e,'winerror',None),'filename':e.filename,'message':str(e)}
def aliases_original():return all(m.signature is ORIGINAL_SIGNATURE for m in(context,geometry,prefetch))

class TrackingAPI:
    """Real Win32 operations except explicitly labelled METHOD_STUB failures."""
    def __init__(self,api):self.api=api;self.opens=0;self.closes=0;self.live=set();self.fake_final=None;self.fake_info_failure=False;self.fake_close_false=False
    def CreateFileW(self,*a):
        h=self.api.CreateFileW(*a)
        if h!=INVALID_HANDLE_VALUE:self.opens+=1;self.live.add(h)
        return h
    def GetFinalPathNameByHandleW(self,h,buffer,count,flags):
        if self.fake_final is not None:buffer.value=self.fake_final;return len(self.fake_final)
        return self.api.GetFinalPathNameByHandleW(h,buffer,count,flags)
    def GetFileInformationByHandle(self,*a):
        if self.fake_info_failure:ctypes.set_last_error(5);return 0
        return self.api.GetFileInformationByHandle(*a)
    def CloseHandle(self,h):
        v=self.api.CloseHandle(h)
        if v:self.closes+=1;self.live.remove(h)
        return 0 if self.fake_close_false else v

def synthetic_signatures():
    root=OUT/'synthetic';root.mkdir()
    paths=[root/'one data.sqlite',root/'iki_ş.sqlite',root/'three.sqlite']
    for i,p in enumerate(paths):p.write_bytes(bytes([65+i])*(17+i))
    expected=[ORIGINAL_SIGNATURE(p)for p in paths];signer=FreshSingleOpenSignature(expected)
    tracker=TrackingAPI(signer.api);signer.api=tracker
    for p,e in zip(paths,expected):require(signer._fresh_query(str(p.resolve()))==e,'Real fresh handle canonical/size/FILETIME equality')
    for _ in range(20):
        for p,e in zip(paths,expected):require(signer.signature(p)==e,'No stale/changed original metadata')
    require(tracker.opens==tracker.closes and not tracker.live,'All real fresh-query handles closed')
    case('REAL_Win32_fresh_handle_exact_three_fields_integer_time_unicode_spaces',{'opens':tracker.opens,'closes':tracker.closes})
    paths[0].write_bytes(b'new actual longer bytes')
    changed=signer.signature(paths[0]);require(changed==ORIGINAL_SIGNATURE(paths[0]) and changed!=expected[0],'No cached successful size/mtime')
    case('REAL_changed_file_current_signature_not_cached')
    (root/'sub').mkdir();alias=root/'sub/../one data.sqlite'
    require(signer.signature(alias)==ORIGINAL_SIGNATURE(alias),'Original fallback for unproved noncanonical spelling')
    case('REAL_dotdot_alias_original_fallback')
    original=signer.original;fallback=[]
    def counted(p):fallback.append(str(p));return original(p)
    signer.original=counted
    tracker.fake_final='\\\\?\\E:\\unproved_different_canonical_name'
    require(signer.signature(paths[1])==original(paths[1])and fallback,'Unproved final spelling uses original full function')
    tracker.fake_final=None;case('METHOD_STUB_final_name_mismatch_uses_ORIGINAL_not_blind_prefix_strip')
    tracker.fake_info_failure=True;n=len(fallback);require(signer.signature(paths[1])==original(paths[1])and len(fallback)==n+1,'Original fallback on API uncertainty')
    tracker.fake_info_failure=False;case('METHOD_STUB_information_API_error_uses_ORIGINAL')
    tracker.fake_close_false=True
    try:signer.signature(paths[1])
    except SignatureHandleCloseFailure:pass
    else:raise AssertionError('A failed close result must never be promoted through fallback')
    tracker.fake_close_false=False;require(not tracker.live,'Real handle actually closed before simulated false result')
    case('METHOD_STUB_CloseHandle_false_stops_no_leaked_real_handle')
    paths[2].unlink();require(error_shape(lambda:signer.signature(paths[2]))==error_shape(lambda:original(paths[2])),'Missing path original OS exception shape')
    case('REAL_deleted_path_original_exception_shape')
    rename_root=root/'rename-parent';rename_root.mkdir();rename_file=rename_root/'a.sqlite';rename_file.write_bytes(b'a')
    others=[root/'other-a',root/'other-b']
    for p in others:p.write_bytes(b'x')
    rename_signer=FreshSingleOpenSignature([ORIGINAL_SIGNATURE(p)for p in(rename_file,*others)])
    renamed=root/'renamed-parent';rename_root.rename(renamed)
    require(error_shape(lambda:rename_signer.signature(rename_file))==error_shape(lambda:original(rename_file)),'Parent rename cannot use cached canonical success')
    case('REAL_parent_rename_original_failure_semantics')
    # Reparse creation is attempted only in owned E scratch. Unsupported E FS or
    # privilege is recorded, not replaced with a claimed real reparse proof.
    target=root/'alias-target';target.mkdir();(target/'a.sqlite').write_bytes(b'b')
    link=root/'alias-parent';capability={'actually_created':False}
    try:
        os.symlink(target,link,target_is_directory=True);capability['actually_created']=True
        require(rename_signer.signature(link/'a.sqlite')==original(link/'a.sqlite'),'Real reparse alias original fallback')
        link.unlink();case('REAL_reparse_alias_ORIGINAL_fallback',capability)
    except OSError as e:
        capability.update(unsupported_type=type(e).__name__,winerror=getattr(e,'winerror',None),errno=e.errno)
        case('REAL_reparse_capability_UNSUPPORTED_no_equivalence_claim',capability)
    require(tracker.opens==tracker.closes and not tracker.live,'All synthetic candidate handles closed')
    return paths[:2]+[renamed/'a.sqlite']

def scaffold(signatures):
    """METHOD_STUB constructor only; real original lock class/paths/handles."""
    c=context.FullRunSourceContext9223.__new__(context.FullRunSourceContext9223);c.closed=False
    r=context.FinalBroadRows9223.__new__(context.FinalBroadRows9223);r.poisoned=False
    r.signatures=deepcopy(signatures[:2]);r.geometry=types.SimpleNamespace(signature=deepcopy(signatures[2]))
    r.metadata_locks=FinalMetadataReadLocks([v['path']for v in signatures]);c.resolver=r
    return c

def integration_negatives(signatures):
    c=scaffold(signatures)
    try:
        try:FreshMetadataSignatureScope(c,private_worker_closed=42)
        except ValueError:pass
        else:raise AssertionError('Noncallable closure check accepted')
        case('METHOD_STUB_noncallable_closure_rejected')
        original_lock=c.resolver.metadata_locks
        c.resolver.metadata_locks=types.SimpleNamespace(handles=list(original_lock.handles),paths=list(original_lock.paths))
        try:
            try:FreshMetadataSignatureScope(c,private_worker_closed=lambda:True)
            except ValueError:pass
            else:raise AssertionError('A fake lock object was accepted')
        finally:c.resolver.metadata_locks=original_lock
        case('METHOD_STUB_nonoriginal_lock_type_rejected')
        oldpaths=c.resolver.metadata_locks.paths;c.resolver.metadata_locks.paths=list(reversed(oldpaths))
        try:FreshMetadataSignatureScope(c,private_worker_closed=lambda:True)
        except ValueError:pass
        else:raise AssertionError('Wrong original lock path identity accepted')
        finally:c.resolver.metadata_locks.paths=oldpaths
        case('METHOD_STUB_wrong_lock_path_order_rejected')
        sentinel=RuntimeError('exact original body exception')
        try:
            with FreshMetadataSignatureScope(c,private_worker_closed=lambda:True):raise sentinel
        except RuntimeError as e:require(e is sentinel,'Original exception identity')
        require(aliases_original(),'Normal joined exception restores aliases')
        case('METHOD_STUB_body_exception_identity_and_restoration')
        r=c.resolver;r.population=context.BROAD_N;r.cache=OrderedDict();r.connections=[]
        def original_guard_failure():raise sentinel
        r.guard=original_guard_failure
        try:
            with FreshMetadataSignatureScope(c,private_worker_closed=lambda:True):r.row(1)
        except RuntimeError as e:require(e is sentinel and r.poisoned,'Original row guard exception and poisoning unchanged')
        else:raise AssertionError('Original row guard was bypassed')
        r.poisoned=False # Only reset this METHOD_STUB constructor fixture for remaining negatives.
        case('METHOD_STUB_constructor_original_row_guard_exception_identity_and_poisoning')
        joined=[False];scope=FreshMetadataSignatureScope(c,private_worker_closed=lambda:joined[0]);scope.__enter__()
        require(scope.__exit__(RuntimeError,sentinel,None)is False and scope.active and scope.deferred_to_abnormal_exit,'Unjoined abnormal scope honestly deferred')
        joined[0]=True;scope.__exit__(None,None,None);require(aliases_original(),'Explicit fixture cleanup after deferred negative')
        case('METHOD_STUB_unjoined_abnormal_defer_NO_normal_closure_claim')
        scope=FreshMetadataSignatureScope(c,private_worker_closed=lambda:True);scope.__enter__();foreign=lambda p:None
        context.signature=foreign
        try:scope.__exit__(None,None,None)
        except ValueError:pass
        else:raise AssertionError('Changed alias accepted as normal closure')
        require(context.signature is foreign,'Foreign replacement is not silently overwritten')
        context.signature=ORIGINAL_SIGNATURE;require(aliases_original(),'Fixture restores its own negative alteration')
        case('METHOD_STUB_changed_alias_strict_failure')
        def close_error():raise sentinel
        scope=FreshMetadataSignatureScope(c,private_worker_closed=close_error);scope.__enter__()
        require(scope.__exit__(RuntimeError,sentinel,None)is False,'Existing original body exception is not replaced by closure failure')
        scope.private_worker_closed=lambda:True;scope.__exit__(None,None,None)
        require(aliases_original(),'Fixture cleanup after callback negative')
        case('METHOD_STUB_closure_callback_failure_preserves_original_exception')
    finally:c.resolver.metadata_locks.close()
    require(not c.resolver.metadata_locks.handles,'All real scaffold locks closed')

def original280(inputs,bindings,signatures,plan):
    closed=threading.Event();c=scaffold(signatures);status={};counts={v['path']:0 for v in signatures};spy_lock=threading.Lock();in_rows=threading.Event()
    original_row=context.FinalBroadRows9223.row;original_declaration=geometry.DeclaredGeometryIndex.declaration
    scope=FreshMetadataSignatureScope(c,private_worker_closed=closed.is_set)
    try:
        with scope:
            query=scope.reader._fresh_query
            def spy(key):
                result=query(key)
                if in_rows.is_set():
                    with spy_lock:counts[key]+=1
                return result
            scope.reader._fresh_query=spy
            def worker():
                reader=None
                try:
                    reader=prefetch._ThreadOwnedBroadRows(bindings,signatures,guard=check)
                    require(context.FinalBroadRows9223.row is original_row and geometry.DeclaredGeometryIndex.declaration is original_declaration,'Original actual row/declaration method objects unchanged')
                    oracle=[r for u in plan['updates']for r in u['rows'][:14]]
                    require(len(oracle)==280 and len(reader.cache)==0,'One original first-pass280, unchanged32LRU')
                    final=plan['final_private_planning_sampler']['broad_cursor'];initial=deepcopy(final);initial['draws']-=280
                    cursor=CompletePassCursor(population=final['population'],identity=final['identity'],seed=final['seed'],passes=final['passes']);cursor.load_state_dict(initial)
                    rows={};ordinals=[];encoded=0;in_rows.set()
                    for group in range(20):
                        batch=cursor.preview(14)
                        require([d['ordinal']for d in batch]==inputs['original_ordered280_ordinals'][group*14:group*14+14],'Same cloned original cursor/order')
                        for expected in inputs['original_ordered_rows'][group*14:group*14+14]:
                            check();row=reader.row(expected['ordinal']);wire=canonical(row);encoded+=len(wire)
                            require(encoded<=prefetch.MAX_ROW_BYTES and row['source_id']==expected['source_id']and hashlib.sha256(wire).hexdigest()==expected['canonical_original_plan_row_sha256'],'Every full original row/labels/geometry matches frozen oracle')
                            rows[expected['ordinal']]=row;ordinals.append(expected['ordinal'])
                        cursor.commit(batch)
                    in_rows.clear();require(cursor.state_dict()==final and ordinals==inputs['original_ordered280_ordinals'],'No authoritative sampler or RNG advance')
                    require(counts=={v['path']:280 for v in signatures},'Original560 catalog-label+280 geometry fresh check counts')
                    status.update(passed=True,rows=280,rows_digest=digest(rows),encoded_bytes=encoded,signature_counts=counts,final_private_cursor_exact=True)
                except BaseException as e:status.update(passed=False,error=repr(e),traceback=traceback.format_exc())
                finally:
                    in_rows.clear()
                    if reader is not None:
                        reader.close();status['actual_worker_denywrite_handles_closed']=not reader.metadata_locks.handles
                    closed.set()
            t=threading.Thread(target=worker,name='candidate-original280-metadata');t.start()
            while t.is_alive():check();t.join(.05)
            require(status.get('passed')is True and closed.is_set(),str(status))
        require(scope.closed and aliases_original(),'Real worker joined before original aliases restored')
    finally:
        c.resolver.metadata_locks.close()
    case('REAL_original280_same_methods_labels_geometry_order_cursor_and_all840_checks',status)
    case('REAL_original280_worker_joined_readonly_connections_and_all_denywrite_closed')
    return status

def concurrent_signature_observation(signatures):
    paths=[v['path']for v in signatures];candidate=FreshSingleOpenSignature(signatures)
    locks=FinalMetadataReadLocks(paths);phases=[]
    try:
        for name,call in [('ORIGINAL_first',ORIGINAL_SIGNATURE),('CANDIDATE_second',candidate.signature)]:
            result=[None,None];start=threading.Barrier(3)
            def worker(i):
                try:
                    start.wait(timeout=10);b=time.perf_counter_ns();cpu=time.thread_time_ns()
                    for j in range(280):
                        if j%20==0:check()
                        k=j%3;require(call(paths[k])==signatures[k],'All same actual locked source signatures/oracle')
                    result[i]={'calls':280,'wall_ns':time.perf_counter_ns()-b,'thread_cpu_ns':time.thread_time_ns()-cpu}
                except BaseException as e:result[i]={'error':repr(e)}
            threads=[threading.Thread(target=worker,args=(i,),name='signature-fixed-'+str(i))for i in range(2)]
            for t in threads:t.start()
            began=time.perf_counter_ns();start.wait(timeout=10)
            for t in threads:
                while t.is_alive():check();t.join(.05)
            elapsed=time.perf_counter_ns()-began
            require(all(v is not None and 'error'not in v for v in result),'Bounded two-thread exact signatures')
            phases.append({'phase':name,'two_threads_total_calls':560,'phase_wall_ns':elapsed,'threads':result})
    finally:locks.close()
    eligible=phases[1]['phase_wall_ns']<=phases[0]['phase_wall_ns']
    case('REAL_fixed_Original_then_Candidate_two_threads_280calls_each_same3paths',{'phases':phases,'candidate_not_slower_in_THIS_bounded_order':eligible,
        'limits':'Fixed order and warm metadata, not randomized live contention/GC or physical disk proof. Concurrent thread wall times not additive. Correctness alone does not select production.'})
    return {'phases':phases,'candidate_not_slower_in_THIS_bounded_order':eligible}

def collect_proof():
    sentinel=object();seen=[];marker=RuntimeError('original exact METHOD_STUB exception')
    def original(*a,**k):seen.append((a,k));return sentinel
    obs=CollectScalarObservation(original);require(obs(2,x=sentinel)is sentinel and seen==[((2,),{'x':sentinel})],'Exact args/kwargs/return passthrough')
    require(obs.take_window()['calls']==1 and obs.take_window()['calls']==0,'Bounded reset scalar count')
    case('METHOD_STUB_collect_exact_once_args_kwargs_result_and_reset')
    def failing(*a,**k):raise marker
    obs=CollectScalarObservation(failing)
    try:obs()
    except RuntimeError as e:require(e is marker,'Original exception object unchanged')
    else:raise AssertionError('Original exception swallowed')
    require(obs.take_window()['raised_calls']==1,'Raised count only, no saved exception')
    case('METHOD_STUB_collect_original_exception_identity')
    def bad_clock():raise marker
    obs=CollectScalarObservation(original,clock=bad_clock);require(obs()is sentinel and obs.take_window()['clock_errors']==1,'Timer errors do not replace original result')
    obs=CollectScalarObservation(failing,clock=bad_clock)
    try:obs()
    except RuntimeError as e:require(e is marker,'Timer failure cannot replace original exception')
    case('METHOD_STUB_clock_errors_preserve_original_outcomes')
    depth=[0];nested=[]
    def recursive():
        nested.append(depth[0])
        if not depth[0]:depth[0]=1;obs();depth[0]=0
        return 7
    obs=CollectScalarObservation(recursive);require(obs()==7 and obs.take_window()['calls']==2 and nested==[0,1],'Reentrant call count, no doubleclaim of losttime')
    case('METHOD_STUB_reentrant_passthrough_nestedwall_explicit_NOT_losttime')
    captured=collect_module.ORIGINAL_COLLECT;collect_module.ORIGINAL_COLLECT=lambda:0
    try:
        try:ObserveOriginalCollectScope()
        except ValueError:pass
        else:raise AssertionError('A substituted non-builtin original was accepted')
    finally:collect_module.ORIGINAL_COLLECT=captured
    case('METHOD_STUB_production_scope_rejects_nonbuiltin_capture')
    before=(gc.get_threshold(),gc.get_freeze_count(),gc.isenabled())
    class Cycle:pass
    x=Cycle();x.self=x;ref=weakref.ref(x);del x
    scope=ObserveOriginalCollectScope()
    with scope as observer:
        value=gc.collect();require(type(value)is int and ref()is None,'Genuine original builtin collects owned temporary cycle')
        stats=observer.take_window();require(stats['calls']==1 and stats['raised_calls']==0,'One actual original builtin invocation')
    require(gc.collect is ORIGINAL_COLLECT and scope.restored and before==(gc.get_threshold(),gc.get_freeze_count(),gc.isenabled()),'Builtin/threshold/freeze/enabled exact on scope cleanup')
    case('REAL_original_builtin_owned_cycle_collect_no_GC_policy_change',stats)
    weak=weakref.ref(observer);del observer,scope;ORIGINAL_COLLECT();require(weak()is None,'Observer/scalar state not retained after scope')
    case('REAL_observer_lifetime_no_retained_frames_exception_or_temporary_cycle')
    try:
        with ObserveOriginalCollectScope():raise marker
    except RuntimeError as e:require(e is marker,'Original body exception through collect scope')
    require(gc.collect is ORIGINAL_COLLECT,'Exception scope cleanup restores builtin')
    case('METHOD_STUB_collect_scope_exception_identity_restoration')

def main():
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);a=p.parse_args();me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    review=load(a.ROOT_review)
    require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_fresh_single_open_signature_and_collect_observation_CPU_v1'
        and review['fixture_source']==pin(__file__)and Path(review['output']).resolve()==OUT.resolve(),'Exact frozen ROOT fixture scope')
    for v in review['input_pins']:require(pin(v['path'])==v,'Every input pin before actual job')
    require(not OUT.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Fresh CPU-only process-local environment')
    OUT.mkdir();status={'kind':'ASR18_fresh_single_open_signature_and_collect_observation_actual_CPU_v1','passed':False,
        'identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'launcher_identity':{'pid':me.ppid(),'create_time':me.parent().create_time(),'ppid':me.parent().ppid()},
        'ROOT_scope':pin(a.ROOT_review),'fixture_source':pin(__file__),'input_pins_before':review['input_pins'],'cases':CASES,
        'constructor_scaffold_METHOD_STUB_not_full_population_readmission':True,'actual_model_audio_GPU_native_or_authoritative_draw_operations':0,
        'sampling_or_GC_policy_or_production_environment_change':False}
    try:
        synthetic_signatures()
        runtime=load(RUNTIME);inputs=load(INPUT);configuration=load(CONFIG);plan=load(inputs['source_plan']['path'])
        require(pin(inputs['source_plan']['path'])==inputs['source_plan']and digest(plan)==inputs['source_plan_canonical_sha256'],'Same already-selected plan/oracle')
        require(configuration['planning_arguments']['DATA_evidence']['BPE9223_population']==pin(RUNTIME),'Same already-admitted source population')
        bindings={'final_catalog':runtime['source_catalog_unchanged'],'final_labels':runtime['databases']['broad_labels'],'declared_geometry_index':runtime['databases']['declared_geometry']}
        signatures=[{k:v[k]for k in('path','bytes','mtime_ns')}for v in bindings.values()]
        require([ORIGINAL_SIGNATURE(v['path'])for v in signatures]==signatures,'Original current immutable signature before proof')
        integration_negatives(signatures)
        status['actual_original280']=original280(inputs,bindings,signatures,plan)
        status['concurrent_observation']=concurrent_signature_observation(signatures)
        collect_proof()
        require(aliases_original()and gc.collect is ORIGINAL_COLLECT,'All scoped aliases restored')
        require([ORIGINAL_SIGNATURE(v['path'])for v in signatures]==signatures,'Actual metadata signatures unchanged')
        require(not any(k=='torch'or k.startswith('torch.')or k=='numpy'or k.startswith('numpy.')for k in sys.modules),'No model/numerical runtime imported')
        status['semantic_passed']=True
        status['candidate_timing_eligible']=status['concurrent_observation']['candidate_not_slower_in_THIS_bounded_order']
        status['passed']=status['candidate_timing_eligible']
        if not status['passed']:status['error']='Correct semantics but candidate slower in predeclared fixed two-thread observation; no production promotion or automatic retry'
    except BaseException as e:status.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        status['input_pins_after']=[pin(v['path'])for v in review['input_pins']]
        status['input_pins_before_after_exact']=status['input_pins_after']==review['input_pins']
        status['aliases_original_at_end']=aliases_original();status['collect_original_at_end']=gc.collect is ORIGINAL_COLLECT
        status['wall_seconds']=time.monotonic()-BEGIN
        status['passed']=bool(status['passed']and status['input_pins_before_after_exact']and status['aliases_original_at_end']and status['collect_original_at_end'])
        with(OUT/'report.json').open('x',encoding='utf-8',newline='\n')as f:json.dump(status,f,indent=2);f.write('\n')
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':status['passed'],'cases':len(CASES),'error':status.get('error')}),flush=True)
    require(status['passed'],'Preserve actual proof/performance failure, no unchanged retry')

if __name__=='__main__':main()
