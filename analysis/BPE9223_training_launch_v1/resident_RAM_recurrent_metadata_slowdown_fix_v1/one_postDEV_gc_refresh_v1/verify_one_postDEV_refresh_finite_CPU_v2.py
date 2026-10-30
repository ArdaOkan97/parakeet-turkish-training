"""UNEXECUTED focused finite CPython proof proposal; NO production operation.

Real CPython builtins/cycles/weakrefs and a joined toy idle worker are tested.
Scientific GPU fingerprints, resource callbacks, DEV and DB authority are
explicit METHOD_STUBs. The original prefetch close method is AST-extracted
without alteration and used with a labelled constructor/reader double.
"""
import argparse, ast, copy, gc, hashlib, json, os, queue, random, sys, threading, time, traceback, weakref
from pathlib import Path
import psutil

F=Path(__file__).parent
FIX=F.parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish'
OUTPUT=F/'finite_one_postDEV_refresh_actual_CPU_01'
sys.path[:0]=[str(F),str(FIX)]


def require(value,message):
    if not value:raise ValueError(message)


def pin(path):
    p=Path(path).resolve();raw=p.read_bytes();require(len(raw)<=8*1024**2,'Compact/source inputs only')
    return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n')as out:
        json.dump(value,out,indent=2,ensure_ascii=False,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())


def barrier(expected=4000):
    value={'kind':'ASR18_actual_first_periodic_DEV_refresh_barrier_v1','updates':expected,
           'capture_contract':'original_bounded_live_fingerprint_and_identity_manifest_v1'}
    for key in ('genuine_periodic_DEV_returned_and_recorded','checkpoint_full_persistence_same_frontier',
        'feature_CPU_actual_wait0_joined_pipes_and_both_credit_sets_closed','DEV_preservation_and_phase_monitor_join_succeeded',
        'optimizer_idle_unpoisoned','scalar_empty_joined_closed_errorNone_worker_DB_and_locks_closed',
        'old_scalar_object_and_frame_dropped','parent_original_context_and_locks_continuous','no_live_packet_or_arrays',
        'checkpoint_writer_and_completed_aliases_dropped','queued_original_same_object_and_pin','main_Monitor_active',
        'stream_is_None','future_ticket_is_None','resolver_is_original'):value[key]=True
    return value


def science_METHOD_STUB():
    # Known acyclic scalar/hash bookkeeping tree only; not a tensor/GPU proof.
    return {'proof':'METHOD_STUB_scientific_GPU_fingerprint','207_ids':list(range(207)),
            'three_Adam_215_slots':[17,6,192],'full854_semantic_hash':'a'*64,
            'four_RNG':['b'*64]*4,'journal':'c'*64,'clocks':[68250,6000,66250],
            'both_samplers_and_nextdraw':'d'*64,'queued_plan':'e'*64}


class StubHarness:
    def __init__(self):self.order=[];self.receipts=[];self.captures=0;self.compare_calls=0
    def guard(self):self.order.append('guard')
    def admit(self,n):
        require(n==16777216,'Exact separate16MiB bookkeeping proposal')
        self.order.append('admit');return {'passed':True,'kind':'METHOD_STUB_full_original_plus_bookkeeping_admission'}
    def capture(self):self.order.append('capture');self.captures+=1;return science_METHOD_STUB()
    def compare(self,a,b):self.compare_calls+=1;require(a==b,'METHOD_STUB scientific equality')
    def op(self,name,result=None):
        def call():self.order.append(name);return result
        return call
    def operations(self):return {'unfreeze':self.op('unfreeze'),'collect':self.op('collect',7),
                                  'freeze':self.op('freeze'),'get_freeze_count':self.op('get_freeze_count',1)}
    def emit(self,value):self.receipts.append(copy.deepcopy(value))
    def run(self,machine,b=None,**extra):
        args={'capture':self.capture,'compare':self.compare,'guard':self.guard,
              'admit':self.admit,'operations':self.operations(),'emit':self.emit};args.update(extra)
        return machine.execute(barrier()if b is None else b,**args)


def expected_error(call,expected=ValueError,identity=None):
    try:call()
    except BaseException as error:
        require(isinstance(error,expected),'Expected strict negative type, observed '+type(error).__name__)
        if identity is not None:require(error is identity,'Exact callback/guard exception identity propagates')
        name=type(error).__name__;message=str(error);error.__traceback__=None
        return {'type':name,'message':message}
    raise AssertionError('Strict negative unexpectedly passed')


class Payload:
    pass


def extracted_original_close():
    path=S/'asr18_BPE9223_private_scalar_prefetch_v1.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='PrivateScalarPrefetch')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='close')
    def main_thread():require(threading.current_thread()is threading.main_thread(),'Original main-thread close')
    names={'main_thread':main_thread,'require':require}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),names)
    return names['close'],ast.dump(node,include_attributes=False)


class ToyIdleWorker:
    """Constructor/reader doubles; real thread with retained original-shaped locals."""
    active=0
    maximum_active=0
    generation=0
    close=None
    def __init__(self,close_method):
        self.original=object();self.context=type('ContextDouble',(),{})();self.context.resolver=self.original
        self.requests=queue.Queue(1);self.results=queue.Queue(1);self.stop=threading.Event()
        self.worker_status={'closed':False,'error':None};self.closed=False;self.held_result=None;self.outstanding=None
        self.adopted=threading.Event();self.started=threading.Event();self.reader_closed=threading.Event()
        ToyIdleWorker.generation+=1;self.generation=ToyIdleWorker.generation
        require(type(self).close is close_method,'Original close method is class-bound, no artificial self/bound-method cycle')
        self.thread=threading.Thread(target=ToyIdleWorker.worker,
            args=(self.requests,self.results,self.stop,self.worker_status,self.adopted,self.started,self.reader_closed),
            name='NEW-toy-idle-scalar-reader',daemon=True)
        self.thread.start();require(self.started.wait(2),'Real toy thread started')
    @staticmethod
    def worker(requests,results,stop,status,adopted,started,reader_closed):
        ToyIdleWorker.active+=1;ToyIdleWorker.maximum_active=max(ToyIdleWorker.maximum_active,ToyIdleWorker.active)
        started.set()
        try:
            request=requests.get(timeout=2)
            cursor=request['cursor'];rows=request['rows'];row=rows[0]
            results.put({'sequence':request['sequence']},timeout=2)
            adopted.set()
            while not stop.wait(.01):pass
        except BaseException as error:status['error']=error
        finally:
            reader_closed.set();status['closed']=True;ToyIdleWorker.active-=1
    def prepare_and_adopt(self,payload):
        self.outstanding={'generation':self.generation,'sequence':1}
        self.requests.put({'sequence':1,'cursor':payload,'rows':[payload]})
        require(self.adopted.wait(2),'Toy result ready')
        result=self.results.get(timeout=2);require(result=={'sequence':1},'Ordered toy result')
        ticket=self.outstanding;self.outstanding=None;self.held_result=None
        return ticket
    def require_current_ticket(self,ticket):
        require(ticket['generation']==self.generation,'Never carry a stale old-worker ticket')


def main():
    p=argparse.ArgumentParser();p.add_argument('--ROOT-review',required=True);a=p.parse_args()
    review=json.loads(Path(a.ROOT_review).read_text(encoding='utf-8'))
    require(review['accepted']is True and review['reviewer']=='/root'
        and review['scope']=='ASR18_one_postDEV_refresh_finite_CPU_v1'
        and review['fixture_source']==pin(__file__)and Path(review['output_path']).resolve()==OUTPUT.resolve(),
        'Exact future ROOT finite scope/source/output; inactive preparation scope is insufficient')
    require(not OUTPUT.exists(),'One fresh finite output; no automatic retry')
    for value in review['input_pins']:require(pin(value['path'])==value,'Frozen source/input equality before proof')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'
        and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
        'Isolated E CPU-only process environment')
    require(not any(x in sys.modules for x in ('torch','numpy','av','transformers')),'No scientific/audio/device runtime')
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    began=time.monotonic();cases=[];workers=[];report=None;failure=None;original_alias=gc.collect
    OUTPUT.mkdir()
    def guard():require(time.monotonic()-began<100,'Finite internal100s; external120s remains authoritative')
    def case(name,call):
        guard();start=time.monotonic();detail=call();cases.append({'case':name,'passed':True,'detail':detail,
                                                                 'wall_seconds':time.monotonic()-start})
    try:
        import asr18_one_postDEV_gc_refresh_v2 as h
        from asr18_observe_original_gc_collect_v2 import ObserveOriginalCollectScope
        require(h.ORIGINAL_COLLECT is original_alias and gc.get_freeze_count()==0,'Fresh isolated CPython generation')
        def basic():
            machine=h.RefreshSequence(0);stub=StubHarness();result=stub.run(machine)
            require(machine.completed and result['operations_completed']==['unfreeze','captured_original_full_collect','freeze']
                and [x for x in stub.order if x in ('unfreeze','collect','freeze')]==['unfreeze','collect','freeze']
                and stub.captures==2 and stub.compare_calls==1,'One exact stub sequence/capture pair')
            expected_error(lambda:stub.run(machine));return result
        case('METHOD_STUB_exact_one_event_sequence_and_second_event_rejection',basic)
        for key,value in barrier().items():
            if type(value)is bool:
                def bad_field(key=key):
                    b=barrier();b[key]=False;m=h.RefreshSequence(0);s=StubHarness()
                    result=expected_error(lambda:s.run(m,b));require(not s.order and not m.attempted,'Reject before GC operation')
                    return {'field':key,'error':result}
                case('METHOD_STUB_barrier_false_'+key,bad_field)
        for value,label in ((3999,'wrong_frontier'),(8000,'second_periodic'),('4000','wrong_type')):
            case('METHOD_STUB_'+label,lambda value=value:expected_error(lambda:StubHarness().run(h.RefreshSequence(0),barrier(value))))
        b=barrier();b['capture_contract']='identity_version_only'
        case('METHOD_STUB_version_only_not_semantic_proof',lambda:expected_error(lambda:StubHarness().run(h.RefreshSequence(0),b)))
        for op in ('unfreeze','collect','freeze'):
            def failed_operation(op=op):
                error=RuntimeError('METHOD_STUB exact '+op);m=h.RefreshSequence(0);s=StubHarness();ops=s.operations()
                def fail():raise error
                ops[op]=fail;result=expected_error(lambda:s.run(m,operations=ops),RuntimeError,error)
                require(m.attempted and not m.completed and s.receipts[-1]['passed']is False,'Fatal recorded partial failure')
                return {'exception':result,'operations_completed':list(m.operations_completed)}
            case('METHOD_STUB_operation_failure_'+op,failed_operation)
        def guard_failure():
            error=InterruptedError('METHOD_STUB original guard');s=StubHarness();m=h.RefreshSequence(0);count=0
            def fail_after_unfreeze():
                nonlocal count;count+=1
                if count==3:raise error
            result=expected_error(lambda:s.run(m,guard=fail_after_unfreeze),InterruptedError,error)
            require(m.operations_completed==['unfreeze'],'No fallback collect/freeze after guard error')
            return result
        case('METHOD_STUB_original_guard_identity_after_partial_unfreeze',guard_failure)
        def compare_failure():
            error=ValueError('METHOD_STUB full scientific mismatch');s=StubHarness();m=h.RefreshSequence(0)
            def fail(a,b):raise error
            result=expected_error(lambda:s.run(m,compare=fail),ValueError,error)
            require(not m.completed and m.operations_completed==['unfreeze','captured_original_full_collect','freeze'],'Fatal after-state mismatch')
            return result
        case('METHOD_STUB_independent_semantic_mismatch_is_fatal',compare_failure)
        def changed_capture_value():
            s=StubHarness();m=h.RefreshSequence(0);count=0
            def capture_changed_frontier():
                nonlocal count;count+=1;value=science_METHOD_STUB()
                if count==2:value['clocks'][0]+=1
                return value
            result=expected_error(lambda:s.run(m,capture=capture_changed_frontier,compare=lambda a,b:None))
            require('Exact independent scientific fingerprint'in result['message']and not m.completed,
                'Actual helper canonical comparison rejects real changed capture value, not a throwing comparison double')
            return result
        case('METHOD_STUB_genuinely_changed_frontier_value_rejected_by_actual_helper',changed_capture_value)
        def changed_real_isolated_python_RNG():
            saved=random.getstate();count=0;s=StubHarness();m=h.RefreshSequence(0)
            try:
                def capture_changed_rng():
                    nonlocal count;count+=1
                    if count==2:random.random()
                    return dict(science_METHOD_STUB(),isolated_actual_Python_RNG=random.getstate())
                result=expected_error(lambda:s.run(m,capture=capture_changed_rng,compare=lambda a,b:None))
                require('Exact independent scientific fingerprint'in result['message'],
                    'Real fixture RNG change reaches actual equality rejection')
                return dict(result,production_RNG_read_or_changed=False,real_isolated_fixture_RNG_change=True)
            finally:random.setstate(saved)
        case('actual_isolated_Python_RNG_changed_value_negative_with_restoration',changed_real_isolated_python_RNG)
        from finite_actual_capture_bridge_METADATA_DOUBLES_v1 import (check_actual_capture_and_compare_sources,
            check_actual_periodic_eligibility_branch)
        case('actual_capture_and_compare_source_executed_under_explicit_207_threeAdam215_METADATA_DOUBLES',
            check_actual_capture_and_compare_sources)
        case('actual_source36_first_opportunity_and_original_terminal_AST_under_METADATA_DOUBLES',
            lambda:check_actual_periodic_eligibility_branch(h,P/'work/research/run_ASR18_BPE9223_resident_RAM_training_v36.py'))
        def callback_failure():
            error=OSError('METHOD_STUB receipt failure');s=StubHarness();m=h.RefreshSequence(0)
            def fail(v):raise error
            return expected_error(lambda:s.run(m,emit=fail),OSError,error)
        case('METHOD_STUB_receipt_exception_not_replaced',callback_failure)
        def other_thread_rejected():
            result=[]
            def work():result.append(expected_error(lambda:StubHarness().run(h.RefreshSequence(0))))
            t=threading.Thread(target=work);t.start();t.join(2);require(not t.is_alive()and len(result)==1,'MainThread-only authority')
            return result[0]
        case('METHOD_STUB_other_thread_cannot_refresh',other_thread_rejected)
        def exact_alias_negative():
            saved=gc.freeze
            try:
                with ObserveOriginalCollectScope()as observation:
                    gc.freeze=lambda:None
                    return expected_error(lambda:h.OnePostDEVRefresh(0).execute_real(barrier(),capture=science_METHOD_STUB,
                        compare=lambda a,b:None,guard=guard,admit=lambda n:{'passed':True},original_collect_observer=observation,emit=lambda v:None))
            finally:gc.freeze=saved
        case('actual_builtin_alias_replacement_rejected_without_GC',exact_alias_negative)
        close_method,close_ast=extracted_original_close()
        ToyIdleWorker.close=close_method
        queued=({'immutable':'same original toy queued plan','draws_consumed':0},{'path':'METHOD_STUB_immutable_pin','sha256':'e'*64})
        queued_identity=tuple(map(id,(queued,queued[0],queued[1])))
        old=ToyIdleWorker(close_method);workers.append(old);payload=Payload();ref=weakref.ref(payload)
        old_ticket=old.prepare_and_adopt(payload);del payload
        require(ref()is not None and old.outstanding is None and old.held_result is None
            and old.requests.empty()and old.results.empty(),'Empty adoption does not drop actual idle frame locals')
        closed=old.close(discard_unconsumed=False)
        require(closed['joined']and not closed['discarded_unconsumed_request']and old.reader_closed.is_set()
            and not old.thread.is_alive()and old.worker_status=={'closed':True,'error':None}and ref()is None,
            'Original close method genuinely joins toy worker and releases retained frame')
        old_ref=weakref.ref(old);workers.remove(old);del old
        require(old_ref()is None,'No old worker object remains before event')
        cases.append({'case':'actual_original_close_AST_with_constructor_reader_double_and_real_idle_frame_release',
                      'passed':True,'original_method_AST_sha256':hashlib.sha256(close_ast.encode()).hexdigest(),
                      'genuine_production_SQLite_or_locks_tested':False,'details':closed})
        h.ORIGINAL_COLLECT();h.ORIGINAL_FREEZE()
        reachable=Payload();reachable.cycle=reachable;cycle_ref=weakref.ref(reachable)
        policy_receipts=[]
        with ObserveOriginalCollectScope()as observation:
            m=h.OnePostDEVRefresh(0)
            actual_isolated_python_rng_before=random.getstate()
            def capture_real_fixture_rng():
                return dict(science_METHOD_STUB(),isolated_actual_Python_RNG=random.getstate())
            result=m.execute_real(barrier(),capture=capture_real_fixture_rng,
                compare=lambda a,b:require(a==b,'METHOD_STUB stable full scientific capture'),guard=guard,
                admit=lambda n:{'passed':n==16777216,'kind':'METHOD_STUB_resource_admission'},
                original_collect_observer=observation,emit=lambda v:policy_receipts.append(copy.deepcopy(v)))
            actual_isolated_python_rng_after=random.getstate()
            require(actual_isolated_python_rng_before==actual_isolated_python_rng_after,
                'Real isolated Python RNG state exactly unchanged across actual CPython policy event')
            require(result['passed']and observation.take_window()['calls']==0,
                'Actual policy collect bypasses original per-update observation; no new alias')
            require(gc.collect is observation,'Original observation alias remains active after refresh')
            require(queued_identity==tuple(map(id,(queued,queued[0],queued[1])))and queued[0]['draws_consumed']==0,
                'Same queued object/pin and no authoritative draw')
            new=ToyIdleWorker(close_method);workers.append(new);new_payload=Payload();new_ref=weakref.ref(new_payload)
            new.prepare_and_adopt(new_payload);del new_payload
            require(ToyIdleWorker.maximum_active==1,'No old/new scalar-worker overlap')
            expected_error(lambda:new.require_current_ticket(old_ticket))
            new.close(discard_unconsumed=False);require(new_ref()is None,'New worker after freeze is outside frozen set')
            workers.remove(new);del new
            del reachable
            h.ORIGINAL_COLLECT()
            require(cycle_ref()is not None,'Safety negative: frozen then unreachable real cycle survives full collect')
            cases.append({'case':'actual_CPython_one_refresh_cycle_retention_and_fresh_worker_lifetime',
                          'passed':True,'event':result,'scientific_GPU_proof':'METHOD_STUB only',
                          'frozen_cycle_retention_risk_genuinely_observed':True,'maximum_simultaneous_toy_workers':ToyIdleWorker.maximum_active,
                          'actual_isolated_python_rng_before':actual_isolated_python_rng_before,
                          'actual_isolated_python_rng_after':actual_isolated_python_rng_after,
                          'real_isolated_Python_RNG_before_after_exact':True,'production_RNG_read_or_changed':False})
        require(gc.collect is h.ORIGINAL_COLLECT,'LIFO exact original collect alias restored before teardown')
        h.ORIGINAL_UNFREEZE();h.ORIGINAL_COLLECT()
        require(gc.get_freeze_count()==0 and cycle_ref()is None,'Original-style final unfreeze/collect reclaims known cycle')
        cases.append({'case':'actual_final_unfreeze_reclamation_and_original_alias_restoration','passed':True})
        report={'passed':True}
    except BaseException as error:
        failure={'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
    finally:
        cleanup=[]
        for worker in workers:
            try:
                if not worker.closed:worker.close(discard_unconsumed=True)
            except BaseException as error:cleanup.append(repr(error))
        # Isolated fixture teardown only, never production. Preserve failure if a
        # partially executed event left a frozen generation/observation alias.
        gc.collect=original_alias
        gc.unfreeze();original_alias()
        inputs_after=[pin(v['path'])for v in review['input_pins']]
        value={'kind':'ASR18_one_postDEV_refresh_finite_CPU_report_v1','passed':bool(report and not failure and not cleanup),
            'cases':cases,'actual_exception':failure,'required_fixture_cleanup_errors':cleanup,
            'fixture_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},
            'input_pins_before':review['input_pins'],'input_pins_after':inputs_after,
            'input_pins_before_after_exact':inputs_after==review['input_pins'],
            'final_frozen_objects':gc.get_freeze_count(),'final_original_collect_alias_restored':gc.collect is original_alias,
            'actual_production_fingerprint_or_GPU_operation':False,'actual_authority_draw_or_state_write':False,
            'real_CPythongc_and_toy_thread_only':True,'real_production_DB_lock_lifecycle_tested':False,
            'wall_seconds':time.monotonic()-began,'opaque_framework_cycle_absence_proved':False}
        value['passed']=value['passed']and value['input_pins_before_after_exact']and value['final_frozen_objects']==0
        write(OUTPUT/'report.json',value)
    print(json.dumps({'report':pin(OUTPUT/'report.json'),'passed':value['passed'],'cases':len(cases)}),flush=True)
    require(value['passed'],'Preserve actual finite failure; no automatic rerun')


if __name__=='__main__':main()
