"""INACTIVE. Exact ROOT CPU scope is required before this fixture executes.

Actual complete source38 metadata selectors and update AST are executed under
explicit METHOD_STUB owners. No production import, journal bytes, restore,
sampler draw, audio, tensor, model, native update, GC, synchronization or GPU.
The accepted identity-helper performance proof is reused, never rebenchmarked.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
import weakref
from contextlib import contextmanager

import psutil

P = Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S = P / 'outputs/parakeet_turkish'
R = P / 'work/research'
W = Path(__file__).resolve().parent
sys.path[:0] = [str(W), str(S), str(R)]
from resource_watchdog_v2 import sample_memory

BEGIN = time.monotonic()
RESERVE = 72 * 1024**2
SCOPE = 'ASR18_source39_actual204800_entire_transition_main_path_metadata_CPU_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path)
    require(path.stat().st_size <= 8388608, 'Bounded admitted source/metadata')
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    return pin(path)


def who(proc=None):
    proc = psutil.Process() if proc is None else proc
    return dict(pid=proc.pid, create_time=proc.create_time(), ppid=proc.ppid())


def guard():
    value = sample_memory(os.getpid(), 'source38_integration_metadata_runtime')
    require(value['paired_sampling_within_one_second'] is True
            and value['system_memory']['commit_available_bytes'] >= 2343932928
            and value['system_memory']['physical_available_bytes'] >= 1843932928
            and value['worker_private_memory']['private_bytes'] < 536870912
            and time.monotonic() - BEGIN < 100
            and shutil.disk_usage('E:/').free >= 100000000000,
            'Original225/FULL plus separately charged72MiB; metadata100s/private512/E100GB')
    return value


def execute(nodes, scope):
    if not isinstance(nodes, list):
        nodes = [nodes]
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<ACTUAL_FROZEN_SOURCE_AST_METHOD_STUB_OWNERS>', 'exec'), scope)


def expression(node, scope):
    return eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),
                        '<ACTUAL_COMPLETE_METADATA_DICTIONARY>', 'eval'), scope)


def assign(nodes, name):
    return next(n for n in nodes if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def reject(kind, call):
    try:
        call()
    except kind as error:
        return dict(type=type(error).__name__, message=str(error))
    raise AssertionError('Expected strict rejection')


def forbidden(*args, **kwargs):
    raise AssertionError('Scientific/production authority forbidden in metadata fixture')


class PrefixDouble:
    def __init__(self, saved, guard):
        self.saved = copy.deepcopy(saved)

    def binding(self):
        return copy.deepcopy(self.saved)


class JournalDouble:
    def __init__(self, binding):
        self.value = copy.deepcopy(binding)

    def binding(self):
        return copy.deepcopy(self.value)


from source39_actual204800_transition_metadata_bridge_v1 import metadata_transition_cases


def update_cases(driver, old_driver, results):
    from main_path_scalar_attribution_v1 import MainPathScalarAttribution
    def update_loop(main):
        return next(n for n in ast.walk(main)if isinstance(n,ast.For)and ast.unparse(n.iter)=="enumerate(plan['updates'])")
    def run(main, active, failure=None, clock_failure=False):
        log=[];sentinel=RuntimeError('METHOD_STUB original injected '+str(failure));refs=[];restored=[];method_refs=[]
        def event(name):
            log.append(name)
            if name in ('propose','native','archive','release'):
                require(all(ref()is None for ref in method_refs),'Bound methods dropped immediately before propose/native/archive/release')
            if failure==name:raise sentinel
        class Packet(dict):
            plan_accesses=0
            def __getitem__(self,key):
                if key=='plan_sha256':
                    self.plan_accesses+=1
                    if self.plan_accesses==1:event('packet_plan_key')
                return super().__getitem__(key)
        entry=dict(update_index=0,native_transaction_sha256='tx',before_sampler_sha256='before',after_sampler_sha256='after',batch_id='batch',
            rows=[dict(duration=1)],draws=[])
        metadata={**entry,'prepared_payload':'augmentation'}
        plan=dict(updates=[entry]);packet=Packet(plan_sha256='plan',metadata=metadata,update=metadata,arrays=[1],
            source=dict(protocol=dict(sha256='CPU')),token='token',timings=dict(preparation_seconds=.125))
        if failure=='missing_plan_key':del packet['plan_sha256']
        if failure in ('missing_metadata_index','shortcircuit_wrong_hash'):metadata.pop('update_index')
        if failure=='shortcircuit_wrong_hash':packet['plan_sha256']='wrong'
        refs.append(weakref.ref(packet));queue=[packet];del packet
        class Stream:
            @property
            def wait_update_ready(self):
                event('ready_lookup');method=self._ready;method_refs.append(weakref.ref(method));return method
            def _ready(self,*a):event('ready_call');require(a==('plan',0),'Original ready args')
            @property
            def next_update(self):
                event('next_lookup');method=self._next;method_refs.append(weakref.ref(method));return method
            def _next(self,*a):event('next_call');require(a==('plan',0),'Original next args');return queue.pop()
            def release(self,token):event('release');require(token=='token','Original release token')
        class Schedule:
            means={'TR':1};poisoned=False
            def propose(self,**kwargs):
                event('propose');require(kwargs['optimizer_updates']==0,'Original propose cursor');return types.SimpleNamespace(identity_digest='tx',before={},phase='TR')
        @contextmanager
        def identity(*a,**kw):
            log.append('identity_enter')
            try:yield True
            finally:restored.append(True);log.append('identity_restore')
        plan_calls=[0]
        def digest(value):
            if value is plan:
                plan_calls[0]+=1;event('digest'+str(plan_calls[0]));return 'plan'
            return 'before' if value=={} else 'after' if value=={'state':1} else 'metadata'
        def native(*a,**kw):event('native');return dict(augmentation='augmentation')
        optimizer=types.SimpleNamespace(updates=0,poisoned=False)
        def commit(*a):event('commit');optimizer.updates+=1;return {'state':1}
        class Archive:
            def encode_record(self,value):event('archive');return b'bounded'
        def failed_clock():raise RuntimeError('METHOD_STUB observation clock failure')
        phase=(MainPathScalarAttribution(wall_clock=failed_clock,thread_clock=failed_clock)if clock_failure else MainPathScalarAttribution())if active else None
        pressure=lambda **kw:dict(wait_seconds=0.,observations=1)
        scope=dict(require=require,sys=sys,time=time,plan=plan,stream=Stream(),phase_observer=phase,throughput_active=active,
            throughput_pressure_wait=pressure,wait_for_native_commit_pressure=pressure,monitor=types.SimpleNamespace(_observe=forbidden),
            guard=lambda:None,native_pressure_policy=dict(parameters={}),throughput_reserve=RESERVE if active else 0,
            metadata_timing={},digest=digest,engine=types.SimpleNamespace(schedule=Schedule(),context=types.SimpleNamespace(resolve_identity=forbidden)),
            current_plan_identity_view=identity,identity_authority=object(),optimizer=optimizer,resident_source=dict(sha256='CPU'),
            native_update=native,model=None,teacher=None,journal=None,torch=None,stage='METHOD_STUB',commit_RAM_transaction=commit,
            native_seconds=0.,audio_seconds=0.,prepared_archive=[],archive_codec=Archive(),future_ticket=None,
            initial_updates=-1000,active_began=time.monotonic(),a=types.SimpleNamespace(queue_bytes=2250000000),
            folder=Path('E:/gpu_training_scratchpad/METHOD_STUB_no_write'),producer={},record=lambda p:{},__file__='METHOD_STUB',
            write_json=forbidden,print=forbidden)
        # Suppress progress branch by making update1 neither initial+1 nor20.
        function=ast.FunctionDef(name='one_update',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
            body=[copy.deepcopy(update_loop(main)),ast.Return(value=ast.Tuple(elts=[ast.Compare(left=ast.Constant('ready_call'),ops=[ast.NotIn()],comparators=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])]),
                ast.Compare(left=ast.Constant('next_call'),ops=[ast.NotIn()],comparators=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])])],ctx=ast.Load()))],decorator_list=[])
        # Assignments in the actual loop use these scalar accumulators as locals.
        function.body[:0]=ast.parse('native_seconds=0.;audio_seconds=0.;future_ticket=None').body
        execute(function,scope)
        function_object=scope.pop('one_update');error=None
        try:deleted=function_object()
        except BaseException as exc:error=exc;deleted=None
        special=failure in ('missing_plan_key','missing_metadata_index','shortcircuit_wrong_hash')
        if special:
            require(isinstance(error,ValueError if failure=='shortcircuit_wrong_hash'else KeyError),'Original shortcircuit/KeyError outcome')
        elif failure is not None:require(error is sentinel,'Original exception OBJECT identity retained')
        else:require(error is None and deleted==(True,True)and plan_calls[0]==3,'Three original plan digest calls and immediate bound-method deletion')
        error_type=None if error is None else type(error).__name__
        if error is not None:error.__traceback__=None
        stream_ref=weakref.ref(scope.pop('stream'));del function_object;queue.clear()
        require(stream_ref()is None and all(ref()is None for ref in method_refs),'No old stream/bound callable retained on normal or exception paths')
        filtered=[x for x in log if not x.startswith('identity_')]
        return dict(log=filtered,plan_calls=plan_calls[0],original_exception_identity=error is sentinel if failure and not special else None,error_type=error_type,
            packet_released=refs[0]()is None,identity_restored=len(restored),phase=phase.snapshot_and_reset()if phase else None)
    baseline=run(old_driver,False);candidate=run(driver,True)
    require(baseline['log']==candidate['log']and candidate['packet_released']and candidate['identity_restored']==1,'Original full update call order and packet weakref release')
    results['actual_complete_update_AST_original_call_order_three_digests']=dict(passed=True,baseline=baseline,candidate=candidate)
    for site in ('ready_lookup','digest1','ready_call','next_lookup','digest2','next_call','packet_plan_key','digest3','propose','native','commit','archive','release'):
        before=run(old_driver,False,site);after=run(driver,True,site)
        require(before['log']==after['log'],'Original failing-path order '+site)
        results['update_original_exception_identity_'+site]=dict(passed=True,log=after['log'],plan_calls=after['plan_calls'])
    for site in ('missing_plan_key','missing_metadata_index','shortcircuit_wrong_hash'):
        before=run(old_driver,False,site);after=run(driver,True,site)
        require(before['log']==after['log']and before['error_type']==after['error_type'],'Original packet KeyError/shortcircuit evaluation order')
        if site=='missing_plan_key':require(after['plan_calls']==2,'Packet key access precedes third fresh digest')
        results['actual_packet_key_shortcircuit_'+site]=dict(passed=True,log=after['log'],plan_calls=after['plan_calls'],error_type=after['error_type'])
    clocked=run(driver,True,clock_failure=True)
    require(clocked['log']==baseline['log']and sum(v['clock_errors']for v in clocked['phase']['phases'].values())>0,'Clock observation never suppresses original calls')
    clocked_error=run(driver,True,'native',clock_failure=True)
    require(clocked_error['original_exception_identity']is True,'Clock failure cannot mask original exception object')
    results['actual_inline_clock_failures_passthrough']=dict(passed=True)
    counts={k:v['count']for k,v in candidate['phase']['phases'].items()}
    require(counts=={k:(2 if k=='host_gate'else 1)for k in counts}and len(counts)==11,'Actual all11 phase counts including both original pressure gates')
    require(candidate['phase']['existing_packet_preparation_seconds']==dict(count=1,seconds=.125,max_seconds=.125),'Actual packet timings.preparation_seconds scalar')
    results['actual_all11_phase_records_and_CPU_preparation_scalar']=dict(passed=True,counts=counts)
    # Static scope completeness does not claim overlapping phase sums are CPU causes.
    labels={n.args[0].value for n in ast.walk(driver)if isinstance(n,ast.Call)and ast.unparse(n.func)=='phase_observer.start'and n.args}
    require(labels=={'host_gate','digest_readiness','readiness_wait','digest_take','packet_take','digest_binding','authoritative_propose',
        'metadata_binding','native_and_commit','archive_drop_release','progress_emit'},'All11 fixed bounded phase keys')
    results['all11_phase_keys_and_actual_packet_preparation_schema']=dict(passed=True,keys=sorted(labels),CPU_scalar=.125,not_exclusive_CPU_decomposition=True)


def resource_cases(driver, driver_tree, output, results, review):
    import asr18_native_update_commit_pressure_bookkeeping_v4 as pressure
    for extra in (0,RESERVE):
        events=[]
        def check():events.append('check')
        def observe():events.append('observe');return dict(system_memory=dict(commit_available_bytes=4518435456+extra,physical_available_bytes=1768435456+extra))
        got=pressure.wait_for_native_update_commit_pressure(observe=observe,check=check,policy=pressure.POLICY,clock=lambda:1.,sleep=forbidden,
            additional_bookkeeping_reservation_bytes=extra)
        require(events==['check','observe','check']and got['required_commit_available_bytes']==4518435456+extra
            and got['required_physical_available_bytes']==1768435456+extra,'Exact callbacks and full original225/FULL plus separate72')
        results['pressure_default0_or72_'+str(extra)]=dict(passed=True,result=got)
    for site in ('check','observe'):
        sentinel=RuntimeError('METHOD_STUB original pressure '+site)
        def fail():raise sentinel
        try:pressure.wait_for_native_update_commit_pressure(observe=fail if site=='observe'else lambda:{},check=fail if site=='check'else lambda:None,
            policy=pressure.POLICY,clock=lambda:0.)
        except RuntimeError as error:require(error is sentinel,'Pressure callback exception identity')
        else:raise AssertionError('Expected callback error')
        results['pressure_original_exception_'+site]=dict(passed=True)
    original_tree=ast.parse(Path(review['original_pressure_source']['path']).read_text(encoding='utf-8-sig'))
    policy_node=next(n for n in original_tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='POLICY'for t in n.targets))
    original_wait=next(n for n in original_tree.body if isinstance(n,ast.FunctionDef)and n.name=='wait_for_native_update_commit_pressure')
    original_scope=dict(time=time);execute([policy_node,original_wait],original_scope)
    def scripted(call):
        events=[];clock_value=[0.];samples=[0]
        def clock():events.append('clock');return clock_value[0]
        def check():events.append('check')
        def observe():
            events.append('observe');samples[0]+=1
            return dict(system_memory=dict(commit_available_bytes=4518435455 if samples[0]==1 else 4518435456,physical_available_bytes=1768435456))
        def sleep(seconds):events.append(('sleep',seconds));clock_value[0]+=seconds
        value=call(observe=observe,check=check,policy=pressure.POLICY,clock=clock,sleep=sleep)
        return events,value
    old_events,old_value=scripted(original_scope['wait_for_native_update_commit_pressure'])
    new_events,new_value=scripted(pressure.wait_for_native_update_commit_pressure)
    require(old_events==new_events and old_value==new_value,'Original default0 complete clock/check/observe/sleep/return order')
    results['original_pressure_default0_full_callback_and_baseline_equivalence']=dict(passed=True,events=new_events)
    for site in ('clock','sleep'):
        sentinel=RuntimeError('METHOD_STUB original pressure '+site)
        def fail(*a):raise sentinel
        try:pressure.wait_for_native_update_commit_pressure(observe=lambda:dict(system_memory=dict(commit_available_bytes=4518435455,physical_available_bytes=1768435456)),
            check=lambda:None,policy=pressure.POLICY,clock=fail if site=='clock'else lambda:0.,sleep=fail if site=='sleep'else forbidden)
        except RuntimeError as error:require(error is sentinel,'Original pressure clock/sleep exception identity')
        else:raise AssertionError('Expected original callback exception')
        results['pressure_original_exception_'+site]=dict(passed=True)
    results['pressure_invalid_reservation']=reject(ValueError,lambda:pressure.wait_for_native_update_commit_pressure(
        observe=forbidden,check=forbidden,policy=pressure.POLICY,additional_bookkeeping_reservation_bytes=True))
    admit=next(n for n in ast.walk(driver)if isinstance(n,ast.FunctionDef)and n.name=='admit')
    for extra in (0,RESERVE):
        sample=dict(system_memory=dict(commit_available_bytes=2000000000+268435456+extra+2100000000,
            physical_available_bytes=1500000000+268435456+extra))
        scope=dict(require=require,guard=lambda:None,monitor=types.SimpleNamespace(_observe=lambda:sample),
            prefetch_policy=dict(additional_host_reservation_bytes=268435456),throughput_reserve=extra,write_json=forbidden,
            invocation=Path('E:/gpu_training_scratchpad/METHOD_STUB_no_write'))
        execute(admit,scope);value=scope['admit']('DEV_original_increment',2100000000)
        require(value['additional_prefetch_reservation_bytes']==268435456 and value['additional_throughput_bookkeeping_reservation_bytes']==extra,'DEV original incremental/fullprefetch unchanged')
        results['actual_admit_DEV_default0_or72_'+str(extra)]=dict(passed=True)
        sample['system_memory']['physical_available_bytes']-=1
        results['actual_admit_one_byte_physical_denial_'+str(extra)]=reject(ValueError,lambda:scope['admit']('DEV_original_increment',2100000000))
    main_calls=[n for n in ast.walk(driver)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='admit'
        and n.args and isinstance(n.args[0],ast.Constant)and n.args[0].value=='throughput_bookkeeping_above_full_CPU_startup']
    require(len(main_calls)==3 and all(n.args[1].value==2500000000 for n in main_calls),'Every one of3 original CPU startups charges full2.5GB+72+FULLprefetch')
    results['all3_CPU_startup_admissions_AST_exact']=dict(passed=True,count=3,increment=2500000000)
    waits=[n for n in ast.walk(driver)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='throughput_pressure_wait']
    require(len(waits)==2 and all(any(k.arg=='additional_bookkeeping_reservation_bytes'and ast.unparse(k.value)=='throughput_reserve'for k in n.keywords)for n in waits),
        'Both original pre-update pressure boundaries separately charge72 above original full225/FULL')
    results['both_actual_driver_pressure_gates_charge72']=dict(passed=True,count=2)
    monitor_node=next(n for n in driver_tree.body if isinstance(n,ast.ClassDef)and n.name=='Monitor')
    scope=dict(require=require,os=os,write_json=lambda *a:None)
    execute(monitor_node,scope)
    import resource_watchdog_v2 as watchdog
    original=watchdog.sample_memory
    try:
        for extra in (0,RESERVE):
            monitor=scope['Monitor'].__new__(scope['Monitor'])
            monitor.folder=output;monitor.additional_reserved_host_bytes=extra;monitor.samples=0;monitor.peak_private=0
            sample=dict(paired_sampling_within_one_second=True,worker_private_memory=dict(private_bytes=100),
                system_memory=dict(commit_available_bytes=2000000000+extra,physical_available_bytes=1500000000+extra))
            watchdog.sample_memory=lambda *a:sample
            monitor._observe()
            require(monitor.samples==1 and monitor.peak_private==100,'Original scalar-only Monitor update')
            sample['system_memory']['commit_available_bytes']-=1
            results['actual_Monitor_one_byte_commit_denial_'+str(extra)]=reject(ValueError,monitor._observe)
            sample['system_memory']['commit_available_bytes']+=1;sample['system_memory']['physical_available_bytes']-=1
            results['actual_Monitor_one_byte_physical_denial_'+str(extra)]=reject(ValueError,monitor._observe)
    finally:watchdog.sample_memory=original


def main(args):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
        'Exact hidden CPU/caps1')
    review=read(args.ROOT_review)
    require(review['scope']==SCOPE and review['accepted']is True and review['reviewer']=='/root'and review['fixture_source']==pin(__file__),'Separate exact CPU admission')
    output=Path(review['fixture_output']);require(not output.exists()and output.drive.lower()=='e:','Fresh E output');output.mkdir()
    entry=dict(identity=who(),launcher=who(psutil.Process().parent()),ROOT_review=pin(args.ROOT_review))
    write(output/'entry.json',entry)
    initial=sample_memory(os.getpid(),'source38_metadata_initial_before_predicate');write(output/'initial_numeric_before_predicate.json',initial)
    require(initial['system_memory']['commit_available_bytes']>=5130803840 and initial['system_memory']['physical_available_bytes']>=2380803840
        and initial['worker_private_memory']['private_bytes']<536870912 and shutil.disk_usage('E:/').free>=100000000000,'Full original225+FULLprefetch+72+512 prospective')
    results={};report=dict(kind=SCOPE,passed=False,entry=entry,results=results,METHOD_STUB_future_transition_and_science=True)
    try:
        for item in review['input_pins']:require(pin(item['path'])==item,'Exact frozen inputs before')
        trees=[ast.parse(Path(review[k]['path']).read_text(encoding='utf-8-sig'))for k in ('driver_source','wrapper_source','old_driver_source')]
        driver,wrapper,old_driver=[next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='main')for t in trees]
        metadata_transition_cases(review,output,driver,wrapper,results)
        update_cases(driver,old_driver,results)
        resource_cases(driver,trees[0],output,results,review)
        for item in review['input_pins']:require(pin(item['path'])==item,'Exact frozen inputs after')
        require('torch'not in sys.modules and 'numpy'not in sys.modules,'No framework/model/audio/GPU operation')
        report.update(passed=True,final_guard=guard(),input_pins_before=review['input_pins'],input_pins_after=[pin(i['path'])for i in review['input_pins']])
    except BaseException as error:
        import traceback
        report['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        raise
    finally:
        report.update(wall_seconds=time.monotonic()-BEGIN,final_identity=who())
        result=write(output/'report.json',report);print(json.dumps(dict(result=result,passed=report['passed'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);main(parser.parse_args())
