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
SCOPE = 'ASR18_source38_entire_transition_main_path_metadata_CPU_v1'


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


def metadata_transition_cases(review, output, driver, wrapper, results):
    """Actual helper and complete AST; all future transition files are doubles."""
    import asr18_current_plan_throughput_checkpoint_transition_v3 as lane
    import asr18_one_postDEV_gc_refresh_checkpoint_transition_v2 as gap_module
    original_record, original_read, original_dead = lane.record, lane.read_json, lane.exact_dead
    base = read(review['source_configuration']['path'])
    proper = lane.legacy_source37_configuration(base)
    lane.checked_current_plan_throughput_policy(base)
    results['genuine_policy_source_and_positive_CPU_gate'] = dict(passed=True, closed_CPU_report_not_read=True)
    runtime = read(lane.BASE_RUNTIME)
    documents, records = {}, {}
    def document(name, value):
        path = str(output / (name + '.METHOD_STUB.json'))
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        item = dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        documents[path] = copy.deepcopy(value); records[path] = item
        return item
    def record_double(path):
        key = str(path)
        return copy.deepcopy(records[key]) if key in records else original_record(path)
    def read_double(path):
        key = str(path)
        return copy.deepcopy(documents[key]) if key in documents else original_read(path)
    saved = dict(path=str(lane.BRANCH / 'journals/commits_000026.jsonl'), offset=0, sha256='a'*64)
    pointer_value = dict(committed_updates=204000, diagnostic_only=False,
                         all_window_updates_committed=True, journal_prefix=saved)
    pointer = document('pointer', pointer_value)
    failure = document('failure', dict(actual_optimizer_updates=204001, last_checkpoint=204000,
        error="ValueError('Actual owner cooperative stop')", traceback='METHOD_STUB guard after published event'))
    monitor = document('monitor', dict(joined=True, clean=True, error=None))
    stage = document('stage', dict(decision='READY', actual=204000, ready_absent=True,
        current_native_descendants=dict(descendants=[dict(identity=dict(pid=900001,create_time=1.25,ppid=19320)),
                                                     dict(identity=dict(pid=900002,create_time=1.5,ppid=900001))])))
    stop = document('stop', dict(kind='ASR18_human_authorized_postDEV204000_cooperative_checkpoint_stop_v1',
        optimizer_updates_observed=204000, normal_exit_or_no_tail_race_claimed=False, evidence=stage))
    watcher = document('watcher', dict(passed=True, stop_published=True, prestop_evidence=stage))
    persistence = document('persistence', dict(passed=True, optimizer_updates=204000))
    journal = output / 'METHOD_STUB_closed_journal_STAT_only.bin'
    with journal.open('xb') as stream:
        stream.write(b'x')
    stat = dict(kind='STAT_ONLY_original_closed_journal_v1', path=str(journal), bytes=1, mtime_ns=journal.stat().st_mtime_ns)
    identities = [dict(pid=19320,create_time=1792803517.7236164,ppid=39856),
        dict(pid=39856,create_time=1792803517.7165897,ppid=21928),
        dict(pid=21928,create_time=1792803516.8916032,ppid=41376),
        dict(pid=42068,create_time=1792830275.0700862,ppid=40536),
        dict(pid=40536,create_time=1792830275.0545292,ppid=19320),
        dict(pid=900001,create_time=1.25,ppid=19320),dict(pid=900002,create_time=1.5,ppid=900001)]
    evidence = dict(old_driver=original_record(R/'run_ASR18_BPE9223_resident_RAM_training_v37.py'),
        old_wrapper=original_record(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v37.py'),
        old_CPU=original_record(S/'fastmss_resident_feature_stream_v14.py'),old_configuration=original_record(lane.BASE_RUNTIME),
        pointer=pointer,actual_failure=failure,stop=stop,monitor_completion=monitor,watcher_result=watcher,
        persistence=persistence,original_journal_stat=stat)
    # Exact production path equality is tested against metadata mappings, not files.
    records[str(lane.BRANCH/'invocations/owner_000026/actual_failure.json')] = failure
    records[str(lane.BRANCH/'invocations/owner_000026/stop.json')] = stop
    documents[str(lane.BRANCH/'latest_checkpoint.json')] = dict(pointer=pointer)
    closure_value = dict(kind='ASR18_actual_source37_postDEV204000_throughput_transition_v1',owned_session=26493,
        old_owner_invocation=26,new_owner_invocation=27,actual_outer_TOOL_exit_code=1,normal0_claimed=False,
        intentional_cooperative_stop_observed=True,evidence=evidence,published_journal_optimizer_updates=204001,
        actual_optimizer_updates=204001,last_fully_saved_optimizer_updates=204000,
        optimizer_vs_journal_frontier=gap_module.validate_intentional_optimizer_journal_gap(documents[failure['path']],204001),
        all_recorded_runtime_identities=identities,saved_journal_prefix=saved,preserved_tail_positive_updates=1,
        unsaved_tail_bytes=1,original_tail_modified=False,tail_is_historical_only_requires_genuine_redo=True)
    closure = document('closure', closure_value)
    base[lane.PRIOR] = closure
    lane.record, lane.read_json, lane.exact_dead = record_double, read_double, lambda identity: True
    try:
        actual = lane.checked_current_plan_throughput_checkpoint_resume(closure,base=base,folder=lane.BRANCH,pointer_pin=pointer)
        results['actual_complete_typed_validator_METHOD_STUB_future204_owners'] = dict(passed=True, value=actual)
        def changed_value(name, mutate):
            modified = copy.deepcopy(closure_value); mutate(modified)
            cp = document(name, modified); cb = copy.deepcopy(base); cb[lane.PRIOR] = cp
            return reject((ValueError,KeyError),lambda:lane.checked_current_plan_throughput_checkpoint_resume(cp,base=cb,folder=lane.BRANCH,pointer_pin=pointer))
        for name, mutate in [
            ('missing_native_launcher',lambda v:v['all_recorded_runtime_identities'].pop(1)),
            ('missing_wrapper',lambda v:v['all_recorded_runtime_identities'].pop(2)),
            ('missing_old_CPU',lambda v:v['all_recorded_runtime_identities'].pop(3)),
            ('missing_captured_child',lambda v:v['all_recorded_runtime_identities'].pop()),
            ('wrong_captured_PPID',lambda v:v['all_recorded_runtime_identities'][-1].__setitem__('ppid',7)),
            ('normal0_alias',lambda v:v.__setitem__('normal0_claimed',True)),
            ('wrong_saved_frontier',lambda v:v.__setitem__('last_fully_saved_optimizer_updates',203800)),
            ('prefix_mismatch',lambda v:v['saved_journal_prefix'].__setitem__('offset',1)),
            ('tail_promoted',lambda v:v.__setitem__('tail_is_historical_only_requires_genuine_redo',False))]:
            results['transition_reject_'+name] = changed_value(name,mutate)
        for name, mutate in [('LR',lambda v:v['full_run_identity'].__setitem__('METHOD_STUB_changed_LR',1)),
            ('FULLprefetch',lambda v:v['private_scalar_prefetch_policy'].__setitem__('additional_host_reservation_bytes',0)),
            ('extra_reserve',lambda v:v[lane.POLICY].__setitem__('additional_host_reservation_bytes',0)),
            ('second_GC',lambda v:v['one_postDEV_gc_refresh_policy'].__setitem__('additional_bookkeeping_reserve_bytes',0))]:
            bad=copy.deepcopy(base);mutate(bad)
            results['policy_reject_'+name]=reject((ValueError,KeyError),lambda:lane.checked_current_plan_throughput_policy(bad))
        for gap in (0,1):
            f=dict(actual_optimizer_updates=204001,error='Actual owner cooperative stop',
                   traceback='in commit_RAM_transaction in append self.namespace.check(); self.binding() self.guard()')
            value=gap_module.validate_intentional_optimizer_journal_gap(f,204001-gap)
            require(value['unpublished_optimizer_counter_advance']==gap,'Actual original0or1 gap code')
        results['actual_original_gap_validator_0_and1_METHOD_STUB_traces']=dict(passed=True)
        results['gap1_without_source_trace_rejected']=reject(ValueError,lambda:gap_module.validate_intentional_optimizer_journal_gap(
            dict(actual_optimizer_updates=204001,error='Actual owner cooperative stop',traceback='unknown'),204000))
        names=('prior_process','ninth_original_resume','gc_refresh_resume','eighth_original_resume','fresh_signature_resume',
            'canonical_archive_resume','lookahead_resume','cushion225_resume','postrestore_selector_resume',
            'packet_credit_resume','native_headroom_resume','abnormal_resume','source_transition_resume',
            'original_interval_resume','DEV_frontend_resume','DEV_admission_resume','periodic_quiescence_resume')
        args=types.SimpleNamespace(**{k:None for k in names},throughput_resume=closure['path'],invocation=27,start_invocation=27,
            configuration=review['source_configuration']['path'],stream_configuration=review['source_configuration']['path'])
        scope=dict(a=args,candidate_base=base,throughput_active=True,folder=lane.BRANCH,pointer_pin=pointer,record=record_double,
            require=require,checked_current_plan_throughput_checkpoint_resume=lane.checked_current_plan_throughput_checkpoint_resume)
        entry=assign(driver.body,'throughput_transition')
        check=next(n for n in driver.body if isinstance(n,ast.If)and ast.unparse(n.test)=='throughput_transition is not None')
        execute([entry,check],scope)
        require(scope['throughput_transition']==actual,'Actual entire pre-restore selector lane')
        wrong=copy.deepcopy(args);wrong.invocation=26
        results['selector_wrong_invocation']=reject(ValueError,lambda:execute(check,{**scope,'a':wrong}))
        wrong=copy.deepcopy(args);wrong.ninth_original_resume='old_lane'
        results['selector_old_lane_mix']=reject(ValueError,lambda:execute(check,{**scope,'a':wrong}))
        common={k:None for k in ('gc_transition','signature_transition','canonical_transition','lookahead_transition',
            'interval_failure','abnormal','transition')}
        scope.update(common);execute(assign(driver.body,'preserved_tail'),scope)
        require(scope['preserved_tail']==actual,'Actual complete shared preserved-tail selector')
        factory=next(n for n in ast.walk(driver)if isinstance(n,ast.FunctionDef)and n.name=='journal_factory')
        fs={**scope,'source_journal':saved,'SavedJournalPrefixView':PrefixDouble,'guard':forbidden}
        execute(factory,fs);view,before=fs['journal_factory'](saved)
        require(view.binding()==saved and before==dict(kind='ASR18_readonly_saved204000_prefix_after_source37_throughput_transition_v1',
            actual_transition=closure,original_tail_modified=False,no_tail_replay=True),'ENTIRE factory return dictionary')
        results['actual_entire_factory_METHOD_STUB_no_read']=dict(passed=True,before=before)
        results['wrong_factory_prefix']=reject(ValueError,lambda:fs['journal_factory']({**saved,'offset':1}))
        post=next(n for n in ast.walk(driver)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='write_json'
            and n.args and 'actual_journal_branch.json'in ast.unparse(n.args[0]))
        newbinding={**saved,'path':str(lane.BRANCH/'journals/commits_000027.jsonl')}
        ps={**fs,'journal':JournalDouble(newbinding),'preserved_tail':actual}
        after=expression(post.args[1],ps)
        require(after==dict(kind='ASR18_actual_saved_prefix_recovery_journal_branch_v1',predecessor=closure,
            saved_prefix=saved,preserved_original=stat,new_active_journal=newbinding,historical_unsaved_tail_updates=1,
            native_updates_or_draws_in_copy=0),'ENTIRE postrestore dict and STAT-only original')
        results['actual_entire_postrestore_dictionary_METHOD_STUB']=dict(passed=True,after=after)
        for name,value in [('None',None),('stale','E:/gpu_training_scratchpad/METHOD_STUB_missing_selector')]:
            wrong=copy.deepcopy(args);wrong.throughput_resume=value
            results['postselector_reject_'+name]=reject((TypeError,FileNotFoundError),lambda:expression(post.args[1],{**ps,'a':wrong}))
        cs={**scope,'candidate_base':base,'copy':copy,'__file__':review['driver_source']['path']}
        cassign=assign(driver.body,'c')
        cupdate=next(n for n in driver.body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
            and ast.unparse(n.value.func)=='c.update')
        execute([cassign,cupdate],cs)
        expected=dict(worker_driver_source=review['driver_source'],parent_driver_source=review['driver_source'],
            resident_RAM_stream_configuration=review['source_configuration'],resident_branch_namespace=str(lane.BRANCH),
            checkpoint_interval_updates=200,source_checkpoint=pointer,source_configuration=review['source_configuration'])
        require(all(cs['c'][k]==v for k,v in expected.items()),'Actual seven runtime fields, both NEW config pins')
        results['actual_complete_runtime7']=dict(passed=True,expected=expected)
        route=next(n for n in wrapper.body if isinstance(n,ast.If)and ast.unparse(n.test)=='a.prior_process is not None')
        structural=dict(throughput_resume=closure,throughput_transition_validator=review['transition_helper'],
            authoritative_current_plan_identity_and_main_path_attribution_policy=base[lane.POLICY],initial_owner_invocation=27)
        ws=dict(a=args,review=structural,candidate_configuration=base,configuration=proper,record=record_double,
                THROUGHPUT_FOLDER=W,require=require,Path=Path,S=S)
        execute(route,ws)
        results['entire_wrapper_route_METHOD_STUB_structural_review']=dict(passed=True)
        wrong=copy.deepcopy(args);wrong.start_invocation=26
        results['wrapper_wrong_invocation']=reject(ValueError,lambda:execute(route,{**ws,'a':wrong}))
        loop=next(n for n in wrapper.body if isinstance(n,ast.While))
        forward=next(n for n in loop.body if isinstance(n,ast.If)and ast.unparse(n.test)=='prior is None')
        fw=dict(a=args,prior=None,invocation=27,command=[]);execute(forward,fw)
        require(fw['command']==['--throughput-resume',closure['path']]and all(type(x)is str for x in fw['command']),
            'Actual complete initial wrapper39 forwarding, no None or legacy fallback')
        results['actual_wrapper_forwarding']=dict(passed=True,command=fw['command'])
        fw=dict(a=args,prior=dict(path='METHOD_STUB_normal_prior'),invocation=28,command=[])
        execute(forward,fw)
        require(fw['command']==['--prior-process',fw['prior']['path']],
            'Original subsequent normal-prior continuation, no repeated throughput selector')
        results['actual_wrapper_normal_prior_continuation']=dict(passed=True,command=fw['command'])
    finally:
        lane.record,lane.read_json,lane.exact_dead=original_record,original_read,original_dead


def update_cases(driver, old_driver, results):
    from main_path_scalar_attribution_v1 import MainPathScalarAttribution
    def update_loop(main):
        return next(n for n in ast.walk(main)if isinstance(n,ast.For)and ast.unparse(n.iter)=="enumerate(plan['updates'])")
    def run(main, active, failure=None):
        log=[];sentinel=RuntimeError('METHOD_STUB original injected '+str(failure));refs=[];restored=[]
        def event(name):
            log.append(name)
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
        refs.append(weakref.ref(packet));queue=[packet];del packet
        class Stream:
            @property
            def wait_update_ready(self):
                event('ready_lookup')
                def call(*a):event('ready_call');require(a==('plan',0),'Original ready args')
                return call
            @property
            def next_update(self):
                event('next_lookup')
                def call(*a):event('next_call');require(a==('plan',0),'Original next args');return queue.pop()
                return call
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
        phase=MainPathScalarAttribution() if active else None
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
        function.body[:0]=[ast.parse('native_seconds=0.;audio_seconds=0.').body[0],ast.parse('audio_seconds=0.').body[0]]
        execute(function,scope)
        error=None
        try:deleted=scope['one_update']()
        except BaseException as exc:error=exc;deleted=None
        if failure is not None:
            require(error is sentinel,'Original exception OBJECT identity retained')
            error.__traceback__=None
        else:require(error is None and deleted==(True,True)and plan_calls[0]==3,'Three original plan digest calls and immediate bound-method deletion')
        filtered=[x for x in log if not x.startswith('identity_')]
        return dict(log=filtered,plan_calls=plan_calls[0],original_exception_identity=error is sentinel if failure else None,
            packet_released=refs[0]()is None,identity_restored=len(restored),phase=phase.snapshot_and_reset()if phase else None)
    baseline=run(old_driver,False);candidate=run(driver,True)
    require(baseline['log']==candidate['log']and candidate['packet_released']and candidate['identity_restored']==1,'Original full update call order and packet weakref release')
    results['actual_complete_update_AST_original_call_order_three_digests']=dict(passed=True,baseline=baseline,candidate=candidate)
    for site in ('ready_lookup','digest1','ready_call','next_lookup','digest2','next_call','packet_plan_key','digest3','propose','native','commit','archive','release'):
        before=run(old_driver,False,site);after=run(driver,True,site)
        require(before['log']==after['log'],'Original failing-path order '+site)
        results['update_original_exception_identity_'+site]=dict(passed=True,log=after['log'],plan_calls=after['plan_calls'])
    # Static scope completeness does not claim overlapping phase sums are CPU causes.
    labels={n.args[0].value for n in ast.walk(driver)if isinstance(n,ast.Call)and ast.unparse(n.func)=='phase_observer.start'and n.args}
    require(labels=={'host_gate','digest_readiness','readiness_wait','digest_take','packet_take','digest_binding','authoritative_propose',
        'metadata_binding','native_and_commit','archive_drop_release','progress_emit'},'All11 fixed bounded phase keys')
    results['all11_phase_keys_and_actual_packet_preparation_schema']=dict(passed=True,keys=sorted(labels),CPU_scalar=.125,not_exclusive_CPU_decomposition=True)


def resource_cases(driver, driver_tree, output, results):
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
        resource_cases(driver,trees[0],output,results)
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
