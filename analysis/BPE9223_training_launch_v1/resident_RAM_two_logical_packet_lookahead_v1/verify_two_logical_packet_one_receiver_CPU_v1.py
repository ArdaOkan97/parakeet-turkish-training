"""INACTIVE finite real Win32/byte-pipe control; SYNTHETIC packets, no audio/model.
Parent harness substitutes constructor/worker and adds assertion-only observers.
Actual CPU11 sender, receiver, readiness/get/next_update/release/close/abort and
helper3 drain are exercised. Weakrefs observe arrays, base ndarrays and memoryviews;
raw bytearray owners cannot be weak-referenced and are identified honestly.
Two admitted complete packets is NOT a full frontend/process/kernel-pipe bound.
This source is NOT permission to execute; requires a separate exact ROOT scope.
"""
import argparse,ast,copy,hashlib,json,os,queue,random,shutil,subprocess,sys,threading,time,traceback,weakref
from collections import deque
from pathlib import Path
F=Path(__file__).resolve().parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
sys.path[:0]=[str(F),str(S)]
import psutil
import fastmss_resident_feature_stream_v11 as transport
from asr18_owned_packet_credit_v2 import PacketCreditOwner,PacketCreditConsumer
from resource_watchdog_v2 import sample_memory
BEGIN=time.monotonic();TRACKED={};PEAK=0
FEATURE_SHAPE=(1,2048,128);MASK_SHAPE=(1,2048)
SYNTHETIC_ARRAY_BYTES=2048*128*4+2048
def require(v,m):
    if not v:raise ValueError(m)
def identity(pid=None):
    p=psutil.Process(os.getpid()if pid is None else pid)
    return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def pin(p):return transport.pin(p)
def write(p,v):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(v,f,indent=2);f.write('\n')
def guard():
    global PEAK
    value=sample_memory(os.getpid(),'inactive_real_kernel_synthetic_packet_control')
    memory=value['system_memory'];family=value['worker_private_memory']['private_bytes']
    for pid,ctime in list(TRACKED.items()):
        try:
            p=psutil.Process(pid)
            if abs(p.create_time()-ctime)<.001:family+=getattr(p.memory_info(),'private',p.memory_info().rss)
        except psutil.NoSuchProcess:pass
    PEAK=max(PEAK,family)
    require(value['paired_sampling_within_one_second'] and family<512*1024**2
        and memory['commit_available_bytes']>=2000000000+268435456
        and memory['physical_available_bytes']>=1500000000+268435456
        and shutil.disk_usage('E:/').free>=100000000000 and time.monotonic()-BEGIN<180,
        'Original512MiB family/180s/full256MiB/headroom/E100GB CPU control bounds')
    return value
def event(folder,v):
    with (Path(folder)/'events.jsonl').open('a',encoding='utf-8')as f:
        f.write(json.dumps(v)+'\n');f.flush()
def read_events(folder):
    p=Path(folder)/'events.jsonl'
    require(not p.exists()or p.stat().st_size<1024**2,'Bounded synthetic event evidence')
    return []if not p.exists()else[json.loads(x)for x in p.read_text().splitlines()]
def until(condition,timeout=10):
    end=time.monotonic()+timeout
    while not condition():
        guard()
        if time.monotonic()>end:raise TimeoutError('Finite synthetic-control observation')
        time.sleep(.025)
def dead(who):
    try:return abs(psutil.Process(who['pid']).create_time()-who['create_time'])>=.001
    except psutil.NoSuchProcess:return True
def expect(kind,call):
    try:call()
    except kind as error:return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('Expected '+kind.__name__)

def setup_numpy():
    import fastmss_BPE9223_cpu_common_v12 as core
    core.numpy_runtime(frontend=False) # Original CPU NumPy path, NO Torch/model.
    import numpy as np
    return np


def payload_weakrefs(values):
    """Weakref actual ndarray/base/memoryview objects, never retain raw payloads."""
    refs=[];seen=set();types={};unweakrefable=set()
    for root in values:
        current=root
        while current is not None and id(current) not in seen:
            seen.add(id(current));name=type(current).__name__;types[name]=types.get(name,0)+1
            try:refs.append(weakref.ref(current))
            except TypeError:unweakrefable.add(name)
            if isinstance(current,memoryview):current=current.obj
            else:current=getattr(current,'base',None)
    return refs,{'weakref_object_types':types,'unweakrefable_owner_types':sorted(unweakrefable)}

class SendOutputObserver:
    """Observe actual _send chunk views without copying or retaining payloads."""
    def __init__(self,stream):self.stream=stream;self.refs=[];self.view_writes=0
    def write(self,value):
        if isinstance(value,memoryview):
            refs,_=payload_weakrefs([value]);self.refs.extend(refs);self.view_writes+=1
        return self.stream.write(value)
    def flush(self):return self.stream.flush()

def full_packet_cap_source_control():
    tree=ast.parse(Path(transport.__file__).read_text(encoding='utf-8-sig'))
    worker_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker_main')
    calls=[n for n in ast.walk(worker_node)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
        and n.func.id=='require'and len(n.args)>1 and isinstance(n.args[1],ast.Constant)
        and n.args[1].value=='Bounded full update packet with all features/masks']
    require(len(calls)==1,'One original post-prepare full packet cap expression')
    expr=compile(ast.Expression(calls[0].args[0]),str(transport.__file__),'eval')
    def accepted(total):return eval(expr,{'len':len,'arrays':[None,None],'metadata':{'rows':[{}]},
        'total':total,'PACKET_LIMIT':transport.PACKET_LIMIT})
    require(accepted(transport.PACKET_LIMIT)and not accepted(transport.PACKET_LIMIT+1),
        'Original full256MiB geometry guard tested WITHOUT allocating256MiB')
    return {'actual_original_guard_AST':ast.dump(calls[0],include_attributes=False),
        'limit_bytes':transport.PACKET_LIMIT,'at_limit_accepted':True,'over_limit_rejected':True,
        'large_allocation_performed':False,'post_prepare_guard_not_a_frontend_growth_bound':True}


def worker(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    np=setup_numpy();folder=Path(a.output)
    credit=PacketCreditConsumer(json.loads(os.environ['ASR18_PACKET_CREDIT_DESCRIPTOR']))
    out=sys.stdout.buffer;commands=sys.stdin.buffer;sys.stdout=sys.stderr
    completed=0;previous=None;who=identity();write(folder/'synthetic_worker_entry.json',who)
    acquiring=[False];original_acquire=credit.acquire;original_close=credit.close
    def checked_acquire(**arguments):
        acquiring[0]=True
        try:return original_acquire(**arguments)
        finally:acquiring[0]=False
    def checked_close():
        require(not acquiring[0],'Child SAME handles never close during its acquire wait')
        proof=original_close();event(folder,{'kind':'child_credit_close_after_wait_unwound','proof':proof})
        return proof
    credit.acquire=checked_acquire;credit.close=checked_close
    try:
        transport._send(out,{'kind':'ready','producer_identity':who,'source':{'protocol':pin(transport.__file__)},
            'startup_seconds':time.monotonic()-BEGIN,'packet_credit':credit.descriptor})
        while True:
            command=transport._header(commands)
            if command['kind']=='close':break
            require(command['kind']=='prepare_plan','Original ordered plan control frame')
            plan=transport.bound(command['plan']);sha=transport.digest(plan)
            require(sha==command['plan_sha256'],'Exact synthetic submitted plan')
            if previous is not None:require(plan['initial_sampler_sha256']==previous,'Exact synthetic ordered plan chain')
            for index,entry in enumerate(plan['updates']):
                credit.acquire(phase='before_prepare_update',token={'plan_sha256':sha,'update_index':index},check=guard)
                event(folder,{'kind':'synthetic_prepare_started','plan':sha,'index':index,'monotonic':time.monotonic()})
                if a.fault=='prepare_error':raise ValueError('SyntheticPrepareMarker')
                features=np.arange(2048*128,dtype=np.float32).reshape(FEATURE_SHAPE)+index
                mask=np.ones(MASK_SHAPE,dtype=np.bool_)
                arrays=[features,mask];refs,owner_types=payload_weakrefs(arrays)
                specs=[{'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.nbytes,
                    'sha256':hashlib.sha256(memoryview(x).cast('B')).hexdigest()}for x in arrays]
                metadata={'kind':'SYNTHETIC_not_audio_or_model','update_index':index,
                    **{k:entry[k]for k in ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')},
                    'rows':[{'input_features':{'shape':specs[0]['shape'],'array_sha256':specs[0]['sha256']},
                             'attention_mask':{'shape':specs[1]['shape'],'array_sha256':specs[1]['sha256']}}]}
                frame={'kind':'update','token':str(who['pid'])+':'+sha+':'+str(index),'plan_sha256':sha,'update_index':index,
                    'metadata':metadata,'metadata_sha256':transport.digest(metadata),'array_descriptors':specs,
                    'array_bytes':sum(x.nbytes for x in arrays),'source':{'protocol':pin(transport.__file__)}}
                if a.fault=='receiver_hash_error':specs[0]['sha256']='f'*64
                if a.fault=='receiver_full_cap':
                    frame['array_bytes']=transport.PACKET_LIMIT+1
                    frame['array_descriptors'][0]['bytes']=transport.PACKET_LIMIT+1-frame['array_descriptors'][1]['bytes']
                    transport._send(out,frame);return 0
                if a.fault=='send_eof':transport._send(out,frame);return 0
                if a.fault=='send_error':
                    class BrokenOutput:
                        def write(self,value):raise OSError('SyntheticSendMarker')
                    transport._send(BrokenOutput(),frame,arrays)
                observed_out=SendOutputObserver(out);outer_views=[]
                def send_profile(actual_frame,kind,arg):
                    if kind=='return'and actual_frame.f_code is transport._send.__code__:
                        view=actual_frame.f_locals.get('view')
                        if view is not None:outer_views.append(weakref.ref(view))
                sys.setprofile(send_profile)
                try:transport._send(observed_out,frame,arrays)
                finally:sys.setprofile(None)
                refs.extend(observed_out.refs);refs.extend(outer_views)
                view_writes=observed_out.view_writes
                del features,mask,arrays,metadata,frame,specs,observed_out
                require(view_writes>=2 and outer_views and all(r()is None for r in refs),
                    'Actual send views and synthetic producer arrays/base owners dead BEFORE next real acquire')
                event(folder,{'kind':'synthetic_sent_payload_refs_dead','plan':sha,'index':index,
                    'monotonic':time.monotonic(),'actual_send_view_writes':view_writes,
                    'actual_send_outer_view_weakrefs':len(outer_views),'all_payload_weakrefs_dead':True,
                    'ownership_types':owner_types})
                completed+=1
            previous=transport.digest(plan['final_private_planning_sampler'])
            transport._send(out,{'kind':'plan_complete','plan_sha256':sha,'updates':len(plan['updates']),
                'final_sampler_sha256':previous,'process_exit_claimed':False})
        credit.close()
        transport._send(out,{'kind':'closed','completed_updates':completed,'packet_credit':credit.receipt(),
            'synthetic_contents_no_audio_model':True})
        return 0
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        try:transport._send(out,{'kind':'error','message':repr(error),
            **({'packet_credit_timeout':error.packet_credit_timeout}if hasattr(error,'packet_credit_timeout')else{})})
        except BaseException:pass
        return 1
    finally:credit.close()

class Harness(transport.ResidentFeatureProducer):
    """Synthetic constructor plus assertion-only observers; actual CPU11 methods run."""
    def __init__(self,folder,fault='none'):
        self.folder=Path(folder);self.folder.mkdir()
        self.queue_bytes=268435456;self.guard=guard;self.max_live_packets=1
        self.peak_live_packets=0;self.credit_wait_seconds=0.;self.protocol_source=pin(transport.__file__)
        self.condition=threading.Condition();self.packets=deque();self.live={};self.used_bytes=0;self.peak_queue_bytes=0
        self.error=None;self.producer_identity=None;self.source=None;self.ready=threading.Event();self.closed_message=None
        self.plan_completions={};self.commands=queue.Queue(maxsize=2);self.submitted=set()
        self.credit=PacketCreditOwner();self.required_dead_refs=[]
        self.receiver_trace=[];self.receiver_raw_active=0;self.receiver_raw_peak=0;self.receiver_view_refs=[]
        self.delivered_ownership=[]
        original_return=self.credit.return_credit
        def checked_return():
            require(all(r()is None for r in self.required_dead_refs),'Parent AND receiver payload refs dead before actual kernel post')
            original_return()
        self.credit.return_credit=checked_return # Observer assertion around REAL kernel post.
        env=os.environ.copy();env['ASR18_PACKET_CREDIT_DESCRIPTOR']=json.dumps(self.credit.descriptor)
        self.log=(self.folder/'CPU_stderr.log').open('xb')
        self.process=subprocess.Popen([sys.executable,'-Xutf8','-B',str(Path(__file__).resolve()),'--role','worker',
            '--output',str(self.folder),'--fault',fault],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            env=env,bufsize=0,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.launcher=identity(self.process.pid);TRACKED[self.launcher['pid']]=self.launcher['create_time']
        self._sender=threading.Thread(target=self._send_commands,daemon=True)
        self._receiver=threading.Thread(target=self._observed_receive,daemon=True);self._sender.start();self._receiver.start()
        original_close=self.credit.close
        def checked_owner_close():
            require(self.process.poll()is not None and not self._sender.is_alive()and not self._receiver.is_alive(),
                'Parent kernel handles close ONLY after actual process exit and transport joins')
            return original_close()
        self.credit.close=checked_owner_close
        until(lambda:self.ready.is_set())
        if self.producer_identity is not None:TRACKED[self.producer_identity['pid']]=self.producer_identity['create_time']
    def _receive_profile(self,frame,kind,arg):
        if frame.f_code is not transport._read_exact.__code__:return
        caller=frame.f_back
        if caller is None or caller.f_code is not transport.ResidentFeatureProducer._receive.__code__:return
        state=caller.f_locals;header=state.get('frame',{});spec=state.get('spec',{})
        if header.get('kind')!='update':return
        if kind=='call':
            count=frame.f_locals['count']
            require(count==spec['bytes']and self.used_bytes==header['array_bytes']
                and not self.live and not self.packets,
                'Actual receiver FULL byte reservation precedes first partial allocation; LIVE1 including partial')
            self.receiver_raw_active+=1;self.receiver_raw_peak=max(self.receiver_raw_peak,self.receiver_raw_active)
            require(self.receiver_raw_active==1,'No second receiver raw allocation/read overlaps')
            self.receiver_trace.append({'kind':'before_actual_raw_allocation','plan':header['plan_sha256'],
                'index':header['update_index'],'token':header['token'],'read_bytes':count,
                'full_packet_reserved_bytes':self.used_bytes,'live_packets_before_allocation':len(self.live),
                'monotonic':time.monotonic()})
        elif kind=='return':
            view=frame.f_locals.get('view')
            if view is not None:self.receiver_view_refs.append(weakref.ref(view))
            self.receiver_raw_active-=1
            require(self.receiver_raw_active==0,'Actual raw read unwound before subsequent allocation')
    def _observed_receive(self):
        sys.setprofile(self._receive_profile)
        try:return transport.ResidentFeatureProducer._receive(self)
        finally:sys.setprofile(None)
    def get(self,timeout=300):
        packet=transport.ResidentFeatureProducer.get(self,timeout)
        self.required_dead_refs,types=payload_weakrefs([x for row in packet['arrays']for x in row.values()])
        self.delivered_ownership.append({'token':packet['token'],**types})
        return packet
    def closure(self,normal=None):
        who=self.producer_identity
        until(lambda:dead(self.launcher)and(who is None or dead(who)))
        trace_path=self.folder/'receiver_lifetime_trace.json';write(trace_path,self.receiver_trace)
        return {'actual_Popen_returncode':self.process.returncode,'launcher':self.launcher,'producer':who,
            'receiver_lifetime_trace':pin(trace_path),'receiver_raw_read_peak':self.receiver_raw_peak,
            'receiver_raw_read_active_at_closure':self.receiver_raw_active,
            'actual_read_exact_view_weakrefs_dead':all(r()is None for r in self.receiver_view_refs),
            'delivered_array_and_base_ownership':self.delivered_ownership,
            'raw_bytearray_not_weakrefable':True,
            'launcher_exact_dead':dead(self.launcher),'producer_exact_dead':None if who is None else dead(who),
            'threads_joined':not self._sender.is_alive()and not self._receiver.is_alive(),
            'pipe_handles_closed':self.process.stdin.closed and self.process.stdout.closed,
            'credit_owner':self.credit.receipt(),'normal':normal,'events':pin(self.folder/'events.jsonl')if(self.folder/'events.jsonl').exists()else None,
            'stderr':pin(self.folder/'CPU_stderr.log')}

def synthetic_plan(folder,name,initial,previous=None):
    # New local RNG objects ONLY; no model/training RNG/sampler is loaded or advanced.
    state={'optimizer_updates':initial+20,'carried_replay':{'replay_rng':random.Random(123).getstate(),
        'Portuguese_rng':random.Random(456).getstate()}}
    updates=[{k:('synthetic_'+k+'_'+str(initial+i+1))for k in
        ('native_transaction_sha256','before_sampler_sha256','after_sampler_sha256','batch_id')}for i in range(20)]
    plan={'kind':'SYNTHETIC_plan_no_authority','initial_committed_updates':initial,
        'initial_sampler_sha256':previous,'final_private_planning_sampler':state,'updates':updates}
    path=folder/(name+'.json');write(path,plan);return plan,pin(path)
def drop_then_release(h,packet):
    # NOT used: cannot drop the caller's reference inside a helper. Caller must del.
    raise AssertionError('Payload lifetime must be explicit at caller, before release')

def owner_process(a):
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    credit=PacketCreditOwner()
    try:
        descriptor_tmp=Path(a.output)/'owner_descriptor.json.tmp'
        write(descriptor_tmp,credit.descriptor)
        os.replace(descriptor_tmp,Path(a.output)/'owner_descriptor.json')
        sys.stdin.buffer.read(1) # Fixture parent sends1byte only after consumer wait starts.
        return 0
    finally:credit.close()

def main(a):
    out=Path(a.output).resolve();require(out.is_relative_to(Path('E:/gpu_training_scratchpad'))and not out.exists(),'Fresh E only')
    out.mkdir();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and all(os.environ.get(k)=='1'for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original CPU-only thread masks')
    require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Apply E runtime first')
    review=transport.bound(a.ROOT_review)
    require(review['accepted'] is True and review['reviewer']=='/root' and review['fixture_source']==pin(__file__)
        and review['helper_source']==pin(F/'asr18_owned_packet_credit_v2.py')
        and review['CPU_source']==pin(transport.__file__),'Separate exact ROOT finite CPU fixture scope required')
    for input_pin in review['input_pins']:require(pin(input_pin['path'])==input_pin,'Exact frozen CPU scope input')
    candidate=transport.bound(review['proposed_configuration'])
    require(candidate['resident_RAM_packet_credit_policy']['actual_CPU_control_review']is None,
        'Inactive production candidate has NO fake completed proof')
    require(candidate['resident_RAM_packet_credit_policy']['human_approval']==review['human_approval']
        and transport.bound(review['human_approval'])['answer']=='Approve lookahead and checkpoint restart',
        'Exact direct human logical-lookahead exception')
    transport.checked_resource_profile(candidate)
    require(candidate['resident_RAM_packet_credit_policy']['maximum_logical_packets']==2,
        'Max2 policy selected only by exact explicit approval')
    initial=guard();require(initial['system_memory']['commit_available_bytes']>=2000000000+268435456+512*1024**2
        and initial['system_memory']['physical_available_bytes']>=1500000000+268435456+512*1024**2,'Full prospective512MiB family')
    setup_numpy();entry={'parent':identity(),'launcher':identity(os.getppid())};write(out/'entry.json',entry)
    results={};owned=[]
    try:
        # Genuine max2/overflow/API/collision and actual blocked third acquisition; no stress/GPU.
        owner=PacketCreditOwner()
        try:
            results['kernel_initial_overflow']=expect(OSError,owner.return_credit)
            saved=owner.handles['credit'];owner.handles['credit']=0
            try:results['kernel_invalid_handle']=expect(OSError,owner.return_credit)
            finally:owner.handles['credit']=saved
            import asr18_owned_packet_credit_v2 as api
            original_uuid=api.uuid.uuid4
            class FixedNonce:hex=owner.descriptor['nonce']
            api.uuid.uuid4=lambda:FixedNonce()
            try:results['preexisting_rejected']=expect(ValueError,PacketCreditOwner)
            finally:api.uuid.uuid4=original_uuid
            consumer=PacketCreditConsumer(owner.descriptor)
            try:
                marker=ValueError('OriginalGuardExceptionIdentity')
                def fail_guard():raise marker
                try:consumer.acquire(phase='guard_identity',token='synthetic',check=fail_guard)
                except ValueError as error:require(error is marker,'Original callback exception identity retained')
                results['original_callback_exception_identity']=True
                consumer.acquire(phase='consume_A',token='synthetic_A',check=guard)
                consumer.acquire(phase='consume_B',token='synthetic_B',check=guard)
                started=threading.Event();finished=threading.Event();third=[]
                def real_third_acquire():
                    started.set()
                    try:third.append(consumer.acquire(phase='REAL_BLOCKED_C',token='synthetic_C',check=guard))
                    except BaseException as error:third.append({'error':repr(error)})
                    finally:finished.set()
                waiter=threading.Thread(target=real_third_acquire);waiter.start();started.wait(2);time.sleep(.1)
                require(not finished.is_set(),'Actual third logical acquire blocked while two permits outstanding')
                owner.return_credit();require(owner.last_release_previous_count==0,'Actual release previous0')
                waiter.join(10);require(not waiter.is_alive()and finished.is_set()and third and 'error'not in third[0],
                    'Third acquire wakes only after real post; waiter joined before handle closure')
                owner.return_credit();owner.return_credit()
                require(owner.last_release_previous_count==1,'Actual final release previous1 restores two permits')
                results['genuine_two_credit_third_wait_and_previous01']={'third':third,'owner':owner.receipt(),'waiter_joined':True}
                for field,value in [('logical_packet_maximum',1),('kind','wrong_kind'),('nonce','0'*32),
                    ('helper_source',{'wrong':True}),('owner',{'pid':owner.descriptor['owner']['pid'],
                    'create_time':owner.descriptor['owner']['create_time']+1})]:
                    altered=copy.deepcopy(owner.descriptor);altered[field]=value
                    results['descriptor_reject_'+field]=expect(ValueError,lambda:PacketCreditConsumer(altered))
                consumer.acquire(phase='consume_A_again',token='synthetic_A',check=guard)
                consumer.acquire(phase='consume_B_again',token='synthetic_B',check=guard)
                ticks=iter([0.,601.])
                try:consumer.acquire(phase='FAKE_CLOCK_DEADLINE_ONLY',token='synthetic_unreleased',check=guard,clock=lambda:next(ticks))
                except TimeoutError as error:results['fake_clock_deadline_not_600s_runtime']=error.packet_credit_timeout
                else:raise AssertionError('Expected typed deadline')
            finally:consumer.close()
        finally:owner.close()
        results['original_full_cap_AST_no_large_allocation']=full_packet_cap_source_control()
        plan1,pin1=synthetic_plan(out,'plan1',0)
        plan2,pin2=synthetic_plan(out,'plan2',20,transport.digest(plan1['final_private_planning_sampler']))
        h=Harness(out/'ordered_two_plans');owned.append(h);h.submit(pin1);h.submit(pin2)
        ready=h.wait_update_ready(transport.digest(plan1),0)
        require(len(h.packets)==1 and h.live[ready['token']]['delivered'] is False,'Readiness does not pop/mark delivered')
        results['unknown_release']=expect(KeyError,lambda:h.release('not_a_token'))
        results['undelivered_release']=expect(ValueError,lambda:h.release(ready['token']))
        until(lambda:len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2)
        time.sleep(.1);guard()
        require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2,
            'A queued plus B preparation allowed; C logical acquire blocked')
        require(all(e['index']==0 and e['plan']==transport.digest(plan1)for e in h.receiver_trace),
            'No B raw receiver allocation while A remains queued')
        p=h.next_update(transport.digest(plan1),0);token=p['token']
        require(any(x.get('memoryview',0)>0 for x in [h.delivered_ownership[-1]['weakref_object_types']]),
            'Actual ndarray base memoryview ownership observed')
        before_posts=h.credit.posts
        results['observer_rejects_premature_payload_release']=expect(ValueError,lambda:h.release(token))
        require(h.credit.posts==before_posts and h.live[token]['delivered'],
            'Assertion-only fixture guard rejects early release WITHOUT kernel post or token mutation')
        results['normal_close_with_live_rejected']=expect(ValueError,h.close)
        time.sleep(.1);guard();require(len([e for e in read_events(h.folder)if e['kind']=='synthetic_prepare_started'])==2,
            'C still blocked with A delivered-unreleased and B allowed')
        require(all(e['index']==0 for e in h.receiver_trace),'No B raw allocation while A delivered-unreleased')
        del p;h.release(token);results['duplicate_release']=expect(KeyError,lambda:h.release(token))
        for index in range(1,20):
            p=h.next_update(transport.digest(plan1),index);token=p['token']
            del p;h.release(token)
        from asr18_resident_periodic_CPU_quiescence_v3 import close_private_preparation_for_DEV
        closed=close_private_preparation_for_DEV(h,(plan2,pin2),committed_updates=20,guard=guard)
        require(closed['private_packets_discarded']==20 and closed['actual_native_updates']==0
            and closed['same_exact_plan_must_be_resubmitted_after_DEV'] is True,'Original full20 private drain, no authority')
        events=read_events(h.folder)
        prepared=[e for e in events if e['kind']=='synthetic_prepare_started']
        sent=[e for e in events if e['kind']=='synthetic_sent_payload_refs_dead']
        require(len(prepared)==len(sent)==40,'All40 ordered synthetic packets sent once')
        expected=[(transport.digest(plan1),i)for i in range(20)]+[(transport.digest(plan2),i)for i in range(20)]
        require([(e['plan'],e['index'])for e in prepared]==expected,'Real serial producer current-to-queued plan order')
        require(all(sent[i]['monotonic']<=prepared[i+1]['monotonic']for i in range(39)),
            'Every actual sent array/base/view dead BEFORE following actual acquire succeeds')
        require(h.credit.last_release_previous_count==1 and h.credit.posts==40,
            'Final last-current/queued-plan drain release previous1 restores initial two permits exactly')
        results['ordered_two_plans_and_periodic_drain']=h.closure(closed);owned.remove(h)
        # Same exact plan/index/digest on fresh owned transport after normal phase.
        h=Harness(out/'same_plan_resubmit');owned.append(h);h.submit(pin2)
        again=close_private_preparation_for_DEV(h,(plan2,pin2),committed_updates=20,guard=guard)
        require(again['unconsumed_private_plan']==pin2 and pin(Path(pin2['path']))==pin2,'Same private plan never regenerated')
        results['same_plan_resubmit']=h.closure(again);owned.remove(h)
        for fault in ['prepare_error','send_error','send_eof','receiver_hash_error','receiver_full_cap']:
            h=Harness(out/fault,fault);owned.append(h);h.submit(pin1)
            until(lambda:h.error is not None)
            actual={'type':type(h.error).__name__,'message':str(h.error)}
            if fault=='receiver_full_cap':require(not h.receiver_trace,
                'Actual receiver rejects over-full-cap header BEFORE payload allocation')
            h.abort();results[fault]={'pre_abort_actual_error':actual,'closure':h.closure()};owned.remove(h)
        h=Harness(out/'cancel_before_first_prepare');owned.append(h);h.credit.cancel();h.submit(pin1)
        until(lambda:h.error is not None)
        require(not any(e['kind']=='synthetic_prepare_started'for e in read_events(h.folder)),
            'Cancelled permit must prevent first preparation')
        h.abort();results['cancel_before_first_prepare']=h.closure();owned.remove(h)
        h=Harness(out/'errored_release');owned.append(h);h.submit(pin1)
        p=h.next_update(transport.digest(plan1),0);token=p['token'];del p
        marker=RuntimeError('SyntheticOriginalParentFailure');h._fail(marker)
        results['errored_release']=expect(ValueError,lambda:h.release(token))
        h.abort();results['errored_release']['closure']=h.closure();owned.remove(h)
        h=Harness(out/'cancel_during_credit_wait');owned.append(h);h.submit(pin1)
        h.wait_update_ready(transport.digest(plan1),0)
        h.abort();results['cancel_during_credit_wait']=h.closure();owned.remove(h)
        # Real owner process death, observed through a separate consumer handle.
        owner_dir=out/'exact_owner_death';owner_dir.mkdir()
        with (owner_dir/'stderr.log').open('xb')as err:
            process=subprocess.Popen([sys.executable,'-Xutf8','-B',str(Path(__file__).resolve()),'--role','owner_process',
                '--output',str(owner_dir)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            launcher=identity(process.pid);TRACKED[launcher['pid']]=launcher['create_time']
            until(lambda:(owner_dir/'owner_descriptor.json').exists())
            descriptor=json.loads((owner_dir/'owner_descriptor.json').read_text());TRACKED[descriptor['owner']['pid']]=descriptor['owner']['create_time']
            consumer=PacketCreditConsumer(descriptor);observed=[]
            try:
                consumer.acquire(phase='consume_A',token='owner_death_synthetic_A',check=guard)
                consumer.acquire(phase='consume_B',token='owner_death_synthetic_B',check=guard)
                def observe_death():
                    try:consumer.acquire(phase='WAIT_REAL_OWNER_DEATH',token='owner_death_synthetic',check=guard)
                    except BaseException as error:observed.append({'type':type(error).__name__,'message':str(error)})
                waiter=threading.Thread(target=observe_death);waiter.start();time.sleep(.1)
                require(waiter.is_alive()and not observed,'Actual third acquire remains blocked before owner exits')
                process.stdin.write(b'x');process.stdin.flush();code=process.wait(timeout=10);waiter.join(timeout=10)
                require(not waiter.is_alive() and observed and 'owner exited' in observed[0]['message'],'Real exact owner-death signal; waiter joined before handle close')
                until(lambda:dead(launcher)and dead(descriptor['owner']))
                results['genuine_exact_owner_death']={'actual_Popen_wait_code':code,'launcher':launcher,'owner':descriptor['owner'],
                    'all_exact_dead':True,'waiter_joined':True,'observed':observed}
            finally:
                if process.poll()is None:process.terminate();process.wait(timeout=10)
                if 'waiter' in locals()and waiter.is_alive():waiter.join(timeout=10)
                require('waiter'not in locals()or not waiter.is_alive(),'Unwind real wait before closing SAME consumer handles')
                process.stdin.close();consumer.close()
        for input_pin in review['input_pins']:require(pin(input_pin['path'])==input_pin,'Frozen input unchanged after actual control')
        write(out/'report.json',{'kind':'ASR18_actual_owned_two_logical_packet_one_receiver_transport_CPU_control_v1','passed':True,
            'maximum_logical_packets':2,'maximum_live_receiver_packets':1,
            'payload_lifetime_and_partial_receive_interleavings_passed':True,
            'constructor_and_small_synthetic_preparation_are_explicit_doubles':True,
            'completed_production_CPU_review_remained_None':True,
            'two_full_copy_claim_conditional_on_original_packet_geometry_admission':True,
            'frontend_growth_allocator_workspace_kernel_pipe_are_extra_not_512MiB_bound':True,
            'synthetic_array_geometry':{'features':FEATURE_SHAPE,'mask':MASK_SHAPE,'bytes':SYNTHETIC_ARRAY_BYTES},
            'fake600s_clock_case_not_real600s_wait':True,'real_audio_frontend_or_GPU_fit_not_proved':True,
            'human_approval':review['human_approval'],'input_pins':review['input_pins'],
            'genuine_named_kernel_credit_and_pipe_flow_passed':True,'synthetic_packets_no_audio_model':True,
            'actual_native_optimizer_updates':0,'actual_audio_decode':0,'actual_authoritative_sampler_or_journal_operation':False,
            'original_CPU11_numerics_not_executed_here':True,'results':results,'parent':entry,'sampled_family_private_peak_bytes':PEAK,
            'initial_guard':initial,'final_guard':guard(),'source':pin(__file__),'CPU_source':pin(transport.__file__),
            'helper_source':pin(F/'asr18_owned_packet_credit_v2.py'),'quiescence_helper':pin(S/'asr18_resident_periodic_CPU_quiescence_v3.py')})
        print(json.dumps({'report':pin(out/'report.json'),'entry':entry}),flush=True)
    except BaseException as error:
        write(out/'actual_failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'partial_results':results,'parent':entry})
        raise
    finally:
        for h in owned:
            try:h.abort()
            except BaseException as error:write(h.folder/'fixture_abort_failure.json',{'error':repr(error)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['main','worker','owner_process'],default='main')
    p.add_argument('--output',required=True);p.add_argument('--ROOT-review');p.add_argument('--fault',default='none')
    a=p.parse_args()
    if a.role=='worker':raise SystemExit(worker(a))
    if a.role=='owner_process':raise SystemExit(owner_process(a))
    main(a)
