"""Finite CPU transport/control fixture, never production audio/model preparation."""
import argparse,ast,copy,gc,hashlib,importlib.util,json,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(__file__).parent;D=N/'resident_RAM_single_packet_resource_source_v1'
CPU=S/'fastmss_resident_feature_stream_v6.py'
sys.path[:0]=[str(S),str(R)]
def require(ok,message):
    if not ok:raise ValueError(message)
def pin(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load():
    spec=importlib.util.spec_from_file_location('fixture_real_resident_stream6',CPU)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def worker(a):
    import numpy as np
    from fastmss_owned_process_closure_v2 import identity
    m=load();out=sys.stdout.buffer;inp=sys.stdin.buffer;sys.stdout=sys.stderr
    source={'protocol':pin(CPU),'fixture_generator':pin(__file__),'production_CPU_preparation':False}
    m._send(out,{'kind':'ready','producer_identity':identity(os.getpid()),'source':source,'fixture':True})
    while True:
        command=m._header(inp)
        if command['kind']=='close':
            m._send(out,{'kind':'closed','completed_updates':a.updates,'fixture':True,'production_CPU_preparation':False});return 0
        require(command['kind']=='prepare_plan','Fixture command order')
        sha=command['plan_sha256']
        if a.mode=='error':
            m._send(out,{'kind':'error','message':'fixture controlled CPU exception',
                'actual_resource_telemetry':{'fixture_only':True,'completed_updates':0}});return 7
        if a.mode=='oversized':
            m._send(out,{'kind':'update','metadata':{'rows':[]},'metadata_sha256':m.digest({'rows':[]}),
                'array_bytes':m.PACKET_LIMIT+1,'array_descriptors':[{'bytes':m.PACKET_LIMIT+1}]});return 8
        for i in range(a.updates):
            if a.mode=='full':f=np.zeros((67108863,),dtype=np.float32);mask=np.ones((4,),dtype=np.bool_)
            else:f=np.full((1,256,128),i+1,dtype=np.float32);mask=np.ones((1,256),dtype=np.bool_)
            arrays=[f,mask]
            descriptors=[{'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.nbytes,
                'sha256':hashlib.sha256(memoryview(x).cast('B')).hexdigest()}for x in arrays]
            row={'input_features':{'shape':list(f.shape),'array_sha256':descriptors[0]['sha256']},
                'attention_mask':{'shape':list(mask.shape),'array_sha256':descriptors[1]['sha256']}}
            metadata={'kind':'fixture_control_metadata_not_scientific_preparation','rows':[row],
                'fixture_update':i,'actual_optimizer_updates':0,'actual_draws':0}
            frame={'kind':'update','token':f'fixture:{os.getpid()}:{i}','plan_sha256':sha,'update_index':i,
                'metadata':metadata,'metadata_sha256':m.digest(metadata),'source':source,
                'array_descriptors':descriptors,'array_bytes':sum(x.nbytes for x in arrays)}
            (Path(a.markers)/f'before_send_{i}').write_text('fixture wire send started',encoding='utf-8')
            if a.mode=='partial':
                m._send(out,frame)
                out.write(memoryview(f).cast('B')[:8]);out.flush();return 9
            m._send(out,frame,arrays)
            (Path(a.markers)/f'after_send_{i}').write_text('fixture wire send completed',encoding='utf-8')
            del f,mask,arrays,descriptors,row,metadata,frame
        m._send(out,{'kind':'plan_complete','plan_sha256':sha,'updates':a.updates,'fixture':True})

def main(a):
    import psutil,types
    from resource_watchdog_v2 import sample_memory
    from fastmss_owned_process_closure_v2 import identity
    folder=Path(a.output).resolve()
    require(folder.is_relative_to(Path(r'E:\gpu_training_scratchpad'))and not folder.exists(),'Fresh E fixture only')
    folder.mkdir();psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    for name in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[name]).resolve().is_relative_to(Path(r'E:\gpu_training_scratchpad')),'E storage environment')
    require(os.environ['CUDA_VISIBLE_DEVICES']=='-1'and all(os.environ[x]=='1'for x in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only four thread caps')
    began=time.monotonic();stop=threading.Event();failure=[];samples=[];peak=0;peak_family=[]
    def guard():
        nonlocal peak,peak_family
        require(time.monotonic()-began<180,'Finite180s CPU fixture')
        value=sample_memory(os.getpid(),'single_live_packet_CPU_fixture');s=value['system_memory']
        require(value['paired_sampling_within_one_second']and s['available'],'Fresh paired actual telemetry')
        require(s['commit_available_bytes']>=2_000_000_000 and s['physical_available_bytes']>=1_500_000_000
            and s['commit_used_bytes']/s['commit_limit_bytes']<.97,'Original actual headroom guards retained')
        family=[]
        for proc in [psutil.Process(),*psutil.Process().children(recursive=True)]:
            try:family.append({'pid':proc.pid,'create_time':proc.create_time(),'private_bytes':proc.memory_info().private})
            except psutil.NoSuchProcess:pass
        total=sum(x['private_bytes']for x in family)
        require(total<2_000_000_000,'Finite CPU fixture whole-family2GB private cap')
        if total>peak:peak=total;peak_family=family
        require(shutil.disk_usage('E:/').free>=100_000_000_000,'Original E100GB reserve')
        if failure:raise RuntimeError('Fixture observer failed: '+repr(failure[0]))
        return value
    initial=guard();require(initial['system_memory']['commit_available_bytes']-2_000_000_000-268435456>=2_000_000_000,
        'Full2GB fixture plus unchanged extra256MiB reservation prospectively admitted; no freed credit')
    write(folder/'initial_admission.json',{'sample':initial,'full_fixture_reservation':2_000_000_000,
        'unchanged_extra_prefetch_reservation':268435456,'freed_credit':0,'source':pin(__file__)})
    def watch():
        while not stop.wait(.05):
            try:samples.append({'monotonic':time.monotonic(),'commit_available_bytes':guard()['system_memory']['commit_available_bytes']})
            except BaseException as exc:failure.append(exc);return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    m=load();handoff=read(D/'source_handoff_v1.json');configuration=handoff['configuration'];processes=[];results=[]
    active=[];actual_Popen=subprocess.Popen
    try:
        # Reconstruct the entire old source from the declared changes, not just selected functions.
        inverse=[]
        for entry in handoff['source_maps'].values():
            require(pin(entry['old']['path'])==entry['old']and pin(entry['new']['path'])==entry['new'],'Frozen exact source pins')
            text=Path(entry['new']['path']).read_text(encoding='utf-8')
            for change in reversed(entry['declared_replacements']):
                require(text.count(change['new'])==change['count'],'Exact inverse occurrence')
                text=text.replace(change['new'],change['old'])
            require(text==Path(entry['old']['path']).read_text(encoding='utf-8'),'Whole source inverse equality')
            inverse.append(entry['new'])
        oldtree=ast.parse(Path(handoff['source_maps'][Path(handoff['sources']['driver']['path']).name]['old']['path']).read_text(encoding='utf-8'))
        newtree=ast.parse(Path(handoff['sources']['driver']['path']).read_text(encoding='utf-8'))
        olddefs={x.name:ast.dump(x,include_attributes=False)for x in oldtree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        newdefs={x.name:ast.dump(x,include_attributes=False)for x in newtree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        preserved=[k for k in olddefs if k not in ('checked_headroom_resume','main')]
        require(all(olddefs[k]==newdefs[k]for k in preserved),'Every non-entry numerical/state/DEV/checkpoint definition unchanged')
        m.checked_resource_profile(m.bound(configuration))
        def expect(cls,callback):
            try:callback()
            except cls as exc:return {'exception_type':type(exc).__name__,'message':str(exc)}
            raise AssertionError('Expected '+cls.__name__)
        # Explicit synthetic resource samples test scheduling branches only; real guard remains above.
        budget=1024;counter=[0]
        def value(available):return {'worker_private_memory':{'private_bytes':1024},'system_memory':{'commit_available_bytes':available}}
        policy=m.AvailableMemoryPolicy();require(policy.allocation_wait_seconds==300 and policy.prospective_reserve_bytes==2500000000,'Exact production reserve/wait')
        def blocked_then_ready():counter[0]+=1;return value(2500000000+budget-(1 if counter[0]<=2 else 0))
        admitted=policy.admit(budget,blocked_then_ready)
        require(counter[0]==3 and policy.allocation_wait_count==1 and admitted['allocation_wait_seconds']>=.19,'Wait before allocation then exact reserve equality pass')
        denied=expect(ValueError,lambda:policy.admit(policy.private_limit,lambda:value(20_000_000_000)))
        marker=RuntimeError('fixture actual-guard propagation')
        def guard_error():raise marker
        require(expect(RuntimeError,lambda:policy.admit(budget,guard_error))['message']==str(marker),'Original actual guard failure propagates')
        timeout=m.AvailableMemoryPolicy();timeout.allocation_wait_seconds=.15 # Fixture-only short deadline, production remains300.
        timed=expect(TimeoutError,lambda:timeout.admit(budget,lambda:value(2500000000+budget-1)))
        results.append({'control':'prospective_resource_scheduling','synthetic_inputs':True,'production_default_seconds':300,
            'stronger_reserve_bytes':2500000000,'wait_then_pass':admitted,'private_cap_rejection':denied,
            'timeout_rejection':timed,'actual_guard_exception_propagation':True,'sampler_or_optimizer_objects_constructed':False})
        def create(case,mode,updates=3):
            casefolder=folder/case;casefolder.mkdir();markers=casefolder/'wire_markers';markers.mkdir()
            plan=casefolder/'plan.json';write(plan,{'kind':'CPU_protocol_fixture_only','case':case,'updates':updates,'draws':0,'native_updates':0})
            capture={}
            def launch(command,**kwargs):
                require(Path(command[3]).resolve()==CPU.resolve()and command[4]=='--worker','Actual production constructor command captured')
                require(kwargs['env']['CUDA_VISIBLE_DEVICES']=='-1','No GPU child')
                actual=[command[0],'-Xutf8','-B',str(Path(__file__).resolve()),'--fixture-worker','--mode',mode,
                    '--updates',str(updates),'--markers',str(markers)]
                kwargs['creationflags']=kwargs.get('creationflags',0)|subprocess.BELOW_NORMAL_PRIORITY_CLASS
                proc=actual_Popen(actual,**kwargs);capture.update(actual_command=actual,production_command=command,
                    launcher_identity=identity(proc.pid),fixture_source=pin(__file__));processes.append(proc);return proc
            # Fixture-only process launch substitution. Receiver/sender/credit/close methods are unmodified source6.
            m.subprocess=types.SimpleNamespace(Popen=launch,PIPE=subprocess.PIPE,CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW)
            prod=m.ResidentFeatureProducer(configuration,casefolder/'CPU',queue_bytes=268435456,guard=guard);active.append(prod)
            deadline=time.monotonic()+15
            while not prod.ready.wait(.05):guard();require(time.monotonic()<deadline,'Bounded fixture ready')
            if prod.error:raise prod.error
            require(prod.source['production_CPU_preparation']is False,'Fixture is not a production preparation claim')
            capture['actual_worker_identity']=prod.producer_identity;write(casefolder/'actual_child_identity.json',capture)
            sha=prod.submit(pin(plan));return prod,sha,casefolder,markers
        def release(prod,packet):
            token=packet['token'];prod.release(token);del packet
        def wait_marker(path):
            deadline=time.monotonic()+10
            while not path.exists():guard();require(time.monotonic()<deadline,'Bounded fixture wire marker');time.sleep(.02)
        def check_closed(prod):
            require(not prod._receiver.is_alive()and not prod._sender.is_alive()and prod.process.poll()is not None,
                'Actual child waited and both real transport threads joined')
            require(prod.process.stdin.closed and prod.process.stdout.closed and prod.log.closed,'All owned pipe/log handles closed')
            active.remove(prod)
        prod,sha,casefolder,markers=create('ordered_release','ordinary')
        duplicate=expect(ValueError,lambda:prod.submit(pin(casefolder/'plan.json')))
        packet=prod.next_update(sha,0,timeout=15);token=packet['token']
        wait_marker(markers/'before_send_1');time.sleep(.35);guard()
        require(len(prod.live)==1 and prod.live[token]['delivered']and not prod.packets and prod.used_bytes==packet['array_bytes'],
            'Delivered-unreleased packet still consumes the sole live credit')
        require(not(markers/'after_send_1').exists(),'Real pipe producer blocked behind receiver credit')
        unreleased=expect(ValueError,prod.close)
        del packet;prod.release(token)
        duplicate_release=expect(KeyError,lambda:prod.release(token))
        observed=[0]
        for index in (1,2):
            packet=prod.next_update(sha,index,timeout=15);observed.append(packet['update_index'])
            require(float(packet['arrays'][0]['input_features'].flat[0])==index+1 and bool(packet['arrays'][0]['attention_mask'].all()),'Exact ordered fixture array bytes')
            token=packet['token'];del packet;prod.release(token)
        closure=prod.close();check_closed(prod)
        require(observed==[0,1,2]and closure['peak_live_receiver_packets']==1 and closure['receiver_credit_wait_seconds']>=.3,'Order/wakeup/single-credit proved')
        results.append({'control':'real_pipe_order_release_backpressure','closure':closure,'ordered_indices':observed,
            'duplicate_plan_rejected':duplicate,'normal_close_unreleased_rejected':unreleased,
            'duplicate_release_rejected':duplicate_release,'delivered_unreleased_blocks':True})
        prod,sha,casefolder,markers=create('full_256MiB_packet','full',1)
        packet=prod.next_update(sha,0,timeout=30)
        require(packet['array_bytes']==268435456 and prod.used_bytes==268435456,'Entire original max packet accepted with no filtering')
        full_descriptor=packet['array_descriptors'];token=packet['token'];del packet;prod.release(token);gc.collect()
        closure=prod.close();check_closed(prod)
        results.append({'control':'full_original_256MiB_packet','entire_packet_bytes':268435456,'descriptors':full_descriptor,'closure':closure})
        for mode,exit_code,expected_type in [('error',7,RuntimeError),('partial',9,EOFError),('oversized',8,ValueError)]:
            prod,sha,casefolder,markers=create('rejection_'+mode,mode,1)
            rejected=expect(expected_type,lambda:prod.get(timeout=15))
            actual_code=prod.process.wait(timeout=15);require(actual_code==exit_code,'Exact fixture-generated abnormal child wait code')
            saved_error=pin(casefolder/'CPU/actual_CPU_failure.json')if mode=='error'else None
            prod.abort();check_closed(prod)
            results.append({'control':'abnormal_'+mode,'actual_fixture_child_exit_code':actual_code,
                'parent_exception':rejected,'original_CPU_error_frame':saved_error,'threads_joined':True,'handles_closed':True,
                'normal_preparation_or_process0_claimed':False})
        prod,sha,casefolder,markers=create('abort_credit_blocked','ordinary')
        packet=prod.next_update(sha,0,timeout=15);wait_marker(markers/'before_send_1');time.sleep(.1)
        prod.abort();actual_code=prod.process.poll();check_closed(prod);del packet;gc.collect()
        require(actual_code is not None and actual_code!=0,'Explicit aborted child is never called normal0')
        results.append({'control':'abort_delivered_unreleased_credit_blocked','actual_fixture_child_exit_code':actual_code,
            'threads_joined':True,'handles_closed':True,'normal_process0_claimed':False})
        # Original call/order validation rejects wrong requested consumption; no authoritative sampler exists in fixture.
        prod,sha,casefolder,markers=create('wrong_update_order','ordinary',1)
        rejected=expect(ValueError,lambda:prod.next_update(sha,1,timeout=15));prod.abort();check_closed(prod);gc.collect()
        results.append({'control':'wrong_update_order','rejected':rejected,'threads_joined':True,'handles_closed':True,'draws':0,'native_updates':0})
        require(not active and all(x.poll()is not None for x in processes),'All fixture children actually observed exited')
        guard()
        report={'kind':'ASR18_single_live_packet_resource_actual_CPU_controls_v1','passed':True,
            'fixture_source':pin(__file__),'cohort_handoff':pin(D/'source_handoff_v1.json'),'sources':handoff['sources'],
            'configuration':configuration,'source_inverse_proof':inverse,'unchanged_driver_AST_definitions':preserved,
            'controls':results,'actual_optimizer_updates':0,'actual_draws':0,'audio_or_model_loaded':False,'GPU_used':False,
            'production_preparation_claimed':False,'resource_guard_changes_or_freed_credit_claimed':False,
            'historical_queue_cause_or_measured_saving_claimed':False,'elapsed_seconds':time.monotonic()-began,
            'family_peak_private_bytes':peak,'family_at_peak':peak_family,'guard_samples':len(samples),'all_fixture_children_waited':True}
        write(folder/'report.json',report);print(json.dumps({'report':pin(folder/'report.json'),'peak':peak,'elapsed':report['elapsed_seconds']}),flush=True)
    finally:
        cleanup=[]
        for prod in list(active):
            try:prod.abort();cleanup.append({'pid':prod.process.pid,'wait':prod.process.poll(),'threads_joined':not prod._sender.is_alive()and not prod._receiver.is_alive()})
            except BaseException as exc:cleanup.append({'pid':prod.process.pid,'cleanup_error':repr(exc)})
        stop.set();watcher.join(timeout=5)
        write(folder/'actual_fixture_completion.json',{'fixture_identity':identity(os.getpid()),'source':pin(__file__),
            'observer_joined':not watcher.is_alive(),'observer_error':repr(failure[0])if failure else None,
            'actual_child_waits':[{'pid':x.pid,'actual_Popen_wait_code':x.poll()}for x in processes],
            'abnormal_cleanup':cleanup,'family_peak_private_bytes':peak,'elapsed_seconds':time.monotonic()-began})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture-worker',action='store_true');p.add_argument('--mode');p.add_argument('--updates',type=int);p.add_argument('--markers');p.add_argument('--output')
    a=p.parse_args()
    if a.fixture_worker:raise SystemExit(worker(a))
    main(a)
