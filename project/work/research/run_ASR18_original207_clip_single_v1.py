"""One new clip-only arm versus actual default oracle; no model/audio/loss.

Each arm starts from the same qualified original projected source state.
Only clip foreach=False changes; default Adam and complete step stay intact.
"""
import argparse,copy,ctypes,gc,hashlib,importlib.util,json,os,random,subprocess,sys,time
from ctypes import wintypes
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';LEASE=P/'work/gpu_training_owner.json';OWNER='/root/memory_diagnosis'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory,PROCESS_MEMORY_COUNTERS_EX
HELPER=P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py'
spec=importlib.util.spec_from_file_location('reviewed_component',HELPER);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
record=H.record

def checked(config,review):
    H.storage();c=read_json(config);r=read_json(review)
    require(c['kind']=='ASR18_original207_clip_single_saved_default_component_v1' and c['arms']==['clip_single']
        and c['production_enabled'] is False and c['actual_training_updates']==0,'Exact optimizer-only pair')
    require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==record(config)
        and r['source_bindings']==c['source_bindings'] and r['resource_policy']==c['resource_policy'],'Exact ROOT review')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Frozen source changed')
    for pin in c['evidence'].values():require(record(pin['path'])==pin,'Exact source evidence')
    require(c['source_bindings']['harness']==record(__file__) and record(c['GPU_python']['path'])==c['GPU_python'],'Exact worker implementation/runtime')
    return c

def highwater(cap):
    k=ctypes.WinDLL('kernel32',use_last_error=True);api=ctypes.WinDLL('psapi',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD];api.GetProcessMemoryInfo.restype=wintypes.BOOL
    v=PROCESS_MEMORY_COUNTERS_EX();v.cb=ctypes.sizeof(v)
    require(api.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(v),v.cb),'Owned OS private/lifetime telemetry')
    require(v.PrivateUsage<=cap and v.PeakPagefileUsage<=cap,'Current AND OS-lifetime worker cap')
    return {'current_private_bytes':int(v.PrivateUsage),'lifetime_private_peak_bytes':int(v.PeakPagefileUsage)}

def comparable(value):
    value=copy.deepcopy(value)
    for part in value['state']['Adam_partitions']:
        for group in part['param_groups']:
            require(group['foreach'] is None or group['foreach'] is False,'Only expected Adam foreach policy differs')
            group['foreach']=None
    value['state_sha256']=digest(value['state'])
    return value

def worker(config,review,folder,arm):
    c=checked(config,review);producer=H.pid_identity();write_json(folder/'boot.json',{'producer':producer,'source':record(__file__),'arm':arm})
    watch=H.Watch(folder,3_350_000_000,worker=True);code=1;torch=None;stages=[];began=time.monotonic()
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            lease=read_json(LEASE)
            if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid():break
            watch.check();time.sleep(.05)
        else:raise ValueError('Actual sole-owner worker handshake')
        def stage(name):
            watch.check();require(time.monotonic()-began<c['maximum_worker_seconds'],'Finite optimizer-only arm')
            if torch is not None and torch.cuda.is_initialized():torch.cuda.synchronize()
            watch.phase=name;v=sample_memory(os.getpid(),name);v['private_high_water']=highwater(3_350_000_000)
            if torch is not None and torch.cuda.is_initialized():
                v['cuda']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                    'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
                require(v['cuda']['peak_allocated']<=1_000_000_000,'Complete optimizer CUDA peak<=1GB')
            stages.append(v)
            with (folder/'stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
            return v
        stage('before_Torch_import')
        import torch as t
        torch=t;torch.set_num_threads(1);stage('after_Torch_import')
        require(torch.__version__=='2.7.1+cu128','Exact existing Torch')
        from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
        capacity=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
        torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0);torch.empty(0,device='cuda')
        ready=stage('after_CUDA_initialization');require(ready['worker_private_memory']['private_bytes']<=1_250_000_000,'Measured import/CUDA prerequisite<=1.25GB')
        from bpe_broad_full207_optimizer_v1 import BPEBroadFull207Optimizer
        from asr18_live_state_fingerprint_v1 import live_state_fingerprint,tensor_digest
        projection=read_json(c['evidence']['source_projection']['path']);preflight=read_json(c['evidence']['optimizer_preflight']['path'])
        require(projection['passed'] and preflight['passed'],'Actual original source projection/preflight')
        source_records={k:v['parent'] for k,v in projection['projections'].items()}
        for k,v in projection['projections'].items():
            require(record(v['projected']['path'])==v['projected'] and source_records[k]==preflight['pinned_sources'][k],'Exact original source tensors')
        stage('before_original_source_load')
        original=torch.load(projection['projections']['original']['projected']['path'],mmap=True,map_location='cpu',weights_only=True)
        bpe=torch.load(projection['projections']['bpe']['projected']['path'],mmap=True,map_location='cpu',weights_only=True)
        names=tuple(original['weights']);lora={n for n in names if n.endswith(('lora_A','lora_B'))}
        compact={'step':original['step'],'selected_step':original['selected_step'],
            'weights':{n:original['weights'][n].clone() if n in lora else None for n in names},
            'optimizer':{'param_groups':copy.deepcopy(original['optimizer']['param_groups']),
                'state':{i:{k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in old.items()}
                    for i,old in original['optimizer']['state'].items() if i>=15}}}
        del original;gc.collect()
        class Named207:
            def __init__(self):self.values={n:torch.nn.Parameter(bpe['weights'][n].detach().to('cuda',copy=True)) for n in names}
            def named_parameters(self):return iter(self.values.items())
        model=Named207();optimizer=BPEBroadFull207Optimizer(model,compact,bpe,source_records=source_records,torch_module=torch)
        require(len(optimizer.parameters)==207 and sum(p.numel()*p.element_size() for p in optimizer.parameters.values())==81_459_760,'Exact complete original207 geometry')
        random.setstate(bpe['python_rng']);torch.set_rng_state(bpe['torch_rng']);torch.cuda.set_rng_state_all(bpe['cuda_rng'])
        import numpy as np
        np.random.seed(1107) # Source bootstrap does not restore NumPy; identical explicit fixture-only state.
        del compact,bpe;gc.collect();torch.cuda.empty_cache();stage('after_source_transfer_release')
        initial=live_state_fingerprint(optimizer,torch=torch,guard=watch.check)
        for parameter in optimizer.parameters.values():
            parameter.grad=torch.empty_like(parameter);flat=parameter.grad.reshape(-1)
            for low in range(0,flat.numel(),1_048_576):
                watch.check();size=min(1_048_576,flat.numel()-low)
                host=torch.arange(low,low+size,dtype=torch.int32).remainder_(257).sub_(128).to(torch.float32).div_(4096)
                flat[low:low+size].copy_(host);del host
        parameter=flat=None;gc.collect();torch.cuda.synchronize()
        gradients={n:tensor_digest(p.grad,torch=torch,guard=watch.check) for n,p in optimizer.parameters.items()}
        before=stage('before_original_global_clip_and_Adam')
        remaining=512_000_000+750_000_000+64_000_000
        require(before['worker_private_memory']['private_bytes']+remaining<=3_350_000_000,'Complete512MB optimizer/cold +UNCHANGED750MB variation +64MB staging inside wholecap')
        phase_admission=[]
        for _ in range(3):
            watch.check();v=sample_memory(os.getpid(),'optimizer_full_remaining_prospective');system=v['system_memory'];extra=remaining+192_000_000
            ok=v['paired_sampling_within_one_second'] and system['available'] and v['worker_private_memory']['available']
            ok=ok and system['commit_used_bytes']+extra<.90*system['commit_limit_bytes'] and system['commit_available_bytes']-extra>2_000_000_000
            phase_admission.append({'passed':ok,'sample':v,'full_worker_remaining':remaining,'full_future_family_allowance':192_000_000});time.sleep(.05)
        write_json(folder/'before_optimizer_fresh_admission.json',{'passed':all(v['passed'] for v in phase_admission),'samples':phase_admission})
        require(all(v['passed'] for v in phase_admission),'Fresh complete optimizer phase/family admission')
        actual_groups=[[{k:copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in o.param_groups] for o in optimizer.optimizers]
        require(all(g['foreach'] is None for groups in actual_groups for g in groups),'Original default foreach=None required')
        require(arm=='clip_single','One new clip-only arm; default Adam None unchanged')
        clip=torch.nn.utils.clip_grad_norm_;original_steps=[]
        def observed_clip(*args,**kwargs):
            require('foreach' not in kwargs,'Original clip must not already override foreach');stage('before_original_global_clip');v=clip(*args,foreach=False,**kwargs);stage('after_original_global_clip');return v
        torch.nn.utils.clip_grad_norm_=observed_clip
        for index,o in enumerate(optimizer.optimizers):
            bound=o.step;original_steps.append(bound)
            def observed_step(*args,_step=bound,_index=index,**kwargs):
                stage(f'before_Adam_partition_{_index}');v=_step(*args,**kwargs);stage(f'after_Adam_partition_{_index}');return v
            o.step=observed_step
        try:norm=optimizer.step()
        finally:
            torch.nn.utils.clip_grad_norm_=clip
            for o,bound in zip(optimizer.optimizers,original_steps):o.step=bound
        stage('after_complete_original_207_step')
        final=live_state_fingerprint(optimizer,torch=torch,guard=watch.check)
        clipped={n:tensor_digest(p.grad,torch=torch,guard=watch.check) for n,p in optimizer.parameters.items()}
        require(initial['state']['RNG']==final['state']['RNG'] and optimizer.updates==1,'All4RNG untouched by deterministic optimizer component')
        write_json(folder/'initial_state.json',initial);write_json(folder/'final_state.json',final)
        result={'execution_completed':True,'producer':producer,'source':record(__file__),'configuration':record(config),'arm':arm,
            'actual_training_updates':0,'isolated_optimizer_updates':1,'original_source_only':True,'full_model_audio_loss_used':False,
            'initial_state':record(folder/'initial_state.json'),'final_state':record(folder/'final_state.json'),
            'initial_comparison_sha256':comparable(initial)['state_sha256'],'final_comparison_sha256':comparable(final)['state_sha256'],
            'full207_gradient_hashes':gradients,'full207_clipped_gradient_hashes':clipped,'norm':tensor_digest(norm,torch=torch,guard=watch.check),
            'actual_Adam_foreach':[o.param_groups[0]['foreach'] for o in optimizer.optimizers],'clip_execution_foreach':False,'all4RNG_unchanged':True,
            'phase_stages':stages,'wall_seconds':time.monotonic()-began,'production_enabled':False}
        optimizer.zero_grad();del norm,model,optimizer,initial,final,original_steps,clip,o,bound,observed_step,observed_clip;gc.collect();torch.cuda.empty_cache();stage('after_owned_cleanup')
        result['phase_stages']=stages;write_json(folder/'worker_report.json',result);code=0
    finally:watch.close(code)

def parent(config,review,folder):
    import psutil
    c=checked(config,review);require(folder.resolve().is_relative_to(E.resolve()) and not folder.exists(),'Fresh E-only pair');folder.mkdir()
    watch=H.Watch(folder,128_000_000);prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Idle soleGPU owner');code=1;reports=[];proofs=[]
    try:
        oracle=c['saved_default_oracle'];proof=read_json(oracle['actual_process0']['path']);baseline=read_json(oracle['report']['path'])
        oldconfig=read_json(oracle['configuration']['path']);oldreview=read_json(oracle['ROOT_review']['path']);oldguard=read_json(proof['guard']['path'])
        require(proof['actual_Popen_wait_code']==0 and proof['all_captured_identities_dead'] and not any(H.alive(v) for v in proof['captured_identities'])
            and proof['report']==oracle['report'] and proof['source']==baseline['source']==oldconfig['source_bindings']['harness']
            and record(proof['source']['path'])==proof['source'] and record(proof['guard']['path'])==proof['guard']
            and baseline['configuration']==oracle['configuration'] and oldreview['accepted'] is True and oldreview['configuration']==oracle['configuration']
            and oldguard['actual_exit_code']==0 and oldguard['monitor_clean'] and oldguard['monitor_joined'] and oldguard['pid']==proof['native']['pid']
            and baseline['producer']==proof['native'] and baseline['arm']=='default' and baseline['actual_Adam_foreach']==[None,None,None],
            'Exact saved actual default oracle0; outer old pair failure is preserved separately')
        for key in ('initial_state','final_state'):
            require(record(baseline[key]['path'])==baseline[key],'Immutable actual original default state fingerprint')
            original=read_json(baseline[key]['path']);require(original['state_sha256']==digest(original['state'])==comparable(original)['state_sha256'],'Default has no differing foreach metadata')
        for arm in c['arms']:
            admission=H.fresh_admit(c);write_json(folder/(arm+'_fresh_admission.json'),admission);require(admission['passed'],'Fresh full3.542GB family, no unchanged retry')
            path=folder/arm;path.mkdir();known={};child=None;native=None
            try:
                with (path/'worker.log').open('xb') as log:
                    child=subprocess.Popen([c['GPU_python']['path'],'-X','utf8','-B',str(Path(__file__).resolve()),'--config',config,'--ROOT-review',review,
                        '--folder',str(path),'--worker',arm],cwd=P,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    began=time.monotonic()
                    while child.poll() is None:
                        watch.check();require(time.monotonic()-began<c['maximum_worker_seconds']+30,'Finite component wall time')
                        try:
                            for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                                try:v=H.pid_identity(process.pid);known[(v['pid'],v['create_time'])]=v
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        if native is None and (path/'boot.json').exists():
                            boot=read_json(path/'boot.json');native=boot['producer'];require(boot['arm']==arm and boot['source']==record(__file__) and H.alive(native) and (native['pid'],native['create_time']) in known,'Exact live owned worker')
                            lease=copy.deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],actual_training_optimizer_running=False);write_json(LEASE,lease)
                        own=sample_memory(os.getpid(),'Adam_parent')['worker_private_memory'];require(own['available'],'Available parent private');total=own['private_bytes'];shims=0
                        for v in known.values():
                            if not H.alive(v):continue
                            w=sample_memory(v['pid'],'Adam_owned_family')['worker_private_memory']
                            if not w['available'] and not H.alive(v):continue
                            require(w['available'],'Available exact worker/shim');total+=w['private_bytes']
                            if native is None or native['pid']!=v['pid']:shims+=w['private_bytes']
                        require(total<=3_542_000_000 and shims<=64_000_000,'Whole family/shim hardcaps');time.sleep(.05)
                    actual=child.wait();deadline=time.monotonic()+2
                    while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.02)
                    require(actual==0 and native is not None and not any(H.alive(v) for v in known.values()),'Real Popen0 and all captured identities dead')
                    g=read_json(path/'guard_closure.json');r=read_json(path/'worker_report.json')
                    require(g['actual_exit_code']==0 and g['pid']==native['pid'] and g['monitor_clean'] and g['monitor_joined'] and g['error'] is None
                        and g['samples']>=3 and r['execution_completed'] and r['producer']==native and r['configuration']==record(config),'Exact clean owned actual0')
                    proof={'actual_Popen_wait_code':0,'native':native,'captured_identities':list(known.values()),'all_captured_identities_dead':True,
                        'source':record(__file__),'configuration':record(config),'report':record(path/'worker_report.json'),'guard':record(path/'guard_closure.json')}
                    write_json(path/'actual_process0.json',proof);reports.append(r);proofs.append(record(path/'actual_process0.json'))
            finally:
                for v in reversed(list(known.values())):
                    if H.alive(v):psutil.Process(v['pid']).terminate()
                deadline=time.monotonic()+10
                while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.05)
                require(not any(H.alive(v) for v in known.values()),'Every actual family identity dead before release')
                if child is not None:
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=5)
                write_json(LEASE,prior)
        require(reports[0]['actual_Adam_foreach']==[None,None,None] and reports[0]['clip_execution_foreach'] is False,'Only clipping foreach policy changed')
        checks={k:baseline[k]==reports[0][k] for k in ('initial_comparison_sha256','final_comparison_sha256','full207_gradient_hashes','full207_clipped_gradient_hashes','norm')}
        write_json(folder/'paired_report.json',{'execution_completed':True,'passed':all(checks.values()),'exact_checks':checks,'arms':proofs,
            'full_raw_state_bytes_compared_no_policy_exclusions':True,'saved_default_oracle':oracle,'all_other_scalar_policies_and_tensor_bytes_compared':True,'actual_training_updates':0,'production_enabled':False})
        require(all(checks.values()),'Strict complete207/threeAdam/all4RNG equality; preserve numerical failure');code=0
    finally:watch.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True);p.add_argument('--worker',choices=('clip_single',));a=p.parse_args()
    if a.worker:worker(a.config,a.ROOT_review,Path(a.folder),a.worker)
    else:parent(a.config,a.ROOT_review,Path(a.folder))
