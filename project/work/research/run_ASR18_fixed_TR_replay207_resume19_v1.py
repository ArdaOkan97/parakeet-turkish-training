"""ONE real short TR diagnostic, original207/threeAdam, no sampler commit.

CPU original-frontend preparation, training and exact recovery are distinct
owned children. No launch without exact ROOT source/config/resource review.
"""
import argparse,copy,ctypes,gc,hashlib,importlib.util,json,os,secrets,shutil,subprocess,sys,time
from ctypes import wintypes
from functools import partial
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');A=E/'analysis/ASR18_memory_diagnosis_v1';LEASE=P/'work/gpu_training_owner.json';OWNER='/root/memory_diagnosis'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory,PROCESS_MEMORY_COUNTERS_EX
spec=importlib.util.spec_from_file_location('component_guard',P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)
record=H.record

def checked(config,review):
    H.storage();c=read_json(config);r=read_json(review)
    require(c['kind']=='ASR18_fixed20_resume_checkpoint1_remaining19_diagnostic_v1' and c['production_enabled'] is False
        and c['native_updates']==20 and c['new_optimizer_updates']==19 and c['sampler_draws']==0 and c['teacher_or_replay_forward'] is True,'Exact one-row unqualified diagnostic')
    require(r['accepted'] is True and r['reviewer']=='/root' and r['configuration']==record(config)
        and r['source_bindings']==c['source_bindings'] and r['resource_policy']==c['resource_policy'],'Exact ROOT component approval')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Frozen source changed')
    require(c['source_bindings']['harness']==record(__file__),'Exact driver source')
    for pin in c['evidence'].values():require(record(pin['path'])==pin,'Exact small prior evidence')
    require(record(c['CPU_python']['path'])==c['CPU_python'] and record(c['GPU_python']['path'])==c['GPU_python'],'Exact existing interpreters')
    return c

def current_lifetime(cap):
    k=ctypes.WinDLL('kernel32',use_last_error=True);p=ctypes.WinDLL('psapi',use_last_error=True);k.GetCurrentProcess.restype=wintypes.HANDLE
    p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    p.GetProcessMemoryInfo.restype=wintypes.BOOL;v=PROCESS_MEMORY_COUNTERS_EX();v.cb=ctypes.sizeof(v)
    require(p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(v),v.cb),'Available owned lifetime telemetry')
    require(v.PrivateUsage<=cap and v.PeakPagefileUsage<=cap,'Actual current AND lifetime private cap, including short spikes')
    return {'current_private_bytes':int(v.PrivateUsage),'lifetime_private_peak_bytes':int(v.PeakPagefileUsage)}

def fresh(increment,guard):
    samples=[]
    for _ in range(3):
        guard();v=sample_memory(os.getpid(),'single_TR207_prospective');s=v['system_memory'];w=v['worker_private_memory']
        fits=v['paired_sampling_within_one_second'] and s['available'] and w['available']
        fits=fits and s['commit_used_bytes']+increment<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-increment>2_000_000_000
        samples.append({'sample':v,'incremental_forecast_bytes':increment,'passed':fits});time.sleep(.05)
    return {'passed':all(v['passed'] for v in samples),'samples':samples}

def prepare(c,folder,guard):
    from asr18_fixed_TR_replay207_resume19_operations_v1 import prepare as operation
    return operation(c,folder,guard,H=H,record=record,fresh=fresh)

def gpu(c,folder,watch,mode,rootfolder):
    from asr18_fixed_TR_replay207_resume19_operations_v1 import execute_gpu
    return execute_gpu(c,folder,watch,mode,rootfolder,H=H,record=record,current_lifetime=current_lifetime,fresh=fresh)

def leaf(config,review,folder,mode,rootfolder,family_nonce):
    c=checked(config,review);c['_configuration']=record(config);c['_family_nonce']=family_nonce;producer=H.pid_identity();write_json(folder/'boot.json',{'producer':producer,'source':record(__file__),'family_nonce':family_nonce})
    cap=1_000_000_000 if mode=='prepare' else 3_000_000_000 if mode=='train' else 3_350_000_000
    watch=H.Watch(folder,cap,worker=True);code=1
    try:
        if mode!='prepare':
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                lease=read_json(LEASE)
                if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid():break
                watch.check();time.sleep(.05)
            else:raise ValueError('Actual exclusive GPU handshake')
        if mode=='prepare':prepare(c,folder,watch.check)
        else:gpu(c,folder,watch,mode,rootfolder)
        code=0
    finally:watch.close(code)

def parent(config,review,folder):
    c=checked(config,review);require(folder.resolve().is_relative_to(E.resolve()) and not folder.exists(),'Fresh E-only run')
    folder.mkdir();watch=H.Watch(folder,128_000_000);prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Idle exclusive GPU owner')
    code=1
    try:
        for mode,forecast in [('train',5_816_813_440),('recover',3_542_000_000)]:
            admission=fresh(forecast,watch.check);write_json(folder/(mode+'_prelaunch.json'),admission);require(admission['passed'],'Fresh phase-owned family admission, no unchanged retry')
            path=folder/mode;path.mkdir();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1' if mode=='prepare' else '0';env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
            require(not env.get('DISABLE_ADDMM_CUDA_LT') and not env.get('TORCH_CUDNN_V8_API_DISABLED'),'No failed backend switches')
            family_nonce=secrets.token_hex(16);observer=H.pid_identity()
            python=c['CPU_python']['path'] if mode=='prepare' else c['GPU_python']['path'];known={};native=None;child=None
            try:
                with (path/'worker.log').open('xb') as log:
                    child=subprocess.Popen([python,'-X','utf8','-B',str(Path(__file__).resolve()),'--config',config,'--ROOT-review',review,'--folder',str(path),
                        '--leaf',mode,'--root-folder',str(folder),'--family-nonce',family_nonce],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    launcher=H.pid_identity(child.pid);began=time.monotonic()
                    while child.poll() is None:
                        watch.check();require(time.monotonic()-began<c['maximum_stage_seconds'][mode],'Finite phase wall budget')
                        try:
                            import psutil
                            for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
                                try:v=H.pid_identity(process.pid);known[(v['pid'],v['create_time'])]=v
                                except psutil.NoSuchProcess:pass
                        except psutil.NoSuchProcess:pass
                        if native is None and (path/'boot.json').exists():
                            boot=read_json(path/'boot.json');native=boot['producer'];require(boot['source']==record(__file__) and boot['family_nonce']==family_nonce and H.alive(native) and (native['pid']==child.pid or native['ppid']==child.pid) and (native['pid'],native['create_time']) in known,'Exact captured live owned child')
                            if mode!='prepare':
                                lease=copy.deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],actual_training_optimizer_running=False);write_json(LEASE,lease)
                        total=sample_memory(os.getpid(),'single_TR_owned_family')['worker_private_memory']['private_bytes'];shim=0
                        for v in known.values():
                            if not H.alive(v):continue
                            w=sample_memory(v['pid'],'single_TR_known_child')['worker_private_memory']
                            if not w['available'] and not H.alive(v):continue
                            require(w['available'],'Available exact owned family');total+=w['private_bytes']
                            if native is None or v['pid']!=native['pid']:shim+=w['private_bytes']
                        if native is not None and H.alive(native):
                            write_json(path/'owned_family_context.json',{'kind':'ASR18_actual_owned_GPU_family_context_v1','source':record(__file__),'configuration':record(config),'nonce':family_nonce,
                                'observer':observer,'launcher':launcher,'native':native,'all_live_shims':[v for v in known.values() if H.alive(v) and v['pid']!=native['pid']],
                                'observed_monotonic':time.monotonic()})
                        require(total<=(1_192_000_000 if mode=='prepare' else 5_816_813_440 if mode=='train' else 3_542_000_000) and shim<=64_000_000,'Exact family and launcher ceilings')
                        time.sleep(.05)
                    actual=child.wait();deadline=time.monotonic()+2
                    while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.02)
                    require(actual==0 and native is not None and not any(H.alive(v) for v in known.values()),'Real handle0 and all exact identities dead')
                    g=read_json(path/'guard_closure.json');require(g['actual_exit_code']==0 and g['pid']==native['pid'] and g['monitor_clean'] and g['monitor_joined'] and g['samples']>=3,'Exact clean native guard0')
                    resultpath=path/('prepared.json' if mode=='prepare' else 'training_result.json' if mode=='train' else 'recovery_result.json')
                    result=read_json(resultpath);require(result['passed'] is True and result['producer']==native and result['source']==record(__file__),'Exact closed source/result/producer')
                    write_json(path/'actual_process0.json',{'kind':'ASR18_single_TR207_owned_phase_process0_v1','actual_Popen_wait_code':0,'all_captured_identities_dead':True,
                        'captured_identities':list(known.values()),'native':native,'source':record(__file__),'configuration':record(config),'report':record(resultpath),'guard':record(path/'guard_closure.json')})
            finally:
                import psutil
                for v in reversed(list(known.values())):
                    if H.alive(v):psutil.Process(v['pid']).terminate()
                deadline=time.monotonic()+10
                while any(H.alive(v) for v in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.05)
                require(not any(H.alive(v) for v in known.values()),'Every owned native dead before lease release')
                if child is not None:
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=5)
                write_json(LEASE,prior)
        write_json(folder/'overall_result.json',{'passed':True,'actual_updates':20,'actual_new_optimizer_updates':19,'source_update1':c['continuation']['failed_actual_tool1'],'actual_training_process0':record(folder/'train/actual_process0.json'),
            'actual_fresh_recovery_process0':record(folder/'recover/actual_process0.json'),'sampler_draws_consumed':0,'qualified':False,'production_enabled':False});code=0
    finally:watch.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--leaf',choices=('prepare','train','recover'));p.add_argument('--root-folder');p.add_argument('--family-nonce');a=p.parse_args()
    if a.leaf:leaf(a.config,a.ROOT_review,Path(a.folder),a.leaf,Path(a.root_folder),a.family_nonce)
    else:parent(a.config,a.ROOT_review,Path(a.folder))
