"""ONE inactive tiny official-TDT cold/warm component; no model or optimizer."""
import argparse
from copy import deepcopy
import ctypes
from ctypes import wintypes
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
E=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis';LEASE=P/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import require,digest
from resource_watchdog_v2 import sample_memory,PROCESS_MEMORY_COUNTERS_EX

# Reuse the already executed guarded component process/telemetry primitives.
HELPER=P/'work/research/run_ASR18_frozen_CNN_lifetime_pair_v8.py'
spec=importlib.util.spec_from_file_location('reviewed_component_telemetry',HELPER)
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
record=helpers.record;Watch=helpers.Watch;identity=helpers.pid_identity;alive=helpers.alive

def checked(config_path,review_path):
    helpers.storage();c=read_json(config_path);review=read_json(review_path)
    require(c['kind']=='ASR18_official_TDT_exact_six_cache_experiment_v1' and c['production_enabled'] is False
            and type(c['actual_training_updates']) is int and c['actual_training_updates']==0
            and c['calls']==2 and c['shape']==[1,4,4,8716] and c['labels']==[5,300,8710], 'Exact tiny full-V component')
    require(review['kind']=='ROOT_official_TDT_cold_host_component_execution_review_v1' and review['accepted'] is True
            and review['reviewer']=='/root' and review['configuration']==record(config_path)
            and review['source_bindings']==c['source_bindings'] and review['resource_policy']==c['resource_policy'],
            'Exact ROOT component source/config/resource review required')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Exact source changed')
    require(c['source_bindings']['harness']==record(__file__) and c['source_bindings']['telemetry_helper']==record(HELPER),'Exact component implementation')
    require(c['GPU_python']==str(P/'work/nemo-native-env-v2/Scripts/python.exe') and record(c['GPU_python'])==c['GPU_python_binary'],'Exact existing runtime')
    for pin in c['saved_baseline'].values():require(record(pin['path'])==pin,'Saved actual original TDT evidence unchanged')
    require(record(c['seed_configuration']['path'])==c['seed_configuration'] and record(c['seed_parent_source']['path'])==c['seed_parent_source'], 'Exact preserved successful seed implementation/configuration')
    seedc=read_json(c['seed_configuration']['path'])
    for pin in seedc['source_bindings'].values():require(record(pin['path'])==pin,'Original seed source unchanged')
    for key in ('shape','labels','GPU_python','GPU_python_binary','resource_policy','cache_directory','seed_folder','seed_actual_process_proof','saved_baseline','scientific_policy'):
        require(c[key]==seedc[key],'Exact unchanged seed scientific/runtime/resource/cache contract')
    require(seedc['source_bindings']['harness']==c['seed_parent_source'] and record(c['seed_actual_process_proof'])==c['seed_actual_process_proof_binding'],'Exact genuine seed source and actual process0 pin')
    return c

def highwater():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    v=PROCESS_MEMORY_COUNTERS_EX();v.cb=ctypes.sizeof(v)
    require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(v),v.cb),'OS current/lifetime telemetry')
    require(v.PrivateUsage<=3_750_000_000 and v.PeakPagefileUsage<=3_750_000_000,'Whole-worker current AND lifetime private<=3.75GB')
    return {'private_bytes':int(v.PrivateUsage),'pagefile_usage_private_commit_bytes':int(v.PagefileUsage),
        'lifetime_peak_private_commit_bytes':int(v.PeakPagefileUsage),'API':'GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX'}

def worker(config_path,review_path,folder,cache_mode):
    c=checked(config_path,review_path);folder=Path(folder);require(Path(sys.executable).resolve()==Path(c['GPU_python']).resolve(),'Exact worker interpreter')
    producer=identity();write_json(folder/'boot.json',{'producer':producer,'source':record(__file__)})
    watch=Watch(folder,c['resource_policy']['worker_private_forecast_bytes'],worker=True);torch=None;code=1;began=time.monotonic();stages=[]
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            lease=read_json(LEASE)
            if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid():break
            watch.check();time.sleep(.05)
        else:raise ValueError('Actual exclusive owned worker handshake missing')
        def stage(name):
            lease=read_json(LEASE);require(lease['owner']==OWNER and lease['actual_GPU_running'] is True
                and lease['actual_owned_worker_pid']==os.getpid() and lease['actual_training_optimizer_running'] is False,'Current component owner')
            watch.phase=name;watch.check();v=sample_memory(os.getpid(),name);v['private_high_water']=highwater();v['at_monotonic']=time.monotonic()
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize();v['cuda']={'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            stages.append(v)
            with (folder/'stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v)+'\n')
            return v
        stage('before_ML_import')
        require('torch' not in sys.modules,'Fresh coherent runtime before Torch')
        from preload_nemo_linker_v1 import preload
        coherent=preload()
        import torch as imported_torch
        torch=imported_torch;torch.set_num_threads(4);stage('after_coherent_preload_Torch_and_Numba_import')
        require(torch.__version__=='2.7.1+cu128' and coherent['actual_binding_linker']['version']==[12,9],'Same existing runtime')
        from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
        cap=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
        torch.empty(0,device='cuda');stage('after_CUDA_initialization')
        from native_tdt_minimal_import_v1 import prepare
        stage('before_official_minimal_namespace_import');minimal=prepare();stage('after_official_minimal_namespace_import')
        stage('before_original_native_adapter_import_and_construction')
        from native_tdt_checkpointed_v4 import _native_loss
        backend=_native_loss(8192,(0,1,2,3,4),0.)
        ready=stage('after_original_native_adapter_construction')
        require(ready['worker_private_memory']['private_bytes']<=c['resource_policy']['maximum_official_import_ready_private_bytes'],
                'Actual official import startup prerequisite missed; no unchanged retry')
        from asr18_exact_TDT_kernel_cache_v3 import ExactTDTCache
        cache=ExactTDTCache(c,mode=cache_mode,torch=torch,coherent=coherent,record=record,guard=watch.check)
        cache_before=cache.snapshot();stage('after_E_only_six_dispatcher_cache_configuration_'+cache_mode)
        import numpy as np
        import random
        random.seed(941177);np.random.seed(1701);torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
        source=torch.arange(4*4*8716,dtype=torch.float32).remainder_(257).sub_(128).div_(128).reshape(c['shape'])
        input_sha=helpers.tensor_hash(source,torch)
        labels=torch.tensor([c['labels']],device='cuda',dtype=torch.long)
        lengths=torch.tensor([4],device='cuda',dtype=torch.long);targets=torch.tensor([3],device='cuda',dtype=torch.long)
        calls=[];heap_trim=None
        for index in range(2):
            watch.check();require(time.monotonic()-began<c['maximum_worker_seconds'],'Finite TDT component')
            logits=source.to('cuda').detach().requires_grad_();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            prospective=[]
            for _ in range(3):
                value=watch.check();s=value['system_memory'];private=value['worker_private_memory']['private_bytes']
                extra=max(0,c['resource_policy']['worker_private_forecast_bytes']-private)+192_000_000
                fits=value['paired_sampling_within_one_second'] and s['available'] and value['worker_private_memory']['available']
                fits=fits and s['commit_used_bytes']+extra<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-extra>2_000_000_000
                prospective.append({'sample':value,'extra_bytes':extra,'passed':fits});time.sleep(.05)
            require(all(x['passed'] for x in prospective),'Three fresh full remaining component admissions')
            rng=helpers.capture_rng(torch,np,random);before=stage('call'+str(index+1)+'_before_forward');started=time.monotonic()
            # SAME exact FP32 split + duration log-softmax + native backend.loss
            # call/RNG restoration as native_tdt_checkpointed_v4. No new loss recipe.
            lexical=logits[...,:8711].contiguous();duration=torch.log_softmax(logits[...,8711:],dim=-1).contiguous()
            torch.cuda.synchronize();stage('call'+str(index+1)+'_after_lexical_split_and_duration_logsoftmax_before_official_backend')
            state=random.getstate()
            try:
                loss=backend.loss(lexical,duration,labels,lengths,targets,backend.blank,backend.durations,backend.reduction,
                    backend.fastemit_lambda,backend.clamp,backend.sigma,backend.omega)
            finally:random.setstate(state)
            torch.cuda.synchronize();forward_seconds=time.monotonic()-started;after_forward=stage('call'+str(index+1)+'_after_forward')
            require(loss.shape==(1,) and bool(torch.isfinite(loss).all()),'Finite original raw loss')
            started=time.monotonic();loss.sum().backward();torch.cuda.synchronize();backward_seconds=time.monotonic()-started
            after_backward=stage('call'+str(index+1)+'_after_backward')
            require(logits.grad is not None and bool(torch.isfinite(logits.grad).all()) and bool((logits.grad!=0).any()),'Actual finite nonzero full-logit backward')
            require(helpers.tensor_hash(logits,torch)==input_sha and helpers.tensor_hash(source,torch)==input_sha,'Original full input unchanged')
            require(digest(helpers.capture_rng(torch,np,random))==digest(rng),'Same frozen native wrapper all4RNG preservation')
            require(torch.cuda.max_memory_allocated()<=c['resource_policy']['CUDA_component_peak_ceiling_bytes'],'Tiny full-V tensor/kernel CUDA ceiling')
            calls.append({'call':index+1,'before_forward':before,'after_forward':after_forward,'after_backward':after_backward,
                'forward_seconds':forward_seconds,'backward_seconds':backward_seconds,'loss':loss.detach().cpu().tolist(),
                'loss_sha256':helpers.tensor_hash(loss,torch),'full_logit_gradient_sha256':helpers.tensor_hash(logits.grad,torch),
                'full_logit_gradient_shape':list(logits.grad.shape),'all4RNG_unchanged':True,'input_unchanged':True,
                'prospective_admissions':prospective})
            del loss,logits,lexical,duration,state,rng;gc.collect();torch.cuda.empty_cache();stage('call'+str(index+1)+'_after_cleanup')
            if index==0:
                def trim_buffers():
                    return {name:{'sha256':helpers.tensor_hash(tensor,torch),'data_ptr':tensor.data_ptr(),'version':tensor._version,
                        'shape':list(tensor.shape),'dtype':str(tensor.dtype)} for name,tensor in
                        {'source':source,'labels':labels,'lengths':lengths,'targets':targets}.items()}
                before_buffers=trim_buffers();trim_rng=helpers.capture_rng(torch,np,random)
                kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                kernel.GetModuleHandleW.argtypes=[wintypes.LPCWSTR];kernel.GetModuleHandleW.restype=wintypes.HMODULE
                kernel.GetProcAddress.argtypes=[wintypes.HMODULE,wintypes.LPCSTR];kernel.GetProcAddress.restype=ctypes.c_void_p
                handle=kernel.GetModuleHandleW('ucrtbase.dll');require(handle,'UCRT must ALREADY be loaded; no LoadLibrary')
                address=kernel.GetProcAddress(handle,b'_heapmin');errno_address=kernel.GetProcAddress(handle,b'_errno')
                require(address and errno_address,'Documented existing owned-process CRT symbols')
                heapmin=ctypes.CFUNCTYPE(ctypes.c_int)(address)
                errno_function=ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_int))(errno_address)
                errno_pointer=errno_function();errno_before=int(errno_pointer[0])
                before_trim=stage('call1_after_cleanup_immediately_before_UCRT_heapmin')
                trim_started=time.monotonic();return_code=int(heapmin());trim_seconds=time.monotonic()-trim_started
                errno_after=int(errno_pointer[0]);after_trim=stage('call1_immediately_after_UCRT_heapmin')
                require(return_code in (0,-1),'Documented _heapmin return range')
                require(trim_buffers()==before_buffers and digest(helpers.capture_rng(torch,np,random))==digest(trim_rng),
                    'Heap trim itself must preserve complete inputs/storage/version/labels/lengths and all4RNG')
                heap_trim={'operation':'Already-loaded ucrtbase!_heapmin, owned process unused host heap only',
                    'return_code':return_code,'errno_before':errno_before,'errno_after':errno_after,'seconds':trim_seconds,
                    'before':before_trim,'after':after_trim,'all_inputs_storage_versions_and_all4RNG_unchanged':True,
                    'source_buffers':before_buffers,'DLL_loaded_unloaded_or_working_set_trimmed':False}
                del trim_rng,before_buffers,heapmin,errno_function,errno_pointer

        require(calls[0]['loss_sha256']==calls[1]['loss_sha256'] and calls[0]['full_logit_gradient_sha256']==calls[1]['full_logit_gradient_sha256'],
                'Identical tiny first/second official loss and full-V gradients required')
        saved=read_json(c['saved_baseline']['worker_report']['path'])
        require(all(x['loss_sha256']==saved['calls'][0]['loss_sha256'] and x['full_logit_gradient_sha256']==saved['calls'][0]['full_logit_gradient_sha256']
                    for x in calls),'Exact saved ORIGINAL loss and every full8711 gradient byte, no baseline rerun')
        cache_after=cache.complete()
        report={'kind':'ASR18_actual_official_TDT_tiny_cold_warm_component_v1','execution_completed':True,'source':record(__file__),
            'configuration':record(config_path),'producer':producer,'shape':c['shape'],'labels':c['labels'],
            'full_V':8711,'blank':8192,'durations':[0,1,2,3,4],'sigma':0.,'omega':0.,'fastemit_lambda':0.,'clamp':0.,
            'input_sha256':input_sha,'calls':calls,'phase_stages':stages,'minimal_namespace_policy':minimal,'coherent_runtime':coherent,'capacity':cap,
            'first_second_loss_full_gradient_byte_exact':True,'model_encoder_Adam_teacher_audio_used':False,
            'actual_training_updates':0,'production_enabled':False,'host_attribution':'phase-tagged component evidence only; not proof of a specific DLL/JIT or full-native host forecast',
            'wall_seconds':time.monotonic()-began}
        report.update(heap_trim=heap_trim,cache_mode=cache_mode,cache_identity=cache.identity,cache_before=cache_before,cache_after=cache_after,
            saved_original_loss_gradient_byte_exact=True,saved_baseline=c['saved_baseline'])
        report['phase_semantics']={'coherent_preload':'Imports Torch AND Numba before returning; no CUDA initialization by preload',
            'official_forward':'backend.loss forward computes native cost AND native gradient internally; phase includes first JIT/kernel work',
            'backward':'PyTorch autograd scales/propagates precomputed native gradients through full FP32 split/logsoftmax; no optimizer'}
        write_json(folder/'worker_report.json',report);code=0
    finally:watch.close(code)

def parent(config_path,review_path,folder,cache_mode):
    import psutil
    require(cache_mode=='hit','This successor consumes only genuine preserved seed2; no reseed')
    c=checked(config_path,review_path);folder=Path(folder).resolve();require(folder.is_relative_to(E.resolve()) and not folder.exists(),'Fresh E component folder')
    folder.mkdir();watch=Watch(folder,128_000_000);prior=read_json(LEASE);known={};child=None;native=None;code=1
    require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Idle sole GPU lease')
    require(str(folder)==c[cache_mode+'_folder'],'Exact reviewed seed/hit evidence path')
    from asr18_exact_TDT_kernel_cache_v3 import inventory
    cache_directory=Path(c['cache_directory']);require(cache_directory.resolve().is_relative_to(E.resolve()),'E-only cache')
    if cache_mode=='seed':
        require(not cache_directory.parent.exists(),'Fresh owned source-keyed cache namespace')
        cache_directory.mkdir(parents=True)
    else:
        seed=read_json(c['seed_actual_process_proof']);sg=read_json(seed['guard']['path']);sr=read_json(seed['report']['path'])
        require(seed['source']==c['seed_parent_source'] and seed['actual_Popen_wait_code']==0 and seed['all_captured_identities_dead'] is True
                and not any(alive(x) for x in seed['captured_identities']) and sg['actual_exit_code']==0 and sg['monitor_clean'] and sg['monitor_joined']
                and sr['cache_mode']=='seed' and sr['saved_original_loss_gradient_byte_exact'] is True
                and sr['configuration']==c['seed_configuration'],'Genuine current-source closed seed0 required before freshhit')
        require(record(seed['report']['path'])==seed['report'] and record(seed['guard']['path'])==seed['guard'],'Exact source-bound seed report/guard')
        require(inventory(cache_directory,record=record,guard=watch.check)==read_json(cache_directory.parent/'seed_inventory.json'),'Immutable exact seed cache files before hit')
    def remember(pid):
        try:
            item=identity(pid);known[(item['pid'],item['create_time'])]=item
            for p in psutil.Process(pid).children(recursive=True):
                try:v=identity(p.pid);known[(v['pid'],v['create_time'])]=v
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
    try:
        admission=helpers.fresh_admit(c);write_json(folder/'fresh_admission.json',admission)
        require(admission['passed'],'Fresh whole component admission failed; no unchanged retry')
        child_folder=folder/'official_TDT';child_folder.mkdir();env=os.environ.copy();env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
        env['NUMBA_CACHE_DIR']=str(cache_directory.resolve())
        command=[c['GPU_python'],'-X','utf8','-B',str(Path(__file__).resolve()),'--worker','--config',str(Path(config_path).resolve()),
            '--ROOT-review',str(Path(review_path).resolve()),'--folder',str(child_folder),'--cache-mode',cache_mode]
        with (child_folder/'worker.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT);remember(child.pid);start=time.monotonic()
            while child.poll() is None:
                watch.check();remember(child.pid);require(time.monotonic()-start<c['maximum_worker_seconds']+30,'Finite component owner')
                if native is None and (child_folder/'boot.json').exists():
                    boot=read_json(child_folder/'boot.json');native=boot['producer']
                    require(boot['source']==record(__file__) and alive(native) and (native['pid'],native['create_time']) in known,'Exact captured owned native')
                    lease=deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],actual_training_optimizer_running=False,
                        current_TDT_component_configuration=record(config_path));write_json(LEASE,lease)
                total=sample_memory(os.getpid(),'TDT_parent_family')['worker_private_memory']['private_bytes'];shim=0
                for item in known.values():
                    if not alive(item):continue
                    v=sample_memory(item['pid'],'TDT_known_owned_family')['worker_private_memory']
                    if not v['available'] and not alive(item):continue
                    require(v['available'],'Available actual known native/shim')
                    total+=v['private_bytes']
                    if native is None or item['pid']!=native['pid']:shim+=v['private_bytes']
                require(total<=c['resource_policy']['whole_owned_family_forecast_bytes'] and shim<=64_000_000,'Whole owned family and launcher bound')
                time.sleep(.05)
            exit_code=child.wait();deadline=time.monotonic()+2
            while any(alive(x) for x in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.02)
            require(exit_code==0 and native is not None and not any(alive(x) for x in known.values()),'Actual handle0 and all captured identities dead')
            g=read_json(child_folder/'guard_closure.json');r=read_json(child_folder/'worker_report.json')
            require(g['actual_exit_code']==0 and g['pid']==native['pid'] and g['monitor_clean'] and g['monitor_joined'] and g['error'] is None
                    and g['samples']>=3 and r['execution_completed'] and r['producer']==native and r['configuration']==record(config_path),'Exact actual native0/report/clean monitor')
            proof={'kind':'ASR18_actual_owned_TDT_cold_component_process0_v1','owner':OWNER,'native':native,'actual_Popen_wait_code':exit_code,
                'captured_identities':list(known.values()),'all_captured_identities_dead':True,'source':record(__file__),
                'report':record(child_folder/'worker_report.json'),'guard':record(child_folder/'guard_closure.json')}
            contents=inventory(cache_directory,record=record,guard=watch.check)
            if cache_mode=='seed':write_json(cache_directory.parent/'seed_inventory.json',contents)
            else:require(contents==read_json(cache_directory.parent/'seed_inventory.json'),'Exact cache files unchanged after freshhit')
            proof.update(cache_mode=cache_mode,cache_inventory=contents,cache_identity=record(cache_directory.parent/'cache_identity.json'))
            write_json(folder/'actual_process0.json',proof);code=0
    finally:
        for item in reversed(list(known.values())):
            if alive(item):psutil.Process(item['pid']).terminate()
        deadline=time.monotonic()+10
        while any(alive(x) for x in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.05)
        require(not any(alive(x) for x in known.values()),'Actual known family death before lease release')
        if child is not None:
            if child.poll() is None:child.terminate()
            child.wait(timeout=5)
        lease=deepcopy(prior);lease.update(actual_GPU_running=False,actual_owned_worker_pid=None,actual_training_optimizer_running=False);write_json(LEASE,lease)
        watch.close(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--ROOT-review',required=True);p.add_argument('--folder',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--cache-mode',choices=('seed','hit'),required=True);a=p.parse_args()
    if a.worker:worker(a.config,a.ROOT_review,a.folder,a.cache_mode)
    else:parent(a.config,a.ROOT_review,a.folder,a.cache_mode)
