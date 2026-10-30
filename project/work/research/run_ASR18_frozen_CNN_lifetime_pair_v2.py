"""Inactive original vs instance-only inplace-ReLU frozen CNN component pair.

No launch without exact ROOT config/resource review. No audio, full encoder,
LoRA, decoder, teacher, optimizer or training/checkpoint state is instantiated.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(SOURCE))
ROOT=Path(r'E:\gpu_training_scratchpad');OWNER='/root/memory_diagnosis'
LEASE=PROJECT/'work/gpu_training_owner.json'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_production_state_v1 import digest,require
from resource_watchdog_v2 import ResourceGuard,sample_memory


def record(path):
    path=Path(path).resolve();require(path.is_relative_to(ROOT.resolve()) or path.is_relative_to(PROJECT.resolve()),'Authorized project/E paths only')
    before=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024**2):h.update(block)
    require((before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns),'File changed while hashing')
    return {'path':str(path),'bytes':before.st_size,'sha256':h.hexdigest()}


def signature(path):
    p=Path(path).resolve();s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def storage():
    require(ROOT.exists() and read_json(ROOT/'storage_config.json')['storage_root']==str(ROOT),'Required E storage')
    for key in ('TEMP','TMP','HF_HOME','TORCH_EXTENSIONS_DIR'):
        require(Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve()),'Apply process-local E storage first')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        require(os.environ.get(key)=='1','Frozen thread caps1')
    require(shutil.disk_usage(ROOT).free>100_000_000_000,'E100GB reserve')


def checked(config_path,review_path):
    storage();c=read_json(config_path);review=read_json(review_path)
    require(c['kind']=='ASR18_frozen_CNN_original_inplace_component_pair_v1' and c['production_enabled'] is False
            and c['arms']==['original','inplace_relu'] and type(c['actual_training_updates']) is int and c['actual_training_updates']==0,
            'One exact bounded zero-training component pair')
    require(review['kind']=='ROOT_frozen_CNN_component_pair_execution_review_v1' and review['accepted'] is True
            and review['reviewer']=='/root' and review['configuration']==record(config_path)
            and review['resource_policy']==c['resource_policy'] and review['source_bindings']==c['source_bindings'],
            'ROOT exact source/config/component numeric resource review required')
    for pin in c['source_bindings'].values():require(record(pin['path'])==pin,'Reviewed source changed')
    require(c['source_bindings']['harness']==record(__file__),'Exact worker/parent source')
    require(c['GPU_python']==str(PROJECT/'work/nemo-native-env-v2/Scripts/python.exe')
            and record(c['GPU_python'])==c['GPU_python_binary'],'Exact existing isolated GPU interpreter')
    require(signature(c['base_archive']['path'])==c['base_archive_signature'],'Immutable original base signature changed')
    require(record(c['bank_report']['path'])==c['bank_report'],'Exact actual frozen bank')
    return c


def pid_identity(pid=None):
    import psutil
    p=psutil.Process(os.getpid() if pid is None else pid)
    return {'pid':p.pid,'ppid':p.ppid(),'create_time':p.create_time()}


def alive(item):
    import psutil
    try:p=psutil.Process(item['pid']);return p.create_time()==item['create_time'] and p.is_running()
    except psutil.NoSuchProcess:return False


def private_high_water():
    import ctypes
    from ctypes import wintypes
    from resource_watchdog_v2 import PROCESS_MEMORY_COUNTERS_EX
    api=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    api.GetProcessMemoryInfo.restype=wintypes.BOOL
    value=PROCESS_MEMORY_COUNTERS_EX();value.cb=ctypes.sizeof(value)
    require(api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb),'Actual owned lifetime counter')
    return {'private_bytes':int(value.PrivateUsage),'pagefile_usage_private_commit_bytes':int(value.PagefileUsage),'lifetime_peak_pagefile_usage_bytes':int(value.PeakPagefileUsage),
        'API':'GetProcessMemoryInfo PROCESS_MEMORY_COUNTERS_EX PrivateUsage/PeakPagefileUsage; process lifetime, not resettable phase peak'}


class Watch:
    def __init__(self,folder,cap,*,worker=False):
        self.folder=Path(folder);self.cap=cap;self.worker=worker;self.done=threading.Event();self.error=None
        self.phase='before_ML_import';self.torch=None;self.count=0;self.last=None;self.peak=0;self.system_peak=0.;self.phases={}
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();deadline=time.monotonic()+3
        while self.count<3:
            require(self.thread.is_alive() and self.error is None and time.monotonic()<deadline,'Live fresh guard before work');time.sleep(.02)
    def loop(self):
        resource=ResourceGuard()
        try:
            while not self.done.is_set():
                value=sample_memory(os.getpid(),'CNN_component_'+self.phase)
                require(value['paired_sampling_within_one_second'] and value['system_memory']['available']
                        and value['worker_private_memory']['available'],'Fresh paired component telemetry')
                private=value['worker_private_memory']['private_bytes'];s=value['system_memory']
                reason,_=resource.observe(s,value['worker_private_memory'])
                require(reason is None and private<self.cap and not (self.folder/'stop.json').exists(),'Resource/owned-stop '+str(reason))
                require(shutil.disk_usage(ROOT).free>100_000_000_000,'E100GB reserve')
                self.count+=1;self.last=time.monotonic();self.peak=max(self.peak,private);self.system_peak=max(self.system_peak,s['commit_fraction'])
                p=self.phases.setdefault(self.phase,{'private_peak_bytes':0,'samples':0,'system_commit_peak_fraction':0.})
                p['private_peak_bytes']=max(p['private_peak_bytes'],private);p['samples']+=1
                p['system_commit_peak_fraction']=max(p['system_commit_peak_fraction'],s['commit_fraction'])
                write_json(self.folder/'live.json',value);self.done.wait(.05)
        except BaseException:
            self.error=traceback.format_exc()
            write_json(self.folder/'resource_stop.json',{'error':self.error,'phase':self.phase})
            if self.worker:os._exit(75)
    def check(self):
        require(self.error is None and self.thread.is_alive() and self.last is not None and time.monotonic()-self.last<2,
                'Clean live guard, no stale/guessed sample')
        return sample_memory(os.getpid(),'CNN_component_owner_check')
    def close(self,code):
        self.done.set();self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count>=3 and time.monotonic()-self.last<2,'Clean joined guard')
        proof={'pid':os.getpid(),'actual_exit_code':code,'monitor_clean':True,'monitor_joined':True,'samples':self.count,
            'peak_private_bytes':self.peak,'system_commit_peak_fraction':self.system_peak,'phases':self.phases,'error':None}
        write_json(self.folder/'guard_closure.json',proof);return proof


def capture_rng(torch,np,random):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),*n[2:]),
        'torch':torch.get_rng_state().tolist(),'cuda':[v.tolist() for v in torch.cuda.get_rng_state_all()]}


def tensor_hash(tensor,torch):
    raw=tensor.detach().reshape(-1).view(torch.uint8);h=hashlib.sha256()
    for low in range(0,raw.numel(),4*1024**2):h.update(raw[low:low+4*1024**2].cpu().numpy().tobytes())
    return h.hexdigest()


def load_CNN(c,torch,guard):
    from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
    from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderSubsamplingConv2D
    from stream_frozen_encoder_matrices_v1 import readonly_shard
    import inspect
    require(record(inspect.getsourcefile(ParakeetEncoderSubsamplingConv2D))==c['source_bindings']['original_CNN'],'Exact original CNN class')
    configuration=read_json(c['base_config']['path'])
    with torch.device('meta'):module=ParakeetEncoderSubsamplingConv2D(ParakeetEncoderConfig(**configuration['encoder_config']))
    require(len(module.layers)==8 and all(p.device.type=='meta' for p in module.parameters()),'Only original CNN constructed on meta')
    state={};maps=[];backing={};handles=[];metrics={'current_matrix_bytes':0,'peak_matrix_bytes':0,'transfers':0,'transfer_bytes':0}
    with Path(c['base_archive']['path']).open('rb') as stream:
        hlen=struct.unpack('<Q',stream.read(8))[0];header_raw=stream.read(hlen);header=json.loads(header_raw);offset=8+hlen
        require(hashlib.sha256(header_raw).hexdigest()==c['base_header_sha256'],'Exact original safetensors header')
        for name,reference in module.state_dict().items():
            guard();full='encoder.subsampling.'+name
            if name in c['CNN_matrices']:
                entry=c['CNN_matrices'][name];require(record(entry['tensor']['path'])==entry['tensor'],'Exact existing frozen CNN shard')
                value,mapping=readonly_shard(Path(entry['tensor']['path']),entry['shape']);maps.append(mapping);state[name]=value
                require(value.dtype==torch.bfloat16 and list(value.shape)==entry['shape'],'Original BF16 shard')
            else:
                entry=c['CNN_biases'][name];info=header[full]
                require(info==entry['descriptor'] and info['dtype']=='F32','Exact original bias descriptor')
                low,high=info['data_offsets'];stream.seek(offset+low);raw=stream.read(high-low)
                require(len(raw)==high-low and hashlib.sha256(raw).hexdigest()==entry['raw_sha256'],'Original bias raw bytes differ')
                state[name]=torch.frombuffer(bytearray(raw),dtype=torch.float32).reshape(info['shape']).to('cuda')
            require(state[name].shape==reference.shape,'CNN state geometry differs')
        require(set(state)==set(module.state_dict()) and len(c['CNN_matrices'])==6 and len(c['CNN_biases'])==6,'All six matrices/biases only')
    module.load_state_dict(state,strict=True,assign=True);module.eval().requires_grad_(False)
    for name in c['CNN_matrices']:
        path,key=name.rsplit('.',1);layer=module.get_submodule(path);cpu_parameter=layer.weight
        backing[name]={'parameter':cpu_parameter,'ptr':cpu_parameter.data_ptr(),'hash':tensor_hash(cpu_parameter,torch)}
        cpu=cpu_parameter.detach();size=cpu.numel()*cpu.element_size()
        def before(current,inputs,cpu=cpu,size=size):
            require(current.weight.device.type=='cpu' and not current.weight.requires_grad,'Original frozen CPU entry')
            current._parameters['weight']=torch.nn.Parameter(cpu.to(device='cuda',non_blocking=False),requires_grad=False)
            metrics['current_matrix_bytes']+=size;metrics['peak_matrix_bytes']=max(metrics['peak_matrix_bytes'],metrics['current_matrix_bytes'])
            metrics['transfers']+=1;metrics['transfer_bytes']+=size
        def after(current,inputs,output,cpu_parameter=cpu_parameter,size=size):
            current._parameters['weight']=cpu_parameter;metrics['current_matrix_bytes']-=size
        handles.append(layer.register_forward_pre_hook(before));handles.append(layer.register_forward_hook(after,always_call=True))
    del state,value;gc.collect();torch.cuda.empty_cache()
    require(signature(c['base_archive']['path'])==c['base_archive_signature'],'Immutable original base source changed')
    return module,maps,backing,handles,metrics


def worker(config_path,review_path,folder,arm):
    c=checked(config_path,review_path);folder=Path(folder).resolve();require(folder.is_relative_to(ROOT.resolve()) and folder.is_dir(),'Fresh owned E child')
    require(Path(sys.executable).resolve()==Path(c['GPU_python']).resolve(),'Exact worker interpreter identity')
    write_json(folder/'boot.json',{'producer':pid_identity(),'source':record(__file__),'arm':arm})
    watch=Watch(folder,c['resource_policy']['worker_private_forecast_bytes'],worker=True);code=1;torch=None;began=time.monotonic()
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            lease=read_json(LEASE)
            if lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid():break
            watch.check();time.sleep(.05)
        else:raise ValueError('Actual sole GPU lease handshake missing')
        stages=[]
        def stage(name):
            lease=read_json(LEASE)
            require(lease['owner']==OWNER and lease['actual_GPU_running'] is True and lease['actual_owned_worker_pid']==os.getpid()
                    and lease['actual_training_optimizer_running'] is False,'Current exclusive component owner')
            watch.phase=name;watch.check();value=sample_memory(os.getpid(),'CNN_'+name);value['private_high_water']=private_high_water()
            value['at_monotonic']=time.monotonic();value['cuda']={'initialized':False}
            if torch is not None and torch.cuda.is_initialized():
                value['cuda']={'initialized':True,'allocated_bytes':torch.cuda.memory_allocated(),
                    'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            stages.append(value)
            with (folder/'stages.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
            return value
        stage('before_ML_import')
        require('torch' not in sys.modules,'Fresh coherent preload before Torch')
        from preload_nemo_linker_v1 import preload
        coherent=preload()
        import torch as imported_torch
        torch=imported_torch;torch.set_num_threads(4);stage('after_Torch_import')
        require(torch.__version__=='2.7.1+cu128' and coherent['actual_binding_linker']['version']==[12,9],'Exact existing native runtime')
        from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
        cap=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
        torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
        torch.empty(0,device='cuda');stage('after_CUDA_initialization')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        import numpy as np
        import random
        module,maps,backing,handles,metrics=load_CNN(c,torch,watch.check)
        for index in (1,4,7):
            require(isinstance(module.layers[index],torch.nn.ReLU) and module.layers[index].inplace is False,'Exact original ReLU layout')
            if arm=='inplace_relu':module.layers[index].inplace=True
        require(arm in c['arms'],'Exact pair arm')
        ready=stage('after_frozen_CNN_mapping_construction')
        require(ready['worker_private_memory']['private_bytes']<=c['resource_policy']['maximum_CNN_ready_private_bytes'],
                'Component startup prerequisite missed; no forward/unchanged retry')
        def fingerprint():
            require(metrics['current_matrix_bytes']==0,'No in-flight streamed matrix')
            result={}
            for name,p in module.named_parameters():
                result[name]={'dtype':str(p.dtype),'shape':list(p.shape),'ptr':p.data_ptr(),'version':p._version,'hash':tensor_hash(p,torch)}
                if name in backing:require(p.device.type=='cpu' and p.data_ptr()==backing[name]['ptr'] and result[name]['hash']==backing[name]['hash'],'Readonly backing changed')
                require(not p.requires_grad and p.grad is None,'No gradient/scientific state')
            return result
        random.seed(1701);np.random.seed(1701);torch.manual_seed(1701);torch.cuda.manual_seed_all(1701)
        initial=fingerprint();observations=[]
        for case in c['cases']:
            watch.check();require(time.monotonic()-began<c['maximum_worker_seconds'],'Finite component time budget')
            f,v=case['physical_frames'],case['valid_frames'];array=np.empty((1,f,128),dtype=np.float32)
            flat=array.reshape(-1)
            for low in range(0,flat.size,16384):
                high=min(low+16384,flat.size);counter=np.arange(low,high,dtype=np.int64)
                flat[low:high]=((counter%257)-128).astype(np.float32)/128
            array[:,v:]=0;mask=np.arange(f)[None,:]<v
            input_sha=hashlib.sha256(array.tobytes()).hexdigest();mask_sha=hashlib.sha256(mask.tobytes()).hexdigest()
            x=torch.from_numpy(array).to('cuda');m=torch.from_numpy(mask).to('cuda')
            rng=capture_rng(torch,np,random);before_parameters=fingerprint();before_metrics=deepcopy(metrics)
            torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            prospective=[]
            for _ in range(3):
                current=watch.check();s=current['system_memory'];private=current['worker_private_memory']['private_bytes']
                extra=max(0,c['resource_policy']['worker_private_forecast_bytes']-private)+c['resource_policy']['parent_private_cap_bytes']+c['resource_policy']['launcher_private_reserve_bytes']
                fits=current['paired_sampling_within_one_second'] and s['available'] and current['worker_private_memory']['available']
                fits=fits and s['commit_used_bytes']+extra<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-extra>2_000_000_000
                prospective.append({'sample':current,'additional_projected_bytes':extra,'passed':fits});time.sleep(.05)
            require(all(item['passed'] for item in prospective),'Fresh full remaining component projection before forward')
            layer_observations=[];instrument_seconds=[0.];layer_handles=[]
            def observe_layer(name,when,inputs,output=None):
                begin=time.monotonic();torch.cuda.synchronize();value=sample_memory(os.getpid(),'CNN_layer_'+name+'_'+when)
                value.update(layer=name,when=when,private_high_water=private_high_water(),
                    cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),
                    inputs=[{'shape':list(item.shape),'dtype':str(item.dtype),'ptr':item.data_ptr()} for item in inputs if isinstance(item,torch.Tensor)])
                if isinstance(output,torch.Tensor):value['output']={'shape':list(output.shape),'dtype':str(output.dtype),'ptr':output.data_ptr()}
                layer_observations.append(value);instrument_seconds[0]+=time.monotonic()-begin
            for layer_name,layer in list(module.layers.named_children())+[('linear',module.linear)]:
                layer_handles.append(layer.register_forward_pre_hook(lambda current,inputs,name=layer_name:observe_layer(name,'before',inputs)))
                layer_handles.append(layer.register_forward_hook(lambda current,inputs,output,name=layer_name:observe_layer(name,'after',inputs,output)))
            before=stage('before_forward_'+case['name']);start=time.monotonic()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):output=module(x,m)
            torch.cuda.synchronize();seconds=time.monotonic()-start;after=stage('after_forward_'+case['name'])
            for handle in layer_handles:handle.remove()
            layer_handles.clear()
            continuous=deepcopy(watch.phases.get('before_forward_'+case['name'],{}))
            sampled_peak=max([before['worker_private_memory']['private_bytes'],after['worker_private_memory']['private_bytes']]+[item['worker_private_memory']['private_bytes'] for item in layer_observations]+[continuous.get('private_peak_bytes',0)])
            pre_peak=before['private_high_water']['lifetime_peak_pagefile_usage_bytes'];post_peak=after['private_high_water']['lifetime_peak_pagefile_usage_bytes']
            require(output.shape==(1,(f+7)//8,1024) and output.dtype==torch.bfloat16 and torch.isfinite(output).all(),
                    'Full physical finite BF16 original output required')
            require(torch.cuda.max_memory_allocated()<=c['resource_policy']['CUDA_tensor_and_workspace_peak_ceiling_bytes'],
                    'CNN tensor/backend-workspace component ceiling exceeded')
            require(digest(capture_rng(torch,np,random))==digest(rng) and fingerprint()==before_parameters==initial,
                    'All4 RNG or exact CNN/backing pointers/weights changed')
            require(tensor_hash(x,torch)==input_sha and tensor_hash(m,torch)==mask_sha,'Input/mask modified')
            raw=output.detach().cpu().view(torch.uint8).reshape(-1).numpy()
            out=folder/(case['name']+'.BF16.bin')
            with out.open('xb') as stream:
                for low in range(0,raw.size,65536):stream.write(raw[low:low+65536].tobytes())
                stream.flush();os.fsync(stream.fileno())
            observations.append({'case':case,'input_sha256':input_sha,'mask_sha256':mask_sha,'output':record(out),
                'output_shape':list(output.shape),'output_dtype':'bfloat16','valid_encoder_frames':(v+7)//8,
                'before':before,'after':after,'forward_wall_seconds':seconds,
                'instrumentation_wall_seconds':instrument_seconds[0],'layer_observations':layer_observations,
                'continuous_forward_phase':continuous,'observed_forward_private_peak_bytes':sampled_peak,
                'observed_forward_private_peak_minus_own_baseline_bytes':sampled_peak-before['worker_private_memory']['private_bytes'],
                'lifetime_peak_record_growth_bytes':post_peak-pre_peak,'lifetime_peak_forward_censored_by_prior_peak':post_peak<=pre_peak,
                'new_lifetime_peak_minus_own_baseline_bytes':post_peak-before['private_high_water']['private_bytes'] if post_peak>pre_peak else None,
                'prospective_full_remaining_admission':prospective,
                'all4_RNG_unchanged':True,'all_backing_weights_biases_pointers_unchanged':True,
                'streamed_matrix_transfers':metrics['transfers']-before_metrics['transfers'],
                'streamed_matrix_transfer_bytes':metrics['transfer_bytes']-before_metrics['transfer_bytes']})
            del output,raw,x,m,array,mask,flat,counter,rng,before_parameters;gc.collect();torch.cuda.empty_cache()
            stage('after_case_cleanup_'+case['name'])
        require(fingerprint()==initial and metrics['current_matrix_bytes']==0,'Final frozen state')
        for handle in handles:handle.remove()
        # Drop all readonly Tensor/Parameter views before closing six ACCESS_READ mappings.
        del layer,handle,layer_handles
        handles.clear();backing.clear();module=None;gc.collect();torch.cuda.empty_cache()
        for mapping in maps:mapping.close()
        stage('after_CNN_cleanup')
        report={'kind':'ASR18_actual_frozen_CNN_component_arm_v1','execution_completed':True,'arm':arm,
            'producer':pid_identity(),'source':record(__file__),'configuration':record(config_path),'observations':observations,
            'coherent_runtime':coherent,'capacity':cap,'matrix_metrics':metrics,'phase_stages':stages,
            'wall_seconds':time.monotonic()-began,'full_encoder_LoRA_Adam_teacher_audio_used':False,
            'actual_training_updates':0,'production_enabled':False}
        write_json(folder/'worker_report.json',report);code=0
    finally:watch.close(code)


def fresh_admit(c):
    samples=[];extra=c['resource_policy']['whole_owned_family_forecast_bytes']
    for index in range(3):
        value=sample_memory(os.getpid(),'CNN_pair_prelaunch');s=value['system_memory']
        passed=value['paired_sampling_within_one_second'] and s['available'] and value['worker_private_memory']['available']
        passed=passed and s['commit_used_bytes']+extra<.90*s['commit_limit_bytes'] and s['commit_available_bytes']-extra>2_000_000_000
        samples.append({'passed':passed,'sample':value});time.sleep(.1)
    return {'passed':all(s['passed'] for s in samples),'whole_owned_family_forecast_bytes':extra,'samples':samples}


def parent(config_path,review_path,folder):
    import psutil
    c=checked(config_path,review_path);folder=Path(folder).resolve()
    require(folder.is_relative_to(ROOT.resolve()) and not folder.exists(),'Fresh E-only component pair')
    folder.mkdir();watch=Watch(folder,c['resource_policy']['parent_private_cap_bytes']);reports=[];closed=[];code=1
    prior=read_json(LEASE);require(prior['owner']==OWNER and prior['actual_GPU_running'] is False,'Idle exclusive owner lease')
    try:
        for arm in c['arms']:
            watch.check();admission=fresh_admit(c);write_json(folder/(arm+'_fresh_admission.json'),admission)
            require(admission['passed'],'Component whole-family admission failed; no unchanged retry')
            child_folder=folder/arm;child_folder.mkdir();known={};native=None;child=None
            env=os.environ.copy();env['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT']='16'
            command=[c['GPU_python'],'-X','utf8','-B',str(Path(__file__).resolve()),'--worker',arm,
                '--config',str(Path(config_path).resolve()),'--ROOT-review',str(Path(review_path).resolve()),'--folder',str(child_folder)]
            def remember(pid):
                try:
                    item=pid_identity(pid);known[(item['pid'],item['create_time'])]=item
                    for p in psutil.Process(pid).children(recursive=True):
                        try:entry=pid_identity(p.pid);known[(entry['pid'],entry['create_time'])]=entry
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
            with (child_folder/'worker.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic();remember(child.pid)
                try:
                    while child.poll() is None:
                        watch.check();remember(child.pid)
                        require(time.monotonic()-start<c['maximum_worker_seconds']+30,'Finite owned component time')
                        if native is None and (child_folder/'boot.json').exists():
                            boot=read_json(child_folder/'boot.json');native=boot['producer']
                            require(boot['source']==record(__file__) and boot['arm']==arm and alive(native)
                                    and (native['pid'],native['create_time']) in known,'Exact captured owned worker identity')
                            lease=deepcopy(prior);lease.update(actual_GPU_running=True,actual_owned_worker_pid=native['pid'],
                                actual_training_optimizer_running=False,component_pair_configuration=record(config_path),current_component_arm=arm)
                            write_json(LEASE,lease)
                        family=sample_memory(os.getpid(),'CNN_parent_family');total=family['worker_private_memory']['private_bytes'];extra=0
                        for item in known.values():
                            if not alive(item):continue
                            value=sample_memory(item['pid'],'CNN_owned_child_family')
                            if not value['worker_private_memory']['available'] and not alive(item):continue
                            require(value['worker_private_memory']['available'],'Available live owned child')
                            size=value['worker_private_memory']['private_bytes'];total+=size
                            if native is None or item['pid']!=native['pid']:extra+=size
                        require(extra<=c['resource_policy']['launcher_private_reserve_bytes']
                                and total<=c['resource_policy']['whole_owned_family_forecast_bytes'],'Actual whole family/shim bound')
                        time.sleep(.05)
                    exit_code=child.wait();deadline=time.monotonic()+2
                    while any(alive(item) for item in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.02)
                    require(not any(alive(item) for item in known.values()),'Captured native/shim identities must actually die before lease release')
                    require(exit_code==0 and native is not None,'Actual handle0 and source-bound native report required')
                    guard=read_json(child_folder/'guard_closure.json');report=read_json(child_folder/'worker_report.json')
                    require(guard['actual_exit_code']==0 and guard['pid']==native['pid'] and guard['monitor_clean']
                            and guard['monitor_joined'] and guard['samples']>=3 and guard['error'] is None
                            and report['producer']==native and report['execution_completed'] and report['arm']==arm
                            and report['configuration']==record(config_path),'Actual native0/clean closure/report identity')
                    receipt={'kind':'ASR18_actual_owned_frozen_CNN_component_process0_v1','owner':OWNER,'arm':arm,
                        'actual_Popen_wait_code':exit_code,'native':native,'captured_identities':list(known.values()),
                        'all_captured_identities_dead':True,'report':record(child_folder/'worker_report.json'),
                        'guard':record(child_folder/'guard_closure.json'),'source':record(__file__)}
                    write_json(child_folder/'actual_process0.json',receipt);closed.append(record(child_folder/'actual_process0.json'));reports.append(report)
                finally:
                    # Retire all captured identities regardless of launcher.poll;
                    # a closed launcher never proves its native child is dead.
                    write_json(child_folder/'stop.json',{'reason':'owner_leaving_component_arm'})
                    for item in reversed(list(known.values())):
                        if alive(item):psutil.Process(item['pid']).terminate()
                    deadline=time.monotonic()+10
                    while any(alive(item) for item in known.values()) and time.monotonic()<deadline:watch.check();time.sleep(.05)
                    require(not any(alive(item) for item in known.values()),'Actual known-family death before lease release')
                    if child.poll() is None:child.terminate()
                    child.wait(timeout=5)
                    lease=deepcopy(prior);lease.update(actual_GPU_running=False,actual_owned_worker_pid=None,actual_training_optimizer_running=False)
                    write_json(LEASE,lease)
        pairs=[]
        for a,b in zip(reports[0]['observations'],reports[1]['observations']):
            require(a['case']==b['case'] and a['input_sha256']==b['input_sha256'] and a['mask_sha256']==b['mask_sha256']
                    and a['output_shape']==b['output_shape'] and a['output_dtype']==b['output_dtype'],'Same complete input/physical/mask case')
            exact=a['output']['sha256']==b['output']['sha256'];max_diff=0.;different=0
            if not exact:
                with Path(a['output']['path']).open('rb') as left,Path(b['output']['path']).open('rb') as right:
                    while block:=left.read(65536):
                        other=right.read(len(block));require(len(other)==len(block),'Same complete output bytes')
                        for x,y in zip(struct.iter_unpack('<H',block),struct.iter_unpack('<H',other)):
                            if x!=y:
                                different+=1;fx=struct.unpack('<f',struct.pack('<I',x[0]<<16))[0];fy=struct.unpack('<f',struct.pack('<I',y[0]<<16))[0]
                                max_diff=max(max_diff,abs(fx-fy))
                    require(not right.read(1),'Same complete EOF')
            pairs.append({'case':a['case'],'byte_exact':exact,'different_elements':different,'max_abs_difference':max_diff,
                'original':a,'inplace_relu':b,
                'original_private_baseline_bytes':a['before']['worker_private_memory']['private_bytes'],
                'inplace_private_baseline_bytes':b['before']['worker_private_memory']['private_bytes'],
                'original_forward_private_sample_bytes':a['after']['worker_private_memory']['private_bytes'],
                'inplace_forward_private_sample_bytes':b['after']['worker_private_memory']['private_bytes']})
        require(len(pairs)==len(c['cases']),'Both complete case lists, never only matched prefix')
        passed=all(pair['byte_exact'] for pair in pairs)
        result={'kind':'ASR18_actual_paired_frozen_CNN_lifetime_component_comparison_v1','execution_completed':True,
            'numeric_gate_passed':passed,'pairs':pairs,'actual_process0':closed,'configuration':record(config_path),
            'host_private_reduction_credited_to_native_forecast':False,'training_forecast_changed':False,
            'actual_training_updates':0,'production_enabled':False}
        write_json(folder/'paired_report.json',result);code=0 if passed else 1
    finally:watch.close(code)
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--ROOT-review',required=True)
    parser.add_argument('--folder',required=True);parser.add_argument('--worker',choices=('original','inplace_relu'))
    args=parser.parse_args()
    if args.worker:worker(args.config,args.ROOT_review,args.folder,args.worker)
    else:parent(args.config,args.ROOT_review,args.folder)
