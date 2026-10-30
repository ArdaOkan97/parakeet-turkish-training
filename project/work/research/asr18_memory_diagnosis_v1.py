"""Bounded, exclusive ASR-18 diagnostics. No automatic scientific continuation."""
from __future__ import annotations
import argparse
import ctypes
import copy
import gc
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'outputs/parakeet_turkish'
STORAGE = Path(r'E:\gpu_training_scratchpad')
AREA = STORAGE / 'analysis/ASR18_memory_diagnosis_v1'
LEASE = PROJECT / 'work/gpu_training_owner.json'
OWNER = '/root/memory_diagnosis'
sys.path.insert(0, str(SOURCE))
from atomic_io_retry_v1 import read_json, write_json
from resource_watchdog_v2 import ResourceGuard, sample_memory, PROCESS_MEMORY_COUNTERS_EX


def now(): return datetime.now(timezone.utc).isoformat()
def require(ok, message):
    if not ok: raise ValueError(message)


def storage():
    config = read_json(STORAGE / 'storage_config.json')
    require(config['storage_root'] == str(STORAGE) and not config['allow_fallback_outside_root'], 'E-only storage required')
    for key in ('TEMP', 'TMP', 'HF_HOME', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH'):
        require(Path(os.environ[key]).resolve().is_relative_to(STORAGE.resolve()), 'E environment required: ' + key)


def record(path):
    path = Path(path)
    with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def concurrent_data_observation():
    """Observe the separately owned CPU data workers; never gate their IO."""
    path=STORAGE/'analysis/BPE_weak_then_clean_v1/current_data_dispatch_v2.json'
    try:
        dispatch=read_json(path)
        items=list(dispatch.get('heavy_successors',[]))
        items.extend(dispatch.get(key,{}) for key in ('metadata','BPE_labels'))
        pids=sorted({item['PID'] for item in items if item.get('PID')})
        workers=[{'pid':pid,'private':sample_memory(pid,'concurrent_data_observation')['worker_private_memory']} for pid in pids]
        return {'at_utc':now(),'dispatch_updated_at_utc':dispatch.get('updated_at_utc'),
                'workers':workers,'data_IO_independently_owned':True,'pause_required':False}
    except (OSError,ValueError,KeyError) as exc:
        return {'at_utc':now(),'telemetry_error':str(exc),'pause_required':False}


def terminate_runtime_worker(pid, launcher):
    """Only the handshake-verified owned child; native use follows stop grace."""
    if pid and pid != launcher.pid:
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(1, False, pid)
        if handle:
            try: kernel.TerminateProcess(handle, 75)
            finally: kernel.CloseHandle(handle)
    if launcher.poll() is None: launcher.terminate()
    launcher.wait(timeout=15)


def owned_backend_modules():
    """Read loaded module basenames/image extents of this process; no file IO."""
    api=ctypes.WinDLL('psapi',use_last_error=True)
    api.EnumProcessModules.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong)]
    api.GetModuleBaseNameW.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_ulong]
    modules=(ctypes.c_void_p*2048)();needed=ctypes.c_ulong();handle=ctypes.c_void_p(-1)
    require(api.EnumProcessModules(handle,modules,ctypes.sizeof(modules),ctypes.byref(needed)),'Owned module telemetry failed')
    result=[]
    for item in modules[:needed.value//ctypes.sizeof(ctypes.c_void_p)]:
        name=ctypes.create_unicode_buffer(256)
        if api.GetModuleBaseNameW(handle,item,name,256) and any(x in name.value.lower() for x in ('cuda','cudnn','cublas','nvrtc','nvidia','nvjit')):
            result.append(name.value)
    return sorted(result)


def stage(folder, phase, torch=None, **extra):
    value = sample_memory(os.getpid(), phase)
    api=ctypes.WinDLL('psapi',use_last_error=True)
    api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),ctypes.c_ulong]
    counters=PROCESS_MEMORY_COUNTERS_EX();counters.cb=ctypes.sizeof(counters)
    require(api.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(counters),counters.cb),'Owned peak-private telemetry required')
    value['worker_private_peak_bytes_lifetime']=int(counters.PeakPagefileUsage)
    value.update(at_utc=now(), pid=os.getpid(), **extra)
    if torch is not None:
        value['cuda'] = {key: int(getattr(torch.cuda, fn)()) for key, fn in
                        [('allocated_bytes','memory_allocated'), ('reserved_bytes','memory_reserved'),
                         ('peak_allocated_bytes','max_memory_allocated'), ('peak_reserved_bytes','max_memory_reserved')]}
    with (folder / 'stages.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(value) + '\n')
    write_json(folder / 'worker_stage.json', value)
    print(json.dumps({'phase': phase, 'pid': os.getpid(),
                      'private_bytes': value['worker_private_memory'].get('private_bytes'),
                      'system_commit_bytes': value['system_memory'].get('commit_used_bytes'),
                      'cuda': value.get('cuda')}), flush=True)
    return value


def admit_stage(folder, name, budget):
    write_json(folder / 'stage_request.json', {'name':name,'incremental_private_budget_bytes':budget,'at_utc':now()})
    until=time.monotonic()+30
    while time.monotonic()<until:
        path=folder/'stage_admission.json'
        if path.exists():
            value=read_json(path)
            if value['name']==name:
                require(value['passed'], 'Fresh bounded stage admission rejected: '+name)
                return
        if (folder/'stop.json').exists(): raise ValueError('Parent stop before stage: '+name)
        time.sleep(.2)
    raise ValueError('Missing fresh parent admission: '+name)


def worker(args):
    storage(); folder = Path(args.folder)
    write_json(folder / 'worker_boot_identity.json', {'pid': os.getpid(), 'ppid': os.getppid(),
               'source': record(__file__), 'at_utc': now(), 'CUDA_imported': False})
    until = time.monotonic() + 15
    while time.monotonic() < until:
        lease = read_json(LEASE)
        if lease.get('owner') == OWNER and lease.get('actual_GPU_running') is True and lease.get('actual_owned_worker_pid') == os.getpid(): break
        time.sleep(.1)
    else: raise ValueError('Actual delegated PID lease required')
    plan=read_json(folder/'plan.json')
    require(all(record(path)==item for path,item in plan['pinned_sources'].items()),'Admitted diagnostic source changed')
    require('torch' not in sys.modules, 'CUDA12.9 preload must precede Torch')
    stage(folder, 'before_coherent_preload')
    from preload_nemo_linker_v1 import preload
    preloaded = preload()
    require(preloaded['actual_binding_linker']['version'] == [12,9] and not preloaded['cuda_initialized'], 'Coherent CUDA12.9 preload required')
    stage(folder, 'after_preload_before_Torch')
    import torch
    torch.set_num_threads(4)  # Preserve the original runtime.load_model CPU policy.
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    cap = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)
    torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'], device=0)
    stage(folder, 'runtime_ready', torch, torch_version=torch.__version__, torch_CUDA=torch.version.cuda,
          torch_CPU_threads=torch.get_num_threads(),
          coherent_linker=preloaded, cap=cap, allocator_cap_bytes=8_000_000_000,
          maximum_full_joint_bytes=1_700_000_000, native_end_cache_bytes=3_000_000_000)
    outcome={'passed':True,'positive_native_update':False,'training_run_started':False}
    if args.mode in ('adam_reference','adam_single'):
        from asr18_adam_policy_probe_v1 import probe
        outcome=probe(folder,single=args.mode=='adam_single',stage=lambda phase,**extra:stage(folder,phase,torch,**extra),
                      admit=lambda name,budget:admit_stage(folder,name,budget))
    minimal_policy=None
    if args.minimal_native:
        require(args.mode in ('native_mixed','native_augmentation'),'Minimal native import is scoped to diagnostic training children')
        proof_path=AREA/'official_native_kernel_minimal_02/kernel_equivalence.json'
        proof=read_json(proof_path)
        require(proof['passed'] and proof['loader']==record(SOURCE/'native_tdt_minimal_import_v1.py')
                and proof['manifest']==record(SOURCE/'native_tdt_minimal_import_manifest_v1.json'),'Pinned actual native equivalence required')
        measurements=[]
        for name in ('official_native_kernel_reference_01','official_native_kernel_minimal_01'):
            require(read_json(AREA/name/'report.json')['passed'],'Both actual kernel probes must have passed')
            stages=[json.loads(line) for line in (AREA/name/'stages.jsonl').read_text().splitlines()]
            measurements.append([item for item in stages if item['phase']=='native_kernel_case_complete'][-1]['worker_private_memory']['private_bytes'])
        saving=measurements[0]-measurements[1]
        require(saving>100_000_000,'Measured post-kernel private saving required')
        reference_stages=[json.loads(line) for line in (AREA/'native_repeated_memory_validation_01/stages.jsonl').read_text().splitlines()]
        before_native=next(item for item in reference_stages if item['phase']=='exact_fingerprint_350baseline_reused')
        after_native=next(item for item in reference_stages if item['phase']=='native_microbatch_complete')
        observed_first_native_increment=(after_native['worker_private_memory']['private_bytes']
                                         -before_native['worker_private_memory']['private_bytes'])
        admit_stage(folder,'minimal_native_import_preload',100_000_000)
        from native_tdt_minimal_import_v1 import prepare
        minimal_policy=prepare()
        minimal_policy.update(kernel_proof=record(proof_path),measured_postkernel_private_saving_bytes=saving,
            first_native_reference=record(AREA/'native_repeated_memory_validation_01/stages.jsonl'),
            observed_first_native_private_increment_bytes=observed_first_native_increment,
            cold_native_budget_bytes=observed_first_native_increment-saving+250_000_000,
            stage_estimate_derivation='Observed firstnative increment minus independentlymeasured postkernel saving plus250MB geometry/safety margin; globalguards unchanged')
        stage(folder,'minimal_native_import_preloaded',torch,policy=minimal_policy)
    if args.mode in ('native_import','native_kernel_reference','native_kernel_minimal'):
        admit_stage(folder,'official_native_import_constructor',1_000_000_000)
        before=stage(folder,'before_official_native_import_constructor',torch)
        if args.mode=='native_kernel_minimal':
            from native_tdt_minimal_import_v1 import prepare
            write_json(folder/'minimal_import_policy.json',prepare())
        from native_tdt_checkpointed_v4 import _native_loss
        native=_native_loss(8192,(0,1,2,3,4),0.)
        after=stage(folder,'after_official_native_import_constructor',torch)
        outcome['official_native_import_private_increment_bytes']=(after['worker_private_memory']['private_bytes']
                                                                  -before['worker_private_memory']['private_bytes'])
        if args.mode!='native_import':
            admit_stage(folder,'official_native_kernel_forward_backward',1_000_000_000)
            tensors=[]
            for frames,tokens,width in ((6,2,8711),(12,5,8711),(24,13,8711)):
                lexical=(torch.arange(frames*(tokens+1)*width,device='cuda',dtype=torch.float32).remainder(997)/997).reshape(1,frames,tokens+1,width).requires_grad_()
                raw=(torch.arange(frames*(tokens+1)*5,device='cuda',dtype=torch.float32).remainder(19)/19).reshape(1,frames,tokens+1,5).requires_grad_()
                duration=torch.nn.functional.log_softmax(raw,dim=-1)
                labels=torch.arange(1,tokens+1,device='cuda',dtype=torch.long)[None]
                value=native.loss(lexical,duration,labels,torch.tensor([frames],device='cuda'),torch.tensor([tokens],device='cuda'),
                                  native.blank,native.durations,native.reduction,native.fastemit_lambda,native.clamp,native.sigma,native.omega)
                value.sum().backward();torch.cuda.synchronize()
                require(bool(torch.isfinite(value).all()) and bool(torch.isfinite(lexical.grad).all()),'Finite actual native loss/gradient required')
                tensors.append({'shape':[frames,tokens,width],'loss':value.detach().cpu(),'lexical_gradient':lexical.grad.cpu(),'duration_gradient':raw.grad.cpu()})
                del lexical,raw,duration,labels,value;torch.cuda.empty_cache()
                stage(folder,'native_kernel_case_complete',torch,frames=frames,tokens=tokens)
            if args.mode=='native_kernel_reference':torch.save(tensors,folder/'reference_tensors.pt')
            else:
                reference=AREA/'official_native_kernel_reference_01/reference_tensors.pt'
                saved=torch.load(reference,map_location='cpu',weights_only=True,mmap=True)
                require(all(actual['shape']==prior['shape'] and all(torch.equal(actual[key].contiguous().view(torch.uint8),prior[key].contiguous().view(torch.uint8))
                    for key in ('loss','lexical_gradient','duration_gradient')) for actual,prior in zip(tensors,saved)),'Minimal import changed official loss/gradients')
                outcome['native_loss_and_both_gradients_byte_exact']=True
                write_json(folder/'kernel_equivalence.json',{'passed':True,'reference':record(reference),'cases':len(tensors),
                    'loader':record(SOURCE/'native_tdt_minimal_import_v1.py'),'manifest':record(SOURCE/'native_tdt_minimal_import_manifest_v1.json'),
                    'loss_and_lexical_duration_gradients_byte_exact':True})
            outcome['real_official_native_kernel_cases']=len(tensors)
        del native
    if args.mode in ('base','teacher_cache','encoder_reference','encoder_streamed','native','native_mixed','native_augmentation','eval_final','forward_cache','storage_equivalence','mixed_load_equivalence'):
        stage(folder,'actual_live_runtime_data_gate_exit_window',torch)
        time.sleep(45 if args.mode=='base' else 2)
        mixed=args.mode in ('teacher_cache','encoder_reference','encoder_streamed','native_mixed','native_augmentation','eval_final')
        # The original4GB estimate was for FP32 loading. The mixed loader's
        # exact1.217GB frozen-matrix storage reduction has passed actual GPU
        # equivalence and full350DEV; subtract those concrete live weight bytes.
        base_budget=4_000_000_000-1_217_066_496 if mixed else 4_000_000_000
        admit_stage(folder,'direct_GPU_mixed_base_and_teacher' if mixed else 'direct_GPU_FP32_base_and_teacher',base_budget)
        from transformers import AutoModelForTDT,AutoProcessor
        base=Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
        require(base.is_relative_to(PROJECT.resolve()) or base.is_relative_to(STORAGE.resolve()), 'Allowed existing base required')
        processor=AutoProcessor.from_pretrained(base,local_files_only=True)
        stage(folder,'before_direct_GPU_FP32_base_load',torch)
        load_policy=None
        if args.stream_frozen or args.mode=='encoder_streamed':
            require(args.mode in ('encoder_streamed','native_mixed','native_augmentation','eval_final'),'Streaming scope is diagnostic only')
            if args.mode!='encoder_streamed':
                proof=read_json(AREA/'streamed_encoder_candidate_06/streaming_equivalence.json')
                require(proof['passed'] and proof['loader']==record(SOURCE/'stream_frozen_encoder_matrices_v1.py'),'Actual output and LoRA gradient proof required')
            from stream_frozen_encoder_matrices_v1 import load
            model,load_policy=load(base)
            from streamed_generation_device_v1 import apply
            load_policy['generation_execution_device']=apply(model)
        elif args.mode in ('teacher_cache','encoder_reference','native_mixed','native_augmentation','eval_final'):
            proof=read_json(AREA/'mixed_load_equivalence_01/mixed_load_equivalence.json')
            require(proof['passed'] and proof['loader']==record(SOURCE/'direct_mixed_encoder_load_v1.py'),'Actual mixed-load numerical proof required')
            from direct_mixed_encoder_load_v1 import load
            model,load_policy=load(base)
        else:
            model=AutoModelForTDT.from_pretrained(base,dtype=torch.float32,local_files_only=True,
                                                low_cpu_mem_usage=True,device_map={'':'cuda:0'})
        require(all(p.device.type=='cuda' or (hasattr(model,'_diagnostic_frozen_backing') and not p.requires_grad) for p in model.parameters()),
                'Only declared frozen mapped weights may remain on CPU')
        torch.cuda.synchronize();stage(folder,'after_direct_GPU_mixed_base_load' if load_policy else 'after_direct_GPU_FP32_base_load',torch)
        teacher={};cached_teacher=None;cached_teacher_policy=None
        if args.cached_teacher:
            require(args.mode=='native_mixed','Cachedteacher is restricted to the fixed diagnostic native workload')
            from asr18_cached_teacher_control_v1 import restore_for_training
            cached_teacher,cached_teacher_policy=restore_for_training(model)
            teacher=cached_teacher.heads
            stage(folder,'exact_cached_teacher_restored_without_live_teacher_copy',torch,policy=cached_teacher_policy)
        elif args.mode not in ('teacher_cache','encoder_reference','encoder_streamed','forward_cache','storage_equivalence','mixed_load_equivalence','eval_final'):
            teacher={n:copy.deepcopy(getattr(model,n)).eval().requires_grad_(False) for n in ('encoder_projector','decoder','joint')}
            torch.cuda.synchronize();stage(folder,'after_original_teacher_copy',torch)
        elif args.mode in ('storage_equivalence','mixed_load_equivalence'):
            admit_stage(folder,'FP32_storage_reference_first_encoder',1_600_000_000)
            from runtime import DATA,inputs_for
            from frozen_encoder_compute_storage_v1 import apply
            rows=[json.loads(line) for line in (DATA/'validation_tr.jsonl').read_text(encoding='utf-8').splitlines() if line]
            selected=[rows[i] for i in (0,25,49)];reference=[];model.eval();model.requires_grad_(False)
            for index,row in enumerate(selected):
                inputs=inputs_for(processor,row)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):encoded=model.encoder(**inputs)
                reference.append((encoded.last_hidden_state.detach().cpu(),encoded.attention_mask.detach().cpu()))
                del inputs,encoded;torch.cuda.empty_cache()
                stage(folder,'FP32_storage_reference_encoder',torch,clip=index+1)
            if args.mode=='mixed_load_equivalence':
                del model;gc.collect();torch.cuda.empty_cache();stage(folder,'after_FP32_reference_model_release',torch)
                from direct_mixed_encoder_load_v1 import load
                model,policy=load(base);model.eval();model.requires_grad_(False)
            else:policy=apply(model)
            stage(folder,'after_frozen_matrix_BF16_storage',torch,policy=policy)
            comparisons=[]
            for index,row in enumerate(selected):
                inputs=inputs_for(processor,row)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):encoded=model.encoder(**inputs)
                actual=encoded.last_hidden_state.detach().cpu();mask=encoded.attention_mask.detach().cpu()
                equal=(torch.equal(reference[index][0].contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8))
                       and torch.equal(reference[index][1],mask))
                comparison={'source_id':row['source_id'],'encoder_frames':int(mask.sum()),'byte_exact':equal,
                            'max_absolute_difference':float((reference[index][0].float()-actual.float()).abs().max())}
                require(equal,'Frozen BF16 storage changed actual autocast encoder outputs')
                comparisons.append(comparison);del inputs,encoded,actual,mask;torch.cuda.empty_cache()
            write_json(folder/'storage_equivalence.json',{'passed':True,'policy':policy,'comparisons':comparisons,
                       'source_weights_modified_on_disk':False,'full_DEV_and_native_update_proof_pending':True})
            if args.mode=='mixed_load_equivalence':
                write_json(folder/'mixed_load_equivalence.json',{'passed':True,'policy':policy,'comparisons':comparisons,
                    'loader':record(SOURCE/'direct_mixed_encoder_load_v1.py'),'full_DEV_and_native_update_proof_pending':True})
            outcome['frozen_storage_encoder_outputs_byte_exact']=True
        elif args.mode=='forward_cache':
            admit_stage(folder,'first_forward_host_runtime_allowance',1_600_000_000)
            from runtime import DATA,inputs_for
            rows=[json.loads(line) for line in (DATA/'validation_tr.jsonl').read_text(encoding='utf-8').splitlines() if line]
            model.eval();model.requires_grad_(False)
            stage(folder,'before_first_encoder_forward',torch,backend_modules=owned_backend_modules(),
                  CUDNN_cache_limit=os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'))
            for index,row in enumerate(rows):
                if (folder/'stop.json').exists():raise ValueError('Parent stopped bounded first-forward probe')
                inputs=inputs_for(processor,row)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    encoded=model.encoder(**inputs)
                    if index==0:
                        torch.cuda.synchronize();stage(folder,'after_first_encoder_before_decoder',torch,backend_modules=owned_backend_modules())
                        ep=model.encoder_projector(encoded.last_hidden_state[:,:1])
                        dp=model.decoder(torch.tensor([[8192]],device='cuda'))
                        torch.cuda.synchronize();stage(folder,'after_first_decoder_before_joint',torch,backend_modules=owned_backend_modules())
                        joint=model.joint(ep[:,:,None],dp[:,None])
                        torch.cuda.synchronize();stage(folder,'after_first_joint',torch,backend_modules=owned_backend_modules())
                        del ep,dp,joint
                del inputs,encoded
                torch.cuda.empty_cache()
                stage(folder,'encoder_shape_complete',torch,clip=index+1)
            outcome['first_forward_and50_encoder_shapes_proved']=True
        if args.mode=='teacher_cache':
            from asr18_cached_teacher_control_v1 import prepare
            outcome=prepare(model,folder,stage=lambda phase,**extra:stage(folder,phase,torch,**extra),
                            admit=lambda name,budget:admit_stage(folder,name,budget))
        if args.mode in ('encoder_reference','encoder_streamed','native','native_mixed','native_augmentation','eval_final'):
            from asr18_diagnostic_bootstrap_v1 import bootstrap
            from asr18_native_validation_v1 import validate
            emit=lambda phase,**details:stage(folder,phase,torch,**details)
            accept=lambda name,budget:admit_stage(folder,name,budget)
            optimizer,provenance=bootstrap(model,processor,torch=torch,stage=emit,admit=accept)
            if load_policy is not None:
                provenance['frozen_encoder_storage_policy']=load_policy
                provenance['frozen_encoder_storage_loader']=record(SOURCE/('stream_frozen_encoder_matrices_v1.py'
                    if hasattr(model,'_diagnostic_frozen_backing') else 'direct_mixed_encoder_load_v1.py'))
                provenance['CUDNN_execution_plan_cache_limit']=os.environ.get('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT')
            if minimal_policy is not None:provenance['minimal_native_import']=minimal_policy
            if cached_teacher_policy is not None:provenance['cached_teacher_policy']=cached_teacher_policy
            write_json(folder/'bootstrap_provenance.json',provenance)
            if args.mode in ('encoder_reference','encoder_streamed'):
                del optimizer;gc.collect();torch.cuda.empty_cache()
                from asr18_streamed_encoder_equivalence_v1 import validate as validate_encoder
                outcome=validate_encoder(model,processor,folder,candidate=args.mode=='encoder_streamed',stage=emit,admit=accept)
            elif args.mode=='eval_final':
                require(plan['checkpoint']==record(args.checkpoint),'Pinned diagnostic checkpoint changed')
                del optimizer;gc.collect();torch.cuda.empty_cache()
                from asr18_final_DEV_v1 import evaluate_checkpoint
                outcome=evaluate_checkpoint(model,processor,args.checkpoint,folder,stage=emit,admit=accept)
            elif args.mode=='native_augmentation':
                from asr18_fastmss_native_smoke_v1 import validate as validate_augmentation
                outcome=validate_augmentation(model,processor,teacher,optimizer,provenance,folder,stage=emit,admit=accept)
                del optimizer
            else:
                outcome=validate(model,processor,teacher,optimizer,provenance,folder,stage=emit,admit=accept,cached_teacher=cached_teacher)
                del optimizer
        del teacher,model,processor;gc.collect();torch.cuda.empty_cache()
        stage(folder,'after_model_release',torch)
    outcome.update(mode=args.mode,pid=os.getpid(),completed_at_utc=now(),
                   sources_unchanged=all(record(path)==item for path,item in plan['pinned_sources'].items()))
    write_json(folder / 'worker_completion.json',outcome)


def controller(args):
    storage(); AREA.mkdir(parents=True, exist_ok=True)
    folder = AREA / args.name
    require(not folder.exists(), 'Fresh diagnostic namespace required')
    lease = read_json(LEASE)
    require(lease.get('owner') == OWNER and lease.get('actual_GPU_running') is False and not lease.get('actual_owned_worker_pid'), 'Exclusive idle delegated GPU lease required')
    ready=concurrent_data_observation()
    baseline = sample_memory(os.getpid(), 'bounded_diagnostic_admission')
    system = baseline['system_memory']; budget = 3_000_000_000
    require(system['available'] and system['commit_used_bytes'] + budget < .9 * system['commit_limit_bytes']
            and system['commit_available_bytes'] - budget > 2_000_000_000, 'Bounded runtime stage fails unchanged commit/headroom admission')
    folder.mkdir()
    dependencies=[Path(__file__).resolve(),*[SOURCE/name for name in ('preload_nemo_linker_v1.py','resource_watchdog_v2.py',
        'native_tdt_checkpointed_v4.py','asr18_diagnostic_bootstrap_v1.py','asr18_native_validation_v1.py',
        'bpe_broad_full207_optimizer_v1.py','batched_lora_tr_metadata_v1.py','runtime.py','encoder_lora.py','extend_vocab.py',
        'train_lora.py','train_lora_batched_tsc_bpe_full_decoder_v1.py','train_lora_batched_tsc_expanded_TRAIN_v1.py',
        'TR_tdt_population_weighting_v1.py','bpe_training_label_catalog_v1.py','frozen_encoder_compute_storage_v1.py',
        'direct_mixed_encoder_load_v1.py','asr18_final_DEV_v1.py','native_tdt_minimal_import_v1.py','native_tdt_minimal_import_manifest_v1.json',
        'asr18_adam_policy_probe_v1.py')]]
    dependencies.append(SOURCE/'asr18_cached_teacher_control_v1.py')
    dependencies.extend(SOURCE/name for name in ('stream_frozen_encoder_matrices_v1.py','asr18_streamed_encoder_equivalence_v1.py'))
    dependencies.append(SOURCE/'streamed_generation_device_v1.py')
    dependencies.append(SOURCE/'asr18_fastmss_native_smoke_v1.py')
    if args.mode=='native_augmentation':
        dependencies.extend([SOURCE/'fastmss_train_augmentation_v1.py',SOURCE/'fastmss_train_augmentation_v1.pilot.disabled.json',
            STORAGE/'analysis/FastMSS_integration_v1/FastMSS_integration_readiness_v1.json',
            STORAGE/'analysis/FastMSS_integration_v1/upstream_RIR_pilot_01/FastMSS_RIR_assets.json'])
    if args.stream_frozen or args.mode=='encoder_streamed':dependencies.append(AREA/'frozen_weight_bank_v1/report.json')
    checkpoint=None
    if args.mode=='eval_final':
        require(args.checkpoint and Path(args.checkpoint).resolve().is_relative_to(AREA.resolve()),'E-only diagnostic checkpoint required')
        checkpoint=record(args.checkpoint)
    write_json(folder / 'plan.json', {'mode': args.mode, 'source': record(__file__),
               'pinned_sources':{str(path):record(path) for path in dependencies},'checkpoint':checkpoint,'baseline': baseline,
               'incremental_runtime_budget_bytes': budget, 'data_ready': ready,
               'historical_12_87GB_worker_not_blindly_admitted': True, 'created_at_utc': now()})
    command = [sys.executable, '-X', 'utf8', '-B', str(Path(__file__).resolve()), '--worker',
               '--mode', args.mode, '--folder', str(folder), '--name', args.name]
    if args.checkpoint:command.extend(['--checkpoint',args.checkpoint])
    if args.minimal_native:command.append('--minimal-native')
    if args.cached_teacher:command.append('--cached-teacher')
    if args.stream_frozen:command.append('--stream-frozen')
    guard = ResourceGuard(); child = None; worker_pid = None; reason = None; samples = [];stop_at=None
    try:
        with (folder / 'worker.log').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            until = time.monotonic() + 10
            while not (folder / 'worker_boot_identity.json').exists() and child.poll() is None and time.monotonic() < until: time.sleep(.1)
            boot = read_json(folder / 'worker_boot_identity.json')
            require(boot['source'] == record(__file__) and (boot['pid'] == child.pid or boot['ppid'] == child.pid), 'Owned launcher-to-actual-worker identity required')
            worker_pid = boot['pid']
            lease.update(actual_GPU_running=True, actual_owned_worker_pid=worker_pid,
                         actual_owned_launcher_pid=child.pid,
                         actual_training_optimizer_running=False, running=True,
                         status='memory_diagnostic_'+args.mode, diagnostic_folder=str(folder), updated_at_utc=now())
            write_json(LEASE, lease)
            print(json.dumps({'owned_GPU_actual_worker_pid': worker_pid, 'launcher_pid': child.pid, 'mode': args.mode, 'folder': str(folder)}), flush=True)
            started = time.monotonic(); admitted=set()
            while child.poll() is None:
                value = sample_memory(worker_pid, 'diagnostic_parent_fresh_poll')
                value['concurrent_data']=concurrent_data_observation()
                stop, detail = guard.observe(value['system_memory'], value['worker_private_memory'])
                samples.append(value)
                write_json(folder / 'parent_live.json', {'sample': value, 'guard': detail, 'stop_reason': stop})
                if (folder/'stage_request.json').exists():
                    request=read_json(folder/'stage_request.json')
                    if request['name'] not in admitted:
                        current=value['system_memory']; incremental=request['incremental_private_budget_bytes']
                        passed=(current['commit_used_bytes']+incremental<.9*current['commit_limit_bytes']
                                and current['commit_available_bytes']-incremental>2_000_000_000)
                        write_json(folder/'stage_admission.json',{'name':request['name'],'passed':passed,'fresh_sample':value,
                                   'incremental_budget_bytes':incremental,'at_utc':now()})
                        admitted.add(request['name'])
                        lease.update(actual_training_optimizer_running=passed and request['name'].startswith('native_update_'),
                                     diagnostic_phase=request['name'],updated_at_utc=now())
                        write_json(LEASE,lease)
                timeout=1800 if args.mode.startswith('native') else (900 if args.mode=='eval_final' else 300)
                if (stop or time.monotonic()-started>timeout) and stop_at is None:
                    reason=stop or 'bounded_diagnostic_timeout';stop_at=time.monotonic()
                    write_json(folder/'stop.json',{'reason':reason,'at_utc':now()})
                    if not args.mode.startswith('native'):terminate_runtime_worker(worker_pid,child);break
                if stop_at is not None and time.monotonic()-stop_at>60:
                    terminate_runtime_worker(worker_pid,child);break
                time.sleep(1)
            code = child.wait()
        report = {'mode': args.mode, 'worker_pid': worker_pid, 'launcher_pid': child.pid, 'exit_code': code, 'stop_reason': reason,
                  'samples': samples, 'source_unchanged': record(__file__) == read_json(folder / 'plan.json')['source'],
                  'completed_at_utc': now(), 'positive_native_update': False}
        completion=read_json(folder/'worker_completion.json') if (folder/'worker_completion.json').exists() else None
        report['completion']=completion
        report['passed']=bool(code==0 and completion and completion['passed'] and completion['sources_unchanged'] and not reason)
        progress=read_json(folder/'native_progress.json') if (folder/'native_progress.json').exists() else None
        report['durable_native_progress']=progress
        report['positive_native_update']=bool((completion and completion.get('positive_native_update')) or (progress and progress.get('completed_updates',0)>0))
        write_json(folder / 'report.json', report)
        write_json(AREA / 'memory_diagnosis_status.json', {'status': 'bounded_'+args.mode+'_probe_closed',
                   'report': record(folder / 'report.json'), 'passed': report['passed'], 'updated_at_utc': now(),
                   'historical_causal_attribution': 'Final system increase mostly outside recorded worker private; precise outside attribution unavailable',
                   'fix_verified':bool(report['passed'] and completion and completion.get('bounded_memory_validation_complete')),
                   'bounded_native_stability_verified':bool(report['passed'] and completion and completion.get('real_native_updates',completion.get('diagnostic_training_completed_updates'))==20),
                   'full_broad_stage_training_admission_or_fit_proved':False,'owned_GPU_child_running':False})
        print(json.dumps({k:v for k,v in report.items() if k != 'samples'}), flush=True)
        require(report['passed'], 'Diagnostic worker did not pass')
    finally:
        if child is not None and child.poll() is None: terminate_runtime_worker(worker_pid, child)
        latest = read_json(LEASE)
        require(latest['owner'] == OWNER and (worker_pid is None or latest.get('actual_owned_worker_pid') == worker_pid), 'GPU lease identity changed')
        latest.update(actual_GPU_running=False, actual_owned_worker_pid=None, actual_training_optimizer_running=False,
                      actual_owned_launcher_pid=None, running=False, status='memory_diagnostic_idle', updated_at_utc=now())
        write_json(LEASE, latest)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--mode', choices=['runtime','adam_reference','adam_single','native_import','native_kernel_reference','native_kernel_minimal','teacher_cache','encoder_reference','encoder_streamed','base','native','native_mixed','native_augmentation','eval_final','forward_cache','storage_equivalence','mixed_load_equivalence'], default='runtime')
    p.add_argument('--name', required=True); p.add_argument('--folder'); p.add_argument('--worker', action='store_true')
    p.add_argument('--checkpoint')
    p.add_argument('--minimal-native',action='store_true')
    p.add_argument('--cached-teacher',action='store_true')
    p.add_argument('--stream-frozen',action='store_true')
    a=p.parse_args(); worker(a) if a.worker else controller(a)
