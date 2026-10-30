"""Exclusive finite production-caller preparation. NO production launch command."""
from __future__ import annotations
import argparse,copy,gc,json,os,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'outputs/parakeet_turkish'
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
LEASE=PROJECT/'work/gpu_training_owner.json';OWNER='/root/memory_diagnosis'
sys.path.insert(0,str(SOURCE))
from asr18_memory_diagnosis_v1 import storage,record,require,now,stage,admit_stage,concurrent_data_observation,terminate_runtime_worker
from resource_watchdog_v2 import ResourceGuard,sample_memory
from atomic_io_retry_v1 import write_json,read_json

def worker(a):
    storage();folder=Path(a.folder);plan=read_json(folder/'plan.json')
    write_json(folder/'worker_boot_identity.json',{'pid':os.getpid(),'ppid':os.getppid(),'source':record(__file__)})
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        lease=read_json(LEASE)
        if lease['owner']==OWNER and lease['actual_GPU_running'] and lease['actual_owned_worker_pid']==os.getpid():break
        time.sleep(.1)
    else:raise ValueError('Actual exclusive worker lease required')
    require(all(record(item['path'])==item for item in plan['pinned_sources']),'Preparation source changed after review/admission')
    runtime=read_json(AREA/'broad_audio_runtime_v1.json')
    require(runtime['version']=='16.0.1' and all(record(item['path'])==item for item in runtime['files']),'Isolated E audio runtime changed')
    for proof_path,implementation in ((AREA/'streamed_encoder_candidate_06/streaming_equivalence.json','stream_frozen_encoder_matrices_v1.py'),
                                     (AREA/'official_native_kernel_minimal_02/kernel_equivalence.json','native_tdt_minimal_import_v1.py')):
        proof=read_json(proof_path);require(proof['passed'] and proof['loader']==record(SOURCE/implementation),'Actual byte-exact proof does not bind admitted implementation')
    require('torch' not in sys.modules,'Coherent preload before Torch')
    from preload_nemo_linker_v1 import preload
    coherent=preload();require(coherent['actual_binding_linker']['version']==[12,9] and not coherent['cuda_initialized'],'Pinned coherent CUDA12.9')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    cap=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
    torch.cuda.set_per_process_memory_fraction(cap['per_process_allocator_fraction'],device=0)
    emit=lambda phase,**extra:stage(folder,phase,torch,**extra)
    admit=lambda name,budget:admit_stage(folder,name,budget)
    emit('production_prep_runtime_ready',coherent=coherent,cap=cap)
    from native_tdt_minimal_import_v1 import prepare
    minimal=prepare();emit('production_prep_narrow_native_ready',policy=minimal)
    admit('readonly_streamed_base_and_original_teacher',900_000_000)
    from transformers import AutoProcessor
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply
    base=Path((PROJECT/'work/base_model_path.txt').read_text().strip())
    model,loader=load(base);device_policy=apply(model);processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    teacher={name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False) for name in ('encoder_projector','decoder','joint')}
    emit('production_prep_streamed_base_teacher_ready',loader=loader,device_policy=device_policy)
    from bpe_broad_streamed_bootstrap_v1 import bootstrap
    optimizer,provenance,replay=bootstrap(model,processor,torch=torch,stage=emit,admit=admit)
    provenance.update(frozen_loader=record(SOURCE/'stream_frozen_encoder_matrices_v1.py'),bootstrap=record(SOURCE/'bpe_broad_streamed_bootstrap_v1.py'),minimal_native=minimal,logical_device=device_policy)
    write_json(folder/'bootstrap_provenance.json',provenance)
    config=read_json(a.config);config['_path']=a.config
    def stop():require(not (folder/'stop.json').exists(),'Fresh parent cooperative stop')
    from tdt_joint_canonical_2d_fp32_v2 import apply as apply_canonical_joint
    provenance['NEW_joint_numerical_policy']=apply_canonical_joint(model.joint)
    from asr18_broad_stream_checkpoint20_probe_v2 import prepare_probe
    result=prepare_probe(model,processor,teacher,optimizer,provenance,replay,folder,config,torch=torch,stage=emit,admit=admit,stop_check=stop)
    del optimizer,teacher,model,processor;gc.collect();torch.cuda.empty_cache();emit('production_caller_probe_models_released')
    result.update(sources_unchanged=all(record(item['path'])==item for item in plan['pinned_sources']))
    write_json(folder/'worker_completion.json',result)

def controller(a):
    storage();folder=AREA/a.name;require(not folder.exists(),'Fresh bounded preparation namespace required')
    lease=read_json(LEASE);require(lease['owner']==OWNER and not lease['actual_GPU_running'] and not lease.get('actual_owned_worker_pid'),'Exclusive idle lease required')
    baseline=sample_memory(os.getpid(),'bounded_new_caller_admission');system=baseline['system_memory']
    require(system['commit_used_bytes']+3_000_000_000<.9*system['commit_limit_bytes'] and system['commit_available_bytes']-3_000_000_000>2_000_000_000,'Unchanged bounded runtime admission failed')
    dependencies=[Path(__file__),Path(a.config),PROJECT/'work/research/asr18_memory_diagnosis_v1.py',PROJECT/'work/research/ASR18_bounded_CPU_phase_v3.py',PROJECT/'work/research/decode_ASR18_broad_batch_v1.py']
    dependencies.append(Path(read_json(a.config)['probe_reader_proof']))
    dependencies.extend(SOURCE/name for name in ('bpe_broad_production_v3.py','bpe_broad_production_state_v1.py','bpe_broad_streamed_bootstrap_v1.py','bpe_broad_audio_v1.py',
        'stream_frozen_encoder_matrices_v1.py','streamed_generation_device_v1.py','native_tdt_minimal_import_v1.py','native_tdt_minimal_import_manifest_v1.json',
        'asr18_diagnostic_bootstrap_v1.py','bpe_broad_full207_optimizer_v1.py','preload_nemo_linker_v1.py','launch_with_memory_guard_TSC_resume_v3.py',
        'native_tdt_checkpointed_v4.py','resource_watchdog_v2.py','fastmss_train_augmentation_v1.py','fastmss_ordered_draw_binding_v1.py',
        'fastmss_broad_optin_v1.json','batched_lora_tr_metadata_v1.py','train_lora_batched_tsc_expanded_TRAIN_v1.py','runtime.py',
        'encoder_lora.py','extend_vocab.py','train_lora.py','train_lora_batched_tsc_bpe_full_decoder_v1.py','tsc_bpe_full_decoder_transition_v1.py',
        'bpe_full_decoder_optimizer_v1.py','train_lora_batched_data_v11.py','tr_frame_metadata_v1.py','atomic_io_retry_v1.py'))
    dependencies.extend([AREA/'frozen_weight_bank_v1/report.json',AREA/'source_projection_v4/report.json',AREA/'optimizer_projected_v4/report.json',
        AREA/'broad_audio_runtime_v1.json',ROOT/'analysis/FastMSS_integration_v1/upstream_RIR_pilot_01/FastMSS_RIR_assets.json'])
    dependencies.extend([AREA/'streamed_encoder_candidate_06/streaming_equivalence.json',AREA/'official_native_kernel_minimal_02/kernel_equivalence.json'])
    dependencies.extend([PROJECT/'work/research/decode_ASR18_broad_batch_v6.py',Path(read_json(a.config)['probe_manifest']),Path(read_json(a.config)['heldout_PCM_binding']['path'])])
    dependencies.extend(SOURCE/name for name in ('bpe_broad_production_v5.py','asr18_broad_integrated_probe_v2.py','bpe_broad_transaction_v1.py','bpe_broad_journal_branch_v1.py','bpe_broad_source_identity_v1.py','bpe_broad_audio_v2.py','bpe_broad_final_population_v2.py','fastmss_cpu_prepared_transport_v1.py','tdt_joint_canonical_2d_fp32_v2.py','tdt_tiled_full_native_v6.py','tdt_tile_native_gradient_v1.py'))
    dependencies.extend([AREA/'real_target_TDT_gradients_06/real_model_equivalence.json',AREA/'real_target_TDT_large_02/full_large_row_validation.json',ROOT/'analysis/FastMSS_integration_v1/CPU_transport_fixture_03/CPU_prepared_transport_validation.json'])
    dependencies.extend([SOURCE/'bpe_broad_audio_FLAC_records_v1.py',AREA/'full14_FLAC_CPU_actual_01/report.json',AREA/'fixture_FLAC_carrier_bindings_v1.json'])
    for item in read_json(AREA/'fixture_FLAC_carrier_bindings_v1.json')['sources']:
        dependencies.extend(Path(item[key]['path']) for key in ('receipt','validation','index'))
    dependencies.extend(SOURCE/name for name in ('asr18_broad_streamed_checkpoint_probe_v1.py','bpe_streaming_checkpoint_v1.py','bpe_streaming_training_state_v1.py'))
    dependencies.append(Path(read_json(a.config)['streaming_checkpoint_CPU_proof']['path']))
    dependencies.extend(Path(v['path']) for v in read_json(a.config)['coherent_diagnostic_parent'].values())
    dependencies.extend(SOURCE/n for n in ('asr18_broad_stream_checkpoint20_probe_v2.py','asr18_repeat_fixture_cursor_v1.py'))
    dependencies.extend([AREA/'legacy_CPU_step_retention_01/report.json',AREA/'streamed_checkpoint_continuation_01/actual_owned_CLOSED0.json'])
    from asr18_owned_source_closure_v1 import owned_closure
    dependencies.append(PROJECT/'work/research/asr18_owned_source_closure_v1.py')
    dependencies.extend(owned_closure(dependencies))
    dependencies=sorted(set(dependencies))
    dependencies.extend([SOURCE/'bpe_broad_production_v6.py',SOURCE/'tdt_exact_denominator_instance_v1.py',AREA/'full_hard_2D_native_01/actual_owned_CLOSED0.json',AREA/'full_hard_2D_native_01/full_hard_native_validation.json',AREA/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',AREA/'instance_denominator_joint_01/actual_joint_precision.json'])
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))
    folder.mkdir();write_json(folder/'plan.json',{'kind':'finite_new_caller_preparation','production_training_started':False,'pinned_sources':[record(p) for p in dependencies],
        'baseline':baseline,'concurrent_data':concurrent_data_observation(),'max_native_updates':20,'review_receipt':a.review})
    review=read_json(a.review);require(review['scope']=='bounded_production_caller_preparation' and review['accepted'] is True and review['production_approval_claimed'] is False,
        'Explicit bounded diagnostic review required; production approval remains separate')
    require(all(record(item['path'])==item for item in review['bindings']),'Reviewed caller binding changed')
    child=None;pid=None;reason=None;stop_at=None;samples=[];guard=ResourceGuard()
    try:
        with (folder/'worker.log').open('x',encoding='utf-8') as log:
            command=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--worker','--folder',str(folder),'--name',a.name,'--config',a.config,'--review',a.review]
            child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+15
            while not (folder/'worker_boot_identity.json').exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.1)
            boot=read_json(folder/'worker_boot_identity.json');require(boot['source']==record(__file__) and (boot['pid']==child.pid or boot['ppid']==child.pid),'Actual owned worker identity')
            pid=boot['pid'];lease.update(actual_GPU_running=True,actual_owned_worker_pid=pid,actual_owned_launcher_pid=child.pid,running=True,
                status='bounded_new_caller_preparation',diagnostic_folder=str(folder),updated_at_utc=now());write_json(LEASE,lease)
            print(json.dumps({'owned_GPU_actual_worker_pid':pid,'launcher_pid':child.pid,'folder':str(folder)}),flush=True)
            began=time.monotonic();admitted=set()
            while child.poll() is None:
                sample=sample_memory(pid,'new_caller_parent_fresh_poll');sample['concurrent_data']=concurrent_data_observation();samples.append(sample)
                stop,detail=guard.observe(sample['system_memory'],sample['worker_private_memory']);write_json(folder/'parent_live.json',{'sample':sample,'guard':detail,'stop_reason':stop})
                if (folder/'stage_request.json').exists():
                    request=read_json(folder/'stage_request.json')
                    if request['name'] not in admitted:
                        system=sample['system_memory'];budget=request['incremental_private_budget_bytes']
                        passed=system['commit_used_bytes']+budget<.9*system['commit_limit_bytes'] and system['commit_available_bytes']-budget>2_000_000_000
                        write_json(folder/'stage_admission.json',{'name':request['name'],'passed':passed,'fresh_sample':sample,'incremental_budget_bytes':budget})
                        admitted.add(request['name']);lease.update(diagnostic_phase=request['name'],actual_training_optimizer_running=passed and request['name'].startswith('native_integrated_update_'),updated_at_utc=now());write_json(LEASE,lease)
                if (stop or time.monotonic()-began>1200) and stop_at is None:
                    reason=stop or 'finite_preparation_timeout';stop_at=time.monotonic();write_json(folder/'stop.json',{'reason':reason})
                if stop_at is not None and time.monotonic()-stop_at>60:terminate_runtime_worker(pid,child);break
                time.sleep(1)
            code=child.wait()
        completion=read_json(folder/'worker_completion.json') if (folder/'worker_completion.json').exists() else None
        report={'actual_exit_code':code,'PID':pid,'launcher_PID':child.pid,'stop_reason':reason,'samples':samples,'completion':completion,
                'passed':bool(code==0 and not reason and completion and completion['passed'] and completion['sources_unchanged'])}
        progress=read_json(folder/'native_progress.json') if (folder/'native_progress.json').exists() else None
        report['durable_native_progress']=progress;report['actual_positive_native_update']=bool(progress and progress['completed_updates']>0)
        write_json(folder/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('samples','completion')}),flush=True)
        require(report['passed'],'Finite production-caller preparation did not pass')
    finally:
        if child is not None and child.poll() is None:terminate_runtime_worker(pid,child)
        current=read_json(LEASE);require(current['owner']==OWNER and current.get('actual_owned_worker_pid') in (pid,None),'Owned lease identity changed')
        current.update(actual_GPU_running=False,actual_owned_worker_pid=None,actual_owned_launcher_pid=None,actual_training_optimizer_running=False,running=False,status='memory_diagnostic_idle',updated_at_utc=now());write_json(LEASE,current)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--folder');p.add_argument('--worker',action='store_true')
    p.add_argument('--config',default=str(SOURCE/'bpe_broad_stream_checkpoint20_v2.disabled.json'));p.add_argument('--review',required=True);args=p.parse_args()
    worker(args) if args.worker else controller(args)
