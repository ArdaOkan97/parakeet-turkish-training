"""Inactive separate GPU recovery runtime; zero training/frontend operations."""
import gc, os, time
from pathlib import Path
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity
from asr18_separated_worker_forecast_v1 import restore_only_worker_forecast,phase_projection
from asr18_phase_private_monitor_v1 import PhasePrivateMonitor
from resource_watchdog_v2 import sample_memory


def run(config_path,folder,*,parent_source,check_bindings,storage,stage,admit_stage):
    began=time.monotonic();storage();config=read_json(config_path);check_bindings(config)
    require(config['worker_mode']=='restore_only' and config['parent_driver_source']==parent_source
            and digest(config['entire_worker_forecast'])==digest(restore_only_worker_forecast())
            and config['production_enabled'] is False and config['diagnostic_only'] is True,
            'Only separately admitted zero-update exact recovery worker')
    from asr18_targeted_cleaner23_recovery_v2 import verified_requests,recover_all
    requests=verified_requests(config['deferred_recovery_requests'],session=config['session'],
        parent_source=parent_source,guard=lambda:require(not (folder/'stop.json').exists(),'Owned recovery stop'))
    producer=identity();write_json(folder/'worker_boot_identity.json',{'producer':producer,'source':parent_source})
    project=Path(parent_source['path']).resolve().parents[2];lease_path=project/'work/gpu_training_owner.json'
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        lease=read_json(lease_path)
        if lease['owner']=='/root/memory_diagnosis' and lease['actual_GPU_running'] and lease['actual_owned_worker_pid']==os.getpid():break
        time.sleep(.1)
    else:raise ValueError('Actual sole-owned recovery lease handshake required')
    import sys
    require('torch' not in sys.modules,'Coherent CUDA preload before Torch required')
    from preload_nemo_linker_v1 import preload
    coherent=preload();require(coherent['actual_binding_linker']['version']==[12,9] and not coherent['cuda_initialized'],'Frozen CUDA12.9 preload')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0)
    def guard():require(not (folder/'stop.json').exists(),'Owned recovery parent stop')
    emit=lambda name,**extra:stage(folder,name,torch,**extra)
    admit=lambda name,budget:admit_stage(folder,name,budget)
    # Keep the already reviewed narrow import/bootstrap path. No loss function,
    # forward/backward, teacher, augmenter, processor frontend or Adam step runs.
    from native_tdt_minimal_import_v1 import prepare
    minimal=prepare();emit('recovery_runtime_ready',coherent=coherent,capacity=capacity,minimal=minimal)
    admit('recovery_readonly_base',900_000_000)
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply
    from asr18_token_only_carrier_v1 import load as token_carrier
    base=Path((project/'work/base_model_path.txt').read_text().strip())
    model,loader=load(base);logical=apply(model);processor=token_carrier(base)
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer,provenance,replay=bootstrap(model,processor,torch=torch,stage=emit,admit=admit)
    from bpe_broad_production_v3 import prohibit_model_transforms
    prohibit_model_transforms(model,torch)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    canonical=canonical_joint(model.joint)
    from asr18_targeted_cleaner_schedule_context_v1 import CleanerContext
    from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
    context=CleanerContext(config['planning_arguments'],config['cleaner_ancestor'],guard=guard)
    require(digest(replay)==digest(context.snapshot) and optimizer.updates==0,
            'Original history/schedule bootstrap required, never diagnostic seed')
    augmentation=MetadataCheckpointBinding(config['augmentation_identity'],guard=guard)
    def private(label):
        value=sample_memory(os.getpid(),label)
        require(value['paired_sampling_within_one_second'] and value['system_memory']['available']
                and value['worker_private_memory']['available'],'Fresh recovery private/system telemetry')
        return value['worker_private_memory']['private_bytes']
    require(private('recovery_ready')<=3_000_000_000,'Actual restore-only pre-stage private must remain<=3GB')
    write_json(folder/'worker_ready.json',{'passed':True,'producer':producer,'worker_mode':'restore_only',
        'worker_startup_and_initial_restore_seconds':time.monotonic()-began,'native_updates':0})
    phase_records=[]
    def run_phase(name,operation):
        guard();before=private(name+'_before');budget=350_000_000
        projection=phase_projection(config['entire_worker_forecast'],phase='checkpoint_restore',
            current_private_bytes=before,increment_bytes=budget)
        admit(name,budget);monitor=PhasePrivateMonitor();monitor.start();start=time.monotonic();returned=False
        try:result=operation();returned=True
        finally:
            proof=monitor.finish();proof['wall_seconds']=time.monotonic()-start
            covered=proof['private_peak_bytes']<=before+budget
            path=folder/(name+'_memory.json');write_json(path,{'before_private_bytes':before,
                'increment_bytes':budget,'projection':projection,'actual':proof,'covered':covered,
                'margin_bytes':before+budget-proof['private_peak_bytes'],'operation_returned':returned})
            phase_records.append(record(path))
        require(covered,'Separate restore350MB prospective forecast missed; no unchanged retry')
        return result,proof
    sequence=0;recovery=None
    try:
        while True:
            guard();require(time.monotonic()-began<config['maximum_GPU_child_wall_seconds'],'Finite recovery time limit')
            if not (folder/'command.json').exists():time.sleep(.1);continue
            command=read_json(folder/'command.json')
            if command['sequence']<=sequence:time.sleep(.1);continue
            require(command['sequence']==sequence+1 and type(command['nonce']) is str and len(command['nonce'])>=16,'Serial recovery command identity')
            sequence=command['sequence'];check_bindings(config)
            if command['kind']=='restore_all_deferred_checkpoints':
                require(recovery is None and command['window']['request']==config['deferred_recovery_requests'],
                        'One exact source-bound recovery chain, never native work')
                recovery=recover_all(config['deferred_recovery_requests'],model=model,optimizer=optimizer,
                    context=context,augmentation=augmentation,session=config['session'],parent_source=parent_source,
                    torch=torch,folder=folder/'actual_recoveries',run_phase=run_phase,guard=guard)
                response={'passed':True,'sequence':sequence,'nonce':command['nonce'],'recovery_report':recovery}
            elif command['kind']=='finish_coherent_native_process':
                require(recovery is not None,'Recovery cannot exit success before every checkpoint recovered')
                write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],'actual_process_exit_pending':True});break
            else:raise ValueError('No native training/idle/feature command in restore-only worker')
            write_json(folder/'response.json',response)
    finally:context.close()
    final_updates=optimizer.updates
    del context,augmentation,optimizer,model,processor;gc.collect();torch.cuda.empty_cache()
    emit('recovery_models_released')
    report={'passed':True,'producer':producer,'worker_source':parent_source,'parent_source':parent_source,
        'worker_mode':'restore_only','recovery_report':recovery,'windows':[],
        'coherent_chain_new_optimizer_updates':final_updates,'actual_native_updates_this_process':0,
        'all_captured_checkpoints_actually_recovered':True,'actual_draws_consumed':0,
        'phase_memory_records':phase_records,'bootstrap_source':record(Path(__file__)),'canonical_joint':canonical,
        'wall_seconds':time.monotonic()-began,'all_sources_unchanged':True,
        'production_enabled':False,'qualified_checkpoint':False}
    check_bindings(config);write_json(folder/'worker_completion.json',report)
