"""Inactive concrete resident-native / CURRENT-only recovery worker.

The reviewed outer owner supplies process-local storage, actual lease, source
configuration and per-operation admission. No CLI starts long production.
"""
import argparse,copy,gc,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_owned_process_closure_v2 import identity,verify
from asr18_memory_diagnosis_v2 import storage,stage,admit_stage
OWNER='/root/memory_diagnosis';LEASE=P/'work/gpu_training_owner.json'


def check_bindings(config):
    manifest=read_json(config['source_manifest']['path'])
    require(record(config['source_manifest']['path'])==config['source_manifest'],'Exact frozen source manifest')
    for pin in manifest['bindings']:require(record(pin['path'])==pin,'Frozen runtime dependency: '+pin['path'])
    require(record(__file__)==config['worker_driver_source'],'Actual isolated GPU runtime source')


def run(config_path,folder):
    storage();folder=Path(folder).resolve()
    require(folder.is_relative_to(Path(r'E:\gpu_training_scratchpad')) and folder.is_dir(),'Parent-created owned E worker directory')
    config=read_json(config_path);config['_path']=str(Path(config_path).resolve());producer=identity()
    # Truthful actual entrypoint BEFORE slow checkpoint/source scans.
    write_json(folder/'entrypoint_boot.json',{'producer':producer,'source':record(__file__),
        'configuration':record(config_path),'parent_source':config['parent_driver_source']})
    check_bindings(config)
    require(config['kind']=='ASR18_full_run_native_owner_configuration_v1' and config['production_enabled'] is False
            and config['diagnostic_only'] is False and config['worker_mode'] in ('resident_training','restore_only'),
            'Exact unarmed native/CURRENT recovery configuration')
    write_json(folder/'worker_boot_identity.json',{'producer':producer,'source':record(__file__),
        'configuration':record(config_path),'parent_source':config['parent_driver_source']})
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        lease=read_json(LEASE)
        if lease['owner']==OWNER and lease['actual_GPU_running'] and lease['actual_owned_worker_pid']==producer['pid']:break
        time.sleep(.1)
    else:raise ValueError('Actual sole-owned source/PID lease handshake')
    require('torch' not in sys.modules,'Coherent preload beforeTorch')
    from preload_nemo_linker_v1 import preload
    coherent=preload();require(coherent['actual_binding_linker']['version']==[12,9] and not coherent['cuda_initialized'],'Unchanged coherent compiler/runtime binding')
    import torch
    torch.set_num_threads(4)
    from launch_with_memory_guard_TSC_resume_v3 import validate_capacity
    capacity=validate_capacity(torch.cuda.get_device_properties(0).total_memory,8.)
    torch.cuda.set_per_process_memory_fraction(capacity['per_process_allocator_fraction'],device=0)
    whole=config['entire_worker_forecast']['entire_worker_private_forecast_bytes']
    def emit(name,**extra):
        value=stage(folder,name,torch,**extra)
        require(value['worker_private_memory']['private_bytes']<=whole
                and value['worker_private_peak_bytes_lifetime']<=whole,'Actual current AND OS-lifetime whole-worker cap')
        require(value['cuda']['peak_allocated_bytes']<=8000000000,'Unchanged decimal8GB allocator/actual peak')
        return value
    def guard():
        require(not(folder/'stop.json').exists(),'Source-owned cooperative stop; no unchanged retry')
    def admit(name,budget):guard();return admit_stage(folder,name,budget)
    from native_tdt_minimal_import_v1 import prepare
    minimal=prepare();emit('full_run_runtime_ready',coherent=coherent,capacity=capacity)
    admit('readonly_base_and_source_bootstrap',900000000)
    from asr18_token_only_carrier_v1 import load as token_carrier
    from stream_frozen_encoder_matrices_v1 import load
    from streamed_generation_device_v1 import apply
    base=Path((P/'work/base_model_path.txt').read_text().strip())
    model,loader=load(base);logical=apply(model);processor=token_carrier(base)
    teacher=({name:copy.deepcopy(getattr(model,name)).eval().requires_grad_(False)
              for name in ('encoder_projector','decoder','joint')} if config['worker_mode']=='resident_training' else None)
    from bpe_broad_streamed_bootstrap_v2 import bootstrap
    optimizer,provenance,snapshot=bootstrap(model,processor,torch=torch,stage=emit,admit=admit)
    from tdt_joint_canonical_2d_fp32_v2 import apply as canonical_joint
    provenance.update(frozen_loader=record(S/'stream_frozen_encoder_matrices_v1.py'),
        bootstrap=record(S/'bpe_broad_streamed_bootstrap_v2.py'),minimal_native=minimal,logical_device=logical,
        NEW_joint_numerical_policy=canonical_joint(model.joint),frozen_loader_policy=loader)
    from bpe_bounded_production_journal_v4 import JournalNamespace
    namespace=JournalNamespace(Path(config['journal_namespace']),identity=config['full_run_identity'],guard=guard)
    def verify_quiescence(actual_exit,saved):
        require(actual_exit==config['source_native_process0'],'Exact current native closure for journal authority')
        closed=read_json(actual_exit['path']);require(record(actual_exit['path'])==actual_exit,'Immutable native0')
        verify(closed,kind='ASR18_owned_GPU_process_completion_v1',owner=OWNER,guard=guard)
        report=read_json(closed['report']['path']);capture=config['source_checkpoint_capture']
        pointer=read_json(capture['pointer']['path'])
        require(record(closed['report']['path'])==closed['report'] and record(capture['pointer']['path'])==capture['pointer']
                and closed['worker_source']==record(__file__) and closed['parent_source']==config['parent_driver_source']
                and report['passed'] is True and report['current_checkpoint']==capture
                and report['run_identity_sha256']==digest(config['full_run_identity'])
                and pointer['journal_prefix']==saved,'Real coherent closed source journal/pointer, never idle-only or process0 alone')
        return True
    def journal_factory(saved):
        return namespace.resume(saved,destination_name='commits_branch_'+str(config['journal_resume_sequence']).zfill(6)+'.jsonl',
            actual_exit=config['source_native_process0'],verify_quiescence=verify_quiescence)
    sequence=0;began=time.monotonic();engine=None;canonical_report=None
    if config['worker_mode']=='resident_training':
        from asr18_full_run_native_owner_v3 import FullRunNativeOwner
        journal=namespace.writer(Path(config['initial_journal_path']))
        engine=FullRunNativeOwner(model,processor,teacher,optimizer,provenance,snapshot,folder,config,
            torch=torch,stage=emit,admit=admit,guard=guard,journal=journal,journal_factory=journal_factory,journal_factory_source=record(__file__))
        if config.get('restore_parent_pointer') is not None:
            emit('before_actual_current_startup_restore')
            require(emit('current_restore_startup_bound')['worker_private_memory']['private_bytes']<=3000000000,'Unchanged3GB before restore')
            engine.restore_parent(config['restore_parent_pointer'],augmentation_identity=config['augmentation_identity'])
    else:
        from asr18_full_run_current_recovery_v2 import recover_current
        canonical_report=recover_current(config['current_recovery_request'],model=model,optimizer=optimizer,bootstrap_snapshot=snapshot,
            torch=torch,stage=emit,admit=admit,guard=guard,journal_factory=journal_factory,journal_factory_source=record(__file__),
            worker_source=record(__file__),parent_source=config['parent_driver_source'],folder=folder/'actual_current_recovery')
        write_json(folder/'current_recovery_ready.json',{'passed':True,'report':canonical_report,'producer':producer})
    write_json(folder/'worker_ready.json',{'passed':True,'producer':producer,'optimizer_updates':optimizer.updates})
    try:
        while True:
            guard();require(time.monotonic()-began<config['maximum_GPU_child_wall_seconds'],'Reviewed finite owned worker time budget')
            path=folder/'command.json'
            if not path.exists():time.sleep(.1);continue
            command=read_json(path)
            if command['sequence']<=sequence:time.sleep(.1);continue
            require(command['sequence']==sequence+1 and type(command['nonce']) is str and len(command['nonce'])>=16,'Exact parent serial command nonce')
            sequence=command['sequence'];check_bindings(config)
            if command['kind']=='finish_coherent_native_process':
                require(canonical_report is not None or(engine is not None and engine.capture is not None),'No fabricated positive native boundary')
                write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],'actual_process_exit_pending':True});break
            require(engine is not None,'Recovery cannot consume native commands')
            if command['kind']=='run_prepared_window':
                complete=engine.run_window(command['window'])
            elif command['kind']=='refresh_coherent_idle':complete=engine.last_window
            else:raise ValueError('Unsupported source-bound full owner command')
            idle=engine.coherent_idle(nonce=command['nonce'],window_completion=complete)
            ip=folder/('idle_ack_'+str(sequence)+'.json');write_json(ip,idle)
            released=engine.checkpoint_storage_release_ack(nonce=command['nonce'],window_completion=complete)
            release_path=folder/('checkpoint_release_ack_'+str(sequence)+'.json');write_json(release_path,released)
            write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],
                'window_completion':complete,'idle_ack':record(ip),'checkpoint_storage_release_ack':record(release_path)})
    finally:
        if engine is not None:engine.close()
    if engine is not None:
        result={'kind':'ASR18_full_run_closed_CURRENT_native_process_report_v1','passed':True,'producer':producer,
            'worker_source':record(__file__),'parent_source':config['parent_driver_source'],'diagnostic_only':False,
            'run_identity_sha256':digest(config['full_run_identity']),'current_checkpoint':copy.deepcopy(engine.capture),
            'final_optimizer_updates':optimizer.updates,'actual_native_updates_this_process':optimizer.updates-engine.process_initial_updates,
            'all_exact_GPU_checkpoint_restores_deferred':True,'current_window_completion':engine.last_window,
            'production_enabled':False,'qualified_checkpoint':False,'wall_seconds':time.monotonic()-began}
    del engine,optimizer,teacher,model,processor;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache()
    require(torch.cuda.memory_reserved()<=3000000000,'Actual final modelrelease/endcache3GB')
    emit('full_run_models_released')
    check_bindings(config)
    if canonical_report is None:write_json(folder/'worker_completion.json',result)
    else:require(record(canonical_report['path'])==canonical_report,'SAME canonical recovery report after release')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--folder',required=True)
    a=p.parse_args();run(a.config,a.folder)
