"""Metadata-only completion request for the genuine saved production40.

Prior durable20 is protocol history only.  No model, source restore, draw,
publication, cleanup or GPU job occurs in this assembler.
"""
import argparse,copy,json,sys
from contextlib import contextmanager
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_memory_diagnosis_v2 import storage
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
from asr18_BPE9223_persistence_prefix_v8 import checked_cleanup_join_failure
from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
from asr18_BPE9223_operational_metadata_v2 import configure,read_bound

ROOT=Path(r'E:/gpu_training_scratchpad')

def bound(path):
    pin=record(path);require(0<pin['bytes']<=16*1024**2,'Bounded metadata input');return pin

@contextmanager
def metadata_phase(name):
    require(name=='read_index','Readonly existing metadata only');yield

def main(a):
    storage();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(ROOT) and not folder.exists(),'One fresh E metadata output only')
    review_pin=bound(a.ROOT_source_review);review=read_json(a.ROOT_source_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_pending_completed40_configuration_v1'
        and review['source']==record(__file__) and review['production_approval_claimed'] is False,
        'Exact metadata-only source review, no GPU/audio/native permission')
    compatibility_pin=bound(a.compatibility);compatibility=read_json(a.compatibility)
    original_pin=compatibility['original_configuration'];original=read_bound(original_pin)
    identity=original['full_run_identity'];checked_compatibility(compatibility_pin,identity)
    failure_pin=compatibility['prior_cleanup_join_failure']
    failed=checked_cleanup_join_failure(failure_pin,identity=identity,operational_compatibility=compatibility_pin)
    require(review['operational_compatibility']==compatibility_pin,'Exact reviewed compatibility')
    frozen_keys=('storage_state','durable_latest','current_registered_journal','pool_registry')
    require(all(record(failed[k]['path'])==failed[k] for k in frozen_keys),
        'Exact stopped40 state/pool/whole journal/head before metadata assembly')
    cpu_pin=bound(a.CPU_handoff);cpu=read_json(a.CPU_handoff)
    dev_pin=bound(a.DEV_handoff);dev=read_json(a.DEV_handoff)
    helper_pin=bound(a.reconciliation);helper=read_json(a.reconciliation)
    require(helper['actual_tool_exit_code']==helper['actual_exec_tool_exit_code']==0
        and helper['all_captured_identities_dead'] is True
        and helper['physical_cleanup_performed_in_this_invocation'] is False,
        'Genuine helper2 closure0; no old cleanup process0 invented')
    require(helper['source']==compatibility['operational_sources']['cleanup_reconciliation']
        and helper['observer_source']==compatibility['CPU_execution_delegate_source'],
        'Actual helper/observer identities')
    cpu_updates=cpu['full_configuration_updates'];dev_updates=dev['full_configuration_updates']
    require({'session','CPU_template_configuration','CPU_controller_source','CPU_bridge_source'}<=set(cpu_updates)
        and set(dev_updates)=={'ROOT_DEV_policy_review','ROOT_DEV_resource_review','ROOT_DEV_static_execution_review',
            'DEV_execution_sources','DEV_native_boundary_sources','DEV_source_bindings'},'Explicit exact derivatives')
    sources=compatibility['operational_sources'];c=copy.deepcopy(original);c.update(cpu_updates);c.update(dev_updates)
    c.update(source_manifest=compatibility['operational_source_manifest'],parent_driver_source=sources['parent'],
        worker_driver_source=sources['runtime'],full_owner_controller_source=sources['controller'],
        operational_compatibility=compatibility_pin,resource_native_closure=compatibility['actual_resource_stop'],
        prior_operational_failure=compatibility['prior_operational_failure'],
        prior_acknowledged_resource_stop=compatibility['prior_acknowledged_resource_stop'],
        CPU_execution_delegate_source=compatibility['CPU_execution_delegate_source'],
        CPU_leaf_execution_delegate_source=compatibility['CPU_leaf_execution_delegate_source'],
        CPU_controller_source=sources['CPU_orchestrator'],
        prior_CPU_reference_failure=compatibility['prior_CPU_reference_failure'],
        current_cleanup_join_failure=failure_pin,cleanup_reconciliation_process0=helper['actual_process0'],
        preserved_failed_metadata_controls=[compatibility['prior_CPU_reference_failure'],
            bound(a.prior_failed_reconciliation)],
        operational_storage_owner=read_json(failed['storage_state']['path'])['owner'],
        operational_control_root=str(Path(c['run_namespace'])/'control'),
        original_genesis_configuration=original_pin,configuration_assembly_source=record(__file__),
        configuration_source_read_review=review_pin)
    for key in ('full_run_identity','planning_arguments','planned_native_updates','initial_sampler_sha256',
        'augmentation_identity','row_migration_plan','entire_worker_forecast','final_recovery_worker_forecast',
        'CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes','CPU_outer_controller_launcher_reserve_bytes',
        'checkpoint_storage_namespace','checkpoint_storage_policy','durable_policy','qualified_source_winner',
        'control_log_policy','CUDA_capacity_policy','maximum_owner_wall_seconds','maximum_GPU_child_wall_seconds',
        'collect_GPU_timeline','owner_validation_only','diagnostic_only'):
        require(c[key]==original[key] and digest(c[key])==digest(original[key]),'Unchanged science/guard/source lineage: '+key)
    require(c['planned_native_updates']==401072 and c['collect_GPU_timeline'] is False
        and c['owner_validation_only'] is False and c['diagnostic_only'] is False,'Original full production recipe')
    control=Path(c['operational_control_root'])
    store=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],guard=lambda:None,
        operational_compatibility=compatibility_pin)
    configure(storage_owner=store.marker_pin,control_root=control,identity=identity,
        operational_compatibility=compatibility_pin,current_native_resource_closure=None,
        guard=lambda:None,metadata_phase=metadata_phase)
    with store.locked():state=store._load(allow_preparing_auxiliary=True)[0]
    require(state['latest']=='p000000001' and state['pending_publication']['id']=='p000000002'
        and state['pending_retirement'] is None,'Exact pending p2; no state mutation or source rollback')
    old=read_bound(failed['resume_request']);owner_state=copy.deepcopy(old['owner_state'])
    launch=read_json(Path(failure_pin['path']).parent/'launch.json')
    require(launch['actual_owner_invocation_count']==5,'Preserve actual owner5 cumulative count')
    owner_state['actual_owner_invocation_count']=5;owner_state['native_sequence']=2
    owner_state['closed_native']=owner_state['closed_recovery']
    pool=control/'worker_control_pool';registry=read_json(pool/'registry.json')
    owner_state['worker_control_pool_state']=failed['pool_registry']
    owner_state['worker_control_pool_binding']={'kind':'ASR18_exact_worker_control_pool_snapshot_v1',
        'owner':record(pool/'owner.json'),'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),
        'archive_prefix':record(pool/'archives.bin'),'archive_bytes':registry['archive_bytes'],
        'archive_count':registry['archive_count'],'reservation_count':registry['reservation_count'],
        'live_folders':[row['folder'] for row in registry['live']],'owner_folders':registry['owner_folders']}
    from asr18_BPE9223_full_run_controller_v10 import _original_proof_sequence,checked_pending40_resume_request
    witnesses=record(control/'storage_operation_witnesses.bin')
    require(owner_state['proof_sequence']==_original_proof_sequence(witnesses,lambda:None),
        'Actual witness sequence preserved, never reset to protocol snapshot')
    response=read_bound(failed['actual_native_response']);completion=read_bound(failed['actual_completed_native_window'])
    command=read_json(Path(failed['actual_native_response']['path']).parent/'command.json')
    completed={'response':response,'completion':failed['actual_completed_native_window'],'packet':command['window'],
        'capture':completion['checkpoints'][0],'release':response['checkpoint_storage_release_ack'],
        'CPU0':failed['actual_CPU_controller0'],'CPU_report':failed['actual_CPU_report'],
        'bridge':read_bound(failed['actual_CPU_report'])['bridge'],
        'cleanup_reconciliation_process0':helper['actual_process0'],'cleanup_reconciliation_report':helper['report']}
    folder.mkdir();config_path=folder/'configuration_pending_execution_review.json';write_json(config_path,c);config_pin=record(config_path)
    request={'kind':'ASR18_exact_completed40_pending_publication_cleanup_join_resume_request_v1',
        'original_configuration':original_pin,'operational_configuration':config_pin,
        'operational_compatibility':compatibility_pin,'current_cleanup_join_failure':failure_pin,
        'authoritative_current_optimizer_updates':40,'prior_durable_protocol_optimizer_updates':20,
        'no_source_optimizer_RNG_sampler_or_history_reset':True,'prior_durable_owner_state':owner_state,
        'completed40':completed,'cleanup_reconciliation_process0':helper['actual_process0'],
        'cleanup_reconciliation_report':helper['report'],'prior_operation_witness_stream':witnesses,
        'current_stopped_state_bindings':{k:failed[k] for k in frozen_keys},'run_identity_sha256':digest(identity)}
    request_path=folder/'resource_resume_request.json';write_json(request_path,request);request_pin=record(request_path)
    checked=checked_pending40_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=lambda:None)
    require(checked['pending40']==completed,'Complete exact pending40 checked before execution')
    import run_ASR18_BPE9223_full_run_owner_v11 as parent
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    (folder/'prospective_DEV').mkdir(exist_ok=False)
    SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=sources['parent'],native_boundary=parent.native_boundary_DEV,
        native_boundary_source=sources['parent'],native_boundary_sources=c['DEV_native_boundary_sources'],
        output_directory=folder/'prospective_DEV')
    require(not any(name in sys.modules for name in ('torch','numpy','scipy')),'Numerical-library-free metadata process')
    require(all(record(failed[k]['path'])==failed[k] for k in frozen_keys)
        and record(witnesses['path'])==witnesses,'No original state/journal/head/pool/counter mutation')
    report={'kind':'ASR18_actual_metadata_pending_completed40_configuration_v1','passed':True,'source':record(__file__),
        'configuration':config_pin,'resource_resume_request':request_pin,'operational_compatibility':compatibility_pin,
        'CPU_handoff':cpu_pin,'DEV_handoff':dev_pin,'cleanup_reconciliation_actual_TOOL0':helper_pin,
        'original_genesis_configuration':original_pin,'current_optimizer_updates':40,'prior_durable_protocol_updates':20,
        'first_GPU_fresh_exact40_required':True,'publication_prefixACK_before_new_CPU_or_native_draw':True,
        'prior_protocol20_restore_or_replay_permitted':False,'proof_counter_preserved':owner_state['proof_sequence'],
        'prospective_SerialDEVCallbacks_constructed_without_invocation':True,
        'new_source_or_RNG_sampler_reset':False,'GPU_audio_NATIVE_jobs':0,'production_launch_approval_claimed':False}
    write_json(folder/'report.json',report)
    print(json.dumps({'passed':True,'configuration':config_pin,'request':request_pin,'report':record(folder/'report.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-source-review',required=True);p.add_argument('--compatibility',required=True)
    p.add_argument('--CPU-handoff',required=True);p.add_argument('--DEV-handoff',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--reconciliation',required=True);p.add_argument('--prior-failed-reconciliation',required=True)
    main(p.parse_args())
