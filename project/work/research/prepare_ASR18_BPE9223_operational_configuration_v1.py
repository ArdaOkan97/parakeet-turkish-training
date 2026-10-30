"""Metadata-only operational config/request; never creates run or draws.

Preserves real production20 and its reserved p2. Only the actual supervisor,
under a separate ROOT execution review, can freshly restore or continue it.
"""
import argparse,copy,gzip,hashlib,inspect,json,sys
from contextlib import contextmanager
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from bpe_streaming_checkpoint_semantics_v2 import decode_scalar_tree,select
from asr18_memory_diagnosis_v2 import storage
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility,checked_resource_stop
from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
from asr18_BPE9223_operational_metadata_v2 import configure,read_bound

ROOT=Path(r'E:/gpu_training_scratchpad')

def bound(path):
    pin=record(path);require(pin['bytes']<=16*1024**2,'Bounded metadata input');return pin

@contextmanager
def metadata_phase(name):
    require(name=='read_index','Readonly existing metadata only');yield

def main(a):
    storage();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(ROOT) and not folder.exists(),'One fresh E metadata output only')
    review_pin=bound(a.ROOT_source_review);review=read_json(a.ROOT_source_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_resource_operational_configuration_v1'
        and review['source']==record(__file__) and review['production_approval_claimed'] is False,
        'Exact metadata-only source review, no GPU/audio/native permission')
    compatibility_pin=bound(a.compatibility);compatibility=read_json(a.compatibility)
    original_pin=compatibility['original_configuration'];original=read_bound(original_pin)
    identity=original['full_run_identity'];checked_compatibility(compatibility_pin,identity)
    stopped=checked_resource_stop(compatibility['actual_resource_stop'],identity=identity,compatibility=compatibility_pin)
    require(review['operational_compatibility']==compatibility_pin,'Exact compatibility metadata scope')
    sources=compatibility['operational_sources']
    cpu_pin=bound(a.CPU_handoff);cpu=read_json(a.CPU_handoff)
    dev_pin=bound(a.DEV_handoff);dev=read_json(a.DEV_handoff)
    cpu_updates=cpu['full_configuration_updates'];dev_updates=dev['full_configuration_updates']
    require({'session','CPU_template_configuration','CPU_controller_source','CPU_bridge_source'}<=set(cpu_updates)
        and set(dev_updates)=={'ROOT_DEV_policy_review','ROOT_DEV_resource_review','ROOT_DEV_static_execution_review',
            'DEV_execution_sources','DEV_native_boundary_sources','DEV_source_bindings'},'Explicit exact metadata derivatives')
    c=copy.deepcopy(original);c.update(cpu_updates);c.update(dev_updates)
    c.update(source_manifest=compatibility['operational_source_manifest'],parent_driver_source=sources['parent'],
        worker_driver_source=sources['runtime'],full_owner_controller_source=sources['controller'],
        operational_compatibility=compatibility_pin,resource_native_closure=compatibility['actual_resource_stop'],
        operational_storage_owner=read_json(stopped['storage_state']['path'])['owner'],
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
        and c['owner_validation_only'] is False and c['diagnostic_only'] is False,'Original full production recipe only')
    folder.mkdir()
    config_path=folder/'configuration_pending_execution_review.json';write_json(config_path,c);config_pin=record(config_path)
    control=Path(c['operational_control_root']);store=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],
        guard=lambda:None,operational_compatibility=compatibility_pin)
    resolver=configure(storage_owner=store.marker_pin,control_root=control,identity=identity,
        operational_compatibility=compatibility_pin,current_native_resource_closure=compatibility['actual_resource_stop'],
        guard=lambda:None,metadata_phase=metadata_phase)
    with store.locked():
        state=store._load(allow_preparing_auxiliary=True)[0]
        p1=store._auxiliary('p000000001',verify_archive=True)
        manifest=store._verify_auxiliary_archive(p1)
        p2=store._auxiliary('p000000002')
        checkpoint=store._entry('p000000001',verify_files=True)
    def member(relative):
        path=str(Path(p1['workspace'])/relative)
        values=[row['file'] for row in manifest['files'] if row['file']['path']==path]
        require(len(values)==1,'One exact registered p1 metadata member: '+relative)
        return values[0]
    response=read_bound(stopped['old_response']);command=read_bound(stopped['old_command'])
    complete=read_bound(response['window_completion']);capture=complete['checkpoints'][0]
    pointer=read_bound(capture['pointer']);metadata=read_bound(pointer['checkpoint'])
    sampler=decode_scalar_tree(select(metadata['tree'],'sampler'))
    latest=read_json(Path(c['run_namespace'])/'durable/latest.json')
    from asr18_full_run_serial_DEV_9223_v3 import read_logical_durable_head
    head=read_logical_durable_head(latest['head'])['head']
    archive_pin=checkpoint['archive'];require(record(archive_pin['path'])==archive_pin,'Exact original publication archive')
    with gzip.open(archive_pin['path'],'rb') as stream:
        raw=stream.read(16*1024**2+1);require(len(raw)<=16*1024**2,'Bounded original publication metadata')
    archived=json.loads(raw)
    require(archived['pointer_binding']==capture['pointer'] and archived['pointer']==pointer,
        'Original publication archive still binds exact p1')
    pool=control/'worker_control_pool';registry=read_json(pool/'registry.json')
    pool_binding={'kind':'ASR18_exact_worker_control_pool_snapshot_v1','owner':record(pool/'owner.json'),
        'registry':record(pool/'registry.json'),'index':record(pool/'index.sqlite'),'archive_prefix':record(pool/'archives.bin'),
        'archive_bytes':registry['archive_bytes'],'archive_count':registry['archive_count'],
        'reservation_count':registry['reservation_count'],'live_folders':[x['folder'] for x in registry['live']],
        'owner_folders':registry['owner_folders']}
    prior=read_json(Path(p2['workspace'])/'prior_window_boundary.json')
    owner_state={'current_capture':capture,'current_completion':response['window_completion'],'current_response':response,
        'current_release':response['checkpoint_storage_release_ack'],'current_recovery_report':None,'current_ack':None,
        'current_coverage':None,'current_packet':command['window'],
        'current_CPU0':member('CPU/actual_owned_process0.json'),'current_CPU_report':member('CPU/payload/report.json'),
        'current_bridge':prior['bridge'],'current_cleanup0':prior['owned_cleanup_actual_closed'],
        'current_cleanup':prior['owned_cleanup_receipt'],'current_workspace':command['window']['auxiliary_workspace'],
        'current_original_plan':command['window']['coordinator_plan'],'current_intent':command['window']['checkpoint_intent'],
        'closed_native':compatibility['actual_resource_stop'],'closed_recovery':None,
        'current_cadence':archived['actual_live_expectation']['publication_cadence_evidence'],
        'native_sequence':1,'recovery_sequence':0,'last_window_index':0,'actual_owner_invocation_count':1,
        'sampler_state':sampler,'lifecycle_state':head['lifecycle'],'durable_head':latest['head'],'durable_owner':latest['owner'],
        'storage_owner':store.marker_pin,'storage_catalog':state['catalog'],'journal_prefix':pointer['journal_prefix'],
        'worker_control_pool_state':record(pool/'registry.json'),'worker_control_pool_binding':pool_binding,
        'next_draw_packet_sha256':digest(read_bound(capture['next_draw_packet'])),
        'next_draw_packet_record_sha256':capture['next_draw_packet']['sha256']}
    from asr18_BPE9223_full_run_controller_v5 import _original_proof_sequence,checked_resource_resume_request
    witnesses=record(control/'storage_operation_witnesses.bin')
    owner_state['proof_sequence']=_original_proof_sequence(witnesses,lambda:None)
    request={'kind':'ASR18_exact_coherent_resource_stop_resume_request_v1','original_configuration':original_pin,
        'operational_configuration':config_pin,'operational_compatibility':compatibility_pin,
        'current_native_resource_closure':compatibility['actual_resource_stop'],'run_identity_sha256':digest(identity),
        'no_source_optimizer_RNG_sampler_or_history_reset':True,'prior_operation_witness_stream':witnesses,
        'owner_state':owner_state,'reserved_preparation':{'intent':state['pending_publication'],'workspace':p2,'plan':p2['plan']}}
    request_path=folder/'resource_resume_request.json';write_json(request_path,request);request_pin=record(request_path)
    checked_resource_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=lambda:None)
    import run_ASR18_BPE9223_full_run_owner_v5 as parent
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    callbacks=SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=sources['parent'],native_boundary=parent.native_boundary_DEV,
        native_boundary_source=sources['parent'],native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=folder/'prospective_DEV')
    require(not any(name in sys.modules for name in ('torch','numpy','scipy')),'Metadata parent remains numerical-library free')
    require(record(stopped['storage_state']['path'])==stopped['storage_state']
        and record(witnesses['path'])==witnesses,'No original storage/journal/counter mutation')
    report={'kind':'ASR18_actual_metadata_operational_configuration_and_resource_request_v1','passed':True,
        'source':record(__file__),'configuration':config_pin,'resource_resume_request':request_pin,
        'operational_compatibility':compatibility_pin,'CPU_handoff':cpu_pin,'DEV_handoff':dev_pin,
        'original_genesis_configuration':original_pin,'current_optimizer_updates':20,'first_GPU_fresh_p1_required':True,
        'original_reserved_p2_preserved':True,'proof_counter_preserved':owner_state['proof_sequence'],
        'prospective_SerialDEVCallbacks_constructed_without_invocation':True,
        'new_source_or_RNG_sampler_reset':False,'GPU_audio_NATIVE_jobs':0,'production_launch_approval_claimed':False}
    write_json(folder/'report.json',report)
    print(json.dumps({'passed':True,'configuration':config_pin,'request':request_pin,'report':record(folder/'report.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-source-review',required=True);p.add_argument('--compatibility',required=True)
    p.add_argument('--CPU-handoff',required=True);p.add_argument('--DEV-handoff',required=True);p.add_argument('--folder',required=True)
    main(p.parse_args())
