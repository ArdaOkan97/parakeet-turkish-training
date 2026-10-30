"""Readonly configuration for actual already-ACKed40 after control cleanup failure.

No normal failed-owner exit is invented.  The original actual fresh40/ACK stays
authoritative; the next resident training worker performs its ordinary restore.
"""
import argparse,copy,json,sys
from pathlib import Path
from contextlib import contextmanager
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S));sys.path.insert(0,str(P/'work/research'))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_memory_diagnosis_v2 import storage
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
from asr18_BPE9223_indexed_checkpoint_storage_v2 import IndexedOwnedFutureCheckpointStore
from asr18_BPE9223_operational_metadata_v2 import configure,read_bound
ROOT=Path(r'E:/gpu_training_scratchpad')

@contextmanager
def metadata_phase(name):
    require(name=='read_index','Readonly current indexed metadata');yield

def bound(path):
    pin=record(path);require(0<pin['bytes']<=16*1024**2,'Bounded exact metadata');return pin

def main(a):
    storage();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(ROOT) and not folder.exists(),'Fresh E output only')
    review_pin=bound(a.ROOT_source_review);review=read_json(a.ROOT_source_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_ACKed40_reserved_p3_reader_configuration_v1'
        and review['source']==record(__file__) and review['production_approval_claimed'] is False,
        'Metadata-only exact ROOT source authority')
    compatibility_pin=bound(a.compatibility);compatibility=read_json(a.compatibility)
    stopped_pin=compatibility['prior_CPU_metadata_reader_failure'];stopped=read_bound(stopped_pin)
    prior=read_bound(stopped['configuration']);identity=prior['full_run_identity']
    checked_compatibility(compatibility_pin,identity)
    require(review['operational_compatibility']==compatibility_pin,'Exact metadata scope')
    frozen={k:stopped[k] for k in ('storage_state','durable_latest','current_registered_journal',
        'pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses')}
    require(all(record(pin['path'])==pin for pin in frozen.values()),'Exact actual post-ACK40 state')
    cpu_pin=bound(a.CPU_handoff);cpu=read_json(a.CPU_handoff)
    dev_pin=bound(a.DEV_handoff);dev=read_json(a.DEV_handoff)
    c=copy.deepcopy(prior);c.update(dev['full_configuration_updates'])
    require(all(c[k]==cpu['full_configuration_updates'][k] for k in ('session','CPU_bridge_source','CPU_template_configuration','ROOT_CPU_execution_review')),
        'Exact frozen CPU session/template/bridge/authority; retain actual orchestrator8')
    sources=compatibility['operational_sources']
    c.update(source_manifest=compatibility['operational_source_manifest'],parent_driver_source=sources['parent'],
        worker_driver_source=sources['runtime'],full_owner_controller_source=sources['controller'],
        operational_compatibility=compatibility_pin,prior_CPU_metadata_reader_failure=stopped_pin,
        process_readonly_metadata_authority=compatibility['process_readonly_metadata_authority'],
        CPU_controller_source=sources['CPU_orchestrator'],
        configuration_assembly_source=record(__file__),configuration_source_read_review=review_pin)
    allowed={'source_manifest','parent_driver_source','worker_driver_source','full_owner_controller_source',
        'operational_compatibility','prior_CPU_metadata_reader_failure','process_readonly_metadata_authority','configuration_assembly_source',
        'configuration_source_read_review','DEV_execution_sources','DEV_native_boundary_sources','DEV_source_bindings',
        'ROOT_DEV_policy_review','ROOT_DEV_resource_review','ROOT_DEV_static_execution_review'}
    changed={k for k in set(c)|set(prior) if c.get(k)!=prior.get(k)}
    require(changed<=allowed and c['planned_native_updates']==401072
        and c['owner_validation_only'] is False and c['diagnostic_only'] is False,
        'Only explicit operational metadata/source derivatives; scientific schedule/guards unchanged')
    store=IndexedOwnedFutureCheckpointStore(c['operational_storage_owner'],guard=lambda:None,
        operational_compatibility=compatibility_pin)
    readonly=c['process_readonly_metadata_authority'];session=read_bound(c['session'])
    require(readonly==session['operational_metadata'] and readonly['storage_owner']==store.marker_pin
        and readonly['control_root']==c['operational_control_root'] and readonly['identity']==identity
        and readonly['current_native_resource_closure'] is None,'One original exact readonly metadata identity')
    configure(**readonly,guard=lambda:None,metadata_phase=metadata_phase)
    from asr18_BPE9223_full_run_controller_v14 import ACKed40_reserved_owner_state,checked_ACKed40_reader_resume_request
    snapshot=ACKed40_reserved_owner_state(stopped,guard=lambda:None)
    require(snapshot['next_draw_packet_sha256']==digest(read_bound(stopped['next_draw_packet'])),
        'Saved actual40 semantic nextdraw, no replay or sampler mutation')
    require(snapshot['sampler_state'] is not None and snapshot['actual_owner_invocation_count']==8
        and snapshot['native_sequence']==2 and snapshot['recovery_sequence']==2
        and snapshot['recovery_control_sequence']==4 and snapshot['proof_sequence']==6,
        'Actual cumulative counters and already-ACKed40, no rollback or reuse')
    folder.mkdir();path=folder/'configuration_pending_execution_review.json';write_json(path,c);config_pin=record(path)
    request={'kind':'ASR18_exact_ACKed40_reserved_p3_readonly_authority_resume_request_v1',
        'operational_configuration':config_pin,'operational_compatibility':compatibility_pin,
        'prior_CPU_metadata_reader_failure':stopped_pin,'owner_state':snapshot,
        'original_configuration':compatibility['original_configuration'],'run_identity_sha256':digest(identity),
        'authoritative_current_optimizer_updates':40,'no_source_optimizer_RNG_sampler_or_history_reset':True,
        'current_stopped_state_bindings':frozen,'prepared_CPU':None,'reserved_preparation':snapshot.get('reserved_preparation')}
    path=folder/'resource_resume_request.json';write_json(path,request);request_pin=record(path)
    checked_ACKed40_reader_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=lambda:None)
    import run_ASR18_BPE9223_full_run_owner_v15 as parent
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    (folder/'prospective_DEV').mkdir()
    SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=sources['parent'],native_boundary=parent.native_boundary_DEV,
        native_boundary_source=sources['parent'],native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=folder/'prospective_DEV')
    require(not any(name in sys.modules for name in ('torch','numpy','scipy')),'Light stdlib metadata process')
    require(all(record(pin['path'])==pin for pin in frozen.values()),'No run/pool/index/journal/head/witness mutation')
    report={'kind':'ASR18_actual_metadata_ACKed40_reserved_p3_reader_configuration_v1','passed':True,
        'source':record(__file__),'configuration':config_pin,'resource_resume_request':request_pin,
        'operational_compatibility':compatibility_pin,'prior_CPU_metadata_reader_failure':stopped_pin,
        'CPU_handoff':cpu_pin,'DEV_handoff':dev_pin,'current_optimizer_updates':40,
        'preserved_actual_counters':{k:snapshot[k] for k in ('actual_owner_invocation_count','native_sequence',
            'recovery_sequence','recovery_control_sequence','proof_sequence')},
        'real_prior_fresh40_and_ACK40_reused_without_repeated_standalone_recovery':True,
        'process_readonly_metadata_authority':readonly,'reserved_p3_frontier_preserved':True,
        'GPU_audio_native_or_state_mutation':False,'production_launch_approval_claimed':False,
        'prospective_SerialDEVCallbacks_constructed_without_invocation':True}
    write_json(folder/'report.json',report)
    print(json.dumps({'passed':True,'configuration':config_pin,'request':request_pin,'report':record(folder/'report.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ROOT-source-review',required=True);p.add_argument('--compatibility',required=True)
    p.add_argument('--CPU-handoff',required=True);p.add_argument('--DEV-handoff',required=True);p.add_argument('--folder',required=True)
    main(p.parse_args())
