"""Read-only exact prepared60 carryforward after actual UTF8 preGPU failure."""
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
from asr18_BPE9223_operational_metadata_v2 import configure,read_bound
ROOT=Path(r'E:/gpu_training_scratchpad')
@contextmanager
def metadata_phase(name):
    require(name=='read_index','Read-only bounded metadata only');yield

def bound(path):
    pin=record(path);require(0<pin['bytes']<=16*1024**2,'Bounded exact metadata');return pin

def main(a):
    storage();folder=Path(a.folder).resolve()
    require(folder.is_relative_to(ROOT) and not folder.exists(),'Fresh E config output only')
    rp=bound(a.ROOT_source_review);review=read_json(a.ROOT_source_review)
    require(review['accepted'] is True and review['reviewer']=='/root'
        and review['scope']=='metadata_BPE9223_prepared60_UTF8_carryforward_configuration_v1'
        and review['source']==record(__file__) and review['production_approval_claimed'] is False,
        'Exact metadata-only ROOT source authority')
    cp=bound(a.compatibility);compat=read_json(a.compatibility)
    failed_pin=compat['prior_committed60_CPU_memory_failure'];failed=read_bound(failed_pin)
    latest_pin=compat['prior_prepared60_UTF8_failure'];latest=read_bound(latest_pin)
    require(latest['current_committed60_CPU_failure']==failed_pin and latest['current_ACKed_optimizer_updates']==60 and latest['failure_before_GPU_Popen'] is True and latest['new_GPU_children']==0,'Exact actual failedowner12/preparedCPU0 and no GPU update; genuineACK60 retained')
    prior=read_bound(latest['configuration']);identity=prior['full_run_identity'];checked_compatibility(cp,identity)
    require(review['operational_compatibility']==cp,'Same exact ROOT source review')
    frozen={k:latest[k] for k in ('storage_state','durable_latest','current_registered_journal',
        'pool_registry','pool_index','pool_archive_prefix','storage_operation_witnesses')}
    require(all(record(pin['path'])==pin for pin in frozen.values()),'Actual saved60/p4 frontier unchanged')
    CPU_pin=bound(a.CPU_handoff);CPU=read_json(a.CPU_handoff);DEV_pin=bound(a.DEV_handoff);DEV=read_json(a.DEV_handoff)
    allowed_CPU={'session','CPU_bridge_source','CPU_controller_source','CPU_template_configuration','ROOT_CPU_execution_review',
        'CPU_leaf_execution_delegate_source'}
    require(set(CPU['full_configuration_updates'])<=allowed_CPU,'Only declared operational CPU cohort derivatives')
    c=copy.deepcopy(prior);c.update(CPU['full_configuration_updates']);c.update(DEV['full_configuration_updates'])
    sources=compat['operational_sources'];policy=compat['merged_CPU_topology']
    c.update(source_manifest=compat['operational_source_manifest'],parent_driver_source=sources['parent'],
        worker_driver_source=sources['runtime'],full_owner_controller_source=sources['controller'],
        operational_compatibility=cp,prior_committed60_CPU_memory_failure=failed_pin,prior_prepared60_UTF8_failure=latest_pin,
        operational_entry_mode='direct_new_controller_with_frozen_parent19_services',parent_process_UTF8_required=True,
        merged_CPU_policy=policy,CPU_owned_allowance_bytes=1867108864,CPU_inner_owned_allowance_bytes=1833554432,
        CPU_outer_controller_launcher_reserve_bytes=33554432,CPU_controller_source=sources['CPU_orchestrator'],
        CPU_leaf_execution_delegate_source=compat['CPU_leaf_execution_delegate_source'],
        configuration_assembly_source=record(__file__),configuration_source_read_review=rp)
    require(policy=={'kind':'ASR18_merged_supervisor_CPU_orchestration_resource_domain_v1',
        'merged_parent_current_cap_bytes':128000000,'leaf_maximum_private_bytes':1800000000,
        'each_shim_private_reserve_bytes':33554432,'inner_including_parent_bytes':1961554432,
        'child_subtree_private_allowance_bytes':1867108864,'whole_owned_CPU_allowance_bytes':1995108864,
        'separate_CPU_orchestrator_process_exists':False,'original_global_CPU_hard_cap_bytes':2000000000},
        'Exact two-role merge; lower aggregate reservation under same2GB and unchanged1.8GB leaf')
    allowed={'source_manifest','parent_driver_source','worker_driver_source','full_owner_controller_source','operational_compatibility',
        'prior_committed60_CPU_memory_failure','prior_prepared60_UTF8_failure','operational_entry_mode','parent_process_UTF8_required','merged_CPU_policy','CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes',
        'CPU_outer_controller_launcher_reserve_bytes','configuration_assembly_source','configuration_source_read_review',
        'DEV_execution_sources','DEV_native_boundary_sources','DEV_source_bindings','ROOT_DEV_policy_review',
        'ROOT_DEV_resource_review','ROOT_DEV_static_execution_review'}|allowed_CPU
    changed={k for k in set(c)|set(prior)if c.get(k)!=prior.get(k)}
    require(changed<=allowed and c['planned_native_updates']==401072 and c['owner_validation_only'] is False
        and c['diagnostic_only'] is False and c['CPU_execution_delegate_source']==prior['CPU_execution_delegate_source'],
        'Only declared operational topology/source change; science/moments/RNG/cursors untouched')
    session=read_bound(c['session']);authority=session['operational_metadata']
    require(authority==c['process_readonly_metadata_authority']==compat['process_readonly_metadata_authority']
        ==read_bound(prior['session'])['operational_metadata'],
        'One exact stable originalcomp2 reader binding, no global configure switch')
    configure(**authority,guard=lambda:None,metadata_phase=metadata_phase)
    from asr18_BPE9223_full_run_controller_v18 import prepared60_UTF8_owner_state,checked_prepared60_UTF8_resume_request
    snapshot=prepared60_UTF8_owner_state(latest_pin,guard=lambda:None)
    require({k:snapshot[k]for k in ('actual_owner_invocation_count','native_sequence','recovery_sequence',
        'recovery_control_sequence','proof_sequence','last_window_index')}=={
        'actual_owner_invocation_count':12,'native_sequence':4,'recovery_sequence':3,
        'recovery_control_sequence':5,'proof_sequence':10,'last_window_index':2},'Exact cumulative actual60 counters')
    pointer=read_bound(latest['checkpoint_pointer'])
    require(pointer['committed_updates']==60 and digest(snapshot['sampler_state'])==pointer['sampler_state_sha256']
        and digest(read_bound(latest['current_checkpoint_capture']['next_draw_packet']))==snapshot['next_draw_packet_sha256'],
        'Exact real60 sampler and semantic nextdraw, never old40 or source0')
    folder.mkdir();out=folder/'configuration_pending_execution_review.json';write_json(out,c);config_pin=record(out)
    request={'kind':'ASR18_exact_prepared60_UTF8_preGPU_failure_resume_request_v1',
        'operational_configuration':config_pin,'operational_compatibility':cp,
        'prior_committed60_CPU_memory_failure':failed_pin,'prior_acknowledged_committed60_stop':c['prior_acknowledged_committed60_stop'],'prior_prepared60_UTF8_failure':latest_pin,'owner_state':snapshot,'current_stopped_state_bindings':frozen,
        'run_identity_sha256':digest(identity),'no_source_optimizer_RNG_sampler_or_history_reset':True,
        'prepared_CPU':snapshot['prepared_CPU'],'reserved_preparation':None}
    out=folder/'resource_resume_request.json';write_json(out,request);request_pin=record(out)
    checked_prepared60_UTF8_resume_request(config_pin,{'resource_resume_request':request_pin},request_pin,guard=lambda:None)
    import run_ASR18_BPE9223_full_run_owner_v19 as parent
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    (folder/'prospective_DEV').mkdir()
    SerialDEVCallbacks(ROOT_policy_review=c['ROOT_DEV_policy_review'],execution_sources=c['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=sources['parent'],native_boundary=parent.native_boundary_DEV,
        native_boundary_source=sources['parent'],native_boundary_sources=c['DEV_native_boundary_sources'],output_directory=folder/'prospective_DEV')
    require(all(record(pin['path'])==pin for pin in frozen.values()),'No actual run/storage/pool/journal/head mutation')
    require(not any(name in sys.modules for name in ('torch','numpy','scipy')),'Stdlib-only config source assembly')
    result={'kind':'ASR18_actual_metadata_prepared60_UTF8_carryforward_configuration_v1','passed':True,
        'source':record(__file__),'configuration':config_pin,'resource_resume_request':request_pin,'operational_compatibility':cp,
        'actual_stop':failed_pin,'actual_latest_prepared60_UTF8_failure':latest_pin,'current_optimizer_updates':60,'preserved_actual_counters':{k:snapshot[k]for k in
        ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence')},
        'CPU_handoff':CPU_pin,'DEV_handoff':DEV_pin,'process_readonly_metadata_authority':authority,
        'merged_CPU_policy':policy,'prepared_p4_frontier_and_actual_CPU0_reused':True,'all_authoritative_pins_unchanged':True,
        'GPU_audio_native_state_or_storage_operations':False,'production_launch_approval_claimed':False,
        'prospective_SerialDEVCallbacks_constructed_without_invocation':True}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'configuration':config_pin,
        'request':request_pin,'report':record(folder/'report.json')}))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ROOT-source-review',required=True);parser.add_argument('--compatibility',required=True)
    parser.add_argument('--CPU-handoff',required=True);parser.add_argument('--DEV-handoff',required=True);parser.add_argument('--folder',required=True)
    main(parser.parse_args())
