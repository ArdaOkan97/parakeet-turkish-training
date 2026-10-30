"""Exact saved ACK260 metadata assembly; never creates an owner or a producer."""
import argparse,copy,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_full_run_controller_v24 import ACKed260_owner_state,checked_ACKed260_resume_request,FullOwner
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
import run_ASR18_BPE9223_full_run_owner_v20 as parent
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')

def main(args):
    parent.storage();began=time.monotonic();calls=0;peak=0
    require(os.environ.get('PYTHONUTF8')=='1' and sys.flags.utf8_mode==1,'UTF8 at process entry')
    def guard():
        nonlocal calls,peak
        v=parent.sample_memory(os.getpid(),'singlepass_configuration_metadata');calls+=1
        peak=max(peak,v['worker_private_memory']['private_bytes'])
        require(time.monotonic()-began<180 and peak<128000000 and v['paired_sampling_within_one_second']
            and v['system_memory']['commit_fraction']<.95 and v['system_memory']['commit_available_bytes']>2000000000,
            'Same128M/180s/System95/2GB headroom metadata guard')
    def load(pin):
        guard();require(record(pin['path'])==pin,'Exact actual input');return read_json(pin['path'])
    def save(path,value):
        guard();require(not path.exists(),'New immutable E metadata only');write_json(path,value);return record(path)
    for _ in range(3):require(parent.fresh_admit(128000000)['passed'],'Fresh whole metadata admission');time.sleep(.2)
    comp_pin=record(N/'pipeline_overlap_operational_bundle_v1/ROOT_compatibility_v1.json');comp=load(comp_pin)
    failure_pin=comp['prior_ACKed260_metadata_path_failure'];failed=load(failure_pin);prior=load(failed['configuration'])
    state=ACKed260_owner_state(failure_pin,guard);checked_compatibility(comp_pin,prior['full_run_identity'],guard)
    dev_pin=record(args.DEV);dev=load(dev_pin)
    cpu_pin=record(N/'pipeline_overlap_source_v1/ROOT_singlepass_CPU_source_and_actual_output_review_v2.json')
    cpu=load(cpu_pin);require(cpu['accepted'] is True and cpu['reviewer']=='/root','Genuine ROOT CPU transport authority')
    review_pin=record(args.metadata_source_review);review=load(review_pin)
    require(review['accepted'] is True and review['reviewer']=='/root' and review['source']==record(__file__),
        'Exact independently reviewed metadata assembler source')
    config=copy.deepcopy(prior);config.update(copy.deepcopy(dev['full_configuration_updates']))
    op=comp['operational_sources']
    config.update(source_manifest=comp['operational_source_manifest'],operational_compatibility=comp_pin,
        full_owner_controller_source=op['controller'],parent_driver_source=op['parent'],worker_driver_source=op['runtime'],
        operational_entry_mode=comp['operational_entry_mode'],prior_ACKed260_metadata_path_failure=failure_pin,
        prior_acknowledged260_preparepath_stop=failure_pin,one_slot_CPU_prefetch_policy=comp['one_slot_CPU_prefetch_policy'],
        singlepass_CPU_worker_source=op['singlepass_CPU_worker'],singlepass_CPU_protocol_source=op['singlepass_CPU_protocol'],
        ROOT_singlepass_CPU_review=cpu_pin,configuration_assembly_source=record(__file__),configuration_source_read_review=review_pin)
    unchanged=('run_namespace','checkpoint_storage_namespace','full_run_identity','initial_sampler_sha256','planning_arguments',
        'augmentation_identity','qualified_source_winner','planned_native_updates','process_readonly_metadata_authority',
        'durable_policy','CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes','CPU_outer_controller_launcher_reserve_bytes',
        'merged_CPU_policy','entire_worker_forecast','CUDA_capacity_policy','control_log_policy','maximum_owner_wall_seconds',
        'session','CPU_template_configuration','ROOT_CPU_execution_review','CPU_controller_source','CPU_leaf_execution_delegate_source')
    require(all(config[k]==prior[k] for k in unchanged),'Exact scientific/state/reader/backend/forecast/guards unchanged')
    parent.check_bindings(config)
    folder=Path(args.folder).resolve();require(folder.is_relative_to(N) and not folder.exists(),'Fresh E configuration folder');folder.mkdir()
    config_pin=save(folder/'configuration_pending_execution_review.json',config)
    request={'kind':'ASR18_exact_ACKed260_singlepass_owner_resume_request_v1','configuration':config_pin,
        'source_manifest':config['source_manifest'],'operational_compatibility':comp_pin,
        'parent_source':config['parent_driver_source'],'controller_source':config['full_owner_controller_source'],
        'run_identity_sha256':digest(config['full_run_identity']),'prior_ACKed260_metadata_path_failure':failure_pin,
        'owner_state':state,'reserved_preparation':state['reserved_preparation'],'prepared_CPU':None,
        'no_source_optimizer_RNG_sampler_reset':True}
    request_pin=save(folder/'resume_request.json',request)
    checked=checked_ACKed260_resume_request(config_pin,{'invocation':'ACKed260_singlepass_resume','resume_request':request_pin},
        request_pin,parent_source=config['parent_driver_source'],guard=guard)
    require(digest(checked)==digest(state),'Real serialized request/snapshot/counters equality')
    from asr18_BPE9223_owned_GPU_runtime_v27 import verify_closed_journal_source_authority
    saved=load(state['current_capture']['pointer'])['journal_prefix']
    verify_closed_journal_source_authority(dict(config,source_native_process0=state['closed_native'],
        source_checkpoint_capture=state['current_capture']),state['closed_native'],saved,guard=guard)
    view=SimpleNamespace(config=config,parent=SimpleNamespace(check=guard),root=Path(config['run_namespace']),
        CPU_execution_delegate_source=config['CPU_execution_delegate_source'],current_capture=state['current_capture'])
    reserved=state['reserved_preparation'];plan=reserved['plan']
    window=load(plan);index=state['last_window_index']+1
    cpu_request=FullOwner._singlepass_request(view,plan,index,index%2)
    cpu_request_pin=save(folder/'prospective_p14_singlepass_request.json',cpu_request)
    from fastmss_singlepass_CPU_v1 import validate_request
    _,_,_,_,actual_plan,rows=validate_request(cpu_request_pin,guard)
    require(actual_plan==window and len(rows)==20 and sum(len(x['rows'])for x in rows)==320,'Real p14 plan/full320rows metadata joins')
    from asr18_full_run_serial_DEV_9223_v3 import SerialDEVCallbacks
    prospective=folder/'prospective_DEV';prospective.mkdir()
    SerialDEVCallbacks(ROOT_policy_review=config['ROOT_DEV_policy_review'],execution_sources=config['DEV_execution_sources'],
        execute_owned=parent.execute_owned_DEV,execute_owned_source=config['parent_driver_source'],
        native_boundary=parent.native_boundary_DEV,native_boundary_source=config['parent_driver_source'],
        native_boundary_sources=config['DEV_native_boundary_sources'],output_directory=prospective)
    for pin in failed['frozen_current_state_pins'].values():require(record(pin['path'])==pin,'No authoritative state/head/journal/pool change')
    result={'kind':'ASR18_actual_ACKed260_singlepass_configuration_metadata_v1','passed':True,'source':record(__file__),
        'configuration':config_pin,'resume_request':request_pin,'DEV_authorities':dev_pin,'operational_compatibility':comp_pin,
        'ROOT_singlepass_CPU_review':cpu_pin,'prospective_p14_request':cpu_request_pin,'unchanged_fields':list(unchanged),
        'actual_serialized_checker_passed':True,'actual_historical_fresh260_journal_source_gate_passed':True,
        'actual_singlepass_request_full320rows_metadata_validation_passed':True,'actual_DEV_callback_constructor_passed':True,
        'actual_source_manifest_bindings_checked':True,'preserved_updates':260,
        'counters':{k:state[k]for k in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence')},
        'owner_ctor_CPU_audio_GPU_ACK_or_authoritative_run_mutation':False,'external_process0_or_joined_guard_claimed':False,
        'guard_calls':calls,'sampled_private_peak_bytes':peak,'elapsed_seconds':time.monotonic()-began}
    report_pin=save(folder/'report.json',result);print(json.dumps({'configuration':config_pin,'request':request_pin,'report':report_pin}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--DEV',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--metadata-source-review',required=True);main(p.parse_args())
