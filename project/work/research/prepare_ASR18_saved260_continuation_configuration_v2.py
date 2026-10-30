"""Exact original-audio260 continuation metadata; no namespace mutation or GPU."""
import argparse,copy,json,os,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from asr18_BPE9223_full_run_controller_v23 import failed_audio260_owner_state,checked_failed_audio260_resume_request
from asr18_BPE9223_resource_operational_compatibility_v1 import checked_compatibility
from asr18_BPE9223_geometry_forecast_v1 import CalibratedGeometryForecast,resident_worker_forecast
from run_ASR18_BPE9223_full_run_owner_v19 import storage,sample_memory,fresh_admit
N=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1')

def main(args):
    storage();began=time.monotonic();calls=0;peak=0
    require(os.environ.get('PYTHONUTF8')=='1' and sys.flags.utf8_mode==1,'UTF8 before any metadata constructor')
    def guard():
        nonlocal calls,peak
        v=sample_memory(os.getpid(),'saved260_continuation_metadata');calls+=1;peak=max(peak,v['worker_private_memory']['private_bytes'])
        require(time.monotonic()-began<180 and peak<128000000 and v['paired_sampling_within_one_second']
            and v['system_memory']['commit_fraction']<.95 and v['system_memory']['commit_available_bytes']>2000000000,'Same128M/180s/system95/headroom')
    def load(pin):
        guard();require(record(pin['path'])==pin,'Exact pinned metadata input');return read_json(pin['path'])
    def save(path,value):
        guard();require(not path.exists(),'New immutable metadata only');write_json(path,value);return record(path)
    for _ in range(3):require(fresh_admit(128000000)['passed'],'Fresh metadata whole admission');time.sleep(.2)
    authority_pin=record(args.authorities);authorities=load(authority_pin)
    candidates=load(authorities['configuration_candidates'])
    compatibility=candidates['genuine_compatibility'];review=load(compatibility)
    failure_pin=review['prior_committed260_original_audio_failure'];failed=load(failure_pin);prior=load(failed['configuration'])
    identity=prior['full_run_identity'];checked_compatibility(compatibility,identity,guard)
    state=failed_audio260_owner_state(failure_pin,guard)
    updates=copy.deepcopy(candidates['full_configuration_candidates'])
    updates.update({key:authorities[key] for key in ('ROOT_CPU_execution_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review')})
    require(all(key in updates for key in ('session','CPU_template_configuration','ROOT_CPU_execution_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review')),'Genuine complete ROOT CPU/DEV authority handoff')
    folder=Path(args.folder).resolve();require(folder.is_relative_to(N) and not folder.exists(),'Fresh E metadata output only');folder.mkdir()
    # Exact genuinely produced p13 firstupdate metadata; referenced arrays were
    # retired by real cleanup0. Only rows geometry is used for startup forecast.
    report=load(state['current_CPU_report']);boundary=load(state['prior_boundary'])
    retained=load(boundary['preserved_metadata']);first=copy.deepcopy(retained['metadata']['update_00.json'])
    require(first['prepared_payload']==report['complete_ordered_prepared_payloads'][0]
        and retained['manifest_binding']==report['manifest'] and first['global_update']==241,
        'Exact real retained p13 first native-input JSON/full rows and completed payload binding')
    first.update(restoration_forecast_geometry_only=True,prepared_feature_arrays_adopted=False,
        actual_source_CPU_report=state['current_CPU_report'],actual_retained_metadata=boundary['preserved_metadata'],
        original_complete_payload_sha256=first['prepared_payload']['payload_sha256'])
    first_pin=save(folder/'actual_p13_first_update_geometry_for_restore_forecast.json',first)
    calibration=CalibratedGeometryForecast(**prior['forecast_calibration'],guard=guard)
    expected=resident_worker_forecast(calibration.budget(rows=first['rows'],current_private_bytes=3000000000,update_in_process=1))
    require(expected==prior['entire_worker_forecast'],'Same full resident worker forecast, no margin/cap shrink')
    config=copy.deepcopy(prior);config.update(updates)
    config.update(operational_compatibility=compatibility,source_manifest=review['operational_source_manifest'],
        full_owner_controller_source=review['operational_sources']['controller'],worker_driver_source=review['operational_sources']['runtime'],
        prior_committed260_original_audio_failure=failure_pin,original_audio260_restore_first_feature_update=first_pin,
        configuration_assembly_source=record(__file__),configuration_source_read_review=record(args.metadata_source_review))
    unchanged=('run_namespace','checkpoint_storage_namespace','full_run_identity','initial_sampler_sha256','planning_arguments',
        'augmentation_identity','qualified_source_winner','planned_native_updates','process_readonly_metadata_authority',
        'durable_policy','CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes','CPU_outer_controller_launcher_reserve_bytes',
        'merged_CPU_policy','entire_worker_forecast','CUDA_capacity_policy','control_log_policy','maximum_owner_wall_seconds')
    require(all(config[k]==prior[k] for k in unchanged),'Exact saved scientific state/namespace/schedule/storage/guard policy preserved')
    session=load(config['session']);require(session['operational_metadata']==config['process_readonly_metadata_authority']
        and session['original_audio260_prior_session']==prior['session']
        and session['operational_compatibility']==compatibility,'Exact new session current-source/stable-reader/old07 predecessor joins')
    require(load(config['ROOT_CPU_execution_review'])['accepted'] is True
        and load(config['ROOT_DEV_policy_review'])['accepted'] is True,'Only genuine ROOT authorities before final config')
    config_pin=save(folder/'configuration_pending_execution_review.json',config)
    request={'kind':'ASR18_exact_committed260_original_Opus_failure_resume_request_v1','configuration':config_pin,
        'source_manifest':config['source_manifest'],'operational_compatibility':compatibility,
        'parent_source':config['parent_driver_source'],'controller_source':config['full_owner_controller_source'],
        'run_identity_sha256':digest(identity),'current_original_audio260_failure':failure_pin,'owner_state':state,
        'reserved_preparation':state['reserved_preparation'],'prepared_CPU':None,'no_source_optimizer_RNG_sampler_reset':True}
    request_pin=save(folder/'resume_request.json',request)
    checked=checked_failed_audio260_resume_request(config_pin,{'invocation':'original_audio260_resume','resume_request':request_pin},
        request_pin,parent_source=config['parent_driver_source'],guard=guard)
    require(digest(checked)==digest(state),'Actual serialized full request rechecks exact scalar/counter/reservation authority')
    # Actual constructors and pure contract derivation, never prepared-feature
    # ownership or native/CPU producer work.
    from fastmss_BPE9223_resource_CPU_boundary_v2 import configure_session
    configure_session(session,guard)
    from fastmss_BPE9223_feature_protocol_v12 import read_bound
    numerical=load(load(config['CPU_template_configuration'])['source_contract_template']['numerical_CPU_contract'])
    from fastmss_BPE9223_cpu_common_v12 import verify_contract
    verify_contract(load(config['CPU_template_configuration'])['source_contract_template']['numerical_CPU_contract'],session,guard=guard)
    for old_pin in failed['evidence'].values():require(record(old_pin['path'])==old_pin,'Every actual saved260 authoritative/failure pin unchanged after assembly')
    result={'kind':'ASR18_actual_saved260_continuation_configuration_metadata_v1','passed':True,'source':record(__file__),
        'configuration':config_pin,'resume_request':request_pin,'ROOT_authorities':authority_pin,'operational_compatibility':compatibility,
        'unchanged_scientific_guard_fields':list(unchanged),'actual_serialized_checker_passed':True,
        'actual_CPU_boundary2_configure_and_core12_contract_passed':True,'actual_geometry_forecast_constructor_and_full_budget_equal':True,
        'optimizer_updates_preserved':260,'all_actual_counters_preserved':{k:state[k] for k in ('actual_owner_invocation_count','native_sequence','recovery_sequence','recovery_control_sequence','proof_sequence')},
        'no_GPU_audio_producer_owner_constructor_or_authoritative_run_mutation':True,'guard_calls':calls,
        'actual_current_private_sample_peak_bytes':peak,'wall_seconds':time.monotonic()-began}
    save(folder/'report.json',result);print(json.dumps({'configuration':config_pin,'request':request_pin,'report':record(folder/'report.json')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--authorities',required=True);parser.add_argument('--folder',required=True);parser.add_argument('--metadata-source-review',required=True)
    main(parser.parse_args())
