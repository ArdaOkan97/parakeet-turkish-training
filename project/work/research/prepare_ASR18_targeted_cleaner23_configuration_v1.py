"""Metadata-only exact new cleaner23 plan/session/config, never audio/GPU."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_row_serial_cpu_v6 import environment,resource_guard,verify_contract,verify_raw_review
from fastmss_phase_aware_plan_bridge_v5 import read_bound,PURPOSE,SESSION_KIND
from asr18_targeted_cleaner_schedule_context_v1 import CleanerContext
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'


def assemble(*,source_manifest,source_review,CPU_resource_review,folder):
    environment();guard=resource_guard
    require(folder.is_relative_to(A) and not folder.exists(),'Fresh E-only metadata namespace')
    sources=read_bound(source_manifest,guard);review=read_bound(source_review,guard)
    require(review['accepted'] is True and review['source_manifest']==source_manifest
            and review['scope']=='source_read_review_for_future_targeted_cleaner23_diagnostic'
            and review['production_approval_claimed'] is False,'Exact new cleaner caller source scope')
    for pin in sources['bindings']:require(record(pin['path'])==pin,'Frozen source closure changed')
    own=record(__file__);require(own in sources['bindings'],'Exact reviewed metadata assembler')
    template=read_bound(sources['configuration_template'],guard)
    require(template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False,
            'No armed template or implicit launch')
    CPU=read_bound(CPU_resource_review,guard)
    require(CPU['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU['reviewer']=='/root'
            and CPU['accepted'] is True and CPU['scope']=='CPU_preparation_only' and CPU['production_enabled'] is False
            and CPU['CPU_source_manifest']==sources['CPU_owner_bundle']
            and CPU['outer_parent_source']==template['parent_driver_source'],'Exact new outer-parent CPU resource binding')
    resource=copy.deepcopy(CPU['resource_review']);raw=copy.deepcopy(CPU['raw_lifetime_review']);verify_raw_review(raw)
    require(resource['passed'] is True and resource['maximum_simultaneous_children']==1
            and resource['whole_owned_private_allowance_bytes']+resource['outer_controller_launcher_reserve_bytes']==1_934_217_728,
            'Unchanged CPU6/controller/all-shim family, serial leaf and raw margins')
    conditional=read_bound(template['cleaner_conditional_CPU_resource_review'],guard)
    require(conditional['accepted'] is True and conditional['source']==record(S/'fastmss_row_serial_cpu_v6.py')
            and conditional['unchanged_CPU_family_allowance_bytes']==1_934_217_728,'Exact cleaner header/raw conditional review')
    context=CleanerContext(template['planning_arguments'],template['cleaner_ancestor'],guard=guard)
    try:
        schedule=context.new_schedule();initial=copy.deepcopy(schedule.state_dict());plan=context.plan(schedule,maximum_updates=1)
        require(digest(schedule.state_dict())==digest(initial)==plan['initial_sampler_sha256']
                and schedule.committed_updates()==22 and schedule.cleaner.draws==0,
                'No original or diagnostic source/replay draw consumed by planning')
        rows=plan['updates'][0]['rows'];require(len(rows)==4,'Two real cleaner TRAIN rows plus exact next carried pair')
        geometry=[];forecast_rows=[]
        for row in rows:
            role=row['augmentation_role'];ids=row['tr_metadata_labels'] if role=='student_tr' else row['clean_target_ids']
            t=row['encoder_frames'];frames=8*t+1;samples=8*t*160+159
            geometry.append({'source_id':row['source_id'],'language':row['language'],'role':role,
                'cleaner_original_ordinal':row.get('cleaner_original_ordinal'),'T':t,'U':len(ids),
                'conditional_physical_frames_upper':frames,'conditional_N_upper':samples,
                'actual_header_N_T_required_before_allocation':True})
            forecast_rows.append({'role':role,'encoder_frames':t,'full_target_ids':ids,'samples':samples,
                'input_features':{'shape':[1,frames,128]}})
        from asr18_calibrated_geometry_forecast_v6 import CalibratedGeometryForecast
        calibration=CalibratedGeometryForecast(**template['forecast_calibration'],guard=guard)
        candidate=calibration.budget(rows=forecast_rows,current_private_bytes=3_000_000_000,update_in_process=1)
    finally:context.close()
    folder.mkdir();plan_path=folder/'targeted_cleaner23_plan.json';write_json(plan_path,plan)
    contract=read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v6.json')
    require(contract['kind']=='FastMSS_row_serial_CPU_preparation_contract_v6' and contract['TEMPLATE_NOT_LAUNCHABLE'] is True,
            'Preserved exact CPU6 contract template')
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(template['planning_arguments']['DATA_evidence']),
        outer_parent_source=template['parent_driver_source'],resource_review=resource,raw_lifetime_review=raw,
        actual_ROOT_CPU_resource_review=CPU_resource_review)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract)
    session={'kind':SESSION_KIND,'purpose':PURPOSE,'production_enabled':False,'diagnostic_only':True,
        'maximum_windows':2,'maximum_native_updates':40,'initial_optimizer_updates':22,
        'initial_sampler_sha256':plan['initial_sampler_sha256'],'ancestor':template['cleaner_ancestor'],
        'coordinator_source':record(S/'bpe_production_window_coordinator_v4.py'),
        'augmentation_identity':template['augmentation_identity'],'scientific_source_state':template['planning_arguments']['snapshot_binding'],
        'preserve_original_all207_threeAdam_allRNG_and_replay_lineage':True,
        'DATA_evidence_sha256':digest(template['planning_arguments']['DATA_evidence']),
        'CPU_preparation_contract':record(cp),'configuration_assembler_source':own,
        'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0,
        'targeted_cleaner_diagnostic_only':True,'production_phase_advance_claimed':False}
    sp=folder/'targeted_cleaner23_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    config={**template,'kind':'ASR18_targeted_cleaner23_diagnostic_config_v1','TEMPLATE_NOT_LAUNCHABLE':False,
        'finite_diagnostic_execution_enabled':True,'source_manifest':source_manifest,'session':record(sp),
        'initial_cleaner_diagnostic_sampler_sha256':plan['initial_sampler_sha256'],'provisional_cleaner_plan':record(plan_path),
        'CPU_owned_allowance_bytes':1_934_217_728,'CPU_inner_owned_allowance_bytes':resource['whole_owned_private_allowance_bytes'],
        'CPU_outer_controller_launcher_reserve_bytes':resource['outer_controller_launcher_reserve_bytes'],
        'configuration_assembly_source':own,'configuration_source_read_review':source_review,
        'actual_ROOT_CPU_resource_review':CPU_resource_review,'raw_lifetime_review':raw,
        'exact_ROOT_execution_configuration_review_still_required':True,
        'CPU_preparation_contract_and_whole_family_allowance_require_separate_actual_review':False}
    cfg=folder/'configuration_pending_execution_review.json';write_json(cfg,config)
    from fastmss_owned_process_closure_v2 import identity
    result={'kind':'ASR18_targeted_cleaner23_actual_metadata_configuration_v1','passed':True,'producer':identity(),
        'worker_source':own,'parent_source':own,'configuration':record(cfg),'session':record(sp),'CPU_contract':record(cp),
        'provisional_cleaner_plan':record(plan_path),'source_manifest':source_manifest,'source_review':source_review,
        'CPU_resource_review':CPU_resource_review,'cleaner_conditional_CPU_review':template['cleaner_conditional_CPU_resource_review'],
        'cleaner_ancestor':template['cleaner_ancestor'],'declared_conditional_geometry':geometry,'same_unchanged_native_forecast':candidate,
        'actual_CPU_frontend_or_GPU_jobs_launched':0,'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0,
        'shared_CACHE_created':False,'production_enabled':False,'production_phase_advance_claimed':False,
        'ROOT_exact_configuration_and_execution_review_still_required':True}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'configuration':result['configuration']}),flush=True)
    return result


def pin(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source-manifest','source-review','CPU-resource-review','folder'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    assemble(source_manifest=pin(args.source_manifest),source_review=pin(args.source_review),
        CPU_resource_review=pin(args.CPU_resource_review),folder=Path(args.folder).resolve())
