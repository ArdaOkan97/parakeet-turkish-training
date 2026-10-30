"""Metadata-only assembly of two one-update diagnostic continuation windows.

The actual20 checkpoint is immutable scientific ancestry, never a production
seed. Planning clones the actual saved sampler and advances only private state.
No PCM/frontend/model/GPU or owned checkpoint namespace is created here.
"""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_row_serial_cpu_v6 import environment,resource_guard,verify_contract,verify_raw_review
from fastmss_phase_aware_plan_bridge_v5 import read_bound,PURPOSE,SESSION_KIND
from bpe_final_DATA_schedule_context_v2 import ScheduleContext,actual_saved_schedule,normalize_schedule_state
from asr18_verified_pipeline_ancestor_v1 import verify_ancestor
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'


def assemble(*,source_manifest,source_review,CPU_resource_review,storage_policy,folder):
    environment();guard=resource_guard
    require(folder.is_relative_to(A) and not folder.exists(),'Fresh E-only metadata output')
    sources=read_bound(source_manifest,guard);review=read_bound(source_review,guard)
    require(review['accepted'] is True and review['source_manifest']==source_manifest
            and review['scope']=='source_read_review_for_future_checkpoint20_continuation21_22_diagnostic'
            and review['production_approval_claimed'] is False,'Exact new continuation caller source review')
    for pin in sources['bindings']:require(record(pin['path'])==pin,'Reviewed source closure changed')
    own=record(__file__);require(own in sources['bindings'],'Exact reviewed assembler source')
    template=read_bound(sources['configuration_template'],guard)
    require(template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False,
            'Only unarmed source template accepted')
    CPU=read_bound(CPU_resource_review,guard)
    require(CPU['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU['reviewer']=='/root'
            and CPU['accepted'] is True and CPU['scope']=='CPU_preparation_only' and CPU['production_enabled'] is False
            and CPU['CPU_source_manifest']==sources['CPU_owner_bundle']
            and CPU['outer_parent_source']==template['parent_driver_source'],
            'Unchanged numerical CPU scope must bind the genuine new outer parent')
    resource=copy.deepcopy(CPU['resource_review']);raw=copy.deepcopy(CPU['raw_lifetime_review']);verify_raw_review(raw)
    require(resource['passed'] is True and resource['maximum_simultaneous_children']==1
            and resource['scope']=='CPU_preparation_only' and resource['native_GPU_fit_claimed'] is False,
            'No audio/native approval inferred from metadata/resource reviews')
    require(resource['whole_owned_private_allowance_bytes']+resource['outer_controller_launcher_reserve_bytes']==1_934_217_728,
            'No changed inner/outer family margin or duplicated parent accounting')
    from bpe_validation_owned_checkpoint_pairs_v1 import policy_values
    policy=policy_values(read_bound(storage_policy,guard))
    require(policy['maximum_whole_namespace_allocation_bytes']==629145600,
            'Exact separately reviewed bounded two-pair physical reservation')
    ancestor=template['continuation_ancestor'];pointer=verify_ancestor(ancestor,guard=guard)
    pointer,saved=actual_saved_schedule(ancestor['checkpoint_pointer'],guard=guard)
    context=ScheduleContext(template['planning_arguments'],guard=guard)
    try:
        schedule=context.new_schedule();schedule.load_state_dict(normalize_schedule_state(saved))
        initial=copy.deepcopy(schedule.state_dict());plan21=context.plan(schedule,maximum_updates=1)
        require(digest(schedule.state_dict())==digest(initial)==plan21['initial_sampler_sha256']==pointer['sampler_state_sha256']
                and plan21['initial_committed_updates']==20,'Private planning cannot consume actual ancestor frontiers')
        preview=context.new_schedule();preview.load_state_dict(normalize_schedule_state(plan21['final_private_planning_sampler']))
        plan22=context.plan(preview,maximum_updates=1)
        require(plan22['initial_committed_updates']==21 and plan22['initial_sampler_sha256']==plan21['updates'][-1]['after_sampler_sha256'],
                'Private preview22 is exact successor of virtual plan21, never actual positive native proof')
        from bpe_final_DATA_window_planner_v4 import feature_upper_bound
        from asr18_calibrated_geometry_forecast_v6 import CalibratedGeometryForecast
        calibration=CalibratedGeometryForecast(**template['forecast_calibration'],guard=guard)
        geometry=[]
        for plan in (plan21,plan22):
            rows=plan['updates'][0]['rows'];groups={};declared_metadata=[]
            for row in rows:
                role=row['augmentation_role'];bucket=groups.setdefault(role,{'rows':0,'MAXT':0,'MAXU':0,'physical_feature_upper_bytes':0})
                ids=row['tr_metadata_labels'] if role=='student_tr' else row['clean_target_ids']
                frames=row['declared_frontend_geometry']['physical_feature_frames'] if role=='student_tr' else 8*row['encoder_frames']+1
                samples=row['declared_geometry']['clean16k_samples'] if role=='student_tr' else 8*row['encoder_frames']*160+159
                bucket['rows']+=1;bucket['MAXT']=max(bucket['MAXT'],row['encoder_frames'])
                bucket['MAXU']=max(bucket['MAXU'],len(ids));bucket['physical_feature_upper_bytes']+=frames*513+8192
                declared_metadata.append({'role':role,'encoder_frames':row['encoder_frames'],'full_target_ids':ids,
                    'input_features':{'shape':[1,frames,128]},'samples':samples})
            # Same actual admissible source geometry policy; this does not
            # lower calibrated private high water or infer actual row fit.
            declared_budget=calibration.budget(rows=declared_metadata,current_private_bytes=3_000_000_000,update_in_process=1)
            geometry.append({'global_update':plan['initial_committed_updates']+1,'groups':groups,
                'feature_window_upper_bytes':feature_upper_bound(plan['phase'],rows)+plan['metadata_reserve_bytes'],
                'forecast_inputs_are_declared_geometry_upper_bounds_not_actual_prepared_features':True,
                'same_unchanged_forecast_policy_budget':declared_budget})
    finally:context.close()
    folder.mkdir()
    p21=folder/'actual20_successor_plan21.json';p22=folder/'private_only_preview_plan22.json'
    write_json(p21,plan21);write_json(p22,plan22)
    contract=read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v6.json')
    require(contract['kind']=='FastMSS_row_serial_CPU_preparation_contract_v6' and contract['TEMPLATE_NOT_LAUNCHABLE'] is True,
            'Unchanged CPU6 public contract schema')
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(template['planning_arguments']['DATA_evidence']),
        outer_parent_source=template['parent_driver_source'],resource_review=resource,raw_lifetime_review=raw,
        actual_ROOT_CPU_resource_review=CPU_resource_review)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract)
    session={'kind':SESSION_KIND,'purpose':PURPOSE,'production_enabled':False,'diagnostic_only':True,
        'maximum_windows':2,'maximum_native_updates':40,'initial_optimizer_updates':20,
        'initial_sampler_sha256':plan21['initial_sampler_sha256'],'ancestor':ancestor,
        'coordinator_source':record(S/'bpe_production_window_coordinator_v4.py'),
        'augmentation_identity':template['augmentation_identity'],
        'scientific_source_state':template['planning_arguments']['snapshot_binding'],
        'preserve_original_all207_threeAdam_allRNG_and_replay_lineage':True,
        'DATA_evidence_sha256':digest(template['planning_arguments']['DATA_evidence']),
        'CPU_preparation_contract':record(cp),'configuration_assembler_source':own,
        'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0,
        'new_diagnostic_orchestration_indices0_1_do_not_reset_global_scientific_updates':True}
    sp=folder/'continuation21_22_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    config={**template,'kind':'ASR18_checkpoint20_two_one_update_windows_diagnostic_config_v1',
        'TEMPLATE_NOT_LAUNCHABLE':False,'finite_diagnostic_execution_enabled':True,
        'source_manifest':source_manifest,'session':record(sp),'provisional_plan21':record(p21),'private_preview_plan22':record(p22),
        'CPU_owned_allowance_bytes':1_934_217_728,'CPU_inner_owned_allowance_bytes':resource['whole_owned_private_allowance_bytes'],
        'CPU_outer_controller_launcher_reserve_bytes':resource['outer_controller_launcher_reserve_bytes'],
        'configuration_assembly_source':own,'configuration_source_read_review':source_review,
        'actual_ROOT_CPU_resource_review':CPU_resource_review,'raw_lifetime_review':raw,
        'validation_checkpoint_policy':storage_policy,
        'exact_ROOT_execution_configuration_review_still_required':True,
        'CPU_preparation_contract_and_whole_family_allowance_require_separate_actual_review':False}
    cfg=folder/'configuration_pending_execution_review.json';write_json(cfg,config)
    from fastmss_owned_process_closure_v2 import identity
    result={'kind':'ASR18_checkpoint20_continuation21_22_actual_metadata_configuration_v1','passed':True,
        'producer':identity(),'worker_source':own,'parent_source':own,'configuration':record(cfg),'session':record(sp),
        'CPU_contract':record(cp),'provisional_plan21':record(p21),'private_only_preview_plan22':record(p22),
        'source_manifest':source_manifest,'source_review':source_review,'CPU_resource_review':CPU_resource_review,
        'storage_policy':storage_policy,'initial_actual20_ancestor':ancestor,'declared_geometry_preview':geometry,
        'actual_CPU_frontend_or_GPU_jobs_launched':0,'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0,
        'shared_CACHE_created':False,'checkpoint_owned_namespace_created':False,'production_enabled':False,
        'ROOT_exact_configuration_storage_and_execution_review_still_required':True}
    write_json(folder/'report.json',result);print(json.dumps({'passed':True,'configuration':result['configuration']}),flush=True)
    return result


def pin(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source-manifest','source-review','CPU-resource-review','storage-policy','folder'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    assemble(source_manifest=pin(args.source_manifest),source_review=pin(args.source_review),
        CPU_resource_review=pin(args.CPU_resource_review),storage_policy=pin(args.storage_policy),folder=Path(args.folder).resolve())
