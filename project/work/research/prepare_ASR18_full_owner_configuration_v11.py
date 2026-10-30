"""Metadata-only DISABLED full-two-broad/one-cleaner configuration and original first20 CPU8 bridge."""
import argparse,copy,hashlib,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_row_serial_cpu_v8 import environment,resource_guard,verify_contract,verify_raw_review
from fastmss_phase_aware_plan_bridge_v7 import read_bound,bridge_plan
from asr18_full_run_source_context_v1 import FullRunSourceContext
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'

def assemble(*,source_manifest,source_review,CPU_resource_review,template_pin,folder):
    environment();guard=resource_guard
    require(folder.is_relative_to(A) and not folder.exists(),'One fresh E-only metadata output')
    manifest=read_bound(source_manifest,guard);review=read_bound(source_review,guard);template=read_bound(template_pin,guard)
    require(review['accepted'] is True and review['source_manifest']==source_manifest
        and review['scope']=='source_read_review_for_disabled_full_run_owner_production_metadata_preparation'
        and review['production_approval_claimed'] is False,'Exact changed full-owner sources separately reviewed')
    for pin in manifest['bindings']:require(record(pin['path'])==pin,'Frozen source closure changed')
    own=record(__file__);require(own in manifest['bindings'] and review['configuration_template']==template_pin,
        'Exact metadata assembler/template')
    require(template['TEMPLATE_NOT_LAUNCHABLE'] and template['production_enabled'] is False
        and template['owner_validation_only'] is False and template['planned_native_updates']==401072
        and template['full_run_identity']['owner_validation_only'] is False
        and template['full_run_identity']['production_source_eligible'] is True
        and template['full_run_launch_enabled'] is False
        and template['normal_boundary_resume_enabled'] is True
        and not any(key in template for key in ('maximum_owner_validation_updates',
            'maximum_validation_updates_per_window','owner_validation_pause_after_updates')),
        'Disabled real-run preparation only; no finite diagnostic fields or long-run launch authority')
    cpu=read_bound(CPU_resource_review,guard)
    require(cpu['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and cpu['accepted'] is True
        and cpu['reviewer']=='/root' and cpu['scope']=='CPU_preparation_only'
        and cpu['CPU_source_manifest']==manifest['CPU_owner_bundle']
        and cpu['outer_parent_source']==template['parent_driver_source'],'Same CPU8 arithmetic, actual new parent identity')
    resource=copy.deepcopy(cpu['resource_review']);raw=copy.deepcopy(cpu['raw_lifetime_review']);verify_raw_review(raw)
    require(resource['whole_owned_private_allowance_bytes']+resource['outer_controller_launcher_reserve_bytes']==1934217728
        and resource['maximum_simultaneous_children']==1,'Original serial wholeCPU domain unchanged')
    identity=template['full_run_identity'];identity_sha=digest(identity)
    for key in ('durable_policy','ROOT_storage_review','ROOT_DEV_policy_review','ROOT_DEV_static_execution_review','ROOT_DEV_resource_review'):
        authority=read_bound(template[key],guard)
        require(authority['accepted'] is True and authority['reviewer']=='/root','Separately genuine immutable authority: '+key)
        if 'run_identity_sha256' in authority:require(authority['run_identity_sha256']==identity_sha,'No foreign identity authority')
    # Actual lightweight constructor/API preflight, NOT native or model proof.
    from asr18_full_run_parent_controller_v13 import FullOwner
    from bpe_full_run_lifecycle_v2 import SourceBoundOwnerCallbacks,CALLBACKS
    from bpe_full_run_durable_owner_adapter_v1 import OwnerProofCallbacks
    from asr18_full_run_persistence_prefix_v1 import OwnerPrefixVerifier
    functions={name:getattr(FullOwner,name) for name in CALLBACKS}
    SourceBoundOwnerCallbacks(sources={name:record(inspect.getsourcefile(fn)) for name,fn in functions.items()},callbacks=functions)
    methods={name:getattr(FullOwner,name) for name in ('verify_pending_window','verify_writer_authority','verify_durable_resume')}
    methods['verify_current_recovery_and_persistence_prefix']=OwnerPrefixVerifier.verify_current_recovery_and_persistence_prefix
    OwnerProofCallbacks(sources={name:record(inspect.getsourcefile(fn)) for name,fn in methods.items()},functions=methods)
    context=FullRunSourceContext(template['planning_arguments'],guard=guard)
    try:
        schedule=context.new_schedule();initial=copy.deepcopy(schedule.state_dict())
        plan=context.plan(schedule,completed_boundaries=0)
        from bpe_full_run_lifecycle_v2 import progress,next_boundary_request
        final=plan['final_private_planning_sampler'];frontier=progress(final)
        require(schedule.committed_updates()==0 and digest(initial)==digest(schedule.state_dict())==plan['initial_sampler_sha256']
            and len(plan['updates'])==20 and frontier['optimizer_updates']==20 and frontier['broad_draws']==280
            and frontier['cleaner_draws']==0 and next_boundary_request(final,0)['kind']=='bounded_training_window',
            'Actual original first20 full-mode private preview; no consumed draw or fabricated pass/DEV')
    finally:context.close()
    folder.mkdir();pp=folder/'prepared_original_first20_plan.json';write_json(pp,plan)
    contract=read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v8.json')
    require(contract['kind']=='FastMSS_source_typed_row_serial_CPU_preparation_contract_v1'
        and contract['mode']=='full_two_broad_one_cleaner' and contract['planned_native_updates']==401072,
        'Exact actual full-mode CPU8 numerical contract, no diagnostic schema projection')
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(template['planning_arguments']['DATA_evidence']),
        outer_parent_source=template['parent_driver_source'],resource_review=resource,raw_lifetime_review=raw,
        actual_ROOT_CPU_resource_review=CPU_resource_review)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract)
    controller=record(template['full_owner_controller_source']['path']);prefix=record(template['full_run_identity']['persistent_prefix_sources']['prefix_verifier']['path'])
    proof_sources={'verify_pending_window':controller,'verify_writer_authority':controller,'verify_durable_resume':controller,
        'verify_current_recovery_and_persistence_prefix':prefix}
    # This is explicitly a deterministic FUTURE binding. Runtime must create
    # that exact owner and compare its actual record3 before any CPU producer.
    owner_value={'kind':'ASR18_future_packed_durable_lifecycle_owner_v2','policy':template['durable_policy'],
        'source':record(S/'bpe_full_run_durable_owner_adapter_v2.py'),
        'transition_source':read_bound(template['durable_policy'],guard)['transition_source'],
        'identity':identity,'proof_sources':proof_sources,'adopt_existing_artifacts':False}
    owner_raw=(json.dumps(owner_value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    expected_owner={'path':str(Path(template['run_namespace'])/'durable/owner.json'),
        'bytes':len(owner_raw),'sha256':hashlib.sha256(owner_raw).hexdigest()}
    session={'kind':'FastMSS_full_run_CPU_preparation_session_v1','mode':'full_two_broad_one_cleaner',
        'purpose':'reviewed_full_two_broad_one_cleaner_production_CPU_preparation','production_enabled':False,
        'diagnostic_only':False,'planned_native_updates':401072,'initial_optimizer_updates':0,
        'initial_sampler_sha256':plan['initial_sampler_sha256'],'full_run_identity_sha256':identity_sha,
        'durable_owner':expected_owner,'durable_owner_is_predicted_not_actual_until_runtime_creation':True,
        'coordinator_source':record(S/'bpe_production_window_coordinator_v4.py'),'augmentation_identity':template['augmentation_identity'],
        'scientific_source_state':template['planning_arguments']['snapshot_binding'],
        'preserve_original_all207_threeAdam_allRNG_and_replay_lineage':True,
        'DATA_evidence_sha256':digest(template['planning_arguments']['DATA_evidence']),
        'CPU_preparation_contract':record(cp),'configuration_assembler_source':own,
        'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0}
    sp=folder/'full_run_CPU_preparation_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    bridge=bridge_plan(record(pp),DATA_evidence=template['planning_arguments']['DATA_evidence'],
        session_binding=record(sp),window_index=0,prior_window_boundary=None,guard=guard)
    require(bridge['diagnostic_only'] is False and len(bridge['updates'])==20
        and bridge['final_sampler_sha256']==plan['updates'][-1]['after_sampler_sha256'],
        'Real source-typed full CPU8 bridge BEFORE any audio/native job')
    bp=folder/'actual_metadata_full_CPU8_bridge.json';write_json(bp,bridge)
    config={**template,'TEMPLATE_NOT_LAUNCHABLE':False,'source_manifest':source_manifest,'session':record(sp),
        'initial_sampler_sha256':plan['initial_sampler_sha256'],'prepared_metadata_plan':record(pp),
        'actual_metadata_bridge':record(bp),'actual_ROOT_CPU_resource_review':CPU_resource_review,
        'CPU_owned_allowance_bytes':1934217728,'CPU_inner_owned_allowance_bytes':resource['whole_owned_private_allowance_bytes'],
        'CPU_outer_controller_launcher_reserve_bytes':resource['outer_controller_launcher_reserve_bytes'],
        'raw_lifetime_review':raw,'configuration_assembly_source':own,'configuration_source_read_review':source_review,
        'exact_ROOT_execution_configuration_review_still_required':True}
    path=folder/'configuration_pending_execution_review.json';write_json(path,config)
    from fastmss_owned_process_closure_v2 import identity as producer
    report={'kind':'ASR18_actual_disabled_full_run_owner_production_metadata_configuration_v1','passed':True,'producer':producer(),
        'worker_source':own,'configuration':record(path),'source_manifest':source_manifest,'source_review':source_review,
        'CPU_resource_review':CPU_resource_review,'session':record(sp),'CPU_contract':record(cp),
        'plan':record(pp),'actual_metadata_bridge':record(bp),'actual_metadata_bridge_call_passed':True,
        'run_identity_sha256':identity_sha,'actual_native_or_audio_jobs':0,'actual_authoritative_draws_consumed':0,
        'lightweight_exact_callback_constructor_preflight_passed':True,
        'future_durable_owner_binding_is_NOT_actual_creation_evidence':True,
        'full_run_launch_enabled':False,'production_execution_admitted':False,
        'legacy_fresh_invocation_token':'genesis_normal_pause4',
        'legacy_token_does_not_enable_validation_pause_in_full_mode':True,'production_enabled':False}
    write_json(folder/'report.json',report);print(json.dumps({'passed':True,'configuration':record(path)}),flush=True)

def argument(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source-manifest','source-review','CPU-resource-review','template','folder'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();assemble(source_manifest=argument(args.source_manifest),source_review=argument(args.source_review),
        CPU_resource_review=argument(args.CPU_resource_review),template_pin=argument(args.template),folder=Path(args.folder).resolve())
