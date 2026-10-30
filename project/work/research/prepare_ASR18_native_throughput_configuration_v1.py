"""Metadata-only original first4 paired-profile configuration; actual CPU8 bridge."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_phase_aware_plan_bridge_v7 import read_bound,bridge_plan
from fastmss_row_serial_cpu_v8 import environment,resource_guard,verify_contract


def assemble(manifest_pin,review_pin,CPU_review_pin,folder):
    environment();guard=resource_guard;m=read_bound(manifest_pin,guard);r=read_bound(review_pin,guard)
    require(r['accepted'] is True and r['reviewer']=='/root' and r['source_manifest']==manifest_pin
        and r['scope']=='source_read_review_for_original_first4_throughput_metadata_v1'
        and r['metadata_builder_source']==record(__file__),'Exact finite metadata source review')
    for value in m['bindings']:require(record(value['path'])==value,'Frozen reused/new profiling source')
    cpu=read_bound(CPU_review_pin,guard);base=read_bound(m['base_configuration'],guard)
    require(cpu['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and cpu['accepted'] is True
        and cpu['scope']=='CPU_preparation_only' and cpu['CPU_source_manifest']==m['CPU_owner_bundle']
        and cpu['outer_parent_source']==m['component_sources']['harness'],'Unchanged CPU8/raw allowance, truthful component parent')
    require(folder.is_relative_to(A) and not folder.exists(),'Fresh owned E metadata output, no audio/cache creation')
    from asr18_full_run_source_context_v1 import FullRunSourceContext
    from bpe_full_run_lifecycle_v2 import progress,next_boundary_request
    context=FullRunSourceContext(base['planning_arguments'],guard=guard)
    try:
        schedule=context.new_schedule();initial=digest(schedule.state_dict())
        plan=context.validation_plan(schedule,completed_boundaries=0,maximum_updates=4)
        final=progress(plan['final_private_planning_sampler'])
        require(schedule.committed_updates()==0 and digest(schedule.state_dict())==initial==plan['initial_sampler_sha256']
            and len(plan['updates'])==4 and final['optimizer_updates']==4 and final['broad_draws']==56
            and next_boundary_request(plan['final_private_planning_sampler'],0)['kind']=='bounded_training_window',
            'Actual original first4/TR56 plus eight carried replay rows; no consumed draw or fabricated pass')
    finally:context.close()
    folder.mkdir();plan_path=folder/'original_first4_plan.json';write_json(plan_path,plan)
    contract=read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v8.json')
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(base['planning_arguments']['DATA_evidence']),
        outer_parent_source=m['component_sources']['harness'],resource_review=copy.deepcopy(cpu['resource_review']),
        raw_lifetime_review=copy.deepcopy(cpu['raw_lifetime_review']),actual_ROOT_CPU_resource_review=CPU_review_pin)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract)
    source_session=read_bound(base['session'],guard)
    # Initial-window CPU8 does not read or create a durable owner. Preserve
    # exactly the already-reviewed expected future owner bytes/SHA, but bind
    # them to a fresh NEVER-created profiling namespace, with no adoption.
    future=copy.deepcopy(source_session['durable_owner'])
    uncreated=Path(r'E:\gpu_training_scratchpad\runs\ASR18_native_throughput_profile_UNCREATED_01')
    require(not uncreated.exists(),'Never create/adopt a full-owner namespace for this component')
    future['path']=str(uncreated/'durable/owner.json')
    session={key:copy.deepcopy(value) for key,value in source_session.items()
        if key not in ('CPU_preparation_contract','durable_owner','configuration_assembler_source',
            'durable_owner_is_predicted_not_actual_until_runtime_creation')}
    session.update(CPU_preparation_contract=record(cp),durable_owner=future,configuration_assembler_source=record(__file__),
        durable_owner_is_predicted_not_created_in_this_component=True,
        native_component_scope='original_first4_private_diagnostic_no_production_namespace',
        actual_optimizer_updates=0,actual_authoritative_draws_consumed=0)
    sp=folder/'CPU_preparation_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    bridge=bridge_plan(record(plan_path),DATA_evidence=base['planning_arguments']['DATA_evidence'],session_binding=record(sp),
        window_index=0,prior_window_boundary=None,guard=guard)
    require(len(bridge['updates'])==4 and bridge['diagnostic_only'] is False,
        'Same actual full-mode CPU8 original first4 preparation path; native scope separately diagnostic')
    bp=folder/'actual_CPU8_metadata_bridge.json';write_json(bp,bridge)
    keys=('CPU_python','GPU_python','CPU_guard_source','CPU_controller_source','planning_arguments','augmentation_identity',
        'CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes','CPU_outer_controller_launcher_reserve_bytes')
    config={key:copy.deepcopy(base[key]) for key in keys}
    config.update(kind='ASR18_original_first4_native_throughput_component_configuration_v1',production_enabled=False,
        native_updates=4,window_count=1,source_manifest=manifest_pin,harness_source=m['component_sources']['harness'],
        resource_policy=m['resource_policy'],maximum_CPU_phase_seconds=480,maximum_GPU_phase_seconds=900,
        plan=record(plan_path),bridge=record(bp),session=record(sp),CPU_contract=record(cp),
        actual_ROOT_CPU_resource_review=CPU_review_pin,candidate_baseline_initial=None,candidate_shared_features=None,
        actual_metadata_source_review=review_pin,source_assembler=record(__file__),
        shared_features_retained_immutable_for_separately_reviewed_pair=True,
        each_arm_separate_native0_then_fresh_all207_threeAdam_all4RNG_recovery_required=True,
        independently_measured_baseline_initial_all4_RNG_required_before_candidate=True)
    path=folder/'configuration_pending_execution_review.json';write_json(path,config)
    from fastmss_owned_process_closure_v2 import identity
    report={'kind':'ASR18_actual_original_first4_throughput_metadata_configuration_v1','passed':True,'producer':identity(),
        'source':record(__file__),'configuration':record(path),'source_manifest':manifest_pin,'source_review':review_pin,
        'CPU_resource_review':CPU_review_pin,'plan':record(plan_path),'session':record(sp),'CPU_contract':record(cp),
        'actual_metadata_bridge':record(bp),'actual_CPU8_bridge_call_passed':True,'original_initial_schedule_unchanged':True,
        'actual_audio_GPU_optimizer_jobs':0,'actual_authoritative_draws_consumed':0,'future_owner_created':False,
        'production_enabled':False}
    write_json(folder/'report.json',report);print(json.dumps({'passed':True,'configuration':record(path)}),flush=True)


def arg(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source-manifest','ROOT-review','CPU-review','folder'):p.add_argument('--'+name,required=True)
    a=p.parse_args();assemble(arg(a.source_manifest),arg(a.ROOT_review),arg(a.CPU_review),Path(a.folder).resolve())
