"""Metadata-only original9223 first20; actual source-bound CPU bridge before audio."""
import argparse,ast,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
A=N.parent/'ASR18_memory_diagnosis_v1';F=N.parent/'FastMSS_integration_v1'
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from atomic_io_retry_v1 import read_json,write_json
from fastmss_BPE9223_cpu_common_v3 import environment,resource_guard,verify_contract
from fastmss_BPE9223_plan_bridge_v3 import bridge_plan,read_bound


def assemble(manifest_pin,review_pin,folder):
    environment();guard=resource_guard;m=read_bound(manifest_pin,guard);r=read_bound(review_pin,guard)
    require(r['accepted'] is True and r['reviewer']=='/root' and r['scope']=='metadata_BPE9223_sustained20_configuration_v1'
        and r['source_manifest']==manifest_pin and r['metadata_builder_source']==record(__file__),
        'Exact bounded metadata source/resource review before no-audio assembly')
    for pin in m['bindings']:require(record(pin['path'])==pin,'Frozen source closure changed')
    base=read_bound(m['base_configuration'],guard);owner=m['harness_source'];cpu=m['CPU_source_handoff']
    cpu=read_bound(cpu,guard);sources=cpu['sources'];data=read_bound(m['population_evidence'],guard)
    identity=read_bound(m['augmentation_identity'],guard)
    require(folder.is_relative_to(N) and not folder.exists(),'Fresh private metadata outputs only')
    arguments=copy.deepcopy(base['planning_arguments']);arguments['DATA_evidence']=data
    from asr18_BPE9223_source_context_v2 import FullRunSourceContext9223
    from asr18_BPE9223_lifecycle_v3 import progress,next_boundary_request
    context=FullRunSourceContext9223(arguments,guard=guard)
    try:
        schedule=context.new_schedule();before=copy.deepcopy(schedule.state_dict())
        plan=context.validation_plan(schedule,completed_boundaries=0,maximum_updates=20)
        final=progress(plan['final_private_planning_sampler'])
        require(len(plan['updates'])==20 and final['optimizer_updates']==20 and final['broad_draws']==280
            and final['cleaner_draws']==0 and digest(before)==plan['initial_sampler_sha256']==digest(schedule.state_dict())
            and next_boundary_request(plan['final_private_planning_sampler'],0)['kind']=='bounded_training_window',
            'Original first280 fullTR draws+40 carried replay, no reset/consumed draw/fake pass')
    finally:context.close()
    folder.mkdir();pp=folder/'original9223_first20_plan.json';write_json(pp,plan)
    contract=read_json(A/'native_throughput_first4_configuration_01/reviewed_CPU_preparation_contract.json')
    remap={'bridge':'bridge','common':'core','sink':'engine','reference_worker':'reference',
        'feature_worker':'frontend','controller':'parent'}
    for key,new in remap.items():contract['sources'][key]=sources[new]
    context_pin=record(S/'asr18_BPE9223_source_context_v2.py')
    contract['sources']['cleaner']=context_pin
    contract.update(TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(data),outer_parent_source=owner,
        vocabulary=9223,source_context_source=context_pin,label_population_contract=data['BPE9223_population'],
        owner_validation_only=True,checkpoint_diagnostic_only=True,augmentation_identity=identity,
        actual_ROOT_CPU_resource_review=review_pin)
    cp=folder/'CPU_numerical_contract.json';write_json(cp,contract)
    oldsession=read_bound(base['session'],guard)
    session=copy.deepcopy(oldsession)
    for key in ('hard_envelope_diagnostic_context','hard_context_verifier_source','maximum_windows','maximum_native_updates'):
        session.pop(key,None)
    future=copy.deepcopy(session['durable_owner']);future['path']=str(Path(r'E:\gpu_training_scratchpad\runs\BPE9223_sustained_UNCREATED_01\durable\owner.json'))
    require(not Path(future['path']).parent.parent.exists(),'Prediction only; no full-owner namespace creation in this component')
    session.update(CPU_preparation_contract=record(cp),augmentation_identity=identity,DATA_evidence_sha256=digest(data),
        vocabulary=9223,source_context_source=context_pin,label_population_contract=data['BPE9223_population'],
        owner_validation_only=True,checkpoint_diagnostic_only=True,diagnostic_only=True,production_source_eligible=False,
        full_run_identity_sha256=digest({'scope':'original9223_sustained20_only','planning_arguments':arguments,'harness':owner,
            'migration':m['migration_plan'],'qualification_only':True}),initial_sampler_sha256=digest(before),
        durable_owner=future,durable_owner_is_predicted_not_created_in_this_component=True,
        configuration_assembler_source=record(__file__),native_component_scope='original9223_sustained20_not_production',
        actual_optimizer_updates=0,actual_authoritative_draws_consumed=0)
    sp=folder/'CPU_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    bridge=bridge_plan(record(pp),DATA_evidence=data,session_binding=record(sp),window_index=0,guard=guard)
    require(len(bridge['updates'])==20 and bridge['diagnostic_only'] is True,'Genuine actual CPU9223 metadata bridge before audio')
    bp=folder/'actual_CPU9223_bridge.json';write_json(bp,bridge)
    old=read_bound(base['recorded_room_source_contract'],guard)
    public=__import__('fastmss_BPE9223_feature_protocol_v3')
    contract_keys={'protocol':'protocol','engine':'engine','reference':'reference','frontend':'frontend',
        'controller':'parent','core':'core','augmentation_adapter':'adapter'}
    for key,new in contract_keys.items():old['sources'][key]=sources[new]
    old.update(kind=public.CONTRACT,mode=public.MODE,numerical_CPU_contract=record(cp),bridge=record(bp),
        augmentation_identity=identity,original_augmentation_identity=identity,augmentation_policy=cpu['policy'],
        vocabulary=9223,owner_validation_only=True,checkpoint_diagnostic_only=True,
        production_source_eligible=False,validation_only=False,native_benchmark_only=False,process_owner='/root/cpu_data_preparation')
    old.pop('numerical_CPU8_contract',None)
    sc=folder/'CPU_source_contract.json';write_json(sc,old);public.verify_source_contract(record(sc),guard=guard)
    output=str(N/'sustained20_CPU_window_actual_01')
    proposal={'kind':public.CONFIGURATION,'production_enabled':False,'native_benchmark_only':False,
        'vocabulary':9223,'owner_validation_only':True,'checkpoint_diagnostic_only':True,
        'sources':{'parent':sources['parent'],'reference':sources['reference'],'frontend':sources['frontend'],
            'core':sources['core'],'guard':old['sources']['guard'],'protocol':sources['protocol']},
        'python':base['CPU_python'],'outer_parent_source':owner,'freeze_after_setup':True,
        'namespace':'BPE9223_sustained20_qualification_01','source_contract':record(sc),
        'output_root':output,'window_id':'sustained20_01','window_index':0,'template_configuration':None}
    wp=folder/'CPU_window_configuration.json';write_json(wp,proposal)
    keys=('CPU_python','GPU_python','CPU_guard_source','CPU_owned_allowance_bytes','CPU_inner_owned_allowance_bytes',
        'CPU_outer_controller_launcher_reserve_bytes','optimization','profile_enabled','grouped_Transformer',
        'grouped_original_head_reference','grouped_oracle_observers_enabled','CPU_phase_parent_cap_bytes',
        'native_phase_parent_cap_bytes','CPU_phase_combined_forecast_bytes','maximum_CPU_phase_seconds',
        'maximum_GPU_phase_seconds','CUDA_capacity_policy','loss_complete_union_cap_bytes','loss_packed_work_unit_cap_bytes')
    config={k:copy.deepcopy(base[k]) for k in keys}
    tree=ast.parse((S/'tdt_blocked_bf16_joint_view_9223_v1.py').read_text())
    policy=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='POLICY' for v in n.targets)))
    resources=copy.deepcopy(base['resource_policy'])
    resources.update(kind='ASR18_BPE9223_sustained20_resource_policy_v1',additional_profiler_host_budget_bytes=0,
        appended512_full_parameters_moments_gradients_and_bounded_metadata_pool_bytes=32_000_000,
        maximum_detached_observed_output_CUDA_bytes=0)
    config.update(kind='ASR18_BPE9223_sustained_original_source_configuration_v1',production_enabled=False,
        native_updates=20,window_count=1,vocabulary=9223,owner_validation_only=True,checkpoint_diagnostic_only=True,
        source_manifest=manifest_pin,harness_source=owner,source_assembler=record(__file__),resource_policy=resources,
        row_migration_plan=m['migration_plan'],planning_arguments=arguments,augmentation_identity=identity,
        plan=record(pp),bridge=record(bp),session=record(sp),CPU_contract=record(cp),declared_loss_joint_policy=policy,
        CPU_controller_source=sources['parent'],recorded_room_CPU_configuration=record(wp),recorded_room_source_contract=record(sc),
        recorded_room_reader_source=sources['protocol'],recorded_room_CPU_output_root=output,
        recorded_room_CPU_execution_review=None,CPU_review_join_required_before_any_execution=True,
        no_diagnostic_baseline_RNG_or_weights_adopted=True)
    out=folder/'native_configuration_pending_CPU_review.json';write_json(out,config)
    report={'kind':'ASR18_BPE9223_actual_sustained20_metadata_configuration_v1','passed':True,'source':record(__file__),
        'source_manifest':manifest_pin,'configuration':record(out),'CPU_configuration':record(wp),
        'plan':record(pp),'session':record(sp),'actual_metadata_bridge':record(bp),'CPU_contract':record(cp),
        'CPU_source_contract':record(sc),'actual_audio_or_GPU_jobs':0,'actual_authoritative_draws_consumed':0,
        'original_initial_sampler_unchanged':True,'full_owner_namespace_created':False,'production_enabled':False}
    write_json(folder/'report.json',report);print(json.dumps(report),flush=True)


def arg(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-manifest','ROOT-review','folder'):p.add_argument('--'+key,required=True)
    a=p.parse_args();assemble(arg(a.source_manifest),arg(a.ROOT_review),Path(a.folder).resolve())
