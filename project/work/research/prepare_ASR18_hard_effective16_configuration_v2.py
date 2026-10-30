"""Exact hard-effective16 metadata configuration; real bridge call before audio."""
import argparse,copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest
from fastmss_row_serial_cpu_v8 import environment,resource_guard,verify_contract,verify_raw_review
from fastmss_phase_aware_plan_bridge_v7 import read_bound,bridge_plan
from asr18_hard_envelope_schedule_context_v1 import HardEnvelopeContext,CONTEXT
from asr18_full_population_geometry_forecast_v1 import CalibratedGeometryForecast
from asr18_full_population_separated_forecast_v1 import resident_worker_forecast
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'

def assemble(*,source_manifest,source_review,CPU_resource_review,folder):
    environment();guard=resource_guard
    require(folder.is_relative_to(A) and not folder.exists(),'Fresh E-only metadata namespace')
    sources=read_bound(source_manifest,guard);review=read_bound(source_review,guard)
    require(review['accepted'] is True and review['source_manifest']==source_manifest
            and review['scope']=='source_read_review_for_hard_effective16_current_CPU_native_diagnostic'
            and review['production_approval_claimed'] is False,'Exact new source read scope')
    for pin in sources['bindings']:require(record(pin['path'])==pin,'Frozen dependency closure changed')
    own=record(__file__);require(own in sources['bindings'],'Exact source-bound assembler')
    template=read_bound(sources['configuration_template'],guard)
    require(template['TEMPLATE_NOT_LAUNCHABLE'] is True and template['finite_diagnostic_execution_enabled'] is False,'Disabled exact template')
    CPU=read_bound(CPU_resource_review,guard)
    require(CPU['kind']=='ASR18_reviewed_row_serial_CPU_resource_contract_v1' and CPU['accepted'] is True
            and CPU['reviewer']=='/root' and CPU['scope']=='CPU_preparation_only' and CPU['production_enabled'] is False
            and CPU['CPU_source_manifest']==sources['CPU_owner_bundle'] and CPU['outer_parent_source']==template['parent_driver_source'],
            'Exact outer parent/CPU8 source identity and unchanged numeric review')
    resource=copy.deepcopy(CPU['resource_review']);raw=copy.deepcopy(CPU['raw_lifetime_review']);verify_raw_review(raw)
    require(resource['whole_owned_private_allowance_bytes']+resource['outer_controller_launcher_reserve_bytes']==1934217728
            and resource['maximum_simultaneous_children']==1,'Same serial full CPU/raw/private/workspace allowance')
    context=HardEnvelopeContext(template['planning_arguments'],template['hard_row_provenance'],guard=guard)
    try:
        schedule=context.new_schedule();initial=schedule.state_dict();plan=context.plan(schedule,maximum_updates=1)
        require(digest(initial)==digest(schedule.state_dict())==plan['initial_sampler_sha256']
                and schedule.committed_updates()==0 and len(plan['updates'])==1,'Provisional metadata never consumes original/diagnostic/replay RNG')
        folder.mkdir()
        document={'kind':CONTEXT,'planning_arguments':template['planning_arguments'],'prior_closed20':template['hard_row_provenance'],
            'initial_diagnostic_sampler':initial,'schedule_adapter_source':record(S/'asr18_hard_envelope_schedule_context_v1.py')}
        dp=folder/'hard_envelope_diagnostic_context.json';write_json(dp,document);plan['hard_envelope_diagnostic_context']=record(dp)
        geometry=[];forecast=[]
        for row in plan['updates'][0]['rows']:
            ids=row['tr_metadata_labels'] if row['language']=='tr' else row['clean_target_ids']
            if row['language']=='tr':
                d=row['declared_geometry'];n=d['clean16k_samples'];frames=n//160+1
            else:n=8*row['encoder_frames']*160+159;frames=8*row['encoder_frames']+1
            role='student_tr' if row['language']=='tr' else 'replay'
            geometry.append({'source_id':row['source_id'],'role':role,'T':row['encoder_frames'],'U':len(ids),
                'declared_or_conditional_N':n,'declared_or_conditional_physical_frames':frames,'runtime_full_PCM_and_geometry_required':True})
            forecast.append({'role':role,'encoder_frames':row['encoder_frames'],'full_target_ids':ids,
                'samples':n,'input_features':{'shape':[1,frames,128]}})
        require(geometry[13]['T']==1991 and geometry[13]['U']==574 and geometry[13]['declared_or_conditional_N']==2547840,
                'Complete extremal retained hard row last among14TR, never capped')
        calibration=CalibratedGeometryForecast(**template['forecast_calibration'],guard=guard)
        budget=calibration.budget(rows=forecast,current_private_bytes=3000000000,update_in_process=1)
        whole=resident_worker_forecast(budget)
        require(whole['entire_worker_private_forecast_bytes']==6748646272,'Exact conservative new whole worker')
    finally:context.close()
    pp=folder/'hard_effective16_plan.json';write_json(pp,plan)
    contract=read_json(F/'FastMSS_row_serial_CPU_contract_TEMPLATE_NOT_LAUNCHABLE_v8.json')
    require(contract['kind']=='FastMSS_source_typed_row_serial_CPU_preparation_contract_v1' and contract['TEMPLATE_NOT_LAUNCHABLE'] is True,
            'Exact CPU8 source-typed contract template')
    contract.pop('planned_native_updates',None)
    contract.update(mode='one_hard_envelope_diagnostic',purpose='bounded_one_full_hard_envelope_diagnostic',
        maximum_session_windows=1,maximum_session_native_updates=1,
        TEMPLATE_NOT_LAUNCHABLE=False,DATA_evidence_sha256=digest(template['planning_arguments']['DATA_evidence']),
        outer_parent_source=template['parent_driver_source'],resource_review=resource,raw_lifetime_review=raw,
        actual_ROOT_CPU_resource_review=CPU_resource_review)
    cp=folder/'reviewed_CPU_preparation_contract.json';write_json(cp,contract)
    session={'kind':'FastMSS_one_full_hard_envelope_diagnostic_CPU_session_v1','mode':'one_hard_envelope_diagnostic',
        'purpose':'bounded_one_full_hard_envelope_diagnostic','production_enabled':False,'diagnostic_only':True,
        'maximum_windows':1,'maximum_native_updates':1,'initial_optimizer_updates':0,'initial_sampler_sha256':plan['initial_sampler_sha256'],
        'hard_envelope_diagnostic_context':record(dp),'hard_context_verifier_source':record(S/'asr18_hard_envelope_schedule_context_v1.py'),
        'coordinator_source':record(S/'bpe_production_window_coordinator_v4.py'),'augmentation_identity':template['augmentation_identity'],
        'scientific_source_state':template['planning_arguments']['snapshot_binding'],
        'preserve_original_all207_threeAdam_allRNG_and_replay_lineage':True,'DATA_evidence_sha256':digest(template['planning_arguments']['DATA_evidence']),
        'CPU_preparation_contract':record(cp),'configuration_assembler_source':own,'actual_optimizer_updates':0,
        'actual_authoritative_draws_consumed':0,'production_source_eligible':False,'production_phase_advance_claimed':False}
    sp=folder/'hard_effective16_session.json';write_json(sp,session);verify_contract(record(cp),session,guard=guard)
    bridged=bridge_plan(record(pp),DATA_evidence=template['planning_arguments']['DATA_evidence'],session_binding=record(sp),
        window_index=0,prior_window_boundary=None,guard=guard)
    require(bridged['phase']==plan['phase'] and len(bridged['updates'])==1 and len(bridged['updates'][0]['rows'])==16
            and bridged['final_sampler_sha256']==plan['updates'][0]['after_sampler_sha256'],
            'Actual real metadata bridge contract must pass before original audio or native work')
    bp=folder/'actual_metadata_bridged_hard_plan.json';write_json(bp,bridged)
    config={**template,'kind':'ASR18_hard_effective16_CPU8_native_diagnostic_config_v1','TEMPLATE_NOT_LAUNCHABLE':False,
        'finite_diagnostic_execution_enabled':True,'source_manifest':source_manifest,'session':record(sp),
        'hard_envelope_diagnostic_context':record(dp),'provisional_plan':record(pp),'initial_sampler_sha256':plan['initial_sampler_sha256'],
        'actual_metadata_bridged_plan':record(bp),'CPU_owned_allowance_bytes':1934217728,
        'CPU_inner_owned_allowance_bytes':resource['whole_owned_private_allowance_bytes'],
        'CPU_outer_controller_launcher_reserve_bytes':resource['outer_controller_launcher_reserve_bytes'],
        'actual_ROOT_CPU_resource_review':CPU_resource_review,'raw_lifetime_review':raw,'configuration_assembly_source':own,
        'configuration_source_read_review':source_review,'entire_worker_forecast':whole,'exact_ROOT_execution_configuration_review_still_required':True}
    cf=folder/'configuration_pending_execution_review.json';write_json(cf,config)
    from fastmss_owned_process_closure_v2 import identity
    report={'kind':'ASR18_hard_effective16_actual_metadata_configuration_v1','passed':True,'producer':identity(),'worker_source':own,
        'configuration':record(cf),'session':record(sp),'CPU_contract':record(cp),'plan':record(pp),'context':record(dp),
        'actual_metadata_bridged_plan':record(bp),'actual_metadata_bridge_call_passed':True,'source_manifest':source_manifest,
        'source_review':source_review,'CPU_resource_review':CPU_resource_review,'geometry':geometry,'native_forecast':budget,
        'whole_worker_forecast':whole,'actual_audio_frontend_GPU_Adam_jobs':0,'actual_authoritative_draws_consumed':0,'production_enabled':False}
    write_json(folder/'report.json',report);print(json.dumps({'passed':True,'configuration':record(cf)}),flush=True)

def argument(value):return json.loads(value) if value.lstrip().startswith('{') else read_json(value)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source-manifest','source-review','CPU-resource-review','folder'):p.add_argument('--'+name,required=True)
    a=p.parse_args();assemble(source_manifest=argument(a.source_manifest),source_review=argument(a.source_review),
        CPU_resource_review=argument(a.CPU_resource_review),folder=Path(a.folder).resolve())
